//! 探索アルゴリズムモジュール
//!
//! Negamax、AlphaBeta、MTD(f)探索を実装し、Phase 3学習システムへの統合APIを提供する。

use crate::board::BitBoard;
use thiserror::Error;

/// 探索エラー型
#[derive(Error, Debug)]
pub enum SearchError {
    /// 置換表のメモリ確保失敗
    #[error("Failed to allocate transposition table: {0}")]
    MemoryAllocation(String),

    /// 評価関数エラー（Phase 1からの伝播）
    #[error("Evaluator error: {0}")]
    EvaluationError(String),

    /// 時間切れ（通常は使用しない、最後の完了深さを返すため）
    #[error("Search timeout after {0}ms")]
    TimeoutExceeded(u64),

    /// 不正な盤面状態（デバッグ用）
    #[error("Invalid board state: {0}")]
    InvalidBoardState(String),
}

/// 探索結果構造体
///
/// 探索完了時に最善手、評価値、探索統計を返す。
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 最善手（0-63、なければNone）
    pub best_move: Option<u8>,
    /// 評価値（石差）
    pub score: f32,
    /// 到達深さ
    pub depth: u8,
    /// 探索ノード数
    pub nodes_searched: u64,
    /// 置換表ヒット数
    pub tt_hits: u64,
    /// 探索時間（ミリ秒）
    pub elapsed_ms: u64,
    /// Principal Variation（オプション）
    pub pv: Option<Vec<u8>>,
}

impl SearchResult {
    /// SearchResultを生成
    pub fn new(
        best_move: Option<u8>,
        score: f32,
        depth: u8,
        nodes_searched: u64,
        tt_hits: u64,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            best_move,
            score,
            depth,
            nodes_searched,
            tt_hits,
            elapsed_ms,
            pv: None,
        }
    }

    /// 置換表ヒット率を計算
    pub fn tt_hit_rate(&self) -> f64 {
        if self.nodes_searched == 0 {
            0.0
        } else {
            (self.tt_hits as f64) / (self.nodes_searched as f64)
        }
    }
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Move: {:?}, Score: {:.2}, Depth: {}, Nodes: {}, TT Hits: {}/{} ({:.1}%), Time: {}ms",
            self.best_move,
            self.score,
            self.depth,
            self.nodes_searched,
            self.tt_hits,
            self.nodes_searched,
            self.tt_hit_rate() * 100.0,
            self.elapsed_ms
        )
    }
}

/// Zobristハッシュテーブル
///
/// BitBoardから64ビットハッシュを計算し、置換表のインデックスとして使用する。
pub struct ZobristTable {
    /// 黒石の乱数（位置0-63）
    black: [u64; 64],
    /// 白石の乱数（位置0-63）
    white: [u64; 64],
    /// 手番の乱数
    turn: u64,
}

impl Default for ZobristTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ZobristTable {
    /// 固定シードで初期化
    ///
    /// # Returns
    /// ZobristTable - 乱数テーブル
    #[must_use]
    pub fn new() -> Self {
        // 固定シード値（再現性確保）
        const SEED: u64 = 0x123456789ABCDEF0;

        let mut black = [0u64; 64];
        let mut white = [0u64; 64];

        // 簡易的なLCG（線形合同法）で乱数生成
        let mut rng = SEED;

        for i in 0..64 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            black[i] = rng;

            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            white[i] = rng;
        }

        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let turn = rng;

        Self { black, white, turn }
    }

    /// BitBoardからハッシュ計算
    ///
    /// # Arguments
    /// * `board` - 盤面
    ///
    /// # Returns
    /// u64 - Zobristハッシュ
    ///
    /// # Preconditions
    /// * `board`は合法な盤面状態
    ///
    /// # Postconditions
    /// * 同じ盤面は常に同じハッシュ（決定性）
    /// * 1ビット異なる盤面は高確率で異なるハッシュ
    pub fn hash(&self, board: &BitBoard) -> u64 {
        let mut hash = 0u64;

        // 黒石をXOR
        let mut black_bits = board.black;
        while black_bits != 0 {
            let pos = black_bits.trailing_zeros() as usize;
            hash ^= self.black[pos];
            black_bits &= black_bits - 1; // 最下位ビットをクリア
        }

        // 白石をXOR
        let mut white_bits = board.white_mask();
        while white_bits != 0 {
            let pos = white_bits.trailing_zeros() as usize;
            hash ^= self.white[pos];
            white_bits &= white_bits - 1; // 最下位ビットをクリア
        }

        // 手番をXOR
        if board.turn() == crate::board::Color::White {
            hash ^= self.turn;
        }

        hash
    }
}

/// 置換表エントリの境界タイプ
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// 正確な評価値
    Exact,
    /// alpha値（下限）
    Lower,
    /// beta値（上限）
    Upper,
}

/// 置換表エントリ
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    /// Zobristハッシュ（完全一致確認用）
    pub hash: u64,
    /// 探索深さ
    pub depth: i8,
    /// 境界タイプ
    pub bound: Bound,
    /// 評価値
    pub score: i16,
    /// 最善手（0-63、255=なし）
    pub best_move: u8,
    /// 世代情報
    pub age: u8,
}

/// 置換表
///
/// 評価済み局面を保存・検索し、探索を高速化する。
pub struct TranspositionTable {
    /// エントリ配列
    entries: Vec<Option<TTEntry>>,
    /// テーブルサイズ
    size: usize,
    /// 現在の世代
    current_age: u8,
}

impl TranspositionTable {
    /// 置換表を初期化
    ///
    /// # Arguments
    /// * `size_mb` - メモリサイズ（128-256MB）
    ///
    /// # Returns
    /// Result<TranspositionTable, SearchError> - 初期化成功時は置換表、失敗時はMemoryAllocationエラー
    pub fn new(size_mb: usize) -> Result<Self, SearchError> {
        if !(128..=256).contains(&size_mb) {
            return Err(SearchError::MemoryAllocation(format!(
                "Invalid table size: {}MB (must be 128-256MB)",
                size_mb
            )));
        }

        const ENTRY_SIZE: usize = std::mem::size_of::<Option<TTEntry>>();
        let num_entries = (size_mb * 1024 * 1024) / ENTRY_SIZE;

        // 2の累乗に丸める（ビットマスク最適化のため）
        let size = num_entries.next_power_of_two();

        let entries = vec![None; size];

        Ok(Self {
            entries,
            size,
            current_age: 0,
        })
    }

    /// 局面を検索
    ///
    /// # Arguments
    /// * `hash` - Zobristハッシュ
    ///
    /// # Returns
    /// Option<TTEntry> - ヒット時はエントリ、ミス時はNone
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let index = (hash as usize) & (self.size - 1);

        if let Some(entry) = self.entries[index] {
            // hash値の完全一致確認（衝突検出）
            if entry.hash == hash {
                return Some(entry);
            }
        }

        None
    }

    /// 局面を保存
    ///
    /// # Arguments
    /// * `hash` - Zobristハッシュ
    /// * `entry` - 保存するエントリ
    pub fn store(&mut self, hash: u64, entry: TTEntry) {
        let index = (hash as usize) & (self.size - 1);

        // 置換戦略: 深さ優先 + 世代管理
        let should_replace = if let Some(existing) = self.entries[index] {
            // 異なる世代なら置換
            if existing.age != self.current_age {
                true
            } else {
                // 同じ世代なら深さで判定
                entry.depth >= existing.depth
            }
        } else {
            // 空エントリなら保存
            true
        };

        if should_replace {
            self.entries[index] = Some(entry);
        }
    }

    /// 世代を更新（新しい探索開始時）
    pub fn increment_age(&mut self) {
        self.current_age = self.current_age.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{BitBoard, make_move};

    #[test]
    fn test_zobrist_deterministic() {
        // 同じ盤面で常に同じハッシュ値を返すことを検証
        let zobrist = ZobristTable::new();
        let board = BitBoard::new();

        let hash1 = zobrist.hash(&board);
        let hash2 = zobrist.hash(&board);

        assert_eq!(hash1, hash2, "同じ盤面は同じハッシュ値を返すべき");
    }

    #[test]
    fn test_zobrist_different_boards() {
        // 異なる盤面で異なるハッシュ値を返すことを検証
        let zobrist = ZobristTable::new();
        let board1 = BitBoard::new();
        let mut board2 = BitBoard::new();

        // 1手進める
        let moves = crate::board::legal_moves(&board2);
        let first_move = moves.trailing_zeros() as u8;
        make_move(&mut board2, first_move).unwrap();

        let hash1 = zobrist.hash(&board1);
        let hash2 = zobrist.hash(&board2);

        assert_ne!(hash1, hash2, "異なる盤面は異なるハッシュ値を返すべき");
    }

    #[test]
    fn test_transposition_table_new() {
        // 置換表の初期化テスト
        let tt = TranspositionTable::new(128);
        assert!(tt.is_ok(), "128MBの置換表は初期化できるべき");

        let tt = tt.unwrap();
        assert!(tt.size > 0, "置換表サイズは0より大きいべき");
        assert_eq!(tt.current_age, 0, "初期世代は0");
    }

    #[test]
    fn test_transposition_table_invalid_size() {
        // 不正なサイズでエラーを返すことを検証
        let tt = TranspositionTable::new(64);
        assert!(tt.is_err(), "64MBは範囲外なのでエラーになるべき");

        let tt = TranspositionTable::new(512);
        assert!(tt.is_err(), "512MBは範囲外なのでエラーになるべき");
    }

    #[test]
    fn test_transposition_table_probe_miss() {
        // 置換表のミステスト
        let tt = TranspositionTable::new(128).unwrap();
        let hash = 0x123456789ABCDEF0;

        let result = tt.probe(hash);
        assert!(result.is_none(), "空の置換表はNoneを返すべき");
    }

    #[test]
    fn test_transposition_table_store_and_probe() {
        // 同一局面で既存エントリを取得するテスト
        let mut tt = TranspositionTable::new(128).unwrap();
        let zobrist = ZobristTable::new();
        let board = BitBoard::new();
        let hash = zobrist.hash(&board);

        let entry = TTEntry {
            hash,
            depth: 6,
            bound: Bound::Exact,
            score: 100,
            best_move: 19,
            age: 0,
        };

        tt.store(hash, entry);

        let result = tt.probe(hash);
        assert!(result.is_some(), "保存したエントリは取得できるべき");

        let retrieved = result.unwrap();
        assert_eq!(retrieved.hash, hash, "ハッシュ値が一致するべき");
        assert_eq!(retrieved.depth, 6, "深さが一致するべき");
        assert_eq!(retrieved.score, 100, "評価値が一致するべき");
        assert_eq!(retrieved.best_move, 19, "最善手が一致するべき");
    }

    #[test]
    fn test_transposition_table_replacement_strategy() {
        // 置換戦略（深さ優先 + 世代管理）のテスト
        let mut tt = TranspositionTable::new(128).unwrap();
        let hash = 0x123456789ABCDEF0;

        // 浅い探索結果を保存
        let entry1 = TTEntry {
            hash,
            depth: 3,
            bound: Bound::Exact,
            score: 50,
            best_move: 10,
            age: 0,
        };
        tt.store(hash, entry1);

        // 深い探索結果で上書き
        let entry2 = TTEntry {
            hash,
            depth: 6,
            bound: Bound::Exact,
            score: 100,
            best_move: 19,
            age: 0,
        };
        tt.store(hash, entry2);

        let result = tt.probe(hash).unwrap();
        assert_eq!(result.depth, 6, "深い探索結果が保存されるべき");
        assert_eq!(result.score, 100, "深い探索の評価値が保存されるべき");
    }

    #[test]
    fn test_transposition_table_age_management() {
        // 世代管理のテスト
        let mut tt = TranspositionTable::new(128).unwrap();
        let hash = 0x123456789ABCDEF0;

        let entry1 = TTEntry {
            hash,
            depth: 6,
            bound: Bound::Exact,
            score: 100,
            best_move: 19,
            age: 0,
        };
        tt.store(hash, entry1);

        // 世代を更新
        tt.increment_age();
        assert_eq!(tt.current_age, 1, "世代が1になるべき");

        // 浅い探索でも異なる世代なら置換される
        let entry2 = TTEntry {
            hash,
            depth: 3,
            bound: Bound::Exact,
            score: 50,
            best_move: 10,
            age: 1,
        };
        tt.store(hash, entry2);

        let result = tt.probe(hash).unwrap();
        assert_eq!(result.age, 1, "新しい世代のエントリが保存されるべき");
        assert_eq!(result.depth, 3, "異なる世代なら浅い探索でも置換される");
    }

    #[test]
    fn test_search_result_display() {
        // SearchResultのDisplay traitテスト
        let result = SearchResult::new(Some(19), 1.5, 6, 10000, 5000, 15);
        let display = format!("{}", result);

        assert!(display.contains("Move: Some(19)"), "最善手が含まれるべき");
        assert!(display.contains("Score: 1.50"), "評価値が含まれるべき");
        assert!(display.contains("Depth: 6"), "深さが含まれるべき");
        assert!(display.contains("Nodes: 10000"), "ノード数が含まれるべき");
        assert!(display.contains("50.0%"), "ヒット率が含まれるべき");
    }

    #[test]
    fn test_search_result_hit_rate() {
        // ヒット率計算のテスト
        let result = SearchResult::new(Some(19), 1.5, 6, 10000, 5000, 15);
        assert_eq!(result.tt_hit_rate(), 0.5, "ヒット率は50%");

        let result_zero = SearchResult::new(None, 0.0, 0, 0, 0, 0);
        assert_eq!(result_zero.tt_hit_rate(), 0.0, "ノード数0の場合はヒット率0");
    }
}
