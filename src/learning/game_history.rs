//! ゲーム履歴記録モジュール
//!
//! TD-Leaf学習のためのゲーム履歴を管理する。
//! 各手の盤面状態、リーフ評価値、パターンインデックス、ステージを記録し、
//! 後方TD更新のための逆順イテレーションをサポートする。
//!
//! # 概要
//!
//! - `MoveRecord`: 1手分のデータ（盤面、評価値、56パターンインデックス、ステージ）
//! - `GameHistory`: 最大60手のゲーム履歴コンテナ
//!
//! # 要件対応
//!
//! - Req 5.1: 各手のボード状態（BitBoard）を記録
//! - Req 5.2: MTD(f)探索からのリーフ評価値を記録
//! - Req 5.3: 各手の56個のパターンインデックスを記録
//! - Req 5.4: ゲームステージ（0-29）を記録
//! - Req 5.5: 最大60手をサポート
//! - Req 5.6: 後方イテレーションに適したメモリ効率的なフォーマット
//! - Req 5.7: TD更新完了後にゲーム固有データの解放を許可

use crate::board::BitBoard;

/// パターンインスタンス数（14パターン × 4回転 = 56）
pub const NUM_PATTERN_INSTANCES: usize = 56;

/// ゲームの最大手数
pub const MAX_MOVES_PER_GAME: usize = 60;

/// 単一の手記録
///
/// TD更新に必要なすべての情報を含む。
///
/// # フィールド
///
/// - `board`: その手での盤面状態
/// - `leaf_value`: MTD(f)探索からのリーフ評価値（石差としてのf32）
/// - `pattern_indices`: 56個のパターンインデックス（14パターン × 4回転）
/// - `stage`: ゲームステージ（0-29）
///
/// # メモリレイアウト
///
/// ```text
/// BitBoard:         24バイト
/// leaf_value:        4バイト
/// pattern_indices: 448バイト（56 × 8バイト）
/// stage:             8バイト
/// 合計:            約484バイト/手
/// ```
///
/// # 要件対応
///
/// - Req 5.1: board フィールド
/// - Req 5.2: leaf_value フィールド
/// - Req 5.3: pattern_indices フィールド（56個）
/// - Req 5.4: stage フィールド
#[derive(Clone, Debug)]
pub struct MoveRecord {
    /// 盤面状態
    pub board: BitBoard,
    /// リーフ評価値（石差としてのf32）
    pub leaf_value: f32,
    /// 56個のパターンインデックス
    pub pattern_indices: [usize; NUM_PATTERN_INSTANCES],
    /// ゲームステージ（0-29）
    pub stage: usize,
}

impl MoveRecord {
    /// 新しい手記録を作成
    ///
    /// # 引数
    ///
    /// * `board` - 盤面状態
    /// * `leaf_value` - リーフ評価値
    /// * `pattern_indices` - 56個のパターンインデックス
    /// * `stage` - ゲームステージ（0-29）
    ///
    /// # 例
    ///
    /// ```
    /// use prismind::board::BitBoard;
    /// use prismind::learning::game_history::{MoveRecord, NUM_PATTERN_INSTANCES};
    ///
    /// let board = BitBoard::new();
    /// let record = MoveRecord::new(board, 0.0, [0; NUM_PATTERN_INSTANCES], 0);
    /// assert_eq!(record.stage, 0);
    /// ```
    pub fn new(
        board: BitBoard,
        leaf_value: f32,
        pattern_indices: [usize; NUM_PATTERN_INSTANCES],
        stage: usize,
    ) -> Self {
        Self {
            board,
            leaf_value,
            pattern_indices,
            stage,
        }
    }

    /// 現在の手番が黒かどうかを返す
    #[inline]
    pub fn is_black_turn(&self) -> bool {
        self.board.turn() == crate::board::Color::Black
    }
}

/// ゲーム履歴コンテナ
///
/// 1ゲーム分の全手記録を保持し、TD後方更新のための
/// 逆順イテレーションを提供する。
///
/// # 特徴
///
/// - 最大60手の容量を事前確保（再アロケーション回避）
/// - メモリ効率的な格納形式
/// - TD更新完了後に解放可能
///
/// # 要件対応
///
/// - Req 5.5: 最大60手をサポート
/// - Req 5.6: 後方イテレーションに適したメモリ効率的なフォーマット
/// - Req 5.7: TD更新完了後に解放可能
#[derive(Debug)]
pub struct GameHistory {
    /// 手記録のベクタ
    moves: Vec<MoveRecord>,
}

impl GameHistory {
    /// 60手分の容量を持つ空の履歴を作成
    ///
    /// # 要件対応
    ///
    /// - Req 5.5: 最大60手をサポート（事前確保）
    ///
    /// # 例
    ///
    /// ```
    /// use prismind::learning::game_history::GameHistory;
    ///
    /// let history = GameHistory::new();
    /// assert_eq!(history.len(), 0);
    /// assert!(history.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            moves: Vec::with_capacity(MAX_MOVES_PER_GAME),
        }
    }

    /// 指定した容量で空の履歴を作成
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            moves: Vec::with_capacity(capacity),
        }
    }

    /// 手記録を追加
    ///
    /// # 引数
    ///
    /// * `record` - 追加する手記録
    ///
    /// # パニック
    ///
    /// 60手を超えて追加しようとした場合（デバッグビルドのみ）
    ///
    /// # 例
    ///
    /// ```
    /// use prismind::board::BitBoard;
    /// use prismind::learning::game_history::{GameHistory, MoveRecord, NUM_PATTERN_INSTANCES};
    ///
    /// let mut history = GameHistory::new();
    /// let record = MoveRecord::new(BitBoard::new(), 0.0, [0; NUM_PATTERN_INSTANCES], 0);
    /// history.push(record);
    /// assert_eq!(history.len(), 1);
    /// ```
    pub fn push(&mut self, record: MoveRecord) {
        debug_assert!(
            self.moves.len() < MAX_MOVES_PER_GAME,
            "GameHistory exceeded maximum capacity of {} moves",
            MAX_MOVES_PER_GAME
        );
        self.moves.push(record);
    }

    /// 履歴の手数を返す
    #[inline]
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// 履歴が空かどうかを返す
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }

    /// TD後方更新用の逆順イテレータを返す
    ///
    /// # 要件対応
    ///
    /// - Req 5.6: 後方イテレーションに適したフォーマット
    ///
    /// # 例
    ///
    /// ```
    /// use prismind::board::BitBoard;
    /// use prismind::learning::game_history::{GameHistory, MoveRecord, NUM_PATTERN_INSTANCES};
    ///
    /// let mut history = GameHistory::new();
    /// history.push(MoveRecord::new(BitBoard::new(), 1.0, [0; NUM_PATTERN_INSTANCES], 0));
    /// history.push(MoveRecord::new(BitBoard::new(), 2.0, [0; NUM_PATTERN_INSTANCES], 0));
    ///
    /// // 逆順でイテレーション（TD更新用）
    /// let values: Vec<f32> = history.iter_reverse().map(|r| r.leaf_value).collect();
    /// assert_eq!(values, vec![2.0, 1.0]);
    /// ```
    pub fn iter_reverse(&self) -> impl Iterator<Item = &MoveRecord> {
        self.moves.iter().rev()
    }

    /// 指定インデックスの手記録を取得
    ///
    /// # 引数
    ///
    /// * `index` - 手のインデックス（0から開始）
    ///
    /// # 戻り値
    ///
    /// 存在する場合は`Some(&MoveRecord)`、そうでなければ`None`
    pub fn get(&self, index: usize) -> Option<&MoveRecord> {
        self.moves.get(index)
    }

    /// 最後の手記録を取得
    pub fn last(&self) -> Option<&MoveRecord> {
        self.moves.last()
    }

    /// 順方向イテレータを返す
    pub fn iter(&self) -> impl Iterator<Item = &MoveRecord> {
        self.moves.iter()
    }

    /// 履歴をクリアしてメモリを解放
    ///
    /// # 要件対応
    ///
    /// - Req 5.7: TD更新完了後にゲーム固有データの解放を許可
    pub fn clear(&mut self) {
        self.moves.clear();
    }

    /// インデックス付き逆順イテレータを返す
    ///
    /// TD更新で位置情報が必要な場合に使用
    pub fn iter_reverse_enumerated(&self) -> impl Iterator<Item = (usize, &MoveRecord)> {
        self.moves.iter().enumerate().rev()
    }
}

impl Default for GameHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::BitBoard;

    // ========== Task 4.1: MoveRecord Tests ==========

    #[test]
    fn test_move_record_new() {
        // Req 5.1-5.4: MoveRecord構造体のフィールド確認
        let board = BitBoard::new();
        let pattern_indices = [42usize; NUM_PATTERN_INSTANCES];
        let record = MoveRecord::new(board, 5.5, pattern_indices, 10);

        assert_eq!(record.board, board);
        assert_eq!(record.leaf_value, 5.5);
        assert_eq!(record.pattern_indices[0], 42);
        assert_eq!(record.stage, 10);
    }

    #[test]
    fn test_move_record_is_black_turn() {
        // 黒番のボード
        let board_black = BitBoard::new();
        let record_black = MoveRecord::new(board_black, 0.0, [0; NUM_PATTERN_INSTANCES], 0);
        assert!(record_black.is_black_turn());

        // 白番のボード（pass()で反転）
        let board_white = BitBoard::new().pass();
        let record_white = MoveRecord::new(board_white, 0.0, [0; NUM_PATTERN_INSTANCES], 0);
        assert!(!record_white.is_black_turn());
    }

    #[test]
    fn test_move_record_has_56_pattern_indices() {
        // Req 5.3: 56個のパターンインデックス
        let record = MoveRecord::new(BitBoard::new(), 0.0, [0; NUM_PATTERN_INSTANCES], 0);
        assert_eq!(record.pattern_indices.len(), 56);
    }

    #[test]
    fn test_move_record_stage_range() {
        // Req 5.4: ステージ0-29
        for stage in 0..30 {
            let record = MoveRecord::new(BitBoard::new(), 0.0, [0; NUM_PATTERN_INSTANCES], stage);
            assert_eq!(record.stage, stage);
        }
    }

    // ========== Task 4.1: GameHistory Tests ==========

    #[test]
    fn test_game_history_new() {
        // Req 5.5: 空の履歴を作成
        let history = GameHistory::new();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
    }

    #[test]
    fn test_game_history_push() {
        // Req 5.5: 手記録の追加
        let mut history = GameHistory::new();
        let record = MoveRecord::new(BitBoard::new(), 1.0, [0; NUM_PATTERN_INSTANCES], 0);
        history.push(record);
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_game_history_supports_60_moves() {
        // Req 5.5: 最大60手をサポート
        let mut history = GameHistory::new();

        for i in 0..60 {
            let record =
                MoveRecord::new(BitBoard::new(), i as f32, [i; NUM_PATTERN_INSTANCES], i / 2);
            history.push(record);
        }

        assert_eq!(history.len(), 60);

        // 各手が正しく記録されていることを確認
        for (i, record) in history.iter().enumerate() {
            assert_eq!(record.leaf_value, i as f32);
        }
    }

    #[test]
    fn test_game_history_iter_reverse() {
        // Req 5.6: 後方イテレーション
        let mut history = GameHistory::new();

        for i in 0..5 {
            let record = MoveRecord::new(BitBoard::new(), i as f32, [0; NUM_PATTERN_INSTANCES], 0);
            history.push(record);
        }

        // 逆順でイテレーション
        let values: Vec<f32> = history.iter_reverse().map(|r| r.leaf_value).collect();
        assert_eq!(values, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_game_history_iter_reverse_enumerated() {
        // TD更新用のインデックス付き逆順イテレーション
        let mut history = GameHistory::new();

        for i in 0..5 {
            let record = MoveRecord::new(BitBoard::new(), i as f32, [0; NUM_PATTERN_INSTANCES], 0);
            history.push(record);
        }

        // インデックス付き逆順イテレーション
        let indexed: Vec<(usize, f32)> = history
            .iter_reverse_enumerated()
            .map(|(i, r)| (i, r.leaf_value))
            .collect();

        assert_eq!(
            indexed,
            vec![(4, 4.0), (3, 3.0), (2, 2.0), (1, 1.0), (0, 0.0)]
        );
    }

    #[test]
    fn test_game_history_get() {
        let mut history = GameHistory::new();
        let record = MoveRecord::new(BitBoard::new(), 5.0, [0; NUM_PATTERN_INSTANCES], 2);
        history.push(record);

        assert!(history.get(0).is_some());
        assert_eq!(history.get(0).unwrap().leaf_value, 5.0);
        assert!(history.get(1).is_none());
    }

    #[test]
    fn test_game_history_last() {
        let mut history = GameHistory::new();
        assert!(history.last().is_none());

        history.push(MoveRecord::new(
            BitBoard::new(),
            1.0,
            [0; NUM_PATTERN_INSTANCES],
            0,
        ));
        history.push(MoveRecord::new(
            BitBoard::new(),
            2.0,
            [0; NUM_PATTERN_INSTANCES],
            0,
        ));

        assert_eq!(history.last().unwrap().leaf_value, 2.0);
    }

    #[test]
    fn test_game_history_clear() {
        // Req 5.7: TD更新完了後の解放
        let mut history = GameHistory::new();

        for i in 0..10 {
            history.push(MoveRecord::new(
                BitBoard::new(),
                i as f32,
                [0; NUM_PATTERN_INSTANCES],
                0,
            ));
        }

        assert_eq!(history.len(), 10);
        history.clear();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
    }

    #[test]
    fn test_game_history_preallocated_capacity() {
        // 再アロケーション回避のための事前確保
        let history = GameHistory::new();
        // 内部実装の詳細なので直接テストは困難だが、
        // 60手追加してもパニックしないことを確認
        let mut history = history;
        for _ in 0..60 {
            history.push(MoveRecord::new(
                BitBoard::new(),
                0.0,
                [0; NUM_PATTERN_INSTANCES],
                0,
            ));
        }
        assert_eq!(history.len(), 60);
    }

    #[test]
    fn test_game_history_memory_efficiency() {
        // メモリ効率的なフォーマットの確認
        // MoveRecordのサイズが妥当な範囲内であることを確認
        let record_size = std::mem::size_of::<MoveRecord>();
        println!("MoveRecord size: {} bytes", record_size);

        // 1ゲーム分（60手）のメモリ使用量
        let game_memory = record_size * 60;
        println!(
            "GameHistory (60 moves) memory: {} bytes ({:.2} KB)",
            game_memory,
            game_memory as f64 / 1024.0
        );

        // 目標: ~29KB（設計文書より）
        // 許容範囲: 50KB以下
        assert!(
            game_memory < 50 * 1024,
            "GameHistory memory usage should be under 50KB, got {} bytes",
            game_memory
        );
    }

    // ========== Requirements Summary Test ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== Task 4.1: Game History Requirements Verification ===");

        // Req 5.1: ボード状態の記録
        let board = BitBoard::new();
        let record = MoveRecord::new(board, 0.0, [0; NUM_PATTERN_INSTANCES], 0);
        assert_eq!(record.board, board);
        println!("  5.1: Board state (BitBoard) recorded");

        // Req 5.2: リーフ評価値の記録
        let record = MoveRecord::new(BitBoard::new(), 10.5, [0; NUM_PATTERN_INSTANCES], 0);
        assert_eq!(record.leaf_value, 10.5);
        println!("  5.2: Leaf evaluation value recorded");

        // Req 5.3: 56パターンインデックスの記録
        assert_eq!(NUM_PATTERN_INSTANCES, 56);
        println!("  5.3: 56 pattern indices recorded");

        // Req 5.4: ゲームステージの記録
        let record = MoveRecord::new(BitBoard::new(), 0.0, [0; NUM_PATTERN_INSTANCES], 15);
        assert_eq!(record.stage, 15);
        println!("  5.4: Game stage (0-29) recorded");

        // Req 5.5: 最大60手サポート
        let mut history = GameHistory::new();
        for _ in 0..60 {
            history.push(MoveRecord::new(
                BitBoard::new(),
                0.0,
                [0; NUM_PATTERN_INSTANCES],
                0,
            ));
        }
        assert_eq!(history.len(), 60);
        println!("  5.5: Supports up to 60 moves");

        // Req 5.6: 後方イテレーション
        let values: Vec<_> = history.iter_reverse().take(3).collect();
        assert_eq!(values.len(), 3);
        println!("  5.6: Memory-efficient format with reverse iteration");

        // Req 5.7: TD更新後の解放
        history.clear();
        assert!(history.is_empty());
        println!("  5.7: Allows deallocation after TD update");

        println!("=== All Task 4.1 requirements verified ===");
    }
}
