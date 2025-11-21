//! Prismind - オセロAIエンジン
//!
//! パターンベース評価関数を用いた高性能オセロ(リバーシ)AIエンジン。
//!
//! # モジュール構成
//!
//! - [`board`] - BitBoard表現による盤面管理と着手生成
//! - [`pattern`] - パターン定義の読み込みと管理
//! - [`evaluator`] - 評価関数とスコア変換
//! - `arm64` - ARM64専用SIMD最適化（条件付きコンパイル）
//!
//! # 使用例
//!
//! ```no_run
//! use prismind::board::{BitBoard, legal_moves, make_move, check_game_state, GameState};
//! use prismind::evaluator::Evaluator;
//!
//! // 盤面の初期化
//! let mut board = BitBoard::new();
//!
//! // 評価関数の初期化
//! let evaluator = Evaluator::new("patterns.csv").unwrap();
//!
//! // ゲームループ
//! loop {
//!     // ゲーム状態を確認
//!     let game_state = check_game_state(&board);
//!     match game_state {
//!         GameState::Playing => {
//!             // 合法手を取得
//!             let moves = legal_moves(&board);
//!             if moves == 0 {
//!                 break;
//!             }
//!
//!             // 盤面を評価
//!             let eval = evaluator.evaluate(&board);
//!             println!("評価値: {}", eval);
//!
//!             // 最初の合法手を実行
//!             let pos = moves.trailing_zeros() as u8;
//!             make_move(&mut board, pos).unwrap();
//!         }
//!         GameState::Pass => {
//!             board = board.flip();
//!         }
//!         GameState::GameOver(score) => {
//!             println!("ゲーム終了: スコア = {}", score);
//!             break;
//!         }
//!     }
//! }
//! ```
//!
//! # 主要な機能
//!
//! ## BitBoard表現
//!
//! 64ビット整数2つで盤面を表現し、ビット演算による高速な着手生成を実現。
//! - 黒石: `u64` ビットマスク
//! - 白石とメタデータ: `u64` (下位8ビットがメタデータ)
//!
//! ## パターンベース評価
//!
//! 14種類のパターンを4方向(0°, 90°, 180°, 270°)に抽出し、
//! 合計56個のパターンインスタンスから評価値を計算。
//!
//! ## ARM64最適化
//!
//! - NEON SIMD命令による評価値変換の並列化
//! - プリフェッチによるキャッシュミス削減
//!
//! # パフォーマンス目標
//!
//! - 合法手生成: 0.5μs以内
//! - 着手実行: 0.3μs以内
//! - 評価関数: 35μs以内 (ARM64)
//!
//! # メモリ使用量
//!
//! - 評価テーブル: 約70-80MB (30ステージ × 14パターン)
//! - BitBoard: 16バイト (スタック配置)

// 公開モジュール
pub mod board;
pub mod evaluator;
pub mod pattern;
pub mod search;

// ARM64専用最適化モジュール（条件付きコンパイル）
#[cfg(target_arch = "aarch64")]
pub mod arm64;

// 再エクスポート: よく使用される型と関数をクレートルートから直接アクセス可能にする
pub use board::{
    BitBoard, Color, DIRECTIONS, GameError, GameState, UndoInfo, check_game_state, display,
    final_score, legal_moves, make_move, shift, undo_move,
};
pub use evaluator::{EvaluationTable, Evaluator, calculate_stage, score_to_u16, u16_to_score};
pub use pattern::{Pattern, PatternError, coord_to_bit, extract_index, load_patterns};
pub use search::{Bound, SearchError, SearchResult, TTEntry, TranspositionTable, ZobristTable};

// ARM64最適化のre-export（条件付き）
#[cfg(target_arch = "aarch64")]
pub use arm64::{prefetch_arm64, u16_to_score_simd_arm64};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitboard_size() {
        // BitBoardは16バイト以内であること
        assert!(std::mem::size_of::<board::BitBoard>() <= 16);
    }

    #[test]
    fn test_color_opposite() {
        use board::Color;

        // 白の反対は黒
        assert_eq!(Color::White.opposite(), Color::Black);
        // 黒の反対は白
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    #[test]
    fn test_bitboard_traits() {
        use board::{BitBoard, Color};

        // Clone, Copy, Debug, PartialEq, Eqトレイトが実装されていること
        let board1 = BitBoard::new();
        let board2 = board1; // Copy
        let board3 = board1;

        assert_eq!(board1, board2); // PartialEq
        assert_eq!(board2, board3); // Eq

        // 初期手番は黒
        assert_eq!(board1.turn(), Color::Black);

        // Debug
        let debug_str = format!("{:?}", board1);
        assert!(!debug_str.is_empty());
    }
}
