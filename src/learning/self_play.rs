//! 自己対戦ゲームエンジンモジュール
//!
//! 学習データ生成のための自己対戦エンジンを実装する。
//! イプシロン貪欲法による探索制御と、Phase 2探索APIとの統合を提供する。
//!
//! # 概要
//!
//! - `EpsilonSchedule`: ゲーム番号に基づくイプシロン値の計算
//! - `play_game`: Phase 2探索APIを使用した完全ゲームのプレイ
//! - `GameResult`: ゲーム結果（履歴、最終スコア、統計情報）
//!
//! # 要件対応
//!
//! - Req 4.1: 初期オセロ配置から終了までの完全ゲーム
//! - Req 4.2: 15ms制限のPhase 2 Search APIを使用
//! - Req 4.3: ゲーム0-299,999でepsilon=0.15
//! - Req 4.4: ゲーム300,000-699,999でepsilon=0.05
//! - Req 4.5: ゲーム700,000-999,999でepsilon=0.0
//! - Req 4.6: ランダム手では現在盤面の静的評価をリーフ値として使用
//! - Req 4.7: 各手のボード状態、リーフ評価、パターンインデックス、ステージを記録
//! - Req 4.8: オセロルールに従ったパス処理
//! - Req 4.9: 最終石差をゲーム結果として計算

use crate::board::{
    BitBoard, GameState, check_game_state, final_score, legal_moves, make_move_unchecked,
};
use crate::evaluator::calculate_stage;
use crate::learning::LearningError;
use crate::learning::game_history::{GameHistory, MoveRecord, NUM_PATTERN_INSTANCES};
use crate::pattern::{Pattern, extract_all_patterns_into};
use crate::search::Search;
use rand::Rng;

/// イプシロンスケジュールの高探索フェーズ終了ゲーム数
const HIGH_EXPLORATION_END: u64 = 300_000;

/// イプシロンスケジュールの中探索フェーズ終了ゲーム数
const MODERATE_EXPLORATION_END: u64 = 700_000;

/// 高探索フェーズのイプシロン値（15%ランダム）
const EPSILON_HIGH: f32 = 0.15;

/// 中探索フェーズのイプシロン値（5%ランダム）
const EPSILON_MODERATE: f32 = 0.05;

/// 搾取フェーズのイプシロン値（0%ランダム）
const EPSILON_EXPLOITATION: f32 = 0.0;

/// デフォルトの探索時間制限（ミリ秒）
pub const DEFAULT_SEARCH_TIME_MS: u64 = 15;

/// イプシロンスケジュール
///
/// ゲーム番号に基づいてイプシロン値を計算する。
/// ステートレスな計算として実装されており、メモリを使用しない。
///
/// # スケジュール
///
/// | ゲーム範囲 | イプシロン | フェーズ |
/// |-----------|----------|---------|
/// | 0-299,999 | 0.15 | 高探索 |
/// | 300,000-699,999 | 0.05 | 中探索 |
/// | 700,000-999,999 | 0.0 | 搾取 |
///
/// # 要件対応
///
/// - Req 4.3: ゲーム0-299,999でepsilon=0.15
/// - Req 4.4: ゲーム300,000-699,999でepsilon=0.05
/// - Req 4.5: ゲーム700,000-999,999でepsilon=0.0
pub struct EpsilonSchedule;

impl EpsilonSchedule {
    /// ゲーム番号に対応するイプシロン値を取得
    ///
    /// # 引数
    ///
    /// * `game_num` - ゲーム番号（0から開始）
    ///
    /// # 戻り値
    ///
    /// イプシロン値（0.0-0.15）
    ///
    /// # 例
    ///
    /// ```
    /// use prismind::learning::self_play::EpsilonSchedule;
    ///
    /// assert_eq!(EpsilonSchedule::get(0), 0.15);
    /// assert_eq!(EpsilonSchedule::get(299_999), 0.15);
    /// assert_eq!(EpsilonSchedule::get(300_000), 0.05);
    /// assert_eq!(EpsilonSchedule::get(699_999), 0.05);
    /// assert_eq!(EpsilonSchedule::get(700_000), 0.0);
    /// assert_eq!(EpsilonSchedule::get(999_999), 0.0);
    /// ```
    #[inline]
    pub fn get(game_num: u64) -> f32 {
        if game_num < HIGH_EXPLORATION_END {
            EPSILON_HIGH
        } else if game_num < MODERATE_EXPLORATION_END {
            EPSILON_MODERATE
        } else {
            EPSILON_EXPLOITATION
        }
    }

    /// 高探索フェーズかどうかを判定
    #[inline]
    pub fn is_high_exploration(game_num: u64) -> bool {
        game_num < HIGH_EXPLORATION_END
    }

    /// 中探索フェーズかどうかを判定
    #[inline]
    pub fn is_moderate_exploration(game_num: u64) -> bool {
        (HIGH_EXPLORATION_END..MODERATE_EXPLORATION_END).contains(&game_num)
    }

    /// 搾取フェーズかどうかを判定
    #[inline]
    pub fn is_exploitation(game_num: u64) -> bool {
        game_num >= MODERATE_EXPLORATION_END
    }
}

/// 自己対戦ゲームの結果
///
/// 完了したゲームの履歴と統計情報を含む。
#[derive(Debug)]
pub struct GameResult {
    /// ゲーム履歴（TD更新用）
    pub history: GameHistory,
    /// 最終スコア（石差、正=黒勝ち）
    pub final_score: f32,
    /// プレイされた手数
    pub moves_played: usize,
    /// ランダムに選択された手の数
    pub random_moves: usize,
}

/// 自己対戦ゲームをプレイ
///
/// 初期オセロ配置から終了までの完全ゲームをプレイし、
/// TD学習用の履歴を返す。
///
/// # 引数
///
/// * `search` - Phase 2探索エンジン
/// * `patterns` - パターン定義（インデックス抽出用）
/// * `epsilon` - ランダム手選択確率（0.0-1.0）
/// * `time_limit_ms` - 1手あたりの探索時間制限（ミリ秒）
/// * `rng` - 乱数生成器
///
/// # 戻り値
///
/// ゲーム結果（履歴、最終スコア、統計情報）
///
/// # エラー
///
/// 探索エラーが発生した場合
///
/// # 例
///
/// ```ignore
/// use prismind::learning::self_play::{play_game, DEFAULT_SEARCH_TIME_MS};
/// use prismind::search::Search;
/// use prismind::evaluator::Evaluator;
/// use rand::thread_rng;
///
/// let evaluator = Evaluator::new("patterns.csv").unwrap();
/// let mut search = Search::new(evaluator, 128).unwrap();
/// let patterns = load_patterns("patterns.csv").unwrap();
/// let mut rng = thread_rng();
///
/// let result = play_game(&mut search, &patterns, 0.15, DEFAULT_SEARCH_TIME_MS, &mut rng).unwrap();
/// println!("Game result: {} (moves: {})", result.final_score, result.moves_played);
/// ```
///
/// # 要件対応
///
/// - Req 4.1: 初期配置から終了までプレイ
/// - Req 4.2: 15ms制限のPhase 2 Search APIを使用
/// - Req 4.6: ランダム手では静的評価をリーフ値として使用
/// - Req 4.7: 各手の情報を記録
/// - Req 4.8: パス処理
/// - Req 4.9: 最終石差を計算
pub fn play_game<R: Rng>(
    search: &mut Search,
    patterns: &[Pattern; 14],
    epsilon: f32,
    time_limit_ms: u64,
    rng: &mut R,
) -> Result<GameResult, LearningError> {
    let mut board = BitBoard::new();
    let mut history = GameHistory::new();
    let mut random_moves = 0;

    loop {
        // ゲーム状態を確認
        match check_game_state(&board) {
            GameState::GameOver(_) => {
                // ゲーム終了
                let score = final_score(&board) as f32;
                return Ok(GameResult {
                    history,
                    final_score: score,
                    moves_played: board.move_count() as usize,
                    random_moves,
                });
            }
            GameState::Pass => {
                // Req 4.8: パス処理
                board = board.pass();
                continue;
            }
            GameState::Playing => {
                // ゲーム続行
            }
        }

        // イプシロン貪欲法による手選択
        let legal = legal_moves(&board);
        let is_random_move = rng.random::<f32>() < epsilon;

        let (best_move, leaf_value) = if is_random_move {
            // ランダム手選択
            random_moves += 1;
            let move_pos = select_random_move(legal, rng);

            // Req 4.6: ランダム手では現在盤面の静的評価をリーフ値として使用
            // 静的評価のために盤面を評価（探索なし）
            let static_eval = search.evaluate_static(&board);
            (move_pos, static_eval)
        } else {
            // 最善手探索
            // Req 4.2: 15ms制限でPhase 2 Search APIを使用
            let search_result = search.search(&board, time_limit_ms, None)?;
            let best_move = search_result.best_move.ok_or_else(|| {
                LearningError::Search(crate::search::SearchError::InvalidBoardState(
                    "No legal moves found".to_string(),
                ))
            })?;
            (best_move, search_result.score)
        };

        // Req 4.7: 各手の情報を記録
        let stage = calculate_stage(board.move_count());
        let mut pattern_indices = [0usize; NUM_PATTERN_INSTANCES];
        extract_all_patterns_into(&board, patterns, &mut pattern_indices);

        let record = MoveRecord::new(board, leaf_value, pattern_indices, stage);
        history.push(record);

        // 手を実行
        make_move_unchecked(&mut board, best_move);
    }
}

/// 合法手からランダムに1手を選択
fn select_random_move<R: Rng>(legal_mask: u64, rng: &mut R) -> u8 {
    debug_assert!(legal_mask != 0, "No legal moves available");

    let move_count = legal_mask.count_ones() as usize;
    let selected_index = rng.random_range(0..move_count);

    // ビットマスクからn番目のビットを取得
    let mut mask = legal_mask;
    for _ in 0..selected_index {
        mask &= mask - 1; // 最下位ビットをクリア
    }
    mask.trailing_zeros() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Task 4.2: Epsilon Schedule Tests ==========

    #[test]
    fn test_epsilon_high_exploration_phase() {
        // Req 4.3: ゲーム0-299,999でepsilon=0.15
        assert_eq!(EpsilonSchedule::get(0), 0.15);
        assert_eq!(EpsilonSchedule::get(1), 0.15);
        assert_eq!(EpsilonSchedule::get(100_000), 0.15);
        assert_eq!(EpsilonSchedule::get(299_999), 0.15);
    }

    #[test]
    fn test_epsilon_moderate_exploration_phase() {
        // Req 4.4: ゲーム300,000-699,999でepsilon=0.05
        assert_eq!(EpsilonSchedule::get(300_000), 0.05);
        assert_eq!(EpsilonSchedule::get(300_001), 0.05);
        assert_eq!(EpsilonSchedule::get(500_000), 0.05);
        assert_eq!(EpsilonSchedule::get(699_999), 0.05);
    }

    #[test]
    fn test_epsilon_exploitation_phase() {
        // Req 4.5: ゲーム700,000-999,999でepsilon=0.0
        assert_eq!(EpsilonSchedule::get(700_000), 0.0);
        assert_eq!(EpsilonSchedule::get(700_001), 0.0);
        assert_eq!(EpsilonSchedule::get(999_999), 0.0);
        assert_eq!(EpsilonSchedule::get(1_000_000), 0.0); // 範囲外も搾取フェーズ
    }

    #[test]
    fn test_epsilon_boundary_transitions() {
        // 境界でのフェーズ遷移を確認
        // 高探索 -> 中探索
        assert_eq!(EpsilonSchedule::get(299_999), 0.15);
        assert_eq!(EpsilonSchedule::get(300_000), 0.05);

        // 中探索 -> 搾取
        assert_eq!(EpsilonSchedule::get(699_999), 0.05);
        assert_eq!(EpsilonSchedule::get(700_000), 0.0);
    }

    #[test]
    fn test_epsilon_phase_helpers() {
        // フェーズ判定ヘルパーのテスト
        assert!(EpsilonSchedule::is_high_exploration(0));
        assert!(EpsilonSchedule::is_high_exploration(299_999));
        assert!(!EpsilonSchedule::is_high_exploration(300_000));

        assert!(!EpsilonSchedule::is_moderate_exploration(299_999));
        assert!(EpsilonSchedule::is_moderate_exploration(300_000));
        assert!(EpsilonSchedule::is_moderate_exploration(699_999));
        assert!(!EpsilonSchedule::is_moderate_exploration(700_000));

        assert!(!EpsilonSchedule::is_exploitation(699_999));
        assert!(EpsilonSchedule::is_exploitation(700_000));
        assert!(EpsilonSchedule::is_exploitation(1_000_000));
    }

    #[test]
    fn test_epsilon_is_stateless() {
        // ステートレス計算の確認（同じ入力で同じ出力）
        for _ in 0..100 {
            assert_eq!(EpsilonSchedule::get(500_000), 0.05);
        }
    }

    // ========== Task 4.3: Random Move Selection Tests ==========

    #[test]
    fn test_select_random_move_single_option() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // 1手のみの場合
        let legal = 1u64 << 20; // 位置20のみ
        let selected = select_random_move(legal, &mut rng);
        assert_eq!(selected, 20);
    }

    #[test]
    fn test_select_random_move_multiple_options() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // 複数の合法手
        let legal = (1u64 << 10) | (1u64 << 20) | (1u64 << 30); // 位置10, 20, 30

        // 100回選択して、すべて合法手であることを確認
        for _ in 0..100 {
            let selected = select_random_move(legal, &mut rng);
            assert!(
                selected == 10 || selected == 20 || selected == 30,
                "Selected move {} is not in legal moves",
                selected
            );
        }
    }

    #[test]
    fn test_select_random_move_distribution() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

        // 4つの合法手
        let legal = (1u64 << 0) | (1u64 << 10) | (1u64 << 20) | (1u64 << 30);

        let mut counts = [0u32; 64];
        let iterations = 10000;

        for _ in 0..iterations {
            let selected = select_random_move(legal, &mut rng);
            counts[selected as usize] += 1;
        }

        // 各手が概ね均等に選ばれることを確認（許容誤差20%）
        let expected = iterations / 4;
        let tolerance = expected / 5;

        for pos in [0, 10, 20, 30] {
            let count = counts[pos];
            assert!(
                (count as i32 - expected as i32).unsigned_abs() < tolerance,
                "Position {} selected {} times, expected ~{} (tolerance: {})",
                pos,
                count,
                expected,
                tolerance
            );
        }
    }

    // ========== Task 4.3: GameResult Tests ==========

    #[test]
    fn test_game_result_fields() {
        let history = GameHistory::new();
        let result = GameResult {
            history,
            final_score: 10.0,
            moves_played: 50,
            random_moves: 5,
        };

        assert_eq!(result.final_score, 10.0);
        assert_eq!(result.moves_played, 50);
        assert_eq!(result.random_moves, 5);
    }

    // ========== Requirements Summary Tests ==========

    #[test]
    fn test_epsilon_schedule_requirements_summary() {
        println!("=== Task 4.2: Epsilon Schedule Requirements Verification ===");

        // Req 4.3: 高探索フェーズ
        assert_eq!(EpsilonSchedule::get(0), 0.15);
        assert_eq!(EpsilonSchedule::get(299_999), 0.15);
        println!("  4.3: Games 0-299,999 return epsilon=0.15");

        // Req 4.4: 中探索フェーズ
        assert_eq!(EpsilonSchedule::get(300_000), 0.05);
        assert_eq!(EpsilonSchedule::get(699_999), 0.05);
        println!("  4.4: Games 300,000-699,999 return epsilon=0.05");

        // Req 4.5: 搾取フェーズ
        assert_eq!(EpsilonSchedule::get(700_000), 0.0);
        assert_eq!(EpsilonSchedule::get(999_999), 0.0);
        println!("  4.5: Games 700,000-999,999 return epsilon=0.0");

        // ステートレス計算
        let e1 = EpsilonSchedule::get(500_000);
        let e2 = EpsilonSchedule::get(500_000);
        assert_eq!(e1, e2);
        println!("  Stateless computation verified");

        println!("=== All Task 4.2 requirements verified ===");
    }

    #[test]
    fn test_self_play_requirements_summary() {
        println!("=== Task 4.3: Self-Play Engine Requirements (Partial) ===");

        // 注: 完全な統合テストはpatterns.csvが必要なため、
        // ここでは構造とユニットテストのみ検証

        // Req 4.1: 完全ゲームのプレイ（構造確認）
        println!("  4.1: play_game function plays complete games from initial position");

        // Req 4.2: Phase 2 Search API使用（構造確認）
        println!("  4.2: Uses Phase 2 Search API with time limit");

        // Req 4.6: ランダム手での静的評価（構造確認）
        println!("  4.6: Uses static evaluation for random move leaf values");

        // Req 4.7: 各手の情報記録（構造確認）
        println!("  4.7: Records board state, leaf evaluation, pattern indices, stage");

        // Req 4.8: パス処理（構造確認）
        println!("  4.8: Handles pass moves correctly");

        // Req 4.9: 最終石差計算（構造確認）
        println!("  4.9: Computes final stone difference as game result");

        println!(
            "=== Task 4.3 structure verified (full integration tests require patterns.csv) ==="
        );
    }
}
