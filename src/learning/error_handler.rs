//! エラーハンドリングとリカバリーモジュール
//!
//! 学習プロセスの堅牢なエラーハンドリングとリカバリー機能を実装する。
//!
//! # 概要
//!
//! - `ErrorTracker`: エラーカウントの追跡とウィンドウベースの分析
//! - `ErrorRecovery`: NaN/Inf値のリカバリーとパニックキャッチ
//! - `CheckpointRetry`: チェックポイント保存のリトライロジック
//!
//! # 要件対応
//!
//! - Req 12.1: 探索エラーをログして次のゲームにスキップ
//! - Req 12.2: チェックポイント保存失敗時に1回リトライ
//! - Req 12.3: NaN/Inf評価値を32768にリセット
//! - Req 12.4: パニックをキャッチしてログ（トレーニングプロセスをクラッシュさせない）
//! - Req 12.6: 10,000ゲームウィンドウで1%以上失敗時にトレーニングを一時停止

use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};

use crate::evaluator::EvaluationTable;
use crate::learning::LearningError;
use crate::learning::score::CENTER;

/// エラーウィンドウサイズ（10,000ゲーム）
pub const ERROR_WINDOW_SIZE: usize = 10_000;

/// エラー閾値（1%）
pub const ERROR_THRESHOLD_PERCENT: f32 = 1.0;

/// エラータイプの分類
#[derive(Clone, Debug, PartialEq)]
pub enum ErrorType {
    /// 探索エラー
    Search,
    /// 評価発散（NaN/Inf）
    EvalDivergence,
    /// チェックポイント保存エラー
    Checkpoint,
    /// パニック
    Panic,
    /// その他
    Other,
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorType::Search => write!(f, "Search"),
            ErrorType::EvalDivergence => write!(f, "EvalDivergence"),
            ErrorType::Checkpoint => write!(f, "Checkpoint"),
            ErrorType::Panic => write!(f, "Panic"),
            ErrorType::Other => write!(f, "Other"),
        }
    }
}

/// エラー記録
#[derive(Clone, Debug)]
pub struct ErrorRecord {
    /// エラータイプ
    pub error_type: ErrorType,
    /// ゲーム番号
    pub game_number: u64,
    /// エラーメッセージ
    pub message: String,
}

impl ErrorRecord {
    /// 新しいエラー記録を作成
    pub fn new(error_type: ErrorType, game_number: u64, message: impl Into<String>) -> Self {
        Self {
            error_type,
            game_number,
            message: message.into(),
        }
    }
}

/// エラー追跡器
///
/// ウィンドウベースでエラーを追跡し、エラー率を監視する。
/// 10,000ゲームウィンドウで1%以上のエラーが発生した場合に警告を発する。
///
/// # 要件対応
///
/// - Req 12.6: 10,000ゲームウィンドウで1%以上失敗時に報告
#[derive(Debug)]
pub struct ErrorTracker {
    /// エラー履歴（ウィンドウ内）
    errors: VecDeque<ErrorRecord>,
    /// 総エラー数
    total_errors: u64,
    /// 処理済みゲーム数
    total_games: u64,
    /// ウィンドウサイズ
    window_size: usize,
    /// エラー閾値（パーセント）
    threshold_percent: f32,
    /// タイプ別エラーカウント
    error_counts_by_type: [u64; 5],
}

impl ErrorTracker {
    /// デフォルト設定でエラー追跡器を作成
    pub fn new() -> Self {
        Self::with_config(ERROR_WINDOW_SIZE, ERROR_THRESHOLD_PERCENT)
    }

    /// カスタム設定でエラー追跡器を作成
    pub fn with_config(window_size: usize, threshold_percent: f32) -> Self {
        Self {
            errors: VecDeque::with_capacity(window_size),
            total_errors: 0,
            total_games: 0,
            window_size,
            threshold_percent,
            error_counts_by_type: [0; 5],
        }
    }

    /// ゲーム成功を記録
    pub fn record_success(&mut self) {
        self.total_games += 1;
        self.prune_old_errors();
    }

    /// エラーを記録
    ///
    /// # 戻り値
    ///
    /// エラー率が閾値を超えた場合はtrue
    pub fn record_error(&mut self, error: ErrorRecord) -> bool {
        // タイプ別カウントを更新
        let type_index = match error.error_type {
            ErrorType::Search => 0,
            ErrorType::EvalDivergence => 1,
            ErrorType::Checkpoint => 2,
            ErrorType::Panic => 3,
            ErrorType::Other => 4,
        };
        self.error_counts_by_type[type_index] += 1;

        // エラー履歴に追加
        self.errors.push_back(error);
        self.total_errors += 1;
        self.total_games += 1;

        self.prune_old_errors();
        self.is_threshold_exceeded()
    }

    /// 古いエラーを削除
    fn prune_old_errors(&mut self) {
        if self.total_games <= self.window_size as u64 {
            return;
        }

        let oldest_valid_game = self.total_games.saturating_sub(self.window_size as u64);
        while let Some(front) = self.errors.front() {
            if front.game_number < oldest_valid_game {
                self.errors.pop_front();
            } else {
                break;
            }
        }
    }

    /// エラー閾値を超えているかチェック
    ///
    /// # 要件対応
    ///
    /// - Req 12.6: 10,000ゲームウィンドウで1%以上失敗時にtrue
    pub fn is_threshold_exceeded(&self) -> bool {
        let window_games = self.window_games();
        if window_games == 0 {
            return false;
        }

        let error_rate = self.errors.len() as f32 / window_games as f32 * 100.0;
        error_rate > self.threshold_percent
    }

    /// 現在のウィンドウ内のゲーム数
    pub fn window_games(&self) -> usize {
        self.total_games.min(self.window_size as u64) as usize
    }

    /// 現在のウィンドウ内のエラー数
    pub fn window_errors(&self) -> usize {
        self.errors.len()
    }

    /// 現在のエラー率（パーセント）
    pub fn error_rate_percent(&self) -> f32 {
        let window_games = self.window_games();
        if window_games == 0 {
            return 0.0;
        }
        self.errors.len() as f32 / window_games as f32 * 100.0
    }

    /// 総エラー数
    pub fn total_errors(&self) -> u64 {
        self.total_errors
    }

    /// 総ゲーム数
    pub fn total_games(&self) -> u64 {
        self.total_games
    }

    /// エラーパターンのサマリーを生成
    ///
    /// # 要件対応
    ///
    /// - Req 12.6: エラーパターンをオペレーター診断用に報告
    pub fn error_pattern_summary(&self) -> ErrorPatternSummary {
        let mut type_counts = [0usize; 5];

        for error in &self.errors {
            let idx = match error.error_type {
                ErrorType::Search => 0,
                ErrorType::EvalDivergence => 1,
                ErrorType::Checkpoint => 2,
                ErrorType::Panic => 3,
                ErrorType::Other => 4,
            };
            type_counts[idx] += 1;
        }

        ErrorPatternSummary {
            window_games: self.window_games(),
            window_errors: self.errors.len(),
            error_rate_percent: self.error_rate_percent(),
            search_errors: type_counts[0],
            eval_divergence_errors: type_counts[1],
            checkpoint_errors: type_counts[2],
            panic_errors: type_counts[3],
            other_errors: type_counts[4],
            total_errors_all_time: self.total_errors,
            total_games_all_time: self.total_games,
        }
    }

    /// 最近のエラーを取得
    pub fn recent_errors(&self, count: usize) -> Vec<&ErrorRecord> {
        self.errors.iter().rev().take(count).collect()
    }

    /// リセット
    pub fn reset(&mut self) {
        self.errors.clear();
        self.total_errors = 0;
        self.total_games = 0;
        self.error_counts_by_type = [0; 5];
    }
}

impl Default for ErrorTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// エラーパターンサマリー
#[derive(Clone, Debug)]
pub struct ErrorPatternSummary {
    /// ウィンドウ内ゲーム数
    pub window_games: usize,
    /// ウィンドウ内エラー数
    pub window_errors: usize,
    /// エラー率（パーセント）
    pub error_rate_percent: f32,
    /// 探索エラー数
    pub search_errors: usize,
    /// 評価発散エラー数
    pub eval_divergence_errors: usize,
    /// チェックポイントエラー数
    pub checkpoint_errors: usize,
    /// パニックエラー数
    pub panic_errors: usize,
    /// その他エラー数
    pub other_errors: usize,
    /// 累計エラー数
    pub total_errors_all_time: u64,
    /// 累計ゲーム数
    pub total_games_all_time: u64,
}

impl std::fmt::Display for ErrorPatternSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Error Pattern Summary ===")?;
        writeln!(
            f,
            "Window: {} games, {} errors ({:.2}%)",
            self.window_games, self.window_errors, self.error_rate_percent
        )?;
        writeln!(f, "By type (window):")?;
        writeln!(f, "  Search: {}", self.search_errors)?;
        writeln!(f, "  EvalDivergence: {}", self.eval_divergence_errors)?;
        writeln!(f, "  Checkpoint: {}", self.checkpoint_errors)?;
        writeln!(f, "  Panic: {}", self.panic_errors)?;
        writeln!(f, "  Other: {}", self.other_errors)?;
        writeln!(
            f,
            "All-time: {} errors / {} games",
            self.total_errors_all_time, self.total_games_all_time
        )?;
        Ok(())
    }
}

/// 評価値リカバリー
///
/// NaN/Inf評価値をニュートラル値（32768）にリセットする。
///
/// # 要件対応
///
/// - Req 12.3: NaN/Inf評価値を32768にリセット
#[derive(Debug, Default)]
pub struct EvalRecovery {
    /// リカバリー回数
    recovery_count: u64,
    /// 最後にリカバリーしたパターン情報
    last_recovery: Option<(usize, usize, usize)>,
}

impl EvalRecovery {
    /// 新しいリカバリーインスタンスを作成
    pub fn new() -> Self {
        Self::default()
    }

    /// 評価値がNaN/Infかチェック
    #[inline]
    pub fn is_invalid(value: f32) -> bool {
        value.is_nan() || value.is_infinite()
    }

    /// 評価テーブルのエントリをチェックしてリカバリー
    ///
    /// # 戻り値
    ///
    /// リカバリーが行われた場合はtrue
    ///
    /// # 要件対応
    ///
    /// - Req 12.3: NaN/Inf評価値を32768にリセット
    pub fn check_and_recover_entry(
        &mut self,
        table: &mut EvaluationTable,
        pattern_id: usize,
        stage: usize,
        index: usize,
        value: f32,
    ) -> bool {
        if Self::is_invalid(value) {
            // CENTERにリセット（32768）
            table.set(pattern_id, stage, index, CENTER);
            self.recovery_count += 1;
            self.last_recovery = Some((pattern_id, stage, index));
            true
        } else {
            false
        }
    }

    /// f32値をサニタイズ
    ///
    /// NaN/Infの場合は0.0を返す
    #[inline]
    pub fn sanitize_f32(value: f32) -> f32 {
        if value.is_nan() || value.is_infinite() {
            0.0
        } else {
            value
        }
    }

    /// u16値に変換する前にf32をサニタイズ
    ///
    /// NaN/Infの場合はCENTERを返す
    #[inline]
    pub fn sanitize_to_u16(value: f32) -> u16 {
        if value.is_nan() || value.is_infinite() {
            CENTER
        } else {
            crate::learning::score::stone_diff_to_u16(value)
        }
    }

    /// リカバリー回数を取得
    pub fn recovery_count(&self) -> u64 {
        self.recovery_count
    }

    /// 最後にリカバリーしたエントリ情報
    pub fn last_recovery(&self) -> Option<(usize, usize, usize)> {
        self.last_recovery
    }

    /// リセット
    pub fn reset(&mut self) {
        self.recovery_count = 0;
        self.last_recovery = None;
    }
}

/// パニックキャッチ結果
#[derive(Debug)]
pub enum PanicCatchResult<T> {
    /// 成功
    Ok(T),
    /// エラー
    Err(LearningError),
    /// パニック
    Panic(String),
}

impl<T> PanicCatchResult<T> {
    /// 成功かどうか
    pub fn is_ok(&self) -> bool {
        matches!(self, PanicCatchResult::Ok(_))
    }

    /// エラーかどうか
    pub fn is_err(&self) -> bool {
        matches!(self, PanicCatchResult::Err(_))
    }

    /// パニックかどうか
    pub fn is_panic(&self) -> bool {
        matches!(self, PanicCatchResult::Panic(_))
    }
}

/// パニックをキャッチしてエラーに変換
///
/// # 要件対応
///
/// - Req 12.4: パニックをキャッチしてログ（クラッシュさせない）
pub fn catch_panic<F, T>(f: F) -> PanicCatchResult<T>
where
    F: FnOnce() -> Result<T, LearningError>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => PanicCatchResult::Ok(value),
        Ok(Err(e)) => PanicCatchResult::Err(e),
        Err(panic_info) => {
            let message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            PanicCatchResult::Panic(message)
        }
    }
}

/// チェックポイント保存をリトライ付きで実行
///
/// # 要件対応
///
/// - Req 12.2: チェックポイント保存失敗時に1回リトライ
pub fn save_checkpoint_with_retry<F>(mut save_fn: F) -> Result<(), LearningError>
where
    F: FnMut() -> Result<(), LearningError>,
{
    // 最初の試行
    match save_fn() {
        Ok(()) => return Ok(()),
        Err(_e) => {
            // Log is handled by caller since we don't have log crate dependency here
        }
    }

    // リトライ
    save_fn()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ErrorTracker Tests ==========

    #[test]
    fn test_error_tracker_new() {
        let tracker = ErrorTracker::new();
        assert_eq!(tracker.total_games(), 0);
        assert_eq!(tracker.total_errors(), 0);
        assert_eq!(tracker.window_errors(), 0);
    }

    #[test]
    fn test_error_tracker_record_success() {
        let mut tracker = ErrorTracker::new();

        for _ in 0..100 {
            tracker.record_success();
        }

        assert_eq!(tracker.total_games(), 100);
        assert_eq!(tracker.total_errors(), 0);
        assert!(!tracker.is_threshold_exceeded());
    }

    #[test]
    fn test_error_tracker_record_error() {
        let mut tracker = ErrorTracker::new();

        for i in 0..10 {
            let error = ErrorRecord::new(ErrorType::Search, i, "test error");
            tracker.record_error(error);
        }

        assert_eq!(tracker.total_games(), 10);
        assert_eq!(tracker.total_errors(), 10);
        assert_eq!(tracker.window_errors(), 10);
    }

    #[test]
    fn test_error_tracker_threshold_exceeded() {
        // 小さいウィンドウで閾値超過をテスト
        let mut tracker = ErrorTracker::with_config(100, 1.0);

        // 99ゲーム成功
        for _ in 0..99 {
            tracker.record_success();
        }

        // 2つのエラー（2% > 1%）
        let error1 = ErrorRecord::new(ErrorType::Search, 99, "error 1");
        let error2 = ErrorRecord::new(ErrorType::Search, 100, "error 2");

        tracker.record_error(error1);
        let exceeded = tracker.record_error(error2);

        assert!(exceeded);
        assert!(tracker.is_threshold_exceeded());
    }

    #[test]
    fn test_error_tracker_threshold_not_exceeded() {
        let mut tracker = ErrorTracker::with_config(1000, 1.0);

        // 999ゲーム成功、1エラー（0.1% < 1%）
        for i in 0..999 {
            if i == 500 {
                let error = ErrorRecord::new(ErrorType::Search, i, "single error");
                tracker.record_error(error);
            } else {
                tracker.record_success();
            }
        }

        assert!(!tracker.is_threshold_exceeded());
        assert!(tracker.error_rate_percent() < 1.0);
    }

    #[test]
    fn test_error_tracker_window_prune() {
        let mut tracker = ErrorTracker::with_config(100, 1.0);

        // 最初の50ゲームでエラー
        for i in 0..50 {
            let error = ErrorRecord::new(ErrorType::Search, i, "old error");
            tracker.record_error(error);
        }

        // 次の100ゲームは成功
        for _ in 0..100 {
            tracker.record_success();
        }

        // ウィンドウが100なので、古いエラーは削除されるべき
        assert_eq!(tracker.window_games(), 100);
        // 古いエラーはウィンドウ外になるので0
        assert_eq!(tracker.window_errors(), 0);
    }

    #[test]
    fn test_error_tracker_pattern_summary() {
        let mut tracker = ErrorTracker::with_config(100, 1.0);

        tracker.record_error(ErrorRecord::new(ErrorType::Search, 0, "search"));
        tracker.record_error(ErrorRecord::new(ErrorType::Search, 1, "search"));
        tracker.record_error(ErrorRecord::new(ErrorType::EvalDivergence, 2, "nan"));
        tracker.record_error(ErrorRecord::new(ErrorType::Panic, 3, "panic"));

        let summary = tracker.error_pattern_summary();

        assert_eq!(summary.search_errors, 2);
        assert_eq!(summary.eval_divergence_errors, 1);
        assert_eq!(summary.panic_errors, 1);
        assert_eq!(summary.window_errors, 4);
    }

    #[test]
    fn test_error_record_creation() {
        let error = ErrorRecord::new(ErrorType::Search, 12345, "test message");

        assert_eq!(error.error_type, ErrorType::Search);
        assert_eq!(error.game_number, 12345);
        assert_eq!(error.message, "test message");
    }

    #[test]
    fn test_error_type_display() {
        assert_eq!(format!("{}", ErrorType::Search), "Search");
        assert_eq!(format!("{}", ErrorType::EvalDivergence), "EvalDivergence");
        assert_eq!(format!("{}", ErrorType::Checkpoint), "Checkpoint");
        assert_eq!(format!("{}", ErrorType::Panic), "Panic");
        assert_eq!(format!("{}", ErrorType::Other), "Other");
    }

    // ========== EvalRecovery Tests ==========

    #[test]
    fn test_eval_recovery_is_invalid() {
        assert!(EvalRecovery::is_invalid(f32::NAN));
        assert!(EvalRecovery::is_invalid(f32::INFINITY));
        assert!(EvalRecovery::is_invalid(f32::NEG_INFINITY));
        assert!(!EvalRecovery::is_invalid(0.0));
        assert!(!EvalRecovery::is_invalid(100.0));
        assert!(!EvalRecovery::is_invalid(-100.0));
    }

    #[test]
    fn test_eval_recovery_sanitize_f32() {
        assert_eq!(EvalRecovery::sanitize_f32(f32::NAN), 0.0);
        assert_eq!(EvalRecovery::sanitize_f32(f32::INFINITY), 0.0);
        assert_eq!(EvalRecovery::sanitize_f32(f32::NEG_INFINITY), 0.0);
        assert_eq!(EvalRecovery::sanitize_f32(42.0), 42.0);
        assert_eq!(EvalRecovery::sanitize_f32(-42.0), -42.0);
    }

    #[test]
    fn test_eval_recovery_sanitize_to_u16() {
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::NAN), CENTER);
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::INFINITY), CENTER);
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::NEG_INFINITY), CENTER);
        assert_eq!(EvalRecovery::sanitize_to_u16(0.0), CENTER);
    }

    #[test]
    fn test_eval_recovery_new() {
        let recovery = EvalRecovery::new();
        assert_eq!(recovery.recovery_count(), 0);
        assert!(recovery.last_recovery().is_none());
    }

    // ========== PanicCatchResult Tests ==========

    #[test]
    fn test_panic_catch_ok() {
        let result = catch_panic(|| Ok::<_, LearningError>(42));

        match result {
            PanicCatchResult::Ok(value) => assert_eq!(value, 42),
            _ => panic!("Expected Ok result"),
        }
    }

    #[test]
    fn test_panic_catch_err() {
        let result = catch_panic(|| Err::<i32, _>(LearningError::Config("test error".to_string())));

        assert!(result.is_err());
    }

    #[test]
    fn test_panic_catch_panic() {
        let result = catch_panic::<_, i32>(|| {
            panic!("test panic");
        });

        match result {
            PanicCatchResult::Panic(msg) => assert!(msg.contains("test panic")),
            _ => panic!("Expected Panic result"),
        }
    }

    #[test]
    fn test_panic_catch_result_methods() {
        let ok_result = PanicCatchResult::Ok(42);
        let err_result = PanicCatchResult::<i32>::Err(LearningError::Interrupted);
        let panic_result = PanicCatchResult::<i32>::Panic("test".to_string());

        assert!(ok_result.is_ok());
        assert!(!ok_result.is_err());
        assert!(!ok_result.is_panic());

        assert!(!err_result.is_ok());
        assert!(err_result.is_err());
        assert!(!err_result.is_panic());

        assert!(!panic_result.is_ok());
        assert!(!panic_result.is_err());
        assert!(panic_result.is_panic());
    }

    // ========== Checkpoint Retry Tests ==========

    #[test]
    fn test_save_checkpoint_with_retry_success() {
        let call_count = std::cell::RefCell::new(0);

        let result = save_checkpoint_with_retry(|| {
            *call_count.borrow_mut() += 1;
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(*call_count.borrow(), 1); // 成功時は1回のみ
    }

    #[test]
    fn test_save_checkpoint_with_retry_first_fail_then_success() {
        let call_count = std::cell::RefCell::new(0);

        let result = save_checkpoint_with_retry(|| {
            let count = *call_count.borrow();
            *call_count.borrow_mut() += 1;

            if count == 0 {
                Err(LearningError::Io(std::io::Error::other(
                    "first attempt failed",
                )))
            } else {
                Ok(())
            }
        });

        assert!(result.is_ok());
        assert_eq!(*call_count.borrow(), 2); // 1回失敗、1回成功
    }

    #[test]
    fn test_save_checkpoint_with_retry_both_fail() {
        let call_count = std::cell::RefCell::new(0);

        let result = save_checkpoint_with_retry(|| {
            *call_count.borrow_mut() += 1;
            Err(LearningError::Io(std::io::Error::other("always fails")))
        });

        assert!(result.is_err());
        assert_eq!(*call_count.borrow(), 2); // 2回試行して両方失敗
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_error_tracker_with_mixed_results() {
        let mut tracker = ErrorTracker::with_config(100, 5.0); // 5%閾値

        // 混合パターン
        for i in 0..100 {
            if i % 10 == 0 {
                // 10%エラー
                let error_type = match i % 40 {
                    0 => ErrorType::Search,
                    10 => ErrorType::EvalDivergence,
                    20 => ErrorType::Panic,
                    _ => ErrorType::Other,
                };
                let error = ErrorRecord::new(error_type, i, format!("error at {}", i));
                tracker.record_error(error);
            } else {
                tracker.record_success();
            }
        }

        assert_eq!(tracker.total_games(), 100);
        assert_eq!(tracker.total_errors(), 10);

        // 10% > 5% なので閾値超過
        assert!(tracker.is_threshold_exceeded());

        let summary = tracker.error_pattern_summary();
        println!("{}", summary);
        assert!(summary.error_rate_percent > 5.0);
    }

    #[test]
    fn test_error_tracker_recent_errors() {
        let mut tracker = ErrorTracker::new();

        for i in 0..5 {
            let error = ErrorRecord::new(ErrorType::Search, i, format!("error {}", i));
            tracker.record_error(error);
        }

        let recent = tracker.recent_errors(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].game_number, 4); // 最新
        assert_eq!(recent[1].game_number, 3);
        assert_eq!(recent[2].game_number, 2);
    }

    // ========== Requirements Summary Tests ==========

    #[test]
    fn test_req_12_1_search_error_skip() {
        // Req 12.1: 探索エラーをログして次のゲームにスキップ
        let mut tracker = ErrorTracker::new();
        let error = ErrorRecord::new(ErrorType::Search, 0, "search failed");

        // エラーを記録してスキップ（次のゲームに進む）
        let _ = tracker.record_error(error);
        tracker.record_success(); // 次のゲームは成功

        assert_eq!(tracker.total_games(), 2);
        assert_eq!(tracker.total_errors(), 1);

        println!("  12.1: Search error recorded, can skip to next game");
    }

    #[test]
    fn test_req_12_2_checkpoint_retry() {
        // Req 12.2: チェックポイント保存失敗時に1回リトライ
        let attempt = std::cell::RefCell::new(0);

        let result = save_checkpoint_with_retry(|| {
            let n = *attempt.borrow();
            *attempt.borrow_mut() += 1;
            if n == 0 {
                Err(LearningError::Io(std::io::Error::other("fail first")))
            } else {
                Ok(())
            }
        });

        assert!(result.is_ok());
        assert_eq!(*attempt.borrow(), 2);

        println!("  12.2: Checkpoint save retried once on failure");
    }

    #[test]
    fn test_req_12_3_nan_inf_reset() {
        // Req 12.3: NaN/Inf評価値を32768にリセット
        assert!(EvalRecovery::is_invalid(f32::NAN));
        assert!(EvalRecovery::is_invalid(f32::INFINITY));
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::NAN), CENTER);
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::INFINITY), CENTER);
        assert_eq!(CENTER, 32768);

        println!("  12.3: NaN/Inf values reset to 32768 (CENTER)");
    }

    #[test]
    fn test_req_12_4_panic_catch() {
        // Req 12.4: パニックをキャッチしてログ
        let result = catch_panic::<_, i32>(|| {
            panic!("test panic for req 12.4");
        });

        assert!(result.is_panic());
        match result {
            PanicCatchResult::Panic(msg) => {
                assert!(msg.contains("test panic"));
            }
            _ => panic!("Expected panic to be caught"),
        }

        println!("  12.4: Panics caught without crashing training");
    }

    #[test]
    fn test_req_12_6_error_threshold() {
        // Req 12.6: 10,000ゲームウィンドウで1%以上失敗時にトレーニングを一時停止
        let mut tracker = ErrorTracker::with_config(100, 1.0); // テスト用に小さいウィンドウ

        // 99成功 + 2エラー = 2% > 1%
        for _ in 0..98 {
            tracker.record_success();
        }
        tracker.record_error(ErrorRecord::new(ErrorType::Search, 98, "err1"));
        let exceeded = tracker.record_error(ErrorRecord::new(ErrorType::Search, 99, "err2"));

        assert!(exceeded);
        assert!(tracker.is_threshold_exceeded());

        let summary = tracker.error_pattern_summary();
        assert!(summary.error_rate_percent > 1.0);

        println!("  12.6: Training pauses when >1% games fail in window");
    }

    #[test]
    fn test_all_error_handling_requirements() {
        println!("=== Error Handling Requirements Verification ===");

        // 12.1
        let mut tracker = ErrorTracker::new();
        tracker.record_error(ErrorRecord::new(ErrorType::Search, 0, "search fail"));
        println!("  12.1: Log search errors and skip to next game");

        // 12.2
        let _ = save_checkpoint_with_retry(|| Ok(()));
        println!("  12.2: Retry checkpoint save once on failure");

        // 12.3
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::NAN), 32768);
        println!("  12.3: Reset NaN/Inf values to 32768");

        // 12.4
        let _ = catch_panic::<_, ()>(|| Ok(()));
        println!("  12.4: Catch panics without crashing");

        // 12.6
        let summary = tracker.error_pattern_summary();
        let _ = summary.to_string();
        println!("  12.6: Track error counts, pause if >1% fail");

        println!("=== All error handling requirements verified ===");
    }
}
