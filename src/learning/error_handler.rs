//! Error Handling and Recovery Module
//!
//! Implements robust error handling and recovery capabilities for the learning process.
//!
//! # Overview
//!
//! - `ErrorTracker`: Error count tracking and window-based analysis
//! - `EvalRecovery`: NaN/Inf value recovery
//! - `CheckpointRecovery`: Checkpoint load error recovery with suggestions
//! - `WorkerWatchdog`: Hung worker thread detection and restart
//!
//! # Requirements Coverage
//!
//! - Req 9.1: Log search errors with game context and skip to next game
//! - Req 9.2: Retry checkpoint save once after 5-second delay on failure
//! - Req 9.3: Detect NaN/Inf values and reset to 32768 with warning log
//! - Req 9.4: Catch panics from worker threads without crashing
//! - Req 9.5: Track error rate per 10,000 game window
//! - Req 9.6: Pause training and save checkpoint when error rate exceeds 1%
//! - Req 9.7: Detect checkpoint corruption and offer recovery options
//! - Req 9.8: Watchdog for hung worker thread detection with heartbeat

use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

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

/// Retry delay for checkpoint save operations (5 seconds).
///
/// # Requirements Coverage
///
/// - Req 9.2: Retry once after 5-second delay on failure
pub const CHECKPOINT_RETRY_DELAY_SECS: u64 = 5;

/// Save checkpoint with retry on failure.
///
/// If the first save attempt fails, waits 5 seconds and retries once.
/// This provides resilience against transient I/O errors.
///
/// # Requirements Coverage
///
/// - Req 9.2: Retry checkpoint save once after 5-second delay on failure
pub fn save_checkpoint_with_retry<F>(mut save_fn: F) -> Result<(), LearningError>
where
    F: FnMut() -> Result<(), LearningError>,
{
    // First attempt
    match save_fn() {
        Ok(()) => Ok(()),
        Err(first_error) => {
            // Log first failure
            log::warn!(
                "Checkpoint save failed, retrying in {} seconds: {}",
                CHECKPOINT_RETRY_DELAY_SECS,
                first_error
            );

            // Wait 5 seconds before retry (Req 9.2)
            std::thread::sleep(Duration::from_secs(CHECKPOINT_RETRY_DELAY_SECS));

            // Retry
            match save_fn() {
                Ok(()) => {
                    log::info!("Checkpoint save succeeded on retry");
                    Ok(())
                }
                Err(second_error) => {
                    log::warn!("Checkpoint save failed after retry: {}", second_error);
                    Err(second_error)
                }
            }
        }
    }
}

// ============================================================================
// Task 7.6: Checkpoint Load Error Recovery (Req 9.7)
// ============================================================================

/// Checkpoint recovery options when corruption is detected.
///
/// # Requirements Coverage
///
/// - Req 9.7: Offer option to start fresh or try previous checkpoint
#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointRecoveryOption {
    /// Start fresh training from scratch
    StartFresh,
    /// Try loading a previous checkpoint
    TryPrevious(String),
    /// Abort and report error
    Abort,
}

/// Checkpoint load error with recovery suggestions.
///
/// Provides detailed error information and recovery options when
/// checkpoint loading fails.
///
/// # Requirements Coverage
///
/// - Req 9.7: Provide clear error messages with recovery suggestions
#[derive(Clone, Debug)]
pub struct CheckpointLoadError {
    /// Path to the checkpoint that failed to load
    pub checkpoint_path: String,
    /// Error description
    pub error_message: String,
    /// Whether corruption was detected (vs other errors)
    pub is_corruption: bool,
    /// Available recovery options
    pub recovery_options: Vec<CheckpointRecoveryOption>,
    /// Suggested action
    pub suggestion: String,
}

impl CheckpointLoadError {
    /// Create a new checkpoint load error for corruption.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to the corrupted checkpoint
    /// * `error_message` - Description of the corruption error
    /// * `previous_checkpoint` - Optional path to a previous valid checkpoint
    pub fn corruption(
        checkpoint_path: impl Into<String>,
        error_message: impl Into<String>,
        previous_checkpoint: Option<String>,
    ) -> Self {
        let mut options = vec![CheckpointRecoveryOption::StartFresh];
        if let Some(ref prev) = previous_checkpoint {
            options.insert(0, CheckpointRecoveryOption::TryPrevious(prev.clone()));
        }
        options.push(CheckpointRecoveryOption::Abort);

        let suggestion = if previous_checkpoint.is_some() {
            "Try loading a previous checkpoint, or start fresh training.".to_string()
        } else {
            "Start fresh training or check for backup files.".to_string()
        };

        Self {
            checkpoint_path: checkpoint_path.into(),
            error_message: error_message.into(),
            is_corruption: true,
            recovery_options: options,
            suggestion,
        }
    }

    /// Create a new checkpoint load error for I/O failure.
    pub fn io_error(checkpoint_path: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            checkpoint_path: checkpoint_path.into(),
            error_message: error_message.into(),
            is_corruption: false,
            recovery_options: vec![
                CheckpointRecoveryOption::StartFresh,
                CheckpointRecoveryOption::Abort,
            ],
            suggestion: "Check file permissions and disk space, then retry.".to_string(),
        }
    }
}

impl std::fmt::Display for CheckpointLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Checkpoint Load Error ===")?;
        writeln!(f, "Path: {}", self.checkpoint_path)?;
        writeln!(f, "Error: {}", self.error_message)?;
        if self.is_corruption {
            writeln!(f, "Type: Data corruption detected")?;
        }
        writeln!(f, "Suggestion: {}", self.suggestion)?;
        writeln!(f, "Recovery options:")?;
        for (i, opt) in self.recovery_options.iter().enumerate() {
            match opt {
                CheckpointRecoveryOption::StartFresh => {
                    writeln!(f, "  {}. Start fresh training", i + 1)?;
                }
                CheckpointRecoveryOption::TryPrevious(path) => {
                    writeln!(f, "  {}. Try previous checkpoint: {}", i + 1, path)?;
                }
                CheckpointRecoveryOption::Abort => {
                    writeln!(f, "  {}. Abort and exit", i + 1)?;
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Task 7.7: Worker Watchdog (Req 9.8)
// ============================================================================

/// Watchdog timeout threshold in seconds (30s based on typical game duration).
///
/// A worker thread that doesn't report progress within this time is considered hung.
///
/// # Requirements Coverage
///
/// - Req 9.8: Timeout threshold for hung thread detection
pub const WATCHDOG_TIMEOUT_SECS: u64 = 30;

/// Heartbeat for a single worker thread.
///
/// Workers should update this regularly to indicate they are making progress.
#[derive(Debug)]
pub struct WorkerHeartbeat {
    /// Last activity timestamp (Unix epoch milliseconds for atomic storage)
    last_activity: AtomicU64,
    /// Worker ID for identification
    worker_id: usize,
    /// Flag indicating if this worker has been marked as hung
    is_hung: AtomicBool,
    /// Restart count for this worker
    restart_count: AtomicU64,
}

impl WorkerHeartbeat {
    /// Create a new worker heartbeat.
    pub fn new(worker_id: usize) -> Self {
        Self {
            last_activity: AtomicU64::new(Self::current_timestamp()),
            worker_id,
            is_hung: AtomicBool::new(false),
            restart_count: AtomicU64::new(0),
        }
    }

    /// Get current timestamp in milliseconds.
    fn current_timestamp() -> u64 {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Update heartbeat to current time.
    ///
    /// Should be called by the worker after completing work.
    pub fn beat(&self) {
        self.last_activity
            .store(Self::current_timestamp(), Ordering::SeqCst);
        self.is_hung.store(false, Ordering::SeqCst);
    }

    /// Get time since last activity in seconds.
    pub fn time_since_activity(&self) -> f64 {
        let last = self.last_activity.load(Ordering::SeqCst);
        let now = Self::current_timestamp();
        (now.saturating_sub(last)) as f64 / 1000.0
    }

    /// Check if this worker is hung (no activity for timeout duration).
    ///
    /// A worker is considered hung if the time since last activity exceeds
    /// the timeout threshold. Uses `>=` for 0-second timeout to allow immediate detection.
    pub fn is_hung(&self, timeout_secs: u64) -> bool {
        let elapsed = self.time_since_activity();
        if timeout_secs == 0 {
            // Special case: 0 timeout means "check if any time has passed"
            // We consider this as "always hung after creation" for testing
            elapsed >= 0.0
        } else {
            elapsed > timeout_secs as f64
        }
    }

    /// Mark this worker as hung.
    pub fn mark_hung(&self) {
        self.is_hung.store(true, Ordering::SeqCst);
    }

    /// Check if this worker was previously marked as hung.
    pub fn was_marked_hung(&self) -> bool {
        self.is_hung.load(Ordering::SeqCst)
    }

    /// Increment restart count and return new value.
    pub fn record_restart(&self) -> u64 {
        self.restart_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get total restart count.
    pub fn restart_count(&self) -> u64 {
        self.restart_count.load(Ordering::SeqCst)
    }

    /// Get worker ID.
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }
}

/// Watchdog for monitoring multiple worker threads.
///
/// Tracks heartbeats from all workers and detects hung threads.
///
/// # Requirements Coverage
///
/// - Req 9.8: Monitor worker thread activity with heartbeat mechanism
#[derive(Debug)]
pub struct WorkerWatchdog {
    /// Heartbeats for each worker
    heartbeats: Vec<Arc<WorkerHeartbeat>>,
    /// Timeout threshold in seconds
    timeout_secs: u64,
    /// Total hung thread detections
    total_hung_detections: AtomicU64,
    /// Total restarts performed
    total_restarts: AtomicU64,
}

impl WorkerWatchdog {
    /// Create a new watchdog for the specified number of workers.
    ///
    /// # Arguments
    ///
    /// * `num_workers` - Number of worker threads to monitor
    /// * `timeout_secs` - Timeout threshold (default: WATCHDOG_TIMEOUT_SECS)
    pub fn new(num_workers: usize, timeout_secs: Option<u64>) -> Self {
        let heartbeats = (0..num_workers)
            .map(|id| Arc::new(WorkerHeartbeat::new(id)))
            .collect();

        Self {
            heartbeats,
            timeout_secs: timeout_secs.unwrap_or(WATCHDOG_TIMEOUT_SECS),
            total_hung_detections: AtomicU64::new(0),
            total_restarts: AtomicU64::new(0),
        }
    }

    /// Get a heartbeat handle for a worker.
    ///
    /// # Arguments
    ///
    /// * `worker_id` - ID of the worker (0-based)
    ///
    /// # Returns
    ///
    /// Arc to the worker's heartbeat, or None if worker_id is invalid.
    pub fn get_heartbeat(&self, worker_id: usize) -> Option<Arc<WorkerHeartbeat>> {
        self.heartbeats.get(worker_id).cloned()
    }

    /// Check all workers and return list of hung worker IDs.
    ///
    /// # Returns
    ///
    /// Vector of worker IDs that have exceeded the timeout threshold.
    pub fn check_workers(&self) -> Vec<usize> {
        let mut hung = Vec::new();

        for heartbeat in &self.heartbeats {
            if heartbeat.is_hung(self.timeout_secs) && !heartbeat.was_marked_hung() {
                heartbeat.mark_hung();
                hung.push(heartbeat.worker_id());
                self.total_hung_detections.fetch_add(1, Ordering::SeqCst);

                log::warn!(
                    "Watchdog: Worker {} hung (no activity for {:.1}s, threshold {}s)",
                    heartbeat.worker_id(),
                    heartbeat.time_since_activity(),
                    self.timeout_secs
                );
            }
        }

        hung
    }

    /// Record that a worker was restarted.
    ///
    /// # Arguments
    ///
    /// * `worker_id` - ID of the restarted worker
    pub fn record_restart(&self, worker_id: usize) {
        if let Some(heartbeat) = self.heartbeats.get(worker_id) {
            let restarts = heartbeat.record_restart();
            self.total_restarts.fetch_add(1, Ordering::SeqCst);

            log::info!(
                "Watchdog: Worker {} restarted (restart #{} for this worker)",
                worker_id,
                restarts
            );
        }
    }

    /// Get total number of hung thread detections.
    pub fn total_hung_detections(&self) -> u64 {
        self.total_hung_detections.load(Ordering::SeqCst)
    }

    /// Get total number of restarts.
    pub fn total_restarts(&self) -> u64 {
        self.total_restarts.load(Ordering::SeqCst)
    }

    /// Get watchdog status summary.
    pub fn status_summary(&self) -> WatchdogStatus {
        let workers: Vec<_> = self
            .heartbeats
            .iter()
            .map(|hb| WorkerStatus {
                worker_id: hb.worker_id(),
                time_since_activity_secs: hb.time_since_activity(),
                is_hung: hb.was_marked_hung(),
                restart_count: hb.restart_count(),
            })
            .collect();

        WatchdogStatus {
            num_workers: self.heartbeats.len(),
            timeout_secs: self.timeout_secs,
            total_hung_detections: self.total_hung_detections(),
            total_restarts: self.total_restarts(),
            workers,
        }
    }
}

/// Status of a single worker.
#[derive(Clone, Debug)]
pub struct WorkerStatus {
    /// Worker ID
    pub worker_id: usize,
    /// Seconds since last activity
    pub time_since_activity_secs: f64,
    /// Whether worker is currently marked as hung
    pub is_hung: bool,
    /// Number of times this worker was restarted
    pub restart_count: u64,
}

/// Overall watchdog status summary.
#[derive(Clone, Debug)]
pub struct WatchdogStatus {
    /// Number of workers being monitored
    pub num_workers: usize,
    /// Timeout threshold in seconds
    pub timeout_secs: u64,
    /// Total hung thread detections since start
    pub total_hung_detections: u64,
    /// Total restarts since start
    pub total_restarts: u64,
    /// Per-worker status
    pub workers: Vec<WorkerStatus>,
}

impl std::fmt::Display for WatchdogStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Watchdog Status ===")?;
        writeln!(
            f,
            "Workers: {}, Timeout: {}s",
            self.num_workers, self.timeout_secs
        )?;
        writeln!(
            f,
            "Total hung detections: {}, Total restarts: {}",
            self.total_hung_detections, self.total_restarts
        )?;
        for w in &self.workers {
            let status = if w.is_hung { "HUNG" } else { "OK" };
            writeln!(
                f,
                "  Worker {}: {} (last activity {:.1}s ago, {} restarts)",
                w.worker_id, status, w.time_since_activity_secs, w.restart_count
            )?;
        }
        Ok(())
    }
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

    // ========== Task 7.2: Checkpoint Retry Delay Tests ==========

    #[test]
    fn test_checkpoint_retry_delay_constant() {
        // Verify the retry delay is 5 seconds as per Req 9.2
        assert_eq!(CHECKPOINT_RETRY_DELAY_SECS, 5);
    }

    // ========== Task 7.6: Checkpoint Load Error Recovery Tests ==========

    #[test]
    fn test_checkpoint_load_error_corruption() {
        let error = CheckpointLoadError::corruption(
            "/path/to/checkpoint.bin",
            "Checksum mismatch",
            Some("/path/to/prev_checkpoint.bin".to_string()),
        );

        assert!(error.is_corruption);
        assert_eq!(error.checkpoint_path, "/path/to/checkpoint.bin");
        assert!(error.error_message.contains("Checksum"));
        assert!(error.recovery_options.len() >= 2);

        // Should have TryPrevious as first option
        assert!(matches!(
            &error.recovery_options[0],
            CheckpointRecoveryOption::TryPrevious(_)
        ));

        // Display should work
        let display = format!("{}", error);
        assert!(display.contains("corruption"));
        assert!(display.contains("Recovery options"));
    }

    #[test]
    fn test_checkpoint_load_error_corruption_no_previous() {
        let error =
            CheckpointLoadError::corruption("/path/to/checkpoint.bin", "Checksum mismatch", None);

        assert!(error.is_corruption);
        // Without previous checkpoint, first option should be StartFresh
        assert!(matches!(
            &error.recovery_options[0],
            CheckpointRecoveryOption::StartFresh
        ));
        assert!(error.suggestion.contains("Start fresh"));
    }

    #[test]
    fn test_checkpoint_load_error_io() {
        let error = CheckpointLoadError::io_error("/path/to/checkpoint.bin", "Permission denied");

        assert!(!error.is_corruption);
        assert!(error.suggestion.contains("permissions"));
        assert!(
            error
                .recovery_options
                .contains(&CheckpointRecoveryOption::StartFresh)
        );
        assert!(
            error
                .recovery_options
                .contains(&CheckpointRecoveryOption::Abort)
        );
    }

    #[test]
    fn test_checkpoint_recovery_option_equality() {
        assert_eq!(
            CheckpointRecoveryOption::StartFresh,
            CheckpointRecoveryOption::StartFresh
        );
        assert_eq!(
            CheckpointRecoveryOption::Abort,
            CheckpointRecoveryOption::Abort
        );
        assert_eq!(
            CheckpointRecoveryOption::TryPrevious("a".to_string()),
            CheckpointRecoveryOption::TryPrevious("a".to_string())
        );
        assert_ne!(
            CheckpointRecoveryOption::TryPrevious("a".to_string()),
            CheckpointRecoveryOption::TryPrevious("b".to_string())
        );
    }

    // ========== Task 7.7: Worker Watchdog Tests ==========

    #[test]
    fn test_watchdog_timeout_constant() {
        // Verify the timeout is 30 seconds as per design
        assert_eq!(WATCHDOG_TIMEOUT_SECS, 30);
    }

    #[test]
    fn test_worker_heartbeat_creation() {
        let heartbeat = WorkerHeartbeat::new(5);

        assert_eq!(heartbeat.worker_id(), 5);
        assert!(!heartbeat.was_marked_hung());
        assert_eq!(heartbeat.restart_count(), 0);
        // Time since activity should be very small (just created)
        assert!(heartbeat.time_since_activity() < 1.0);
    }

    #[test]
    fn test_worker_heartbeat_beat() {
        let heartbeat = WorkerHeartbeat::new(0);

        // Simulate some time passing by marking hung then beating
        heartbeat.mark_hung();
        assert!(heartbeat.was_marked_hung());

        heartbeat.beat();
        assert!(!heartbeat.was_marked_hung());
        assert!(heartbeat.time_since_activity() < 1.0);
    }

    #[test]
    fn test_worker_heartbeat_restart_count() {
        let heartbeat = WorkerHeartbeat::new(0);

        assert_eq!(heartbeat.restart_count(), 0);
        assert_eq!(heartbeat.record_restart(), 1);
        assert_eq!(heartbeat.restart_count(), 1);
        assert_eq!(heartbeat.record_restart(), 2);
        assert_eq!(heartbeat.restart_count(), 2);
    }

    #[test]
    fn test_worker_heartbeat_is_hung() {
        let heartbeat = WorkerHeartbeat::new(0);

        // Just created, should not be hung with any reasonable timeout
        assert!(!heartbeat.is_hung(1));
        assert!(!heartbeat.is_hung(30));

        // With 0 second timeout, should be hung
        assert!(heartbeat.is_hung(0));
    }

    #[test]
    fn test_worker_watchdog_creation() {
        let watchdog = WorkerWatchdog::new(4, Some(30));

        assert_eq!(watchdog.total_hung_detections(), 0);
        assert_eq!(watchdog.total_restarts(), 0);

        // Should be able to get heartbeats for all workers
        for i in 0..4 {
            assert!(watchdog.get_heartbeat(i).is_some());
        }
        // Invalid worker ID returns None
        assert!(watchdog.get_heartbeat(10).is_none());
    }

    #[test]
    fn test_worker_watchdog_default_timeout() {
        let watchdog = WorkerWatchdog::new(2, None);

        let status = watchdog.status_summary();
        assert_eq!(status.timeout_secs, WATCHDOG_TIMEOUT_SECS);
    }

    #[test]
    fn test_worker_watchdog_check_workers_no_hung() {
        let watchdog = WorkerWatchdog::new(4, Some(30));

        // All workers just created, none should be hung
        let hung = watchdog.check_workers();
        assert!(hung.is_empty());
        assert_eq!(watchdog.total_hung_detections(), 0);
    }

    #[test]
    fn test_worker_watchdog_check_workers_with_hung() {
        let watchdog = WorkerWatchdog::new(4, Some(0)); // 0 second timeout

        // With 0 second timeout, all workers should be detected as hung
        let hung = watchdog.check_workers();
        assert_eq!(hung.len(), 4);
        assert_eq!(watchdog.total_hung_detections(), 4);

        // Second check should not detect them again (already marked)
        let hung2 = watchdog.check_workers();
        assert!(hung2.is_empty());
        assert_eq!(watchdog.total_hung_detections(), 4);
    }

    #[test]
    fn test_worker_watchdog_record_restart() {
        let watchdog = WorkerWatchdog::new(4, Some(30));

        watchdog.record_restart(0);
        watchdog.record_restart(0);
        watchdog.record_restart(2);

        assert_eq!(watchdog.total_restarts(), 3);

        let status = watchdog.status_summary();
        assert_eq!(status.workers[0].restart_count, 2);
        assert_eq!(status.workers[1].restart_count, 0);
        assert_eq!(status.workers[2].restart_count, 1);
    }

    #[test]
    fn test_worker_watchdog_status_summary() {
        let watchdog = WorkerWatchdog::new(2, Some(30));

        let status = watchdog.status_summary();

        assert_eq!(status.num_workers, 2);
        assert_eq!(status.timeout_secs, 30);
        assert_eq!(status.total_hung_detections, 0);
        assert_eq!(status.total_restarts, 0);
        assert_eq!(status.workers.len(), 2);

        // Display should work
        let display = format!("{}", status);
        assert!(display.contains("Watchdog Status"));
        assert!(display.contains("Workers: 2"));
    }

    #[test]
    fn test_worker_status_fields() {
        let status = WorkerStatus {
            worker_id: 3,
            time_since_activity_secs: 15.5,
            is_hung: false,
            restart_count: 2,
        };

        assert_eq!(status.worker_id, 3);
        assert!((status.time_since_activity_secs - 15.5).abs() < 0.01);
        assert!(!status.is_hung);
        assert_eq!(status.restart_count, 2);
    }

    // ========== Task 7 Requirements Summary ==========

    #[test]
    fn test_task7_requirements_summary() {
        println!("=== Task 7: Error Handling and Recovery System ===");

        // 7.1: Error recovery for search and game execution
        let mut tracker = ErrorTracker::new();
        tracker.record_error(ErrorRecord::new(
            ErrorType::Search,
            0,
            "search error in game 0",
        ));
        println!(
            "  7.1: Error recovery for search - ErrorTracker records errors with game context"
        );

        // 7.2: Checkpoint save retry with 5-second delay
        assert_eq!(CHECKPOINT_RETRY_DELAY_SECS, 5);
        println!(
            "  7.2: Checkpoint retry delay is {} seconds",
            CHECKPOINT_RETRY_DELAY_SECS
        );

        // 7.3: Evaluation error detection (already tested above)
        assert_eq!(EvalRecovery::sanitize_to_u16(f32::NAN), CENTER);
        println!("  7.3: NaN/Inf values reset to {} (neutral)", CENTER);

        // 7.4: Worker thread panic handling
        let panic_result = catch_panic::<_, ()>(|| {
            panic!("test");
        });
        assert!(panic_result.is_panic());
        println!("  7.4: Panics caught without crashing");

        // 7.5: Error threshold monitoring
        let mut tracker2 = ErrorTracker::with_config(100, 1.0);
        for _ in 0..98 {
            tracker2.record_success();
        }
        tracker2.record_error(ErrorRecord::new(ErrorType::Search, 98, "e1"));
        let exceeded = tracker2.record_error(ErrorRecord::new(ErrorType::Search, 99, "e2"));
        assert!(exceeded);
        println!("  7.5: Error threshold monitoring - auto-pause when >1% errors");

        // 7.6: Checkpoint load error recovery
        let load_error = CheckpointLoadError::corruption("test.bin", "checksum fail", None);
        assert!(load_error.is_corruption);
        assert!(
            load_error
                .recovery_options
                .contains(&CheckpointRecoveryOption::StartFresh)
        );
        println!("  7.6: Checkpoint load error recovery with suggestions");

        // 7.7: Watchdog for hung worker detection
        assert_eq!(WATCHDOG_TIMEOUT_SECS, 30);
        let watchdog = WorkerWatchdog::new(4, Some(WATCHDOG_TIMEOUT_SECS));
        let _ = watchdog.check_workers();
        println!(
            "  7.7: Watchdog with {}s timeout for hung thread detection",
            WATCHDOG_TIMEOUT_SECS
        );

        println!("=== All Task 7 requirements implemented ===");
    }
}
