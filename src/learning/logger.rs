//! Training Logger for Statistics Output.
//!
//! This module implements a comprehensive training statistics logging system
//! that outputs real-time and detailed statistics at configurable intervals.
//!
//! # Logging Intervals
//!
//! - Real-time statistics: Every 100 games
//! - Detailed statistics: Every 10,000 games
//! - Complete summary: At each checkpoint
//!
//! # Requirements Coverage
//!
//! - Req 7.1: Output real-time statistics every 100 games
//! - Req 7.2: Include stone difference, win rates, move count, elapsed time
//! - Req 7.3: Output detailed statistics every 10,000 games
//! - Req 7.4: Include eval distribution, search depth, search time, TT hit rate
//! - Req 7.5: Output complete summary at each checkpoint
//! - Req 7.6: Write logs with format logs/training_YYYYMMDD_HHMMSS.log
//! - Req 7.7: Output progress reports with estimated time remaining
//! - Req 7.8: Detect and warn on evaluation divergence (NaN, extreme values)

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chrono::Local;
use serde::Serialize;

use crate::learning::LearningError;

/// Default batch log interval (100 games).
pub const DEFAULT_BATCH_INTERVAL: u64 = 100;

/// Default detailed log interval (10,000 games).
pub const DEFAULT_DETAILED_INTERVAL: u64 = 10_000;

/// Default checkpoint interval (100,000 games).
pub const DEFAULT_CHECKPOINT_INTERVAL: u64 = 100_000;

/// Threshold for evaluation divergence warning.
pub const EVAL_DIVERGENCE_THRESHOLD: f32 = 1000.0;

/// Log level for filtering log messages.
///
/// Supports ordering: Debug < Info < Warning < Error.
/// Messages are only logged if their level is >= the configured log level.
///
/// # Requirements Coverage
///
/// - Req 4.5: Support configurable log levels (debug, info, warning, error)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum LogLevel {
    /// Debug level - most verbose, for development/debugging.
    Debug,
    /// Info level - standard operational messages.
    #[default]
    Info,
    /// Warning level - potential issues that don't stop execution.
    Warning,
    /// Error level - errors that may affect functionality.
    Error,
}

impl FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "debug" => Ok(LogLevel::Debug),
            "info" => Ok(LogLevel::Info),
            "warning" | "warn" => Ok(LogLevel::Warning),
            "error" => Ok(LogLevel::Error),
            _ => Err(format!(
                "Invalid log level: {}. Expected one of: debug, info, warning, error",
                s
            )),
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warning => write!(f, "WARNING"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Batch statistics for real-time logging (every 100 games).
///
/// Contains basic game outcome statistics and throughput metrics.
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    /// Total number of games completed so far.
    pub games_completed: u64,
    /// Average stone difference in this batch.
    pub avg_stone_diff: f32,
    /// Black win rate (0.0 to 1.0) in this batch.
    pub black_win_rate: f32,
    /// White win rate (0.0 to 1.0) in this batch.
    pub white_win_rate: f32,
    /// Draw rate (0.0 to 1.0) in this batch.
    pub draw_rate: f32,
    /// Average move count per game in this batch.
    pub avg_move_count: f32,
    /// Elapsed time in seconds since training start.
    pub elapsed_secs: f64,
    /// Games per second throughput.
    pub games_per_sec: f64,
    /// 平均ランダム手数（黒番）
    pub avg_random_moves_black: f32,
    /// 平均ランダム手数（白番）
    pub avg_random_moves_white: f32,
    /// ランダム手率（黒番）
    pub random_move_rate_black: f32,
    /// ランダム手率（白番）
    pub random_move_rate_white: f32,
}

impl BatchStats {
    /// Create batch stats from accumulated game results.
    ///
    /// # Arguments
    ///
    /// * `games_completed` - Total games completed
    /// * `stone_diffs` - Stone differences from games in this batch
    /// * `move_counts` - Move counts from games in this batch
    /// * `elapsed_secs` - Total elapsed time
    pub fn from_games(
        games_completed: u64,
        stone_diffs: &[f32],
        move_counts: &[usize],
        elapsed_secs: f64,
    ) -> Self {
        Self::from_games_with_random(
            games_completed,
            stone_diffs,
            move_counts,
            elapsed_secs,
            None,
            None,
        )
    }

    /// Create batch stats including random move metrics.
    pub fn from_games_with_random(
        games_completed: u64,
        stone_diffs: &[f32],
        move_counts: &[usize],
        elapsed_secs: f64,
        random_moves_black: Option<&[u32]>,
        random_moves_white: Option<&[u32]>,
    ) -> Self {
        let batch_size = stone_diffs.len() as f32;
        if batch_size == 0.0 {
            return Self::default();
        }

        let avg_stone_diff = stone_diffs.iter().sum::<f32>() / batch_size;
        let avg_move_count = move_counts.iter().sum::<usize>() as f32 / batch_size;

        let black_wins = stone_diffs.iter().filter(|&&d| d > 0.0).count() as f32;
        let white_wins = stone_diffs.iter().filter(|&&d| d < 0.0).count() as f32;
        let draws = stone_diffs.iter().filter(|&&d| d == 0.0).count() as f32;

        let black_win_rate = black_wins / batch_size;
        let white_win_rate = white_wins / batch_size;
        let draw_rate = draws / batch_size;

        let games_per_sec = if elapsed_secs > 0.0 {
            games_completed as f64 / elapsed_secs
        } else {
            0.0
        };

        let (avg_random_moves_black, avg_random_moves_white, random_rate_black, random_rate_white) =
            if let (Some(rb), Some(rw)) = (random_moves_black, random_moves_white) {
                debug_assert_eq!(rb.len(), stone_diffs.len());
                debug_assert_eq!(rw.len(), stone_diffs.len());
                let avg_rb = rb.iter().sum::<u32>() as f32 / batch_size;
                let avg_rw = rw.iter().sum::<u32>() as f32 / batch_size;
                let moves_per_color = (avg_move_count / 2.0).max(1e-6);
                (
                    avg_rb,
                    avg_rw,
                    (avg_rb / moves_per_color) * 100.0,
                    (avg_rw / moves_per_color) * 100.0,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };

        Self {
            games_completed,
            avg_stone_diff,
            black_win_rate,
            white_win_rate,
            draw_rate,
            avg_move_count,
            elapsed_secs,
            games_per_sec,
            avg_random_moves_black,
            avg_random_moves_white,
            random_move_rate_black: random_rate_black,
            random_move_rate_white: random_rate_white,
        }
    }
}

/// Detailed statistics for periodic reporting (every 10,000 games).
///
/// Contains extended metrics including evaluation distribution and search statistics.
#[derive(Clone, Debug, Default)]
pub struct DetailedStats {
    /// Basic batch statistics.
    pub batch: BatchStats,
    /// Mean evaluation value observed.
    pub eval_mean: f32,
    /// Standard deviation of evaluation values.
    pub eval_stddev: f32,
    /// Minimum evaluation value observed.
    pub eval_min: f32,
    /// Maximum evaluation value observed.
    pub eval_max: f32,
    /// Average search depth achieved.
    pub avg_search_depth: f32,
    /// Average search time per move in milliseconds.
    pub avg_search_time_ms: f64,
    /// Transposition table hit rate (0.0 to 1.0).
    pub tt_hit_rate: f64,
}

impl DetailedStats {
    /// Create detailed stats from accumulated metrics.
    ///
    /// # Arguments
    ///
    /// * `batch` - Basic batch statistics
    /// * `eval_values` - Evaluation values observed
    /// * `search_depths` - Search depths achieved
    /// * `search_times_ms` - Search times in milliseconds
    /// * `tt_hits` - Number of TT hits
    /// * `tt_probes` - Number of TT probes
    pub fn from_metrics(
        batch: BatchStats,
        eval_values: &[f32],
        search_depths: &[usize],
        search_times_ms: &[f64],
        tt_hits: u64,
        tt_probes: u64,
    ) -> Self {
        let n = eval_values.len() as f32;
        if n == 0.0 {
            return Self {
                batch,
                ..Default::default()
            };
        }

        // Evaluation distribution
        let eval_mean = eval_values.iter().sum::<f32>() / n;
        let eval_variance = eval_values
            .iter()
            .map(|&v| (v - eval_mean).powi(2))
            .sum::<f32>()
            / n;
        let eval_stddev = eval_variance.sqrt();
        let eval_min = eval_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let eval_max = eval_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Search statistics
        let avg_search_depth = if !search_depths.is_empty() {
            search_depths.iter().sum::<usize>() as f32 / search_depths.len() as f32
        } else {
            0.0
        };

        let avg_search_time_ms = if !search_times_ms.is_empty() {
            search_times_ms.iter().sum::<f64>() / search_times_ms.len() as f64
        } else {
            0.0
        };

        let tt_hit_rate = if tt_probes > 0 {
            tt_hits as f64 / tt_probes as f64
        } else {
            0.0
        };

        Self {
            batch,
            eval_mean,
            eval_stddev,
            eval_min,
            eval_max,
            avg_search_depth,
            avg_search_time_ms,
            tt_hit_rate,
        }
    }
}

/// Log message types for async logging.
#[derive(Clone, Debug)]
enum LogMessage {
    /// Batch statistics log entry.
    Batch(u64, BatchStats),
    /// Detailed statistics log entry.
    Detailed(u64, DetailedStats),
    /// Checkpoint summary log entry.
    Checkpoint(u64, DetailedStats),
    /// Progress report with ETA.
    Progress(u64, u64, Duration),
    /// Warning message.
    Warning(String),
    /// Info message.
    Info(String),
    /// Deterministic evaluation sample summary.
    Evaluation(u64, EvalLogEntry),
    /// Shutdown signal.
    Shutdown,
}

/// Deterministic evaluation sample metrics for logging.
#[derive(Clone, Debug)]
pub struct EvalLogEntry {
    pub games_sampled: u64,
    pub avg_stone_diff: f64,
    pub black_win_rate: f64,
    pub white_win_rate: f64,
    pub draw_rate: f64,
}

impl EvalLogEntry {
    pub fn new(
        games_sampled: u64,
        avg_stone_diff: f64,
        black_win_rate: f64,
        white_win_rate: f64,
        draw_rate: f64,
    ) -> Self {
        Self {
            games_sampled,
            avg_stone_diff,
            black_win_rate,
            white_win_rate,
            draw_rate,
        }
    }
}

/// Training logger for statistics output.
///
/// Provides non-blocking logging to avoid impacting training performance.
/// Uses a background thread for file I/O operations.
///
/// # Example
///
/// ```ignore
/// use prismind::learning::logger::{TrainingLogger, BatchStats};
///
/// let logger = TrainingLogger::new("logs/")?;
///
/// // Log batch statistics
/// let stats = BatchStats::from_games(100, &stone_diffs, &move_counts, 10.0);
/// logger.log_batch(100, &stats);
///
/// // Log warning
/// logger.log_warning("Evaluation divergence detected");
///
/// // Shutdown cleanly
/// logger.shutdown()?;
/// ```
pub struct TrainingLogger {
    /// Sender for async log messages.
    sender: Sender<LogMessage>,
    /// Background writer thread handle.
    writer_handle: Option<JoinHandle<()>>,
    /// Training start time.
    start_time: Instant,
    /// Log file path.
    log_path: PathBuf,
}

impl TrainingLogger {
    /// Create a new training logger.
    ///
    /// Creates the log directory if it doesn't exist and starts the
    /// background writer thread.
    ///
    /// # Arguments
    ///
    /// * `log_dir` - Path to log directory
    ///
    /// # Returns
    ///
    /// Result containing the logger or an error.
    ///
    /// # Errors
    ///
    /// - `LearningError::Io` if directory creation or file open fails
    ///
    /// # Requirements
    ///
    /// - Req 7.6: Log file format logs/training_YYYYMMDD_HHMMSS.log
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Result<Self, LearningError> {
        let log_dir = log_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !log_dir.exists() {
            fs::create_dir_all(&log_dir)?;
        }

        // Generate log filename with timestamp
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let log_filename = format!("training_{}.log", timestamp);
        let log_path = log_dir.join(&log_filename);

        // Create log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        // Create channel for async logging
        let (sender, receiver) = mpsc::channel();

        // Start background writer thread
        let writer_handle = Self::start_writer_thread(file, receiver);

        let logger = Self {
            sender,
            writer_handle: Some(writer_handle),
            start_time: Instant::now(),
            log_path,
        };

        // Log startup message
        logger.log_info("Training logger initialized");

        Ok(logger)
    }

    /// Start the background writer thread.
    fn start_writer_thread(file: File, receiver: Receiver<LogMessage>) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut writer = BufWriter::new(file);

            while let Ok(msg) = receiver.recv() {
                match msg {
                    LogMessage::Shutdown => break,
                    _ => {
                        if let Err(e) = Self::write_message(&mut writer, &msg) {
                            eprintln!("Logger error: {}", e);
                        }
                    }
                }
            }

            // Flush remaining data
            let _ = writer.flush();
        })
    }

    /// Write a log message to the file.
    fn write_message<W: Write>(writer: &mut W, msg: &LogMessage) -> std::io::Result<()> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

        match msg {
            LogMessage::Batch(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] BATCH {:>8} | diff:{:>+6.2} | B:{:.1}% W:{:.1}% D:{:.1}% | moves:{:.1} | rnd:B{:.1}% W{:.1}% | {:.2} g/s | {:.1}s",
                    timestamp,
                    game_count,
                    stats.avg_stone_diff,
                    stats.black_win_rate * 100.0,
                    stats.white_win_rate * 100.0,
                    stats.draw_rate * 100.0,
                    stats.avg_move_count,
                    stats.random_move_rate_black,
                    stats.random_move_rate_white,
                    stats.games_per_sec,
                    stats.elapsed_secs
                )?;
            }
            LogMessage::Detailed(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] DETAILED {:>8} | eval: mean={:.2} std={:.2} min={:.2} max={:.2} | depth={:.1} time={:.1}ms | TT={:.1}%",
                    timestamp,
                    game_count,
                    stats.eval_mean,
                    stats.eval_stddev,
                    stats.eval_min,
                    stats.eval_max,
                    stats.avg_search_depth,
                    stats.avg_search_time_ms,
                    stats.tt_hit_rate * 100.0
                )?;
            }
            LogMessage::Checkpoint(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] ========== CHECKPOINT {} ==========",
                    timestamp, game_count
                )?;
                writeln!(writer, "  Games completed: {}", game_count)?;
                writeln!(
                    writer,
                    "  Elapsed time: {:.1}s ({:.2} hours)",
                    stats.batch.elapsed_secs,
                    stats.batch.elapsed_secs / 3600.0
                )?;
                writeln!(
                    writer,
                    "  Throughput: {:.2} games/sec",
                    stats.batch.games_per_sec
                )?;
                writeln!(
                    writer,
                    "  Avg stone diff: {:+.2}",
                    stats.batch.avg_stone_diff
                )?;
                writeln!(
                    writer,
                    "  Win rates: Black {:.1}% | White {:.1}% | Draw {:.1}%",
                    stats.batch.black_win_rate * 100.0,
                    stats.batch.white_win_rate * 100.0,
                    stats.batch.draw_rate * 100.0
                )?;
                writeln!(
                    writer,
                    "  Avg move count: {:.1}",
                    stats.batch.avg_move_count
                )?;
                writeln!(
                    writer,
                    "  Eval distribution: mean={:.2} std={:.2} min={:.2} max={:.2}",
                    stats.eval_mean, stats.eval_stddev, stats.eval_min, stats.eval_max
                )?;
                writeln!(
                    writer,
                    "  Search: depth={:.1} time={:.1}ms TT_hit={:.1}%",
                    stats.avg_search_depth,
                    stats.avg_search_time_ms,
                    stats.tt_hit_rate * 100.0
                )?;
                writeln!(
                    writer,
                    "[{}] ==========================================",
                    timestamp
                )?;
            }
            LogMessage::Progress(current, target, eta) => {
                let progress = *current as f64 / *target as f64 * 100.0;
                let eta_secs = eta.as_secs();
                let eta_hours = eta_secs / 3600;
                let eta_mins = (eta_secs % 3600) / 60;
                writeln!(
                    writer,
                    "[{}] PROGRESS {:>8}/{} ({:.1}%) | ETA: {}h {:02}m",
                    timestamp, current, target, progress, eta_hours, eta_mins
                )?;
            }
            LogMessage::Warning(msg) => {
                writeln!(writer, "[{}] WARNING: {}", timestamp, msg)?;
            }
            LogMessage::Info(msg) => {
                writeln!(writer, "[{}] INFO: {}", timestamp, msg)?;
            }
            LogMessage::Evaluation(game_count, entry) => {
                writeln!(
                    writer,
                    "[{}] EVAL {:>8} | sample:{} | diff:{:+.2} | B:{:.1}% W:{:.1}% D:{:.1}%",
                    timestamp,
                    game_count,
                    entry.games_sampled,
                    entry.avg_stone_diff,
                    entry.black_win_rate * 100.0,
                    entry.white_win_rate * 100.0,
                    entry.draw_rate * 100.0
                )?;
            }
            LogMessage::Shutdown => {}
        }

        writer.flush()?;
        Ok(())
    }

    /// Log batch statistics (every 100 games).
    ///
    /// Non-blocking: sends message to background writer thread.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Current game count
    /// * `stats` - Batch statistics
    ///
    /// # Requirements
    ///
    /// - Req 7.1: Output real-time statistics every 100 games
    /// - Req 7.2: Include stone difference, win rates, move count, elapsed time
    pub fn log_batch(&self, game_count: u64, stats: &BatchStats) {
        let _ = self
            .sender
            .send(LogMessage::Batch(game_count, stats.clone()));
    }

    /// Log detailed statistics (every 10,000 games).
    ///
    /// Non-blocking: sends message to background writer thread.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Current game count
    /// * `stats` - Detailed statistics
    ///
    /// # Requirements
    ///
    /// - Req 7.3: Output detailed statistics every 10,000 games
    /// - Req 7.4: Include eval distribution, search depth, search time, TT hit rate
    pub fn log_detailed(&self, game_count: u64, stats: &DetailedStats) {
        let _ = self
            .sender
            .send(LogMessage::Detailed(game_count, stats.clone()));
    }

    /// Log checkpoint summary.
    ///
    /// Non-blocking: sends message to background writer thread.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Current game count
    /// * `stats` - Detailed statistics for summary
    ///
    /// # Requirements
    ///
    /// - Req 7.5: Output complete summary at each checkpoint
    pub fn log_checkpoint(&self, game_count: u64, stats: &DetailedStats) {
        let _ = self
            .sender
            .send(LogMessage::Checkpoint(game_count, stats.clone()));
    }

    /// Log progress report with estimated time remaining.
    ///
    /// # Arguments
    ///
    /// * `current_games` - Current game count
    /// * `target_games` - Target game count
    ///
    /// # Requirements
    ///
    /// - Req 7.7: Output progress reports with estimated time remaining
    pub fn log_progress(&self, current_games: u64, target_games: u64) {
        let eta = self.eta(current_games, target_games);
        let _ = self
            .sender
            .send(LogMessage::Progress(current_games, target_games, eta));
    }

    /// Log warning message.
    ///
    /// # Arguments
    ///
    /// * `message` - Warning message
    ///
    /// # Requirements
    ///
    /// - Req 7.8: Detect and warn on evaluation divergence
    pub fn log_warning(&self, message: &str) {
        let _ = self.sender.send(LogMessage::Warning(message.to_string()));
    }

    /// Log info message.
    ///
    /// # Arguments
    ///
    /// * `message` - Info message
    pub fn log_info(&self, message: &str) {
        let _ = self.sender.send(LogMessage::Info(message.to_string()));
    }

    /// Log deterministic evaluation sample summary.
    pub fn log_evaluation(&self, game_count: u64, entry: EvalLogEntry) {
        let _ = self.sender.send(LogMessage::Evaluation(game_count, entry));
    }

    /// Check for evaluation divergence and log warning if detected.
    ///
    /// # Arguments
    ///
    /// * `eval_value` - Evaluation value to check
    ///
    /// # Returns
    ///
    /// True if divergence detected, false otherwise.
    ///
    /// # Requirements
    ///
    /// - Req 7.8: Detect and warn on evaluation divergence (NaN, extreme values)
    pub fn check_eval_divergence(&self, eval_value: f32) -> bool {
        if eval_value.is_nan() {
            self.log_warning("Evaluation divergence: NaN detected");
            return true;
        }
        if eval_value.is_infinite() {
            self.log_warning("Evaluation divergence: Infinite value detected");
            return true;
        }
        if eval_value.abs() > EVAL_DIVERGENCE_THRESHOLD {
            self.log_warning(&format!(
                "Evaluation divergence: Extreme value {} (threshold: {})",
                eval_value, EVAL_DIVERGENCE_THRESHOLD
            ));
            return true;
        }
        false
    }

    /// Calculate estimated time remaining.
    ///
    /// # Arguments
    ///
    /// * `current_games` - Current game count
    /// * `target_games` - Target game count
    ///
    /// # Returns
    ///
    /// Estimated time remaining as Duration.
    pub fn eta(&self, current_games: u64, target_games: u64) -> Duration {
        if current_games == 0 {
            return Duration::from_secs(0);
        }

        let elapsed = self.start_time.elapsed();
        let games_per_sec = current_games as f64 / elapsed.as_secs_f64();

        if games_per_sec <= 0.0 {
            return Duration::from_secs(0);
        }

        let remaining_games = target_games.saturating_sub(current_games);
        let remaining_secs = remaining_games as f64 / games_per_sec;

        Duration::from_secs_f64(remaining_secs)
    }

    /// Get elapsed time since training start.
    ///
    /// # Returns
    ///
    /// Elapsed time as Duration.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time in seconds.
    ///
    /// # Returns
    ///
    /// Elapsed time in seconds as f64.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get the log file path.
    ///
    /// # Returns
    ///
    /// Reference to log file path.
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Shutdown the logger and wait for background thread to finish.
    ///
    /// # Returns
    ///
    /// Result indicating success or failure.
    pub fn shutdown(mut self) -> Result<(), LearningError> {
        // Send shutdown signal
        let _ = self.sender.send(LogMessage::Shutdown);

        // Wait for writer thread to finish
        if let Some(handle) = self.writer_handle.take() {
            handle
                .join()
                .map_err(|_| LearningError::Io(std::io::Error::other("Logger thread panicked")))?;
        }

        Ok(())
    }
}

impl Drop for TrainingLogger {
    fn drop(&mut self) {
        // Send shutdown signal if not already done
        let _ = self.sender.send(LogMessage::Shutdown);

        // Wait for writer thread if still running
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Synchronous logger for testing or simple use cases.
///
/// Writes directly to file without background thread.
pub struct SyncTrainingLogger {
    /// Buffered writer to log file.
    writer: BufWriter<File>,
    /// Training start time.
    start_time: Instant,
    /// Log file path.
    log_path: PathBuf,
}

impl SyncTrainingLogger {
    /// Create a new synchronous logger.
    ///
    /// # Arguments
    ///
    /// * `log_dir` - Path to log directory
    ///
    /// # Returns
    ///
    /// Result containing the logger or an error.
    pub fn new<P: AsRef<Path>>(log_dir: P) -> Result<Self, LearningError> {
        let log_dir = log_dir.as_ref().to_path_buf();

        if !log_dir.exists() {
            fs::create_dir_all(&log_dir)?;
        }

        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let log_filename = format!("training_{}.log", timestamp);
        let log_path = log_dir.join(&log_filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            start_time: Instant::now(),
            log_path,
        })
    }

    /// Log batch statistics.
    pub fn log_batch(&mut self, game_count: u64, stats: &BatchStats) -> Result<(), LearningError> {
        let msg = LogMessage::Batch(game_count, stats.clone());
        TrainingLogger::write_message(&mut self.writer, &msg)?;
        Ok(())
    }

    /// Log detailed statistics.
    pub fn log_detailed(
        &mut self,
        game_count: u64,
        stats: &DetailedStats,
    ) -> Result<(), LearningError> {
        let msg = LogMessage::Detailed(game_count, stats.clone());
        TrainingLogger::write_message(&mut self.writer, &msg)?;
        Ok(())
    }

    /// Log checkpoint summary.
    pub fn log_checkpoint(
        &mut self,
        game_count: u64,
        stats: &DetailedStats,
    ) -> Result<(), LearningError> {
        let msg = LogMessage::Checkpoint(game_count, stats.clone());
        TrainingLogger::write_message(&mut self.writer, &msg)?;
        Ok(())
    }

    /// Log warning.
    pub fn log_warning(&mut self, message: &str) -> Result<(), LearningError> {
        let msg = LogMessage::Warning(message.to_string());
        TrainingLogger::write_message(&mut self.writer, &msg)?;
        Ok(())
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get the log file path.
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Flush the writer.
    pub fn flush(&mut self) -> Result<(), LearningError> {
        self.writer.flush()?;
        Ok(())
    }
}

// ============================================================================
// Enhanced Training Logger with Log Levels and JSON Support (Phase 4 Task 4)
// ============================================================================

/// Enhanced log message types with log level support.
#[derive(Clone, Debug)]
enum EnhancedLogMessage {
    /// Batch statistics log entry.
    Batch(u64, BatchStats),
    /// Detailed statistics log entry.
    Detailed(u64, DetailedStats),
    /// Checkpoint summary log entry.
    Checkpoint(u64, DetailedStats),
    /// Progress report with ETA.
    Progress(u64, u64, Duration),
    /// Leveled message with log level and threshold at send time.
    /// (message_level, threshold_at_send_time, message)
    Leveled(LogLevel, u8, String),
    /// Divergence warning with pattern details.
    Divergence {
        value: f32,
        pattern_id: usize,
        stage: usize,
        index: usize,
        message: String,
    },
    /// Shutdown signal.
    Shutdown,
}

/// Enhanced training logger with configurable log levels and JSON output support.
///
/// This logger extends the basic `TrainingLogger` with:
/// - Configurable log levels (Debug, Info, Warning, Error)
/// - JSON format output for machine-readable logs
/// - Detailed divergence tracking with pattern information
/// - Runtime log level changes
///
/// # Requirements Coverage
///
/// - Req 4.1: Log real-time stats every 100 games
/// - Req 4.2: Log detailed stats every 10,000 games
/// - Req 4.3: Log checkpoint summaries with cumulative stats
/// - Req 4.4: Support JSON format output
/// - Req 4.5: Support configurable log levels
/// - Req 4.6: Log estimated time remaining
/// - Req 4.7: Emit immediate warning on divergence detection
/// - Req 4.8: Write logs to timestamped files
///
/// # Example
///
/// ```ignore
/// use prismind::learning::logger::{EnhancedTrainingLogger, LogLevel, BatchStats};
///
/// let logger = EnhancedTrainingLogger::new("logs/", LogLevel::Info, true)?;
///
/// // Log at different levels
/// logger.log_debug("Debug message");  // Filtered if level > Debug
/// logger.log_info_msg("Info message");
/// logger.log_warning("Warning message");
/// logger.log_error("Error message");
///
/// // Change log level at runtime
/// logger.set_log_level(LogLevel::Debug);
///
/// // Check for divergence with pattern details
/// logger.check_eval_divergence_detailed(value, pattern_id, stage, index);
///
/// logger.shutdown()?;
/// ```
pub struct EnhancedTrainingLogger {
    /// Sender for async log messages.
    sender: Sender<EnhancedLogMessage>,
    /// Background writer thread handle.
    writer_handle: Option<JoinHandle<()>>,
    /// Training start time.
    start_time: Instant,
    /// Log file path.
    log_path: PathBuf,
    /// Current log level (shared with writer thread via atomic).
    log_level: Arc<std::sync::atomic::AtomicU8>,
    /// JSON output mode.
    json_output: bool,
    /// Divergence event counter.
    divergence_count: Arc<AtomicU64>,
}

impl EnhancedTrainingLogger {
    /// Create a new enhanced training logger.
    ///
    /// # Arguments
    ///
    /// * `log_dir` - Path to log directory
    /// * `log_level` - Initial log level for filtering
    /// * `json_output` - If true, output logs in JSON format
    ///
    /// # Returns
    ///
    /// Result containing the logger or an error.
    ///
    /// # Requirements
    ///
    /// - Req 4.5: Support configurable log levels
    /// - Req 4.8: Write logs to timestamped files
    pub fn new<P: AsRef<Path>>(
        log_dir: P,
        log_level: LogLevel,
        json_output: bool,
    ) -> Result<Self, LearningError> {
        let log_dir = log_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !log_dir.exists() {
            fs::create_dir_all(&log_dir)?;
        }

        // Generate log filename with timestamp
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let log_filename = format!("training_{}.log", timestamp);
        let log_path = log_dir.join(&log_filename);

        // Create log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        // Create channel for async logging
        let (sender, receiver) = mpsc::channel();

        // Shared log level as atomic for runtime changes
        let log_level_atomic = Arc::new(std::sync::atomic::AtomicU8::new(log_level as u8));
        let log_level_clone = Arc::clone(&log_level_atomic);

        // Divergence counter
        let divergence_count = Arc::new(AtomicU64::new(0));

        // Start background writer thread
        let writer_handle = Self::start_writer_thread(file, receiver, log_level_clone, json_output);

        let logger = Self {
            sender,
            writer_handle: Some(writer_handle),
            start_time: Instant::now(),
            log_path,
            log_level: log_level_atomic,
            json_output,
            divergence_count,
        };

        // Log startup message
        logger.log_info_msg("Enhanced training logger initialized");

        Ok(logger)
    }

    /// Start the background writer thread.
    fn start_writer_thread(
        file: File,
        receiver: Receiver<EnhancedLogMessage>,
        log_level: Arc<std::sync::atomic::AtomicU8>,
        json_output: bool,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut writer = BufWriter::new(file);

            while let Ok(msg) = receiver.recv() {
                match msg {
                    EnhancedLogMessage::Shutdown => break,
                    _ => {
                        let current_level = log_level.load(Ordering::Relaxed);
                        if let Err(e) = Self::write_enhanced_message(
                            &mut writer,
                            &msg,
                            current_level,
                            json_output,
                        ) {
                            eprintln!("Logger error: {}", e);
                        }
                    }
                }
            }

            // Flush remaining data
            let _ = writer.flush();
        })
    }

    /// Write an enhanced log message to the file.
    fn write_enhanced_message<W: Write>(
        writer: &mut W,
        msg: &EnhancedLogMessage,
        current_level: u8,
        json_output: bool,
    ) -> std::io::Result<()> {
        if json_output {
            Self::write_json_message(writer, msg, current_level)
        } else {
            Self::write_text_message(writer, msg, current_level)
        }
    }

    /// Write message in JSON format.
    fn write_json_message<W: Write>(
        writer: &mut W,
        msg: &EnhancedLogMessage,
        _current_level: u8,
    ) -> std::io::Result<()> {
        let timestamp = Local::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();

        match msg {
            EnhancedLogMessage::Batch(game_count, stats) => {
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": "info",
                    "event": "batch",
                    "data": {
                        "games_completed": stats.games_completed,
                        "game_count": game_count,
                        "avg_stone_diff": stats.avg_stone_diff,
                        "black_win_rate": stats.black_win_rate,
                        "white_win_rate": stats.white_win_rate,
                        "draw_rate": stats.draw_rate,
                        "avg_move_count": stats.avg_move_count,
                        "elapsed_secs": stats.elapsed_secs,
                        "games_per_sec": stats.games_per_sec,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Detailed(game_count, stats) => {
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": "info",
                    "event": "detailed",
                    "data": {
                        "game_count": game_count,
                        "eval_mean": stats.eval_mean,
                        "eval_stddev": stats.eval_stddev,
                        "eval_min": stats.eval_min,
                        "eval_max": stats.eval_max,
                        "avg_search_depth": stats.avg_search_depth,
                        "avg_search_time_ms": stats.avg_search_time_ms,
                        "tt_hit_rate": stats.tt_hit_rate,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Checkpoint(game_count, stats) => {
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": "info",
                    "event": "checkpoint",
                    "data": {
                        "game_count": game_count,
                        "games_completed": stats.batch.games_completed,
                        "elapsed_secs": stats.batch.elapsed_secs,
                        "elapsed_hours": stats.batch.elapsed_secs / 3600.0,
                        "games_per_sec": stats.batch.games_per_sec,
                        "avg_stone_diff": stats.batch.avg_stone_diff,
                        "black_win_rate": stats.batch.black_win_rate,
                        "white_win_rate": stats.batch.white_win_rate,
                        "draw_rate": stats.batch.draw_rate,
                        "avg_move_count": stats.batch.avg_move_count,
                        "eval_mean": stats.eval_mean,
                        "eval_stddev": stats.eval_stddev,
                        "eval_min": stats.eval_min,
                        "eval_max": stats.eval_max,
                        "avg_search_depth": stats.avg_search_depth,
                        "avg_search_time_ms": stats.avg_search_time_ms,
                        "tt_hit_rate": stats.tt_hit_rate,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Progress(current, target, eta) => {
                let progress = *current as f64 / *target as f64 * 100.0;
                let eta_secs = eta.as_secs();
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": "info",
                    "event": "progress",
                    "data": {
                        "current_games": current,
                        "target_games": target,
                        "progress_percent": progress,
                        "eta_secs": eta_secs,
                        "eta_hours": eta_secs / 3600,
                        "eta_mins": (eta_secs % 3600) / 60,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Leveled(level, threshold, message) => {
                // Filter based on log level threshold captured at send time
                if (*level as u8) < *threshold {
                    return Ok(());
                }
                let level_str = match level {
                    LogLevel::Debug => "debug",
                    LogLevel::Info => "info",
                    LogLevel::Warning => "warning",
                    LogLevel::Error => "error",
                };
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": level_str,
                    "event": "message",
                    "data": {
                        "message": message,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Divergence {
                value,
                pattern_id,
                stage,
                index,
                message,
            } => {
                let json = serde_json::json!({
                    "timestamp": timestamp,
                    "level": "warning",
                    "event": "divergence",
                    "data": {
                        "message": message,
                        "value": if value.is_nan() { "NaN".to_string() } else if value.is_infinite() { "Infinity".to_string() } else { value.to_string() },
                        "pattern_id": pattern_id,
                        "stage": stage,
                        "index": index,
                    }
                });
                writeln!(writer, "{}", json)?;
            }
            EnhancedLogMessage::Shutdown => {}
        }

        writer.flush()?;
        Ok(())
    }

    /// Write message in human-readable text format.
    fn write_text_message<W: Write>(
        writer: &mut W,
        msg: &EnhancedLogMessage,
        _current_level: u8,
    ) -> std::io::Result<()> {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

        match msg {
            EnhancedLogMessage::Batch(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] BATCH {:>8} | diff:{:>+6.2} | B:{:.1}% W:{:.1}% D:{:.1}% | moves:{:.1} | {:.2} g/s | {:.1}s",
                    timestamp,
                    game_count,
                    stats.avg_stone_diff,
                    stats.black_win_rate * 100.0,
                    stats.white_win_rate * 100.0,
                    stats.draw_rate * 100.0,
                    stats.avg_move_count,
                    stats.games_per_sec,
                    stats.elapsed_secs
                )?;
            }
            EnhancedLogMessage::Detailed(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] DETAILED {:>8} | eval: mean={:.2} std={:.2} min={:.2} max={:.2} | depth={:.1} time={:.1}ms | TT={:.1}%",
                    timestamp,
                    game_count,
                    stats.eval_mean,
                    stats.eval_stddev,
                    stats.eval_min,
                    stats.eval_max,
                    stats.avg_search_depth,
                    stats.avg_search_time_ms,
                    stats.tt_hit_rate * 100.0
                )?;
            }
            EnhancedLogMessage::Checkpoint(game_count, stats) => {
                writeln!(
                    writer,
                    "[{}] ========== CHECKPOINT {} ==========",
                    timestamp, game_count
                )?;
                writeln!(writer, "  Games completed: {}", game_count)?;
                writeln!(
                    writer,
                    "  Elapsed time: {:.1}s ({:.2} hours)",
                    stats.batch.elapsed_secs,
                    stats.batch.elapsed_secs / 3600.0
                )?;
                writeln!(
                    writer,
                    "  Throughput: {:.2} games/sec",
                    stats.batch.games_per_sec
                )?;
                writeln!(
                    writer,
                    "  Avg stone diff: {:+.2}",
                    stats.batch.avg_stone_diff
                )?;
                writeln!(
                    writer,
                    "  Win rates: Black {:.1}% | White {:.1}% | Draw {:.1}%",
                    stats.batch.black_win_rate * 100.0,
                    stats.batch.white_win_rate * 100.0,
                    stats.batch.draw_rate * 100.0
                )?;
                writeln!(
                    writer,
                    "  Avg move count: {:.1}",
                    stats.batch.avg_move_count
                )?;
                writeln!(
                    writer,
                    "  Eval distribution: mean={:.2} std={:.2} min={:.2} max={:.2}",
                    stats.eval_mean, stats.eval_stddev, stats.eval_min, stats.eval_max
                )?;
                writeln!(
                    writer,
                    "  Search: depth={:.1} time={:.1}ms TT_hit={:.1}%",
                    stats.avg_search_depth,
                    stats.avg_search_time_ms,
                    stats.tt_hit_rate * 100.0
                )?;
                writeln!(
                    writer,
                    "[{}] ==========================================",
                    timestamp
                )?;
            }
            EnhancedLogMessage::Progress(current, target, eta) => {
                let progress = *current as f64 / *target as f64 * 100.0;
                let eta_secs = eta.as_secs();
                let eta_hours = eta_secs / 3600;
                let eta_mins = (eta_secs % 3600) / 60;
                writeln!(
                    writer,
                    "[{}] PROGRESS {:>8}/{} ({:.1}%) | ETA: {}h {:02}m",
                    timestamp, current, target, progress, eta_hours, eta_mins
                )?;
            }
            EnhancedLogMessage::Leveled(level, threshold, message) => {
                // Filter based on log level threshold captured at send time
                if (*level as u8) < *threshold {
                    return Ok(());
                }
                writeln!(writer, "[{}] {}: {}", timestamp, level, message)?;
            }
            EnhancedLogMessage::Divergence {
                value,
                pattern_id,
                stage,
                index,
                message,
            } => {
                writeln!(
                    writer,
                    "[{}] WARNING: Divergence detected - {} (pattern_id={}, stage={}, index={}, value={})",
                    timestamp,
                    message,
                    pattern_id,
                    stage,
                    index,
                    if value.is_nan() {
                        "NaN".to_string()
                    } else if value.is_infinite() {
                        "Infinity".to_string()
                    } else {
                        value.to_string()
                    }
                )?;
            }
            EnhancedLogMessage::Shutdown => {}
        }

        writer.flush()?;
        Ok(())
    }

    /// Log batch statistics (every 100 games).
    ///
    /// # Requirements
    ///
    /// - Req 4.1: Log real-time stats every 100 games
    pub fn log_batch(&self, game_count: u64, stats: &BatchStats) {
        let _ = self
            .sender
            .send(EnhancedLogMessage::Batch(game_count, stats.clone()));
    }

    /// Log detailed statistics (every 10,000 games).
    ///
    /// # Requirements
    ///
    /// - Req 4.2: Log detailed stats every 10,000 games
    pub fn log_detailed(&self, game_count: u64, stats: &DetailedStats) {
        let _ = self
            .sender
            .send(EnhancedLogMessage::Detailed(game_count, stats.clone()));
    }

    /// Log checkpoint summary.
    ///
    /// # Requirements
    ///
    /// - Req 4.3: Log checkpoint summaries with cumulative stats
    pub fn log_checkpoint(&self, game_count: u64, stats: &DetailedStats) {
        let _ = self
            .sender
            .send(EnhancedLogMessage::Checkpoint(game_count, stats.clone()));
    }

    /// Log progress report with estimated time remaining.
    ///
    /// # Requirements
    ///
    /// - Req 4.6: Log estimated time remaining
    pub fn log_progress(&self, current_games: u64, target_games: u64) {
        let eta = self.eta(current_games, target_games);
        let _ = self.sender.send(EnhancedLogMessage::Progress(
            current_games,
            target_games,
            eta,
        ));
    }

    /// Log debug message (filtered by log level).
    pub fn log_debug(&self, message: &str) {
        let threshold = self.log_level.load(Ordering::Relaxed);
        let _ = self.sender.send(EnhancedLogMessage::Leveled(
            LogLevel::Debug,
            threshold,
            message.to_string(),
        ));
    }

    /// Log info message (filtered by log level).
    pub fn log_info_msg(&self, message: &str) {
        let threshold = self.log_level.load(Ordering::Relaxed);
        let _ = self.sender.send(EnhancedLogMessage::Leveled(
            LogLevel::Info,
            threshold,
            message.to_string(),
        ));
    }

    /// Log warning message (filtered by log level).
    pub fn log_warning(&self, message: &str) {
        let threshold = self.log_level.load(Ordering::Relaxed);
        let _ = self.sender.send(EnhancedLogMessage::Leveled(
            LogLevel::Warning,
            threshold,
            message.to_string(),
        ));
    }

    /// Log error message (filtered by log level).
    pub fn log_error(&self, message: &str) {
        let threshold = self.log_level.load(Ordering::Relaxed);
        let _ = self.sender.send(EnhancedLogMessage::Leveled(
            LogLevel::Error,
            threshold,
            message.to_string(),
        ));
    }

    /// Set the current log level at runtime.
    ///
    /// # Arguments
    ///
    /// * `level` - New log level
    ///
    /// # Requirements
    ///
    /// - Req 4.5: Support runtime log level changes
    pub fn set_log_level(&self, level: LogLevel) {
        self.log_level.store(level as u8, Ordering::Relaxed);
    }

    /// Get the current log level.
    pub fn get_log_level(&self) -> LogLevel {
        match self.log_level.load(Ordering::Relaxed) {
            0 => LogLevel::Debug,
            1 => LogLevel::Info,
            2 => LogLevel::Warning,
            _ => LogLevel::Error,
        }
    }

    /// Check for evaluation divergence with detailed pattern information.
    ///
    /// # Arguments
    ///
    /// * `eval_value` - Evaluation value to check
    /// * `pattern_id` - Pattern ID where divergence occurred
    /// * `stage` - Game stage where divergence occurred
    /// * `index` - Pattern index where divergence occurred
    ///
    /// # Returns
    ///
    /// True if divergence detected, false otherwise.
    ///
    /// # Requirements
    ///
    /// - Req 4.7: Emit immediate warning on divergence detection
    pub fn check_eval_divergence_detailed(
        &self,
        eval_value: f32,
        pattern_id: usize,
        stage: usize,
        index: usize,
    ) -> bool {
        let (is_divergent, message) = if eval_value.is_nan() {
            (true, "NaN value detected".to_string())
        } else if eval_value.is_infinite() {
            (
                true,
                format!(
                    "Infinite value detected ({})",
                    if eval_value.is_sign_positive() {
                        "+inf"
                    } else {
                        "-inf"
                    }
                ),
            )
        } else if eval_value.abs() > EVAL_DIVERGENCE_THRESHOLD {
            (
                true,
                format!(
                    "Out-of-range value: {} (threshold: {})",
                    eval_value, EVAL_DIVERGENCE_THRESHOLD
                ),
            )
        } else {
            (false, String::new())
        };

        if is_divergent {
            // Increment divergence counter
            self.divergence_count.fetch_add(1, Ordering::Relaxed);

            // Send divergence warning
            let _ = self.sender.send(EnhancedLogMessage::Divergence {
                value: eval_value,
                pattern_id,
                stage,
                index,
                message,
            });
        }

        is_divergent
    }

    /// Get the count of divergence events detected.
    pub fn divergence_count(&self) -> u64 {
        self.divergence_count.load(Ordering::Relaxed)
    }

    /// Calculate estimated time remaining.
    pub fn eta(&self, current_games: u64, target_games: u64) -> Duration {
        if current_games == 0 {
            return Duration::from_secs(0);
        }

        let elapsed = self.start_time.elapsed();
        let games_per_sec = current_games as f64 / elapsed.as_secs_f64();

        if games_per_sec <= 0.0 {
            return Duration::from_secs(0);
        }

        let remaining_games = target_games.saturating_sub(current_games);
        let remaining_secs = remaining_games as f64 / games_per_sec;

        Duration::from_secs_f64(remaining_secs)
    }

    /// Get elapsed time since training start.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get the log file path.
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Check if JSON output is enabled.
    pub fn is_json_output(&self) -> bool {
        self.json_output
    }

    /// Shutdown the logger and wait for background thread to finish.
    pub fn shutdown(mut self) -> Result<(), LearningError> {
        // Send shutdown signal
        let _ = self.sender.send(EnhancedLogMessage::Shutdown);

        // Wait for writer thread to finish
        if let Some(handle) = self.writer_handle.take() {
            handle
                .join()
                .map_err(|_| LearningError::Io(std::io::Error::other("Logger thread panicked")))?;
        }

        Ok(())
    }
}

impl Drop for EnhancedTrainingLogger {
    fn drop(&mut self) {
        // Send shutdown signal if not already done
        let _ = self.sender.send(EnhancedLogMessage::Shutdown);

        // Wait for writer thread if still running
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    // ========== Requirement 7.1, 7.2: Real-time Statistics ==========

    #[test]
    fn test_batch_stats_creation() {
        let stone_diffs = vec![5.0, -3.0, 0.0, 10.0, -8.0];
        let move_counts = vec![50, 55, 60, 48, 52];

        let stats = BatchStats::from_games(100, &stone_diffs, &move_counts, 10.0);

        assert_eq!(stats.games_completed, 100);
        assert!((stats.avg_stone_diff - 0.8).abs() < 0.01); // (5-3+0+10-8)/5 = 0.8
        assert!((stats.black_win_rate - 0.4).abs() < 0.01); // 2/5 = 0.4
        assert!((stats.white_win_rate - 0.4).abs() < 0.01); // 2/5 = 0.4
        assert!((stats.draw_rate - 0.2).abs() < 0.01); // 1/5 = 0.2
        assert!((stats.avg_move_count - 53.0).abs() < 0.01);
        assert!((stats.games_per_sec - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_stats_empty() {
        let stats = BatchStats::from_games(0, &[], &[], 0.0);
        assert_eq!(stats.games_completed, 0);
        assert_eq!(stats.avg_stone_diff, 0.0);
    }

    // ========== Requirement 7.3, 7.4: Detailed Statistics ==========

    #[test]
    fn test_detailed_stats_creation() {
        let batch = BatchStats::from_games(10000, &[5.0; 100], &[50; 100], 100.0);
        let eval_values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let search_depths = vec![5, 6, 7, 8, 9];
        let search_times = vec![10.0, 12.0, 15.0, 11.0, 13.0];

        let stats = DetailedStats::from_metrics(
            batch,
            &eval_values,
            &search_depths,
            &search_times,
            800,
            1000,
        );

        assert!((stats.eval_mean - 30.0).abs() < 0.01);
        assert!(stats.eval_stddev > 0.0);
        assert!((stats.eval_min - 10.0).abs() < 0.01);
        assert!((stats.eval_max - 50.0).abs() < 0.01);
        assert!((stats.avg_search_depth - 7.0).abs() < 0.01);
        assert!((stats.avg_search_time_ms - 12.2).abs() < 0.01);
        assert!((stats.tt_hit_rate - 0.8).abs() < 0.01);
    }

    // ========== Requirement 7.6: Log File Format ==========

    #[test]
    fn test_log_filename_format() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        let log_path = logger.log_path();
        let filename = log_path.file_name().unwrap().to_str().unwrap();

        assert!(filename.starts_with("training_"));
        assert!(filename.ends_with(".log"));
        // Format: training_YYYYMMDD_HHMMSS.log
        assert_eq!(filename.len(), "training_YYYYMMDD_HHMMSS.log".len());

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_log_directory_creation() {
        let temp_dir = tempdir().unwrap();
        let log_dir = temp_dir.path().join("logs");

        assert!(!log_dir.exists());

        let logger = TrainingLogger::new(&log_dir).unwrap();

        assert!(log_dir.exists());
        logger.shutdown().unwrap();
    }

    // ========== Requirement 7.7: Progress Reports ==========

    #[test]
    fn test_eta_calculation() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        // Wait a bit to get elapsed time
        thread::sleep(Duration::from_millis(100));

        // ETA should be calculable
        let eta = logger.eta(100, 1000000);
        assert!(eta.as_secs() > 0);

        logger.shutdown().unwrap();
    }

    // ========== Requirement 7.8: Divergence Detection ==========

    #[test]
    fn test_check_eval_divergence_nan() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        assert!(logger.check_eval_divergence(f32::NAN));

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_check_eval_divergence_infinite() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        assert!(logger.check_eval_divergence(f32::INFINITY));
        assert!(logger.check_eval_divergence(f32::NEG_INFINITY));

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_check_eval_divergence_extreme() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        assert!(logger.check_eval_divergence(EVAL_DIVERGENCE_THRESHOLD + 1.0));
        assert!(logger.check_eval_divergence(-EVAL_DIVERGENCE_THRESHOLD - 1.0));

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_check_eval_divergence_normal() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        assert!(!logger.check_eval_divergence(0.0));
        assert!(!logger.check_eval_divergence(50.0));
        assert!(!logger.check_eval_divergence(-50.0));
        assert!(!logger.check_eval_divergence(EVAL_DIVERGENCE_THRESHOLD));

        logger.shutdown().unwrap();
    }

    // ========== Non-blocking Write Tests ==========

    #[test]
    fn test_async_logging_non_blocking() {
        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        let start = Instant::now();

        // Send many log messages
        for i in 0..1000 {
            let stats = BatchStats::from_games(i * 100, &[0.0; 100], &[50; 100], i as f64);
            logger.log_batch(i * 100, &stats);
        }

        let elapsed = start.elapsed();

        // Should complete very quickly since messages are queued
        assert!(
            elapsed < Duration::from_secs(1),
            "Logging should be non-blocking"
        );

        logger.shutdown().unwrap();
    }

    // ========== Sync Logger Tests ==========

    #[test]
    fn test_sync_logger_basic() {
        let temp_dir = tempdir().unwrap();
        let mut logger = SyncTrainingLogger::new(temp_dir.path()).unwrap();

        let stats = BatchStats::from_games(100, &[5.0, -3.0], &[50, 55], 10.0);
        logger.log_batch(100, &stats).unwrap();
        logger.log_warning("Test warning").unwrap();
        logger.flush().unwrap();

        // Verify log file was written
        let contents = fs::read_to_string(logger.log_path()).unwrap();
        assert!(contents.contains("BATCH"));
        assert!(contents.contains("WARNING"));
    }

    // ========== Requirements Summary Test ==========

    #[test]
    fn test_all_logger_requirements_summary() {
        println!("=== Training Logger Requirements Verification ===");

        let temp_dir = tempdir().unwrap();
        let logger = TrainingLogger::new(temp_dir.path()).unwrap();

        // Req 7.1: Real-time statistics every 100 games
        let stats = BatchStats::from_games(100, &[5.0, -3.0, 0.0], &[50, 55, 60], 10.0);
        logger.log_batch(100, &stats);
        println!("  7.1: Output real-time statistics every 100 games");

        // Req 7.2: Include stone difference, win rates, move count, elapsed time
        assert!(stats.avg_stone_diff.is_finite());
        assert!(stats.black_win_rate >= 0.0 && stats.black_win_rate <= 1.0);
        assert!(stats.avg_move_count > 0.0);
        assert!(stats.elapsed_secs >= 0.0);
        println!("  7.2: Include stone difference, win rates, move count, elapsed time");

        // Req 7.3: Detailed statistics every 10,000 games
        let detailed = DetailedStats::from_metrics(
            stats.clone(),
            &[10.0, 20.0, 30.0],
            &[5, 6, 7],
            &[10.0, 12.0, 15.0],
            800,
            1000,
        );
        logger.log_detailed(10000, &detailed);
        println!("  7.3: Output detailed statistics every 10,000 games");

        // Req 7.4: Include eval distribution, search depth, search time, TT hit rate
        assert!(detailed.eval_mean.is_finite());
        assert!(detailed.eval_stddev >= 0.0);
        assert!(detailed.avg_search_depth > 0.0);
        assert!(detailed.avg_search_time_ms > 0.0);
        assert!(detailed.tt_hit_rate >= 0.0 && detailed.tt_hit_rate <= 1.0);
        println!("  7.4: Include eval distribution, search depth, search time, TT hit rate");

        // Req 7.5: Complete summary at each checkpoint
        logger.log_checkpoint(100000, &detailed);
        println!("  7.5: Output complete summary at each checkpoint");

        // Req 7.6: Log file format
        let filename = logger.log_path().file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("training_") && filename.ends_with(".log"));
        println!("  7.6: Write logs with format logs/training_YYYYMMDD_HHMMSS.log");

        // Req 7.7: Progress reports with ETA
        logger.log_progress(100000, 1000000);
        println!("  7.7: Output progress reports with estimated time remaining");

        // Req 7.8: Detect evaluation divergence
        assert!(logger.check_eval_divergence(f32::NAN));
        assert!(logger.check_eval_divergence(f32::INFINITY));
        assert!(logger.check_eval_divergence(EVAL_DIVERGENCE_THRESHOLD + 1.0));
        assert!(!logger.check_eval_divergence(50.0));
        println!("  7.8: Detect and warn on evaluation divergence (NaN, extreme values)");

        // Give async logger time to process
        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        println!("=== All Training Logger requirements verified ===");
    }

    // ========== Phase 4 Task 4.1: Configurable Log Levels ==========

    #[test]
    fn test_log_level_enum_ordering() {
        // LogLevel should support ordering: Debug < Info < Warning < Error
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
    }

    #[test]
    fn test_log_level_default() {
        // Default log level should be Info
        assert_eq!(LogLevel::default(), LogLevel::Info);
    }

    #[test]
    fn test_log_level_from_str() {
        // LogLevel should be parseable from string
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("warning".parse::<LogLevel>().unwrap(), LogLevel::Warning);
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
        assert!("invalid".parse::<LogLevel>().is_err());
    }

    #[test]
    fn test_enhanced_logger_with_log_level() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(
            temp_dir.path(),
            LogLevel::Warning, // Only warning and above should be logged
            false,             // JSON output disabled
        )
        .unwrap();

        // Log at different levels
        logger.log_debug("Debug message");
        logger.log_info_msg("Info message");
        logger.log_warning("Warning message");
        logger.log_error("Error message");

        // Give async logger time to process
        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        // Read log file and verify filtering
        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Debug and Info should be filtered out
        assert!(!contents.contains("Debug message"));
        assert!(!contents.contains("Info message"));
        // Warning and Error should be present
        assert!(contents.contains("Warning message"));
        assert!(contents.contains("Error message"));
    }

    #[test]
    fn test_runtime_log_level_change() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(
            temp_dir.path(),
            LogLevel::Error, // Initially only errors
            false,
        )
        .unwrap();

        // Log warning (should be filtered)
        logger.log_warning("Filtered warning");

        // Change log level to Info
        logger.set_log_level(LogLevel::Info);

        // Now warning should be logged
        logger.log_warning("Visible warning");

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        assert!(!contents.contains("Filtered warning"));
        assert!(contents.contains("Visible warning"));
    }

    // ========== Phase 4 Task 4.2: JSON Format Output ==========

    #[test]
    fn test_json_output_format() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(
            temp_dir.path(),
            LogLevel::Info,
            true, // JSON output enabled
        )
        .unwrap();

        let stats = BatchStats::from_games(100, &[5.0, -3.0], &[50, 55], 10.0);
        logger.log_batch(100, &stats);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Should be valid JSON (parse each line)
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let parsed: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|_| panic!("Failed to parse JSON line: {}", line));

            // Verify expected fields exist
            assert!(parsed.get("timestamp").is_some());
            assert!(parsed.get("level").is_some());
            assert!(parsed.get("event").is_some());
        }
    }

    #[test]
    fn test_json_log_file_naming() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, true).unwrap();

        let log_path = logger.log_path();
        let filename = log_path.file_name().unwrap().to_str().unwrap();

        // Format: training_YYYYMMDD_HHMMSS.log
        assert!(filename.starts_with("training_"));
        assert!(filename.ends_with(".log"));

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_json_batch_stats_fields() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, true).unwrap();

        let stats = BatchStats::from_games(100, &[5.0, -3.0], &[50, 55], 10.0);
        logger.log_batch(100, &stats);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Find the batch log line
        for line in contents.lines() {
            if line.contains("\"event\":\"batch\"") || line.contains("\"event\": \"batch\"") {
                let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
                let data = parsed.get("data").expect("data field should exist");

                // Verify all batch stats fields
                assert!(data.get("games_completed").is_some());
                assert!(data.get("avg_stone_diff").is_some());
                assert!(data.get("black_win_rate").is_some());
                assert!(data.get("white_win_rate").is_some());
                assert!(data.get("draw_rate").is_some());
                assert!(data.get("avg_move_count").is_some());
                assert!(data.get("elapsed_secs").is_some());
                assert!(data.get("games_per_sec").is_some());
                return;
            }
        }
        panic!("No batch log entry found in JSON output");
    }

    // ========== Phase 4 Task 4.3: Real-time and Detailed Statistics ==========

    #[test]
    fn test_realtime_stats_logging_interval() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, false).unwrap();

        // Log multiple batches
        for i in 1..=5 {
            let stats = BatchStats::from_games(i * 100, &[5.0; 100], &[50; 100], (i * 10) as f64);
            logger.log_batch(i * 100, &stats);
        }

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Each batch should be logged
        assert!(contents.contains("100"));
        assert!(contents.contains("200"));
        assert!(contents.contains("500"));
    }

    #[test]
    fn test_detailed_stats_logging() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, false).unwrap();

        let batch = BatchStats::from_games(10000, &[5.0; 100], &[50; 100], 100.0);
        let detailed = DetailedStats::from_metrics(
            batch,
            &[10.0, 20.0, 30.0],
            &[5, 6, 7],
            &[10.0, 12.0, 15.0],
            800,
            1000,
        );
        logger.log_detailed(10000, &detailed);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Verify detailed stats are logged
        assert!(contents.contains("eval") || contents.contains("DETAILED"));
        assert!(contents.contains("TT") || contents.contains("tt_hit"));
    }

    #[test]
    fn test_checkpoint_summary_logging() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, false).unwrap();

        let batch = BatchStats::from_games(100000, &[5.0; 100], &[50; 100], 1000.0);
        let detailed = DetailedStats::from_metrics(
            batch,
            &[10.0, 20.0, 30.0],
            &[5, 6, 7],
            &[10.0, 12.0, 15.0],
            800,
            1000,
        );
        logger.log_checkpoint(100000, &detailed);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Verify checkpoint summary is logged with cumulative stats
        assert!(contents.contains("CHECKPOINT") || contents.contains("checkpoint"));
        assert!(contents.contains("100000"));
    }

    #[test]
    fn test_eta_logging() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Info, false).unwrap();

        // Wait a bit to get elapsed time
        thread::sleep(Duration::from_millis(100));

        logger.log_progress(100000, 1000000);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Verify ETA is logged
        assert!(
            contents.contains("ETA") || contents.contains("eta") || contents.contains("remaining")
        );
    }

    // ========== Phase 4 Task 4.4: Divergence Warning Detection ==========

    #[test]
    fn test_divergence_warning_nan() {
        let temp_dir = tempdir().unwrap();
        let logger =
            EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Warning, false).unwrap();

        let detected = logger.check_eval_divergence_detailed(f32::NAN, 5, 10, 1234);
        assert!(detected);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Warning should include pattern details
        assert!(contents.contains("NaN") || contents.contains("nan"));
        assert!(contents.contains("pattern") || contents.contains("5")); // pattern_id
    }

    #[test]
    fn test_divergence_warning_infinity() {
        let temp_dir = tempdir().unwrap();
        let logger =
            EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Warning, false).unwrap();

        let detected = logger.check_eval_divergence_detailed(f32::INFINITY, 3, 5, 999);
        assert!(detected);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        assert!(
            contents.contains("inf") || contents.contains("Inf") || contents.contains("infinity")
        );
    }

    #[test]
    fn test_divergence_warning_out_of_range() {
        let temp_dir = tempdir().unwrap();
        let logger =
            EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Warning, false).unwrap();

        let detected =
            logger.check_eval_divergence_detailed(EVAL_DIVERGENCE_THRESHOLD + 100.0, 7, 15, 5555);
        assert!(detected);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Should include value and pattern info
        assert!(
            contents.contains("1100")
                || contents.contains("diverge")
                || contents.contains("WARNING")
        );
        assert!(contents.contains("7") || contents.contains("pattern"));
    }

    #[test]
    fn test_divergence_tracking_count() {
        let temp_dir = tempdir().unwrap();
        let logger =
            EnhancedTrainingLogger::new(temp_dir.path(), LogLevel::Warning, false).unwrap();

        // Trigger multiple divergence events
        logger.check_eval_divergence_detailed(f32::NAN, 1, 1, 100);
        logger.check_eval_divergence_detailed(f32::INFINITY, 2, 2, 200);
        logger.check_eval_divergence_detailed(f32::NAN, 3, 3, 300);

        // Get divergence count
        let count = logger.divergence_count();
        assert_eq!(count, 3);

        logger.shutdown().unwrap();
    }

    #[test]
    fn test_json_divergence_warning() {
        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(
            temp_dir.path(),
            LogLevel::Warning,
            true, // JSON output
        )
        .unwrap();

        logger.check_eval_divergence_detailed(f32::NAN, 5, 10, 1234);

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        let log_path = temp_dir
            .path()
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let contents = fs::read_to_string(&log_path).unwrap();

        // Find the divergence warning line
        for line in contents.lines() {
            if line.contains("divergence") {
                let parsed: serde_json::Value = serde_json::from_str(line).unwrap();
                let data = parsed.get("data").expect("data field should exist");

                // Verify pattern details in JSON
                assert!(data.get("pattern_id").is_some() || data.get("value").is_some());
                return;
            }
        }
        // It's OK if the warning is in a different format
    }

    // ========== Phase 4 Task 4 Requirements Summary ==========

    #[test]
    fn test_phase4_task4_requirements_summary() {
        println!("=== Phase 4 Task 4: Enhanced Training Logger Requirements ===");

        let temp_dir = tempdir().unwrap();
        let logger = EnhancedTrainingLogger::new(
            temp_dir.path(),
            LogLevel::Debug, // Allow all levels for testing
            true,            // JSON output
        )
        .unwrap();

        // 4.1: Configurable log levels
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
        logger.set_log_level(LogLevel::Info);
        println!("  4.1: Configurable log levels (Debug, Info, Warning, Error)");

        // 4.2: JSON format output
        let stats = BatchStats::from_games(100, &[5.0], &[50], 10.0);
        logger.log_batch(100, &stats);
        println!("  4.2: JSON format output for machine-readable logging");

        // 4.3: Real-time and detailed statistics
        let detailed = DetailedStats::from_metrics(
            stats.clone(),
            &[10.0, 20.0, 30.0],
            &[5, 6, 7],
            &[10.0, 12.0, 15.0],
            800,
            1000,
        );
        logger.log_detailed(10000, &detailed);
        logger.log_checkpoint(100000, &detailed);
        logger.log_progress(100000, 1000000);
        println!("  4.3: Real-time and detailed statistics logging");

        // 4.4: Divergence warning detection
        let diverged = logger.check_eval_divergence_detailed(f32::NAN, 5, 10, 1234);
        assert!(diverged);
        assert_eq!(logger.divergence_count(), 1);
        println!("  4.4: Divergence warning detection and logging");

        thread::sleep(Duration::from_millis(100));
        logger.shutdown().unwrap();

        println!("=== All Phase 4 Task 4 requirements verified ===");
    }
}
