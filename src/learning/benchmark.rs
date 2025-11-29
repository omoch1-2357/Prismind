//! Performance benchmarks for the learning module.
//!
//! This module implements performance benchmarks for verifying system targets.
//!
//! # Performance Targets
//!
//! - Game throughput: 4.6 games/second minimum
//! - TD update latency: Under 10ms per game
//! - Checkpoint save time: Under 30 seconds
//! - Search operations: 15ms per move average
//! - CPU utilization: 80% or higher
//! - Total training time: 50-60 hours for 1M games
//!
//! # Requirements Coverage
//!
//! - Req 13.1: Complete 1M games within 60 hours on 4-core ARM64
//! - Req 13.2: Achieve 4.6 games/sec throughput
//! - Req 13.3: TD updates within 10ms per game
//! - Req 13.4: Checkpoint saves within 30 seconds
//! - Req 13.7: Search operations maintain 15ms per move average

use std::time::{Duration, Instant};

/// Minimum game throughput target (games per second).
pub const TARGET_GAMES_PER_SEC: f64 = 4.6;

/// Maximum TD update latency (milliseconds).
pub const MAX_TD_UPDATE_MS: f64 = 10.0;

/// Maximum checkpoint save time (seconds).
pub const MAX_CHECKPOINT_SAVE_SECS: f64 = 30.0;

/// Target search time per move (milliseconds).
pub const TARGET_SEARCH_TIME_MS: u64 = 15;

/// Minimum CPU utilization target (percentage).
pub const MIN_CPU_UTILIZATION_PCT: f64 = 80.0;

/// Target total training time (hours).
pub const TARGET_TRAINING_HOURS: f64 = 60.0;

/// Total target games.
pub const TARGET_TOTAL_GAMES: u64 = 1_000_000;

/// Number of parallel threads.
pub const NUM_THREADS: usize = 4;

/// Performance measurement result for a benchmark.
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Total elapsed time.
    pub elapsed: Duration,
    /// Number of operations performed.
    pub operations: u64,
    /// Operations per second.
    pub ops_per_sec: f64,
    /// Average time per operation in milliseconds.
    pub avg_ms_per_op: f64,
    /// Whether the benchmark passed the target.
    pub passed: bool,
    /// Target value (for comparison).
    pub target: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result from timing data.
    ///
    /// # Arguments
    ///
    /// * `name` - Benchmark name
    /// * `elapsed` - Total elapsed time
    /// * `operations` - Number of operations
    /// * `target` - Target value (depends on benchmark type)
    /// * `is_rate_target` - True if target is ops/sec, false if target is ms/op
    pub fn new(
        name: impl Into<String>,
        elapsed: Duration,
        operations: u64,
        target: f64,
        is_rate_target: bool,
    ) -> Self {
        let elapsed_secs = elapsed.as_secs_f64();
        let ops_per_sec = if elapsed_secs > 0.0 {
            operations as f64 / elapsed_secs
        } else {
            0.0
        };
        let avg_ms_per_op = if operations > 0 {
            elapsed.as_secs_f64() * 1000.0 / operations as f64
        } else {
            0.0
        };

        let passed = if is_rate_target {
            ops_per_sec >= target
        } else {
            avg_ms_per_op <= target
        };

        Self {
            name: name.into(),
            elapsed,
            operations,
            ops_per_sec,
            avg_ms_per_op,
            passed,
            target,
        }
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        writeln!(f, "=== {} ===", self.name)?;
        writeln!(f, "  Status: {} (target: {:.2})", status, self.target)?;
        writeln!(f, "  Operations: {}", self.operations)?;
        writeln!(f, "  Elapsed: {:.3}s", self.elapsed.as_secs_f64())?;
        writeln!(f, "  Rate: {:.2} ops/sec", self.ops_per_sec)?;
        writeln!(f, "  Avg time: {:.3} ms/op", self.avg_ms_per_op)?;
        Ok(())
    }
}

/// Game throughput benchmark.
///
/// Measures the rate at which complete self-play games can be executed.
///
/// # Requirements
///
/// - Req 13.2: Achieve 4.6 games/sec throughput
pub struct GameThroughputBenchmark {
    /// Measured games per second.
    pub games_per_sec: f64,
    /// Total games executed.
    pub total_games: u64,
    /// Total elapsed time.
    pub elapsed: Duration,
}

impl GameThroughputBenchmark {
    /// Create a new throughput measurement.
    pub fn new(total_games: u64, elapsed: Duration) -> Self {
        let games_per_sec = if elapsed.as_secs_f64() > 0.0 {
            total_games as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        Self {
            games_per_sec,
            total_games,
            elapsed,
        }
    }

    /// Check if the target throughput is met.
    pub fn meets_target(&self) -> bool {
        self.games_per_sec >= TARGET_GAMES_PER_SEC
    }

    /// Convert to BenchmarkResult.
    pub fn to_result(&self) -> BenchmarkResult {
        BenchmarkResult::new(
            "Game Throughput",
            self.elapsed,
            self.total_games,
            TARGET_GAMES_PER_SEC,
            true, // rate target
        )
    }
}

/// TD update latency benchmark.
///
/// Measures the time to perform TD updates for a single game.
///
/// # Requirements
///
/// - Req 13.3: TD updates within 10ms per game
pub struct TDUpdateBenchmark {
    /// Average update time in milliseconds.
    pub avg_update_ms: f64,
    /// Maximum update time observed.
    pub max_update_ms: f64,
    /// Total updates performed.
    pub total_updates: u64,
    /// Total elapsed time.
    pub elapsed: Duration,
}

impl TDUpdateBenchmark {
    /// Create a new TD update benchmark from measurements.
    pub fn from_measurements(update_times_ms: &[f64]) -> Self {
        let total_updates = update_times_ms.len() as u64;
        let total_ms: f64 = update_times_ms.iter().sum();
        let avg_update_ms = if total_updates > 0 {
            total_ms / total_updates as f64
        } else {
            0.0
        };
        let max_update_ms = update_times_ms.iter().cloned().fold(0.0, f64::max);
        let elapsed = Duration::from_secs_f64(total_ms / 1000.0);

        Self {
            avg_update_ms,
            max_update_ms,
            total_updates,
            elapsed,
        }
    }

    /// Check if the target latency is met.
    pub fn meets_target(&self) -> bool {
        self.avg_update_ms <= MAX_TD_UPDATE_MS
    }

    /// Convert to BenchmarkResult.
    pub fn to_result(&self) -> BenchmarkResult {
        BenchmarkResult::new(
            "TD Update Latency",
            self.elapsed,
            self.total_updates,
            MAX_TD_UPDATE_MS,
            false, // latency target
        )
    }
}

/// Checkpoint save time benchmark.
///
/// Measures the time to save a complete checkpoint.
///
/// # Requirements
///
/// - Req 13.4: Checkpoint saves within 30 seconds
pub struct CheckpointBenchmark {
    /// Save time in seconds.
    pub save_time_secs: f64,
    /// Total saves performed.
    pub total_saves: u64,
    /// Total elapsed time.
    pub elapsed: Duration,
}

impl CheckpointBenchmark {
    /// Create a new checkpoint benchmark from measurements.
    pub fn from_measurements(save_times_secs: &[f64]) -> Self {
        let total_saves = save_times_secs.len() as u64;
        let total_secs: f64 = save_times_secs.iter().sum();
        let save_time_secs = if total_saves > 0 {
            total_secs / total_saves as f64
        } else {
            0.0
        };
        let elapsed = Duration::from_secs_f64(total_secs);

        Self {
            save_time_secs,
            total_saves,
            elapsed,
        }
    }

    /// Check if the target save time is met.
    pub fn meets_target(&self) -> bool {
        self.save_time_secs <= MAX_CHECKPOINT_SAVE_SECS
    }

    /// Convert to BenchmarkResult.
    pub fn to_result(&self) -> BenchmarkResult {
        BenchmarkResult::new(
            "Checkpoint Save Time",
            self.elapsed,
            self.total_saves,
            MAX_CHECKPOINT_SAVE_SECS * 1000.0, // Convert to ms for comparison
            false,                             // latency target
        )
    }
}

/// Search performance benchmark.
///
/// Measures the time for search operations per move.
///
/// # Requirements
///
/// - Req 13.7: Search operations maintain 15ms per move average
pub struct SearchBenchmark {
    /// Average search time in milliseconds.
    pub avg_search_ms: f64,
    /// Maximum search time observed.
    pub max_search_ms: f64,
    /// Total searches performed.
    pub total_searches: u64,
    /// Total elapsed time.
    pub elapsed: Duration,
}

impl SearchBenchmark {
    /// Create a new search benchmark from measurements.
    pub fn from_measurements(search_times_ms: &[f64]) -> Self {
        let total_searches = search_times_ms.len() as u64;
        let total_ms: f64 = search_times_ms.iter().sum();
        let avg_search_ms = if total_searches > 0 {
            total_ms / total_searches as f64
        } else {
            0.0
        };
        let max_search_ms = search_times_ms.iter().cloned().fold(0.0, f64::max);
        let elapsed = Duration::from_secs_f64(total_ms / 1000.0);

        Self {
            avg_search_ms,
            max_search_ms,
            total_searches,
            elapsed,
        }
    }

    /// Check if the target search time is met.
    pub fn meets_target(&self) -> bool {
        self.avg_search_ms <= TARGET_SEARCH_TIME_MS as f64
    }

    /// Convert to BenchmarkResult.
    pub fn to_result(&self) -> BenchmarkResult {
        BenchmarkResult::new(
            "Search Performance",
            self.elapsed,
            self.total_searches,
            TARGET_SEARCH_TIME_MS as f64,
            false, // latency target
        )
    }
}

/// CPU utilization measurement.
///
/// Estimates CPU utilization based on work time vs total time.
///
/// # Requirements
///
/// - Req 13.6: 80%+ CPU utilization
pub struct CPUUtilizationBenchmark {
    /// Estimated CPU utilization percentage.
    pub utilization_pct: f64,
    /// Total work time (busy time).
    pub work_time: Duration,
    /// Total elapsed time (wall time).
    pub total_time: Duration,
}

impl CPUUtilizationBenchmark {
    /// Create a new CPU utilization measurement.
    ///
    /// # Arguments
    ///
    /// * `work_time` - Total CPU work time across all threads
    /// * `total_time` - Wall clock time
    /// * `num_threads` - Number of threads used
    pub fn new(work_time: Duration, total_time: Duration, num_threads: usize) -> Self {
        let max_work_time = total_time.as_secs_f64() * num_threads as f64;
        let utilization_pct = if max_work_time > 0.0 {
            (work_time.as_secs_f64() / max_work_time) * 100.0
        } else {
            0.0
        };

        Self {
            utilization_pct,
            work_time,
            total_time,
        }
    }

    /// Check if the target utilization is met.
    pub fn meets_target(&self) -> bool {
        self.utilization_pct >= MIN_CPU_UTILIZATION_PCT
    }
}

/// Training time estimator.
///
/// Estimates total training time based on current throughput.
///
/// # Requirements
///
/// - Req 13.1: Complete 1M games within 60 hours
pub struct TrainingTimeEstimator {
    /// Current games per second.
    pub games_per_sec: f64,
    /// Estimated total hours for 1M games.
    pub estimated_hours: f64,
    /// Games completed so far.
    pub games_completed: u64,
    /// Elapsed time so far.
    pub elapsed: Duration,
}

impl TrainingTimeEstimator {
    /// Create a new training time estimator.
    pub fn new(games_completed: u64, elapsed: Duration) -> Self {
        let games_per_sec = if elapsed.as_secs_f64() > 0.0 {
            games_completed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let estimated_hours = if games_per_sec > 0.0 {
            TARGET_TOTAL_GAMES as f64 / games_per_sec / 3600.0
        } else {
            f64::INFINITY
        };

        Self {
            games_per_sec,
            estimated_hours,
            games_completed,
            elapsed,
        }
    }

    /// Check if the target training time is achievable.
    pub fn meets_target(&self) -> bool {
        self.estimated_hours <= TARGET_TRAINING_HOURS
    }

    /// Get remaining time estimate.
    pub fn remaining_hours(&self, current_games: u64) -> f64 {
        if self.games_per_sec > 0.0 {
            let remaining_games = TARGET_TOTAL_GAMES.saturating_sub(current_games);
            remaining_games as f64 / self.games_per_sec / 3600.0
        } else {
            f64::INFINITY
        }
    }
}

/// Performance benchmark runner.
///
/// Provides convenience methods for running benchmarks.
pub struct BenchmarkRunner {
    /// Collected benchmark results.
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Run a timed benchmark.
    ///
    /// # Arguments
    ///
    /// * `name` - Benchmark name
    /// * `iterations` - Number of iterations to run
    /// * `target` - Target value
    /// * `is_rate_target` - True if target is rate (higher is better)
    /// * `operation` - Operation to benchmark
    ///
    /// # Returns
    ///
    /// Benchmark result
    pub fn run_benchmark<F>(
        &mut self,
        name: &str,
        iterations: u64,
        target: f64,
        is_rate_target: bool,
        mut operation: F,
    ) -> BenchmarkResult
    where
        F: FnMut(),
    {
        let start = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        let elapsed = start.elapsed();

        let result = BenchmarkResult::new(name, elapsed, iterations, target, is_rate_target);
        self.results.push(result.clone());
        result
    }

    /// Run a benchmark with individual timing.
    ///
    /// Returns detailed timing for each iteration.
    pub fn run_detailed_benchmark<F>(
        &mut self,
        _name: &str,
        iterations: u64,
        mut operation: F,
    ) -> Vec<Duration>
    where
        F: FnMut(),
    {
        let mut timings = Vec::with_capacity(iterations as usize);

        for _ in 0..iterations {
            let start = Instant::now();
            operation();
            timings.push(start.elapsed());
        }

        timings
    }

    /// Get all passed benchmarks.
    pub fn passed(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| r.passed).collect()
    }

    /// Get all failed benchmarks.
    pub fn failed(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Check if all benchmarks passed.
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Print summary of all benchmarks.
    pub fn print_summary(&self) {
        println!("=== Benchmark Summary ===");
        for result in &self.results {
            println!("{}", result);
        }
        println!(
            "Total: {}/{} passed",
            self.passed().len(),
            self.results.len()
        );
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Task 11.1: Performance Benchmark Tests ==========

    // ----- Game Throughput Tests -----

    #[test]
    fn test_game_throughput_target_constant() {
        // Req 13.2: Target 4.6 games/second
        assert_eq!(TARGET_GAMES_PER_SEC, 4.6);
    }

    #[test]
    fn test_game_throughput_benchmark_creation() {
        let benchmark = GameThroughputBenchmark::new(100, Duration::from_secs(20));

        assert_eq!(benchmark.total_games, 100);
        assert!((benchmark.games_per_sec - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_game_throughput_meets_target() {
        // 5 games/sec >= 4.6 target
        let good = GameThroughputBenchmark::new(50, Duration::from_secs(10));
        assert!(good.meets_target());

        // 4 games/sec < 4.6 target
        let bad = GameThroughputBenchmark::new(40, Duration::from_secs(10));
        assert!(!bad.meets_target());
    }

    #[test]
    fn test_game_throughput_to_result() {
        let benchmark = GameThroughputBenchmark::new(46, Duration::from_secs(10));
        let result = benchmark.to_result();

        assert_eq!(result.name, "Game Throughput");
        assert!(result.passed);
        assert_eq!(result.target, TARGET_GAMES_PER_SEC);
    }

    // ----- TD Update Latency Tests -----

    #[test]
    fn test_td_update_latency_target_constant() {
        // Req 13.3: Target under 10ms per game
        assert_eq!(MAX_TD_UPDATE_MS, 10.0);
    }

    #[test]
    fn test_td_update_benchmark_creation() {
        let times = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let benchmark = TDUpdateBenchmark::from_measurements(&times);

        assert_eq!(benchmark.total_updates, 5);
        assert!((benchmark.avg_update_ms - 7.0).abs() < 0.001);
        assert!((benchmark.max_update_ms - 9.0).abs() < 0.001);
    }

    #[test]
    fn test_td_update_meets_target() {
        // Average 8ms <= 10ms target
        let good_times = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let good = TDUpdateBenchmark::from_measurements(&good_times);
        assert!(good.meets_target());

        // Average 12ms > 10ms target
        let bad_times = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let bad = TDUpdateBenchmark::from_measurements(&bad_times);
        assert!(!bad.meets_target());
    }

    // ----- Checkpoint Save Time Tests -----

    #[test]
    fn test_checkpoint_save_time_target_constant() {
        // Req 13.4: Target under 30 seconds
        assert_eq!(MAX_CHECKPOINT_SAVE_SECS, 30.0);
    }

    #[test]
    fn test_checkpoint_benchmark_creation() {
        let times = vec![20.0, 25.0, 22.0];
        let benchmark = CheckpointBenchmark::from_measurements(&times);

        assert_eq!(benchmark.total_saves, 3);
        assert!((benchmark.save_time_secs - 22.333).abs() < 0.01);
    }

    #[test]
    fn test_checkpoint_meets_target() {
        // Average 25s <= 30s target
        let good_times = vec![20.0, 25.0, 30.0];
        let good = CheckpointBenchmark::from_measurements(&good_times);
        assert!(good.meets_target());

        // Average 35s > 30s target
        let bad_times = vec![30.0, 35.0, 40.0];
        let bad = CheckpointBenchmark::from_measurements(&bad_times);
        assert!(!bad.meets_target());
    }

    // ----- Search Performance Tests -----

    #[test]
    fn test_search_time_target_constant() {
        // Req 13.7: Target 15ms per move
        assert_eq!(TARGET_SEARCH_TIME_MS, 15);
    }

    #[test]
    fn test_search_benchmark_creation() {
        let times = vec![10.0, 12.0, 15.0, 14.0, 11.0];
        let benchmark = SearchBenchmark::from_measurements(&times);

        assert_eq!(benchmark.total_searches, 5);
        assert!((benchmark.avg_search_ms - 12.4).abs() < 0.001);
        assert!((benchmark.max_search_ms - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_search_meets_target() {
        // Average 12ms <= 15ms target
        let good_times = vec![10.0, 12.0, 14.0];
        let good = SearchBenchmark::from_measurements(&good_times);
        assert!(good.meets_target());

        // Average 18ms > 15ms target
        let bad_times = vec![16.0, 18.0, 20.0];
        let bad = SearchBenchmark::from_measurements(&bad_times);
        assert!(!bad.meets_target());
    }

    // ----- CPU Utilization Tests -----

    #[test]
    fn test_cpu_utilization_target_constant() {
        // Req 13.6: Target 80%+ CPU utilization
        assert_eq!(MIN_CPU_UTILIZATION_PCT, 80.0);
    }

    #[test]
    fn test_cpu_utilization_calculation() {
        // 4 threads, 10 seconds wall time, 32 seconds work = 80%
        let benchmark =
            CPUUtilizationBenchmark::new(Duration::from_secs(32), Duration::from_secs(10), 4);

        assert!((benchmark.utilization_pct - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_cpu_utilization_meets_target() {
        // 85% >= 80% target
        let good =
            CPUUtilizationBenchmark::new(Duration::from_secs(34), Duration::from_secs(10), 4);
        assert!(good.meets_target());

        // 70% < 80% target
        let bad = CPUUtilizationBenchmark::new(Duration::from_secs(28), Duration::from_secs(10), 4);
        assert!(!bad.meets_target());
    }

    // ----- Training Time Estimator Tests -----

    #[test]
    fn test_training_time_target_constants() {
        // Req 13.1: Complete 1M games within 60 hours
        assert_eq!(TARGET_TOTAL_GAMES, 1_000_000);
        assert_eq!(TARGET_TRAINING_HOURS, 60.0);
    }

    #[test]
    fn test_training_time_estimator_calculation() {
        // 1000 games in 200 seconds = 5 games/sec
        // 1M games / 5 games/sec = 200,000 seconds = 55.56 hours
        let estimator = TrainingTimeEstimator::new(1000, Duration::from_secs(200));

        assert!((estimator.games_per_sec - 5.0).abs() < 0.001);
        assert!((estimator.estimated_hours - 55.56).abs() < 0.1);
    }

    #[test]
    fn test_training_time_meets_target() {
        // 5 games/sec -> ~55 hours (< 60 hours target)
        let good = TrainingTimeEstimator::new(1000, Duration::from_secs(200));
        assert!(good.meets_target());

        // 3 games/sec -> ~92 hours (> 60 hours target)
        let bad = TrainingTimeEstimator::new(600, Duration::from_secs(200));
        assert!(!bad.meets_target());
    }

    #[test]
    fn test_training_time_remaining_hours() {
        let estimator = TrainingTimeEstimator::new(1000, Duration::from_secs(200));

        // At 100,000 games, remaining = 900,000 / 5 / 3600 = 50 hours
        let remaining = estimator.remaining_hours(100_000);
        assert!((remaining - 50.0).abs() < 0.1);
    }

    // ----- Benchmark Result Tests -----

    #[test]
    fn test_benchmark_result_rate_target() {
        let result = BenchmarkResult::new(
            "Test Rate",
            Duration::from_secs(10),
            50,
            4.0,  // target 4 ops/sec
            true, // rate target
        );

        assert_eq!(result.operations, 50);
        assert!((result.ops_per_sec - 5.0).abs() < 0.001);
        assert!(result.passed); // 5 >= 4
    }

    #[test]
    fn test_benchmark_result_latency_target() {
        let result = BenchmarkResult::new(
            "Test Latency",
            Duration::from_millis(100),
            10,
            15.0,  // target 15ms/op
            false, // latency target
        );

        assert_eq!(result.operations, 10);
        assert!((result.avg_ms_per_op - 10.0).abs() < 0.001);
        assert!(result.passed); // 10ms <= 15ms
    }

    #[test]
    fn test_benchmark_result_display() {
        let result = BenchmarkResult::new("Display Test", Duration::from_secs(1), 100, 50.0, true);

        let display = format!("{}", result);
        assert!(display.contains("Display Test"));
        assert!(display.contains("PASS"));
    }

    // ----- Benchmark Runner Tests -----

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchmarkRunner::new();
        assert!(runner.results.is_empty());
    }

    #[test]
    fn test_benchmark_runner_run_benchmark() {
        let mut runner = BenchmarkRunner::new();
        let mut counter = 0;

        let result = runner.run_benchmark(
            "Counter Test",
            100,
            1000.0, // target rate
            true,
            || {
                counter += 1;
            },
        );

        assert_eq!(counter, 100);
        assert_eq!(result.operations, 100);
        assert_eq!(runner.results.len(), 1);
    }

    #[test]
    fn test_benchmark_runner_passed_failed() {
        let mut runner = BenchmarkRunner::new();

        // Add a passing benchmark (fast operation, high rate target easily met)
        runner.run_benchmark("Pass", 100, 0.1, true, || {});

        // Add a failing benchmark (impossible rate target)
        runner.run_benchmark("Fail", 1, 1_000_000.0, true, || {
            std::thread::sleep(Duration::from_millis(10));
        });

        assert_eq!(runner.passed().len(), 1);
        assert_eq!(runner.failed().len(), 1);
        assert!(!runner.all_passed());
    }

    #[test]
    fn test_benchmark_runner_detailed_benchmark() {
        let mut runner = BenchmarkRunner::new();

        let timings = runner.run_detailed_benchmark("Detailed", 5, || {
            std::thread::sleep(Duration::from_millis(1));
        });

        assert_eq!(timings.len(), 5);
        for timing in timings {
            assert!(timing >= Duration::from_millis(1));
        }
    }

    // ========== Requirements Summary Tests ==========

    #[test]
    fn test_performance_targets_summary() {
        println!("=== Task 11.1: Performance Benchmark Targets ===");

        // Req 13.1: Training time target
        assert_eq!(TARGET_TOTAL_GAMES, 1_000_000);
        assert_eq!(TARGET_TRAINING_HOURS, 60.0);
        println!(
            "  13.1: Complete {} games within {} hours",
            TARGET_TOTAL_GAMES, TARGET_TRAINING_HOURS
        );

        // Req 13.2: Game throughput target
        assert_eq!(TARGET_GAMES_PER_SEC, 4.6);
        println!(
            "  13.2: Achieve {} games/sec throughput",
            TARGET_GAMES_PER_SEC
        );

        // Req 13.3: TD update latency target
        assert_eq!(MAX_TD_UPDATE_MS, 10.0);
        println!("  13.3: TD updates within {}ms per game", MAX_TD_UPDATE_MS);

        // Req 13.4: Checkpoint save time target
        assert_eq!(MAX_CHECKPOINT_SAVE_SECS, 30.0);
        println!(
            "  13.4: Checkpoint saves within {}s",
            MAX_CHECKPOINT_SAVE_SECS
        );

        // Req 13.6: CPU utilization target
        assert_eq!(MIN_CPU_UTILIZATION_PCT, 80.0);
        println!("  13.6: CPU utilization >= {}%", MIN_CPU_UTILIZATION_PCT);

        // Req 13.7: Search time target
        assert_eq!(TARGET_SEARCH_TIME_MS, 15);
        println!(
            "  13.7: Search operations at {}ms per move",
            TARGET_SEARCH_TIME_MS
        );

        println!("=== All performance targets defined ===");
    }

    #[test]
    fn test_benchmark_structures_complete() {
        // Verify all benchmark structures are usable
        let _ = GameThroughputBenchmark::new(100, Duration::from_secs(10));
        let _ = TDUpdateBenchmark::from_measurements(&[5.0, 6.0, 7.0]);
        let _ = CheckpointBenchmark::from_measurements(&[20.0, 25.0]);
        let _ = SearchBenchmark::from_measurements(&[10.0, 12.0, 14.0]);
        let _ = CPUUtilizationBenchmark::new(Duration::from_secs(32), Duration::from_secs(10), 4);
        let _ = TrainingTimeEstimator::new(1000, Duration::from_secs(200));
        let _ = BenchmarkRunner::new();

        println!("All benchmark structures instantiated successfully");
    }
}
