//! Convergence Monitoring System.
//!
//! This module implements a training convergence monitoring system that tracks
//! learning progress and detects potential issues during self-play training.
//!
//! # Monitoring Metrics
//!
//! - Stone difference trends: Average stone difference across training
//! - Evaluation stability: Variance of evaluation values over time
//! - Pattern update coverage: Percentage of pattern entries receiving updates
//! - Win rate tracking: Win rate against baseline (random) play
//! - Stagnation detection: No improvement for extended periods
//!
//! # Requirements Coverage
//!
//! - Req 10.1: Track average stone difference trends across training
//! - Req 10.2: Track evaluation value stability (variance over time)
//! - Req 10.3: Track percentage of pattern entries receiving updates (target: 200+ avg)
//! - Req 10.4: Warn if win rate against random drops below 95%
//! - Req 10.5: Output convergence metrics every 100,000 games
//! - Req 10.6: Detect and report learning stagnation (no improvement for 50,000+ games)

use std::collections::HashSet;

/// Default interval for convergence metric output (100,000 games).
pub const CONVERGENCE_REPORT_INTERVAL: u64 = 100_000;

/// Target minimum average updates per pattern entry.
pub const TARGET_UPDATES_PER_ENTRY: u64 = 200;

/// Stagnation detection window size (50,000 games).
pub const STAGNATION_WINDOW: u64 = 50_000;

/// Minimum win rate against random baseline (95%).
pub const MIN_WIN_RATE_VS_RANDOM: f32 = 0.95;

/// Improvement threshold for stagnation detection.
/// Stone difference improvement less than this is considered no improvement.
pub const IMPROVEMENT_THRESHOLD: f32 = 0.5;

/// Window size for computing running averages.
pub const RUNNING_AVERAGE_WINDOW: usize = 1000;

/// Convergence metrics snapshot.
///
/// Contains all monitored metrics at a specific point in training.
#[derive(Clone, Debug, Default)]
pub struct ConvergenceMetrics {
    /// Total games played at this point.
    pub games_played: u64,
    /// Average stone difference in recent window.
    pub avg_stone_diff: f32,
    /// Stone difference trend (positive = improving for Black).
    pub stone_diff_trend: f32,
    /// Evaluation value mean.
    pub eval_mean: f32,
    /// Evaluation value variance (stability metric).
    pub eval_variance: f32,
    /// Total unique pattern entries updated.
    pub unique_entries_updated: u64,
    /// Total pattern entry updates.
    pub total_updates: u64,
    /// Average updates per entry (total_updates / unique_entries).
    pub avg_updates_per_entry: f64,
    /// Percentage of all pattern entries that received updates.
    pub entry_coverage_pct: f64,
    /// Win rate vs random in recent games.
    pub win_rate_vs_random: f32,
    /// Whether stagnation is detected.
    pub is_stagnating: bool,
    /// Games since last significant improvement.
    pub games_since_improvement: u64,
}

impl ConvergenceMetrics {
    /// Check if training health is good (no warnings).
    pub fn is_healthy(&self) -> bool {
        !self.is_stagnating && self.win_rate_vs_random >= MIN_WIN_RATE_VS_RANDOM
    }

    /// Get warning messages for any issues detected.
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if self.is_stagnating {
            warnings.push(format!(
                "Learning stagnation detected: no improvement for {} games",
                self.games_since_improvement
            ));
        }

        if self.win_rate_vs_random < MIN_WIN_RATE_VS_RANDOM {
            warnings.push(format!(
                "Win rate vs random ({:.1}%) below threshold ({:.1}%)",
                self.win_rate_vs_random * 100.0,
                MIN_WIN_RATE_VS_RANDOM * 100.0
            ));
        }

        warnings
    }
}

/// Convergence monitoring system for training progress tracking.
///
/// Tracks multiple metrics over time to detect convergence, stagnation,
/// and other training issues.
///
/// # Example
///
/// ```ignore
/// use prismind::learning::convergence::ConvergenceMonitor;
///
/// let total_entries = 14_000_000; // Approximate total pattern entries
/// let mut monitor = ConvergenceMonitor::new(total_entries);
///
/// // Record game results
/// monitor.record_game(5.0, &[0, 1, 2], &[0.5, 0.3, 0.1]);
///
/// // Check for stagnation
/// if monitor.is_stagnating() {
///     println!("Warning: Learning has stagnated!");
/// }
///
/// // Get convergence report
/// let metrics = monitor.get_metrics();
/// println!("Coverage: {:.1}%", metrics.entry_coverage_pct);
/// ```
#[derive(Debug)]
pub struct ConvergenceMonitor {
    /// Total number of pattern entries across all stages and patterns.
    total_pattern_entries: u64,

    /// Games completed.
    games_played: u64,

    /// Running window of stone differences.
    stone_diff_window: Vec<f32>,

    /// Running window of evaluation values.
    eval_window: Vec<f32>,

    /// Set of unique (pattern_id, stage, index) tuples that have been updated.
    /// Uses a hash set for efficient deduplication.
    unique_entries: HashSet<(usize, usize, usize)>,

    /// Total number of pattern entry updates.
    total_updates: u64,

    /// Stone difference at last improvement checkpoint.
    last_improvement_stone_diff: f32,

    /// Games at last improvement checkpoint.
    last_improvement_games: u64,

    /// Best average stone difference seen.
    best_avg_stone_diff: f32,

    /// Win rate tracking (wins / total vs random).
    wins_vs_random: u64,
    games_vs_random: u64,
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor.
    ///
    /// # Arguments
    ///
    /// * `total_pattern_entries` - Total number of pattern entries across all stages
    ///
    /// # Requirements
    ///
    /// - Req 10.3: Track percentage of entries updated
    pub fn new(total_pattern_entries: u64) -> Self {
        Self {
            total_pattern_entries,
            games_played: 0,
            stone_diff_window: Vec::with_capacity(RUNNING_AVERAGE_WINDOW),
            eval_window: Vec::with_capacity(RUNNING_AVERAGE_WINDOW),
            unique_entries: HashSet::new(),
            total_updates: 0,
            last_improvement_stone_diff: 0.0,
            last_improvement_games: 0,
            best_avg_stone_diff: f32::NEG_INFINITY,
            wins_vs_random: 0,
            games_vs_random: 0,
        }
    }

    /// Record a completed game's results.
    ///
    /// # Arguments
    ///
    /// * `stone_diff` - Final stone difference (positive = Black wins)
    /// * `updated_entries` - List of (pattern_id, stage, index) tuples that were updated
    /// * `eval_values` - Evaluation values from the game
    ///
    /// # Requirements
    ///
    /// - Req 10.1: Track stone difference trends
    /// - Req 10.2: Track evaluation stability
    /// - Req 10.3: Track pattern entry updates
    pub fn record_game(
        &mut self,
        stone_diff: f32,
        updated_entries: &[(usize, usize, usize)],
        eval_values: &[f32],
    ) {
        self.games_played += 1;

        // Update stone difference window
        if self.stone_diff_window.len() >= RUNNING_AVERAGE_WINDOW {
            self.stone_diff_window.remove(0);
        }
        self.stone_diff_window.push(stone_diff);

        // Update evaluation window
        for &eval in eval_values {
            if self.eval_window.len() >= RUNNING_AVERAGE_WINDOW {
                self.eval_window.remove(0);
            }
            self.eval_window.push(eval);
        }

        // Track unique entries and total updates
        for &entry in updated_entries {
            self.unique_entries.insert(entry);
        }
        self.total_updates += updated_entries.len() as u64;

        // Check for improvement
        let current_avg = self.avg_stone_diff();
        if current_avg > self.best_avg_stone_diff + IMPROVEMENT_THRESHOLD {
            self.best_avg_stone_diff = current_avg;
            self.last_improvement_stone_diff = current_avg;
            self.last_improvement_games = self.games_played;
        }
    }

    /// Record a game result against random baseline.
    ///
    /// # Arguments
    ///
    /// * `won` - Whether the trained AI won against random
    ///
    /// # Requirements
    ///
    /// - Req 10.4: Track win rate against random
    pub fn record_vs_random(&mut self, won: bool) {
        self.games_vs_random += 1;
        if won {
            self.wins_vs_random += 1;
        }
    }

    /// Get the current average stone difference.
    ///
    /// # Returns
    ///
    /// Average stone difference from the running window.
    ///
    /// # Requirements
    ///
    /// - Req 10.1: Track average stone difference trends
    pub fn avg_stone_diff(&self) -> f32 {
        if self.stone_diff_window.is_empty() {
            return 0.0;
        }
        self.stone_diff_window.iter().sum::<f32>() / self.stone_diff_window.len() as f32
    }

    /// Get the stone difference trend.
    ///
    /// Compares first half and second half of the window to determine trend.
    ///
    /// # Returns
    ///
    /// Positive value if improving, negative if declining.
    pub fn stone_diff_trend(&self) -> f32 {
        if self.stone_diff_window.len() < 2 {
            return 0.0;
        }

        let mid = self.stone_diff_window.len() / 2;
        let first_half: f32 = self.stone_diff_window[..mid].iter().sum::<f32>() / mid as f32;
        let second_half: f32 = self.stone_diff_window[mid..].iter().sum::<f32>()
            / (self.stone_diff_window.len() - mid) as f32;

        second_half - first_half
    }

    /// Get the evaluation value mean.
    ///
    /// # Returns
    ///
    /// Mean of evaluation values in the running window.
    pub fn eval_mean(&self) -> f32 {
        if self.eval_window.is_empty() {
            return 0.0;
        }
        self.eval_window.iter().sum::<f32>() / self.eval_window.len() as f32
    }

    /// Get the evaluation value variance.
    ///
    /// # Returns
    ///
    /// Variance of evaluation values (stability metric).
    ///
    /// # Requirements
    ///
    /// - Req 10.2: Track evaluation value stability (variance over time)
    pub fn eval_variance(&self) -> f32 {
        if self.eval_window.len() < 2 {
            return 0.0;
        }

        let mean = self.eval_mean();
        let sum_sq_diff: f32 = self.eval_window.iter().map(|&v| (v - mean).powi(2)).sum();
        sum_sq_diff / self.eval_window.len() as f32
    }

    /// Get the number of unique pattern entries updated.
    ///
    /// # Returns
    ///
    /// Count of unique (pattern_id, stage, index) combinations updated.
    ///
    /// # Requirements
    ///
    /// - Req 10.3: Track percentage of pattern entries receiving updates
    pub fn unique_entries_updated(&self) -> u64 {
        self.unique_entries.len() as u64
    }

    /// Get the total number of pattern entry updates.
    ///
    /// # Returns
    ///
    /// Total updates applied (including duplicates).
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Get the average updates per pattern entry.
    ///
    /// # Returns
    ///
    /// Average number of updates per unique entry.
    ///
    /// # Requirements
    ///
    /// - Req 10.3: Target 200+ updates per entry average
    pub fn avg_updates_per_entry(&self) -> f64 {
        let unique = self.unique_entries.len() as f64;
        if unique == 0.0 {
            return 0.0;
        }
        self.total_updates as f64 / unique
    }

    /// Get the percentage of pattern entries that have been updated.
    ///
    /// # Returns
    ///
    /// Percentage (0.0 to 100.0) of total entries that received updates.
    ///
    /// # Requirements
    ///
    /// - Req 10.3: Track percentage of pattern entries receiving updates
    pub fn entry_coverage_pct(&self) -> f64 {
        if self.total_pattern_entries == 0 {
            return 0.0;
        }
        (self.unique_entries.len() as f64 / self.total_pattern_entries as f64) * 100.0
    }

    /// Get the win rate against random baseline.
    ///
    /// # Returns
    ///
    /// Win rate as fraction (0.0 to 1.0).
    ///
    /// # Requirements
    ///
    /// - Req 10.4: Track win rate against random
    pub fn win_rate_vs_random(&self) -> f32 {
        if self.games_vs_random == 0 {
            return 1.0; // Assume 100% if no games played
        }
        self.wins_vs_random as f32 / self.games_vs_random as f32
    }

    /// Check if training is stagnating.
    ///
    /// Stagnation is detected when there's no significant improvement
    /// for STAGNATION_WINDOW games.
    ///
    /// # Returns
    ///
    /// True if learning has stagnated, false otherwise.
    ///
    /// # Requirements
    ///
    /// - Req 10.6: Detect and report learning stagnation (no improvement for 50,000+ games)
    pub fn is_stagnating(&self) -> bool {
        if self.games_played < STAGNATION_WINDOW {
            return false;
        }
        self.games_played - self.last_improvement_games >= STAGNATION_WINDOW
    }

    /// Get the number of games since last improvement.
    ///
    /// # Returns
    ///
    /// Number of games since stone difference improved significantly.
    pub fn games_since_improvement(&self) -> u64 {
        self.games_played - self.last_improvement_games
    }

    /// Check if win rate is below minimum threshold.
    ///
    /// # Returns
    ///
    /// True if win rate vs random is below 95%.
    ///
    /// # Requirements
    ///
    /// - Req 10.4: Warn if win rate against random drops below 95%
    pub fn is_win_rate_low(&self) -> bool {
        self.games_vs_random > 0 && self.win_rate_vs_random() < MIN_WIN_RATE_VS_RANDOM
    }

    /// Get complete convergence metrics snapshot.
    ///
    /// # Returns
    ///
    /// ConvergenceMetrics containing all tracked metrics.
    ///
    /// # Requirements
    ///
    /// - Req 10.5: Output convergence metrics every 100,000 games
    pub fn get_metrics(&self) -> ConvergenceMetrics {
        ConvergenceMetrics {
            games_played: self.games_played,
            avg_stone_diff: self.avg_stone_diff(),
            stone_diff_trend: self.stone_diff_trend(),
            eval_mean: self.eval_mean(),
            eval_variance: self.eval_variance(),
            unique_entries_updated: self.unique_entries_updated(),
            total_updates: self.total_updates,
            avg_updates_per_entry: self.avg_updates_per_entry(),
            entry_coverage_pct: self.entry_coverage_pct(),
            win_rate_vs_random: self.win_rate_vs_random(),
            is_stagnating: self.is_stagnating(),
            games_since_improvement: self.games_since_improvement(),
        }
    }

    /// Check if it's time to output convergence metrics.
    ///
    /// # Returns
    ///
    /// True if games_played is a multiple of CONVERGENCE_REPORT_INTERVAL.
    ///
    /// # Requirements
    ///
    /// - Req 10.5: Output convergence metrics every 100,000 games
    pub fn should_report(&self) -> bool {
        self.games_played > 0
            && self
                .games_played
                .is_multiple_of(CONVERGENCE_REPORT_INTERVAL)
    }

    /// Get the total games played.
    pub fn games_played(&self) -> u64 {
        self.games_played
    }

    /// Reset the monitor state.
    pub fn reset(&mut self) {
        self.games_played = 0;
        self.stone_diff_window.clear();
        self.eval_window.clear();
        self.unique_entries.clear();
        self.total_updates = 0;
        self.last_improvement_stone_diff = 0.0;
        self.last_improvement_games = 0;
        self.best_avg_stone_diff = f32::NEG_INFINITY;
        self.wins_vs_random = 0;
        self.games_vs_random = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Requirement 10.1: Track average stone difference trends ==========

    #[test]
    fn test_avg_stone_diff_empty() {
        let monitor = ConvergenceMonitor::new(1000);
        assert_eq!(monitor.avg_stone_diff(), 0.0);
    }

    #[test]
    fn test_avg_stone_diff_single_game() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(5.0, &[], &[]);
        assert!((monitor.avg_stone_diff() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_avg_stone_diff_multiple_games() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(10.0, &[], &[]);
        monitor.record_game(20.0, &[], &[]);
        monitor.record_game(30.0, &[], &[]);
        // Average of 10, 20, 30 = 20
        assert!((monitor.avg_stone_diff() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_stone_diff_trend_improving() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // First half: low values
        for _ in 0..5 {
            monitor.record_game(0.0, &[], &[]);
        }
        // Second half: high values
        for _ in 0..5 {
            monitor.record_game(10.0, &[], &[]);
        }
        // Trend should be positive (improving)
        assert!(monitor.stone_diff_trend() > 0.0);
    }

    #[test]
    fn test_stone_diff_trend_declining() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // First half: high values
        for _ in 0..5 {
            monitor.record_game(10.0, &[], &[]);
        }
        // Second half: low values
        for _ in 0..5 {
            monitor.record_game(0.0, &[], &[]);
        }
        // Trend should be negative (declining)
        assert!(monitor.stone_diff_trend() < 0.0);
    }

    // ========== Requirement 10.2: Track evaluation value stability ==========

    #[test]
    fn test_eval_variance_empty() {
        let monitor = ConvergenceMonitor::new(1000);
        assert_eq!(monitor.eval_variance(), 0.0);
    }

    #[test]
    fn test_eval_variance_single_value() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(0.0, &[], &[5.0]);
        // Variance with single value should be 0
        assert_eq!(monitor.eval_variance(), 0.0);
    }

    #[test]
    fn test_eval_variance_constant_values() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // All same values - variance should be 0
        for _ in 0..10 {
            monitor.record_game(0.0, &[], &[5.0, 5.0, 5.0]);
        }
        assert!(monitor.eval_variance().abs() < 0.001);
    }

    #[test]
    fn test_eval_variance_varying_values() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Varying values - variance should be > 0
        monitor.record_game(0.0, &[], &[0.0, 10.0, 20.0, 30.0]);
        assert!(monitor.eval_variance() > 0.0);
    }

    #[test]
    fn test_eval_mean() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(0.0, &[], &[10.0, 20.0, 30.0]);
        assert!((monitor.eval_mean() - 20.0).abs() < 0.001);
    }

    // ========== Requirement 10.3: Track pattern entry updates ==========

    #[test]
    fn test_unique_entries_empty() {
        let monitor = ConvergenceMonitor::new(1000);
        assert_eq!(monitor.unique_entries_updated(), 0);
    }

    #[test]
    fn test_unique_entries_single_update() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(0.0, &[(0, 0, 0)], &[]);
        assert_eq!(monitor.unique_entries_updated(), 1);
        assert_eq!(monitor.total_updates(), 1);
    }

    #[test]
    fn test_unique_entries_multiple_updates_same_entry() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Same entry updated multiple times
        monitor.record_game(0.0, &[(0, 0, 0)], &[]);
        monitor.record_game(0.0, &[(0, 0, 0)], &[]);
        monitor.record_game(0.0, &[(0, 0, 0)], &[]);
        assert_eq!(monitor.unique_entries_updated(), 1); // Still only 1 unique
        assert_eq!(monitor.total_updates(), 3); // But 3 total updates
    }

    #[test]
    fn test_unique_entries_multiple_different_entries() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(0.0, &[(0, 0, 0), (1, 0, 0), (0, 1, 0)], &[]);
        assert_eq!(monitor.unique_entries_updated(), 3);
        assert_eq!(monitor.total_updates(), 3);
    }

    #[test]
    fn test_avg_updates_per_entry() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // 2 unique entries, 6 total updates
        monitor.record_game(0.0, &[(0, 0, 0), (1, 0, 0)], &[]);
        monitor.record_game(0.0, &[(0, 0, 0), (1, 0, 0)], &[]);
        monitor.record_game(0.0, &[(0, 0, 0), (1, 0, 0)], &[]);
        assert_eq!(monitor.unique_entries_updated(), 2);
        assert_eq!(monitor.total_updates(), 6);
        assert!((monitor.avg_updates_per_entry() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_entry_coverage_pct() {
        let mut monitor = ConvergenceMonitor::new(100); // 100 total entries
        monitor.record_game(0.0, &[(0, 0, 0), (1, 0, 0), (2, 0, 0)], &[]); // 3 entries
        // 3/100 = 3%
        assert!((monitor.entry_coverage_pct() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_entry_coverage_pct_zero_total() {
        let monitor = ConvergenceMonitor::new(0);
        assert_eq!(monitor.entry_coverage_pct(), 0.0);
    }

    // ========== Requirement 10.4: Win rate against random ==========

    #[test]
    fn test_win_rate_no_games() {
        let monitor = ConvergenceMonitor::new(1000);
        // No games = 100% assumed
        assert_eq!(monitor.win_rate_vs_random(), 1.0);
    }

    #[test]
    fn test_win_rate_all_wins() {
        let mut monitor = ConvergenceMonitor::new(1000);
        for _ in 0..100 {
            monitor.record_vs_random(true);
        }
        assert!((monitor.win_rate_vs_random() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_win_rate_no_wins() {
        let mut monitor = ConvergenceMonitor::new(1000);
        for _ in 0..100 {
            monitor.record_vs_random(false);
        }
        assert_eq!(monitor.win_rate_vs_random(), 0.0);
    }

    #[test]
    fn test_win_rate_mixed() {
        let mut monitor = ConvergenceMonitor::new(1000);
        for _ in 0..80 {
            monitor.record_vs_random(true);
        }
        for _ in 0..20 {
            monitor.record_vs_random(false);
        }
        assert!((monitor.win_rate_vs_random() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_is_win_rate_low() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // 90% win rate (below 95% threshold)
        for _ in 0..90 {
            monitor.record_vs_random(true);
        }
        for _ in 0..10 {
            monitor.record_vs_random(false);
        }
        assert!(monitor.is_win_rate_low());
    }

    #[test]
    fn test_is_win_rate_ok() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // 96% win rate (above 95% threshold)
        for _ in 0..96 {
            monitor.record_vs_random(true);
        }
        for _ in 0..4 {
            monitor.record_vs_random(false);
        }
        assert!(!monitor.is_win_rate_low());
    }

    // ========== Requirement 10.5: Output convergence metrics every 100,000 games ==========

    #[test]
    fn test_should_report_at_interval() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Not at interval yet
        for _ in 0..99_999 {
            monitor.record_game(0.0, &[], &[]);
        }
        assert!(!monitor.should_report());

        // At interval
        monitor.record_game(0.0, &[], &[]);
        assert!(monitor.should_report());

        // Past interval
        monitor.record_game(0.0, &[], &[]);
        assert!(!monitor.should_report());
    }

    #[test]
    fn test_get_metrics_complete() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(5.0, &[(0, 0, 0)], &[10.0]);
        monitor.record_vs_random(true);

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.games_played, 1);
        assert!((metrics.avg_stone_diff - 5.0).abs() < 0.001);
        assert_eq!(metrics.unique_entries_updated, 1);
        assert_eq!(metrics.total_updates, 1);
        assert!((metrics.win_rate_vs_random - 1.0).abs() < 0.001);
    }

    // ========== Requirement 10.6: Detect learning stagnation ==========

    #[test]
    fn test_is_stagnating_early() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Not enough games to detect stagnation
        for _ in 0..1000 {
            monitor.record_game(0.0, &[], &[]);
        }
        assert!(!monitor.is_stagnating());
    }

    #[test]
    fn test_is_stagnating_no_improvement() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Fill window with baseline values
        for _ in 0..RUNNING_AVERAGE_WINDOW {
            monitor.record_game(5.0, &[], &[]);
        }
        // Now run STAGNATION_WINDOW games with no improvement
        for _ in 0..STAGNATION_WINDOW {
            monitor.record_game(5.0, &[], &[]); // Same value, no improvement
        }
        assert!(monitor.is_stagnating());
    }

    #[test]
    fn test_is_stagnating_with_improvement() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // Initial games
        for _ in 0..STAGNATION_WINDOW / 2 {
            monitor.record_game(5.0, &[], &[]);
        }
        // Significant improvement
        for i in 0..RUNNING_AVERAGE_WINDOW {
            monitor.record_game(5.0 + (i as f32 * 0.1), &[], &[]);
        }
        // More games but still improving
        for _ in 0..STAGNATION_WINDOW / 2 {
            monitor.record_game(15.0, &[], &[]);
        }
        // Should not be stagnating because improvement happened recently
        assert!(!monitor.is_stagnating());
    }

    #[test]
    fn test_games_since_improvement() {
        let mut monitor = ConvergenceMonitor::new(1000);
        // First game triggers improvement (from -infinity to 0.0)
        monitor.record_game(0.0, &[], &[]);
        // Additional games with no improvement
        for _ in 0..999 {
            monitor.record_game(0.0, &[], &[]); // Same value, no improvement
        }
        let games_since = monitor.games_since_improvement();
        // First game was an improvement, so games_since = 1000 - 1 = 999
        assert_eq!(games_since, 999);
    }

    // ========== ConvergenceMetrics tests ==========

    #[test]
    fn test_metrics_is_healthy() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.96,
            is_stagnating: false,
            ..Default::default()
        };
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_metrics_not_healthy_stagnating() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.96,
            is_stagnating: true,
            ..Default::default()
        };
        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_metrics_not_healthy_low_win_rate() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.90,
            is_stagnating: false,
            ..Default::default()
        };
        assert!(!metrics.is_healthy());
    }

    #[test]
    fn test_metrics_warnings_empty() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.96,
            is_stagnating: false,
            ..Default::default()
        };
        assert!(metrics.warnings().is_empty());
    }

    #[test]
    fn test_metrics_warnings_stagnation() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.96,
            is_stagnating: true,
            games_since_improvement: 60000,
            ..Default::default()
        };
        let warnings = metrics.warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("stagnation"));
    }

    #[test]
    fn test_metrics_warnings_low_win_rate() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.90,
            is_stagnating: false,
            ..Default::default()
        };
        let warnings = metrics.warnings();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Win rate"));
    }

    #[test]
    fn test_metrics_warnings_multiple() {
        let metrics = ConvergenceMetrics {
            win_rate_vs_random: 0.90,
            is_stagnating: true,
            games_since_improvement: 60000,
            ..Default::default()
        };
        let warnings = metrics.warnings();
        assert_eq!(warnings.len(), 2);
    }

    // ========== Reset test ==========

    #[test]
    fn test_reset() {
        let mut monitor = ConvergenceMonitor::new(1000);
        monitor.record_game(5.0, &[(0, 0, 0)], &[10.0]);
        monitor.record_vs_random(true);

        assert!(monitor.games_played() > 0);
        assert!(monitor.unique_entries_updated() > 0);

        monitor.reset();

        assert_eq!(monitor.games_played(), 0);
        assert_eq!(monitor.unique_entries_updated(), 0);
        assert_eq!(monitor.total_updates(), 0);
        assert_eq!(monitor.avg_stone_diff(), 0.0);
    }

    // ========== Requirements summary test ==========

    #[test]
    fn test_all_convergence_requirements_summary() {
        println!("=== Convergence Monitor Requirements Verification ===");

        let mut monitor = ConvergenceMonitor::new(1_000_000);

        // Req 10.1: Track average stone difference trends
        for i in 0..100 {
            monitor.record_game(i as f32 * 0.1, &[], &[]);
        }
        assert!(monitor.avg_stone_diff() > 0.0);
        assert!(monitor.stone_diff_trend() > 0.0);
        println!("  10.1: Track average stone difference trends");

        // Req 10.2: Track evaluation value stability
        for _ in 0..100 {
            monitor.record_game(0.0, &[], &[10.0, 20.0, 30.0]);
        }
        assert!(monitor.eval_variance() > 0.0);
        println!("  10.2: Track evaluation value stability (variance)");

        // Req 10.3: Track percentage of pattern entries updated
        for i in 0..100 {
            monitor.record_game(0.0, &[(i % 14, i % 30, i)], &[]);
        }
        assert!(monitor.unique_entries_updated() > 0);
        assert!(monitor.entry_coverage_pct() > 0.0);
        println!("  10.3: Track percentage of pattern entries (target 200+ avg)");

        // Req 10.4: Win rate against random
        for _ in 0..100 {
            monitor.record_vs_random(true);
        }
        assert!((monitor.win_rate_vs_random() - 1.0).abs() < 0.001);
        println!("  10.4: Warn if win rate against random drops below 95%");

        // Req 10.5: Output convergence metrics every 100,000 games
        assert_eq!(CONVERGENCE_REPORT_INTERVAL, 100_000);
        let metrics = monitor.get_metrics();
        assert!(metrics.games_played > 0);
        println!("  10.5: Output convergence metrics every 100,000 games");

        // Req 10.6: Detect stagnation
        assert_eq!(STAGNATION_WINDOW, 50_000);
        // Can't easily test stagnation in summary test, but constant is correct
        println!("  10.6: Detect learning stagnation (no improvement for 50,000+ games)");

        println!("=== All convergence monitor requirements verified ===");
    }
}
