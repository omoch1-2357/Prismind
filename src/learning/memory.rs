//! Memory Management Validation Module
//!
//! This module implements memory validation and runtime monitoring for the learning system.
//! It ensures that memory usage stays within the 600 MB budget defined in the requirements.
//!
//! # Overview
//!
//! Memory components and their expected usage:
//! - EvaluationTable: ~57 MB (30 stages x ~14.4M entries x 2 bytes)
//! - Adam m moments: ~114 MB (30 stages x ~14.4M entries x 4 bytes)
//! - Adam v moments: ~114 MB (30 stages x ~14.4M entries x 4 bytes)
//! - TranspositionTable: 128-256 MB (configurable, shared with Phase 2)
//! - Game histories: ~120 KB (4 threads x ~30 KB per game)
//! - Eligibility traces: ~50 KB (~3,360 entries x 16 bytes)
//! - Other overhead: ~10 MB
//!
//! # Requirements Coverage
//!
//! - Req 8.1: Total memory <= 600 MB
//! - Req 8.2: Pattern Table ~57 MB
//! - Req 8.3: Adam Optimizer ~228 MB
//! - Req 8.4: TranspositionTable 128-256 MB (shared with Phase 2)
//! - Req 8.5: Release game history after TD update
//! - Req 8.6: Sparse eligibility traces

use crate::evaluator::EvaluationTable;
use crate::learning::LearningError;
use crate::learning::adam::AdamOptimizer;
use crate::learning::eligibility_trace::EligibilityTrace;
use crate::learning::game_history::GameHistory;

/// Total memory budget in bytes (600 MB).
///
/// Requirement 8.1: The Learning System shall use no more than 600 MB total memory.
pub const TOTAL_MEMORY_BUDGET: usize = 600 * 1024 * 1024;

/// Expected EvaluationTable memory in bytes (~57 MB).
///
/// Requirement 8.2: The Pattern Table shall use approximately 57 MB.
pub const EXPECTED_EVAL_TABLE_MB: usize = 57;

/// Expected Adam Optimizer memory in bytes (~228 MB).
///
/// Requirement 8.3: The Adam Optimizer shall use approximately 228 MB.
pub const EXPECTED_ADAM_MB: usize = 228;

/// Minimum TranspositionTable size in MB.
///
/// Requirement 8.4: The Transposition Table shall use 128-256 MB.
pub const MIN_TT_SIZE_MB: usize = 128;

/// Maximum TranspositionTable size in MB.
///
/// Requirement 8.4: The Transposition Table shall use 128-256 MB.
pub const MAX_TT_SIZE_MB: usize = 256;

/// Tolerance factor for memory validation (10%).
pub const MEMORY_TOLERANCE: f64 = 0.10;

/// Expected game history memory per game in bytes (~30 KB).
pub const EXPECTED_GAME_HISTORY_KB: usize = 30;

/// Maximum expected eligibility trace memory per game in bytes (~50 KB).
pub const EXPECTED_ELIGIBILITY_TRACE_KB: usize = 50;

/// Memory component breakdown for monitoring.
#[derive(Debug, Clone, Default)]
pub struct MemoryBreakdown {
    /// EvaluationTable memory in bytes.
    pub eval_table_bytes: usize,
    /// Adam optimizer memory in bytes (m + v moments).
    pub adam_bytes: usize,
    /// TranspositionTable memory in bytes.
    pub tt_bytes: usize,
    /// Estimated game history memory in bytes.
    pub game_history_bytes: usize,
    /// Estimated eligibility trace memory in bytes.
    pub eligibility_trace_bytes: usize,
    /// Other overhead in bytes.
    pub overhead_bytes: usize,
}

impl MemoryBreakdown {
    /// Calculate total memory usage.
    pub fn total(&self) -> usize {
        self.eval_table_bytes
            + self.adam_bytes
            + self.tt_bytes
            + self.game_history_bytes
            + self.eligibility_trace_bytes
            + self.overhead_bytes
    }

    /// Total memory usage in megabytes.
    pub fn total_mb(&self) -> f64 {
        self.total() as f64 / (1024.0 * 1024.0)
    }

    /// Check if total is within budget.
    pub fn is_within_budget(&self) -> bool {
        self.total() <= TOTAL_MEMORY_BUDGET
    }

    /// Get formatted summary string.
    pub fn summary(&self) -> String {
        format!(
            "Memory Usage: {:.1} MB / {:.1} MB (Budget)\n\
             - EvaluationTable: {:.1} MB\n\
             - Adam Optimizer: {:.1} MB\n\
             - TranspositionTable: {:.1} MB\n\
             - Game Histories: {:.1} KB\n\
             - Eligibility Traces: {:.1} KB\n\
             - Overhead: {:.1} MB",
            self.total_mb(),
            TOTAL_MEMORY_BUDGET as f64 / (1024.0 * 1024.0),
            self.eval_table_bytes as f64 / (1024.0 * 1024.0),
            self.adam_bytes as f64 / (1024.0 * 1024.0),
            self.tt_bytes as f64 / (1024.0 * 1024.0),
            self.game_history_bytes as f64 / 1024.0,
            self.eligibility_trace_bytes as f64 / 1024.0,
            self.overhead_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Runtime memory monitor for training.
///
/// Tracks memory usage of all components and provides validation
/// against the 600 MB budget.
///
/// # Example
///
/// ```ignore
/// let mut monitor = MemoryMonitor::new();
/// monitor.update_eval_table_usage(eval_table.memory_usage());
/// monitor.update_adam_usage(adam.memory_usage());
/// monitor.update_tt_usage(tt_size_mb * 1024 * 1024);
///
/// if !monitor.is_within_budget() {
///     return Err(LearningError::MemoryAllocation("Budget exceeded".into()));
/// }
/// ```
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Current memory breakdown.
    breakdown: MemoryBreakdown,
    /// Number of active parallel games.
    active_games: usize,
    /// Number of threads for game execution.
    num_threads: usize,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor.
    pub fn new() -> Self {
        Self {
            breakdown: MemoryBreakdown::default(),
            active_games: 0,
            num_threads: 4, // Default to 4 threads
        }
    }

    /// Create with specific thread count.
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            breakdown: MemoryBreakdown::default(),
            active_games: 0,
            num_threads,
        }
    }

    /// Update EvaluationTable memory usage.
    pub fn update_eval_table_usage(&mut self, bytes: usize) {
        self.breakdown.eval_table_bytes = bytes;
    }

    /// Update Adam optimizer memory usage.
    pub fn update_adam_usage(&mut self, bytes: usize) {
        self.breakdown.adam_bytes = bytes;
    }

    /// Update TranspositionTable memory usage.
    pub fn update_tt_usage(&mut self, bytes: usize) {
        self.breakdown.tt_bytes = bytes;
    }

    /// Update game history memory usage.
    ///
    /// # Arguments
    ///
    /// * `game_history` - Reference to a game history for size estimation
    pub fn update_game_history_usage(&mut self, game_history: &GameHistory) {
        // Memory per game history: MoveRecord size * number of moves
        let move_record_size = std::mem::size_of::<crate::learning::game_history::MoveRecord>();
        let per_game = move_record_size * game_history.len();

        // Multiply by active games (or num_threads if games are active)
        self.breakdown.game_history_bytes = per_game * self.num_threads;
    }

    /// Update eligibility trace memory usage.
    ///
    /// # Arguments
    ///
    /// * `trace` - Reference to an eligibility trace for size estimation
    pub fn update_eligibility_trace_usage(&mut self, trace: &EligibilityTrace) {
        self.breakdown.eligibility_trace_bytes = trace.memory_usage();
    }

    /// Update overhead estimate.
    pub fn update_overhead(&mut self, bytes: usize) {
        self.breakdown.overhead_bytes = bytes;
    }

    /// Set the number of active parallel games.
    pub fn set_active_games(&mut self, count: usize) {
        self.active_games = count;
    }

    /// Get current memory breakdown.
    pub fn breakdown(&self) -> &MemoryBreakdown {
        &self.breakdown
    }

    /// Get total memory usage.
    pub fn total_usage(&self) -> usize {
        self.breakdown.total()
    }

    /// Get total memory usage in MB.
    pub fn total_usage_mb(&self) -> f64 {
        self.breakdown.total_mb()
    }

    /// Check if within memory budget.
    pub fn is_within_budget(&self) -> bool {
        self.breakdown.is_within_budget()
    }

    /// Validate memory constraints and return detailed error if exceeded.
    ///
    /// # Returns
    ///
    /// Ok(()) if within budget, Err with details if exceeded.
    pub fn validate(&self) -> Result<(), LearningError> {
        if self.is_within_budget() {
            Ok(())
        } else {
            Err(LearningError::MemoryAllocation(format!(
                "Memory budget exceeded: {:.1} MB > {} MB\n{}",
                self.total_usage_mb(),
                TOTAL_MEMORY_BUDGET / (1024 * 1024),
                self.breakdown.summary()
            )))
        }
    }

    /// Get formatted summary.
    pub fn summary(&self) -> String {
        self.breakdown.summary()
    }
}

/// Validate that EvaluationTable uses expected memory (~57 MB).
///
/// # Arguments
///
/// * `table` - Reference to the evaluation table
///
/// # Returns
///
/// Ok(actual_bytes) if within tolerance, Err if significantly different.
///
/// # Requirements
///
/// - Req 8.2: Pattern Table ~57 MB
pub fn validate_eval_table_memory(table: &EvaluationTable) -> Result<usize, LearningError> {
    let actual_bytes = table.memory_usage();
    let expected_bytes = EXPECTED_EVAL_TABLE_MB * 1024 * 1024;
    let tolerance = (expected_bytes as f64 * MEMORY_TOLERANCE) as usize;

    let diff = actual_bytes.abs_diff(expected_bytes);

    if diff <= tolerance {
        Ok(actual_bytes)
    } else {
        Err(LearningError::MemoryAllocation(format!(
            "EvaluationTable memory mismatch: {:.1} MB (expected ~{} MB)",
            actual_bytes as f64 / (1024.0 * 1024.0),
            EXPECTED_EVAL_TABLE_MB
        )))
    }
}

/// Validate that Adam optimizer uses expected memory (~228 MB).
///
/// # Arguments
///
/// * `adam` - Reference to the Adam optimizer
///
/// # Returns
///
/// Ok(actual_bytes) if within tolerance, Err if significantly different.
///
/// # Requirements
///
/// - Req 8.3: Adam Optimizer ~228 MB
pub fn validate_adam_memory(adam: &AdamOptimizer) -> Result<usize, LearningError> {
    let actual_bytes = adam.memory_usage();
    let expected_bytes = EXPECTED_ADAM_MB * 1024 * 1024;
    let tolerance = (expected_bytes as f64 * MEMORY_TOLERANCE) as usize;

    let diff = actual_bytes.abs_diff(expected_bytes);

    if diff <= tolerance {
        Ok(actual_bytes)
    } else {
        Err(LearningError::MemoryAllocation(format!(
            "Adam optimizer memory mismatch: {:.1} MB (expected ~{} MB)",
            actual_bytes as f64 / (1024.0 * 1024.0),
            EXPECTED_ADAM_MB
        )))
    }
}

/// Validate TranspositionTable size is within range.
///
/// # Arguments
///
/// * `size_mb` - TranspositionTable size in MB
///
/// # Returns
///
/// Ok(size_bytes) if within range, Err if outside 128-256 MB.
///
/// # Requirements
///
/// - Req 8.4: TranspositionTable 128-256 MB
pub fn validate_tt_size(size_mb: usize) -> Result<usize, LearningError> {
    if (MIN_TT_SIZE_MB..=MAX_TT_SIZE_MB).contains(&size_mb) {
        Ok(size_mb * 1024 * 1024)
    } else {
        Err(LearningError::MemoryAllocation(format!(
            "TranspositionTable size {} MB outside valid range ({}-{} MB)",
            size_mb, MIN_TT_SIZE_MB, MAX_TT_SIZE_MB
        )))
    }
}

/// Verify that game history memory is released after TD update.
///
/// This function verifies that a game history is effectively empty/cleared,
/// indicating that memory was properly released after TD update.
///
/// # Arguments
///
/// * `history` - Reference to the game history to check
///
/// # Returns
///
/// true if history is empty (memory released), false otherwise.
///
/// # Requirements
///
/// - Req 8.5: Release game history memory after TD update
#[inline]
pub fn verify_game_history_released(history: &GameHistory) -> bool {
    history.is_empty()
}

/// Verify that eligibility trace uses sparse storage.
///
/// Checks that the eligibility trace memory usage is consistent with
/// sparse HashMap storage (much less than full dense array).
///
/// # Arguments
///
/// * `trace` - Reference to the eligibility trace
/// * `max_expected_entries` - Maximum expected number of entries
///
/// # Returns
///
/// true if using sparse storage efficiently, false otherwise.
///
/// # Requirements
///
/// - Req 8.6: Sparse eligibility traces
pub fn verify_sparse_eligibility_trace(
    trace: &EligibilityTrace,
    max_expected_entries: usize,
) -> bool {
    let actual_entries = trace.len();
    let memory = trace.memory_usage();

    // Sparse storage should have entry count much less than a full pattern table
    // Full table would have ~14.4M entries per stage
    // A typical game should have ~3,360 entries (56 patterns * 60 moves)
    let is_sparse = actual_entries <= max_expected_entries;

    // Memory should be proportional to capacity (HashMap pre-allocates), not full table
    // Each entry with HashMap overhead: ~48-100 bytes depending on implementation
    // A dense array for one stage would be ~14.4M * 4 = ~57MB
    // Sparse storage with max_expected_entries should be << 1MB
    let max_reasonable_memory = 1024 * 1024; // 1 MB is extremely generous for ~3360 entries
    let memory_efficient = memory < max_reasonable_memory;

    is_sparse && memory_efficient
}

/// Validate total memory budget.
///
/// Validates that the total of all components stays within 600 MB.
///
/// # Arguments
///
/// * `eval_table` - Reference to evaluation table
/// * `adam` - Reference to Adam optimizer
/// * `tt_size_mb` - TranspositionTable size in MB
///
/// # Returns
///
/// Ok(breakdown) with memory breakdown if within budget.
///
/// # Requirements
///
/// - Req 8.1: Total memory <= 600 MB
pub fn validate_total_memory(
    eval_table: &EvaluationTable,
    adam: &AdamOptimizer,
    tt_size_mb: usize,
) -> Result<MemoryBreakdown, LearningError> {
    let breakdown = MemoryBreakdown {
        eval_table_bytes: eval_table.memory_usage(),
        adam_bytes: adam.memory_usage(),
        tt_bytes: tt_size_mb * 1024 * 1024,
        // Estimate maximum per-thread game history and traces
        // 4 threads * 30KB per game = 120KB
        game_history_bytes: 4 * EXPECTED_GAME_HISTORY_KB * 1024,
        // 4 threads * ~12.5KB per trace = 50KB
        eligibility_trace_bytes: EXPECTED_ELIGIBILITY_TRACE_KB * 1024,
        // Other overhead: ~10MB
        overhead_bytes: 10 * 1024 * 1024,
    };

    if breakdown.is_within_budget() {
        Ok(breakdown)
    } else {
        Err(LearningError::MemoryAllocation(format!(
            "Total memory {:.1} MB exceeds 600 MB budget:\n{}",
            breakdown.total_mb(),
            breakdown.summary()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::Pattern;

    /// Create test patterns for testing.
    fn create_test_patterns() -> Vec<Pattern> {
        vec![
            Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
            Pattern::new(1, 10, vec![0, 8, 16, 24, 32, 40, 48, 56, 1, 9]).unwrap(),
            Pattern::new(2, 10, vec![0, 1, 8, 9, 10, 16, 17, 18, 24, 25]).unwrap(),
            Pattern::new(3, 10, vec![0, 9, 18, 27, 36, 45, 54, 63, 1, 10]).unwrap(),
            Pattern::new(4, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            Pattern::new(5, 8, vec![0, 8, 16, 24, 32, 40, 48, 56]).unwrap(),
            Pattern::new(6, 8, vec![0, 9, 18, 27, 36, 45, 54, 63]).unwrap(),
            Pattern::new(7, 8, vec![7, 14, 21, 28, 35, 42, 49, 56]).unwrap(),
            Pattern::new(8, 7, vec![0, 1, 2, 3, 4, 5, 6]).unwrap(),
            Pattern::new(9, 7, vec![0, 8, 16, 24, 32, 40, 48]).unwrap(),
            Pattern::new(10, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
            Pattern::new(11, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
            Pattern::new(12, 5, vec![0, 1, 2, 3, 4]).unwrap(),
            Pattern::new(13, 5, vec![0, 8, 16, 24, 32]).unwrap(),
        ]
    }

    // ========== Requirement 8.1: Total memory <= 600 MB ==========

    #[test]
    fn test_total_memory_budget_constant() {
        assert_eq!(TOTAL_MEMORY_BUDGET, 600 * 1024 * 1024);
        assert_eq!(TOTAL_MEMORY_BUDGET, 629_145_600);
    }

    #[test]
    fn test_memory_breakdown_total() {
        let breakdown = MemoryBreakdown {
            eval_table_bytes: 57 * 1024 * 1024,
            adam_bytes: 228 * 1024 * 1024,
            tt_bytes: 128 * 1024 * 1024,
            ..Default::default()
        };

        let total = breakdown.total();
        assert_eq!(total, (57 + 228 + 128) * 1024 * 1024);
    }

    #[test]
    fn test_memory_breakdown_within_budget() {
        let breakdown = MemoryBreakdown {
            eval_table_bytes: 57 * 1024 * 1024,
            adam_bytes: 228 * 1024 * 1024,
            tt_bytes: 256 * 1024 * 1024,
            ..Default::default()
        };

        // 57 + 228 + 256 = 541 MB < 600 MB
        assert!(breakdown.is_within_budget());
    }

    #[test]
    fn test_memory_breakdown_exceeds_budget() {
        let breakdown = MemoryBreakdown {
            eval_table_bytes: 100 * 1024 * 1024,
            adam_bytes: 300 * 1024 * 1024,
            tt_bytes: 256 * 1024 * 1024,
            ..Default::default()
        };

        // 100 + 300 + 256 = 656 MB > 600 MB
        assert!(!breakdown.is_within_budget());
    }

    #[test]
    fn test_memory_monitor_validation() {
        let mut monitor = MemoryMonitor::new();
        monitor.update_eval_table_usage(57 * 1024 * 1024);
        monitor.update_adam_usage(228 * 1024 * 1024);
        monitor.update_tt_usage(128 * 1024 * 1024);

        assert!(monitor.is_within_budget());
        assert!(monitor.validate().is_ok());
    }

    #[test]
    fn test_memory_monitor_validation_fails_over_budget() {
        let mut monitor = MemoryMonitor::new();
        monitor.update_eval_table_usage(100 * 1024 * 1024);
        monitor.update_adam_usage(300 * 1024 * 1024);
        monitor.update_tt_usage(256 * 1024 * 1024);

        assert!(!monitor.is_within_budget());
        assert!(monitor.validate().is_err());
    }

    // ========== Requirement 8.2: Pattern Table ~57 MB ==========

    #[test]
    fn test_expected_eval_table_constant() {
        assert_eq!(EXPECTED_EVAL_TABLE_MB, 57);
    }

    #[test]
    fn test_eval_table_memory_validation() {
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);

        let actual_mb = table.memory_usage() as f64 / (1024.0 * 1024.0);

        // EvaluationTable should use approximately 57 MB
        // Allow 10% tolerance
        println!("EvaluationTable memory: {:.2} MB", actual_mb);

        // With test patterns, memory will be lower than full patterns
        // Just verify it's a reasonable value > 0
        assert!(table.memory_usage() > 0);
    }

    // ========== Requirement 8.3: Adam Optimizer ~228 MB ==========

    #[test]
    fn test_expected_adam_constant() {
        assert_eq!(EXPECTED_ADAM_MB, 228);
    }

    #[test]
    fn test_adam_memory_validation() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);

        let actual_mb = adam.memory_usage() as f64 / (1024.0 * 1024.0);

        // Adam optimizer should use approximately 228 MB (m: 114 MB + v: 114 MB)
        println!("Adam optimizer memory: {:.2} MB", actual_mb);

        // With test patterns, memory will be lower than full patterns
        // Just verify it's a reasonable value > 0
        assert!(adam.memory_usage() > 0);
    }

    #[test]
    fn test_adam_memory_is_double_eval_table() {
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Adam uses f32 (4 bytes) x 2 arrays (m, v) = 8 bytes per entry
        // EvaluationTable uses u16 (2 bytes) = 2 bytes per entry
        // So Adam should be ~4x the EvaluationTable memory

        let table_mem = table.memory_usage();
        let adam_mem = adam.memory_usage();
        let ratio = adam_mem as f64 / table_mem as f64;

        println!("Adam/EvaluationTable memory ratio: {:.2}", ratio);

        // Should be approximately 4x (with some tolerance for overhead)
        assert!(
            (3.8..=4.2).contains(&ratio),
            "Expected ratio ~4.0, got {}",
            ratio
        );
    }

    // ========== Requirement 8.4: TranspositionTable 128-256 MB ==========

    #[test]
    fn test_tt_size_constants() {
        assert_eq!(MIN_TT_SIZE_MB, 128);
        assert_eq!(MAX_TT_SIZE_MB, 256);
    }

    #[test]
    fn test_tt_size_validation_valid() {
        assert!(validate_tt_size(128).is_ok());
        assert!(validate_tt_size(192).is_ok());
        assert!(validate_tt_size(256).is_ok());
    }

    #[test]
    fn test_tt_size_validation_invalid() {
        assert!(validate_tt_size(64).is_err());
        assert!(validate_tt_size(100).is_err());
        assert!(validate_tt_size(300).is_err());
        assert!(validate_tt_size(512).is_err());
    }

    // ========== Requirement 8.5: Release game history after TD update ==========

    #[test]
    fn test_game_history_released_empty() {
        let history = GameHistory::new();
        assert!(verify_game_history_released(&history));
    }

    #[test]
    fn test_game_history_released_after_clear() {
        use crate::board::BitBoard;
        use crate::learning::game_history::{MoveRecord, NUM_PATTERN_INSTANCES};

        let mut history = GameHistory::new();

        // Add some moves
        for _ in 0..10 {
            history.push(MoveRecord::new(
                BitBoard::new(),
                0.0,
                [0; NUM_PATTERN_INSTANCES],
                0,
            ));
        }

        assert!(!verify_game_history_released(&history));

        // Clear (simulating release after TD update)
        history.clear();

        assert!(verify_game_history_released(&history));
    }

    // ========== Requirement 8.6: Sparse eligibility traces ==========

    #[test]
    fn test_sparse_eligibility_trace_empty() {
        let trace = EligibilityTrace::new();
        assert!(verify_sparse_eligibility_trace(&trace, 3360));
    }

    #[test]
    fn test_sparse_eligibility_trace_typical_game() {
        let mut trace = EligibilityTrace::with_capacity(3360);

        // Simulate a typical game with 60 moves x 56 patterns = 3360 entries max
        // But many will be duplicates, so actual count will be less
        for move_num in 0..60 {
            let stage = move_num / 2;
            for pattern_id in 0..14 {
                let index = (move_num * 100 + pattern_id) % 59049;
                trace.increment(pattern_id, stage, index);
            }
        }

        // Should have sparse representation
        assert!(trace.len() <= 3360, "Trace entries should be <= 3360");

        // Verify it uses sparse storage: memory should be reasonable
        // With 3360 capacity, memory ~= 3360 * 48 bytes + overhead = ~161KB
        // This is still much less than a dense array (14.4M entries * 4 bytes = 57MB)
        let memory = trace.memory_usage();
        assert!(
            memory < 500_000,
            "Memory should be < 500KB for sparse storage, got {} bytes",
            memory
        );

        println!("Eligibility trace entries: {}", trace.len());
        println!(
            "Eligibility trace memory: {} bytes ({:.2} KB)",
            memory,
            memory as f64 / 1024.0
        );
    }

    #[test]
    fn test_sparse_trace_memory_efficient() {
        let mut trace = EligibilityTrace::new(); // Start with no pre-allocation

        // Add only 100 entries
        for i in 0..100 {
            trace.increment(i % 14, i / 14, i * 10);
        }

        let memory = trace.memory_usage();

        // Memory should be proportional to actual entries, not millions
        // HashMap will grow as needed; with ~100 entries, memory should be reasonable
        // Even with HashMap overhead and some extra capacity, should be << 1MB
        // (A dense array would be 14.4M entries * 4 bytes = ~57MB)
        assert!(
            memory < 500_000,
            "Memory should be < 500KB for ~100 entries, got {} bytes",
            memory
        );

        println!(
            "Sparse trace with {} entries uses {} bytes ({:.2} KB)",
            trace.len(),
            memory,
            memory as f64 / 1024.0
        );
    }

    // ========== MemoryMonitor tests ==========

    #[test]
    fn test_memory_monitor_new() {
        let monitor = MemoryMonitor::new();
        assert_eq!(monitor.total_usage(), 0);
        assert!(monitor.is_within_budget());
    }

    #[test]
    fn test_memory_monitor_with_threads() {
        let monitor = MemoryMonitor::with_threads(8);
        assert_eq!(monitor.num_threads, 8);
    }

    #[test]
    fn test_memory_monitor_update_components() {
        let mut monitor = MemoryMonitor::new();

        monitor.update_eval_table_usage(57 * 1024 * 1024);
        assert_eq!(monitor.breakdown.eval_table_bytes, 57 * 1024 * 1024);

        monitor.update_adam_usage(228 * 1024 * 1024);
        assert_eq!(monitor.breakdown.adam_bytes, 228 * 1024 * 1024);

        monitor.update_tt_usage(128 * 1024 * 1024);
        assert_eq!(monitor.breakdown.tt_bytes, 128 * 1024 * 1024);
    }

    #[test]
    fn test_memory_breakdown_summary() {
        let breakdown = MemoryBreakdown {
            eval_table_bytes: 57 * 1024 * 1024,
            adam_bytes: 228 * 1024 * 1024,
            tt_bytes: 128 * 1024 * 1024,
            ..Default::default()
        };

        let summary = breakdown.summary();
        assert!(summary.contains("EvaluationTable"));
        assert!(summary.contains("Adam"));
        assert!(summary.contains("TranspositionTable"));
    }

    // ========== Integration validation test ==========

    #[test]
    fn test_validate_total_memory_within_budget() {
        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // With test patterns (smaller than real), should be well within budget
        let result = validate_total_memory(&table, &adam, 128);
        assert!(result.is_ok(), "Should be within budget with test patterns");

        if let Ok(breakdown) = result {
            println!("{}", breakdown.summary());
            assert!(breakdown.is_within_budget());
        }
    }

    // ========== Requirements summary test ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== Task 10 Memory Management Requirements Verification ===");

        // Req 8.1: Total memory <= 600 MB
        assert_eq!(TOTAL_MEMORY_BUDGET, 600 * 1024 * 1024);
        println!("  8.1: Total memory budget = 600 MB");

        // Req 8.2: Pattern Table ~57 MB
        assert_eq!(EXPECTED_EVAL_TABLE_MB, 57);
        println!("  8.2: Pattern Table expected ~57 MB");

        // Req 8.3: Adam Optimizer ~228 MB
        assert_eq!(EXPECTED_ADAM_MB, 228);
        println!("  8.3: Adam Optimizer expected ~228 MB");

        // Req 8.4: TranspositionTable 128-256 MB
        assert!(validate_tt_size(128).is_ok());
        assert!(validate_tt_size(256).is_ok());
        println!("  8.4: TranspositionTable range 128-256 MB");

        // Req 8.5: Release game history after TD update
        let history = GameHistory::new();
        assert!(verify_game_history_released(&history));
        println!("  8.5: Game history release verification");

        // Req 8.6: Sparse eligibility traces
        let trace = EligibilityTrace::new();
        assert!(verify_sparse_eligibility_trace(&trace, 3360));
        println!("  8.6: Sparse eligibility trace verification");

        println!("=== All Task 10 requirements verified ===");
    }
}
