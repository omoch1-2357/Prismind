//! Eligibility Trace for TD(lambda)-Leaf learning.
//!
//! This module implements a sparse storage system for eligibility traces,
//! which track the contribution of each pattern entry to the learning signal.
//!
//! # Overview
//!
//! Eligibility traces are used for credit assignment in temporal difference learning.
//! When a pattern entry is visited during a game, its trace is incremented.
//! At each time step during backward updates, all traces are decayed by lambda.
//!
//! # Memory Efficiency
//!
//! Uses HashMap for sparse storage to minimize memory usage. Only visited
//! pattern entries have non-zero traces stored.
//!
//! # Requirements Coverage
//!
//! - Req 2.1: Maintain eligibility trace for each visited pattern entry
//! - Req 2.2: Increment trace by 1.0 on pattern visit
//! - Req 2.3: Decay all traces by lambda (0.3) at each reverse step
//! - Req 2.4: Compute gradient as td_error * eligibility_trace
//! - Req 2.5: Reset all traces to zero for new game
//! - Req 2.6: Use sparse HashMap storage to minimize memory

use std::collections::HashMap;

/// Key type for eligibility trace entries.
///
/// Represents a unique pattern table entry as (pattern_id, stage, index).
pub type TraceKey = (usize, usize, usize);

/// Sparse eligibility trace storage.
///
/// Maintains eligibility traces for visited pattern entries using a HashMap.
/// This sparse representation minimizes memory usage by only storing non-zero traces.
///
/// # Lifecycle
///
/// Traces are per-game:
/// 1. Create new trace at start of TD update
/// 2. Increment traces as positions are processed in reverse
/// 3. Decay all traces at each time step
/// 4. Reset traces for next game
///
/// # Example
///
/// ```
/// use prismind::learning::eligibility_trace::EligibilityTrace;
///
/// let mut trace = EligibilityTrace::new();
///
/// // Increment trace for visited pattern entry
/// trace.increment(0, 5, 100);
/// assert_eq!(trace.get(0, 5, 100), 1.0);
///
/// // Decay all traces by lambda
/// trace.decay(0.3);
/// assert!((trace.get(0, 5, 100) - 0.3).abs() < 0.001);
///
/// // Reset for next game
/// trace.reset();
/// assert_eq!(trace.get(0, 5, 100), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct EligibilityTrace {
    /// (pattern_id, stage, index) -> trace value
    traces: HashMap<TraceKey, f32>,
}

impl Default for EligibilityTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl EligibilityTrace {
    /// Create new empty eligibility trace.
    ///
    /// # Returns
    ///
    /// An empty EligibilityTrace with no stored entries.
    ///
    /// # Requirements
    ///
    /// - Req 2.5: Start with all traces at zero
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }

    /// Create with pre-allocated capacity.
    ///
    /// Useful when the approximate number of entries is known.
    /// A typical game has ~60 moves x 56 patterns = ~3,360 entries.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Initial capacity for the HashMap
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            traces: HashMap::with_capacity(capacity),
        }
    }

    /// Increment trace for pattern entry by 1.0.
    ///
    /// When a pattern entry is visited during the game, its eligibility trace
    /// should be incremented by 1.0 to record the visit.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern ID (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index (0 to 3^k - 1)
    ///
    /// # Requirements
    ///
    /// - Req 2.2: Increment trace by 1.0 on pattern visit
    #[inline]
    pub fn increment(&mut self, pattern_id: usize, stage: usize, index: usize) {
        let key = (pattern_id, stage, index);
        *self.traces.entry(key).or_insert(0.0) += 1.0;
    }

    /// Get trace value for pattern entry.
    ///
    /// Returns 0.0 for entries that have not been visited (sparse storage).
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern ID (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index (0 to 3^k - 1)
    ///
    /// # Returns
    ///
    /// Trace value (0.0 if not present).
    ///
    /// # Requirements
    ///
    /// - Req 2.1: Access eligibility trace for pattern entry
    #[inline]
    pub fn get(&self, pattern_id: usize, stage: usize, index: usize) -> f32 {
        let key = (pattern_id, stage, index);
        self.traces.get(&key).copied().unwrap_or(0.0)
    }

    /// Decay all traces by multiplying with lambda.
    ///
    /// At each time step during backward TD updates, all eligibility traces
    /// are decayed by multiplying with lambda (typically 0.3).
    ///
    /// # Arguments
    ///
    /// * `lambda` - Decay factor (typically 0.3)
    ///
    /// # Requirements
    ///
    /// - Req 2.3: Decay all traces by lambda at each reverse step
    pub fn decay(&mut self, lambda: f32) {
        for value in self.traces.values_mut() {
            *value *= lambda;
        }
    }

    /// Reset all traces to zero.
    ///
    /// Called at the start of a new game to clear all eligibility traces.
    ///
    /// # Requirements
    ///
    /// - Req 2.5: Reset all traces for new game
    pub fn reset(&mut self) {
        self.traces.clear();
    }

    /// Get number of non-zero entries.
    ///
    /// # Returns
    ///
    /// Count of entries with non-zero traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if trace storage is empty.
    ///
    /// # Returns
    ///
    /// `true` if no traces are stored.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Iterate over all trace entries.
    ///
    /// # Returns
    ///
    /// Iterator over ((pattern_id, stage, index), trace_value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&TraceKey, &f32)> {
        self.traces.iter()
    }

    /// Get estimated memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage including HashMap overhead.
    pub fn memory_usage(&self) -> usize {
        // HashMap overhead: approximately 48 bytes per entry (key + value + bucket overhead)
        // Key: 3 * size_of::<usize>() = 24 bytes
        // Value: size_of::<f32>() = 4 bytes
        // Bucket overhead: ~20 bytes
        let per_entry = std::mem::size_of::<TraceKey>() + std::mem::size_of::<f32>() + 20;
        let base_size = std::mem::size_of::<Self>();
        base_size + self.traces.capacity() * per_entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Requirement 2.1: Maintain eligibility trace for each visited entry ==========

    #[test]
    fn test_new_creates_empty_trace() {
        let trace = EligibilityTrace::new();
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
    }

    #[test]
    fn test_with_capacity_creates_empty_trace() {
        let trace = EligibilityTrace::with_capacity(1000);
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
    }

    #[test]
    fn test_maintains_trace_for_visited_entries() {
        let mut trace = EligibilityTrace::new();

        // Visit multiple entries
        trace.increment(0, 5, 100);
        trace.increment(1, 10, 200);
        trace.increment(13, 29, 59049);

        // All visited entries should have traces
        assert_eq!(trace.len(), 3);
        assert!(trace.get(0, 5, 100) > 0.0);
        assert!(trace.get(1, 10, 200) > 0.0);
        assert!(trace.get(13, 29, 59049) > 0.0);
    }

    // ========== Requirement 2.2: Increment trace by 1.0 on pattern visit ==========

    #[test]
    fn test_increment_adds_one_to_trace() {
        let mut trace = EligibilityTrace::new();

        // First visit
        trace.increment(0, 0, 0);
        assert_eq!(trace.get(0, 0, 0), 1.0);

        // Second visit (same entry)
        trace.increment(0, 0, 0);
        assert_eq!(trace.get(0, 0, 0), 2.0);

        // Third visit
        trace.increment(0, 0, 0);
        assert_eq!(trace.get(0, 0, 0), 3.0);
    }

    #[test]
    fn test_increment_creates_new_entry_if_not_exists() {
        let mut trace = EligibilityTrace::new();

        assert_eq!(trace.get(5, 15, 1000), 0.0);
        trace.increment(5, 15, 1000);
        assert_eq!(trace.get(5, 15, 1000), 1.0);
    }

    #[test]
    fn test_increment_different_entries_are_independent() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);
        trace.increment(1, 0, 0);
        trace.increment(0, 1, 0);
        trace.increment(0, 0, 1);

        assert_eq!(trace.get(0, 0, 0), 1.0);
        assert_eq!(trace.get(1, 0, 0), 1.0);
        assert_eq!(trace.get(0, 1, 0), 1.0);
        assert_eq!(trace.get(0, 0, 1), 1.0);
        assert_eq!(trace.len(), 4);
    }

    // ========== Requirement 2.3: Decay all traces by lambda ==========

    #[test]
    fn test_decay_multiplies_all_traces_by_lambda() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);
        trace.increment(1, 1, 1);
        trace.increment(2, 2, 2);

        // Initial values are all 1.0
        assert_eq!(trace.get(0, 0, 0), 1.0);
        assert_eq!(trace.get(1, 1, 1), 1.0);
        assert_eq!(trace.get(2, 2, 2), 1.0);

        // Decay by lambda = 0.3
        trace.decay(0.3);

        assert!((trace.get(0, 0, 0) - 0.3).abs() < 1e-6);
        assert!((trace.get(1, 1, 1) - 0.3).abs() < 1e-6);
        assert!((trace.get(2, 2, 2) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_decay_with_lambda_zero() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);
        trace.decay(0.0);

        assert_eq!(trace.get(0, 0, 0), 0.0);
    }

    #[test]
    fn test_decay_with_lambda_one() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);
        trace.decay(1.0);

        assert_eq!(trace.get(0, 0, 0), 1.0);
    }

    #[test]
    fn test_decay_after_increment() {
        let mut trace = EligibilityTrace::new();

        // Simulate visiting entries over multiple time steps
        trace.increment(0, 0, 0);
        trace.decay(0.3);
        // After decay: 0.3

        trace.increment(0, 0, 0);
        // After increment: 0.3 + 1.0 = 1.3
        assert!((trace.get(0, 0, 0) - 1.3).abs() < 1e-6);

        trace.decay(0.3);
        // After second decay: 1.3 * 0.3 = 0.39
        assert!((trace.get(0, 0, 0) - 0.39).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_decays() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);

        // Multiple decays
        for _ in 0..5 {
            trace.decay(0.3);
        }

        // 1.0 * 0.3^5 = 0.00243
        let expected = 0.3_f32.powi(5);
        assert!((trace.get(0, 0, 0) - expected).abs() < 1e-6);
    }

    // ========== Requirement 2.4: Gradient computation (tested via get) ==========

    #[test]
    fn test_get_returns_zero_for_unvisited() {
        let trace = EligibilityTrace::new();

        // Unvisited entries return 0.0
        assert_eq!(trace.get(0, 0, 0), 0.0);
        assert_eq!(trace.get(13, 29, 59048), 0.0);
    }

    #[test]
    fn test_gradient_computation_example() {
        let mut trace = EligibilityTrace::new();

        // Visit an entry
        trace.increment(0, 5, 100);

        // Simulate gradient computation: gradient = td_error * eligibility_trace
        let td_error = 0.5;
        let eligibility = trace.get(0, 5, 100);
        let gradient = td_error * eligibility;

        // gradient = 0.5 * 1.0 = 0.5
        assert!((gradient - 0.5).abs() < 1e-6);
    }

    // ========== Requirement 2.5: Reset all traces for new game ==========

    #[test]
    fn test_reset_clears_all_traces() {
        let mut trace = EligibilityTrace::new();

        // Add multiple entries
        trace.increment(0, 0, 0);
        trace.increment(1, 1, 1);
        trace.increment(2, 2, 2);
        assert_eq!(trace.len(), 3);

        // Reset
        trace.reset();

        // All traces should be cleared
        assert!(trace.is_empty());
        assert_eq!(trace.len(), 0);
        assert_eq!(trace.get(0, 0, 0), 0.0);
        assert_eq!(trace.get(1, 1, 1), 0.0);
        assert_eq!(trace.get(2, 2, 2), 0.0);
    }

    #[test]
    fn test_reset_allows_reuse() {
        let mut trace = EligibilityTrace::new();

        // First game
        trace.increment(0, 0, 0);
        trace.reset();

        // Second game
        trace.increment(1, 1, 1);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.get(0, 0, 0), 0.0);
        assert_eq!(trace.get(1, 1, 1), 1.0);
    }

    // ========== Requirement 2.6: Sparse storage (HashMap) ==========

    #[test]
    fn test_sparse_storage_only_stores_visited() {
        let mut trace = EligibilityTrace::new();

        // Visit only 3 entries out of potentially millions
        trace.increment(0, 0, 0);
        trace.increment(7, 15, 10000);
        trace.increment(13, 29, 59048);

        // Only 3 entries stored
        assert_eq!(trace.len(), 3);

        // Memory usage should be minimal
        let memory = trace.memory_usage();
        println!("Memory usage for 3 entries: {} bytes", memory);
        assert!(memory < 1000); // Should be well under 1KB for 3 entries
    }

    #[test]
    fn test_sparse_storage_typical_game_size() {
        let mut trace = EligibilityTrace::with_capacity(3360);

        // Simulate a typical game: ~60 moves x 56 patterns = 3360 entries
        for move_num in 0..60 {
            let stage = move_num / 2;
            for pattern_id in 0..14 {
                // Simulate 4 rotations
                for rotation in 0..4 {
                    let index = (move_num * 100 + pattern_id * 10 + rotation) % 59049;
                    trace.increment(pattern_id, stage, index);
                }
            }
        }

        // Should have at most 3360 unique entries (likely fewer due to duplicates)
        println!("Entries after simulated game: {}", trace.len());
        assert!(trace.len() <= 3360);

        // Memory usage should be reasonable
        let memory = trace.memory_usage();
        println!(
            "Memory usage for simulated game: {} bytes ({:.2} KB)",
            memory,
            memory as f64 / 1024.0
        );

        // Should be well under 500KB for sparse storage
        assert!(memory < 500_000);
    }

    // ========== Iterator tests ==========

    #[test]
    fn test_iter_returns_all_entries() {
        let mut trace = EligibilityTrace::new();

        trace.increment(0, 0, 0);
        trace.increment(1, 1, 100);
        trace.increment(2, 2, 200);

        let entries: Vec<_> = trace.iter().collect();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_iter_empty_trace() {
        let trace = EligibilityTrace::new();
        let entries: Vec<_> = trace.iter().collect();
        assert!(entries.is_empty());
    }

    // ========== Clone and Default tests ==========

    #[test]
    fn test_clone() {
        let mut trace = EligibilityTrace::new();
        trace.increment(0, 0, 0);
        trace.increment(1, 1, 1);

        let cloned = trace.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get(0, 0, 0), 1.0);
        assert_eq!(cloned.get(1, 1, 1), 1.0);
    }

    #[test]
    fn test_default() {
        let trace = EligibilityTrace::default();
        assert!(trace.is_empty());
    }

    // ========== Requirements summary test ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== Eligibility Trace Requirements Verification ===");

        let mut trace = EligibilityTrace::new();

        // Req 2.1: Maintain trace for each visited entry
        trace.increment(0, 5, 100);
        assert!(trace.get(0, 5, 100) > 0.0);
        println!("  2.1: Maintain eligibility trace for visited entries");

        // Req 2.2: Increment by 1.0
        trace.increment(0, 5, 100);
        assert_eq!(trace.get(0, 5, 100), 2.0);
        println!("  2.2: Increment trace by 1.0 on pattern visit");

        // Req 2.3: Decay by lambda
        trace.decay(0.3);
        assert!((trace.get(0, 5, 100) - 0.6).abs() < 1e-6);
        println!("  2.3: Decay all traces by lambda (0.3)");

        // Req 2.4: Gradient = td_error * trace
        let gradient = 0.5 * trace.get(0, 5, 100);
        assert!((gradient - 0.3).abs() < 1e-6);
        println!("  2.4: Gradient = td_error * eligibility_trace");

        // Req 2.5: Reset for new game
        trace.reset();
        assert!(trace.is_empty());
        println!("  2.5: Reset all traces for new game");

        // Req 2.6: Sparse HashMap storage
        trace.increment(0, 0, 0);
        assert!(trace.memory_usage() < 1000);
        println!("  2.6: Sparse HashMap storage minimizes memory");

        println!("=== All eligibility trace requirements verified ===");
    }
}
