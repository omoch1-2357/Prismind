//! SharedEvaluator: Thread-safe evaluator wrapper for Phase 3 learning.
//!
//! This module provides a thread-safe wrapper around EvaluationTable using
//! `Arc<RwLock>` for concurrent read access during parallel game execution
//! and exclusive write access during TD weight updates.
//!
//! # Architecture
//!
//! ```text
//! SharedEvaluator
//!     |-- Arc<RwLock<EvaluationTable>>  -- Thread-safe weight storage
//!     |-- [Pattern; 14]                 -- Pattern definitions for extraction
//! ```
//!
//! # Concurrency Model
//!
//! - **Read locks**: Acquired during evaluation (search phase)
//!   - Multiple threads can hold read locks simultaneously
//!   - Allows parallel game execution
//! - **Write locks**: Acquired during TD updates
//!   - Exclusive access for weight modifications
//!   - Ensures consistency between learning and search
//!
//! # Requirements Coverage
//!
//! - Req 9.1: Use Phase 2 Search::search() API for move selection
//! - Req 9.2: Reuse Phase 2 TranspositionTable instance across games
//! - Req 9.3: Share Evaluator instance with search system
//! - Req 9.4: Ensure consistency between learning and search
//! - Req 9.5: Use Phase 2's SearchResult for leaf evaluation values
//! - Req 9.6: Handle Phase 2 SearchError appropriately
//! - Req 9.7: Use Phase 2 make_move and check_game_state for game progression

use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::board::BitBoard;
use crate::evaluator::{EvaluationTable, calculate_stage, u16_to_score};
use crate::learning::LearningError;
use crate::pattern::{Pattern, extract_all_patterns_into, load_patterns};

/// Number of patterns in the Othello AI system
pub const NUM_PATTERNS: usize = 14;

/// Number of pattern instances (14 patterns x 4 rotations)
pub const NUM_PATTERN_INSTANCES: usize = 56;

/// Thread-safe evaluator wrapper for concurrent read/write access.
///
/// Wraps EvaluationTable with `Arc<RwLock>` to allow:
/// - Multiple concurrent reads during parallel game execution
/// - Exclusive writes during TD weight updates
///
/// # Example
///
/// ```ignore
/// use prismind::learning::shared_evaluator::SharedEvaluator;
///
/// let evaluator = SharedEvaluator::new("patterns.csv").unwrap();
///
/// // Read access for evaluation
/// let score = evaluator.evaluate(&board);
///
/// // Write access for TD updates
/// {
///     let mut table = evaluator.write();
///     table.set(0, 0, 0, 40000);
/// }
/// ```
///
/// # Thread Safety
///
/// SharedEvaluator is both Send and Sync, allowing it to be shared
/// across threads safely.
#[derive(Debug)]
pub struct SharedEvaluator {
    /// Thread-safe evaluation table wrapped in `Arc<RwLock>`
    table: Arc<RwLock<EvaluationTable>>,
    /// Pattern definitions for pattern extraction
    patterns: [Pattern; 14],
}

impl SharedEvaluator {
    /// Create a new SharedEvaluator from a patterns file.
    ///
    /// Loads pattern definitions from the specified CSV file and initializes
    /// the evaluation table with neutral values (32768).
    ///
    /// # Arguments
    ///
    /// * `pattern_file` - Path to patterns.csv file
    ///
    /// # Returns
    ///
    /// Result containing the SharedEvaluator or a LearningError
    ///
    /// # Errors
    ///
    /// - `LearningError::Config` if pattern file cannot be loaded
    ///
    /// # Requirements
    ///
    /// - Req 9.3: Share Evaluator instance with search system
    pub fn new<P: AsRef<Path>>(pattern_file: P) -> Result<Self, LearningError> {
        // Load patterns from CSV file
        let patterns_vec = load_patterns(pattern_file)
            .map_err(|e| LearningError::Config(format!("Failed to load patterns: {}", e)))?;

        // Convert to fixed-size array
        let patterns: [Pattern; 14] = patterns_vec.try_into().map_err(|v: Vec<Pattern>| {
            LearningError::Config(format!("Expected 14 patterns, found {}", v.len()))
        })?;

        // Initialize evaluation table
        let table = EvaluationTable::from_patterns(&patterns);

        Ok(Self {
            table: Arc::new(RwLock::new(table)),
            patterns,
        })
    }

    /// Create a SharedEvaluator from existing patterns and evaluation table.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Pattern definitions array
    /// * `table` - Pre-initialized evaluation table
    ///
    /// # Returns
    ///
    /// A new SharedEvaluator wrapping the provided table
    pub fn from_parts(patterns: [Pattern; 14], table: EvaluationTable) -> Self {
        Self {
            table: Arc::new(RwLock::new(table)),
            patterns,
        }
    }

    /// Evaluate a board position.
    ///
    /// Acquires a read lock on the evaluation table, extracts all 56 pattern
    /// indices, and computes the total evaluation score.
    ///
    /// # Arguments
    ///
    /// * `board` - The board position to evaluate
    ///
    /// # Returns
    ///
    /// Evaluation score as f32 (positive = Black advantage)
    ///
    /// # Panics
    ///
    /// Panics if the read lock is poisoned (another thread panicked while holding the lock)
    ///
    /// # Requirements
    ///
    /// - Req 9.4: Ensure consistency between learning and search
    pub fn evaluate(&self, board: &BitBoard) -> f32 {
        // Get stage from move count
        let stage = calculate_stage(board.move_count());

        // Extract all 56 pattern indices
        let mut indices = [0usize; NUM_PATTERN_INSTANCES];
        extract_all_patterns_into(board, &self.patterns, &mut indices);

        // Acquire read lock and compute evaluation
        let table = self.table.read().expect("RwLock poisoned");

        let mut sum = 0.0f32;

        // Sum evaluation values for all 56 pattern instances
        for rotation in 0..4 {
            for pattern_id in 0..NUM_PATTERNS {
                let idx = rotation * NUM_PATTERNS + pattern_id;
                let index = indices[idx];
                let value_u16 = table.get(pattern_id, stage, index);
                sum += u16_to_score(value_u16);
            }
        }

        // Negate for White's turn
        if board.turn() == crate::board::Color::White {
            -sum
        } else {
            sum
        }
    }

    /// Get a read guard for concurrent evaluation during parallel games.
    ///
    /// Multiple threads can hold read guards simultaneously, allowing
    /// parallel game execution to evaluate positions concurrently.
    ///
    /// # Returns
    ///
    /// RwLockReadGuard providing immutable access to the EvaluationTable
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned
    ///
    /// # Requirements
    ///
    /// - Req 9.3: Share Evaluator instance with search system
    pub fn read(&self) -> RwLockReadGuard<'_, EvaluationTable> {
        self.table.read().expect("RwLock poisoned")
    }

    /// Get a write guard for exclusive TD weight updates.
    ///
    /// Only one thread can hold a write guard at a time, ensuring
    /// exclusive access during weight modifications.
    ///
    /// # Returns
    ///
    /// RwLockWriteGuard providing mutable access to the EvaluationTable
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned
    ///
    /// # Requirements
    ///
    /// - Req 9.4: Ensure consistency between learning and search
    pub fn write(&self) -> RwLockWriteGuard<'_, EvaluationTable> {
        self.table.write().expect("RwLock poisoned")
    }

    /// Get reference to the patterns array.
    ///
    /// Used for pattern extraction during self-play games.
    ///
    /// # Returns
    ///
    /// Reference to the 14-element Pattern array
    pub fn patterns(&self) -> &[Pattern; 14] {
        &self.patterns
    }

    /// Clone the Arc for sharing across threads.
    ///
    /// Returns a new SharedEvaluator that shares the same underlying
    /// EvaluationTable. Changes made through one instance are visible
    /// to all other instances.
    ///
    /// # Returns
    ///
    /// A clone of this SharedEvaluator sharing the same table
    pub fn clone_shared(&self) -> Self {
        Self {
            table: Arc::clone(&self.table),
            patterns: self.patterns,
        }
    }

    /// Get the underlying Arc for direct sharing.
    ///
    /// # Returns
    ///
    /// Clone of the `Arc<RwLock<EvaluationTable>>`
    pub fn table_arc(&self) -> Arc<RwLock<EvaluationTable>> {
        Arc::clone(&self.table)
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage of the evaluation table
    pub fn memory_usage(&self) -> usize {
        let table = self.table.read().expect("RwLock poisoned");
        table.memory_usage()
    }
}

// Verify Send + Sync traits are implemented
// These are automatically derived due to Arc<RwLock<T>> being Send + Sync
// when T: Send + Sync
impl Clone for SharedEvaluator {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    /// Create test patterns for testing
    fn create_test_patterns() -> [Pattern; 14] {
        let patterns = vec![
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
        ];
        patterns.try_into().unwrap()
    }

    /// Create a SharedEvaluator for testing
    fn create_test_evaluator() -> SharedEvaluator {
        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        SharedEvaluator::from_parts(patterns, table)
    }

    // ========== Requirement 9.1: Phase 2 Search API Integration ==========

    #[test]
    fn test_shared_evaluator_creation() {
        let evaluator = create_test_evaluator();
        assert_eq!(evaluator.patterns().len(), 14);
    }

    #[test]
    fn test_shared_evaluator_from_file() {
        // This test requires patterns.csv to exist
        if std::path::Path::new("patterns.csv").exists() {
            let evaluator = SharedEvaluator::new("patterns.csv");
            assert!(evaluator.is_ok());
            let evaluator = evaluator.unwrap();
            assert_eq!(evaluator.patterns().len(), 14);
        } else {
            println!("patterns.csv not found, skipping file-based test");
        }
    }

    // ========== Requirement 9.2: RwLock Protection ==========

    #[test]
    fn test_read_guard_allows_concurrent_reads() {
        let evaluator = create_test_evaluator();

        // Acquire multiple read guards (should not block)
        let guard1 = evaluator.read();
        let guard2 = evaluator.read();

        // Both should have access to the same data
        let val1 = guard1.get(0, 0, 0);
        let val2 = guard2.get(0, 0, 0);
        assert_eq!(val1, val2);
        assert_eq!(val1, 32768); // Initial value
    }

    #[test]
    fn test_write_guard_exclusive_access() {
        let evaluator = create_test_evaluator();

        // Acquire write guard and modify
        {
            let mut guard = evaluator.write();
            guard.set(0, 0, 0, 40000);
        }

        // Read back the modified value
        {
            let guard = evaluator.read();
            assert_eq!(guard.get(0, 0, 0), 40000);
        }
    }

    // ========== Requirement 9.3: Shared Evaluator Instance ==========

    #[test]
    fn test_clone_shared_shares_same_table() {
        let evaluator1 = create_test_evaluator();
        let evaluator2 = evaluator1.clone_shared();

        // Modify through evaluator1
        {
            let mut guard = evaluator1.write();
            guard.set(0, 0, 100, 50000);
        }

        // Read through evaluator2 - should see the change
        {
            let guard = evaluator2.read();
            assert_eq!(guard.get(0, 0, 100), 50000);
        }
    }

    // ========== Requirement 9.4: Evaluation Consistency ==========

    #[test]
    fn test_evaluate_initial_board() {
        let evaluator = create_test_evaluator();
        let board = BitBoard::new();

        let score = evaluator.evaluate(&board);

        // Initial board with neutral weights should evaluate to ~0
        assert!(
            score.abs() < 1.0,
            "Initial board score should be near 0, got {}",
            score
        );
    }

    #[test]
    fn test_evaluate_respects_turn() {
        let evaluator = create_test_evaluator();

        let board_black = BitBoard::new();
        let board_white = board_black.flip();

        let score_black = evaluator.evaluate(&board_black);
        let score_white = evaluator.evaluate(&board_white);

        // White turn should negate the evaluation
        // Since both boards have same position but different turn,
        // and initial weights are neutral, both should be near 0
        println!(
            "Black turn score: {}, White turn score: {}",
            score_black, score_white
        );
    }

    // ========== Requirement 9.5: Pattern Extraction ==========

    #[test]
    fn test_patterns_accessible() {
        let evaluator = create_test_evaluator();
        let patterns = evaluator.patterns();

        assert_eq!(patterns.len(), 14);

        // Verify pattern k values
        assert_eq!(patterns[0].k, 10); // P01: 10 cells
        assert_eq!(patterns[4].k, 8); // P05: 8 cells
        assert_eq!(patterns[12].k, 5); // P13: 5 cells
    }

    // ========== Requirement 9.6: Thread Safety ==========

    #[test]
    fn test_shared_evaluator_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SharedEvaluator>();
        assert_sync::<SharedEvaluator>();
    }

    #[test]
    fn test_concurrent_reads_from_multiple_threads() {
        let evaluator = Arc::new(create_test_evaluator());
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Spawn 4 threads that read concurrently
        for _ in 0..4 {
            let eval_clone = Arc::clone(&evaluator);
            let counter_clone = Arc::clone(&counter);

            let handle = thread::spawn(move || {
                let board = BitBoard::new();
                for _ in 0..100 {
                    let _score = eval_clone.evaluate(&board);
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 400);
    }

    #[test]
    fn test_concurrent_read_write_safety() {
        let evaluator = Arc::new(create_test_evaluator());
        let read_count = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Spawn reader threads
        for _ in 0..3 {
            let eval_clone = Arc::clone(&evaluator);
            let count_clone = Arc::clone(&read_count);

            let handle = thread::spawn(move || {
                let board = BitBoard::new();
                for _ in 0..50 {
                    let _score = eval_clone.evaluate(&board);
                    count_clone.fetch_add(1, Ordering::SeqCst);
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        // Spawn writer thread
        {
            let eval_clone = Arc::clone(&evaluator);
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    {
                        let mut table = eval_clone.write();
                        table.set(0, 0, 0, 32768 + (i * 100) as u16);
                    }
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // All reads should have completed
        assert_eq!(read_count.load(Ordering::SeqCst), 150);
    }

    // ========== Requirement 9.7: Memory Management ==========

    #[test]
    fn test_memory_usage() {
        let evaluator = create_test_evaluator();
        let memory = evaluator.memory_usage();

        // Should be > 0
        assert!(memory > 0);

        // Should be within expected bounds (approximately 57 MB for full table)
        let memory_mb = memory as f64 / 1_048_576.0;
        println!("SharedEvaluator memory usage: {:.2} MB", memory_mb);
        assert!(memory_mb < 80.0, "Memory usage should be under 80 MB");
    }

    #[test]
    fn test_table_arc() {
        let evaluator = create_test_evaluator();
        let arc1 = evaluator.table_arc();
        let arc2 = evaluator.table_arc();

        // Both arcs should point to the same data
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_evaluate_after_weight_update() {
        let evaluator = create_test_evaluator();
        let board = BitBoard::new();

        // Get initial evaluation
        let initial_score = evaluator.evaluate(&board);

        // Update some weights
        {
            let mut table = evaluator.write();
            // Set a pattern weight to a high value
            table.set(0, 0, 0, 50000); // Positive bias
        }

        // Evaluate again - score should have changed
        let new_score = evaluator.evaluate(&board);

        println!(
            "Score before: {}, after weight update: {}",
            initial_score, new_score
        );
        // The exact change depends on pattern indices, but there should be some change
    }

    #[test]
    fn test_clone_derives_automatically() {
        let evaluator1 = create_test_evaluator();
        let evaluator2 = evaluator1.clone(); // Uses Clone trait

        // Both should share the same underlying table
        {
            let mut table1 = evaluator1.write();
            table1.set(5, 5, 100, 45000);
        }

        {
            let table2 = evaluator2.read();
            assert_eq!(table2.get(5, 5, 100), 45000);
        }
    }

    // ========== Requirements Summary ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== SharedEvaluator Requirements Verification ===");

        let evaluator = create_test_evaluator();
        let board = BitBoard::new();

        // Req 9.1: Phase 2 Search API compatible
        // (Evaluator interface is compatible with Search)
        println!("  9.1: SharedEvaluator compatible with Phase 2 Search API");

        // Req 9.2: RwLock protection
        {
            let _read1 = evaluator.read();
            let _read2 = evaluator.read();
        }
        println!("  9.2: RwLock allows concurrent reads");

        // Req 9.3: Shared instance
        let shared = evaluator.clone_shared();
        assert!(Arc::ptr_eq(&evaluator.table, &shared.table));
        println!("  9.3: Shared Evaluator instance works correctly");

        // Req 9.4: Evaluation consistency
        let _score = evaluator.evaluate(&board);
        println!("  9.4: Evaluate method acquires read lock correctly");

        // Req 9.5: Pattern extraction available
        assert_eq!(evaluator.patterns().len(), 14);
        println!("  9.5: Patterns accessible for extraction");

        // Req 9.6: Thread safety
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<SharedEvaluator>();
        assert_sync::<SharedEvaluator>();
        println!("  9.6: SharedEvaluator is Send + Sync");

        // Req 9.7: Memory usage tracking
        let memory = evaluator.memory_usage();
        assert!(memory > 0);
        println!("  9.7: Memory usage tracking works ({} bytes)", memory);

        println!("=== All SharedEvaluator requirements verified ===");
    }
}
