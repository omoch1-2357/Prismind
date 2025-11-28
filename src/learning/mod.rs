//! Phase 3: TD(lambda)-Leaf Reinforcement Learning System
//!
//! This module implements a self-play learning system that trains the evaluation function
//! through temporal difference learning with eligibility traces.
//!
//! # Overview
//!
//! The learning system uses TD(lambda)-Leaf with the following components:
//! - **TD-Leaf Learning**: Backward updates from game outcomes to update pattern weights
//! - **Eligibility Traces**: Credit assignment across visited states with lambda decay
//! - **Adam Optimizer**: Adaptive learning rate for stable gradient updates
//! - **Self-Play Engine**: Generates training data through epsilon-greedy exploration
//!
//! # Architecture
//!
//! ```text
//! TrainingEngine
//!     |-- SelfPlayGame (parallel execution)
//!     |       |-- Search API (Phase 2)
//!     |       |-- SharedEvaluator (read lock)
//!     |-- TDLearner (sequential updates)
//!     |       |-- AdamOptimizer
//!     |       |-- EligibilityTrace
//!     |       |-- SharedEvaluator (write lock)
//!     |-- CheckpointManager
//!     |-- TrainingLogger
//! ```
//!
//! # Requirements Coverage
//!
//! - Req 12.1-12.6: Error handling with `LearningError` enum
//! - Req 1.1-1.8: TD(lambda)-Leaf algorithm (TDLearner)
//! - Req 2.1-2.6: Eligibility traces (EligibilityTrace)
//! - Req 3.1-3.10: Adam optimizer (AdamOptimizer)
//! - Req 4.1-4.10: Self-play engine (SelfPlayGame)
//! - Req 5.1-5.7: Game history recording (GameHistory)
//! - Req 9.1-9.7: Phase 2 integration (SharedEvaluator)
//! - Req 6.1-6.10: Checkpoint management (CheckpointManager)
//! - Req 7.1-7.8: Training logging (TrainingLogger)
//!
//! # Example
//!
//! ```ignore
//! use prismind::learning::{LearningError, TrainingConfig, TrainingEngine};
//!
//! let config = TrainingConfig::default();
//! let mut engine = TrainingEngine::new(config)?;
//! let stats = engine.train(1_000_000)?;
//! ```

use crate::search::SearchError;
use thiserror::Error;

// Submodules
pub mod adam;
pub mod checkpoint;
pub mod convergence;
pub mod eligibility_trace;
pub mod error_handler;
pub mod game_history;
pub mod logger;
pub mod score;
pub mod self_play;
pub mod shared_evaluator;
pub mod td_learner;
pub mod training_engine;

// Re-export public types
pub use adam::{AdamMoments, AdamOptimizer};
pub use checkpoint::{CHECKPOINT_MAGIC, CheckpointManager, CheckpointMeta};
pub use convergence::{
    CONVERGENCE_REPORT_INTERVAL, ConvergenceMetrics, ConvergenceMonitor, IMPROVEMENT_THRESHOLD,
    MIN_WIN_RATE_VS_RANDOM, RUNNING_AVERAGE_WINDOW, STAGNATION_WINDOW, TARGET_UPDATES_PER_ENTRY,
};
pub use eligibility_trace::EligibilityTrace;
pub use error::LearningError;
pub use error_handler::{
    ERROR_THRESHOLD_PERCENT, ERROR_WINDOW_SIZE, ErrorPatternSummary, ErrorRecord, ErrorTracker,
    ErrorType, EvalRecovery, PanicCatchResult, catch_panic, save_checkpoint_with_retry,
};
pub use game_history::{
    GameHistory, MAX_MOVES_PER_GAME, MoveRecord as HistoryMoveRecord, NUM_PATTERN_INSTANCES,
};
pub use logger::{
    BatchStats, DEFAULT_BATCH_INTERVAL, DEFAULT_CHECKPOINT_INTERVAL, DEFAULT_DETAILED_INTERVAL,
    DetailedStats, EVAL_DIVERGENCE_THRESHOLD, SyncTrainingLogger, TrainingLogger,
};
pub use score::{CENTER, SCALE, initial_value, stone_diff_to_u16, u16_to_stone_diff};
pub use self_play::{DEFAULT_SEARCH_TIME_MS, EpsilonSchedule, GameResult, play_game};
pub use shared_evaluator::SharedEvaluator;
pub use td_learner::{MoveRecord, TDLearner, TDUpdateStats};
pub use training_engine::{
    DEFAULT_NUM_THREADS, DEFAULT_TT_SIZE_MB, MAX_MEMORY_BUDGET, MIN_TT_SIZE_MB,
    TOTAL_PATTERN_ENTRIES, TrainingConfig, TrainingEngine, TrainingStats,
};

/// Error types for the learning module
mod error {
    use super::*;

    /// Learning system error type
    ///
    /// Provides structured error variants for all learning operations,
    /// enabling appropriate handling and recovery strategies.
    ///
    /// # Error Categories
    ///
    /// - **I/O Errors**: File operations, checkpoint read/write
    /// - **Validation Errors**: Checkpoint corruption, configuration issues
    /// - **Runtime Errors**: Search failures, evaluation divergence
    /// - **Resource Errors**: Memory allocation failures
    /// - **Control Errors**: Training interruption (SIGINT/SIGTERM)
    ///
    /// # Recovery Strategies
    ///
    /// | Variant | Recovery |
    /// |---------|----------|
    /// | `Io` | Retry once for saves; log and continue for logging |
    /// | `InvalidCheckpoint` | Report error, allow fresh start |
    /// | `Search` | Log error, skip game, continue training |
    /// | `EvaluationDivergence` | Reset affected entries to 32768, log warning |
    /// | `MemoryAllocation` | Reduce TT size, retry |
    /// | `Config` | Report configuration issue, abort |
    /// | `Interrupted` | Save checkpoint, exit gracefully |
    #[derive(Error, Debug)]
    pub enum LearningError {
        /// I/O errors (file, checkpoint)
        ///
        /// Occurs during file operations such as reading/writing checkpoints
        /// or log files. Implements `From<std::io::Error>` for automatic conversion.
        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),

        /// Checkpoint corruption or version mismatch
        ///
        /// Occurs when loading a checkpoint with:
        /// - Invalid magic header
        /// - Version mismatch
        /// - Truncated or corrupted data
        #[error("Invalid checkpoint: {0}")]
        InvalidCheckpoint(String),

        /// Search system error (from Phase 2)
        ///
        /// Propagates errors from the Phase 2 search system.
        /// Implements `From<SearchError>` for automatic conversion.
        #[error("Search error: {0}")]
        Search(#[from] SearchError),

        /// Evaluation divergence (NaN, extreme values)
        ///
        /// Occurs when evaluation values become:
        /// - NaN (Not a Number)
        /// - Infinite
        /// - Extremely large (outside reasonable bounds)
        ///
        /// Recovery: Reset affected pattern entries to neutral value (32768).
        #[error("Evaluation diverged: {0}")]
        EvaluationDivergence(String),

        /// Memory allocation failure
        ///
        /// Occurs when unable to allocate required memory for:
        /// - Adam optimizer moments (~228 MB)
        /// - Transposition table (128-256 MB)
        /// - Other data structures
        ///
        /// Recovery: Reduce transposition table size and retry.
        #[error("Memory allocation failed: {0}")]
        MemoryAllocation(String),

        /// Configuration error
        ///
        /// Occurs when configuration values are invalid:
        /// - Invalid hyperparameters (lambda, learning rate)
        /// - Invalid file paths
        /// - Missing required files
        #[error("Configuration error: {0}")]
        Config(String),

        /// Training interrupted (SIGINT/SIGTERM)
        ///
        /// Occurs when the training process receives an interrupt signal.
        /// The system should save a checkpoint before exiting.
        #[error("Training interrupted")]
        Interrupted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let learning_err: LearningError = io_err.into();

        match learning_err {
            LearningError::Io(_) => {}
            _ => panic!("Expected Io variant"),
        }
    }

    #[test]
    fn test_search_error_conversion() {
        let search_err = SearchError::MemoryAllocation("test".to_string());
        let learning_err: LearningError = search_err.into();

        match learning_err {
            LearningError::Search(_) => {}
            _ => panic!("Expected Search variant"),
        }
    }

    #[test]
    fn test_invalid_checkpoint_error() {
        let err = LearningError::InvalidCheckpoint("bad magic".to_string());
        assert!(err.to_string().contains("checkpoint"));
        assert!(err.to_string().contains("bad magic"));
    }

    #[test]
    fn test_evaluation_divergence_error() {
        let err = LearningError::EvaluationDivergence("NaN in pattern 5".to_string());
        assert!(err.to_string().contains("diverged"));
    }

    #[test]
    fn test_memory_allocation_error() {
        let err = LearningError::MemoryAllocation("failed to allocate 228MB".to_string());
        assert!(err.to_string().contains("Memory"));
    }

    #[test]
    fn test_config_error() {
        let err = LearningError::Config("lambda must be in [0, 1]".to_string());
        assert!(err.to_string().contains("Configuration"));
    }

    #[test]
    fn test_interrupted_error() {
        let err = LearningError::Interrupted;
        assert!(err.to_string().contains("interrupted"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // LearningError should be Send + Sync for use with parallel execution
        assert_send::<LearningError>();
        assert_sync::<LearningError>();
    }
}
