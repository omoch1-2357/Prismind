//! PyO3 Python bindings for Prismind Othello AI
//!
//! This module provides Python bindings for the Prismind Othello AI engine,
//! exposing pattern-based evaluation, training management, and debugging utilities.
//!
//! # Module Structure
//!
//! The Python module `prismind._prismind` exposes the following classes:
//!
//! - [`PyEvaluator`] - Board evaluation using pattern tables
//! - [`PyTrainingManager`] - Training session control with pause/resume
//! - [`PyCheckpointManager`] - Checkpoint save/load with integrity verification
//! - [`PyLearningState`] - Training state container for checkpoint operations
//! - [`PyStatisticsManager`] - Statistics and monitoring aggregation
//! - [`PyDebugModule`] - Debugging and diagnostic utilities
//!
//! # Example
//!
//! ```python
//! from prismind import PyEvaluator
//!
//! evaluator = PyEvaluator()
//! board = [0] * 64  # Empty board
//! board[27] = 2  # White at D4
//! board[28] = 1  # Black at E4
//! score = evaluator.evaluate(board, 1)
//! print(f"Evaluation: {score}")
//! ```

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
mod evaluator;

#[cfg(feature = "pyo3")]
mod checkpoint_manager;

#[cfg(feature = "pyo3")]
mod training_manager;

#[cfg(feature = "pyo3")]
mod statistics_manager;

#[cfg(feature = "pyo3")]
mod debug_module;

#[cfg(feature = "pyo3")]
pub use evaluator::PyEvaluator;

#[cfg(feature = "pyo3")]
pub use checkpoint_manager::{PyCheckpointManager, PyLearningState};

#[cfg(feature = "pyo3")]
pub use training_manager::{PyTrainingManager, PyTrainingResult};

#[cfg(feature = "pyo3")]
pub use statistics_manager::PyStatisticsManager;

#[cfg(feature = "pyo3")]
pub use debug_module::PyDebugModule;

/// PyO3 module entry point for the Prismind Python extension.
///
/// This function is called by Python when importing `prismind._prismind`.
/// It registers all PyO3 classes and module metadata.
#[cfg(feature = "pyo3")]
#[pymodule]
fn _prismind(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register PyEvaluator class
    m.add_class::<PyEvaluator>()?;

    // Register PyCheckpointManager and PyLearningState (Task 3 - implemented)
    m.add_class::<PyCheckpointManager>()?;
    m.add_class::<PyLearningState>()?;

    // Register PyTrainingManager and PyTrainingResult (Task 6 - implemented)
    m.add_class::<PyTrainingManager>()?;
    m.add_class::<PyTrainingResult>()?;

    // Register PyStatisticsManager (Task 8 - implemented)
    m.add_class::<PyStatisticsManager>()?;

    // Register placeholder classes (implemented as simple PyO3 classes)
    // These will be fully implemented in subsequent tasks
    m.add_class::<PyDebugModule>()?;

    // Add module version from Cargo package
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

// PyDebugModule is now fully implemented in debug_module.rs (Task 10)

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // This test verifies the module structure compiles correctly
        // Actual functionality is tested in integration tests
        // If this test runs, the module compiled successfully
    }

    #[cfg(feature = "pyo3")]
    #[test]
    fn test_pyo3_classes_exist() {
        // Verify all PyO3 classes are defined
        use super::*;
        let _ = std::any::type_name::<PyEvaluator>();
        let _ = std::any::type_name::<PyTrainingManager>();
        let _ = std::any::type_name::<PyCheckpointManager>();
        let _ = std::any::type_name::<PyLearningState>();
        let _ = std::any::type_name::<PyStatisticsManager>();
        let _ = std::any::type_name::<PyDebugModule>();
    }
}
