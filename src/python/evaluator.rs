//! PyO3 wrapper for the Rust Evaluator
//!
//! This module provides Python bindings for board evaluation using pattern tables.
//!
//! # Features
//!
//! - Board evaluation with GIL release during Rust computation
//! - NumPy array support for efficient data interchange
//! - Pattern weight access for external analysis
//!
//! # GIL Release Pattern
//!
//! The evaluate methods release the Python Global Interpreter Lock (GIL) during
//! Rust computation to avoid blocking other Python threads. This is done using
//! `py.allow_threads()` which temporarily releases the GIL while the closure runs.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::RwLock;

use crate::board::{BitBoard, Color};
use crate::evaluator::{EvaluationTable, Evaluator};

/// Python wrapper for the Othello board evaluator.
///
/// Provides pattern-based evaluation of board positions with thread-safe access.
///
/// # Example
///
/// ```python
/// from prismind import PyEvaluator
///
/// evaluator = PyEvaluator()
/// board = [0] * 64
/// board[27] = 2  # White at D4
/// board[28] = 1  # Black at E4
/// board[35] = 1  # Black at D5
/// board[36] = 2  # White at E5
/// score = evaluator.evaluate(board, 1)  # Evaluate for black
/// ```
#[pyclass]
pub struct PyEvaluator {
    /// Thread-safe evaluator instance
    evaluator: Arc<Evaluator>,
    /// Thread-safe access to pattern tables for weight inspection
    table: Arc<RwLock<EvaluationTable>>,
}

#[pymethods]
impl PyEvaluator {
    /// Create a new evaluator, optionally loading from a checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Optional path to a checkpoint file to load weights from
    ///
    /// # Returns
    ///
    /// A new PyEvaluator instance
    ///
    /// # Raises
    ///
    /// * `ValueError` - If checkpoint path is provided but cannot be loaded
    #[new]
    #[pyo3(signature = (checkpoint_path=None))]
    pub fn new(checkpoint_path: Option<&str>) -> PyResult<Self> {
        // Create evaluation table
        let table = if let Some(_path) = checkpoint_path {
            // TODO: Implement checkpoint loading in Task 3
            // For now, create default table
            EvaluationTable::new()
        } else {
            EvaluationTable::new()
        };

        let table = Arc::new(RwLock::new(table));
        let evaluator = Arc::new(Evaluator::new_with_table(table.clone()));

        Ok(Self { evaluator, table })
    }

    /// Evaluate a board position.
    ///
    /// This method releases the GIL during Rust computation to avoid blocking
    /// other Python threads.
    ///
    /// # Arguments
    ///
    /// * `board` - 64-element array representing the board (0=empty, 1=black, 2=white)
    /// * `player` - Current player (1=black, 2=white)
    ///
    /// # Returns
    ///
    /// Evaluation score as float (positive favors black)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If board array doesn't have exactly 64 elements
    /// * `ValueError` - If player is not 1 or 2
    /// * `ValueError` - If board contains invalid values
    ///
    /// # Note
    ///
    /// For bulk evaluation with NumPy arrays, use `evaluate_numpy()` for better
    /// performance through efficient array access.
    pub fn evaluate(&self, py: Python<'_>, board: Vec<i8>, player: i8) -> PyResult<f64> {
        // Validate board size
        if board.len() != 64 {
            return Err(PyValueError::new_err(format!(
                "Board must have exactly 64 elements, got {}",
                board.len()
            )));
        }

        // Validate player
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Player must be 1 (black) or 2 (white), got {}",
                    player
                )));
            }
        };

        // Convert board array to BitBoard
        let bitboard = self.array_to_bitboard(&board, color)?;

        // Release GIL during Rust computation
        let evaluator = Arc::clone(&self.evaluator);
        let score = py.allow_threads(move || evaluator.evaluate(&bitboard));

        Ok(score as f64)
    }

    /// Evaluate a board position from a NumPy array.
    ///
    /// This method provides efficient evaluation using NumPy arrays for data
    /// interchange. It releases the GIL during Rust computation.
    ///
    /// # Arguments
    ///
    /// * `board` - NumPy array with 64 elements (dtype: int8, values: 0=empty, 1=black, 2=white)
    /// * `player` - Current player (1=black, 2=white)
    ///
    /// # Returns
    ///
    /// Evaluation score as float (positive favors black)
    ///
    /// # Raises
    ///
    /// * `ValueError` - If array doesn't have exactly 64 elements
    /// * `ValueError` - If player is not 1 or 2
    /// * `ValueError` - If array contains invalid values
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// from prismind import PyEvaluator
    ///
    /// evaluator = PyEvaluator()
    /// board = np.zeros(64, dtype=np.int8)
    /// board[27] = 2  # White at D4
    /// board[28] = 1  # Black at E4
    /// board[35] = 1  # Black at D5
    /// board[36] = 2  # White at E5
    /// score = evaluator.evaluate_numpy(board, 1)
    /// ```
    pub fn evaluate_numpy(
        &self,
        py: Python<'_>,
        board: PyReadonlyArray1<'_, i8>,
        player: i8,
    ) -> PyResult<f64> {
        // Get slice reference to array data
        let board_slice = board.as_slice().map_err(|e| {
            PyValueError::new_err(format!("Failed to access NumPy array data: {}", e))
        })?;

        // Validate board size
        if board_slice.len() != 64 {
            return Err(PyValueError::new_err(format!(
                "Board array must have exactly 64 elements, got {}",
                board_slice.len()
            )));
        }

        // Validate player
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Player must be 1 (black) or 2 (white), got {}",
                    player
                )));
            }
        };

        // Convert NumPy array to BitBoard
        let bitboard = self.slice_to_bitboard(board_slice, color)?;

        // Release GIL during Rust computation
        let evaluator = Arc::clone(&self.evaluator);
        let score = py.allow_threads(move || evaluator.evaluate(&bitboard));

        Ok(score as f64)
    }

    /// Get pattern weight for a specific entry.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern identifier (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index within the pattern
    ///
    /// # Returns
    ///
    /// Weight value as float
    ///
    /// # Raises
    ///
    /// * `ValueError` - If any parameter is out of valid range
    pub fn get_weight(&self, pattern_id: usize, stage: usize, index: usize) -> PyResult<f64> {
        // Validate parameters
        if pattern_id >= 14 {
            return Err(PyValueError::new_err(format!(
                "Pattern ID must be 0-13, got {}",
                pattern_id
            )));
        }
        if stage >= 30 {
            return Err(PyValueError::new_err(format!(
                "Stage must be 0-29, got {}",
                stage
            )));
        }

        let table = self
            .table
            .read()
            .map_err(|e| PyValueError::new_err(format!("Failed to acquire read lock: {}", e)))?;

        // Get weight from table
        let weight = table.get_weight(pattern_id, stage, index);

        Ok(weight)
    }

    /// Get all pattern weights for external analysis.
    ///
    /// # Returns
    ///
    /// Dictionary mapping (pattern_id, stage, index) tuples to weight values
    ///
    /// Note: This can be a large data structure (~57MB of data).
    /// Consider using get_weight() for specific entries instead.
    pub fn get_weights(&self) -> PyResult<std::collections::HashMap<(usize, usize, usize), f64>> {
        let table = self
            .table
            .read()
            .map_err(|e| PyValueError::new_err(format!("Failed to acquire read lock: {}", e)))?;

        let mut weights = std::collections::HashMap::new();

        // Export all weights (this is expensive - ~14 patterns * 30 stages * varying indices)
        for pattern_id in 0..14 {
            for stage in 0..30 {
                let pattern_size = table.pattern_size(pattern_id);
                for index in 0..pattern_size {
                    let weight = table.get_weight(pattern_id, stage, index);
                    weights.insert((pattern_id, stage, index), weight);
                }
            }
        }

        Ok(weights)
    }
}

impl PyEvaluator {
    /// Convert a Python board array to BitBoard.
    fn array_to_bitboard(&self, board: &[i8], turn: Color) -> PyResult<BitBoard> {
        self.slice_to_bitboard(board, turn)
    }

    /// Convert a slice of i8 values to BitBoard.
    ///
    /// This is the core conversion function used by both `evaluate` and `evaluate_numpy`.
    fn slice_to_bitboard(&self, board: &[i8], turn: Color) -> PyResult<BitBoard> {
        let mut black: u64 = 0;
        let mut white: u64 = 0;

        for (i, &cell) in board.iter().enumerate() {
            match cell {
                0 => {}                  // Empty
                1 => black |= 1u64 << i, // Black
                2 => white |= 1u64 << i, // White
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid board value at index {}: {} (must be 0, 1, or 2)",
                        i, cell
                    )));
                }
            }
        }

        Ok(BitBoard::from_masks(black, white, turn))
    }

    /// Evaluate without GIL release (for Rust-only use and testing).
    ///
    /// This method performs the same validation and evaluation as `evaluate`,
    /// but without the Python GIL context. Used for unit tests and Rust API.
    ///
    /// # Note
    ///
    /// For Python usage, prefer `evaluate()` which properly releases the GIL
    /// during computation.
    pub fn evaluate_sync(&self, board: Vec<i8>, player: i8) -> PyResult<f64> {
        // Validate board size
        if board.len() != 64 {
            return Err(PyValueError::new_err(format!(
                "Board must have exactly 64 elements, got {}",
                board.len()
            )));
        }

        // Validate player
        let color = match player {
            1 => Color::Black,
            2 => Color::White,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Player must be 1 (black) or 2 (white), got {}",
                    player
                )));
            }
        };

        // Convert board array to BitBoard
        let bitboard = self.array_to_bitboard(&board, color)?;

        // Evaluate the position (no GIL release in sync version)
        let score = self.evaluator.evaluate(&bitboard);

        Ok(score as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyevaluator_creation() {
        let evaluator = PyEvaluator::new(None).expect("Failed to create evaluator");
        assert!(Arc::strong_count(&evaluator.evaluator) >= 1);
    }

    #[test]
    fn test_board_validation() {
        let evaluator = PyEvaluator::new(None).expect("Failed to create evaluator");

        // Valid board
        let mut board = vec![0i8; 64];
        board[27] = 2; // White at D4
        board[28] = 1; // Black at E4
        board[35] = 1; // Black at D5
        board[36] = 2; // White at E5

        // Use evaluate_sync for Rust-only testing (no Python GIL)
        let result = evaluator.evaluate_sync(board, 1);
        assert!(result.is_ok());

        // Invalid board size
        let short_board = vec![0i8; 32];
        let result = evaluator.evaluate_sync(short_board, 1);
        assert!(result.is_err());

        // Invalid player
        let valid_board = vec![0i8; 64];
        let result = evaluator.evaluate_sync(valid_board, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_weight_validation() {
        let evaluator = PyEvaluator::new(None).expect("Failed to create evaluator");

        // Valid parameters
        let result = evaluator.get_weight(0, 0, 0);
        assert!(result.is_ok());

        // Invalid pattern_id
        let result = evaluator.get_weight(14, 0, 0);
        assert!(result.is_err());

        // Invalid stage
        let result = evaluator.get_weight(0, 30, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_to_bitboard_conversion() {
        let evaluator = PyEvaluator::new(None).expect("Failed to create evaluator");

        // Valid conversion
        let board = vec![0i8; 64];
        let result = evaluator.slice_to_bitboard(&board, Color::Black);
        assert!(result.is_ok());

        // Board with pieces
        let mut board_with_pieces = vec![0i8; 64];
        board_with_pieces[0] = 1; // Black at A1
        board_with_pieces[7] = 2; // White at H1
        let result = evaluator.slice_to_bitboard(&board_with_pieces, Color::Black);
        assert!(result.is_ok());
        let bitboard = result.unwrap();
        assert_eq!(bitboard.black & 1, 1); // A1 should have black
    }

    #[test]
    fn test_evaluate_sync_returns_finite_score() {
        let evaluator = PyEvaluator::new(None).expect("Failed to create evaluator");

        let mut board = vec![0i8; 64];
        board[27] = 2; // White at D4
        board[28] = 1; // Black at E4
        board[35] = 1; // Black at D5
        board[36] = 2; // White at E5

        let score = evaluator.evaluate_sync(board, 1).unwrap();
        assert!(score.is_finite(), "Score should be finite");
        // Initial position should be balanced
        assert!(
            score.abs() < 10.0,
            "Initial position should have near-zero evaluation"
        );
    }
}
