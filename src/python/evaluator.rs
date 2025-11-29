//! PyO3 wrapper for the Rust Evaluator
//!
//! This module provides Python bindings for board evaluation using pattern tables.

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
    pub fn evaluate(&self, board: Vec<i8>, player: i8) -> PyResult<f64> {
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

        // Evaluate the position
        let score = self.evaluator.evaluate(&bitboard);

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

        let result = evaluator.evaluate(board, 1);
        assert!(result.is_ok());

        // Invalid board size
        let short_board = vec![0i8; 32];
        let result = evaluator.evaluate(short_board, 1);
        assert!(result.is_err());

        // Invalid player
        let valid_board = vec![0i8; 64];
        let result = evaluator.evaluate(valid_board, 3);
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
}
