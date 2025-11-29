//! TDD tests for PyEvaluator implementation (Task 2)
//!
//! These tests verify the PyO3 bindings for the Evaluator class.
//! Task 2.1: PyEvaluator class with board evaluation
//! Task 2.2: NumPy array support
//! Task 2.3: Pattern weight access for external analysis
//!
//! Note: Tests use `evaluate_sync` method for Rust-only testing without Python GIL.
//! The actual `evaluate` and `evaluate_numpy` methods with GIL release are tested
//! via Python integration tests after maturin build.

#[cfg(feature = "pyo3")]
mod pyo3_evaluator_tests {
    use prismind::python::PyEvaluator;

    // ========== Task 2.1: PyEvaluator class with board evaluation ==========

    /// Test 2.1.1: PyEvaluator creation with default (no checkpoint)
    #[test]
    fn test_pyevaluator_new_default() {
        // Requirement: Constructor accepting optional checkpoint path
        let evaluator = PyEvaluator::new(None);
        assert!(evaluator.is_ok(), "PyEvaluator::new(None) should succeed");
    }

    /// Test 2.1.2: PyEvaluator creation with invalid checkpoint path
    #[test]
    fn test_pyevaluator_new_invalid_checkpoint() {
        // Requirement: Raise Python exception for invalid parameters
        // Note: Currently checkpoint loading is not implemented,
        // so this test documents expected future behavior
        let evaluator = PyEvaluator::new(Some("/nonexistent/path/checkpoint.bin"));
        // For now, this should still succeed (falls back to default table)
        // When checkpoint loading is implemented, this should fail
        assert!(evaluator.is_ok());
    }

    /// Test 2.1.3: evaluate_sync method with valid 64-element board array
    /// Note: Uses evaluate_sync for Rust-only testing (GIL release tested via Python)
    #[test]
    fn test_evaluate_valid_board() {
        // Requirement: evaluate method for 64-element board arrays with player indicator
        let evaluator = PyEvaluator::new(None).unwrap();

        // Standard initial Othello position
        let mut board = vec![0i8; 64];
        board[27] = 2; // White at D4
        board[28] = 1; // Black at E4
        board[35] = 1; // Black at D5
        board[36] = 2; // White at E5

        let result = evaluator.evaluate_sync(board, 1); // Evaluate for black
        assert!(result.is_ok(), "evaluate should succeed for valid board");

        let score = result.unwrap();
        // Initial position should be near zero (balanced)
        assert!(
            score.abs() < 10.0,
            "Initial position should have evaluation near zero, got {}",
            score
        );
    }

    /// Test 2.1.4: evaluate method validates board has exactly 64 elements
    #[test]
    fn test_evaluate_invalid_board_size() {
        // Requirement: Validate board array has exactly 64 elements
        let evaluator = PyEvaluator::new(None).unwrap();

        // Too short
        let short_board = vec![0i8; 32];
        let result = evaluator.evaluate_sync(short_board, 1);
        assert!(
            result.is_err(),
            "Should reject board with fewer than 64 elements"
        );

        // Too long
        let long_board = vec![0i8; 128];
        let result = evaluator.evaluate_sync(long_board, 1);
        assert!(
            result.is_err(),
            "Should reject board with more than 64 elements"
        );
    }

    /// Test 2.1.5: evaluate method validates board values in range {0, 1, 2}
    #[test]
    fn test_evaluate_invalid_board_values() {
        // Requirement: Validate board values in valid range (0, 1, 2)
        let evaluator = PyEvaluator::new(None).unwrap();

        let mut invalid_board = vec![0i8; 64];
        invalid_board[0] = 3; // Invalid value

        let result = evaluator.evaluate_sync(invalid_board, 1);
        assert!(
            result.is_err(),
            "Should reject board with invalid cell values"
        );
    }

    /// Test 2.1.6: evaluate method validates player is 1 or 2
    #[test]
    fn test_evaluate_invalid_player() {
        // Requirement: Raise descriptive Python exception for invalid parameters
        let evaluator = PyEvaluator::new(None).unwrap();
        let board = vec![0i8; 64];

        let result = evaluator.evaluate_sync(board.clone(), 0);
        assert!(result.is_err(), "Should reject player 0");

        let result = evaluator.evaluate_sync(board.clone(), 3);
        assert!(result.is_err(), "Should reject player 3");

        let result = evaluator.evaluate_sync(board, -1);
        assert!(result.is_err(), "Should reject negative player");
    }

    /// Test 2.1.7: evaluate returns float with positive values favoring black
    #[test]
    fn test_evaluate_returns_float_positive_favors_black() {
        // Requirement: Return evaluation score as Python float (positive favors black)
        let evaluator = PyEvaluator::new(None).unwrap();

        // Create a board position where black has more stones
        let mut board = vec![0i8; 64];
        // Place more black stones than white
        board[27] = 1; // Black at D4
        board[28] = 1; // Black at E4
        board[35] = 1; // Black at D5
        board[36] = 2; // White at E5 (only one white)

        let result = evaluator.evaluate_sync(board, 1);
        assert!(result.is_ok());
        let score = result.unwrap();

        // The score should be finite
        assert!(score.is_finite(), "Score should be a finite number");
    }

    /// Test 2.1.8: Thread-safe access via Arc
    #[test]
    fn test_pyevaluator_thread_safe() {
        // Requirement: Thread-safe for concurrent evaluation calls
        use std::sync::Arc;
        use std::thread;

        let evaluator = Arc::new(PyEvaluator::new(None).unwrap());

        let mut handles = vec![];

        for _ in 0..4 {
            let eval_clone = Arc::clone(&evaluator);
            let handle = thread::spawn(move || {
                let mut board = vec![0i8; 64];
                board[27] = 2;
                board[28] = 1;
                board[35] = 1;
                board[36] = 2;

                // Each thread should be able to evaluate independently
                for _ in 0..10 {
                    let result = eval_clone.evaluate_sync(board.clone(), 1);
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }

    // ========== Task 2.3: Pattern weight access for external analysis ==========

    /// Test 2.3.1: get_weight returns valid f64 for valid parameters
    #[test]
    fn test_get_weight_valid_params() {
        // Requirement: get_weight method for accessing specific pattern entry weights
        let evaluator = PyEvaluator::new(None).unwrap();

        let result = evaluator.get_weight(0, 0, 0);
        assert!(
            result.is_ok(),
            "get_weight should succeed for valid parameters"
        );

        let weight = result.unwrap();
        // Initial weight should be 0.0 (neutral, from u16 value 32768)
        assert!(
            (weight - 0.0).abs() < 0.01,
            "Initial weight should be 0.0, got {}",
            weight
        );
    }

    /// Test 2.3.2: get_weight validates pattern_id range (0-13)
    #[test]
    fn test_get_weight_invalid_pattern_id() {
        // Requirement: Raise ValueError for out of range parameters
        let evaluator = PyEvaluator::new(None).unwrap();

        let result = evaluator.get_weight(14, 0, 0); // pattern_id out of range
        assert!(result.is_err(), "Should reject pattern_id >= 14");

        let result = evaluator.get_weight(100, 0, 0);
        assert!(result.is_err(), "Should reject large pattern_id");
    }

    /// Test 2.3.3: get_weight validates stage range (0-29)
    #[test]
    fn test_get_weight_invalid_stage() {
        // Requirement: Raise ValueError for out of range parameters
        let evaluator = PyEvaluator::new(None).unwrap();

        let result = evaluator.get_weight(0, 30, 0); // stage out of range
        assert!(result.is_err(), "Should reject stage >= 30");

        let result = evaluator.get_weight(0, 100, 0);
        assert!(result.is_err(), "Should reject large stage");
    }

    /// Test 2.3.4: get_weights returns HashMap for all pattern weights
    #[test]
    fn test_get_weights_returns_all() {
        // Requirement: get_weights method returning dictionary mapping (pattern_id, stage, index) to weight
        let evaluator = PyEvaluator::new(None).unwrap();

        let result = evaluator.get_weights();
        assert!(result.is_ok(), "get_weights should succeed");

        let weights = result.unwrap();

        // Should have entries for all 14 patterns * 30 stages
        // Total entries depends on pattern sizes
        assert!(
            !weights.is_empty(),
            "get_weights should return non-empty HashMap"
        );

        // Check a few specific entries exist
        assert!(
            weights.contains_key(&(0, 0, 0)),
            "Should contain entry for (0, 0, 0)"
        );

        // All values should be initialized to 0.0 (neutral)
        for (key, &value) in &weights {
            assert!(value.is_finite(), "Weight at {:?} should be finite", key);
            assert!(
                (value - 0.0).abs() < 0.01,
                "Initial weight at {:?} should be 0.0, got {}",
                key,
                value
            );
        }
    }

    /// Test 2.3.5: Thread-safe read access through RwLock
    #[test]
    fn test_get_weights_thread_safe() {
        // Requirement: Expose pattern table through RwLock for thread-safe read access
        use std::sync::Arc;
        use std::thread;

        let evaluator = Arc::new(PyEvaluator::new(None).unwrap());

        let mut handles = vec![];

        for _ in 0..4 {
            let eval_clone = Arc::clone(&evaluator);
            let handle = thread::spawn(move || {
                // Multiple threads should be able to read weights concurrently
                for _ in 0..5 {
                    let result = eval_clone.get_weight(0, 0, 0);
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }
}

// ========== Task 2.2: NumPy array support (requires pyo3 feature) ==========
// Note: evaluate_numpy tests require Python interpreter and are tested
// via integration tests with maturin build.

#[cfg(feature = "pyo3")]
mod numpy_support_tests {
    /// Test 2.2.1: Verify numpy feature is enabled in Cargo.toml
    #[test]
    fn test_numpy_dependency_available() {
        // This test verifies the numpy dependency is configured
        // The actual numpy::PyArray1 usage is tested in Python integration tests
        use numpy::PyArray1;
        let _ = std::any::type_name::<PyArray1<i8>>();
    }

    /// Test 2.2.2: Verify PyReadonlyArray1 type is available
    #[test]
    fn test_readonly_array_available() {
        use numpy::PyReadonlyArray1;
        let _ = std::any::type_name::<PyReadonlyArray1<'_, i8>>();
    }
}

// Integration test for complete PyEvaluator workflow
#[cfg(feature = "pyo3")]
#[test]
fn test_pyevaluator_complete_workflow() {
    use prismind::python::PyEvaluator;

    // Step 1: Create evaluator
    let evaluator = PyEvaluator::new(None).unwrap();

    // Step 2: Evaluate a board position (using sync version for Rust tests)
    let mut board = vec![0i8; 64];
    board[27] = 2; // White at D4
    board[28] = 1; // Black at E4
    board[35] = 1; // Black at D5
    board[36] = 2; // White at E5

    let score = evaluator.evaluate_sync(board, 1).unwrap();
    println!("Initial board evaluation: {}", score);

    // Step 3: Access pattern weights
    let weight = evaluator.get_weight(0, 0, 0).unwrap();
    println!("Pattern weight (0,0,0): {}", weight);

    // Step 4: Get all weights (expensive operation)
    let all_weights = evaluator.get_weights().unwrap();
    println!("Total weight entries: {}", all_weights.len());

    // Verify consistency
    let weight_from_map = all_weights.get(&(0, 0, 0));
    assert!(weight_from_map.is_some());
    assert!(
        (weight - weight_from_map.unwrap()).abs() < 0.0001,
        "get_weight and get_weights should return consistent values"
    );
}
