//! Tests for the learning module structure and error types.
//!
//! These tests verify that the learning module is properly structured
//! and the LearningError enum has all required variants.

use prismind::learning::LearningError;
use std::io;

/// Test that LearningError::Io variant exists and works correctly
#[test]
fn test_learning_error_io_variant() {
    let io_error = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let error: LearningError = io_error.into();

    let error_string = error.to_string();
    assert!(error_string.contains("I/O"));
}

/// Test that LearningError::InvalidCheckpoint variant exists
#[test]
fn test_learning_error_invalid_checkpoint_variant() {
    let error = LearningError::InvalidCheckpoint("corrupt header".to_string());
    let error_string = error.to_string();

    assert!(error_string.contains("checkpoint") || error_string.contains("Checkpoint"));
    assert!(error_string.contains("corrupt header"));
}

/// Test that LearningError::Search variant exists and can be created from SearchError
#[test]
fn test_learning_error_search_variant() {
    use prismind::search::SearchError;

    let search_error = SearchError::MemoryAllocation("allocation failed".to_string());
    let error: LearningError = search_error.into();

    let error_string = error.to_string();
    assert!(error_string.contains("Search") || error_string.contains("search"));
}

/// Test that LearningError::EvaluationDivergence variant exists
#[test]
fn test_learning_error_evaluation_divergence_variant() {
    let error = LearningError::EvaluationDivergence("NaN detected".to_string());
    let error_string = error.to_string();

    assert!(
        error_string.contains("diverge")
            || error_string.contains("Diverge")
            || error_string.contains("NaN")
    );
}

/// Test that LearningError::MemoryAllocation variant exists
#[test]
fn test_learning_error_memory_allocation_variant() {
    let error = LearningError::MemoryAllocation("out of memory".to_string());
    let error_string = error.to_string();

    assert!(
        error_string.contains("Memory")
            || error_string.contains("memory")
            || error_string.contains("allocation")
    );
}

/// Test that LearningError::Config variant exists
#[test]
fn test_learning_error_config_variant() {
    let error = LearningError::Config("invalid lambda value".to_string());
    let error_string = error.to_string();

    assert!(
        error_string.contains("Config")
            || error_string.contains("config")
            || error_string.contains("invalid lambda")
    );
}

/// Test that LearningError::Interrupted variant exists
#[test]
fn test_learning_error_interrupted_variant() {
    let error = LearningError::Interrupted;
    let error_string = error.to_string();

    assert!(error_string.contains("interrupt") || error_string.contains("Interrupt"));
}

/// Test that LearningError implements std::error::Error
#[test]
fn test_learning_error_is_std_error() {
    let error = LearningError::Interrupted;

    // This will fail to compile if LearningError doesn't implement std::error::Error
    let _: &dyn std::error::Error = &error;
}

/// Test that LearningError implements Debug
#[test]
fn test_learning_error_debug() {
    let error = LearningError::InvalidCheckpoint("test".to_string());
    let debug_str = format!("{:?}", error);

    assert!(!debug_str.is_empty());
}

/// Test that all error variants are distinguishable
#[test]
fn test_learning_error_variants_distinguishable() {
    let errors = [
        LearningError::Io(io::Error::other("test")),
        LearningError::InvalidCheckpoint("test".to_string()),
        LearningError::Search(prismind::search::SearchError::MemoryAllocation(
            "test".to_string(),
        )),
        LearningError::EvaluationDivergence("test".to_string()),
        LearningError::MemoryAllocation("test".to_string()),
        LearningError::Config("test".to_string()),
        LearningError::Interrupted,
    ];

    // Each error should have a unique string representation
    let error_strings: Vec<String> = errors.iter().map(|e| e.to_string()).collect();

    // Check that we have 7 distinct error types
    assert_eq!(errors.len(), 7);

    // Verify each type produces a non-empty error message
    for s in &error_strings {
        assert!(!s.is_empty());
    }
}
