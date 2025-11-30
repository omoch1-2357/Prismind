//! PyO3 wrapper for EnhancedCheckpointManager
//!
//! This module provides Python bindings for checkpoint save/load operations
//! with CRC32 integrity verification, optional compression, and retention policies.
//!
//! # Features
//!
//! - Atomic checkpoint saves using write-to-temp-then-rename
//! - CRC32 checksum verification for data integrity
//! - Optional gzip compression to reduce storage
//! - Configurable retention policy (keep last N checkpoints)
//!
//! # Requirements Coverage
//!
//! - Req 3.1: Atomic save with write-to-temp-then-rename
//! - Req 3.5: Configurable checkpoint retention
//! - Req 3.7: CRC32 checksum for data integrity
//! - Req 3.9: Optional compression via flate2
//! - Req 3.10: Log checkpoint operations with stats

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::evaluator::EvaluationTable;
use crate::learning::adam::AdamOptimizer;
use crate::learning::checkpoint::EnhancedCheckpointManager;
use crate::pattern::Pattern;

/// Python wrapper for the enhanced checkpoint manager.
///
/// Provides atomic checkpoint save/load with CRC32 verification,
/// optional compression, and configurable retention policy.
///
/// # Example
///
/// ```python
/// from prismind import PyCheckpointManager
///
/// manager = PyCheckpointManager("checkpoints/", 5, True)
///
/// # Save checkpoint (need PyLearningState)
/// # path, size, duration = manager.save(learning_state)
///
/// # Load latest checkpoint
/// state = manager.load_latest()
///
/// # List available checkpoints
/// checkpoints = manager.list_checkpoints()
/// for path, games, timestamp, size in checkpoints:
///     print(f"{path}: {games} games, {size} bytes")
/// ```
#[pyclass]
pub struct PyCheckpointManager {
    /// The wrapped EnhancedCheckpointManager
    inner: Arc<EnhancedCheckpointManager>,
    /// Checkpoint directory path
    checkpoint_dir: PathBuf,
    /// Retention count
    retention_count: usize,
    /// Compression enabled flag
    compression_enabled: bool,
}

/// Python wrapper for learning state (evaluation table + Adam optimizer + metadata).
///
/// This struct holds the complete training state that can be saved/loaded.
/// Uses Arc<RwLock<>> for thread-safe shared access without requiring Clone.
#[pyclass]
pub struct PyLearningState {
    /// Number of games completed
    #[pyo3(get, set)]
    pub games_completed: u64,
    /// Elapsed training time in seconds
    #[pyo3(get, set)]
    pub elapsed_time_secs: u64,
    /// Adam optimizer timestep
    #[pyo3(get, set)]
    pub adam_timestep: u64,
    /// Unix timestamp when checkpoint was created
    #[pyo3(get, set)]
    pub created_at: u64,
    /// Internal evaluation table (thread-safe access)
    eval_table: Arc<RwLock<EvaluationTable>>,
    /// Internal Adam optimizer state (thread-safe access)
    adam: Arc<RwLock<AdamOptimizer>>,
}

#[pymethods]
impl PyLearningState {
    /// Create a new learning state with default initialization.
    #[new]
    pub fn new() -> PyResult<Self> {
        let patterns = create_default_patterns();
        let eval_table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        Ok(Self {
            games_completed: 0,
            elapsed_time_secs: 0,
            adam_timestep: 0,
            created_at: 0,
            eval_table: Arc::new(RwLock::new(eval_table)),
            adam: Arc::new(RwLock::new(adam)),
        })
    }

    /// Get a summary of the learning state.
    pub fn summary(&self) -> String {
        format!(
            "PyLearningState(games={}, elapsed={}s, adam_t={}, created_at={})",
            self.games_completed, self.elapsed_time_secs, self.adam_timestep, self.created_at
        )
    }
}

impl PyLearningState {
    /// Create from internal components.
    pub fn from_components(
        eval_table: EvaluationTable,
        adam: AdamOptimizer,
        games_completed: u64,
        elapsed_time_secs: u64,
        adam_timestep: u64,
        created_at: u64,
    ) -> Self {
        Self {
            games_completed,
            elapsed_time_secs,
            adam_timestep,
            created_at,
            eval_table: Arc::new(RwLock::new(eval_table)),
            adam: Arc::new(RwLock::new(adam)),
        }
    }

    /// Get read lock on internal evaluation table.
    pub fn eval_table_read(&self) -> std::sync::RwLockReadGuard<'_, EvaluationTable> {
        self.eval_table.read().unwrap()
    }

    /// Get read lock on internal Adam optimizer.
    pub fn adam_read(&self) -> std::sync::RwLockReadGuard<'_, AdamOptimizer> {
        self.adam.read().unwrap()
    }
}

#[pymethods]
impl PyCheckpointManager {
    /// Create a new checkpoint manager.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory path for storing checkpoints
    /// * `retention_count` - Number of checkpoints to retain (default: 5)
    /// * `compression_enabled` - Whether to compress checkpoints (default: false)
    ///
    /// # Returns
    ///
    /// A new PyCheckpointManager instance
    ///
    /// # Raises
    ///
    /// * `IOError` - If checkpoint directory cannot be created
    #[new]
    #[pyo3(signature = (checkpoint_dir="checkpoints", retention_count=5, compression_enabled=false))]
    pub fn new(
        checkpoint_dir: &str,
        retention_count: usize,
        compression_enabled: bool,
    ) -> PyResult<Self> {
        let inner =
            EnhancedCheckpointManager::new(checkpoint_dir, retention_count, compression_enabled)
                .map_err(|e| {
                    PyIOError::new_err(format!("Failed to create checkpoint manager: {}", e))
                })?;

        Ok(Self {
            inner: Arc::new(inner),
            checkpoint_dir: PathBuf::from(checkpoint_dir),
            retention_count,
            compression_enabled,
        })
    }

    /// Save checkpoint with current training state.
    ///
    /// Uses atomic save (write-to-temp-then-rename) with CRC32 integrity.
    ///
    /// # Arguments
    ///
    /// * `state` - PyLearningState containing evaluation table and Adam optimizer
    ///
    /// # Returns
    ///
    /// Tuple of (checkpoint_path, file_size_bytes, save_duration_secs)
    ///
    /// # Raises
    ///
    /// * `IOError` - If save operation fails
    pub fn save(&self, state: &PyLearningState) -> PyResult<(String, u64, f64)> {
        let patterns = create_default_patterns();

        // Acquire read locks on the internal state
        let eval_table = state.eval_table_read();
        let adam = state.adam_read();

        let (path, size, duration) = self
            .inner
            .save(
                state.games_completed,
                &eval_table,
                &adam,
                &patterns,
                state.elapsed_time_secs,
            )
            .map_err(|e| PyIOError::new_err(format!("Failed to save checkpoint: {}", e)))?;

        Ok((path.to_string_lossy().to_string(), size, duration))
    }

    /// Load checkpoint from a specific path.
    ///
    /// Verifies CRC32 checksum and decompresses if necessary.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to checkpoint file
    ///
    /// # Returns
    ///
    /// PyLearningState containing restored training state
    ///
    /// # Raises
    ///
    /// * `IOError` - If load operation fails
    /// * `ValueError` - If checkpoint is corrupted or version mismatch
    pub fn load(&self, checkpoint_path: &str) -> PyResult<PyLearningState> {
        let patterns = create_default_patterns();
        let path = PathBuf::from(checkpoint_path);

        let (eval_table, adam, meta) = self
            .inner
            .load(&path, &patterns)
            .map_err(|e| PyValueError::new_err(format!("Failed to load checkpoint: {}", e)))?;

        Ok(PyLearningState::from_components(
            eval_table,
            adam,
            meta.game_count,
            meta.elapsed_time_secs,
            meta.adam_timestep,
            meta.created_at,
        ))
    }

    /// Load the latest checkpoint in the directory.
    ///
    /// # Returns
    ///
    /// Optional PyLearningState if a checkpoint exists, None otherwise
    ///
    /// # Raises
    ///
    /// * `IOError` - If directory read fails
    /// * `ValueError` - If latest checkpoint is corrupted
    pub fn load_latest(&self) -> PyResult<Option<PyLearningState>> {
        let patterns = create_default_patterns();

        match self.inner.load_latest(&patterns) {
            Ok(Some((eval_table, adam, meta))) => Ok(Some(PyLearningState::from_components(
                eval_table,
                adam,
                meta.game_count,
                meta.elapsed_time_secs,
                meta.adam_timestep,
                meta.created_at,
            ))),
            Ok(None) => Ok(None),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to load latest checkpoint: {}",
                e
            ))),
        }
    }

    /// List available checkpoints with metadata.
    ///
    /// # Returns
    ///
    /// List of tuples: (path, games_completed, timestamp_str, size_bytes)
    ///
    /// # Raises
    ///
    /// * `IOError` - If directory read fails
    pub fn list_checkpoints(&self) -> PyResult<Vec<(String, u64, String, u64)>> {
        self.inner
            .list_checkpoints()
            .map_err(|e| PyIOError::new_err(format!("Failed to list checkpoints: {}", e)))
    }

    /// Set the checkpoint retention count.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of checkpoints to retain
    pub fn set_retention(&mut self, count: usize) {
        // We need to recreate the inner manager with the new retention count
        // This is a limitation since EnhancedCheckpointManager doesn't expose mutable set_retention
        // For now, we store the new value and recreate on next save if needed
        self.retention_count = count;

        // Recreate the inner manager with new settings
        if let Ok(new_inner) =
            EnhancedCheckpointManager::new(&self.checkpoint_dir, count, self.compression_enabled)
        {
            self.inner = Arc::new(new_inner);
        }
    }

    /// Enable or disable compression.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable compression
    pub fn set_compression(&mut self, enabled: bool) {
        self.compression_enabled = enabled;

        // Recreate the inner manager with new settings
        if let Ok(new_inner) =
            EnhancedCheckpointManager::new(&self.checkpoint_dir, self.retention_count, enabled)
        {
            self.inner = Arc::new(new_inner);
        }
    }

    /// Verify checkpoint integrity without full load.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to checkpoint file
    ///
    /// # Returns
    ///
    /// True if checkpoint is valid, False otherwise
    ///
    /// # Raises
    ///
    /// * `IOError` - If file read fails
    pub fn verify(&self, checkpoint_path: &str) -> PyResult<bool> {
        let path = PathBuf::from(checkpoint_path);

        self.inner
            .verify(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to verify checkpoint: {}", e)))
    }

    /// Get the checkpoint directory path.
    pub fn checkpoint_dir(&self) -> String {
        self.checkpoint_dir.to_string_lossy().to_string()
    }

    /// Get the current retention count.
    pub fn get_retention(&self) -> usize {
        self.retention_count
    }

    /// Check if compression is enabled.
    pub fn is_compression_enabled(&self) -> bool {
        self.compression_enabled
    }
}

/// Create default patterns for checkpoint operations.
///
/// These patterns match the standard Othello pattern configuration
/// used throughout the system.
fn create_default_patterns() -> Vec<Pattern> {
    vec![
        Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
        Pattern::new(1, 10, vec![0, 8, 16, 24, 32, 40, 48, 56, 1, 9]).unwrap(),
        Pattern::new(2, 10, vec![0, 1, 8, 9, 10, 16, 17, 18, 24, 25]).unwrap(),
        Pattern::new(3, 10, vec![0, 9, 18, 27, 36, 45, 54, 63, 1, 10]).unwrap(),
        Pattern::new(4, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
        Pattern::new(5, 8, vec![0, 8, 16, 24, 32, 40, 48, 56]).unwrap(),
        Pattern::new(6, 8, vec![0, 9, 18, 27, 36, 45, 54, 63]).unwrap(),
        Pattern::new(7, 8, vec![7, 14, 21, 28, 35, 42, 49, 56]).unwrap(),
        Pattern::new(8, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
        Pattern::new(9, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
        Pattern::new(10, 5, vec![0, 1, 2, 3, 4]).unwrap(),
        Pattern::new(11, 5, vec![0, 8, 16, 24, 32]).unwrap(),
        Pattern::new(12, 4, vec![0, 1, 2, 3]).unwrap(),
        Pattern::new(13, 4, vec![0, 8, 16, 24]).unwrap(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // ========== Task 3.6: PyCheckpointManager Tests (TDD) ==========

    #[test]
    fn test_pycheckpointmanager_creation() {
        // Test creating PyCheckpointManager with default parameters
        let temp_dir = tempdir().unwrap();
        let manager = PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false);

        assert!(manager.is_ok(), "Should create manager successfully");
        let manager = manager.unwrap();
        assert_eq!(manager.get_retention(), 5);
        assert!(!manager.is_compression_enabled());
    }

    #[test]
    fn test_pycheckpointmanager_with_compression() {
        // Test creating PyCheckpointManager with compression enabled
        let temp_dir = tempdir().unwrap();
        let manager = PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 3, true).unwrap();

        assert!(manager.is_compression_enabled());
        assert_eq!(manager.get_retention(), 3);
    }

    #[test]
    fn test_pylearningstate_creation() {
        // Test creating PyLearningState
        let state = PyLearningState::new();
        assert!(state.is_ok());

        let state = state.unwrap();
        assert_eq!(state.games_completed, 0);
        assert_eq!(state.elapsed_time_secs, 0);
        assert_eq!(state.adam_timestep, 0);
    }

    #[test]
    fn test_pylearningstate_summary() {
        // Test learning state summary string
        let state = PyLearningState::new().unwrap();
        let summary = state.summary();

        assert!(summary.contains("PyLearningState"));
        assert!(summary.contains("games=0"));
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        // Test saving and loading a checkpoint
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        // Create a learning state
        let mut state = PyLearningState::new().unwrap();
        state.games_completed = 100000;
        state.elapsed_time_secs = 3600;

        // Save checkpoint
        let result = manager.save(&state);
        assert!(result.is_ok(), "Save should succeed: {:?}", result.err());

        let (path, size, duration) = result.unwrap();
        assert!(!path.is_empty(), "Path should not be empty");
        assert!(size > 0, "Size should be greater than 0");
        assert!(duration >= 0.0, "Duration should be non-negative");

        // Load checkpoint
        let loaded = manager.load(&path);
        assert!(loaded.is_ok(), "Load should succeed: {:?}", loaded.err());

        let loaded_state = loaded.unwrap();
        assert_eq!(loaded_state.games_completed, 100000);
        assert_eq!(loaded_state.elapsed_time_secs, 3600);
    }

    #[test]
    fn test_save_with_compression() {
        // Test saving with compression enabled
        let temp_dir = tempdir().unwrap();
        let manager = PyCheckpointManager::new(
            temp_dir.path().to_str().unwrap(),
            5,
            true, // Compression enabled
        )
        .unwrap();

        let mut state = PyLearningState::new().unwrap();
        state.games_completed = 50000;

        let result = manager.save(&state);
        assert!(result.is_ok(), "Compressed save should succeed");

        // Load and verify
        let (path, _, _) = result.unwrap();
        let loaded = manager.load(&path);
        assert!(loaded.is_ok(), "Compressed load should succeed");
        assert_eq!(loaded.unwrap().games_completed, 50000);
    }

    #[test]
    fn test_load_latest_empty_directory() {
        // Test load_latest when no checkpoints exist
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        let result = manager.load_latest();
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "Should return None for empty directory"
        );
    }

    #[test]
    fn test_load_latest_with_checkpoints() {
        // Test load_latest returns the most recent checkpoint
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        // Save multiple checkpoints
        let mut state = PyLearningState::new().unwrap();

        state.games_completed = 10000;
        manager.save(&state).unwrap();

        state.games_completed = 20000;
        manager.save(&state).unwrap();

        state.games_completed = 30000;
        manager.save(&state).unwrap();

        // Load latest should return the one with 30000 games
        let latest = manager.load_latest().unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().games_completed, 30000);
    }

    #[test]
    fn test_list_checkpoints() {
        // Test listing checkpoints
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        // Initially empty
        let list = manager.list_checkpoints().unwrap();
        assert!(list.is_empty());

        // Save a checkpoint
        let mut state = PyLearningState::new().unwrap();
        state.games_completed = 100000;
        manager.save(&state).unwrap();

        // Should now have one checkpoint
        let list = manager.list_checkpoints().unwrap();
        assert_eq!(list.len(), 1);

        let (path, games, _timestamp, size) = &list[0];
        assert!(path.contains("100000"));
        assert_eq!(*games, 100000);
        assert!(*size > 0);
    }

    #[test]
    fn test_set_retention() {
        // Test setting retention count
        let temp_dir = tempdir().unwrap();
        let mut manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        assert_eq!(manager.get_retention(), 5);

        manager.set_retention(3);
        assert_eq!(manager.get_retention(), 3);
    }

    #[test]
    fn test_set_compression() {
        // Test enabling/disabling compression
        let temp_dir = tempdir().unwrap();
        let mut manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        assert!(!manager.is_compression_enabled());

        manager.set_compression(true);
        assert!(manager.is_compression_enabled());

        manager.set_compression(false);
        assert!(!manager.is_compression_enabled());
    }

    #[test]
    fn test_verify_valid_checkpoint() {
        // Test verifying a valid checkpoint
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        let mut state = PyLearningState::new().unwrap();
        state.games_completed = 100000;

        let (path, _, _) = manager.save(&state).unwrap();

        let is_valid = manager.verify(&path);
        assert!(is_valid.is_ok());
        assert!(
            is_valid.unwrap(),
            "Valid checkpoint should verify successfully"
        );
    }

    #[test]
    fn test_verify_corrupted_checkpoint() {
        // Test verifying a corrupted checkpoint
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        // Create a corrupted file
        let corrupted_path = temp_dir.path().join("corrupted.bin");
        std::fs::write(&corrupted_path, b"corrupted data").unwrap();

        let result = manager.verify(corrupted_path.to_str().unwrap());
        // Should either return error or false for corrupted file
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_checkpoint_dir_accessor() {
        // Test getting checkpoint directory
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_str().unwrap();

        let manager = PyCheckpointManager::new(dir_path, 5, false).unwrap();

        let returned_dir = manager.checkpoint_dir();
        assert!(
            returned_dir.contains(
                temp_dir
                    .path()
                    .to_str()
                    .unwrap()
                    .split('/')
                    .next_back()
                    .unwrap_or("")
            )
        );
    }

    #[test]
    fn test_retention_policy_applied() {
        // Test that retention policy deletes old checkpoints
        let temp_dir = tempdir().unwrap();
        let manager = PyCheckpointManager::new(
            temp_dir.path().to_str().unwrap(),
            2, // Keep only 2 checkpoints
            false,
        )
        .unwrap();

        let mut state = PyLearningState::new().unwrap();

        // Save 4 checkpoints
        for games in [10000u64, 20000, 30000, 40000] {
            state.games_completed = games;
            manager.save(&state).unwrap();
        }

        // Should only have 2 checkpoints (30000 and 40000)
        let list = manager.list_checkpoints().unwrap();
        assert_eq!(
            list.len(),
            2,
            "Should only have 2 checkpoints after retention"
        );

        // Verify we have the most recent ones
        let games: Vec<u64> = list.iter().map(|(_, g, _, _)| *g).collect();
        assert!(games.contains(&30000));
        assert!(games.contains(&40000));
    }

    #[test]
    fn test_save_returns_correct_tuple_format() {
        // Test that save returns (path: str, size: int, duration: float)
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        let mut state = PyLearningState::new().unwrap();
        state.games_completed = 100000;

        let (path, size, duration) = manager.save(&state).unwrap();

        // Path should be a string containing the checkpoint filename
        assert!(path.ends_with(".bin"), "Path should end with .bin");
        assert!(path.contains("100000"), "Path should contain game count");

        // Size should be positive (checkpoint file size)
        assert!(size > 0, "Size should be positive");

        // Duration should be non-negative
        assert!(duration >= 0.0, "Duration should be non-negative");
    }

    #[test]
    fn test_load_nonexistent_checkpoint_fails() {
        // Test loading a non-existent checkpoint
        let temp_dir = tempdir().unwrap();
        let manager =
            PyCheckpointManager::new(temp_dir.path().to_str().unwrap(), 5, false).unwrap();

        let result = manager.load("/nonexistent/path/checkpoint.bin");
        assert!(
            result.is_err(),
            "Loading non-existent checkpoint should fail"
        );
    }
}
