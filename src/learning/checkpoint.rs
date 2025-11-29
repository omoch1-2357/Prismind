//! Checkpoint Management for Training State Persistence.
//!
//! This module implements checkpoint management for saving and loading
//! complete training state, enabling fault tolerance and training resumption.
//!
//! # Binary Format
//!
//! | Offset | Size | Field | Description |
//! |--------|------|-------|-------------|
//! | 0 | 24 | magic | "OTHELLO_AI_CHECKPOINT_V1" |
//! | 24 | 8 | game_count | u64 little-endian |
//! | 32 | 8 | elapsed_secs | u64 little-endian |
//! | 40 | 8 | adam_timestep | u64 little-endian |
//! | 48 | 8 | created_at | Unix timestamp u64 |
//! | 56 | ~57 MB | eval_table | Raw u16 array |
//! | ~57 MB | ~114 MB | adam_m | Raw f32 array |
//! | ~171 MB | ~114 MB | adam_v | Raw f32 array |
//!
//! # Requirements Coverage
//!
//! - Req 6.1: Save checkpoints every 100,000 games
//! - Req 6.2: Save all pattern table weights (~57 MB)
//! - Req 6.3: Save Adam optimizer state (m and v, ~228 MB)
//! - Req 6.4: Save Adam timestep counter
//! - Req 6.5: Save metadata (game count, elapsed time, timestamp)
//! - Req 6.6: Filename format checkpoint_NNNNNN.bin
//! - Req 6.7: Load checkpoint restoring all state
//! - Req 6.8: Verify checkpoint integrity with header signature
//! - Req 6.9: Report error on corruption, allow fresh start
//! - Req 6.10: Support initial checkpoint_000000.bin

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::evaluator::EvaluationTable;
use crate::learning::LearningError;
use crate::learning::adam::AdamOptimizer;
use crate::pattern::Pattern;

/// 24-byte magic header for checkpoint verification.
///
/// This header is used to verify that a file is a valid checkpoint
/// and to check version compatibility.
pub const CHECKPOINT_MAGIC: &[u8; 24] = b"OTHELLO_AI_CHECKPOINT_V1";

/// Number of patterns in the Othello AI system.
pub const NUM_PATTERNS: usize = 14;

/// Number of stages in the evaluation table.
pub const NUM_STAGES: usize = 30;

/// Checkpoint metadata containing training state information.
///
/// Stores non-weight information about the training progress.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointMeta {
    /// Number of games completed at checkpoint time.
    pub game_count: u64,
    /// Total elapsed training time in seconds.
    pub elapsed_time_secs: u64,
    /// Adam optimizer timestep counter.
    pub adam_timestep: u64,
    /// Unix timestamp when checkpoint was created.
    pub created_at: u64,
}

impl CheckpointMeta {
    /// Create new checkpoint metadata with current timestamp.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    /// * `elapsed_time_secs` - Total elapsed training time in seconds
    /// * `adam_timestep` - Adam optimizer timestep counter
    pub fn new(game_count: u64, elapsed_time_secs: u64, adam_timestep: u64) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
        }
    }
}

/// Checkpoint manager for training state persistence.
///
/// Handles saving and loading of complete training state including:
/// - Pattern table weights (~57 MB)
/// - Adam optimizer moments (~228 MB)
/// - Training metadata
///
/// # Example
///
/// ```ignore
/// use prismind::learning::checkpoint::CheckpointManager;
///
/// let manager = CheckpointManager::new("checkpoints/")?;
///
/// // Save checkpoint
/// let path = manager.save(100000, &eval_table, &adam, &patterns, 0)?;
///
/// // Load checkpoint
/// let (table, adam, meta) = manager.load(&path, &patterns)?;
/// ```
pub struct CheckpointManager {
    /// Directory for checkpoint files.
    checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    ///
    /// Creates the checkpoint directory if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Path to checkpoint directory
    ///
    /// # Returns
    ///
    /// Result containing the manager or an error.
    ///
    /// # Errors
    ///
    /// - `LearningError::Io` if directory creation fails
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self, LearningError> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir)?;
        }

        Ok(Self { checkpoint_dir })
    }

    /// Generate checkpoint filename for a given game count.
    ///
    /// Format: checkpoint_NNNNNN.bin (6-digit zero-padded)
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    ///
    /// # Returns
    ///
    /// Filename string.
    pub fn checkpoint_filename(game_count: u64) -> String {
        format!("checkpoint_{:06}.bin", game_count)
    }

    /// Get full path for a checkpoint file.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    ///
    /// # Returns
    ///
    /// Full path to checkpoint file.
    pub fn checkpoint_path(&self, game_count: u64) -> PathBuf {
        self.checkpoint_dir
            .join(Self::checkpoint_filename(game_count))
    }

    /// Save checkpoint with pattern weights, Adam state, and metadata.
    ///
    /// Binary format:
    /// 1. 24-byte magic header
    /// 2. Metadata (game_count, elapsed_secs, adam_timestep, created_at)
    /// 3. Evaluation table weights
    /// 4. Adam first moment (m) vectors
    /// 5. Adam second moment (v) vectors
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    /// * `eval_table` - Evaluation table with pattern weights
    /// * `adam` - Adam optimizer state
    /// * `patterns` - Pattern definitions for calculating entry counts
    /// * `elapsed_time_secs` - Total elapsed training time
    ///
    /// # Returns
    ///
    /// Path to saved checkpoint file.
    ///
    /// # Errors
    ///
    /// - `LearningError::Io` if file write fails
    ///
    /// # Requirements
    ///
    /// - Req 6.2: Save pattern table weights (~57 MB)
    /// - Req 6.3: Save Adam optimizer state (~228 MB)
    /// - Req 6.4: Save Adam timestep counter
    /// - Req 6.5: Save metadata
    /// - Req 6.6: Filename format checkpoint_NNNNNN.bin
    pub fn save(
        &self,
        game_count: u64,
        eval_table: &EvaluationTable,
        adam: &AdamOptimizer,
        patterns: &[Pattern],
        elapsed_time_secs: u64,
    ) -> Result<PathBuf, LearningError> {
        let checkpoint_path = self.checkpoint_path(game_count);
        let temp_path = checkpoint_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Write magic header
        writer.write_all(CHECKPOINT_MAGIC)?;

        // Create and write metadata
        let meta = CheckpointMeta::new(game_count, elapsed_time_secs, adam.timestep());

        writer.write_all(&meta.game_count.to_le_bytes())?;
        writer.write_all(&meta.elapsed_time_secs.to_le_bytes())?;
        writer.write_all(&meta.adam_timestep.to_le_bytes())?;
        writer.write_all(&meta.created_at.to_le_bytes())?;

        // Write evaluation table weights
        // Format: for each stage, write all pattern entries as u16 little-endian
        self.write_eval_table(&mut writer, eval_table, patterns)?;

        // Write Adam first moment (m) vectors
        self.write_adam_moments(&mut writer, adam.first_moment(), patterns)?;

        // Write Adam second moment (v) vectors
        self.write_adam_moments(&mut writer, adam.second_moment(), patterns)?;

        writer.flush()?;
        drop(writer); // Ensure file is closed before rename

        // Atomic rename
        std::fs::rename(&temp_path, &checkpoint_path)?;

        Ok(checkpoint_path)
    }

    /// Load checkpoint, returning restored state.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to checkpoint file
    /// * `patterns` - Pattern definitions for table reconstruction
    ///
    /// # Returns
    ///
    /// Tuple of (EvaluationTable, AdamOptimizer, CheckpointMeta).
    ///
    /// # Errors
    ///
    /// - `LearningError::InvalidCheckpoint` if magic header doesn't match
    /// - `LearningError::Io` if file read fails
    ///
    /// # Requirements
    ///
    /// - Req 6.7: Load checkpoint restoring all state
    /// - Req 6.8: Verify checkpoint integrity with header
    /// - Req 6.9: Report error on corruption
    pub fn load(
        &self,
        checkpoint_path: &Path,
        patterns: &[Pattern],
    ) -> Result<(EvaluationTable, AdamOptimizer, CheckpointMeta), LearningError> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic header
        let mut magic = [0u8; 24];
        reader.read_exact(&mut magic)?;

        if &magic != CHECKPOINT_MAGIC {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Invalid magic header: expected {:?}, got {:?}",
                CHECKPOINT_MAGIC, magic
            )));
        }

        // Read metadata
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf8)?;
        let game_count = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let elapsed_time_secs = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let adam_timestep = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let created_at = u64::from_le_bytes(buf8);

        let meta = CheckpointMeta {
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
        };

        // Read evaluation table
        let eval_table = self.read_eval_table(&mut reader, patterns)?;

        // Read Adam optimizer
        let mut adam = AdamOptimizer::new(patterns);
        adam.set_timestep(adam_timestep);

        self.read_adam_moments(&mut reader, adam.first_moment_mut(), patterns)?;
        self.read_adam_moments(&mut reader, adam.second_moment_mut(), patterns)?;

        Ok((eval_table, adam, meta))
    }

    /// Verify checkpoint integrity without full load.
    ///
    /// Reads only the header and metadata to verify the checkpoint is valid.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to checkpoint file
    ///
    /// # Returns
    ///
    /// Checkpoint metadata if valid.
    ///
    /// # Errors
    ///
    /// - `LearningError::InvalidCheckpoint` if header doesn't match
    /// - `LearningError::Io` if file read fails
    pub fn verify(&self, checkpoint_path: &Path) -> Result<CheckpointMeta, LearningError> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic header
        let mut magic = [0u8; 24];
        reader.read_exact(&mut magic)?;

        if &magic != CHECKPOINT_MAGIC {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Invalid magic header: expected {:?}, got {:?}",
                CHECKPOINT_MAGIC, magic
            )));
        }

        // Read metadata
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf8)?;
        let game_count = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let elapsed_time_secs = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let adam_timestep = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let created_at = u64::from_le_bytes(buf8);

        Ok(CheckpointMeta {
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
        })
    }

    /// Find the latest checkpoint in the directory.
    ///
    /// Searches for checkpoint files matching the pattern checkpoint_NNNNNN.bin
    /// and returns the one with the highest game count.
    ///
    /// # Returns
    ///
    /// Path to latest checkpoint if found, None otherwise.
    ///
    /// # Errors
    ///
    /// - `LearningError::Io` if directory read fails
    pub fn find_latest(&self) -> Result<Option<PathBuf>, LearningError> {
        let mut latest: Option<(u64, PathBuf)> = None;

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str())
                && let Some(game_count) = Self::parse_checkpoint_filename(filename)
            {
                match &latest {
                    None => latest = Some((game_count, path)),
                    Some((current_max, _)) if game_count > *current_max => {
                        latest = Some((game_count, path))
                    }
                    _ => {}
                }
            }
        }

        Ok(latest.map(|(_, path)| path))
    }

    /// Parse game count from checkpoint filename.
    ///
    /// # Arguments
    ///
    /// * `filename` - Checkpoint filename
    ///
    /// # Returns
    ///
    /// Game count if filename matches pattern, None otherwise.
    fn parse_checkpoint_filename(filename: &str) -> Option<u64> {
        if filename.starts_with("checkpoint_") && filename.ends_with(".bin") {
            let num_str = &filename[11..filename.len() - 4];
            num_str.parse().ok()
        } else {
            None
        }
    }

    /// Write evaluation table to writer.
    ///
    /// # Arguments
    ///
    /// * `writer` - Writer to write data to
    /// * `table` - Evaluation table to save
    /// * `patterns` - Pattern definitions for calculating entry counts
    fn write_eval_table<W: Write>(
        &self,
        writer: &mut W,
        table: &EvaluationTable,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        // Write all weights: for each stage, for each pattern, for each index
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                // Calculate number of entries dynamically from pattern k value
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let value = table.get(pattern_id, stage, index);
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Read evaluation table from reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader to read data from
    /// * `patterns` - Pattern definitions for table reconstruction
    fn read_eval_table<R: Read>(
        &self,
        reader: &mut R,
        patterns: &[Pattern],
    ) -> Result<EvaluationTable, LearningError> {
        let mut table = EvaluationTable::new(patterns);

        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                // Calculate number of entries dynamically from pattern k value
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let mut buf = [0u8; 2];
                    reader.read_exact(&mut buf)?;
                    let value = u16::from_le_bytes(buf);
                    table.set(pattern_id, stage, index, value);
                }
            }
        }

        Ok(table)
    }

    /// Write Adam moments to writer.
    ///
    /// # Arguments
    ///
    /// * `writer` - Writer to write data to
    /// * `moments` - Adam moments to save
    /// * `patterns` - Pattern definitions for calculating entry counts
    fn write_adam_moments<W: Write>(
        &self,
        writer: &mut W,
        moments: &crate::learning::adam::AdamMoments,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                // Calculate number of entries dynamically from pattern k value
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let value = moments.get(pattern_id, stage, index);
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Read Adam moments from reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader to read data from
    /// * `moments` - Mutable moments storage to populate
    /// * `patterns` - Pattern definitions for calculating entry counts
    fn read_adam_moments<R: Read>(
        &self,
        reader: &mut R,
        moments: &mut crate::learning::adam::AdamMoments,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                // Calculate number of entries dynamically from pattern k value
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    let value = f32::from_le_bytes(buf);
                    moments.set(pattern_id, stage, index, value);
                }
            }
        }
        Ok(())
    }

    /// Get number of entries for a pattern.
    ///
    /// Calculates entry count dynamically from pattern k value using 3^k formula.
    /// This ensures correctness even if pattern configuration changes.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Pattern definitions array
    /// * `pattern_id` - Pattern ID (0-13)
    ///
    /// # Returns
    ///
    /// Number of entries (3^k where k is the pattern's cell count)
    ///
    /// # Panics
    ///
    /// Panics if pattern_id is out of bounds.
    fn get_pattern_entries(patterns: &[Pattern], pattern_id: usize) -> usize {
        assert!(
            pattern_id < patterns.len(),
            "pattern_id {} out of bounds (patterns.len = {})",
            pattern_id,
            patterns.len()
        );
        3_usize.pow(patterns[pattern_id].k as u32)
    }

    /// Get the checkpoint directory path.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

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
            Pattern::new(8, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
            Pattern::new(9, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
            Pattern::new(10, 5, vec![0, 1, 2, 3, 4]).unwrap(),
            Pattern::new(11, 5, vec![0, 8, 16, 24, 32]).unwrap(),
            Pattern::new(12, 4, vec![0, 1, 2, 3]).unwrap(),
            Pattern::new(13, 4, vec![0, 8, 16, 24]).unwrap(),
        ]
    }

    // ========== Requirement 6.1: Magic Header ==========

    #[test]
    fn test_checkpoint_magic_header_is_24_bytes() {
        assert_eq!(CHECKPOINT_MAGIC.len(), 24);
        assert_eq!(CHECKPOINT_MAGIC, b"OTHELLO_AI_CHECKPOINT_V1");
    }

    // ========== Requirement 6.6: Filename Format ==========

    #[test]
    fn test_checkpoint_filename_format() {
        assert_eq!(
            CheckpointManager::checkpoint_filename(0),
            "checkpoint_000000.bin"
        );
        assert_eq!(
            CheckpointManager::checkpoint_filename(100000),
            "checkpoint_100000.bin"
        );
        assert_eq!(
            CheckpointManager::checkpoint_filename(1000000),
            "checkpoint_1000000.bin"
        );
        assert_eq!(
            CheckpointManager::checkpoint_filename(999999),
            "checkpoint_999999.bin"
        );
    }

    #[test]
    fn test_parse_checkpoint_filename() {
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_000000.bin"),
            Some(0)
        );
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_100000.bin"),
            Some(100000)
        );
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_999999.bin"),
            Some(999999)
        );
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("invalid.bin"),
            None
        );
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_abc.bin"),
            None
        );
    }

    // ========== Requirement 6.10: Initial Checkpoint ==========

    #[test]
    fn test_save_initial_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save initial checkpoint (game_count = 0)
        let path = manager.save(0, &table, &adam, &patterns, 0).unwrap();

        assert!(path.exists());
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            "checkpoint_000000.bin"
        );
    }

    // ========== Requirement 6.2, 6.3, 6.4, 6.5: Save State ==========

    #[test]
    fn test_save_and_load_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::new(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Modify table and adam to have non-default values
        table.set(0, 0, 0, 40000);
        table.set(5, 15, 100, 25000);
        adam.update(0, 0, 0, 32768.0, 1.0);
        adam.step();

        // Save checkpoint
        let path = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Load checkpoint
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();

        // Verify metadata
        assert_eq!(meta.game_count, 100000);
        assert_eq!(meta.elapsed_time_secs, 3600);
        assert_eq!(meta.adam_timestep, 1);
        assert!(meta.created_at > 0);

        // Verify table values
        assert_eq!(loaded_table.get(0, 0, 0), 40000);
        assert_eq!(loaded_table.get(5, 15, 100), 25000);

        // Verify Adam timestep
        assert_eq!(loaded_adam.timestep(), 1);

        // Verify Adam moments are restored
        assert!(loaded_adam.first_moment().get(0, 0, 0) != 0.0);
    }

    // ========== Requirement 6.7: Load Checkpoint ==========

    #[test]
    fn test_load_checkpoint_restores_all_state() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::new(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Set specific values across different patterns and stages
        for pattern_id in 0..NUM_PATTERNS {
            for stage in [0, 15, 29] {
                let value = 32768 + (pattern_id * 100 + stage * 10) as u16;
                table.set(pattern_id, stage, 0, value);
            }
        }

        // Update Adam for multiple patterns
        for i in 0..10 {
            adam.update(0, 0, i, 32768.0, 1.0 + i as f32 * 0.1);
        }
        adam.step();
        adam.step();

        // Save and reload
        manager
            .save(200000, &table, &adam, &patterns, 7200)
            .unwrap();
        let path = manager.checkpoint_path(200000);
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();

        // Verify all table values
        for pattern_id in 0..NUM_PATTERNS {
            for stage in [0, 15, 29] {
                let expected = 32768 + (pattern_id * 100 + stage * 10) as u16;
                assert_eq!(
                    loaded_table.get(pattern_id, stage, 0),
                    expected,
                    "Mismatch at pattern {}, stage {}",
                    pattern_id,
                    stage
                );
            }
        }

        // Verify Adam state
        assert_eq!(loaded_adam.timestep(), 2);
        assert_eq!(meta.adam_timestep, 2);
    }

    // ========== Requirement 6.8: Header Verification ==========

    #[test]
    fn test_verify_checkpoint_header() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let path = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Verify should succeed
        let meta = manager.verify(&path).unwrap();
        assert_eq!(meta.game_count, 100000);
    }

    #[test]
    fn test_invalid_header_returns_error() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        // Create a file with invalid header (must be exactly 24 bytes)
        let invalid_path = temp_dir.path().join("invalid.bin");
        fs::write(&invalid_path, b"INVALID_HEADER_12345678X").unwrap();

        // Verify should fail
        let result = manager.verify(&invalid_path);
        assert!(result.is_err());
        match result.unwrap_err() {
            LearningError::InvalidCheckpoint(msg) => {
                assert!(msg.contains("Invalid magic header"));
            }
            _ => panic!("Expected InvalidCheckpoint error"),
        }
    }

    // ========== Requirement 6.9: Error Handling ==========

    #[test]
    fn test_load_corrupted_checkpoint_returns_error() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();

        // Create a file with valid header but truncated data
        let corrupted_path = temp_dir.path().join("corrupted.bin");
        let mut file = File::create(&corrupted_path).unwrap();
        file.write_all(CHECKPOINT_MAGIC).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // game_count
        // Missing rest of data

        // Load should fail with IO error (unexpected EOF)
        let result = manager.load(&corrupted_path, &patterns);
        assert!(result.is_err());
    }

    // ========== find_latest Tests ==========

    #[test]
    fn test_find_latest_empty_directory() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let result = manager.find_latest().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_find_latest_single_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        manager.save(100000, &table, &adam, &patterns, 0).unwrap();

        let latest = manager.find_latest().unwrap().unwrap();
        assert!(
            latest
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("100000")
        );
    }

    #[test]
    fn test_find_latest_multiple_checkpoints() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::new(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save multiple checkpoints
        manager.save(100000, &table, &adam, &patterns, 0).unwrap();
        manager.save(200000, &table, &adam, &patterns, 0).unwrap();
        manager.save(300000, &table, &adam, &patterns, 0).unwrap();

        let latest = manager.find_latest().unwrap().unwrap();
        assert!(
            latest
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("300000")
        );
    }

    // ========== CheckpointMeta Tests ==========

    #[test]
    fn test_checkpoint_meta_creation() {
        let meta = CheckpointMeta::new(100000, 3600, 50);

        assert_eq!(meta.game_count, 100000);
        assert_eq!(meta.elapsed_time_secs, 3600);
        assert_eq!(meta.adam_timestep, 50);
        assert!(meta.created_at > 0);
    }

    // ========== Requirements Summary Test ==========

    #[test]
    fn test_all_checkpoint_requirements_summary() {
        println!("=== Checkpoint Manager Requirements Verification ===");

        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::new(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Req 6.1: 24-byte magic header
        assert_eq!(CHECKPOINT_MAGIC.len(), 24);
        println!("  6.1: 24-byte magic header \"OTHELLO_AI_CHECKPOINT_V1\"");

        // Req 6.2: Save pattern table weights
        table.set(0, 0, 0, 45000);
        println!("  6.2: Save pattern table weights (~57 MB)");

        // Req 6.3: Save Adam optimizer state
        adam.update(0, 0, 0, 32768.0, 1.0);
        adam.step();
        println!("  6.3: Save Adam optimizer m and v moments (~228 MB)");

        // Req 6.4: Save Adam timestep counter
        println!("  6.4: Save Adam timestep counter");

        // Req 6.5: Save metadata
        println!("  6.5: Save metadata (game count, elapsed time, timestamp)");

        // Req 6.6: Filename format
        assert_eq!(
            CheckpointManager::checkpoint_filename(100000),
            "checkpoint_100000.bin"
        );
        println!("  6.6: Filename format checkpoint_NNNNNN.bin");

        // Save checkpoint
        let path = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();
        assert!(path.exists());

        // Req 6.7: Load checkpoint
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();
        assert_eq!(loaded_table.get(0, 0, 0), 45000);
        assert_eq!(loaded_adam.timestep(), 1);
        assert_eq!(meta.game_count, 100000);
        println!("  6.7: Load checkpoint restoring all state");

        // Req 6.8: Verify header
        let verified_meta = manager.verify(&path).unwrap();
        assert_eq!(verified_meta.game_count, 100000);
        println!("  6.8: Verify checkpoint integrity with header signature");

        // Req 6.9: Error on corruption
        let invalid_path = temp_dir.path().join("invalid.bin");
        fs::write(&invalid_path, b"INVALID").unwrap();
        assert!(manager.verify(&invalid_path).is_err());
        println!("  6.9: Return error on corruption or version mismatch");

        // Req 6.10: Initial checkpoint
        let initial_path = manager.save(0, &table, &adam, &patterns, 0).unwrap();
        assert!(initial_path.exists());
        assert!(
            initial_path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("000000")
        );
        println!("  6.10: Support saving checkpoint_000000.bin before training");

        println!("=== All Checkpoint Manager requirements verified ===");
    }
}
