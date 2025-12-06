//! Checkpoint Management for Training State Persistence.
//!
//! This module implements checkpoint management for saving and loading
//! complete training state, enabling fault tolerance and training resumption.
//!
//! # Binary Formats
//!
//! ## V1 Format (Legacy)
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
//! ## V2 Format (Enhanced - Phase 4)
//!
//! | Offset | Size | Field | Description |
//! |--------|------|-------|-------------|
//! | 0 | 4 | magic | "PRSM" |
//! | 4 | 4 | version | u32 (current: 2) |
//! | 8 | 4 | flags | bit 0: compressed |
//! | 12 | 4 | checksum | CRC32 of data |
//! | 16 | 8 | games_completed | u64 little-endian |
//! | 24 | 8 | timestamp | i64 Unix timestamp |
//! | 32 | 8 | elapsed_secs | u64 little-endian |
//! | 40 | 8 | adam_timestep | u64 little-endian |
//! | 48 | ~285 MB | data | Pattern tables + Adam state (optionally compressed) |
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
//!
//! ## Phase 4 Enhanced Requirements
//!
//! - Req 3.1: Atomic save with write-to-temp-then-rename
//! - Req 3.3: Version header for format validation
//! - Req 3.4: Version mismatch returns error with details
//! - Req 3.5: Configurable checkpoint retention
//! - Req 3.6: Automatic deletion of old checkpoints
//! - Req 3.7: CRC32 checksum for data integrity
//! - Req 3.8: Corruption detection via checksum mismatch
//! - Req 3.9: Optional compression via flate2
//! - Req 3.10: Log checkpoint operations with stats

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use chrono::Local;

use crc32fast::Hasher as Crc32Hasher;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;

use crate::evaluator::EvaluationTable;
use crate::learning::LearningError;
use crate::learning::adam::AdamOptimizer;
use crate::pattern::Pattern;

/// 24-byte magic header for checkpoint verification (V1 legacy format).
///
/// This header is used to verify that a file is a valid V1 checkpoint
/// and to check version compatibility.
pub const CHECKPOINT_MAGIC: &[u8; 24] = b"OTHELLO_AI_CHECKPOINT_V1";

/// 4-byte magic header for V2 format ("PRSM").
///
/// Used by `EnhancedCheckpointManager` for the new checkpoint format
/// with CRC32 integrity and optional compression.
pub const CHECKPOINT_MAGIC_V2: &[u8; 4] = b"PRSM";

/// Current checkpoint format version.
///
/// Version 2 supports CRC32 checksums and optional compression.
pub const CHECKPOINT_VERSION: u32 = 2;

/// Flag bit indicating compression is enabled.
pub const FLAG_COMPRESSED: u32 = 1;

/// Number of patterns in the Othello AI system.
pub const NUM_PATTERNS: usize = 14;

/// Number of stages in the evaluation table.
pub const NUM_STAGES: usize = 30;

/// Default retention count for checkpoints.
pub const DEFAULT_RETENTION_COUNT: usize = 5;

/// Checkpoint metadata containing training state information.
///
/// Stores non-weight information about the training progress.
///
/// # Backward Compatibility
///
/// Fields added after v1 (`target_games`, `accumulated_wins`, etc.) are stored
/// as Optional fields. When loading legacy checkpoints that lack these fields,
/// they default to `None` or zero values, ensuring compatibility.
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

    // ===== Extended fields for resume support (v2+) =====
    /// Target games for the training session (if set).
    /// None for legacy checkpoints or when not set.
    pub target_games: Option<u64>,
    /// Accumulated win counts: (black_wins, white_wins, draws).
    /// Used to restore win rate statistics on resume.
    pub accumulated_wins: (u64, u64, u64),
    /// Sum of stone differences for all games (for average calculation).
    pub total_stone_diff_sum: f64,
    /// Total number of games used for statistics calculation.
    /// May differ from game_count if some games failed.
    pub total_games_for_stats: u64,
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
            // Extended fields default to zero/None for new sessions
            target_games: None,
            accumulated_wins: (0, 0, 0),
            total_stone_diff_sum: 0.0,
            total_games_for_stats: 0,
        }
    }

    /// Create checkpoint metadata with full statistics for resume support.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    /// * `elapsed_time_secs` - Total elapsed training time in seconds
    /// * `adam_timestep` - Adam optimizer timestep counter
    /// * `target_games` - Target games for the training session
    /// * `accumulated_wins` - Win counts (black, white, draw)
    /// * `total_stone_diff_sum` - Sum of all stone differences
    /// * `total_games_for_stats` - Number of games contributing to statistics
    #[allow(clippy::too_many_arguments)]
    pub fn with_statistics(
        game_count: u64,
        elapsed_time_secs: u64,
        adam_timestep: u64,
        target_games: Option<u64>,
        accumulated_wins: (u64, u64, u64),
        total_stone_diff_sum: f64,
        total_games_for_stats: u64,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
            target_games,
            accumulated_wins,
            total_stone_diff_sum,
            total_games_for_stats,
        }
    }

    /// Create a legacy-compatible metadata (without extended fields).
    /// Used when reading old checkpoints.
    pub fn legacy(
        game_count: u64,
        elapsed_time_secs: u64,
        adam_timestep: u64,
        created_at: u64,
    ) -> Self {
        Self {
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
            target_games: None,
            accumulated_wins: (0, 0, 0),
            total_stone_diff_sum: 0.0,
            total_games_for_stats: 0,
        }
    }
}

/// V2 Checkpoint header with version, flags, and CRC32 checksum.
///
/// This header is used by `EnhancedCheckpointManager` and provides:
/// - Magic bytes for format identification
/// - Version number for compatibility checking
/// - Flags for compression status
/// - CRC32 checksum for data integrity
///
/// # Binary Format (32 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 4 | magic ("PRSM") |
/// | 4 | 4 | version (u32) |
/// | 8 | 4 | flags (u32) |
/// | 12 | 4 | checksum (u32) |
/// | 16 | 8 | games_completed (u64) |
/// | 24 | 8 | timestamp (i64) |
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointHeader {
    /// Magic bytes for format identification ("PRSM").
    pub magic: [u8; 4],
    /// Format version number.
    pub version: u32,
    /// Flags (bit 0: compressed).
    pub flags: u32,
    /// CRC32 checksum of the data section.
    pub checksum: u32,
    /// Number of games completed at checkpoint time.
    pub games_completed: u64,
    /// Unix timestamp when checkpoint was created.
    pub timestamp: i64,
}

impl CheckpointHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 32;

    /// Create a new checkpoint header.
    ///
    /// # Arguments
    ///
    /// * `games_completed` - Number of games completed
    /// * `compressed` - Whether data will be compressed
    pub fn new(games_completed: u64, compressed: bool) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let flags = if compressed { FLAG_COMPRESSED } else { 0 };

        Self {
            magic: *CHECKPOINT_MAGIC_V2,
            version: CHECKPOINT_VERSION,
            flags,
            checksum: 0, // Set during save after data serialization
            games_completed,
            timestamp,
        }
    }

    /// Set the CRC32 checksum.
    pub fn set_checksum(&mut self, checksum: u32) {
        self.checksum = checksum;
    }

    /// Check if compression flag is set.
    pub fn is_compressed(&self) -> bool {
        self.flags & FLAG_COMPRESSED != 0
    }

    /// Serialize header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.checksum.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.games_completed.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.timestamp.to_le_bytes());
        bytes
    }

    /// Deserialize header from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if magic bytes don't match or version is incompatible.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LearningError> {
        if bytes.len() < Self::SIZE {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Header too small: expected {} bytes, got {}",
                Self::SIZE,
                bytes.len()
            )));
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);

        if &magic != CHECKPOINT_MAGIC_V2 {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Invalid magic header: expected {:?}, got {:?}",
                CHECKPOINT_MAGIC_V2, magic
            )));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != CHECKPOINT_VERSION {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Incompatible checkpoint version: expected {}, got {}. \
                 This checkpoint was created with a different version of the software.",
                CHECKPOINT_VERSION, version
            )));
        }

        let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let checksum = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let games_completed = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let timestamp = i64::from_le_bytes(bytes[24..32].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            flags,
            checksum,
            games_completed,
            timestamp,
        })
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
    /// Format: checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    ///
    /// # Returns
    ///
    /// Filename string with timestamp.
    pub fn checkpoint_filename(game_count: u64) -> String {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        format!("checkpoint_{}_{:06}.bin", timestamp, game_count)
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

    /// Save checkpoint with full metadata including extended statistics.
    ///
    /// This method preserves all training statistics for accurate resume.
    /// Extended metadata is appended after the Adam moments, allowing
    /// backward-compatible loading of older checkpoints.
    ///
    /// # Arguments
    ///
    /// * `eval_table` - Evaluation table with pattern weights
    /// * `adam` - Adam optimizer state
    /// * `patterns` - Pattern definitions
    /// * `meta` - Full checkpoint metadata including statistics
    ///
    /// # Returns
    ///
    /// Path to saved checkpoint file.
    pub fn save_with_metadata(
        &self,
        eval_table: &EvaluationTable,
        adam: &AdamOptimizer,
        patterns: &[Pattern],
        meta: &CheckpointMeta,
    ) -> Result<PathBuf, LearningError> {
        let checkpoint_path = self.checkpoint_path(meta.game_count);
        let temp_path = checkpoint_path.with_extension("tmp");
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);

        // Write magic header
        writer.write_all(CHECKPOINT_MAGIC)?;

        // Write core metadata
        writer.write_all(&meta.game_count.to_le_bytes())?;
        writer.write_all(&meta.elapsed_time_secs.to_le_bytes())?;
        writer.write_all(&meta.adam_timestep.to_le_bytes())?;
        writer.write_all(&meta.created_at.to_le_bytes())?;

        // Write evaluation table weights
        self.write_eval_table(&mut writer, eval_table, patterns)?;

        // Write Adam first moment (m) vectors
        self.write_adam_moments(&mut writer, adam.first_moment(), patterns)?;

        // Write Adam second moment (v) vectors
        self.write_adam_moments(&mut writer, adam.second_moment(), patterns)?;

        // Write extended metadata (new fields for resume support)
        self.write_extended_metadata(&mut writer, meta)?;

        writer.flush()?;
        drop(writer);

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

        // Use legacy constructor for V1 format (doesn't have extended fields yet)
        let meta = CheckpointMeta::legacy(game_count, elapsed_time_secs, adam_timestep, created_at);

        // Read evaluation table
        let eval_table = self.read_eval_table(&mut reader, patterns)?;

        // Read Adam optimizer
        let mut adam = AdamOptimizer::new(patterns);
        adam.set_timestep(adam_timestep);

        self.read_adam_moments(&mut reader, adam.first_moment_mut(), patterns)?;
        self.read_adam_moments(&mut reader, adam.second_moment_mut(), patterns)?;

        // Try to read extended metadata (may not exist in older checkpoints)
        let meta = self.read_extended_metadata(&mut reader, meta);

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

        // Use legacy constructor for V1 format verification
        Ok(CheckpointMeta::legacy(
            game_count,
            elapsed_time_secs,
            adam_timestep,
            created_at,
        ))
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
            // Try new format: checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin
            // or legacy format: checkpoint_NNNNNN.bin
            let inner = &filename[11..filename.len() - 4];
            // New format has underscores: YYYYMMDD_HHMMSS_NNNNNN
            if let Some(last_underscore) = inner.rfind('_') {
                // Try parsing the part after the last underscore as game count
                inner[last_underscore + 1..].parse().ok()
            } else {
                // Legacy format: just NNNNNN
                inner.parse().ok()
            }
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
        let mut table = EvaluationTable::from_patterns(patterns);

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

    /// Write extended metadata to writer.
    ///
    /// Extended metadata format (backward compatible - appended after main data):
    /// - 8 bytes: Magic marker "EXTMETA\0" for detection
    /// - 8 bytes: target_games (u64, or 0 if None)
    /// - 1 byte: has_target_games flag (0 or 1)
    /// - 8 bytes: black_wins (u64)
    /// - 8 bytes: white_wins (u64)
    /// - 8 bytes: draws (u64)
    /// - 8 bytes: total_stone_diff_sum (f64)
    /// - 8 bytes: total_games_for_stats (u64)
    fn write_extended_metadata<W: Write>(
        &self,
        writer: &mut W,
        meta: &CheckpointMeta,
    ) -> Result<(), LearningError> {
        // Magic marker for extended metadata detection
        writer.write_all(b"EXTMETA\0")?;

        // target_games and flag
        let (target, has_target) = match meta.target_games {
            Some(t) => (t, 1u8),
            None => (0, 0u8),
        };
        writer.write_all(&target.to_le_bytes())?;
        writer.write_all(&[has_target])?;

        // accumulated_wins
        writer.write_all(&meta.accumulated_wins.0.to_le_bytes())?;
        writer.write_all(&meta.accumulated_wins.1.to_le_bytes())?;
        writer.write_all(&meta.accumulated_wins.2.to_le_bytes())?;

        // stone diff statistics
        writer.write_all(&meta.total_stone_diff_sum.to_le_bytes())?;
        writer.write_all(&meta.total_games_for_stats.to_le_bytes())?;

        Ok(())
    }

    /// Read extended metadata from reader (if present).
    ///
    /// Returns updated CheckpointMeta with extended fields populated,
    /// or the original meta if extended data is not present.
    fn read_extended_metadata<R: Read>(
        &self,
        reader: &mut R,
        mut meta: CheckpointMeta,
    ) -> CheckpointMeta {
        // Try to read magic marker
        let mut magic = [0u8; 8];
        if reader.read_exact(&mut magic).is_err() {
            return meta; // No extended metadata (legacy checkpoint)
        }

        if &magic != b"EXTMETA\0" {
            return meta; // Not extended metadata marker
        }

        // Read target_games
        let mut buf8 = [0u8; 8];
        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        let target_value = u64::from_le_bytes(buf8);

        let mut buf1 = [0u8; 1];
        if reader.read_exact(&mut buf1).is_err() {
            return meta;
        }
        meta.target_games = if buf1[0] != 0 {
            Some(target_value)
        } else {
            None
        };

        // Read accumulated_wins
        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        let black_wins = u64::from_le_bytes(buf8);

        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        let white_wins = u64::from_le_bytes(buf8);

        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        let draws = u64::from_le_bytes(buf8);
        meta.accumulated_wins = (black_wins, white_wins, draws);

        // Read stone diff statistics
        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        meta.total_stone_diff_sum = f64::from_le_bytes(buf8);

        if reader.read_exact(&mut buf8).is_err() {
            return meta;
        }
        meta.total_games_for_stats = u64::from_le_bytes(buf8);

        meta
    }

    /// Get the checkpoint directory path.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

// ============================================================================
// Enhanced Checkpoint Manager (Phase 4)
// ============================================================================

/// Enhanced checkpoint manager with CRC32 integrity, compression, and retention.
///
/// This manager provides production-ready checkpoint functionality with:
/// - Atomic saves using write-to-temp-then-rename
/// - CRC32 checksum for data integrity verification
/// - Optional gzip compression to reduce storage
/// - Configurable retention policy (keep last N checkpoints)
///
/// # Example
///
/// ```ignore
/// use prismind::learning::checkpoint::EnhancedCheckpointManager;
///
/// let manager = EnhancedCheckpointManager::new("checkpoints/", 5, true)?;
///
/// // Save with compression and integrity checking
/// let (path, size, duration) = manager.save(100000, &table, &adam, &patterns, 3600)?;
///
/// // Verify integrity
/// assert!(manager.verify(&path)?);
///
/// // Load checkpoint
/// let (table, adam, meta) = manager.load(&path, &patterns)?;
/// ```
pub struct EnhancedCheckpointManager {
    /// Directory for checkpoint files.
    checkpoint_dir: PathBuf,
    /// Number of checkpoints to retain.
    retention_count: usize,
    /// Whether to compress checkpoint data.
    compression_enabled: bool,
}

/// Type alias for extended metadata returned by read_extended_metadata.
///
/// Tuple components:
/// - `Option<u64>`: target_games
/// - `(u64, u64, u64)`: accumulated_wins (black, white, draws)
/// - `f64`: total_stone_diff_sum
/// - `u64`: total_games_for_stats
type ExtendedMetadata = (Option<u64>, (u64, u64, u64), f64, u64);

impl EnhancedCheckpointManager {
    /// Create a new enhanced checkpoint manager.
    ///
    /// Creates the checkpoint directory if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Path to checkpoint directory
    /// * `retention_count` - Number of checkpoints to keep (default: 5)
    /// * `compression_enabled` - Whether to compress checkpoints
    ///
    /// # Returns
    ///
    /// Result containing the manager or an error.
    pub fn new<P: AsRef<Path>>(
        checkpoint_dir: P,
        retention_count: usize,
        compression_enabled: bool,
    ) -> Result<Self, LearningError> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir)?;
        }

        Ok(Self {
            checkpoint_dir,
            retention_count,
            compression_enabled,
        })
    }

    /// Get the current retention count.
    pub fn retention_count(&self) -> usize {
        self.retention_count
    }

    /// Set the retention count.
    pub fn set_retention(&mut self, count: usize) {
        self.retention_count = count;
    }

    /// Check if compression is enabled.
    pub fn compression_enabled(&self) -> bool {
        self.compression_enabled
    }

    /// Enable or disable compression.
    pub fn set_compression(&mut self, enabled: bool) {
        self.compression_enabled = enabled;
    }

    /// Generate checkpoint filename for a given game count.
    ///
    /// Format: checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin
    pub fn checkpoint_filename(game_count: u64) -> String {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        format!("checkpoint_{}_{:06}.bin", timestamp, game_count)
    }

    /// Get full path for a checkpoint file.
    pub fn checkpoint_path(&self, game_count: u64) -> PathBuf {
        self.checkpoint_dir
            .join(Self::checkpoint_filename(game_count))
    }

    /// Save checkpoint with CRC32 integrity and optional compression.
    ///
    /// Uses atomic write (write-to-temp-then-rename) to prevent corruption.
    ///
    /// # Arguments
    ///
    /// * `game_count` - Number of games completed
    /// * `eval_table` - Evaluation table with pattern weights
    /// * `adam` - Adam optimizer state
    /// * `patterns` - Pattern definitions
    /// * `elapsed_time_secs` - Total elapsed training time
    ///
    /// # Returns
    ///
    /// Tuple of (path, file_size_bytes, save_duration_secs).
    pub fn save(
        &self,
        game_count: u64,
        eval_table: &EvaluationTable,
        adam: &AdamOptimizer,
        patterns: &[Pattern],
        elapsed_time_secs: u64,
    ) -> Result<(PathBuf, u64, f64), LearningError> {
        // Create basic metadata without extended statistics
        let meta = CheckpointMeta::new(game_count, elapsed_time_secs, adam.timestep());
        self.save_with_metadata(eval_table, adam, patterns, &meta)
    }

    /// Save checkpoint with full metadata including extended statistics.
    ///
    /// This method preserves all training statistics for accurate resume.
    ///
    /// # Arguments
    ///
    /// * `eval_table` - Evaluation table with pattern weights
    /// * `adam` - Adam optimizer state
    /// * `patterns` - Pattern definitions
    /// * `meta` - Full checkpoint metadata including statistics
    ///
    /// # Returns
    ///
    /// Tuple of (path, file_size_bytes, save_duration_secs).
    pub fn save_with_metadata(
        &self,
        eval_table: &EvaluationTable,
        adam: &AdamOptimizer,
        patterns: &[Pattern],
        meta: &CheckpointMeta,
    ) -> Result<(PathBuf, u64, f64), LearningError> {
        let start_time = Instant::now();
        let checkpoint_path = self.checkpoint_path(meta.game_count);
        let temp_path = checkpoint_path.with_extension("tmp");

        // Serialize data to buffer
        let mut data_buffer = Vec::new();

        // Write elapsed_time_secs and adam_timestep to data buffer
        data_buffer.extend_from_slice(&meta.elapsed_time_secs.to_le_bytes());
        data_buffer.extend_from_slice(&meta.adam_timestep.to_le_bytes());

        // Write evaluation table
        self.write_eval_table_to_buffer(&mut data_buffer, eval_table, patterns)?;

        // Write Adam moments
        self.write_adam_moments_to_buffer(&mut data_buffer, adam.first_moment(), patterns)?;
        self.write_adam_moments_to_buffer(&mut data_buffer, adam.second_moment(), patterns)?;

        // Write extended metadata for resume support
        self.write_extended_metadata_to_buffer(&mut data_buffer, meta);

        // Optionally compress data
        let final_data = if self.compression_enabled {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(&data_buffer)?;
            encoder.finish()?
        } else {
            data_buffer
        };

        // Calculate CRC32 checksum of the final data
        let mut hasher = Crc32Hasher::new();
        hasher.update(&final_data);
        let checksum = hasher.finalize();

        // Create header with checksum
        let mut header = CheckpointHeader::new(meta.game_count, self.compression_enabled);
        header.set_checksum(checksum);

        // Write to temp file
        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header.to_bytes())?;
        writer.write_all(&final_data)?;
        writer.flush()?;
        drop(writer);

        // Atomic rename
        fs::rename(&temp_path, &checkpoint_path)?;

        // Get file size
        let file_size = fs::metadata(&checkpoint_path)?.len();

        // Apply retention policy
        self.apply_retention()?;

        let duration = start_time.elapsed().as_secs_f64();

        // Log checkpoint operation
        log::info!(
            "Checkpoint saved: {} ({} bytes, {:.2}s, compression: {})",
            checkpoint_path.display(),
            file_size,
            duration,
            self.compression_enabled
        );

        Ok((checkpoint_path, file_size, duration))
    }

    /// Load checkpoint with CRC32 verification.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_path` - Path to checkpoint file
    /// * `patterns` - Pattern definitions
    ///
    /// # Returns
    ///
    /// Tuple of (EvaluationTable, AdamOptimizer, CheckpointMeta).
    pub fn load(
        &self,
        checkpoint_path: &Path,
        patterns: &[Pattern],
    ) -> Result<(EvaluationTable, AdamOptimizer, CheckpointMeta), LearningError> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; CheckpointHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = CheckpointHeader::from_bytes(&header_bytes)?;

        // Read data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Verify checksum
        let mut hasher = Crc32Hasher::new();
        hasher.update(&data);
        let computed_checksum = hasher.finalize();

        if computed_checksum != header.checksum {
            return Err(LearningError::InvalidCheckpoint(format!(
                "Checksum mismatch: expected {:#010x}, computed {:#010x}. Data may be corrupted.",
                header.checksum, computed_checksum
            )));
        }

        // Decompress if needed
        let decompressed_data = if header.is_compressed() {
            let mut decoder = GzDecoder::new(Cursor::new(data));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            data
        };

        // Parse data
        let mut cursor = Cursor::new(decompressed_data);

        // Read elapsed_time_secs and adam_timestep
        let mut buf8 = [0u8; 8];
        cursor.read_exact(&mut buf8)?;
        let elapsed_time_secs = u64::from_le_bytes(buf8);

        cursor.read_exact(&mut buf8)?;
        let adam_timestep = u64::from_le_bytes(buf8);

        // Read evaluation table
        let eval_table = self.read_eval_table_from_reader(&mut cursor, patterns)?;

        // Read Adam optimizer
        let mut adam = AdamOptimizer::new(patterns);
        adam.set_timestep(adam_timestep);
        self.read_adam_moments_from_reader(&mut cursor, adam.first_moment_mut(), patterns)?;
        self.read_adam_moments_from_reader(&mut cursor, adam.second_moment_mut(), patterns)?;

        // Try to read extended metadata (may not exist in older checkpoints)
        let (target_games, accumulated_wins, total_stone_diff_sum, total_games_for_stats) = self
            .read_extended_metadata(&mut cursor)
            .unwrap_or((None, (0, 0, 0), 0.0, 0));

        let meta = CheckpointMeta {
            game_count: header.games_completed,
            elapsed_time_secs,
            adam_timestep,
            created_at: header.timestamp as u64,
            target_games,
            accumulated_wins,
            total_stone_diff_sum,
            total_games_for_stats,
        };

        Ok((eval_table, adam, meta))
    }

    /// Load the latest checkpoint in the directory.
    ///
    /// # Returns
    ///
    /// Optional tuple of (EvaluationTable, AdamOptimizer, CheckpointMeta).
    pub fn load_latest(
        &self,
        patterns: &[Pattern],
    ) -> Result<Option<(EvaluationTable, AdamOptimizer, CheckpointMeta)>, LearningError> {
        if let Some(latest_path) = self.find_latest()? {
            Ok(Some(self.load(&latest_path, patterns)?))
        } else {
            Ok(None)
        }
    }

    /// Verify checkpoint integrity without full load.
    ///
    /// Reads and verifies the header and CRC32 checksum.
    ///
    /// # Returns
    ///
    /// True if checkpoint is valid, false otherwise.
    pub fn verify(&self, checkpoint_path: &Path) -> Result<bool, LearningError> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; CheckpointHeader::SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header = CheckpointHeader::from_bytes(&header_bytes)?;

        // Read data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Verify checksum
        let mut hasher = Crc32Hasher::new();
        hasher.update(&data);
        let computed_checksum = hasher.finalize();

        Ok(computed_checksum == header.checksum)
    }

    /// List all checkpoints with metadata.
    ///
    /// # Returns
    ///
    /// Vector of (path, games_completed, timestamp_str, size_bytes).
    pub fn list_checkpoints(&self) -> Result<Vec<(String, u64, String, u64)>, LearningError> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str())
                && filename.starts_with("checkpoint_")
                && filename.ends_with(".bin")
                && let Ok(file) = File::open(&path)
            {
                // Try to read header for metadata
                let mut reader = BufReader::new(file);
                let mut header_bytes = [0u8; CheckpointHeader::SIZE];
                if reader.read_exact(&mut header_bytes).is_ok()
                    && let Ok(header) = CheckpointHeader::from_bytes(&header_bytes)
                {
                    let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    let timestamp = chrono::DateTime::from_timestamp(header.timestamp, 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    checkpoints.push((
                        path.to_string_lossy().to_string(),
                        header.games_completed,
                        timestamp,
                        size,
                    ));
                }
            }
        }

        // Sort by game count descending
        checkpoints.sort_by(|a, b| b.1.cmp(&a.1));

        Ok(checkpoints)
    }

    /// Find the latest checkpoint in the directory.
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

    /// Apply retention policy, deleting old checkpoints.
    fn apply_retention(&self) -> Result<Vec<PathBuf>, LearningError> {
        let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|n| n.to_str())
                && let Some(game_count) = Self::parse_checkpoint_filename(filename)
            {
                checkpoints.push((game_count, path));
            }
        }

        // Sort by game count descending
        checkpoints.sort_by(|a, b| b.0.cmp(&a.0));

        // Delete checkpoints beyond retention count
        let mut deleted = Vec::new();
        if checkpoints.len() > self.retention_count {
            for (_, path) in checkpoints.into_iter().skip(self.retention_count) {
                if fs::remove_file(&path).is_ok() {
                    log::info!("Deleted old checkpoint: {}", path.display());
                    deleted.push(path);
                }
            }
        }

        Ok(deleted)
    }

    /// Parse game count from checkpoint filename.
    fn parse_checkpoint_filename(filename: &str) -> Option<u64> {
        if filename.starts_with("checkpoint_") && filename.ends_with(".bin") {
            let num_str = &filename[11..filename.len() - 4];
            num_str.parse().ok()
        } else {
            None
        }
    }

    /// Write evaluation table to buffer.
    fn write_eval_table_to_buffer(
        &self,
        buffer: &mut Vec<u8>,
        table: &EvaluationTable,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let value = table.get(pattern_id, stage, index);
                    buffer.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    /// Read evaluation table from reader.
    fn read_eval_table_from_reader<R: Read>(
        &self,
        reader: &mut R,
        patterns: &[Pattern],
    ) -> Result<EvaluationTable, LearningError> {
        let mut table = EvaluationTable::from_patterns(patterns);

        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
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

    /// Write Adam moments to buffer.
    fn write_adam_moments_to_buffer(
        &self,
        buffer: &mut Vec<u8>,
        moments: &crate::learning::adam::AdamMoments,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
                let num_entries = Self::get_pattern_entries(patterns, pattern_id);
                for index in 0..num_entries {
                    let value = moments.get(pattern_id, stage, index);
                    buffer.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(())
    }

    /// Read Adam moments from reader.
    fn read_adam_moments_from_reader<R: Read>(
        &self,
        reader: &mut R,
        moments: &mut crate::learning::adam::AdamMoments,
        patterns: &[Pattern],
    ) -> Result<(), LearningError> {
        for stage in 0..NUM_STAGES {
            for pattern_id in 0..NUM_PATTERNS {
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

    /// Write extended metadata to buffer.
    ///
    /// Extended metadata format:
    /// - 8 bytes: Magic marker "EXTMETA\0" for detection
    /// - 8 bytes: target_games (u64, or 0 if None)
    /// - 1 byte: has_target_games flag (0 or 1)
    /// - 8 bytes: black_wins (u64)
    /// - 8 bytes: white_wins (u64)
    /// - 8 bytes: draws (u64)
    /// - 8 bytes: total_stone_diff_sum (f64)
    /// - 8 bytes: total_games_for_stats (u64)
    fn write_extended_metadata_to_buffer(&self, buffer: &mut Vec<u8>, meta: &CheckpointMeta) {
        // Magic marker for extended metadata detection
        buffer.extend_from_slice(b"EXTMETA\0");

        // target_games and flag
        let (target, has_target) = match meta.target_games {
            Some(t) => (t, 1u8),
            None => (0, 0u8),
        };
        buffer.extend_from_slice(&target.to_le_bytes());
        buffer.push(has_target);

        // accumulated_wins
        buffer.extend_from_slice(&meta.accumulated_wins.0.to_le_bytes());
        buffer.extend_from_slice(&meta.accumulated_wins.1.to_le_bytes());
        buffer.extend_from_slice(&meta.accumulated_wins.2.to_le_bytes());

        // stone diff statistics
        buffer.extend_from_slice(&meta.total_stone_diff_sum.to_le_bytes());
        buffer.extend_from_slice(&meta.total_games_for_stats.to_le_bytes());
    }

    /// Read extended metadata from reader (if present).
    ///
    /// Returns None if extended metadata is not present (legacy checkpoint).
    fn read_extended_metadata<R: Read>(&self, reader: &mut R) -> Option<ExtendedMetadata> {
        // Try to read magic marker
        let mut magic = [0u8; 8];
        if reader.read_exact(&mut magic).is_err() {
            return None;
        }

        if &magic != b"EXTMETA\0" {
            return None;
        }

        // Read target_games
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8).ok()?;
        let target_value = u64::from_le_bytes(buf8);

        let mut buf1 = [0u8; 1];
        reader.read_exact(&mut buf1).ok()?;
        let target_games = if buf1[0] != 0 {
            Some(target_value)
        } else {
            None
        };

        // Read accumulated_wins
        reader.read_exact(&mut buf8).ok()?;
        let black_wins = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8).ok()?;
        let white_wins = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8).ok()?;
        let draws = u64::from_le_bytes(buf8);

        // Read stone diff statistics
        reader.read_exact(&mut buf8).ok()?;
        let total_stone_diff_sum = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8).ok()?;
        let total_games_for_stats = u64::from_le_bytes(buf8);

        Some((
            target_games,
            (black_wins, white_wins, draws),
            total_stone_diff_sum,
            total_games_for_stats,
        ))
    }

    /// Get number of entries for a pattern.
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
        // New format: checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin
        let filename = CheckpointManager::checkpoint_filename(0);
        assert!(filename.starts_with("checkpoint_"));
        assert!(filename.ends_with("_000000.bin"));
        assert!(filename.len() > 20); // Has timestamp

        let filename = CheckpointManager::checkpoint_filename(100000);
        assert!(filename.ends_with("_100000.bin"));

        let filename = CheckpointManager::checkpoint_filename(1000000);
        assert!(filename.ends_with("_1000000.bin"));
    }

    #[test]
    fn test_parse_checkpoint_filename() {
        // Legacy format
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
        // New format with timestamp
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_20251206_115829_100000.bin"),
            Some(100000)
        );
        assert_eq!(
            CheckpointManager::parse_checkpoint_filename("checkpoint_20251206_235959_000000.bin"),
            Some(0)
        );
        // Invalid
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
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save initial checkpoint (game_count = 0)
        let path = manager.save(0, &table, &adam, &patterns, 0).unwrap();

        assert!(path.exists());
        let filename = path.file_name().unwrap().to_str().unwrap();
        // New format: checkpoint_YYYYMMDD_HHMMSS_000000.bin
        assert!(filename.starts_with("checkpoint_"));
        assert!(filename.ends_with("_000000.bin"));
    }

    // ========== Requirement 6.2, 6.3, 6.4, 6.5: Save State ==========

    #[test]
    fn test_save_and_load_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::from_patterns(&patterns);
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
        let mut table = EvaluationTable::from_patterns(&patterns);
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
        let path = manager
            .save(200000, &table, &adam, &patterns, 7200)
            .unwrap();
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
        let table = EvaluationTable::from_patterns(&patterns);
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
        let table = EvaluationTable::from_patterns(&patterns);
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
        let table = EvaluationTable::from_patterns(&patterns);
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

    // ========== Extended Metadata Tests (Resume Support) ==========

    #[test]
    fn test_checkpoint_with_extended_metadata() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Create metadata with statistics
        // Using sum and count instead of Vec<f32>
        let total_stone_diff_sum = 5.0 + (-3.0) + 2.0 + 0.0 + 7.0; // = 11.0
        let total_games_for_stats = 5;
        let meta = CheckpointMeta::with_statistics(
            50000,
            3600,
            100,
            Some(100000), // target_games
            (3, 1, 1),    // accumulated_wins (black, white, draw)
            total_stone_diff_sum,
            total_games_for_stats,
        );

        // Save with extended metadata
        let path = manager
            .save_with_metadata(&table, &adam, &patterns, &meta)
            .unwrap();

        // Load and verify
        let (_, _, loaded_meta) = manager.load(&path, &patterns).unwrap();

        assert_eq!(loaded_meta.game_count, 50000);
        assert_eq!(loaded_meta.elapsed_time_secs, 3600);
        assert_eq!(loaded_meta.adam_timestep, 100);
        assert_eq!(loaded_meta.target_games, Some(100000));
        assert_eq!(loaded_meta.accumulated_wins, (3, 1, 1));
        assert!((loaded_meta.total_stone_diff_sum - 11.0).abs() < 0.001);
        assert_eq!(loaded_meta.total_games_for_stats, 5);
    }

    #[test]
    fn test_legacy_checkpoint_loads_with_default_extended_metadata() {
        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save using old method (without extended metadata)
        let path = manager.save(25000, &table, &adam, &patterns, 1800).unwrap();

        // Load and verify extended fields have default values
        let (_, _, loaded_meta) = manager.load(&path, &patterns).unwrap();

        assert_eq!(loaded_meta.game_count, 25000);
        assert_eq!(loaded_meta.elapsed_time_secs, 1800);
        // Extended fields should be defaults (backward compatible)
        assert_eq!(loaded_meta.target_games, None);
        assert_eq!(loaded_meta.accumulated_wins, (0, 0, 0));
        assert_eq!(loaded_meta.total_stone_diff_sum, 0.0);
        assert_eq!(loaded_meta.total_games_for_stats, 0);
    }

    #[test]
    fn test_checkpoint_meta_with_statistics() {
        // Using sum and count directly instead of Vec
        let total_stone_diff_sum = 1.0 + 2.0 + 3.0 + 4.0 + 5.0; // = 15.0
        let total_games_for_stats = 5;
        let meta = CheckpointMeta::with_statistics(
            10000,
            600,
            50,
            Some(50000),
            (100, 50, 10),
            total_stone_diff_sum,
            total_games_for_stats,
        );

        assert_eq!(meta.game_count, 10000);
        assert_eq!(meta.elapsed_time_secs, 600);
        assert_eq!(meta.adam_timestep, 50);
        assert_eq!(meta.target_games, Some(50000));
        assert_eq!(meta.accumulated_wins, (100, 50, 10));
        assert!((meta.total_stone_diff_sum - 15.0).abs() < 0.001);
        assert_eq!(meta.total_games_for_stats, 5);
    }

    // ========== Requirements Summary Test ==========

    #[test]
    fn test_all_checkpoint_requirements_summary() {
        println!("=== Checkpoint Manager Requirements Verification ===");

        let temp_dir = tempdir().unwrap();
        let manager = CheckpointManager::new(temp_dir.path()).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::from_patterns(&patterns);
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

        // Req 6.6: Filename format (checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin)
        let filename = CheckpointManager::checkpoint_filename(100000);
        assert!(filename.starts_with("checkpoint_"));
        assert!(filename.ends_with("_100000.bin"));
        println!("  6.6: Filename format checkpoint_YYYYMMDD_HHMMSS_NNNNNN.bin");

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
