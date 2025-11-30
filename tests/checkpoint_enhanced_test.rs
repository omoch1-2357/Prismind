//! Enhanced Checkpoint Manager Integration Tests (Task 3)
//!
//! Tests for Phase 4 checkpoint enhancements including:
//! - CheckpointHeader format with version and integrity (3.1)
//! - Atomic checkpoint save with write-to-temp-then-rename (3.2)
//! - CRC32 checksum calculation and verification (3.3)
//! - Optional checkpoint compression (3.4)
//! - Checkpoint retention policy (3.5)
//! - PyCheckpointManager PyO3 wrapper (3.6)

use prismind::evaluator::EvaluationTable;
use prismind::learning::AdamOptimizer;
use prismind::pattern::Pattern;
use std::fs;
use tempfile::tempdir;

/// Create test patterns for checkpoint testing.
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

// ============================================================================
// Task 3.1: CheckpointHeader format with version and integrity
// Requirements: 3.3, 3.4
// ============================================================================

mod header_format_tests {
    use prismind::learning::checkpoint::{
        CHECKPOINT_MAGIC_V2, CHECKPOINT_VERSION, CheckpointHeader,
    };

    #[test]
    fn test_checkpoint_header_has_magic_bytes() {
        // Magic bytes should be "PRSM" (4 bytes)
        assert_eq!(CHECKPOINT_MAGIC_V2.len(), 4);
        assert_eq!(CHECKPOINT_MAGIC_V2, b"PRSM");
    }

    #[test]
    fn test_checkpoint_header_has_version_number() {
        // Version should be defined and be >= 1
        let version = CHECKPOINT_VERSION;
        assert!(version >= 1, "Version must be >= 1, got {}", version);
    }

    #[test]
    fn test_checkpoint_header_structure() {
        let header = CheckpointHeader::new(100000, false);

        assert_eq!(header.magic, *CHECKPOINT_MAGIC_V2);
        assert_eq!(header.version, CHECKPOINT_VERSION);
        assert_eq!(header.games_completed, 100000);
        assert!(header.timestamp > 0);
        // Flags: bit 0 = compressed
        assert_eq!(header.flags & 1, 0); // Not compressed
    }

    #[test]
    fn test_checkpoint_header_with_compression_flag() {
        let header = CheckpointHeader::new(200000, true);

        // Flags bit 0 should be set for compression
        assert_eq!(header.flags & 1, 1);
    }

    #[test]
    fn test_checkpoint_header_checksum_field() {
        let header = CheckpointHeader::new(100000, false);

        // Checksum should initially be 0, set during save
        assert_eq!(header.checksum, 0);
    }

    #[test]
    fn test_checkpoint_header_serialization() {
        let header = CheckpointHeader::new(300000, false);
        let bytes = header.to_bytes();

        // Header should serialize to fixed size
        // magic(4) + version(4) + flags(4) + checksum(4) + games_completed(8) + timestamp(8) = 32 bytes
        assert_eq!(bytes.len(), 32);

        // Should be able to deserialize back
        let parsed = CheckpointHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, header.magic);
        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.games_completed, header.games_completed);
    }

    #[test]
    fn test_version_mismatch_returns_error() {
        // Create a header with wrong version
        let mut bytes = CheckpointHeader::new(100000, false).to_bytes();
        // Modify version bytes (bytes 4-7) to an incompatible version
        bytes[4] = 255;
        bytes[5] = 255;
        bytes[6] = 255;
        bytes[7] = 255;

        let result = CheckpointHeader::from_bytes(&bytes);
        assert!(result.is_err());

        if let Err(e) = result {
            // Error message should contain version details
            let msg = e.to_string();
            assert!(
                msg.contains("version"),
                "Error should mention version: {}",
                msg
            );
        }
    }
}

// ============================================================================
// Task 3.2: Atomic checkpoint save with write-to-temp-then-rename
// Requirements: 3.1, 3.2, 3.10
// ============================================================================

mod atomic_save_tests {
    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_save_creates_final_file_atomically() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _size, _duration) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Final file should exist
        assert!(path.exists());

        // Temp file should NOT exist (was renamed)
        let temp_path = path.with_extension("tmp");
        assert!(!temp_path.exists());
    }

    #[test]
    fn test_no_partial_file_on_disk() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save checkpoint
        manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // List all files in directory
        let files: Vec<_> = fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();

        // No .tmp files should exist
        for entry in &files {
            let path = entry.path();
            assert!(
                path.extension().is_none_or(|ext| ext != "tmp"),
                "Found temporary file: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_save_returns_file_size_and_duration() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, size_bytes, duration_secs) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Size should be non-zero
        assert!(size_bytes > 0);

        // Duration should be non-negative
        assert!(duration_secs >= 0.0);

        // Size should match actual file size
        let actual_size = fs::metadata(&path).unwrap().len();
        assert_eq!(size_bytes, actual_size);
    }
}

// ============================================================================
// Task 3.3: CRC32 checksum calculation and verification
// Requirements: 3.7, 3.8
// ============================================================================

mod crc32_tests {
    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_save_includes_crc32_checksum() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Verify checkpoint returns true for valid file
        assert!(manager.verify(&path).unwrap());
    }

    #[test]
    fn test_corrupted_data_fails_verification() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Corrupt the file by modifying a byte in the data section
        let mut data = fs::read(&path).unwrap();
        if data.len() > 100 {
            data[100] ^= 0xFF; // Flip bits
        }
        fs::write(&path, &data).unwrap();

        // Verification should fail with checksum mismatch
        let result = manager.verify(&path);
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_load_fails_on_checksum_mismatch() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Corrupt the file
        let mut data = fs::read(&path).unwrap();
        if data.len() > 100 {
            data[100] ^= 0xFF;
        }
        fs::write(&path, &data).unwrap();

        // Load should fail with corruption error
        let result = manager.load(&path, &patterns);
        assert!(result.is_err());

        if let Err(e) = result {
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("checksum") || msg.contains("corrupt") || msg.contains("integrity"),
                "Error should mention checksum/corruption: {}",
                e
            );
        }
    }

    #[test]
    fn test_verify_without_full_load() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Verify should work without loading full state
        let is_valid = manager.verify(&path).unwrap();
        assert!(is_valid);
    }
}

// ============================================================================
// Task 3.4: Optional checkpoint compression
// Requirements: 3.9
// ============================================================================

mod compression_tests {
    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_save_with_compression_enabled() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, true).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (path, _size_compressed, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        assert!(path.exists());

        // File should be loadable
        let result = manager.load(&path, &patterns);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compressed_file_is_smaller() {
        let temp_dir = tempdir().unwrap();

        // Save without compression
        let manager_uncompressed =
            EnhancedCheckpointManager::new(temp_dir.path().join("uncompressed"), 5, false).unwrap();

        // Save with compression
        let manager_compressed =
            EnhancedCheckpointManager::new(temp_dir.path().join("compressed"), 5, true).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        let (_, size_uncompressed, _) = manager_uncompressed
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        let (_, size_compressed, _) = manager_compressed
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Compressed should be smaller (pattern data compresses well)
        assert!(
            size_compressed < size_uncompressed,
            "Compressed ({}) should be smaller than uncompressed ({})",
            size_compressed,
            size_uncompressed
        );
    }

    #[test]
    fn test_load_compressed_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, true).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::from_patterns(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Set some non-default values
        table.set(0, 0, 0, 40000);
        adam.update(0, 0, 0, 32768.0, 1.0);
        adam.step();

        let (path, _, _) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Load and verify values are preserved
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();

        assert_eq!(loaded_table.get(0, 0, 0), 40000);
        assert_eq!(loaded_adam.timestep(), 1);
        assert_eq!(meta.game_count, 100000);
    }

    #[test]
    fn test_set_compression_at_runtime() {
        let temp_dir = tempdir().unwrap();
        let mut manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        // Initially compression is off
        assert!(!manager.compression_enabled());

        // Enable compression
        manager.set_compression(true);
        assert!(manager.compression_enabled());

        // Disable compression
        manager.set_compression(false);
        assert!(!manager.compression_enabled());
    }
}

// ============================================================================
// Task 3.5: Checkpoint retention policy
// Requirements: 3.5, 3.6
// ============================================================================

mod retention_tests {
    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_default_retention_is_five() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        assert_eq!(manager.retention_count(), 5);
    }

    #[test]
    fn test_retention_deletes_oldest_checkpoints() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 3, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save 5 checkpoints (retention = 3)
        for i in 0..5 {
            manager
                .save(i * 100000, &table, &adam, &patterns, 0)
                .unwrap();
        }

        // List checkpoints
        let checkpoints = manager.list_checkpoints().unwrap();

        // Should only have 3 (retention count)
        assert_eq!(checkpoints.len(), 3, "Should retain only {} checkpoints", 3);

        // Should have the latest 3 (200000, 300000, 400000)
        let game_counts: Vec<u64> = checkpoints.iter().map(|(_, gc, _, _)| *gc).collect();
        assert!(game_counts.contains(&200000));
        assert!(game_counts.contains(&300000));
        assert!(game_counts.contains(&400000));

        // Oldest (0, 100000) should be deleted
        assert!(!game_counts.contains(&0));
        assert!(!game_counts.contains(&100000));
    }

    #[test]
    fn test_set_retention_at_runtime() {
        let temp_dir = tempdir().unwrap();
        let mut manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        assert_eq!(manager.retention_count(), 5);

        manager.set_retention(10);
        assert_eq!(manager.retention_count(), 10);

        manager.set_retention(2);
        assert_eq!(manager.retention_count(), 2);
    }

    #[test]
    fn test_apply_retention_after_save() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 2, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Save first checkpoint
        let (path1, _, _) = manager.save(100000, &table, &adam, &patterns, 0).unwrap();
        assert!(path1.exists());

        // Save second checkpoint
        let (path2, _, _) = manager.save(200000, &table, &adam, &patterns, 0).unwrap();
        assert!(path2.exists());
        assert!(path1.exists()); // First should still exist (retention = 2)

        // Save third checkpoint - should delete oldest
        let (path3, _, _) = manager.save(300000, &table, &adam, &patterns, 0).unwrap();
        assert!(path3.exists());
        assert!(path2.exists());
        assert!(!path1.exists()); // First should be deleted now
    }
}

// ============================================================================
// Task 3.6: PyCheckpointManager PyO3 wrapper
// Requirements: 3.1, 3.5, 3.7, 3.9, 3.10
// ============================================================================

#[cfg(feature = "pyo3")]
mod pyo3_wrapper_tests {
    // PyO3 wrapper tests would require Python runtime
    // These tests verify the Rust-side interface that PyO3 will wrap

    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_manager_interface_for_pyo3() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        // Interface should return tuple suitable for PyO3
        let (path, size, duration) = manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // Path should be convertible to string
        let path_str = path.to_string_lossy().to_string();
        assert!(!path_str.is_empty());

        // Size and duration are already primitive types
        assert!(size > 0);
        assert!(duration >= 0.0);
    }

    #[test]
    fn test_list_checkpoints_returns_tuple_format() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);

        manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        // list_checkpoints returns Vec<(String, u64, String, u64)>
        // (path, games_completed, timestamp_str, size_bytes)
        let checkpoints = manager.list_checkpoints().unwrap();

        assert_eq!(checkpoints.len(), 1);
        let (path, games, timestamp, size) = &checkpoints[0];

        assert!(!path.is_empty());
        assert_eq!(*games, 100000);
        assert!(!timestamp.is_empty()); // Timestamp string
        assert!(*size > 0);
    }

    #[test]
    fn test_load_latest_returns_option() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();

        // Empty directory returns None
        let result = manager.load_latest(&patterns).unwrap();
        assert!(result.is_none());

        // After saving, returns Some
        let table = EvaluationTable::from_patterns(&patterns);
        let adam = AdamOptimizer::new(&patterns);
        manager
            .save(100000, &table, &adam, &patterns, 3600)
            .unwrap();

        let result = manager.load_latest(&patterns).unwrap();
        assert!(result.is_some());
    }
}

// ============================================================================
// Integration tests combining all Task 3 features
// ============================================================================

mod integration_tests {
    use super::*;
    use prismind::learning::checkpoint::EnhancedCheckpointManager;

    #[test]
    fn test_full_checkpoint_workflow() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 3, true).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::from_patterns(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Modify state
        table.set(0, 0, 0, 45000);
        table.set(5, 15, 100, 25000);
        for i in 0..10 {
            adam.update(0, 0, i, 32768.0, 1.0 + i as f32 * 0.1);
        }
        adam.step();
        adam.step();

        // Save checkpoint
        let (path, size, duration) = manager
            .save(100000, &table, &adam, &patterns, 7200)
            .unwrap();

        // Verify checkpoint
        assert!(manager.verify(&path).unwrap());

        // Load checkpoint
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();

        // Verify all state is preserved
        assert_eq!(loaded_table.get(0, 0, 0), 45000);
        assert_eq!(loaded_table.get(5, 15, 100), 25000);
        assert_eq!(loaded_adam.timestep(), 2);
        assert_eq!(meta.game_count, 100000);
        assert_eq!(meta.elapsed_time_secs, 7200);

        println!("Checkpoint saved: {} bytes in {:.3}s", size, duration);
    }

    #[test]
    fn test_checkpoint_preserves_state_exactly() {
        let temp_dir = tempdir().unwrap();
        let manager = EnhancedCheckpointManager::new(temp_dir.path(), 5, false).unwrap();

        let patterns = create_test_patterns();
        let mut table = EvaluationTable::from_patterns(&patterns);
        let mut adam = AdamOptimizer::new(&patterns);

        // Set specific values across all patterns and stages
        for pattern_id in 0..14 {
            for stage in 0..30 {
                let value = 32768 + (pattern_id * 100 + stage * 10) as u16;
                table.set(pattern_id, stage, 0, value);
            }
        }

        // Update Adam state
        for _ in 0..5 {
            adam.update(0, 0, 0, 32768.0, 1.5);
            adam.step();
        }

        // Save
        let (path, _, _) = manager
            .save(500000, &table, &adam, &patterns, 18000)
            .unwrap();

        // Load
        let (loaded_table, loaded_adam, meta) = manager.load(&path, &patterns).unwrap();

        // Verify all values
        for pattern_id in 0..14 {
            for stage in 0..30 {
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

        assert_eq!(loaded_adam.timestep(), 5);
        assert_eq!(meta.game_count, 500000);
    }
}
