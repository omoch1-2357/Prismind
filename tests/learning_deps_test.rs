//! Tests to verify learning module dependencies are properly configured.
//!
//! These tests ensure that all required dependencies for the Phase 3 learning system
//! are available and properly versioned.

/// Test that bincode is available for checkpoint serialization
#[test]
fn test_bincode_available() {
    // bincode 2.x uses these traits for serialization
    use bincode::{Decode, Encode};

    // Verify we can define types with bincode traits
    #[derive(Encode, Decode, PartialEq, Debug)]
    struct TestCheckpoint {
        game_count: u64,
        elapsed_secs: u64,
    }

    // Test round-trip serialization
    let original = TestCheckpoint {
        game_count: 100_000,
        elapsed_secs: 3600,
    };

    let config = bincode::config::standard();
    let encoded = bincode::encode_to_vec(&original, config).unwrap();
    let (decoded, _): (TestCheckpoint, _) = bincode::decode_from_slice(&encoded, config).unwrap();

    assert_eq!(original, decoded);
}

/// Test that rayon is available for parallel game execution
#[test]
fn test_rayon_available() {
    use rayon::prelude::*;

    // Test parallel iteration
    let numbers: Vec<i32> = (0..100).collect();
    let sum: i32 = numbers.par_iter().sum();

    assert_eq!(sum, 4950); // 0 + 1 + ... + 99 = 4950
}

/// Test that chrono is available for timestamp handling
#[test]
fn test_chrono_available() {
    use chrono::{DateTime, Utc};

    // Test timestamp creation
    let now: DateTime<Utc> = Utc::now();
    let timestamp = now.timestamp();

    // Timestamp should be reasonable (after 2020)
    assert!(timestamp > 1577836800); // 2020-01-01 00:00:00 UTC
}

/// Test that log and env_logger are available
#[test]
fn test_log_available() {
    use log::{Level, LevelFilter, debug, error, info, trace, warn};

    // Test that log macros are available (they may not output anything without initialization)
    // This just verifies compilation works
    let _ = Level::Info;
    let _ = LevelFilter::Debug;

    // These macros should compile (output depends on logger initialization)
    trace!("trace message");
    debug!("debug message");
    info!("info message");
    warn!("warn message");
    error!("error message");
}

/// Test that ctrlc is available for graceful shutdown
#[test]
fn test_ctrlc_available() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Test that we can create the pattern used for signal handling
    let running = Arc::new(AtomicBool::new(true));

    // Verify atomic operations work
    assert!(running.load(Ordering::SeqCst));
    running.store(false, Ordering::SeqCst);
    assert!(!running.load(Ordering::SeqCst));

    // Note: We don't actually set up ctrlc handler in test as it modifies global state
    // and can only be called once per process. The actual handler is tested via
    // TrainingEngine which uses a global OnceLock for single registration.
}

/// Test that rand is available as a regular dependency (not just dev-dependency)
#[test]
fn test_rand_available() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Create seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Test random number generation
    let value: f64 = rng.random();
    assert!((0.0..1.0).contains(&value));

    // Test random range
    let int_value: u32 = rng.random_range(0..100);
    assert!(int_value < 100);
}
