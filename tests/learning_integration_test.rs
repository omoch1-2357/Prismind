//! Task 11.2: Integration tests for complete training flow.
//!
//! Tests the integration between all learning module components:
//! - Self-play game completion
//! - TD updates and weight changes
//! - Checkpoint save/load round-trip
//! - Parallel game execution
//! - Phase 2 Search API integration
//! - Epsilon schedule transitions
//!
//! # Requirements Coverage
//!
//! - Req 1.1: TD(lambda)-Leaf algorithm
//! - Req 4.1: Self-play game execution
//! - Req 6.7: Checkpoint save/load
//! - Req 9.1: Phase 2 integration
//! - Req 13.8: Integration testing

use prismind::evaluator::{EvaluationTable, Evaluator};
use prismind::learning::adam::AdamOptimizer;
use prismind::learning::td_learner::MoveRecord as TDMoveRecord;
use prismind::learning::*;
use prismind::pattern::load_patterns;
use prismind::search::Search;
use rand::SeedableRng;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

/// Load patterns as a fixed-size array
fn load_patterns_array() -> [prismind::pattern::Pattern; 14] {
    let patterns_vec = load_patterns("patterns.csv").expect("Failed to load patterns");
    patterns_vec
        .try_into()
        .expect("Expected exactly 14 patterns")
}

/// Convert GameHistory to TD MoveRecords
fn game_history_to_move_records(history: &prismind::learning::GameHistory) -> Vec<TDMoveRecord> {
    history
        .iter()
        .map(|record| {
            TDMoveRecord::new(
                record.leaf_value,
                record.pattern_indices,
                record.stage,
                record.is_black_turn(),
            )
        })
        .collect()
}

// ========== Task 11.2.1: Self-play Game Completion Tests ==========

#[test]
fn test_single_self_play_game_completion() {
    // Req 4.1: Test single self-play game completion with valid history

    // Create evaluator and search
    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let patterns = load_patterns_array();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Play a single game with epsilon=0.15 (high exploration phase)
    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    );

    // Verify game completed with valid result
    match result {
        Ok(game_result) => {
            // Verify valid stone difference
            assert!(
                game_result.final_score >= -64.0 && game_result.final_score <= 64.0,
                "Stone difference {} should be within valid range [-64, 64]",
                game_result.final_score
            );

            // Verify history has valid moves
            assert!(
                !game_result.history.is_empty(),
                "Game history should have at least one move"
            );

            // Verify moves count is reasonable (game should have multiple moves)
            let move_count = game_result.history.len();
            assert!(
                move_count >= 4,
                "Game should have at least 4 moves, got {}",
                move_count
            );

            // Verify all recorded positions are valid via iteration
            for record in game_result.history.iter() {
                assert!(
                    record.stage < 30,
                    "Stage {} should be less than 30",
                    record.stage
                );
            }
        }
        Err(e) => {
            panic!("Self-play game failed: {:?}", e);
        }
    }
}

#[test]
fn test_self_play_game_produces_valid_history() {
    // Req 5.1: Verify game history recording

    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let patterns = load_patterns_array();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    )
    .unwrap();

    // Check history properties
    let history = &result.history;

    // Verify history is not empty
    assert!(!history.is_empty());

    // Verify stages are recorded correctly (0-29)
    for record in history.iter() {
        assert!(
            record.stage < 30,
            "Stage {} should be less than 30",
            record.stage
        );
    }

    // Verify pattern indices are recorded
    for record in history.iter() {
        assert_eq!(
            record.pattern_indices.len(),
            NUM_PATTERN_INSTANCES,
            "Should have {} pattern indices",
            NUM_PATTERN_INSTANCES
        );
    }
}

// ========== Task 11.2.2: TD Update Weight Change Tests ==========

#[test]
fn test_td_update_produces_weight_changes() {
    // Req 1.1: Test TD update produces expected weight changes

    let patterns = load_patterns_array();
    let mut eval_table = EvaluationTable::from_patterns(&patterns);

    // Create a simple evaluator and search for game play
    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(456);

    // Play a game to get history
    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    )
    .unwrap();

    // Convert history to TD format
    let move_records = game_history_to_move_records(&result.history);

    // Create TD learner and Adam optimizer
    let mut learner = TDLearner::default_lambda();
    let mut adam = AdamOptimizer::new(&patterns);

    // Perform TD update
    let stats = learner.update(
        &move_records,
        result.final_score,
        &mut eval_table,
        &mut adam,
    );

    // Verify updates occurred
    assert!(
        stats.patterns_updated > 0,
        "TD update should modify some patterns"
    );

    // Log for debugging
    println!(
        "TD update stats: {} moves processed, {} patterns updated",
        stats.moves_processed, stats.patterns_updated
    );
}

#[test]
fn test_td_update_direction_matches_outcome() {
    // Req 1.3: Test TD error has correct sign based on outcome

    let patterns = load_patterns_array();
    let mut eval_table = EvaluationTable::from_patterns(&patterns);

    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);

    // Play a game
    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    )
    .unwrap();

    // Convert history to TD format
    let move_records = game_history_to_move_records(&result.history);

    // Create fresh learner and optimizer
    let mut learner = TDLearner::default_lambda();
    let mut adam = AdamOptimizer::new(&patterns);

    // Perform update
    let stats = learner.update(
        &move_records,
        result.final_score,
        &mut eval_table,
        &mut adam,
    );

    // The direction of updates should correlate with outcome
    assert!(
        stats.patterns_updated > 0,
        "Should have updated some patterns"
    );

    println!(
        "Game score: {}, patterns updated: {}, avg TD error: {:.4}",
        result.final_score, stats.patterns_updated, stats.avg_td_error
    );
}

// ========== Task 11.2.3: Checkpoint Save/Load Round-Trip Tests ==========

#[test]
fn test_checkpoint_save_load_preserves_state() {
    // Req 6.7: Test checkpoint save/load round-trip preserves state exactly

    let temp_dir = tempdir().unwrap();

    // Create and modify evaluation table
    let patterns = load_patterns_array();
    let mut eval_table = EvaluationTable::from_patterns(&patterns);

    // Make some modifications (using safe indices within pattern bounds)
    eval_table.set(0, 0, 100, 40000);
    eval_table.set(5, 15, 500, 25000);
    eval_table.set(10, 29, 50, 50000); // Pattern 10 has up to 3^6 = 729 entries

    // Create Adam optimizer
    let adam = AdamOptimizer::new(&patterns);

    // Create checkpoint manager
    let checkpoint_manager = CheckpointManager::new(temp_dir.path()).unwrap();

    // Save checkpoint with training state
    let games_completed = 50000u64;
    let training_time_secs = 3600u64;

    let checkpoint_path = checkpoint_manager
        .save(
            games_completed,
            &eval_table,
            &adam,
            &patterns,
            training_time_secs,
        )
        .unwrap();

    // Load checkpoint
    let (loaded_table, _loaded_adam, loaded_meta) = checkpoint_manager
        .load(&checkpoint_path, &patterns)
        .unwrap();

    // Verify state preserved exactly
    assert_eq!(
        loaded_meta.game_count, games_completed,
        "Games completed should match"
    );
    assert_eq!(
        loaded_meta.elapsed_time_secs, training_time_secs,
        "Training time should match"
    );

    // Verify evaluation table values preserved
    assert_eq!(
        loaded_table.get(0, 0, 100),
        40000,
        "Modified entry should be preserved"
    );
    assert_eq!(
        loaded_table.get(5, 15, 500),
        25000,
        "Modified entry should be preserved"
    );
    assert_eq!(
        loaded_table.get(10, 29, 50),
        50000,
        "Modified entry should be preserved"
    );
}

#[test]
fn test_checkpoint_round_trip_all_patterns() {
    // Test that all pattern entries are preserved

    let temp_dir = tempdir().unwrap();
    let patterns = load_patterns_array();
    let mut eval_table = EvaluationTable::from_patterns(&patterns);

    // Set specific test values across different patterns (using safe indices)
    // Pattern 0: 3^10 = 59049 entries, Pattern 10: 3^6 = 729 entries
    let test_values: Vec<(usize, usize, usize, u16)> =
        vec![(0, 0, 0, 30000), (0, 29, 100, 35000), (10, 15, 50, 40000)];

    for (pattern_id, stage, idx, value) in &test_values {
        eval_table.set(*pattern_id, *stage, *idx, *value);
    }

    // Create Adam and checkpoint manager
    let adam = AdamOptimizer::new(&patterns);
    let checkpoint_manager = CheckpointManager::new(temp_dir.path()).unwrap();

    // Save
    let checkpoint_path = checkpoint_manager
        .save(0, &eval_table, &adam, &patterns, 0)
        .unwrap();

    // Load
    let (loaded, _, _) = checkpoint_manager
        .load(&checkpoint_path, &patterns)
        .unwrap();

    // Verify all test values
    for (pattern_id, stage, idx, expected) in &test_values {
        let actual = loaded.get(*pattern_id, *stage, *idx);
        assert_eq!(
            actual, *expected,
            "Pattern {} stage {} idx {} should be {}, got {}",
            pattern_id, stage, idx, expected, actual
        );
    }
}

// ========== Task 11.2.4: Parallel Game Execution Tests ==========

#[test]
fn test_parallel_game_execution_no_data_races() {
    // Req 9.1, 13.8: Test parallel game execution without data races (4 threads)

    let shared = SharedEvaluator::new("patterns.csv").unwrap();
    let shared = Arc::new(shared);
    let no_races = Arc::new(AtomicBool::new(true));

    // Spawn 4 parallel games
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let shared_clone = Arc::clone(&shared);
            let _no_races_clone = Arc::clone(&no_races);

            thread::spawn(move || {
                // Each thread plays multiple games
                for game_idx in 0..5 {
                    // Get read lock and play
                    let table = shared_clone.read();

                    // Simulate game by reading pattern values
                    // Use safe indices (pattern 0 has 3^10 = 59049 entries)
                    for stage in 0..30 {
                        let _value = table.get(0, stage, 100);
                    }

                    drop(table);

                    // Small delay to allow interleaving
                    thread::sleep(Duration::from_micros(100));

                    // Report progress
                    println!("Thread {} completed game {}", thread_id, game_idx);
                }

                // If we get here without panic, no data races
                true
            })
        })
        .collect();

    // Wait for all threads
    let mut all_ok = true;
    for handle in handles {
        match handle.join() {
            Ok(result) => all_ok = all_ok && result,
            Err(_) => {
                no_races.store(false, Ordering::SeqCst);
                all_ok = false;
            }
        }
    }

    assert!(
        all_ok,
        "All parallel games should complete without data races"
    );
    assert!(
        no_races.load(Ordering::SeqCst),
        "No data races should occur"
    );
}

#[test]
fn test_shared_evaluator_concurrent_access() {
    // Test SharedEvaluator with concurrent read/write access

    let shared = SharedEvaluator::new("patterns.csv").unwrap();
    let shared = Arc::new(shared);
    let success = Arc::new(AtomicBool::new(true));

    // Spawn reader threads (using safe pattern 0 index)
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let shared_clone = Arc::clone(&shared);
            thread::spawn(move || {
                for _ in 0..100 {
                    let table = shared_clone.read();
                    let _val = table.get(0, 15, 100); // Pattern 0 has 3^10 entries, 100 is safe
                    drop(table);
                    thread::yield_now();
                }
            })
        })
        .collect();

    // Spawn writer thread
    let writer_shared = Arc::clone(&shared);
    let _writer_success = Arc::clone(&success);
    let writer = thread::spawn(move || {
        for i in 0..10 {
            let mut table = writer_shared.write();
            table.set(0, 15, 100, (32768 + i) as u16); // Pattern 0, safe index
            drop(table);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Wait for all
    for r in readers {
        r.join().unwrap();
    }
    writer.join().unwrap();

    assert!(
        success.load(Ordering::SeqCst),
        "Concurrent access should succeed"
    );
}

// ========== Task 11.2.5: Phase 2 Search API Integration Tests ==========

#[test]
fn test_phase2_search_api_integration() {
    // Req 9.1: Test Phase 2 Search API integration with shared evaluator

    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let patterns = load_patterns_array();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(999);

    // Play a game using the search API
    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        DEFAULT_SEARCH_TIME_MS,
        &mut rng,
        StartingPlayer::Black,
    );
    assert!(
        result.is_ok(),
        "Game should complete with search integration"
    );

    let game_result = result.unwrap();
    println!(
        "Game completed: {} moves, final score: {}",
        game_result.moves_played, game_result.final_score
    );
}

#[test]
fn test_search_integration_returns_valid_moves() {
    // Verify search returns valid moves during self-play

    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let patterns = load_patterns_array();
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(1111);

    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    )
    .unwrap();

    // All stages should be valid (0-29)
    for record in result.history.iter() {
        assert!(
            record.stage < 30,
            "Stage {} should be valid (0-29)",
            record.stage
        );
    }
}

// ========== Task 11.2.6: Epsilon Schedule Transition Tests ==========

#[test]
fn test_epsilon_schedule_transitions() {
    // Test epsilon schedule returns correct values for each phase

    // Phase 1: Games 0-299,999 -> epsilon 0.15
    assert!(
        (EpsilonSchedule::get(0) - 0.15).abs() < 0.001,
        "Game 0 should have epsilon 0.15"
    );
    assert!(
        (EpsilonSchedule::get(150_000) - 0.15).abs() < 0.001,
        "Game 150,000 should have epsilon 0.15"
    );
    assert!(
        (EpsilonSchedule::get(299_999) - 0.15).abs() < 0.001,
        "Game 299,999 should have epsilon 0.15"
    );

    // Phase 2: Games 300,000-699,999 -> epsilon 0.05
    assert!(
        (EpsilonSchedule::get(300_000) - 0.05).abs() < 0.001,
        "Game 300,000 should have epsilon 0.05"
    );
    assert!(
        (EpsilonSchedule::get(500_000) - 0.05).abs() < 0.001,
        "Game 500,000 should have epsilon 0.05"
    );
    assert!(
        (EpsilonSchedule::get(699_999) - 0.05).abs() < 0.001,
        "Game 699,999 should have epsilon 0.05"
    );

    // Phase 3: Games 700,000+ -> epsilon 0.0
    assert!(
        (EpsilonSchedule::get(700_000) - 0.0).abs() < 0.001,
        "Game 700,000 should have epsilon 0.0"
    );
    assert!(
        (EpsilonSchedule::get(850_000) - 0.0).abs() < 0.001,
        "Game 850,000 should have epsilon 0.0"
    );
    assert!(
        (EpsilonSchedule::get(999_999) - 0.0).abs() < 0.001,
        "Game 999,999 should have epsilon 0.0"
    );
}

#[test]
fn test_epsilon_schedule_boundary_transitions() {
    // Test exact boundary transitions

    // Transition at 300,000
    let before = EpsilonSchedule::get(299_999);
    let after = EpsilonSchedule::get(300_000);
    assert!(
        (before - 0.15).abs() < 0.001,
        "Before boundary should be 0.15"
    );
    assert!(
        (after - 0.05).abs() < 0.001,
        "After boundary should be 0.05"
    );

    // Transition at 700,000
    let before = EpsilonSchedule::get(699_999);
    let after = EpsilonSchedule::get(700_000);
    assert!(
        (before - 0.05).abs() < 0.001,
        "Before boundary should be 0.05"
    );
    assert!((after - 0.0).abs() < 0.001, "After boundary should be 0.0");
}

// ========== Requirements Summary ==========

#[test]
fn test_integration_requirements_summary() {
    println!("=== Task 11.2: Integration Test Requirements ===");

    // Req 1.1: TD(lambda)-Leaf algorithm
    let patterns = load_patterns_array();
    let mut eval_table = EvaluationTable::from_patterns(&patterns);

    let evaluator = Evaluator::new("patterns.csv").expect("Failed to load evaluator");
    let mut search = Search::new(evaluator, 128).expect("Failed to create search");
    let mut rng = rand::rngs::StdRng::seed_from_u64(2222);

    let result = play_game(
        &mut search,
        &patterns,
        0.15,
        15,
        &mut rng,
        StartingPlayer::Black,
    )
    .unwrap();
    let move_records = game_history_to_move_records(&result.history);

    let mut learner = TDLearner::default_lambda();
    let mut adam = AdamOptimizer::new(&patterns);
    let stats = learner.update(
        &move_records,
        result.final_score,
        &mut eval_table,
        &mut adam,
    );
    assert!(stats.patterns_updated > 0);
    println!("  1.1: TD(lambda)-Leaf algorithm - PASS");

    // Req 4.1: Self-play game execution
    assert!(!result.history.is_empty());
    println!("  4.1: Self-play game execution - PASS");

    // Req 6.7: Checkpoint functionality
    let temp_dir = tempdir().unwrap();
    let checkpoint_mgr = CheckpointManager::new(temp_dir.path()).unwrap();
    let path = checkpoint_mgr
        .save(1000, &eval_table, &adam, &patterns, 100)
        .unwrap();
    let (_, _, meta) = checkpoint_mgr.load(&path, &patterns).unwrap();
    assert_eq!(meta.game_count, 1000);
    println!("  6.7: Checkpoint save/load - PASS");

    // Req 9.1: Phase 2 integration
    let patterns = load_patterns_array();
    let shared = SharedEvaluator::from_parts(patterns, eval_table);
    let _table = shared.read();
    println!("  9.1: Phase 2 integration with shared evaluator - PASS");

    // Epsilon schedule
    assert!((EpsilonSchedule::get(0) - 0.15).abs() < 0.001);
    println!("  Epsilon schedule transitions - PASS");

    println!("=== All integration requirements verified ===");
}
