# Implementation Plan

## Task 1. Project Setup and Dependencies
- [x] 1.1 (P) Configure Cargo dependencies for learning module
  - Add bincode 2.0+ for checkpoint serialization
  - Add rayon 1.10+ for parallel game execution
  - Add chrono for timestamp handling
  - Add log 0.4 and env_logger for training statistics output
  - Add ctrlc for graceful shutdown signal handling
  - Promote rand from dev-dependency to regular dependency
  - _Requirements: 6.1, 7.6, 12.7, 13.8_

- [x] 1.2 (P) Create learning module structure
  - Create src/learning/mod.rs with module declarations
  - Define LearningError enum with all error variants (Io, InvalidCheckpoint, Search, EvaluationDivergence, MemoryAllocation, Config, Interrupted)
  - Export public types for external access
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

## Task 2. Score Representation and Conversion Utilities
- [x] 2. Implement score conversion functions
  - Define CENTER constant as 32768 for zero stone-difference
  - Define SCALE constant as 256.0 for conversion factor
  - Implement u16 to stone difference conversion: (value - 32768) / 256.0
  - Implement stone difference to u16 conversion with clamping to [0, 65535]
  - Initialize pattern table entries to 32768 (neutral evaluation)
  - Add unit tests for conversion edge cases and clamping behavior
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

## Task 3. Core Algorithm Components
- [x] 3.1 (P) Implement Eligibility Trace storage
  - Create sparse HashMap-based storage keyed by (pattern_id, stage, index) tuples
  - Implement increment operation that adds 1.0 to trace value on pattern visit
  - Implement decay operation that multiplies all traces by lambda (0.3)
  - Implement get operation returning 0.0 for unvisited entries
  - Implement reset operation to clear all traces for new game
  - Minimize memory usage through sparse representation
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 3.2 (P) Implement Adam optimizer with moment vectors
  - Initialize with hyperparameters: alpha=0.025, beta1=0.9, beta2=0.999, epsilon=1e-8
  - Allocate first moment (m) vectors matching EvaluationTable layout (~114 MB)
  - Allocate second moment (v) vectors matching EvaluationTable layout (~114 MB)
  - Maintain global timestep counter for bias correction
  - Implement bias-corrected update: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
  - Compute parameter update: delta = alpha * m_hat / (sqrt(v_hat) + epsilon)
  - Initialize all moment values to 0.0
  - Add memory usage tracking method
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 3.3 Implement TD-Leaf learner with backward updates
  - Configure lambda=0.3 as trace decay parameter
  - Iterate game history in reverse order from final to initial position
  - Compute TD error as difference between target value and current evaluation
  - Apply target formula: lambda * final_score + (1 - lambda) * next_value for non-terminal positions
  - Use actual game result (stone difference) as target at final position
  - Update all 56 pattern instances (14 patterns x 4 rotations) per position
  - Account for side-to-move by negating values for White's perspective
  - Integrate with Adam optimizer for gradient updates
  - Integrate with Eligibility Trace for credit assignment
  - Clamp updated weights to valid u16 range
  - Depends on Task 3.1 (EligibilityTrace) and Task 3.2 (AdamOptimizer)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

## Task 4. Game History and Self-Play Engine
- [x] 4.1 (P) Implement game history recording
  - Create MoveRecord structure with board state, leaf value, 56 pattern indices, and stage
  - Create GameHistory container supporting up to 60 moves per game
  - Implement push operation for adding move records
  - Implement reverse iteration for TD backward updates
  - Use memory-efficient format suitable for deallocation after TD update
  - Pre-allocate capacity for 60 moves to avoid reallocation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [x] 4.2 Implement epsilon schedule for exploration control
  - Return epsilon=0.15 for games 0-299,999 (high exploration phase)
  - Return epsilon=0.05 for games 300,000-699,999 (moderate exploration)
  - Return epsilon=0.0 for games 700,000-999,999 (exploitation phase)
  - Implement as stateless computation based on game number
  - _Requirements: 4.3, 4.4, 4.5_

- [x] 4.3 Implement self-play game engine
  - Play complete games from initial Othello position to termination
  - Use Phase 2 Search API with 15ms time limit per move
  - Apply epsilon-greedy move selection using epsilon schedule
  - Record board state, leaf evaluation, pattern indices, and stage for each move
  - Use current board's static evaluation as leaf value for random moves
  - Handle pass moves correctly according to Othello rules
  - Compute final stone difference as game result
  - Return complete GameHistory with final score
  - Depends on Task 4.1 (GameHistory) and Task 4.2 (EpsilonSchedule)
  - _Requirements: 4.1, 4.2, 4.6, 4.7, 4.8, 4.9_

## Task 5. Thread-Safe Evaluation Integration
- [x] 5. Implement SharedEvaluator with RwLock protection
  - Wrap EvaluationTable with Arc<RwLock<>> for thread-safe access
  - Provide read guard for concurrent evaluation during parallel games
  - Provide write guard for exclusive TD weight updates
  - Implement evaluate method acquiring read lock
  - Store patterns reference for pattern extraction
  - Reuse Phase 2 Evaluator interface for Search compatibility
  - Ensure consistency between learning and search evaluation access
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

## Task 6. Persistence Layer
- [x] 6.1 (P) Implement checkpoint manager for training state
  - Define binary format with 24-byte magic header "OTHELLO_AI_CHECKPOINT_V1"
  - Save pattern table weights (~57 MB) using bincode serialization
  - Save Adam optimizer m and v moments (~228 MB)
  - Save Adam timestep counter
  - Save metadata: game count, elapsed time, creation timestamp
  - Use filename format checkpoint_NNNNNN.bin (6-digit zero-padded)
  - Support saving initial checkpoint_000000.bin before training
  - Implement checkpoint load with header verification
  - Return appropriate error on corruption or version mismatch
  - Allow fresh start if checkpoint loading fails
  - Implement find_latest to locate most recent checkpoint
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10_

- [x] 6.2 (P) Implement training logger for statistics output
  - Output real-time statistics every 100 games: stone difference, win rates, move count, elapsed time
  - Output detailed statistics every 10,000 games: evaluation distribution, search depth, search time, TT hit rate
  - Output complete summary at each checkpoint
  - Write logs to files with format logs/training_YYYYMMDD_HHMMSS.log
  - Output progress reports with estimated time remaining
  - Detect and warn on evaluation divergence (NaN, extreme values)
  - Ensure non-blocking file writes to avoid blocking main training loop
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

## Task 7. Training Monitoring
- [x] 7. Implement convergence monitoring system
  - Track average stone difference trends across training
  - Track evaluation value stability (variance over time)
  - Track percentage of pattern entries receiving updates
  - Target 200+ updates per entry on average
  - Output convergence metrics every 100,000 games
  - Detect learning stagnation (no improvement for 50,000+ games)
  - Warn if win rate against random baseline drops below 95%
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

## Task 8. Training Engine Orchestration
- [x] 8.1 Implement training engine core with parallel execution
  - Manage rayon thread pool for 4-thread parallel game execution
  - Coordinate sequential TD updates after batch of games completes
  - Acquire read locks for parallel evaluation during games
  - Acquire write lock exclusively during TD weight updates
  - Trigger checkpoint save every 100,000 games
  - Trigger batch logging every 100 games
  - Trigger detailed logging every 10,000 games
  - Enforce total memory budget of 600 MB
  - Depends on Tasks 3.3, 4.3, 5, 6.1, 6.2, 7
  - _Requirements: 4.10, 13.1, 13.2, 13.5, 13.6, 13.8_

- [x] 8.2 Implement checkpoint resume and graceful shutdown
  - Support initialization from checkpoint file for training resume
  - Restore pattern weights, Adam state, and metadata on resume
  - Validate loaded state consistency
  - Register SIGINT/SIGTERM handler with ctrlc
  - Save checkpoint before exit on interrupt signal
  - Reduce transposition table size and retry on memory allocation failure
  - _Requirements: 12.5, 12.7, 8.7_

## Task 9. Error Handling and Recovery
- [ ] 9. Implement comprehensive error handling
  - Log search errors and skip to next game on search failure
  - Retry checkpoint save once on failure, log warning if still failing
  - Reset affected entries to 32768 on NaN/infinite evaluation values
  - Catch and log panics without crashing training process
  - Track error counts per 10,000 game window
  - Pause training and report if >1% of games fail in window
  - Report error pattern for operator diagnosis
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6_

## Task 10. Memory Management Validation
- [ ] 10. Validate and enforce memory constraints
  - Verify pattern table uses ~57 MB
  - Verify Adam optimizer uses ~228 MB total
  - Configure transposition table to 128-256 MB (shared with Phase 2)
  - Release game history memory after each TD update
  - Use sparse eligibility traces to minimize per-game memory
  - Validate total memory stays within 600 MB budget
  - Add runtime memory monitoring during training
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

## Task 11. Performance Optimization and Testing
- [ ] 11.1 Implement performance benchmarks
  - Measure game throughput targeting 4.6 games/second minimum
  - Measure TD update latency targeting under 10ms per game
  - Measure checkpoint save time targeting under 30 seconds
  - Verify search operations maintain 15ms per move average
  - Measure CPU utilization targeting 80% or higher
  - Track total training time progress toward 50-60 hour target
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.7_

- [ ] 11.2 Integration tests for complete training flow
  - Test single self-play game completion with valid history
  - Test TD update produces expected weight changes
  - Test checkpoint save/load round-trip preserves state exactly
  - Test parallel game execution without data races (4 threads)
  - Test Phase 2 Search API integration with shared evaluator
  - Test epsilon schedule transitions at boundary game numbers
  - _Requirements: 1.1, 4.1, 6.7, 9.1, 13.8_

- [ ]* 11.3 Unit tests for algorithm correctness
  - Test TD error computation for known game positions
  - Test eligibility trace increment, decay, and reset operations
  - Test Adam optimizer bias correction computation
  - Test score conversion clamping at u16 boundaries
  - Test epsilon schedule returns correct values for each phase
  - Test convergence monitor detects stagnation condition
  - _Requirements: 1.3, 2.2, 2.3, 3.8, 3.9, 10.6, 11.4, 11.5_
