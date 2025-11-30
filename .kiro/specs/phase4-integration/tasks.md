# Implementation Plan

## Task 1. Build Configuration and PyO3 Module Foundation

- [x] 1.1 (P) Configure build system for PyO3 Python bindings
  - Add PyO3 dependency with extension-module feature and abi3-py38 for Python version compatibility
  - Add numpy, crc32fast, and flate2 dependencies for enhanced functionality
  - Configure release profile with opt-level=3, LTO, and codegen-units=1 for maximum performance
  - Set crate-type to include both cdylib (for Python) and rlib (for Rust library use)
  - _Requirements: 11.1, 11.3_

- [x] 1.2 (P) Create maturin build configuration
  - Create pyproject.toml with maturin as build backend
  - Configure Python source directory and module naming conventions
  - Add cross-compilation support for ARM64 target (aarch64-unknown-linux-gnu)
  - Enable NEON SIMD optimizations via target-cpu flags when building for ARM64
  - _Requirements: 11.2, 11.4, 11.5_

- [x] 1.3 Create PyO3 module entry point with class registration
  - Create the prismind Python module structure with proper PyO3 annotations
  - Register all PyO3 classes (PyEvaluator, PyTrainingManager, PyCheckpointManager, PyStatisticsManager, PyDebugModule)
  - Add module metadata including version from cargo package
  - Handle module initialization errors with descriptive Python exceptions
  - _Requirements: 1.1, 1.8_

## Task 2. PyEvaluator Implementation

- [x] 2.1 Implement PyEvaluator class with board evaluation
  - Create PyEvaluator class wrapping the Rust Evaluator with thread-safe access via Arc
  - Implement constructor accepting optional checkpoint path for loading pre-trained weights
  - Implement evaluate method accepting 64-element board array (Python list) with player indicator
  - Validate board array has exactly 64 elements with values in valid range (0, 1, 2)
  - Return evaluation score as Python float with positive values favoring black
  - Raise descriptive Python exception for invalid parameters
  - Note: GIL release pattern for evaluate() - release GIL during Rust computation to avoid blocking Python
  - _Requirements: 1.1, 1.2, 1.8_

- [x] 2.2 (P) Add NumPy array support to PyEvaluator
  - Implement evaluate_numpy method accepting NumPy array for efficient data interchange
  - Handle NumPy array conversion to internal board representation
  - Ensure thread-safety for concurrent evaluation calls from Python
  - _Requirements: 1.9_

- [x] 2.3 (P) Implement pattern weight access for external analysis
  - Implement get_weights method returning dictionary mapping (pattern_id, stage, index) to weight value
  - Implement get_weight method for accessing specific pattern entry weights
  - Expose pattern table through RwLock for thread-safe read access
  - _Requirements: 1.10_

## Task 3. Enhanced Checkpoint Manager

- [x] 3.1 Implement checkpoint header format with version and integrity
  - Define CheckpointHeader structure with magic bytes ("PRSM"), version number, and flags
  - Add CRC32 checksum field for data integrity verification
  - Include games_completed and timestamp in header for metadata
  - Implement version validation that returns error with version details on mismatch
  - Note: Design review issue - document version migration strategy for future format changes
  - _Requirements: 3.3, 3.4_

- [x] 3.2 Implement atomic checkpoint save with write-to-temp-then-rename
  - Serialize training state (pattern tables, Adam optimizer state, metadata) using bincode
  - Write to temporary file first to prevent partial writes
  - Rename temporary file to final checkpoint path atomically
  - Log checkpoint operations with file size, save duration, and storage location
  - _Requirements: 3.1, 3.2, 3.10_

- [x] 3.3 Add CRC32 checksum calculation and verification
  - Calculate CRC32 checksum of serialized data before writing
  - Store checksum in checkpoint header
  - Verify checksum on load and return corruption error on mismatch
  - Use crc32fast library for efficient checksum computation
  - _Requirements: 3.7, 3.8_

- [x] 3.4 (P) Implement optional checkpoint compression
  - Add configurable compression using flate2 library
  - Set compression flag in checkpoint header when enabled
  - Decompress data during load when compression flag is set
  - Balance compression ratio against CPU overhead for save performance
  - _Requirements: 3.9_

- [x] 3.5 Implement checkpoint retention policy
  - Track checkpoint files in directory by timestamp
  - Implement configurable retention count (default: 5 checkpoints)
  - Automatically delete oldest checkpoints when count exceeds limit
  - Apply retention policy after each successful save operation
  - _Requirements: 3.5, 3.6_

- [x] 3.6 Create PyCheckpointManager PyO3 wrapper
  - Wrap CheckpointManager for Python access with constructor accepting directory, retention count, compression flag
  - Implement save method returning tuple of (path, file_size_bytes, save_duration_secs)
  - Implement load method returning training state as Python dictionary
  - Implement load_latest method for resuming from most recent checkpoint
  - Implement list_checkpoints method returning checkpoint metadata list
  - Add set_retention and set_compression methods for runtime configuration
  - Implement verify method for checking checkpoint integrity without full load
  - _Requirements: 3.1, 3.5, 3.7, 3.9, 3.10_

## Task 4. Enhanced Training Logger

- [x] 4.1 Extend TrainingLogger with configurable log levels
  - Add LogLevel enum supporting debug, info, warning, error levels
  - Implement log filtering based on configured level
  - Add constructor parameter for setting initial log level
  - Support runtime log level changes
  - _Requirements: 4.5_

- [x] 4.2 Implement JSON format output for machine-readable logging
  - Add json_output configuration flag to TrainingLogger
  - Format log entries as JSON objects with timestamp, level, event type, and data fields
  - Write JSON logs to timestamped files with format logs/training_YYYYMMDD_HHMMSS.log
  - Support both JSON and human-readable formats based on configuration
  - _Requirements: 4.4, 4.8_

- [x] 4.3 (P) Implement real-time and detailed statistics logging
  - Log real-time statistics every 100 games: stone difference, win rates, move count, elapsed time
  - Log detailed statistics every 10,000 games: evaluation distribution (mean, std, min, max), search depth stats, TT hit rate
  - Log checkpoint summaries with cumulative training statistics
  - Log estimated time remaining based on current throughput
  - _Requirements: 4.1, 4.2, 4.3, 4.6_

- [x] 4.4 (P) Add divergence warning detection and logging
  - Detect NaN, infinity, or out-of-range evaluation values
  - Emit immediate warning with pattern ID, stage, index, and anomalous value
  - Track divergence events for error summary
  - _Requirements: 4.7_

## Task 5. Training Engine Enhancements

- [x] 5.1 Implement training state machine with pause/resume capability
  - Add state enum (Idle, Training, Paused) with AtomicU8 representation for thread-safe access
  - Implement pause_flag as AtomicBool checked after each game batch
  - Ensure pause operation completes within 5 seconds by finishing current batch
  - Save checkpoint automatically on pause
  - _Requirements: 2.2, 2.3, 2.5_

- [x] 5.2 Implement progress callback mechanism for Python integration
  - Add callback channel using crossbeam-channel for progress updates
  - Invoke progress callbacks at configurable intervals (default: every 100 games)
  - Include games completed, average stone difference, win rate, and elapsed time in callbacks
  - Release GIL during Rust computation using py.allow_threads() to avoid blocking Python
  - _Requirements: 2.4_

- [x] 5.3 Implement training completion and interrupt handling
  - Detect training completion when target game count is reached
  - Save final checkpoint on completion with completion statistics
  - Handle interrupt signals (Ctrl+C) with graceful shutdown
  - Save checkpoint before exiting on interrupt
  - _Requirements: 2.1, 2.6, 2.7_

- [x] 5.4 Implement train_game and train_batch methods for PyO3 interface
  - Implement train_game method executing single self-play game with configurable time and epsilon
  - Return game statistics including stone difference, move count, and winner
  - Implement train_batch method executing multiple games with rayon parallel execution
  - Expose configure method for runtime adjustment of search time, epsilon, and logging verbosity
  - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7_

## Task 6. PyTrainingManager Implementation

- [x] 6.1 Create PyTrainingManager class with training control
  - Create PyTrainingManager class wrapping TrainingEngine with Mutex for thread-safe access
  - Implement start_training method accepting target games, checkpoint interval, callback interval, search time, and epsilon
  - Implement is_training_active method returning current training status
  - Implement get_state method returning string state ("idle", "training", "paused")
  - _Requirements: 2.1, 2.5_

- [x] 6.2 Implement pause and resume functionality
  - Implement pause_training method that signals pause and waits for current batch
  - Return game count when paused for tracking progress
  - Implement resume_training method that loads latest checkpoint and continues
  - _Requirements: 2.2, 2.3_

- [x] 6.3 (P) Implement progress callback and configuration
  - Implement set_progress_callback method accepting Python callable
  - Define callback signature as fn(games: int, stone_diff: float, win_rate: float, elapsed_secs: float)
  - Implement configure method for runtime parameter adjustment
  - _Requirements: 2.4, 1.7_

- [x] 6.4 Create PyTrainingResult class for completion data
  - Create PyTrainingResult class with fields for completion statistics
  - Include games_completed, final_stone_diff, win rates (black/white/draw), elapsed time, games per second, error count
  - Use pyo3(get) attributes for Python attribute access
  - _Requirements: 2.6_

## Task 7. Error Handling and Recovery System

- [ ] 7.1 Implement error recovery for search and game execution
  - Catch search errors during self-play and log with game context
  - Skip to next game without crashing on recoverable errors
  - Implement error counting per 10,000 game window
  - _Requirements: 9.1, 9.5_

- [ ] 7.2 Implement checkpoint save retry and failure handling
  - Retry checkpoint save once after 5-second delay on failure
  - Log warning and continue training if retry fails
  - Track checkpoint failures in error summary
  - _Requirements: 9.2_

- [ ] 7.3 Implement evaluation error detection and recovery
  - Detect NaN or infinite values in evaluation results
  - Reset affected pattern entries to neutral value (32768)
  - Log warning with position and pattern details for diagnosis
  - _Requirements: 9.3_

- [ ] 7.4 Implement worker thread panic handling
  - Catch panics from rayon worker threads using catch_unwind where appropriate
  - Log panic information without crashing main training process
  - Track panic counts in error monitoring
  - _Requirements: 9.4_

- [ ] 7.5 Implement error threshold monitoring and auto-pause
  - Track error rate per 10,000 game window
  - Pause training automatically if error rate exceeds 1%
  - Save checkpoint and report error pattern when pausing
  - _Requirements: 9.5, 9.6_

- [ ] 7.6 Implement checkpoint load error recovery
  - Detect corruption during checkpoint load via checksum mismatch
  - Offer option to start fresh or try previous checkpoint
  - Provide clear error messages with recovery suggestions
  - _Requirements: 9.7_

- [ ] 7.7 Implement watchdog for hung worker thread detection
  - Monitor worker thread activity with heartbeat mechanism
  - Detect threads that exceed timeout threshold without progress
  - Restart hung threads and log restart events
  - Note: Design review issue - specify timeout threshold (recommend 30s based on typical game duration)
  - _Requirements: 9.8_

## Task 8. PyStatisticsManager Implementation

- [ ] 8.1 Create PyStatisticsManager class aggregating monitoring components
  - Create PyStatisticsManager wrapping ConvergenceMonitor, BenchmarkRunner, MemoryMonitor, and TrainingLogger
  - Use Arc for shared access to monitoring components
  - Provide unified interface for all statistics and metrics
  - _Requirements: 4.1, 5.8, 6.8, 7.8_

- [ ] 8.2 (P) Implement convergence metrics access
  - Implement get_convergence_metrics returning dictionary with stone_diff_avg, eval_variance, pattern_coverage, stagnation_detected
  - Calculate rolling average of stone difference over last 10,000 games
  - Track evaluation stability via variance over time windows
  - Report pattern update coverage percentage
  - Detect stagnation when variance shows no decrease for 50,000 consecutive games
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 8.3 (P) Implement pattern update coverage monitoring
  - Count pattern table entries that have received updates
  - Warn when coverage falls below 90% after 500,000 games
  - Report average update count per pattern entry at checkpoints
  - Warn about undertrained patterns when average updates fall significantly below expected rate (233 at 1M games)
  - _Requirements: 5.3, 5.4, 5.6, 5.7_

- [ ] 8.4 (P) Implement memory usage reporting
  - Implement get_memory_report returning breakdown by component
  - Report total usage and per-component values (pattern tables, Adam state, TT, misc)
  - Express values in megabytes for readability
  - _Requirements: 7.1, 7.2, 7.4, 7.8_

- [ ] 8.5 (P) Implement benchmark execution and reporting
  - Implement run_benchmarks method with iteration count parameter
  - Measure and report games per second, TD update latency, checkpoint save/load duration, CPU utilization
  - Compare against target thresholds and identify bottlenecks when below targets
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ] 8.6 Implement statistics export and ETA calculation
  - Implement export_json method writing statistics to file
  - Implement get_eta method calculating estimated time remaining based on throughput
  - Support continuous profiling mode for regression detection
  - Implement get_statistics method returning current training metrics dictionary
  - _Requirements: 1.6, 4.6, 6.8_

## Task 9. Memory Management Enhancements

- [ ] 9.1 (P) Implement configurable memory budget enforcement
  - Add configurable total memory budget (default: 600 MB)
  - Track allocation by component during initialization
  - Automatically reduce transposition table size if budget would be exceeded
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 9.2 (P) Implement memory allocation tracking
  - Track memory allocation by component: pattern tables, Adam optimizer, transposition table
  - Provide memory report method showing current usage breakdown
  - Detect and log memory fragmentation with metrics
  - _Requirements: 7.4, 7.7_

- [ ] 9.3 (P) Implement sparse eligibility trace storage
  - Optimize eligibility trace memory with sparse storage approach
  - Minimize per-game memory overhead during TD updates
  - Ensure game history is deallocated promptly after TD update completion
  - _Requirements: 7.5, 7.6_

## Task 10. PyDebugModule Implementation

- [ ] 10.1 (P) Implement board and pattern visualization
  - Create PyDebugModule class wrapping EvaluationTable and ErrorTracker
  - Implement visualize_board method outputting ASCII representation of board state
  - Implement visualize_pattern method showing pattern indices and current weights
  - _Requirements: 8.1, 8.2_

- [ ] 10.2 (P) Implement trace and weight inspection
  - Implement inspect_trace method showing eligibility trace values for a position
  - Implement compare_checkpoints method diffing two checkpoints and reporting changed entries
  - Implement get_weight method for specific pattern entry access
  - _Requirements: 8.3, 8.6_

- [ ] 10.3 (P) Implement game replay with detailed logging
  - Implement replay_game method re-executing game from history with verbose logging
  - Log all TD updates with before/after weight values when debug mode enabled
  - _Requirements: 8.4, 8.5_

- [ ] 10.4 (P) Implement data export and anomaly detection
  - Implement export_csv method supporting training data export for external analysis
  - Implement detect_anomalies method identifying sudden large weight changes
  - Log position and pattern details for anomalous values
  - Implement get_error_summary method returning recent error information
  - _Requirements: 8.7, 8.8_

## Task 11. Python Training Scripts

- [ ] 11.1 Create train.py script with command-line interface
  - Accept command-line arguments for target games, checkpoint interval, search time, and epsilon schedule
  - Support --resume flag to continue from latest checkpoint
  - Output progress to both console and log file
  - Implement proper signal handling for graceful shutdown on Ctrl+C
  - _Requirements: 10.1, 10.2, 10.3, 10.8_

- [ ] 11.2 (P) Create monitor.py real-time dashboard script
  - Display real-time training statistics in formatted terminal dashboard
  - Support plotting training curves showing stone difference and win rate over games
  - Connect to training via progress callbacks
  - _Requirements: 10.4, 10.5_

- [ ] 11.3 (P) Create evaluate.py model evaluation script
  - Test trained model against random player and report win rate
  - Test trained model against simple heuristic (corner/edge priority) and report win rate
  - Accept checkpoint path as command-line argument
  - _Requirements: 10.6, 10.7_

## Task 12. Build Scripts and Deployment Configuration

- [ ] 12.1 (P) Create build scripts with common targets
  - Create Makefile or build script providing targets: build, test, bench, release, install
  - Add development mode build for rapid iteration
  - Add release mode build with all optimizations enabled
  - _Requirements: 11.6_

- [ ] 12.2 (P) Create systemd service file for background training
  - Define systemd service unit for running training as background service
  - Configure automatic restart on failure
  - Set appropriate resource limits and working directory
  - _Requirements: 11.7_

- [ ] 12.3 (P) Document OCI Ampere A1 deployment setup
  - Create deployment instructions for OCI Ampere A1 environment
  - Include Python environment setup and dependency installation
  - Document ARM64-specific build steps and optimizations
  - _Requirements: 11.8_

## Task 13. Integration Tests

- [ ] 13.1 Implement Python-to-Rust round-trip tests
  - Test all PyO3 methods return valid results from Python calls
  - Verify data type conversions work correctly (board arrays, statistics dictionaries)
  - Test error handling returns appropriate Python exceptions
  - _Requirements: 12.1_

- [ ] 13.2 Implement checkpoint state preservation tests
  - Save checkpoint, load checkpoint, verify all training state is identical
  - Test compression and non-compression modes
  - Verify checksum validation detects corruption
  - _Requirements: 12.2_

- [ ] 13.3 Implement parallel training correctness tests
  - Execute parallel training with multiple threads
  - Verify no data races using MIRI or thread sanitizer where applicable
  - Confirm game statistics are consistent across runs
  - _Requirements: 12.3_

- [ ] 13.4 Implement memory budget tests
  - Monitor memory usage during extended test runs
  - Verify total usage stays within 600 MB budget
  - Test automatic TT size reduction when approaching limit
  - _Requirements: 12.4_

- [ ] 13.5 Implement graceful shutdown tests
  - Send interrupt signal during training
  - Verify checkpoint is saved before exit
  - Verify training can resume from saved state
  - _Requirements: 12.5_

- [ ] 13.6 Implement convergence metrics tests
  - Run 1,000 game training sample
  - Verify convergence metrics are computed and reported correctly
  - Test stagnation detection with controlled input
  - _Requirements: 12.6_

- [ ] 13.7 Implement performance threshold tests
  - Run benchmarks and verify minimum thresholds are met
  - Test game throughput >= 4.6 games/second
  - Test checkpoint operations complete within time limits
  - Verify CI completion within 10 minutes for full suite
  - _Requirements: 12.7, 12.8_
