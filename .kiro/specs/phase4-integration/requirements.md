# Requirements Document

## Introduction

Phase 4 (Integration and Optimization) serves as the bridge between the Rust core (Phase 1-3) and the Python training infrastructure. This phase implements Python bindings using PyO3, enhances checkpoint management for production training, adds comprehensive logging and monitoring capabilities, and provides debugging and optimization utilities. The goal is to prepare the system for Phase 5: 1 million game training execution on OCI Ampere A1.

## Requirements

### Requirement 1: Python Bindings (PyO3)

**Objective:** As a training operator, I want to control the Othello AI training from Python scripts, so that I can orchestrate training with flexible parameters, monitor progress, and integrate with Python-based analysis tools.

#### Acceptance Criteria

1. The PyO3 module shall expose a PyEvaluator class that wraps the Rust Evaluator with pattern table access.
2. When the Python client calls the evaluate method with a 64-element board array, the PyO3 module shall return the evaluation score as a float.
3. When the Python client calls the train_game method with time_ms and epsilon parameters, the PyO3 module shall execute one self-play game and return statistics (stone difference, move count, winner).
4. When the Python client calls the train_batch method with game count and epsilon, the PyO3 module shall execute multiple games with parallel execution using rayon.
5. The PyO3 module shall expose checkpoint save and load methods that persist and restore complete training state.
6. When the Python client calls get_statistics, the PyO3 module shall return current training metrics (games played, average stone difference, win rates, elapsed time).
7. The PyO3 module shall expose a configure method allowing runtime adjustment of search time, epsilon, and logging verbosity.
8. If invalid parameters are provided to any PyO3 method, the module shall raise a Python exception with descriptive error message.
9. The PyO3 module shall support NumPy arrays for board representation to enable efficient data interchange.
10. The PyO3 module shall expose pattern table weights for external analysis and visualization through a get_weights method.

### Requirement 2: Training Session Management

**Objective:** As a training operator, I want to manage long-running training sessions from Python, so that I can start, pause, resume, and monitor 1 million game training runs reliably.

#### Acceptance Criteria

1. When the Python client calls start_training with target_games parameter, the PyO3 module shall begin training from the current state toward the target.
2. When the Python client calls pause_training, the PyO3 module shall save a checkpoint and halt game execution within 5 seconds.
3. When the Python client calls resume_training, the PyO3 module shall load the latest checkpoint and continue from the saved state.
4. While training is active, the PyO3 module shall provide progress callbacks to Python at configurable intervals (default: every 100 games).
5. The PyO3 module shall expose an is_training_active method returning the current training status.
6. When training completes the target game count, the PyO3 module shall save a final checkpoint and return completion statistics.
7. If the training process receives an interrupt signal, the PyO3 module shall invoke graceful shutdown and save checkpoint before exiting.

### Requirement 3: Enhanced Checkpoint Mechanism

**Objective:** As a training operator, I want robust checkpoint management, so that I can recover from failures and manage checkpoint storage efficiently during multi-day training runs.

#### Acceptance Criteria

1. The Checkpoint Manager shall save checkpoints atomically using write-to-temp-then-rename strategy to prevent corruption.
2. When checkpoint save is requested, the Checkpoint Manager shall write pattern tables (~57 MB), Adam optimizer state (~228 MB), and training metadata.
3. The Checkpoint Manager shall include version information and a magic header for format validation during load.
4. When loading a checkpoint with version mismatch, the Checkpoint Manager shall return an error with version details rather than attempting to load.
5. The Checkpoint Manager shall support configurable checkpoint retention (keep last N checkpoints, default: 5).
6. When checkpoint count exceeds retention limit, the Checkpoint Manager shall delete the oldest checkpoint files automatically.
7. The Checkpoint Manager shall calculate and store a checksum (CRC32) for data integrity verification.
8. When loading a checkpoint with checksum mismatch, the Checkpoint Manager shall return a corruption error.
9. The Checkpoint Manager shall support checkpoint compression (optional, configurable) to reduce storage requirements.
10. The Checkpoint Manager shall log checkpoint operations with file size, save duration, and storage location.

### Requirement 4: Training Statistics and Logging

**Objective:** As a training operator, I want comprehensive logging of training progress, so that I can monitor convergence, detect issues, and analyze training dynamics.

#### Acceptance Criteria

1. The Training Logger shall output real-time statistics every 100 games: average stone difference, black/white/draw rates, average move count, elapsed time.
2. The Training Logger shall output detailed statistics every 10,000 games: evaluation distribution (mean, std, min, max), search depth statistics, transposition table hit rate.
3. When a checkpoint is saved, the Training Logger shall output a complete training summary with cumulative statistics.
4. The Training Logger shall write logs to timestamped files with format logs/training_YYYYMMDD_HHMMSS.log.
5. The Training Logger shall support configurable log levels (debug, info, warning, error) for different output verbosity.
6. The Training Logger shall output estimated time remaining based on current throughput and remaining games.
7. If evaluation values diverge (NaN, infinity, or values outside valid range), the Training Logger shall emit an immediate warning.
8. The Training Logger shall support JSON-format output for machine-readable log parsing.

### Requirement 5: Convergence Monitoring

**Objective:** As a training operator, I want automated convergence monitoring, so that I can detect training problems early and ensure learning is progressing as expected.

#### Acceptance Criteria

1. The Convergence Monitor shall track rolling average of stone difference over the last 10,000 games.
2. The Convergence Monitor shall track evaluation stability (variance of evaluations over time windows).
3. The Convergence Monitor shall count pattern table entries that have received updates and report coverage percentage.
4. When pattern update coverage falls below 90% after 500,000 games, the Convergence Monitor shall emit a warning.
5. When stone difference variance shows no decrease for 50,000 consecutive games, the Convergence Monitor shall emit a stagnation warning.
6. The Convergence Monitor shall report average update count per pattern entry at each checkpoint.
7. When average updates per entry falls significantly below expected rate (233 updates at 1M games), the Convergence Monitor shall warn about potential undertrained patterns.
8. The Convergence Monitor shall expose metrics via Python API for external monitoring and visualization.

### Requirement 6: Performance Benchmarking Utilities

**Objective:** As a developer, I want performance benchmarking tools, so that I can measure and optimize training throughput to meet the 50-60 hour target for 1 million games.

#### Acceptance Criteria

1. The Benchmark Suite shall measure game throughput in games per second, targeting minimum 4.6 games/second.
2. The Benchmark Suite shall measure TD update latency per game, targeting under 10ms.
3. The Benchmark Suite shall measure checkpoint save duration, targeting under 30 seconds for full state.
4. The Benchmark Suite shall measure checkpoint load duration, targeting under 30 seconds.
5. The Benchmark Suite shall measure memory usage breakdown by component (pattern tables, Adam state, TT, misc).
6. The Benchmark Suite shall measure CPU utilization across all 4 cores, targeting 80% or higher.
7. When benchmark results fall below targets, the Benchmark Suite shall report specific bottleneck identification.
8. The Benchmark Suite shall support continuous profiling mode for identifying performance regressions.

### Requirement 7: Memory Optimization

**Objective:** As a developer, I want memory optimization utilities, so that I can ensure the system operates within the 600 MB budget on OCI Ampere A1.

#### Acceptance Criteria

1. The Memory Manager shall enforce a configurable total memory budget (default: 600 MB).
2. The Memory Manager shall track allocation by component: pattern tables (~57 MB), Adam optimizer (~228 MB), transposition table (128-256 MB).
3. If total memory allocation would exceed budget, the Memory Manager shall reduce transposition table size automatically.
4. The Memory Manager shall provide a memory report method showing current usage by component.
5. The Memory Manager shall implement sparse eligibility trace storage to minimize per-game memory overhead.
6. The Memory Manager shall ensure game history is deallocated promptly after TD update completion.
7. When memory fragmentation is detected, the Memory Manager shall log a warning with fragmentation metrics.
8. The Memory Manager shall expose memory metrics via Python API for external monitoring.

### Requirement 8: Debugging Utilities

**Objective:** As a developer, I want debugging utilities, so that I can diagnose training issues and verify correctness of learning algorithms.

#### Acceptance Criteria

1. The Debug Module shall provide a board visualization function outputting ASCII representation of board state.
2. The Debug Module shall provide a pattern visualization function showing pattern indices and their current weights.
3. The Debug Module shall provide a trace inspection function showing current eligibility trace values for a given position.
4. When debug mode is enabled, the Debug Module shall log all TD updates with before/after weight values.
5. The Debug Module shall provide a replay function that re-executes a game from history with detailed logging.
6. The Debug Module shall provide a weight diff function comparing two checkpoints and reporting changed entries.
7. The Debug Module shall support exporting training data to CSV format for external analysis.
8. If anomalous weight values are detected (sudden large changes), the Debug Module shall log the position and pattern details.

### Requirement 9: Error Handling and Recovery

**Objective:** As a training operator, I want robust error handling, so that multi-day training runs can recover from transient failures without human intervention.

#### Acceptance Criteria

1. If a search error occurs during self-play, the Training Engine shall log the error and skip to the next game without crashing.
2. If checkpoint save fails, the Training Engine shall retry once after 5 seconds, then log warning and continue training.
3. If evaluation produces NaN or infinite values, the Training Engine shall reset affected pattern entries to neutral (32768) and log warning.
4. The Training Engine shall catch and log panics from worker threads without crashing the main training process.
5. The Training Engine shall track error counts per 10,000 game window and pause training if error rate exceeds 1%.
6. When training pauses due to high error rate, the Training Engine shall save checkpoint and report error pattern for diagnosis.
7. If checkpoint load fails due to corruption, the Training Engine shall offer option to start fresh or try previous checkpoint.
8. The Training Engine shall implement watchdog functionality that detects hung worker threads and restarts them.

### Requirement 10: Python Training Scripts

**Objective:** As a training operator, I want ready-to-use Python training scripts, so that I can launch and manage 1 million game training with minimal setup.

#### Acceptance Criteria

1. The train.py script shall accept command-line arguments for target games, checkpoint interval, search time, and epsilon schedule.
2. The train.py script shall support --resume flag to continue from latest checkpoint.
3. The train.py script shall output progress to both console and log file.
4. The monitor.py script shall display real-time training statistics in a formatted dashboard.
5. The monitor.py script shall support plotting training curves (stone difference, win rate over games).
6. The evaluate.py script shall test trained model against random player and report win rate.
7. The evaluate.py script shall test trained model against simple heuristic (corner/edge only) and report win rate.
8. The Python scripts shall include proper signal handling for graceful shutdown on Ctrl+C.

### Requirement 11: Build and Deployment Configuration

**Objective:** As a developer, I want proper build configuration, so that the PyO3 module builds correctly and achieves optimal performance on ARM64.

#### Acceptance Criteria

1. The Cargo.toml shall include PyO3 dependency with extension-module feature for Python binding.
2. The pyproject.toml shall configure maturin for building the Python wheel.
3. The build configuration shall enable release optimizations: opt-level=3, LTO, codegen-units=1.
4. The build configuration shall support cross-compilation for ARM64 target (aarch64-unknown-linux-gnu).
5. When building for ARM64, the build shall enable NEON SIMD optimizations where applicable.
6. The Makefile or build script shall provide targets for: build, test, bench, release, and install.
7. The deployment configuration shall include systemd service file for running training as background service.
8. The deployment configuration shall include instructions for OCI Ampere A1 setup.

### Requirement 12: Integration Testing

**Objective:** As a developer, I want comprehensive integration tests, so that I can verify the complete training pipeline works correctly end-to-end.

#### Acceptance Criteria

1. The integration test suite shall verify Python-to-Rust round-trip for all PyO3 methods.
2. The integration test suite shall verify checkpoint save/load preserves training state exactly.
3. The integration test suite shall verify parallel training executes without data races.
4. The integration test suite shall verify memory usage stays within budget during extended runs.
5. The integration test suite shall verify graceful shutdown saves checkpoint correctly.
6. The integration test suite shall verify convergence metrics are computed correctly over 1,000 game sample.
7. The integration test suite shall verify performance benchmarks meet minimum thresholds.
8. When running on CI, the integration tests shall complete within 10 minutes.
