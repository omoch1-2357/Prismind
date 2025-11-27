# Requirements Document

## Introduction

This document defines the requirements for Phase 3 of the Othello AI project: TD(lambda)-Leaf Reinforcement Learning System. Building upon Phase 1 (BitBoard, Pattern, Evaluator) and Phase 2 (Search algorithms: Negamax, AlphaBeta, MTD(f), Transposition Table, Move Ordering, Iterative Deepening), Phase 3 implements the self-play learning system that trains the evaluation function through 1 million games of self-play.

The learning system uses TD(lambda)-Leaf with Eligibility Trace for temporal difference learning, Adam optimizer for stable gradient updates, and epsilon-greedy exploration for diverse game experiences. The target environment is OCI Always Free (ARM64, 4 cores, 24GB RAM), with an estimated training time of 50-60 hours.

## Requirements

### Requirement 1: TD(lambda)-Leaf Learning Algorithm

**Objective:** As a machine learning engineer, I want a TD(lambda)-Leaf learning algorithm that updates pattern evaluation weights based on game outcomes and intermediate evaluations, so that the AI learns optimal move evaluation through self-play.

#### Acceptance Criteria

1. The Learning System shall implement TD(lambda)-Leaf learning with lambda=0.3 as the trace decay parameter.
2. When a game ends, the Learning System shall perform backward updates from the final position to the initial position.
3. The Learning System shall compute TD error as the difference between target value and current evaluation.
4. When computing the target value at time step t, the Learning System shall use the formula: target = lambda * final_score + (1 - lambda) * next_value for non-terminal positions.
5. When computing the target value at the final position, the Learning System shall use the actual game result (stone difference) as the target.
6. The Learning System shall update all 56 pattern instances (14 patterns x 4 rotations) for each board position.
7. The Learning System shall perform approximately 33.6 billion weight updates over 1 million games (56 patterns x 60 moves x 1M games).
8. The Learning System shall account for the side-to-move when computing evaluation values and targets (negating values for White's perspective).

### Requirement 2: Eligibility Trace Management

**Objective:** As a machine learning engineer, I want an Eligibility Trace system that tracks the contribution of each pattern entry to the learning signal, so that credit assignment is properly distributed across visited states.

#### Acceptance Criteria

1. The Learning System shall maintain an Eligibility Trace for each pattern table entry that was visited during a game.
2. When a pattern entry is visited, the Learning System shall increment its trace value by 1.0.
3. When processing each time step in reverse order, the Learning System shall decay all trace values by multiplying with lambda (0.3).
4. The Learning System shall compute the gradient for each entry as: gradient = td_error * eligibility_trace.
5. When a new game starts, the Learning System shall reset all Eligibility Traces to zero.
6. The Learning System shall use a sparse data structure (HashMap) for traces to minimize memory usage.

### Requirement 3: Adam Optimizer

**Objective:** As a machine learning engineer, I want an Adam optimizer that provides stable and adaptive learning rate adjustments, so that weight updates converge reliably without manual learning rate tuning.

#### Acceptance Criteria

1. The Learning System shall implement the Adam optimizer with learning rate alpha=0.025.
2. The Learning System shall use beta1=0.9 for the first moment (gradient moving average) decay rate.
3. The Learning System shall use beta2=0.999 for the second moment (squared gradient moving average) decay rate.
4. The Learning System shall use epsilon=1e-8 for numerical stability in division.
5. The Learning System shall maintain first moment (m) vectors for all pattern table entries, requiring approximately 114 MB storage.
6. The Learning System shall maintain second moment (v) vectors for all pattern table entries, requiring approximately 114 MB storage.
7. The Learning System shall maintain a global timestep counter (t) for bias correction.
8. When updating a parameter, the Learning System shall apply bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t).
9. When updating a parameter, the Learning System shall compute the update as: delta = alpha * m_hat / (sqrt(v_hat) + epsilon).
10. The Learning System shall initialize all first and second moment values to 0.0.

### Requirement 4: Self-Play Game Engine

**Objective:** As a system operator, I want a self-play game engine that generates diverse training data through controlled exploration, so that the AI learns from a wide variety of game situations.

#### Acceptance Criteria

1. The Self-Play Engine shall play complete games from the initial Othello position to game end.
2. The Self-Play Engine shall use the Phase 2 Search API (Search::search()) with 15ms time limit per move for move selection.
3. When the game number is between 0 and 300,000, the Self-Play Engine shall use epsilon=0.15 (15% random moves).
4. When the game number is between 300,000 and 700,000, the Self-Play Engine shall use epsilon=0.05 (5% random moves).
5. When the game number is between 700,000 and 1,000,000, the Self-Play Engine shall use epsilon=0.0 (always best move).
6. When a random move is selected, the Self-Play Engine shall use the current board's static evaluation as the leaf value.
7. The Self-Play Engine shall record for each move: board state, leaf evaluation value, pattern indices (56 values), and stage number.
8. The Self-Play Engine shall handle pass moves correctly according to Othello rules.
9. When a game ends, the Self-Play Engine shall compute the final stone difference as the game result.
10. The Self-Play Engine shall target completion of 1 million games within 50-60 hours on 4-core ARM64.

### Requirement 5: Game History Recording

**Objective:** As a machine learning engineer, I want comprehensive game history recording that captures all information needed for TD learning, so that weight updates can be computed correctly after each game.

#### Acceptance Criteria

1. The Game History module shall record the board state (BitBoard) for each move in the game.
2. The Game History module shall record the leaf evaluation value from MTD(f) search for each move.
3. The Game History module shall record all 56 pattern indices for each move.
4. The Game History module shall record the game stage (0-29) for each move.
5. The Game History module shall support games with up to 60 moves.
6. The Game History module shall store data in memory-efficient format suitable for backward iteration.
7. When TD update is complete, the Game History module shall allow deallocation of game-specific data.

### Requirement 6: Checkpoint Management

**Objective:** As a system operator, I want a checkpoint management system that saves training state periodically, so that training can be resumed from any checkpoint if interrupted.

#### Acceptance Criteria

1. The Checkpoint Manager shall save checkpoints every 100,000 games (10 checkpoints total for 1M games).
2. The Checkpoint Manager shall save all pattern table weights (14 patterns x 30 stages x entries, approximately 57 MB).
3. The Checkpoint Manager shall save all Adam optimizer state (m and v vectors, approximately 228 MB).
4. The Checkpoint Manager shall save the Adam timestep counter (t).
5. The Checkpoint Manager shall save metadata including: current game count, elapsed training time, and statistics summary.
6. The Checkpoint Manager shall use the filename format: checkpoint_NNNNNN.bin where NNNNNN is the game count (6 digits, zero-padded).
7. When loading a checkpoint, the Checkpoint Manager shall restore all pattern weights, Adam state, and metadata.
8. The Checkpoint Manager shall verify checkpoint integrity using a header signature (e.g., "OTHELLO_AI_CHECKPOINT_V1").
9. If checkpoint loading fails, the Checkpoint Manager shall report an error and allow fresh start.
10. The Checkpoint Manager shall support saving checkpoint_000000.bin as the initial state before training.

### Requirement 7: Training Statistics Logging

**Objective:** As a system operator, I want comprehensive training statistics logging, so that I can monitor training progress and detect potential issues.

#### Acceptance Criteria

1. The Logging System shall output real-time statistics every 100 games.
2. When outputting real-time statistics, the Logging System shall include: average stone difference, black win rate, white win rate, draw rate, average move count, and elapsed time.
3. The Logging System shall output detailed statistics every 10,000 games.
4. When outputting detailed statistics, the Logging System shall include: evaluation value distribution (mean, standard deviation, min, max), average search depth (by game phase), average search time per move, and transposition table hit rate.
5. When saving a checkpoint, the Logging System shall output a complete statistics summary.
6. The Logging System shall write logs to files with format: logs/training_YYYYMMDD_HHMMSS.log.
7. The Logging System shall output progress reports with estimated time remaining.
8. If evaluation values diverge (extreme values or NaN), the Logging System shall output a warning.

### Requirement 8: Memory Management

**Objective:** As a system operator, I want efficient memory management that stays within the OCI Always Free limits, so that training can complete without memory-related failures.

#### Acceptance Criteria

1. The Learning System shall use no more than 600 MB total memory.
2. The Pattern Table shall use approximately 57 MB (14 patterns x 30 stages x ~14.4M entries x 2 bytes).
3. The Adam Optimizer shall use approximately 228 MB (m: 114 MB + v: 114 MB, using f32).
4. The Transposition Table shall use 128-256 MB (configurable, shared with Phase 2).
5. While training, the Learning System shall release game history memory after each TD update.
6. The Learning System shall use sparse data structures for Eligibility Traces to minimize per-game memory.
7. If memory allocation fails, the Learning System shall reduce transposition table size and retry.

### Requirement 9: Integration with Phase 2 Search

**Objective:** As a developer, I want seamless integration with the Phase 2 search system, so that TD-Leaf learning can leverage the existing high-performance search implementation.

#### Acceptance Criteria

1. The Learning System shall use the Phase 2 Search::search() API for move selection.
2. The Learning System shall reuse the Phase 2 TranspositionTable instance across games.
3. The Learning System shall share the Evaluator instance with the search system.
4. When the Evaluator weights are updated, the Learning System shall ensure consistency between learning and search.
5. The Learning System shall use Phase 2's SearchResult to obtain leaf evaluation values.
6. The Learning System shall handle Phase 2 SearchError appropriately with fallback behavior.
7. The Learning System shall use the Phase 2 make_move and check_game_state functions for game progression.

### Requirement 10: Training Convergence Monitoring

**Objective:** As a machine learning engineer, I want training convergence monitoring that tracks learning progress, so that I can verify the model is improving over time.

#### Acceptance Criteria

1. The Monitoring System shall track average stone difference trends across training.
2. The Monitoring System shall track evaluation value stability (decreasing variance over time).
3. The Monitoring System shall track the percentage of pattern entries updated (target: 200+ updates per entry average).
4. When win rate against random drops below 95%, the Monitoring System shall output a warning.
5. The Monitoring System shall output convergence metrics every 100,000 games.
6. The Monitoring System shall detect and report if learning has stagnated (no improvement for 50,000+ games).

### Requirement 11: Score Representation and Conversion

**Objective:** As a developer, I want consistent score representation between learning and evaluation systems, so that weight updates are properly scaled and bounded.

#### Acceptance Criteria

1. The Learning System shall use u16 representation for pattern table entries (range 0-65535).
2. The Learning System shall use CENTER=32768 as the zero stone-difference value.
3. The Learning System shall use SCALE=256.0 for converting between stone difference and u16.
4. When converting u16 to stone difference, the Learning System shall compute: score = (value - 32768) / 256.0.
5. When converting stone difference to u16, the Learning System shall compute: value = clamp(score * 256.0 + 32768, 0, 65535).
6. When updating weights, the Learning System shall clamp values to the valid u16 range [0, 65535].
7. The Learning System shall initialize all pattern table entries to 32768 (stone difference = 0).

### Requirement 12: Error Handling and Recovery

**Objective:** As a system operator, I want robust error handling and recovery mechanisms, so that training can continue despite transient failures.

#### Acceptance Criteria

1. If a search operation fails, the Learning System shall log the error and skip to the next game.
2. If a checkpoint save fails, the Learning System shall retry once and log a warning if still failing.
3. If evaluation values become NaN or infinite, the Learning System shall reset the affected entries to 32768 and log a warning.
4. The Learning System shall catch and log all panics without crashing the entire training process.
5. When resuming from checkpoint, the Learning System shall validate that the loaded state is consistent.
6. If more than 1% of games fail in a 10,000 game window, the Learning System shall pause and report the error pattern.
7. The Learning System shall implement graceful shutdown on SIGINT/SIGTERM, saving a checkpoint before exit.

### Requirement 13: Performance Requirements

**Objective:** As a system operator, I want the learning system to meet performance targets, so that 1 million games complete within the expected timeframe.

#### Acceptance Criteria

1. The Learning System shall complete 1 million games within 60 hours on 4-core ARM64.
2. The Learning System shall achieve an average game throughput of at least 4.6 games per second.
3. The Learning System shall complete TD updates for a single game within 10ms.
4. The Learning System shall complete checkpoint saves within 30 seconds.
5. The Learning System shall not block the main training loop during logging operations.
6. The Learning System shall achieve 80% or higher CPU utilization during training.
7. The Search operations shall maintain the Phase 2 performance target of 15ms per move average.
8. The Learning System shall utilize all 4 CPU cores through parallel game execution.
