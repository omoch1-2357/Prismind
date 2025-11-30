# Project Structure

## Organization Philosophy

**Modular by feature phase**: The codebase is organized around development phases (foundation, search, learning), with each phase building upon previous capabilities. Core modules live in `src/`, with submodules for complex features.

## Directory Patterns

### Source Code (`src/`)
**Purpose**: Rust library and binary source
**Pattern**: Top-level modules for core concepts, subdirectories for complex subsystems

| Module | Responsibility |
|--------|---------------|
| `board.rs` | BitBoard representation, move generation, game state |
| `pattern.rs` | Pattern definitions, CSV loading, index extraction |
| `evaluator.rs` | Evaluation function, stage management, score conversion |
| `search.rs` | Negamax, AlphaBeta, MTD(f), transposition table |
| `learning/` | TD-Leaf, Adam optimizer, self-play, checkpoints |
| `arm64.rs` | Platform-specific SIMD optimizations (conditional compile) |

### Learning Submodule (`src/learning/`)
**Purpose**: Reinforcement learning system components
**Pattern**: One file per algorithm/feature, re-exported through `mod.rs`

Key components: `td_learner.rs`, `adam.rs`, `eligibility_trace.rs`, `self_play.rs`, `checkpoint.rs`, `training_engine.rs`

### Tests (`tests/`)
**Purpose**: Integration and end-to-end tests
**Pattern**: `*_test.rs` or `*_integration.rs` naming

### Benchmarks (`benches/`)
**Purpose**: Criterion performance benchmarks
**Pattern**: `*_bench.rs` naming, one benchmark per critical operation

### Documentation (`docs/`)
**Purpose**: Design documents and technical specifications
**Pattern**: Markdown files for architecture decisions and API guides

## Naming Conventions

- **Files**: snake_case (`td_learner.rs`, `game_history.rs`)
- **Types**: PascalCase (`BitBoard`, `SearchResult`, `TDLearner`)
- **Functions**: snake_case (`make_move`, `extract_index`, `play_game`)
- **Constants**: SCREAMING_SNAKE_CASE (`CHECKPOINT_MAGIC`, `DEFAULT_SEARCH_TIME_MS`)

## Import Organization

```rust
// Standard library
use std::sync::Arc;

// External crates
use thiserror::Error;
use serde::{Serialize, Deserialize};

// Internal modules (crate-level)
use crate::board::{BitBoard, Color};
use crate::evaluator::Evaluator;

// Submodule imports
use super::adam::AdamOptimizer;
```

**Module Re-exports**: Public API types are re-exported from `lib.rs` for convenient access.

## Code Organization Principles

- **Phase boundaries**: Each development phase has corresponding spec documents in `.kiro/specs/`
- **Error propagation**: Custom error types per module, converted via `From` trait
- **Conditional compilation**: ARM64 optimizations gated by `#[cfg(target_arch = "aarch64")]`
- **Documentation**: Every public item has doc comments; modules have `//!` header documentation

---
_Document patterns, not file trees. New files following patterns should not require updates_
