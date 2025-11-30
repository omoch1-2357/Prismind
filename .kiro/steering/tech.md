# Technology Stack

## Architecture

Modular Rust library with distinct phases: core engine (Phase 1), search algorithms (Phase 2), and reinforcement learning (Phase 3). Designed for high-throughput training on cloud infrastructure.

## Core Technologies

- **Language**: Rust (Edition 2024)
- **Platform**: Cross-platform with ARM64 (aarch64) optimization focus
- **Target Environment**: Oracle Cloud Infrastructure Ampere A1 (24GB RAM, 4 OCPU)

## Key Libraries

| Library | Purpose |
|---------|---------|
| `serde` + `bincode` | Checkpoint serialization |
| `rayon` | Parallel game execution during training |
| `thiserror` | Structured error handling |
| `criterion` | Performance benchmarking |

## Development Standards

### Type Safety
- Rust strict mode with all compiler warnings as errors
- `#[repr(C)]` for memory layout guarantees on critical structures
- Exhaustive pattern matching

### Code Quality
- `cargo fmt` for formatting
- `cargo clippy -- -D warnings` (warnings as errors)
- Comprehensive doc comments with examples

### Testing
- Unit tests within each module
- Integration tests in `tests/` directory
- Performance benchmarks in `benches/`
- Target coverage: core algorithms fully tested

## Development Environment

### Required Tools
- Rust stable toolchain (Edition 2024)
- Optional: ARM64 cross-compilation target

### Common Commands
```bash
# Dev build: cargo build
# Release build: cargo build --release
# Test: cargo test --all-features
# Bench: cargo bench
# Lint: cargo clippy -- -D warnings
```

## Key Technical Decisions

- **BitBoard representation**: Two u64 values + metadata byte for minimal memory (24 bytes)
- **Pattern scoring**: u16 fixed-point representation (0-65535 maps to -128 to +128 stone difference)
- **Evaluation tables**: 30 stages x 14 patterns with 4-rotation symmetry (shared tables via color swap)
- **Parallel training**: Read-locked evaluator for game execution, write-locked for TD updates
- **Checkpoint format**: Binary with magic header, version control, and atomic save via rename

---
_Document standards and patterns, not every dependency_
