# Lint/Type-Check Bypass Policy

Pre-commit hooks block common bypass patterns. This document defines approved exceptions.

## Approved Exceptions

### Python

| Pattern | Allowed In | Reason |
|---------|------------|--------|
| `Any` type | `tests/` only | **Blocked in production code** - use `Protocol` or concrete types |
| `# type: ignore` | `tests/` only | Test mocking/patching |
| `cast()` | - | **Not allowed** - refactor types instead |

### Rust

| Pattern | Allowed In | Reason |
|---------|------------|--------|
| `unsafe {}` | `arm64.rs`, `src/python/*.rs`, `evaluator.rs` | SIMD intrinsics, FFI, performance-critical |
| `#[allow(clippy::too_many_arguments)]` | Existing public API | Backward compatibility |
| `.unwrap()` / `.expect()` | Doctests (`///`), `#[cfg(test)]`, `tests/`, `benches/`, `examples/` | Documentation examples, tests |
| Other `#[allow(...)]` | - | **Not allowed** |

## Enforcement Layers

### Layer 1: Pre-commit hooks (`.pre-commit-config.yaml`)
- Pattern-based detection using pygrep
- Blocks obvious bypass attempts before commit
- Cannot distinguish doctest from production code

### Layer 2: Clippy lints (`Cargo.toml`)
```toml
[lints.clippy]
unwrap_used = "warn"  # Enforced in non-test code only
expect_used = "warn"
```
- Respects `#[cfg(test)]` boundaries
- Properly ignores doctests
- More accurate than regex patterns

### Layer 3: CI verification
- Runs full `cargo clippy` with `-D warnings`
- Final defense against `--no-verify` commits

## How to Request an Exception

1. Open a PR with the bypass and clear justification
2. Get approval from at least one reviewer
3. Add file to `exclude` pattern in `.pre-commit-config.yaml` if persistent

## Rationale

- `type: ignore` hides real bugs - fix the types instead
- `noqa` bypasses are lazy - fix the lint issue
- `#[allow(unused)]` hides dead code - remove it
- `unsafe` must be audited - only in designated modules
- `unwrap()`/`expect()` panic in production - use `?` or proper error handling
