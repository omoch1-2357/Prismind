# ARM64 Performance Benchmarks

## Overview

This document describes the ARM64 benchmark workflow for Task 14.1 (Core Operations Final Benchmark).

## Purpose

Since the development environment is x86-64, we use GitHub Actions with ARM64 runners to measure actual ARM64-specific optimizations:
- ARM64 REV instruction effect (in `rotate_180()`)
- ARM64 CLZ/CTZ instructions (in `legal_moves()`)
- ARM64 NEON SIMD optimizations

## Workflow: `.github/workflows/arm64-bench.yml`

### Trigger Conditions

The workflow runs on:
- **Pull Requests** to `main` or `phase1-foundation` branches
- Changes to `src/**`, `benches/**`, `Cargo.toml`, or the workflow file itself
- **Manual trigger** via GitHub Actions UI

### Jobs

#### 1. `arm64-core-benchmarks` (Primary)

**Runner**: `macos-latest` (Apple Silicon M1/M2/M3 - ARM64)

**Note**: We use macOS runners for broader compatibility. They're available on free GitHub accounts and provide genuine ARM64 performance measurements.

**Benchmarks** (Task 14.1 requirements):
1. `legal_moves()` - Target: 500ns average
2. `make_move()` - Target: 1.5μs average
3. `rotate_180()` - Target: 200ns average (ARM64 REV instruction)

**Measurements**:
- Average execution time (mean)
- Standard deviation
- p99 percentile (99th percentile)
- 1000 iterations per benchmark

**Outputs**:
- Criterion HTML reports uploaded as artifacts (30-day retention)
- Benchmark summary posted as PR comment
- Architecture verification logs

#### 2. `x86-64-comparison` (Reference)

**Runner**: `ubuntu-latest` (x86-64)

Runs the same benchmarks on x86-64 for comparison purposes.

## Usage

### Running Locally (x86-64)

```bash
# Run specific Task 14.1 benchmarks
cargo bench --bench legal_moves_bench -- legal_moves_1000_iters
cargo bench --bench make_move_bench -- make_move_1000_iters
cargo bench --bench rotation_bench -- rotate_180_1000_iters
```

**Note**: Local x86-64 results are for functional verification only. ARM64 performance targets apply only to ARM64 hardware.

### Viewing ARM64 Results

After creating a PR:

1. **PR Comment**: Automatic summary posted by GitHub Actions
2. **Detailed Reports**: Download from Actions → Artifacts → `arm64-benchmark-results`
3. **Criterion HTML**: Open `target/criterion/*/report/index.html` from artifacts

### Manual Trigger

Go to Actions → "ARM64 Performance Benchmarks" → "Run workflow"

## Performance Targets (ARM64)

| Function | Target | Measurement | Notes |
|----------|--------|-------------|-------|
| `legal_moves()` | 500ns | Mean/StdDev/p99 | ARM64 CLZ/CTZ instructions |
| `make_move()` | 1.5μs | Mean/StdDev/p99 | Full move execution |
| `rotate_180()` | 200ns | Mean/StdDev/p99 | ARM64 REV instruction effect |

## Implementation Details

### ARM64 Optimizations

From `Cargo.toml`:
```toml
[target.aarch64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=neoverse-n1", "-C", "target-feature=+neon,+crc,+crypto"]
```

These flags enable:
- **Neoverse-N1 CPU tuning**: Optimized instruction scheduling
- **NEON**: ARM64 SIMD instructions
- **CRC**: Hardware CRC instructions
- **Crypto**: Cryptography extensions

### Benchmark Configuration

From benchmark files:
- `sample_size(1000)`: Exactly 1000 iterations as specified in Task 14.1
- `black_box()`: Prevents compiler optimizations that would invalidate measurements
- Criterion automatically calculates mean, stddev, and percentiles

## Troubleshooting

### Alternative ARM64 Runners

Current configuration uses `macos-latest` (Apple Silicon). Alternatives:

1. **Linux ARM64** (GitHub Enterprise only):
   ```yaml
   runs-on: ubuntu-24.04-arm
   ```

2. **Self-hosted ARM64**:
   ```yaml
   runs-on: [self-hosted, linux, ARM64]
   ```

### If benchmarks fail

1. Check Cargo.toml ARM64 configuration
2. Verify benchmark code compiles: `cargo bench --no-run`
3. Check GitHub Actions logs for detailed error messages

## Next Steps (Task 14.2, 14.3)

After Task 14.1 completion:
- Task 14.2: Evaluation system final performance verification
- Task 14.3: Performance report creation

These will be added to the same workflow as additional jobs.

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [GitHub Actions ARM64 Runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
