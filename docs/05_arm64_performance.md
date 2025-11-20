# ARM64 Performance Benchmarking Guide

## Overview

Phase 1 of Prismind is optimized for ARM64 architecture (specifically OCI Ampere A1 with Neoverse N1 cores). This document describes how to measure and verify performance on ARM64 systems.

## Performance Targets

### Phase 1 Requirements

| Operation | Target Latency | Requirements Reference |
|-----------|----------------|----------------------|
| Legal moves generation | < 500ns | REQ-15.1, REQ-2.6 |
| Pattern extraction (56 patterns) | < 25μs | REQ-15.2, REQ-8.6 |
| Evaluation function | < 35μs | REQ-15.3, REQ-11.6 |
| Make move (with flip) | < 1.5μs | REQ-15.4, REQ-3.6 |
| BitBoard rotation | < 200ns | REQ-15.5, REQ-5.6 |

### Cache Performance Targets

- **Cache miss rate**: 30-40% or lower (REQ-15.6)
- **Memory usage**: < 80MB for SoA format evaluation table (REQ-13.2)
- **IPC (Instructions Per Cycle)**: > 0.85

## GitHub Actions Workflow

### Automated ARM64 Benchmarking

The `.github/workflows/arm64-performance.yml` workflow automatically runs performance benchmarks on pull requests to `phase1-foundation` and `main` branches.

#### Workflow Features

1. **All Benchmarks**: Runs all 6 benchmark suites
   - `rotation_bench` - BitBoard rotation operations
   - `legal_moves_bench` - Legal move generation
   - `make_move_bench` - Move execution with stone flipping
   - `extract_patterns_bench` - Pattern index extraction
   - `score_conversion_bench` - u16 ↔ f32 conversion
   - `evaluate_bench` - Full evaluation function

2. **Performance Targets**: Displays target latencies in GitHub Actions summary

3. **Artifact Upload**: Saves Criterion HTML reports for detailed analysis

4. **Baseline Comparison**: Optional comparison with previous commits (via workflow_dispatch)

#### Viewing Results

After a PR is created:

1. Go to the **Actions** tab in GitHub
2. Find the **ARM64 Performance Benchmarks** workflow run
3. Click on the run to see the summary with performance targets
4. Download **arm64-benchmark-results** artifact for detailed Criterion reports

## Running Benchmarks Locally

### On ARM64 Linux (Native)

```bash
# Build with ARM64 optimizations
RUSTFLAGS="-C target-cpu=neoverse-n1 -C target-feature=+neon,+crc,+crypto" \
  cargo build --release --benches

# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench evaluate_bench

# Save baseline for comparison
cargo bench -- --save-baseline my-baseline

# Compare with baseline
cargo bench -- --baseline my-baseline
```

### On x86_64 (Cross-compilation)

```bash
# Install ARM64 target
rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation toolchain
sudo apt-get install gcc-aarch64-linux-gnu

# Build (check compilation only, cannot run)
cargo build --target aarch64-unknown-linux-gnu --release
```

## Cache Performance Analysis with `perf`

### Prerequisites (Linux ARM64 only)

```bash
# Install perf tools
sudo apt-get update
sudo apt-get install linux-tools-generic linux-tools-$(uname -r)
```

### Running Cache Analysis

```bash
# Build release binary with ARM64 optimizations
RUSTFLAGS="-C target-cpu=neoverse-n1 -C target-feature=+neon,+crc,+crypto" \
  cargo build --release

# Run evaluation benchmark with perf
sudo perf stat \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  cargo bench --bench evaluate_bench

# Example expected output:
#   Performance counter stats:
#
#   12,345,678,901      cycles                    #    3.000 GHz
#   10,234,567,890      instructions              #    0.83  insn per cycle
#      234,567,890      cache-references
#       82,098,765      cache-misses              #   35.00% of all cache refs
#    2,345,678,901      branches
#       12,345,678      branch-misses             #    0.53% of all branches
```

### Interpreting Results

#### Cache Miss Rate
```
cache_miss_rate = (cache-misses / cache-references) × 100%
```

**Target**: < 40%

- **Good** (30-35%): SoA optimization working well
- **Acceptable** (35-40%): Within target range
- **Poor** (>40%): Review memory layout, consider prefetching improvements

#### Instructions Per Cycle (IPC)
```
IPC = instructions / cycles
```

**Target**: > 0.85

- **Good** (>0.9): Efficient execution, low stalls
- **Acceptable** (0.85-0.9): Within target range
- **Poor** (<0.85): Pipeline stalls, review branch prediction

#### Branch Miss Rate
```
branch_miss_rate = (branch-misses / branches) × 100%
```

**Target**: < 2%

- Branchless implementations should achieve <1%
- Pattern extraction should benefit from branchless design

## ARM64 Optimization Features

### CPU-Specific Optimizations

The project uses the following ARM64-specific optimizations (enabled via `RUSTFLAGS` in `Cargo.toml`):

1. **Target CPU**: `neoverse-n1`
   - Optimized for OCI Ampere A1 processors
   - Enables Neoverse N1-specific instruction scheduling

2. **Target Features**:
   - `+neon`: SIMD vector instructions for batch processing
   - `+crc`: CRC32 instructions (future use)
   - `+crypto`: Cryptographic extensions (future use)

### Architecture-Specific Code

#### 1. REV Instruction (180° Rotation)

File: `src/board.rs`

```rust
pub fn rotate_180(board: u64) -> u64 {
    board.reverse_bits()  // Compiles to ARM64 REV instruction (1 cycle)
}
```

**Expected**: < 200ns (measured in `rotation_bench`)

#### 2. CLZ/CTZ Instructions (Legal Moves)

File: `src/board.rs`

```rust
// Find first legal move
let first_move = legal_moves.trailing_zeros();  // ARM64 CTZ instruction
```

**Expected**: < 500ns total for legal move generation

#### 3. NEON SIMD (Score Conversion - Optional)

File: `src/arm64.rs`

```rust
#[cfg(target_arch = "aarch64")]
pub fn u16_to_score_simd(values: &[u16; 8]) -> [f32; 8] {
    // Uses NEON intrinsics for 8-wide parallel conversion
}
```

**Expected**: 4-8x speedup over scalar version

#### 4. Prefetch Directives (Evaluation)

File: `src/evaluator.rs`

```rust
#[cfg(target_arch = "aarch64")]
unsafe {
    std::arch::aarch64::_builtin_prefetch(next_ptr, 0, 3);
}
```

**Expected**: Reduced cache miss rate (contribution to 30-40% target)

## Self-Hosted ARM64 Runner Setup (Optional)

For testing on actual OCI Ampere A1 instances:

### 1. Provision ARM64 Instance

```bash
# OCI Ampere A1 (4 cores, 24GB RAM recommended)
# OS: Ubuntu 22.04 ARM64
```

### 2. Install GitHub Actions Runner

```bash
# Download ARM64 runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-arm64-2.311.0.tar.gz \
  -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-arm64-2.311.0.tar.gz

tar xzf ./actions-runner-linux-arm64-2.311.0.tar.gz

# Configure runner (follow GitHub instructions)
./config.sh --url https://github.com/YOUR_ORG/Prismind --token YOUR_TOKEN

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

### 3. Update Workflow

In `.github/workflows/arm64-performance.yml`, change:

```yaml
runs-on: ubuntu-latest  # Current (cross-compile only)
```

to:

```yaml
runs-on: [self-hosted, linux, ARM64, ampere-a1]  # Self-hosted runner
```

## Continuous Performance Monitoring

### Regression Detection

1. **Baseline Benchmarks**: Run benchmarks on `main` branch regularly
2. **PR Comparison**: Compare PR benchmarks against baseline
3. **Alert on Regression**: Fail CI if performance degrades >10%

### Performance History Tracking

Consider using [Bencher.dev](https://bencher.dev) or similar tools for:
- Historical performance tracking
- Regression visualization
- Automated performance reports

## Troubleshooting

### Issue: Benchmarks run too slow

**Possible Causes**:
- Not running on ARM64 hardware
- Missing ARM64 optimization flags
- Debug build instead of release

**Solution**:
```bash
# Verify architecture
uname -m  # Should show "aarch64"

# Verify RUSTFLAGS
echo $RUSTFLAGS  # Should include target-cpu and target-feature

# Ensure release build
cargo bench --release
```

### Issue: Cache miss rate > 40%

**Possible Causes**:
- Incorrect SoA memory layout
- Missing prefetch directives
- Large working set size

**Solution**:
- Review `EvaluationTable` structure in `src/evaluator.rs`
- Verify prefetch implementation in evaluation loop
- Check memory usage with `cargo build --release && size target/release/prismind`

### Issue: IPC < 0.85

**Possible Causes**:
- Branch mispredictions
- Data dependencies
- Pipeline stalls

**Solution**:
- Use `perf record` for detailed profiling
- Review branchless implementations
- Check for unnecessary data dependencies

## References

- [ARM Neoverse N1 Optimization Guide](https://developer.arm.com/documentation/swog309707/latest/)
- [Rust ARM64 Performance Guide](https://doc.rust-lang.org/rustc/platform-support/aarch64-unknown-linux-gnu.html)
- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [Linux perf Examples](https://www.brendangregg.com/perf.html)

## Next Steps

After Phase 1 performance validation:

1. **Phase 2**: Search algorithm optimization (MTD(f), AlphaBeta)
2. **Phase 3**: Learning system (TD(λ)-Leaf with Adam optimizer)
3. **Phase 4**: Python bindings with PyO3

See `.kiro/specs/phase1-foundation/tasks.md` for complete implementation checklist.
