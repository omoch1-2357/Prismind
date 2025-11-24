# Perf Profiling Guide for Prismind Search Engine

## Overview

This guide explains how to use Linux `perf` tools to profile the Prismind search engine and measure cache miss rates, branch prediction miss rates, and other performance metrics.

**Target Metrics (Phase 2, Task 10.2):**
- **Cache Miss Rate**: ≤50%
- **Branch Prediction Miss Rate**: ≤1%

---

## Prerequisites

### Platform Requirements

- **Operating System**: Linux (Ubuntu, Debian, Fedora, Arch, etc.)
- **Architecture**: x86_64 or ARM64 (aarch64)
- **Kernel**: Modern Linux kernel with perf support (3.0+)

**Note:** The `perf` tool is Linux-specific. For profiling on other platforms:
- **macOS**: Use `instruments` or `dtrace`
- **Windows**: Use Windows Performance Analyzer (WPA)

### Installing Perf Tools

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
```

#### Fedora/RHEL/CentOS

```bash
sudo dnf install perf
```

#### Arch Linux

```bash
sudo pacman -S perf
```

#### Verify Installation

```bash
perf --version
```

Expected output:
```
perf version 5.x.x
```

---

## Setting Up Permissions

By default, `perf` requires root privileges or specific capabilities to access hardware performance counters.

### Option 1: Run with sudo (Simplest)

```bash
sudo ./scripts/perf_profile.sh
```

### Option 2: Adjust sysctl (Session-Based)

```bash
# Allow non-root users to access perf events
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
```

This setting is reset on reboot. To make it permanent:

```bash
# Add to /etc/sysctl.conf
sudo tee -a /etc/sysctl.conf << EOF
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0
EOF

# Apply changes
sudo sysctl -p
```

### Option 3: Add CAP_PERFMON Capability (Linux 5.8+)

```bash
# Grant CAP_PERFMON to specific user
sudo setcap cap_perfmon=ep /usr/bin/perf
```

---

## Usage

### Quick Start: Using the Profiling Script

The easiest way to run perf profiling is using the provided script:

```bash
# Make script executable
chmod +x scripts/perf_profile.sh

# Run profiling
./scripts/perf_profile.sh
```

This script will:
1. Check for Linux and perf installation
2. Build benchmarks in release mode
3. Run perf stat with cache and branch events
4. Parse results and check against targets
5. Save detailed report to `perf_results/perf_report_*.txt`

### Manual Profiling

If you prefer to run perf manually:

#### Step 1: Build Benchmarks

```bash
cargo build --release --benches
```

#### Step 2: Run Perf Stat

```bash
perf stat -e cache-references,cache-misses,branches,branch-misses,instructions,cycles \
  -- cargo bench --bench search_bench --no-fail-fast -- --test
```

Or directly on the benchmark binary:

```bash
# Find the compiled benchmark binary
BENCH_BINARY=$(ls -t target/release/deps/search_bench-* | grep -v '\.d$' | head -1)

# Run perf on the binary
perf stat -e cache-references,cache-misses,branches,branch-misses,instructions,cycles \
  -- "$BENCH_BINARY" --bench --test
```

---

## Understanding Perf Output

### Sample Perf Stat Output

```
Performance counter stats for 'cargo bench --bench search_bench -- --test':

     1,234,567,890      cache-references
       123,456,789      cache-misses              #   10.00 % of all cache refs
     9,876,543,210      branches
        98,765,432      branch-misses             #    1.00% of all branches
    50,000,000,000      instructions              #    2.00  insn per cycle
    25,000,000,000      cycles

       5.123456789 seconds time elapsed

       4.567890123 seconds user
       0.555666777 seconds sys
```

### Key Metrics Explained

#### Cache Miss Rate

```
Cache Miss Rate = (cache-misses / cache-references) × 100%
```

- **cache-references**: Total number of cache accesses
- **cache-misses**: Number of cache accesses that missed (had to fetch from main memory)

**Target**: ≤50%

**Interpretation**:
- **<10%**: Excellent cache locality
- **10-30%**: Good cache usage
- **30-50%**: Acceptable, room for optimization
- **>50%**: Poor cache locality, significant optimization needed

#### Branch Miss Rate

```
Branch Miss Rate = (branch-misses / branches) × 100%
```

- **branches**: Total number of conditional branches executed
- **branch-misses**: Number of branches that were mispredicted

**Target**: ≤1%

**Interpretation**:
- **<0.5%**: Excellent branch predictability
- **0.5-1%**: Good, typical for well-optimized code
- **1-2%**: Acceptable, some room for branchless optimizations
- **>2%**: Poor predictability, consider branchless alternatives

#### IPC (Instructions Per Cycle)

```
IPC = instructions / cycles
```

**Interpretation**:
- **IPC > 2**: Excellent CPU utilization (superscalar execution)
- **IPC 1-2**: Good efficiency
- **IPC < 1**: CPU is stalling (likely due to cache misses or data dependencies)

---

## Advanced Profiling

### Profiling Specific Functions

Use `perf record` to collect call graph data:

```bash
# Record with call graph
perf record -e cache-misses --call-graph dwarf -- cargo bench --bench search_bench -- --test

# View report
perf report
```

Navigate the report with arrow keys:
- Press `Enter` to expand function calls
- Press `a` to annotate assembly
- Press `q` to quit

### Cache Line Analysis

```bash
# Record cache misses with address sampling
perf record -e mem:0:r -- cargo bench --bench search_bench -- --test

# Analyze cache line accesses
perf mem report
```

### Branch Misprediction Hotspots

```bash
# Record branch misses
perf record -e branch-misses:u -- cargo bench --bench search_bench -- --test

# View annotated source
perf annotate
```

---

## Optimization Strategies

### Reducing Cache Misses

1. **Data Structure Alignment**:
   ```rust
   #[repr(C, align(64))]  // Align to cache line
   struct TTEntry {
       // ...
   }
   ```

2. **Structure of Arrays (SoA) Layout**:
   - Already implemented in Phase 1 evaluation tables
   - Improves SIMD efficiency and cache locality

3. **Prefetching**:
   ```rust
   #[cfg(target_arch = "aarch64")]
   unsafe {
       core::arch::aarch64::_prefetch(ptr, core::arch::aarch64::_PREFETCH_READ, core::arch::aarch64::_PREFETCH_LOCALITY3);
   }
   ```

4. **Access Patterns**:
   - Sequential access is better than random
   - Reorder data to match access patterns

### Reducing Branch Mispredictions

1. **Branchless Techniques**:
   ```rust
   // Before (branching)
   let priority = if is_corner(pos) { 100 } else { 50 };

   // After (branchless)
   let priority = 50 + (is_corner(pos) as i32) * 50;
   ```

2. **Likely/Unlikely Hints**:
   ```rust
   #[cold]
   fn unlikely_case() { /* ... */ }
   ```

3. **Loop Unrolling**:
   - Let LLVM handle most cases
   - Manual unroll for critical loops if needed

4. **Lookup Tables**:
   - Replace complex conditionals with array lookups
   - Trade memory for predictability

---

## Comparing ARM64 vs x86_64

### Running on Both Architectures

1. **x86_64 Linux**:
   ```bash
   ./scripts/perf_profile.sh
   ```

2. **ARM64 Linux** (e.g., OCI Ampere A1, AWS Graviton):
   ```bash
   ./scripts/perf_profile.sh
   ```

3. **Compare Results**:
   - Save outputs to separate files
   - Use the performance report template (docs/perf_report_template.md)
   - Analyze differences

### Expected Differences

**ARM64 Advantages**:
- Better IPC due to wider pipelines
- More efficient NEON SIMD
- Often better branch prediction
- Lower cache miss rates with larger caches

**x86_64 Advantages**:
- Higher clock speeds
- Better single-threaded performance
- More mature compiler optimizations

---

## CI/CD Integration

### GitHub Actions Workflow

Perf profiling is automatically run in CI/CD via `.github/workflows/perf-profile.yml`:

- **Triggered on**:
  - Pull requests to `phase2-search` or `main` branch
  - Manual workflow dispatch
  - Weekly schedule (Sunday midnight UTC)

- **Runs on**:
  - x86_64 Linux (ubuntu-latest)
  - ARM64 macOS (uses Criterion instead of perf)

- **Outputs**:
  - Perf stat results as artifacts
  - Performance comparison report
  - PR comment with summary

### Manual Workflow Trigger

Go to GitHub Actions → "Perf Profiling (Linux)" → "Run workflow"

---

## Troubleshooting

### Issue: "perf: not found"

**Solution**: Install perf tools (see Prerequisites section)

### Issue: "Permission denied"

**Solution**: Adjust permissions (see Setting Up Permissions section)

### Issue: "No data collected"

**Causes**:
- Benchmark ran too quickly
- Wrong perf event names for your architecture

**Solution**:
```bash
# List available events
perf list

# Test with basic events
perf stat -e cycles,instructions -- cargo bench --bench search_bench -- --test
```

### Issue: "Kernel too old"

Some perf features require newer kernels. Upgrade if possible, or use alternative profiling methods.

### Issue: ARM64 macOS "perf not available"

**Solution**: macOS doesn't have perf. Use Instruments or dtrace, or run on Linux ARM64.

---

## Performance Targets Verification

After running perf profiling, verify that results meet targets:

```bash
# Example output from script
Cache Statistics:
  References: 1,234,567,890
  Misses:     123,456,789
  Miss Rate:  10.00%
  Status:     ✓ PASS (target: ≤50%)

Branch Prediction Statistics:
  Branches:   9,876,543,210
  Misses:     98,765,432
  Miss Rate:  1.00%
  Status:     ✓ PASS (target: ≤1%)
```

Both targets must pass for Task 10.2 completion.

---

## References

- [Linux Perf Wiki](https://perf.wiki.kernel.org/)
- [Brendan Gregg's Perf Examples](http://www.brendangregg.com/perf.html)
- [ARM Performance Monitoring Unit](https://developer.arm.com/documentation/ddi0500/latest/)
- [Intel Performance Counter Monitor](https://software.intel.com/content/www/us/en/develop/articles/intel-performance-counter-monitor.html)
- [Optimizing Software in C++](https://www.agner.org/optimize/optimizing_cpp.pdf) - Chapter on cache and branch optimization

---

**Last Updated**: 2025-11-23
**Phase**: Phase 2 - Search Algorithm Implementation
**Task**: 10.2 - Perf profiling infrastructure
