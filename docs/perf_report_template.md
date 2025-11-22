# Performance Profiling Report: ARM64 vs x86_64

## Executive Summary

This report compares the performance characteristics of the Prismind search engine on ARM64 and x86_64 architectures using Linux perf profiling tools.

**Report Date:** [YYYY-MM-DD]
**Prismind Version:** [Version/Commit]
**Phase:** Phase 2 - Search Algorithm Implementation

---

## Performance Targets (Task 10.2)

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Cache Miss Rate** | ≤50% | Efficient memory access patterns, good cache locality |
| **Branch Prediction Miss Rate** | ≤1% | Branchless implementations, predictable control flow |

---

## Test Environment

### x86_64 Linux

- **Platform:** [Ubuntu/Debian/Fedora version]
- **CPU:** [CPU model, e.g., Intel Xeon E5-2686 v4]
- **Cores:** [Number of cores]
- **Cache:** [L1/L2/L3 cache sizes]
- **RAM:** [Total RAM]
- **Kernel:** [Kernel version]
- **Rust Version:** [rustc version]
- **Perf Version:** [perf --version]

### ARM64 Linux

- **Platform:** [Ubuntu/Debian version on ARM64]
- **CPU:** [CPU model, e.g., Ampere Altra, AWS Graviton2, Apple M1/M2]
- **Cores:** [Number of cores]
- **Cache:** [L1/L2/L3 cache sizes]
- **RAM:** [Total RAM]
- **Kernel:** [Kernel version]
- **Rust Version:** [rustc version]
- **Perf Version:** [perf --version]

---

## Benchmark Configuration

### Search Benchmarks Profiled

```bash
# Benchmarks from benches/search_bench.rs
- bench_initial_position (depths 1-6)
- bench_midgame_position (depths 6-8)
- bench_tt_hit_rate
- bench_pruning_efficiency
- bench_nodes_per_second
```

### Perf Command

```bash
perf stat -e cache-references,cache-misses,branches,branch-misses,instructions,cycles \
  -- cargo bench --bench search_bench --no-fail-fast -- --test
```

---

## Results: x86_64 Linux

### Cache Performance

| Metric | Value | Status |
|--------|-------|--------|
| Cache References | [e.g., 1,234,567,890] | - |
| Cache Misses | [e.g., 123,456,789] | - |
| **Cache Miss Rate** | **[e.g., 10.0%]** | ✅ PASS / ❌ FAIL |

### Branch Prediction Performance

| Metric | Value | Status |
|--------|-------|--------|
| Total Branches | [e.g., 9,876,543,210] | - |
| Branch Misses | [e.g., 98,765,432] | - |
| **Branch Miss Rate** | **[e.g., 1.0%]** | ✅ PASS / ❌ FAIL |

### CPU Efficiency

| Metric | Value |
|--------|-------|
| Instructions | [e.g., 50,000,000,000] |
| Cycles | [e.g., 25,000,000,000] |
| **IPC (Instructions Per Cycle)** | **[e.g., 2.0]** |

### Detailed Perf Output (x86_64)

```
[Paste full perf stat output here]

Performance counter stats for '...':

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

---

## Results: ARM64 Linux

### Cache Performance

| Metric | Value | Status |
|--------|-------|--------|
| Cache References | [e.g., 1,234,567,890] | - |
| Cache Misses | [e.g., 61,728,395] | - |
| **Cache Miss Rate** | **[e.g., 5.0%]** | ✅ PASS / ❌ FAIL |

### Branch Prediction Performance

| Metric | Value | Status |
|--------|-------|--------|
| Total Branches | [e.g., 9,876,543,210] | - |
| Branch Misses | [e.g., 49,382,716] | - |
| **Branch Miss Rate** | **[e.g., 0.5%]** | ✅ PASS / ❌ FAIL |

### CPU Efficiency

| Metric | Value |
|--------|-------|
| Instructions | [e.g., 50,000,000,000] |
| Cycles | [e.g., 20,000,000,000] |
| **IPC (Instructions Per Cycle)** | **[e.g., 2.5]** |

### Detailed Perf Output (ARM64)

```
[Paste full perf stat output here]

Performance counter stats for '...':

     1,234,567,890      cache-references
        61,728,395      cache-misses              #    5.00 % of all cache refs
     9,876,543,210      branches
        49,382,716      branch-misses             #    0.50% of all branches
    50,000,000,000      instructions              #    2.50  insn per cycle
    20,000,000,000      cycles

       4.123456789 seconds time elapsed

       3.567890123 seconds user
       0.455666777 seconds sys
```

---

## Comparison: ARM64 vs x86_64

### Performance Comparison Table

| Metric | x86_64 | ARM64 | Winner | Improvement |
|--------|--------|-------|--------|-------------|
| Cache Miss Rate | [e.g., 10.0%] | [e.g., 5.0%] | ARM64 | 50% lower |
| Branch Miss Rate | [e.g., 1.0%] | [e.g., 0.5%] | ARM64 | 50% lower |
| IPC | [e.g., 2.0] | [e.g., 2.5] | ARM64 | 25% higher |
| Execution Time | [e.g., 5.12s] | [e.g., 4.12s] | ARM64 | 19.5% faster |

### Visualization (Optional)

```
Cache Miss Rate Comparison:
x86_64:  ████████████ 10.0%
ARM64:   ██████ 5.0%
Target:  ██████████████████████████████ 50.0% (max)

Branch Miss Rate Comparison:
x86_64:  ████████████ 1.0%
ARM64:   ██████ 0.5%
Target:  ████████████ 1.0% (max)
```

---

## Analysis

### Cache Miss Rate

**x86_64:**
- [Analysis of x86_64 cache performance]
- [Identify hotspots or inefficiencies]

**ARM64:**
- [Analysis of ARM64 cache performance]
- [Explain ARM64-specific optimizations that helped]

**Comparison:**
- [Compare cache behavior between architectures]
- [Explain differences (e.g., larger caches, better prefetching)]

### Branch Prediction Miss Rate

**x86_64:**
- [Analysis of x86_64 branch prediction]
- [Identify problematic branches]

**ARM64:**
- [Analysis of ARM64 branch prediction]
- [Explain branchless optimizations or better prediction]

**Comparison:**
- [Compare branch behavior]
- [Explain architectural differences in branch predictors]

### CPU Efficiency (IPC)

**x86_64:**
- [Analysis of instruction-level parallelism]

**ARM64:**
- [Analysis of ARM64 IPC advantages]
- [Explain wider pipelines, better out-of-order execution, etc.]

---

## Optimization Recommendations

### General Optimizations (Both Architectures)

1. **Cache Optimization:**
   - [Specific recommendations for cache locality]
   - [Data structure alignment suggestions]

2. **Branch Reduction:**
   - [Identify branches that can be eliminated]
   - [Suggest branchless alternatives]

3. **Algorithmic Improvements:**
   - [Suggest algorithmic changes to reduce cache misses or branches]

### ARM64-Specific Optimizations

1. **NEON SIMD:**
   - [Already implemented or suggestions for further SIMD usage]

2. **Prefetching:**
   - [Use ARM64 PRFM instruction for manual prefetching]

3. **Cache Line Alignment:**
   - [Ensure critical data structures are cache-aligned]

### x86_64-Specific Optimizations

1. **SSE/AVX:**
   - [Suggestions for x86 SIMD usage]

2. **Cache Prefetch:**
   - [Use x86 prefetch instructions]

---

## Conclusion

### Performance Target Achievement

| Target | x86_64 | ARM64 |
|--------|--------|-------|
| Cache Miss Rate ≤50% | ✅ PASS / ❌ FAIL | ✅ PASS / ❌ FAIL |
| Branch Miss Rate ≤1% | ✅ PASS / ❌ FAIL | ✅ PASS / ❌ FAIL |

### Summary

[Overall summary of performance comparison]

[Key takeaways]

[Recommendations for production deployment]

---

## Appendix

### Reproduction Steps

1. **Setup Linux Environment:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install linux-tools-common linux-tools-generic

   # Set perf permissions
   echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
   ```

2. **Run Profiling:**
   ```bash
   # Clone repository
   git clone https://github.com/your-org/Prismind.git
   cd Prismind

   # Run perf profiling script
   chmod +x scripts/perf_profile.sh
   ./scripts/perf_profile.sh
   ```

3. **Analyze Results:**
   ```bash
   # View generated report
   cat perf_results/perf_report_*.txt
   ```

### References

- [Linux perf documentation](https://perf.wiki.kernel.org/)
- [ARM64 Performance Monitoring](https://developer.arm.com/documentation/)
- [Intel Performance Counter Monitor](https://software.intel.com/content/www/us/en/develop/articles/intel-performance-counter-monitor.html)
- [Prismind Phase 2 Design Document](../design.md)

---

**Report Generated:** [Timestamp]
**Author:** [Your Name / GitHub Actions]
**Contact:** [Email / GitHub]
