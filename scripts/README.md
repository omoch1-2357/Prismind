# Prismind Scripts

This directory contains utility scripts for profiling and performance analysis.

## perf_profile.sh

**Task 10.2**: Linux perf profiling script for measuring cache miss rates and branch prediction miss rates.

### Usage

```bash
# Make executable (first time only)
chmod +x scripts/perf_profile.sh

# Run profiling
./scripts/perf_profile.sh
```

### Prerequisites

- Linux operating system
- `perf` tools installed (`sudo apt-get install linux-tools-common linux-tools-generic`)
- Permissions to access perf events (see [docs/perf_profiling.md](../docs/perf_profiling.md))

### Output

Results are saved to `perf_results/perf_report_<arch>_<timestamp>.txt`

### Performance Targets

- Cache Miss Rate: ≤50%
- Branch Prediction Miss Rate: ≤1%

### Documentation

For detailed documentation, see:
- [Perf Profiling Guide](../docs/perf_profiling.md)
- [Performance Report Template](../docs/perf_report_template.md)

### CI/CD

This script is automatically run on GitHub Actions via `.github/workflows/perf-profile.yml`
