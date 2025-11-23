#!/bin/bash
# scripts/perf_profile.sh
#
# Task 10.2: Perf profiling script for search performance measurement
#
# This script runs perf stat to measure cache miss rate and branch prediction miss rate
# for the Prismind search benchmarks.
#
# Requirements:
# - Linux environment with perf tools installed
# - Root or CAP_PERFMON capability
# - Cargo and Rust toolchain
#
# Targets:
# - Cache miss rate: ≤50%
# - Branch prediction miss rate: ≤1%

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Prismind Perf Profiling"
echo "Task 10.2: Cache and Branch Prediction Analysis"
echo "======================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: perf profiling only works on Linux${NC}"
    echo "Current OS: $OSTYPE"
    echo "Please run this script in a Linux environment (GitHub Actions, Docker, WSL, etc.)"
    exit 1
fi

# Check if perf is installed
if ! command -v perf &> /dev/null; then
    echo -e "${RED}Error: perf tool not found${NC}"
    echo "Install perf with:"
    echo "  Ubuntu/Debian: sudo apt-get install linux-tools-common linux-tools-generic"
    echo "  Fedora/RHEL:   sudo dnf install perf"
    echo "  Arch:          sudo pacman -S perf"
    exit 1
fi

# Check perf permissions
if ! perf stat -e cycles true &> /dev/null; then
    echo -e "${YELLOW}Warning: perf may require elevated permissions${NC}"
    echo "Try one of:"
    echo "  1. Run with sudo: sudo ./scripts/perf_profile.sh"
    echo "  2. Set sysctl: echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
    echo "  3. Add CAP_PERFMON capability"
    echo ""
fi

# Architecture detection
ARCH=$(uname -m)
echo "Architecture: $ARCH"
echo "Rust target: $(rustc -vV | grep host | cut -d' ' -f2)"
echo ""

# Build benchmarks in release mode
echo "Building benchmarks in release mode..."
cargo build --release --benches
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Output directory
OUTPUT_DIR="perf_results"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$OUTPUT_DIR/perf_report_${ARCH}_${TIMESTAMP}.txt"

echo "Output will be saved to: $REPORT_FILE"
echo ""

# Perf events to measure
# Cache events
CACHE_EVENTS="cache-references,cache-misses"
# Branch prediction events
BRANCH_EVENTS="branches,branch-misses"
# Instruction events for context
INST_EVENTS="instructions,cycles"

# All events combined
ALL_EVENTS="${CACHE_EVENTS},${BRANCH_EVENTS},${INST_EVENTS}"

echo "======================================"
echo "Running perf stat on search benchmarks"
echo "======================================"
echo ""
echo "Measured events:"
echo "  - Cache: cache-references, cache-misses"
echo "  - Branch: branches, branch-misses"
echo "  - Instructions: instructions, cycles"
echo ""

# Run perf stat on the search benchmark binary
# Note: We use the compiled benchmark binary directly to avoid overhead
# Find the most recent search_bench binary (excluding .d files)
BENCH_BINARY=$(find target/release/deps -maxdepth 1 -name 'search_bench-*' -type f ! -name '*.d' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- | xargs basename)

if [[ ! -f "target/release/deps/${BENCH_BINARY}" ]]; then
    echo -e "${YELLOW}Warning: Compiled benchmark binary not found${NC}"
    echo "Attempting to run via cargo bench..."

    # Run via cargo bench with perf
    perf stat -e "$ALL_EVENTS" -o "$REPORT_FILE" \
        cargo bench --bench search_bench --no-fail-fast -- --test
else
    echo "Using benchmark binary: $BENCH_BINARY"

    # Run the benchmark binary directly with perf
    perf stat -e "$ALL_EVENTS" -o "$REPORT_FILE" \
        "target/release/deps/${BENCH_BINARY}" --bench --test
fi

echo ""
echo -e "${GREEN}✓ Perf profiling complete${NC}"
echo ""

# Parse and display results
echo "======================================"
echo "Performance Analysis"
echo "======================================"
echo ""

# Extract metrics from perf output
parse_perf_output() {
    local report=$1

    # Cache metrics (declare separately for better error handling)
    local cache_refs
    cache_refs=$(grep -oP '\d+(?=\s+cache-references)' "$report" | head -1)
    local cache_misses
    cache_misses=$(grep -oP '\d+(?=\s+cache-misses)' "$report" | head -1)

    # Branch metrics
    local branches=$(grep -oP '\d+(?=\s+branches)' "$report" | head -1)
    local branch_misses=$(grep -oP '\d+(?=\s+branch-misses)' "$report" | head -1)

    # Instruction metrics
    local instructions=$(grep -oP '\d+(?=\s+instructions)' "$report" | head -1)
    local cycles=$(grep -oP '\d+(?=\s+cycles)' "$report" | head -1)

    # Calculate miss rates
    if [[ -n "$cache_refs" && -n "$cache_misses" && "$cache_refs" -gt 0 ]]; then
        cache_miss_rate=$(echo "scale=4; ($cache_misses / $cache_refs) * 100" | bc)
        echo "Cache Statistics:"
        echo "  References: $(printf "%'d" $cache_refs)"
        echo "  Misses:     $(printf "%'d" $cache_misses)"
        echo "  Miss Rate:  ${cache_miss_rate}%"

        # Check against target (≤50%)
        if (( $(echo "$cache_miss_rate <= 50" | bc -l) )); then
            echo -e "  Status:     ${GREEN}✓ PASS (target: ≤50%)${NC}"
        else
            echo -e "  Status:     ${RED}✗ FAIL (target: ≤50%)${NC}"
        fi
        echo ""
    fi

    if [[ -n "$branches" && -n "$branch_misses" && "$branches" -gt 0 ]]; then
        branch_miss_rate=$(echo "scale=4; ($branch_misses / $branches) * 100" | bc)
        echo "Branch Prediction Statistics:"
        echo "  Branches:   $(printf "%'d" $branches)"
        echo "  Misses:     $(printf "%'d" $branch_misses)"
        echo "  Miss Rate:  ${branch_miss_rate}%"

        # Check against target (≤1%)
        if (( $(echo "$branch_miss_rate <= 1" | bc -l) )); then
            echo -e "  Status:     ${GREEN}✓ PASS (target: ≤1%)${NC}"
        else
            echo -e "  Status:     ${RED}✗ FAIL (target: ≤1%)${NC}"
        fi
        echo ""
    fi

    if [[ -n "$instructions" && -n "$cycles" && "$cycles" -gt 0 ]]; then
        ipc=$(echo "scale=4; $instructions / $cycles" | bc)
        echo "CPU Efficiency:"
        echo "  Instructions: $(printf "%'d" $instructions)"
        echo "  Cycles:       $(printf "%'d" $cycles)"
        echo "  IPC:          $ipc (Instructions Per Cycle)"
        echo ""
    fi
}

# Parse and display results
if [[ -f "$REPORT_FILE" ]]; then
    parse_perf_output "$REPORT_FILE"

    echo "======================================"
    echo "Full perf stat output saved to:"
    echo "  $REPORT_FILE"
    echo "======================================"
else
    echo -e "${RED}Error: Perf report file not found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Perf profiling completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review the detailed report: cat $REPORT_FILE"
echo "  2. Compare with other architectures (run on x86_64 and ARM64)"
echo "  3. Update docs/perf_report_template.md with results"
