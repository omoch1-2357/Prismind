#!/bin/bash
#
# build.sh - Build script for Prismind Othello AI Engine
#
# Provides common build targets for development and deployment.
# Requirements Coverage: 11.6 (Build scripts with common targets)
#
# Usage:
#   ./build.sh build      # Development build
#   ./build.sh release    # Release build
#   ./build.sh test       # Run all tests
#   ./build.sh bench      # Run benchmarks
#   ./build.sh install    # Install Python package
#   ./build.sh clean      # Clean build artifacts
#   ./build.sh help       # Show help
#

set -e  # Exit on error

# Configuration
CARGO="${CARGO:-cargo}"
MATURIN="${MATURIN:-maturin}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
PYTEST="${PYTEST:-pytest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo -e "${CYAN}=== $1 ===${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

command_exists() {
    command -v "$1" &> /dev/null
}

# Detect architecture
detect_arch() {
    local arch=$(uname -m)
    case $arch in
        aarch64|arm64)
            echo "arm64"
            ;;
        x86_64|amd64)
            echo "x86_64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

ARCH=$(detect_arch)

# Target implementations

do_build() {
    print_header "Development Build"
    $CARGO build
    print_success "Development build complete"
}

do_release() {
    print_header "Release Build"
    $CARGO build --release
    print_success "Release build complete"

    echo ""
    echo "Binary info:"
    ls -lh target/release/libprismind.* 2>/dev/null || true
    ls -lh target/release/prismind-cli 2>/dev/null || true
}

do_arm64_release() {
    print_header "ARM64 Release Build"

    if [ "$ARCH" = "arm64" ]; then
        echo "Building native ARM64 release..."
        $CARGO build --release
    else
        echo "Cross-compiling for ARM64..."
        $CARGO build --release --target aarch64-unknown-linux-gnu
    fi

    print_success "ARM64 release build complete"
}

do_test_rust() {
    print_header "Running Rust Tests"
    $CARGO test --all-features
    print_success "Rust tests passed"
}

do_test_python() {
    print_header "Running Python Tests"
    if command_exists $PYTEST; then
        $PYTEST python/tests/ -v --tb=short || print_warning "Python tests failed or not available"
    else
        print_warning "pytest not found, skipping Python tests"
    fi
}

do_test() {
    do_test_rust
    do_test_python
    print_success "All tests complete"
}

do_bench() {
    print_header "Running Benchmarks"
    $CARGO bench
    print_success "Benchmarks complete"
}

do_install() {
    print_header "Installing Python Package (Development Mode)"
    if ! command_exists $MATURIN; then
        print_error "maturin not found. Install with: pip install maturin"
        exit 1
    fi
    $MATURIN develop --release
    print_success "Installation complete. You can now: import prismind"
}

do_install_release() {
    print_header "Installing Release Build"
    if ! command_exists $MATURIN; then
        print_error "maturin not found. Install with: pip install maturin"
        exit 1
    fi
    $MATURIN build --release
    $PIP install target/wheels/*.whl --force-reinstall
    print_success "Release installation complete"
}

do_clean() {
    print_header "Cleaning Build Artifacts"

    $CARGO clean

    rm -rf target/ 2>/dev/null || true
    rm -rf .pytest_cache/ 2>/dev/null || true
    rm -rf dist/ 2>/dev/null || true
    rm -rf build/ 2>/dev/null || true
    rm -rf *.egg-info/ 2>/dev/null || true

    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    print_success "Clean complete"
}

do_clippy() {
    print_header "Running Clippy"
    $CARGO clippy --all-features -- -D warnings
    print_success "Clippy passed"
}

do_lint() {
    do_clippy

    print_header "Running Python Linters"
    if command_exists ruff; then
        ruff check python/ || print_warning "Ruff found issues"
    else
        print_warning "ruff not found, skipping Python linting"
    fi

    if command_exists mypy; then
        mypy python/ || print_warning "mypy found issues"
    else
        print_warning "mypy not found, skipping type checking"
    fi

    print_success "Linting complete"
}

do_check() {
    print_header "Checking Code"
    $CARGO check --all-features
    print_success "Check complete"
}

do_fmt() {
    print_header "Formatting Code"

    $CARGO fmt

    if command_exists ruff; then
        ruff format python/ || true
        ruff check --fix python/ || true
    else
        print_warning "ruff not found, skipping Python formatting"
    fi

    print_success "Formatting complete"
}

do_docs() {
    print_header "Generating Documentation"
    $CARGO doc --no-deps --all-features
    print_success "Documentation generated at: target/doc/prismind/index.html"
}

do_pre_commit() {
    print_header "Running Pre-commit Hooks"
    if command_exists pre-commit; then
        pre-commit run --all-files
        print_success "Pre-commit hooks passed"
    else
        print_warning "pre-commit not found. Install with: pip install pre-commit"
    fi
}

do_all() {
    do_release
    do_test
    do_lint
    do_docs
    print_success "Full build complete"
}

show_help() {
    echo ""
    echo -e "${CYAN}Prismind Build System${NC}"
    echo "====================="
    echo ""
    echo "Usage: ./build.sh <target>"
    echo ""
    echo -e "${YELLOW}Development Targets:${NC}"
    echo "  build         - Development build (debug mode, fast compilation)"
    echo "  dev           - Alias for build"
    echo "  release       - Release build with all optimizations"
    echo "  test          - Run all tests (Rust + Python)"
    echo "  bench         - Run performance benchmarks"
    echo "  lint          - Run linters (cargo clippy, ruff, mypy)"
    echo "  fmt           - Format code (cargo fmt, ruff)"
    echo "  check         - Check code without building (fast feedback)"
    echo ""
    echo -e "${YELLOW}Installation Targets:${NC}"
    echo "  install       - Install Python package (development mode)"
    echo "  install-dev   - Alias for install"
    echo "  install-release - Install optimized release build"
    echo ""
    echo -e "${YELLOW}Build Targets:${NC}"
    echo "  arm64-release - Build release for ARM64 target"
    echo ""
    echo -e "${YELLOW}Utility Targets:${NC}"
    echo "  clean         - Remove all build artifacts"
    echo "  docs          - Generate documentation"
    echo "  pre-commit    - Run pre-commit hooks on all files"
    echo "  all           - Full build (release + test + lint + docs)"
    echo ""
    echo "Architecture detected: $ARCH"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./build.sh build      # Quick development build"
    echo "  ./build.sh release    # Optimized release build"
    echo "  ./build.sh test       # Run test suite"
    echo "  ./build.sh install    # Install for development"
    echo ""
}

# Main execution
TARGET="${1:-help}"

case "$TARGET" in
    build|dev)
        do_build
        ;;
    release)
        do_release
        ;;
    arm64-release)
        do_arm64_release
        ;;
    test)
        do_test
        ;;
    test-rust)
        do_test_rust
        ;;
    test-python)
        do_test_python
        ;;
    bench)
        do_bench
        ;;
    install|install-dev)
        do_install
        ;;
    install-release)
        do_install_release
        ;;
    clean)
        do_clean
        ;;
    lint)
        do_lint
        ;;
    clippy)
        do_clippy
        ;;
    check)
        do_check
        ;;
    fmt)
        do_fmt
        ;;
    docs)
        do_docs
        ;;
    pre-commit)
        do_pre_commit
        ;;
    all)
        do_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown target: $TARGET"
        show_help
        exit 1
        ;;
esac
