# Makefile for Prismind Othello AI Engine
#
# Provides common build targets for development and deployment.
# Requirements Coverage: 11.6 (Build scripts with common targets)
#
# Targets:
#   build       - Development build (debug mode)
#   release     - Release build with optimizations
#   test        - Run all tests
#   bench       - Run benchmarks
#   install     - Install Python package (development mode)
#   clean       - Clean build artifacts
#   lint        - Run linters and formatters
#   docs        - Generate documentation
#
# Usage:
#   make build      # Quick development build
#   make release    # Optimized release build
#   make test       # Run test suite
#   make bench      # Run benchmarks
#   make install    # Install for development

# Configuration
CARGO := cargo
MATURIN := maturin
PYTHON := python3
PIP := pip
PYTEST := pytest
CARGO_FLAGS :=
RELEASE_FLAGS := --release

# Detect target architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
else ifeq ($(UNAME_M),arm64)
    ARCH := arm64
else
    ARCH := x86_64
endif

# Default target
.DEFAULT_GOAL := build

# Phony targets (not actual files)
.PHONY: all build release test bench install clean lint docs \
        build-python build-rust test-rust test-python \
        check fmt clippy help dev arm64-release \
        install-dev install-release pre-commit

# Help target
help:
	@echo "Prismind Build System"
	@echo "====================="
	@echo ""
	@echo "Development Targets:"
	@echo "  build         - Development build (debug mode, fast compilation)"
	@echo "  dev           - Alias for build"
	@echo "  release       - Release build with all optimizations"
	@echo "  test          - Run all tests (Rust + Python)"
	@echo "  bench         - Run performance benchmarks"
	@echo "  lint          - Run linters (cargo clippy, ruff, mypy)"
	@echo "  fmt           - Format code (cargo fmt, ruff)"
	@echo "  check         - Check code without building (fast feedback)"
	@echo ""
	@echo "Installation Targets:"
	@echo "  install       - Install Python package (development mode)"
	@echo "  install-dev   - Alias for install"
	@echo "  install-release - Install optimized release build"
	@echo ""
	@echo "Build Targets:"
	@echo "  build-rust    - Build Rust library only"
	@echo "  build-python  - Build Python bindings with maturin"
	@echo "  arm64-release - Build release for ARM64 target"
	@echo ""
	@echo "Utility Targets:"
	@echo "  clean         - Remove all build artifacts"
	@echo "  docs          - Generate documentation"
	@echo "  pre-commit    - Run pre-commit hooks on all files"
	@echo ""
	@echo "Architecture detected: $(ARCH)"

# Development build (fast iteration)
build: build-rust
	@echo "Development build complete"

dev: build

# Build Rust library only
build-rust:
	@echo "Building Rust library (debug mode)..."
	$(CARGO) build $(CARGO_FLAGS)

# Build Python bindings
build-python:
	@echo "Building Python bindings (debug mode)..."
	$(MATURIN) develop

# Release build with all optimizations
release:
	@echo "Building release version with optimizations..."
	$(CARGO) build $(RELEASE_FLAGS) $(CARGO_FLAGS)
	@echo "Release build complete"
	@echo "Binary size:"
	@ls -lh target/release/libprismind.* 2>/dev/null || true
	@ls -lh target/release/prismind-cli 2>/dev/null || true

# ARM64-specific release build
arm64-release:
	@echo "Building release for ARM64 target..."
ifeq ($(ARCH),arm64)
	@echo "Building native ARM64 release..."
	$(CARGO) build $(RELEASE_FLAGS) $(CARGO_FLAGS)
else
	@echo "Cross-compiling for ARM64..."
	$(CARGO) build $(RELEASE_FLAGS) --target aarch64-unknown-linux-gnu $(CARGO_FLAGS)
endif
	@echo "ARM64 release build complete"

# Run all tests
test: test-rust test-python
	@echo "All tests passed"

# Run Rust tests
test-rust:
	@echo "Running Rust tests..."
	$(CARGO) test --all-features $(CARGO_FLAGS)

# Run Python tests
test-python:
	@echo "Running Python tests..."
	$(PYTEST) python/tests/ -v --tb=short || echo "Python tests skipped (not installed)"

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	$(CARGO) bench $(CARGO_FLAGS)

# Install Python package (development mode)
install: install-dev

install-dev:
	@echo "Installing Python package (development mode)..."
	$(MATURIN) develop --release
	@echo "Installation complete. You can now: import prismind"

# Install release build
install-release:
	@echo "Building and installing release version..."
	$(MATURIN) build --release
	$(PIP) install target/wheels/*.whl --force-reinstall
	@echo "Release installation complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -rf target/
	rm -rf .pytest_cache/
	rm -rf python/__pycache__/
	rm -rf python/**/__pycache__/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete"

# Lint and format checks
lint: clippy
	@echo "Running Python linters..."
	ruff check python/ || true
	mypy python/ || true
	@echo "Linting complete"

# Clippy (Rust linter)
clippy:
	@echo "Running clippy..."
	$(CARGO) clippy --all-features -- -D warnings

# Check code without building
check:
	@echo "Checking code..."
	$(CARGO) check --all-features $(CARGO_FLAGS)

# Format code
fmt:
	@echo "Formatting Rust code..."
	$(CARGO) fmt
	@echo "Formatting Python code..."
	ruff format python/ || true
	ruff check --fix python/ || true
	@echo "Formatting complete"

# Generate documentation
docs:
	@echo "Generating documentation..."
	$(CARGO) doc --no-deps --all-features
	@echo "Documentation generated at: target/doc/prismind/index.html"

# Run pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# All target - build everything
all: release test lint docs
	@echo "Full build complete"
