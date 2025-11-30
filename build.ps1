<#
.SYNOPSIS
    Build script for Prismind Othello AI Engine (Windows PowerShell)

.DESCRIPTION
    Provides common build targets for development and deployment on Windows.
    Requirements Coverage: 11.6 (Build scripts with common targets)

.PARAMETER Target
    The build target to execute. Valid values:
    - build      : Development build (debug mode)
    - release    : Release build with optimizations
    - test       : Run all tests
    - bench      : Run benchmarks
    - install    : Install Python package (development mode)
    - clean      : Clean build artifacts
    - lint       : Run linters
    - fmt        : Format code
    - check      : Check code without building
    - docs       : Generate documentation
    - help       : Show this help message

.EXAMPLE
    .\build.ps1 build
    Quick development build

.EXAMPLE
    .\build.ps1 release
    Optimized release build

.EXAMPLE
    .\build.ps1 test
    Run all tests

.EXAMPLE
    .\build.ps1 install
    Install Python package for development

.NOTES
    Author: Prismind Project
    License: MIT
#>

param(
    [Parameter(Position=0)]
    [ValidateSet('build', 'release', 'test', 'bench', 'install', 'clean', 'lint', 'fmt', 'check', 'docs', 'help', 'dev', 'install-dev', 'install-release', 'test-rust', 'test-python', 'clippy', 'pre-commit', 'all')]
    [string]$Target = 'help'
)

# Configuration
$ErrorActionPreference = 'Stop'
$Cargo = 'cargo'
$Maturin = 'maturin'
$Python = 'python'
$Pip = 'pip'
$Pytest = 'pytest'

# Helper functions
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Target implementations
function Invoke-Build {
    Write-Header "Development Build"
    & $Cargo build
    if ($LASTEXITCODE -ne 0) { throw "Rust build failed" }
    Write-Success "Development build complete"
}

function Invoke-Release {
    Write-Header "Release Build"
    & $Cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "Release build failed" }
    Write-Success "Release build complete"

    Write-Host ""
    Write-Host "Binary info:" -ForegroundColor Cyan
    $releaseDir = "target\release"
    if (Test-Path "$releaseDir\prismind.dll") {
        Get-Item "$releaseDir\prismind.dll" | Format-Table Name, @{N='Size (KB)';E={[math]::Round($_.Length/1KB)}} -AutoSize
    }
    if (Test-Path "$releaseDir\prismind-cli.exe") {
        Get-Item "$releaseDir\prismind-cli.exe" | Format-Table Name, @{N='Size (KB)';E={[math]::Round($_.Length/1KB)}} -AutoSize
    }
}

function Invoke-TestRust {
    Write-Header "Running Rust Tests"
    & $Cargo test --all-features
    if ($LASTEXITCODE -ne 0) { throw "Rust tests failed" }
    Write-Success "Rust tests passed"
}

function Invoke-TestPython {
    Write-Header "Running Python Tests"
    if (Test-CommandExists $Pytest) {
        & $Pytest python/tests/ -v --tb=short
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "Python tests failed or not available"
        } else {
            Write-Success "Python tests passed"
        }
    } else {
        Write-Warning-Custom "pytest not found, skipping Python tests"
    }
}

function Invoke-Test {
    Invoke-TestRust
    Invoke-TestPython
    Write-Success "All tests complete"
}

function Invoke-Bench {
    Write-Header "Running Benchmarks"
    & $Cargo bench
    if ($LASTEXITCODE -ne 0) { throw "Benchmarks failed" }
    Write-Success "Benchmarks complete"
}

function Invoke-Install {
    Write-Header "Installing Python Package (Development Mode)"
    if (-not (Test-CommandExists $Maturin)) {
        Write-Warning-Custom "maturin not found. Install with: pip install maturin"
        throw "maturin required for installation"
    }
    & $Maturin develop --release
    if ($LASTEXITCODE -ne 0) { throw "Installation failed" }
    Write-Success "Installation complete. You can now: import prismind"
}

function Invoke-InstallRelease {
    Write-Header "Installing Release Build"
    if (-not (Test-CommandExists $Maturin)) {
        throw "maturin required for installation"
    }
    & $Maturin build --release
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }

    $wheel = Get-ChildItem "target\wheels\*.whl" | Select-Object -First 1
    if ($wheel) {
        & $Pip install $wheel.FullName --force-reinstall
        if ($LASTEXITCODE -ne 0) { throw "Installation failed" }
        Write-Success "Release installation complete"
    } else {
        throw "No wheel found in target\wheels"
    }
}

function Invoke-Clean {
    Write-Header "Cleaning Build Artifacts"

    & $Cargo clean

    $pathsToRemove = @(
        'target',
        '.pytest_cache',
        'dist',
        'build',
        '*.egg-info'
    )

    foreach ($path in $pathsToRemove) {
        if ($path.Contains('*')) {
            Get-ChildItem $path -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        } elseif (Test-Path $path) {
            Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # Remove __pycache__ directories
    Get-ChildItem -Path . -Directory -Recurse -Filter '__pycache__' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # Remove .pyc files
    Get-ChildItem -Path . -Recurse -Filter '*.pyc' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue

    Write-Success "Clean complete"
}

function Invoke-Clippy {
    Write-Header "Running Clippy"
    & $Cargo clippy --all-features -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw "Clippy found issues" }
    Write-Success "Clippy passed"
}

function Invoke-Lint {
    Invoke-Clippy

    Write-Header "Running Python Linters"
    if (Test-CommandExists 'ruff') {
        & ruff check python/
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "Ruff found issues"
        }
    } else {
        Write-Warning-Custom "ruff not found, skipping Python linting"
    }

    if (Test-CommandExists 'mypy') {
        & mypy python/
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "mypy found issues"
        }
    } else {
        Write-Warning-Custom "mypy not found, skipping type checking"
    }

    Write-Success "Linting complete"
}

function Invoke-Check {
    Write-Header "Checking Code"
    & $Cargo check --all-features
    if ($LASTEXITCODE -ne 0) { throw "Check failed" }
    Write-Success "Check complete"
}

function Invoke-Fmt {
    Write-Header "Formatting Code"

    & $Cargo fmt
    if ($LASTEXITCODE -ne 0) { throw "Cargo fmt failed" }

    if (Test-CommandExists 'ruff') {
        & ruff format python/
        & ruff check --fix python/
    } else {
        Write-Warning-Custom "ruff not found, skipping Python formatting"
    }

    Write-Success "Formatting complete"
}

function Invoke-Docs {
    Write-Header "Generating Documentation"
    & $Cargo doc --no-deps --all-features
    if ($LASTEXITCODE -ne 0) { throw "Documentation generation failed" }
    Write-Success "Documentation generated at: target\doc\prismind\index.html"
}

function Invoke-PreCommit {
    Write-Header "Running Pre-commit Hooks"
    if (Test-CommandExists 'pre-commit') {
        & pre-commit run --all-files
        if ($LASTEXITCODE -ne 0) { throw "Pre-commit hooks failed" }
        Write-Success "Pre-commit hooks passed"
    } else {
        Write-Warning-Custom "pre-commit not found. Install with: pip install pre-commit"
    }
}

function Invoke-All {
    Invoke-Release
    Invoke-Test
    Invoke-Lint
    Invoke-Docs
    Write-Success "Full build complete"
}

function Show-Help {
    Write-Host ""
    Write-Host "Prismind Build System (Windows)" -ForegroundColor Cyan
    Write-Host "===============================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 <target>" -ForegroundColor White
    Write-Host ""
    Write-Host "Development Targets:" -ForegroundColor Yellow
    Write-Host "  build         - Development build (debug mode, fast compilation)"
    Write-Host "  dev           - Alias for build"
    Write-Host "  release       - Release build with all optimizations"
    Write-Host "  test          - Run all tests (Rust + Python)"
    Write-Host "  bench         - Run performance benchmarks"
    Write-Host "  lint          - Run linters (cargo clippy, ruff, mypy)"
    Write-Host "  fmt           - Format code (cargo fmt, ruff)"
    Write-Host "  check         - Check code without building (fast feedback)"
    Write-Host ""
    Write-Host "Installation Targets:" -ForegroundColor Yellow
    Write-Host "  install       - Install Python package (development mode)"
    Write-Host "  install-dev   - Alias for install"
    Write-Host "  install-release - Install optimized release build"
    Write-Host ""
    Write-Host "Utility Targets:" -ForegroundColor Yellow
    Write-Host "  clean         - Remove all build artifacts"
    Write-Host "  docs          - Generate documentation"
    Write-Host "  pre-commit    - Run pre-commit hooks on all files"
    Write-Host "  all           - Full build (release + test + lint + docs)"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1 build     # Quick development build"
    Write-Host "  .\build.ps1 release   # Optimized release build"
    Write-Host "  .\build.ps1 test      # Run test suite"
    Write-Host "  .\build.ps1 install   # Install for development"
    Write-Host ""
}

# Main execution
try {
    switch ($Target) {
        'build' { Invoke-Build }
        'dev' { Invoke-Build }
        'release' { Invoke-Release }
        'test' { Invoke-Test }
        'test-rust' { Invoke-TestRust }
        'test-python' { Invoke-TestPython }
        'bench' { Invoke-Bench }
        'install' { Invoke-Install }
        'install-dev' { Invoke-Install }
        'install-release' { Invoke-InstallRelease }
        'clean' { Invoke-Clean }
        'lint' { Invoke-Lint }
        'clippy' { Invoke-Clippy }
        'check' { Invoke-Check }
        'fmt' { Invoke-Fmt }
        'docs' { Invoke-Docs }
        'pre-commit' { Invoke-PreCommit }
        'all' { Invoke-All }
        'help' { Show-Help }
        default { Show-Help }
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
}
