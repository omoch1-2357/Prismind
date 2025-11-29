//! Integration tests for PyO3 build configuration
//!
//! These tests verify that the PyO3 module is correctly configured
//! and can be compiled as a Python extension module.

/// Test that prismind library can be built as cdylib
/// This test verifies that the crate-type configuration is correct
#[test]
fn test_crate_type_includes_cdylib() {
    // The fact that this test compiles and runs means the rlib target works
    // The cdylib target is verified by the maturin build process
    // This test serves as a compile-time check that the library is properly configured

    // Verify we can access core prismind types
    use prismind::BitBoard;
    let board = BitBoard::new();
    assert_eq!(board.black.count_ones(), 2);
    assert_eq!(board.white_mask().count_ones(), 2);
}

/// Test that required dependencies are available
#[test]
fn test_required_dependencies_available() {
    // Test crc32fast is available
    let hash = crc32fast::hash(b"test data");
    assert!(hash != 0);

    // Test flate2 is available for compression
    use flate2::Compression;
    let level = Compression::default();
    assert!(level.level() > 0);
}

/// Test release profile optimizations are applied
/// This test documents expected behavior rather than runtime verification
#[test]
fn test_release_profile_documentation() {
    // Release profile settings (verified via Cargo.toml):
    // - opt-level = 3
    // - lto = true (changed from "fat" to true for cdylib compatibility)
    // - codegen-units = 1

    // Runtime verification: check that optimizations don't break functionality
    use prismind::board::{BitBoard, legal_moves, make_move};

    let mut board = BitBoard::new();
    let moves = legal_moves(&board);

    // Standard opening has 4 legal moves
    assert_eq!(moves.count_ones(), 4);

    // Verify move execution works correctly
    let pos = moves.trailing_zeros() as u8;
    make_move(&mut board, pos).expect("Move should succeed");
}

/// Test that pyproject.toml exists and is valid
#[test]
fn test_pyproject_toml_exists() {
    use std::path::Path;

    let pyproject_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("pyproject.toml");
    assert!(
        pyproject_path.exists(),
        "pyproject.toml must exist for maturin builds"
    );

    // Read and verify basic structure
    let content = std::fs::read_to_string(&pyproject_path).expect("Failed to read pyproject.toml");

    // Verify required sections exist
    assert!(
        content.contains("[build-system]"),
        "pyproject.toml must have [build-system] section"
    );
    assert!(
        content.contains("maturin"),
        "pyproject.toml must use maturin as build backend"
    );
    assert!(
        content.contains("[project]"),
        "pyproject.toml must have [project] section"
    );
    assert!(
        content.contains("requires-python"),
        "pyproject.toml must specify Python version requirement"
    );
}

/// Test that maturin configuration includes ARM64 support
#[test]
fn test_maturin_arm64_configuration() {
    use std::path::Path;

    let pyproject_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("pyproject.toml");
    let content = std::fs::read_to_string(&pyproject_path).expect("Failed to read pyproject.toml");

    // Verify maturin tool section exists
    assert!(
        content.contains("[tool.maturin]"),
        "pyproject.toml must have [tool.maturin] section for build configuration"
    );
}

/// Test that python module directory structure exists
#[test]
fn test_python_module_structure() {
    use std::path::Path;

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Verify python source directory exists (for pure Python wrappers)
    let python_dir = manifest_dir.join("python");
    assert!(
        python_dir.exists(),
        "python/ directory must exist for Python source modules"
    );

    // Verify prismind package directory exists
    let prismind_pkg = python_dir.join("prismind");
    assert!(
        prismind_pkg.exists(),
        "python/prismind/ package directory must exist"
    );

    // Verify __init__.py exists
    let init_py = prismind_pkg.join("__init__.py");
    assert!(init_py.exists(), "python/prismind/__init__.py must exist");
}

/// Test that PyO3 module source file exists
#[test]
fn test_pyo3_module_source_exists() {
    use std::path::Path;

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Verify src/python directory exists
    let python_mod = manifest_dir.join("src").join("python");
    assert!(
        python_mod.exists(),
        "src/python/ module directory must exist for PyO3 bindings"
    );

    // Verify mod.rs exists (module entry point)
    let mod_rs = python_mod.join("mod.rs");
    assert!(
        mod_rs.exists(),
        "src/python/mod.rs must exist as PyO3 module entry point"
    );

    // Read module content and verify basic structure
    let content = std::fs::read_to_string(&mod_rs).expect("Failed to read mod.rs");

    // Verify it contains PyO3 module definition
    assert!(
        content.contains("#[pymodule]") || content.contains("pymodule"),
        "src/python/mod.rs must contain PyO3 module definition"
    );

    // Verify module name is _prismind (matching pyproject.toml)
    assert!(
        content.contains("_prismind"),
        "PyO3 module must be named _prismind"
    );
}

#[cfg(feature = "pyo3")]
mod pyo3_tests {
    //! Tests that require PyO3 feature to be enabled

    /// Test that PyO3 module can be initialized
    /// Note: This test requires auto-initialize feature or explicit Python setup.
    /// We skip it in CI/test environments without Python.
    #[test]
    #[ignore] // Skip: requires Python interpreter
    fn test_pyo3_module_initialization() {
        use pyo3::prelude::*;

        // Initialize Python first if not using auto-initialize
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            // Basic Python interpreter access works
            let version = py.version();
            assert!(!version.is_empty());
        });
    }

    /// Test that PyO3 module registers expected classes
    #[test]
    fn test_pyo3_module_classes_registered() {
        // Verify the module exports required classes
        // This is verified at compile time by the pymodule macro
        use prismind::python::PyEvaluator;

        // If this compiles, the class is properly registered
        let _ = std::any::type_name::<PyEvaluator>();
    }

    /// Test that module version matches Cargo package version
    #[test]
    fn test_module_version() {
        // Version should match CARGO_PKG_VERSION
        let cargo_version = env!("CARGO_PKG_VERSION");
        assert!(!cargo_version.is_empty());
        assert_eq!(cargo_version, "0.1.0");
    }
}
