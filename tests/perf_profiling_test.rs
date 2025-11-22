// tests/perf_profiling_test.rs
//
// Task 10.2: perf profiling infrastructure tests
// These tests verify that perf profiling scripts and configuration are correct

#[cfg(test)]
mod perf_profiling_tests {
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_perf_script_exists() {
        // Verify that perf profiling script exists
        let script_path = Path::new("scripts/perf_profile.sh");
        assert!(
            script_path.exists(),
            "Perf profiling script should exist at scripts/perf_profile.sh"
        );
    }

    #[test]
    fn test_perf_script_is_executable() {
        // On Unix systems, verify script has executable permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let script_path = Path::new("scripts/perf_profile.sh");
            if script_path.exists() {
                let metadata = fs::metadata(script_path).expect("Failed to read script metadata");
                let permissions = metadata.permissions();
                assert!(
                    permissions.mode() & 0o111 != 0,
                    "Perf script should be executable"
                );
            }
        }
    }

    #[test]
    fn test_perf_workflow_exists() {
        // Verify that GitHub Actions workflow for perf profiling exists
        let workflow_path = Path::new(".github/workflows/perf-profile.yml");
        assert!(
            workflow_path.exists(),
            "Perf profiling workflow should exist at .github/workflows/perf-profile.yml"
        );
    }

    #[test]
    fn test_perf_workflow_contains_required_sections() {
        let workflow_path = Path::new(".github/workflows/perf-profile.yml");
        if workflow_path.exists() {
            let content = fs::read_to_string(workflow_path).expect("Failed to read workflow file");

            // Check for required sections
            assert!(
                content.contains("perf stat"),
                "Workflow should contain 'perf stat' command"
            );
            assert!(
                content.contains("cache-misses"),
                "Workflow should measure cache-misses"
            );
            assert!(
                content.contains("branch-misses"),
                "Workflow should measure branch-misses"
            );
            assert!(
                content.contains("aarch64") || content.contains("arm64"),
                "Workflow should support ARM64 architecture"
            );
            assert!(
                content.contains("x86_64") || content.contains("amd64"),
                "Workflow should support x86_64 architecture"
            );
        }
    }

    #[test]
    fn test_perf_report_template_exists() {
        // Verify that performance comparison report template exists
        let template_path = Path::new("docs/perf_report_template.md");
        assert!(
            template_path.exists(),
            "Performance report template should exist at docs/perf_report_template.md"
        );
    }

    #[test]
    fn test_perf_report_template_structure() {
        let template_path = Path::new("docs/perf_report_template.md");
        if template_path.exists() {
            let content = fs::read_to_string(template_path).expect("Failed to read template file");

            // Check for required sections in report template
            assert!(
                content.contains("Cache Miss Rate"),
                "Report should include Cache Miss Rate section"
            );
            assert!(
                content.contains("Branch Prediction Miss Rate"),
                "Report should include Branch Prediction Miss Rate section"
            );
            assert!(
                content.contains("ARM64"),
                "Report should include ARM64 section"
            );
            assert!(
                content.contains("x86_64") || content.contains("x86-64"),
                "Report should include x86_64 section"
            );
            assert!(
                content.contains("Performance Targets"),
                "Report should include Performance Targets section"
            );
        }
    }

    #[test]
    fn test_perf_documentation_exists() {
        // Verify that perf profiling documentation exists
        let docs_path = Path::new("docs/perf_profiling.md");
        assert!(
            docs_path.exists(),
            "Perf profiling documentation should exist at docs/perf_profiling.md"
        );
    }

    #[test]
    fn test_perf_documentation_contains_usage() {
        let docs_path = Path::new("docs/perf_profiling.md");
        if docs_path.exists() {
            let content = fs::read_to_string(docs_path).expect("Failed to read documentation");

            // Check for key documentation sections
            assert!(
                content.contains("perf stat") || content.contains("Usage"),
                "Documentation should explain perf stat usage"
            );
            assert!(
                content.contains("cache miss") || content.contains("Cache Miss"),
                "Documentation should explain cache miss measurement"
            );
            assert!(
                content.contains("branch prediction") || content.contains("Branch Prediction"),
                "Documentation should explain branch prediction measurement"
            );
            assert!(
                content.contains("Linux"),
                "Documentation should mention Linux requirement"
            );
        }
    }
}
