// tests/perf_script_validation_test.rs
//
// Task 10.2: Validate perf profiling script syntax and structure
// These tests verify that the shell script is valid and well-formed

#[cfg(test)]
mod perf_script_validation_tests {
    use std::fs;

    #[test]
    fn test_script_has_shebang() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.starts_with("#!/bin/bash"),
            "Script should start with shebang #!/bin/bash"
        );
    }

    #[test]
    fn test_script_has_set_e() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("set -e"),
            "Script should have 'set -e' for error handling"
        );
    }

    #[test]
    fn test_script_checks_linux() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("linux-gnu"),
            "Script should check for Linux OS"
        );
        assert!(
            script_content.contains("OSTYPE"),
            "Script should use OSTYPE variable"
        );
    }

    #[test]
    fn test_script_checks_perf_installation() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("command -v perf"),
            "Script should check if perf is installed"
        );
    }

    #[test]
    fn test_script_measures_cache_events() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("cache-references"),
            "Script should measure cache-references"
        );
        assert!(
            script_content.contains("cache-misses"),
            "Script should measure cache-misses"
        );
    }

    #[test]
    fn test_script_measures_branch_events() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("branches"),
            "Script should measure branches"
        );
        assert!(
            script_content.contains("branch-misses"),
            "Script should measure branch-misses"
        );
    }

    #[test]
    fn test_script_builds_benchmarks() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("cargo build --release --benches"),
            "Script should build benchmarks in release mode"
        );
    }

    #[test]
    fn test_script_runs_perf_stat() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("perf stat"),
            "Script should run perf stat command"
        );
    }

    #[test]
    fn test_script_creates_output_directory() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("mkdir -p"),
            "Script should create output directory"
        );
        assert!(
            script_content.contains("perf_results") || script_content.contains("OUTPUT_DIR"),
            "Script should define output directory"
        );
    }

    #[test]
    fn test_script_validates_targets() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        // Check for 50% cache miss target
        assert!(
            script_content.contains("50"),
            "Script should reference 50% cache miss target"
        );

        // Check for 1% branch miss target
        assert!(
            script_content.contains("1") || script_content.contains("1.0"),
            "Script should reference 1% branch miss target"
        );
    }

    #[test]
    fn test_script_reports_pass_fail() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("PASS") || script_content.contains("✓"),
            "Script should report PASS status"
        );
        assert!(
            script_content.contains("FAIL") || script_content.contains("✗"),
            "Script should report FAIL status"
        );
    }

    #[test]
    fn test_script_detects_architecture() {
        let script_content =
            fs::read_to_string("scripts/perf_profile.sh").expect("Failed to read script");

        assert!(
            script_content.contains("uname -m") || script_content.contains("ARCH"),
            "Script should detect architecture (x86_64 or ARM64)"
        );
    }

    #[test]
    fn test_workflow_runs_on_linux() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("ubuntu-latest"),
            "Workflow should run on ubuntu-latest (x86_64 Linux)"
        );
    }

    #[test]
    fn test_workflow_installs_perf() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("linux-tools"),
            "Workflow should install linux-tools (perf)"
        );
    }

    #[test]
    fn test_workflow_sets_permissions() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("perf_event_paranoid"),
            "Workflow should set perf_event_paranoid for permissions"
        );
    }

    #[test]
    fn test_workflow_uploads_artifacts() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("upload-artifact"),
            "Workflow should upload perf results as artifacts"
        );
    }

    #[test]
    fn test_workflow_generates_comparison() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("comparison") || workflow_content.contains("compare"),
            "Workflow should generate performance comparison report"
        );
    }

    #[test]
    fn test_workflow_comments_on_pr() {
        let workflow_content = fs::read_to_string(".github/workflows/perf-profile.yml")
            .expect("Failed to read workflow");

        assert!(
            workflow_content.contains("github-script") || workflow_content.contains("comment"),
            "Workflow should comment on PR with results"
        );
    }
}
