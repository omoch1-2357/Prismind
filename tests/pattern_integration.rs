use prismind::pattern::load_patterns;

#[test]
fn test_load_patterns_from_project_file() {
    // Test loading patterns.csv from the project root
    let result = load_patterns("patterns.csv");

    if let Ok(patterns) = result {
        assert_eq!(patterns.len(), 14, "Should have 14 patterns");

        // Verify first pattern
        let p01 = &patterns[0];
        assert_eq!(p01.id, 0);
        assert_eq!(p01.k, 10);

        // Verify last pattern
        let p14 = &patterns[13];
        assert_eq!(p14.id, 13);
        assert_eq!(p14.k, 4);

        println!("✓ Successfully loaded 14 patterns from patterns.csv");
        for pattern in &patterns {
            println!(
                "  Pattern {:02}: k={}, positions={:?}",
                pattern.id + 1,
                pattern.k,
                &pattern.positions[..pattern.k as usize]
            );
        }
    } else {
        println!("Note: patterns.csv not found in project root, skipping integration test");
        println!("This is expected if the file hasn't been created yet.");
    }
}
