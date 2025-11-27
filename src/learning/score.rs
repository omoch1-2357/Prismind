//! Score representation and conversion utilities for the learning module.
//!
//! This module provides score conversion functions and constants for the
//! TD(lambda)-Leaf learning system. It defines the mapping between u16
//! pattern table entries and f32 stone difference values.
//!
//! # Score Representation
//!
//! Pattern table entries use u16 values with the following mapping:
//! - `CENTER` (32768): Represents zero stone difference
//! - `SCALE` (256.0): Conversion factor between u16 and stone difference
//!
//! # Conversion Formulas
//!
//! - u16 to stone difference: `(value - CENTER) / SCALE`
//! - Stone difference to u16: `clamp(score * SCALE + CENTER, 0, 65535)`
//!
//! # Requirements Coverage
//!
//! - Req 11.1: u16 representation for pattern table entries (range 0-65535)
//! - Req 11.2: CENTER=32768 as zero stone-difference value
//! - Req 11.3: SCALE=256.0 for conversion factor
//! - Req 11.4: u16 to stone difference conversion formula
//! - Req 11.5: Stone difference to u16 conversion with clamping
//! - Req 11.6: Clamp updated weights to valid u16 range [0, 65535]
//! - Req 11.7: Initialize pattern table entries to 32768 (neutral evaluation)

/// Center value representing zero stone difference.
///
/// This is the neutral evaluation value for pattern table entries.
/// When a pattern entry has this value, it indicates no advantage
/// for either player.
///
/// # Value
///
/// 32768 (midpoint of u16 range 0-65535)
///
/// # Usage
///
/// - Initialize all pattern table entries to this value
/// - Use as reference point for score conversion
///
/// # Requirements
///
/// - Req 11.2: CENTER=32768 as zero stone-difference value
/// - Req 11.7: Initialize pattern table entries to 32768
pub const CENTER: u16 = 32768;

/// Scale factor for converting between u16 values and stone difference.
///
/// This determines the precision of the score representation:
/// - 1 u16 unit = 1/256 stone difference
/// - Range of representable scores: -128.0 to +127.996
///
/// # Value
///
/// 256.0
///
/// # Requirements
///
/// - Req 11.3: SCALE=256.0 for conversion factor
pub const SCALE: f32 = 256.0;

/// Converts a u16 pattern table value to stone difference (f32).
///
/// # Formula
///
/// `score = (value - 32768) / 256.0`
///
/// # Mapping
///
/// | u16 value | Stone difference |
/// |-----------|-----------------|
/// | 0         | -128.0          |
/// | 32768     | 0.0             |
/// | 65535     | +127.996        |
///
/// # Arguments
///
/// * `value` - u16 pattern table entry
///
/// # Returns
///
/// f32 stone difference (positive = black advantage, negative = white advantage)
///
/// # Examples
///
/// ```
/// use prismind::learning::score::{u16_to_stone_diff, CENTER};
///
/// assert_eq!(u16_to_stone_diff(0), -128.0);
/// assert_eq!(u16_to_stone_diff(CENTER), 0.0);
/// assert!((u16_to_stone_diff(65535) - 127.996).abs() < 0.001);
/// ```
///
/// # Requirements
///
/// - Req 11.4: u16 to stone difference: (value - 32768) / 256.0
#[inline]
pub fn u16_to_stone_diff(value: u16) -> f32 {
    (value as f32 - CENTER as f32) / SCALE
}

/// Converts stone difference (f32) to u16 pattern table value with clamping.
///
/// # Formula
///
/// `value = clamp(score * 256.0 + 32768.0, 0.0, 65535.0) as u16`
///
/// # Clamping Behavior
///
/// - Values below -128.0 are clamped to 0
/// - Values above +127.996 are clamped to 65535
///
/// # Arguments
///
/// * `score` - f32 stone difference
///
/// # Returns
///
/// u16 pattern table value in range [0, 65535]
///
/// # Examples
///
/// ```
/// use prismind::learning::score::{stone_diff_to_u16, CENTER};
///
/// assert_eq!(stone_diff_to_u16(-128.0), 0);
/// assert_eq!(stone_diff_to_u16(0.0), CENTER);
///
/// // Maximum representable value (127.99609375 maps to 65535)
/// let max_score = stone_diff_to_u16(127.99609375);
/// assert_eq!(max_score, 65535);
///
/// // Clamping behavior
/// assert_eq!(stone_diff_to_u16(-200.0), 0);      // Below range
/// assert_eq!(stone_diff_to_u16(200.0), 65535);   // Above range
/// ```
///
/// # Requirements
///
/// - Req 11.5: Stone difference to u16 with clamping to [0, 65535]
/// - Req 11.6: Clamp updated weights to valid u16 range
#[inline]
pub fn stone_diff_to_u16(score: f32) -> u16 {
    let raw = score * SCALE + CENTER as f32;
    raw.clamp(0.0, 65535.0) as u16
}

/// Returns the initial (neutral) value for pattern table entries.
///
/// All pattern table entries should be initialized to this value
/// before training begins. It represents zero stone difference.
///
/// # Returns
///
/// 32768 (CENTER value)
///
/// # Examples
///
/// ```
/// use prismind::learning::score::{initial_value, CENTER};
///
/// assert_eq!(initial_value(), CENTER);
/// assert_eq!(initial_value(), 32768);
/// ```
///
/// # Requirements
///
/// - Req 11.7: Initialize pattern table entries to 32768
#[inline]
pub const fn initial_value() -> u16 {
    CENTER
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Requirement 11.2: CENTER constant ==========

    #[test]
    fn test_center_constant_is_32768() {
        // Req 11.2: CENTER=32768 as zero stone-difference value
        assert_eq!(CENTER, 32768, "CENTER constant should be 32768");
    }

    #[test]
    fn test_center_is_midpoint_of_u16_range() {
        // CENTER should be the midpoint of u16 range
        assert_eq!(
            CENTER,
            u16::MAX / 2 + 1,
            "CENTER should be midpoint of u16 range"
        );
    }

    // ========== Requirement 11.3: SCALE constant ==========

    #[test]
    fn test_scale_constant_is_256() {
        // Req 11.3: SCALE=256.0 for conversion factor
        assert_eq!(SCALE, 256.0, "SCALE constant should be 256.0");
    }

    // ========== Requirement 11.4: u16 to stone difference conversion ==========

    #[test]
    fn test_u16_to_stone_diff_zero_maps_to_minus_128() {
        // Req 11.4: u16 value 0 -> stone difference -128.0
        let score = u16_to_stone_diff(0);
        assert_eq!(score, -128.0, "u16 value 0 should map to stone diff -128.0");
    }

    #[test]
    fn test_u16_to_stone_diff_center_maps_to_zero() {
        // Req 11.4: u16 value 32768 -> stone difference 0.0
        let score = u16_to_stone_diff(CENTER);
        assert_eq!(
            score, 0.0,
            "u16 value CENTER (32768) should map to stone diff 0.0"
        );
    }

    #[test]
    fn test_u16_to_stone_diff_max_maps_to_approx_128() {
        // Req 11.4: u16 value 65535 -> stone difference ~127.996
        let score = u16_to_stone_diff(65535);
        let expected = (65535.0 - 32768.0) / 256.0;
        assert!(
            (score - expected).abs() < 0.001,
            "u16 value 65535 should map to stone diff ~127.996, got {}",
            score
        );
    }

    #[test]
    fn test_u16_to_stone_diff_formula() {
        // Req 11.4: Verify formula (value - 32768) / 256.0
        // Calculate expected values: (value - 32768) / 256.0
        // 10000: (10000 - 32768) / 256.0 = -22768 / 256.0 = -88.9375
        // 60000: (60000 - 32768) / 256.0 = 27232 / 256.0 = 106.375
        let test_values: [(u16, f32); 7] = [
            (0, -128.0),
            (10000, -88.9375),
            (16384, -64.0),
            (32768, 0.0),
            (49152, 64.0),
            (60000, 106.375),
            (65535, 127.996_09),
        ];

        for (u16_val, expected) in test_values {
            let score = u16_to_stone_diff(u16_val);
            assert!(
                (score - expected).abs() < 0.0001,
                "u16_to_stone_diff({}) should be {}, got {}",
                u16_val,
                expected,
                score
            );
        }
    }

    // ========== Requirement 11.5: Stone difference to u16 conversion ==========

    #[test]
    fn test_stone_diff_to_u16_minus_128_maps_to_zero() {
        // Req 11.5: stone difference -128.0 -> u16 value 0
        let u16_val = stone_diff_to_u16(-128.0);
        assert_eq!(u16_val, 0, "Stone diff -128.0 should map to u16 value 0");
    }

    #[test]
    fn test_stone_diff_to_u16_zero_maps_to_center() {
        // Req 11.5: stone difference 0.0 -> u16 value 32768
        let u16_val = stone_diff_to_u16(0.0);
        assert_eq!(
            u16_val, CENTER,
            "Stone diff 0.0 should map to u16 value CENTER (32768)"
        );
    }

    #[test]
    fn test_stone_diff_to_u16_max_maps_to_65535() {
        // Req 11.5: stone difference 127.996 -> u16 value 65535
        let u16_val = stone_diff_to_u16(127.996);
        assert!(
            u16_val >= 65534,
            "Stone diff 127.996 should map to u16 value ~65535, got {}",
            u16_val
        );
    }

    #[test]
    fn test_stone_diff_to_u16_formula() {
        // Req 11.5: Verify formula clamp(score * 256.0 + 32768.0, 0.0, 65535.0)
        let test_scores: [(f32, u16); 5] = [
            (-128.0, 0),
            (-64.0, 16384),
            (0.0, 32768),
            (64.0, 49152),
            (127.0, 65280),
        ];

        for (score, expected) in test_scores {
            let u16_val = stone_diff_to_u16(score);
            assert_eq!(
                u16_val, expected,
                "stone_diff_to_u16({}) should be {}, got {}",
                score, expected, u16_val
            );
        }
    }

    // ========== Requirement 11.6: Clamping behavior ==========

    #[test]
    fn test_stone_diff_to_u16_clamps_below_range() {
        // Req 11.6: Values below -128.0 are clamped to 0
        assert_eq!(
            stone_diff_to_u16(-200.0),
            0,
            "Score -200.0 should clamp to 0"
        );
        assert_eq!(
            stone_diff_to_u16(-500.0),
            0,
            "Score -500.0 should clamp to 0"
        );
        assert_eq!(
            stone_diff_to_u16(-128.1),
            0,
            "Score -128.1 should clamp to 0"
        );
    }

    #[test]
    fn test_stone_diff_to_u16_clamps_above_range() {
        // Req 11.6: Values above +127.996 are clamped to 65535
        assert_eq!(
            stone_diff_to_u16(200.0),
            65535,
            "Score 200.0 should clamp to 65535"
        );
        assert_eq!(
            stone_diff_to_u16(500.0),
            65535,
            "Score 500.0 should clamp to 65535"
        );
        assert_eq!(
            stone_diff_to_u16(128.0),
            65535,
            "Score 128.0 should clamp to 65535"
        );
    }

    #[test]
    fn test_stone_diff_to_u16_boundary_clamping() {
        // Req 11.6: Test exact boundary clamping
        // At -128.0: score * 256.0 + 32768.0 = -32768.0 + 32768.0 = 0.0
        assert_eq!(stone_diff_to_u16(-128.0), 0);

        // Just below -128.0 should also be 0
        assert_eq!(stone_diff_to_u16(-128.001), 0);

        // Just at max representable score
        let max_score = (65535.0 - 32768.0) / 256.0;
        let u16_at_max = stone_diff_to_u16(max_score);
        assert!(
            u16_at_max >= 65534,
            "Max representable score should map to ~65535"
        );
    }

    // ========== Requirement 11.7: Initial value ==========

    #[test]
    fn test_initial_value_is_center() {
        // Req 11.7: Initialize pattern table entries to 32768
        assert_eq!(initial_value(), CENTER);
        assert_eq!(initial_value(), 32768);
    }

    #[test]
    fn test_initial_value_represents_zero_stone_diff() {
        // Req 11.7: Initial value should represent zero stone difference
        let score = u16_to_stone_diff(initial_value());
        assert_eq!(score, 0.0, "Initial value should represent stone diff 0.0");
    }

    // ========== Round-trip conversion tests ==========

    #[test]
    fn test_round_trip_conversion_u16_to_f32_to_u16() {
        // Test that u16 -> f32 -> u16 preserves the value (within +-1)
        let test_values: [u16; 10] = [
            0, 100, 1000, 10000, 16384, 32768, 49152, 60000, 65000, 65535,
        ];

        for original in test_values {
            let score = u16_to_stone_diff(original);
            let back_to_u16 = stone_diff_to_u16(score);

            let diff = original.abs_diff(back_to_u16);

            assert!(
                diff <= 1,
                "Round trip failed: {} -> {} -> {}, diff = {}",
                original,
                score,
                back_to_u16,
                diff
            );
        }
    }

    #[test]
    fn test_round_trip_conversion_f32_to_u16_to_f32() {
        // Test that f32 -> u16 -> f32 preserves the value (within precision limits)
        let test_scores: [f32; 9] = [-128.0, -64.0, -10.0, -1.0, 0.0, 1.0, 10.0, 64.0, 127.0];

        for original in test_scores {
            let u16_val = stone_diff_to_u16(original);
            let back_to_f32 = u16_to_stone_diff(u16_val);

            // Precision is 1/256 = ~0.0039
            let precision = 1.0 / SCALE;
            let diff = (original - back_to_f32).abs();

            assert!(
                diff < precision,
                "Round trip failed: {} -> {} -> {}, diff = {}",
                original,
                u16_val,
                back_to_f32,
                diff
            );
        }
    }

    // ========== Edge case tests ==========

    #[test]
    fn test_conversion_handles_small_positive_scores() {
        // Test small positive stone differences
        // Note: Precision is 1/256 = ~0.0039, so very small values may round to CENTER
        // 0.001 * 256 = 0.256 -> rounds to 0 difference -> still CENTER
        // 0.01 * 256 = 2.56 -> adds ~3 to CENTER
        let small_scores = [0.01, 0.1, 0.5, 1.0];

        for score in small_scores {
            let u16_val = stone_diff_to_u16(score);
            assert!(
                u16_val > CENTER,
                "Small positive score {} should map to > CENTER",
                score
            );
        }

        // Very small score (below precision) may equal CENTER
        let tiny_score = 0.001;
        let u16_val = stone_diff_to_u16(tiny_score);
        assert!(
            u16_val >= CENTER,
            "Tiny positive score {} should map to >= CENTER, got {}",
            tiny_score,
            u16_val
        );
    }

    #[test]
    fn test_conversion_handles_small_negative_scores() {
        // Test small negative stone differences
        let small_scores = [-0.001, -0.01, -0.1, -0.5, -1.0];

        for score in small_scores {
            let u16_val = stone_diff_to_u16(score);
            assert!(
                u16_val < CENTER,
                "Small negative score {} should map to < CENTER",
                score
            );
        }
    }

    #[test]
    fn test_u16_to_stone_diff_is_monotonic() {
        // Verify that the conversion is monotonically increasing
        let mut prev_score = f32::NEG_INFINITY;

        for u16_val in (0..=65535u16).step_by(1000) {
            let score = u16_to_stone_diff(u16_val);
            assert!(
                score > prev_score,
                "Conversion should be monotonic: {} -> {} should be > {}",
                u16_val,
                score,
                prev_score
            );
            prev_score = score;
        }
    }

    #[test]
    fn test_stone_diff_to_u16_is_monotonic_in_valid_range() {
        // Verify that the conversion is monotonically increasing in valid range
        let mut prev_u16 = 0u16;

        // Test scores in valid range [-128, 127.996]
        let mut score = -128.0;
        while score < 127.0 {
            let u16_val = stone_diff_to_u16(score);
            assert!(
                u16_val >= prev_u16,
                "Conversion should be monotonic: {} -> {} should be >= {}",
                score,
                u16_val,
                prev_u16
            );
            prev_u16 = u16_val;
            score += 10.0;
        }
    }

    // ========== Special value tests ==========

    #[test]
    fn test_stone_diff_to_u16_handles_nan() {
        // NaN should be clamped (behavior depends on f32::clamp implementation)
        let u16_val = stone_diff_to_u16(f32::NAN);
        // NaN comparisons always return false, so clamp returns 0.0
        assert!(
            u16_val == 0 || u16_val == 65535,
            "NaN should be clamped to boundary"
        );
    }

    #[test]
    fn test_stone_diff_to_u16_handles_infinity() {
        // Positive infinity should clamp to 65535
        assert_eq!(stone_diff_to_u16(f32::INFINITY), 65535);

        // Negative infinity should clamp to 0
        assert_eq!(stone_diff_to_u16(f32::NEG_INFINITY), 0);
    }

    // ========== Requirements summary test ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== Score Representation Requirements Verification ===");

        // Req 11.1: u16 representation
        let _: u16 = stone_diff_to_u16(0.0);
        println!("  11.1: u16 representation for pattern table entries");

        // Req 11.2: CENTER constant
        assert_eq!(CENTER, 32768);
        println!("  11.2: CENTER=32768 as zero stone-difference value");

        // Req 11.3: SCALE constant
        assert_eq!(SCALE, 256.0);
        println!("  11.3: SCALE=256.0 for conversion factor");

        // Req 11.4: u16 to stone difference
        assert_eq!(u16_to_stone_diff(32768), 0.0);
        println!("  11.4: u16 to stone difference: (value - 32768) / 256.0");

        // Req 11.5: Stone difference to u16 with clamping
        assert_eq!(stone_diff_to_u16(0.0), 32768);
        println!("  11.5: Stone difference to u16 with clamping to [0, 65535]");

        // Req 11.6: Clamping behavior
        assert_eq!(stone_diff_to_u16(-200.0), 0);
        assert_eq!(stone_diff_to_u16(200.0), 65535);
        println!("  11.6: Clamp updated weights to valid u16 range");

        // Req 11.7: Initial value
        assert_eq!(initial_value(), 32768);
        println!("  11.7: Initialize pattern table entries to 32768 (neutral)");

        println!("=== All score requirements verified ===");
    }
}
