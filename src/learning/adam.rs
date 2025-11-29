//! Adam Optimizer for TD(lambda)-Leaf learning.
//!
//! This module implements the Adam optimizer with momentum and adaptive learning rate
//! for stable gradient updates during reinforcement learning.
//!
//! # Overview
//!
//! Adam (Adaptive Moment Estimation) maintains two moment estimates:
//! - First moment (m): Exponential moving average of gradients
//! - Second moment (v): Exponential moving average of squared gradients
//!
//! # Memory Layout
//!
//! Moment vectors are stored in Structure of Arrays (SoA) format matching
//! the EvaluationTable layout for cache efficiency:
//! - Each stage has a flat array containing all pattern entries
//! - pattern_offsets track where each pattern starts within the flat array
//!
//! # Requirements Coverage
//!
//! - Req 3.1: Learning rate alpha=0.025
//! - Req 3.2: beta1=0.9 for first moment decay
//! - Req 3.3: beta2=0.999 for second moment decay
//! - Req 3.4: epsilon=1e-8 for numerical stability
//! - Req 3.5: First moment vectors (~114 MB)
//! - Req 3.6: Second moment vectors (~114 MB)
//! - Req 3.7: Global timestep counter for bias correction
//! - Req 3.8: Bias correction: m_hat = m / (1 - beta1^t)
//! - Req 3.9: Parameter update: delta = alpha * m_hat / (sqrt(v_hat) + epsilon)
//! - Req 3.10: Initialize all moments to 0.0

use crate::pattern::Pattern;

/// Default learning rate (alpha)
pub const DEFAULT_ALPHA: f32 = 0.025;

/// Default first moment decay rate (beta1)
pub const DEFAULT_BETA1: f32 = 0.9;

/// Default second moment decay rate (beta2)
pub const DEFAULT_BETA2: f32 = 0.999;

/// Default epsilon for numerical stability
pub const DEFAULT_EPSILON: f32 = 1e-8;

/// Number of stages in the evaluation table
pub const NUM_STAGES: usize = 30;

/// Number of patterns
pub const NUM_PATTERNS: usize = 14;

/// Moment storage in Structure of Arrays (SoA) format.
///
/// Matches the EvaluationTable layout: \[stage\]\[flat_array\]
/// where flat_array contains all pattern entries concatenated.
#[derive(Debug, Clone)]
pub struct AdamMoments {
    /// \[stage\]\[flat_array\] of moment values
    data: Vec<Box<[f32]>>,
    /// Offset for each pattern within the flat array
    pattern_offsets: [usize; NUM_PATTERNS],
    /// Total entries per stage
    entries_per_stage: usize,
}

impl AdamMoments {
    /// Create new moment storage initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Pattern definitions (14 patterns)
    ///
    /// # Requirements
    ///
    /// - Req 3.10: Initialize all moment values to 0.0
    pub fn new(patterns: &[Pattern]) -> Self {
        assert_eq!(patterns.len(), NUM_PATTERNS, "Expected 14 patterns");

        // Calculate pattern offsets
        let mut pattern_offsets = [0usize; NUM_PATTERNS];
        let mut offset = 0;
        for (i, pattern) in patterns.iter().enumerate() {
            pattern_offsets[i] = offset;
            offset += 3_usize.pow(pattern.k as u32);
        }
        let entries_per_stage = offset;

        // Initialize data for 30 stages, all zeros
        let mut data = Vec::with_capacity(NUM_STAGES);
        for _ in 0..NUM_STAGES {
            let stage_data = vec![0.0f32; entries_per_stage].into_boxed_slice();
            data.push(stage_data);
        }

        Self {
            data,
            pattern_offsets,
            entries_per_stage,
        }
    }

    /// Get moment value at specified location.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern ID (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index
    #[inline]
    pub fn get(&self, pattern_id: usize, stage: usize, index: usize) -> f32 {
        debug_assert!(pattern_id < NUM_PATTERNS);
        debug_assert!(stage < NUM_STAGES);
        let offset = self.pattern_offsets[pattern_id] + index;
        self.data[stage][offset]
    }

    /// Set moment value at specified location.
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern ID (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index
    /// * `value` - New moment value
    #[inline]
    pub fn set(&mut self, pattern_id: usize, stage: usize, index: usize, value: f32) {
        debug_assert!(pattern_id < NUM_PATTERNS);
        debug_assert!(stage < NUM_STAGES);
        let offset = self.pattern_offsets[pattern_id] + index;
        self.data[stage][offset] = value;
    }

    /// Get memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        NUM_STAGES * self.entries_per_stage * std::mem::size_of::<f32>()
    }

    /// Reset all moment values to zero.
    pub fn reset(&mut self) {
        for stage_data in &mut self.data {
            for value in stage_data.iter_mut() {
                *value = 0.0;
            }
        }
    }
}

/// Adam optimizer state and update logic.
///
/// Implements the Adam optimization algorithm with:
/// - Exponential moving average of gradients (first moment)
/// - Exponential moving average of squared gradients (second moment)
/// - Bias correction for early timesteps
///
/// # Memory Usage
///
/// Total: ~228 MB
/// - First moment (m): ~114 MB
/// - Second moment (v): ~114 MB
///
/// # Example
///
/// ```ignore
/// use prismind::learning::adam::AdamOptimizer;
///
/// let patterns = load_patterns("patterns.csv")?;
/// let mut adam = AdamOptimizer::new(&patterns);
///
/// // Update a parameter
/// let current_value = 32768.0;
/// let gradient = 0.5;
/// let new_value = adam.update(0, 5, 100, current_value, gradient);
/// ```
#[derive(Debug)]
pub struct AdamOptimizer {
    /// Learning rate
    alpha: f32,
    /// First moment decay rate
    beta1: f32,
    /// Second moment decay rate
    beta2: f32,
    /// Numerical stability term
    epsilon: f32,
    /// Global timestep counter
    t: u64,
    /// First moment vectors
    m: AdamMoments,
    /// Second moment vectors
    v: AdamMoments,
}

impl AdamOptimizer {
    /// Create new optimizer with default hyperparameters.
    ///
    /// Default values:
    /// - alpha = 0.025
    /// - beta1 = 0.9
    /// - beta2 = 0.999
    /// - epsilon = 1e-8
    ///
    /// # Arguments
    ///
    /// * `patterns` - Pattern definitions (14 patterns)
    ///
    /// # Requirements
    ///
    /// - Req 3.1: alpha=0.025
    /// - Req 3.2: beta1=0.9
    /// - Req 3.3: beta2=0.999
    /// - Req 3.4: epsilon=1e-8
    /// - Req 3.5, 3.6: Allocate moment vectors
    /// - Req 3.10: Initialize all to 0.0
    pub fn new(patterns: &[Pattern]) -> Self {
        Self::with_params(
            DEFAULT_ALPHA,
            DEFAULT_BETA1,
            DEFAULT_BETA2,
            DEFAULT_EPSILON,
            patterns,
        )
    }

    /// Create optimizer with custom hyperparameters.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Learning rate
    /// * `beta1` - First moment decay rate
    /// * `beta2` - Second moment decay rate
    /// * `epsilon` - Numerical stability term
    /// * `patterns` - Pattern definitions (14 patterns)
    pub fn with_params(
        alpha: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        patterns: &[Pattern],
    ) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: AdamMoments::new(patterns),
            v: AdamMoments::new(patterns),
        }
    }

    /// Apply Adam update for a single parameter.
    ///
    /// Computes the bias-corrected update using the Adam algorithm:
    /// 1. Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
    /// 2. Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
    /// 3. Compute bias-corrected moments: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    /// 4. Compute update: delta = alpha * m_hat / (sqrt(v_hat) + epsilon)
    /// 5. Return: new_value = current_value + delta
    ///
    /// # Arguments
    ///
    /// * `pattern_id` - Pattern ID (0-13)
    /// * `stage` - Game stage (0-29)
    /// * `index` - Pattern index
    /// * `current_value` - Current parameter value (as f32)
    /// * `gradient` - Gradient for this parameter
    ///
    /// # Returns
    ///
    /// New parameter value after Adam update.
    ///
    /// # Requirements
    ///
    /// - Req 3.8: Bias correction
    /// - Req 3.9: Parameter update formula
    #[inline]
    pub fn update(
        &mut self,
        pattern_id: usize,
        stage: usize,
        index: usize,
        current_value: f32,
        gradient: f32,
    ) -> f32 {
        // Get current moment values
        let m_old = self.m.get(pattern_id, stage, index);
        let v_old = self.v.get(pattern_id, stage, index);

        // Update biased first moment estimate
        let m_new = self.beta1 * m_old + (1.0 - self.beta1) * gradient;

        // Update biased second moment estimate
        let v_new = self.beta2 * v_old + (1.0 - self.beta2) * gradient * gradient;

        // Store updated moments
        self.m.set(pattern_id, stage, index, m_new);
        self.v.set(pattern_id, stage, index, v_new);

        // Compute bias-corrected moment estimates
        // Use t+1 since we increment after batch
        let t_effective = (self.t + 1) as f32;
        let m_hat = m_new / (1.0 - self.beta1.powf(t_effective));
        let v_hat = v_new / (1.0 - self.beta2.powf(t_effective));

        // Compute update
        let delta = self.alpha * m_hat / (v_hat.sqrt() + self.epsilon);

        // Return new value
        current_value + delta
    }

    /// Increment the global timestep counter.
    ///
    /// Should be called once per game (or batch) after all updates.
    ///
    /// # Requirements
    ///
    /// - Req 3.7: Global timestep counter for bias correction
    pub fn step(&mut self) {
        self.t += 1;
    }

    /// Get current timestep.
    ///
    /// # Returns
    ///
    /// Current timestep value.
    pub fn timestep(&self) -> u64 {
        self.t
    }

    /// Set the timestep counter.
    ///
    /// Used when restoring from checkpoint.
    ///
    /// # Arguments
    ///
    /// * `t` - Timestep value to set
    pub fn set_timestep(&mut self, t: u64) {
        self.t = t;
    }

    /// Get total memory usage in bytes.
    ///
    /// # Returns
    ///
    /// Approximate memory usage in bytes.
    ///
    /// # Requirements
    ///
    /// - Req 3.5: First moment ~114 MB
    /// - Req 3.6: Second moment ~114 MB
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage() + self.v.memory_usage()
    }

    /// Get learning rate (alpha).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get first moment decay rate (beta1).
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get second moment decay rate (beta2).
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get epsilon value.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get reference to first moment storage.
    pub fn first_moment(&self) -> &AdamMoments {
        &self.m
    }

    /// Get reference to second moment storage.
    pub fn second_moment(&self) -> &AdamMoments {
        &self.v
    }

    /// Get mutable reference to first moment storage.
    ///
    /// Used for checkpoint restore.
    pub fn first_moment_mut(&mut self) -> &mut AdamMoments {
        &mut self.m
    }

    /// Get mutable reference to second moment storage.
    ///
    /// Used for checkpoint restore.
    pub fn second_moment_mut(&mut self) -> &mut AdamMoments {
        &mut self.v
    }

    /// Reset optimizer state.
    ///
    /// Clears all moment values and resets timestep to 0.
    pub fn reset(&mut self) {
        self.t = 0;
        self.m.reset();
        self.v.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test patterns for testing.
    fn create_test_patterns() -> Vec<Pattern> {
        vec![
            Pattern::new(0, 10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap(),
            Pattern::new(1, 10, vec![0, 8, 16, 24, 32, 40, 48, 56, 1, 9]).unwrap(),
            Pattern::new(2, 10, vec![0, 1, 8, 9, 10, 16, 17, 18, 24, 25]).unwrap(),
            Pattern::new(3, 10, vec![0, 9, 18, 27, 36, 45, 54, 63, 1, 10]).unwrap(),
            Pattern::new(4, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            Pattern::new(5, 8, vec![0, 8, 16, 24, 32, 40, 48, 56]).unwrap(),
            Pattern::new(6, 8, vec![0, 9, 18, 27, 36, 45, 54, 63]).unwrap(),
            Pattern::new(7, 8, vec![7, 14, 21, 28, 35, 42, 49, 56]).unwrap(),
            Pattern::new(8, 7, vec![0, 1, 2, 3, 4, 5, 6]).unwrap(),
            Pattern::new(9, 7, vec![0, 8, 16, 24, 32, 40, 48]).unwrap(),
            Pattern::new(10, 6, vec![0, 1, 2, 3, 4, 5]).unwrap(),
            Pattern::new(11, 6, vec![0, 8, 16, 24, 32, 40]).unwrap(),
            Pattern::new(12, 5, vec![0, 1, 2, 3, 4]).unwrap(),
            Pattern::new(13, 5, vec![0, 8, 16, 24, 32]).unwrap(),
        ]
    }

    // ========== Requirement 3.1: alpha=0.025 ==========

    #[test]
    fn test_default_alpha() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);
        assert_eq!(adam.alpha(), DEFAULT_ALPHA);
        assert_eq!(adam.alpha(), 0.025);
    }

    // ========== Requirement 3.2: beta1=0.9 ==========

    #[test]
    fn test_default_beta1() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);
        assert_eq!(adam.beta1(), DEFAULT_BETA1);
        assert_eq!(adam.beta1(), 0.9);
    }

    // ========== Requirement 3.3: beta2=0.999 ==========

    #[test]
    fn test_default_beta2() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);
        assert_eq!(adam.beta2(), DEFAULT_BETA2);
        assert_eq!(adam.beta2(), 0.999);
    }

    // ========== Requirement 3.4: epsilon=1e-8 ==========

    #[test]
    fn test_default_epsilon() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);
        assert_eq!(adam.epsilon(), DEFAULT_EPSILON);
        assert_eq!(adam.epsilon(), 1e-8);
    }

    // ========== Requirement 3.5, 3.6: Moment vector allocation ==========

    #[test]
    fn test_moment_vectors_allocated() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);

        // Both moment vectors should be allocated
        assert!(adam.memory_usage() > 0);

        // Should have data for all stages
        let m_mem = adam.first_moment().memory_usage();
        let v_mem = adam.second_moment().memory_usage();

        // Each should be roughly half of total
        assert_eq!(m_mem, v_mem);
        assert_eq!(adam.memory_usage(), m_mem + v_mem);
    }

    #[test]
    fn test_moment_vector_memory_size() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);

        // Calculate expected size
        let total_entries_per_stage: usize = patterns.iter().map(|p| 3_usize.pow(p.k as u32)).sum();
        let expected_per_vector = NUM_STAGES * total_entries_per_stage * std::mem::size_of::<f32>();

        let m_mem = adam.first_moment().memory_usage();
        assert_eq!(m_mem, expected_per_vector);

        // Total should be approximately 228 MB for full patterns
        // For test patterns, should be smaller
        println!(
            "Moment vector size: {} bytes ({:.2} MB)",
            m_mem,
            m_mem as f64 / 1_048_576.0
        );
        println!(
            "Total Adam memory: {} bytes ({:.2} MB)",
            adam.memory_usage(),
            adam.memory_usage() as f64 / 1_048_576.0
        );
    }

    // ========== Requirement 3.7: Global timestep counter ==========

    #[test]
    fn test_timestep_counter_initial() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);
        assert_eq!(adam.timestep(), 0);
    }

    #[test]
    fn test_timestep_counter_increment() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        adam.step();
        assert_eq!(adam.timestep(), 1);

        adam.step();
        assert_eq!(adam.timestep(), 2);

        for _ in 0..100 {
            adam.step();
        }
        assert_eq!(adam.timestep(), 102);
    }

    #[test]
    fn test_timestep_counter_set() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        adam.set_timestep(1000);
        assert_eq!(adam.timestep(), 1000);
    }

    // ========== Requirement 3.8: Bias correction ==========

    #[test]
    fn test_bias_correction_early_timestep() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        // At t=1, bias correction should be significant
        let current_value = 0.0;
        let gradient = 1.0;

        let new_value = adam.update(0, 0, 0, current_value, gradient);
        adam.step();

        // With bias correction, the update should be larger than without
        // m_hat = m / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
        // v_hat = v / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
        // delta = 0.025 * 1.0 / (sqrt(1.0) + 1e-8) = 0.025
        assert!(
            (new_value - 0.025).abs() < 1e-5,
            "Expected ~0.025, got {}",
            new_value
        );
    }

    // ========== Requirement 3.9: Parameter update formula ==========

    #[test]
    fn test_update_positive_gradient() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        let current_value = 32768.0;
        let gradient = 1.0;

        let new_value = adam.update(0, 0, 0, current_value, gradient);

        // Update should increase the value for positive gradient
        assert!(new_value > current_value);
    }

    #[test]
    fn test_update_negative_gradient() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        let current_value = 32768.0;
        let gradient = -1.0;

        let new_value = adam.update(0, 0, 0, current_value, gradient);

        // Update should decrease the value for negative gradient
        assert!(new_value < current_value);
    }

    #[test]
    fn test_update_zero_gradient() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        let current_value = 32768.0;
        let gradient = 0.0;

        let new_value = adam.update(0, 0, 0, current_value, gradient);

        // With zero gradient, value should not change (or minimal change)
        assert!(
            (new_value - current_value).abs() < 1e-5,
            "Expected no change, got {} -> {}",
            current_value,
            new_value
        );
    }

    #[test]
    fn test_update_multiple_times() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        let mut value = 0.0;
        let gradient = 1.0;

        // Apply multiple updates with same gradient
        for _ in 0..10 {
            value = adam.update(0, 0, 0, value, gradient);
            adam.step();
        }

        // Value should have increased significantly
        assert!(value > 0.1, "Value should increase after multiple updates");
        println!("Value after 10 updates with gradient=1.0: {}", value);
    }

    // ========== Requirement 3.10: Initialize all to 0.0 ==========

    #[test]
    fn test_moments_initialized_to_zero() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::new(&patterns);

        // Check various locations are initialized to 0.0
        for pattern_id in 0..NUM_PATTERNS {
            for stage in [0, 15, 29] {
                assert_eq!(adam.first_moment().get(pattern_id, stage, 0), 0.0);
                assert_eq!(adam.second_moment().get(pattern_id, stage, 0), 0.0);
            }
        }
    }

    #[test]
    fn test_moments_updated_after_update() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        // Before update, moments are zero
        assert_eq!(adam.first_moment().get(0, 0, 0), 0.0);
        assert_eq!(adam.second_moment().get(0, 0, 0), 0.0);

        // After update
        adam.update(0, 0, 0, 0.0, 1.0);

        // Moments should be non-zero
        // m = 0.9 * 0 + 0.1 * 1.0 = 0.1
        // v = 0.999 * 0 + 0.001 * 1.0 = 0.001
        let m = adam.first_moment().get(0, 0, 0);
        let v = adam.second_moment().get(0, 0, 0);

        assert!((m - 0.1).abs() < 1e-6, "m should be 0.1, got {}", m);
        assert!((v - 0.001).abs() < 1e-6, "v should be 0.001, got {}", v);
    }

    // ========== Reset tests ==========

    #[test]
    fn test_reset() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        // Update some values
        adam.update(0, 0, 0, 0.0, 1.0);
        adam.step();
        adam.step();

        assert!(adam.first_moment().get(0, 0, 0) != 0.0);
        assert!(adam.timestep() > 0);

        // Reset
        adam.reset();

        // Everything should be zero
        assert_eq!(adam.first_moment().get(0, 0, 0), 0.0);
        assert_eq!(adam.second_moment().get(0, 0, 0), 0.0);
        assert_eq!(adam.timestep(), 0);
    }

    // ========== Custom hyperparameters test ==========

    #[test]
    fn test_custom_hyperparameters() {
        let patterns = create_test_patterns();
        let adam = AdamOptimizer::with_params(0.001, 0.8, 0.99, 1e-7, &patterns);

        assert_eq!(adam.alpha(), 0.001);
        assert_eq!(adam.beta1(), 0.8);
        assert_eq!(adam.beta2(), 0.99);
        assert_eq!(adam.epsilon(), 1e-7);
    }

    // ========== Different pattern/stage/index tests ==========

    #[test]
    fn test_update_different_locations() {
        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        // Update different locations
        adam.update(0, 0, 0, 0.0, 1.0);
        adam.update(5, 15, 100, 0.0, -1.0);
        adam.update(13, 29, 200, 0.0, 0.5);

        // Check that moments are updated independently
        let m1 = adam.first_moment().get(0, 0, 0);
        let m2 = adam.first_moment().get(5, 15, 100);
        let m3 = adam.first_moment().get(13, 29, 200);

        assert!(m1 > 0.0);
        assert!(m2 < 0.0);
        assert!(m3 > 0.0);

        // Other locations should still be zero
        assert_eq!(adam.first_moment().get(0, 0, 1), 0.0);
        assert_eq!(adam.first_moment().get(1, 0, 0), 0.0);
    }

    // ========== AdamMoments tests ==========

    #[test]
    fn test_adam_moments_get_set() {
        let patterns = create_test_patterns();
        let mut moments = AdamMoments::new(&patterns);

        assert_eq!(moments.get(0, 0, 0), 0.0);

        moments.set(0, 0, 0, 1.5);
        assert_eq!(moments.get(0, 0, 0), 1.5);

        moments.set(5, 15, 100, -2.5);
        assert_eq!(moments.get(5, 15, 100), -2.5);

        // Original location unchanged
        assert_eq!(moments.get(0, 0, 0), 1.5);
    }

    #[test]
    fn test_adam_moments_reset() {
        let patterns = create_test_patterns();
        let mut moments = AdamMoments::new(&patterns);

        moments.set(0, 0, 0, 1.0);
        moments.set(5, 15, 100, 2.0);

        moments.reset();

        assert_eq!(moments.get(0, 0, 0), 0.0);
        assert_eq!(moments.get(5, 15, 100), 0.0);
    }

    // ========== Requirements summary test ==========

    #[test]
    fn test_all_requirements_summary() {
        println!("=== Adam Optimizer Requirements Verification ===");

        let patterns = create_test_patterns();
        let mut adam = AdamOptimizer::new(&patterns);

        // Req 3.1: alpha=0.025
        assert_eq!(adam.alpha(), 0.025);
        println!("  3.1: alpha=0.025");

        // Req 3.2: beta1=0.9
        assert_eq!(adam.beta1(), 0.9);
        println!("  3.2: beta1=0.9");

        // Req 3.3: beta2=0.999
        assert_eq!(adam.beta2(), 0.999);
        println!("  3.3: beta2=0.999");

        // Req 3.4: epsilon=1e-8
        assert_eq!(adam.epsilon(), 1e-8);
        println!("  3.4: epsilon=1e-8");

        // Req 3.5, 3.6: Moment vectors allocated
        assert!(adam.memory_usage() > 0);
        println!(
            "  3.5, 3.6: Moment vectors allocated ({:.2} MB)",
            adam.memory_usage() as f64 / 1_048_576.0
        );

        // Req 3.7: Global timestep counter
        assert_eq!(adam.timestep(), 0);
        adam.step();
        assert_eq!(adam.timestep(), 1);
        println!("  3.7: Global timestep counter");

        // Req 3.8: Bias correction
        let new_val = adam.update(0, 0, 0, 0.0, 1.0);
        assert!(new_val > 0.0);
        println!("  3.8: Bias correction applied");

        // Req 3.9: Parameter update
        let m = adam.first_moment().get(0, 0, 0);
        let v = adam.second_moment().get(0, 0, 0);
        assert!(m > 0.0);
        assert!(v > 0.0);
        println!("  3.9: Parameter update: delta = alpha * m_hat / (sqrt(v_hat) + epsilon)");

        // Req 3.10: Initialize all to 0.0
        let adam2 = AdamOptimizer::new(&patterns);
        assert_eq!(adam2.first_moment().get(0, 0, 0), 0.0);
        assert_eq!(adam2.second_moment().get(0, 0, 0), 0.0);
        println!("  3.10: Initialize all moments to 0.0");

        println!("=== All Adam optimizer requirements verified ===");
    }
}
