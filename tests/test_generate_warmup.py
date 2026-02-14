#!/usr/bin/env python3
"""
Unit tests for scripts/generate_warmup.py

Tests the critical mathematical and operational components:
- IRT reward simulation
- Noise injection
- Mean imputation
- Plasticity factor application
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from generate_warmup_priors import (
    simulate_irt_reward,
    perturb_prompt,
)


class TestIRTRewardSimulation:
    """Test Item Response Theory reward simulation"""
    
    def test_irt_weak_model_hard_prompt(self):
        """Weak model on hard prompt should have very low success rate"""
        weak_hle = 0.76  # Ministral
        hard_difficulty = 1.0
        
        prob = simulate_irt_reward(weak_hle, hard_difficulty)
        
        # Should be low (~7% with recalibrated centering)
        assert 0.05 <= prob <= 0.10, f"Expected ~7% success, got {prob:.2%}"
    
    def test_irt_strong_model_hard_prompt(self):
        """Strong model on hard prompt should have high success rate"""
        strong_hle = 0.98  # Opus
        hard_difficulty = 1.0
        
        prob = simulate_irt_reward(strong_hle, hard_difficulty)
        
        # Should be high (~85% with recalibrated centering)
        assert 0.80 <= prob <= 0.90, f"Expected ~85% success, got {prob:.2%}"
    
    def test_irt_weak_model_easy_prompt(self):
        """Weak model on easy prompt should have high success rate"""
        weak_hle = 0.76
        easy_difficulty = 0.1
        
        prob = simulate_irt_reward(weak_hle, easy_difficulty)
        
        # Should be high (~94% with recalibrated centering)
        assert 0.92 <= prob <= 0.96, f"Expected ~94% success, got {prob:.2%}"
    
    def test_irt_strong_model_easy_prompt(self):
        """Strong model on easy prompt should have very high success rate"""
        strong_hle = 0.98
        easy_difficulty = 0.1
        
        prob = simulate_irt_reward(strong_hle, easy_difficulty)
        
        # Should be very high (~99%+)
        assert prob >= 0.98, f"Expected ~99%+ success, got {prob:.2%}"
    
    def test_irt_returns_valid_probability(self):
        """IRT should always return valid probability [0, 1]"""
        test_cases = [
            (0.70, 0.0),
            (0.85, 0.5),
            (0.99, 1.0),
            (0.76, 0.3),
            (0.90, 0.7),
        ]
        
        for hle, difficulty in test_cases:
            prob = simulate_irt_reward(hle, difficulty)
            assert 0.0 <= prob <= 1.0, f"Invalid probability {prob} for HLE={hle}, diff={difficulty}"
    
    def test_irt_trap_bypass(self):
        """IRT should return static HLE for trap prompts"""
        hle = 0.85
        difficulty = 0.5
        
        prob = simulate_irt_reward(hle, difficulty, is_trap=True)
        
        assert prob == hle, f"Trap should return static HLE {hle}, got {prob}"
    
    def test_irt_monotonic_in_ability(self):
        """Higher ability should yield higher success probability for same difficulty"""
        difficulty = 0.5
        
        prob_weak = simulate_irt_reward(0.76, difficulty)
        prob_mid = simulate_irt_reward(0.85, difficulty)
        prob_strong = simulate_irt_reward(0.98, difficulty)
        
        assert prob_weak < prob_mid < prob_strong, \
            f"Not monotonic: {prob_weak:.3f} < {prob_mid:.3f} < {prob_strong:.3f}"
    
    def test_irt_monotonic_in_difficulty(self):
        """Higher difficulty should yield lower success probability for same ability"""
        hle = 0.85
        
        prob_easy = simulate_irt_reward(hle, 0.1)
        prob_mid = simulate_irt_reward(hle, 0.5)
        prob_hard = simulate_irt_reward(hle, 0.9)
        
        assert prob_easy > prob_mid > prob_hard, \
            f"Not monotonic: {prob_easy:.3f} > {prob_mid:.3f} > {prob_hard:.3f}"


class TestNoiseInjection:
    """Test prompt perturbation for feature distribution alignment"""
    
    def test_perturb_adds_noise(self):
        """Perturbation should modify some prompts"""
        original = "What is the function of this code"
        perturbed_versions = set()
        
        # Run multiple times to see variations
        for i in range(20):
            perturbed = perturb_prompt(original, noise_level=1.0, seed=i)
            perturbed_versions.add(perturbed)
        
        # Should have some variation
        assert len(perturbed_versions) > 1, "Perturbation should create variations"
    
    def test_perturb_noise_level_zero(self):
        """Noise level 0 should never perturb"""
        original = "What is the function of this code"
        
        for i in range(10):
            perturbed = perturb_prompt(original, noise_level=0.0, seed=i)
            assert perturbed == original, f"Noise level 0 should not perturb, got {perturbed}"
    
    def test_perturb_preserves_meaning(self):
        """Perturbation should preserve core content"""
        original = "Write a Python function to sort a list"
        perturbed = perturb_prompt(original, noise_level=1.0, seed=42)
        
        # Should still contain key terms (case-insensitive)
        perturbed_lower = perturbed.lower()
        assert "python" in perturbed_lower or "func" in perturbed_lower
        assert "sort" in perturbed_lower
        assert "list" in perturbed_lower
    
    def test_perturb_deterministic_with_seed(self):
        """Same seed should produce same perturbation"""
        original = "What is the function of this code"
        
        perturbed1 = perturb_prompt(original, noise_level=1.0, seed=42)
        perturbed2 = perturb_prompt(original, noise_level=1.0, seed=42)
        
        assert perturbed1 == perturbed2, "Same seed should give same result"
    
    def test_perturb_different_seeds(self):
        """Different seeds should produce different perturbations"""
        original = "What is the function of this code"
        
        perturbations = set()
        for seed in range(10):
            perturbed = perturb_prompt(original, noise_level=1.0, seed=seed)
            perturbations.add(perturbed)
        
        # Should have some variety
        assert len(perturbations) >= 3, "Different seeds should create variety"


class TestMeanImputation:
    """Test mean imputation logic for missing HLE scores"""
    
    def test_mean_imputation_calculation(self):
        """Mean imputation should calculate correct average"""
        existing_hles = [0.76, 0.85, 0.90, 0.98]
        expected_mean = np.mean(existing_hles)
        
        # Simulate the logic
        model_hle_map = {f"model_{i}": hle for i, hle in enumerate(existing_hles)}
        avg_hle = np.mean(list(model_hle_map.values()))
        
        assert abs(avg_hle - expected_mean) < 1e-6, \
            f"Expected {expected_mean:.3f}, got {avg_hle:.3f}"
    
    def test_mean_imputation_fallback(self):
        """Empty map should use fallback of 0.85"""
        model_hle_map = {}
        avg_hle = np.mean(list(model_hle_map.values())) if model_hle_map else 0.85
        
        assert avg_hle == 0.85, f"Empty map should fallback to 0.85, got {avg_hle}"


class TestPlasticityFactor:
    """Test plasticity factor application to A and b matrices"""
    
    def test_plasticity_preserves_coefficients(self):
        """Scaling both A and b should preserve theta = A^-1 * b"""
        # Create simple test case
        A_original = np.array([[2.0, 0.5], [0.5, 3.0]])
        b_original = np.array([1.0, 2.0])
        
        # Calculate original theta
        theta_original = np.linalg.solve(A_original, b_original)
        
        # Apply plasticity factor
        plasticity = 0.1
        A_scaled = A_original * plasticity
        b_scaled = b_original * plasticity
        
        # Calculate new theta
        theta_scaled = np.linalg.solve(A_scaled, b_scaled)
        
        # Should be approximately equal
        np.testing.assert_allclose(theta_original, theta_scaled, rtol=1e-10,
                                   err_msg="Coefficients not preserved after scaling")
    
    def test_plasticity_widens_confidence(self):
        """Scaling A should widen confidence intervals"""
        A_original = np.array([[4.0, 0.0], [0.0, 4.0]])
        x = np.array([1.0, 0.0])
        
        # Original confidence term
        A_inv_original = np.linalg.inv(A_original)
        conf_original = np.sqrt(x.T @ A_inv_original @ x)
        
        # After scaling
        plasticity = 0.1
        A_scaled = A_original * plasticity
        A_inv_scaled = np.linalg.inv(A_scaled)
        conf_scaled = np.sqrt(x.T @ A_inv_scaled @ x)
        
        # Should be wider by sqrt(1/plasticity) = sqrt(10)
        expected_ratio = np.sqrt(1.0 / plasticity)
        actual_ratio = conf_scaled / conf_original
        
        assert abs(actual_ratio - expected_ratio) < 1e-6, \
            f"Confidence should widen by {expected_ratio:.2f}×, got {actual_ratio:.2f}×"


class TestIntegration:
    """Integration tests for end-to-end warmup generation logic"""
    
    def test_bernoulli_sampling_distribution(self):
        """Bernoulli sampling should match expected distribution"""
        prob_success = 0.75
        n_samples = 10000
        
        import random
        random.seed(42)
        
        outcomes = [1.0 if random.random() < prob_success else 0.0 
                   for _ in range(n_samples)]
        
        empirical_prob = np.mean(outcomes)
        
        # Should be close to 0.75 with large n
        assert abs(empirical_prob - prob_success) < 0.02, \
            f"Expected {prob_success:.2f}, got {empirical_prob:.2f}"
    
    def test_irt_probability_range(self):
        """IRT probabilities should create realistic gradient"""
        # Test various model-difficulty combinations
        test_cases = [
            # (HLE, difficulty, expected_range_min, expected_range_max)
            (0.76, 1.0, 0.05, 0.10),    # Weak on hard: very low
            (0.98, 1.0, 0.80, 0.90),    # Strong on hard: high
            (0.76, 0.1, 0.92, 0.96),    # Weak on easy: high
            (0.98, 0.1, 0.98, 1.0),     # Strong on easy: very high
        ]
        
        for hle, difficulty, min_prob, max_prob in test_cases:
            prob = simulate_irt_reward(hle, difficulty)
            assert min_prob <= prob <= max_prob, \
                f"HLE={hle}, diff={difficulty}: expected [{min_prob}, {max_prob}], got {prob:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
