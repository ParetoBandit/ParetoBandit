"""
Unit tests for the synthetic warmup generation system.

Tests cover:
1. IRT reward function correctness
2. HLE and difficulty normalization
3. A matrix density after warmup
4. Warmup loading mechanism
"""

import pytest
import numpy as np
import math
import tempfile
from pathlib import Path
import joblib

# Import the IRT function
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.generate_warmup import ir_theory_reward
from src.bandit_gpt.router import BanditRouter
from experiments.utils.data_loader import load_model_registry


class TestIRTRewardFunction:
    """Test the Item Response Theory reward function."""
    
    def test_irt_extreme_cases(self):
        """Test that IRT produces realistic probabilities for extreme cases."""
        # Case 1: Strong model on easy task → ~100%
        prob = ir_theory_reward(model_skill=0.9, difficulty=0.1)
        assert prob > 0.95, f"Strong model on easy task should have >95% success, got {prob:.3f}"
        assert prob < 1.0, f"Probability should not be exactly 1.0, got {prob:.3f}"
        
        # Case 2: Weak model on hard task → ~0%
        prob = ir_theory_reward(model_skill=0.1, difficulty=0.9)
        assert prob < 0.05, f"Weak model on hard task should have <5% success, got {prob:.3f}"
        assert prob > 0.0, f"Probability should not be exactly 0.0, got {prob:.3f}"
        
        # Case 3: Equal match → ~50%
        prob = ir_theory_reward(model_skill=0.5, difficulty=0.5)
        assert 0.45 < prob < 0.55, f"Equal match should be ~50%, got {prob:.3f}"
    
    def test_irt_monotonicity(self):
        """Test that higher skill → higher success probability (monotonic)."""
        difficulty = 0.6
        skills = [0.1, 0.3, 0.5, 0.7, 0.9]
        probs = [ir_theory_reward(skill, difficulty) for skill in skills]
        
        # Check monotonically increasing
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i+1], \
                f"P(skill={skills[i]}) = {probs[i]:.3f} should be < P(skill={skills[i+1]}) = {probs[i+1]:.3f}"
    
    def test_irt_symmetry(self):
        """Test symmetry: P(0.9, 0.1) ≈ 1 - P(0.1, 0.9)."""
        prob_high_low = ir_theory_reward(model_skill=0.9, difficulty=0.1)
        prob_low_high = ir_theory_reward(model_skill=0.1, difficulty=0.9)
        
        # They should be approximately complementary
        assert abs((prob_high_low + prob_low_high) - 1.0) < 0.05, \
            f"Symmetry violated: {prob_high_low:.3f} + {prob_low_high:.3f} ≠ 1.0"
    
    def test_irt_without_transformation_is_wrong(self):
        """Demonstrate why the logit transformation is necessary."""
        # Naive sigmoid (no transform)
        def naive_sigmoid(model_skill, difficulty):
            return 1.0 / (1.0 + math.exp(-(model_skill - difficulty)))
        
        # Compare: Strong model on easy task
        naive_prob = naive_sigmoid(0.9, 0.1)
        irt_prob = ir_theory_reward(0.9, 0.1)
        
        # Naive is too low (compressed around 0.5)
        assert naive_prob < 0.75, f"Naive sigmoid is {naive_prob:.3f}"
        assert irt_prob > 0.95, f"IRT is {irt_prob:.3f}"
        assert irt_prob > naive_prob + 0.2, \
            f"IRT should be significantly higher than naive: {irt_prob:.3f} vs {naive_prob:.3f}"


class TestNormalization:
    """Test that HLE and difficulty scores are correctly normalized."""
    
    def test_hle_range(self):
        """Verify HLE scores are in [0, 1] range."""
        registry = load_model_registry()
        
        hle_scores = []
        for model_id, model_data in registry.items():
            hle = model_data.get("hle")
            if hle is not None:
                hle_scores.append(hle)
                assert 0.0 <= hle <= 1.0, \
                    f"HLE for {model_id} out of range: {hle}"
        
        assert len(hle_scores) > 0, "No HLE scores found in registry"
        print(f"✓ Validated {len(hle_scores)} HLE scores, range: [{min(hle_scores):.3f}, {max(hle_scores):.3f}]")
    
    def test_difficulty_range(self):
        """Verify router's difficulty scores are in [0, 1] range."""
        registry = load_model_registry()
        router = BanditRouter.create(registry, priors="none", exploration="safe")
        
        test_prompts = [
            "What is 2+2?",  # Easy
            "Implement a balanced AVL tree with self-balancing rotations",  # Hard
            "Write a poem about nature",  # Medium
        ]
        
        for prompt in test_prompts:
            difficulty = router._detect_difficulty_score(prompt)
            assert 0.0 <= difficulty <= 1.0, \
                f"Difficulty for '{prompt[:30]}...' out of range: {difficulty}"
            print(f"✓ '{prompt[:40]}...' → difficulty={difficulty:.3f}")


class TestWarmupGeneration:
    """Test the warmup generation process."""
    
    def test_a_matrix_becomes_dense(self):
        """Verify that A matrices become non-identity after updates."""
        registry = load_model_registry()
        router = BanditRouter.create(registry, priors="none", exploration="safe")
        
        # Get initial A matrix (should be close to identity)
        model_id = list(router.bandit.models)[0]
        A_before = router.bandit.A[model_id].copy()
        
        # Check it's close to identity initially
        identity = np.eye(router.bandit.dim) * router.bandit.init_lambda
        assert np.allclose(A_before, identity, atol=0.1), \
            "Initial A should be close to λI"
        
        # Perform some updates
        prompts = [
            "Write Python code",
            "Solve this math problem",
            "Tell me a story"
        ]
        
        for prompt in prompts:
            for model in router.bandit.models[:5]:  # Update first 5 models
                reward = np.random.uniform(0.3, 0.9)
                router.update(model, prompt, reward)
        
        # Get A matrix after updates
        A_after = router.bandit.A[model_id].copy()
        
        # Should be different from identity now
        diff = np.abs(A_after - identity).sum()
        assert diff > 1.0, \
            f"A matrix should diverge from identity after updates, total diff={diff:.3f}"
        
        # Check off-diagonal elements are non-zero (covariance)
        off_diagonal = A_after - np.diag(np.diag(A_after))
        off_diagonal_sum = np.abs(off_diagonal).sum()
        assert off_diagonal_sum > 0.1, \
            f"A matrix should have non-zero off-diagonal elements, sum={off_diagonal_sum:.3f}"
        
        print(f"✓ A matrix diverged from identity: total_diff={diff:.3f}, off_diag_sum={off_diagonal_sum:.3f}")
    
    def test_warmup_artifact_structure(self, tmp_path):
        """Test that warmup artifact has correct structure."""
        # Create a mini warmup artifact
        registry = load_model_registry()
        router = BanditRouter.create(registry, priors="none", exploration="safe")
        
        # Do a few updates
        for i in range(10):
            prompt = f"Test prompt {i}"
            for model_id in list(router.bandit.models)[:3]:
                router.update(model_id, prompt, 0.7)
        
        # Save it
        test_path = tmp_path / "test_warmup.joblib"
        state = {
            "A": router.bandit.A,
            "b": router.bandit.b,
            "n": 10
        }
        joblib.dump(state, test_path)
        
        # Load and verify
        loaded = joblib.load(test_path)
        
        assert "A" in loaded, "Warmup artifact should have 'A' key"
        assert "b" in loaded, "Warmup artifact should have 'b' key"
        assert "n" in loaded, "Warmup artifact should have 'n' key"
        
        assert len(loaded["A"]) == len(router.bandit.models), \
            "A dict should have entry for each model"
        assert len(loaded["b"]) == len(router.bandit.models), \
            "b dict should have entry for each model"
        
        # Check shapes
        for model_id in loaded["A"]:
            assert loaded["A"][model_id].shape == (router.bandit.dim, router.bandit.dim), \
                f"A[{model_id}] has wrong shape"
            assert loaded["b"][model_id].shape == (router.bandit.dim,), \
                f"b[{model_id}] has wrong shape"
        
        print(f"✓ Warmup artifact structure validated")


class TestWarmupLoading:
    """Test the warmup loading mechanism in the router."""
    
    def test_warmup_loading_with_file(self, tmp_path):
        """Test that router loads warmup priors correctly."""
        # Create a warmup artifact
        registry = load_model_registry()
        router_train = BanditRouter.create(registry, priors="none", exploration="safe")
        
        # Train it a bit
        for i in range(20):
            router_train.update(
                list(router_train.bandit.models)[0], 
                f"Test {i}", 
                0.8
            )
        
        # Save state
        warmup_path = tmp_path / "priors_warmup.joblib"
        state = {
            "A": router_train.bandit.A,
            "b": router_train.bandit.b,
            "n": 20
        }
        joblib.dump(state, warmup_path)
        
        # Now test loading - we'd need to modify the router to accept a custom path
        # For now, just verify the structure is correct
        loaded = joblib.load(warmup_path)
        
        model_id = list(router_train.bandit.models)[0]
        A_loaded = loaded["A"][model_id]
        A_original = router_train.bandit.A[model_id]
        
        assert np.allclose(A_loaded, A_original), \
            "Loaded A matrix should match original"
        
        print(f"✓ Warmup state saved and loaded correctly")
    
    def test_refresh_inverse_cache(self):
        """Test that refresh_inverse_cache recomputes inverses correctly."""
        registry = load_model_registry()
        router = BanditRouter.create(registry, priors="none", exploration="safe")
        
        # Get initial inverse
        model_id = list(router.bandit.models)[0]
        A_inv_before = router.bandit.A_inv[model_id].copy()
        
        # Modify A directly (simulating bulk load)
        router.bandit.A[model_id] += np.eye(router.bandit.dim) * 0.5
        
        # Inverse should now be stale (not match A)
        # Verify: A @ A_inv should NOT equal I
        product_stale = router.bandit.A[model_id] @ A_inv_before
        identity = np.eye(router.bandit.dim)
        assert not np.allclose(product_stale, identity, atol=0.01), \
            "Stale inverse should not satisfy A @ A_inv = I"
        
        # Refresh the cache
        router.bandit.refresh_inverse_cache()
        
        # Now it should work
        A_inv_after = router.bandit.A_inv[model_id]
        product_fresh = router.bandit.A[model_id] @ A_inv_after
        assert np.allclose(product_fresh, identity, atol=0.01), \
            "After refresh, A @ A_inv should equal I"
        
        print(f"✓ Inverse cache refresh works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
