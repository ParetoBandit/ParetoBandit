"""
Test for Expert Death Prevention via Mixing Parameter (Paper Reviewer Fix)

This test verifies that the mixing parameter (gamma) prevents Expert Death
in non-stationary environments where expert performance can change over time.
"""

import numpy as np
import pytest
from typing import List
from src.bandit_gpt.router import CorrallingRouter


class MockExpert:
    """Mock expert that returns a fixed model and can have varying performance."""
    
    def __init__(self, model_id: str, models: List[str]):
        self.model_id = model_id
        self.models = models
        self.selections = 0
        
    def select_model(self, context: np.ndarray, total_steps: int = 0, **kwargs) -> str:
        """Always return the same model."""
        self.selections += 1
        return self.model_id
    
    def update(self, context: np.ndarray, model: str, reward: float, weight: float = 1.0):
        """No-op update for mock."""
        pass


def test_mixing_parameter_prevents_zero_probability():
    """
    Test that gamma ensures no expert's probability drops to zero.
    
    Scenario:
    - Expert 0 gets consistently good rewards (0.9)
    - Expert 1 gets consistently bad rewards (0.1)
    - After many iterations, Expert 1 should still have probability >= gamma/K
    """
    models = ["model_a", "model_b"]
    gamma = 0.05
    
    # Create two mock experts
    expert_0 = MockExpert("model_a", models)
    expert_1 = MockExpert("model_b", models)
    
    # Create Corralling router with mixing parameter
    router = CorrallingRouter(
        experts=[expert_0, expert_1],
        models=models,
        learning_rate=0.5,  # Higher learning rate to accelerate weight changes
        gamma=gamma
    )
    
    # Simulate 1000 iterations where Expert 0 is consistently better
    context = np.random.randn(10)
    np.random.seed(42)
    
    for _ in range(1000):
        model, token = router.select_model(context)
        
        # Expert 0 gets high reward, Expert 1 gets low reward
        if model == "model_a":
            reward = 0.9
        else:
            reward = 0.1
        
        router.update(context, model, reward, selection_token=token)
    
    # Check that Expert 1's probability is at least gamma/K
    min_prob = gamma / router.n_experts
    mixed_probs = router._get_mixed_distribution()
    
    print(f"\nFinal expert weights: {router.weights}")
    print(f"Final mixed probabilities: {mixed_probs}")
    print(f"Minimum guaranteed probability: {min_prob}")
    
    # Verify that even the worst expert has at least gamma/K probability
    assert np.all(mixed_probs >= min_prob - 1e-10), \
        f"Expert probability {mixed_probs.min()} dropped below minimum {min_prob}"
    
    # Verify that Expert 1 was still selected at least a few times
    # (with gamma=0.05 and 1000 iterations, we expect ~25 selections)
    assert expert_1.selections > 0, "Expert 1 was never selected (Expert Death occurred!)"
    print(f"Expert 1 was selected {expert_1.selections} times (expected ~25)")


def test_recovery_in_nonstationary_environment():
    """
    Test that the router can recover when expert performance changes.
    
    Scenario:
    - Phase 1 (steps 0-500): Expert 0 is better (0.8 vs 0.2)
    - Phase 2 (steps 500-1000): Expert 1 becomes better (0.2 vs 0.8)
    - The router should adapt and increase Expert 1's weight in Phase 2
    """
    models = ["model_a", "model_b"]
    gamma = 0.1  # Higher gamma for faster recovery
    
    expert_0 = MockExpert("model_a", models)
    expert_1 = MockExpert("model_b", models)
    
    router = CorrallingRouter(
        experts=[expert_0, expert_1],
        models=models,
        learning_rate=0.3,
        gamma=gamma
    )
    
    context = np.random.randn(10)
    np.random.seed(42)
    
    phase_1_weights = []
    phase_2_weights = []
    
    for step in range(1000):
        model, token = router.select_model(context)
        
        # Phase transition at step 500
        if step < 500:
            # Phase 1: Expert 0 is better
            reward = 0.8 if model == "model_a" else 0.2
            phase_1_weights.append(router.weights.copy())
        else:
            # Phase 2: Expert 1 is better
            reward = 0.2 if model == "model_a" else 0.8
            phase_2_weights.append(router.weights.copy())
        
        router.update(context, model, reward, selection_token=token)
    
    # Check that weights adapted to the phase change
    avg_weight_expert_1_phase_1 = np.mean([w[1] for w in phase_1_weights[-100:]])
    avg_weight_expert_1_phase_2 = np.mean([w[1] for w in phase_2_weights[-100:]])
    
    print(f"\nExpert 1 weight in Phase 1 (last 100 steps): {avg_weight_expert_1_phase_1:.4f}")
    print(f"Expert 1 weight in Phase 2 (last 100 steps): {avg_weight_expert_1_phase_2:.4f}")
    
    # Expert 1's weight should increase significantly in Phase 2
    assert avg_weight_expert_1_phase_2 > avg_weight_expert_1_phase_1, \
        "Router failed to recover when Expert 1 became better"
    
    print("✅ Router successfully adapted to non-stationary environment")


def test_gamma_zero_causes_expert_death():
    """
    Test that gamma=0 (no mixing) leads to much lower weights than gamma>0.
    
    This is a negative test to confirm the problem exists without the fix.
    """
    models = ["model_a", "model_b"]
    
    # Test with gamma=0
    expert_0_no_mix = MockExpert("model_a", models)
    expert_1_no_mix = MockExpert("model_b", models)
    
    router_no_mix = CorrallingRouter(
        experts=[expert_0_no_mix, expert_1_no_mix],
        models=models,
        learning_rate=0.5,
        gamma=0.0  # No mixing - pure exponential weighting
    )
    
    # Test with gamma=0.05
    expert_0_mix = MockExpert("model_a", models)
    expert_1_mix = MockExpert("model_b", models)
    
    router_mix = CorrallingRouter(
        experts=[expert_0_mix, expert_1_mix],
        models=models,
        learning_rate=0.5,
        gamma=0.05  # With mixing
    )
    
    context = np.random.randn(10)
    
    # Simulate scenario where Expert 0 is consistently better
    for i in range(1000):
        np.random.seed(42 + i)
        
        # No mixing
        model, token = router_no_mix.select_model(context)
        reward = 0.9 if model == "model_a" else 0.1
        router_no_mix.update(context, model, reward, selection_token=token)
        
        # With mixing
        model, token = router_mix.select_model(context)
        reward = 0.9 if model == "model_a" else 0.1
        router_mix.update(context, model, reward, selection_token=token)
    
    # Get final probabilities
    prob_no_mix = router_no_mix._get_mixed_distribution()[1]
    prob_mix = router_mix._get_mixed_distribution()[1]
    
    print(f"\nExpert 1 weight with gamma=0: {router_no_mix.weights[1]:.2e}")
    print(f"Expert 1 probability with gamma=0: {prob_no_mix:.2e}")
    print(f"Expert 1 weight with gamma=0.05: {router_mix.weights[1]:.2e}")
    print(f"Expert 1 probability with gamma=0.05: {prob_mix:.4f}")
    
    # With gamma=0, Expert 1's weight should be much smaller
    assert router_no_mix.weights[1] < 1e-4, \
        "Expected very small weight with gamma=0"
    
    # With gamma>0, Expert 1 should maintain minimum probability
    min_prob = 0.05 / 2  # gamma / K
    assert prob_mix >= min_prob - 1e-10, \
        f"Expected probability >= {min_prob}, got {prob_mix}"
    
    # The mixing parameter should provide significantly higher probability
    assert prob_mix > prob_no_mix * 10, \
        "Mixing parameter should provide much higher probability"
    
    print("✅ Confirmed: gamma>0 prevents Expert Death")


def test_importance_weighting_uses_mixed_probability():
    """
    Test that the importance-weighted loss estimator uses the mixed probability.
    
    This ensures the estimator is unbiased.
    """
    models = ["model_a", "model_b"]
    gamma = 0.1
    
    expert_0 = MockExpert("model_a", models)
    expert_1 = MockExpert("model_b", models)
    
    router = CorrallingRouter(
        experts=[expert_0, expert_1],
        models=models,
        learning_rate=0.1,
        gamma=gamma
    )
    
    context = np.random.randn(10)
    
    # Force selection of Expert 0
    np.random.seed(0)
    model, token = router.select_model(context)
    
    # Store the probability from the selection token
    selected_prob = token["expert_prob"]
    expert_idx = token["expert_idx"]
    
    # Compute what the mixed probability should be
    mixed_probs = router._get_mixed_distribution()
    expected_prob = mixed_probs[expert_idx]
    
    print(f"\nRaw weight: {router.weights[expert_idx]:.4f}")
    print(f"Token probability: {selected_prob:.4f}")
    print(f"Expected mixed probability: {expected_prob:.4f}")
    
    # Verify that the token probability matches the mixed distribution
    assert abs(selected_prob - expected_prob) < 1e-10, \
        "Selection token probability doesn't match the mixed distribution!"
    
    # Verify that mixed probability is higher than raw weight (due to mixing)
    assert selected_prob >= router.weights[expert_idx] - 1e-10, \
        "Mixed probability should be >= raw weight"
    
    print("✅ Selection token correctly carries mixed probability")


def test_gamma_parameter_bounds():
    """Test that gamma parameter is properly bounded."""
    models = ["model_a", "model_b"]
    
    expert_0 = MockExpert("model_a", models)
    expert_1 = MockExpert("model_b", models)
    
    # Test valid gamma values
    for gamma in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        router = CorrallingRouter(
            experts=[expert_0, expert_1],
            models=models,
            gamma=gamma
        )
        assert router.gamma == gamma
        
        # Verify mixed distribution sums to 1
        mixed_probs = router._get_mixed_distribution()
        assert abs(mixed_probs.sum() - 1.0) < 1e-10
        
        # Verify minimum probability is gamma/K
        min_prob = gamma / router.n_experts
        assert np.all(mixed_probs >= min_prob - 1e-10)
    
    print("✅ Gamma parameter bounds are correct")


if __name__ == "__main__":
    # Run tests
    print("=" * 80)
    print("Testing Expert Death Prevention (Paper Reviewer Fix)")
    print("=" * 80)
    
    test_mixing_parameter_prevents_zero_probability()
    print()
    
    test_recovery_in_nonstationary_environment()
    print()
    
    test_gamma_zero_causes_expert_death()
    print()
    
    test_importance_weighting_uses_mixed_probability()
    print()
    
    test_gamma_parameter_bounds()
    print()
    
    print("=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)

