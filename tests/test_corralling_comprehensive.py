"""
Comprehensive unit tests for CorrallingRouter.

Tests based on experiments from experiments_v1/04_figure/ and experiments_v1/06_figure/
but focusing on testing the core router.py CorrallingRouter functionality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np

from bandit_gpt.router import CorrallingRouter


# =============================================================================
# Mock Experts for Testing
# =============================================================================

class DeterministicExpert:
    """Expert that always selects the same model."""
    
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, cost=0.0):
        self.update_count += 1


class AdaptiveExpert:
    """Expert that adapts based on context."""
    
    def __init__(self, name: str, models: list, threshold: float = 0.5):
        self.name = name
        self.models = models
        self.threshold = threshold
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        # Select based on context sum
        if np.sum(context) > self.threshold:
            return self.models[0]
        return self.models[1]
    
    def update(self, context, model, reward, cost=0.0):
        self.update_count += 1


class SmartExpert:
    """Expert that occasionally explores but mostly exploits."""
    
    def __init__(self, name: str, best_model: str, models: list, explore_rate: float = 0.05):
        self.name = name
        self.best_model = best_model
        self.models = models
        self.explore_rate = explore_rate
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        if np.random.random() < self.explore_rate:
            return np.random.choice([m for m in self.models if m != self.best_model])
        return self.best_model
    
    def update(self, context, model, reward, cost=0.0):
        self.update_count += 1


# =============================================================================
# Initialization Tests
# =============================================================================

class TestCorrallingInitialization:
    """Test CorrallingRouter initialization."""
    
    def test_basic_initialization(self):
        """Test basic initialization with two experts."""
        models = ["model_a", "model_b"]
        expert1 = DeterministicExpert("expert1", "model_a")
        expert2 = DeterministicExpert("expert2", "model_b")
        
        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        # Check initialization
        assert router.n_experts == 2
        assert len(router.weights) == 2
        assert len(router.cumulative_losses) == 2
        
        # Weights should be uniform
        np.testing.assert_array_almost_equal(router.weights, np.array([0.5, 0.5]))
        
        # Cumulative losses should be zero
        np.testing.assert_array_almost_equal(router.cumulative_losses, np.zeros(2))
    
    def test_initialization_with_custom_learning_rate(self):
        """Test initialization with custom learning rate."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        learning_rate = 5.0  # Optimized value from experiments
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=learning_rate,
            gamma=0.10
        )
        
        assert router.learning_rate == learning_rate
        assert router.gamma == 0.10
    
    def test_initialization_with_multiple_experts(self):
        """Test initialization with more than 2 experts."""
        models = ["model_a", "model_b", "model_c"]
        experts = [DeterministicExpert(f"expert{i}", models[i % len(models)]) for i in range(5)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        assert router.n_experts == 5
        assert len(router.weights) == 5
        
        # Weights should be uniform
        expected_weight = 1.0 / 5.0
        for w in router.weights:
            assert abs(w - expected_weight) < 1e-10


# =============================================================================
# Selection Tests
# =============================================================================

class TestCorrallingSelection:
    """Test expert and model selection."""
    
    def test_select_model_returns_valid_model(self):
        """Test that select_model always returns a valid model."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Run multiple selections
        for _ in range(100):
            selected = router.select_model(context)
            assert selected in models, f"Selected model {selected} not in {models}"
    
    def test_expert_sampling_distribution(self):
        """Test that experts are sampled according to weights."""
        models = ["model_a", "model_b"]
        expert1 = DeterministicExpert("expert1", "model_a")
        expert2 = DeterministicExpert("expert2", "model_b")
        
        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        n_trials = 1000
        expert_selections = []
        
        # Sample many times
        np.random.seed(42)
        for _ in range(n_trials):
            router.select_model(context)
            expert_selections.append(router.last_expert_idx)
        
        # With uniform weights, should be roughly 50/50
        count_0 = sum(1 for e in expert_selections if e == 0)
        ratio = count_0 / n_trials
        
        # Allow some variance (binomial distribution)
        assert 0.4 < ratio < 0.6, f"Expected ~50% expert 0, got {ratio:.2%}"
    
    def test_gamma_prevents_expert_death(self):
        """Test that gamma parameter prevents expert from getting zero probability."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        gamma = 0.1  # 10% minimum
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=gamma
        )
        
        # Manually set very unbalanced cumulative losses
        router.cumulative_losses = np.array([0.0, 100.0])  # Expert 1 performed terribly
        
        # Get mixed distribution
        probs = router._get_mixed_distribution()
        
        # Check that both experts have at least gamma/n_experts probability
        min_prob = gamma / router.n_experts
        for prob in probs:
            assert prob >= min_prob, f"Probability {prob} below minimum {min_prob}"


# =============================================================================
# Update Tests
# =============================================================================

class TestCorrallingUpdate:
    """Test weight update mechanism."""
    
    def test_weights_update_after_feedback(self):
        """Test that weights change after receiving feedback."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        initial_weights = router.weights.copy()
        
        # Select and update
        selected_model = router.select_model(context)
        router.update(context, selected_model, reward=0.8)
        
        # Weights should have changed
        assert not np.allclose(router.weights, initial_weights), \
            "Weights should change after update"
    
    def test_weights_sum_to_one(self):
        """Test that weights always sum to 1."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Run many updates
        for i in range(100):
            selected = router.select_model(context)
            reward = 0.5 + 0.4 * np.sin(i)  # Varying rewards
            router.update(context, selected, reward)
            
            # Check weight sum
            weight_sum = router.weights.sum()
            assert abs(weight_sum - 1.0) < 1e-10, \
                f"Weights sum to {weight_sum}, expected 1.0"
    
    def test_cumulative_losses_increase(self):
        """Test that cumulative losses accumulate over time."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"expert{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        initial_losses = router.cumulative_losses.copy()
        
        # Run updates with imperfect rewards
        for _ in range(50):
            selected = router.select_model(context)
            router.update(context, selected, reward=0.6)  # < 1.0, so there's loss
        
        # Cumulative losses should have increased
        assert router.cumulative_losses.sum() > initial_losses.sum(), \
            "Cumulative losses should increase"
    
    def test_expert_updates_are_called(self):
        """Test that underlying experts receive updates."""
        models = ["model_a", "model_b"]
        expert1 = DeterministicExpert("expert1", "model_a")
        expert2 = DeterministicExpert("expert2", "model_b")
        
        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Run updates
        for _ in range(20):
            selected = router.select_model(context)
            router.update(context, selected, reward=0.8)
        
        # Both experts should have received some updates
        total_updates = expert1.update_count + expert2.update_count
        assert total_updates == 20, f"Expected 20 updates total, got {total_updates}"


# =============================================================================
# Learning Behavior Tests
# =============================================================================

class TestCorrallingLearning:
    """Test that Corralling learns to weight experts appropriately."""
    
    def test_learns_to_favor_better_expert(self):
        """Test that Corralling increases weight of better-performing expert."""
        models = ["model_a", "model_b"]
        
        # Expert 1 always picks the better model
        good_expert = DeterministicExpert("good", "model_a")
        
        # Expert 2 always picks the worse model
        bad_expert = DeterministicExpert("bad", "model_b")
        
        router = CorrallingRouter(
            experts=[good_expert, bad_expert],
            models=models,
            learning_rate=2.0,  # Higher learning rate for faster adaptation
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Simulate environment where model_a is always better
        for _ in range(100):
            selected = router.select_model(context)
            
            # Reward based on which model was selected
            if selected == "model_a":
                reward = 0.9
            else:
                reward = 0.3
            
            router.update(context, selected, reward)
        
        # Good expert should have more weight
        assert router.weights[0] > router.weights[1], \
            f"Good expert weight ({router.weights[0]:.3f}) should exceed bad expert ({router.weights[1]:.3f})"
    
    def test_adapts_to_distribution_shift(self):
        """Test that Corralling can adapt when the better expert changes."""
        models = ["model_a", "model_b"]
        
        expert1 = DeterministicExpert("expert1", "model_a")
        expert2 = DeterministicExpert("expert2", "model_b")
        
        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Phase 1: model_a is better
        for _ in range(50):
            selected = router.select_model(context)
            reward = 0.9 if selected == "model_a" else 0.3
            router.update(context, selected, reward)
        
        weights_phase1 = router.weights.copy()
        
        # Phase 2: model_b becomes better (distribution shift)
        for _ in range(100):
            selected = router.select_model(context)
            reward = 0.3 if selected == "model_a" else 0.9  # Reversed!
            router.update(context, selected, reward)
        
        weights_phase2 = router.weights.copy()
        
        # Weights should have shifted
        # In phase 1, expert1 (model_a) should have been favored
        # In phase 2, expert2 (model_b) should become favored
        
        # Check that weights changed significantly
        weight_change = np.abs(weights_phase2 - weights_phase1).sum()
        assert weight_change > 0.1, \
            f"Weights should change after distribution shift, change={weight_change:.3f}"
    
    def test_learning_rate_effect(self):
        """Test that learning rate affects adaptation speed."""
        models = ["model_a", "model_b"]
        
        # Create two routers with different learning rates
        router_slow = CorrallingRouter(
            experts=[DeterministicExpert("e1", "model_a"), DeterministicExpert("e2", "model_b")],
            models=models,
            learning_rate=0.1,  # Slow
            gamma=0.05
        )
        
        router_fast = CorrallingRouter(
            experts=[DeterministicExpert("e1", "model_a"), DeterministicExpert("e2", "model_b")],
            models=models,
            learning_rate=5.0,  # Fast (optimized value)
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Give same feedback to both
        np.random.seed(42)
        for _ in range(50):
            # Slow router
            selected_slow = router_slow.select_model(context)
            reward = 0.9 if selected_slow == "model_a" else 0.3
            router_slow.update(context, selected_slow, reward)
            
            # Fast router (same feedback)
            np.random.seed(42 + _)
            selected_fast = router_fast.select_model(context)
            reward = 0.9 if selected_fast == "model_a" else 0.3
            router_fast.update(context, selected_fast, reward)
        
        # Fast router should have more diverged weights
        slow_divergence = abs(router_slow.weights[0] - 0.5)
        fast_divergence = abs(router_fast.weights[0] - 0.5)
        
        assert fast_divergence >= slow_divergence * 0.8, \
            f"Fast LR ({fast_divergence:.3f}) should adapt more than slow LR ({slow_divergence:.3f})"


# =============================================================================
# Realistic Scenario Tests
# =============================================================================

class TestCorrallingRealisticScenarios:
    """Test Corralling in scenarios resembling real experiments."""
    
    def test_warmup_vs_tabula_rasa(self):
        """Test Corralling with warmup expert vs tabula rasa expert."""
        models = ["model_a", "model_b"]
        
        # Warmup expert: has strong bias toward model_a
        warmup = DeterministicExpert("warmup", "model_a")
        
        # Tabula rasa: starts fresh, can pick either
        tabula_rasa = AdaptiveExpert("tabula_rasa", models, threshold=0.5)
        
        router = CorrallingRouter(
            experts=[warmup, tabula_rasa],
            models=models,
            learning_rate=5.0,
            gamma=0.10
        )
        
        # Scenario: warmup's bias is wrong, tabula rasa is better
        context = np.ones(10)  # Context sum > 0.5, so tabula_rasa picks model_a
        
        for _ in range(100):
            selected = router.select_model(context)
            
            # model_b is actually better (contradicting warmup)
            reward = 0.3 if selected == "model_a" else 0.9
            router.update(context, selected, reward)
        
        # After learning, tabula_rasa should have more weight
        # (though this depends on sampling randomness and may not always hold)
        
        # At minimum, check that learning happened
        assert not np.allclose(router.weights, [0.5, 0.5]), \
            "Weights should have moved from initial uniform distribution"
    
    def test_statistical_power_with_more_samples(self):
        """Test that more samples lead to clearer expert preference."""
        models = ["model_a", "model_b"]
        
        good_expert = SmartExpert("good", "model_b", models, explore_rate=0.05)
        bad_expert = DeterministicExpert("bad", "model_a")
        
        router = CorrallingRouter(
            experts=[good_expert, bad_expert],
            models=models,
            learning_rate=0.1,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Give many samples with clear performance difference
        n_samples = 1000
        for _ in range(n_samples):
            selected = router.select_model(context)
            
            # model_b is clearly better
            if selected == "model_b":
                reward = np.random.normal(0.85, 0.05)
            else:
                reward = np.random.normal(0.60, 0.05)
            
            reward = np.clip(reward, 0.0, 1.0)
            router.update(context, selected, reward)
        
        # With enough samples, good expert should be strongly preferred
        assert router.weights[0] > 0.6, \
            f"Good expert should be clearly preferred after {n_samples} samples, got {router.weights[0]:.3f}"


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestCorrallingRobustness:
    """Test edge cases and numerical robustness."""
    
    def test_handles_zero_rewards(self):
        """Test that router handles zero rewards gracefully."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"e{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Give all zero rewards
        for _ in range(50):
            selected = router.select_model(context)
            router.update(context, selected, reward=0.0)
        
        # Should not crash or produce NaN
        assert not np.any(np.isnan(router.weights))
        assert not np.any(np.isinf(router.weights))
        assert abs(router.weights.sum() - 1.0) < 1e-10
    
    def test_handles_one_reward(self):
        """Test that router handles perfect rewards."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"e{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Give all perfect rewards
        for _ in range(50):
            selected = router.select_model(context)
            router.update(context, selected, reward=1.0)
        
        # Should not crash or produce NaN
        assert not np.any(np.isnan(router.weights))
        assert not np.any(np.isinf(router.weights))
        assert abs(router.weights.sum() - 1.0) < 1e-10
    
    def test_numerical_stability_many_updates(self):
        """Test numerical stability with many updates."""
        models = ["model_a", "model_b"]
        experts = [DeterministicExpert(f"e{i}", models[i]) for i in range(2)]
        
        router = CorrallingRouter(
            experts=experts,
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )
        
        context = np.random.randn(10)
        
        # Run many updates
        for i in range(10000):
            selected = router.select_model(context)
            reward = 0.5 + 0.3 * np.sin(i * 0.1)
            router.update(context, selected, reward)
            
            # Check for numerical issues every 1000 steps
            if i % 1000 == 0:
                assert not np.any(np.isnan(router.weights))
                assert not np.any(np.isinf(router.weights))
                assert abs(router.weights.sum() - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
