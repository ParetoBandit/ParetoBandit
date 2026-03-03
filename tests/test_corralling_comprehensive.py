"""
Comprehensive unit tests for CorrallingRouter.

Tests based on experiments from experiments/03_figure/ and experiments/appendix/04_figure/
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
    
    def select_model(self, context: np.ndarray, total_steps: int = 0, **kwargs) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, weight=1.0, cost=0.0):
        self.update_count += 1


class AdaptiveExpert:
    """Expert that adapts based on context."""
    
    def __init__(self, name: str, models: list, threshold: float = 0.5):
        self.name = name
        self.models = models
        self.threshold = threshold
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0, **kwargs) -> str:
        # Select based on context sum
        if np.sum(context) > self.threshold:
            return self.models[0]
        return self.models[1]
    
    def update(self, context, model, reward, weight=1.0, cost=0.0):
        self.update_count += 1


class SmartExpert:
    """Expert that occasionally explores but mostly exploits."""
    
    def __init__(self, name: str, best_model: str, models: list, explore_rate: float = 0.05):
        self.name = name
        self.best_model = best_model
        self.models = models
        self.explore_rate = explore_rate
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0, **kwargs) -> str:
        if np.random.random() < self.explore_rate:
            return np.random.choice([m for m in self.models if m != self.best_model])
        return self.best_model
    
    def update(self, context, model, reward, weight=1.0, cost=0.0):
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
        
        # Cumulative losses initialized from weights: L_i = -ln(w_i) / η
        # With uniform weights [0.5, 0.5] and η=1.0: L_i = -ln(0.5) / 1.0 ≈ 0.693147
        expected_losses = -np.log(np.array([0.5, 0.5])) / 1.0
        np.testing.assert_array_almost_equal(router.cumulative_losses, expected_losses)
    
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
            selected, _token = router.select_model(context)
            assert selected in models, f"Selected model {selected} not in {models}"
    
    def test_action_sampling_distribution(self):
        """Test that actions are sampled according to marginal probabilities."""
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
        action_counts = {"model_a": 0, "model_b": 0}
        
        np.random.seed(42)
        for _ in range(n_trials):
            model, token = router.select_model(context)
            action_counts[model] += 1
            assert "action_prob" in token
            assert "endorsing_experts" in token
        
        # With uniform weights and disjoint experts, should be roughly 50/50
        ratio = action_counts["model_a"] / n_trials
        assert 0.4 < ratio < 0.6, f"Expected ~50% model_a, got {ratio:.2%}"
    
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
        selected_model, token = router.select_model(context)
        router.update(context, selected_model, reward=0.8, selection_token=token)
        
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
            selected, token = router.select_model(context)
            reward = 0.5 + 0.4 * np.sin(i)  # Varying rewards
            router.update(context, selected, reward, selection_token=token)
            
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
            selected, token = router.select_model(context)
            router.update(context, selected, reward=0.6, selection_token=token)  # < 1.0, so there's loss
        
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
            selected, token = router.select_model(context)
            router.update(context, selected, reward=0.8, selection_token=token)
        
        # All experts observe every update (base algorithms must see all
        # feedback).  With 2 experts and 20 rounds, each expert
        # gets 20 updates → 40 total.
        total_updates = expert1.update_count + expert2.update_count
        assert total_updates == 40, f"Expected 40 updates total (2 experts × 20 rounds), got {total_updates}"
        assert expert1.update_count == 20, f"Expert 1 should have 20 updates, got {expert1.update_count}"
        assert expert2.update_count == 20, f"Expert 2 should have 20 updates, got {expert2.update_count}"


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
            selected, token = router.select_model(context)
            
            # Reward based on which model was selected
            if selected == "model_a":
                reward = 0.9
            else:
                reward = 0.3
            
            router.update(context, selected, reward, selection_token=token)
        
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
            selected, token = router.select_model(context)
            reward = 0.9 if selected == "model_a" else 0.3
            router.update(context, selected, reward, selection_token=token)
        
        weights_phase1 = router.weights.copy()
        
        # Phase 2: model_b becomes better (distribution shift)
        for _ in range(100):
            selected, token = router.select_model(context)
            reward = 0.3 if selected == "model_a" else 0.9  # Reversed!
            router.update(context, selected, reward, selection_token=token)
        
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
            selected_slow, token_slow = router_slow.select_model(context)
            reward = 0.9 if selected_slow == "model_a" else 0.3
            router_slow.update(context, selected_slow, reward, selection_token=token_slow)
            
            # Fast router (same feedback)
            np.random.seed(42 + _)
            selected_fast, token_fast = router_fast.select_model(context)
            reward = 0.9 if selected_fast == "model_a" else 0.3
            router_fast.update(context, selected_fast, reward, selection_token=token_fast)
        
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
        
        # Warmup expert: always picks model_a
        warmup = DeterministicExpert("warmup", "model_a")
        
        # Tabula rasa: picks model_b (context sum < threshold)
        tabula_rasa = AdaptiveExpert("tabula_rasa", models, threshold=0.5)
        
        router = CorrallingRouter(
            experts=[warmup, tabula_rasa],
            models=models,
            learning_rate=5.0,
            gamma=0.10
        )
        
        # Context sum = 0.3 < 0.5, so tabula_rasa picks model_b.
        # Experts disagree → Exp4 can differentiate them.
        context = np.array([0.03] * 10)
        
        for _ in range(100):
            selected, token = router.select_model(context)
            
            # model_b is actually better (contradicting warmup)
            reward = 0.3 if selected == "model_a" else 0.9
            router.update(context, selected, reward, selection_token=token)
        
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
            selected, token = router.select_model(context)
            
            # model_b is clearly better
            if selected == "model_b":
                reward = np.random.normal(0.85, 0.05)
            else:
                reward = np.random.normal(0.60, 0.05)
            
            reward = np.clip(reward, 0.0, 1.0)
            router.update(context, selected, reward, selection_token=token)
        
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
            selected, token = router.select_model(context)
            router.update(context, selected, reward=0.0, selection_token=token)
        
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
            selected, token = router.select_model(context)
            router.update(context, selected, reward=1.0, selection_token=token)
        
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
            selected, token = router.select_model(context)
            reward = 0.5 + 0.3 * np.sin(i * 0.1)
            router.update(context, selected, reward, selection_token=token)
            
            # Check for numerical issues every 1000 steps
            if i % 1000 == 0:
                assert not np.any(np.isnan(router.weights))
                assert not np.any(np.isinf(router.weights))
                assert abs(router.weights.sum() - 1.0) < 1e-6


# =============================================================================
# Exp4 Loss Attribution Tests
# =============================================================================

class TestExp4LossAttribution:
    """Test that the Exp4 estimator distributes loss correctly."""

    def test_agreeing_experts_share_loss(self):
        """When both experts endorse the same action, both accumulate loss."""
        models = ["model_a", "model_b"]
        expert1 = DeterministicExpert("e1", "model_a")
        expert2 = DeterministicExpert("e2", "model_a")  # Same recommendation

        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.05
        )

        context = np.random.randn(10)
        initial_losses = router.cumulative_losses.copy()

        model, token = router.select_model(context)
        assert model == "model_a"
        assert token["action_prob"] == pytest.approx(1.0)
        assert set(token["endorsing_experts"]) == {0, 1}

        router.update(context, model, reward=0.0, selection_token=token)

        # Both experts should have accumulated loss
        assert router.cumulative_losses[0] > initial_losses[0]
        assert router.cumulative_losses[1] > initial_losses[1]
        # And the same amount (both endorsed the same action)
        delta = router.cumulative_losses - initial_losses
        assert abs(delta[0] - delta[1]) < 1e-10

    def test_disagreeing_experts_single_loss(self):
        """When experts disagree, only the endorsing expert gets loss."""
        models = ["model_a", "model_b"]
        expert1 = DeterministicExpert("e1", "model_a")
        expert2 = DeterministicExpert("e2", "model_b")

        router = CorrallingRouter(
            experts=[expert1, expert2],
            models=models,
            learning_rate=1.0,
            gamma=0.0,
            loss_decay=1.0,
            meta_lr_halflife=float("inf"),
        )

        initial_losses = router.cumulative_losses.copy()
        context = np.random.randn(10)

        np.random.seed(1)
        model, token = router.select_model(context)
        endorsing = token["endorsing_experts"]
        non_endorsing = [j for j in range(2) if j not in endorsing]

        router.update(context, model, reward=0.0, selection_token=token)

        for j in endorsing:
            assert router.cumulative_losses[j] > initial_losses[j]
        for j in non_endorsing:
            assert router.cumulative_losses[j] == pytest.approx(
                initial_losses[j], abs=1e-10
            )

    def test_consensus_reduces_loss_magnitude(self):
        """π(a)=1 when experts agree → loss/1.0 < loss/0.5 when they disagree."""
        models = ["model_a", "model_b"]
        agree_1 = DeterministicExpert("a1", "model_a")
        agree_2 = DeterministicExpert("a2", "model_a")
        disagree_1 = DeterministicExpert("d1", "model_a")
        disagree_2 = DeterministicExpert("d2", "model_b")

        router_agree = CorrallingRouter(
            experts=[agree_1, agree_2], models=models,
            learning_rate=1.0, gamma=0.05,
            meta_lr_halflife=float("inf"),
        )
        router_disagree = CorrallingRouter(
            experts=[disagree_1, disagree_2], models=models,
            learning_rate=1.0, gamma=0.05,
            meta_lr_halflife=float("inf"),
        )

        initial_agree = router_agree.cumulative_losses.copy()
        initial_disagree = router_disagree.cumulative_losses.copy()

        context = np.random.randn(10)
        _, tok_a = router_agree.select_model(context)
        # Force selection of model_a for the disagreeing router
        np.random.seed(1)
        model_d, tok_d = router_disagree.select_model(context)
        while model_d != "model_a":
            np.random.seed(np.random.randint(1000))
            model_d, tok_d = router_disagree.select_model(context)

        router_agree.update(context, "model_a", reward=0.0, selection_token=tok_a)
        router_disagree.update(context, model_d, reward=0.0, selection_token=tok_d)

        # Agreeing: IW loss = 1.0 / 1.0 = 1.0 to both experts
        # Disagreeing: IW loss = 1.0 / ~0.525 ≈ 1.9 to the endorsing expert
        delta_agree = router_agree.cumulative_losses - initial_agree
        delta_disagree = router_disagree.cumulative_losses - initial_disagree
        assert max(delta_agree) < max(delta_disagree)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
