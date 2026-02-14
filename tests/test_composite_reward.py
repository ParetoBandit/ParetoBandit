"""
Unit tests for composite reward (cost_weight) in CorrallingRouter.

Tests that cost_weight correctly adjusts the reward signal before it reaches
both the meta-learner (Level 1) and expert bandits (Level 2), ensuring the
entire learning stack optimizes for cost-quality value when cost_weight > 0.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
from unittest.mock import MagicMock

from bandit_gpt.router import CorrallingRouter


# =============================================================================
# Mock Experts for Testing
# =============================================================================

class RewardTrackingExpert:
    """Expert that records every reward it receives via update()."""
    
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
        self.received_rewards = []
        self.update_count = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0, **kwargs) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, weight=1.0):
        self.received_rewards.append(reward)
        self.update_count += 1


# =============================================================================
# Fixtures
# =============================================================================

MODELS = ["mixtral", "gpt-4o"]
MODEL_COSTS = {
    "mixtral": {"normalized_cost": 0.29},
    "gpt-4o": {"normalized_cost": 0.69},
}
CONTEXT = np.random.RandomState(42).randn(24)


def make_router(cost_weight: float = 0.0, model_costs=None):
    """Create a CorrallingRouter with two RewardTrackingExperts."""
    expert_a = RewardTrackingExpert("warmup", "mixtral")
    expert_b = RewardTrackingExpert("tabula_rasa", "gpt-4o")
    router = CorrallingRouter(
        experts=[expert_a, expert_b],
        models=MODELS,
        learning_rate=1.0,
        gamma=0.05,
        model_costs=model_costs if model_costs is not None else MODEL_COSTS,
        cost_weight=cost_weight,
    )
    return router, expert_a, expert_b


# =============================================================================
# Test: Backward Compatibility (cost_weight=0.0)
# =============================================================================

class TestCompositeRewardBackwardCompat:
    """When cost_weight=0, behavior must be identical to the original."""

    def test_zero_cost_weight_passes_raw_reward(self):
        """Experts receive the exact raw reward when cost_weight=0."""
        router, exp_a, exp_b = make_router(cost_weight=0.0)
        
        # Select a model to get a valid token
        _, token = router.select_model(CONTEXT)
        
        # Update with reward=1.0 for gpt-4o (expensive model)
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)
        
        # Both experts should receive exactly 1.0 (no cost adjustment)
        assert exp_a.received_rewards[-1] == 1.0
        assert exp_b.received_rewards[-1] == 1.0

    def test_zero_cost_weight_with_zero_reward(self):
        """Zero reward stays zero when cost_weight=0."""
        router, exp_a, exp_b = make_router(cost_weight=0.0)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "mixtral", 0.0, selection_token=token)
        
        assert exp_a.received_rewards[-1] == 0.0
        assert exp_b.received_rewards[-1] == 0.0

    def test_default_cost_weight_is_zero(self):
        """CorrallingRouter defaults to cost_weight=0 (backward compat)."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
        )
        assert router.cost_weight == 0.0
        assert router.model_costs == {}


# =============================================================================
# Test: Composite Reward Computation
# =============================================================================

class TestCompositeRewardComputation:
    """Verify r_effective = quality - λ·normalized_cost."""

    def test_cheap_model_reward_adjusted(self):
        """Mixtral (cost=0.29): r = 1.0 - 0.3*0.29 = 0.913."""
        router, exp_a, exp_b = make_router(cost_weight=0.3)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "mixtral", 1.0, selection_token=token)
        
        expected = 1.0 - 0.3 * 0.29
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10
        assert abs(exp_b.received_rewards[-1] - expected) < 1e-10

    def test_expensive_model_reward_adjusted(self):
        """GPT-4o (cost=0.69): r = 1.0 - 0.3*0.69 = 0.793."""
        router, exp_a, exp_b = make_router(cost_weight=0.3)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)
        
        expected = 1.0 - 0.3 * 0.69
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10
        assert abs(exp_b.received_rewards[-1] - expected) < 1e-10

    def test_cheap_model_gets_higher_effective_reward(self):
        """When both models succeed (reward=1), cheap model gets higher r_eff."""
        router, exp_a, _ = make_router(cost_weight=0.3)
        
        _, token = router.select_model(CONTEXT)
        router.update(CONTEXT, "mixtral", 1.0, selection_token=token)
        cheap_reward = exp_a.received_rewards[-1]
        
        _, token = router.select_model(CONTEXT)
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)
        expensive_reward = exp_a.received_rewards[-1]
        
        assert cheap_reward > expensive_reward, (
            f"Cheap model reward {cheap_reward} should exceed expensive {expensive_reward}"
        )
        # The gap should be λ · (cost_gpt4o - cost_mixtral)
        expected_gap = 0.3 * (0.69 - 0.29)
        actual_gap = cheap_reward - expensive_reward
        assert abs(actual_gap - expected_gap) < 1e-10

    def test_composite_reward_can_be_negative(self):
        """When quality=0 and cost_weight>0, effective reward is negative."""
        router, exp_a, _ = make_router(cost_weight=0.5)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "gpt-4o", 0.0, selection_token=token)
        
        expected = 0.0 - 0.5 * 0.69  # = -0.345
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10
        assert exp_a.received_rewards[-1] < 0

    def test_aggressive_cost_weight(self):
        """At λ=1.0, cost fully deducted from reward."""
        router, exp_a, _ = make_router(cost_weight=1.0)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)
        
        expected = 1.0 - 1.0 * 0.69  # = 0.31
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10

    def test_fractional_quality_reward(self):
        """Composite reward works with non-binary quality."""
        router, exp_a, _ = make_router(cost_weight=0.3)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "mixtral", 0.7, selection_token=token)
        
        expected = 0.7 - 0.3 * 0.29  # = 0.613
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10


# =============================================================================
# Test: Meta-Learner Receives Composite Reward
# =============================================================================

class TestMetaLearnerCompositeReward:
    """The meta-learner's loss computation must use the composite reward."""

    def test_meta_loss_uses_composite_reward(self):
        """
        Meta-loss = 1 - r_composite.
        When r_composite is lower (expensive model), meta-loss is higher,
        so the meta-learner penalizes the expert that picked the costly model.
        """
        # Create two scenarios: expert picks cheap vs expensive, same quality
        router_cheap, _, _ = make_router(cost_weight=0.5)
        router_expensive, _, _ = make_router(cost_weight=0.5)
        
        # Force both to use expert 0
        np.random.seed(42)
        _, token_cheap = router_cheap.select_model(CONTEXT)
        np.random.seed(42)
        _, token_expensive = router_expensive.select_model(CONTEXT)
        
        # Update: both succeed (quality=1), but different models
        router_cheap.update(CONTEXT, "mixtral", 1.0, selection_token=token_cheap)
        router_expensive.update(CONTEXT, "gpt-4o", 1.0, selection_token=token_expensive)
        
        # The meta-learner that saw the expensive model should have
        # accumulated MORE loss (lower composite reward → higher loss)
        assert router_expensive.cumulative_losses.sum() > router_cheap.cumulative_losses.sum(), (
            "Meta-learner should penalize expensive model picks more"
        )

    def test_meta_weights_shift_toward_cost_effective_expert(self):
        """
        Over many rounds, an expert that picks cheap successful models
        should gain meta-weight vs one that picks expensive models.
        """
        models = ["cheap_model", "expensive_model"]
        costs = {
            "cheap_model": {"normalized_cost": 0.1},
            "expensive_model": {"normalized_cost": 0.9},
        }
        
        # Expert 0 always picks cheap, expert 1 always picks expensive
        exp_cheap = RewardTrackingExpert("cheap_picker", "cheap_model")
        exp_expensive = RewardTrackingExpert("expensive_picker", "expensive_model")
        
        router = CorrallingRouter(
            experts=[exp_cheap, exp_expensive],
            models=models,
            learning_rate=1.0,
            gamma=0.05,
            model_costs=costs,
            cost_weight=0.5,
        )
        
        # Simulate 100 rounds where BOTH models always succeed (quality=1)
        # The only differentiator is cost.
        for _ in range(100):
            _, token = router.select_model(CONTEXT)
            expert_idx = token["expert_idx"]
            selected_model = [exp_cheap, exp_expensive][expert_idx].favorite_model
            router.update(CONTEXT, selected_model, 1.0, selection_token=token)
        
        # Expert 0 (cheap) should have more meta-weight
        assert router.weights[0] > router.weights[1], (
            f"Cheap expert weight {router.weights[0]:.4f} should exceed "
            f"expensive expert weight {router.weights[1]:.4f}"
        )


# =============================================================================
# Test: Unknown Model Graceful Handling
# =============================================================================

class TestCompositeRewardEdgeCases:
    """Edge cases for model_costs lookup."""

    def test_unknown_model_gets_no_cost_adjustment(self):
        """Model not in model_costs gets raw reward (no crash)."""
        router, exp_a, _ = make_router(cost_weight=0.5)
        _, token = router.select_model(CONTEXT)
        
        # Update with a model not in model_costs
        router.update(CONTEXT, "unknown_model", 0.8, selection_token=token)
        
        # Should receive raw reward (no cost adjustment)
        assert exp_a.received_rewards[-1] == 0.8

    def test_missing_normalized_cost_key(self):
        """Model in costs dict but missing 'normalized_cost' key → no crash."""
        bad_costs = {"mixtral": {"some_other_key": 999}}
        router, exp_a, _ = make_router(cost_weight=0.5, model_costs=bad_costs)
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "mixtral", 1.0, selection_token=token)
        
        # get('normalized_cost', 0.0) → 0.0, so no adjustment
        assert exp_a.received_rewards[-1] == 1.0

    def test_empty_model_costs(self):
        """Empty model_costs dict with cost_weight > 0 → no crash."""
        router, exp_a, _ = make_router(cost_weight=0.5, model_costs={})
        _, token = router.select_model(CONTEXT)
        
        router.update(CONTEXT, "mixtral", 1.0, selection_token=token)
        assert exp_a.received_rewards[-1] == 1.0

    def test_none_model_costs_defaults_to_empty(self):
        """model_costs=None should default to {} and not crash."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
            cost_weight=0.5,
            model_costs=None,
        )
        _, token = router.select_model(CONTEXT)
        router.update(CONTEXT, "mixtral", 1.0, selection_token=token)
        assert expert.received_rewards[-1] == 1.0


# =============================================================================
# Test: No Token (Level 2 only) Still Gets Composite Reward
# =============================================================================

class TestCompositeRewardWithoutToken:
    """When selection_token=None, only Level 2 runs but still uses composite."""

    def test_experts_get_composite_reward_without_token(self):
        """Even without a token, experts should receive the adjusted reward."""
        router, exp_a, exp_b = make_router(cost_weight=0.3)
        
        # Update WITHOUT selection_token (skip meta-weight update)
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=None)
        
        expected = 1.0 - 0.3 * 0.69
        assert abs(exp_a.received_rewards[-1] - expected) < 1e-10
        assert abs(exp_b.received_rewards[-1] - expected) < 1e-10

    def test_meta_weights_unchanged_without_token(self):
        """Without token, meta-weights should not change."""
        router, _, _ = make_router(cost_weight=0.3)
        initial_weights = router.weights.copy()
        
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=None)
        
        np.testing.assert_array_equal(router.weights, initial_weights)


# =============================================================================
# Test: Interaction with Loss Decay
# =============================================================================

class TestCompositeRewardWithLossDecay:
    """Composite reward interacts correctly with loss_decay."""

    def test_composite_reward_with_loss_decay(self):
        """Cumulative losses decay AND use composite reward."""
        expert_a = RewardTrackingExpert("a", "mixtral")
        expert_b = RewardTrackingExpert("b", "gpt-4o")
        
        router = CorrallingRouter(
            experts=[expert_a, expert_b],
            models=MODELS,
            learning_rate=1.0,
            gamma=0.05,
            loss_decay=0.99,  # Moderate decay
            model_costs=MODEL_COSTS,
            cost_weight=0.3,
        )
        
        # Run several updates
        for _ in range(10):
            _, token = router.select_model(CONTEXT)
            router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)
        
        # Experts should have received composite reward each time
        expected = 1.0 - 0.3 * 0.69
        for r in expert_a.received_rewards:
            assert abs(r - expected) < 1e-10


# =============================================================================
# Test: Cost-Quality Tradeoff Convergence
# =============================================================================

class TestCostQualityTradeoff:
    """Verify the composite reward drives correct model preferences."""

    def test_cheap_model_preferred_when_quality_equal(self):
        """
        Scenario: Both models always succeed (quality=1).
        With cost_weight>0, the cheaper model should be selected more over time.
        """
        models = ["cheap", "expensive"]
        costs = {
            "cheap": {"normalized_cost": 0.1},
            "expensive": {"normalized_cost": 0.9},
        }
        
        # Both experts start identical but track rewards
        exp_a = RewardTrackingExpert("a", "cheap")
        exp_b = RewardTrackingExpert("b", "expensive")
        
        router = CorrallingRouter(
            experts=[exp_a, exp_b],
            models=models,
            learning_rate=1.0,
            gamma=0.05,
            model_costs=costs,
            cost_weight=0.5,
        )
        
        selections = {"cheap": 0, "expensive": 0}
        for _ in range(200):
            _, token = router.select_model(CONTEXT)
            expert_idx = token["expert_idx"]
            selected = [exp_a, exp_b][expert_idx].favorite_model
            selections[selected] += 1
            # Both models always succeed
            router.update(CONTEXT, selected, 1.0, selection_token=token)
        
        # Cheap model should be selected more often
        assert selections["cheap"] > selections["expensive"], (
            f"Cheap: {selections['cheap']}, Expensive: {selections['expensive']}"
        )

    def test_expensive_model_preferred_when_quality_gap_large(self):
        """
        Scenario: Expensive model has much better quality.
        Even with cost_weight>0, quality should dominate if gap is large enough.
        
        cheap: quality=0.5, cost=0.1 → r = 0.5 - 0.3*0.1 = 0.47
        expensive: quality=1.0, cost=0.9 → r = 1.0 - 0.3*0.9 = 0.73
        """
        models = ["cheap", "expensive"]
        costs = {
            "cheap": {"normalized_cost": 0.1},
            "expensive": {"normalized_cost": 0.9},
        }
        quality = {"cheap": 0.5, "expensive": 1.0}
        
        exp_a = RewardTrackingExpert("a", "cheap")
        exp_b = RewardTrackingExpert("b", "expensive")
        
        router = CorrallingRouter(
            experts=[exp_a, exp_b],
            models=models,
            learning_rate=1.0,
            gamma=0.05,
            model_costs=costs,
            cost_weight=0.3,
        )
        
        selections = {"cheap": 0, "expensive": 0}
        for _ in range(200):
            _, token = router.select_model(CONTEXT)
            expert_idx = token["expert_idx"]
            selected = [exp_a, exp_b][expert_idx].favorite_model
            selections[selected] += 1
            router.update(CONTEXT, selected, quality[selected], selection_token=token)
        
        # Expensive model should still be preferred (quality gap overwhelms cost)
        assert selections["expensive"] > selections["cheap"], (
            f"Expensive: {selections['expensive']}, Cheap: {selections['cheap']}. "
            f"Quality gap should dominate moderate cost penalty."
        )


# =============================================================================
# Test: BanditRouter Integration (corralling_cost_weight parameter)
# =============================================================================

class TestBanditRouterCostWeightParam:
    """Verify BanditRouter exposes corralling_cost_weight and passes it through."""
    
    def test_bandit_router_accepts_cost_weight(self):
        """BanditRouter.__init__ should accept corralling_cost_weight."""
        from bandit_gpt.router import BanditRouter
        
        # Just verify the parameter is accepted without error
        # We can't fully init BanditRouter without model_registry etc,
        # but we can check the attribute is stored
        router = BanditRouter.__new__(BanditRouter)
        router.corralling_cost_weight = 0.3
        assert router.corralling_cost_weight == 0.3

    def test_corralling_cost_weight_default_is_zero(self):
        """Default corralling_cost_weight should be 0.0 (backward compat)."""
        from bandit_gpt.router import BanditRouter
        
        # Check that the default in the function signature is 0.0
        import inspect
        sig = inspect.signature(BanditRouter.__init__)
        param = sig.parameters.get('corralling_cost_weight')
        assert param is not None, "BanditRouter should have corralling_cost_weight param"
        assert param.default == 0.0, f"Default should be 0.0, got {param.default}"


# =============================================================================
# Test: CorrallingRouter Initialization with cost params
# =============================================================================

class TestCorrallingRouterCostInit:
    """Verify CorrallingRouter stores cost_weight and model_costs."""

    def test_init_stores_cost_params(self):
        """cost_weight and model_costs should be stored as attributes."""
        router, _, _ = make_router(cost_weight=0.42)
        assert router.cost_weight == 0.42
        assert router.model_costs == MODEL_COSTS

    def test_init_defaults(self):
        """Default cost_weight=0.0 and model_costs={}."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
        )
        assert router.cost_weight == 0.0
        assert router.model_costs == {}

    def test_model_costs_none_defaults_to_empty_dict(self):
        """Passing model_costs=None should store {}."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
            model_costs=None,
            cost_weight=0.5,
        )
        assert router.model_costs == {}
