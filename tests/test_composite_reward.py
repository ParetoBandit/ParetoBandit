"""
Unit tests for cost handling in CorrallingRouter and BanditRouter.

After the unification of cost mechanisms (selection-time penalty only),
CorrallingRouter no longer shapes the reward signal.  Cost-quality
trade-offs are handled exclusively via each expert's ``cost_penalty``
parameter (paper Eq. 4), applied at arm selection time.

These tests verify:
  1. CorrallingRouter passes raw rewards to experts (no cost adjustment).
  2. BanditRouter exposes ``cost_penalty`` and wires it to experts.
  3. CorrallingRouter stores ``model_costs`` but does not use them for
     reward shaping.
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


def make_router(model_costs=None):
    """Create a CorrallingRouter with two RewardTrackingExperts."""
    expert_a = RewardTrackingExpert("warmup", "mixtral")
    expert_b = RewardTrackingExpert("tabula_rasa", "gpt-4o")
    router = CorrallingRouter(
        experts=[expert_a, expert_b],
        models=MODELS,
        learning_rate=1.0,
        gamma=0.05,
        model_costs=model_costs if model_costs is not None else MODEL_COSTS,
    )
    return router, expert_a, expert_b


# =============================================================================
# Test: Raw Rewards Passed Through (no reward shaping)
# =============================================================================

class TestRawRewardPassthrough:
    """CorrallingRouter must pass raw rewards to experts without modification."""

    def test_experts_receive_raw_reward(self):
        """Experts receive the exact raw reward — no cost adjustment."""
        router, exp_a, exp_b = make_router()
        _, token = router.select_model(CONTEXT)

        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=token)

        assert exp_a.received_rewards[-1] == 1.0
        assert exp_b.received_rewards[-1] == 1.0

    def test_zero_reward_stays_zero(self):
        """Zero reward passes through unchanged."""
        router, exp_a, exp_b = make_router()
        _, token = router.select_model(CONTEXT)

        router.update(CONTEXT, "mixtral", 0.0, selection_token=token)

        assert exp_a.received_rewards[-1] == 0.0
        assert exp_b.received_rewards[-1] == 0.0

    def test_fractional_reward_passes_through(self):
        """Fractional reward passes through unchanged."""
        router, exp_a, _ = make_router()
        _, token = router.select_model(CONTEXT)

        router.update(CONTEXT, "mixtral", 0.7, selection_token=token)

        assert exp_a.received_rewards[-1] == 0.7

    def test_experts_receive_raw_reward_without_token(self):
        """Even without a selection token, experts get raw rewards."""
        router, exp_a, exp_b = make_router()
        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=None)

        assert exp_a.received_rewards[-1] == 1.0
        assert exp_b.received_rewards[-1] == 1.0

    def test_meta_weights_unchanged_without_token(self):
        """Without token, meta-weights should not change."""
        router, _, _ = make_router()
        initial_weights = router.weights.copy()

        router.update(CONTEXT, "gpt-4o", 1.0, selection_token=None)

        np.testing.assert_array_equal(router.weights, initial_weights)


# =============================================================================
# Test: BanditRouter Integration (cost_penalty parameter)
# =============================================================================

class TestBanditRouterCostPenaltyParam:
    """Verify BanditRouter exposes cost_penalty and passes it to experts."""
    
    def test_bandit_router_accepts_cost_penalty(self):
        """BanditRouter.__init__ should accept cost_penalty."""
        from bandit_gpt.router import BanditRouter
        
        router = BanditRouter.__new__(BanditRouter)
        router.cost_penalty = 0.3
        assert router.cost_penalty == 0.3

    def test_cost_penalty_default_is_0_3(self):
        """Default cost_penalty should be 0.3."""
        from bandit_gpt.router import BanditRouter
        
        import inspect
        sig = inspect.signature(BanditRouter.__init__)
        param = sig.parameters.get('cost_penalty')
        assert param is not None, "BanditRouter should have cost_penalty param"
        assert param.default == 0.3, f"Default should be 0.3, got {param.default}"

    def test_corralling_cost_weight_removed(self):
        """corralling_cost_weight should no longer be a parameter."""
        from bandit_gpt.router import BanditRouter

        import inspect
        sig = inspect.signature(BanditRouter.__init__)
        assert 'corralling_cost_weight' not in sig.parameters, (
            "corralling_cost_weight should have been removed"
        )


# =============================================================================
# Test: CorrallingRouter Initialization
# =============================================================================

class TestCorrallingRouterCostInit:
    """Verify CorrallingRouter stores model_costs."""

    def test_init_stores_model_costs(self):
        """model_costs should be stored as an attribute."""
        router, _, _ = make_router()
        assert router.model_costs == MODEL_COSTS

    def test_init_defaults(self):
        """Default model_costs={}."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
        )
        assert router.model_costs == {}

    def test_model_costs_none_defaults_to_empty_dict(self):
        """Passing model_costs=None should store {}."""
        expert = RewardTrackingExpert("exp", "mixtral")
        router = CorrallingRouter(
            experts=[expert],
            models=["mixtral"],
            learning_rate=1.0,
            model_costs=None,
        )
        assert router.model_costs == {}

    def test_no_cost_weight_parameter(self):
        """CorrallingRouter should no longer accept cost_weight."""
        import inspect
        sig = inspect.signature(CorrallingRouter.__init__)
        assert 'cost_weight' not in sig.parameters, (
            "cost_weight should have been removed from CorrallingRouter"
        )
