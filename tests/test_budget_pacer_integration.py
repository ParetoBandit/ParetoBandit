"""Integration tests: BudgetPacer wired into BanditRouter.

Tests verify that:
  1. The pacer is correctly threaded through route() and process_feedback().
  2. Existing behaviour is preserved when no pacer is provided (no-regression).
  3. Hard mode excludes expensive models once the dual variable rises.
  4. Soft mode shifts preference toward cheaper models.
  5. RoutingLog contains pacer diagnostics when a pacer is active.
  6. process_feedback() calls pacer.observe().
  7. Hard ceiling relaxation fallback prevents NoEligibleModelsError.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from pareto_bandit import BanditRouter
from pareto_bandit.budget_pacer import BudgetPacer, PacingMode

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def two_model_registry():
    """Two-model portfolio with a 100x cost gap."""
    return {
        "expensive/model-a": {
            "model_id": "expensive/model-a",
            "display_name": "Expensive A",
            "scores": {"hle": 0.90},
            "hallucination_rate": 1.0,
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
        },
        "cheap/model-b": {
            "model_id": "cheap/model-b",
            "display_name": "Cheap B",
            "scores": {"hle": 0.50},
            "hallucination_rate": 5.0,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1,
        },
    }


def _make_router(
    registry,
    budget_pacer=None,
    cost_penalty: float = 0.0,
):
    """Build a minimal BanditRouter for testing."""
    return BanditRouter.create(
        model_registry=registry,
        priors="none",
        cost_penalty=cost_penalty,
        budget_pacer=budget_pacer,
    )


# ======================================================================
# 1. No-regression: pacer=None preserves existing behaviour
# ======================================================================


class TestNoRegression:
    """Router with budget_pacer=None behaves identically to the original."""

    def test_no_pacer_route_unchanged(self, two_model_registry):
        router = _make_router(two_model_registry)
        assert router.budget_pacer is None

        model, log = router.route("Hello world")
        assert model in two_model_registry
        assert log.pacer_lambda_t is None
        assert log.pacer_cost_ema is None

    def test_no_pacer_feedback_works(self, two_model_registry):
        router = _make_router(two_model_registry)
        _, log = router.route("Test feedback")
        router.process_feedback(log.request_id, 0.8)


# ======================================================================
# 2. RoutingLog contains pacer state
# ======================================================================


class TestRoutingLogPacerState:
    """RoutingLog should contain pacer_lambda_t and pacer_cost_ema."""

    def test_log_has_pacer_fields(self, two_model_registry):
        pacer = BudgetPacer(target_avg_spend_usd=0.001)
        router = _make_router(two_model_registry, budget_pacer=pacer)

        _, log = router.route("Check log fields")
        assert log.pacer_lambda_t is not None
        assert log.pacer_cost_ema is not None
        assert log.pacer_lambda_t == 0.0
        assert log.pacer_cost_ema == 0.001


# ======================================================================
# 3. process_feedback calls pacer.observe()
# ======================================================================


class TestPacerObserve:
    """process_feedback must call budget_pacer.observe() with the cost."""

    def test_observe_called(self, two_model_registry):
        pacer = BudgetPacer(target_avg_spend_usd=0.01)
        router = _make_router(two_model_registry, budget_pacer=pacer)
        assert pacer.n_observations == 0

        _, log = router.route("Observe test")
        router.process_feedback(log.request_id, 0.9)
        assert pacer.n_observations == 1

    def test_observe_accumulates(self, two_model_registry):
        pacer = BudgetPacer(target_avg_spend_usd=0.01)
        router = _make_router(two_model_registry, budget_pacer=pacer)

        for i in range(10):
            _, log = router.route(f"Prompt {i}")
            router.process_feedback(log.request_id, 0.8)

        assert pacer.n_observations == 10


# ======================================================================
# 4. Hard mode: expensive model excluded when overspending
# ======================================================================


class TestHardMode:
    """HARD pacing should tighten the cost ceiling and exclude expensive models."""

    def test_hard_mode_excludes_expensive_after_overspend(self, two_model_registry):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01,
            mode=PacingMode.HARD,
            lr=0.1,
        )
        router = _make_router(two_model_registry, budget_pacer=pacer)

        for _ in range(50):
            pacer.observe(0.05)

        assert pacer.lambda_t > 0.0

        expensive_blended_per_1k = (10.0 + 30.0) / 2.0 / 1000.0  # 0.02
        cheap_blended_per_1k = (0.1 + 0.1) / 2.0 / 1000.0        # 0.0001
        ceiling = pacer.get_cost_ceiling_per_1k(expensive_blended_per_1k)
        assert ceiling is not None
        assert ceiling < expensive_blended_per_1k, (
            f"Ceiling {ceiling} should exclude expensive model "
            f"(cost={expensive_blended_per_1k})"
        )
        assert ceiling > cheap_blended_per_1k, (
            f"Ceiling {ceiling} should still allow cheap model "
            f"(cost={cheap_blended_per_1k})"
        )

        selections = []
        for i in range(20):
            model, _ = router.route(f"Hard test {i}")
            selections.append(model)
        assert all(
            m == "cheap/model-b" for m in selections
        ), f"Expected only cheap model, got: {set(selections)}"

    def test_hard_pacing_with_user_max_cost_takes_minimum(self, two_model_registry):
        """User-supplied max_cost and pacer ceiling should use the tighter one."""
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01,
            mode=PacingMode.HARD,
            lr=0.1,
        )
        router = _make_router(two_model_registry, budget_pacer=pacer)

        for _ in range(50):
            pacer.observe(0.05)

        user_max_cost = 100.0
        model, _ = router.route("Max cost test", max_cost=user_max_cost)
        assert model == "cheap/model-b", (
            "Pacer ceiling should be tighter than generous user max_cost"
        )


# ======================================================================
# 5. Soft mode: preference shifts toward cheaper model
# ======================================================================


class TestSoftMode:
    """SOFT pacing should penalize the expensive model in UCB scoring."""

    def test_soft_mode_shifts_preference(self, two_model_registry):
        pacer_inactive = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.SOFT
        )
        router_baseline = _make_router(
            two_model_registry, budget_pacer=pacer_inactive
        )

        pacer_active = BudgetPacer(
            target_avg_spend_usd=0.00001,
            mode=PacingMode.SOFT,
            lr=1.0,
        )
        router_paced = _make_router(
            two_model_registry, budget_pacer=pacer_active
        )
        for _ in range(50):
            pacer_active.observe(0.01)

        assert pacer_active.lambda_t > 0.0

        n_trials = 100
        np.random.seed(42)

        cheap_count_baseline = 0
        for i in range(n_trials):
            m, _ = router_baseline.route(f"Baseline {i}")
            if m == "cheap/model-b":
                cheap_count_baseline += 1

        cheap_count_paced = 0
        for i in range(n_trials):
            m, _ = router_paced.route(f"Paced {i}")
            if m == "cheap/model-b":
                cheap_count_paced += 1

        assert cheap_count_paced >= cheap_count_baseline, (
            f"Soft pacing should increase cheap model selection: "
            f"paced={cheap_count_paced}, baseline={cheap_count_baseline}"
        )


# ======================================================================
# 6. Adaptive mode: both mechanisms active
# ======================================================================


class TestAdaptiveMode:
    """ADAPTIVE mode should enable both hard and soft mechanisms."""

    def test_adaptive_uses_both(self, two_model_registry):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01,
            mode=PacingMode.ADAPTIVE,
            lr=0.1,
        )
        router = _make_router(two_model_registry, budget_pacer=pacer)

        assert pacer.uses_hard
        assert pacer.uses_soft

        for _ in range(50):
            pacer.observe(0.05)

        assert pacer.lambda_t > 0.0

        expensive_blended_per_1k = (10.0 + 30.0) / 2.0 / 1000.0
        ceiling = pacer.get_cost_ceiling_per_1k(expensive_blended_per_1k)
        assert ceiling is not None

        model, log = router.route("Adaptive test")
        assert model in two_model_registry
        assert log.pacer_lambda_t > 0.0


# ======================================================================
# 7. Budget pacer passthrough: extra_cost_penalties passed through
# ======================================================================


class TestBudgetPacerPassthrough:
    """extra_cost_penalties should flow through the routing path."""

    def test_routing_with_pacer(self, two_model_registry):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01,
            mode=PacingMode.SOFT,
        )
        router = _make_router(
            two_model_registry,
            budget_pacer=pacer,
        )

        model, log = router.route("Budget pacer test")
        assert model in two_model_registry
        router.process_feedback(log.request_id, 0.9)
        assert pacer.n_observations == 1


# ======================================================================
# 8. Hard ceiling relaxation fallback
# ======================================================================


class TestHardCeilingFallback:
    """When the hard ceiling excludes all models, the router should
    fall back to the user's max_cost (or no ceiling) instead of raising."""

    def test_relaxation_when_ceiling_too_tight(self, two_model_registry):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.0000001,
            mode=PacingMode.HARD,
            lr=1.0,
            lambda_max=100.0,
        )
        router = _make_router(two_model_registry, budget_pacer=pacer)

        for _ in range(100):
            pacer.observe(1.0)

        assert pacer.lambda_t == 100.0

        model, log = router.route("Fallback test")
        assert model in two_model_registry, (
            "Router should fall back gracefully when ceiling is too tight"
        )
