"""Unit tests for the BudgetPacer (Primal-Dual CBwK).

Tests are organised into seven groups:
  1. **Initialization** — validate defaults, mode properties, and input validation.
  2. **Dual-variable dynamics** — verify lambda_t and cost_ema evolve correctly
     under overspend, underspend, and mixed regimes.
  3. **Hard ceiling** — verify the cost ceiling returned by the HARD mechanism.
  4. **Soft penalties** — verify per-model penalty scaling in SOFT mode.
  5. **Reset** — verify state re-initialization.
  6. **Thread safety** — concurrent observe/read does not corrupt state.
  7. **Diagnostics** — verify get_state() and __repr__().
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from bandit_gpt.budget_pacer import BudgetPacer, PacingMode


# ======================================================================
# 1. Initialization and mode properties
# ======================================================================


class TestPacerInit:
    """Verify constructor defaults and input validation."""

    def test_init_defaults(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01)
        assert pacer.lambda_t == 0.0
        assert pacer.cost_ema == 0.01
        assert pacer.n_observations == 0
        assert pacer.mode is PacingMode.ADAPTIVE
        assert pacer.target_avg_spend_usd == 0.01
        assert pacer.lambda_max == 5.0

    def test_mode_properties_hard(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.HARD)
        assert pacer.uses_hard is True
        assert pacer.uses_soft is False

    def test_mode_properties_soft(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.SOFT)
        assert pacer.uses_hard is False
        assert pacer.uses_soft is True

    def test_mode_properties_adaptive(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.ADAPTIVE)
        assert pacer.uses_hard is True
        assert pacer.uses_soft is True

    def test_zero_target_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BudgetPacer(target_avg_spend_usd=0.0)

    def test_negative_target_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BudgetPacer(target_avg_spend_usd=-0.01)

    def test_invalid_ema_alpha_raises(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            BudgetPacer(target_avg_spend_usd=0.01, ema_alpha=0.0)

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="lr"):
            BudgetPacer(target_avg_spend_usd=0.01, lr=-0.1)

    def test_invalid_lambda_max_raises(self):
        with pytest.raises(ValueError, match="lambda_max"):
            BudgetPacer(target_avg_spend_usd=0.01, lambda_max=0.0)


# ======================================================================
# 2. Dual-variable dynamics
# ======================================================================


class TestDualVariableDynamics:
    """Verify lambda_t and cost_ema evolve correctly."""

    def test_lambda_increases_on_overspend(self):
        target = 0.01
        pacer = BudgetPacer(target_avg_spend_usd=target, lr=0.05)

        lambdas = []
        for _ in range(20):
            pacer.observe(target * 5.0)
            lambdas.append(pacer.lambda_t)

        assert pacer.lambda_t > 0.0, "lambda must increase after overspend"
        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1], (
                f"lambda must be monotonically non-decreasing under "
                f"sustained overspend (step {i})"
            )

    def test_lambda_stays_zero_on_underspend(self):
        target = 0.01
        pacer = BudgetPacer(target_avg_spend_usd=target, lr=0.05)

        for _ in range(50):
            pacer.observe(target * 0.5)
            assert pacer.lambda_t == 0.0, (
                "lambda must remain 0 when spending below target"
            )

    def test_lambda_recovers_after_burst(self):
        target = 0.01
        pacer = BudgetPacer(target_avg_spend_usd=target, lr=0.05)

        for _ in range(20):
            pacer.observe(target * 5.0)
        peak_lambda = pacer.lambda_t
        assert peak_lambda > 0.0

        for _ in range(200):
            pacer.observe(target * 0.1)
        assert pacer.lambda_t < peak_lambda, (
            "lambda must decrease after sustained underspend"
        )

    def test_lambda_returns_to_zero_after_long_underspend(self):
        target = 0.01
        pacer = BudgetPacer(target_avg_spend_usd=target, lr=0.05)

        for _ in range(10):
            pacer.observe(target * 3.0)
        assert pacer.lambda_t > 0.0

        for _ in range(500):
            pacer.observe(target * 0.01)
        assert pacer.lambda_t == 0.0, (
            "lambda must clamp back to 0 after enough underspend"
        )

    def test_ema_tracks_constant_cost(self):
        target = 0.01
        constant_cost = 0.005
        pacer = BudgetPacer(
            target_avg_spend_usd=target, ema_alpha=0.1
        )

        for _ in range(200):
            pacer.observe(constant_cost)

        assert abs(pacer.cost_ema - constant_cost) < 1e-6, (
            f"EMA should converge to {constant_cost}, got {pacer.cost_ema}"
        )

    def test_ema_warm_start_at_target(self):
        """cost_ema should start at target, not 0."""
        pacer = BudgetPacer(target_avg_spend_usd=0.025)
        assert pacer.cost_ema == 0.025

    def test_observation_count(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01)
        for _ in range(37):
            pacer.observe(0.005)
        assert pacer.n_observations == 37

    def test_normalized_dual_update_is_scale_invariant(self):
        """Same lr should produce equivalent lambda trajectories
        regardless of absolute cost scale, when costs are proportionally
        identical relative to their targets.
        """
        lr = 0.1

        target_a = 0.001
        pacer_a = BudgetPacer(target_avg_spend_usd=target_a, lr=lr)
        for _ in range(20):
            pacer_a.observe(target_a * 3.0)

        target_b = 10.0
        pacer_b = BudgetPacer(target_avg_spend_usd=target_b, lr=lr)
        for _ in range(20):
            pacer_b.observe(target_b * 3.0)

        assert abs(pacer_a.lambda_t - pacer_b.lambda_t) < 1e-10, (
            f"Scale invariance violated: lambda_a={pacer_a.lambda_t}, "
            f"lambda_b={pacer_b.lambda_t}"
        )

    def test_lambda_capped_at_lambda_max(self):
        """lambda_t should never exceed lambda_max."""
        target = 0.001
        pacer = BudgetPacer(
            target_avg_spend_usd=target, lr=1.0, lambda_max=3.0
        )

        for _ in range(1000):
            pacer.observe(target * 100.0)

        assert pacer.lambda_t == 3.0, (
            f"lambda_t={pacer.lambda_t} should be capped at lambda_max=3.0"
        )

    def test_lambda_max_custom_value(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, lambda_max=1.5
        )
        assert pacer.lambda_max == 1.5


# ======================================================================
# 3. Hard ceiling
# ======================================================================

_MAX_COST_PER_1K = 0.02  # Reference: portfolio's most expensive model


class TestHardCeiling:
    """Verify the cost ceiling mechanism."""

    def test_ceiling_none_when_underspending(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.HARD
        )
        for _ in range(20):
            pacer.observe(0.005)
        assert pacer.get_cost_ceiling_per_1k(_MAX_COST_PER_1K) is None, (
            "Ceiling should be None when not overspending"
        )

    def test_ceiling_none_at_init(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.HARD
        )
        assert pacer.get_cost_ceiling_per_1k(_MAX_COST_PER_1K) is None, (
            "Ceiling should be None at initialization (lambda=0)"
        )

    def test_ceiling_finite_on_overspend(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.HARD, lr=0.1
        )
        for _ in range(10):
            pacer.observe(0.05)

        ceiling = pacer.get_cost_ceiling_per_1k(_MAX_COST_PER_1K)
        assert ceiling is not None, "Ceiling should be set after overspend"
        assert ceiling > 0.0, "Ceiling should be positive"

    def test_ceiling_tightens_as_lambda_grows(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.HARD, lr=0.1
        )

        ceilings = []
        for _ in range(20):
            pacer.observe(0.05)
            c = pacer.get_cost_ceiling_per_1k(_MAX_COST_PER_1K)
            if c is not None:
                ceilings.append(c)

        assert len(ceilings) >= 2, "Should have multiple ceiling readings"
        for i in range(1, len(ceilings)):
            assert ceilings[i] <= ceilings[i - 1] + 1e-12, (
                f"Ceiling must tighten monotonically under sustained "
                f"overspend (step {i}: {ceilings[i]} > {ceilings[i-1]})"
            )

    def test_ceiling_formula(self):
        """Verify ceiling = max_cost / (1 + multiplier * lambda_t)
        with target-normalized dual update."""
        target = 0.01
        cost = 0.05
        max_cost_1k = 0.03
        pacer = BudgetPacer(
            target_avg_spend_usd=target,
            mode=PacingMode.HARD,
            lr=0.1,
            hard_ceiling_multiplier=2.0,
        )
        pacer.observe(cost)

        # Dual update uses the EMA (not raw cost).
        # After 1 observation: ema = (1-0.05)*target + 0.05*cost
        expected_ema = (1 - 0.05) * target + 0.05 * cost
        expected_lambda = 0.1 * (expected_ema / target - 1.0)
        expected_ceiling = max_cost_1k / (1.0 + 2.0 * expected_lambda)
        actual_ceiling = pacer.get_cost_ceiling_per_1k(max_cost_1k)

        assert actual_ceiling is not None
        assert abs(actual_ceiling - expected_ceiling) < 1e-10, (
            f"Expected ceiling {expected_ceiling}, got {actual_ceiling}"
        )

    def test_ceiling_bounded_by_lambda_max(self):
        """Even under extreme overspend, the ceiling should not go
        below max_cost / (1 + multiplier * lambda_max)."""
        target = 0.001
        max_cost_1k = 0.05
        pacer = BudgetPacer(
            target_avg_spend_usd=target,
            mode=PacingMode.HARD,
            lr=1.0,
            hard_ceiling_multiplier=1.0,
            lambda_max=4.0,
        )
        for _ in range(1000):
            pacer.observe(target * 1000)

        ceiling = pacer.get_cost_ceiling_per_1k(max_cost_1k)
        expected_floor = max_cost_1k / (1.0 + 1.0 * 4.0)  # 0.01
        assert ceiling is not None
        assert abs(ceiling - expected_floor) < 1e-10, (
            f"Ceiling {ceiling} should equal floor {expected_floor} "
            f"when lambda is at cap"
        )


# ======================================================================
# 4. Soft penalties
# ======================================================================


class TestSoftPenalties:
    """Verify per-model penalty scaling."""

    def test_penalties_zero_when_lambda_zero(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.SOFT
        )
        model_costs = {"cheap": 0.1, "expensive": 0.8}
        penalties = pacer.get_extra_cost_penalties(model_costs)
        assert penalties == {"cheap": 0.0, "expensive": 0.0}

    def test_penalties_proportional_to_lambda(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.SOFT, lr=0.1
        )
        model_costs = {"cheap": 0.1, "expensive": 0.8}

        pacer.observe(0.05)

        penalties = pacer.get_extra_cost_penalties(model_costs)
        assert penalties["expensive"] > penalties["cheap"], (
            "More expensive model should have higher penalty"
        )
        ratio = penalties["expensive"] / penalties["cheap"]
        expected_ratio = 0.8 / 0.1
        assert abs(ratio - expected_ratio) < 1e-10, (
            f"Penalty ratio should equal cost ratio ({expected_ratio}), "
            f"got {ratio}"
        )

    def test_penalties_scale_with_lambda(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01, mode=PacingMode.SOFT, lr=0.1
        )
        model_costs = {"model_a": 0.5}

        pacer.observe(0.05)
        pen_1 = pacer.get_extra_cost_penalties(model_costs)["model_a"]

        pacer.observe(0.05)
        pen_2 = pacer.get_extra_cost_penalties(model_costs)["model_a"]

        assert pen_2 > pen_1, (
            "Penalty should increase as lambda grows from continued overspend"
        )

    def test_penalties_empty_dict(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.SOFT)
        pacer.observe(0.05)
        assert pacer.get_extra_cost_penalties({}) == {}


# ======================================================================
# 5. Reset
# ======================================================================


class TestReset:
    """Verify state re-initialization."""

    def test_reset_clears_state(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, lr=0.1)

        for _ in range(50):
            pacer.observe(0.05)

        assert pacer.lambda_t > 0.0
        assert pacer.n_observations == 50

        pacer.reset()

        assert pacer.lambda_t == 0.0
        assert pacer.cost_ema == 0.01
        assert pacer.n_observations == 0

    def test_reset_preserves_config(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.02,
            mode=PacingMode.HARD,
            lr=0.1,
            ema_alpha=0.1,
            hard_ceiling_multiplier=3.0,
            lambda_max=8.0,
        )
        for _ in range(10):
            pacer.observe(0.05)

        pacer.reset()

        assert pacer.target_avg_spend_usd == 0.02
        assert pacer.mode is PacingMode.HARD
        assert pacer.lr == 0.1
        assert pacer.ema_alpha == 0.1
        assert pacer.hard_ceiling_multiplier == 3.0
        assert pacer.lambda_max == 8.0

    def test_reset_allows_fresh_start(self):
        """After reset, pacer should behave identically to a fresh instance."""
        pacer = BudgetPacer(target_avg_spend_usd=0.01, lr=0.05)
        for _ in range(30):
            pacer.observe(0.05)
        pacer.reset()

        fresh = BudgetPacer(target_avg_spend_usd=0.01, lr=0.05)
        assert pacer.lambda_t == fresh.lambda_t
        assert pacer.cost_ema == fresh.cost_ema
        assert pacer.n_observations == fresh.n_observations
        max_cost = 0.02
        assert (
            pacer.get_cost_ceiling_per_1k(max_cost)
            == fresh.get_cost_ceiling_per_1k(max_cost)
        )


# ======================================================================
# 6. Thread safety
# ======================================================================


class TestThreadSafety:
    """Concurrent access must not corrupt state."""

    def test_concurrent_observe_and_read(self):
        pacer = BudgetPacer(
            target_avg_spend_usd=0.01,
            mode=PacingMode.ADAPTIVE,
            lr=0.01,
        )
        errors: list[str] = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                cost = 0.005 + 0.01 * (i % 5)
                pacer.observe(cost)
                i += 1

        def reader():
            while not stop.is_set():
                ceiling = pacer.get_cost_ceiling_per_1k(0.02)
                if ceiling is not None and (ceiling <= 0 or ceiling != ceiling):
                    errors.append(f"Invalid ceiling: {ceiling}")
                penalties = pacer.get_extra_cost_penalties({"m": 0.5})
                pen = penalties["m"]
                if pen != pen:  # NaN check
                    errors.append(f"NaN penalty: {pen}")
                lam = pacer.lambda_t
                if lam != lam or lam < 0:
                    errors.append(f"Invalid lambda: {lam}")

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer, daemon=True))
        for _ in range(5):
            threads.append(threading.Thread(target=reader, daemon=True))

        for t in threads:
            t.start()

        stop.wait(timeout=2.0)
        stop.set()

        for t in threads:
            t.join(timeout=3.0)

        assert not errors, f"Thread safety violations: {errors}"
        assert pacer.lambda_t >= 0.0
        assert pacer.cost_ema == pacer.cost_ema  # not NaN


# ======================================================================
# 7. Diagnostics
# ======================================================================


class TestDiagnostics:
    """Verify get_state() and __repr__()."""

    def test_get_state_keys(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01)
        state = pacer.get_state()
        expected_keys = {
            "lambda_t", "cost_ema", "n_observations",
            "target_avg_spend_usd", "mode", "lr", "ema_alpha",
            "lambda_max", "uses_hard", "uses_soft",
        }
        assert set(state.keys()) == expected_keys

    def test_repr_contains_key_info(self):
        pacer = BudgetPacer(target_avg_spend_usd=0.01, mode=PacingMode.SOFT)
        r = repr(pacer)
        assert "0.01" in r
        assert "soft" in r
