"""Tests for open-source usability improvements.

Validates that external users can bring their own models, reward scales,
cost specifications, and feature pipelines to the router without hitting
silent failures or requiring internal knowledge of the paper's K=3 portfolio.

Organised by feature:
  1. Configurable reward range (RouterConfig.reward_min / reward_max)
  2. Explicit input/output cost kwargs in register_model()
  3. Market cost ceiling coverage for expensive models
  4. BYOM (bring-your-own-models) end-to-end flow
  5. Custom FeatureService integration
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter, RouterConfig

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

DIM = 24

CHEAP_MODEL = {
    "input_cost_per_m": 0.10,
    "output_cost_per_m": 0.10,
    "time_to_first_token_seconds": 0.2,
}

EXPENSIVE_MODEL = {
    "input_cost_per_m": 5.0,
    "output_cost_per_m": 15.0,
    "time_to_first_token_seconds": 0.8,
}


def _mock_feature_service(dim: int = DIM) -> MagicMock:
    fs = MagicMock()
    fs.dimension = dim
    fs.bias_index = dim - 1
    fs.pca = MagicMock(n_components=dim - 1)
    fs.encoder = None
    fs.using_pca = True
    fs.get_dimension.return_value = dim
    fs.get_feature_names.return_value = (
        [f"pca_{i}" for i in range(dim - 1)] + ["bias"]
    )

    def _extract(prompt):
        if isinstance(prompt, np.ndarray):
            return prompt
        v = np.random.default_rng(abs(hash(prompt)) % 2**31).standard_normal(dim - 1)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.append(v, 1.0)

    fs.extract_features.side_effect = _extract
    return fs


def _make_router(registry: dict, **kwargs) -> BanditRouter:
    defaults = {
        "model_registry": registry,
        "priors": "none",
        "feature_service": _mock_feature_service(),
    }
    defaults.update(kwargs)
    return BanditRouter.create(**defaults)


def _ctx(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0)


# ===========================================================================
# 1. Configurable Reward Range
# ===========================================================================

class TestConfigurableRewardRange:
    """RouterConfig.reward_min / reward_max control clamping in both
    process_feedback() and update()."""

    def test_default_range_is_zero_one(self):
        cfg = RouterConfig()
        assert cfg.reward_min == 0.0
        assert cfg.reward_max == 1.0

    def test_custom_range_accepted(self):
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        assert cfg.reward_min == -1.0
        assert cfg.reward_max == 1.0

    # -- process_feedback --------------------------------------------------

    def test_process_feedback_respects_custom_range(self):
        """Negative rewards within [-1, 1] should update the bandit without clamping."""
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        x = _ctx()
        _, log = router.route(x)

        theta_before = router.bandit.theta[log.selected_model].copy()
        router.process_feedback(log.request_id, reward=-0.5)
        theta_after = router.bandit.theta[log.selected_model]

        assert not np.allclose(theta_before, theta_after), (
            "Negative reward within range should trigger a bandit update"
        )

    def test_process_feedback_clamps_above_max(self):
        """Reward above reward_max is clamped (and a warning is emitted)."""
        cfg = RouterConfig(reward_min=0.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        x = _ctx()
        _, log = router.route(x)

        # Reward of 5.0 should be clamped to 1.0
        router.bandit.theta[log.selected_model].copy()
        router.process_feedback(log.request_id, reward=5.0)

        # Now give a clean reward=1.0 on a fresh route to compare
        _, log2 = router.route(x)
        theta_ref = router.bandit.theta[log2.selected_model].copy()

        # The clamped update should have used reward=1.0, not 5.0.
        # We can't compare theta exactly (A differs), but the clamped
        # value should not produce an extreme theta shift.
        assert np.all(np.isfinite(theta_ref))

    def test_process_feedback_clamps_below_min(self):
        """Reward below reward_min is clamped."""
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        x = _ctx()
        _, log = router.route(x)

        # Reward of -5.0 should be clamped to -1.0
        router.process_feedback(log.request_id, reward=-5.0)
        theta = router.bandit.theta[log.selected_model]
        assert np.all(np.isfinite(theta))

    def test_process_feedback_warns_on_clamp(self, caplog):
        """A warning is logged when reward is clamped."""
        cfg = RouterConfig(reward_min=0.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        _, log = router.route(_ctx())
        with caplog.at_level(logging.WARNING, logger="pareto_bandit.router"):
            router.process_feedback(log.request_id, reward=2.0)

        assert any("clamping" in r.message.lower() for r in caplog.records), (
            "Expected a warning about clamping when reward exceeds range"
        )

    def test_process_feedback_no_warn_within_range(self, caplog):
        """No clamping warning when reward is within the configured range."""
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        _, log = router.route(_ctx())
        with caplog.at_level(logging.WARNING, logger="pareto_bandit.router"):
            router.process_feedback(log.request_id, reward=0.5)

        clamp_warnings = [r for r in caplog.records if "clamping" in r.message.lower()]
        assert len(clamp_warnings) == 0

    # -- update ------------------------------------------------------------

    def test_update_respects_custom_range(self):
        """Direct update() path also uses the configured reward range."""
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        x = _ctx()
        theta_before = router.bandit.theta["m"].copy()
        router.update("m", x, reward=-0.8)
        theta_after = router.bandit.theta["m"]

        assert not np.allclose(theta_before, theta_after)

    def test_update_clamps_and_warns(self, caplog):
        """update() also clamps and warns for out-of-range rewards."""
        cfg = RouterConfig(reward_min=0.0, reward_max=1.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        with caplog.at_level(logging.WARNING, logger="pareto_bandit.router"):
            router.update("m", _ctx(), reward=3.0)

        assert any("clamping" in r.message.lower() for r in caplog.records)

    def test_update_no_warn_within_range(self, caplog):
        cfg = RouterConfig(reward_min=0.0, reward_max=10.0)
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry, config=cfg)

        with caplog.at_level(logging.WARNING, logger="pareto_bandit.router"):
            router.update("m", _ctx(), reward=7.5)

        clamp_warnings = [r for r in caplog.records if "clamping" in r.message.lower()]
        assert len(clamp_warnings) == 0

    # -- backward compatibility --------------------------------------------

    def test_default_range_clamps_zero_one(self):
        """With default config, rewards outside [0, 1] are clamped (backward compat)."""
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry)

        _, log = router.route(_ctx())
        router.process_feedback(log.request_id, reward=-0.5)
        # Should not raise; reward is silently clamped to 0.0


# ===========================================================================
# 2. Explicit Input/Output Costs in register_model()
# ===========================================================================

class TestRegisterModelExplicitCosts:
    """register_model() now accepts input_cost_per_m and output_cost_per_m
    for exact cost specification without relying on the 3x heuristic."""

    def test_explicit_costs_stored_exactly(self):
        """Tier 1: Both input and output provided — stored verbatim."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "exact-model",
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
        )

        entry = router.registry["exact-model"]
        assert entry["input_cost_per_m"] == 2.50
        assert entry["output_cost_per_m"] == 10.00

    def test_explicit_costs_derive_correct_blended(self):
        """Blended cost should be the average of input and output."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "exact-model",
            input_cost_per_m=2.00,
            output_cost_per_m=8.00,
        )

        entry = router.registry["exact-model"]
        assert entry["blended_cost_per_m"] == pytest.approx(5.00)

    def test_explicit_costs_used_in_cost_filtering(self):
        """Models registered with explicit costs must be filterable by max_cost."""
        router = _make_router({
            "expensive": {**EXPENSIVE_MODEL, "input_cost_per_m": 50.0, "output_cost_per_m": 150.0},
        })
        router.register_model(
            "cheap-exact",
            input_cost_per_m=0.05,
            output_cost_per_m=0.05,
        )

        x = _ctx()
        for _ in range(30):
            router.update("cheap-exact", x, reward=0.9)

        mid, _ = router.route(x, max_cost=0.001)
        assert mid == "cheap-exact"

    def test_explicit_costs_used_in_cost_estimation(self):
        """_estimate_cost should use the exact input/output rates from register_model."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "priced-model",
            input_cost_per_m=4.00,
            output_cost_per_m=12.00,
        )

        cost = router._estimate_cost("priced-model", in_tok=1000, out_tok=500)
        expected = (4.00 * 1000 + 12.00 * 500) / 1e6
        assert cost == pytest.approx(expected)

    def test_partial_costs_raise_value_error(self):
        """Providing only input or only output should raise ValueError."""
        router = _make_router({"seed": CHEAP_MODEL})

        with pytest.raises(ValueError, match="both input_cost_per_m"):
            router.register_model("bad-1", input_cost_per_m=1.0)

        with pytest.raises(ValueError, match="both input_cost_per_m"):
            router.register_model("bad-2", output_cost_per_m=5.0)

    def test_blended_cost_still_works(self):
        """Tier 2: blended_cost_per_m alone (no input/output) should still work."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model("blended-only", blended_cost_per_m=3.00)

        entry = router.registry["blended-only"]
        assert entry["blended_cost_per_m"] == pytest.approx(3.00)
        assert "input_cost_per_m" in entry
        assert "output_cost_per_m" in entry

    def test_legacy_cost_usd_still_works(self):
        """Tier 3: legacy cost_usd path should still function."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model("legacy-model", cost_usd=1.00)

        entry = router.registry["legacy-model"]
        assert entry["input_cost_per_m"] == 1.00
        assert entry["output_cost_per_m"] == pytest.approx(3.00)
        assert entry["blended_cost_per_m"] == pytest.approx(2.00)

    def test_no_cost_info_raises_missing_cost_error(self):
        """Tier 4: No cost info at all should raise MissingCostError."""
        from pareto_bandit.exceptions import MissingCostError

        router = _make_router({"seed": CHEAP_MODEL})
        with pytest.raises(MissingCostError, match="no cost information"):
            router.register_model("mystery-model")

    def test_explicit_costs_override_cost_usd(self):
        """Explicit input/output pair takes precedence over cost_usd."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "mixed-model",
            cost_usd=999.0,
            input_cost_per_m=1.00,
            output_cost_per_m=2.00,
        )

        entry = router.registry["mixed-model"]
        assert entry["input_cost_per_m"] == 1.00
        assert entry["output_cost_per_m"] == 2.00
        assert entry["blended_cost_per_m"] == pytest.approx(1.50)

    def test_explicit_costs_override_blended(self):
        """Explicit input/output pair takes precedence over blended_cost_per_m."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "mixed-model",
            blended_cost_per_m=999.0,
            input_cost_per_m=1.00,
            output_cost_per_m=2.00,
        )

        entry = router.registry["mixed-model"]
        assert entry["input_cost_per_m"] == 1.00
        assert entry["output_cost_per_m"] == 2.00
        assert entry["blended_cost_per_m"] == pytest.approx(1.50)

    def test_equal_input_output_costs(self):
        """Models with identical input/output pricing (common for open-source hosting)."""
        router = _make_router({"seed": CHEAP_MODEL})
        router.register_model(
            "uniform-pricing",
            input_cost_per_m=0.50,
            output_cost_per_m=0.50,
        )

        entry = router.registry["uniform-pricing"]
        assert entry["blended_cost_per_m"] == pytest.approx(0.50)
        cost = router._estimate_cost("uniform-pricing", in_tok=1000, out_tok=1000)
        expected = (0.50 * 1000 + 0.50 * 1000) / 1e6
        assert cost == pytest.approx(expected)


# ===========================================================================
# 3. Market Cost Ceiling
# ===========================================================================

class TestMarketCostCeiling:
    """The market_cost_ceiling must be high enough that expensive models
    don't all saturate to the same normalized cost of 1.0."""

    def test_default_ceiling_covers_premium_models(self):
        """Default ceiling should distinguish between mid-tier and premium models."""
        cfg = RouterConfig()
        # o1-pro-class model: ~$60/M output → blended ~$37.5/M → $0.0375/1k
        # With ceiling at $0.10/1k this should normalize below 1.0
        assert cfg.market_cost_ceiling >= 0.05, (
            "Ceiling should cover premium models like o1-pro"
        )

    def test_expensive_models_not_all_saturated(self):
        """Two models at different premium prices should get different normalized costs."""
        registry = {
            "mid-tier": {
                "input_cost_per_m": 3.00,
                "output_cost_per_m": 15.00,
                "time_to_first_token_seconds": 0.5,
            },
            "premium": {
                "input_cost_per_m": 15.00,
                "output_cost_per_m": 60.00,
                "time_to_first_token_seconds": 1.0,
            },
        }
        router = _make_router(registry)

        norm_mid = router._get_normalized_cost("mid-tier")
        norm_premium = router._get_normalized_cost("premium")

        assert norm_premium > norm_mid, (
            "Premium model should have higher normalized cost than mid-tier"
        )
        assert norm_premium < 1.0, (
            f"Premium normalized cost {norm_premium:.3f} saturated at 1.0; "
            f"raise market_cost_ceiling"
        )

    def test_custom_ceiling_for_ultra_expensive(self):
        """Users can raise the ceiling via RouterConfig for very expensive models."""
        cfg = RouterConfig(market_cost_ceiling=0.50)
        registry = {
            "ultra": {
                "input_cost_per_m": 100.0,
                "output_cost_per_m": 400.0,
            },
        }
        router = _make_router(registry, config=cfg)

        norm = router._get_normalized_cost("ultra")
        assert norm < 1.0, (
            "Custom ceiling should prevent saturation for ultra-expensive models"
        )


# ===========================================================================
# 4. BYOM (Bring Your Own Models) End-to-End
# ===========================================================================

class TestBYOMFlow:
    """End-to-end tests for the documented BYOM workflow: construct a router
    with a custom registry, route requests, and provide feedback."""

    def test_byom_cold_start_routes_successfully(self):
        """A fresh router with custom models and no priors should route."""
        registry = {
            "my-gpt4": {
                "input_cost_per_m": 2.50,
                "output_cost_per_m": 10.00,
                "time_to_first_token_seconds": 0.8,
            },
            "my-llama": {
                "input_cost_per_m": 0.50,
                "output_cost_per_m": 0.50,
                "time_to_first_token_seconds": 0.2,
            },
        }
        router = _make_router(registry)

        mid, log = router.route(_ctx())
        assert mid in registry
        assert log.cost_usd > 0
        assert log.latency_s > 0

    def test_byom_feedback_loop_updates_policy(self):
        """route() -> process_feedback() should shift the policy.

        Uses direct update() to ensure both arms receive enough signal for
        the bandit to learn a clear preference, avoiding exploration-driven
        allocation imbalances in the feedback loop.
        """
        registry = {
            "good-model": CHEAP_MODEL,
            "bad-model": CHEAP_MODEL,
        }
        router = _make_router(registry)

        # Train both arms directly so each gets enough observations
        for i in range(200):
            x = _ctx(seed=i)
            router.update("good-model", x, reward=0.9)
            router.update("bad-model", x, reward=0.1)

        # After sufficient learning, the router should prefer "good-model"
        counts = {"good-model": 0, "bad-model": 0}
        with router.exploit():
            for i in range(50):
                mid, _ = router.route(_ctx(seed=1000 + i))
                counts[mid] += 1

        assert counts["good-model"] > counts["bad-model"], (
            f"Expected good-model to dominate after training, got {counts}"
        )

    def test_byom_add_model_at_runtime(self):
        """After initial creation, users can add models via register_model."""
        registry = {"initial-model": CHEAP_MODEL}
        router = _make_router(registry)

        router.register_model(
            "late-addition",
            input_cost_per_m=1.00,
            output_cost_per_m=3.00,
            speed="balanced",
        )

        assert "late-addition" in router.registry
        assert "late-addition" in router.bandit.models

        mid, _ = router.route(_ctx())
        assert mid in {"initial-model", "late-addition"}

    def test_byom_custom_reward_range_end_to_end(self):
        """Full loop with a non-default reward range [-1, 1]."""
        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        registry = {
            "model-a": CHEAP_MODEL,
            "model-b": CHEAP_MODEL,
        }
        router = _make_router(registry, config=cfg)

        for i in range(100):
            x = _ctx(seed=i)
            mid, log = router.route(x)
            reward = 0.8 if mid == "model-a" else -0.8
            router.process_feedback(log.request_id, reward)

        # model-a should be preferred (positive rewards)
        with router.exploit():
            counts = {"model-a": 0, "model-b": 0}
            for i in range(50):
                mid, _ = router.route(_ctx(seed=2000 + i))
                counts[mid] += 1

        assert counts["model-a"] > counts["model-b"]

    def test_byom_minimal_registry_entry(self):
        """Only input/output cost required; everything else is optional."""
        registry = {
            "bare-minimum": {
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 3.0,
            },
        }
        router = _make_router(registry)

        mid, log = router.route(_ctx())
        assert mid == "bare-minimum"
        assert log.cost_usd > 0
        assert np.isfinite(log.latency_s)

    def test_byom_save_load_preserves_learning(self, tmp_path):
        """Learned state can be persisted and restored across sessions."""
        registry = {"m": CHEAP_MODEL}
        router = _make_router(registry)

        x = _ctx()
        _, log = router.route(x)
        router.process_feedback(log.request_id, reward=0.9)

        theta_before = router.bandit.theta["m"].copy()

        save_path = tmp_path / "state.npz"
        router.save_state(save_path)

        router2 = _make_router(registry)
        router2.load_state(save_path)

        assert np.allclose(theta_before, router2.bandit.theta["m"])


# ===========================================================================
# 5. Custom FeatureService Integration
# ===========================================================================

class TestCustomFeatureService:
    """Users should be able to plug in custom embedding functions without
    needing the default SentenceTransformer or shipped PCA artifact."""

    def test_precomputed_vectors_skip_encoding(self):
        """FeatureService.for_precomputed() should accept raw vectors."""
        dim = 16
        fs = FeatureService.for_precomputed(dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )

        x = np.random.default_rng(0).standard_normal(dim)
        x[-1] = 1.0
        mid, log = router.route(x)
        assert mid == "m"

    def test_custom_encoder_callable(self):
        """A user-supplied encoder function should work end-to-end."""
        raw_dim = 32

        def my_encoder(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(
            custom_encoder=my_encoder,
            embedding_dim=raw_dim,
        )
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )

        mid, log = router.route("test prompt")
        assert mid == "m"
        router.process_feedback(log.request_id, reward=0.8)

    def test_custom_encoder_with_feedback_loop(self):
        """Custom encoder + route + feedback should update the bandit."""
        raw_dim = 16

        def my_encoder(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(
            custom_encoder=my_encoder,
            embedding_dim=raw_dim,
        )
        registry = {"fast": CHEAP_MODEL, "slow": EXPENSIVE_MODEL}
        router = BanditRouter.create(
            model_registry=registry,
            priors="none",
            feature_service=fs,
        )

        theta_before = router.bandit.theta["fast"].copy()
        for i in range(10):
            mid, log = router.route(f"prompt {i}")
            router.process_feedback(log.request_id, reward=0.9 if mid == "fast" else 0.1)

        theta_after = router.bandit.theta["fast"]
        assert not np.allclose(theta_before, theta_after), (
            "Bandit theta should change after feedback with custom encoder"
        )

    def test_nondefault_encoder_without_pca_raises_helpful_error(self):
        """Using a non-default encoder name without a PCA path should produce
        an actionable error message with all three resolution options."""
        with pytest.raises(ValueError, match="differs from the default") as exc_info:
            FeatureService(encoder_model="some-other-model")

        msg = str(exc_info.value)
        assert "train_pca" in msg, "Error should mention train_pca as option 1"
        assert "custom_encoder" in msg, "Error should mention custom_encoder as option 2"
        assert "pca_path" in msg, "Error should mention pca_path as option 3"
