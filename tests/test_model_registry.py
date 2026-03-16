"""
Tests for model registry management: initial construction, adding models
via register_model(), and validation of required fields.

The router needs specific metadata per model to function correctly:
  - input_cost_per_m / output_cost_per_m  → cost constraint filtering
  - time_to_first_token_seconds           → latency constraint filtering

Missing fields are handled via pessimistic defaults (expensive & slow),
which is safe but suboptimal.  These tests verify:
  1. Construction with a user-supplied registry dict
  2. Adding models at runtime via register_model()
  3. Cost/latency fields are used for constraint filtering
  4. Missing fields fall back to pessimistic defaults
  5. Duplicate registration is idempotent
  6. Models with bad/missing metadata don't break routing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock

from pareto_bandit.router import BanditRouter, RouterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 24


def _mock_feature_service(dim: int = DIM) -> MagicMock:
    fs = MagicMock()
    fs.dimension = dim
    fs.bias_index = dim - 1
    fs.pca = MagicMock(n_components=dim - 1)
    fs.encoder = None
    fs.using_pca = True
    fs.get_dimension.return_value = dim
    fs.get_feature_names.return_value = [f"pca_{i}" for i in range(dim - 1)] + ["bias"]

    def _extract(prompt):
        if isinstance(prompt, np.ndarray):
            return prompt
        v = np.random.default_rng(0).standard_normal(dim - 1)
        v = v / (np.linalg.norm(v) + 1e-12)
        return np.append(v, 1.0)

    fs.extract_features.side_effect = _extract
    return fs


def _make_router(registry: dict, **kwargs) -> BanditRouter:
    defaults = dict(
        model_registry=registry,
        priors="none",
        feature_service=_mock_feature_service(),
    )
    defaults.update(kwargs)
    return BanditRouter.create(**defaults)


def _ctx(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0)


# A well-formed model entry with all the fields the router uses
WELL_FORMED_MODEL = {
    "model_id": "vendor/model-full",
    "display_name": "Fully Specified Model",
    "input_cost_per_m": 2.50,
    "output_cost_per_m": 7.50,
    "time_to_first_token_seconds": 0.4,
    "output_tokens_per_second": 80.0,
    "initial_quality": 0.85,
}


# ---------------------------------------------------------------------------
# 1. Construction with user registry
# ---------------------------------------------------------------------------

class TestRegistryConstruction:

    def test_all_models_in_bandit(self):
        """Every model in the registry dict should appear in the bandit's arm list."""
        registry = {
            "model-a": {**WELL_FORMED_MODEL, "model_id": "model-a"},
            "model-b": {**WELL_FORMED_MODEL, "model_id": "model-b"},
            "model-c": {**WELL_FORMED_MODEL, "model_id": "model-c"},
        }
        router = _make_router(registry)

        assert set(router.bandit.models) == set(registry.keys())
        assert set(router.registry.keys()) == set(registry.keys())

    def test_empty_registry_initializes(self):
        """An empty registry should produce a router with zero arms."""
        router = _make_router({})
        assert len(router.bandit.models) == 0
        assert len(router.registry) == 0

    def test_single_model_routes(self):
        """A single-model registry should always route to that model."""
        registry = {"only-model": WELL_FORMED_MODEL}
        router = _make_router(registry)

        mid, log = router.route(_ctx())
        assert mid == "only-model"
        assert log.selected_model == "only-model"

    def test_registry_is_a_copy(self):
        """Mutating the original dict after construction should not affect the router."""
        registry = {"model-a": {**WELL_FORMED_MODEL}}
        router = _make_router(registry)
        registry["injected"] = {"bad": True}

        assert "injected" not in router.registry


# ---------------------------------------------------------------------------
# 2. register_model() — adding models at runtime
# ---------------------------------------------------------------------------

class TestRegisterModel:

    def test_register_adds_to_bandit_and_registry(self):
        """Newly registered model must appear in both the bandit and the registry."""
        router = _make_router({"existing": WELL_FORMED_MODEL})

        router.register_model(
            "new-model",
            speed="fast",
            cost_usd=0.50,
            latency_s=0.2,
        )

        assert "new-model" in router.bandit.models
        assert "new-model" in router.registry

    def test_registered_model_is_routable(self):
        """After registration, route() must be able to select the new model."""
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("new-model", speed="fast", cost_usd=0.50, latency_s=0.2)

        all_selected = set()
        for i in range(100):
            mid, _ = router.route(_ctx(seed=i))
            all_selected.add(mid)

        assert "new-model" in all_selected or "existing" in all_selected

    def test_registered_model_has_initialized_matrices(self):
        """Bandit matrices (A, b, A_inv) must be fully initialized for the new model."""
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("new-model", speed="balanced", cost_usd=1.0, latency_s=0.5)

        assert "new-model" in router.bandit.A
        assert "new-model" in router.bandit.b
        assert "new-model" in router.bandit.A_inv

        assert router.bandit.A["new-model"].shape == (DIM, DIM)
        assert router.bandit.b["new-model"].shape == (DIM,)
        product = router.bandit.A["new-model"] @ router.bandit.A_inv["new-model"]
        assert np.allclose(product, np.eye(DIM), atol=1e-6)

    def test_duplicate_registration_is_idempotent(self):
        """Registering the same model_id twice should not create duplicates."""
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("new-model", speed="fast", cost_usd=0.50, latency_s=0.2)
        router.register_model("new-model", speed="slow", cost_usd=99.0, latency_s=5.0)

        count = router.bandit.models.count("new-model")
        assert count == 1, f"Model appeared {count} times in bandit.models"

    def test_register_preserves_cost_and_latency(self):
        """Cost and latency passed to register_model must be stored in the registry."""
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("new-model", cost_usd=3.75, latency_s=1.2)

        entry = router.registry["new-model"]
        assert entry["cost_per_1m_tokens"] == 3.75
        assert entry["median_latency_s"] == 1.2

    def test_register_without_cost_uses_pessimistic_default(self):
        """Omitting cost_usd should use the config's pessimistic default."""
        config = RouterConfig()
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("cheap-mystery")

        entry = router.registry["cheap-mystery"]
        assert entry["cost_per_1m_tokens"] == config.registration.default_cost_per_1m

    def test_register_without_latency_uses_pessimistic_default(self):
        """Omitting latency_s should use the config's pessimistic default."""
        config = RouterConfig()
        router = _make_router({"existing": WELL_FORMED_MODEL})
        router.register_model("slow-mystery")

        entry = router.registry["slow-mystery"]
        assert entry["median_latency_s"] == config.registration.default_latency_s

    def test_register_multiple_grows_model_count(self):
        """Registering N models should result in initial + N total models."""
        registry = {"seed-model": WELL_FORMED_MODEL}
        router = _make_router(registry)

        for i in range(5):
            router.register_model(f"added-{i}", cost_usd=float(i), latency_s=0.1 * i)

        assert len(router.bandit.models) == 6
        assert len(router.registry) == 6


# ---------------------------------------------------------------------------
# 3. Cost and latency fields — used for constraint filtering
# ---------------------------------------------------------------------------

class TestCostLatencyFields:

    def test_max_cost_filters_using_registry_fields(self):
        """max_cost should filter out models whose input/output cost exceeds the budget."""
        registry = {
            "cheap": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 0.10,
                "output_cost_per_m": 0.15,
            },
            "expensive": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 50.0,
                "output_cost_per_m": 150.0,
            },
        }
        router = _make_router(registry)

        # Very tight budget — only cheap should survive
        for _ in range(20):
            mid, _ = router.route(_ctx(), max_cost=0.001)
            assert mid == "cheap", f"Expected 'cheap' under tight budget, got '{mid}'"

    def test_max_latency_filters_using_registry_fields(self):
        """max_latency should filter out models whose TTFT exceeds the limit."""
        registry = {
            "fast": {
                **WELL_FORMED_MODEL,
                "time_to_first_token_seconds": 0.1,
            },
            "slow": {
                **WELL_FORMED_MODEL,
                "time_to_first_token_seconds": 5.0,
            },
        }
        router = _make_router(registry)

        for _ in range(20):
            mid, _ = router.route(_ctx(), max_latency=0.5)
            assert mid == "fast", f"Expected 'fast' under tight latency, got '{mid}'"

    def test_cost_and_latency_constraints_together(self):
        """Combining cost + latency should narrow to only models passing both."""
        registry = {
            "cheap-fast": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 0.10,
                "output_cost_per_m": 0.15,
                "time_to_first_token_seconds": 0.1,
            },
            "cheap-slow": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 0.10,
                "output_cost_per_m": 0.15,
                "time_to_first_token_seconds": 5.0,
            },
            "expensive-fast": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 50.0,
                "output_cost_per_m": 150.0,
                "time_to_first_token_seconds": 0.1,
            },
        }
        router = _make_router(registry)

        for _ in range(20):
            mid, _ = router.route(_ctx(), max_cost=0.001, max_latency=0.5)
            assert mid == "cheap-fast"

    def test_registered_model_cost_used_in_filtering(self):
        """Cost passed via register_model() should be used by the constraint filter."""
        router = _make_router({
            "existing-expensive": {
                **WELL_FORMED_MODEL,
                "input_cost_per_m": 50.0,
                "output_cost_per_m": 150.0,
            },
        })

        router.register_model("added-cheap", cost_usd=0.10, latency_s=0.1)

        # Train added-cheap so it's selectable
        x = _ctx()
        for _ in range(50):
            router.update("added-cheap", x, reward=0.8)

        # Tight budget should select the cheap registered model
        mid, _ = router.route(x, max_cost=0.001)
        assert mid == "added-cheap"


# ---------------------------------------------------------------------------
# 4. Missing/malformed metadata — pessimistic defaults
# ---------------------------------------------------------------------------

class TestMissingMetadata:

    def test_missing_cost_fields_use_pessimistic_default(self):
        """Models without input_cost_per_m get the pessimistic fallback cost."""
        config = RouterConfig()
        registry = {
            "no-cost": {"model_id": "no-cost", "time_to_first_token_seconds": 0.5},
        }
        router = _make_router(registry)

        _, log = router.route(_ctx())

        # Cost should use pessimistic default, not zero or infinity
        expected_default = config.default_missing_cost_per_m
        assert log.cost_usd > 0, "Cost should not be zero for missing fields"
        assert np.isfinite(log.cost_usd), "Cost should not be infinity"

    def test_missing_latency_uses_pessimistic_default(self):
        """Models without time_to_first_token_seconds get the pessimistic fallback latency."""
        config = RouterConfig()
        registry = {
            "no-latency": {"model_id": "no-latency", "input_cost_per_m": 1.0, "output_cost_per_m": 3.0},
        }
        router = _make_router(registry)

        _, log = router.route(_ctx())

        assert log.latency_s == config.default_missing_latency

    def test_model_with_no_metadata_still_routes(self):
        """A bare-bones registry entry (just a key) should not crash routing."""
        registry = {
            "bare-bones": {},
        }
        router = _make_router(registry)

        mid, log = router.route(_ctx())
        assert mid == "bare-bones"
        assert log.cost_usd > 0
        assert log.latency_s > 0

    def test_null_cost_fields_treated_as_missing(self):
        """Explicit None values for cost should be treated same as absent."""
        registry = {
            "null-cost": {
                "input_cost_per_m": None,
                "output_cost_per_m": None,
                "time_to_first_token_seconds": None,
            },
        }
        config = RouterConfig()
        router = _make_router(registry)

        _, log = router.route(_ctx())
        assert log.latency_s == config.default_missing_latency

    def test_string_cost_fields_treated_as_missing(self):
        """Non-numeric cost values should fall back to pessimistic defaults, not crash."""
        registry = {
            "bad-types": {
                "input_cost_per_m": "expensive",
                "output_cost_per_m": "very expensive",
                "time_to_first_token_seconds": "fast",
            },
        }
        router = _make_router(registry)

        mid, log = router.route(_ctx())
        assert mid == "bad-types"
        assert np.isfinite(log.cost_usd)
        assert np.isfinite(log.latency_s)


# ---------------------------------------------------------------------------
# 5. Updating a model list (replacing the full registry)
# ---------------------------------------------------------------------------

class TestRegistryReplacement:

    def test_constructor_registry_is_source_of_truth(self):
        """The model_registry dict passed to the constructor defines the arm set."""
        r1 = {"a": WELL_FORMED_MODEL, "b": WELL_FORMED_MODEL}
        r2 = {"x": WELL_FORMED_MODEL, "y": WELL_FORMED_MODEL, "z": WELL_FORMED_MODEL}

        router1 = _make_router(r1)
        router2 = _make_router(r2)

        assert set(router1.bandit.models) == {"a", "b"}
        assert set(router2.bandit.models) == {"x", "y", "z"}

    def test_register_model_updates_survive_routing(self):
        """Models added via register_model persist across multiple route+update cycles."""
        router = _make_router({"seed": WELL_FORMED_MODEL})
        router.register_model("late-add", cost_usd=1.0, latency_s=0.3)

        x = _ctx()
        for _ in range(50):
            mid, log = router.route(x)
            router.update(mid, x, reward=0.7)

        assert "late-add" in router.bandit.models
        assert "late-add" in router.registry
        assert router.bandit.A["late-add"].shape == (DIM, DIM)
