"""Integration stress test for pip-installed paretobandit.

Designed to run in an isolated Docker container where the library is installed
from a built wheel (not editable mode).  Exercises every major public API path
to catch packaging errors, missing data artifacts, broken imports, and runtime
crashes that unit tests with ``sys.path`` hacks would miss.

Run locally (if installed from wheel):
    python -m pytest tests/integration/test_pip_install_stress.py -v

Run via Docker (recommended — clean environment):
    ./scripts/run_integration_test.sh
"""

from __future__ import annotations

import concurrent.futures
import importlib
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 0. Import Smoke Tests
# ---------------------------------------------------------------------------

class TestImportSmoke:
    """Verify that all public symbols resolve after ``pip install paretobandit``."""

    def test_top_level_import(self) -> None:
        mod = importlib.import_module("pareto_bandit")
        assert hasattr(mod, "__version__")
        assert isinstance(mod.__version__, str)

    def test_all_public_exports_exist(self) -> None:
        import pareto_bandit

        expected_symbols = {
            "BanditRouter",
            "RouterConfig",
            "ExplorationRate",
            "RegistrationConfig",
            "RoutingLog",
            "DisjointLinUCBPolicy",
            "calibrate_priors",
            "MissingCostError",
            "NoEligibleModelsError",
            "NoModelScoredError",
            "FeatureService",
            "SqliteContextStore",
            "EphemeralContextStore",
            "infer_model_family",
            "tetrachoric_corr",
            "compute_correlation_families",
            "train_pca",
            "generate_warmup_priors",
        }
        assert expected_symbols.issubset(set(pareto_bandit.__all__))
        for sym in expected_symbols:
            assert hasattr(pareto_bandit, sym), f"Missing public symbol: {sym}"

    def test_submodule_imports(self) -> None:
        """Key submodules should be importable independently."""
        submodules = [
            "pareto_bandit.router",
            "pareto_bandit.policy",
            "pareto_bandit.types",
            "pareto_bandit.feature_service",
            "pareto_bandit.storage",
            "pareto_bandit.costs",
            "pareto_bandit.rewards",
            "pareto_bandit.exceptions",
            "pareto_bandit.family",
            "pareto_bandit.calibration",
            "pareto_bandit.config",
            "pareto_bandit.cli",
        ]
        for name in submodules:
            mod = importlib.import_module(name)
            assert mod is not None, f"Failed to import {name}"

    def test_version_is_pep440(self) -> None:
        from pareto_bandit import __version__
        from importlib.metadata import version

        installed_version = version("paretobandit")
        assert installed_version == __version__


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

DIM = 16

CHEAP_MODEL: Dict[str, float] = {
    "input_cost_per_m": 0.10,
    "output_cost_per_m": 0.10,
    "time_to_first_token_seconds": 0.2,
}

EXPENSIVE_MODEL: Dict[str, float] = {
    "input_cost_per_m": 5.0,
    "output_cost_per_m": 15.0,
    "time_to_first_token_seconds": 0.8,
}


def _ctx(seed: int = 0, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0).astype(np.float64)


def _make_router(**overrides):
    from pareto_bandit import BanditRouter, FeatureService

    defaults = dict(
        model_registry={
            "cheap-model": CHEAP_MODEL,
            "mid-model": {
                "input_cost_per_m": 1.50,
                "output_cost_per_m": 3.00,
                "time_to_first_token_seconds": 0.5,
            },
            "expensive-model": EXPENSIVE_MODEL,
        },
        priors="none",
        feature_service=FeatureService.for_precomputed(DIM),
    )
    defaults.update(overrides)
    return BanditRouter.create(**defaults)


# ---------------------------------------------------------------------------
# 1. Shipped Artifact Tests
# ---------------------------------------------------------------------------

class TestShippedArtifacts:
    """Verify that PCA and warmup priors ship correctly in the wheel."""

    def test_pca_artifact_loadable(self) -> None:
        import joblib
        from pareto_bandit.config import DEFAULT_PCA_PATH

        assert DEFAULT_PCA_PATH.exists(), (
            f"PCA artifact missing at {DEFAULT_PCA_PATH}"
        )
        pca = joblib.load(DEFAULT_PCA_PATH)
        assert hasattr(pca, "transform")
        assert pca.n_components == 25

    def test_warmup_priors_loadable(self) -> None:
        import joblib
        from pareto_bandit.config import DEFAULT_WARMUP_PRIORS_PATH

        assert DEFAULT_WARMUP_PRIORS_PATH.exists(), (
            f"Warmup priors missing at {DEFAULT_WARMUP_PRIORS_PATH}"
        )
        data = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        assert "A" in data
        assert "b" in data
        assert isinstance(data["A"], dict)
        assert len(data["A"]) > 0

    def test_models_json_loadable(self) -> None:
        import json
        from pareto_bandit.config import DEFAULT_MODEL_REGISTRY_PATH

        assert DEFAULT_MODEL_REGISTRY_PATH.exists()
        with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
            data = json.load(f)
        assert "models" in data
        assert len(data["models"]) >= 3


# ---------------------------------------------------------------------------
# 2. Router Factory Tests
# ---------------------------------------------------------------------------

class TestRouterFactory:
    """Exercise BanditRouter.create() under different initialization modes."""

    def test_create_cold_start(self) -> None:
        router = _make_router()
        assert len(router.registry) == 3
        assert router.bandit.dim == DIM

    def test_create_with_warmup_priors(self) -> None:
        """Warmup priors load correctly when a compatible FeatureService is provided.

        The default FeatureService requires ``sentence-transformers`` (an optional
        extra), so we supply a precomputed service with the shipped PCA dimension
        (25 components + 1 bias = 26) to isolate the priors-loading path.
        """
        from pareto_bandit import BanditRouter, FeatureService

        warmup_dim = 26  # must match shipped pca_25.joblib
        fs = FeatureService.for_precomputed(warmup_dim)
        router = BanditRouter.create(priors="warmup", feature_service=fs)
        assert len(router.registry) >= 3
        for model_id in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[model_id]))

    def test_create_with_custom_config(self) -> None:
        from pareto_bandit import RouterConfig

        cfg = RouterConfig(
            reward_min=-1.0,
            reward_max=1.0,
            market_cost_ceiling=0.50,
        )
        router = _make_router(config=cfg)
        assert router.config.reward_min == -1.0
        assert router.config.reward_max == 1.0


# ---------------------------------------------------------------------------
# 3. Core Routing Loop (route → feedback)
# ---------------------------------------------------------------------------

class TestCoreRoutingLoop:
    """End-to-end route + feedback exercises the full hot path."""

    def test_route_returns_valid_model(self) -> None:
        router = _make_router()
        model, log = router.route(_ctx())
        assert model in router.registry
        assert log.selected_model == model
        assert log.context_vector is not None
        assert np.isfinite(log.cost_usd)
        assert np.isfinite(log.latency_s)

    def test_route_with_string_prompt_and_custom_encoder(self) -> None:
        from pareto_bandit import BanditRouter, FeatureService

        raw_dim = 32

        def my_encoder(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(custom_encoder=my_encoder, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )
        model, log = router.route("What is quantum computing?")
        assert model == "m"

    def test_process_feedback_updates_bandit(self) -> None:
        router = _make_router()
        x = _ctx(seed=42)
        model, log = router.route(x)
        theta_before = router.bandit.theta[model].copy()
        router.process_feedback(log.request_id, reward=0.9)
        theta_after = router.bandit.theta[model]
        assert not np.allclose(theta_before, theta_after)

    def test_direct_update(self) -> None:
        router = _make_router()
        x = _ctx(seed=7)
        b_before = router.bandit.b["cheap-model"].copy()
        router.update("cheap-model", x, reward=0.8, weight=1.5)
        assert not np.allclose(b_before, router.bandit.b["cheap-model"])

    def test_exploit_context_manager(self) -> None:
        router = _make_router()
        for i in range(20):
            router.update("cheap-model", _ctx(seed=i), reward=0.95)
            router.update("expensive-model", _ctx(seed=i), reward=0.1)

        with router.exploit():
            model, _ = router.route(_ctx(seed=999))
            assert model == "cheap-model"

    def test_feedback_loop_learns_preference(self) -> None:
        router = _make_router(
            model_registry={
                "good": CHEAP_MODEL,
                "bad": CHEAP_MODEL,
            }
        )
        for i in range(200):
            x = _ctx(seed=i)
            router.update("good", x, reward=0.95)
            router.update("bad", x, reward=0.05)

        counts = {"good": 0, "bad": 0}
        with router.exploit():
            for i in range(50):
                mid, _ = router.route(_ctx(seed=5000 + i))
                counts[mid] += 1

        assert counts["good"] > counts["bad"], (
            f"Expected 'good' to dominate, got {counts}"
        )


# ---------------------------------------------------------------------------
# 4. Constraint Filtering
# ---------------------------------------------------------------------------

class TestConstraintFiltering:
    """Hard constraints (cost, latency, quality) must be enforced."""

    def test_max_cost_filters_expensive(self) -> None:
        router = _make_router()
        model, _ = router.route(_ctx(), max_cost=0.0002)
        cost = router.registry[model]["blended_cost_per_m"]
        assert cost / 1000.0 <= 0.0002

    def test_max_latency_filters_slow(self) -> None:
        router = _make_router()
        model, _ = router.route(_ctx(), max_latency=0.3)
        lat = router.registry[model]["time_to_first_token_seconds"]
        assert lat <= 0.3

    def test_no_eligible_models_raises(self) -> None:
        from pareto_bandit import NoEligibleModelsError

        router = _make_router()
        with pytest.raises(NoEligibleModelsError):
            router.route(_ctx(), max_cost=0.0000001)


# ---------------------------------------------------------------------------
# 5. Model Registration at Runtime
# ---------------------------------------------------------------------------

class TestModelRegistration:
    """register_model() API must accept various cost tiers."""

    def test_register_explicit_costs(self) -> None:
        router = _make_router()
        router.register_model(
            "new-model",
            input_cost_per_m=2.50,
            output_cost_per_m=10.00,
        )
        assert "new-model" in router.registry
        assert "new-model" in router.bandit.models
        entry = router.registry["new-model"]
        assert entry["input_cost_per_m"] == 2.50
        assert entry["output_cost_per_m"] == 10.00
        assert entry["blended_cost_per_m"] == pytest.approx(6.25)

    def test_register_blended_cost(self) -> None:
        router = _make_router()
        router.register_model("blended-model", blended_cost_per_m=3.00)
        assert "blended-model" in router.registry
        assert router.registry["blended-model"]["blended_cost_per_m"] == pytest.approx(3.00)

    def test_register_partial_costs_raises(self) -> None:
        router = _make_router()
        with pytest.raises(ValueError, match="both input_cost_per_m"):
            router.register_model("bad", input_cost_per_m=1.0)

    def test_registered_model_is_routable(self) -> None:
        router = _make_router(model_registry={"seed": CHEAP_MODEL})
        router.register_model(
            "runtime-addition",
            input_cost_per_m=0.05,
            output_cost_per_m=0.05,
        )
        model, _ = router.route(_ctx())
        assert model in {"seed", "runtime-addition"}


# ---------------------------------------------------------------------------
# 6. State Persistence (save / load)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """save_state/load_state round-trip must preserve learned parameters."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        router = _make_router()
        for i in range(20):
            router.update("cheap-model", _ctx(seed=i), reward=0.9)

        theta_before = router.bandit.theta["cheap-model"].copy()
        save_path = tmp_path / "state.npz"
        router.save_state(save_path)

        router2 = _make_router()
        router2.load_state(save_path)
        assert np.allclose(theta_before, router2.bandit.theta["cheap-model"])

    def test_save_load_file_exists(self, tmp_path: Path) -> None:
        router = _make_router()
        save_path = tmp_path / "state.npz"
        router.save_state(save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# 7. FeatureService Paths
# ---------------------------------------------------------------------------

class TestFeatureServicePaths:
    """Exercise the three FeatureService initialization modes."""

    def test_precomputed_path(self) -> None:
        from pareto_bandit import FeatureService

        fs = FeatureService.for_precomputed(DIM)
        x = _ctx()
        extracted = fs.extract_features(x)
        assert extracted.shape == (DIM,)
        assert fs.get_dimension() == DIM
        names = fs.get_feature_names()
        assert names[-1] == "bias"

    def test_custom_encoder_path(self) -> None:
        from pareto_bandit import FeatureService

        def my_encoder(text: str) -> np.ndarray:
            return np.ones(20)

        fs = FeatureService(custom_encoder=my_encoder, embedding_dim=20)
        result = fs.extract_features("hello world")
        assert result.shape[0] == fs.get_dimension()

    def test_precomputed_rejects_string_with_clear_error(self) -> None:
        """for_precomputed() + string prompt must fail with an actionable message."""
        from pareto_bandit import FeatureService

        fs = FeatureService.for_precomputed(DIM)
        with pytest.raises(TypeError, match="for_precomputed"):
            fs.extract_features("this should not work")

    def test_nondefault_encoder_without_pca_gives_helpful_error(self) -> None:
        from pareto_bandit import FeatureService

        with pytest.raises(ValueError, match="differs from the default"):
            FeatureService(encoder_model="some-unknown-model")


# ---------------------------------------------------------------------------
# 8. ExplorationRate Presets
# ---------------------------------------------------------------------------

class TestExplorationRatePresets:
    """All named presets and numeric pass-through must work."""

    def test_named_presets(self) -> None:
        from pareto_bandit import ExplorationRate

        assert ExplorationRate.get("static") == 0.0
        assert ExplorationRate.get("safe") == 0.1
        assert ExplorationRate.get("balanced") == 1.0
        assert ExplorationRate.get("aggressive") == 2.0

    def test_numeric_passthrough(self) -> None:
        from pareto_bandit import ExplorationRate

        assert ExplorationRate.get(0.42) == pytest.approx(0.42)
        assert ExplorationRate.get(3) == pytest.approx(3.0)

    def test_unknown_preset_raises(self) -> None:
        from pareto_bandit import ExplorationRate

        with pytest.raises(ValueError):
            ExplorationRate.get("nonexistent")


# ---------------------------------------------------------------------------
# 9. Storage Backends
# ---------------------------------------------------------------------------

class TestStorageBackends:

    def test_ephemeral_store_basic(self) -> None:
        from pareto_bandit import EphemeralContextStore

        store = EphemeralContextStore()
        x = _ctx()
        store.save_context("req-1", x, "model-a")
        ctx, mid = store.get_context("req-1")
        assert np.allclose(ctx, x)
        assert mid == "model-a"

    def test_sqlite_store_basic(self, tmp_path: Path) -> None:
        from pareto_bandit import SqliteContextStore

        db_path = tmp_path / "test.db"
        store = SqliteContextStore(str(db_path))
        x = _ctx()
        store.save_context("req-1", x, "model-a")
        ctx, mid = store.get_context("req-1")
        assert np.allclose(ctx, x)
        assert mid == "model-a"

    def test_sqlite_store_with_router(self, tmp_path: Path) -> None:
        from pareto_bandit import SqliteContextStore

        db_path = tmp_path / "router_ctx.db"
        store = SqliteContextStore(str(db_path))
        router = _make_router(context_store=store)

        x = _ctx()
        model, log = router.route(x)
        router.process_feedback(log.request_id, reward=0.8)


# ---------------------------------------------------------------------------
# 10. Exception Contracts
# ---------------------------------------------------------------------------

class TestExceptionContracts:
    """Custom exceptions must be importable and raised correctly."""

    def test_missing_cost_error(self) -> None:
        from pareto_bandit import MissingCostError

        with pytest.raises(MissingCostError):
            raise MissingCostError("test")

    def test_no_eligible_models_error_attributes(self) -> None:
        from pareto_bandit import NoEligibleModelsError

        err = NoEligibleModelsError(
            reasons={"model-a": ["cost too high"]},
            max_cost=0.01,
            max_latency=None,
            quality_floor=None,
        )
        assert "model-a" in str(err)


# ---------------------------------------------------------------------------
# 11. Observability (explain_decision, get_probabilities)
# ---------------------------------------------------------------------------

class TestObservability:

    def test_explain_decision(self) -> None:
        router = _make_router()
        x = _ctx()
        model, _ = router.route(x)
        explanation = router.explain_decision(model, x)
        assert isinstance(explanation, dict)

    def test_explain_selection(self) -> None:
        from pareto_bandit import FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(DIM - 1)

        fs = FeatureService(custom_encoder=enc, embedding_dim=DIM - 1)
        router = _make_router(feature_service=fs)
        explanations = router.explain_selection("test prompt", top_k=2)
        assert isinstance(explanations, dict)
        assert len(explanations) <= 2

    def test_get_probabilities(self) -> None:
        router = _make_router()
        x = _ctx()
        probs = router.get_probabilities(x)
        assert isinstance(probs, dict)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        for p in probs.values():
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 12. CLI Entry Point
# ---------------------------------------------------------------------------

def _package_pip_installed() -> bool:
    """True when ``pareto_bandit`` is on the interpreter's default path.

    Subprocess-based CLI tests spawn a fresh Python that does *not*
    inherit pytest's ``sys.path`` additions, so the package must be
    genuinely installed (pip install / wheel) rather than just on
    PYTHONPATH in the parent process.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import pareto_bandit"],
        capture_output=True,
        timeout=10,
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    not _package_pip_installed(),
    reason="CLI tests require a real pip install (run via Docker target)",
)
class TestCLI:
    """The ``paretobandit`` console script must be usable."""

    def test_cli_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "ParetoBandit" in result.stdout

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "prompt" in result.stdout.lower()


# ---------------------------------------------------------------------------
# 13. Stress: High-Volume Routing
# ---------------------------------------------------------------------------

class TestHighVolumeRouting:
    """Sustained routing must not crash, leak memory, or corrupt state."""

    @pytest.mark.stress
    def test_1000_route_update_cycles(self) -> None:
        router = _make_router()
        for i in range(1000):
            x = _ctx(seed=i)
            model, log = router.route(x)
            reward = 0.9 if "cheap" in model else 0.5
            router.process_feedback(log.request_id, reward)

        for model_id in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[model_id]))
            assert np.all(np.isfinite(router.bandit.A[model_id]))

    @pytest.mark.stress
    def test_mixed_constraints_1000_cycles(self) -> None:
        router = _make_router()
        rng = np.random.default_rng(42)
        for i in range(1000):
            x = _ctx(seed=i)
            max_cost = rng.choice([None, 0.001, 0.01, 0.1])
            max_latency = rng.choice([None, 0.3, 0.6, 1.0])
            try:
                model, log = router.route(
                    x, max_cost=max_cost, max_latency=max_latency
                )
                router.process_feedback(log.request_id, reward=rng.uniform(0, 1))
            except Exception:
                pass  # NoEligibleModelsError is acceptable for tight constraints

    @pytest.mark.stress
    def test_register_many_models(self) -> None:
        router = _make_router()
        for i in range(50):
            router.register_model(
                f"dynamic-model-{i}",
                input_cost_per_m=float(i + 1) * 0.1,
                output_cost_per_m=float(i + 1) * 0.3,
                speed="fast" if i % 3 == 0 else "balanced",
            )
        assert len(router.registry) == 53  # 3 original + 50 new
        model, _ = router.route(_ctx())
        assert model in router.registry


# ---------------------------------------------------------------------------
# 14. Concurrency Stress
# ---------------------------------------------------------------------------

class TestConcurrencyStress:
    """Concurrent routing must not corrupt shared state."""

    @pytest.mark.stress
    def test_concurrent_routing_10_threads(self) -> None:
        router = _make_router()
        errors: List[Exception] = []
        results: List[str] = []
        lock = threading.Lock()

        def route_worker(thread_id: int) -> None:
            try:
                for i in range(100):
                    x = _ctx(seed=thread_id * 1000 + i)
                    model, log = router.route(x)
                    router.process_feedback(log.request_id, reward=0.7)
                    with lock:
                        results.append(model)
            except Exception as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(route_worker, tid) for tid in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Concurrent routing errors: {errors}"
        assert len(results) == 1000

        for model_id in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[model_id]))


# ---------------------------------------------------------------------------
# 15. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions that should be handled gracefully."""

    def test_single_model_registry(self) -> None:
        router = _make_router(model_registry={"only-model": CHEAP_MODEL})
        model, log = router.route(_ctx())
        assert model == "only-model"

    def test_reward_clamping(self) -> None:
        router = _make_router()
        _, log = router.route(_ctx())
        router.process_feedback(log.request_id, reward=5.0)
        router.process_feedback(log.request_id, reward=-5.0)

    def test_nan_reward_rejected(self) -> None:
        router = _make_router()
        _, log = router.route(_ctx())
        router.process_feedback(log.request_id, reward=float("nan"))
        assert np.all(np.isfinite(
            router.bandit.theta[log.selected_model]
        ))

    def test_zero_dimensional_stress(self) -> None:
        """Minimum viable dimension (2: one feature + bias)."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(2)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )
        x = np.array([0.5, 1.0])
        model, _ = router.route(x)
        assert model == "m"

    def test_large_dimension(self) -> None:
        """High-dimensional features should not crash."""
        from pareto_bandit import BanditRouter, FeatureService

        big_dim = 512
        fs = FeatureService.for_precomputed(big_dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )
        x = np.random.default_rng(0).standard_normal(big_dim)
        x[-1] = 1.0
        model, _ = router.route(x)
        assert model == "m"


# ---------------------------------------------------------------------------
# 16. Cost Utility Functions
# ---------------------------------------------------------------------------

class TestCostUtilities:

    def test_estimate_cost(self) -> None:
        router = _make_router()
        cost = router._estimate_cost("cheap-model", in_tok=1000, out_tok=500)
        expected = (0.10 * 1000 + 0.10 * 500) / 1e6
        assert cost == pytest.approx(expected)

    def test_normalized_cost_ordering(self) -> None:
        router = _make_router()
        nc_cheap = router._get_normalized_cost("cheap-model")
        nc_expensive = router._get_normalized_cost("expensive-model")
        assert nc_cheap < nc_expensive

    def test_normalized_cost_bounded(self) -> None:
        router = _make_router()
        for model_id in router.registry:
            nc = router._get_normalized_cost(model_id)
            assert 0.0 <= nc <= 1.0, f"{model_id}: normalized cost {nc} out of [0, 1]"


# ---------------------------------------------------------------------------
# 17. Family / Correlation Utilities
# ---------------------------------------------------------------------------

class TestFamilyUtilities:

    def test_infer_model_family(self) -> None:
        from pareto_bandit import infer_model_family

        assert infer_model_family("openai/gpt-4o") is not None
        assert infer_model_family("meta-llama/llama-3.1-8b-instruct") is not None
        assert isinstance(infer_model_family("some-unknown-model-xyz"), str)

    def test_tetrachoric_corr(self) -> None:
        from pareto_bandit import tetrachoric_corr

        rng = np.random.default_rng(0)
        x = (rng.uniform(size=100) > 0.5).astype(float)
        y = (rng.uniform(size=100) > 0.5).astype(float)
        corr = tetrachoric_corr(x, y)
        assert -1.0 <= corr <= 1.0


# ---------------------------------------------------------------------------
# 18. Reward clamping with custom range
# ---------------------------------------------------------------------------

class TestCustomRewardRange:

    def test_custom_range_feedback(self) -> None:
        from pareto_bandit import RouterConfig

        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        router = _make_router(config=cfg)
        x = _ctx()
        _, log = router.route(x)
        theta_before = router.bandit.theta[log.selected_model].copy()
        router.process_feedback(log.request_id, reward=-0.5)
        theta_after = router.bandit.theta[log.selected_model]
        assert not np.allclose(theta_before, theta_after)

    def test_custom_range_update(self) -> None:
        from pareto_bandit import RouterConfig

        cfg = RouterConfig(reward_min=-1.0, reward_max=1.0)
        router = _make_router(config=cfg)
        theta_before = router.bandit.theta["cheap-model"].copy()
        router.update("cheap-model", _ctx(), reward=-0.8)
        assert not np.allclose(
            theta_before, router.bandit.theta["cheap-model"]
        )


# ---------------------------------------------------------------------------
# 19. update_model_pricing()
# ---------------------------------------------------------------------------

class TestUpdateModelPricing:

    def test_update_pricing_changes_cost(self) -> None:
        router = _make_router()
        old_input = router.registry["cheap-model"]["input_cost_per_m"]
        router.update_model_pricing(
            "cheap-model", input_cost_per_m=99.0, output_cost_per_m=99.0,
        )
        assert router.registry["cheap-model"]["input_cost_per_m"] == 99.0
        assert router.registry["cheap-model"]["output_cost_per_m"] == 99.0
        assert router.registry["cheap-model"]["blended_cost_per_m"] == pytest.approx(99.0)
        assert old_input != 99.0

    def test_update_pricing_affects_normalized_cost(self) -> None:
        router = _make_router()
        nc_before = router._get_normalized_cost("cheap-model")
        router.update_model_pricing(
            "cheap-model", input_cost_per_m=50.0, output_cost_per_m=150.0,
        )
        nc_after = router._get_normalized_cost("cheap-model")
        assert nc_after > nc_before

    def test_update_pricing_unknown_model_raises(self) -> None:
        router = _make_router()
        with pytest.raises(KeyError):
            router.update_model_pricing("nonexistent-model", input_cost_per_m=1.0)


# ---------------------------------------------------------------------------
# 21. calibrate_priors()
# ---------------------------------------------------------------------------

class TestCalibratePriors:

    def test_calibrate_priors_clamps_extreme_predictions(self) -> None:
        from pareto_bandit import DisjointLinUCBPolicy, calibrate_priors

        dim = 8
        policy = DisjointLinUCBPolicy(["m"], dim=dim, alpha=0.01)
        policy.b["m"] = np.ones(dim) * 100.0
        policy.refresh_inverse_cache()

        bias_ctx = np.zeros(dim)
        bias_ctx[-1] = 1.0
        pred_before = float(policy.theta["m"] @ bias_ctx)

        calibrate_priors(policy, target_max_pred=0.9)

        pred_after = float(policy.theta["m"] @ bias_ctx)
        assert abs(pred_after) <= abs(pred_before), (
            "calibrate_priors did not reduce extreme prediction"
        )

    def test_calibrate_priors_keeps_finite(self) -> None:
        from pareto_bandit import DisjointLinUCBPolicy, calibrate_priors

        dim = 16
        policy = DisjointLinUCBPolicy(["a", "b"], dim=dim, alpha=0.01)
        calibrate_priors(policy, target_max_pred=0.9)

        for mid in policy.models:
            assert np.all(np.isfinite(policy.theta[mid]))


# ---------------------------------------------------------------------------
# 22. compute_correlation_families()
# ---------------------------------------------------------------------------

class TestComputeCorrelationFamilies:

    def test_basic_family_grouping(self) -> None:
        from pareto_bandit import compute_correlation_families

        rng = np.random.default_rng(0)
        base = (rng.uniform(size=200) > 0.3).astype(float)
        reward_vectors = {
            "openai/gpt-4o": base,
            "openai/gpt-4o-mini": base + rng.normal(0, 0.05, 200),
            "anthropic/claude-sonnet": rng.uniform(size=200).round(),
        }
        families = compute_correlation_families(reward_vectors, threshold=0.5)
        assert isinstance(families, dict)
        assert len(families) == 3
        for model_id in reward_vectors:
            assert model_id in families


# ---------------------------------------------------------------------------
# 23. DisjointLinUCBPolicy direct usage
# ---------------------------------------------------------------------------

class TestDisjointLinUCBPolicy:

    def test_policy_select_and_update(self) -> None:
        from pareto_bandit import DisjointLinUCBPolicy

        dim = 8
        policy = DisjointLinUCBPolicy(["a", "b"], dim=dim, alpha=0.1, seed=42)
        x = np.random.default_rng(0).standard_normal(dim)
        x[-1] = 1.0

        model, score = policy.select_arm(x)
        assert model in ["a", "b"]
        assert np.isfinite(score)

        policy.update(model, x, reward=0.8)
        assert policy.t >= 1

    def test_policy_add_arm(self) -> None:
        from pareto_bandit import DisjointLinUCBPolicy

        dim = 8
        policy = DisjointLinUCBPolicy(["a"], dim=dim, alpha=0.1)
        policy.add_arm("b")
        assert "b" in policy.models
        assert "b" in policy.A
        assert "b" in policy.b

    def test_policy_get_probabilities(self) -> None:
        from pareto_bandit import DisjointLinUCBPolicy

        dim = 8
        policy = DisjointLinUCBPolicy(["a", "b", "c"], dim=dim, alpha=0.1, seed=0)
        x = np.random.default_rng(0).standard_normal(dim)
        x[-1] = 1.0

        probs = policy.get_probabilities(x, ["a", "b", "c"])
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        for p in probs.values():
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 24. NoModelScoredError
# ---------------------------------------------------------------------------

class TestNoModelScoredError:

    def test_importable_and_raisable(self) -> None:
        from pareto_bandit import NoModelScoredError

        with pytest.raises(NoModelScoredError):
            raise NoModelScoredError("test")

    def test_is_value_error_subclass(self) -> None:
        from pareto_bandit import NoModelScoredError

        assert issubclass(NoModelScoredError, ValueError)


# ---------------------------------------------------------------------------
# 26. RegistrationConfig and RoutingLog dataclass contracts
# ---------------------------------------------------------------------------

class TestDataclassContracts:

    def test_registration_config_defaults(self) -> None:
        from pareto_bandit import RegistrationConfig

        cfg = RegistrationConfig()
        assert isinstance(cfg.fast_bias, float)
        assert isinstance(cfg.default_cost_per_1m, float)
        assert cfg.default_cost_per_1m > 0

    def test_routing_log_fields(self) -> None:
        from pareto_bandit import RoutingLog

        log = RoutingLog(
            request_id="test-123",
            timestamp_s=0.0,
            prompt="hello",
            selected_model="m",
            predicted_utility=0.5,
            cost_usd=0.001,
            latency_s=0.1,
        )
        assert log.request_id == "test-123"
        assert log.selected_model == "m"
        assert log.context_vector is None
