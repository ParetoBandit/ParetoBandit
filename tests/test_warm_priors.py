"""
Tests for user-supplied warm priors via BanditRouter.create().

The warm-prior path in create() loads a joblib file containing:
    {"A": {model_id: ndarray}, "b": {model_id: ndarray}, "n": int}

These tests verify that user-provided priors are correctly scaled,
applied to the bandit matrices, influence routing decisions, and
degrade gracefully when partial or malformed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile

import joblib
import numpy as np
import pytest
from unittest.mock import MagicMock

from bandit_gpt.router import BanditRouter


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def two_model_registry():
    return {
        "fast/model-a": {
            "model_id": "fast/model-a",
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1,
            "initial_quality": 0.6,
        },
        "premium/model-b": {
            "model_id": "premium/model-b",
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 15.0,
            "initial_quality": 0.9,
        },
    }


def _write_priors(path: Path, A: dict, b: dict, n: int) -> None:
    joblib.dump({"A": A, "b": b, "n": n}, path)


def _create_router(registry, priors_path, n_effective=100.0, **kwargs):
    return BanditRouter.create(
        model_registry=registry,
        priors=str(priors_path),
        warmup_path=str(priors_path),
        prior_n_effective=n_effective,
        feature_service=_mock_feature_service(),
        use_corralling=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Core: priors are loaded, scaled, and applied
# ---------------------------------------------------------------------------

class TestPriorLoading:

    def test_user_priors_are_scaled_correctly(self, two_model_registry, tmp_path):
        """A and b from the joblib are multiplied by (n_effective / n)."""
        n_warmup = 1000
        n_effective = 50.0
        scale = n_effective / n_warmup  # 0.05

        raw_A = {mid: np.eye(DIM) * 200.0 for mid in two_model_registry}
        raw_b = {mid: np.ones(DIM) * 40.0 for mid in two_model_registry}
        _write_priors(tmp_path / "priors.joblib", raw_A, raw_b, n_warmup)

        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=n_effective)

        init_lambda = router.bandit.init_lambda  # post-warmup regularization adds λI
        for mid in two_model_registry:
            expected_A_diag = 200.0 * scale + init_lambda
            assert np.isclose(router.bandit.A[mid][0, 0], expected_A_diag), (
                f"{mid}: A[0,0]={router.bandit.A[mid][0, 0]}, expected {expected_A_diag}"
            )
            expected_b = 40.0 * scale
            assert np.isclose(router.bandit.b[mid][0], expected_b), (
                f"{mid}: b[0]={router.bandit.b[mid][0]}, expected {expected_b}"
            )

    def test_theta_reflects_priors(self, two_model_registry, tmp_path):
        """theta = A_inv @ b should reflect the user-supplied prior beliefs."""
        n_warmup = 100
        n_effective = 100.0  # scale = 1.0

        A_dict = {}
        b_dict = {}
        for mid in two_model_registry:
            A_dict[mid] = np.eye(DIM) * 50.0
            b_dict[mid] = np.zeros(DIM)

        # Give model-b a strong positive bias term
        b_dict["premium/model-b"][-1] = 50.0 * 0.9  # bias * quality
        b_dict["fast/model-a"][-1] = 50.0 * 0.3

        _write_priors(tmp_path / "priors.joblib", A_dict, b_dict, n_warmup)
        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=n_effective)

        theta_a = router.bandit.A_inv["fast/model-a"] @ router.bandit.b["fast/model-a"]
        theta_b = router.bandit.A_inv["premium/model-b"] @ router.bandit.b["premium/model-b"]

        assert theta_b[-1] > theta_a[-1], (
            f"Expected premium theta bias ({theta_b[-1]:.4f}) > "
            f"fast theta bias ({theta_a[-1]:.4f})"
        )

    def test_inverse_cache_is_refreshed(self, two_model_registry, tmp_path):
        """A_inv must be consistent with A after prior loading."""
        raw_A = {mid: np.eye(DIM) * 500.0 for mid in two_model_registry}
        raw_b = {mid: np.ones(DIM) * 10.0 for mid in two_model_registry}
        _write_priors(tmp_path / "priors.joblib", raw_A, raw_b, 1000)

        router = _create_router(two_model_registry, tmp_path / "priors.joblib")

        for mid in two_model_registry:
            product = router.bandit.A[mid] @ router.bandit.A_inv[mid]
            assert np.allclose(product, np.eye(DIM), atol=1e-6), (
                f"A @ A_inv != I for {mid}"
            )


# ---------------------------------------------------------------------------
# Behavioral: priors influence routing decisions
# ---------------------------------------------------------------------------

class TestPriorInfluence:

    def test_strong_prior_steers_selection(self, two_model_registry, tmp_path):
        """A model with a much higher prior reward should be selected consistently."""
        A_dict = {mid: np.eye(DIM) * 10.0 for mid in two_model_registry}
        b_dict = {}

        # Give premium/model-b an overwhelmingly positive prior
        b_dict["premium/model-b"] = np.zeros(DIM)
        b_dict["premium/model-b"][-1] = 10.0 * 0.95  # high quality belief

        # Give fast/model-a a very low prior
        b_dict["fast/model-a"] = np.zeros(DIM)
        b_dict["fast/model-a"][-1] = 10.0 * 0.10

        _write_priors(tmp_path / "priors.joblib", A_dict, b_dict, 100)
        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=100.0, alpha=0.01)

        x = np.zeros(DIM)
        x[-1] = 1.0  # just the bias term, no semantic content

        selections = []
        for _ in range(30):
            mid, _ = router.route(x)
            selections.append(mid)

        premium_count = selections.count("premium/model-b")
        assert premium_count > 20, (
            f"Expected premium model to dominate, got {premium_count}/30"
        )

    def test_no_priors_gives_cold_start(self, two_model_registry):
        """With priors='none', both models start identically (A=λI, b=0)."""
        router = BanditRouter.create(
            model_registry=two_model_registry,
            priors="none",
            feature_service=_mock_feature_service(),
            use_corralling=False,
        )

        for mid in two_model_registry:
            init_lambda = router.bandit.init_lambda
            assert np.allclose(router.bandit.A[mid], np.eye(DIM) * init_lambda)
            assert np.allclose(router.bandit.b[mid], np.zeros(DIM))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestPriorEdgeCases:

    def test_partial_coverage_uses_heuristic_fallback(self, two_model_registry, tmp_path):
        """Models missing from joblib fall back to heuristic initialization."""
        A_dict = {"fast/model-a": np.eye(DIM) * 200.0}
        b_dict = {"fast/model-a": np.ones(DIM) * 20.0}
        _write_priors(tmp_path / "priors.joblib", A_dict, b_dict, 1000)

        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=50.0)

        # model-a: from joblib (scaled)
        scale = 50.0 / 1000
        init_lambda = router.bandit.init_lambda
        expected = 200.0 * scale + init_lambda
        assert np.isclose(router.bandit.A["fast/model-a"][0, 0], expected)

        # model-b: heuristic fallback → A = init_lambda*I (from heuristic) + init_lambda*I (post-warmup)
        expected_heuristic = init_lambda * 2.0
        assert np.isclose(router.bandit.A["premium/model-b"][0, 0], expected_heuristic), (
            f"Heuristic A[0,0]={router.bandit.A['premium/model-b'][0, 0]}, expected {expected_heuristic}"
        )

        # Heuristic sets b[-1] = initial_quality * n_effective
        quality = two_model_registry["premium/model-b"]["initial_quality"]
        assert np.isclose(router.bandit.b["premium/model-b"][-1], quality * 50.0)

    def test_n_zero_does_not_crash(self, two_model_registry, tmp_path):
        """n=0 in joblib must not cause ZeroDivisionError."""
        raw_A = {mid: np.eye(DIM) * 10.0 for mid in two_model_registry}
        raw_b = {mid: np.ones(DIM) for mid in two_model_registry}
        _write_priors(tmp_path / "priors.joblib", raw_A, raw_b, n=0)

        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=50.0)

        # n=0 is clamped to 1, so scale = 50 / 1 = 50
        init_lambda = router.bandit.init_lambda
        expected = 10.0 * 50.0 + init_lambda
        assert np.isclose(router.bandit.A["fast/model-a"][0, 0], expected)

    def test_missing_n_key_uses_default(self, two_model_registry, tmp_path):
        """If 'n' key is absent, default of 20000 is used."""
        raw_A = {mid: np.eye(DIM) * 1000.0 for mid in two_model_registry}
        raw_b = {mid: np.ones(DIM) * 100.0 for mid in two_model_registry}
        joblib.dump({"A": raw_A, "b": raw_b}, tmp_path / "priors.joblib")  # no "n"

        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=100.0)

        scale = 100.0 / 20000  # default n
        init_lambda = router.bandit.init_lambda
        expected = 1000.0 * scale + init_lambda
        assert np.isclose(router.bandit.A["fast/model-a"][0, 0], expected)

    def test_nonexistent_path_falls_back_to_cold_start(self, two_model_registry):
        """A path that doesn't exist should silently fall back to cold start."""
        router = BanditRouter.create(
            model_registry=two_model_registry,
            priors="warmup",
            warmup_path="/nonexistent/priors.joblib",
            feature_service=_mock_feature_service(),
            use_corralling=False,
        )

        init_lambda = router.bandit.init_lambda
        for mid in two_model_registry:
            assert np.allclose(router.bandit.A[mid], np.eye(DIM) * init_lambda)

    def test_custom_n_effective_controls_prior_strength(self, two_model_registry, tmp_path):
        """Higher n_effective should produce a larger-magnitude theta."""
        raw_A = {mid: np.eye(DIM) * 100.0 for mid in two_model_registry}
        raw_b = {mid: np.zeros(DIM) for mid in two_model_registry}
        raw_b["fast/model-a"][-1] = 80.0
        _write_priors(tmp_path / "priors.joblib", raw_A, raw_b, 1000)

        router_weak = _create_router(
            two_model_registry, tmp_path / "priors.joblib", n_effective=10.0
        )
        router_strong = _create_router(
            two_model_registry, tmp_path / "priors.joblib", n_effective=500.0
        )

        theta_weak = (router_weak.bandit.A_inv["fast/model-a"]
                      @ router_weak.bandit.b["fast/model-a"])
        theta_strong = (router_strong.bandit.A_inv["fast/model-a"]
                        @ router_strong.bandit.b["fast/model-a"])

        assert abs(theta_strong[-1]) > abs(theta_weak[-1]), (
            f"|theta_strong|={abs(theta_strong[-1]):.4f} should > "
            f"|theta_weak|={abs(theta_weak[-1]):.4f}"
        )


# ---------------------------------------------------------------------------
# Integration: priors + online learning
# ---------------------------------------------------------------------------

class TestPriorsWithOnlineLearning:

    def test_online_updates_override_priors(self, two_model_registry, tmp_path):
        """Strong negative online signal should eventually flip prior preference."""
        A_dict = {mid: np.eye(DIM) * 10.0 for mid in two_model_registry}
        b_dict = {
            "premium/model-b": np.zeros(DIM),
            "fast/model-a": np.zeros(DIM),
        }
        b_dict["premium/model-b"][-1] = 10.0 * 0.9  # prior says premium is great
        b_dict["fast/model-a"][-1] = 10.0 * 0.2

        _write_priors(tmp_path / "priors.joblib", A_dict, b_dict, 100)
        router = _create_router(two_model_registry, tmp_path / "priors.joblib",
                                n_effective=100.0, alpha=0.01)

        x = np.zeros(DIM)
        x[-1] = 1.0

        # Confirm premium starts ahead
        mid_before, _ = router.route(x)
        assert mid_before == "premium/model-b"

        # Pummel premium with bad feedback, reward fast
        for _ in range(500):
            router.update("premium/model-b", x, reward=0.05)
            router.update("fast/model-a", x, reward=0.95)

        selections = [router.route(x)[0] for _ in range(20)]
        fast_count = selections.count("fast/model-a")
        assert fast_count > 15, (
            f"Expected fast model to win after negative feedback, got {fast_count}/20"
        )
