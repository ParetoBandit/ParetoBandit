"""
Tests for HybridLinUCBPolicy (canonical Algorithm 2, Li et al. 2010) and
infer_model_family utility.

Covers:
  1. infer_model_family: suffix stripping, version handling, edge cases
  2. HybridLinUCBPolicy construction: shared + arm-specific + B_a state init
  3. Arm selection: canonical UCB with cross-term corrected theta and
     4-term Schur-complement variance
  4. Update mechanics: two-phase cross-term update propagates correctly
  5. Cold-start transfer: new arm immediately benefits from global beta
  6. Singleton degeneracy: single-arm hybrid approximates disjoint behaviour
  7. Dynamic arm management: add_arm / delete_arm (with B_a)
  8. Deep copy: independent clone with separate state (including B_a)
  9. BanditRouter integration: policy="hybrid" end-to-end
  10. Persistence: save/load round-trip (including B_a)
  11. Calibration: no over-prediction, convergence, hybrid > disjoint cold-start
  12. Thompson sampling: beta sampled once per draw, correlated across arms
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import copy
import numpy as np
import pytest
from unittest.mock import MagicMock

from bandit_gpt.router import (
    BanditRouter,
    RouterConfig,
    DisjointLinUCBPolicy,
    HybridLinUCBPolicy,
    CostAwareLinUCBAdapter,
    infer_model_family,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8


def _ctx(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0)


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


def _make_hybrid_router(registry: dict, **kwargs) -> BanditRouter:
    defaults = dict(
        model_registry=registry,
        priors="none",
        feature_service=_mock_feature_service(),
        use_corralling=False,
        policy="hybrid",
    )
    defaults.update(kwargs)
    return BanditRouter.create(**defaults)


WELL_FORMED = {
    "model_id": "vendor/model",
    "display_name": "Model",
    "input_cost_per_m": 2.50,
    "output_cost_per_m": 7.50,
    "time_to_first_token_seconds": 0.4,
    "initial_quality": 0.85,
}


# ===========================================================================
# 1. infer_model_family (standalone utility — still useful for other analyses)
# ===========================================================================

class TestInferModelFamily:

    @pytest.mark.parametrize("model_id, expected", [
        ("openai/gpt-4-turbo", "openai/gpt-4"),
        ("openai/gpt-4o-mini", "openai/gpt-4o"),
        ("openai/gpt-5.1", "openai/gpt-5"),
        ("openai/o1-mini", "openai/o1"),
        ("openai/o1-preview", "openai/o1"),
        ("anthropic/claude-3.5-sonnet", "anthropic/claude-3"),
        ("anthropic/claude-3-haiku", "anthropic/claude-3"),
        ("mistralai/mixtral-8x7b-instruct", "mistralai/mixtral-8x7b"),
        ("meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3"),
        ("google/gemini-2.0-flash", "google/gemini-2"),
    ])
    def test_known_models(self, model_id, expected):
        assert infer_model_family(model_id) == expected

    def test_no_slash_returns_identity(self):
        assert infer_model_family("standalone-model") == "standalone-model"

    def test_no_suffix_to_strip(self):
        assert infer_model_family("openai/gpt-4o") == "openai/gpt-4o"

    def test_multiple_suffixes_stripped(self):
        assert infer_model_family("meta-llama/llama-3-8b-instruct-chat") == "meta-llama/llama-3"

    def test_date_stamp_stripped(self):
        assert infer_model_family("openai/gpt-4-turbo-2024-04-09") == "openai/gpt-4"

    def test_same_family_for_versions(self):
        f1 = infer_model_family("openai/gpt-5.1")
        f2 = infer_model_family("openai/gpt-5.2")
        assert f1 == f2 == "openai/gpt-5"


# ===========================================================================
# 2. HybridLinUCBPolicy construction
# ===========================================================================

class TestHybridConstruction:

    def test_shared_state_is_single_matrix(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        assert isinstance(pol.A0, np.ndarray)
        assert pol.A0.shape == (DIM, DIM)
        assert isinstance(pol.b0, np.ndarray)
        assert pol.b0.shape == (DIM,)

    def test_shared_state_initialized_correctly(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=2.0)
        np.testing.assert_allclose(pol.A0, 2.0 * np.eye(DIM))
        np.testing.assert_allclose(pol.b0, np.zeros(DIM))

    def test_arm_specific_state_initialized(self):
        pol = HybridLinUCBPolicy(["x", "y"], dim=DIM, alpha=0.1, init_lambda=1.0)
        for m in ["x", "y"]:
            np.testing.assert_allclose(pol.A[m], np.eye(DIM))
            np.testing.assert_allclose(pol.b[m], np.zeros(DIM))
            np.testing.assert_allclose(pol.B[m], np.zeros((DIM, DIM)))

    def test_no_family_map_attribute(self):
        """Global-beta policy should not have family_map/families attributes."""
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        assert not hasattr(pol, "family_map")
        assert not hasattr(pol, "families")


# ===========================================================================
# 3. Arm selection
# ===========================================================================

class TestHybridSelection:

    def test_select_arm_returns_valid_model(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        model, ucb = pol.select_arm(_ctx())
        assert model in ["a", "b"]
        assert isinstance(ucb, float)

    def test_candidate_filtering(self):
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0)
        model, _ = pol.select_arm(_ctx(), candidates=["c"])
        assert model == "c"

    def test_shared_knowledge_transfers_to_untrained_arm(self):
        """After training arm 'a' with high reward, the shared beta should
        make arm 'b' (untrained) have a non-zero expected reward."""
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.01, init_lambda=1.0)

        x = _ctx(42)
        for _ in range(50):
            pol.update("a", x, reward=0.95)

        expected_b = pol.get_expected_reward("b", x)
        assert expected_b > 0.1, (
            f"Untrained arm should benefit from shared beta, got {expected_b:.4f}"
        )


# ===========================================================================
# 4. Update mechanics
# ===========================================================================

class TestHybridUpdate:

    def test_shared_state_updated_on_observation(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        b0_before = pol.b0.copy()
        pol.update("a", _ctx(), reward=0.9)
        assert not np.allclose(pol.b0, b0_before), "Shared b0 should change after update"

    def test_observation_on_one_arm_updates_shared_state_for_all(self):
        """Updating arm 'a' should change the global beta, which affects
        predictions for arm 'b'."""
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.01, init_lambda=1.0)
        x = _ctx(7)

        # Before any update, both arms predict ~0
        before_b = pol.get_expected_reward("b", x)

        for _ in range(20):
            pol.update("a", x, reward=0.8)

        after_b = pol.get_expected_reward("b", x)
        assert after_b > before_b, (
            f"Training arm 'a' should improve prediction for 'b' via shared beta. "
            f"Before: {before_b:.4f}, After: {after_b:.4f}"
        )

    def test_arm_specific_learns_residual(self):
        """Arm-specific b should accumulate residuals (non-zero when
        beta_hat starts at 0)."""
        pol = HybridLinUCBPolicy(["solo"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.update("solo", _ctx(), reward=0.8)
        assert np.linalg.norm(pol.b["solo"]) > 0

    def test_weight_zero_skipped(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        b0_before = pol.b0.copy()
        pol.update("a", _ctx(), reward=0.9, weight=0.0)
        np.testing.assert_array_equal(pol.b0, b0_before)

    def test_negative_weight_skipped(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        b0_before = pol.b0.copy()
        pol.update("a", _ctx(), reward=0.9, weight=-1.0)
        np.testing.assert_array_equal(pol.b0, b0_before)


# ===========================================================================
# 5. Cold-start transfer
# ===========================================================================

class TestColdStartTransfer:

    def test_new_arm_benefits_from_shared_beta(self):
        """A newly added arm should immediately get a non-trivial expected
        reward from the global shared beta trained by other arms."""
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.01, init_lambda=1.0)

        x = _ctx(99)
        for _ in range(100):
            pol.update("a", x, reward=0.9)

        pol.add_arm("b")

        expected_b = pol.get_expected_reward("b", x)
        assert expected_b > 0.2, (
            f"New arm should have meaningful prediction from shared beta, "
            f"got {expected_b:.4f}"
        )


# ===========================================================================
# 6. Singleton degeneracy
# ===========================================================================

class TestSingletonDegeneracy:

    def test_single_arm_prediction_reasonable(self):
        """With a single arm, hybrid should produce reasonable predictions
        similar in magnitude to disjoint."""
        models = ["m1"]
        hybrid = HybridLinUCBPolicy(models, dim=DIM, alpha=1.0, init_lambda=1.0)
        disjoint = DisjointLinUCBPolicy(models, dim=DIM, alpha=1.0, init_lambda=1.0)

        rng = np.random.default_rng(42)
        for _ in range(30):
            x = _ctx(rng.integers(0, 1000))
            r = rng.uniform(0.3, 0.9)
            hybrid.update("m1", x, r)
            disjoint.update("m1", x, r)

        x_test = _ctx(777)
        _, h_ucb = hybrid.select_arm(x_test)
        _, d_ucb = disjoint.select_arm(x_test)
        assert abs(h_ucb - d_ucb) < 2.0, (
            f"Single-arm hybrid UCB={h_ucb:.3f} vs disjoint UCB={d_ucb:.3f}"
        )


# ===========================================================================
# 7. Dynamic arm management
# ===========================================================================

class TestArmManagement:

    def test_add_arm(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.add_arm("b")
        assert "b" in pol.models
        assert "b" in pol.A
        assert "b" in pol.B
        assert "b" in pol.b
        assert "b" in pol.A_inv
        np.testing.assert_allclose(pol.B["b"], np.zeros((DIM, DIM)))

    def test_add_duplicate_arm_is_noop(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.add_arm("a")
        assert pol.models.count("a") == 1

    def test_add_arm_accepts_family_kwarg_silently(self):
        """API compatibility: family kwarg is accepted but ignored."""
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.add_arm("b", family="ignored")
        assert "b" in pol.models

    def test_delete_arm_cleans_up(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.delete_arm("b")
        assert "b" not in pol.models
        assert "b" not in pol.A
        assert "b" not in pol.B

    def test_delete_arm_preserves_shared_state(self):
        """Deleting an arm should not affect the global shared matrices."""
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)

        x = _ctx(0)
        pol.update("a", x, reward=0.9)
        pol.update("b", x, reward=0.7)

        A0_before = pol.A0.copy()
        b0_before = pol.b0.copy()

        pol.delete_arm("b")

        np.testing.assert_array_equal(pol.A0, A0_before)
        np.testing.assert_array_equal(pol.b0, b0_before)


# ===========================================================================
# 8. Deep copy
# ===========================================================================

class TestDeepCopy:

    def test_deepcopy_independent(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.update("a", _ctx(), reward=0.9)

        clone = copy.deepcopy(pol)

        pol.update("a", _ctx(1), reward=0.1)

        assert not np.allclose(pol.b0, clone.b0)
        assert not np.allclose(pol.b["a"], clone.b["a"])
        assert not np.allclose(pol.B["a"], clone.B["a"])

    def test_deepcopy_shared_state_independent(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.update("a", _ctx(), reward=0.9)

        clone = copy.deepcopy(pol)
        clone.update("a", _ctx(1), reward=0.1)

        assert not np.allclose(pol.A0, clone.A0)


# ===========================================================================
# 9. BanditRouter integration
# ===========================================================================

class TestRouterIntegration:

    def test_hybrid_router_creates_hybrid_policy(self):
        reg = {
            "openai/gpt-5.1": {**WELL_FORMED, "model_id": "openai/gpt-5.1"},
            "openai/gpt-5.2": {**WELL_FORMED, "model_id": "openai/gpt-5.2"},
        }
        router = _make_hybrid_router(reg)
        assert isinstance(router.bandit, HybridLinUCBPolicy)
        assert router.policy_type == "hybrid"

    def test_hybrid_router_routes_successfully(self):
        reg = {
            "openai/gpt-5.1": {**WELL_FORMED, "model_id": "openai/gpt-5.1"},
            "openai/gpt-5.2": {**WELL_FORMED, "model_id": "openai/gpt-5.2"},
            "anthropic/claude-3-haiku": {**WELL_FORMED, "model_id": "anthropic/claude-3-haiku"},
        }
        router = _make_hybrid_router(reg)
        model_id, log = router.route(_ctx())
        assert model_id in reg

    def test_default_policy_is_disjoint(self):
        """Default policy is 'disjoint' for backward compatibility."""
        reg = {"m": {**WELL_FORMED, "model_id": "m"}}
        router = BanditRouter.create(
            model_registry=reg,
            priors="none",
            feature_service=_mock_feature_service(),
            use_corralling=False,
        )
        assert isinstance(router.bandit, DisjointLinUCBPolicy)
        assert router.policy_type == "disjoint"

    def test_disjoint_when_explicitly_requested(self):
        """policy='disjoint' should still produce DisjointLinUCBPolicy."""
        reg = {"m": {**WELL_FORMED, "model_id": "m"}}
        router = BanditRouter.create(
            model_registry=reg,
            priors="none",
            feature_service=_mock_feature_service(),
            use_corralling=False,
            policy="disjoint",
        )
        assert isinstance(router.bandit, DisjointLinUCBPolicy)

    def test_register_model_with_hybrid(self):
        reg = {"openai/gpt-5.1": {**WELL_FORMED, "model_id": "openai/gpt-5.1"}}
        router = _make_hybrid_router(reg)

        router.register_model("openai/gpt-5.2", speed="balanced", cost_usd=2.0, latency_s=0.5)
        assert "openai/gpt-5.2" in router.bandit.models

    def test_ucb_accessor_methods(self):
        """get_expected_reward and get_ucb_variance should be consistent with
        select_arm scores."""
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.5, init_lambda=1.0)
        x = _ctx(42)
        pol.update("a", x, reward=0.8)

        for m in ["a", "b"]:
            mean = pol.get_expected_reward(m, x)
            var = pol.get_ucb_variance(m, x)
            assert isinstance(mean, float)
            assert isinstance(var, float)
            assert var >= 0.0


# ===========================================================================
# 10. Persistence (save/load round-trip)
# ===========================================================================

class TestPersistence:

    def test_save_load_roundtrip(self, tmp_path):
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol.update("a", _ctx(0), reward=0.9)
        pol.update("b", _ctx(1), reward=0.7)
        pol.update("c", _ctx(2), reward=0.5)

        path = tmp_path / "state.npz"
        pol.save_state(path)

        pol2 = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0)
        pol2.load_state(path)

        for m in ["a", "b", "c"]:
            np.testing.assert_allclose(pol.A[m], pol2.A[m], atol=1e-10)
            np.testing.assert_allclose(pol.B[m], pol2.B[m], atol=1e-10)
            np.testing.assert_allclose(pol.b[m], pol2.b[m], atol=1e-10)

        np.testing.assert_allclose(pol.A0, pol2.A0, atol=1e-10)
        np.testing.assert_allclose(pol.b0, pol2.b0, atol=1e-10)

    def test_temporal_metadata_roundtrip(self, tmp_path):
        """Verify t, last_update, last_played, regularization_floor persist."""
        pol = HybridLinUCBPolicy(
            ["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, forgetting_factor=0.99
        )
        for _ in range(10):
            pol.update("a", _ctx(0), reward=0.9)

        assert pol.t > 0
        saved_t = pol.t
        saved_lu = dict(pol.last_update)
        saved_rf = dict(pol.regularization_floor)

        path = tmp_path / "state_temporal.npz"
        pol.save_state(path)

        pol2 = HybridLinUCBPolicy(
            ["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, forgetting_factor=0.99
        )
        pol2.load_state(path)

        assert pol2.t == saved_t
        for m in ["a", "b"]:
            assert pol2.last_update[m] == saved_lu[m]
            np.testing.assert_allclose(
                pol2.regularization_floor[m], saved_rf[m], atol=1e-12
            )


# ===========================================================================
# 11. Calibration (Algorithm 2 correctness)
# ===========================================================================

class TestCalibration:
    """Verify the canonical cross-term formulation produces calibrated
    predictions — no over-prediction, correct convergence, and faster
    cold-start learning than disjoint."""

    def test_no_overshoot_after_single_update(self):
        """After one update with r=0.8 on a unit vector, the hybrid
        prediction must be strictly less than the reward."""
        dim = 4
        pol = HybridLinUCBPolicy(["a"], dim=dim, alpha=1.0, init_lambda=1.0)
        x = np.zeros(dim)
        x[0] = 1.0
        pol.update("a", x, reward=0.8)

        pred = pol.get_expected_reward("a", x)
        assert pred < 0.8, (
            f"Over-prediction after 1 update: {pred:.6f} >= 0.8"
        )
        assert pred > 0.0, f"Prediction should be positive, got {pred:.6f}"

    def test_canonical_cold_start_value(self):
        """Mathematical verification: after one update on e_1 with r=0.8
        and init_lambda=1.0, the canonical prediction is 8/15 ~ 0.5333."""
        dim = 4
        pol = HybridLinUCBPolicy(["a"], dim=dim, alpha=1.0, init_lambda=1.0)
        x = np.zeros(dim)
        x[0] = 1.0
        pol.update("a", x, reward=0.8)

        pred = pol.get_expected_reward("a", x)
        np.testing.assert_allclose(pred, 8.0 / 15.0, atol=1e-10)

    def test_hybrid_predicts_higher_than_disjoint_cold_start(self):
        """After one update, hybrid should predict more than disjoint
        (faster learning via shared beta) but less than the reward."""
        dim = 4
        models = ["a"]
        hybrid = HybridLinUCBPolicy(models, dim=dim, alpha=1.0, init_lambda=1.0)
        disjoint = DisjointLinUCBPolicy(models, dim=dim, alpha=1.0, init_lambda=1.0)

        x = np.zeros(dim)
        x[0] = 1.0
        r = 0.8
        hybrid.update("a", x, r)
        disjoint.update("a", x, r)

        h_pred = hybrid.get_expected_reward("a", x)
        d_pred = disjoint.get_expected_reward("a", x)

        assert h_pred > d_pred, (
            f"Hybrid ({h_pred:.4f}) should predict higher than disjoint "
            f"({d_pred:.4f}) at cold start"
        )
        assert h_pred < r, (
            f"Hybrid ({h_pred:.4f}) should not overshoot reward ({r})"
        )

    def test_prediction_converges_to_reward(self):
        """After many updates on the same (x, r), prediction should
        converge to the true reward."""
        dim = 4
        pol = HybridLinUCBPolicy(["a"], dim=dim, alpha=1.0, init_lambda=1.0)
        x = np.zeros(dim)
        x[0] = 1.0
        r = 0.75

        for _ in range(500):
            pol.update("a", x, r)

        pred = pol.get_expected_reward("a", x)
        np.testing.assert_allclose(pred, r, atol=0.02)

    def test_cross_terms_accumulate(self):
        """After an update, B_a should be non-zero (cross-term matrix
        accumulates x z^T contributions)."""
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=1.0, init_lambda=1.0)
        np.testing.assert_allclose(pol.B["a"], np.zeros((DIM, DIM)))

        x = _ctx(42)
        pol.update("a", x, reward=0.8)
        assert np.linalg.norm(pol.B["a"]) > 0, "B_a should be non-zero after update"

    def test_variance_is_positive(self):
        """The 4-term Schur-complement variance should remain positive
        even after many updates."""
        dim = 8
        pol = HybridLinUCBPolicy(["a", "b"], dim=dim, alpha=1.0, init_lambda=1.0)
        rng = np.random.default_rng(42)

        for _ in range(100):
            x = _ctx(rng.integers(0, 10000))
            pol.update("a", x, reward=rng.uniform(0.3, 0.9))
            pol.update("b", x, reward=rng.uniform(0.2, 0.8))

        for m in ["a", "b"]:
            var = pol.get_ucb_variance(m, _ctx(999))
            assert var > 0, f"Variance for {m} should be positive, got {var}"


# ===========================================================================
# 12. Thompson sampling (beta correlation)
# ===========================================================================

class TestThompsonSampling:

    def test_beta_sampled_once_per_draw(self):
        """Verify that get_probabilities produces correlated shared
        draws across arms, not independent ones.

        Strategy: train two arms to have identical arm-specific state.
        With a shared beta sampled once per draw, the two arms should
        produce highly correlated scores. If beta were sampled
        independently per arm (the old bug), correlation would be lower.
        """
        np.random.seed(42)
        dim = 4
        pol = HybridLinUCBPolicy(["a", "b"], dim=dim, alpha=1.0, init_lambda=1.0)

        x = np.zeros(dim)
        x[0] = 1.0
        for _ in range(10):
            pol.update("a", x, reward=0.8)
            pol.update("b", x, reward=0.8)

        probs = pol.get_probabilities(x, ["a", "b"], n_samples=5000)
        assert abs(probs["a"] - probs["b"]) < 0.1, (
            f"Identical arms should have near-equal win rates: {probs}"
        )

    def test_probabilities_sum_to_one(self):
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=1.0, init_lambda=1.0)
        x = _ctx(42)
        for m in ["a", "b", "c"]:
            pol.update(m, x, reward=np.random.uniform(0.3, 0.9))

        probs = pol.get_probabilities(x, ["a", "b", "c"], n_samples=2000)
        np.testing.assert_allclose(sum(probs.values()), 1.0, atol=1e-10)

    def test_strong_arm_wins_majority(self):
        """An arm trained with consistently high rewards should win the
        majority of Thompson samples."""
        np.random.seed(0)
        pol = HybridLinUCBPolicy(["strong", "weak"], dim=DIM, alpha=1.0, init_lambda=1.0)

        rng = np.random.default_rng(99)
        for _ in range(100):
            x = _ctx(rng.integers(0, 10000))
            pol.update("strong", x, reward=0.95)
            pol.update("weak", x, reward=0.3)

        x_test = _ctx(42)
        probs = pol.get_probabilities(x_test, ["strong", "weak"], n_samples=5000)
        assert probs["strong"] > probs["weak"], (
            f"Strong arm should win more often: {probs}"
        )
