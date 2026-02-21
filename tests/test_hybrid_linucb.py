"""
Tests for HybridLinUCBPolicy and model family inference.

Covers:
  1. infer_model_family: suffix stripping, version handling, edge cases
  2. HybridLinUCBPolicy construction: family mapping, state initialization
  3. Arm selection: UCB with shared + arm-specific components
  4. Update mechanics: shared update propagates to family, residual learning
  5. Cold-start transfer: new arm in existing family gets meaningful prediction
  6. Singleton degeneracy: single-member families approximate disjoint behaviour
  7. Dynamic arm management: add_arm / delete_arm with family bookkeeping
  8. Deep copy: independent clone with separate state
  9. BanditRouter integration: policy="hybrid" end-to-end
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
    HybridLinUCBPolicy,
    DisjointLinUCBPolicy,
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


def _make_hybrid_router(registry: dict, family_map=None, **kwargs) -> BanditRouter:
    defaults = dict(
        model_registry=registry,
        priors="none",
        feature_service=_mock_feature_service(),
        use_corralling=False,
        policy="hybrid",
        family_map=family_map,
    )
    defaults.update(kwargs)
    return BanditRouter.create(**defaults)


WELL_FORMED = {
    "openrouter_id": "vendor/model",
    "display_name": "Model",
    "input_cost_per_m": 2.50,
    "output_cost_per_m": 7.50,
    "time_to_first_token_seconds": 0.4,
    "initial_quality": 0.85,
}


# ===========================================================================
# 1. infer_model_family
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

    def test_explicit_family_map(self):
        fmap = {"a": "fam1", "b": "fam1", "c": "fam2"}
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        assert pol.family_map == fmap
        assert set(pol.families["fam1"]) == {"a", "b"}
        assert pol.families["fam2"] == ["c"]

    def test_auto_family_inference(self):
        models = ["openai/gpt-5.1", "openai/gpt-5.2", "anthropic/claude-3-haiku"]
        pol = HybridLinUCBPolicy(models, dim=DIM, alpha=0.1, init_lambda=1.0)
        assert pol.family_map["openai/gpt-5.1"] == "openai/gpt-5"
        assert pol.family_map["openai/gpt-5.2"] == "openai/gpt-5"
        assert "openai/gpt-5" in pol.families
        assert len(pol.families["openai/gpt-5"]) == 2

    def test_shared_state_initialized_per_family(self):
        fmap = {"a": "f1", "b": "f1", "c": "f2"}
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=2.0, family_map=fmap)
        assert set(pol.A0.keys()) == {"f1", "f2"}
        np.testing.assert_allclose(pol.A0["f1"], 2.0 * np.eye(DIM))

    def test_arm_specific_state_initialized(self):
        pol = HybridLinUCBPolicy(["x", "y"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"x": "f", "y": "f"})
        for m in ["x", "y"]:
            np.testing.assert_allclose(pol.A[m], np.eye(DIM))
            np.testing.assert_allclose(pol.b[m], np.zeros(DIM))


# ===========================================================================
# 3. Arm selection
# ===========================================================================

class TestHybridSelection:

    def test_select_arm_returns_valid_model(self):
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"a": "f", "b": "f"})
        model, ucb = pol.select_arm(_ctx())
        assert model in ["a", "b"]
        assert isinstance(ucb, float)

    def test_candidate_filtering(self):
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0,
                                  family_map={"a": "f1", "b": "f1", "c": "f2"})
        model, _ = pol.select_arm(_ctx(), candidates=["c"])
        assert model == "c"

    def test_shared_knowledge_influences_selection(self):
        """After training arm 'a' with high reward, the shared beta should
        make arm 'b' (same family, untrained) have a higher UCB than arm 'c'
        (different family, untrained)."""
        fmap = {"a": "fam_ab", "b": "fam_ab", "c": "fam_c"}
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.01, init_lambda=1.0, family_map=fmap)

        x = _ctx(42)
        for _ in range(50):
            pol.update("a", x, reward=0.95)

        # b should benefit from family shared knowledge
        _, ucb_b = pol.select_arm(x, candidates=["b"])
        _, ucb_c = pol.select_arm(x, candidates=["c"])
        assert ucb_b > ucb_c, "Family member b should have higher UCB than unrelated c"


# ===========================================================================
# 4. Update mechanics
# ===========================================================================

class TestHybridUpdate:

    def test_shared_state_updated_on_observation(self):
        fmap = {"a": "f", "b": "f"}
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        b0_before = pol.b0["f"].copy()

        pol.update("a", _ctx(), reward=0.9)

        assert not np.allclose(pol.b0["f"], b0_before), "Family b0 should change after update"

    def test_arm_specific_learns_residual_for_singleton(self):
        """Singleton family: arm-specific b should accumulate residuals (non-zero initially)."""
        pol = HybridLinUCBPolicy(["solo"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"solo": "lone_fam"})
        pol.update("solo", _ctx(), reward=0.8)
        assert np.linalg.norm(pol.b["solo"]) > 0, "Singleton should learn residual (nonzero when beta_hat starts at 0)"

    def test_residual_learning_for_multi_member(self):
        """Multi-member family: arm-specific should learn residuals (smaller magnitude)."""
        fmap = {"a": "f", "b": "f"}
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)

        x = _ctx(7)
        for _ in range(20):
            pol.update("a", x, reward=0.8)

        # Now b's arm-specific should be smaller than shared
        beta_norm = np.linalg.norm(pol.A0_inv["f"] @ pol.b0["f"])
        theta_b_norm = np.linalg.norm(pol.A_inv["b"] @ pol.b["b"])
        assert theta_b_norm < beta_norm, "Untrained arm-specific should be smaller than shared"

    def test_weight_zero_skipped(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"a": "f"})
        b0_before = pol.b0["f"].copy()
        pol.update("a", _ctx(), reward=0.9, weight=0.0)
        np.testing.assert_array_equal(pol.b0["f"], b0_before)

    def test_negative_weight_skipped(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"a": "f"})
        b0_before = pol.b0["f"].copy()
        pol.update("a", _ctx(), reward=0.9, weight=-1.0)
        np.testing.assert_array_equal(pol.b0["f"], b0_before)


# ===========================================================================
# 5. Cold-start transfer
# ===========================================================================

class TestColdStartTransfer:

    def test_new_arm_benefits_from_family(self):
        """A newly added arm in a trained family should immediately get a
        higher mean prediction than a cold-start arm in a new family."""
        fmap = {"a": "trained_fam"}
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.01, init_lambda=1.0, family_map=fmap)

        x = _ctx(99)
        for _ in range(100):
            pol.update("a", x, reward=0.9)

        pol.add_arm("b", family="trained_fam")
        pol.add_arm("c", family="cold_fam")

        # b should have higher mean than c because it shares family with a
        _, ucb_b = pol.select_arm(x, candidates=["b"])
        _, ucb_c = pol.select_arm(x, candidates=["c"])
        assert ucb_b > ucb_c

    def test_add_arm_creates_family_state(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"a": "f1"})
        pol.add_arm("b", family="f2")
        assert "f2" in pol.A0
        assert "f2" in pol.b0
        assert "f2" in pol.A0_inv


# ===========================================================================
# 6. Singleton degeneracy
# ===========================================================================

class TestSingletonDegeneracy:

    def test_singleton_prediction_matches_disjoint(self):
        """With singleton families, the shared component alone should produce
        predictions similar to a standard disjoint LinUCB."""
        models = ["m1", "m2"]
        fmap = {"m1": "fam_m1", "m2": "fam_m2"}
        hybrid = HybridLinUCBPolicy(models, dim=DIM, alpha=1.0, init_lambda=1.0, family_map=fmap)
        disjoint = DisjointLinUCBPolicy(models, dim=DIM, alpha=1.0, init_lambda=1.0)

        rng = np.random.default_rng(42)
        for _ in range(30):
            x = _ctx(rng.integers(0, 1000))
            r = rng.uniform(0.3, 0.9)
            chosen = rng.choice(models)
            hybrid.update(chosen, x, r)
            disjoint.update(chosen, x, r)

        # Check predictions are within reasonable range
        x_test = _ctx(777)
        _, h_ucb = hybrid.select_arm(x_test)
        _, d_ucb = disjoint.select_arm(x_test)
        # Not expecting exact match due to different variance terms, but
        # the selected models and UCBs should be in the same ballpark
        assert abs(h_ucb - d_ucb) < 2.0, f"Singleton hybrid UCB={h_ucb:.3f} vs disjoint UCB={d_ucb:.3f}"


# ===========================================================================
# 7. Dynamic arm management
# ===========================================================================

class TestArmManagement:

    def test_add_arm_to_existing_family(self):
        fmap = {"a": "f1"}
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol.add_arm("b", family="f1")
        assert "b" in pol.models
        assert "b" in pol.family_map
        assert "b" in pol.families["f1"]

    def test_add_duplicate_arm_is_noop(self):
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map={"a": "f"})
        pol.add_arm("a")
        assert pol.models.count("a") == 1

    def test_delete_arm_cleans_up(self):
        fmap = {"a": "f1", "b": "f1"}
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol.delete_arm("b")
        assert "b" not in pol.models
        assert "b" not in pol.A
        assert "f1" in pol.A0, "Family should survive when members remain"

    def test_delete_last_member_removes_family(self):
        fmap = {"a": "f1"}
        pol = HybridLinUCBPolicy(["a"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol.delete_arm("a")
        assert "f1" not in pol.A0
        assert "f1" not in pol.families


# ===========================================================================
# 8. Deep copy
# ===========================================================================

class TestDeepCopy:

    def test_deepcopy_independent(self):
        fmap = {"a": "f", "b": "f"}
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol.update("a", _ctx(), reward=0.9)

        clone = copy.deepcopy(pol)

        # Mutate original
        pol.update("a", _ctx(1), reward=0.1)

        # Clone should not be affected
        assert not np.allclose(pol.b0["f"], clone.b0["f"])
        assert not np.allclose(pol.b["a"], clone.b["a"])

    def test_deepcopy_family_structure(self):
        fmap = {"a": "f1", "b": "f2"}
        pol = HybridLinUCBPolicy(["a", "b"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        clone = copy.deepcopy(pol)
        assert clone.family_map == pol.family_map
        assert set(clone.families.keys()) == set(pol.families.keys())


# ===========================================================================
# 9. BanditRouter integration
# ===========================================================================

class TestRouterIntegration:

    def test_hybrid_router_creates_hybrid_policy(self):
        reg = {
            "openai/gpt-5.1": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.1"},
            "openai/gpt-5.2": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.2"},
        }
        router = _make_hybrid_router(reg)
        assert isinstance(router.bandit, HybridLinUCBPolicy)
        assert router.policy_type == "hybrid"

    def test_hybrid_router_routes_successfully(self):
        reg = {
            "openai/gpt-5.1": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.1"},
            "openai/gpt-5.2": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.2"},
            "anthropic/claude-3-haiku": {**WELL_FORMED, "openrouter_id": "anthropic/claude-3-haiku"},
        }
        router = _make_hybrid_router(reg)
        model_id, log = router.route(_ctx())
        assert model_id in reg

    def test_family_inferred_from_registry(self):
        reg = {
            "openai/gpt-5.1": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.1"},
            "openai/gpt-5.2": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.2"},
        }
        router = _make_hybrid_router(reg)
        assert router.bandit.family_map["openai/gpt-5.1"] == "openai/gpt-5"
        assert router.bandit.family_map["openai/gpt-5.2"] == "openai/gpt-5"
        assert len(router.bandit.families) == 1

    def test_explicit_family_field_in_registry(self):
        reg = {
            "model-a": {**WELL_FORMED, "openrouter_id": "model-a", "family": "custom-fam"},
            "model-b": {**WELL_FORMED, "openrouter_id": "model-b", "family": "custom-fam"},
        }
        router = _make_hybrid_router(reg)
        assert router.bandit.family_map["model-a"] == "custom-fam"
        assert router.bandit.family_map["model-b"] == "custom-fam"

    def test_explicit_family_map_overrides_registry(self):
        reg = {
            "model-a": {**WELL_FORMED, "openrouter_id": "model-a", "family": "reg-fam"},
        }
        router = _make_hybrid_router(reg, family_map={"model-a": "override-fam"})
        assert router.bandit.family_map["model-a"] == "override-fam"

    def test_disjoint_is_default(self):
        reg = {"m": {**WELL_FORMED, "openrouter_id": "m"}}
        router = BanditRouter.create(
            model_registry=reg,
            priors="none",
            feature_service=_mock_feature_service(),
            use_corralling=False,
        )
        assert isinstance(router.bandit, DisjointLinUCBPolicy)

    def test_register_model_with_hybrid(self):
        reg = {"openai/gpt-5.1": {**WELL_FORMED, "openrouter_id": "openai/gpt-5.1"}}
        router = _make_hybrid_router(reg)

        router.register_model("openai/gpt-5.2", speed="balanced", cost_usd=2.0, latency_s=0.5)
        assert "openai/gpt-5.2" in router.bandit.models
        assert router.bandit.family_map["openai/gpt-5.2"] == "openai/gpt-5"
        assert "openai/gpt-5.2" in router.bandit.families["openai/gpt-5"]


# ===========================================================================
# 10. Persistence (save/load round-trip)
# ===========================================================================

class TestPersistence:

    def test_save_load_roundtrip(self, tmp_path):
        fmap = {"a": "f1", "b": "f1", "c": "f2"}
        pol = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol.update("a", _ctx(0), reward=0.9)
        pol.update("b", _ctx(1), reward=0.7)
        pol.update("c", _ctx(2), reward=0.5)

        path = tmp_path / "state.npz"
        pol.save_state(path)

        pol2 = HybridLinUCBPolicy(["a", "b", "c"], dim=DIM, alpha=0.1, init_lambda=1.0, family_map=fmap)
        pol2.load_state(path)

        for m in ["a", "b", "c"]:
            np.testing.assert_allclose(pol.A[m], pol2.A[m], atol=1e-10)
            np.testing.assert_allclose(pol.b[m], pol2.b[m], atol=1e-10)

        for f in ["f1", "f2"]:
            np.testing.assert_allclose(pol.A0[f], pol2.A0[f], atol=1e-10)
            np.testing.assert_allclose(pol.b0[f], pol2.b0[f], atol=1e-10)
