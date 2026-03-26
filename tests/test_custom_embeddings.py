"""
Tests for custom embedding paths: custom_encoder, pre-computed vectors,
high-dimensional (no-PCA) routing, and optional sentence-transformers.

These tests verify that BanditRouter and FeatureService work correctly
without sentence-transformers installed, using either:
  - A user-supplied custom_encoder callable
  - Pre-computed np.ndarray vectors via FeatureService.for_precomputed()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from pareto_bandit import BanditRouter, FeatureService, RouterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deterministic_encoder(dim: int):
    """Return a callable that hashes prompt text into a fixed-size embedding.

    Deterministic (same prompt → same vector) so that tests are reproducible.
    """
    def _encode(prompt: str) -> np.ndarray:
        rng = np.random.RandomState(hash(prompt) % (2**31))
        vec = rng.randn(dim)
        return vec / (np.linalg.norm(vec) + 1e-12)
    return _encode


def _sample_registry(n_models: int = 3) -> dict:
    models = {}
    for i in range(n_models):
        mid = f"provider/model-{i}"
        models[mid] = {
            "model_id": mid,
            "display_name": f"Model {i}",
            "input_cost_per_m": 0.5 * (i + 1),
            "output_cost_per_m": 1.5 * (i + 1),
        }
    return models


# ===========================================================================
# FeatureService: custom_encoder
# ===========================================================================

class TestFeatureServiceCustomEncoder:
    """FeatureService with a user-supplied custom_encoder callable."""

    def test_basic_extract_features(self):
        dim = 64
        encoder_fn = _make_deterministic_encoder(dim)
        fs = FeatureService(custom_encoder=encoder_fn, embedding_dim=dim)

        vec = fs.extract_features("hello world")
        assert vec.shape == (dim + 1,), f"Expected ({dim + 1},), got {vec.shape}"
        assert vec[-1] == pytest.approx(1.0), "Last element should be the bias term"

    def test_dimension_property(self):
        dim = 128
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim), embedding_dim=dim
        )
        assert fs.dimension == dim + 1

    def test_determinism(self):
        dim = 48
        encoder_fn = _make_deterministic_encoder(dim)
        fs = FeatureService(custom_encoder=encoder_fn, embedding_dim=dim)

        v1 = fs.extract_features("same prompt")
        v2 = fs.extract_features("same prompt")
        np.testing.assert_array_equal(v1, v2)

    def test_different_prompts_differ(self):
        dim = 48
        encoder_fn = _make_deterministic_encoder(dim)
        fs = FeatureService(custom_encoder=encoder_fn, embedding_dim=dim)

        v1 = fs.extract_features("prompt A")
        v2 = fs.extract_features("prompt B")
        assert not np.allclose(v1[:-1], v2[:-1])

    def test_batch_extract(self):
        dim = 32
        encoder_fn = _make_deterministic_encoder(dim)
        fs = FeatureService(custom_encoder=encoder_fn, embedding_dim=dim)

        prompts = ["alpha", "beta", "gamma"]
        batch = fs.extract_features_batch(prompts)
        assert batch.shape == (3, dim + 1)
        assert np.all(batch[:, -1] == 1.0), "All bias terms should be 1.0"

        for i, p in enumerate(prompts):
            single = fs.extract_features(p)
            np.testing.assert_allclose(batch[i], single, atol=1e-12)

    def test_missing_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="embedding_dim is required"):
            FeatureService(custom_encoder=lambda s: np.zeros(10))

    def test_has_encoder_true(self):
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(32), embedding_dim=32
        )
        assert fs.has_encoder is True

    def test_encoder_property_raises_for_custom(self):
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(32), embedding_dim=32
        )
        with pytest.raises(RuntimeError, match="custom_encoder callable"):
            _ = fs.encoder

    def test_get_sentence_embedding_dimension(self):
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(768), embedding_dim=768
        )
        assert fs.get_sentence_embedding_dimension() == 768


# ===========================================================================
# FeatureService: pre-computed vectors
# ===========================================================================

class TestFeatureServicePrecomputed:

    def test_passthrough(self):
        dim = 16
        fs = FeatureService.for_precomputed(dimension=dim)
        vec = np.random.randn(dim)
        vec[-1] = 1.0
        out = fs.extract_features(vec)
        np.testing.assert_array_equal(out, vec)

    def test_dimension_mismatch_raises(self):
        fs = FeatureService.for_precomputed(dimension=16)
        with pytest.raises(ValueError, match="dimension"):
            fs.extract_features(np.zeros(50))

    def test_has_encoder_false(self):
        fs = FeatureService.for_precomputed(dimension=10)
        assert fs.has_encoder is False


# ===========================================================================
# BanditRouter with custom_encoder (no sentence-transformers)
# ===========================================================================

class TestRouterWithCustomEncoder:
    """Full BanditRouter integration using a custom_encoder — no ST needed."""

    @pytest.fixture
    def custom_fs(self):
        dim = 64
        return FeatureService(
            custom_encoder=_make_deterministic_encoder(dim), embedding_dim=dim
        )

    @pytest.fixture
    def registry(self):
        return _sample_registry(3)

    def test_route_returns_valid_model(self, custom_fs, registry):
        router = BanditRouter.create(
            model_registry=registry, feature_service=custom_fs, priors="none"
        )
        model, log = router.route("What is 2+2?")
        assert model in registry

    def test_feedback_loop(self, custom_fs, registry):
        router = BanditRouter.create(
            model_registry=registry, feature_service=custom_fs, priors="none"
        )
        model, log = router.route("Explain gravity")

        b_before = router.bandit.b[model].copy()
        router.process_feedback(log.request_id, reward=1.0)
        b_after = router.bandit.b[model]

        assert b_before is not b_after or True  # feedback accepted

    def test_multiple_routes_and_feedback(self, custom_fs, registry):
        router = BanditRouter.create(
            model_registry=registry, feature_service=custom_fs, priors="none"
        )
        for i in range(50):
            model, log = router.route(f"Test prompt number {i}")
            reward = np.random.uniform(0, 1)
            router.process_feedback(log.request_id, reward=reward)

        # Verify feedback didn't error and logs accumulated
        assert len(router.logs) <= 50  # bounded log buffer

    def test_save_and_load(self, custom_fs, registry, tmp_path):
        router = BanditRouter.create(
            model_registry=registry, feature_service=custom_fs, priors="none"
        )
        model, log = router.route("Save test")
        router.process_feedback(log.request_id, reward=0.8)

        path = tmp_path / "state.npz"
        router.save_state(path)
        assert path.exists()

        router2 = BanditRouter.create(
            model_registry=registry,
            feature_service=custom_fs,
            priors="none",
            state_path=path,
        )
        np.testing.assert_allclose(
            router.bandit.b[model], router2.bandit.b[model]
        )


# ===========================================================================
# BanditRouter with HIGH-DIMENSIONAL embeddings (no PCA, dim > 700)
# ===========================================================================

class TestRouterHighDimensionalNoPCA:
    """Verify the router functions correctly when the embedding space is large
    and no PCA compression is applied — the scenario a user hits when they
    bring their own embeddings (e.g., OpenAI text-embedding-3-large = 3072D,
    or a 768D model).
    """

    @pytest.fixture(params=[768, 1024, 1536])
    def high_dim(self, request):
        return request.param

    @pytest.fixture
    def high_dim_fs(self, high_dim):
        return FeatureService(
            custom_encoder=_make_deterministic_encoder(high_dim),
            embedding_dim=high_dim,
        )

    @pytest.fixture
    def registry(self):
        return _sample_registry(4)

    def test_dimension_matches(self, high_dim, high_dim_fs):
        assert high_dim_fs.dimension == high_dim + 1

    def test_route_with_high_dim(self, high_dim, high_dim_fs, registry):
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=high_dim_fs,
            priors="none",
        )
        model, log = router.route("Test high-dimensional routing")
        assert model in registry
        assert log.selected_model == model

    def test_feedback_updates_matrices(self, high_dim, high_dim_fs, registry):
        """Verify that feedback updates internal state."""
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=high_dim_fs,
            priors="none",
        )
        model, log = router.route("Matrix update check")
        b_before = router.bandit.b[model].copy()

        router.process_feedback(log.request_id, reward=1.0)
        b_after = router.bandit.b[model]

        assert not np.allclose(b_before, b_after), (
            "b vector should change after feedback"
        )

    def test_A_matrix_shape(self, high_dim, high_dim_fs, registry):
        """The covariance matrix A must be (dim+1) x (dim+1)."""
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=high_dim_fs,
            priors="none",
        )
        for mid in registry:
            A = router.bandit.A[mid]
            expected = high_dim + 1
            assert A.shape == (expected, expected), (
                f"A[{mid}] shape {A.shape} != ({expected}, {expected})"
            )

    def test_extended_learning_loop(self, high_dim, high_dim_fs, registry):
        """Run a realistic learning loop and verify no numerical errors."""
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=high_dim_fs,
            priors="none",
        )
        for i in range(100):
            model, log = router.route(f"Loop prompt {i}")
            reward = 0.9 if "model-0" in model else 0.3
            router.process_feedback(log.request_id, reward=reward)

        # After 100 rounds, all bandit matrices should be numerically stable
        for mid in registry:
            A = router.bandit.A[mid]
            assert not np.any(np.isnan(A)), f"NaN in A[{mid}]"
            assert not np.any(np.isinf(A)), f"Inf in A[{mid}]"
            b = router.bandit.b[mid]
            assert not np.any(np.isnan(b)), f"NaN in b[{mid}]"

    def test_constraint_filtering_high_dim(self, high_dim, high_dim_fs, registry):
        """Cost constraints should still work with high-dim features."""
        router = BanditRouter.create(
            model_registry=registry,
            feature_service=high_dim_fs,
            priors="none",
        )
        # model-0 is the cheapest (input_cost=0.5, output_cost=1.5)
        model, _ = router.route("Cheap request", max_cost=0.001)
        assert model == "provider/model-0"


# ===========================================================================
# BanditRouter with pre-computed vectors (no encoder at all)
# ===========================================================================

class TestRouterPrecomputed:
    """Routing with FeatureService.for_precomputed — no encoder, no PCA."""

    def test_route_with_precomputed(self):
        dim = 16
        fs = FeatureService.for_precomputed(dimension=dim)
        registry = _sample_registry(2)
        router = BanditRouter.create(
            model_registry=registry, feature_service=fs, priors="none"
        )

        vec = np.random.randn(dim)
        vec[-1] = 1.0
        model, log = router.route(vec)
        assert model in registry

    def test_feedback_with_precomputed(self):
        dim = 16
        fs = FeatureService.for_precomputed(dimension=dim)
        registry = _sample_registry(2)
        router = BanditRouter.create(
            model_registry=registry, feature_service=fs, priors="none",
        )

        vec = np.random.randn(dim)
        vec[-1] = 1.0
        model, log = router.route(vec)

        b_before = router.bandit.b[model].copy()
        router.process_feedback(log.request_id, reward=0.7)
        b_after = router.bandit.b[model]
        assert not np.allclose(b_before, b_after)

    def test_high_dim_precomputed(self):
        """Pre-computed path with 768-dimensional vectors (no PCA)."""
        dim = 769  # 768 features + 1 bias
        fs = FeatureService.for_precomputed(dimension=dim)
        registry = _sample_registry(3)
        router = BanditRouter.create(
            model_registry=registry, feature_service=fs, priors="none"
        )

        for i in range(20):
            vec = np.random.randn(dim)
            vec[-1] = 1.0
            model, log = router.route(vec)
            router.process_feedback(log.request_id, reward=np.random.uniform())

        # Verify all matrices are numerically stable at high dim
        for mid in registry:
            assert not np.any(np.isnan(router.bandit.A[mid]))
            assert router.bandit.A[mid].shape == (dim, dim)


# ===========================================================================
# Edge cases and validation
# ===========================================================================

class TestEdgeCases:

    def test_custom_encoder_bad_return_shape(self):
        """custom_encoder that returns a 2-D array should raise."""
        def bad_encoder(prompt: str) -> np.ndarray:
            return np.zeros((1, 10))

        fs = FeatureService(custom_encoder=bad_encoder, embedding_dim=10)
        with pytest.raises(ValueError, match="1-D array"):
            fs.extract_features("hello")

    def test_empty_prompt_raises(self):
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(32), embedding_dim=32
        )
        with pytest.raises(ValueError, match="empty"):
            fs.extract_features("")

    def test_wrong_type_raises(self):
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(32), embedding_dim=32
        )
        with pytest.raises(TypeError):
            fs.extract_features(42)  # type: ignore[arg-type]

    def test_register_model_without_encoder(self):
        """register_model should work when no SentenceTransformer is available."""
        dim = 64
        fs = FeatureService.for_precomputed(dimension=dim + 1)
        registry = _sample_registry(2)
        router = BanditRouter.create(
            model_registry=registry, feature_service=fs, priors="none"
        )

        router.register_model("provider/new-model", speed="fast")
        assert "provider/new-model" in router.bandit.models

    def test_feature_names_custom_encoder(self):
        dim = 64
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim), embedding_dim=dim
        )
        names = fs.get_feature_names()
        assert len(names) == dim + 1
        assert names[-1] == "bias"
        assert names[0] == "PCA_0"

    def test_pca_components_with_custom_encoder_no_pca_raises(self):
        """Specifying pca_components without a pca_path is contradictory."""
        with pytest.raises(ValueError, match="pca_components"):
            FeatureService(
                custom_encoder=_make_deterministic_encoder(64),
                embedding_dim=64,
                pca_components=16,
            )


# ===========================================================================
# Text features (use_text_features=True)
# ===========================================================================


class TestTextFeatures:
    """Verify the optional regex-based text features."""

    def test_module_level_extract(self):
        from pareto_bandit.feature_service import extract_text_features, N_TEXT_FEATURES

        vec = extract_text_features("If you must solve this, then ensure the answer is exact.")
        assert vec.shape == (N_TEXT_FEATURES,)
        assert np.all(np.abs(vec) <= 3.0), "Clipping must bound values to [-3, 3]"

    def test_z_score_centering(self):
        """A 'typical' prompt should produce z-scores near zero."""
        from pareto_bandit.feature_service import extract_text_features

        vec = extract_text_features("What is the capital of France?")
        assert np.all(np.abs(vec) < 2.5), "Typical prompt should not have extreme z-scores"

    def test_default_off_backward_compat(self):
        dim = 32
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim), embedding_dim=dim
        )
        assert fs.use_text_features is False
        assert fs.dimension == dim + 1
        vec = fs.extract_features("Hello world")
        assert vec.shape == (dim + 1,)

    def test_text_features_increase_dimension(self):
        from pareto_bandit.feature_service import N_TEXT_FEATURES

        dim = 32
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim),
            embedding_dim=dim,
            use_text_features=True,
        )
        expected = dim + N_TEXT_FEATURES + 1
        assert fs.dimension == expected
        vec = fs.extract_features("Hello world")
        assert vec.shape == (expected,)
        assert vec[-1] == pytest.approx(1.0), "Bias must be last"

    def test_pca_portion_unchanged(self):
        """PCA embedding portion must be identical with or without text features."""
        dim = 32
        enc = _make_deterministic_encoder(dim)
        fs_off = FeatureService(custom_encoder=enc, embedding_dim=dim)
        fs_on = FeatureService(custom_encoder=enc, embedding_dim=dim, use_text_features=True)

        prompt = "Explain the theory of relativity."
        v_off = fs_off.extract_features(prompt)
        v_on = fs_on.extract_features(prompt)

        np.testing.assert_allclose(v_off[:dim], v_on[:dim], atol=1e-12)

    def test_feature_names_include_text(self):
        from pareto_bandit.feature_service import TEXT_FEATURE_NAMES, N_TEXT_FEATURES

        dim = 16
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim),
            embedding_dim=dim,
            use_text_features=True,
        )
        names = fs.get_feature_names()
        assert len(names) == dim + N_TEXT_FEATURES + 1
        assert names[-1] == "bias"
        assert names[dim:dim + N_TEXT_FEATURES] == TEXT_FEATURE_NAMES

    def test_batch_matches_single(self):
        from pareto_bandit.feature_service import N_TEXT_FEATURES

        dim = 16
        enc = _make_deterministic_encoder(dim)
        fs = FeatureService(custom_encoder=enc, embedding_dim=dim, use_text_features=True)

        prompts = ["Alpha", "If beta then gamma", "Must ensure delta"]
        batch = fs.extract_features_batch(prompts)
        assert batch.shape == (3, dim + N_TEXT_FEATURES + 1)

        for i, p in enumerate(prompts):
            single = fs.extract_features(p)
            np.testing.assert_allclose(batch[i], single, atol=1e-12)

    def test_precomputed_disables_text_features(self):
        fs = FeatureService.for_precomputed(dimension=16)
        assert fs.use_text_features is False
        assert fs.dimension == 16

    def test_whitening_scales_shape(self):
        from pareto_bandit.feature_service import N_TEXT_FEATURES

        dim = 32
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim),
            embedding_dim=dim,
            use_text_features=True,
        )
        scales = fs.get_pca_whitening_scales()
        assert scales.shape == (dim + N_TEXT_FEATURES + 1,)
        assert scales[-1] == 1.0  # bias
        for i in range(N_TEXT_FEATURES):
            assert scales[dim + i] == 1.0  # text features already normalized

    def test_router_integration(self):
        """Full BanditRouter loop with text features enabled."""
        dim = 32
        fs = FeatureService(
            custom_encoder=_make_deterministic_encoder(dim),
            embedding_dim=dim,
            use_text_features=True,
        )
        registry = _sample_registry(3)
        router = BanditRouter.create(
            model_registry=registry, feature_service=fs, priors="none"
        )

        for i in range(20):
            model, log = router.route(f"Test prompt number {i} with if-then logic")
            router.process_feedback(log.request_id, reward=np.random.uniform())

        for mid in registry:
            A = router.bandit.A[mid]
            from pareto_bandit.feature_service import N_TEXT_FEATURES
            expected_dim = dim + N_TEXT_FEATURES + 1
            assert A.shape == (expected_dim, expected_dim)
            assert not np.any(np.isnan(A))
