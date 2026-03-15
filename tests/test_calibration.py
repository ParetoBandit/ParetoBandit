"""
Tests for the calibration API and custom-encoder guards.

Covers:
1. train_pca() produces a valid PCA artifact
2. generate_warmup_priors() produces a valid priors dict
3. Default PCA is reproducible via the calibration API (slow)
4. FeatureService guard blocks custom encoder without explicit pca_path
5. BanditRouter.create() guard blocks custom encoder without warmup_path
6. Custom encoder is accepted when explicit artifacts are supplied
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bandit_gpt.calibration import train_pca, generate_warmup_priors
from bandit_gpt.feature_service import FeatureService, DEFAULT_CONTEXT_MODEL
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_prompts(n: int = 200) -> list[str]:
    """Return a small set of synthetic prompts for testing."""
    fs = FeatureService.__new__(FeatureService)
    fs.calibration_file = None
    return fs._generate_synthetic_fallback(n)


def _minimal_rewards_data(prompts: list[str], models: list[str]) -> list[dict]:
    """Create minimal rewards_data from prompts and model ids."""
    rng = np.random.RandomState(42)
    data = []
    for prompt in prompts:
        rewards = {m: float(rng.uniform(0.0, 1.0)) for m in models}
        data.append({"prompt": prompt, "rewards": rewards})
    return data


# ---------------------------------------------------------------------------
# Test 1: train_pca produces a valid artifact
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainPCA:
    def test_produces_valid_artifact(self, tmp_path):
        prompts = _synthetic_prompts(200)
        pca = train_pca(
            prompts,
            encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
            n_components=16,
            output_path=tmp_path / "pca_test.joblib",
        )

        assert pca.n_components_ == 16
        assert pca.n_features_in_ == 384
        assert float(np.sum(pca.explained_variance_ratio_)) > 0
        assert (tmp_path / "pca_test.joblib").exists()

    def test_rejects_empty_prompts(self):
        with pytest.raises(ValueError, match="non-empty"):
            train_pca([], encoder_model=DEFAULT_SENTENCE_TRANSFORMER)

    def test_rejects_too_few_prompts(self):
        with pytest.raises(ValueError, match="at least"):
            train_pca(
                ["hello"],
                encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
                n_components=32,
            )


# ---------------------------------------------------------------------------
# Test 2: generate_warmup_priors produces a valid artifact
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateWarmupPriors:
    def test_produces_valid_artifact(self, tmp_path):
        prompts = _synthetic_prompts(50)
        models = ["model-a", "model-b"]

        pca = train_pca(
            prompts,
            encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
            n_components=8,
        )

        rewards_data = _minimal_rewards_data(prompts, models)
        priors = generate_warmup_priors(
            rewards_data,
            encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
            pca=pca,
            plasticity=0.1,
            output_path=tmp_path / "priors_test.joblib",
        )

        assert set(priors["models"]) == set(models)
        assert priors["context_dim"] == 8 + 1  # PCA + bias
        assert priors["pca_components"] == 8
        assert priors["n_prompts"] == 50

        for m in models:
            assert priors["A"][m].shape == (9, 9)
            assert priors["b"][m].shape == (9,)
            assert not np.isnan(priors["A"][m]).any()
            assert not np.isnan(priors["b"][m]).any()

        assert (tmp_path / "priors_test.joblib").exists()

    def test_rejects_empty_rewards_data(self):
        with pytest.raises(ValueError, match="non-empty"):
            generate_warmup_priors(
                [],
                encoder_model=DEFAULT_SENTENCE_TRANSFORMER,
                pca=np.eye(3),  # dummy, should fail before use
            )


# ---------------------------------------------------------------------------
# Test 3: Guard blocks custom encoder without PCA
# ---------------------------------------------------------------------------

class TestFeatureServiceGuard:
    def test_blocks_custom_encoder_without_pca(self):
        with pytest.raises(ValueError, match="Custom encoder"):
            FeatureService(encoder_model="sentence-transformers/all-mpnet-base-v2")

    def test_allows_default_encoder_without_pca(self):
        fs = FeatureService(encoder_model=DEFAULT_CONTEXT_MODEL)
        assert fs.encoder_model == DEFAULT_CONTEXT_MODEL

    def test_allows_custom_encoder_with_explicit_pca_path(self):
        fs = FeatureService(
            encoder_model="sentence-transformers/all-mpnet-base-v2",
            pca_path="/tmp/some_custom_pca.joblib",
        )
        assert fs.encoder_model == "sentence-transformers/all-mpnet-base-v2"
        assert fs.pca_path == Path("/tmp/some_custom_pca.joblib")
        assert fs.allow_jit_training is False


# ---------------------------------------------------------------------------
# Test 5: Guard blocks custom encoder without warmup priors
# ---------------------------------------------------------------------------

class TestRouterCreateGuard:
    def test_blocks_custom_encoder_without_warmup(self):
        from bandit_gpt.router import BanditRouter

        with pytest.raises(ValueError, match="Custom encoder"):
            BanditRouter.create(
                context_model="sentence-transformers/all-mpnet-base-v2",
            )

    def test_allows_custom_encoder_with_priors_none(self):
        """priors='none' is an explicit opt-out from warmup, so it should
        not trigger the guard even with a custom encoder.  The FeatureService
        guard still requires pca_path, so we inject a precomputed service."""
        from bandit_gpt.router import BanditRouter

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            context_model="sentence-transformers/all-mpnet-base-v2",
            priors="none",
            feature_service=fs,
        )
        assert router is not None


# ---------------------------------------------------------------------------
# Test 6: Custom encoder accepted with explicit artifacts
# ---------------------------------------------------------------------------

class TestCustomEncoderWithArtifacts:
    def test_feature_service_accepts_custom_pca_path(self):
        """FeatureService should not raise when pca_path is explicitly
        provided, even for a custom encoder.  Path existence is validated
        lazily at load time, not at construction."""
        fs = FeatureService(
            encoder_model="custom-org/my-encoder",
            pca_path="/tmp/fake_pca.joblib",
        )
        assert fs.encoder_model == "custom-org/my-encoder"
        assert fs.allow_jit_training is False
