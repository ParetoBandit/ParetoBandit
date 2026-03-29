"""
Unit tests for self-healing PCA via FeatureService.

PCA lifecycle is owned by FeatureService (not BanditRouter).  These tests
validate that the FeatureService can:
1. Auto-train PCA when the artifact is missing (JIT path)
2. Load an existing valid PCA artifact
3. Validate variance capture
4. Create a BanditRouter with a JIT-trained PCA
5. Persist JIT-trained PCA for subsequent startups
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter


class TestSelfHealingPCA:
    """Test self-healing PCA functionality via FeatureService."""

    def test_missing_pca_auto_trains(self) -> None:
        """FeatureService auto-trains PCA when the artifact path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent_pca.joblib"
            fs = FeatureService(pca_path=missing_path, allow_jit_training=True)

            pca = fs.pca
            assert pca is not None, "PCA should be auto-trained when missing"
            assert pca.n_components_ == 25, "JIT PCA defaults to 25 components"

            explained_var = float(np.sum(pca.explained_variance_ratio_))
            assert explained_var > 0.5, (
                f"PCA should capture >50% variance, got {explained_var:.1%}"
            )

    def test_valid_pca_loads_successfully(self) -> None:
        """FeatureService loads a previously-saved PCA from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "test_pca.joblib"

            fs1 = FeatureService(pca_path=pca_path, allow_jit_training=True)
            _ = fs1.pca
            assert pca_path.exists(), "JIT PCA should be persisted"

            fs2 = FeatureService(pca_path=pca_path, allow_jit_training=False)
            pca2 = fs2.pca
            assert pca2 is not None, "PCA should load from disk"
            assert pca2.n_components_ == 25

    def test_pca_variance_validation(self) -> None:
        """JIT-trained PCA captures reasonable variance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FeatureService(
                pca_path=Path(tmpdir) / "pca.joblib",
                allow_jit_training=True,
            )
            pca = fs.pca
            assert pca is not None
            explained_var = float(np.sum(pca.explained_variance_ratio_))
            assert explained_var > 0.3, f"PCA variance too low: {explained_var:.1%}"

    def test_default_feature_service_works(self) -> None:
        """Default FeatureService (shipped PCA) initializes without error."""
        fs = FeatureService()
        assert fs.dimension > 0
        vec = fs.extract_features("What is machine learning?")
        assert np.all(np.isfinite(vec))

    def test_jit_pca_persistence(self) -> None:
        """JIT-trained PCA is persisted and reloaded identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "jit_pca.joblib"

            fs1 = FeatureService(pca_path=pca_path, allow_jit_training=True)
            var1 = float(np.sum(fs1.pca.explained_variance_ratio_))

            assert pca_path.exists(), "JIT PCA should be saved"

            fs2 = FeatureService(pca_path=pca_path, allow_jit_training=False)
            var2 = float(np.sum(fs2.pca.explained_variance_ratio_))

            assert abs(var1 - var2) < 0.01, "Loaded PCA should match saved PCA"


class TestPCAIntegration:
    """Test PCA integration with BanditRouter routing workflow."""

    def test_routing_with_default_pca(self) -> None:
        """BanditRouter routes correctly with the shipped default PCA."""
        registry = {
            "test/model1": {
                "model_id": "test/model1",
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 2.0,
                "initial_quality": 0.8,
            },
            "test/model2": {
                "model_id": "test/model2",
                "input_cost_per_m": 0.5,
                "output_cost_per_m": 1.0,
                "initial_quality": 0.6,
            },
        }
        router = BanditRouter.create(
            model_registry=registry,
            priors="none",
        )

        selected, log = router.route("Write a Python function to sort a list")
        assert selected in registry
        assert log.context_vector is not None
        assert len(log.context_vector) > 0
