"""
Tests that the PCA artifact ships correctly inside the package.

Validates:
1. DEFAULT_PCA_PATH resolves to a file inside the bandit_gpt package tree.
2. The artifact exists on disk and is a loadable sklearn PCA object.
3. The artifact dimensions match the default SentenceTransformer encoder.
4. FeatureService loads the shipped PCA without JIT retraining.
5. The MANIFEST.in and pyproject.toml include .joblib files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import re
import tempfile

import joblib
import numpy as np
import pytest

from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    _PACKAGE_ARTIFACTS_DIR,
    _PACKAGE_DIR,
)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPCAPathResolution:
    """Ensure DEFAULT_PCA_PATH lives inside the installed package."""

    def test_default_pca_path_is_inside_package(self) -> None:
        """DEFAULT_PCA_PATH must be a descendant of the bandit_gpt package dir."""
        assert DEFAULT_PCA_PATH.is_relative_to(
            _PACKAGE_DIR
        ), f"Expected PCA path under {_PACKAGE_DIR}, got {DEFAULT_PCA_PATH}"

    def test_default_pca_path_is_in_data_artifacts(self) -> None:
        """Path should be in data/artifacts/ for wheel inclusion."""
        assert DEFAULT_PCA_PATH.parent == _PACKAGE_ARTIFACTS_DIR

    def test_default_pca_path_filename(self) -> None:
        assert DEFAULT_PCA_PATH.name == "pca_32.joblib"


# ---------------------------------------------------------------------------
# Artifact existence and validity
# ---------------------------------------------------------------------------


class TestPCAArtifact:
    """Verify the shipped pca_32.joblib is a valid sklearn PCA."""

    def test_artifact_exists(self) -> None:
        assert DEFAULT_PCA_PATH.exists(), (
            f"PCA artifact missing at {DEFAULT_PCA_PATH}. "
            "Did you forget to copy it into src/bandit_gpt/data/artifacts/?"
        )

    def test_artifact_loadable(self) -> None:
        pca = joblib.load(DEFAULT_PCA_PATH)
        assert hasattr(pca, "transform"), "Loaded object is not a fitted PCA"
        assert hasattr(pca, "n_components_"), "PCA has no n_components_ attribute"

    def test_artifact_has_32_components(self) -> None:
        pca = joblib.load(DEFAULT_PCA_PATH)
        assert pca.n_components_ == 32, (
            f"Expected 32 PCA components, got {pca.n_components_}"
        )

    def test_artifact_input_dimension_matches_default_encoder(self) -> None:
        """PCA input dim must equal the default SentenceTransformer output dim (384)."""
        pca = joblib.load(DEFAULT_PCA_PATH)
        assert pca.n_features_in_ == 384, (
            f"PCA trained on {pca.n_features_in_}D embeddings, "
            "expected 384 (all-MiniLM-L6-v2)"
        )

    def test_artifact_explained_variance_is_reasonable(self) -> None:
        pca = joblib.load(DEFAULT_PCA_PATH)
        explained = float(np.sum(pca.explained_variance_ratio_))
        assert explained > 0.20, (
            f"Shipped PCA captures only {explained:.1%} variance — "
            "artifact may be corrupted"
        )

    def test_artifact_can_transform_random_vector(self) -> None:
        """Smoke test: transform a random 384-d vector without error."""
        pca = joblib.load(DEFAULT_PCA_PATH)
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, pca.n_features_in_))
        out = pca.transform(x)
        assert out.shape == (1, 32)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# FeatureService integration
# ---------------------------------------------------------------------------


class TestFeatureServiceLoadsShippedPCA:
    """FeatureService should load the shipped PCA (no JIT retraining)."""

    def test_feature_service_uses_shipped_pca(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        fs = FeatureService()
        # Trigger lazy loading
        pca = fs.pca
        assert pca is not None, "FeatureService.pca is None — JIT fallback fired"
        assert pca.n_components_ == 32

    def test_no_jit_warning_when_shipped_pca_exists(self, caplog: pytest.LogCaptureFixture) -> None:
        """Loading the shipped artifact should not emit a JIT CRITICAL log."""
        from bandit_gpt.feature_service import FeatureService

        with caplog.at_level(logging.WARNING, logger="bandit_gpt.feature_service"):
            fs = FeatureService()
            _ = fs.pca

        jit_messages = [r for r in caplog.records if "JIT PCA TRAINING" in r.message]
        assert len(jit_messages) == 0, (
            "JIT PCA training triggered despite shipped artifact existing"
        )

    def test_feature_service_dimension(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        fs = FeatureService()
        assert fs.dimension == 33, f"Expected 33 (32 PCA + 1 bias), got {fs.dimension}"

    def test_extract_features_shape(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        fs = FeatureService()
        vec = fs.extract_features("Explain quantum entanglement in simple terms")
        assert vec.shape == (33,), f"Expected (33,), got {vec.shape}"
        assert vec[-1] == 1.0, "Last element should be bias term = 1.0"
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# Packaging manifests
# ---------------------------------------------------------------------------


class TestPackagingManifests:
    """MANIFEST.in and pyproject.toml correctly include the PCA artifact."""

    _project_root = Path(__file__).parent.parent

    def test_manifest_includes_joblib(self) -> None:
        manifest = (self._project_root / "MANIFEST.in").read_text()
        assert "*.joblib" in manifest, (
            "MANIFEST.in missing *.joblib in recursive-include for bandit_gpt/data"
        )

    def test_pyproject_force_includes_data_dir(self) -> None:
        pyproject = (self._project_root / "pyproject.toml").read_text()
        assert "bandit_gpt/data" in pyproject, (
            "pyproject.toml missing force-include for bandit_gpt/data"
        )

    def test_build_would_include_pca(self) -> None:
        """Verify the PCA file lives under the force-included data/ tree."""
        rel = DEFAULT_PCA_PATH.relative_to(_PACKAGE_DIR)
        assert str(rel).startswith("data"), (
            f"PCA path {rel} is not under data/ — wheel will not include it"
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Experiment-only constants still point to the source-tree artifacts dir."""

    def test_artifacts_dir_unchanged(self) -> None:
        from bandit_gpt.config import ARTIFACTS_DIR

        assert ARTIFACTS_DIR.name == "artifacts"
        assert ARTIFACTS_DIR.parent.name == "src"

    def test_generic_pca_path_still_source_tree(self) -> None:
        from bandit_gpt.config import GENERIC_PCA_PATH

        assert "src" in str(GENERIC_PCA_PATH) and "artifacts" in str(GENERIC_PCA_PATH)

    def test_warmup_priors_path_still_source_tree(self) -> None:
        from bandit_gpt.config import DEFAULT_WARMUP_PRIORS_PATH

        assert "src" in str(DEFAULT_WARMUP_PRIORS_PATH)


# ---------------------------------------------------------------------------
# JIT retraining when artifact is absent
# ---------------------------------------------------------------------------


class TestJITFallbackStillWorks:
    """Self-healing JIT path must still work when the shipped artifact is absent."""

    def test_jit_trains_when_path_missing(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent_pca.joblib"
            fs = FeatureService(pca_path=missing_path, allow_jit_training=True)
            pca = fs.pca
            assert pca is not None, "JIT should have trained a PCA"
            assert pca.n_components_ == 32

    def test_jit_persists_artifact(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "jit_pca.joblib"
            fs = FeatureService(pca_path=pca_path, allow_jit_training=True)
            _ = fs.pca
            assert pca_path.exists(), "JIT-trained PCA should be persisted to disk"

    def test_jit_disabled_raises_on_missing_artifact(self) -> None:
        from bandit_gpt.feature_service import FeatureService

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "nonexistent.joblib"
            fs = FeatureService(pca_path=missing_path, allow_jit_training=False)
            with pytest.raises(RuntimeError, match="JIT training is disabled"):
                _ = fs.pca
