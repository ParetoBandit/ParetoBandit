"""Validate every code example from README.md against the installed library.

Each test class corresponds to a fenced code block in the README.  Tests are
named after the section heading they appear under.  Placeholder values
(``my_encode_fn``, ``"your-model-name"``, ``my_prompt_corpus``) are replaced
with concrete stubs so the examples actually execute.

Run locally::

    python -m pytest tests/integration/test_readme_examples.py -v

Run via Docker (recommended)::

    docker build -f Dockerfile.readme -t paretobandit-readme .
    docker run --rm paretobandit-readme
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HAS_ST = True
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _HAS_ST = False


# ═══════════════════════════════════════════════════════════════════════════
# §Quick Start — BanditRouter.create() + route() + process_feedback()
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestQuickStart:
    """README § Quick Start."""

    def test_quick_start_example(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create()

        model, log = router.route("Explain the transformer architecture", max_cost=0.01)
        print(f"Model: {model}, Cost: ${log.cost_usd:.6f}")

        router.process_feedback(log.request_id, reward=0.85)

        assert model in router.registry
        assert log.cost_usd >= 0.0
        assert log.request_id


# ═══════════════════════════════════════════════════════════════════════════
# §CLI Usage
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestCLI:
    """README § CLI usage."""

    def test_cli_route_prompt(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.cli",
             "Summarize this document", "--max-cost", "0.005"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "Selected Model" in result.stdout

    def test_cli_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.cli", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "ParetoBandit" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════
# §Feature Engineering — Default pipeline
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestDefaultPipeline:
    """README § Feature Engineering — 1. Default pipeline."""

    def test_default_pipeline(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create()
        assert router is not None
        assert router.features.using_pca


# ═══════════════════════════════════════════════════════════════════════════
# §Feature Engineering — Custom encoder
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomEncoder:
    """README § Feature Engineering — 2. Custom encoder."""

    def test_custom_encoder_without_pca(self) -> None:
        from pareto_bandit import BanditRouter
        from pareto_bandit.feature_service import FeatureService

        def my_encode_fn(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(768)

        fs = FeatureService(custom_encoder=my_encode_fn, embedding_dim=768)

        router = BanditRouter.create(feature_service=fs, priors="none")

        model, log = router.route("Test prompt")
        assert model in router.registry

    def test_custom_encoder_with_pca(self, tmp_path: Path) -> None:
        from pareto_bandit.feature_service import FeatureService

        def my_encode_fn(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(768)

        import joblib
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(42)
        pca = PCA(n_components=25).fit(rng.standard_normal((100, 768)))
        pca_path = tmp_path / "my_pca.joblib"
        joblib.dump(pca, pca_path)

        fs = FeatureService(
            custom_encoder=my_encode_fn, embedding_dim=768, pca_path=str(pca_path)
        )
        vec = fs.extract_features("Test prompt")
        assert vec.shape == (26,)  # 25 PCA + 1 bias


# ═══════════════════════════════════════════════════════════════════════════
# §Feature Engineering — Precomputed feature vectors
# ═══════════════════════════════════════════════════════════════════════════


class TestPrecomputedFeatures:
    """README § Feature Engineering — 3. Precomputed feature vectors."""

    def test_precomputed_features(self) -> None:
        import numpy as np
        from pareto_bandit import BanditRouter
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService.for_precomputed(dimension=25)
        router = BanditRouter.create(feature_service=fs, priors="none")

        features = np.random.randn(25)
        model, log = router.route(features, max_cost=0.01)

        assert model in router.registry
        assert log.cost_usd >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# §Feature Engineering — Training a custom PCA
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestTrainCustomPCA:
    """README § Feature Engineering — Training a custom PCA."""

    def test_train_pca_and_create_router(self, tmp_path: Path) -> None:
        from pareto_bandit import train_pca

        my_prompt_corpus = [
            "Explain quantum computing",
            "Write a Python quicksort",
            "Solve x^2 + 2x + 1 = 0",
            "What is the meaning of life?",
            "Debug this code",
            "Compare merge sort and quicksort",
            "Translate hello to French",
            "Summarize the theory of relativity",
            "How does photosynthesis work?",
            "Write a haiku about AI",
        ] * 10  # 100 prompts (>= 100 recommended)

        output_path = str(tmp_path / "my_pca.joblib")
        pca = train_pca(
            prompts=my_prompt_corpus,
            encoder_model="all-MiniLM-L6-v2",
            n_components=25,
            output_path=output_path,
        )

        assert 0.0 < sum(pca.explained_variance_ratio_) <= 1.0
        assert Path(output_path).exists()

        from pareto_bandit import BanditRouter

        router = BanditRouter.create(
            context_model="all-MiniLM-L6-v2",
            pca_path=output_path,
        )
        assert router is not None
