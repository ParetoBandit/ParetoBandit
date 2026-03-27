"""Integration stress test for the embeddings pipeline.

Requires ``pip install paretobandit[embeddings]`` (sentence-transformers + torch).
Exercises the default FeatureService end-to-end: model download, encoding,
PCA projection, and routing with string prompts through the shipped artifacts.

Also validates that user-supplied encoders and model registries are actually
*incorporated* into routing decisions — not silently ignored.

Run via Docker (recommended):
    ./scripts/run_integration_test.sh --embeddings

Skip marker:
    These tests are automatically skipped when sentence-transformers is not
    installed, so the core-only Docker target can safely collect this file.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Dict, List

import numpy as np
import pytest

_st_available = True
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _st_available = False

requires_embeddings = pytest.mark.skipif(
    not _st_available,
    reason="sentence-transformers not installed (pip install paretobandit[embeddings])",
)


def _package_pip_installed() -> bool:
    """True when ``pareto_bandit`` is on the interpreter's default path."""
    probe = subprocess.run(
        [sys.executable, "-c", "import pareto_bandit"],
        capture_output=True,
        timeout=10,
    )
    return probe.returncode == 0


requires_pip_install = pytest.mark.skipif(
    not _package_pip_installed(),
    reason="CLI tests require a real pip install (run via Docker target)",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. SentenceTransformer Encoder Loading
# ---------------------------------------------------------------------------

@requires_embeddings
class TestEncoderLoading:
    """Verify that the default encoder loads and produces sane embeddings."""

    def test_default_feature_service_loads_encoder(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        assert fs.has_encoder
        dim = fs.get_sentence_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2

    def test_encode_single_prompt(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        vec = fs.encode_prompt("What is quantum computing?")
        assert vec.ndim == 1
        assert vec.shape[0] == 384
        assert np.all(np.isfinite(vec))
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)

    def test_encode_batch(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        prompts = [
            "Explain relativity.",
            "Write a Python sort function.",
            "What is the capital of France?",
        ]
        matrix = fs.encode_prompts_batch(prompts)
        assert matrix.shape == (3, 384)
        assert np.all(np.isfinite(matrix))

    def test_different_prompts_produce_different_embeddings(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        v1 = fs.encode_prompt("Explain quantum entanglement.")
        v2 = fs.encode_prompt("How to bake chocolate chip cookies?")
        assert not np.allclose(v1, v2), "Distinct prompts should produce distinct embeddings"


# ---------------------------------------------------------------------------
# 2. PCA Pipeline (shipped artifact)
# ---------------------------------------------------------------------------

@requires_embeddings
class TestPCAPipeline:
    """The shipped PCA artifact must integrate with live encoder output."""

    def test_pca_loads_and_transforms(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        assert fs.using_pca
        assert fs.pca is not None
        assert fs.pca.n_components == 25

    def test_extract_features_with_string(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        features = fs.extract_features("Hello world")
        expected_dim = fs.get_dimension()
        assert features.shape == (expected_dim,)
        assert np.all(np.isfinite(features))
        assert features[-1] == pytest.approx(1.0)  # bias term

    def test_feature_dimension_matches_config(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        assert fs.dimension == 26  # 25 PCA + 1 bias

    def test_feature_names_match_dimension(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        names = fs.get_feature_names()
        assert len(names) == fs.dimension
        assert names[-1] == "bias"


# ---------------------------------------------------------------------------
# 3. Full Router with Default Encoder (the "just works" path)
# ---------------------------------------------------------------------------

@requires_embeddings
class TestDefaultRouterEndToEnd:
    """BanditRouter.create() with default settings + string prompts."""

    def test_create_default_router(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        assert router.bandit.dim == 26
        assert len(router.registry) >= 3

    def test_route_string_prompt_default_registry(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        model, log = router.route("Explain the theory of relativity in simple terms.")
        assert model in router.registry
        assert log.context_vector is not None
        assert log.context_vector.shape == (26,)
        assert np.all(np.isfinite(log.context_vector))

    def test_route_and_feedback_with_string_prompts(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        prompts = [
            "Write a recursive Fibonacci function in Python.",
            "What is the meaning of life?",
            "Compare merge sort and quicksort.",
            "Explain how transformers work in NLP.",
            "Write a haiku about programming.",
        ]
        for prompt in prompts:
            model, log = router.route(prompt)
            assert model in router.registry
            router.process_feedback(log.request_id, reward=0.8)

        for mid in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[mid]))

    def test_route_custom_registry_string_prompts(self) -> None:
        from pareto_bandit import BanditRouter

        registry = {
            "my-cheap": CHEAP_MODEL,
            "my-expensive": EXPENSIVE_MODEL,
        }
        router = BanditRouter.create(
            model_registry=registry,
            priors="none",
        )
        model, log = router.route("Debug this segfault in my C++ code.")
        assert model in registry

    def test_get_probabilities_with_string(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        probs = router.get_probabilities("Solve this calculus integral.")
        assert isinstance(probs, dict)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_explain_selection_with_string(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        explanations = router.explain_selection("Write a unit test for my API.", top_k=2)
        assert isinstance(explanations, dict)
        assert len(explanations) <= 2


# ---------------------------------------------------------------------------
# 4. Stress: repeated encoding stability
# ---------------------------------------------------------------------------

@requires_embeddings
class TestEncoderStress:
    """Repeated encoding should produce stable, finite results."""

    @pytest.mark.stress
    def test_100_sequential_encodes(self) -> None:
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        for i in range(100):
            vec = fs.encode_prompt(f"Test prompt number {i} for stress testing.")
            assert np.all(np.isfinite(vec))
            assert vec.shape[0] == 384

    @pytest.mark.stress
    def test_deterministic_encoding(self) -> None:
        """Same prompt must produce identical embeddings across calls."""
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        prompt = "Is this encoding deterministic?"
        v1 = fs.encode_prompt(prompt)
        v2 = fs.encode_prompt(prompt)
        assert np.allclose(v1, v2), "Encoding the same prompt twice should be deterministic"

    @pytest.mark.stress
    def test_200_route_cycles_with_strings(self) -> None:
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(priors="warmup")
        for i in range(200):
            model, log = router.route(f"Stress test prompt iteration {i}.")
            router.process_feedback(log.request_id, reward=np.random.default_rng(i).uniform(0, 1))

        for mid in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[mid]))
            assert np.all(np.isfinite(router.bandit.A[mid]))


# ---------------------------------------------------------------------------
# 5. CLI --download-models (model installation)
# ---------------------------------------------------------------------------

@requires_embeddings
@requires_pip_install
class TestCLIDownloadModels:
    """The ``paretobandit --download-models`` command must succeed.

    After the download, a fresh FeatureService() must be able to encode
    without any additional setup.
    """

    def test_download_models_cli_succeeds(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pareto_bandit.cli", "--download-models"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"--download-models failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "ready" in result.stdout.lower() or "downloaded" in result.stdout.lower()

    def test_encoder_usable_after_download(self) -> None:
        """After --download-models, a default FeatureService must encode immediately."""
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        vec = fs.encode_prompt("Test after download.")
        assert vec.ndim == 1
        assert vec.shape[0] == 384
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# 6. BYOE: Bring Your Own Embedder — *actually* incorporated
# ---------------------------------------------------------------------------

@requires_embeddings
class TestBringYourOwnEmbedder:
    """Verify that a user-supplied encoder is actually invoked and its output
    flows through to the bandit's theta update — not silently replaced by the
    default encoder or a zero vector.
    """

    def test_custom_encoder_output_reaches_bandit_state(self) -> None:
        """The exact vector from the custom encoder must appear in the
        context_vector stored on the RoutingLog and in the bandit's A/b update.
        """
        from pareto_bandit import BanditRouter, FeatureService

        call_log: List[str] = []
        raw_dim = 20

        def spy_encoder(text: str) -> np.ndarray:
            call_log.append(text)
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(custom_encoder=spy_encoder, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )

        prompt = "Does my encoder actually get called?"
        _, log = router.route(prompt)

        assert prompt in call_log, "Custom encoder was never called"
        assert log.context_vector is not None
        assert log.context_vector.shape == (raw_dim + 1,)  # raw_dim + bias
        assert log.context_vector[-1] == pytest.approx(1.0)

    def test_custom_encoder_output_differs_from_default(self) -> None:
        """Prove the custom encoder is *not* silently replaced by the default
        SentenceTransformer by checking that the context vectors differ.
        """
        from pareto_bandit import BanditRouter, FeatureService

        constant_vec = np.ones(30) * 0.42

        def constant_encoder(text: str) -> np.ndarray:
            return constant_vec.copy()

        fs_custom = FeatureService(custom_encoder=constant_encoder, embedding_dim=30)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs_custom,
        )

        _, log = router.route("hello")
        # The context vector should be the L2-normalized constant + bias,
        # NOT a SentenceTransformer embedding through PCA.
        pca_dims = log.context_vector[:-1]  # everything except bias
        expected_norm = constant_vec / (np.linalg.norm(constant_vec) + 1e-12)
        assert np.allclose(pca_dims, expected_norm, atol=1e-6), (
            "Custom encoder output was not used — context vector doesn't "
            "match the normalized constant vector"
        )

    def test_custom_encoder_feedback_updates_theta(self) -> None:
        """Feedback from routing with a custom encoder must shift theta."""
        from pareto_bandit import BanditRouter, FeatureService

        raw_dim = 16

        def my_encoder(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(custom_encoder=my_encoder, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )

        theta_before = router.bandit.theta["m"].copy()
        A_before = router.bandit.A["m"].copy()

        for i in range(20):
            _, log = router.route(f"feedback test prompt {i}")
            router.process_feedback(log.request_id, reward=0.9)

        assert not np.allclose(theta_before, router.bandit.theta["m"]), (
            "theta unchanged after 20 feedback rounds with custom encoder"
        )
        assert not np.allclose(A_before, router.bandit.A["m"]), (
            "A matrix unchanged after 20 feedback rounds with custom encoder"
        )

    def test_two_different_encoders_produce_different_policies(self) -> None:
        """Training with two distinct encoders on the same prompts+rewards
        should produce different learned policies (theta vectors).
        """
        from pareto_bandit import BanditRouter, FeatureService

        raw_dim = 16
        registry = {"m": CHEAP_MODEL}

        def encoder_a(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash("A" + text)) % 2**31)
            return rng.standard_normal(raw_dim)

        def encoder_b(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash("B" + text)) % 2**31)
            return rng.standard_normal(raw_dim)

        routers = {}
        for name, enc in [("a", encoder_a), ("b", encoder_b)]:
            fs = FeatureService(custom_encoder=enc, embedding_dim=raw_dim)
            r = BanditRouter.create(
                model_registry=registry, priors="none",
                feature_service=fs, bandit_seed=0,
            )
            for i in range(50):
                _, log = r.route(f"prompt {i}")
                r.process_feedback(log.request_id, reward=0.8)
            routers[name] = r

        assert not np.allclose(
            routers["a"].bandit.theta["m"],
            routers["b"].bandit.theta["m"],
        ), "Different encoders produced identical theta — encoder output is being ignored"

    def test_precomputed_vector_bypasses_encoder(self) -> None:
        """When the user passes an np.ndarray, no encoder should be called."""
        from pareto_bandit import BanditRouter, FeatureService

        call_count = [0]
        raw_dim = 10

        def counting_encoder(text: str) -> np.ndarray:
            call_count[0] += 1
            return np.zeros(raw_dim)

        fs = FeatureService(custom_encoder=counting_encoder, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry={"m": CHEAP_MODEL},
            priors="none",
            feature_service=fs,
        )

        x = np.random.default_rng(42).standard_normal(raw_dim + 1)
        x[-1] = 1.0
        _, log = router.route(x)

        assert call_count[0] == 0, "Encoder was called when a raw vector was passed"
        assert np.allclose(log.context_vector, x)


# ---------------------------------------------------------------------------
# 7. BYOM: Bring Your Own Model Config — *actually* incorporated
# ---------------------------------------------------------------------------

@requires_embeddings
class TestBringYourOwnModelConfig:
    """Verify that user-supplied model registries are actually used for
    routing decisions — costs filter correctly, latency constraints apply,
    and the bandit learns *per-model* preferences.
    """

    def test_custom_registry_costs_used_in_filtering(self) -> None:
        """A tight max_cost must exclude expensive custom models."""
        from pareto_bandit import BanditRouter

        registry = {
            "my-cheap": {
                "input_cost_per_m": 0.05,
                "output_cost_per_m": 0.05,
                "time_to_first_token_seconds": 0.1,
            },
            "my-expensive": {
                "input_cost_per_m": 50.0,
                "output_cost_per_m": 150.0,
                "time_to_first_token_seconds": 0.5,
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        for _ in range(30):
            model, _ = router.route(
                "Which model do I get under a tight budget?",
                max_cost=0.0001,
            )
            assert model == "my-cheap", (
                f"Expensive model '{model}' was selected despite tight max_cost"
            )

    def test_custom_registry_latency_used_in_filtering(self) -> None:
        """max_latency must respect the user's time_to_first_token_seconds."""
        from pareto_bandit import BanditRouter

        registry = {
            "fast-model": {
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 3.0,
                "time_to_first_token_seconds": 0.1,
            },
            "slow-model": {
                "input_cost_per_m": 1.0,
                "output_cost_per_m": 3.0,
                "time_to_first_token_seconds": 5.0,
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        for _ in range(30):
            model, _ = router.route("Low latency please.", max_latency=0.5)
            assert model == "fast-model", (
                f"Slow model '{model}' selected despite max_latency=0.5"
            )

    def test_custom_registry_quality_floor_used_in_filtering(self) -> None:
        """quality_floor must filter based on custom scores in registry."""
        from pareto_bandit import BanditRouter

        registry = {
            "high-quality": {
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 15.0,
                "scores": {"hle": 0.90},
            },
            "low-quality": {
                "input_cost_per_m": 0.10,
                "output_cost_per_m": 0.10,
                "scores": {"hle": 0.20},
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        for _ in range(30):
            model, _ = router.route(
                "I need a high quality answer.",
                quality_floor={"hle": 0.80},
            )
            assert model == "high-quality", (
                f"Low-quality model selected despite quality_floor hle=0.80"
            )

    def test_custom_registry_cost_estimation_uses_exact_rates(self) -> None:
        """_estimate_cost must use the exact input/output rates the user provided."""
        from pareto_bandit import BanditRouter

        registry = {
            "priced-model": {
                "input_cost_per_m": 4.00,
                "output_cost_per_m": 12.00,
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        cost = router._estimate_cost("priced-model", in_tok=1000, out_tok=500)
        expected = (4.00 * 1000 + 12.00 * 500) / 1e6
        assert cost == pytest.approx(expected), (
            f"Cost estimate {cost} doesn't match expected {expected} "
            f"from user-provided rates"
        )

    def test_custom_registry_models_appear_in_bandit(self) -> None:
        """Every model in the user's registry must be a bandit arm."""
        from pareto_bandit import BanditRouter

        registry = {
            f"user-model-{i}": {
                "input_cost_per_m": float(i + 1),
                "output_cost_per_m": float(i + 1) * 3,
            }
            for i in range(5)
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        for model_id in registry:
            assert model_id in router.bandit.models, (
                f"User model '{model_id}' missing from bandit arms"
            )
            assert model_id in router.registry

    def test_bandit_learns_per_model_preference_from_custom_registry(self) -> None:
        """After training, the bandit should prefer the model that receives
        consistently higher rewards — proving the custom registry models are
        properly wired into the learning loop.
        """
        from pareto_bandit import BanditRouter, FeatureService

        raw_dim = 16
        registry = {
            "reward-winner": CHEAP_MODEL,
            "reward-loser": CHEAP_MODEL,
        }

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(custom_encoder=enc, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry=registry, priors="none",
            feature_service=fs, bandit_seed=42,
        )

        for i in range(200):
            x_str = f"training prompt {i}"
            router.update("reward-winner", x_str, reward=0.95)
            router.update("reward-loser", x_str, reward=0.05)

        counts: Dict[str, int] = {"reward-winner": 0, "reward-loser": 0}
        with router.exploit():
            for i in range(50):
                model, _ = router.route(f"evaluation prompt {i}")
                counts[model] += 1

        assert counts["reward-winner"] > counts["reward-loser"], (
            f"Bandit did not learn user-registry model preference: {counts}"
        )

    def test_runtime_registered_model_gets_routed(self) -> None:
        """A model added via register_model() must be eligible and eventually
        selected by the bandit.
        """
        from pareto_bandit import BanditRouter, FeatureService

        raw_dim = 16

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(raw_dim)

        fs = FeatureService(custom_encoder=enc, embedding_dim=raw_dim)
        router = BanditRouter.create(
            model_registry={"seed-model": EXPENSIVE_MODEL},
            priors="none",
            feature_service=fs,
        )

        router.register_model(
            "late-joiner",
            input_cost_per_m=0.01,
            output_cost_per_m=0.01,
            speed="fast",
        )

        for i in range(100):
            router.update("late-joiner", f"prompt {i}", reward=0.99)
            router.update("seed-model", f"prompt {i}", reward=0.01)

        selected_models = set()
        with router.exploit():
            for i in range(30):
                model, _ = router.route(f"test prompt {i}")
                selected_models.add(model)

        assert "late-joiner" in selected_models, (
            "Runtime-registered model was never selected after heavy training"
        )

    def test_normalized_cost_reflects_custom_prices(self) -> None:
        """Normalized cost ordering must respect user-supplied pricing."""
        from pareto_bandit import BanditRouter

        registry = {
            "dirt-cheap": {
                "input_cost_per_m": 0.01,
                "output_cost_per_m": 0.01,
            },
            "mid-tier": {
                "input_cost_per_m": 2.00,
                "output_cost_per_m": 6.00,
            },
            "ultra-premium": {
                "input_cost_per_m": 50.00,
                "output_cost_per_m": 150.00,
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")

        nc_cheap = router._get_normalized_cost("dirt-cheap")
        nc_mid = router._get_normalized_cost("mid-tier")
        nc_premium = router._get_normalized_cost("ultra-premium")

        assert nc_cheap < nc_mid < nc_premium, (
            f"Normalized cost order wrong: cheap={nc_cheap:.4f}, "
            f"mid={nc_mid:.4f}, premium={nc_premium:.4f}"
        )


# ---------------------------------------------------------------------------
# 8. Calibration API: train_pca + generate_warmup_priors
# ---------------------------------------------------------------------------

@requires_embeddings
class TestCalibrationAPI:
    """Exercise the offline calibration tools that generate PCA and warmup
    priors from scratch — the user-facing API for custom encoder pipelines.
    """

    def test_train_pca_produces_valid_artifact(self, tmp_path) -> None:
        from pareto_bandit import train_pca

        prompts = [f"Sample prompt number {i} for PCA training." for i in range(100)]
        pca = train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=10,
            output_path=tmp_path / "test_pca.joblib",
        )
        assert pca.n_components == 10
        assert hasattr(pca, "transform")
        assert (tmp_path / "test_pca.joblib").exists()

        import joblib
        loaded = joblib.load(tmp_path / "test_pca.joblib")
        assert loaded.n_components == 10

    def test_train_pca_empty_prompts_raises(self) -> None:
        from pareto_bandit import train_pca

        with pytest.raises(ValueError, match="non-empty"):
            train_pca([], encoder_model="all-MiniLM-L6-v2")

    def test_generate_warmup_priors_roundtrip(self, tmp_path) -> None:
        from pareto_bandit import train_pca, generate_warmup_priors

        prompts = [f"Warmup prompt {i}" for i in range(100)]
        pca = train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=8,
        )

        rewards_data = [
            {
                "prompt": f"Reward prompt {i}",
                "rewards": {"model-a": 0.8, "model-b": 0.4},
            }
            for i in range(50)
        ]
        priors = generate_warmup_priors(
            rewards_data,
            encoder_model="all-MiniLM-L6-v2",
            pca=pca,
            output_path=tmp_path / "test_priors.joblib",
        )

        assert "A" in priors
        assert "b" in priors
        assert "model-a" in priors["A"]
        assert "model-b" in priors["A"]
        assert priors["n_prompts"] == 50
        assert priors["context_dim"] == 9  # 8 PCA + 1 bias
        assert (tmp_path / "test_priors.joblib").exists()

    def test_generated_priors_loadable_by_router(self, tmp_path) -> None:
        """Full roundtrip: train PCA, generate priors, load into a router."""
        from pareto_bandit import BanditRouter, FeatureService, train_pca, generate_warmup_priors

        prompts = [f"Roundtrip prompt {i}" for i in range(100)]
        pca = train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=8,
            output_path=tmp_path / "pca.joblib",
        )

        rewards_data = [
            {
                "prompt": f"Training prompt {i}",
                "rewards": {"fast-model": 0.9, "slow-model": 0.5},
            }
            for i in range(50)
        ]
        generate_warmup_priors(
            rewards_data,
            encoder_model="all-MiniLM-L6-v2",
            pca=pca,
            output_path=tmp_path / "priors.joblib",
        )

        registry = {
            "fast-model": CHEAP_MODEL,
            "slow-model": EXPENSIVE_MODEL,
        }
        fs = FeatureService(
            encoder_model="all-MiniLM-L6-v2",
            pca_path=tmp_path / "pca.joblib",
        )
        router = BanditRouter.create(
            model_registry=registry,
            priors=str(tmp_path / "priors.joblib"),
            feature_service=fs,
        )

        model, log = router.route("Test after loading custom priors.")
        assert model in registry
        assert np.all(np.isfinite(log.context_vector))

        for mid in router.bandit.models:
            assert np.all(np.isfinite(router.bandit.theta[mid]))
