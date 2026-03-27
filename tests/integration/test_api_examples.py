"""Validate every code example from docs/API_REFERENCE.md against the installed library.

Uses the shipped demo data (test_holdout.jsonl, models.json, pca_25.joblib,
priors_k3_25comp.joblib) so the tests are self-contained and run in Docker
after a plain ``pip install paretobandit[embeddings]``.

Run locally:
    python -m pytest tests/integration/test_api_examples.py -v

Run via Docker (recommended):
    docker build -f Dockerfile.examples --target examples -t paretobandit-examples .
    docker run --rm paretobandit-examples
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

K3_REGISTRY = {
    "meta-llama/llama-3.1-8b-instruct": {
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "input_cost_per_m": 0.10,
        "output_cost_per_m": 0.10,
        "time_to_first_token_seconds": 0.2,
    },
    "mistralai/mistral-large-2512": {
        "model_id": "mistralai/mistral-large-2512",
        "input_cost_per_m": 0.50,
        "output_cost_per_m": 1.50,
        "time_to_first_token_seconds": 0.4,
    },
    "google/gemini-2.5-pro": {
        "model_id": "google/gemini-2.5-pro",
        "input_cost_per_m": 1.25,
        "output_cost_per_m": 10.00,
        "time_to_first_token_seconds": 0.8,
    },
}

TWO_MODEL_REGISTRY = {
    "openai/gpt-4o": {
        "model_id": "openai/gpt-4o",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "time_to_first_token_seconds": 0.5,
    },
    "mistralai/mixtral-8x7b": {
        "model_id": "mistralai/mixtral-8x7b",
        "input_cost_per_m": 0.24,
        "output_cost_per_m": 0.24,
        "time_to_first_token_seconds": 0.3,
    },
}


def _load_holdout_prompts(n: int = 50) -> list[str]:
    """Load *n* prompt strings from the shipped holdout."""
    from pareto_bandit.data import get_example_holdout_path

    path = get_example_holdout_path()
    prompts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def _load_holdout_records(n: int = 50) -> list[dict]:
    """Load *n* records (prompt + arms) from the shipped holdout."""
    from pareto_bandit.data import get_example_holdout_path

    path = get_example_holdout_path()
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 1. Version Check (API Ref line 5–8)
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionExample:

    def test_version_string(self) -> None:
        import pareto_bandit

        assert isinstance(pareto_bandit.__version__, str)
        assert pareto_bandit.__version__  # non-empty


# ═══════════════════════════════════════════════════════════════════════════
# 2. BanditRouter.create() Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestBanditRouterCreateExamples:

    def test_default_warm_start(self) -> None:
        """API Ref: Default usage — warm-start with shipped priors."""
        from pareto_bandit import BanditRouter

        registry = dict(TWO_MODEL_REGISTRY)
        router = BanditRouter.create(registry)
        assert len(router.registry) == 2

    def test_cold_start(self) -> None:
        """API Ref: Cold-start (priors='none')."""
        from pareto_bandit import BanditRouter

        router = BanditRouter.create(TWO_MODEL_REGISTRY, priors="none")
        assert len(router.registry) == 2

    def test_custom_reward_scale(self) -> None:
        """API Ref: Custom reward scale [-1, 1]."""
        from pareto_bandit import BanditRouter, RouterConfig

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            priors="none",
            config=RouterConfig(reward_min=-1.0, reward_max=1.0),
        )
        assert router.config.reward_min == -1.0
        assert router.config.reward_max == 1.0

    def test_custom_encoder(self) -> None:
        """API Ref: Custom encoder callable."""
        from pareto_bandit import BanditRouter, FeatureService

        def my_encoder_fn(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(768)

        fs = FeatureService(
            custom_encoder=my_encoder_fn,
            embedding_dim=768,
        )
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )
        model_id, log = router.route("Test prompt")
        assert model_id in TWO_MODEL_REGISTRY

    def test_precomputed_vectors(self) -> None:
        """API Ref: Pre-computed vectors."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=33)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )
        vec = np.random.default_rng(0).standard_normal(33)
        vec[-1] = 1.0
        model_id, log = router.route(vec)
        assert model_id in TWO_MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# 3. route() Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestRouteExamples:

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        self.fs = FeatureService(custom_encoder=enc, embedding_dim=64)
        self.router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=self.fs, priors="none"
        )

    def test_basic_routing(self) -> None:
        """API Ref: Basic routing."""
        model_id, log = self.router.route("Write a Python function to parse JSON")

        assert model_id in TWO_MODEL_REGISTRY
        assert log.request_id
        assert np.isfinite(log.cost_usd)

    def test_route_with_constraints(self) -> None:
        """API Ref: Route with constraints."""
        model_id, log = self.router.route(
            "Explain the Riemann hypothesis",
            max_cost=5.0,
            output_tokens=200,
        )
        assert model_id in TWO_MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# 4. process_feedback() Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestProcessFeedbackExamples:

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        self.router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

    def test_standard_route_feedback_loop(self) -> None:
        """API Ref: Standard route-feedback loop (with stubs for call_llm/evaluate)."""
        model_id, log = self.router.route("Explain quantum entanglement")
        reward = 0.85  # stub for evaluate_quality(call_llm(model_id, ...))
        self.router.process_feedback(log.request_id, reward=reward)

    def test_online_learning_loop(self) -> None:
        """API Ref: Online learning loop."""
        prompts = [
            "Write a haiku about AI",
            "Solve x^2 - 4 = 0",
            "Debug this Python code",
        ]
        for prompt in prompts:
            model_id, log = self.router.route(prompt)
            reward = np.random.default_rng(abs(hash(prompt)) % 2**31).uniform()
            self.router.process_feedback(log.request_id, reward=reward)


# ═══════════════════════════════════════════════════════════════════════════
# 5. update() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestUpdateExample:

    def test_ingest_historical_data(self) -> None:
        """API Ref: Ingest historical data (update example)."""
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

        historical_data = [
            ("openai/gpt-4o", "Write a Python quicksort", 0.95),
            ("mistralai/mixtral-8x7b", "Tell me a joke", 0.72),
            ("openai/gpt-4o", "Explain relativity", 0.88),
        ]

        for model, prompt, reward in historical_data:
            router.update(model, prompt, reward)


# ═══════════════════════════════════════════════════════════════════════════
# 6. get_probabilities() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestGetProbabilitiesExample:

    def test_get_probabilities(self) -> None:
        """API Ref: get_probabilities example."""
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

        probs = router.get_probabilities("Write a SQL query to find active users")
        for model, prob in sorted(probs.items(), key=lambda x: -x[1]):
            assert 0.0 <= prob <= 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 7. explain_decision() / explain_selection() Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestExplainExamples:

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        self.router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

    def test_explain_decision(self) -> None:
        """API Ref: explain_decision example."""
        model_id, log = self.router.route("Write SQL to get active users")
        explanation = self.router.explain_decision(model_id, log.context_vector)

        assert isinstance(explanation, dict)
        for feature, contribution in sorted(
            explanation.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            assert isinstance(feature, str)
            assert np.isfinite(contribution)

    def test_explain_selection(self) -> None:
        """API Ref: explain_selection example."""
        explanations = self.router.explain_selection(
            "Debug this Python code", top_k=2
        )
        for model, features in explanations.items():
            assert isinstance(features, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 8. register_model() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestRegisterModelExample:

    def test_register_model_variants(self) -> None:
        """API Ref: register_model examples (exact pricing, blended, mystery)."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )

        router.register_model(
            "google/gemini-2.0-flash",
            speed="fast",
            input_cost_per_m=0.10,
            output_cost_per_m=0.40,
        )
        assert "google/gemini-2.0-flash" in router.registry

        router.register_model(
            "local/llama-3-8b", speed="fast", blended_cost_per_m=0.2
        )
        assert "local/llama-3-8b" in router.registry

        router.register_model("mystery/new-model")
        assert "mystery/new-model" in router.registry


# ═══════════════════════════════════════════════════════════════════════════
# 9. update_model_pricing() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestUpdateModelPricingExample:

    def test_update_model_pricing(self) -> None:
        """API Ref: update_model_pricing example."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(K3_REGISTRY, feature_service=fs, priors="none")

        router.register_model(
            "google/gemini-2.5-pro",
            input_cost_per_m=1.25,
            output_cost_per_m=10.00,
        )

        router.update_model_pricing(
            "google/gemini-2.5-pro",
            input_cost_per_m=0.10,
            output_cost_per_m=0.10,
        )
        assert router.registry["google/gemini-2.5-pro"]["input_cost_per_m"] == 0.10


# ═══════════════════════════════════════════════════════════════════════════
# 10. exploit() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestExploitExample:

    def test_exploit_context_manager(self) -> None:
        """API Ref: exploit() context manager."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )

        x = np.random.default_rng(0).standard_normal(16)
        x[-1] = 1.0

        with router.exploit():
            model, log = router.route(x)
            assert model in TWO_MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# 11. save_state() / load_state() Example
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveLoadExample:

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """API Ref: save_state / load_state."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )

        save_path = tmp_path / "router_state.npz"
        router.save_state(str(save_path))
        assert save_path.exists()

        router.load_state(str(save_path))


# ═══════════════════════════════════════════════════════════════════════════
# 12. RoutingLog Example
# ═══════════════════════════════════════════════════════════════════════════


class TestRoutingLogExample:

    def test_inspect_routing_log(self) -> None:
        """API Ref: Inspecting the routing log."""
        from pareto_bandit import BanditRouter, FeatureService

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

        model_id, log = router.route("Solve x^2 + 2x + 1 = 0")

        assert log.selected_model == model_id
        assert log.request_id
        assert np.isfinite(log.predicted_utility)
        assert np.isfinite(log.cost_usd)
        assert np.isfinite(log.latency_s)
        assert log.context_vector is not None
        assert log.context_vector.shape == (65,)  # 64 enc + 1 bias


# ═══════════════════════════════════════════════════════════════════════════
# 13. FeatureService Examples (requires sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════

_HAS_ST = True
try:
    import sentence_transformers  # noqa: F401
except ImportError:
    _HAS_ST = False


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestFeatureServiceSTExamples:
    """Examples that require the default SentenceTransformer encoder."""

    def test_default_usage_bundled_pca(self) -> None:
        """API Ref: Default usage — bundled PCA."""
        from pareto_bandit import FeatureService

        fs = FeatureService()
        vector = fs.extract_features("Explain the Pythagorean theorem")
        assert vector.shape == (26,)
        assert vector[-1] == pytest.approx(1.0)

    def test_with_text_features(self) -> None:
        """API Ref: With text features."""
        from pareto_bandit import FeatureService

        fs = FeatureService(use_text_features=True)
        vector = fs.extract_features("If x > 5 and y < 3, find the minimum")
        assert vector.shape == (30,)

    def test_extract_features_batch(self) -> None:
        """API Ref: extract_features_batch."""
        from pareto_bandit import FeatureService

        fs = FeatureService()
        prompts = [
            "Write a Python quicksort",
            "Explain the Riemann hypothesis",
            "Tell me a joke about programmers",
        ]
        vectors = fs.extract_features_batch(prompts)
        assert vectors.shape == (3, 26)

    def test_get_feature_names(self) -> None:
        """API Ref: get_feature_names."""
        from pareto_bandit import FeatureService

        fs = FeatureService()
        names = fs.get_feature_names()
        assert names[:3] == ["PCA_0", "PCA_1", "PCA_2"]
        assert names[-1] == "bias"
        assert len(names) == 26


class TestFeatureServicePrecomputedExample:

    def test_precomputed_passthrough(self) -> None:
        """API Ref: for_precomputed example."""
        from pareto_bandit import FeatureService

        fs = FeatureService.for_precomputed(dimension=26)
        vector = np.random.randn(26)
        vector[-1] = 1.0
        result = fs.extract_features(vector)
        assert np.allclose(result, vector)

    def test_precomputed_rejects_string(self) -> None:
        """API Ref: for_precomputed rejects string prompts."""
        from pareto_bandit import FeatureService

        fs = FeatureService.for_precomputed(dimension=26)
        with pytest.raises(TypeError):
            fs.extract_features("this should not work")


class TestFeatureServiceCustomEncoderExample:

    def test_custom_encoder_no_pca(self) -> None:
        """API Ref: Custom encoder — no PCA."""
        from pareto_bandit import BanditRouter, FeatureService

        def fake_openai_embed(prompt: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(prompt)) % 2**31)
            return rng.standard_normal(1536)

        fs = FeatureService(custom_encoder=fake_openai_embed, embedding_dim=1536)
        router = BanditRouter.create(
            model_registry=TWO_MODEL_REGISTRY,
            feature_service=fs,
            priors="none",
        )
        model_id, log = router.route("Explain quantum computing")
        assert model_id in TWO_MODEL_REGISTRY
        assert log.context_vector.shape == (1537,)

    def test_custom_encoder_with_pca(self, tmp_path: Path) -> None:
        """API Ref: Custom encoder with PCA compression."""
        import joblib
        from sklearn.decomposition import PCA
        from pareto_bandit import FeatureService

        def fake_openai_embed(prompt: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(prompt)) % 2**31)
            return rng.standard_normal(1536)

        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((50, 1536))
        pca = PCA(n_components=32).fit(embeddings)
        pca_path = tmp_path / "openai_pca_32.joblib"
        joblib.dump(pca, pca_path)

        fs = FeatureService(
            custom_encoder=fake_openai_embed,
            embedding_dim=1536,
            pca_path=str(pca_path),
        )
        vec = fs.extract_features("Test prompt")
        assert vec.shape == (33,)  # 32 PCA + 1 bias


# ═══════════════════════════════════════════════════════════════════════════
# 14. reference_model Example
# ═══════════════════════════════════════════════════════════════════════════


class TestReferenceModelExample:

    def test_reference_model_property(self) -> None:
        """API Ref: reference_model property."""
        from pareto_bandit import BanditRouter, FeatureService

        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY, feature_service=fs, priors="none"
        )
        ref = router.reference_model
        assert "id" in ref
        assert ref.get("initial_quality", 0) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# 15. RouterConfig Example
# ═══════════════════════════════════════════════════════════════════════════


class TestRouterConfigExample:

    def test_router_config_custom(self) -> None:
        """API Ref: RouterConfig with custom values."""
        from pareto_bandit import BanditRouter, RouterConfig

        config = RouterConfig(
            max_log_size=5_000,
            init_lambda=2.0,
            stability_check_interval=500,
            reward_min=-1.0,
            reward_max=1.0,
        )
        router = BanditRouter(
            model_registry=TWO_MODEL_REGISTRY, config=config
        )
        assert router.config.max_log_size == 5_000
        assert router.config.reward_min == -1.0


# ═══════════════════════════════════════════════════════════════════════════
# 16. ExplorationRate Example
# ═══════════════════════════════════════════════════════════════════════════


class TestExplorationRateExample:

    def test_exploration_rate_presets(self) -> None:
        """API Ref: ExplorationRate named presets."""
        from pareto_bandit import BanditRouter, ExplorationRate

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            alpha=ExplorationRate.SAFE,
            priors="none",
        )
        assert router.bandit.alpha == 0.1

        alpha = ExplorationRate.get("balanced")
        assert alpha == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 17. BudgetPacer Example
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetPacerExample:

    def test_budget_constrained_routing(self) -> None:
        """API Ref: Budget-constrained routing."""
        from pareto_bandit import BanditRouter
        from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
        from pareto_bandit import FeatureService

        pacer = BudgetPacer(
            target_avg_spend_usd=0.001,
            mode=PacingMode.ADAPTIVE,
        )

        def enc(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % 2**31)
            return rng.standard_normal(64)

        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            budget_pacer=pacer,
            feature_service=FeatureService(custom_encoder=enc, embedding_dim=64),
            priors="none",
        )

        model_id, log = router.route("Explain relativity")
        router.process_feedback(log.request_id, reward=0.85)

        state = pacer.get_state()
        assert isinstance(state, dict)
        assert "lambda_t" in state


# ═══════════════════════════════════════════════════════════════════════════
# 18. train_pca() Example (uses shipped holdout data)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestTrainPcaExample:

    def test_train_pca_from_demo_data(self, tmp_path: Path) -> None:
        """API Ref: train_pca — uses shipped holdout prompts instead of 3-item stub."""
        from pareto_bandit import train_pca, FeatureService, BanditRouter

        prompts = _load_holdout_prompts(n=50)
        assert len(prompts) >= 50

        pca_path = tmp_path / "my_pca.joblib"
        pca = train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=25,
            output_path=str(pca_path),
        )
        explained = sum(pca.explained_variance_ratio_)
        assert 0.0 < explained <= 1.0
        assert pca_path.exists()

        fs = FeatureService(pca_path=str(pca_path))
        router = BanditRouter.create(feature_service=fs)
        assert router is not None


# ═══════════════════════════════════════════════════════════════════════════
# 19. generate_warmup_priors() Example (uses shipped holdout data)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestGenerateWarmupPriorsExample:

    def test_generate_warmup_priors_from_demo_data(
        self, tmp_path: Path
    ) -> None:
        """API Ref: generate_warmup_priors — uses shipped holdout records."""
        from pareto_bandit import train_pca, generate_warmup_priors

        records = _load_holdout_records(n=50)
        prompts = [r["prompt"] for r in records]

        pca_path = tmp_path / "pca.joblib"
        train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=25,
            output_path=str(pca_path),
        )

        rewards_data = []
        for rec in records:
            entry = {
                "prompt": rec["prompt"],
                "rewards": {
                    arm: info["reward"]
                    for arm, info in rec["arms"].items()
                },
            }
            rewards_data.append(entry)

        priors_path = tmp_path / "my_priors.joblib"
        priors = generate_warmup_priors(
            rewards_data,
            encoder_model="all-MiniLM-L6-v2",
            pca=str(pca_path),
            plasticity=0.1,
            output_path=str(priors_path),
        )
        assert "A" in priors
        assert "b" in priors
        assert "models" in priors
        assert priors["n_prompts"] == len(records)
        assert priors_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
# 20. Storage Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestStorageExamples:

    def test_sqlite_context_store(self, tmp_path: Path) -> None:
        """API Ref: SqliteContextStore example."""
        from pareto_bandit import BanditRouter
        from pareto_bandit.storage import SqliteContextStore
        from pareto_bandit import FeatureService

        db_path = tmp_path / "bandit_router.db"
        store = SqliteContextStore(
            db_path=str(db_path),
            ttl_seconds=86400 * 30,
        )
        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            context_store=store,
            feature_service=fs,
            priors="none",
        )

        x = np.random.default_rng(0).standard_normal(16)
        x[-1] = 1.0
        model_id, log = router.route(x)
        router.process_feedback(log.request_id, reward=0.9)

        stats = store.stats()
        assert "total_contexts" in stats

        pruned = store.prune()
        assert pruned >= 0

    def test_ephemeral_context_store(self) -> None:
        """API Ref: EphemeralContextStore example."""
        from pareto_bandit import BanditRouter
        from pareto_bandit.storage import EphemeralContextStore
        from pareto_bandit import FeatureService

        store = EphemeralContextStore(max_size=100)
        fs = FeatureService.for_precomputed(dimension=16)
        router = BanditRouter.create(
            TWO_MODEL_REGISTRY,
            context_store=store,
            feature_service=fs,
            priors="none",
        )
        x = np.random.default_rng(0).standard_normal(16)
        x[-1] = 1.0
        model_id, _ = router.route(x)
        assert model_id in TWO_MODEL_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
# 21. Utility Function Examples
# ═══════════════════════════════════════════════════════════════════════════


class TestUtilityFunctionExamples:

    def test_infer_model_family(self) -> None:
        """API Ref: infer_model_family examples."""
        from pareto_bandit import infer_model_family

        f1 = infer_model_family("openai/gpt-4o")
        assert isinstance(f1, str)
        assert f1 == "openai/gpt-4o"

        f2 = infer_model_family("anthropic/claude-3.5-sonnet")
        assert isinstance(f2, str)
        assert f2 == "anthropic/claude-3"

    def test_compute_correlation_families(self) -> None:
        """API Ref: compute_correlation_families (using shipped model IDs)."""
        from pareto_bandit import compute_correlation_families

        rng = np.random.default_rng(42)
        n = 200
        reward_vectors = {
            "meta-llama/llama-3.1-8b-instruct": (
                rng.uniform(size=n) > 0.3
            ).astype(float),
            "mistralai/mistral-large-2512": (
                rng.uniform(size=n) > 0.2
            ).astype(float),
            "google/gemini-2.5-pro": (
                rng.uniform(size=n) > 0.1
            ).astype(float),
        }
        families = compute_correlation_families(
            reward_vectors, threshold=0.6, method="tetrachoric"
        )
        assert isinstance(families, dict)
        assert len(families) == 3
        for mid in reward_vectors:
            assert mid in families


# ═══════════════════════════════════════════════════════════════════════════
# 22. Demo Module Examples (uses shipped holdout data)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestDemoModuleExamples:

    def test_demo_config_dataclass(self) -> None:
        """API Ref: DemoConfig construction."""
        from pareto_bandit.demo import DemoConfig

        cfg = DemoConfig(
            n_seeds=10,
            alpha=0.05,
            cost_penalty=0.5,
            scenario=2,
        )
        assert cfg.n_seeds == 10
        assert cfg.scenario == 2

    def test_load_demo_splits(self) -> None:
        """API Ref: load_demo_splits with shipped val + holdout."""
        from pareto_bandit.demo import load_demo_splits, DemoConfig
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        cfg = DemoConfig()
        train, holdout = load_demo_splits(
            val_file=cfg.val_file,
            holdout_file=cfg.holdout_file,
            feature_service=fs,
        )
        assert train.n > 0
        assert holdout.n > 0
        assert train.embeddings.shape[1] == 26

    def test_run_trial(self) -> None:
        """API Ref: run_trial with shipped val + holdout."""
        from pareto_bandit.demo import load_demo_splits, run_trial, DemoConfig
        from pareto_bandit.feature_service import FeatureService

        fs = FeatureService()
        cfg = DemoConfig()
        train, holdout = load_demo_splits(
            val_file=cfg.val_file,
            holdout_file=cfg.holdout_file,
            feature_service=fs,
        )

        trial = run_trial(
            train, holdout, alpha=0.05, cost_penalty=0.0, seed=7
        )
        assert 0.0 <= trial.mean_reward <= 1.0
        assert trial.mean_cost >= 0.0
        assert isinstance(trial.model_fractions, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 23. End-to-End: Full Pipeline with Shipped Data
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestEndToEndWithShippedData:
    """Complete pipeline exercising shipped artifacts: PCA, priors, holdout."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from pareto_bandit import (
            BanditRouter,
            FeatureService,
            train_pca,
            generate_warmup_priors,
        )
        from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
        from pareto_bandit.storage import SqliteContextStore

        records = _load_holdout_records(n=60)
        prompts = [r["prompt"] for r in records]

        pca_path = tmp_path / "pipeline_pca.joblib"
        pca = train_pca(
            prompts,
            encoder_model="all-MiniLM-L6-v2",
            n_components=25,
            output_path=str(pca_path),
        )

        rewards_data = [
            {
                "prompt": r["prompt"],
                "rewards": {
                    arm: info["reward"] for arm, info in r["arms"].items()
                },
            }
            for r in records
        ]
        priors_path = tmp_path / "pipeline_priors.joblib"
        generate_warmup_priors(
            rewards_data,
            encoder_model="all-MiniLM-L6-v2",
            pca=str(pca_path),
            output_path=str(priors_path),
        )

        fs = FeatureService(pca_path=str(pca_path))
        pacer = BudgetPacer(
            target_avg_spend_usd=0.001, mode=PacingMode.ADAPTIVE
        )
        store = SqliteContextStore(
            db_path=str(tmp_path / "ctx.db"), ttl_seconds=3600
        )

        router = BanditRouter.create(
            model_registry=K3_REGISTRY,
            feature_service=fs,
            priors=str(priors_path),
            budget_pacer=pacer,
            context_store=store,
        )

        for rec in records[:20]:
            model_id, log = router.route(rec["prompt"])
            reward = rec["arms"].get(model_id, {}).get("reward", 0.5)
            router.process_feedback(log.request_id, reward=reward)

        probs = router.get_probabilities(records[0]["prompt"])
        assert abs(sum(probs.values()) - 1.0) < 1e-6

        state_path = tmp_path / "checkpoint.npz"
        router.save_state(str(state_path))
        assert state_path.exists()

        router.load_state(str(state_path))
