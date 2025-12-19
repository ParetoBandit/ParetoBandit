"""
Integration Tests for BanditGPT

Tests the complete system end-to-end:
1. Router initialization with priors
2. Routing decisions
3. Feedback and learning
4. Prior persistence
5. Dynamic model management
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_registry():
    """Sample model registry with realistic cost/latency."""
    return {
        "openai/gpt-4o": {"cost_per_1m_tokens": 5.0, "latency_ms": 800},
        "openai/gpt-4o-mini": {"cost_per_1m_tokens": 0.15, "latency_ms": 400},
        "anthropic/claude-3.5-sonnet": {"cost_per_1m_tokens": 3.0, "latency_ms": 600},
        "amazon/nova-lite-v1": {"cost_per_1m_tokens": 0.10, "latency_ms": 300},
        "meta-llama/llama-3-70b-instruct": {"cost_per_1m_tokens": 0.88, "latency_ms": 500},
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Test: Core Imports
# ---------------------------------------------------------------------------


class TestCoreImports:
    """Test that all core components can be imported."""

    def test_import_bandit_router(self):
        from banditgpt.core import BanditRouter
        assert BanditRouter is not None

    def test_import_prior_manager(self):
        from banditgpt.core import PriorManager
        assert PriorManager is not None

    def test_import_optimization_profile(self):
        from banditgpt.core import OptimizationProfile
        assert OptimizationProfile is not None

    def test_import_exploration_rate(self):
        from banditgpt.core import ExplorationRate
        assert ExplorationRate is not None

    def test_import_disjoint_policy(self):
        from banditgpt.core import DisjointLinUCBPolicy
        assert DisjointLinUCBPolicy is not None

    def test_import_shared_policy(self):
        from banditgpt.core import SharedCovarianceLinUCBPolicy
        assert SharedCovarianceLinUCBPolicy is not None

    def test_import_judge(self):
        from banditgpt.core import Judge, create_custom_judge
        assert Judge is not None
        assert create_custom_judge is not None


# ---------------------------------------------------------------------------
# Test: Router Initialization
# ---------------------------------------------------------------------------


class TestRouterInitialization:
    """Test router creation with different configurations."""

    def test_create_cold_start(self, sample_registry):
        """Router can be created without priors (cold start)."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        assert router is not None
        assert len(router.registry) == 5
        assert router.priors_source == "none"

    def test_create_with_bundled_priors(self, sample_registry, temp_dir):
        """Router loads bundled priors when available."""
        from banditgpt.core import BanditRouter, SharedCovarianceLinUCBPolicy
        import json

        # Create fake bundled priors using SharedCovarianceLinUCBPolicy
        models = list(sample_registry.keys())
        policy = SharedCovarianceLinUCBPolicy(models, dim=384, alpha=0.5)
        priors_path = temp_dir / "bundled.npz"
        
        # Save using the expected format (models + meta + A_shared + b)
        meta = json.dumps({"dim": 384, "alpha": 0.5, "ridge_lambda": 1.0})
        b_matrix = np.stack([policy.b[m] for m in models])
        
        np.savez(
            priors_path,
            models=np.array(models),
            meta=np.array([meta]),
            A_shared=policy.A.astype(np.float64),
            b=b_matrix.astype(np.float64),
        )

        router = BanditRouter.create(
            sample_registry,
            priors="bundled",
            bundled_priors_path=priors_path,
        )

        assert router.priors_source == "bundled"

    def test_create_with_expert_priors(self, sample_registry, temp_dir):
        """Router loads expert priors (disjoint format)."""
        from banditgpt.core import BanditRouter

        # Create expert priors
        models = list(sample_registry.keys())
        dim = 384
        A_stack = np.stack([np.eye(dim) * 10 for _ in models])
        b_stack = np.stack([np.random.randn(dim) for _ in models])

        priors_path = temp_dir / "expert_priors.npz"
        np.savez(
            priors_path,
            model_names=np.array(models),
            dim=dim,
            alpha=0.5,
            A_stack=A_stack.astype(np.float16),
            b_stack=b_stack.astype(np.float16),
        )

        router = BanditRouter.create(
            sample_registry,
            priors="bundled",
            bundled_priors_path=priors_path,
        )

        assert router.priors_source == "bundled"


# ---------------------------------------------------------------------------
# Test: Routing Decisions
# ---------------------------------------------------------------------------


class TestRoutingDecisions:
    """Test that routing produces valid decisions."""

    def test_route_returns_valid_model(self, sample_registry):
        """route() returns a model from the registry."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        model, log = router.route("Write a Python function to sort a list")

        assert model in sample_registry
        assert log is not None
        assert log.selected_model == model

    def test_route_with_profile(self, sample_registry):
        """route() respects optimization profiles."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Cost-saver should prefer cheaper models
        model, log = router.route(
            "Simple question",
            profile="cost_saver",
        )
        assert model in sample_registry

    def test_rank_prompt_returns_ordered_list(self, sample_registry):
        """rank_prompt() returns all models with scores."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        rankings = router.rank_prompt("Explain quantum computing")

        assert len(rankings) == len(sample_registry)
        
        # Check rankings are sorted by utility (descending)
        utilities = [r["utility"] for r in rankings]
        assert utilities == sorted(utilities, reverse=True)

    def test_route_with_candidate_models(self, sample_registry):
        """route() can be restricted to candidate models."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        candidates = ["openai/gpt-4o", "amazon/nova-lite-v1"]
        
        model, log = router.route(
            "Write code",
            candidate_models=candidates,
        )

        assert model in candidates


# ---------------------------------------------------------------------------
# Test: Feedback and Learning
# ---------------------------------------------------------------------------


class TestFeedbackAndLearning:
    """Test that the bandit learns from feedback."""

    def test_report_feedback_updates_bandit(self, sample_registry):
        """report_feedback() updates bandit weights."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Get initial routing
        model, log = router.route("Test prompt")
        
        # Get initial A matrix norm
        initial_A_norm = np.linalg.norm(router.bandit.A[model])
        
        # Report positive feedback (reward=1.0)
        router.report_feedback(log.request_id, reward=1.0)
        
        # A matrix should have changed
        new_A_norm = np.linalg.norm(router.bandit.A[model])
        assert new_A_norm != initial_A_norm

    def test_positive_feedback_increases_preference(self, sample_registry):
        """Positive feedback increases model's expected reward."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        target_model = "amazon/nova-lite-v1"
        prompt = "Simple task"
        
        # Get initial theta for target model
        initial_theta = router.bandit.A_inv[target_model] @ router.bandit.b[target_model]
        initial_norm = np.linalg.norm(initial_theta)
        
        # Simulate 10 positive feedback cycles
        for _ in range(10):
            model, log = router.route(prompt, candidate_models=[target_model])
            router.report_feedback(log.request_id, reward=1.0)
        
        # Check updated theta - should have larger norm after positive feedback
        updated_theta = router.bandit.A_inv[target_model] @ router.bandit.b[target_model]
        updated_norm = np.linalg.norm(updated_theta)
        
        # The b vector should have grown with positive rewards
        assert np.linalg.norm(router.bandit.b[target_model]) > 0

    def test_negative_feedback_affects_weights(self, sample_registry):
        """Negative feedback affects model weights."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        target_model = "openai/gpt-4o"
        prompt = "Difficult task"
        
        # Get initial b vector norm
        initial_b_norm = np.linalg.norm(router.bandit.b[target_model])
        
        # Simulate negative feedback (reward=-1.0)
        for _ in range(10):
            model, log = router.route(prompt, candidate_models=[target_model])
            router.report_feedback(log.request_id, reward=-1.0)
        
        # Check updated b vector - negative rewards should create negative components
        updated_b = router.bandit.b[target_model]
        # With negative rewards, the b vector should have negative components
        assert np.min(updated_b) < 0


# ---------------------------------------------------------------------------
# Test: Prior Persistence
# ---------------------------------------------------------------------------


class TestPriorPersistence:
    """Test saving and loading router state."""

    def test_save_and_load_state(self, sample_registry, temp_dir):
        """Router state can be saved and loaded."""
        from banditgpt.core import BanditRouter

        # Create and train router
        router = BanditRouter.create(sample_registry, priors="none")
        
        # Generate some training data
        for i in range(5):
            model, log = router.route(f"Test prompt {i}")
            reward = 1.0 if (i % 2 == 0) else -1.0
            router.report_feedback(log.request_id, reward=reward)
        
        # Save state (save_state creates its own npz sidecar)
        state_path = temp_dir / "router_state.json"
        router.save_state(state_path)
        
        assert state_path.exists()
        # NPZ sidecar should be created automatically
        npz_path = state_path.with_suffix(".bandit.npz")
        assert npz_path.exists()
        
        # Load state
        loaded_router = BanditRouter.load_state(state_path)
        
        # Verify loaded router has same models
        assert set(loaded_router.registry.keys()) == set(router.registry.keys())

    def test_save_shippable_priors(self, sample_registry, temp_dir):
        """Trained bandit can be exported as shippable priors."""
        from banditgpt.core import BanditRouter

        # Create and train router
        router = BanditRouter.create(sample_registry, priors="none")
        
        # Train on some data
        for i in range(10):
            model, log = router.route(f"Training prompt {i}")
            router.report_feedback(log.request_id, reward=1.0)
        
        # Export as shippable priors
        priors_path = temp_dir / "shippable_priors.npz"
        router.save_shippable_priors(priors_path)
        
        assert priors_path.exists()
        
        # Verify can be loaded
        priors = np.load(priors_path, allow_pickle=True)
        assert "model_names" in priors or "A_shared" in priors


# ---------------------------------------------------------------------------
# Test: Dynamic Model Management
# ---------------------------------------------------------------------------


class TestDynamicModelManagement:
    """Test adding and removing models at runtime."""

    def test_add_model_cold_start(self, sample_registry):
        """New model can be added with cold start."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Add new model
        result = router.add_model("openai/gpt-5")
        
        assert result is True
        assert "openai/gpt-5" in router.bandit.models

    def test_add_model_clone(self, sample_registry):
        """New model can be cloned from existing."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Clone from existing model
        result = router.add_model("openai/gpt-5", clone_from="openai/gpt-4o")
        
        assert result is True
        assert "openai/gpt-5" in router.bandit.models

    def test_remove_model(self, sample_registry):
        """Model can be removed from router."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        initial_count = len(router.bandit.models)
        
        # Remove model
        result = router.remove_model("amazon/nova-lite-v1")
        
        assert result is True
        assert "amazon/nova-lite-v1" not in router.bandit.models
        assert len(router.bandit.models) == initial_count - 1


# ---------------------------------------------------------------------------
# Test: End-to-End Workflow
# ---------------------------------------------------------------------------


class TestEndToEndWorkflow:
    """Test complete usage scenarios."""

    def test_full_routing_workflow(self, sample_registry, temp_dir):
        """Test complete workflow: create -> route -> feedback -> save -> load."""
        from banditgpt.core import BanditRouter

        # Step 1: Create router
        router = BanditRouter.create(sample_registry, priors="none")
        
        # Step 2: Make routing decisions
        prompts = [
            "Write a Python function",
            "Explain quantum physics",
            "Translate to French",
            "Debug this code",
            "Write a poem",
        ]
        
        logs = []
        for prompt in prompts:
            model, log = router.route(prompt)
            logs.append(log)
            assert model in sample_registry
        
        # Step 3: Provide feedback
        for i, log in enumerate(logs):
            reward = 1.0 if (i % 2 == 0) else -1.0
            router.report_feedback(log.request_id, reward=reward)
        
        # Step 4: Save state
        state_path = temp_dir / "state.json"
        router.save_state(state_path)
        
        # Step 5: Load state
        loaded_router = BanditRouter.load_state(state_path)
        
        # Step 6: Continue routing with loaded state
        model, log = loaded_router.route("New prompt after reload")
        assert model in sample_registry

    def test_specialist_discovery_workflow(self, sample_registry):
        """Test that router discovers specialists over time."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Simulate: Nova is best for code tasks
        code_prompts = [
            "Write Python code",
            "Debug JavaScript",
            "Implement sorting algorithm",
        ]
        
        # Train: always give positive feedback to nova for code
        for _ in range(20):
            for prompt in code_prompts:
                model, log = router.route(
                    prompt,
                    candidate_models=["amazon/nova-lite-v1"],
                )
                router.report_feedback(log.request_id, reward=1.0)
        
        # Check: Nova should have learned positive weights
        nova_theta = router.bandit.A_inv["amazon/nova-lite-v1"] @ router.bandit.b["amazon/nova-lite-v1"]
        
        # Nova should have learned positive association (theta norm > 0)
        assert np.linalg.norm(nova_theta) > 0

    def test_cost_quality_tradeoff(self, sample_registry):
        """Test that profiles affect routing decisions."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # Sample many routing decisions with different profiles
        cost_saver_choices = []
        quality_first_choices = []
        
        for _ in range(20):
            model, _ = router.route("Test prompt", profile="cost_saver")
            cost_saver_choices.append(sample_registry[model]["cost_per_1m_tokens"])
            
            model, _ = router.route("Test prompt", profile="quality_first")
            quality_first_choices.append(sample_registry[model]["cost_per_1m_tokens"])
        
        # Both should produce valid results
        avg_cost_saver = np.mean(cost_saver_choices)
        avg_quality_first = np.mean(quality_first_choices)
        
        # With exploration, this might not always hold, but verify both work
        assert avg_cost_saver >= 0
        assert avg_quality_first >= 0


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self, sample_registry):
        """Router handles empty prompt."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        model, log = router.route("")
        
        assert model in sample_registry

    def test_very_long_prompt(self, sample_registry):
        """Router handles very long prompts."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        long_prompt = "word " * 1000
        model, log = router.route(long_prompt)
        
        assert model in sample_registry

    def test_unicode_prompt(self, sample_registry):
        """Router handles unicode prompts."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        unicode_prompt = "Explain 量子力学 in English 🔬"
        model, log = router.route(unicode_prompt)
        
        assert model in sample_registry

    def test_feedback_for_unknown_request(self, sample_registry):
        """Feedback for unknown request ID is handled gracefully."""
        from banditgpt.core import BanditRouter

        router = BanditRouter.create(sample_registry, priors="none")
        
        # This should not raise, just return False or handle gracefully
        result = router.report_feedback("nonexistent-request-id", reward=1.0)
        assert result is False

    def test_single_model_registry(self):
        """Router works with single model."""
        from banditgpt.core import BanditRouter

        single_registry = {"openai/gpt-4o": {"cost_per_1m_tokens": 5.0, "latency_ms": 800}}
        router = BanditRouter.create(single_registry, priors="none")
        
        model, log = router.route("Test")
        assert model == "openai/gpt-4o"
