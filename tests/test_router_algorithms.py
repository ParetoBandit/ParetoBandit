"""
Comprehensive unit tests for core algorithms in BanditRouter.

This test suite covers:
1. DisjointLinUCBPolicy - Core bandit algorithm
2. Feature extraction and normalization
3. Cost/latency penalty calculations
4. Pareto frontier filtering
5. Exploration-exploitation tradeoffs
6. Model admission and uniform prior initialization
7. CorrallingRouter - Expert mixing
8. Numerical stability and regularization
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
from collections import defaultdict
from bandit_gpt.router import (
    DisjointLinUCBPolicy,
    BanditRouter,
    RouterConfig,
    CorrallingRouter,
    CostAwareLinUCBAdapter,
    CostAwareTabulaRasaRouter,
    NoEligibleModelsError,
    calibrate_priors,
    l2_normalize,
    estimate_tokens_rough,
)


# =============================================================================
# DisjointLinUCBPolicy Tests - Core Bandit Algorithm
# =============================================================================

class TestDisjointLinUCBPolicy:
    """Test suite for the core LinUCB bandit algorithm."""
    
    def test_initialization(self):
        """Test basic initialization of LinUCB policy."""
        models = ["model_a", "model_b", "model_c"]
        dim = 10
        alpha = 0.5
        init_lambda = 1.0
        
        policy = DisjointLinUCBPolicy(
            models, 
            dim=dim, 
            alpha=alpha,
            init_lambda=init_lambda
        )
        
        assert policy.models == models
        assert policy.dim == dim
        assert policy.alpha == alpha
        assert policy.init_lambda == init_lambda
        
        # Check initialization: A = λI, b = 0
        for model in models:
            assert model in policy.A
            assert model in policy.b
            assert model in policy.A_inv
            
            # A should be identity scaled by lambda
            expected_A = np.eye(dim) * init_lambda
            np.testing.assert_array_almost_equal(policy.A[model], expected_A)
            
            # b should be zero
            np.testing.assert_array_almost_equal(policy.b[model], np.zeros(dim))
            
            # A_inv should be inverse of A
            expected_A_inv = np.eye(dim) / init_lambda
            np.testing.assert_array_almost_equal(policy.A_inv[model], expected_A_inv)
    
    def test_select_arm_basic(self):
        """Test arm selection with simple context."""
        models = ["model_a", "model_b"]
        dim = 5
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=1.0)
        
        # Create a simple context vector
        context = np.array([1.0, 0.5, 0.0, 0.0, 1.0])
        
        # Select arm (should not crash)
        selected_model, ucb_score = policy.select_arm(context)
        
        assert selected_model in models
        assert isinstance(ucb_score, float)
        assert not np.isnan(ucb_score)
        assert not np.isinf(ucb_score)
    
    def test_update_basic(self):
        """Test basic update operation."""
        models = ["model_a"]
        dim = 3
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.1, init_lambda=1.0)
        
        model = "model_a"
        context = np.array([1.0, 0.0, 1.0])
        reward = 0.8
        
        # Store initial state
        A_before = policy.A[model].copy()
        b_before = policy.b[model].copy()
        
        # Update
        policy.update(model, context, reward)
        
        # A should have increased (added x x^T)
        A_after = policy.A[model]
        assert not np.allclose(A_after, A_before)
        
        # b should have increased (added reward * x)
        b_after = policy.b[model]
        assert not np.allclose(b_after, b_before)
        
        # Verify mathematical correctness: A += x x^T
        expected_A = A_before + np.outer(context, context)
        np.testing.assert_array_almost_equal(A_after, expected_A)
        
        # Verify b += reward * x
        expected_b = b_before + reward * context
        np.testing.assert_array_almost_equal(b_after, expected_b)
    
    def test_update_with_weight(self):
        """Test weighted update (importance sampling)."""
        models = ["model_a"]
        dim = 3
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.1, init_lambda=1.0)
        
        model = "model_a"
        context = np.array([1.0, 0.0, 1.0])
        reward = 0.8
        weight = 0.5  # Half weight
        
        b_before = policy.b[model].copy()
        
        # Update with weight
        policy.update(model, context, reward, weight=weight)
        
        b_after = policy.b[model]
        
        # Verify b += weight * reward * x
        expected_b = b_before + weight * reward * context
        np.testing.assert_array_almost_equal(b_after, expected_b)
    
    def test_exploration_bonus(self):
        """Test that exploration bonus (alpha * std) increases with uncertainty."""
        models = ["model_a", "model_b"]
        dim = 5
        
        # High alpha = more exploration
        policy_high_alpha = DisjointLinUCBPolicy(models, dim=dim, alpha=2.0)
        
        # Low alpha = less exploration
        policy_low_alpha = DisjointLinUCBPolicy(models, dim=dim, alpha=0.1)
        
        context = np.ones(dim)
        
        # Get UCB scores
        _, ucb_high = policy_high_alpha.select_arm(context)
        _, ucb_low = policy_low_alpha.select_arm(context)
        
        # With same initialization, high alpha should give higher UCB
        # (more optimistic due to larger exploration bonus)
        assert ucb_high > ucb_low
    
    def test_forgetting_factor(self):
        """Test exponential forgetting (temporal decay)."""
        models = ["model_a"]
        dim = 3
        gamma = 0.9  # 10% decay per step
        
        policy = DisjointLinUCBPolicy(
            models, 
            dim=dim, 
            alpha=0.1,
            forgetting_factor=gamma
        )
        
        model = "model_a"
        context = np.array([1.0, 0.0, 1.0])
        
        # First update
        policy.update(model, context, reward=1.0)
        A_after_first = policy.A[model].copy()
        
        # Advance time without updating this model
        policy.t += 10
        
        # Second update (should apply decay)
        policy.update(model, context, reward=1.0)
        A_after_second = policy.A[model]
        
        # With forgetting, A should have decayed before the second update
        # So A_after_second < 2 * A_after_first (if no decay, it would be ~2x)
        assert np.linalg.norm(A_after_second) < 2 * np.linalg.norm(A_after_first)
    
    def test_add_arm_dynamically(self):
        """Test adding new arms after initialization."""
        models = ["model_a", "model_b"]
        dim = 5
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.5)
        
        # Add new arm
        new_model = "model_c"
        policy.add_arm(new_model)
        
        assert new_model in policy.models
        assert new_model in policy.A
        assert new_model in policy.b
        assert new_model in policy.A_inv
        
        # Should be initialized with identity
        expected_A = np.eye(dim) * policy.init_lambda
        np.testing.assert_array_almost_equal(policy.A[new_model], expected_A)
    
    def test_delete_arm(self):
        """Test removing arms."""
        models = ["model_a", "model_b", "model_c"]
        dim = 5
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.5)
        
        # Delete arm
        policy.delete_arm("model_b")
        
        assert "model_b" not in policy.models
        assert "model_b" not in policy.A
        assert "model_b" not in policy.b
        assert "model_b" not in policy.A_inv
        
        # Other models should remain
        assert "model_a" in policy.models
        assert "model_c" in policy.models
    
    def test_save_load_state(self, tmp_path):
        """Test saving and loading bandit state."""
        models = ["model_a", "model_b"]
        dim = 5
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.5)
        
        # Run some updates
        context = np.ones(dim)
        policy.update("model_a", context, reward=0.8)
        policy.update("model_b", context, reward=0.6)
        
        # Save state
        save_path = tmp_path / "policy_state.npz"
        policy.save_state(save_path)
        
        assert save_path.exists()
        
        # Create new policy and load
        policy2 = DisjointLinUCBPolicy(models, dim=dim, alpha=0.5)
        policy2.load_state(save_path)
        
        # Check that state matches
        for model in models:
            np.testing.assert_array_almost_equal(policy.A[model], policy2.A[model])
            np.testing.assert_array_almost_equal(policy.b[model], policy2.b[model])
    
    def test_dimension_mismatch_detection(self, tmp_path):
        """Test that loading state with wrong dimension raises clear error."""
        models = ["model_a"]
        dim_original = 5
        dim_new = 10
        
        # Create and save with dim=5
        policy1 = DisjointLinUCBPolicy(models, dim=dim_original, alpha=0.5)
        save_path = tmp_path / "policy_state.npz"
        policy1.save_state(save_path)
        
        # Try to load with dim=10
        policy2 = DisjointLinUCBPolicy(models, dim=dim_new, alpha=0.5)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            policy2.load_state(save_path)


# =============================================================================
# Feature Extraction and Normalization Tests
# =============================================================================

class TestFeatureExtraction:
    """Test feature extraction and normalization utilities."""
    
    def test_l2_normalize(self):
        """Test L2 normalization."""
        # Test basic normalization
        x = np.array([3.0, 4.0])
        normalized = l2_normalize(x)
        
        # Should have unit norm
        assert np.abs(np.linalg.norm(normalized) - 1.0) < 1e-6
        
        # Should preserve direction
        assert np.allclose(normalized, np.array([0.6, 0.8]))
    
    def test_l2_normalize_zero_vector(self):
        """Test L2 normalization with zero vector."""
        x = np.array([0.0, 0.0, 0.0])
        normalized = l2_normalize(x)
        
        # Should return zero vector (not crash)
        np.testing.assert_array_almost_equal(normalized, x)
    
    def test_estimate_tokens_rough(self):
        """Test rough token estimation."""
        # Empty string
        assert estimate_tokens_rough("") == 0
        
        # Simple text (roughly 1.3 tokens per word)
        text = "Hello world this is a test"
        tokens = estimate_tokens_rough(text)
        word_count = len(text.split())
        
        # Should be approximately 1.3x word count (rounded)
        expected = max(0, round(word_count * 1.3))
        assert tokens == expected
        
        # Longer text
        long_text = " ".join(["word"] * 100)
        tokens_long = estimate_tokens_rough(long_text)
        assert tokens_long == max(0, round(100 * 1.3))  # 100 * 1.3 rounded


# =============================================================================
# Cost and Latency Penalty Tests
# =============================================================================

class TestCostLatencyPenalties:
    """Test cost and latency penalty calculations."""
    
    @pytest.fixture
    def sample_registry(self):
        """Create sample registry with cost/latency data."""
        return {
            "cheap_fast": {
                "model_id": "provider/cheap-fast",
                "input_cost_per_m": 0.1,
                "output_cost_per_m": 0.3,
                "time_to_first_token_seconds": 0.1,
                "hle": 0.5
            },
            "expensive_slow": {
                "model_id": "provider/expensive-slow",
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 30.0,
                "time_to_first_token_seconds": 3.0,
                "hle": 0.9
            },
            "balanced": {
                "model_id": "provider/balanced",
                "input_cost_per_m": 2.0,
                "output_cost_per_m": 6.0,
                "time_to_first_token_seconds": 0.5,
                "hle": 0.7
            }
        }
    
    def test_cost_penalty_calculation(self, sample_registry):
        """Test absolute cost penalty calculation."""
        router = BanditRouter.create(model_registry=sample_registry, priors="none")
        
        # Test cheap model (should have low penalty)
        cost_cheap = router._estimate_cost("cheap_fast", in_tok=1000, out_tok=500)
        total_tokens = 1500
        cost_per_1k_cheap = (cost_cheap / total_tokens) * 1000
        penalty_cheap = router._calculate_absolute_penalty(cost_per_1k_cheap)
        
        # Test expensive model (should have high penalty)
        cost_expensive = router._estimate_cost("expensive_slow", in_tok=1000, out_tok=500)
        cost_per_1k_expensive = (cost_expensive / total_tokens) * 1000
        penalty_expensive = router._calculate_absolute_penalty(cost_per_1k_expensive)
        
        # Expensive should have higher penalty (or equal if both at ceiling)
        assert penalty_expensive >= penalty_cheap
        
        # Penalties should be in [0, 1]
        assert 0 <= penalty_cheap <= 1
        assert 0 <= penalty_expensive <= 1
    
    def test_latency_estimation(self, sample_registry):
        """Test latency estimation."""
        router = BanditRouter.create(model_registry=sample_registry, priors="none")
        
        # Fast model
        latency_fast = router._estimate_latency("cheap_fast", out_tok=500)
        assert latency_fast == 0.1
        
        # Slow model
        latency_slow = router._estimate_latency("expensive_slow", out_tok=500)
        assert latency_slow == 3.0
        
        # Balanced
        latency_balanced = router._estimate_latency("balanced", out_tok=500)
        assert latency_balanced == 0.5


# =============================================================================
# Model Admission Tests (Uniform Prior Initialization)
# =============================================================================

class TestModelAdmission:
    """Test that register_model() initialises new arms with a uniform prior.

    New models start with A = λI and b = λ·θ (T-shirt prior), relying on
    Hybrid LinUCB's family-level β_F sharing for continuous knowledge transfer.
    """

    @pytest.fixture
    def admission_registry(self):
        """Two-model registry for dynamic admission tests."""
        return {
            "gpt-4": {
                "model_id": "openai/gpt-4",
                "display_name": "GPT-4",
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 15.0,
                "hle": 0.85,
                "capabilities": ["reasoning", "coding"],
                "speed_profile": "slow"
            },
            "claude-opus": {
                "model_id": "anthropic/claude-opus",
                "display_name": "Claude Opus",
                "input_cost_per_m": 6.0,
                "output_cost_per_m": 18.0,
                "hle": 0.88,
                "capabilities": ["reasoning", "creative"],
                "speed_profile": "slow"
            }
        }

    def test_register_model_uses_identity_precision(self, admission_registry):
        """New model's A matrix should be λI (maximum uncertainty)."""
        router = BanditRouter.create(model_registry=admission_registry, priors="none")

        router.register_model(
            "gpt-4-turbo",
            speed="fast",
            capabilities=["reasoning", "coding"],
        )

        A_new = router.bandit.A["gpt-4-turbo"]
        lam = router.bandit.init_lambda

        off_diag = A_new - np.diag(np.diag(A_new))
        assert np.allclose(off_diag, 0, atol=1e-10), "A should be diagonal"
        assert np.allclose(np.diag(A_new), lam, rtol=1e-6), \
            f"Diagonal should equal init_lambda={lam}"

    def test_register_model_applies_tshirt_prior(self, admission_registry):
        """New model's b vector should encode the T-shirt prior, not zeros."""
        router = BanditRouter.create(model_registry=admission_registry, priors="none")

        router.register_model(
            "gpt-4-turbo",
            speed="slow",
            capabilities=["reasoning"],
        )

        b_new = router.bandit.b["gpt-4-turbo"]
        assert np.linalg.norm(b_new) > 1e-6, \
            "b should be non-zero (T-shirt prior applied)"

    def test_register_model_no_neighbor_theta_leakage(self, admission_registry):
        """After training model A, registering model B must NOT inherit A's θ."""
        router = BanditRouter.create(model_registry=admission_registry, priors="none")

        for i in range(100):
            ctx = np.random.randn(router.bandit.dim)
            ctx = ctx / np.linalg.norm(ctx)
            router.bandit.update("gpt-4", ctx, reward=0.5 + 0.4 * np.tanh(ctx[0]))

        theta_gpt4 = router.bandit.A_inv["gpt-4"] @ router.bandit.b["gpt-4"]
        assert np.linalg.norm(theta_gpt4) > 0.01, "Precondition: GPT-4 has learned"

        router.register_model(
            "gpt-4-turbo",
            speed="fast",
            capabilities=["reasoning", "coding"],
        )

        theta_new = router.bandit.A_inv["gpt-4-turbo"] @ router.bandit.b["gpt-4-turbo"]

        if np.linalg.norm(theta_new) > 1e-6 and np.linalg.norm(theta_gpt4) > 1e-6:
            cosine = np.dot(theta_gpt4, theta_new) / (
                np.linalg.norm(theta_gpt4) * np.linalg.norm(theta_new)
            )
            assert cosine < 0.95, (
                f"New model's θ should NOT mirror a trained neighbor's θ "
                f"(cosine={cosine:.3f})"
            )


# =============================================================================
# Corralling Router Tests
# =============================================================================

class TestCorrallingRouter:
    """Test Corralling algorithm for expert mixing."""
    
    @pytest.fixture
    def simple_experts(self):
        """Create simple mock experts for testing."""
        class MockExpert:
            def __init__(self, name, bias=0.0):
                self.name = name
                self.bias = bias
                self.updates = []
            
            def select_model(self, context, total_steps=0, **kwargs):
                # Simple logic: return "model_a" if context sum > 0.5 + bias
                if np.sum(context) > 0.5 + self.bias:
                    return "model_a"
                return "model_b"
            
            def update(self, context, model, reward, weight=1.0, advance_time=True):
                self.updates.append((context, model, reward))
        
        expert1 = MockExpert("optimistic", bias=-0.2)  # Favors model_a
        expert2 = MockExpert("pessimistic", bias=0.2)  # Favors model_b
        
        return [expert1, expert2]
    
    def test_corralling_initialization(self, simple_experts):
        """Test Corralling initialization."""
        models = ["model_a", "model_b"]
        router = CorrallingRouter(
            experts=simple_experts,
            models=models,
            learning_rate=0.1,
            gamma=0.05
        )
        
        # Should start with uniform weights
        assert len(router.weights) == 2
        np.testing.assert_array_almost_equal(router.weights, np.array([0.5, 0.5]))
        
        # sum_squared_losses initialized to zero (no history yet)
        np.testing.assert_array_almost_equal(router.sum_squared_losses, np.zeros(2))
    
    def test_corralling_action_distribution(self, simple_experts):
        """Test that action distribution follows marginal π(a) when experts disagree."""
        models = ["model_a", "model_b"]
        router = CorrallingRouter(
            experts=simple_experts,
            models=models,
            learning_rate=0.1,
            gamma=0.05
        )
        
        # Context where experts disagree: sum=0.6
        # optimistic (bias=-0.2): 0.6 > 0.3 → model_a
        # pessimistic (bias=0.2): 0.6 > 0.7 → model_b
        context = np.array([0.1, 0.2, 0.3])
        n_trials = 1000
        action_counts = {"model_a": 0, "model_b": 0}
        
        for _ in range(n_trials):
            model, token = router.select_model(context)
            action_counts[model] += 1
        
        ratio = action_counts["model_a"] / n_trials
        assert 0.4 < ratio < 0.6, f"Expected ~50% split, got {ratio:.2%}"
    
    def test_corralling_expert_death_prevention(self, simple_experts):
        """Test that gamma prevents expert death."""
        models = ["model_a", "model_b"]
        router = CorrallingRouter(
            experts=simple_experts,
            models=models,
            learning_rate=1.0,  # Very high learning rate
            gamma=0.1  # 10% minimum probability
        )
        
        context = np.array([0.6, 0.4, 0.5])
        
        # Heavily penalize expert 1 using explicit Corralling selection tokens.
        # Simulate scenario where only expert 1 endorsed the action
        # with marginal probability 0.5.
        for _ in range(100):
            token = {"action_prob": 0.5, "endorsing_experts": [1]}
            router.update(context, "model_b", reward=0.0, selection_token=token)
        
        # Even with terrible performance, expert 1 should maintain minimum probability
        mixed_probs = router._get_mixed_distribution()
        min_prob = router.gamma / router.n_experts
        
        assert mixed_probs[1] >= min_prob, \
            f"Expert 1 probability {mixed_probs[1]:.4f} below minimum {min_prob:.4f}"


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability and regularization."""
    
    def test_sherman_morrison_stability(self):
        """Test Sherman-Morrison update doesn't cause numerical issues."""
        models = ["model_a"]
        dim = 10
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.1, init_lambda=1.0)
        
        model = "model_a"
        
        # Run many updates with same context (stress test)
        context = np.ones(dim)
        for i in range(1000):
            policy.update(model, context, reward=0.5)
            
            # Check that A_inv is still valid
            A_inv = policy.A_inv[model]
            assert not np.any(np.isnan(A_inv)), f"NaN detected at iteration {i}"
            assert not np.any(np.isinf(A_inv)), f"Inf detected at iteration {i}"
            
            # Check that A @ A_inv ≈ I
            identity_check = policy.A[model] @ A_inv
            identity_error = np.linalg.norm(identity_check - np.eye(dim))
            assert identity_error < 1e-6, f"Identity check failed at iteration {i}"
    
    def test_regularization_floor_maintenance(self):
        """Test proactive regularization floor maintenance."""
        models = ["model_a"]
        dim = 5
        
        # Use forgetting factor to trigger regularization maintenance
        policy = DisjointLinUCBPolicy(
            models, 
            dim=dim, 
            alpha=0.1,
            init_lambda=1.0,
            forgetting_factor=0.95  # 5% decay per step
        )
        
        model = "model_a"
        context = np.ones(dim)
        
        # Run updates and let regularization decay
        for _ in range(100):
            policy.update(model, context, reward=0.5)
            policy.t += 10  # Advance time to trigger decay
        
        # Check that regularization floor is maintained
        assert model in policy.regularization_floor
        
        # Matrix should still be well-conditioned
        A = policy.A[model]
        eigenvalues = np.linalg.eigvals(A)
        min_eigenvalue = np.min(eigenvalues)
        
        # Should not have collapsed to zero
        assert min_eigenvalue > 1e-6, \
            f"Regularization floor failed: min eigenvalue = {min_eigenvalue}"
    
    def test_stability_check_trace(self):
        """Test O(d) stability check using trace."""
        models = ["model_a"]
        dim = 5
        config = RouterConfig()
        
        policy = DisjointLinUCBPolicy(models, dim=dim, alpha=0.1, init_lambda=1.0)
        
        model = "model_a"
        
        # Artificially create unstable matrix (very small eigenvalues)
        policy.A[model] = np.eye(dim) * 1e-8
        policy.A_inv[model] = np.eye(dim) * 1e8
        
        # Stability check should detect this
        trace = np.trace(policy.A_inv[model])
        assert trace > config.stability_threshold, \
            "Stability check should detect near-singular matrix"


# =============================================================================
# CostAwareLinUCBAdapter Tests
# =============================================================================

class TestCostAwareLinUCBAdapter:
    """Test cost-aware adapter wrapping a shared DisjointLinUCBPolicy."""

    @pytest.fixture
    def warmup_params(self):
        """Create minimal warmup prior matrices for testing."""
        models = ["model_a", "model_b"]
        dim = 5

        np.random.seed(42)
        A = {m: np.eye(dim) * 10.0 for m in models}
        b = {m: np.random.randn(dim) * 2.0 for m in models}

        return {"models": models, "dim": dim, "A": A, "b": b}

    @pytest.fixture
    def model_costs(self):
        """Create cost metadata."""
        return {
            "model_a": {"normalized_cost": 0.1},
            "model_b": {"normalized_cost": 0.9},
        }

    def _make_adapter(self, warmup_params, model_costs, **overrides):
        """Helper: build a DisjointLinUCBPolicy + CostAwareLinUCBAdapter."""
        models = warmup_params["models"]
        dim = warmup_params["dim"]

        bandit = DisjointLinUCBPolicy(
            models, dim=dim, alpha=0.5, init_lambda=10.0,
        )
        for m in models:
            bandit.b[m] = warmup_params["b"][m].copy()
        bandit.refresh_inverse_cache()

        kwargs = dict(
            bandit=bandit,
            model_costs=model_costs,
            alpha_start=2.0,
            alpha_end=0.1,
            cost_penalty=0.5,
        )
        kwargs.update(overrides)
        return CostAwareLinUCBAdapter(**kwargs), bandit

    def test_cost_aware_initialization(self, warmup_params, model_costs):
        """Test initialization with warmup priors loaded into the bandit."""
        adapter, bandit = self._make_adapter(warmup_params, model_costs)

        for model in warmup_params["models"]:
            assert model in adapter.A
            assert model in adapter.b

            np.testing.assert_array_almost_equal(
                adapter.A[model],
                np.eye(warmup_params["dim"]) * 10.0,
            )
            np.testing.assert_array_almost_equal(
                adapter.b[model],
                warmup_params["b"][model],
            )

    def test_alpha_decay_schedule(self, warmup_params, model_costs):
        """Test linear alpha decay from start to end."""
        adapter, _ = self._make_adapter(
            warmup_params, model_costs, cost_penalty=0.0,
        )

        total_steps = 1000

        alpha_0 = adapter.get_current_alpha(total_steps)
        assert alpha_0 == 2.0

        adapter.t = 500
        alpha_mid = adapter.get_current_alpha(total_steps)
        expected_mid = 2.0 + 0.5 * (0.1 - 2.0)
        assert abs(alpha_mid - expected_mid) < 1e-6

        adapter.t = 1000
        alpha_end = adapter.get_current_alpha(total_steps)
        assert abs(alpha_end - 0.1) < 1e-9, f"Expected 0.1, got {alpha_end}"

    def test_cost_penalty_integration(self, warmup_params, model_costs):
        """Test that cost penalty affects selection."""
        adapter, _ = self._make_adapter(
            warmup_params,
            model_costs,
            alpha_start=0.1,
            alpha_end=0.1,
            cost_penalty=10.0,
        )

        context = np.ones(warmup_params["dim"])

        selected = adapter.select_model(context, total_steps=0)
        assert selected == "model_a", "High cost penalty should favor cheap model"

    def test_prior_calibration(self, model_costs):
        """Test standalone calibrate_priors() for scale explosion."""
        models = ["model_a"]
        dim = 5

        bandit = DisjointLinUCBPolicy(
            models, dim=dim, alpha=0.5, init_lambda=0.01,
        )
        bandit.b["model_a"] = np.ones(dim) * 100.0
        bandit.refresh_inverse_cache()

        calibrate_priors(bandit)

        dummy_context = np.zeros(dim)
        dummy_context[-1] = 1.0

        A_inv = np.linalg.inv(bandit.A["model_a"])
        theta = A_inv @ bandit.b["model_a"]
        pred = theta @ dummy_context

        assert abs(pred) < 1.5, \
            f"Prior calibration failed: prediction = {pred}"


# =============================================================================
# Integration Tests
# =============================================================================

class TestRouterIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def full_registry(self):
        """Create comprehensive registry for integration testing."""
        return {
            "gpt-4": {
                "model_id": "openai/gpt-4",
                "display_name": "GPT-4",
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 15.0,
                "time_to_first_token_seconds": 1.0,
                "hle": 0.85,
                "initial_quality": 0.85,
                "capabilities": ["reasoning", "coding"],
                "speed_profile": "slow"
            },
            "gpt-3.5": {
                "model_id": "openai/gpt-3.5-turbo",
                "display_name": "GPT-3.5 Turbo",
                "input_cost_per_m": 0.5,
                "output_cost_per_m": 1.5,
                "time_to_first_token_seconds": 0.3,
                "hle": 0.65,
                "initial_quality": 0.65,
                "capabilities": ["general", "coding"],
                "speed_profile": "fast"
            },
            "claude-opus": {
                "model_id": "anthropic/claude-opus",
                "display_name": "Claude Opus",
                "input_cost_per_m": 6.0,
                "output_cost_per_m": 18.0,
                "time_to_first_token_seconds": 1.2,
                "hle": 0.88,
                "initial_quality": 0.88,
                "capabilities": ["reasoning", "creative"],
                "speed_profile": "slow"
            }
        }
    
    def test_full_routing_pipeline(self, full_registry):
        """Test complete routing pipeline from prompt to feedback."""
        router = BanditRouter.create(model_registry=full_registry, priors="none")
        
        # Route a prompt
        prompt = "Explain quantum computing"
        model, log = router.route(prompt, profile="auto")
        
        # Verify routing log
        assert model in full_registry.keys()
        assert log.selected_model == model
        assert log.prompt == prompt
        assert log.cost_usd > 0
        assert log.latency_s > 0
        assert log.context_vector is not None
        
        # Provide feedback
        router.process_feedback(log.request_id, reward=0.8)
        
        # Verify update happened (feedback processed without error)
    
    def test_progressive_model_registration(self, full_registry):
        """Test registering new models with progressive API."""
        router = BanditRouter.create(model_registry=full_registry, priors="none")
        
        # Register new model with capabilities
        router.register_model(
            model_id="gemini-pro",
            capabilities=["reasoning", "math"],
            speed="balanced",
            cost_usd=2.0,
            latency_s=0.5
        )
        
        # Should be added to registry and bandit
        assert "gemini-pro" in router.registry
        assert "gemini-pro" in router.bandit.models
        
        # Should be routable
        model, log = router.route("Solve this equation", profile="auto")
        # Model might or might not be selected, but should not crash
        assert model in router.registry.keys()
    
    def test_constraint_filtering(self, full_registry):
        """Test that constraints properly filter candidates."""
        router = BanditRouter.create(model_registry=full_registry, priors="none")

        # max_cost is in $/1k tokens (blended).  Blended costs in full_registry:
        #   gpt-4: 10.0/M  -> 0.010/1k
        #   gpt-3.5: 1.0/M -> 0.001/1k
        #   claude-opus: 12.0/M -> 0.012/1k
        # Setting max_cost=0.002 keeps only gpt-3.5.
        model, log = router.route(
            "Simple question",
            profile="auto",
            max_cost=0.002,
        )

        assert model == "gpt-3.5", \
            f"Expected cheapest model gpt-3.5, got {model}"

    def test_constraint_filtering_no_eligible(self, full_registry):
        """Test that NoEligibleModelsError is raised when no model passes."""
        router = BanditRouter.create(model_registry=full_registry, priors="none")

        with pytest.raises(NoEligibleModelsError):
            router.route("Simple question", profile="auto", max_cost=1e-6)
    
# =============================================================================
# Performance and Robustness Tests
# =============================================================================

class TestPerformanceFixes:
    """Test performance fixes for O(d³) matrix inversion caching."""
    
    def test_cost_aware_linucb_a_inv_caching(self):
        """Test that CostAwareLinUCBAdapter caches A_inv and updates it efficiently."""
        models = ["model_a", "model_b"]
        dim = 10

        np.random.seed(0)
        bandit = DisjointLinUCBPolicy(
            models, dim=dim, alpha=0.5, init_lambda=5.0,
        )
        for model in models:
            bandit.b[model] = np.random.randn(dim)
        bandit.refresh_inverse_cache()

        model_costs = {
            "model_a": {"normalized_cost": 0.2},
            "model_b": {"normalized_cost": 0.8},
        }

        adapter = CostAwareLinUCBAdapter(
            bandit=bandit,
            model_costs=model_costs,
            alpha_start=1.0,
            alpha_end=0.1,
            cost_penalty=0.5,
        )

        for model in models:
            assert model in adapter.A_inv
            identity_check = adapter.A[model] @ adapter.A_inv[model]
            np.testing.assert_array_almost_equal(identity_check, np.eye(dim), decimal=6)

        context = np.ones(dim)
        A_inv_before = adapter.A_inv["model_a"].copy()

        for _ in range(10):
            selected = adapter.select_model(context, total_steps=100)
            assert selected in models

        np.testing.assert_array_almost_equal(adapter.A_inv["model_a"], A_inv_before)

        adapter.update(context, "model_a", reward=0.8)

        assert not np.allclose(adapter.A_inv["model_a"], A_inv_before)

        identity_check = adapter.A["model_a"] @ adapter.A_inv["model_a"]
        np.testing.assert_array_almost_equal(identity_check, np.eye(dim), decimal=5)
    
    def test_cost_aware_tabula_rasa_a_inv_caching(self):
        """Test that CostAwareTabulaRasaRouter also caches A_inv correctly."""
        models = ["model_a", "model_b"]
        dim = 8
        
        model_costs = {
            "model_a": {"normalized_cost": 0.3},
            "model_b": {"normalized_cost": 0.7}
        }
        
        router = CostAwareTabulaRasaRouter(
            models=models,
            context_dim=dim,
            model_costs=model_costs,
            alpha_start=2.0,
            alpha_end=0.1,
            cost_penalty=0.5,
            ridge_lambda=1.0
        )
        
        # Verify A_inv is initialized
        for model in models:
            assert model in router.A_inv
            # Verify A_inv is the inverse of A
            identity_check = router.A[model] @ router.A_inv[model]
            np.testing.assert_array_almost_equal(identity_check, np.eye(dim), decimal=6)
        
        # Test Sherman-Morrison update
        context = np.random.randn(dim)
        context = context / np.linalg.norm(context)  # Normalize
        
        A_inv_before = router.A_inv["model_a"].copy()
        router.update(context, "model_a", reward=0.7)
        
        # A_inv should have been updated
        assert not np.allclose(router.A_inv["model_a"], A_inv_before)
        
        # Verify A_inv is still correct
        identity_check = router.A["model_a"] @ router.A_inv["model_a"]
        np.testing.assert_array_almost_equal(identity_check, np.eye(dim), decimal=5)
    
    def test_sherman_morrison_fallback_on_singularity(self):
        """Test that Sherman-Morrison falls back to full inversion when needed."""
        models = ["model_a"]
        dim = 5

        bandit = DisjointLinUCBPolicy(
            models, dim=dim, alpha=0.5, init_lambda=1.0,
        )

        model_costs = {"model_a": {"normalized_cost": 0.5}}

        adapter = CostAwareLinUCBAdapter(
            bandit=bandit,
            model_costs=model_costs,
        )

        context = np.ones(dim)
        context = context / np.linalg.norm(context)

        for i in range(100):
            adapter.update(context, "model_a", reward=0.5)

            assert not np.any(np.isnan(adapter.A_inv["model_a"]))
            assert not np.any(np.isinf(adapter.A_inv["model_a"]))

            if i % 10 == 0:
                identity_check = adapter.A["model_a"] @ adapter.A_inv["model_a"]
                identity_error = np.linalg.norm(identity_check - np.eye(dim))
                assert identity_error < 1e-4, f"Inverse accuracy degraded at iteration {i}"


class TestRobustnessFixes:
    """Test robustness fixes for non-stationary environments and hyperparameter sensitivity."""
    
    def test_corralling_loss_decay(self):
        """Test that Corralling applies exponential decay to cumulative losses."""
        class FixedExpert:
            def __init__(self, model_id: str):
                self.model_id = model_id
            def select_model(self, context, total_steps=0, **kwargs):
                return self.model_id
            def update(self, context, model, reward, weight=1.0, advance_time=True):
                pass
        
        experts_d = [FixedExpert("model_a"), FixedExpert("model_b")]
        experts_s = [FixedExpert("model_a"), FixedExpert("model_b")]
        models = ["model_a", "model_b"]
        
        router_with_decay = CorrallingRouter(
            experts=experts_d, models=models,
            learning_rate=0.1, gamma=0.05, loss_decay=0.9,
        )
        router_stationary = CorrallingRouter(
            experts=experts_s, models=models,
            learning_rate=0.1, gamma=0.05, loss_decay=1.0,
        )
        
        context = np.array([0.6, 0.4, 0.5])
        
        np.random.seed(42)
        for i in range(100):
            _, token_d = router_with_decay.select_model(context)
            reward_d = 0.3 if token_d.get("endorsing_experts") == [0] else 0.8
            router_with_decay.update(
                context, "model_a", reward=reward_d, selection_token=token_d,
            )
            _, token_s = router_stationary.select_model(context)
            reward_s = 0.3 if token_s.get("endorsing_experts") == [0] else 0.8
            router_stationary.update(
                context, "model_a", reward=reward_s, selection_token=token_s,
            )
        
        # With decay, sum_squared_losses should be bounded (recent history matters more)
        assert router_with_decay.sum_squared_losses.sum() < router_stationary.sum_squared_losses.sum(), \
            "Decayed squared losses should be smaller than non-decayed"
        
        # With Log-Barrier OMD, the adaptive learning rate already keeps weights
        # well-balanced.  Verify no collapse.
        assert router_with_decay.weights.min() > 0.01, \
            "Router with decay should maintain balanced weights"
        assert router_stationary.weights.min() > 0.01, \
            "Stationary router should also maintain balanced weights (log-barrier protects)"
    
    def test_register_model_initializes_stable_precision(self):
        """Dynamically registered models get A = λI (well-conditioned)."""
        registry = {
            "gpt-4": {
                "model_id": "openai/gpt-4",
                "display_name": "GPT-4",
                "input_cost_per_m": 5.0,
                "output_cost_per_m": 15.0,
                "hle": 0.85,
                "capabilities": ["reasoning"],
                "speed_profile": "slow"
            }
        }

        router = BanditRouter.create(model_registry=registry, priors="none")

        router.register_model("gpt-4-turbo", speed="fast", capabilities=["reasoning"])

        A = router.bandit.A["gpt-4-turbo"]
        lam = router.bandit.init_lambda

        assert A.shape == (router.bandit.dim, router.bandit.dim)
        off_diag = A - np.diag(np.diag(A))
        assert np.allclose(off_diag, 0, atol=1e-10), "A should be diagonal (λI)"
        assert np.allclose(np.diag(A), lam, rtol=1e-6), \
            f"Diagonal should equal init_lambda={lam}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

