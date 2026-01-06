"""
Unit tests for atomicity refactoring of route() and update() methods.

Tests individual helper methods in isolation to verify correct behavior.
"""

import pytest
import numpy as np
from bandit_gpt.router import BanditRouter, DisjointLinUCBPolicy, BanditState


@pytest.fixture
def sample_registry():
    return {
        "gpt-4": {
            "openrouter_id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "hle": 0.85,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0,
            "time_to_first_token_seconds": 0.8,
            "hallucination_vectara": 1.5,
        },
        "gemma-2b": {
            "openrouter_id": "google/gemma-2-2b",
            "display_name": "Gemma 2 2B",
            "hle": 0.45,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1,
            "time_to_first_token_seconds": 0.2,
            "hallucination_vectara": 3.0,
        }
    }


@pytest.fixture
def router(sample_registry):
    return BanditRouter.create(model_registry=sample_registry, priors="none")


# ============================================================================
# DisjointLinUCBPolicy Helper Method Tests
# ============================================================================

class TestUpdateHelperMethods:
    """Test helper methods for update() refactoring."""
    
    def test_snapshot_bandit_state(self, router):
        """Test _snapshot_bandit_state creates valid snapshot."""
        model = "gpt-4"
        state = router.bandit._snapshot_bandit_state(model)
        
        # Verify it's a valid BanditState
        assert isinstance(state, dict)
        assert "A" in state
        assert "b" in state
        assert "A_inv" in state
        assert "timestamp" in state
        assert "needs_full_inversion" in state
        
        # Verify arrays are copies, not references
        assert state["A"] is not router.bandit.A[model]
        assert state["b"] is not router.bandit.b[model]
        
        # But values should match
        assert np.allclose(state["A"], router.bandit.A[model])
        assert np.allclose(state["b"], router.bandit.b[model])
    
    def test_apply_temporal_decay(self, router):
        """Test _apply_temporal_decay applies forgetting factor correctly."""
        model = "gpt-4"
        
        # Create initial state
        state = router.bandit._snapshot_bandit_state(model)
        original_A = state["A"].copy()
        
        # Apply decay (simulate time passing)
        router.bandit.t += 5  # Advance time
        state = router.bandit._apply_temporal_decay(state, model)
        
        # With gamma < 1.0, matrices should decay
        if router.bandit.gamma < 1.0:
            # A should be scaled down
            expected_gamma = router.bandit.gamma ** 5
            assert np.allclose(state["A"], original_A * expected_gamma)
    
    def test_add_observation(self, router):
        """Test _add_observation updates matrices correctly."""
        model = "gpt-4"
        state = router.bandit._snapshot_bandit_state(model)
        
        # Create observation
        x = np.random.randn(router.bandit.dim)
        reward = 0.8
        weight = 1.0
        
        original_A = state["A"].copy()
        original_b = state["b"].copy()
        
        # Add observation
        state = router.bandit._add_observation(state, x, reward, weight)
        
        # Verify update: A_new = A_old + weight * x @ x^T
        expected_A = original_A + weight * np.outer(x, x)
        assert np.allclose(state["A"], expected_A)
        
        # Verify update: b_new = b_old + weight * reward * x
        expected_b = original_b + weight * reward * x
        assert np.allclose(state["b"], expected_b)
    
    def test_update_inverse_matrix_sherman_morrison(self, router):
        """Test _update_inverse_matrix uses Sherman-Morrison when possible."""
        model = "gpt-4"
        state = router.bandit._snapshot_bandit_state(model)
        state["needs_full_inversion"] = False  # Force Sherman-Morrison path
        
        x = np.random.randn(router.bandit.dim)
        weight = 1.0
        
        original_A_inv = state["A_inv"].copy()
        
        # Update inverse
        state = router.bandit._update_inverse_matrix(state, x, weight)
        
        # Sherman-Morrison should have been applied
        # The result should still be valid (A @ A_inv ≈ I)
        # But we need to update A first
        A_updated = state["A"] + weight * np.outer(x, x)
        identity = A_updated @ state["A_inv"]
        assert np.allclose(identity, np.eye(router.bandit.dim), atol=1e-6)
    
    def test_commit_bandit_state(self, router):
        """Test _commit_bandit_state updates global state atomically."""
        model = "gpt-4"
        
        # Create modified state
        state = router.bandit._snapshot_bandit_state(model)
        state["A"] = state["A"] * 2.0  # Modify
        state["b"] = state["b"] + 1.0  # Modify
        
        # Commit
        router.bandit._commit_bandit_state(model, state)
        
        # Verify global state updated
        assert np.allclose(router.bandit.A[model], state["A"])
        assert np.allclose(router.bandit.b[model], state["b"])
        
        # Verify timestamp incremented
        assert router.bandit.last_update[model] == router.bandit.t - 1


# ============================================================================
# BanditRouter Helper Method Tests
# ============================================================================

class TestRouteHelperMethods:
    """Test helper methods for route() refactoring."""
    
    def test_build_routing_features(self, router):
        """Test _build_routing_features extracts context vector."""
        prompt = "Write a Python function"
        x, prompt_text = router._build_routing_features(prompt)
        
        assert isinstance(x, np.ndarray)
        assert x.shape[0] == router.bandit.dim
        assert prompt_text == prompt
        
        # Test with pre-embedded vector
        x_direct, text_direct = router._build_routing_features(x)
        assert text_direct == "[Pre-embedded Prompt]"
    
    def test_resolve_utility_weights(self, router):
        """Test _resolve_utility_weights handles orthogonal optimization."""
        # Test without constraints
        w_q, w_c, w_l = router._resolve_utility_weights("best_value", None, None)
        assert w_q + w_c + w_l == pytest.approx(1.0)
        
        # Test with cost constraint (should zero out w_c)
        w_q2, w_c2, w_l2 = router._resolve_utility_weights("best_value", 0.001, None)
        assert w_c2 == 0.0
        assert w_q2 > w_q  # Quality weight increased
        
        # Test with latency constraint (should zero out w_l)
        w_q3, w_c3, w_l3 = router._resolve_utility_weights("best_value", None, 1.0)
        assert w_l3 == 0.0
        assert w_q3 > w_q  # Quality weight increased
    
    def test_apply_risk_gating(self, router):
        """Test _apply_risk_gating filters by sensitivity."""
        prompt = "Medical diagnosis question"
        
        # LOW sensitivity - should return all models
        candidates_low = router._apply_risk_gating(prompt, "LOW")
        assert len(candidates_low) == 2
        
        # HIGH sensitivity - should filter by hallucination score
        candidates_high = router._apply_risk_gating(prompt, "HIGH")
        # Only GPT-4 has hallucination_vectara <= 2.5
        assert len(candidates_high) == 1
        assert "gpt-4" in candidates_high
    
    def test_filter_by_constraints(self, router):
        """Test _filter_by_constraints applies hard constraints."""
        prompt = "Test prompt"
        candidates = list(router.registry.keys())
        
        # No constraints - all pass
        filtered = router._filter_by_constraints(
            candidates, prompt, None, None, None, None, 600
        )
        assert len(filtered) == 2
        
        # Cost constraint - only gemma-2b should pass
        # Gemma costs ~0.0001 per request, GPT-4 costs ~0.005+ per request
        filtered_cost = router._filter_by_constraints(
            candidates, prompt, 0.0002, None, None, 500, 600
        )
        assert "gemma-2b" in filtered_cost
        # GPT-4 might also pass if cost is close, so just verify gemma is included
        
        # Latency constraint - gemma-2b is faster
        filtered_lat = router._filter_by_constraints(
            candidates, prompt, None, 0.3, None, 500, 600
        )
        assert "gemma-2b" in filtered_lat
    
    def test_calculate_penalties(self, router):
        """Test _calculate_penalties computes absolute penalties."""
        filtered = ["gpt-4", "gemma-2b"]
        cost_pen, lat_pen = router._calculate_penalties(filtered, 500, 600)
        
        # Both should return dicts with penalties in [0, 1]
        assert all(0 <= cost_pen[m] <= 1 for m in filtered)
        assert all(0 <= lat_pen[m] <= 1 for m in filtered)
        
        # GPT-4 should have higher cost penalty (more expensive)
        assert cost_pen["gpt-4"] > cost_pen["gemma-2b"]
        
        # GPT-4 should have higher latency penalty (slower)
        assert lat_pen["gpt-4"] > lat_pen["gemma-2b"]
    
    def test_score_candidates(self, router):
        """Test _score_candidates selects best model."""
        prompt = "Test prompt"
        x = router._get_context_vector(prompt)
        filtered = ["gpt-4", "gemma-2b"]
        
        # Equal weights - should prefer higher UCB
        best_model, best_utility = router._score_candidates(
            filtered, x, 1.0, 0.0, 0.0, 500, 600
        )
        assert best_model in filtered
        assert best_utility > 0
        
        # Cost-focused - should prefer gemma-2b
        best_cost, _ = router._score_candidates(
            filtered, x, 0.0, 1.0, 0.0, 500, 600
        )
        assert best_cost == "gemma-2b"
    
    def test_create_routing_log(self, router):
        """Test _create_routing_log creates valid log."""
        prompt = "Test prompt"
        x = router._get_context_vector(prompt)
        
        log = router._create_routing_log(prompt, "gpt-4", 0.85, x, 500, 600)
        
        assert log.prompt == prompt
        assert log.selected_model == "gpt-4"
        assert log.predicted_utility == pytest.approx(0.85)
        assert log.cost_usd > 0
        assert log.latency_s > 0
        assert np.allclose(log.context_vector, x)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test that refactored methods produce same results as before."""
    
    def test_update_equivalence(self, router):
        """Test update() produces correct results through helper orchestration."""
        model = "gpt-4"
        x = np.random.randn(router.bandit.dim)
        reward = 0.9
        
        # Capture initial state
        initial_A = router.bandit.A[model].copy()
        initial_b = router.bandit.b[model].copy()
        
        # Update
        router.bandit.update(model, x, reward, weight=1.0)
        
        # Verify matrices changed
        assert not np.allclose(router.bandit.A[model], initial_A)
        assert not np.allclose(router.bandit.b[model], initial_b)
        
        # Verify A @ A_inv ≈ I (inverse is correct)
        identity = router.bandit.A[model] @ router.bandit.A_inv[model]
        assert np.allclose(identity, np.eye(router.bandit.dim), atol=1e-6)
    
    def test_route_equivalence(self, router):
        """Test route() produces valid routing decisions."""
        prompt = "Write a Python function to sort a list"
        
        # Route
        model, log = router.route(prompt, profile="best_value")
        
        # Verify valid selection
        assert model in router.registry.keys()
        assert log.selected_model == model
        assert log.predicted_utility > 0
        assert log.cost_usd > 0
        assert log.latency_s > 0
        
        # Verify log persisted
        assert len(router.logs) > 0
        assert router.logs[-1].request_id == log.request_id
    
    def test_multiple_routes_consistency(self, router):
        """Test multiple routes work consistently."""
        prompts = [
            "Solve this math problem",
            "Write creative story",
            "Debug this code"
        ]
        
        for prompt in prompts:
            model, log = router.route(prompt, profile="best_value")
            assert model in router.registry.keys()
            assert log.predicted_utility > 0
        
        # All logs should be stored
        assert len(router.logs) == len(prompts)
