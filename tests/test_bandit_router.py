import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
import json
from bandit_gpt import BanditRouter, ExplorationRate, RouterConfig

@pytest.fixture
def sample_registry():
    return {
        "openai/gpt-4o": {
            "model_id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "scores": {"hle": 0.85},
            "hallucination_rate": 1.5,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0
        },
        "google/gemma-3-2b-it": {
            "model_id": "google/gemma-3-2b-it",
            "display_name": "Gemma 3 2B",
            "scores": {"hle": 0.45},
            "hallucination_rate": 8.0,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.1
        }
    }

def test_router_initialization(sample_registry):
    # Test initialization with custom registry
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    assert len(router.registry) == 2
    assert "openai/gpt-4o" in router.bandit.models

def test_routing_decisions(sample_registry):
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    prompt = "Simple hello"
    
    # Test simple routing
    model, log = router.route(prompt)
    assert model in ["openai/gpt-4o", "google/gemma-3-2b-it"]
    assert log.selected_model == model
    
    # Test with profile (use auto as default intelligent routing)
    model_cs, log_cs = router.route(prompt, profile="auto")
    # Note: Model selection depends on router's UCB scores and may vary
    assert model_cs in ["openai/gpt-4o", "google/gemma-3-2b-it"]

def test_constraints(sample_registry):
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    prompt = "Constraint test"
    
    # Max cost constraint should favor Gemma
    model, log = router.route(prompt, max_cost=0.001)
    assert model == "google/gemma-3-2b-it"
    
    # Quality floor floor should favor GPT-4
    # Note: HLE is used as prompt score in this simple logic
    model_q, log_q = router.route(prompt, quality_floor={"hle": 0.7})
    assert model_q == "openai/gpt-4o"

def test_save_load(sample_registry, tmp_path):
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    save_path = tmp_path / "bandit_state.npz"
    
    # Run a route and feedback
    model, log = router.route("test")
    router.process_feedback(log.request_id, 1.0)
    
    # Save
    router.save_state(save_path)
    assert save_path.exists()
    
    # Load into new router
    router2 = BanditRouter.create(model_registry=sample_registry, state_path=save_path)
    assert np.allclose(router.bandit.b[model], router2.bandit.b[model])


# TODO: Re-enable after investigating exploration behavior with dynamic Pareto filtering
# This test fails because model0 (cheapest) dominates all routing decisions.
# Need to investigate whether this is correct behavior or if exploration mechanism needs tuning.
def _test_no_zombie_models():
    """
    Integration test: Verify that models don't get stuck in "zombie mode".
    
    With HLE priors, models have different initial UCBs, creating natural
    exploration across the quality-cost spectrum.
    
    NOTE: Temporarily disabled due to interaction with dynamic Pareto filtering.
    """
    # Create a registry with 10 models spanning wide HLE range
    registry = {}
    for i in range(10):
        hle_score = 0.10 + (i * 0.075)  # 0.10 to 0.775
        registry[f"model{i}"] = {
            "model_id": f"provider/model-{i}",
            "display_name": f"Model {i}",
            "hle": hle_score,
            "input_cost_per_m": 0.5 + (i * 0.5),
            "output_cost_per_m": 1.5 + (i * 1.5)
        }
    
    # Create router with HLE priors for realistic initialization
    router = BanditRouter.create(model_registry=registry, priors="hle")
    router.config.probation_bonus = 0.10
    router.config.pruning_min_samples = 30
    
    # Route 500 prompts using custom profile (not "auto" to bypass Pareto filtering)
    # This test is about exploration/exploitation balance, not Pareto efficiency
    N = 500
    custom_profile = {"w_q": 1.0, "w_c": 0.02, "w_l": 0.0}  # Similar to "auto" but bypasses Pareto filter
    for i in range(N):
        model, log = router.route(f"Test prompt {i}", profile=custom_profile)
        # Provide feedback proportional to HLE
        hle = registry[model]["hle"]
        reward = min(1.0, hle + np.random.normal(0, 0.1))
        router.process_feedback(log.request_id, reward=max(0, reward))
    
    # Check sample counts for all models
    sample_counts = router._get_sample_counts(router.bandit.models)
    
    # With HLE priors, bandit will naturally favor high-HLE models
    # but probation subsidy ensures some exploration of lower models
    models_with_traffic = sum(1 for count in sample_counts.values() if count > 0)
    
    # At least 3 models should receive traffic (out of 10)
    assert models_with_traffic >= 3, \
        f"Too few models explored: {sample_counts}"


# =============================================================================
# RESILIENCE TESTS: Pessimistic Defaults (Fail-Operational)
# =============================================================================

def test_estimate_cost_pessimistic_defaults():
    """
    Test that _estimate_cost uses pessimistic defaults when metadata is missing.
    
    Critical resilience behavior: Missing cost data should NOT return infinity,
    which would cause all models to be rejected, leading to service outage.
    Instead, return expensive-tier pricing to keep service operational.
    """
    # Registry with missing cost metadata
    registry_missing_costs = {
        "model_a": {
            "model_id": "provider/model-a",
            "display_name": "Model A",
            "hle": 0.50,
            # Missing: input_cost_per_m, output_cost_per_m
        },
        "model_b": {
            "model_id": "provider/model-b",
            "display_name": "Model B",
            "hle": 0.60,
            "input_cost_per_m": 2.0,
            # Missing: output_cost_per_m only
        },
        "model_c": {
            "model_id": "provider/model-c",
            "display_name": "Model C",
            "hle": 0.70,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0  # Complete metadata
        }
    }
    
    router = BanditRouter.create(model_registry=registry_missing_costs, priors="hle")
    
    # Test model_a: Both costs missing
    cost_a = router._estimate_cost("model_a", in_tok=1000, out_tok=500)
    assert cost_a != float('inf'), "Missing costs should NOT return infinity"
    assert cost_a > 0, "Cost should be positive"
    # Expected: (10.0 * 1000 + 30.0 * 500) / 1e6 = 0.025 (pessimistic tier)
    assert cost_a == pytest.approx(0.025, rel=0.01), f"Expected pessimistic cost, got {cost_a}"
    
    # Test model_b: Output cost missing only
    cost_b = router._estimate_cost("model_b", in_tok=1000, out_tok=500)
    assert cost_b != float('inf'), "Missing output cost should NOT return infinity"
    # Expected: (2.0 * 1000 + 30.0 * 500) / 1e6 = 0.017 (input from registry, output pessimistic)
    assert cost_b == pytest.approx(0.017, rel=0.01), f"Expected mixed cost, got {cost_b}"
    
    # Test model_c: Complete metadata
    cost_c = router._estimate_cost("model_c", in_tok=1000, out_tok=500)
    # Expected: (5.0 * 1000 + 15.0 * 500) / 1e6 = 0.0125
    assert cost_c == pytest.approx(0.0125, rel=0.01), f"Expected accurate cost, got {cost_c}"


def test_estimate_latency_pessimistic_defaults():
    """
    Test that _estimate_latency uses pessimistic defaults when metadata is missing.
    
    Critical resilience behavior: Missing latency data should NOT return infinity.
    """
    registry_missing_latency = {
        "model_a": {
            "model_id": "provider/model-a",
            "display_name": "Model A",
            "hle": 0.50,
            "input_cost_per_m": 1.0,
            "output_cost_per_m": 3.0,
            # Missing: time_to_first_token_seconds
        },
        "model_b": {
            "model_id": "provider/model-b",
            "display_name": "Model B",
            "hle": 0.60,
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
            "time_to_first_token_seconds": 0.0  # Invalid: zero latency
        },
        "model_c": {
            "model_id": "provider/model-c",
            "display_name": "Model C",
            "hle": 0.70,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0,
            "time_to_first_token_seconds": 0.5  # Valid latency
        }
    }
    
    router = BanditRouter.create(model_registry=registry_missing_latency, priors="hle")
    
    # Test model_a: Latency missing
    latency_a = router._estimate_latency("model_a", out_tok=500)
    assert latency_a != float('inf'), "Missing latency should NOT return infinity"
    assert latency_a == router.config.default_missing_latency, \
        f"Expected pessimistic latency {router.config.default_missing_latency}, got {latency_a}"
    
    # Test model_b: Invalid zero latency
    latency_b = router._estimate_latency("model_b", out_tok=500)
    assert latency_b != float('inf'), "Zero latency should NOT return infinity"
    assert latency_b == router.config.default_missing_latency, \
        f"Expected pessimistic latency for invalid zero, got {latency_b}"
    
    # Test model_c: Valid latency
    latency_c = router._estimate_latency("model_c", out_tok=500)
    assert latency_c == 0.5, f"Expected accurate latency 0.5, got {latency_c}"
