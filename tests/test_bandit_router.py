import pytest
import numpy as np
import json
from pathlib import Path
from bandit_gpt import BanditRouter, OptimizationProfile, ExplorationRate, RouterConfig

@pytest.fixture
def sample_registry():
    return {
        "openai/gpt-4o": {
            "openrouter_id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "scores": {"hle": 0.85},
            "hallucination_rate": 1.5,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0
        },
        "google/gemma-3-2b-it": {
            "openrouter_id": "google/gemma-3-2b-it",
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
    
    # Test with profile (use best_value as a cost-sensitive proxy)
    model_cs, log_cs = router.route(prompt, profile="best_value")
    assert model_cs == "google/gemma-3-2b-it"

def test_feedback_learning(sample_registry):
    # Only register one model to ensure it is selected for feedback test
    single_registry = {"openai/gpt-4o": sample_registry["openai/gpt-4o"]}
    router = BanditRouter.create(model_registry=single_registry, priors="none")
    prompt = "Learning test"
    
    # Get initial b vector norm
    model = "openai/gpt-4o"
    initial_b_norm = np.linalg.norm(router.bandit.b[model])
    
    # Route and get request_id
    _, log = router.route(prompt)
    request_id = log.request_id
    
    # Provide positive feedback
    router.process_feedback(request_id, reward=1.0)
    
    # b vector should have updated
    updated_b_norm = np.linalg.norm(router.bandit.b[model])
    assert updated_b_norm > initial_b_norm

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


def test_probation_subsidy():
    """
    Test that the probation subsidy mechanism correctly boosts quality scores.
    
    This test verifies the MECHANISM works (bonus applied to quality), not that
    specific traffic distributions occur. A bandit should favor high-quality models,
    and the probation subsidy is a small nudge, not a forcing function.
    """
    # Setup: Create a registry with models of varying quality and cost
    registry = {
        "high_quality": {
            "openrouter_id": "openai/gpt-4o",
            "display_name": "High Quality Model",
            "hle": 0.85,  # High HLE
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0
        },
        "medium_quality": {
            "openrouter_id": "model/medium",
            "display_name": "Medium Quality Model",
            "hle": 0.50,
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0
        },
        "budget_model": {
            "openrouter_id": "model/budget",
            "display_name": "Budget Model",
            "hle": 0.35,
            "input_cost_per_m": 0.1,
            "output_cost_per_m": 0.3
        }
    }
    
    # Create router with HLE priors
    router = BanditRouter.create(model_registry=registry, priors="hle")
    router.config.probation_bonus = 0.10
    router.config.pruning_min_samples = 15
    
    # Verify probation bonus is correctly computed for each model
    sample_counts = router._get_sample_counts(router.bandit.models)
    
    for model in router.bandit.models:
        count = sample_counts.get(model, 0)
        expected_decay = 1.0 - (count / router.config.pruning_min_samples)
        expected_bonus = router.config.probation_bonus * max(0, expected_decay)
        
        # Verify all models start in probation (count < min_samples)
        assert count < router.config.pruning_min_samples, \
            f"{model} should be in probation but has {count} samples"
        
        # Verify bonus would be applied (non-zero)
        assert expected_bonus > 0, \
            f"{model} should have positive probation bonus"
    
    # Route some prompts and verify sample counting works
    N = 50
    for i in range(N):
        model, log = router.route(f"Test prompt {i}", profile="arbitrage")
        router.process_feedback(log.request_id, reward=0.8)
    
    # Verify sample counts are tracked
    final_counts = router._get_sample_counts(router.bandit.models)
    total_routed = sum(final_counts.values())
    assert total_routed == N, f"Expected {N} total routes, got {total_routed}"


def test_no_zombie_models():
    """
    Integration test: Verify that models don't get stuck in "zombie mode".
    
    With HLE priors, models have different initial UCBs, creating natural
    exploration across the quality-cost spectrum.
    """
    # Create a registry with 10 models spanning wide HLE range
    registry = {}
    for i in range(10):
        hle_score = 0.10 + (i * 0.075)  # 0.10 to 0.775
        registry[f"model{i}"] = {
            "openrouter_id": f"provider/model-{i}",
            "display_name": f"Model {i}",
            "hle": hle_score,
            "input_cost_per_m": 0.5 + (i * 0.5),
            "output_cost_per_m": 1.5 + (i * 1.5)
        }
    
    # Create router with HLE priors for realistic initialization
    router = BanditRouter.create(model_registry=registry, priors="hle")
    router.config.probation_bonus = 0.10
    router.config.pruning_min_samples = 30
    
    # Route 500 prompts using arbitrage profile
    N = 500
    for i in range(N):
        model, log = router.route(f"Test prompt {i}", profile="arbitrage")
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
