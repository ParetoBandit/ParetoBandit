import pytest
import numpy as np
import json
from pathlib import Path
from bandit_gpt import BanditRouter, OptimizationProfile, ExplorationRate

@pytest.fixture
def sample_registry():
    return {
        "gpt-4": {
            "openrouter_id": "openai/gpt-4o",
            "display_name": "GPT-4o",
            "hle": 0.85,
            "price_1m_blended": 5.0,
            "time_to_first_token_seconds": 0.8,
            "hallucination_composite": 1.5,
            "input_cost_per_m": 5.0,
            "output_cost_per_m": 15.0
        },
        "gemma-2b": {
            "openrouter_id": "google/gemma-3-2b-it",
            "display_name": "Gemma 3 2B",
            "hle": 0.45,
            "price_1m_blended": 0.1,
            "time_to_first_token_seconds": 0.2,
            "hallucination_composite": 8.0,
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
    
    # Test with profile
    model_cs, log_cs = router.route(prompt, profile="cost_saver")
    assert model_cs == "google/gemma-3-2b-it"

def test_feedback_learning(sample_registry):
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    prompt = "Learning test"
    
    # Get initial b vector norm
    model = "openai/gpt-4o"
    initial_b_norm = np.linalg.norm(router.bandit.b[model])
    
    # Route and get trace_id
    _, log = router.route(prompt, candidate_models=[model])
    trace_id = log.trace_id
    
    # Provide positive feedback
    router.process_feedback(trace_id, reward_logit=2.0)
    
    # b vector should have updated
    updated_b_norm = np.linalg.norm(router.bandit.b[model])
    assert updated_b_norm > initial_b_norm

def test_constraints(sample_registry):
    router = BanditRouter.create(model_registry=sample_registry, priors="none")
    prompt = "Constraint test"
    
    # Max cost constraint should favor Gemma
    model, log = router.route(prompt, max_cost=0.00000001)
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
    router.process_feedback(log.trace_id, 1.0)
    
    # Save
    router.save_state(save_path)
    assert save_path.exists()
    
    # Load into new router
    router2 = BanditRouter.create(model_registry=sample_registry, state_path=save_path)
    assert np.allclose(router.bandit.b[model], router2.bandit.b[model])
