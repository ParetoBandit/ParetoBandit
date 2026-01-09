import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from src.bandit_gpt.utils.experiment import ExperimentBurnIn
from src.bandit_gpt.router import BanditRouter

@pytest.fixture
def mock_registry():
    return {
        "model_a": {"hle": 0.1, "cost_per_1m": 0.5},
        "model_b": {"hle": 0.3, "cost_per_1m": 2.0},
    }

@pytest.fixture
def mock_rewards():
    return {
        "p1": {"model_a": 0.8, "model_b": 0.2},  # Contentious (Var > 0.05)
        "p2": {"model_a": 0.5, "model_b": 0.5},  # Not contentious
        "p3": {"model_a": 0.9, "model_b": 0.9},  # Not contentious
    }

@pytest.fixture
def temp_splits(tmp_path):
    splits_file = tmp_path / "splits.json"
    data = {
        "dev_pool": ["p1", "p2"],
        "holdout_pool": ["p3"]
    }
    with open(splits_file, "w") as f:
        json.dump(data, f)
    return splits_file

def test_get_splits(mock_registry, mock_rewards, temp_splits):
    burner = ExperimentBurnIn(mock_registry, mock_rewards, temp_splits)
    dev, test = burner.get_splits()
    assert dev == ["p1", "p2"]
    assert test == ["p3"]

def test_get_splits_missing_file(mock_registry, mock_rewards, tmp_path):
    missing_file = tmp_path / "nonexistent.json"
    burner = ExperimentBurnIn(mock_registry, mock_rewards, missing_file)
    with pytest.raises(FileNotFoundError):
        burner.get_splits()

def test_get_splits_leakage(mock_registry, mock_rewards, tmp_path):
    leaky_file = tmp_path / "leaky_splits.json"
    data = {
        "dev_pool": ["p1", "p2"],
        "holdout_pool": ["p1", "p3"]
    }
    with open(leaky_file, "w") as f:
        json.dump(data, f)
        
    burner = ExperimentBurnIn(mock_registry, mock_rewards, leaky_file)
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        burner.get_splits()

def test_generate_curriculum(mock_registry, mock_rewards, temp_splits):
    # p1 is hard (Var(0.8, 0.2) = 0.09 > 0.05)
    # p2 is easy (Var(0.5, 0.5) = 0.0)
    burner = ExperimentBurnIn(mock_registry, mock_rewards, temp_splits)
    curriculum = burner.generate_curriculum(["p1", "p2"])
    
    # p1 should be present 3 times
    assert curriculum.count("p1") == 3
    # p2 should be present at least once (random choice)
    assert "p2" in curriculum
    assert len(curriculum) > 3

def test_perform_burn_in(mock_registry, mock_rewards, temp_splits):
    burner = ExperimentBurnIn(mock_registry, mock_rewards, temp_splits)
    
    mock_router = MagicMock()
    mock_router.route.return_value = ("model_a", {})
    
    curriculum = ["p1", "p2"]
    burner.perform_burn_in(mock_router, curriculum)
    
    # Should have called route and update for each prompt
    assert mock_router.route.call_count == 2
    assert mock_router.update.call_count == 2
    
    # Verify update was called with correct arguments
    # For p1, model_a reward is 0.8
    mock_router.update.assert_any_call("model_a", "p1", 0.8)

def test_create_burned_in_router(mock_registry, mock_rewards, temp_splits):
    burner = ExperimentBurnIn(mock_registry, mock_rewards, temp_splits)
    
    # Create a small router to speed up test
    router, test_prompts = burner.create_burned_in_router(priors="none", alpha=0.1)
    
    assert isinstance(router, BanditRouter)
    assert test_prompts == ["p3"]
    assert router.bandit.alpha == 0.1
    
    # Check if update was called (A should not be identity if it was updated)
    # Since we used priors="none", A starts as init_lambda * I
    # After updates, trace(A) should increase.
    # Note: ExperimentBurnIn.perform_burn_in uses registry which has model_a and model_b
    has_update = False
    for m in ["model_a", "model_b"]:
        if np.trace(router.bandit.A[m]) > router.bandit.dim * router.bandit.init_lambda:
            has_update = True
            break
    assert has_update, "Router should have been updated during burn-in"
