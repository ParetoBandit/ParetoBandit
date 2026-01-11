import json
import gzip
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


# ============================================================================
# NEW TESTS FOR SPLIT GENERATION AND REWARD JOINING
# ============================================================================

def test_create_canonical_splits(mock_rewards, tmp_path):
    """Test the new create_canonical_splits() static method."""
    splits_path = tmp_path / "new_splits.json"
    
    # Create splits
    dev, holdout = ExperimentBurnIn.create_canonical_splits(
        oracle_rewards=mock_rewards,
        splits_path=splits_path,
        test_ratio=0.33,  # 1 of 3 prompts for holdout
        random_state=42
    )
    
    # Verify splits were created
    assert splits_path.exists()
    assert len(dev) == 2
    assert len(holdout) == 1
    
    # Verify disjointness
    assert set(dev).isdisjoint(set(holdout))
    
    # Verify file contents
    with open(splits_path) as f:
        data = json.load(f)
    assert "dev_pool" in data
    assert "holdout_pool" in data
    assert data["dev_pool"] == dev
    assert data["holdout_pool"] == holdout


def test_create_canonical_splits_reproducible(mock_rewards, tmp_path):
    """Test that create_canonical_splits() is reproducible with same seed."""
    splits_path1 = tmp_path / "splits1.json"
    splits_path2 = tmp_path / "splits2.json"
    
    # Create splits twice with same seed
    dev1, holdout1 = ExperimentBurnIn.create_canonical_splits(
        mock_rewards, splits_path1, test_ratio=0.33, random_state=42
    )
    dev2, holdout2 = ExperimentBurnIn.create_canonical_splits(
        mock_rewards, splits_path2, test_ratio=0.33, random_state=42
    )
    
    # Should be identical
    assert dev1 == dev2
    assert holdout1 == holdout2


def test_create_canonical_splits_data_leakage(tmp_path):
    """Test that create_canonical_splits() would catch data leakage (edge case)."""
    # This is more of a sanity check - with proper implementation,
    # there should never be leakage
    oracle_rewards = {f"p{i}": {"model_a": 0.5} for i in range(100)}
    splits_path = tmp_path / "splits.json"
    
    dev, holdout = ExperimentBurnIn.create_canonical_splits(
        oracle_rewards, splits_path, test_ratio=0.4, random_state=42
    )
    
    # Verify no overlap
    overlap = set(dev).intersection(set(holdout))
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping prompts"


def test_get_splits_backward_compatible_signature(mock_registry, mock_rewards, temp_splits):
    """Test that get_splits() maintains backward compatible signature."""
    burner = ExperimentBurnIn(mock_registry, mock_rewards, temp_splits)
   
    # Test default behavior (no argument)
    result_default = burner.get_splits()
    assert isinstance(result_default, tuple)
    assert len(result_default) == 2
    assert isinstance(result_default[0], list)
    assert isinstance(result_default[1], list)
    
    # Test explicit False
    result_false = burner.get_splits(load_rewards=False)
    assert result_default == result_false


def test_create_canonical_splits_stratified(tmp_path):
    """Test that stratified splits actually balance the distributions."""
    # Create 100 prompts: 50 STEM, 50 CODE
    oracle_rewards = {}
    for i in range(50):
        oracle_rewards[f"STEM calculation theorem {i}"] = {"m1": 0.9}
        oracle_rewards[f"CODE function debug {i}"] = {"m1": 0.5}
        
    splits_path = tmp_path / "stratified_splits.json"
    dev, holdout = ExperimentBurnIn.create_canonical_splits(
        oracle_rewards, splits_path, test_ratio=0.5, random_state=42
    )
    
    # In a perfect stratified split, we should have 25 STEM and 25 CODE in each
    dev_stem = [p for p in dev if "STEM" in p]
    dev_code = [p for p in dev if "CODE" in p]
    holdout_stem = [p for p in holdout if "STEM" in p]
    holdout_code = [p for p in holdout if "CODE" in p]
    
    assert len(dev_stem) == 25
    assert len(dev_code) == 25
    assert len(holdout_stem) == 25
    assert len(holdout_code) == 25
