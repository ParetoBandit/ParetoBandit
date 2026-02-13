import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bandit_gpt.router import BanditRouter

def test_pareto_frontier_filtering():
    """
    Verify that the router correctly prunes dominated models based on mean quality.
    
    [Paper ARCHITECTURAL FIX]: Filter uses ONLY mean quality, not UCB.
    - Pareto filtering = hard exclusion → miscalibration causes permanent damage
    - UCB selection = soft exploration → miscalibration self-corrects with data
    
    Uncertainty/exploration happens in the SELECTION phase, not the filtering phase.
    """
    # 1. Setup a dummy router
    router = BanditRouter.create()
    
    # Mock specific stats for 3 models to test the logic
    # Scenario:
    # - Model A: Cheap & Okay (The Baseline)
    # - Model B: Expensive & Same Quality as A (Dominated -> Should be Pruned)
    # - Model C: Expensive but BETTER Quality (Not Dominated -> Should Survive)
    
    # We mock _get_contextual_stats to return controlled values
    original_get_stats = router._get_contextual_stats
    
    mock_stats = {
        "model_a": {"id": "model_a", "mean_quality": 0.80, "uncertainty": 0.01, "cost": 0.50},
        "model_b": {"id": "model_b", "mean_quality": 0.80, "uncertainty": 0.01, "cost": 5.00}, # Dominated by A (same quality, higher cost)
        "model_c": {"id": "model_c", "mean_quality": 0.90, "uncertainty": 0.01, "cost": 5.00}, # Not dominated (better quality justifies cost)
    }
    
    router._get_contextual_stats = lambda m, x, i, o: mock_stats[m]
    
    # 2. Run Filter
    candidates = ["model_a", "model_b", "model_c"]
    # Dummy context/tokens (ignored by our mock)
    survivors = router._filter_pareto_frontier(candidates, np.zeros(10), 0, 0)
    
    # 3. Assertions
    assert "model_a" in survivors, "Baseline should survive (cheap)"
    assert "model_b" not in survivors, "Expensive duplicate should be pruned (dominated by A)"
    assert "model_c" in survivors, "Better quality model should survive (not dominated)"
    
    # Restore method
    router._get_contextual_stats = original_get_stats

def test_smart_shopper_selection():
    """Verify 'smart_shopper' profile picks the best utility score."""
    router = BanditRouter.create()
    
    # Mock stats again
    # Smart Shopper Lambda = 0.5
    # Utility = Quality - (0.5 * Cost)
    mock_stats = {
        "cheap": {"id": "cheap", "mean_quality": 0.90, "uncertainty": 0.0, "cost": 0.10}, 
        # Util = 0.90 - (0.5 * 0.10) = 0.85
        
        "luxury": {"id": "luxury", "mean_quality": 0.95, "uncertainty": 0.0, "cost": 2.00},
        # Util = 0.95 - (0.5 * 2.00) = -0.05
    }
    router._get_contextual_stats = lambda m, x, i, o: mock_stats[m]
    
    # We also need to mock _filter_pareto_frontier to just return everyone for this test
    router._filter_pareto_frontier = lambda c, x, i, o: c
    
    # Run Route
    router.registry = {"cheap": {}, "luxury": {}} # Minimal registry for candidates
    
    # Force the profile to be recognized
    router.PARETO_PROFILES["test_shopper"] = 0.5
    
    selected, _ = router.route("test prompt", profile="auto")
    
    assert selected == "cheap", "Should pick 'cheap' model due to high cost penalty"
