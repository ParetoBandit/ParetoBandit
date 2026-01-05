#!/usr/bin/env python3
"""
Test script for Progressive Registration API.

Validates that the three tiers of knowledge (Archetypes, T-Shirt Sizing, Agnostic)
correctly translate to theta vectors with appropriate biases and feature weights.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.experiments.new_bandit.bandit_v2 import BanditRouter

def test_tier_a_archetypes():
    """Test Tier A: Archetypes (capabilities-based registration)"""
    print("\n" + "="*70)
    print("TEST 1: Tier A - Archetypes (Coding Specialist)")
    print("="*70)
    
    # Create minimal router
    router = BanditRouter.create(
        model_registry={},
        priors="none",  # Start with cold start
        context_encoder=None
    )
    
    # Register a coding specialist
    router.register_model(
        "deepseek-coder",
        capabilities=["coding"],
        speed="slow",
        cost_usd=2.0,
        latency_s=3.0
    )
    
    # Verify model was registered
    assert "deepseek-coder" in router.bandit.models, "Model not in bandit"
    assert "deepseek-coder" in router.registry, "Model not in registry"
    
    # Check theta vector (b / ridge_lambda)
    theta = router.bandit.b["deepseek-coder"] / router.bandit.ridge_lambda
    
    # Check bias (last index)
    bias = theta[-1]
    print(f"Bias: {bias:.2f} (expected: -0.5 for slow model)")
    assert abs(bias - (-0.5)) < 0.01, f"Unexpected bias: {bias}"
    
    # Check anchor_coding boost
    anchor_coding_idx = router._feature_map.get("anchor_coding")
    if anchor_coding_idx is not None:
        anchor_val = theta[anchor_coding_idx]
        print(f"anchor_coding weight: {anchor_val:.2f} (expected: 2.0)")
        assert abs(anchor_val - 2.0) < 0.01, f"Unexpected anchor_coding: {anchor_val}"
    
    # Check has_code_binarize boost
    code_binary_idx = router._feature_map.get("has_code_binarize")
    if code_binary_idx is not None:
        code_val = theta[code_binary_idx]
        print(f"has_code_binarize weight: {code_val:.2f} (expected: 1.5)")
        assert abs(code_val - 1.5) < 0.01, f"Unexpected has_code_binarize: {code_val}"
    
    # Check complexity_score (should be positive for slow model)
    complexity_idx = router._feature_map.get("complexity_score")
    if complexity_idx is not None:
        complexity_val = theta[complexity_idx]
        print(f"complexity_score weight: {complexity_val:.2f} (expected: 2.0 for slow)")
        assert abs(complexity_val - 2.0) < 0.01, f"Unexpected complexity_score: {complexity_val}"
    
    print("✅ Tier A test passed!")
    
def test_tier_b_tshirt_sizing():
    """Test Tier B: T-Shirt Sizing (speed-based registration)"""
    print("\n" + "="*70)
    print("TEST 2: Tier B - T-Shirt Sizing (Fast Model)")
    print("="*70)
    
    router = BanditRouter.create(
        model_registry={},
        priors="none",
        context_encoder=None
    )
    
    # Register a fast model
    router.register_model(
        "llama-3-8b",
        speed="fast",
        capabilities=["general"],
        cost_usd=0.1,
        latency_s=0.5
    )
    
    theta = router.bandit.b["llama-3-8b"] / router.bandit.ridge_lambda
    
    # Check bias (should be positive for fast model)
    bias = theta[-1]
    print(f"Bias: {bias:.2f} (expected: 1.5 for fast model)")
    assert abs(bias - 1.5) < 0.01, f"Unexpected bias: {bias}"
    
    # Check complexity_score (should be negative for fast model)
    complexity_idx = router._feature_map.get("complexity_score")
    if complexity_idx is not None:
        complexity_val = theta[complexity_idx]
        print(f"complexity_score weight: {complexity_val:.2f} (expected: -2.0 for fast)")
        assert abs(complexity_val - (-2.0)) < 0.01, f"Unexpected complexity_score: {complexity_val}"
    
    # Check that general capability boosted all anchors slightly
    for anchor in ["coding", "math", "creative", "jokes", "reasoning"]:
        anchor_idx = router._feature_map.get(f"anchor_{anchor}")
        if anchor_idx is not None:
            anchor_val = theta[anchor_idx]
            print(f"anchor_{anchor} weight: {anchor_val:.2f} (expected: 0.5 for general)")
            assert abs(anchor_val - 0.5) < 0.01, f"Unexpected anchor_{anchor}: {anchor_val}"
    
    print("✅ Tier B test passed!")

def test_tier_c_agnostic():
    """Test Tier C: Agnostic (minimal information)"""
    print("\n" + "="*70)
    print("TEST 3: Tier C - Agnostic (Mystery Model)")
    print("="*70)
    
    router = BanditRouter.create(
        model_registry={},
        priors="none",
        context_encoder=None
    )
    
    # Register a mystery model with minimal info
    router.register_model(
        "model-x",
        speed="balanced"
    )
    
    theta = router.bandit.b["model-x"] / router.bandit.ridge_lambda
    
    # Check bias (should be neutral for balanced)
    bias = theta[-1]
    print(f"Bias: {bias:.2f} (expected: 0.5 for balanced)")
    assert abs(bias - 0.5) < 0.01, f"Unexpected bias: {bias}"
    
    # Check complexity_score
    complexity_idx = router._feature_map.get("complexity_score")
    if complexity_idx is not None:
        complexity_val = theta[complexity_idx]
        print(f"complexity_score weight: {complexity_val:.2f} (expected: 0.5 for balanced)")
        assert abs(complexity_val - 0.5) < 0.01, f"Unexpected complexity_score: {complexity_val}"
    
    # Most other features should be zero
    non_zero_features = np.count_nonzero(theta)
    print(f"Non-zero features: {non_zero_features} (should be minimal: ~2)")
    
    print("✅ Tier C test passed!")

def test_power_user_override():
    """Test Power User: Explicit weight override"""
    print("\n" + "="*70)
    print("TEST 4: Power User - Explicit Weights")
    print("="*70)
    
    router = BanditRouter.create(
        model_registry={},
        priors="none",
        context_encoder=None
    )
    
    # Register with explicit weights that override defaults
    router.register_model(
        "custom-model",
        speed="fast",  # This would set complexity_score = -2.0
        initial_weights={
            "complexity_score": 3.5,  # Override to 3.5
            "anchor_math": 4.0
        }
    )
    
    theta = router.bandit.b["custom-model"] / router.bandit.ridge_lambda
    
    # Check that override worked
    complexity_idx = router._feature_map.get("complexity_score")
    if complexity_idx is not None:
        complexity_val = theta[complexity_idx]
        print(f"complexity_score weight: {complexity_val:.2f} (expected: 3.5 from override)")
        assert abs(complexity_val - 3.5) < 0.01, f"Override failed: {complexity_val}"
    
    # Check custom anchor_math
    math_idx = router._feature_map.get("anchor_math")
    if math_idx is not None:
        math_val = theta[math_idx]
        print(f"anchor_math weight: {math_val:.2f} (expected: 4.0 from override)")
        assert abs(math_val - 4.0) < 0.01, f"Override failed: {math_val}"
    
    print("✅ Power User test passed!")

def test_duplicate_registration():
    """Test that duplicate registration is prevented"""
    print("\n" + "="*70)
    print("TEST 5: Duplicate Registration Prevention")
    print("="*70)
    
    router = BanditRouter.create(
        model_registry={},
        priors="none",
        context_encoder=None
    )
    
    # Register once
    router.register_model("test-model", speed="fast")
    
    # Try to register again (should warn and skip)
    print("Attempting duplicate registration (should warn)...")
    router.register_model("test-model", speed="slow")
    
    # Verify the original registration persists (bias should still be 1.5, not -0.5)
    theta = router.bandit.b["test-model"] / router.bandit.ridge_lambda
    bias = theta[-1]
    print(f"Bias after duplicate attempt: {bias:.2f} (should still be 1.5)")
    assert abs(bias - 1.5) < 0.01, "Duplicate registration changed the model!"
    
    print("✅ Duplicate prevention test passed!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PROGRESSIVE REGISTRATION API - TEST SUITE")
    print("="*70)
    
    try:
        test_tier_a_archetypes()
        test_tier_b_tshirt_sizing()
        test_tier_c_agnostic()
        test_power_user_override()
        test_duplicate_registration()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
