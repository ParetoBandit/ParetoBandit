#!/usr/bin/env python3
"""
Test script for OptimizationProfile.from_reference() method.

This validates the Reference Point Normalization implementation with the
example scenarios from the user's request.
"""

import sys
sys.path.insert(0, '/Users/annette/repostitories/banditGPT/src')

from bandit_gpt.router import OptimizationProfile


def test_arbitrageur():
    """Scenario A: The Arbitrageur
    
    "I want a model that is basically GPT-4 (99% quality) but at half price (50% savings)."
    
    Expected: w_q = 50.0, w_c = 1.0, w_l = 0.0
    """
    print("=" * 70)
    print("Scenario A: The Arbitrageur")
    print("=" * 70)
    print("Input: 1% quality tolerance, 50% cost savings")
    
    profile = OptimizationProfile.from_reference(
        quality_tolerance=0.01,  # 1% quality drop
        cost_savings=0.50        # 50% cost reduction
    )
    
    print(f"Result: {profile}")
    print(f"  w_q = {profile['w_q']:.1f}")
    print(f"  w_c = {profile['w_c']:.1f}")
    print(f"  w_l = {profile['w_l']:.1f}")
    print("Interpretation: EXTREMELY sensitive to quality drops (50x cost weight)")
    print()
    
    # Validate
    assert profile['w_q'] == 50.0, f"Expected w_q=50.0, got {profile['w_q']}"
    assert profile['w_c'] == 1.0, f"Expected w_c=1.0, got {profile['w_c']}"
    assert profile['w_l'] == 0.0, f"Expected w_l=0.0, got {profile['w_l']}"
    print("✓ Test passed!\n")


def test_budget_user():
    """Scenario B: The Budget User
    
    "I can accept a 20% drop in quality if it saves me 90% of the cost."
    
    Expected: w_q = 4.5, w_c = 1.0, w_l = 0.0
    """
    print("=" * 70)
    print("Scenario B: The Budget User")
    print("=" * 70)
    print("Input: 20% quality tolerance, 90% cost savings")
    
    profile = OptimizationProfile.from_reference(
        quality_tolerance=0.20,  # 20% quality drop
        cost_savings=0.90        # 90% cost reduction
    )
    
    print(f"Result: {profile}")
    print(f"  w_q = {profile['w_q']:.1f}")
    print(f"  w_c = {profile['w_c']:.1f}")
    print(f"  w_l = {profile['w_l']:.1f}")
    print("Interpretation: Quality still matters (4.5x), but router has room to pick cheaper models")
    print()
    
    # Validate
    assert profile['w_q'] == 4.5, f"Expected w_q=4.5, got {profile['w_q']}"
    assert profile['w_c'] == 1.0, f"Expected w_c=1.0, got {profile['w_c']}"
    assert profile['w_l'] == 0.0, f"Expected w_l=0.0, got {profile['w_l']}"
    print("✓ Test passed!\n")


def test_speed_matters():
    """Scenario C: Speed Matters
    
    "I can lose 10% quality for 50% cost savings, but also want 30% faster responses."
    
    Expected: w_q = 5.0, w_c = 1.0, w_l = 3.0
    """
    print("=" * 70)
    print("Scenario C: Speed Matters")
    print("=" * 70)
    print("Input: 10% quality tolerance, 50% cost savings, 30% latency savings")
    
    profile = OptimizationProfile.from_reference(
        quality_tolerance=0.10,
        cost_savings=0.50,
        latency_savings=0.30
    )
    
    print(f"Result: {profile}")
    print(f"  w_q = {profile['w_q']:.1f}")
    print(f"  w_c = {profile['w_c']:.1f}")
    print(f"  w_l = {profile['w_l']:.1f}")
    print("Interpretation: Quality 5x cost, latency 3x cost")
    print()
    
    # Validate
    assert profile['w_q'] == 5.0, f"Expected w_q=5.0, got {profile['w_q']}"
    assert profile['w_c'] == 1.0, f"Expected w_c=1.0, got {profile['w_c']}"
    assert abs(profile['w_l'] - 3.0) < 1e-10, f"Expected w_l≈3.0, got {profile['w_l']}"
    print("✓ Test passed!\n")


def test_integration_with_get():
    """Test that from_reference() output works with OptimizationProfile.get()"""
    print("=" * 70)
    print("Integration Test: from_reference() -> get()")
    print("=" * 70)
    
    # Create a profile using from_reference
    custom_profile = OptimizationProfile.from_reference(
        quality_tolerance=0.05,
        cost_savings=0.50
    )
    
    print(f"Custom profile from from_reference: {custom_profile}")
    
    # Pass it to get() (should pass through as-is)
    resolved = OptimizationProfile.get(custom_profile)
    
    print(f"After passing through get(): {resolved}")
    print()
    
    # Validate pass-through
    assert resolved == custom_profile, "get() should pass through dict unchanged"
    assert 'w_q' in resolved and 'w_c' in resolved and 'w_l' in resolved
    print("✓ Integration test passed!\n")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("=" * 70)
    print("Edge Case Tests")
    print("=" * 70)
    
    # Test 1: Very small quality tolerance (should use min threshold)
    print("Test 1: Very small quality tolerance (0.0001)")
    profile = OptimizationProfile.from_reference(
        quality_tolerance=0.0001,
        cost_savings=0.50
    )
    # Should use max(0.0001, 0.001) = 0.001 → w_q = 0.50/0.001 = 500
    print(f"  Result: w_q={profile['w_q']:.1f} (should be 500.0)")
    assert profile['w_q'] == 500.0, f"Expected w_q=500.0, got {profile['w_q']}"
    print("  ✓ Passed")
    
    # Test 2: Invalid negative quality tolerance
    print("\nTest 2: Negative quality tolerance (should raise ValueError)")
    try:
        OptimizationProfile.from_reference(
            quality_tolerance=-0.05,
            cost_savings=0.50
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 3: Invalid negative cost savings
    print("\nTest 3: Negative cost savings (should raise ValueError)")
    try:
        OptimizationProfile.from_reference(
            quality_tolerance=0.05,
            cost_savings=-0.50
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing OptimizationProfile.from_reference()")
    print("=" * 70 + "\n")
    
    test_arbitrageur()
    test_budget_user()
    test_speed_matters()
    test_integration_with_get()
    test_edge_cases()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
