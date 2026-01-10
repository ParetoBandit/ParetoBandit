#!/usr/bin/env python3
"""
Integration test: Verify from_reference() works with BanditRouter.route()

This demonstrates real-world usage of the Reference Point Normalization feature.
"""

import sys
sys.path.insert(0, '/Users/annette/repostitories/banditGPT/src')

from bandit_gpt.router import BanditRouter, OptimizationProfile


def test_router_integration():
    """Test that from_reference() profiles work with router.route()"""
    
    print("=" * 70)
    print("Integration Test: from_reference() with BanditRouter.route()")
    print("=" * 70)
    print()
    
    # Create a minimal router instance
    print("1. Creating BanditRouter instance...")
    router = BanditRouter.create(
        priors="cold",  # Cold start for quick initialization
        alpha=0.1
    )
    print(f"   Router initialized with {len(router.registry)} models")
    print()
    
    # Create a custom profile using from_reference
    print("2. Creating custom profile: Arbitrageur")
    print("   (1% quality tolerance, 50% cost savings)")
    arbitrage_profile = OptimizationProfile.from_reference(
        quality_tolerance=0.01,
        cost_savings=0.50
    )
    print(f"   Profile: {arbitrage_profile}")
    print()
    
    # Test routing with this profile
    print("3. Testing route() with custom profile...")
    test_prompt = "Write a Python function to calculate Fibonacci numbers"
    
    try:
        model_id, log = router.route(
            prompt=test_prompt,
            profile=arbitrage_profile  # Pass the dict directly
        )
        
        print(f"   ✓ Routing successful!")
        print(f"   Selected model: {model_id}")
        print(f"   Predicted utility: {log.predicted_utility:.4f}")
        print(f"   Context vector dimension: {len(log.context_vector)}")
        print()
        
    except Exception as e:
        print(f"   ✗ Routing failed: {e}")
        raise
    
    # Test with Budget User profile
    print("4. Testing Budget User profile")
    print("   (20% quality tolerance, 90% cost savings)")
    budget_profile = OptimizationProfile.from_reference(
        quality_tolerance=0.20,
        cost_savings=0.90
    )
    print(f"   Profile: {budget_profile}")
    
    model_id2, log2 = router.route(
        prompt=test_prompt,
        profile=budget_profile
    )
    
    print(f"   ✓ Routing successful!")
    print(f"   Selected model: {model_id2}")
    print(f"   Predicted utility: {log2.predicted_utility:.4f}")
    print()
    
    # Compare the two
    print("5. Comparison:")
    print(f"   Arbitrageur chose: {model_id}")
    print(f"   Budget User chose: {model_id2}")
    
    if model_id == model_id2:
        print(f"   → Both selected the same model (common with cold start)")
    else:
        print(f"   ✓ Different models selected based on weight preferences")
    
    print()
    print("=" * 70)
    print("Integration test completed successfully! ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_router_integration()
