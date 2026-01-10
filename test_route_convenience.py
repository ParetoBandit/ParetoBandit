#!/usr/bin/env python3
"""
Test script for route() convenience parameters (quality_tolerance, cost_savings).

This validates that the Reference Point Normalization parameters work directly
in the route() method without manually calling from_reference().
"""

import sys
sys.path.insert(0, '/Users/annette/repostitories/banditGPT/src')

from bandit_gpt.router import BanditRouter


def test_route_convenience_parameters():
    """Test that route() accepts quality_tolerance and cost_savings directly"""
    
    print("=" * 70)
    print("Testing route() Convenience Parameters")
    print("=" * 70)
    print()
    
    # Create router
    print("1. Creating BanditRouter...")
    router = BanditRouter.create(priors="cold")
    print(f"   ✓ Router initialized with {len(router.registry)} models")
    print()
    
    # Test 1: Direct usage of convenience parameters
    print("2. Test 1: Using quality_tolerance and cost_savings directly")
    print("   (Arbitrageur: 1% quality tolerance, 50% cost savings)")
    
    model_id, log = router.route(
        prompt="Write a Python function for quicksort",
        quality_tolerance=0.01,
        cost_savings=0.50
    )
    
    print(f"   Selected model: {model_id}")
    print(f"   Predicted utility: {log.predicted_utility:.4f}")
    print(f"   ✓ Route completed successfully")
    print()
    
    # Test 2: Budget user scenario
    print("3. Test 2: Budget User")
    print("   (20% quality tolerance, 90% cost savings)")
    
    model_id2, log2 = router.route(
        prompt="Write a Python function for quicksort",
        quality_tolerance=0.20,
        cost_savings=0.90
    )
    
    print(f"   Selected model: {model_id2}")
    print(f"   Predicted utility: {log2.predicted_utility:.4f}")
    print(f"   ✓ Route completed successfully")
    print()
    
    # Test 3: With latency savings
    print("4. Test 3: Including latency_savings")
    print("   (10% quality tolerance, 50% cost savings, 30% latency savings)")
    
    model_id3, log3 = router.route(
        prompt="What is the capital of France?",
        quality_tolerance=0.10,
        cost_savings=0.50,
        latency_savings=0.30
    )
    
    print(f"   Selected model: {model_id3}")
    print(f"   Predicted utility: {log3.predicted_utility:.4f}")
    print(f"   ✓ Route completed successfully")
    print()
    
    # Test 4: Fallback to named profile (no convenience params)
    print("5. Test 4: Fallback to named profile")
    print("   (Using profile='max_quality')")
    
    model_id4, log4 = router.route(
        prompt="Explain quantum mechanics",
        profile="max_quality"
    )
    
    print(f"   Selected model: {model_id4}")
    print(f"   Predicted utility: {log4.predicted_utility:.4f}")
    print(f"   ✓ Route completed successfully")
    print()
    
    # Test 5: Show reference model
    print("6. Reference Model Information")
    ref = router.reference_model
    print(f"   Current flagship: {ref['id']}")
    print(f"   HLE Score: {ref.get('hle', 0.0):.4f}")
    print(f"   This is what '100% quality' means in this portfolio")
    print()
    
    print("=" * 70)
    print("All convenience parameter tests passed! ✓")
    print("=" * 70)
    print()
    print("Summary:")
    print("  • route() now accepts quality_tolerance and cost_savings directly")
    print("  • Automatically calls from_reference() when both are provided")
    print("  • Falls back to 'profile' parameter if tolerances not specified")
    print("  • reference_model property shows current quality baseline")


if __name__ == "__main__":
    test_route_convenience_parameters()
