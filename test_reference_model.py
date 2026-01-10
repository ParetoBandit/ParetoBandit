#!/usr/bin/env python3
"""
Test script for BanditRouter.reference_model property.

This validates that the reference_model property correctly identifies the
flagship model (highest HLE score) for Reference Point Normalization.
"""

import sys
sys.path.insert(0, '/Users/annette/repostitories/banditGPT/src')

from bandit_gpt.router import BanditRouter


def test_reference_model():
    """Test that reference_model identifies the highest HLE model"""
    
    print("=" * 70)
    print("Testing BanditRouter.reference_model Property")
    print("=" * 70)
    print()
    
    # Create router with default registry
    print("1. Creating BanditRouter with default registry...")
    router = BanditRouter.create(priors="cold")
    print(f"   Router loaded with {len(router.registry)} models")
    print()
    
    # Get the reference model
    print("2. Identifying reference model (highest HLE)...")
    ref = router.reference_model
    
    print(f"   Reference Model ID: {ref['id']}")
    print(f"   HLE Score: {ref.get('hle', 'N/A'):.4f}")
    print(f"   Input Cost: ${ref.get('input_cost_per_m', 0.0):.4f}/1M tokens")
    print(f"   Output Cost: ${ref.get('output_cost_per_m', 0.0):.4f}/1M tokens")
    print()
    
    # Verify it's actually the highest
    print("3. Verifying it's the model with highest HLE score...")
    all_hle_scores = {
        model_id: data.get('hle', 0.0) or 0.0
        for model_id, data in router.registry.items()
    }
    
    max_hle = max(all_hle_scores.values())
    expected_id = max(all_hle_scores, key=all_hle_scores.get)
    
    print(f"   Maximum HLE in registry: {max_hle:.4f}")
    print(f"   Expected model ID: {expected_id}")
    print(f"   Actual reference ID: {ref['id']}")
    
    assert ref['id'] == expected_id, f"Reference model mismatch!"
    assert ref.get('hle', 0.0) == max_hle, f"HLE score mismatch!"
    print("   ✓ Verified!")
    print()
    
    # Show top 5 models by HLE
    print("4. Top 5 models by HLE score:")
    sorted_models = sorted(
        all_hle_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for i, (model_id, hle) in enumerate(sorted_models, 1):
        marker = "← REFERENCE" if model_id == ref['id'] else ""
        print(f"   {i}. {model_id}: {hle:.4f} {marker}")
    
    print()
    print("=" * 70)
    print("reference_model property test passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_reference_model()
