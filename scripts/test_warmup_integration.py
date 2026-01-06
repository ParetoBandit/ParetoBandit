#!/usr/bin/env python3
"""
Quick test to verify the warmup loading mechanism works correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.bandit_gpt.router import BanditRouter

def test_warmup_fallback():
    """Test that warmup loading falls back gracefully when file doesn't exist."""
    print("Testing warmup loading (file doesn't exist yet)...")
    
    # This should fall back to cold start with a warning
    router = BanditRouter.create(
        priors="warmup",
        exploration="safe"
    )
    
    print(f"✓ Router created successfully")
    print(f"  Models loaded: {len(router.bandit.models)}")
    print(f"  Feature dimension: {router.bandit.dim}")
    
    # Verify it's a valid router by routing a test prompt
    test_prompt = "Write a Python function to sort a list"
    model_id, log = router.route(test_prompt, profile="best_value")
    
    print(f"✓ Routing works: selected '{model_id}'")
    print(f"  Context vector shape: {log.context_vector.shape}")
    
    return router

def test_hle_priors():
    """Test that HLE priors still work as expected."""
    print("\nTesting HLE priors (baseline)...")
    
    router = BanditRouter.create(
        priors="hle",
        exploration="safe"
    )
    
    print(f"✓ HLE router created successfully")
    print(f"  Models loaded: {len(router.bandit.models)}")
    
    return router

def test_cold_start():
    """Test cold start for comparison."""
    print("\nTesting cold start...")
    
    router = BanditRouter.create(
        priors="none",
        exploration="safe"
    )
    
    print(f"✓ Cold start router created successfully")
    print(f"  Models loaded: {len(router.bandit.models)}")
    
    return router

if __name__ == "__main__":
    print("=" * 60)
    print("Warmup System Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Warmup loading (should fallback gracefully)
        warmup_router = test_warmup_fallback()
        
        # Test 2: HLE priors (baseline)
        hle_router = test_hle_priors()
        
        # Test 3: Cold start (for comparison)
        cold_router = test_cold_start()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run: python scripts/generate_warmup.py")
        print("2. Re-run this test to verify warmup loading works")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
