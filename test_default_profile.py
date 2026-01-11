#!/usr/bin/env python3
"""
Verify that the default profile uses the updated Arbitrage weights.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bandit_gpt.router import BanditRouter, OptimizationProfile

def main():
    print("="*60)
    print("Verifying Default Profile Uses Updated Weights")
    print("="*60)
    
    # Test 1: Check ARBITRAGE profile weights
    print("\n1. Checking OptimizationProfile.ARBITRAGE weights...")
    arbitrage = OptimizationProfile.ARBITRAGE
    print(f"   ✓ ARBITRAGE = {arbitrage}")
    assert arbitrage == {"w_q": 0.80, "w_c": 0.20, "w_l": 0.00}, \
        f"Expected updated weights, got {arbitrage}"
    
    # Test 2: Verify route() method signature shows "arbitrage" as default
    print("\n2. Checking BanditRouter.route() default parameter...")
    import inspect
    sig = inspect.signature(BanditRouter.route)
    profile_param = sig.parameters['profile']
    print(f"   ✓ Default profile parameter: {profile_param.default}")
    assert profile_param.default == "arbitrage", \
        f"Expected 'arbitrage', got {profile_param.default}"
    
    # Test 3: Verify that calling route without profile uses arbitrage
    print("\n3. Verifying that default routing uses ARBITRAGE weights...")
    router = BanditRouter.create()
    
    # Mock the _resolve_utility_weights to capture what gets passed
    original_resolve = router._resolve_utility_weights
    captured_weights = {}
    
    def mock_resolve(profile, max_cost, max_latency):
        # Capture the profile before resolving
        captured_weights['profile'] = profile
        return original_resolve(profile, max_cost, max_latency)
    
    router._resolve_utility_weights = mock_resolve
    
    # Route without specifying profile (should use default "arbitrage")
    try:
        router.route("Test prompt")
    except:
        pass  # We don't care if it fails, we just want to see the profile
    
    print(f"   ✓ Default profile used: {captured_weights.get('profile')}")
    assert captured_weights.get('profile') == "arbitrage", \
        f"Expected 'arbitrage', got {captured_weights.get('profile')}"
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("Default profile uses updated Arbitrage weights:")
    print(f"   w_q: 0.80, w_c: 0.20, w_l: 0.00")
    print("="*60)

if __name__ == "__main__":
    main()
