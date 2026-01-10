#!/usr/bin/env python3
"""
Test script for updated Reference Point Normalization-based named profiles.

Validates that the new MAX_QUALITY, ARBITRAGE, BEST_VALUE, and COST_SAVER
profiles use the correct weights derived from the exchange rate formula.
"""

import sys
sys.path.insert(0, '/Users/annette/repostitories/banditGPT/src')

from bandit_gpt.router import OptimizationProfile, BanditRouter


def test_updated_profiles():
    """Test that named profiles match Reference Point Normalization logic"""
    
    print("=" * 70)
    print("Testing Updated Named Optimization Profiles")
    print("=" * 70)
    print()
    
    # Test 1: MAX_QUALITY
    print("1. MAX_QUALITY Profile")
    print("   Logic: 'Only 1% quality drop if it's FREE (100% savings)'")
    print("   Math: 1.00 / 0.01 = 100")
    
    max_qual = OptimizationProfile.get("max_quality")
    print(f"   Weights: {max_qual}")
    assert max_qual["w_q"] == 100.0, f"Expected w_q=100.0, got {max_qual['w_q']}"
    assert max_qual["w_c"] == 1.0, f"Expected w_c=1.0, got {max_qual['w_c']}"
    print("   ✓ Validated")
    print()
    
    # Test 2: ARBITRAGE
    print("2. ARBITRAGE Profile")
    print("   Logic: 'Flagship intelligence without brand prices (2.5% drop, 90% savings)'")
    print("   Math: 0.90 / 0.025 = 36")
    
    arb = OptimizationProfile.get("arbitrage")
    print(f"   Weights: {arb}")
    assert arb["w_q"] == 36.0, f"Expected w_q=36.0, got {arb['w_q']}"
    assert arb["w_c"] == 1.0, f"Expected w_c=1.0, got {arb['w_c']}"
    print("   ✓ Validated")
    print()
    
    # Test 3: BEST_VALUE
    print("3. BEST_VALUE Profile")
    print("   Logic: 'Solid trade-off (5% quality drop for 50% cost savings)'")
    print("   Math: 0.50 / 0.05 = 10")
    
    best = OptimizationProfile.get("best_value")
    print(f"   Weights: {best}")
    assert best["w_q"] == 10.0, f"Expected w_q=10.0, got {best['w_q']}"
    assert best["w_c"] == 1.0, f"Expected w_c=1.0, got {best['w_c']}"
    print("   ✓ Validated")
    print()
    
    # Test 4: COST_SAVER
    print("4. COST_SAVER Profile")
    print("   Logic: 'Strict budget (25% quality drop for 90% savings)'")
    print("   Math: 0.90 / 0.25 = 3.6")
    
    cost = OptimizationProfile.get("cost_saver")
    print(f"   Weights: {cost}")
    assert cost["w_q"] == 3.6, f"Expected w_q=3.6, got {cost['w_q']}"
    assert cost["w_c"] == 1.0, f"Expected w_c=1.0, got {cost['w_c']}"
    print("   ✓ Validated")
    print()
    
    # Test 5: Verify equivalence with from_reference
    print("5. Equivalence Test: Named profiles vs from_reference()")
    
    # MAX_QUALITY: quality_tolerance=0.01, cost_savings=1.00
    max_ref = OptimizationProfile.from_reference(
        quality_tolerance=0.01,
        cost_savings=1.00
    )
    assert max_ref["w_q"] == max_qual["w_q"], "MAX_QUALITY mismatch"
    print("   ✓ MAX_QUALITY ≡ from_reference(0.01, 1.00)")
    
    # ARBITRAGE: quality_tolerance=0.025, cost_savings=0.90
    arb_ref = OptimizationProfile.from_reference(
        quality_tolerance=0.025,
        cost_savings=0.90
    )
    assert arb_ref["w_q"] == arb["w_q"], "ARBITRAGE mismatch"
    print("   ✓ ARBITRAGE ≡ from_reference(0.025, 0.90)")
    
    # BEST_VALUE: quality_tolerance=0.05, cost_savings=0.50
    best_ref = OptimizationProfile.from_reference(
        quality_tolerance=0.05,
        cost_savings=0.50
    )
    assert best_ref["w_q"] == best["w_q"], "BEST_VALUE mismatch"
    print("   ✓ BEST_VALUE ≡ from_reference(0.05, 0.50)")
    
    # COST_SAVER: quality_tolerance=0.25, cost_savings=0.90
    cost_ref = OptimizationProfile.from_reference(
        quality_tolerance=0.25,
        cost_savings=0.90
    )
    assert cost_ref["w_q"] == cost["w_q"], "COST_SAVER mismatch"
    print("   ✓ COST_SAVER ≡ from_reference(0.25, 0.90)")
    print()
    
    # Test 6: Integration with router
    print("6. Integration Test: Using updated profiles with router")
    router = BanditRouter.create(priors="cold")
    
    for profile_name in ["max_quality", "arbitrage", "best_value", "cost_saver"]:
        model_id, log = router.route(
            prompt="Explain machine learning",
            profile=profile_name
        )
        print(f"   {profile_name:12s} → {model_id:30s} (utility: {log.predicted_utility:6.2f})")
    
    print()
    print("=" * 70)
    print("All named profile tests passed! ✓")
    print("=" * 70)
    print()
    print("Profile Summary (Pareto Frontier):")
    print(f"  MAX_QUALITY:  w_q={max_qual['w_q']:5.1f}  (Premium: demand perfection)")
    print(f"  ARBITRAGE:    w_q={arb['w_q']:5.1f}  (Smart: flagship quality, budget price)")
    print(f"  BEST_VALUE:   w_q={best['w_q']:5.1f}  (Balanced: solid trade-off)")
    print(f"  COST_SAVER:   w_q={cost['w_q']:5.1f}  (Budget: strict cost constraint)")


if __name__ == "__main__":
    test_updated_profiles()
