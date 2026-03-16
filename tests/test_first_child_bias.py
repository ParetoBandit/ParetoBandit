"""
Test: First-Child Bias Correction

Validates that late-arriving models receive manual priors when
no suitable neighbor exists for bootstrapping.

Failure scenario:
1. Register model A with speed="balanced"
2. Register model B with speed="fast" (unrelated description)
   - Since no neighbor found, b_init = zeros
   - Manual prior (positive bias for "fast") was NOT applied
   - Result: Model started with no positive bias signal

Expected behavior:
1. Register model A with speed="balanced"
2. Register model B with speed="fast" (unrelated description)
   - No neighbor found, b_init = zeros
   - is_bootstrapped = False
   - Manual prior (positive bias for "fast") IS applied
   - Result: Model starts with correct positive bias
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from pareto_bandit.router import BanditRouter

def test_late_model_receives_manual_priors():
    """Test that late-arriving models receive T-shirt sizing priors when no neighbor exists."""
    
    # Create minimal registry for first model
    registry = {
        "financial-analyst-model": {
            "display_name": "Financial Analyst GPT - Expert in stocks, bonds, and derivatives",
            "cost_per_1m_tokens": 5.0,
            "median_latency_s": 2.0
        }
    }
    
    # Initialize router with the first model
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0,
    )
    
    # Get initial state for first model
    first_model_b = router.bandit.b["financial-analyst-model"].copy()
    print(f"✅ First model registered")
    print(f"   b vector bias term: {first_model_b[-1]:.4f}")
    
    # Now register a completely unrelated model with speed="fast"
    # This should NOT find a suitable neighbor (similarity < 0.5)
    print("\n📝 Registering second model with speed='fast'...")
    router.register_model(
        model_id="deepseek-v3",
        speed="fast",  # Should receive positive bias
        cost_usd=0.5,
        latency_s=0.8
    )
    
    # Check the b vector for the new model
    second_model_b = router.bandit.b["deepseek-v3"]
    bias_term = second_model_b[-1]  # Last element is the bias term
    
    print(f"\n✅ Second model registered")
    print(f"   b vector bias term: {bias_term:.4f}")
    print(f"   Expected: positive value from speed='fast' prior")
    
    # The bias term should be non-zero (from the "fast" speed profile)
    # RouterConfig.registration.fast_bias is typically positive (e.g., +2.0)
    expected_bias = router.config.registration.fast_bias * router.bandit.init_lambda
    
    print(f"\n🔍 Verification:")
    print(f"   Expected bias: {expected_bias:.4f}")
    print(f"   Actual bias:   {bias_term:.4f}")
    
    # Verification
    if abs(bias_term - expected_bias) < 0.01:
        print(f"\n✅ TEST PASSED: Late-arriving model received manual prior!")
        print(f"   The 'First-Child' bias issue is resolved.")
        return True
    else:
        print(f"\n❌ TEST FAILED: Manual prior was not applied correctly")
        print(f"   Difference: {abs(bias_term - expected_bias):.4f}")
        return False

if __name__ == "__main__":
    success = test_late_model_receives_manual_priors()
    sys.exit(0 if success else 1)
