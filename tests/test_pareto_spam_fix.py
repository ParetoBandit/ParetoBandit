"""
Test to verify the Pareto Spam vulnerability fix

This test validates that near-duplicate models (high embedding similarity)
are rejected by the Pareto filter, preventing "feature spam" attacks.

Attack Scenario (Pre-Fix):
1. Provider registers "GPT-4-Base" at $10.00/1M
2. Provider spams 100 variants:
   - "GPT-4-Base-v1" at $9.9999/1M
   - "GPT-4-Base-v2" at $9.9998/1M
   - ... (all near-identical, tiny price differences)
3. Bug: All pass Pareto check (u_existing > u_new + 0.05 fails)
4. Result: Registry flooded with near-duplicates

Expected Behavior (Post-Fix):
1. Provider registers "GPT-4-Base" at $10.00/1M
2. Provider tries to register near-duplicate
3. Novelty check detects similarity > 0.9
4. Model rejected with "Feature spam protection"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from bandit_gpt.router import BanditRouter

def test_feature_spam_protection():
    """Test that near-duplicate models are allowed when probation has room."""
    
    # Create a minimal registry with one model
    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "openrouter_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        }
    }
    
    # Initialize router with probation limit = 10 (has room)
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0
    )
    router.config.max_probation_models = 10  # Ensure we have room
    
    print("✅ Router initialized with 1 model")
    print(f"   Model: {list(router.registry.keys())[0]}")
    print(f"   Probation limit: {router.config.max_probation_models}")
    
    # Try to add a near-duplicate model (almost identical name, tiny price diff)
    spam_model = {
        "display_name": "GPT-4 Base Model for Complex Reasoning v1",  # Very similar
        "openrouter_id": "gpt-4-base-v1",
        "cost_per_1m_tokens": 9999.0,  # Only $0.001 cheaper
        "median_latency_s": 2.0,
        "initial_quality": 0.85
    }
    
    print(f"\n📝 Testing novelty check for near-duplicate model:")
    print(f"   Original: '{registry['gpt-4-base']['display_name']}'")
    print(f"   Spam:     '{spam_model['display_name']}'")
    print(f"   Price diff: ${(10000.0 - 9999.0)/1000.0:.4f}/1M")
    
    # Test the Pareto filter - should NOT reject because probation has room
    is_rejected = router._is_pareto_dominated(spam_model)
    
    print(f"\n🔍 Pareto Filter Result:")
    print(f"   Rejected: {is_rejected}")
    
    if not is_rejected:
        print(f"\n✅ TEST PASSED: Near-duplicate was allowed (probation has room)!")
        print(f"   The variation will compete in probation to prove its worth.")
        return True
    else:
        print(f"\n❌ TEST FAILED: Near-duplicate was rejected despite probation having room")
        print(f"   Expected: Allow through when probation capacity available")
        return False
    
def test_legitimate_new_model():
    """Test that genuinely different models are NOT rejected."""
    
    # Create a minimal registry with one model
    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "openrouter_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        }
    }
    
    # Initialize router
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0
    )
    
    print("\n" + "="*60)
    print("✅ Router initialized with 1 model (GPT-4)")
    
    # Try to add a legitimately different model
    legit_model = {
        "display_name": "Claude 3.5 Sonnet - Fast Creative Writing Assistant",  # Very different
        "openrouter_id": "claude-3.5-sonnet",
        "cost_per_1m_tokens": 3000.0,  # Much cheaper
        "median_latency_s": 1.0,
        "initial_quality": 0.80
    }
    
    print(f"\n📝 Testing novelty check for legitimate new model:")
    print(f"   Existing: '{registry['gpt-4-base']['display_name']}'")
    print(f"   New:      '{legit_model['display_name']}'")
    
    # Test the Pareto filter
    is_rejected = router._is_pareto_dominated(legit_model)
    
    print(f"\n🔍 Pareto Filter Result:")
    print(f"   Rejected: {is_rejected}")
    
    if not is_rejected:
        print(f"\n✅ TEST PASSED: Legitimate model was correctly accepted!")
        print(f"   Different models pass the novelty check.")
        return True
    else:
        print(f"\n❌ TEST FAILED: Legitimate model was incorrectly rejected")
        return False

def test_spam_rejection_when_probation_full():
    """Test that near-duplicate models ARE rejected when probation is full."""
    
    # Create a registry with multiple models already in probation
    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "openrouter_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        },
        "claude-3-opus": {
            "display_name": "Claude 3 Opus - Advanced Reasoning",
            "openrouter_id": "claude-3-opus",
            "cost_per_1m_tokens": 15000.0,
            "median_latency_s": 2.5,
            "initial_quality": 0.88
        }
    }
    
    # Initialize router with very small probation limit
    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0
    )
    router.config.max_probation_models = 1  # Very tight limit
    
    # Manually add both models to probation to fill it
    router.probation_models["gpt-4-base"] = {
        "start_t": 0,
        "status": "PROBATION",
        "immune_until": 500  # Still immune (t=0 < 500)
    }
    
    print("\n" + "="*60)
    print("✅ Router initialized with 2 models")
    print(f"   Probation limit: {router.config.max_probation_models}")
    print(f"   Current probation count: 1/1 (FULL)")
    
    # Try to add a near-duplicate model
    spam_model = {
        "display_name": "GPT-4 Base Model for Complex Reasoning Plus",  # Very similar
        "openrouter_id": "gpt-4-base-plus",
        "cost_per_1m_tokens": 9998.0,  # Slightly cheaper spam
        "median_latency_s": 2.0,
        "initial_quality": 0.85
    }
    
    print(f"\n📝 Testing spam rejection when probation is FULL:")
    print(f"   Original: '{registry['gpt-4-base']['display_name']}'")
    print(f"   Spam:     '{spam_model['display_name']}'")
    
    # Test the Pareto filter - SHOULD reject because probation is full
    is_rejected = router._is_pareto_dominated(spam_model)
    
    print(f"\n🔍 Pareto Filter Result:")
    print(f"   Rejected: {is_rejected}")
    
    if is_rejected:
        print(f"\n✅ TEST PASSED: Spam was rejected (probation full)!")
        print(f"   Feature spam protection is active when probation at capacity.")
        return True
    else:
        print(f"\n❌ TEST FAILED: Spam was allowed despite probation being full")
        print(f"   Expected: Reject when probation at capacity")
        return False

if __name__ == "__main__":
    success1 = test_feature_spam_protection()
    success2 = test_legitimate_new_model()
    success3 = test_spam_rejection_when_probation_full()
    
    if success1 and success2 and success3:
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED")
        print("   - Near-duplicates allowed when probation has room")
        print("   - Legitimate models are accepted")
        print("   - Spam blocked when probation is full")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("❌ SOME TESTS FAILED")
        sys.exit(1)

