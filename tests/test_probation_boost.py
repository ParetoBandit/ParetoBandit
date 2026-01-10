#!/usr/bin/env python3
"""
Test: Probation Boost Against "Initial Bad Luck" Trap

Verifies that the probation bonus mechanism overcomes the scenario where
a model receives 3 consecutive hard prompts, fails, and then never gets
selected again despite having potential.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from bandit_gpt import BanditRouter


def test_probation_boost_overcomes_bad_luck():
    """
    Simulate "initial bad luck" trap and verify probation boost enables recovery.
    
    Scenario:
    - Model A: Gets 3 easy prompts, succeeds (reward=1.0)
    - Model B: Gets 3 hard prompts, fails (reward=0.0)
    - Without probation boost: Model B never selected again
    - With probation boost: Model B gets rediscovered and can recover
    """
    print("=" * 70)
    print("PROBATION BOOST TEST: Initial Bad Luck Recovery")
    print("=" * 70)
    
    # Create registry with two similar models
    registry = {
        "model_a": {
            "openrouter_id": "provider/model-a",
            "display_name": "Model A",
            "hle": 0.70,
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0
        },
        "model_b": {
            "openrouter_id": "provider/model-b",  
            "display_name": "Model B",
            "hle": 0.70,  # Same HLE score
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0
        }
    }
    
    # Create router with probation bonus enabled
    router = BanditRouter.create(model_registry=registry, priors="none")
    
    # Verify probation bonus is configured
    print(f"\nConfiguration:")
    print(f"  probation_bonus: {router.config.probation_bonus}")
    print(f"  pruning_min_samples: {router.config.pruning_min_samples}")
    
    # Phase 1: Give each model 3 initial samples
    print(f"\n{'Phase':<15} {'Model':<10} {'Reward':<8} {'Description':<30}")
    print("-" * 70)
    
    # Model A: Gets 3 easy prompts (succeeds)
    for i in range(3):
        model, log = router.route("Easy task", candidates=None)
        # Force selection to model_a for controlled test
        if i < 3:
            # Manually route to ensure we test both models
            test_model = "model_a"
            model, log = router.route(f"Easy task {i}")
            if model != test_model:
                # Override for test
                log.selected_model = test_model
                model = test_model
            
            router.process_feedback(log.request_id, reward=1.0)  # Success
            print(f"{'Initial (easy)':<15} {test_model:<10} {1.0:<8.1f} {'Success on easy prompt':<30}")
    
    # Model B: Gets 3 hard prompts (fails)  
    for i in range(3):
        test_model = "model_b"
        model, log = router.route(f"Hard task {i}")
        if model != test_model:
            # Override for test
            log.selected_model = test_model
            model = test_model
            
        router.process_feedback(log.request_id, reward=0.0)  # Failure
        print(f"{'Initial (hard)':<15} {test_model:<10} {0.0:<8.1f} {'Failure on hard prompt':<30}")
    
    # Phase 2: Route 50 times and track model B recovery
    print(f"\n{'='*70}")
    print("Phase 2: Recovery Check (50 requests)")
    print("="*70)
    
    model_b_selections = 0
    total_requests = 50
    
    for i in range(total_requests):
        model, log = router.route(f"Normal task {i}")
        if model == "model_b":
            model_b_selections += 1
            # Give model_b fair rewards to show it can recover
            router.process_feedback(log.request_id, reward=0.7)
        else:
            router.process_feedback(log.request_id, reward=0.7)
    
    model_b_percentage = (model_b_selections / total_requests) * 100
    
    print(f"\nResults:")
    print(f"  Model B selected: {model_b_selections}/{total_requests} ({model_b_percentage:.1f}%)")
    print(f"  Model A selected: {total_requests - model_b_selections}/{total_requests} ({100 - model_b_percentage:.1f}%)")
    
    # With probation boost, Model B should get rediscovered (>5% selection rate)
    # Without probation boost, it would be < 1% due to initial bad luck
    recovery_threshold = 5.0  # At least 5% selection rate indicates recovery
    
    if model_b_percentage > recovery_threshold:
        print(f"\n✅ PASS: Model B recovered from initial bad luck")
        print(f"  Probation boost enabled rediscovery ({model_b_percentage:.1f}% > {recovery_threshold}%)")
        return True
    else:
        print(f"\n❌ FAIL: Model B stuck in 'bad luck trap'")
        print(f"  Selection rate {model_b_percentage:.1f}% < {recovery_threshold}%")
        print(f"  Probation boost may be insufficient")
        return False


if __name__ == "__main__":
    success = test_probation_boost_overcomes_bad_luck()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Probation boost working correctly!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("⚠️  Probation boost may need tuning")
        print("=" * 70)
        sys.exit(1)
