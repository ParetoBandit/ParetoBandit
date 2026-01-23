import numpy as np
import os
import sys
import time
from collections import deque

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bandit_gpt.router import BanditRouter, RouterConfig

def test_persistent_probation_bonus():
    """
    Verifies that the probation bonus does NOT reappear after log eviction.
    """
    # 1. Setup with small log size to trigger eviction quickly
    config = RouterConfig()
    config.max_log_size = 10 
    config.probation_bonus = 0.5
    config.pruning_min_samples = 5
    
    router = BanditRouter.create(config=config)
    model_id = "meta-llama/llama-3.1-8b-instruct"
    
    # Ensure model is in the router
    if model_id not in router.registry:
        router.registry[model_id] = {"openrouter_id": model_id, "cost_per_1m_tokens": 0.1, "initial_quality": 0.8}
        router.bandit.add_arm(model_id)

    # 2. Add some requests and feedback to "graduate" the model
    print(f"--- Phase 1: Graduating {model_id} ---")
    x = np.random.randn(router.bandit.dim)
    x /= np.linalg.norm(x) # Normalize for stable utilities
    for i in range(10):
        # Manually perform the work of process_feedback to graduate the model
        router.bandit.update(model_id, x, reward=1.0)
        router.model_counts[model_id] += 1
        
    # Check count
    count = router.model_counts[model_id]
    print(f"Model count: {count} (Goal: 10)")
    assert count >= 10
    
    # Verify NO probation bonus is applied anymore
    # Use alpha_scale=0 to focus on (mean + probation_bonus)
    _, utility, _ = router._score_candidates([model_id], x, w_q=1.0, w_c=0.0, w_l=0.0, alpha_scale=0.0, input_tokens=100, output_tokens=100)
    print(f"Graduate Utility: {utility:.4f}")
    # With unit vector x and 10 updates of reward 1.0, mean_quality ~1.0
    # No probation bonus (+0.5) should be present.

    # 3. Evict logs by flooding with other models
    print(f"--- Phase 2: Flooding logs to evict {model_id} ---")
    # Add other models
    other_models = ["model_a", "model_b"]
    for m in other_models:
        router.registry[m] = {"openrouter_id": m, "cost_per_1m_tokens": 1.0}
        router.bandit.add_arm(m)

    for i in range(20): # More than max_log_size (10)
        # route() creates logs and adds them to self.logs, triggering eviction
        router.route(prompt="other prompt", profile="auto")
        
    # Verify log eviction
    log_models = [log.selected_model for log in router.logs]
    print(f"Models in logs: {len(log_models)} entries")
    assert model_id not in log_models
    
    # 4. CRITICAL CHECK: Does the mature model get a bonus again?
    print(f"--- Phase 3: Checking if {model_id} regains bonus ---")
    # Persistent count should still be 10
    print(f"Model count after log eviction: {router.model_counts[model_id]}")
    assert router.model_counts[model_id] == 10
    
    _, utility_after_eviction, _ = router._score_candidates([model_id], x, w_q=1.0, w_c=0.0, w_l=0.0, alpha_scale=0.0, input_tokens=100, output_tokens=100)
    print(f"Utility after eviction: {utility_after_eviction:.4f}")
    
    # If the bug were present, utility would be utility + 0.5 (probation bonus)
    # because _get_sample_counts() would have returned 0 from the evicted logs.
    assert utility_after_eviction == utility
    print("✅ SUCCESS: Probation bonus did not reappear after log eviction.")
    print("✅ SUCCESS: Probation bonus did not reappear after log eviction.")

if __name__ == "__main__":
    test_persistent_probation_bonus()
