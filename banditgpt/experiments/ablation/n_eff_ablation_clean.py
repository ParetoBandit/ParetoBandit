#!/usr/bin/env python3
"""
Clean N_eff Ablation Test: CSR vs HLE Priors

Tests whether task-specific Cluster Success Rate (CSR) priors outperform
generic Hard Label Evaluation (HLE) priors when both are normalized to
have equivalent prior strength.

Configuration:
- prior_n_effective = 40 (moderate prior strength)
- prior_structure_n_effective = 20 (fixed covariance strength)
- Test prompts: 500 (to see learning behavior)
- Metric: Cumulative regret over time

Expected Result:
CSR should show lower cumulative regret than HLE because it has
task-specific cluster information, even though both have equal prior strength.
"""

import sys
from pathlib import Path
import json
import numpy as np
import random
from collections import defaultdict

# Add parent to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from banditgpt import BanditRouter

def load_test_data():
    """Load test rewards from banditgpt/data"""
    test_rewards_path = repo_root / "banditgpt" / "data" / "test_rewards_pareto_dedup.jsonl"
    
    rewards_data = []
    with open(test_rewards_path) as f:
        for line in f:
            rewards_data.append(json.loads(line))
    
    # Build ground truth dictionary
    prompt_to_rewards = defaultdict(dict)
    for entry in rewards_data:
        if entry.get("ok"):
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            score = entry["raw_score"]
            prompt_to_rewards[prompt][model_id] = score
    
    prompts = list(prompt_to_rewards.keys())
    ground_truth = {p: prompt_to_rewards[p] for p in prompts}
    
    return prompts, ground_truth

def run_simulation(router, prompts, ground_truth, label="Router"):
    """
    Run bandit simulation and track cumulative regret over time
    
    Returns:
        cumulative_regrets: List of cumulative regret after each prompt
        final_regret: Total cumulative regret
    """
    cumulative_regret = 0.0
    cumulative_regrets = []
    
    for i, prompt in enumerate(prompts):
        # Get router's choice
        selected_model, log = router.route(prompt, profile="balanced", input_tokens=100)
        
        # Calculate regret
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        actual_reward = true_rewards.get(selected_model, 0.0)
        regret = best_reward - actual_reward
        
        cumulative_regret += regret
        cumulative_regrets.append(cumulative_regret)
        
        # Update router with feedback
        router.process_feedback(log.request_id, actual_reward)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  {label}: {i+1}/{len(prompts)} prompts, cumulative regret: {cumulative_regret:.1f}")
    
    return cumulative_regrets, cumulative_regret

def main():
    print("=" * 70)
    print("N_EFF ABLATION: CSR vs HLE Priors (Clean Test)")
    print("=" * 70)
    
    # Configuration
    N_EFF = 40
    STRUCTURE_N_EFF = 20
    NUM_PROMPTS = 500
    SEED = 42
    
    print(f"\nConfiguration:")
    print(f"  prior_n_effective: {N_EFF}")
    print(f"  prior_structure_n_effective: {STRUCTURE_N_EFF}")
    print(f"  Test prompts: {NUM_PROMPTS}")
    print(f"  Random seed: {SEED}")
    
    # Load data
    print(f"\n[1/4] Loading test data...")
    prompts, ground_truth = load_test_data()
    
    # Select and shuffle prompts
    random.seed(SEED)
    selected_prompts = prompts[:NUM_PROMPTS]
    random.shuffle(selected_prompts)
    
    print(f"  Loaded {len(prompts)} total prompts")
    print(f"  Using {len(selected_prompts)} prompts for test")
    print(f"  Models per prompt: {len(next(iter(ground_truth.values())))}")
    
    # Load registry
    print(f"\n[2/4] Loading model registry...")
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    print(f"  Loaded {len(registry)} models")
    
    # Create routers
    print(f"\n[3/4] Creating routers...")
    print(f"  CSR: Task-specific cluster success rates")
    csr_router = BanditRouter.create(
        registry,
        priors="csr",
        prior_n_effective=float(N_EFF),
        prior_structure_n_effective=float(STRUCTURE_N_EFF),
        exploration="safe"
    )
    
    print(f"  HLE: Generic hard label evaluation scores")
    hle_router = BanditRouter.create(
        registry,
        priors="hle",
        prior_n_effective=float(N_EFF),
        prior_structure_n_effective=float(STRUCTURE_N_EFF),
        exploration="safe"
    )
    
    # Verify normalization
    sample_model = list(registry.keys())[10]  # Use model with HLE data
    csr_b_norm = np.linalg.norm(csr_router.bandit.b[sample_model])
    hle_b_norm = np.linalg.norm(hle_router.bandit.b[sample_model])
    ratio = csr_b_norm / max(hle_b_norm, 1e-10)
    
    print(f"\n  Normalization check (sample model: {sample_model.split('/')[-1]}):")
    print(f"    CSR b norm: {csr_b_norm:.4f}")
    print(f"    HLE b norm: {hle_b_norm:.4f}")
    print(f"    Ratio: {ratio:.2f}x")
    
    if ratio > 10:
        print(f"    ⚠️  Warning: Ratio > 10x, this model may have low HLE score")
    else:
        print(f"    ✓ Normalized priors have similar magnitude")
    
    # Run simulations
    print(f"\n[4/4] Running simulations...")
    print(f"\nCSR Router:")
    csr_regrets, csr_final = run_simulation(csr_router, selected_prompts, ground_truth, "CSR")
    
    print(f"\nHLE Router:")
    hle_regrets, hle_final = run_simulation(hle_router, selected_prompts, ground_truth, "HLE")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nFinal Cumulative Regret (after {NUM_PROMPTS} prompts):")
    print(f"  CSR: {csr_final:.1f}")
    print(f"  HLE: {hle_final:.1f}")
    print(f"  Difference: {abs(csr_final - hle_final):.1f}")
    
    improvement = ((hle_final - csr_final) / max(hle_final, 1)) * 100
    print(f"  CSR improvement: {improvement:+.1f}%")
    
    # Check regret at different points
    print(f"\nRegret at key milestones:")
    for milestone in [100, 250, 500]:
        if milestone <= len(csr_regrets):
            idx = milestone - 1
            print(f"  After {milestone} prompts:")
            print(f"    CSR: {csr_regrets[idx]:.1f}")
            print(f"    HLE: {hle_regrets[idx]:.1f}")
            print(f"    Gap: {abs(csr_regrets[idx] - hle_regrets[idx]):.1f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if improvement > 5:
        print(f"\n✅ SUCCESS: CSR shows {improvement:.1f}% improvement over HLE")
        print("   Task-specific cluster priors are more effective than generic HLE")
    elif improvement > 0:
        print(f"\n⚠️  MARGINAL: CSR shows {improvement:.1f}% improvement")
        print("   Benefit exists but is smaller than expected")
    elif improvement > -5:
        print(f"\n⚠️  EQUIVALENT: CSR and HLE perform similarly ({improvement:+.1f}%)")
        print("   May need more prompts or different test conditions")
    else:
        print(f"\n❌ UNEXPECTED: HLE outperforms CSR by {-improvement:.1f}%")
        print("   This suggests a potential issue with CSR implementation")
    
    print(f"\nNote: With binary rewards (0/1), many prompts have all models")
    print(f"succeeding (score=1), resulting in 0 regret regardless of choice.")
    print(f"The advantage of CSR is most visible on harder prompts where")
    print(f"cluster-specific knowledge helps select the right model.")

if __name__ == "__main__":
    main()
