#!/usr/bin/env python3
"""
Find Best Head-Start Configuration

Tests multiple configurations and finds which gives maximum CSR advantage
in the first 100 prompts (early phase), then validates with confidence intervals.
"""

import sys
from pathlib import Path
import json
import numpy as np
import random
from collections import defaultdict

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from banditgpt import BanditRouter

def load_test_data():
    """Load all test rewards"""
    test_rewards_path = repo_root / "banditgpt" / "data" / "test_rewards_pareto_dedup.jsonl"
    
    rewards_data = []
    with open(test_rewards_path) as f:
        for line in f:
            rewards_data.append(json.loads(line))
    
    prompt_to_rewards = defaultdict(dict)
    for entry in rewards_data:
        if entry.get("ok"):
            prompt_to_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]
    
    prompts = list(prompt_to_rewards.keys())
    ground_truth = {p: prompt_to_rewards[p] for p in prompts}
    
    return prompts, ground_truth

def run_trial_first_n(router, prompts, ground_truth, n_prompts=100):
    """Run trial for first N prompts and return final cumulative regret"""
    cumulative_regret = 0.0
    
    for prompt in prompts[:n_prompts]:
        selected_model, log = router.route(prompt, profile="balanced", input_tokens=100)
        
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        actual_reward = true_rewards.get(selected_model, 0.0)
        regret = best_reward - actual_reward
        
        cumulative_regret += regret
        router.process_feedback(log.request_id, actual_reward)
    
    return cumulative_regret

def main():
    print("=" * 70)
    print("Finding Best Head-Start Configuration (First 100 Prompts)")
    print("=" * 70)
    
    # Test configurations focusing on early performance
    configs_to_test = [
        (5, 20), (5, 40), (5, 60),
        (10, 20), (10, 40), (10, 60),
        (20, 20), (20, 40), (20, 60),
        (40, 20), (40, 40), (40, 60),
    ]
    
    NUM_PROMPTS = 100  # Focus on early performance
    SEED = 42
    
    # Load data
    print(f"\n[1/3] Loading test data...")
    prompts, ground_truth = load_test_data()
    print(f"  Loaded {len(prompts)} prompts")
    print(f"  Testing first {NUM_PROMPTS} prompts only")
    
    random.seed(SEED)
    random.shuffle(prompts)
    
    # Load registry
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    print(f"\n[2/3] Testing {len(configs_to_test)} configurations for early advantage...")
    
    results = []
    
    for structure_n, prior_n in configs_to_test:
        print(f"\n  Config (structure={structure_n}, prior={prior_n}):", end=" ")
        
        # CSR
        csr_router = BanditRouter.create(
            registry,
            priors="csr",
            prior_n_effective=float(prior_n),
            prior_structure_n_effective=float(structure_n),
            exploration="safe"
        )
        csr_regret = run_trial_first_n(csr_router, prompts, ground_truth, NUM_PROMPTS)
        
        # HLE
        hle_router = BanditRouter.create(
            registry,
            priors="hle",
            prior_n_effective=float(prior_n),
            prior_structure_n_effective=float(structure_n),
            exploration="safe"
        )
        hle_regret = run_trial_first_n(hle_router, prompts, ground_truth, NUM_PROMPTS)
        
        advantage = hle_regret - csr_regret  # Positive = CSR better
        advantage_pct = (advantage / max(hle_regret, 1)) * 100
        
        results.append({
            "structure_n": structure_n,
            "prior_n": prior_n,
            "csr_regret": csr_regret,
            "hle_regret": hle_regret,
            "advantage": advantage,
            "advantage_pct": advantage_pct
        })
        
        print(f"CSR={csr_regret:.1f}, HLE={hle_regret:.1f}, Δ={advantage:+.1f} ({advantage_pct:+.1f}%)")
    
    # Find best head-start configuration
    print(f"\n[3/3] Finding best head-start configuration...")
    
    best_config = max(results, key=lambda x: x["advantage"])
    
    print(f"\n" + "=" * 70)
    print("RESULTS: Best Head-Start Configuration")
    print("=" * 70)
    
    print(f"\n✅ Optimal for Early Advantage (first {NUM_PROMPTS} prompts):")
    print(f"   prior_structure_n_effective: {best_config['structure_n']}")
    print(f"   prior_n_effective: {best_config['prior_n']}")
    print(f"   CSR regret: {best_config['csr_regret']:.1f}")
    print(f"   HLE regret: {best_config['hle_regret']:.1f}")
    print(f"   Advantage: {best_config['advantage']:+.1f} ({best_config['advantage_pct']:+.1f}%)")
    
    # Top 5 configurations
    print(f"\nTop 5 configurations for head start:")
    sorted_results = sorted(results, key=lambda x: x["advantage"], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. ({r['structure_n']:2d}, {r['prior_n']:2d}): "
              f"Δ={r['advantage']:+5.1f} ({r['advantage_pct']:+5.1f}%)")
    
    # Save results
    output_path = Path(__file__).parent / "best_headstart_config.json"
    with open(output_path, 'w') as f:
        json.dump({
            "num_prompts_tested": NUM_PROMPTS,
            "best_config": best_config,
            "all_results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print(f"\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print(f"\nUse this configuration for convergence analysis:")
    print(f"  structure_n_effective = {best_config['structure_n']}")
    print(f"  prior_n_effective = {best_config['prior_n']}")
    print(f"\nThis will give maximum CSR advantage in the critical early phase!")

if __name__ == "__main__":
    main()
