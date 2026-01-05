#!/usr/bin/env python3
"""
2D Grid Search: Find Optimal (structure_N, prior_N) for CSR and HLE

Performs grid search over both knobs:
- prior_structure_n_effective (A matrix strength)
- prior_n_effective (b vector strength)

Goal: Find optimal settings for CSR and HLE separately, then compare
at their respective optimal points for fair comparison.
"""

import sys
from pathlib import Path
import json
import numpy as np
import random
from collections import defaultdict
from itertools import product

repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from banditgpt import BanditRouter

def load_test_data(num_prompts=500):
    """Load test rewards"""
    test_rewards_path = repo_root / "banditgpt" / "data" / "test_rewards_pareto_dedup.jsonl"
    
    rewards_data = []
    with open(test_rewards_path) as f:
        for line in f:
            rewards_data.append(json.loads(line))
    
    prompt_to_rewards = defaultdict(dict)
    for entry in rewards_data:
        if entry.get("ok"):
            prompt_to_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]
    
    prompts = list(prompt_to_rewards.keys())[:num_prompts]
    ground_truth = {p: prompt_to_rewards[p] for p in prompts}
    
    return prompts, ground_truth

def run_trial(router, prompts, ground_truth):
    """Run single trial and return cumulative regret"""
    cumulative_regret = 0.0
    
    for prompt in prompts:
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
    print("2D GRID SEARCH: Optimal (structure_N, prior_N) for CSR vs HLE")
    print("=" * 70)
    
    # Grid configuration
    STRUCTURE_N_VALUES = [5, 10, 20, 40]  # Covariance strength
    PRIOR_N_VALUES = [0, 10, 20, 40, 60]  # Prior mean strength
    NUM_PROMPTS = 500  # Faster for grid search
    SEED = 42
    
    print(f"\nConfiguration:")
    print(f"  prior_structure_n_effective: {STRUCTURE_N_VALUES}")
    print(f"  prior_n_effective: {PRIOR_N_VALUES}")
    print(f"  Grid size: {len(STRUCTURE_N_VALUES)} × {len(PRIOR_N_VALUES)} = {len(STRUCTURE_N_VALUES) * len(PRIOR_N_VALUES)} points")
    print(f"  Prompts per trial: {NUM_PROMPTS}")
    
    # Load data
    print(f"\n[1/4] Loading test data...")
    prompts, ground_truth = load_test_data(NUM_PROMPTS)
    print(f"  Loaded {len(prompts)} prompts")
    
    # Shuffle once
    random.seed(SEED)
    random.shuffle(prompts)
    
    # Load registry
    print(f"\n[2/4] Loading model registry...")
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    print(f"  Loaded {len(registry)} models")
    
    # Grid search
    print(f"\n[3/4] Running grid search...")
    
    results_csr = {}
    results_hle = {}
    
    for structure_n, prior_n in product(STRUCTURE_N_VALUES, PRIOR_N_VALUES):
        print(f"\n  Testing (structure={structure_n}, prior={prior_n}):")
        
        # CSR
        csr_router = BanditRouter.create(
            registry,
            priors="csr",
            prior_n_effective=float(prior_n),
            prior_structure_n_effective=float(structure_n),
            exploration="safe"
        )
        csr_regret = run_trial(csr_router, prompts, ground_truth)
        results_csr[(structure_n, prior_n)] = csr_regret
        
        # HLE
        hle_router = BanditRouter.create(
            registry,
            priors="hle",
            prior_n_effective=float(prior_n),
            prior_structure_n_effective=float(structure_n),
            exploration="safe"
        )
        hle_regret = run_trial(hle_router, prompts, ground_truth)
        results_hle[(structure_n, prior_n)] = hle_regret
        
        print(f"    CSR: {csr_regret:.1f}, HLE: {hle_regret:.1f}")
    
    # Find optimal points
    print(f"\n[4/4] Finding optimal parameters...")
    
    optimal_csr = min(results_csr.items(), key=lambda x: x[1])
    optimal_hle = min(results_hle.items(), key=lambda x: x[1])
    
    csr_structure, csr_prior = optimal_csr[0]
    hle_structure, hle_prior = optimal_hle[0]
    
    # Results
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    
    print(f"\nOptimal CSR Configuration:")
    print(f"  prior_structure_n_effective: {csr_structure}")
    print(f"  prior_n_effective: {csr_prior}")
    print(f"  Cumulative regret: {optimal_csr[1]:.1f}")
    
    print(f"\nOptimal HLE Configuration:")
    print(f"  prior_structure_n_effective: {hle_structure}")
    print(f"  prior_n_effective: {hle_prior}")
    print(f"  Cumulative regret: {optimal_hle[1]:.1f}")
    
    improvement = ((optimal_hle[1] - optimal_csr[1]) / optimal_hle[1]) * 100
    print(f"\nCSR improvement at optimal points: {improvement:+.1f}%")
    
    # Heatmaps
    print(f"\n" + "=" * 70)
    print("CSR REGRET HEATMAP")
    print("=" * 70)
    print(f"\n{'prior_N →':>10s}", end="")
    for pn in PRIOR_N_VALUES:
        print(f"{pn:>8d}", end="")
    print()
    print("struct_N ↓" + "-" * (10 + 8 * len(PRIOR_N_VALUES)))
    
    for sn in STRUCTURE_N_VALUES:
        print(f"{sn:>10d}", end="")
        for pn in PRIOR_N_VALUES:
            regret = results_csr[(sn, pn)]
            marker = " *" if (sn, pn) == optimal_csr[0] else "  "
            print(f"{regret:>6.1f}{marker}", end="")
        print()
    
    print(f"\n" + "=" * 70)
    print("HLE REGRET HEATMAP")
    print("=" * 70)
    print(f"\n{'prior_N →':>10s}", end="")
    for pn in PRIOR_N_VALUES:
        print(f"{pn:>8d}", end="")
    print()
    print("struct_N ↓" + "-" * (10 + 8 * len(PRIOR_N_VALUES)))
    
    for sn in STRUCTURE_N_VALUES:
        print(f"{sn:>10d}", end="")
        for pn in PRIOR_N_VALUES:
            regret = results_hle[(sn, pn)]
            marker = " *" if (sn, pn) == optimal_hle[0] else "  "
            print(f"{regret:>6.1f}{marker}", end="")
        print()
    
    # Save results
    output_path = Path(__file__).parent / "grid_search_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "config": {
                "structure_n_values": STRUCTURE_N_VALUES,
                "prior_n_values": PRIOR_N_VALUES,
                "num_prompts": NUM_PROMPTS
            },
            "results_csr": {f"{k[0]},{k[1]}": v for k, v in results_csr.items()},
            "results_hle": {f"{k[0]},{k[1]}": v for k, v in results_hle.items()},
            "optimal_csr": {
                "structure_n": csr_structure,
                "prior_n": csr_prior,
                "regret": optimal_csr[1]
            },
            "optimal_hle": {
                "structure_n": hle_structure,
                "prior_n": hle_prior,
                "regret": optimal_hle[1]
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == "__main__":
    main()
