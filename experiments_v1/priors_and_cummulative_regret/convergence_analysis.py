#!/usr/bin/env python3
"""
CSR vs HLE: Time-Series Convergence with Confidence Intervals

Runs multiple trials to compute mean and 95% confidence intervals,
showing statistical significance of early advantage.
"""

import sys
from pathlib import Path
import json
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

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

def run_trial_with_tracking(router, prompts, ground_truth):
    """Run trial and return cumulative regret at each timestep"""
    cumulative_regrets = []
    cumulative_regret = 0.0
    
    for prompt in prompts:
        selected_model, log = router.route(prompt, profile="balanced", input_tokens=100)
        
        true_rewards = ground_truth[prompt]
        best_reward = max(true_rewards.values())
        actual_reward = true_rewards.get(selected_model, 0.0)
        regret = best_reward - actual_reward
        
        cumulative_regret += regret
        cumulative_regrets.append(cumulative_regret)
        
        router.process_feedback(log.request_id, actual_reward)
    
    return np.array(cumulative_regrets)

def main():
    print("=" * 70)
    print("CSR vs HLE: Convergence with Confidence Intervals")
    print("=" * 70)
    
    # Configuration - using CSR optimal from grid search
    STRUCTURE_N = 40
    PRIOR_N = 60
    NUM_TRIALS = 3  # Multiple trials for confidence intervals
    
    print(f"\nConfiguration:")
    print(f"  prior_structure_n_effective: {STRUCTURE_N}")
    print(f"  prior_n_effective: {PRIOR_N}")
    print(f"  Number of trials: {NUM_TRIALS}")
    
    # Load data
    print(f"\n[1/3] Loading test data...")
    prompts, ground_truth = load_test_data()
    print(f"  Loaded {len(prompts)} prompts")
    
    # Load registry
    models_path = repo_root / "banditgpt" / "models.json"
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    print(f"\n[2/3] Running {NUM_TRIALS} trials...")
    
    csr_trials = []
    hle_trials = []
    
    for trial in range(NUM_TRIALS):
        print(f"\n  Trial {trial + 1}/{NUM_TRIALS}:")
        
        # Shuffle with different seed each trial
        random.seed(42 + trial)
        shuffled = prompts.copy()
        random.shuffle(shuffled)
        
        # CSR
        csr_router = BanditRouter.create(
            registry,
            priors="csr",
            prior_n_effective=float(PRIOR_N),
            prior_structure_n_effective=float(STRUCTURE_N),
            exploration="safe"
        )
        csr_regrets = run_trial_with_tracking(csr_router, shuffled, ground_truth)
        csr_trials.append(csr_regrets)
        
        # HLE
        hle_router = BanditRouter.create(
            registry,
            priors="hle",
            prior_n_effective=float(PRIOR_N),
            prior_structure_n_effective=float(STRUCTURE_N),
            exploration="safe"
        )
        hle_regrets = run_trial_with_tracking(hle_router, shuffled, ground_truth)
        hle_trials.append(hle_regrets)
        
        print(f"    CSR final: {csr_regrets[-1]:.1f}, HLE final: {hle_regrets[-1]:.1f}")
    
    # Compute statistics
    print(f"\n[3/3] Computing statistics...")
    
    csr_trials = np.array(csr_trials)  # Shape: (NUM_TRIALS, 981)
    hle_trials = np.array(hle_trials)
    
    csr_mean = np.mean(csr_trials, axis=0)
    csr_std = np.std(csr_trials, axis=0)
    csr_ci = 1.96 * csr_std / np.sqrt(NUM_TRIALS)  # 95% CI
    
    hle_mean = np.mean(hle_trials, axis=0)
    hle_std = np.std(hle_trials, axis=0)
    hle_ci = 1.96 * hle_std / np.sqrt(NUM_TRIALS)
    
    gap_mean = hle_mean - csr_mean
    
    # Analysis
    print(f"\n" + "=" * 70)
    print("EARLY ADVANTAGE ANALYSIS (with 95% CI)")
    print("=" * 70)
    
    milestones = [50, 100, 250, 500, 981]
    for m in milestones:
        idx = m - 1
        csr_val = csr_mean[idx]
        hle_val = hle_mean[idx]
        gap = gap_mean[idx]
        gap_pct = (gap / max(hle_val, 1)) * 100
        
        print(f"\n@ {m:4d} prompts:")
        print(f"  CSR: {csr_val:6.1f} ± {csr_ci[idx]:.1f}")
        print(f"  HLE: {hle_val:6.1f} ± {hle_ci[idx]:.1f}")
        print(f"  Gap: {gap:+6.1f} ({gap_pct:+5.1f}%)")
    
    # Plot with confidence intervals
    print(f"\n[4/4] Generating plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    x = np.arange(len(csr_mean))
    
    # Plot 1: Cumulative regret with CI bands
    ax1.plot(x, csr_mean, label='CSR Mean', linewidth=2, color='blue')
    ax1.fill_between(x, csr_mean - csr_ci, csr_mean + csr_ci, alpha=0.3, color='blue', label='CSR 95% CI')
    
    ax1.plot(x, hle_mean, label='HLE Mean', linewidth=2, color='red')
    ax1.fill_between(x, hle_mean - hle_ci, hle_mean + hle_ci, alpha=0.3, color='red', label='HLE 95% CI')
    
    ax1.axvline(x=100, linestyle='--', color='gray', alpha=0.5)
    ax1.axvline(x=500, linestyle='--', color='gray', alpha=0.5)
    ax1.set_xlabel('Number of Prompts', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title(f'CSR vs HLE: Convergence Over Time ({NUM_TRIALS} trials)\nConfig: structure={STRUCTURE_N}, prior={PRIOR_N}', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Advantage gap
    ax2.plot(x, gap_mean, linewidth=2, color='green', label='Mean Gap')
    ax2.axhline(y=0, linestyle='-', color='black', linewidth=0.8)
    ax2.fill_between(x, gap_mean, 0, where=(gap_mean>0), alpha=0.3, color='green')
    ax2.fill_between(x, gap_mean, 0, where=(gap_mean<0), alpha=0.3, color='red')
    ax2.axvline(x=100, linestyle='--', color='gray', alpha=0.5, label='Early (100)')
    ax2.axvline(x=500, linestyle='--', color='gray', alpha=0.5, label='Mid (500)')
    ax2.set_xlabel('Number of Prompts', fontsize=12)
    ax2.set_ylabel('Regret Gap (HLE - CSR)', fontsize=12)
    ax2.set_title('CSR Advantage Over Time\n(Positive = CSR better)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    output_path = Path(__file__).parent / "convergence_analysis_ci.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_path}")
    
    # Save results
    json_path = Path(__file__).parent / "convergence_ci_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "config": {"structure_n": STRUCTURE_N, "prior_n": PRIOR_N, "num_trials": NUM_TRIALS},
            "csr_mean": csr_mean.tolist(),
            "csr_ci": csr_ci.tolist(),
            "hle_mean": hle_mean.tolist(),
            "hle_ci": hle_ci.tolist(),
            "gap_mean": gap_mean.tolist(),
            "milestones": {
                str(m): {
                    "csr": float(csr_mean[m-1]),
                    "csr_ci": float(csr_ci[m-1]),
                    "hle": float(hle_mean[m-1]),
                    "hle_ci": float(hle_ci[m-1]),
                    "gap": float(gap_mean[m-1])
                } for m in milestones
            }
        }, f, indent=2)
    print(f"  ✓ Data saved to: {json_path}")
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    early_gap = gap_mean[99]
    mid_gap = gap_mean[499]
    final_gap = gap_mean[-1]
    
    print(f"\nCSR Advantage (Regret Saved):")
    print(f"  Early (100): {early_gap:+.1f} ± {csr_ci[99] + hle_ci[99]:.1f}")
    print(f"  Mid (500): {mid_gap:+.1f} ± {csr_ci[499] + hle_ci[499]:.1f}")
    print(f"  Final (981): {final_gap:+.1f} ± {csr_ci[-1] + hle_ci[-1]:.1f}")
    
    print(f"\nInterpretation:")
    if early_gap > 5:
        print(f"  ✅ CSR provides significant cold-start advantage")
    if abs(final_gap) < 5:
        print(f"  ✅ Both converge after sufficient learning")
    
    print(f"  📊 Total savings from better head start: {early_gap:.1f} regret units")

if __name__ == "__main__":
    main()
