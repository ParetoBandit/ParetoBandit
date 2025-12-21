#!/usr/bin/env python3
"""
5-Fold Cross-Validation: Cold Start vs. Warm Start (Dense Priors)

This script performs a robust statistical evaluation using 5-fold CV.
For each fold, it:
1.  Splits data into Train (80%) and Test (20%).
2.  Generates fresh dense priors using ONLY the Train split.
3.  Evaluates Cold vs. Warm start on the Test split.
4.  Aggregates results to report Mean Regret Reduction ± CI and p-value.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from tqdm import tqdm

from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path
from experiments.generate_expert_priors import generate_dense_priors

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_regret_cv")


def load_all_data() -> Tuple[List[dict], List[dict]]:
    """Load and merge all available archetype and reward data."""
    prompts = []
    rewards = []
    
    # Files to merge
    files = [
        ("train_archetypes.jsonl", "train_rewards.jsonl"),
        ("test_archetypes.jsonl", "test_rewards.jsonl"),
    ]
    
    for p_file, r_file in files:
        p_path = get_priors_path(p_file)
        r_path = get_priors_path(r_file)
        
        with open(p_path) as f:
            for line in f:
                prompts.append(json.loads(line))
                
        with open(r_path) as f:
            for line in f:
                rewards.append(json.loads(line))
                
    return prompts, rewards


def write_temp_split(prompts: List[dict], rewards: List[dict], indices: np.ndarray, p_path: Path, r_path: Path):
    """Write a subset of data to temporary files."""
    # Create a set of cluster_ids in this split for fast filtering
    valid_clusters = set()
    with open(p_path, "w") as f:
        for idx in indices:
            item = prompts[idx]
            f.write(json.dumps(item) + "\n")
            valid_clusters.add(item["cluster_id"])
            
    with open(r_path, "w") as f:
        for item in rewards:
            if item["cluster_id"] in valid_clusters:
                f.write(json.dumps(item) + "\n")


def run_simulation(
    policy: DisjointLinUCBPolicy,
    embeddings: np.ndarray,
    cluster_ids: List[int],
    truth: Dict[int, Dict[str, float]],
    model_names: List[str],
) -> List[float]:
    """Run simulation and return cumulative regret."""
    cumulative_regret = []
    total_regret = 0.0
    
    for i in range(len(cluster_ids)):
        x = embeddings[i]
        cluster = cluster_ids[i]
        cluster_rewards = truth.get(cluster, {})
        
        # Oracle
        best_reward = 0.0
        for m in model_names:
            r = cluster_rewards.get(m, 0.0)
            if r > best_reward:
                best_reward = r
        
        # Agent
        chosen_model, _, _ = policy.select_arm(x)
        observed_reward = cluster_rewards.get(chosen_model, 0.0)
        
        # Update
        policy.update(chosen_model, x, observed_reward)
        
        # Regret
        regret = best_reward - observed_reward
        total_regret += regret
        cumulative_regret.append(total_regret)
        
    return cumulative_regret


def main():
    print("=" * 60)
    print("5-Fold Cross-Validation: Prior Strength Sweep")
    print("=" * 60)
    
    # 1. Load All Data
    all_prompts, all_rewards = load_all_data()
    print(f"Total Prompts: {len(all_prompts)}")
    
    # Extract text for embedding
    prompt_texts = [p["prompt"] for p in all_prompts]
    cluster_ids_all = [p["cluster_id"] for p in all_prompts]
    
    # Embed everything once
    print("Embedding all prompts...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings_all = encoder.encode(prompt_texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings_all = np.asarray(embeddings_all, dtype=np.float64)
    dim = embeddings_all.shape[1]
    
    # Build truth dictionary for fast lookup
    truth_all = {}
    model_set = set()
    for item in all_rewards:
        if item.get("ok", False):
            m = item["model_id"]
            c = item["cluster_id"]
            logit = item.get("reward_logit", 0.0)
            r = 1.0 / (1.0 + np.exp(-logit))
            if c not in truth_all:
                truth_all[c] = {}
            truth_all[c][m] = r
            model_set.add(m)
    model_names = sorted(model_set)
    
    # 2. K-Fold CV with Sweep
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    prior_strengths = [1.0, 5.0, 10.0, 25.0, 50.0]
    
    sweep_results = {}  # strength -> list of reductions
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_prompts)):
            print(f"\n[Fold {fold+1}/5]")
            
            # Paths for this fold
            train_p_path = temp_path / f"train_p_{fold}.jsonl"
            train_r_path = temp_path / f"train_r_{fold}.jsonl"
            priors_path = temp_path / f"priors_{fold}.npz"
            
            # Write Train Split
            write_temp_split(all_prompts, all_rewards, train_idx, train_p_path, train_r_path)
            
            # Generate Priors (Train Only)
            print(f"  Generating priors from {len(train_idx)} samples...")
            generate_dense_priors(
                prompts_path=train_p_path,
                rewards_path=train_r_path,
                output_path=priors_path,
                alpha=0.5,
                seed=42 + fold,
            )
            
            # Prepare Test Data
            test_embeddings = embeddings_all[test_idx]
            test_clusters = [cluster_ids_all[i] for i in test_idx]
            
            # Load Priors Base
            priors = np.load(priors_path, allow_pickle=True)
            A_stack = priors["A_stack"]
            b_stack = priors["b_stack"]
            prior_models = priors["model_names"]
            model_to_idx = {m: i for i, m in enumerate(prior_models)}
            
            # Evaluate Cold Start (Once per fold)
            cold_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
            cold_curve = run_simulation(cold_policy, test_embeddings, test_clusters, truth_all, model_names)
            final_cold = cold_curve[-1]
            
            # Evaluate Warm Start for each Strength
            for strength in prior_strengths:
                if strength not in sweep_results:
                    sweep_results[strength] = []
                
                warm_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
                
                # Inflate
                for m in model_names:
                    if m in model_to_idx:
                        idx = model_to_idx[m]
                        warm_policy.A[m] = A_stack[idx].astype(np.float64) * strength
                        warm_policy.b[m] = b_stack[idx].astype(np.float64) * strength
                        warm_policy.A_inv[m] = np.linalg.inv(warm_policy.A[m])
                
                warm_curve = run_simulation(warm_policy, test_embeddings, test_clusters, truth_all, model_names)
                final_warm = warm_curve[-1]
                
                red = (final_cold - final_warm) / final_cold * 100
                sweep_results[strength].append(red)
                print(f"    Strength={strength}: Red={red:.1f}%")

    # 3. Statistical Analysis & Plotting
    print("\n" + "=" * 60)
    print("Prior Strength Sweep Results")
    print("=" * 60)
    
    means = []
    cis = []
    
    for strength in prior_strengths:
        reds = np.array(sweep_results[strength])
        mean_red = np.mean(reds)
        se_red = np.std(reds, ddof=1) / np.sqrt(len(reds))
        ci_95 = stats.t.ppf(0.975, len(reds)-1) * se_red
        
        means.append(mean_red)
        cis.append(ci_95)
        
        print(f"Strength {strength:4.1f}: {mean_red:6.2f}% ± {ci_95:.2f}%")
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(prior_strengths, means, yerr=cis, fmt='-o', capsize=5, linewidth=2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel("Prior Strength (λ)")
    plt.ylabel("Mean Regret Reduction (%)")
    plt.title("Impact of Prior Strength on Regret Reduction (5-Fold CV)")
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(prior_strengths, [str(s) for s in prior_strengths])
    
    output_dir = Path("results/cv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prior_strength_sweep.png")
    print(f"\nSaved plot to {output_dir / 'prior_strength_sweep.png'}")


if __name__ == "__main__":
    main()
