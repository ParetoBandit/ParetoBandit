#!/usr/bin/env python3
"""
5-Fold Cross-Validation: Cold Start vs. Global Prior (Task-Specific)

This script implements the "Task-Specific Global Prior" strategy:
1.  Compute empirical mean reward for each model from the TRAINING split.
2.  Train the bandit on a synthetic dataset where target = global_mean.
    -   This forces the bandit to learn a "flat" prior at the level of the mean.
3.  Evaluate on TEST split.

This approach ignores noisy context and relies on the strong global signal.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_regret_global")


def load_all_data() -> Tuple[List[dict], List[dict]]:
    """Load and merge all available archetype and reward data."""
    prompts = []
    rewards = []
    
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


def generate_global_priors(
    prompts: List[dict],
    rewards: List[dict],
    embeddings: np.ndarray,
    model_names: List[str],
    output_path: Path,
    dim: int,
    alpha: float = 0.5,
    seed: int = 42,
):
    """
    Generate priors that encode the GLOBAL MEAN reward for each model.
    """
    # 1. Compute Global Mean per Model
    model_sums = {m: 0.0 for m in model_names}
    model_counts = {m: 0 for m in model_names}
    
    # Map cluster -> {model -> reward}
    truth = {}
    for item in rewards:
        if item.get("ok", False):
            c = item["cluster_id"]
            m = item["model_id"]
            logit = item.get("reward_logit", 0.0)
            r = 1.0 / (1.0 + np.exp(-logit))
            if c not in truth:
                truth[c] = {}
            truth[c][m] = r

    for p in prompts:
        c = p["cluster_id"]
        cluster_rewards = truth.get(c, {})
        for m, r in cluster_rewards.items():
            if m in model_sums:
                model_sums[m] += r
                model_counts[m] += 1
                
    global_means = {}
    for m in model_names:
        if model_counts[m] > 0:
            global_means[m] = model_sums[m] / model_counts[m]
        else:
            global_means[m] = 0.5  # Default
            
    # 2. Train Bandit on Synthetic "Global" Data
    # We update every model with its global mean for every prompt in the training set.
    # This forces the Ridge Regression to find the best constant approximation.
    
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=alpha)
    
    # To ensure the prior is truly "global" and robust to context,
    # we update using the actual training contexts but with the CONSTANT global mean target.
    for i in range(len(prompts)):
        x = embeddings[i]
        for m in model_names:
            target = global_means[m]
            policy.update(m, x, target)
            
    # 3. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    A_stack = np.stack([policy.A[m] for m in model_names], axis=0)
    b_stack = np.stack([policy.b[m] for m in model_names], axis=0)
    
    np.savez_compressed(
        output_path,
        model_names=np.array(model_names, dtype=object),
        dim=dim,
        alpha=alpha,
        A_stack=A_stack.astype(np.float16),
        b_stack=b_stack.astype(np.float16),
        seed=seed,
    )


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
    print("5-Fold CV: Task-Specific Global Prior")
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
    
    # 2. K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "cold_regret": [],
        "global_regret": [],
        "reduction": []
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_prompts)):
            print(f"\n[Fold {fold+1}/5]")
            
            priors_path = temp_path / f"global_priors_{fold}.npz"
            
            # Prepare Train Data
            train_prompts = [all_prompts[i] for i in train_idx]
            train_embeddings = embeddings_all[train_idx]
            
            # Generate Global Priors (Train Only)
            print(f"  Generating Global Priors from {len(train_idx)} samples...")
            generate_global_priors(
                prompts=train_prompts,
                rewards=all_rewards, # Pass all rewards, function filters by cluster
                embeddings=train_embeddings,
                model_names=model_names,
                output_path=priors_path,
                dim=dim,
                alpha=0.5,
                seed=42 + fold,
            )
            
            # Prepare Test Data
            test_embeddings = embeddings_all[test_idx]
            test_clusters = [cluster_ids_all[i] for i in test_idx]
            
            # Initialize Agents
            cold_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
            
            global_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
            priors = np.load(priors_path, allow_pickle=True)
            
            # Inflate Global Policy
            # We use a moderate prior strength to enforce the mean but allow adaptation
            prior_strength = 20.0 
            A_stack = priors["A_stack"]
            b_stack = priors["b_stack"]
            prior_models = priors["model_names"]
            model_to_idx = {m: i for i, m in enumerate(prior_models)}
            
            for m in model_names:
                if m in model_to_idx:
                    idx = model_to_idx[m]
                    global_policy.A[m] = A_stack[idx].astype(np.float64) * prior_strength
                    global_policy.b[m] = b_stack[idx].astype(np.float64) * prior_strength
                    global_policy.A_inv[m] = np.linalg.inv(global_policy.A[m])
            
            # Run Simulation
            print(f"  Evaluating on {len(test_idx)} test samples...")
            cold_curve = run_simulation(cold_policy, test_embeddings, test_clusters, truth_all, model_names)
            global_curve = run_simulation(global_policy, test_embeddings, test_clusters, truth_all, model_names)
            
            final_cold = cold_curve[-1]
            final_global = global_curve[-1]
            red = (final_cold - final_global) / final_cold * 100
            
            results["cold_regret"].append(final_cold)
            results["global_regret"].append(final_global)
            results["reduction"].append(red)
            
            print(f"  Result: Cold={final_cold:.1f}, Global={final_global:.1f}, Red={red:.1f}%")

    # 3. Statistical Analysis
    print("\n" + "=" * 60)
    print("5-Fold CV Results (Global Prior)")
    print("=" * 60)
    
    cold_arr = np.array(results["cold_regret"])
    global_arr = np.array(results["global_regret"])
    red_arr = np.array(results["reduction"])
    
    mean_red = np.mean(red_arr)
    std_red = np.std(red_arr, ddof=1)
    se_red = std_red / np.sqrt(len(red_arr))
    ci_95 = stats.t.ppf(0.975, len(red_arr)-1) * se_red
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(cold_arr, global_arr)
    
    print(f"Mean Regret Reduction: {mean_red:.2f}% ± {ci_95:.2f}% (95% CI)")
    print(f"P-value (paired t-test): {p_val:.5f}")
    if p_val < 0.05:
        print("Result is STATISTICALLY SIGNIFICANT.")
    else:
        print("Result is NOT statistically significant.")
        
    print("-" * 60)
    print(f"Cold Regret:   {np.mean(cold_arr):.1f} ± {np.std(cold_arr):.1f}")
    print(f"Global Regret: {np.mean(global_arr):.1f} ± {np.std(global_arr):.1f}")
    print("=" * 60)
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([cold_arr, global_arr], labels=["Cold Start", "Global Prior"])
    plt.ylabel("Cumulative Regret")
    plt.title(f"Global Prior vs Cold Start (p={p_val:.4f})")
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("results/cv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "global_prior_cv.png")
    print(f"Saved plot to {output_dir / 'global_prior_cv.png'}")


if __name__ == "__main__":
    main()
