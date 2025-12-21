#!/usr/bin/env python3
"""
5-Fold Cross-Validation: Cold Start vs. Math-500 Priors

This script evaluates using "Math-500" scores as priors.
Math-500 is a benchmark for mathematical reasoning.

Methodology:
1.  Load `banditgpt/data/models_cache_with_hle.json`.
2.  Extract `math_500` score for each model.
3.  Initialize the bandit with these scores as the prior mean.
4.  Evaluate on 5-Fold CV.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from tqdm import tqdm

from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path, get_package_data_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_regret_math500")


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


def load_math500_scores(model_names: List[str]) -> Dict[str, float]:
    """
    Load Math-500 scores from models_cache_with_hle.json.
    Returns dict: {model_id: score (0.0-1.0)}
    """
    # Use the file with HLE as it also contains math_500 and is the most up to date
    cache_path = get_package_data_dir() / "models_cache_with_hle.json"
    with open(cache_path) as f:
        data = json.load(f)
        
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    scores = {}
    print("\n[Math-500 Scores]")
    print(f"{'Model':<40} {'Math-500':<10}")
    print("-" * 55)
    
    for m in model_names:
        if m not in registry:
            scores[m] = 0.0 # Default low for unknown
            continue
            
        reg = registry[m]
        score = float(reg.get("math_500") or 0.0)
        
        scores[m] = score
        print(f"{m[:40]:<40} {score:.4f}")
        
    return scores


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
    print("5-Fold CV: Math-500 Priors")
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
    
    # Build truth dictionary
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
    
    # 2. Load Math-500 Scores
    math_scores = load_math500_scores(model_names)
    
    # 3. K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "cold_regret": [],
        "math_regret": [],
        "reduction": []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_prompts)):
        print(f"\n[Fold {fold+1}/5]")
        
        # Prepare Train Data (for grounding)
        train_embeddings = embeddings_all[train_idx]
        
        # Initialize Math Policy
        print(f"  Grounding Math-500 Priors on {len(train_idx)} samples...")
        math_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
        
        # Prior Strength
        prior_strength = 20.0 
        
        # Grounding Loop: Update with Math-500 score as target
        for i in range(len(train_idx)):
            x = train_embeddings[i]
            for m in model_names:
                target = math_scores[m]
                math_policy.update(m, x, target)
                
        # Apply Prior Strength Scaling
        for m in model_names:
            math_policy.A[m] *= prior_strength
            math_policy.b[m] *= prior_strength
            math_policy.A_inv[m] = np.linalg.inv(math_policy.A[m])
            
        # Prepare Test Data
        test_embeddings = embeddings_all[test_idx]
        test_clusters = [cluster_ids_all[i] for i in test_idx]
        
        # Initialize Cold Policy
        cold_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
        
        # Run Simulation
        print(f"  Evaluating on {len(test_idx)} test samples...")
        cold_curve = run_simulation(cold_policy, test_embeddings, test_clusters, truth_all, model_names)
        math_curve = run_simulation(math_policy, test_embeddings, test_clusters, truth_all, model_names)
        
        final_cold = cold_curve[-1]
        final_math = math_curve[-1]
        red = (final_cold - final_math) / final_cold * 100
        
        results["cold_regret"].append(final_cold)
        results["math_regret"].append(final_math)
        results["reduction"].append(red)
        
        print(f"  Result: Cold={final_cold:.1f}, Math={final_math:.1f}, Red={red:.1f}%")

    # 4. Statistical Analysis
    print("\n" + "=" * 60)
    print("5-Fold CV Results (Math-500 Prior)")
    print("=" * 60)
    
    cold_arr = np.array(results["cold_regret"])
    math_arr = np.array(results["math_regret"])
    red_arr = np.array(results["reduction"])
    
    mean_red = np.mean(red_arr)
    std_red = np.std(red_arr, ddof=1)
    se_red = std_red / np.sqrt(len(red_arr))
    ci_95 = stats.t.ppf(0.975, len(red_arr)-1) * se_red
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(cold_arr, math_arr)
    
    print(f"Mean Regret Reduction: {mean_red:.2f}% ± {ci_95:.2f}% (95% CI)")
    print(f"P-value (paired t-test): {p_val:.5f}")
    if p_val < 0.05:
        print("Result is STATISTICALLY SIGNIFICANT.")
    else:
        print("Result is NOT statistically significant.")
        
    print("-" * 60)
    print(f"Cold Regret: {np.mean(cold_arr):.1f} ± {np.std(cold_arr):.1f}")
    print(f"Math Regret: {np.mean(math_arr):.1f} ± {np.std(math_arr):.1f}")
    print("=" * 60)
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([cold_arr, math_arr], labels=["Cold Start", "Math-500 Prior"])
    plt.ylabel("Cumulative Regret")
    plt.title(f"Math-500 Prior vs Cold Start (p={p_val:.4f})")
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("results/cv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "math500_prior_cv.png")
    print(f"Saved plot to {output_dir / 'math500_prior_cv.png'}")


if __name__ == "__main__":
    main()
