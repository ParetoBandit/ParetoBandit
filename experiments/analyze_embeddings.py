#!/usr/bin/env python3
"""
Analyze Embedding Quality & Reward Smoothness

This script investigates why dense priors might be failing by analyzing:
1.  **Reward Smoothness**: Do similar prompts (in embedding space) have similar optimal models?
    -   We compute the "local consistency": for each point, what % of its k-nearest neighbors share the same optimal model?
2.  **Global vs. Local**: Does a simple "Global Best" prior outperform the "Contextual" prior?
    -   If Global > Contextual, the embeddings are adding noise.

Usage:
    python -m experiments.analyze_embeddings
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from banditgpt._resources import get_priors_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("analyze_embeddings")


def load_all_data():
    """Load all prompts and rewards."""
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


def get_optimal_models(prompts, rewards):
    """Get the optimal model ID for each prompt."""
    # Map cluster_id -> {model -> reward}
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
            
    optimal_models = []
    for p in prompts:
        c = p["cluster_id"]
        cluster_rewards = truth.get(c, {})
        if not cluster_rewards:
            optimal_models.append(None)
            continue
            
        best_model = max(cluster_rewards.items(), key=lambda x: x[1])[0]
        optimal_models.append(best_model)
        
    return optimal_models, truth


def analyze_smoothness(embeddings, optimal_models, k=5):
    """
    Compute local consistency: fraction of k-NN sharing the same optimal model.
    """
    print(f"\nAnalyzing Reward Smoothness (k={k})...")
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    consistencies = []
    
    for i in range(len(embeddings)):
        my_best = optimal_models[i]
        if my_best is None:
            continue
            
        matches = 0
        # Skip self (index 0)
        for neighbor_idx in indices[i][1:]:
            neighbor_best = optimal_models[neighbor_idx]
            if neighbor_best == my_best:
                matches += 1
        
        consistency = matches / k
        consistencies.append(consistency)
        
    avg_consistency = np.mean(consistencies)
    print(f"Average Local Consistency: {avg_consistency:.2%}")
    print(f"  (Random baseline would be ~1/{len(set(filter(None, optimal_models)))} ≈ 1-2%)")
    
    return consistencies


def evaluate_global_prior(prompts, truth, model_names):
    """
    Evaluate a 'Global Best' prior: always pick the model with highest AVERAGE reward across all data.
    """
    print("\nEvaluating Global Prior Baseline...")
    
    # Compute global average reward for each model
    all_models = set()
    for rewards in truth.values():
        all_models.update(rewards.keys())
    
    model_scores = {m: [] for m in all_models}
    for c, rewards in truth.items():
        for m, r in rewards.items():
            model_scores[m].append(r)
            
    avg_scores = {m: np.mean(scores) if scores else 0.0 for m, scores in model_scores.items()}
    global_best_model = max(avg_scores.items(), key=lambda x: x[1])[0]
    print(f"Global Best Model: {global_best_model} (Avg Reward: {avg_scores[global_best_model]:.4f})")
    
    # Calculate Regret of Global Best
    total_regret = 0.0
    n_samples = 0
    
    for p in prompts:
        c = p["cluster_id"]
        cluster_rewards = truth.get(c, {})
        if not cluster_rewards:
            continue
            
        best_reward = max(cluster_rewards.values())
        observed_reward = cluster_rewards.get(global_best_model, 0.0)
        
        total_regret += (best_reward - observed_reward)
        n_samples += 1
        
    avg_regret = total_regret / n_samples
    print(f"Global Prior Avg Regret: {avg_regret:.4f}")
    return avg_regret


def main():
    print("=" * 60)
    print("Embedding Quality Analysis")
    print("=" * 60)
    
    # Load Data
    prompts, rewards = load_all_data()
    print(f"Loaded {len(prompts)} prompts")
    
    # Get Optimal Models
    optimal_models, truth = get_optimal_models(prompts, rewards)
    model_names = sorted(list(set(m for m in optimal_models if m)))
    print(f"Unique Optimal Models: {len(model_names)}")
    
    # Embed
    print("Embedding prompts...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode([p["prompt"] for p in prompts], normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    
    # 1. Smoothness Analysis
    consistencies = analyze_smoothness(embeddings, optimal_models, k=5)
    
    # Plot Consistency Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(consistencies, bins=6, range=(0, 1.1), rwidth=0.8, alpha=0.7)
    plt.title("Local Consistency of Optimal Models (k=5)")
    plt.xlabel("Fraction of Neighbors with Same Optimal Model")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/embedding_smoothness.png")
    print("Saved smoothness plot to results/embedding_smoothness.png")
    
    # 2. Global Prior Baseline
    global_regret = evaluate_global_prior(prompts, truth, model_names)
    
    # Compare with Cold Start (approximate from previous runs ~0.14-0.16 avg regret)
    print(f"\nComparison:")
    print(f"  Global Prior Regret: {global_regret:.4f}")
    print(f"  Cold Start Regret (Approx): ~0.1400")
    
    if global_regret < 0.14:
        print("\n[!] Global Prior beats Cold Start.")
        print("    This suggests the task is 'easy' if we ignore context.")
        print("    The Contextual Bandit is likely overfitting noise in the embeddings.")
    else:
        print("\n[+] Global Prior is worse than Cold Start.")
        print("    Context matters, but our current embeddings aren't capturing it well.")


if __name__ == "__main__":
    main()
