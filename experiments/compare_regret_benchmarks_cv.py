#!/usr/bin/env python3
"""
5-Fold Cross-Validation: Cold Start vs. Benchmark Priors

This script evaluates using EXTERNAL BENCHMARK SCORES from the model registry
as priors for the bandit. This is a "Zero-Shot" or "Transfer Learning" approach.

Methodology:
1.  Load `banditgpt/data/models_cache.json`.
2.  Compute a composite score for each model:
    Score = Average(MMLU-Pro, HumanEval/100, GPQA)
3.  Initialize the bandit with these scores as the prior mean (b vector).
    -   A = Identity * Prior Strength
    -   b = Score * Prior Strength * Identity
4.  Evaluate on 5-Fold CV (Test Splits).
"""

import json
import logging
import tempfile
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
logger = logging.getLogger("compare_regret_benchmarks")


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


def load_benchmark_scores(model_names: List[str]) -> Dict[str, float]:
    """
    Load benchmark scores from models_cache.json and compute composite score.
    Returns dict: {model_id: score (0.0-1.0)}
    """
    cache_path = get_package_data_dir() / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
        
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    scores = {}
    print("\n[Benchmark Scores]")
    print(f"{'Model':<40} {'MMLU':<6} {'Code':<6} {'GPQA':<6} | {'Avg':<6}")
    print("-" * 70)
    
    for m in model_names:
        if m not in registry:
            scores[m] = 0.5 # Default
            continue
            
        reg = registry[m]
        
        # Extract scores (handle missing/None)
        mmlu = float(reg.get("mmlu_pro") or 0.0)
        code = float(reg.get("humaneval_score") or 0.0) / 100.0
        gpqa = float(reg.get("gpqa") or 0.0)
        
        # Filter out zeros for average if possible, or just avg all 3
        components = [s for s in [mmlu, code, gpqa] if s > 0]
        if components:
            avg = sum(components) / len(components)
        else:
            avg = 0.5 # Fallback if no benchmarks
            
        scores[m] = avg
        print(f"{m[:40]:<40} {mmlu:.2f}   {code:.2f}   {gpqa:.2f}   | {avg:.2f}")
        
    return scores


def generate_benchmark_priors(
    model_names: List[str],
    scores: Dict[str, float],
    output_path: Path,
    dim: int,
    alpha: float = 0.5,
    prior_strength: float = 20.0,
):
    """
    Generate priors based on benchmark scores.
    We construct a 'flat' prior where the mean reward is the benchmark score.
    """
    A_stack = []
    b_stack = []
    
    for m in model_names:
        score = scores.get(m, 0.5)
        
        # A = lambda * I
        # b = lambda * score * unit_vector? No, b should be such that A_inv @ b = score
        # If A = lambda * I, then A_inv = (1/lambda) * I
        # theta = score (scalar) is not right, theta is a vector.
        # We want theta^T x = score for any normalized x.
        # This implies theta should be aligned with the average x? 
        # OR, we just set the bias term if we had one.
        # Since we don't have a bias term, we can't easily set a constant prior 
        # without assuming something about x.
        
        # ALTERNATIVE: Use the "Global Prior" trick.
        # We simulate training on a dataset where x is random (or average) and y = score.
        # But simpler:
        # Just set A = lambda * I
        # And b = 0.
        # Wait, that gives theta = 0.
        
        # We want theta to predict 'score'.
        # If we assume x is roughly unit length and distributed around 0, 
        # we can't enforce a constant positive shift without a bias feature.
        # BUT, usually there's a bias feature or the embeddings are not centered at 0.
        
        # Let's assume we want to bias the bandit towards these models.
        # We can simulate 100 "synthetic" examples where we update the model with reward = score.
        # Since we don't know x, we can't do this perfectly.
        
        # ACTUALLY: The previous "Global Prior" script worked by using REAL x's from training set.
        # Here we don't want to use training set x's (Zero Shot).
        # So we will just use the SCORES to rank them?
        # No, we need to initialize A and b.
        
        # Strategy:
        # We can't set a "constant" prior in a linear bandit without a bias term or data.
        # However, we can use the "Prior Strength" to just initialize A.
        # And b?
        # If we leave b=0, it's Cold Start with high regularization.
        
        # Let's use the "Global Prior" approach but substitute the "Empirical Mean" with "Benchmark Score".
        # AND we need some X's. We can use the TEST set X's? No, that's leakage.
        # We can use a small set of "generic" X's (e.g. random unit vectors) to "burn in" the prior?
        # No, that would make theta random.
        
        # REVISION: We MUST use the Training Set X's to "ground" the benchmark scores.
        # So this is "Transfer Learning":
        # 1. Take Training Prompts (X).
        # 2. Instead of Real Rewards (Y), use Benchmark Score (Y_bench).
        # 3. Train bandit on (X, Y_bench).
        # This teaches the bandit: "For these kinds of prompts, this model gets Y_bench reward."
        pass

    # Implementation of the REVISION:
    # We will do this inside the CV loop since we need the train_idx.
    pass


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
    print("5-Fold CV: Benchmark Priors (Zero-Shot / Transfer)")
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
    
    # 2. Load Benchmark Scores
    bench_scores = load_benchmark_scores(model_names)
    
    # 3. K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "cold_regret": [],
        "bench_regret": [],
        "reduction": []
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_prompts)):
        print(f"\n[Fold {fold+1}/5]")
        
        # Prepare Train Data (for grounding the priors)
        train_embeddings = embeddings_all[train_idx]
        
        # Initialize Benchmark Policy
        # We train it on (Train_X, Benchmark_Score)
        print(f"  Grounding Benchmark Priors on {len(train_idx)} samples...")
        bench_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
        
        # Prior Strength (how many times we repeat the data or scale updates)
        # We simulate "pre-training" on the benchmark scores
        prior_strength = 20.0 
        
        # We can implement prior strength by just running the update loop multiple times?
        # Or better: Update once, then scale A and b.
        
        for i in range(len(train_idx)):
            x = train_embeddings[i]
            for m in model_names:
                target = bench_scores[m]
                bench_policy.update(m, x, target)
                
        # Apply Prior Strength Scaling
        # A <- A * strength? No, that's not exactly right for "updates".
        # If we want to say "I am confident", we scale A.
        # But if we scaled A, we must scale b too to keep theta same.
        # Yes, A_new = lambda * A, b_new = lambda * b preserves theta but reduces variance.
        for m in model_names:
            bench_policy.A[m] *= prior_strength
            bench_policy.b[m] *= prior_strength
            bench_policy.A_inv[m] = np.linalg.inv(bench_policy.A[m])
            
        # Prepare Test Data
        test_embeddings = embeddings_all[test_idx]
        test_clusters = [cluster_ids_all[i] for i in test_idx]
        
        # Initialize Cold Policy
        cold_policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
        
        # Run Simulation
        print(f"  Evaluating on {len(test_idx)} test samples...")
        cold_curve = run_simulation(cold_policy, test_embeddings, test_clusters, truth_all, model_names)
        bench_curve = run_simulation(bench_policy, test_embeddings, test_clusters, truth_all, model_names)
        
        final_cold = cold_curve[-1]
        final_bench = bench_curve[-1]
        red = (final_cold - final_bench) / final_cold * 100
        
        results["cold_regret"].append(final_cold)
        results["bench_regret"].append(final_bench)
        results["reduction"].append(red)
        
        print(f"  Result: Cold={final_cold:.1f}, Bench={final_bench:.1f}, Red={red:.1f}%")

    # 4. Statistical Analysis
    print("\n" + "=" * 60)
    print("5-Fold CV Results (Benchmark Prior)")
    print("=" * 60)
    
    cold_arr = np.array(results["cold_regret"])
    bench_arr = np.array(results["bench_regret"])
    red_arr = np.array(results["reduction"])
    
    mean_red = np.mean(red_arr)
    std_red = np.std(red_arr, ddof=1)
    se_red = std_red / np.sqrt(len(red_arr))
    ci_95 = stats.t.ppf(0.975, len(red_arr)-1) * se_red
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(cold_arr, bench_arr)
    
    print(f"Mean Regret Reduction: {mean_red:.2f}% ± {ci_95:.2f}% (95% CI)")
    print(f"P-value (paired t-test): {p_val:.5f}")
    if p_val < 0.05:
        print("Result is STATISTICALLY SIGNIFICANT.")
    else:
        print("Result is NOT statistically significant.")
        
    print("-" * 60)
    print(f"Cold Regret:  {np.mean(cold_arr):.1f} ± {np.std(cold_arr):.1f}")
    print(f"Bench Regret: {np.mean(bench_arr):.1f} ± {np.std(bench_arr):.1f}")
    print("=" * 60)
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([cold_arr, bench_arr], labels=["Cold Start", "Benchmark Prior"])
    plt.ylabel("Cumulative Regret")
    plt.title(f"Benchmark Prior vs Cold Start (p={p_val:.4f})")
    plt.grid(True, alpha=0.3)
    
    output_dir = Path("results/cv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "benchmark_prior_cv.png")
    print(f"Saved plot to {output_dir / 'benchmark_prior_cv.png'}")


if __name__ == "__main__":
    main()
