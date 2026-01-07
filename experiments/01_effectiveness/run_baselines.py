#!/usr/bin/env python3
"""
Experiment 01: Effectiveness Comparison (KDD-Compliant Offline Replay)

Compares BanditGPT against baselines using REAL data:
- Random selection (no learning)
- ε-greedy (ε=0.1, learns running averages)
- Vanilla LinUCB (bias-only context, learns)
- BanditGPT (full semantic features, learns)

Critical: All methods use ORACLE REWARD LOOKUP, not random generation.
Output: results/effectiveness_results.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_loader import load_oracle_rewards, load_model_registry
from utils.metrics import calculate_cumulative_regret

# Import BanditRouter for full system evaluation
from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from src.bandit_gpt.storage import SqliteContextStore
from sentence_transformers import SentenceTransformer


# =============================================================================
# BASELINE IMPLEMENTATIONS (All use Oracle Lookup)
# =============================================================================

def run_random_baseline(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    seed: int = 42
) -> Dict:
    """
    Random model selection baseline (no learning).
    
    The simplest baseline: uniform random selection across all models.
    No update step since random has no state to learn.
    """
    print(f"Running Random baseline (seed={seed})...")
    rng = np.random.RandomState(seed)
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="Random", leave=False):
        # Select random model
        model_id = rng.choice(available_models)
        
        # ORACLE LOOKUP (not random generation!)
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": "random",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_epsilon_greedy(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    epsilon: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    ε-greedy baseline with online learning.
    
    Maintains running mean reward for each model.
    With probability ε: explore (random)
    With probability 1-ε: exploit (best empirical mean)
    """
    print(f"Running ε-greedy (ε={epsilon}, seed={seed})...")
    rng = np.random.RandomState(seed)
    
    # Initialize running statistics
    model_counts = {m: 0 for m in available_models}
    model_means = {m: 0.5 for m in available_models}  # Optimistic prior
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="ε-greedy", leave=False):
        # ε-greedy selection
        if rng.random() < epsilon:
            # Explore: random selection
            model_id = rng.choice(available_models)
        else:
            # Exploit: best empirical mean
            model_id = max(available_models, key=lambda m: model_means[m])
        
        # ORACLE LOOKUP
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # UPDATE (incremental mean)
        model_counts[model_id] += 1
        n = model_counts[model_id]
        model_means[model_id] += (reward - model_means[model_id]) / n
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": f"epsilon_greedy_{epsilon}",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_vanilla_linucb(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    available_models: List[str],
    alpha: float = 1.0,
    seed: int = 42
) -> Dict:
    """
    Vanilla LinUCB baseline (bias-only context, no semantic features).
    
    Uses context vector x = [1.0] (just bias term).
    This tests whether semantic features provide lift over context-blind LinUCB.
    """
    print(f"Running Vanilla LinUCB (α={alpha}, seed={seed})...")
    
    # Disjoint LinUCB with d=1 (bias only)
    d = 1
    A = {m: np.eye(d) for m in available_models}  # d×d identity matrices
    b = {m: np.zeros(d) for m in available_models}  # d-dimensional vectors
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="LinUCB", leave=False):
        # Context: bias-only
        x = np.array([1.0])
        
        # Compute UCB for each arm
        ucb_scores = {}
        for m in available_models:
            A_inv = np.linalg.inv(A[m])
            theta = A_inv @ b[m]
            ucb = theta @ x + alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[m] = ucb
        
        # Select best UCB
        model_id = max(ucb_scores, key=lambda m: ucb_scores[m])
        
        # ORACLE LOOKUP
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # UPDATE (Sherman-Morrison style, but simple for d=1)
        A[model_id] += np.outer(x, x)
        b[model_id] += reward * x
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": "vanilla_linucb",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


def run_banditgpt(
    prompts: List[str],
    oracle_rewards: Dict[str, Dict[str, float]],
    registry: Dict[str, Dict],
    encoder,
    seed: int = 42
) -> Dict:
    """
    Full BanditGPT system with semantic features and online learning.
    
    Uses the complete feature set:
    - BERT embeddings (384-d) → PCA (32-d)
    - Handcrafted features (code blocks, LaTeX, length, etc.)
    - Virtual anchor similarities
    - Complexity score
    
    This is the method we're trying to prove works!
    """
    print(f"Running BanditGPT (seed={seed})...")
    
    # Fresh router for this trial (clean slate)
    router = BanditRouter.create(
        registry,
        exploration="safe",  # α=0.1
        priors="hle",
        prior_n_effective=10.0,
        context_encoder=encoder,
    )
    
    selected_models = []
    selected_rewards = []
    
    for prompt in tqdm(prompts, desc="BanditGPT", leave=False):
        # ROUTE (uses full semantic features)
        model_id, log = router.route(prompt, profile="arbitrage")
        
        # ORACLE LOOKUP
        reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        
        # UPDATE (the critical learning step!)
        # Pass the prompt text; router will re-encode to get context vector
        router.update(model_id, prompt, reward)
        
        selected_models.append(model_id)
        selected_rewards.append(reward)
    
    return {
        "method": "banditgpt",
        "selected_models": selected_models,
        "rewards": selected_rewards
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    """Run all baseline comparisons with real data."""
    print("=" * 70)
    print("EXPERIMENT 01: EFFECTIVENESS COMPARISON")
    print("KDD-Compliant Offline Replay Evaluation")
    print("=" * 70)
    
    # Load oracle rewards (real data from test set)
    print("\n📦 Loading data...")
    oracle_rewards = load_oracle_rewards("test_rewards_hle_models.jsonl")
    
    # Get prompts from oracle rewards keys
    prompts = list(oracle_rewards.keys())
    print(f"  ✓ {len(prompts)} test prompts")
    
    # Load model registry to get available models
    registry = load_model_registry()
    
    # Filter to models that have rewards in at least 50% of prompts
    model_coverage = defaultdict(int)
    for prompt_rewards in oracle_rewards.values():
        for model_id in prompt_rewards:
            model_coverage[model_id] += 1
    
    min_coverage = len(prompts) * 0.5
    available_models = [
        m for m in registry.keys() 
        if model_coverage.get(m, 0) >= min_coverage
    ]
    print(f"  ✓ {len(available_models)} models with ≥50% coverage")
    
    # Initialize shared encoder (avoid reloading for each trial)
    print("\n🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # Calculate oracle best (upper bound)
    oracle_best = []
    for prompt in prompts:
        rewards = oracle_rewards.get(prompt, {})
        if rewards:
            oracle_best.append(max(rewards.values()))
        else:
            oracle_best.append(0.0)
    oracle_best = np.array(oracle_best)
    print(f"  ✓ Oracle best computed (mean={np.mean(oracle_best):.3f})")
    
    # Run experiments with multiple seeds
    n_seeds = 10
    results = {}
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed + 1}/{n_seeds}")
        print("=" * 70)
        
        # Shuffle prompts for this seed
        rng = np.random.RandomState(seed)
        prompt_order = rng.permutation(len(prompts))
        shuffled_prompts = [prompts[i] for i in prompt_order]
        shuffled_oracle_best = oracle_best[prompt_order]
        
        # Run all methods
        random_result = run_random_baseline(
            shuffled_prompts, oracle_rewards, available_models, seed=seed
        )
        epsilon_result = run_epsilon_greedy(
            shuffled_prompts, oracle_rewards, available_models, seed=seed
        )
        linucb_result = run_vanilla_linucb(
            shuffled_prompts, oracle_rewards, available_models, seed=seed
        )
        banditgpt_result = run_banditgpt(
            shuffled_prompts, oracle_rewards, registry, encoder, seed=seed
        )
        
        # Calculate cumulative regret for each method
        for result in [random_result, epsilon_result, linucb_result, banditgpt_result]:
            method = result["method"]
            cum_regret = calculate_cumulative_regret(
                result["rewards"],
                shuffled_oracle_best
            )
            
            if method not in results:
                results[method] = []
            results[method].append(cum_regret.tolist())
            
            # Print summary for this run
            final_regret = cum_regret[-1]
            avg_reward = np.mean(result["rewards"])
            print(f"  {method:20s}: regret={final_regret:7.1f}, reward={avg_reward:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "effectiveness_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY (Average Final Cumulative Regret)")
    print("=" * 70)
    for method, regret_runs in results.items():
        final_regrets = [run[-1] for run in regret_runs]
        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        print(f"  {method:20s}: {mean_regret:7.1f} ± {std_regret:.1f}")
    
    print("\n📊 Next step: Run `python plot_regret.py` to generate figures")


if __name__ == "__main__":
    main()
