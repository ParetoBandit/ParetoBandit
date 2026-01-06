#!/usr/bin/env python3
"""
Experiment 08: Regret Convergence (Cold Start Defense)

Proves that BanditGPT's priors solve the cold-start problem that makes
standard bandits unusable in production.

Comparison:
- Cold Start LinUCB (N=0): Steep initial slope (thrashing)
- ε-Greedy (ε=0.1): Linear slope (never stops exploring)
- BanditGPT (N=100 Priors): Flat slope (starts competent)

Takeaway: "We solve the Cold Start problem."

Output: results/convergence_results.json
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING (100% REAL DATA)
# =============================================================================

def load_real_data():
    """
    Load train rewards and model registry from actual files.
    NO FALLBACKS. NO SYNTHETIC DATA.
    """
    data_dir = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    models_path = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "config" / "models.json"
    
    train_rewards_path = data_dir / "train_rewards_hle_models.jsonl"
    
    # Verify all files exist
    assert train_rewards_path.exists(), f"Train rewards not found: {train_rewards_path}"
    assert models_path.exists(), f"Models registry not found: {models_path}"
    
    print("📦 Loading real data...")
    
    # Load model registry
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    print(f"  ✓ Registry: {len(registry)} models")
    
    # Load train rewards
    train_data = load_rewards(train_rewards_path, "Training")
    
    return train_data, registry


def load_rewards(path: Path, label: str) -> Dict:
    """Load rewards from JSONL file."""
    prompt_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}})
    
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                cluster_id = entry.get("cluster_id", 0)
                
                prompt_data[prompt]["cluster_id"] = cluster_id
                prompt_data[prompt]["rewards"][model_id] = entry["raw_score"]
    
    print(f"  ✓ {label}: {len(prompt_data)} prompts")
    return dict(prompt_data)


# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

def run_epsilon_greedy(
    prompts: List[str],
    oracle_rewards: Dict,
    available_models: List[str],
    epsilon: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    ε-greedy baseline with online learning.
    Returns full cumulative regret curve.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize running statistics
    model_counts = {m: 0 for m in available_models}
    model_means = {m: 0.5 for m in available_models}  # Optimistic prior
    
    cumulative_regret = 0.0
    regret_curve = []
    
    for prompt in prompts:
        data = oracle_rewards.get(prompt, {})
        rewards = data.get("rewards", {})
        
        if not rewards:
            regret_curve.append(cumulative_regret)
            continue
        
        # ε-greedy selection
        if rng.random() < epsilon:
            # Explore: random selection
            model_id = rng.choice(available_models)
        else:
            # Exploit: best empirical mean
            model_id = max(available_models, key=lambda m: model_means[m])
        
        # ORACLE LOOKUP
        oracle_reward = max(rewards.values())
        actual_reward = rewards.get(model_id, 0.0)
        
        regret = oracle_reward - actual_reward
        cumulative_regret += regret
        regret_curve.append(cumulative_regret)
        
        # UPDATE (incremental mean)
        if model_id in rewards:
            model_counts[model_id] += 1
            n = model_counts[model_id]
            model_means[model_id] += (actual_reward - model_means[model_id]) / n
    
    return np.array(regret_curve), []


def run_cold_start_linucb(
    prompts: List[str],
    oracle_rewards: Dict,
    registry: Dict,
    encoder,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    Cold Start LinUCB: BanditRouter with NO priors (prior_n_effective=0).
    This simulates the "thrashing" behavior of an uninformed contextual bandit.
    """
    random.seed(seed)
    
    # Initialize router with ZERO priors (cold start)
    router = BanditRouter.create(
        registry,
        exploration="safe",  # α=0.1
        priors="hle",  # Structure provided but...
        prior_n_effective=0.0,  # NO strength! Cold start.
        prior_structure_n_effective=0.0,  # Also zero for pure cold start
        context_encoder=encoder
    )
    
    from src.bandit_gpt.router import OptimizationProfile
    profile = OptimizationProfile.MAX_QUALITY
    
    cumulative_regret = 0.0
    regret_curve = []
    
    for prompt in prompts:
        data = oracle_rewards.get(prompt, {})
        rewards = data.get("rewards", {})
        
        if not rewards:
            regret_curve.append(cumulative_regret)
            continue
        
        # Route using the bandit
        selected, log = router.route(prompt, profile=profile, input_tokens=100)
        
        # ORACLE LOOKUP
        oracle_reward = max(rewards.values())
        actual_reward = rewards.get(selected, 0.0)
        
        regret = oracle_reward - actual_reward
        cumulative_regret += regret
        regret_curve.append(cumulative_regret)
        
        # UPDATE using cached context vector directly to avoid re-encoding
        # Use bandit.update() directly since log.context_vector already includes bias term
        if selected in rewards and log.context_vector is not None:
            router.bandit.update(selected, log.context_vector, actual_reward)
    
    return np.array(regret_curve), []


def run_banditgpt_with_priors(
    prompts: List[str],
    oracle_rewards: Dict,
    registry: Dict,
    encoder,
    prior_n: float = 100.0,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    BanditGPT with strong HLE priors (N=100).
    Should show flat slope from the start.
    """
    random.seed(seed)
    
    # Initialize router with STRONG priors
    router = BanditRouter.create(
        registry,
        exploration="safe",
        priors="hle",
        prior_n_effective=prior_n,  # Strong priors!
        prior_structure_n_effective=250.0,  # Keep structure priors
        context_encoder=encoder
    )
    
    from src.bandit_gpt.router import OptimizationProfile
    profile = OptimizationProfile.MAX_QUALITY
    
    cumulative_regret = 0.0
    regret_curve = []
    
    for prompt in prompts:
        data = oracle_rewards.get(prompt, {})
        rewards = data.get("rewards", {})
        
        if not rewards:
            regret_curve.append(cumulative_regret)
            continue
        
        # Route using the bandit
        selected, log = router.route(prompt, profile=profile, input_tokens=100)
        
        # ORACLE LOOKUP
        oracle_reward = max(rewards.values())
        actual_reward = rewards.get(selected, 0.0)
        
        regret = oracle_reward - actual_reward
        cumulative_regret += regret
        regret_curve.append(cumulative_regret)
        
        # UPDATE using cached context vector directly to avoid re-encoding
        # Use bandit.update() directly since log.context_vector already includes bias term
        if selected in rewards and log.context_vector is not None:
            router.bandit.update(selected, log.context_vector, actual_reward)
    
    return np.array(regret_curve), []


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_convergence_experiment(
    train_data: Dict,
    registry: Dict,
    encoder,
    n_trials: int = 5
) -> Dict:
    """
    Run all three algorithms and collect regret curves.
    """
    print("\n" + "="*70)
    print("REGRET CONVERGENCE EXPERIMENT")
    print("="*70)
    
    prompts = list(train_data.keys())
    available_models = list(registry.keys())
    
    results = {
        "cold_start_linucb": [],
        "epsilon_greedy": [],
        "banditgpt_n100": []
    }
    
    for trial in range(n_trials):
        print(f"\n📊 Trial {trial+1}/{n_trials}")
        
        # Shuffle prompts for this trial
        random.seed(42 + trial)
        shuffled_prompts = prompts.copy()
        random.shuffle(shuffled_prompts)
        
        # Run Cold Start LinUCB
        print("  → Cold Start LinUCB (N=0)...", end=" ", flush=True)
        linucb_curve, _ = run_cold_start_linucb(
            shuffled_prompts, train_data, registry, encoder, seed=42+trial
        )
        results["cold_start_linucb"].append(linucb_curve.tolist())
        print(f"Final Regret: {linucb_curve[-1]:.2f}")
        
        # Run ε-Greedy
        print("  → ε-Greedy (ε=0.1)...", end=" ", flush=True)
        egreedy_curve, _ = run_epsilon_greedy(
            shuffled_prompts, train_data, available_models, seed=42+trial
        )
        results["epsilon_greedy"].append(egreedy_curve.tolist())
        print(f"Final Regret: {egreedy_curve[-1]:.2f}")
        
        # Run BanditGPT with N=100 Priors
        print("  → BanditGPT (N=100 Priors)...", end=" ", flush=True)
        bandit_curve, _ = run_banditgpt_with_priors(
            shuffled_prompts, train_data, registry, encoder, prior_n=100.0, seed=42+trial
        )
        results["banditgpt_n100"].append(bandit_curve.tolist())
        print(f"Final Regret: {bandit_curve[-1]:.2f}")
    
    return results


def main():
    """Execute regret convergence experiment."""
    print("="*70)
    print("EXPERIMENT 08: REGRET CONVERGENCE (COLD START DEFENSE)")
    print("="*70)
    
    # Load real data
    train_data, registry = load_real_data()
    
    # Initialize encoder (shared)
    print("\n🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    print(f"  ✓ Encoder: {DEFAULT_CONTEXT_MODEL}")
    
    # Run experiment
    results = run_convergence_experiment(
        train_data, registry, encoder, n_trials=5
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "convergence_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "08_regret_convergence",
            "description": "Regret Convergence (Cold Start Defense)",
            "algorithms": {
                "cold_start_linucb": "LinUCB with prior_n_effective=0 (no priors)",
                "epsilon_greedy": "ε-greedy with ε=0.1",
                "banditgpt_n100": "BanditGPT with prior_n_effective=100 (strong priors)"
            },
            "results": results,
            "metadata": {
                "n_trials": 5,
                "n_prompts": len(train_data),
                "n_models": len(registry)
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (Final Cumulative Regret)")
    print("="*70)
    
    for method, curves in results.items():
        final_regrets = [c[-1] for c in curves]
        mean = np.mean(final_regrets)
        std = np.std(final_regrets)
        print(f"  {method:25s}: {mean:7.2f} ± {std:.2f}")
    
    print(f"\n📁 Next step: Run plot_convergence.py to visualize the results")


if __name__ == "__main__":
    main()
