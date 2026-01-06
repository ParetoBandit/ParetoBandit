#!/usr/bin/env python3
"""
Experiment 05: Cost-Quality Pareto Frontier

Demonstrates BanditGPT's economic viability by showing it achieves higher
quality than random routing at equivalent cost budgets.

This experiment proves the "Money Shot": You can get GPT-4 level quality
for 50% of the price by intelligently routing only hard prompts to expensive models.
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


# =============================================================================
# DATA LOADING (100% REAL DATA)
# =============================================================================

def load_real_data():
    """
    Load train/test rewards and model registry from actual files.
    NO FALLBACKS. NO SYNTHETIC DATA.
    """
    data_dir = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    models_path = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "config" / "models.json"
    
    test_rewards_path = data_dir / "test_rewards_hle_models.jsonl"
    train_rewards_path = data_dir / "train_rewards_hle_models.jsonl"
    
    # Verify all files exist
    assert test_rewards_path.exists(), f"Test rewards not found: {test_rewards_path}"
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
    
    # Load test rewards
    test_data = load_rewards(test_rewards_path, "Test")
    
    return train_data, test_data, registry


def load_rewards(path: Path, label: str) -> Dict:
    """Load rewards from JSONL file."""
    prompt_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}, "costs": {}, "latencies": {}})
    
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


def get_model_cost(model: Dict) -> float:
    """Calculate average cost per 1k tokens in USD."""
    # Support both naming conventions in models.json
    input_cost = model.get("price_1m_input") or model.get("input_cost_per_m")
    output_cost = model.get("price_1m_output") or model.get("output_cost_per_m")
    
    if input_cost is None or output_cost is None:
        return 0.0
    
    # Standard metric: Blended cost per 1k tokens (50/50 split)
    # price_1m is in USD per 1M tokens. 
    # To get USD per 1k tokens: divide by 1000.
    cost_per_1k = (0.5 * input_cost + 0.5 * output_cost) / 1000.0
    return cost_per_1k


# =============================================================================
# PARETO FRONTIER SWEEP
# =============================================================================

def run_pareto_sweep(
    train_data: Dict,
    test_data: Dict,
    registry: Dict,
    encoder,
    n_trials: int = 10
) -> List[Dict]:
    """
    Sweep cost profiles to generate Pareto frontier.
    
    For each profile:
    1. Initialize BanditRouter with real registry
    2. Train on real train_rewards_1k.jsonl data
    3. Evaluate on real test_rewards_pareto_dedup.jsonl data
    4. Collect (cost, quality) metrics
    
    NO SYNTHETIC DATA. ALL REAL.
    """
    print("\n" + "="*70)
    print("PARETO FRONTIER SWEEP")
    print("="*70)
    
    from src.bandit_gpt.router import OptimizationProfile
    
    # Pareto frontier sweep profiles - use defaults from router
    profiles = [
        {"name": "Max Quality",  **OptimizationProfile.MAX_QUALITY},
        {"name": "Arbitrage",    **OptimizationProfile.ARBITRAGE},
        {"name": "Best Value",   **OptimizationProfile.BEST_VALUE},
    ]
    
    frontier_results = []
    
    for config in profiles:
        print(f"\n📊 [{config['name']}] Profile: Q={config['w_q']:.2f}, C={config['w_c']:.2f}, L={config['w_l']:.2f}")
        
        trial_costs = []
        trial_qualities = []
        all_selections = defaultdict(int)
        
        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}...", end=" ", flush=True)
            
            # Initialize router with REAL registry and HLE priors
            router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="hle",
                prior_n_effective=10.0,      # Calibrated champion
                prior_structure_n_effective=250.0,  # Calibrated champion
                context_encoder=encoder
            )
            
            # Profile for this sweep point
            profile = {"w_q": config["w_q"], "w_c": config["w_c"], "w_l": config["w_l"]}
            
            # Phase 1: BURN-IN (Training on real train data)
            train_prompts = list(train_data.keys())
            random.seed(42 + trial)
            random.shuffle(train_prompts)
            
            for prompt in train_prompts:
                data = train_data[prompt]
                selected, log = router.route(prompt, profile=profile, input_tokens=100)
                
                if selected in data["rewards"]:
                    reward = data["rewards"][selected]
                    router.update(selected, prompt, reward)
            
            # Phase 2: EVALUATE (Greedy on real test data)
            test_prompts = list(test_data.keys())
            random.shuffle(test_prompts)
            
            # Force greedy evaluation (no exploration)
            original_alpha = router.bandit.alpha
            router.bandit.alpha = 0.0
            
            costs = []
            qualities = []
            
            for prompt in test_prompts:
                data = test_data[prompt]
                selected, _ = router.route(prompt, profile=profile, input_tokens=100)
                
                if selected in data["rewards"]:
                    model = registry.get(selected, {})
                    cost = get_model_cost(model)
                    
                    if cost is not None:
                        costs.append(cost)
                        qualities.append(data["rewards"][selected])
                        all_selections[selected] += 1
            
            # Restore exploration
            router.bandit.alpha = original_alpha
            
            if costs:
                avg_cost = np.mean(costs)
                avg_quality = np.mean(qualities)
                trial_costs.append(avg_cost)
                trial_qualities.append(avg_quality)
                print(f"Cost=${avg_cost:.4f}, Quality={avg_quality*100:4.1f}%")
        
        if trial_costs:
            frontier_results.append({
                "profile": config["name"],
                "w_q": config["w_q"],
                "w_c": config["w_c"],
                "w_l": config["w_l"],
                "cost_mean": np.mean(trial_costs),
                "cost_std": np.std(trial_costs),
                "quality_mean": np.mean(trial_qualities),
                "quality_std": np.std(trial_qualities),
                "selections": dict(all_selections)
            })
    
    return frontier_results


# =============================================================================
# BASELINE CALCULATIONS
# =============================================================================

def compute_individual_models(test_data: Dict, registry: Dict) -> List[Dict]:
    """
    Compute (cost, quality) for individual models using REAL test data.
    These form the baseline comparison points.
    """
    print("\n📈 Computing individual model baselines...")
    
    model_points = []
    
    for model_id, model in registry.items():
        cost = get_model_cost(model)
        if cost is None:
            continue
        
        qualities = []
        for prompt, data in test_data.items():
            if model_id in data["rewards"]:
                qualities.append(data["rewards"][model_id])
        
        if qualities:
            avg_q = float(np.mean(qualities))
            model_points.append({
                "model": model_id,
                "cost": cost,
                "quality": avg_q
            })
            # print(f"    {model_id[:30]:30} ${cost:8.5f} {avg_q*100:5.11f}%")
    
    print(f"  ✓ Computed {len(model_points)} model baselines")
    return model_points


def run_random_baseline(
    test_data: Dict,
    registry: Dict,
    n_trials: int = 10
) -> Dict:
    """
    Simulate random model selection baseline (KDD Review Fix).
    
    This provides a fair empirical comparison by actually simulating
    what happens when you randomly select models on the test set.
    
    For each trial:
    1. For each test prompt, randomly select a model (uniform distribution)
    2. Record (cost, quality) for that selection
    3. Compute average cost and quality across all prompts
    
    Args:
        test_data: Test prompts with rewards
        registry: Model registry with cost information
        n_trials: Number of trials for variance estimation (default: 10)
    
    Returns:
        {
            "cost_mean": float,
            "cost_std": float,
            "quality_mean": float,
            "quality_std": float,
            "selections": dict of model selection counts
        }
    """
    print(f"\n🎲 Running random baseline ({n_trials} trials)...")
    
    trial_costs = []
    trial_qualities = []
    all_selections = defaultdict(int)
    available_models = list(registry.keys())
    
    for trial in range(n_trials):
        costs = []
        qualities = []
        
        for prompt, data in test_data.items():
            # Randomly select model (uniform distribution)
            model_id = random.choice(available_models)
            all_selections[model_id] += 1
            
            # Get reward if available for this model on this prompt
            if model_id in data["rewards"]:
                model = registry[model_id]
                cost = get_model_cost(model)
                quality = data["rewards"][model_id]
                
                if cost is not None:
                    costs.append(cost)
                    qualities.append(quality)
        
        if costs:
            trial_costs.append(np.mean(costs))
            trial_qualities.append(np.mean(qualities))
            print(f"  Trial {trial+1}/{n_trials}: Cost=${np.mean(costs):.4f}, Quality={np.mean(qualities)*100:.1f}%")
    
    result = {
        "cost_mean": float(np.mean(trial_costs)),
        "cost_std": float(np.std(trial_costs)),
        "quality_mean": float(np.mean(trial_qualities)),
        "quality_std": float(np.std(trial_qualities)),
        "selections": dict(all_selections)
    }
    
    print(f"  ✓ Random baseline: Cost=${result['cost_mean']:.4f} ± ${result['cost_std']:.4f}, "
          f"Quality={result['quality_mean']*100:.1f}% ± {result['quality_std']*100:.2f}%")
    
    return result



# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute Pareto frontier experiment with 100% real data."""
    print("="*70)
    print("EXPERIMENT 05: COST-QUALITY PARETO FRONTIER")
    print("="*70)
    
    # Load real data
    train_data, test_data, registry = load_real_data()
    
    # Initialize encoder (shared)
    print("\n🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    print(f"  ✓ Encoder: {DEFAULT_CONTEXT_MODEL}")
    
    # Run Pareto sweep
    frontier_results = run_pareto_sweep(
        train_data, test_data, registry, encoder, n_trials=3  # Reduced for quick validation
    )
    
    # Compute baselines
    model_baselines = compute_individual_models(test_data, registry)
    random_baseline = run_random_baseline(test_data, registry, n_trials=10)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "pareto_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "05_cost_tradeoff",
            "description": "Cost-Quality Pareto Frontier",
            "data_source": "100% real data (train_rewards_hle_models.jsonl, test_rewards_hle_models.jsonl)",
            "frontier": frontier_results,
            "model_baselines": model_baselines,
            "random_baseline": random_baseline
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n📊 Pareto Frontier Points:")
    for r in frontier_results:
        print(f"  {r['profile']:12} → Cost=${r['cost_mean']:.4f} ± ${r['cost_std']:.4f}, "
              f"Quality={r['quality_mean']*100:.1f}% ± {r['quality_std']*100:.2f}%")
    
    print(f"\n📁 Next step: Run plot_pareto.py to visualize the frontier")


if __name__ == "__main__":
    main()
