#!/usr/bin/env python3
"""
Benchmark: Procedural Warmup Convergence

Proves that 'Procedural Warmup' accelerates learning compared to Cold Start
using REAL test data and the full BanditRouter with embeddings.

KDD Claim: "Procedural Warmup achieves +15.8% improvement over cold start"

Usage:
    python benchmark_convergence.py
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "offline_dataset"
MODELS_PATH = PROJECT_ROOT / "models.json"
TEST_REWARDS_PATH = DATA_DIR / "test_rewards_pareto_dedup.jsonl"


def load_test_data() -> list:
    """
    Load ALL test data with prompts and rewards.
    
    Returns:
        List of dicts with {prompt, model_rewards: {model_id: raw_score}}
    """
    # Group by exact prompt text
    prompt_data = defaultdict(lambda: {"prompt": None, "model_rewards": {}})
    
    print(f"   Loading from {TEST_REWARDS_PATH}...")
    with open(TEST_REWARDS_PATH) as f:
        for line_num, line in enumerate(f):
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                prompt_data[prompt]["prompt"] = prompt
                prompt_data[prompt]["model_rewards"][entry["model_id"]] = entry["raw_score"]
            
            if (line_num + 1) % 10000 == 0:
                print(f"   Processed {line_num + 1} entries...")
    
    # Filter to prompts with multiple model responses (for meaningful regret)
    result = [v for v in prompt_data.values() if len(v["model_rewards"]) >= 3]
    return result


def load_registry() -> dict:
    """Load model registry."""
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return {m["openrouter_id"]: m for m in data["models"]}


def simulate_with_router(
    test_data: list,
    registry: dict,
    use_warmup: bool,
    n_steps: int = 500
) -> dict:
    """
    Simulate routing using the FULL BanditRouter with embeddings.
    """
    from bandit_v2 import BanditRouter
    
    # Select models that appear in test data
    all_models = set()
    for item in test_data:
        all_models.update(item["model_rewards"].keys())
    
    # Filter registry to models in test data
    filtered_registry = {k: v for k, v in registry.items() if k in all_models}
    
    if len(filtered_registry) < 3:
        raise ValueError(f"Need at least 3 models, found {len(filtered_registry)}")
    
    # Create router with or without warmup
    priors_mode = "hle" if use_warmup else "none"
    
    router = BanditRouter.create(
        model_registry=filtered_registry,
        priors=priors_mode,
        alpha=0.1  # SAFE exploration
    )
    
    cumulative_regret = 0.0
    regret_history = []
    
    # Shuffle test data for randomness
    data_copy = test_data.copy()
    np.random.shuffle(data_copy)
    
    step = 0
    for item in data_copy:
        if step >= n_steps:
            break
            
        prompt = item["prompt"]
        model_rewards = item["model_rewards"]
        
        # Filter to models in router
        available = {m: r for m, r in model_rewards.items() if m in router.registry}
        
        if len(available) < 2:
            continue
        
        # Oracle: best possible reward
        oracle_reward = max(available.values())
        
        # Router selection (does embedding internally)
        try:
            selected_model, log = router.route(prompt)
        except Exception as e:
            continue
        
        # Get actual reward (0 if model not in test set for this prompt)
        actual_reward = available.get(selected_model, 0.0)
        
        # Calculate regret
        instant_regret = oracle_reward - actual_reward
        cumulative_regret += instant_regret
        regret_history.append(cumulative_regret)
        
        # Process feedback to update bandit
        router.process_feedback(log.request_id, actual_reward)
        
        step += 1
    
    return {
        "use_warmup": use_warmup,
        "n_steps": len(regret_history),
        "final_regret": cumulative_regret,
        "regret_history": regret_history
    }


def main():
    print("=" * 70)
    print("BENCHMARK: Procedural Warmup Convergence (FULL ROUTER)")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading real test data...")
    
    if not TEST_REWARDS_PATH.exists():
        print(f"   ❌ Test rewards not found: {TEST_REWARDS_PATH}")
        sys.exit(1)
    
    test_data = load_test_data()
    print(f"   ✓ Loaded {len(test_data)} unique prompts with model rewards")
    
    registry = load_registry()
    print(f"   ✓ Loaded {len(registry)} models from registry")
    
    # Run trials
    n_trials = 3
    n_steps = 500
    
    print(f"\n⚔️  Running {n_trials} trials × {n_steps} steps each...")
    print("   (Using full BanditRouter with embeddings)\n")
    
    cold_results = []
    warm_results = []
    
    for trial in range(n_trials):
        print(f"   Trial {trial + 1}/{n_trials}:")
        
        print(f"      Cold start...", end=" ", flush=True)
        cold = simulate_with_router(
            test_data, registry,
            use_warmup=False, n_steps=n_steps
        )
        print(f"regret={cold['final_regret']:.1f}")
        
        print(f"      Warm start...", end=" ", flush=True)
        warm = simulate_with_router(
            test_data, registry,
            use_warmup=True, n_steps=n_steps
        )
        print(f"regret={warm['final_regret']:.1f}")
        
        cold_results.append(cold["final_regret"])
        warm_results.append(warm["final_regret"])
    
    # Calculate statistics
    cold_mean = np.mean(cold_results)
    cold_std = np.std(cold_results)
    warm_mean = np.mean(warm_results)
    warm_std = np.std(warm_results)
    
    improvement = ((cold_mean - warm_mean) / cold_mean) * 100 if cold_mean > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Cumulative Regret at T={n_steps}:")
    print(f"   Cold Start (priors='none'):  {cold_mean:.2f} ± {cold_std:.2f}")
    print(f"   Warm Start (priors='hle'):   {warm_mean:.2f} ± {warm_std:.2f}")
    print(f"   🚀 Improvement: {improvement:+.1f}%")
    
    # Validation
    print("\n" + "=" * 70)
    print("KDD CLAIM VALIDATION")
    print("=" * 70)
    
    claim_target = 15.8
    print(f"\nClaimed improvement: +{claim_target}%")
    print(f"Measured improvement: {improvement:+.1f}%")
    
    if warm_mean < cold_mean:
        print("\n✅ PASS: Procedural Warmup reduces cumulative regret.")
        if improvement >= claim_target * 0.8:
            print(f"✅ PASS: Improvement ({improvement:+.1f}%) ≥ 80% of claim")
        else:
            print(f"⚠️  PARTIAL: Improvement ({improvement:+.1f}%) < 80% of claim")
    else:
        print("\n⚠️  Warmup did not reduce regret in this run.")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return warm_mean <= cold_mean


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
