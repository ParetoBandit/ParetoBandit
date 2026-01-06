#!/usr/bin/env python3
"""
Experiment 06: Prior Strength Sensitivity Analysis

Validates the default prior_n_effective hyperparameter by sweeping
prior strength values and measuring cumulative regret.

This directly addresses the KDD reviewer's critique about the lack of
empirical validation for the prior strength hyperparameter.
"""

import sys
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
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
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_sweep(
    train_data: Dict,
    test_data: Dict,
    registry: Dict,
    encoder,
    prior_values: List[float],
    n_trials: int = 3
) -> List[Dict]:
    """
    Sweep prior_n_effective values and measure cumulative regret.
    
    For each prior strength:
    1. Initialize BanditRouter with specified prior_n_effective
    2. Train on real training data (burn-in)
    3. Evaluate on real test data (greedy)
    4. Calculate cumulative regret vs. oracle
    
    Args:
        train_data: Training prompts with rewards
        test_data: Test prompts with rewards
        registry: Model registry
        encoder: Pre-initialized sentence encoder
        prior_values: List of prior_n_effective values to test
        n_trials: Number of trials per value for variance estimation
    
    Returns:
        List of results for each prior value
    """
    print("\n" + "="*70)
    print("PRIOR STRENGTH SENSITIVITY ANALYSIS")
    print("="*70)
    
    from src.bandit_gpt.router import OptimizationProfile
    
    # Use Max Quality profile for evaluation
    profile = OptimizationProfile.MAX_QUALITY
    
    results = []
    
    for prior_n in prior_values:
        print(f"\n📊 Prior Strength N={prior_n:.1f}")
        
        trial_regrets = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}...", end=" ", flush=True)
            
            # Initialize router with specified prior strength
            router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="hle",
                prior_n_effective=prior_n,
                prior_structure_n_effective=250.0,  # Keep structure constant
                context_encoder=encoder
            )
            
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
            
            cumulative_regret = 0.0
            
            for prompt in test_prompts:
                data = test_data[prompt]
                selected, _ = router.route(prompt, profile=profile, input_tokens=100)
                
                if selected in data["rewards"]:
                    # Oracle: best possible reward for this prompt
                    oracle_reward = max(data["rewards"].values())
                    actual_reward = data["rewards"][selected]
                    
                    regret = oracle_reward - actual_reward
                    cumulative_regret += regret
            
            # Restore exploration
            router.bandit.alpha = original_alpha
            
            trial_regrets.append(cumulative_regret)
            print(f"Cumulative Regret={cumulative_regret:.2f}")
        
        results.append({
            "prior_n_effective": prior_n,
            "regret_mean": float(np.mean(trial_regrets)),
            "regret_std": float(np.std(trial_regrets)),
            "regret_trials": trial_regrets
        })
        
        print(f"  → Mean Regret: {np.mean(trial_regrets):.2f} ± {np.std(trial_regrets):.2f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Execute sensitivity analysis experiment."""
    print("="*70)
    print("EXPERIMENT 06: PRIOR STRENGTH SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Load real data
    train_data, test_data, registry = load_real_data()
    
    # Initialize encoder (shared)
    print("\n🔧 Initializing encoder...")
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    print(f"  ✓ Encoder: {DEFAULT_CONTEXT_MODEL}")
    
    # Prior strength values to test
    prior_values = [0, 10, 20, 50, 100, 250]
    
    # Run sensitivity sweep
    results = run_sensitivity_sweep(
        train_data, test_data, registry, encoder, 
        prior_values=prior_values,
        n_trials=3
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_path = output_dir / "sensitivity_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "06_sensitivity_analysis",
            "description": "Prior Strength Sensitivity Analysis",
            "data_source": "100% real data (train_rewards_hle_models.jsonl, test_rewards_hle_models.jsonl)",
            "prior_values": prior_values,
            "results": results,
            "metadata": {
                "n_trials": 3,
                "n_train_prompts": len(train_data),
                "n_test_prompts": len(test_data),
                "n_models": len(registry)
            }
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n📊 Prior Strength vs. Cumulative Regret:")
    for r in results:
        marker = "  ← DEFAULT" if r["prior_n_effective"] == 10.0 else ""
        print(f"  N={r['prior_n_effective']:5.0f} → Regret={r['regret_mean']:6.2f} ± {r['regret_std']:5.2f}{marker}")
    
    print(f"\n📁 Next step: Run plot_sensitivity.py to visualize the results")


if __name__ == "__main__":
    main()
