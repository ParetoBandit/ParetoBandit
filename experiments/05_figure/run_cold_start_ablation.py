#!/usr/bin/env python3
"""
Cold-Start Ablation: Fair Comparison with RouteLLM

This experiment addresses conference reviewer concern about fairness:
- RouteLLM uses pre-trained weights (no access to dev set)
- banditGPT normally trains on dev set first
- This ablation tests banditGPT WITHOUT dev set training (cold-start)

Expected Result:
- Cold-start performance should be LOWER than warm-start
- But still competitive due to strong priors from 80k RouteLLM battles
"""

import sys
from pathlib import Path
import json
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split,
    banditgpt_hybrid_routing
)
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*70)
    logger.info("COLD-START ABLATION: Fair Comparison with RouteLLM")
    logger.info("="*70)
    logger.info("\nResearcher Note:")
    logger.info("  This addresses reviewer concern about fairness:")
    logger.info("  - RouteLLM: Pre-trained on different 100k dataset")
    logger.info("  - banditGPT (warm-start): Trains on 1,121 dev prompts")
    logger.info("  - banditGPT (cold-start): NO dev training (this experiment)")
    
    # Load data and models
    logger.info("\n📦 Loading data and models...")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    # Use sanitized priors
    sanitized_priors_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    if sanitized_priors_path.exists():
        warmup_priors = joblib.load(sanitized_priors_path)
        logger.info(f"  ✓ Using sanitized priors: {sanitized_priors_path}")
    else:
        warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
        logger.info(f"  ⚠️  Using original priors")
    
    # Normalize costs
    models = list(eval_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    
    normalized_costs = {}
    for model_id in models:
        raw_cost = model_costs[model_id]["cost"]
        normalized = (raw_cost - min_cost) / cost_range if cost_range > 0 else 0.0
        normalized_costs[model_id] = {
            "cost": raw_cost,
            "normalized_cost": normalized
        }
    
    # Run ablation: Test 3 key lambda values
    cost_penalties = [0.0, 0.1, 1.0]  # Quality-focused, Balanced, Cost-focused
    
    results_warm = []
    results_cold = []
    
    logger.info("\n" + "="*70)
    logger.info("RUNNING ABLATION")
    logger.info("="*70)
    
    for i, lambda_val in enumerate(cost_penalties, 1):
        logger.info(f"\n[{i}/{len(cost_penalties)}] Testing λ={lambda_val:.1f}")
        
        # Warm-start (standard protocol)
        logger.info("  Warm-start (with dev training)...")
        warm_rewards = []
        warm_costs = []
        for trial in range(5):
            np.random.seed(42 + trial)
            r, c = banditgpt_hybrid_routing(
                train_data, eval_data, encoder, pca, warmup_priors, 
                normalized_costs, lambda_val, debug=False, cold_start=False
            )
            warm_rewards.append(r)
            warm_costs.append(c)
        
        warm_r = np.mean(warm_rewards)
        warm_c = np.mean(warm_costs)
        warm_r_std = np.std(warm_rewards, ddof=1)
        warm_c_std = np.std(warm_costs, ddof=1)
        
        logger.info(f"    Reward: {warm_r:.4f}±{warm_r_std:.4f}")
        logger.info(f"    Cost:   ${warm_c:.6f}±${warm_c_std:.6f}")
        
        results_warm.append({
            "lambda": lambda_val,
            "reward": warm_r,
            "cost": warm_c,
            "reward_std": warm_r_std,
            "cost_std": warm_c_std
        })
        
        # Cold-start (fair comparison)
        logger.info("  Cold-start (NO dev training)...")
        cold_rewards = []
        cold_costs = []
        for trial in range(5):
            np.random.seed(42 + trial)
            r, c = banditgpt_hybrid_routing(
                train_data, eval_data, encoder, pca, warmup_priors, 
                normalized_costs, lambda_val, debug=False, cold_start=True
            )
            cold_rewards.append(r)
            cold_costs.append(c)
        
        cold_r = np.mean(cold_rewards)
        cold_c = np.mean(cold_costs)
        cold_r_std = np.std(cold_rewards, ddof=1)
        cold_c_std = np.std(cold_costs, ddof=1)
        
        logger.info(f"    Reward: {cold_r:.4f}±{cold_r_std:.4f}")
        logger.info(f"    Cost:   ${cold_c:.6f}±${cold_c_std:.6f}")
        
        # Compute degradation
        reward_drop = warm_r - cold_r
        logger.info(f"    Degradation: {reward_drop:.4f} ({100*reward_drop/warm_r:.1f}%)")
        
        results_cold.append({
            "lambda": lambda_val,
            "reward": cold_r,
            "cost": cold_c,
            "reward_std": cold_r_std,
            "cost_std": cold_c_std,
            "degradation": reward_drop
        })
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_file = output_dir / "cold_start_ablation.json"
    
    ablation_data = {
        "metadata": {
            "description": "Cold-start ablation for fair comparison with RouteLLM",
            "n_eval_prompts": len(eval_data),
            "n_trials": 5,
            "note": "RouteLLM has no dev training; this tests banditGPT without dev training"
        },
        "warm_start": results_warm,
        "cold_start": results_cold
    }
    
    with open(output_file, 'w') as f:
        json.dump(ablation_data, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\n{'Lambda':<8} | {'Warm-Start':<12} | {'Cold-Start':<12} | {'Degradation':<12}")
    logger.info("-"*60)
    for w, c in zip(results_warm, results_cold):
        logger.info(f"{w['lambda']:<8.1f} | {w['reward']:.4f}±{w['reward_std']:.4f} | "
                   f"{c['reward']:.4f}±{c['reward_std']:.4f} | "
                   f"{c['degradation']:+.4f} ({100*c['degradation']/w['reward']:+.1f}%)")
    
    logger.info(f"\n✅ Saved results: {output_file}")
    logger.info("\nKEY FINDING:")
    logger.info("  Cold-start performance quantifies the value of dev set training.")
    logger.info("  This provides a fair baseline comparison with RouteLLM.")


if __name__ == "__main__":
    main()
