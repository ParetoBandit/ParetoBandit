#!/usr/bin/env python3
"""
Baseline Ablations: Testing Simpler Routing Strategies

This addresses KDD reviewer concern about missing ablations.
We test:
1. Random routing (uniform selection)
2. ε-greedy (exploit best, explore randomly)
3. UCB only (no corralling, just warmup expert)
4. Tabula rasa only (no priors)

Expected Results:
- Random: Performs between Mixtral and GPT-4 static baselines
- ε-greedy: Better than random but worse than UCB
- UCB only: Competitive but worse than Corralling (lacks adaptivity)
- Tabula rasa: Slower learning than warmup expert
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from generate_pareto_frontier import (
    load_model_costs,
    load_dataset_with_split
)
from bandit_gpt.router import CostAwareLinUCBRouter, CostAwareTabulaRasaRouter
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def random_routing(train_data: List[Dict], eval_data: List[Dict], 
                   model_costs: Dict) -> Tuple[float, float]:
    """Random routing: Select each model with 50% probability."""
    models = list(eval_data[0]["rewards"].keys())
    total_reward = 0.0
    total_cost = 0.0
    
    for p in eval_data:
        # Random selection
        selected = np.random.choice(models)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]
    
    return total_reward / len(eval_data), total_cost / len(eval_data)


def epsilon_greedy_routing(train_data: List[Dict], eval_data: List[Dict],
                           encoder: SentenceTransformer, pca,
                           model_costs: Dict, epsilon: float = 0.1,
                           lambda_penalty: float = 0.0) -> Tuple[float, float]:
    """
    ε-greedy: Exploit best model (1-ε)% of time, explore randomly ε% of time.
    
    Args:
        epsilon: Exploration rate (0.1 = 10% exploration)
    """
    models = list(train_data[0]["rewards"].keys())
    dim = len(embed_prompt(train_data[0]["prompt"], encoder, pca))
    
    # Initialize simple UCB (no corralling)
    router = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=0.1, alpha_end=0.1, cost_penalty=lambda_penalty
    )
    
    # Normalization bounds from train data
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    
    # Training phase
    for p in train_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        
        # ε-greedy selection
        if np.random.random() < epsilon:
            # Explore: random
            selected = np.random.choice(models)
        else:
            # Exploit: best model
            selected = router.select_model(x, total_steps=len(train_data))
        
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)
    
    # Evaluation phase (pure exploitation: ε=0)
    total_reward = 0.0
    total_cost = 0.0
    
    for p in eval_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        selected = router.select_model(x, total_steps=len(train_data))
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]
    
    return total_reward / len(eval_data), total_cost / len(eval_data)


def ucb_only_routing(train_data: List[Dict], eval_data: List[Dict],
                     encoder: SentenceTransformer, pca, warmup_priors: Dict,
                     model_costs: Dict, lambda_penalty: float = 0.0) -> Tuple[float, float]:
    """UCB with warmup priors only (no corralling, no tabula rasa expert)."""
    from generate_pareto_frontier import normalize_prior_strength
    
    models = list(train_data[0]["rewards"].keys())
    scaled_priors = normalize_prior_strength(warmup_priors, target_sample_size=10.0)
    
    # Single warmup expert (no corralling)
    router = CostAwareLinUCBRouter(
        models=models, warmup_priors=scaled_priors, model_costs=model_costs,
        alpha_start=2.0, alpha_end=0.1, cost_penalty=lambda_penalty
    )
    
    # Normalization bounds
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    
    # Training phase
    burn_in_steps = len(train_data)
    for p in train_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        selected = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)
    
    # Evaluation phase
    total_reward = 0.0
    total_cost = 0.0
    
    for p in eval_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]
    
    return total_reward / len(eval_data), total_cost / len(eval_data)


def tabula_rasa_only_routing(train_data: List[Dict], eval_data: List[Dict],
                              encoder: SentenceTransformer, pca,
                              model_costs: Dict, lambda_penalty: float = 0.0) -> Tuple[float, float]:
    """Tabula rasa only (no priors, no corralling)."""
    models = list(train_data[0]["rewards"].keys())
    dim = len(embed_prompt(train_data[0]["prompt"], encoder, pca))
    
    # Single tabula rasa expert (no corralling)
    router = CostAwareTabulaRasaRouter(
        models=models, context_dim=dim, model_costs=model_costs,
        alpha_start=2.0, alpha_end=0.1, cost_penalty=lambda_penalty
    )
    
    # Normalization bounds
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    
    # Training phase
    burn_in_steps = len(train_data)
    for p in train_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        selected = router.select_model(x, total_steps=burn_in_steps)
        norm_r = (p["rewards"][selected] - r_min) / r_range
        router.update(x, selected, norm_r)
    
    # Evaluation phase
    total_reward = 0.0
    total_cost = 0.0
    
    for p in eval_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        selected = router.select_model(x, total_steps=burn_in_steps)
        total_reward += p["rewards"][selected]
        total_cost += model_costs[selected]["cost"]
    
    return total_reward / len(eval_data), total_cost / len(eval_data)


def main():
    logger.info("="*70)
    logger.info("BASELINE ABLATIONS: Testing Simpler Routing Strategies")
    logger.info("="*70)
    logger.info("\nThis addresses reviewer concern about missing baselines:")
    logger.info("  1. Random routing (uniform selection)")
    logger.info("  2. ε-greedy (explore/exploit)")
    logger.info("  3. UCB only (warmup expert, no corralling)")
    logger.info("  4. Tabula rasa only (no priors, no corralling)")
    
    # Load data and models
    logger.info("\n📦 Loading data and models...")
    model_costs = load_model_costs()
    train_data, eval_data, stats = load_dataset_with_split()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    # Use sanitized priors
    sanitized_priors_path = Path(DEFAULT_WARMUP_PRIORS_PATH).parent / "priors_warmup_normalized.joblib"
    warmup_priors = joblib.load(sanitized_priors_path if sanitized_priors_path.exists() 
                                  else DEFAULT_WARMUP_PRIORS_PATH)
    
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
    
    # Test configurations
    test_lambdas = [0.0, 0.1, 1.0]  # Quality-focused, Balanced, Cost-focused
    results = {}
    
    logger.info("\n" + "="*70)
    logger.info("RUNNING ABLATIONS")
    logger.info("="*70)
    
    # 1. Random routing (lambda-independent)
    logger.info("\n[1/4] Random Routing...")
    random_rewards = []
    random_costs = []
    for trial in range(5):
        np.random.seed(42 + trial)
        r, c = random_routing(train_data, eval_data, normalized_costs)
        random_rewards.append(r)
        random_costs.append(c)
    
    results["Random"] = {
        "reward": np.mean(random_rewards),
        "cost": np.mean(random_costs),
        "reward_std": np.std(random_rewards, ddof=1),
        "cost_std": np.std(random_costs, ddof=1)
    }
    logger.info(f"  Reward: {results['Random']['reward']:.4f}±{results['Random']['reward_std']:.4f}")
    logger.info(f"  Cost:   ${results['Random']['cost']:.6f}±${results['Random']['cost_std']:.6f}")
    
    # 2-4. Other methods (lambda-dependent)
    for method_name, method_func in [
        ("ε-greedy (ε=0.1)", lambda t, e, l: epsilon_greedy_routing(t, e, encoder, pca, normalized_costs, 0.1, l)),
        ("UCB Only", lambda t, e, l: ucb_only_routing(t, e, encoder, pca, warmup_priors, normalized_costs, l)),
        ("Tabula Rasa Only", lambda t, e, l: tabula_rasa_only_routing(t, e, encoder, pca, normalized_costs, l))
    ]:
        logger.info(f"\n[{list(results.keys()).index(method_name) + 2 if method_name in results else len(results) + 2}/4] {method_name}...")
        method_results = []
        
        for lambda_val in test_lambdas:
            trial_rewards = []
            trial_costs = []
            
            for trial in range(5):
                np.random.seed(42 + trial)
                r, c = method_func(train_data, eval_data, lambda_val)
                trial_rewards.append(r)
                trial_costs.append(c)
            
            avg_r = np.mean(trial_rewards)
            avg_c = np.mean(trial_costs)
            std_r = np.std(trial_rewards, ddof=1)
            std_c = np.std(trial_costs, ddof=1)
            
            method_results.append({
                "lambda": lambda_val,
                "reward": avg_r,
                "cost": avg_c,
                "reward_std": std_r,
                "cost_std": std_c
            })
            
            logger.info(f"  λ={lambda_val:.1f}: Reward={avg_r:.4f}±{std_r:.4f}, Cost=${avg_c:.6f}±${std_c:.6f}")
        
        results[method_name] = method_results
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_file = output_dir / "baseline_ablations.json"
    
    ablation_data = {
        "metadata": {
            "description": "Baseline ablations: Random, ε-greedy, UCB only, Tabula rasa only",
            "n_eval_prompts": len(eval_data),
            "n_trials": 5
        },
        "baselines": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(ablation_data, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info("\nComparison at λ=0.0 (quality-focused):")
    logger.info(f"{'Method':<25} | {'Reward':<15} | {'Cost':<15}")
    logger.info("-"*60)
    logger.info(f"{'Random':<25} | {results['Random']['reward']:.4f}±{results['Random']['reward_std']:.4f}   | ${results['Random']['cost']:.6f}")
    for method in ["ε-greedy (ε=0.1)", "UCB Only", "Tabula Rasa Only"]:
        r = results[method][0]  # λ=0.0 is first
        logger.info(f"{method:<25} | {r['reward']:.4f}±{r['reward_std']:.4f}   | ${r['cost']:.6f}")
    
    logger.info(f"\n✅ Saved results: {output_file}")
    logger.info("\nKEY FINDING:")
    logger.info("  These ablations establish lower bounds for comparison.")
    logger.info("  banditGPT-Hybrid (Corralling) should outperform all of these.")


if __name__ == "__main__":
    main()
