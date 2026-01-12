#!/usr/bin/env python3
"""
diagnose_quality_scale.py

Diagnostic script to measure the actual range of LinUCB's mean_quality (θ^T x) 
predictions on training prompts. This helps verify whether a scaling issue exists
between quality predictions and cost/latency penalties.

Usage:
    python diagnose_quality_scale.py [--warmup-path PATH] [--splits-path PATH] [--pca-path PATH]
"""

import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL
from src.bandit_gpt.utils import ExperimentBurnIn
from sentence_transformers import SentenceTransformer


def analyze_quality_predictions(router: BanditRouter, prompts: list, 
                                oracle_rewards: dict, models: list) -> dict:
    """
    Analyze the distribution of LinUCB's quality predictions (θ^T x).
    
    Args:
        router: Trained BanditRouter instance
        prompts: List of prompt strings
        oracle_rewards: Dict of {prompt_id: {model_id: reward}}
        models: List of model IDs to analyze
        
    Returns:
        Dict with statistics on prediction distribution
    """
    
    # Store predictions for each model
    predictions_by_model = defaultdict(list)
    actual_rewards_by_model = defaultdict(list)
    
    print("\n📊 Analyzing θ^T x predictions vs actual rewards...")
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Get context vector for this prompt
        x = router._get_context_vector(prompt)
        
        # For each model, compute θ^T x and compare to actual reward
        for model_id in models:
            if model_id not in router.bandit.A:
                continue
                
            # Get LinUCB prediction (θ^T x)
            with router.bandit._lock:
                theta = router.bandit.A_inv[model_id] @ router.bandit.b[model_id]
                mean_quality = float(theta.dot(x))
            
            predictions_by_model[model_id].append(mean_quality)
            
            # Get actual reward for this prompt-model pair
            actual_reward = oracle_rewards.get(prompt, {}).get(model_id, 0.0)
            actual_rewards_by_model[model_id].append(actual_reward)
    
    # Compute statistics
    all_predictions = []
    all_rewards = []
    
    for model_id in models:
        if model_id in predictions_by_model:
            all_predictions.extend(predictions_by_model[model_id])
            all_rewards.extend(actual_rewards_by_model[model_id])
    
    all_predictions = np.array(all_predictions)
    all_rewards = np.array(all_rewards)
    
    stats = {
        "predictions": {
            "min": float(np.min(all_predictions)),
            "max": float(np.max(all_predictions)),
            "mean": float(np.mean(all_predictions)),
            "median": float(np.median(all_predictions)),
            "std": float(np.std(all_predictions)),
            "p25": float(np.percentile(all_predictions, 25)),
            "p75": float(np.percentile(all_predictions, 75)),
        },
        "rewards": {
            "min": float(np.min(all_rewards)),
            "max": float(np.max(all_rewards)),
            "mean": float(np.mean(all_rewards)),
            "median": float(np.median(all_rewards)),
            "std": float(np.std(all_rewards)),
            "p25": float(np.percentile(all_rewards, 25)),
            "p75": float(np.percentile(all_rewards, 75)),
        },
        "per_model": {}
    }
    
    # Per-model statistics
    for model_id in sorted(models):
        if model_id not in predictions_by_model or len(predictions_by_model[model_id]) == 0:
            continue
            
        preds = np.array(predictions_by_model[model_id])
        rewards = np.array(actual_rewards_by_model[model_id])
        
        stats["per_model"][model_id] = {
            "prediction_mean": float(np.mean(preds)),
            "prediction_std": float(np.std(preds)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "n_samples": len(preds)
        }
    
    return stats, predictions_by_model, actual_rewards_by_model


def main(warmup_path: str = None, splits_path: str = None, pca_path: str = None,
         prior_n_effective: float = 20.0):
    """
    Main diagnostic routine.
    
    Args:
        warmup_path: Path to warmup priors .joblib file
        splits_path: Path to splits.json file
        pca_path: Path to PCA model .joblib file
        prior_n_effective: Prior N effective for router initialization
    """
    
    print("=" * 70)
    print("QUALITY SCALE DIAGNOSTIC: Measuring θ^T x Prediction Range")
    print("=" * 70)
    
    # 1. Setup paths
    project_root = Path(__file__).parent.parent
    
    if splits_path is None:
        splits_path = project_root / "experiments" / "01_effectiveness" / "results" / "splits.json"
    else:
        splits_path = Path(splits_path)
    
    if warmup_path is None:
        warmup_path = project_root / "artifacts" / "priors_warmup.joblib"
    else:
        warmup_path = Path(warmup_path)
    
    if pca_path is None:
        pca_path = project_root / "artifacts" / "pca_23.joblib"
    else:
        pca_path = Path(pca_path)
    
    print(f"\n📦 Configuration:")
    print(f"  - Warmup: {warmup_path}")
    print(f"  - Splits: {splits_path}")
    print(f"  - PCA: {pca_path}")
    print(f"  - Prior N: {prior_n_effective}")
    
    # 2. Load data
    print("\n📊 Loading data...")
    from utils.data_loader import load_model_registry
    
    registry = load_model_registry()
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    burner = ExperimentBurnIn(registry, {}, splits_path, encoder=encoder)
    
    # Get dev prompts with rewards
    (dev_prompts, dev_rewards), _ = burner.get_splits(load_rewards=True)
    print(f"  ✓ Loaded {len(dev_prompts)} dev prompts")
    print(f"  ✓ Loaded {len(dev_rewards)} reward entries")
    
    # Get list of models
    models = list(registry.keys())
    print(f"  ✓ {len(models)} models in registry")
    
    # 3. Initialize router with warmup
    print(f"\n🤖 Initializing router with warmup...")
    router = BanditRouter.create(
        registry,
        context_encoder=encoder,
        priors="warmup",
        warmup_path=str(warmup_path),
        prior_n_effective=prior_n_effective,
        pca_path=str(pca_path)
    )
    print("  ✓ Router initialized")
    
    # 4. Train router on dev set (burn-in)
    print(f"\n🔥 Training router on {len(dev_prompts)} dev prompts...")
    router.bandit.alpha = 2.0  # High exploration during training
    
    for prompt in tqdm(dev_prompts, desc="  Training"):
        # Use ARBITRAGE profile for balanced training
        model_id, _ = router.route(prompt, profile="arbitrage")
        reward = burner.oracle_rewards.get(prompt, {}).get(model_id, 0.0)
        router.update(model_id, prompt, reward)
    
    print("  ✓ Training complete")
    
    # 5. Analyze predictions on the same dev set
    stats, preds_by_model, rewards_by_model = analyze_quality_predictions(
        router, dev_prompts, burner.oracle_rewards, models
    )
    
    # 6. Print results
    print("\n" + "=" * 70)
    print("RESULTS: θ^T x Prediction Statistics")
    print("=" * 70)
    
    print("\n📈 OVERALL DISTRIBUTION:")
    print(f"  Predictions (θ^T x):")
    print(f"    - Range: [{stats['predictions']['min']:.4f}, {stats['predictions']['max']:.4f}]")
    print(f"    - Mean: {stats['predictions']['mean']:.4f} ± {stats['predictions']['std']:.4f}")
    print(f"    - Median: {stats['predictions']['median']:.4f}")
    print(f"    - IQR: [{stats['predictions']['p25']:.4f}, {stats['predictions']['p75']:.4f}]")
    
    print(f"\n  Actual Rewards:")
    print(f"    - Range: [{stats['rewards']['min']:.4f}, {stats['rewards']['max']:.4f}]")
    print(f"    - Mean: {stats['rewards']['mean']:.4f} ± {stats['rewards']['std']:.4f}")
    print(f"    - Median: {stats['rewards']['median']:.4f}")
    print(f"    - IQR: [{stats['rewards']['p25']:.4f}, {stats['rewards']['p75']:.4f}]")
    
    # Check for scale mismatch
    pred_range = stats['predictions']['max'] - stats['predictions']['min']
    reward_range = stats['rewards']['max'] - stats['rewards']['min']
    
    print(f"\n⚖️ SCALE ANALYSIS:")
    print(f"  Prediction range width: {pred_range:.4f}")
    print(f"  Reward range width: {reward_range:.4f}")
    print(f"  Ratio: {pred_range / reward_range:.2f}x")
    
    if stats['predictions']['mean'] < 0.5:
        print("\n⚠️ WARNING: Mean prediction is LOW (< 0.5)")
        print("   This suggests predictions are NOT on the same scale as rewards [0, 1]")
        print("   → SCALING ISSUE CONFIRMED")
    else:
        print("\n✅ Mean prediction is reasonable (≥ 0.5)")
        print("   Predictions appear to be on similar scale to binary rewards")
    
    # Show top 10 models by prediction mean
    print("\n" + "=" * 70)
    print("TOP 10 MODELS BY PREDICTION MEAN (θ^T x)")
    print("=" * 70)
    print(f"| {'Model':<40} | {'Pred Mean':<10} | {'Reward Mean':<12} | {'Samples':<8} |")
    print(f"| {'-'*40} | {'-'*10} | {'-'*12} | {'-'*8} |")
    
    sorted_models = sorted(
        stats['per_model'].items(),
        key=lambda x: x[1]['prediction_mean'],
        reverse=True
    )[:10]
    
    for model_id, model_stats in sorted_models:
        display_name = registry[model_id].get('display_name', model_id)[:40]
        print(f"| {display_name:<40} | {model_stats['prediction_mean']:>10.4f} | "
              f"{model_stats['reward_mean']:>12.4f} | {model_stats['n_samples']:>8} |")
    
    # Statistical test: correlation between predictions and rewards
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    all_preds = []
    all_rews = []
    for model_id in models:
        if model_id in preds_by_model:
            all_preds.extend(preds_by_model[model_id])
            all_rews.extend(rewards_by_model[model_id])
    
    correlation = np.corrcoef(all_preds, all_rews)[0, 1]
    print(f"  Pearson correlation (θ^T x vs actual reward): {correlation:.4f}")
    
    if correlation > 0.3:
        print("  ✅ Moderate to strong positive correlation - LinUCB is learning!")
    else:
        print("  ⚠️ Weak correlation - LinUCB may not be learning effectively")
    
    # Save detailed results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "quality_scale_diagnostic.json"
    
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose quality prediction scale")
    parser.add_argument("--warmup-path", type=str, default=None,
                        help="Path to warmup priors .joblib file")
    parser.add_argument("--splits-path", type=str, default=None,
                        help="Path to splits.json file")
    parser.add_argument("--pca-path", type=str, default=None,
                        help="Path to PCA model .joblib file")
    parser.add_argument("--N", type=float, default=20.0,
                        help="Prior N effective (default: 20.0)")
    
    args = parser.parse_args()
    main(warmup_path=args.warmup_path, splits_path=args.splits_path,
         pca_path=args.pca_path, prior_n_effective=args.N)
