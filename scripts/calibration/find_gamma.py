#!/usr/bin/env python3
"""
Find Optimal Gamma Calibration Factor (CLI Tool)

This script helps users determine the best gamma (covariance inflation) factor
for their specific calibration dataset. It tests multiple gamma values and
visualizes the adaptation curves.

Usage:
    python3 find_gamma.py --calibration-data my_data.jsonl --output results/

Required data format (calibration-data):
    {"prompt": "...", "rewards": {"model_a": 0.85, "model_b": 0.95}}
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt


def run_calibration_experiment(
    calibration_data: List[dict],
    warmup_priors: dict,
    encoder: SentenceTransformer,
    pca_model,
    gamma: float
) -> Tuple[List[Dict], float]:
    """Run calibration with a specific gamma value."""
    
    # Apply gamma scaling
    priors_scaled = apply_gamma_scaling(warmup_priors, gamma)
    
    # Initialize router
    router = SimpleLinUCBRouter(
        models=warmup_priors['models'],
        warmup_priors=priors_scaled,
        alpha=1.0
    )
    
    # Track metrics over time
    metrics = []
    
    for i, item in enumerate(calibration_data):
        # Embed
        context = embed_prompt(item['prompt'], encoder, pca_model)
        
        # Select and update
        selected_model = router.select_model(context)
        reward = item['rewards'].get(selected_model, 0.0)
        router.update(context, selected_model, reward)
        
        # Record metrics every 10 samples
        if i % 10 == 0 or i == len(calibration_data) - 1:
            usage = router.get_model_usage()
            metrics.append({
                'sample': i + 1,
                'model_usage': usage,
                'strong_pct': usage.get(warmup_priors['models'][1], 0.0)  # Assume 2nd is strong
            })
    
    # Final strong model usage
    final_strong_pct = metrics[-1]['strong_pct']
    
    return metrics, final_strong_pct


def main():
    parser = argparse.ArgumentParser(description="Find optimal gamma calibration factor")
    parser.add_argument(
        "--calibration-data", type=str, required=True,
        help="Path to calibration data (JSONL with 'prompt' and 'rewards' fields)"
    )
    parser.add_argument(
        "--warmup-priors", type=str,
        default="data/routellm/artifacts/priors_warmup_routellm_pca24.joblib",
        help="Path to warmup priors"
    )
    parser.add_argument(
        "--pca", type=str,
        default="artifacts/pca_23_routellm.joblib",
        help="Path to PCA model"
    )
    parser.add_argument(
        "--output", type=str, default="calibration_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--gamma-values", type=float, nargs='+',
        default=[1.0, 0.1, 0.01, 0.005, 0.002, 0.001],
        help="Gamma values to test"
    )
    parser.add_argument(
        "--target-usage", type=float, default=None,
        help="Target strong model usage %% (if known from oracle)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FIND OPTIMAL GAMMA CALIBRATION FACTOR")
    print("="*80)
    
    # Load resources
    print("\n📥 Loading resources...")
    warmup_priors = joblib.load(Path(args.warmup_priors))
    pca_model = joblib.load(Path(args.pca))
    from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Warmup priors: {warmup_priors['n_prompts']:,} samples")
    print(f"   ✅ PCA: {pca_model.n_components} components")
    print(f"   ✅ Models: {', '.join(warmup_priors['models'])}")
    
    # Load calibration data
    print(f"\n📊 Loading calibration data from: {args.calibration_data}")
    with open(args.calibration_data) as f:
        calibration_data = [json.loads(line) for line in f]
    print(f"   ✅ Loaded {len(calibration_data)} calibration samples")
    
    # Validate data format
    if not calibration_data:
        print("❌ No calibration data found!")
        return
    
    first_item = calibration_data[0]
    if 'prompt' not in first_item or 'rewards' not in first_item:
        print("❌ Invalid format! Expected: {'prompt': '...', 'rewards': {'model': 0.0}}")
        return
    
    # Run experiments
    print(f"\n🔬 Testing {len(args.gamma_values)} gamma values...")
    results = {}
    
    for gamma in tqdm(args.gamma_values, desc="Gamma experiments"):
        metrics, final_usage = run_calibration_experiment(
            calibration_data,
            warmup_priors,
            encoder,
            pca_model,
            gamma
        )
        results[gamma] = {
            'metrics': metrics,
            'final_strong_pct': final_usage,
            'eff_n': int(warmup_priors['n_prompts'] * gamma),
            'calib_prior_ratio': len(calibration_data) / (warmup_priors['n_prompts'] * gamma)
        }
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison table
    print("\n" + "="*80)
    print("RESULTS: Gamma Factor Comparison")
    print("="*80)
    print(f"\n{'Gamma':>8} {'Eff. N':>10} {'Calib/Prior':>12} {'Final Strong%':>14} {'Delta':>10}")
    print("-"*80)
    
    baseline_usage = results[1.0]['final_strong_pct']
    
    for gamma in sorted(args.gamma_values):
        r = results[gamma]
        delta = r['final_strong_pct'] - baseline_usage
        print(f"{gamma:>8.3f} {r['eff_n']:>10,} {r['calib_prior_ratio']:>12.3f} {r['final_strong_pct']:>13.1f}% {delta:>+9.1f}%")
    
    print("="*80)
    
    # Find optimal gamma
    if args.target_usage is not None:
        print(f"\n🎯 Target strong model usage: {args.target_usage:.1f}%")
        best_gamma = min(args.gamma_values, 
                        key=lambda g: abs(results[g]['final_strong_pct'] - args.target_usage))
        gap = abs(results[best_gamma]['final_strong_pct'] - args.target_usage)
        print(f"   Best gamma: {best_gamma:.3f} (gap: {gap:.1f}%)")
    else:
        # Recommend based on maximum adaptation
        deltas = {g: abs(results[g]['final_strong_pct'] - baseline_usage) 
                 for g in args.gamma_values if g < 1.0}
        best_gamma = max(deltas, key=deltas.get) if deltas else 0.002
        print(f"\n💡 Recommended gamma: {best_gamma:.3f}")
        print(f"   (Maximum adaptation from baseline: {deltas[best_gamma]:.1f}%)")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Calibration curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(args.gamma_values)))
    for gamma, color in zip(sorted(args.gamma_values), colors):
        metrics = results[gamma]['metrics']
        samples = [m['sample'] for m in metrics]
        strong_pct = [m['strong_pct'] for m in metrics]
        label = f'γ={gamma:.3f}' if gamma < 0.1 else f'γ={gamma}'
        axes[0, 0].plot(samples, strong_pct, linewidth=2, color=color, label=label)
    
    if args.target_usage:
        axes[0, 0].axhline(args.target_usage, color='red', linestyle='--', 
                          linewidth=2, label=f'Target ({args.target_usage:.1f}%)')
    
    axes[0, 0].set_xlabel('Calibration Samples', fontsize=12)
    axes[0, 0].set_ylabel('Strong Model Usage (%)', fontsize=12)
    axes[0, 0].set_title('Calibration Adaptation by Gamma', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Final usage vs gamma
    gammas = sorted(args.gamma_values)
    final_usages = [results[g]['final_strong_pct'] for g in gammas]
    
    axes[0, 1].plot(gammas, final_usages, 'o-', linewidth=2, markersize=10, color='steelblue')
    if args.target_usage:
        axes[0, 1].axhline(args.target_usage, color='red', linestyle='--', 
                          linewidth=2, label='Target')
    axes[0, 1].set_xlabel('Gamma Factor (γ)', fontsize=12)
    axes[0, 1].set_ylabel('Final Strong Model Usage (%)', fontsize=12)
    axes[0, 1].set_title('Effect of Prior Weakening', fontsize=14, fontweight='bold')
    axes[0, 1].set_xscale('log')
    if args.target_usage:
        axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Calibration/Prior ratio
    ratios = [results[g]['calib_prior_ratio'] for g in gammas]
    
    axes[1, 0].bar(range(len(gammas)), ratios, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Equal influence')
    axes[1, 0].set_xticks(range(len(gammas)))
    axes[1, 0].set_xticklabels([f'{g:.3f}' if g < 0.1 else f'{g}' for g in gammas], rotation=45)
    axes[1, 0].set_xlabel('Gamma Factor (γ)', fontsize=12)
    axes[1, 0].set_ylabel('Calibration/Prior Ratio', fontsize=12)
    axes[1, 0].set_title('Balance of Calibration vs Prior Influence', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Adaptation magnitude
    deltas = [results[g]['final_strong_pct'] - baseline_usage for g in gammas]
    
    bars = axes[1, 1].bar(range(len(gammas)), deltas, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_xticks(range(len(gammas)))
    axes[1, 1].set_xticklabels([f'{g:.3f}' if g < 0.1 else f'{g}' for g in gammas], rotation=45)
    axes[1, 1].set_xlabel('Gamma Factor (γ)', fontsize=12)
    axes[1, 1].set_ylabel('Change from Baseline (%)', fontsize=12)
    axes[1, 1].set_title('Adaptation Magnitude', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{delta:+.1f}%',
                       ha='center', va='bottom' if delta > 0 else 'top', 
                       fontsize=9, fontweight='bold')
    
    plt.suptitle(
        f'Gamma Calibration Analysis\n'
        f'{len(calibration_data)} calibration samples | '
        f'{warmup_priors["n_prompts"]:,} warmup samples | '
        f'Recommended: γ={best_gamma:.3f}',
        fontsize=13, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_file = output_dir / "gamma_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {plot_file}")
    
    # Save results
    results_file = output_dir / "gamma_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'gamma_values_tested': args.gamma_values,
            'calibration_samples': len(calibration_data),
            'warmup_samples': warmup_priors['n_prompts'],
            'results': {str(k): {
                'final_strong_pct': v['final_strong_pct'],
                'eff_n': v['eff_n'],
                'calib_prior_ratio': v['calib_prior_ratio']
            } for k, v in results.items()},
            'recommended_gamma': float(best_gamma),
            'target_usage': args.target_usage
        }, f, indent=2)
    print(f"   ✅ Saved: {results_file}")
    
    print("\n" + "="*80)
    print("✅ GAMMA ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Use gamma = {best_gamma:.3f} for your domain")
    print(f"   This gives Calibration/Prior ratio = {results[best_gamma]['calib_prior_ratio']:.3f}")
    print(f"   Final strong model usage = {results[best_gamma]['final_strong_pct']:.1f}%")
    print(f"\n📋 Next steps:")
    print(f"   1. Review plots in: {output_dir}/gamma_analysis.png")
    print(f"   2. Use calibrate_router.py with --gamma {best_gamma:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()

