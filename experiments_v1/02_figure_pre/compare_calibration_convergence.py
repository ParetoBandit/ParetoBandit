#!/usr/bin/env python3
"""
Calibration Convergence Analysis

Shows that convergence happens DURING calibration, not during holdout.

Compares:
1. Warmup-only policy (before calibration)
2. Gamma-scaled policy (simulated impact of gamma without calibration data)
3. Fully calibrated (after 1,121 calibration samples)

Uses the actual BanditRouter from src/bandit_gpt/router.py to validate
real-world performance.
"""

import argparse
import gzip
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bandit_gpt.router import BanditRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER, 
    STRONG_MODEL_EQUIVALENTS,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    CANONICAL_CALIBRATED_ROUTER_PATH,
    PROJECT_ROOT
)


def create_model_mapper(router_models: List[str], eval_data_sample: dict) -> Dict[str, str]:
    """Create model name mapping between router and evaluation data."""
    available_models = list(eval_data_sample['rewards'].keys())
    
    mapper = {}
    weak_models = ["mistralai/mixtral-8x7b-instruct"]
    strong_models = STRONG_MODEL_EQUIVALENTS
    
    for router_model in router_models:
        if router_model in weak_models:
            mapper[router_model] = router_model
        elif router_model in strong_models:
            for strong in strong_models:
                if strong in available_models:
                    mapper[router_model] = strong
                    break
        else:
            mapper[router_model] = router_model
    
    return mapper


def evaluate_router(router: BanditRouter, eval_data: List[dict], model_mapper: Dict[str, str]) -> Dict:
    """
    Evaluate BanditRouter on holdout data with frozen policy.
    
    Args:
        router: BanditRouter instance (no learning, just routing)
        eval_data: List of evaluation samples with prompts and rewards
        model_mapper: Maps router model IDs to evaluation data model IDs
        
    Returns:
        Dict with evaluation metrics
    """
    # Determine strong model (typically the second model in the list)
    models = list(router.registry.keys())
    strong_model = models[1] if len(models) > 1 else models[0]
    
    model_selections = {m: 0 for m in models}
    total_reward = 0.0
    
    for item in tqdm(eval_data, desc="Evaluating"):
        # Route using the actual BanditRouter (frozen, no feedback)
        selected_model, log = router.route(
            item['prompt'], 
            profile="auto",  # Use default routing profile
            output_tokens=600
        )
        
        # Map to evaluation data model name
        eval_model = model_mapper.get(selected_model, selected_model)
        reward = item['rewards'].get(eval_model, 0.0)
        
        model_selections[selected_model] += 1
        total_reward += reward
    
    return {
        'model_usage': model_selections,
        'total_reward': total_reward,
        'avg_reward': total_reward / len(eval_data),
        'strong_pct': (model_selections[strong_model] / len(eval_data)) * 100
    }


def get_effective_n(router: BanditRouter, models: List[str] = None) -> Dict[str, float]:
    """Get effective sample size for each model in the router."""
    n_eff = {}
    model_list = models if models else router.bandit.models
    for model in model_list:
        if model in router.bandit.A:
            # Trace of A matrix / context_dim ≈ effective N
            n_eff[model] = np.trace(router.bandit.A[model]) / router.bandit.dim
    return n_eff


def main():
    parser = argparse.ArgumentParser(description="Compare before/after calibration convergence")
    parser.add_argument("--warmup-priors", type=str, 
                       default=str(DEFAULT_WARMUP_PRIORS_PATH))
    parser.add_argument("--calibrated-router", type=str, 
                       default=str(CANONICAL_CALIBRATED_ROUTER_PATH))
    parser.add_argument("--holdout-data", type=str, 
                       default=str(CANONICAL_HOLDOUT_DATA_PATH))
    parser.add_argument("--pca", type=str, default=str(DEFAULT_PCA_PATH))
    parser.add_argument("--registry", type=str, 
                       default=str(PROJECT_ROOT / "src" / "bandit_gpt" / "config" / "models.json"),
                       help="Model registry JSON file")
    parser.add_argument("--output", type=str, default="calibration_convergence_comparison")
    parser.add_argument("--gamma", type=float, default=0.010, 
                       help="Gamma value used during calibration")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CALIBRATION CONVERGENCE ANALYSIS")
    print("Showing that convergence happens DURING calibration, not during holdout")
    print("Using actual BanditRouter from src/bandit_gpt/router.py")
    print("="*80)
    
    # Load resources
    print("\n📥 Loading resources...")
    warmup_state = joblib.load(Path(args.warmup_priors))
    calibrated_state = joblib.load(Path(args.calibrated_router))
    
    # Load holdout data (handles both .jsonl and .jsonl.gz)
    if args.holdout_data.endswith('.gz'):
        with gzip.open(args.holdout_data, 'rt') as f:
            raw_data = [json.loads(line) for line in f]
    else:
        with open(args.holdout_data) as f:
            raw_data = [json.loads(line) for line in f]
    
    # Transform data: group by prompt and create rewards dict
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for item in raw_data:
        by_prompt[item['prompt']].append(item)
    
    holdout_data = []
    for prompt, items in by_prompt.items():
        rewards = {}
        for item in items:
            # Use raw_score as reward (0.0-1.0 scale)
            rewards[item['model_id']] = item['raw_score']
        holdout_data.append({
            'prompt': prompt,
            'rewards': rewards
        })
    
    print(f"   ✅ Loaded {len(holdout_data)} holdout samples ({len(raw_data)} model responses)")
    
    # Load model registry
    with open(args.registry) as f:
        registry_data = json.load(f)
        model_registry = {m["openrouter_id"]: m for m in registry_data["models"]}
    
    models = warmup_state['models']
    strong_model = models[1]
    weak_model = models[0]
    
    model_mapper = create_model_mapper(models, holdout_data[0])
    
    print(f"\n📋 Models: {weak_model.split('/')[-1]} (weak), {strong_model.split('/')[-1]} (strong)")
    print(f"📋 Registry: {len(model_registry)} models loaded")
    
    # ========================================================================
    # SCENARIO 1: Warmup-only (before calibration)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: WARMUP-ONLY POLICY (Before Calibration)")
    print("="*80)
    
    # Create router with warmup state only (no gamma scaling, no calibration)
    router_warmup = BanditRouter(
        model_registry=model_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca_path=args.pca,
        alpha=0.0,  # No exploration (frozen policy for evaluation)
        forgetting_factor=1.0,  # No decay
    )
    
    # Load warmup state directly into bandit
    for model_id in models:
        if model_id in warmup_state['A'] and model_id in warmup_state['b']:
            router_warmup.bandit.A[model_id] = warmup_state['A'][model_id].copy()
            router_warmup.bandit.b[model_id] = warmup_state['b'][model_id].copy()
    
    router_warmup.bandit.refresh_inverse_cache()
    
    n_eff_warmup = get_effective_n(router_warmup, models)
    print(f"\n📊 Effective Sample Sizes:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_warmup[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_warmup[strong_model]:,.0f}")
    
    print(f"\n🤖 Evaluating warmup-only policy on holdout...")
    results_warmup = evaluate_router(router_warmup, holdout_data, model_mapper)
    
    print(f"\n📈 Results:")
    print(f"   Strong model usage: {results_warmup['strong_pct']:.1f}%")
    print(f"   Quality: {results_warmup['avg_reward']:.4f}")
    
    # ========================================================================
    # SCENARIO 2: Warmup + Gamma scaling (simulated)
    # ========================================================================
    print("\n" + "="*80)
    print(f"SCENARIO 2: WARMUP + GAMMA SCALING (γ={args.gamma})")
    print("="*80)
    
    # Create router with gamma-scaled warmup state
    router_gamma = BanditRouter(
        model_registry=model_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca_path=args.pca,
        alpha=0.0,  # No exploration (frozen policy)
        forgetting_factor=1.0,
    )
    
    # Apply gamma scaling to warmup matrices before loading
    for model_id in models:
        if model_id in warmup_state['A'] and model_id in warmup_state['b']:
            router_gamma.bandit.A[model_id] = args.gamma * warmup_state['A'][model_id].copy()
            router_gamma.bandit.b[model_id] = args.gamma * warmup_state['b'][model_id].copy()
    
    router_gamma.bandit.refresh_inverse_cache()
    
    n_eff_gamma = get_effective_n(router_gamma, models)
    print(f"\n📊 Effective Sample Sizes After Gamma Scaling:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_gamma[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_gamma[strong_model]:,.0f}")
    print(f"   Reduction: {(1 - n_eff_gamma[strong_model]/n_eff_warmup[strong_model])*100:.1f}%")
    
    print(f"\n🤖 Evaluating gamma-scaled policy on holdout...")
    results_gamma = evaluate_router(router_gamma, holdout_data, model_mapper)
    
    print(f"\n📈 Results:")
    print(f"   Strong model usage: {results_gamma['strong_pct']:.1f}%")
    print(f"   Quality: {results_gamma['avg_reward']:.4f}")
    
    # ========================================================================
    # SCENARIO 3: Fully calibrated (after 1,121 calibration samples)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: FULLY CALIBRATED (After 1,121 Dev Samples)")
    print("="*80)
    
    # Create router with fully calibrated state
    router_calibrated = BanditRouter(
        model_registry=model_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        pca_path=args.pca,
        alpha=0.0,  # No exploration (frozen policy)
        forgetting_factor=1.0,
    )
    
    # Load calibrated state
    for model_id in models:
        if model_id in calibrated_state['A'] and model_id in calibrated_state['b']:
            router_calibrated.bandit.A[model_id] = calibrated_state['A'][model_id].copy()
            router_calibrated.bandit.b[model_id] = calibrated_state['b'][model_id].copy()
    
    router_calibrated.bandit.refresh_inverse_cache()
    
    n_eff_calibrated = get_effective_n(router_calibrated, models)
    print(f"\n📊 Effective Sample Sizes After Calibration:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_calibrated[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_calibrated[strong_model]:,.0f}")
    
    # Calculate calibration contribution
    calib_contribution = n_eff_calibrated[strong_model] - n_eff_gamma[strong_model]
    print(f"\n📊 Calibration Data Contribution:")
    print(f"   Added effective N: {calib_contribution:,.0f}")
    print(f"   Calibration/Prior ratio: {calib_contribution / n_eff_gamma[strong_model]:.3f}")
    
    print(f"\n🤖 Evaluating calibrated policy on holdout...")
    results_calibrated = evaluate_router(router_calibrated, holdout_data, model_mapper)
    
    print(f"\n📈 Results:")
    print(f"   Strong model usage: {results_calibrated['strong_pct']:.1f}%")
    print(f"   Quality: {results_calibrated['avg_reward']:.4f}")
    
    # ========================================================================
    # Create Comparison Plot
    # ========================================================================
    print(f"\n📊 Generating comparison visualization...")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use publication-quality settings for KDD
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'legend.framealpha': 0.9,
        'figure.titlesize': 13,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,  # TrueType fonts for PDF
        'ps.fonttype': 42
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Shorter, cleaner labels for publication
    scenarios = ['Warmup\nOnly', 
                f'γ-Scaled\n($\\gamma$={args.gamma})',
                'Calibrated\n(1.1K samples)']
    
    strong_pcts = [results_warmup['strong_pct'], 
                   results_gamma['strong_pct'], 
                   results_calibrated['strong_pct']]
    qualities = [results_warmup['avg_reward'], 
                results_gamma['avg_reward'], 
                results_calibrated['avg_reward']]
    n_effs = [n_eff_warmup[strong_model], 
              n_eff_gamma[strong_model], 
              n_eff_calibrated[strong_model]]
    
    # Professional color scheme: muted, colorblind-friendly
    # Blue (baseline) -> Orange (intermediate) -> Green (target)
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bar_width = 0.65
    
    # Plot 1: Strong Model Usage - THE CLIFF EFFECT
    ax1 = axes[0]
    bars1 = ax1.bar(range(3), strong_pcts, color=colors, 
                   edgecolor='white', linewidth=1.5, alpha=0.9, width=bar_width)
    
    # Add subtle shadow effect for depth
    for i, bar in enumerate(bars1):
        ax1.bar(i, strong_pcts[i], color='black', alpha=0.1, 
               width=bar_width, zorder=0, linewidth=0)
    
    # Cleaner annotations - focus on the key insight
    # Bracket showing the dramatic drop
    bracket_y = max(results_gamma['strong_pct'], results_calibrated['strong_pct']) + 8
    ax1.plot([1, 1], [results_gamma['strong_pct']+2, bracket_y], 'k-', lw=1.5, alpha=0.6)
    ax1.plot([2, 2], [results_calibrated['strong_pct']+2, bracket_y], 'k-', lw=1.5, alpha=0.6)
    ax1.plot([1, 2], [bracket_y, bracket_y], 'k-', lw=1.5, alpha=0.6)
    
    delta = results_gamma['strong_pct'] - results_calibrated['strong_pct']
    ax1.text(1.5, bracket_y + 3, f'Δ = −{delta:.1f} pp', ha='center', fontsize=10, 
            fontweight='bold', color='#C44E52')
    
    # Add value labels on bars with better positioning
    for i, val in enumerate(strong_pcts):
        y_offset = 3 if val > 5 else 1
        ax1.text(i, val + y_offset,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Strong Model Usage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Policy Convergence', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax1.set_ylim([0, 115])
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Quality Score - THE TRADEOFF
    ax2 = axes[1]
    bars2 = ax2.bar(range(3), qualities, color=colors, 
                   edgecolor='white', linewidth=1.5, alpha=0.9, width=bar_width)
    
    # Add subtle shadow effect
    for i, bar in enumerate(bars2):
        ax2.bar(i, qualities[i], color='black', alpha=0.1, 
               width=bar_width, zorder=0, linewidth=0)
    
    # Show the quality tradeoff with a cleaner line
    quality_drop = results_warmup['avg_reward'] - results_calibrated['avg_reward']
    quality_drop_pct = (quality_drop / results_warmup['avg_reward']) * 100
    
    # Connecting line showing tradeoff
    ax2.plot([0, 2], [results_warmup['avg_reward'], results_calibrated['avg_reward']], 
            color='#8C564B', linestyle='--', alpha=0.5, linewidth=2, zorder=1)
    
    # Annotation with arrow
    mid_y = (results_warmup['avg_reward'] + results_calibrated['avg_reward']) / 2
    ax2.annotate(f'−{quality_drop_pct:.1f}%', 
                xy=(1, mid_y), xytext=(1, mid_y - 0.02),
                ha='center', fontsize=9, fontweight='bold', color='#8C564B',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#8C564B', linewidth=1.5, alpha=0.9))
    
    # Add value labels on bars
    for i, val in enumerate(qualities):
        ax2.text(i, val + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Quality Score', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Quality-Cost Tradeoff', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(scenarios, fontsize=9)
    ax2.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax2.set_ylim([0.80, 0.99])
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Effective N - BAYESIAN PLASTICITY
    ax3 = axes[2]
    bars3 = ax3.bar(range(3), n_effs, color=colors, 
                   edgecolor='white', linewidth=1.5, alpha=0.9, width=bar_width)
    
    # Add subtle shadow effect
    for i, bar in enumerate(bars3):
        ax3.bar(i, n_effs[i], color='black', alpha=0.1, 
               width=bar_width, zorder=0, linewidth=0)
    
    # Cleaner annotations on log scale
    # (1) -> (2): 99% reduction with bracket
    reduction_pct = (1 - n_eff_gamma[strong_model]/n_eff_warmup[strong_model]) * 100
    
    # Geometric mean for log scale positioning
    mid_y_12 = np.sqrt(n_eff_warmup[strong_model] * n_eff_gamma[strong_model])
    ax3.annotate('', xy=(1, n_eff_gamma[strong_model]*1.4), xytext=(0, n_eff_warmup[strong_model]/1.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#4C72B0', alpha=0.7))
    ax3.text(0.5, mid_y_12, f'$\\gamma$-scale\n−{reduction_pct:.0f}%', 
            ha='center', fontsize=8, color='#4C72B0', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     edgecolor='#4C72B0', linewidth=1.2, alpha=0.9))
    
    # (2) -> (3): Calibration adds data
    calib_contribution = n_eff_calibrated[strong_model] - n_eff_gamma[strong_model]
    calib_ratio = calib_contribution / n_eff_gamma[strong_model]
    
    mid_y_23 = np.sqrt(n_eff_gamma[strong_model] * n_eff_calibrated[strong_model])
    ax3.annotate('', xy=(2, n_eff_calibrated[strong_model]/1.3), xytext=(1, n_eff_gamma[strong_model]*1.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#55A868', alpha=0.7))
    ax3.text(1.5, mid_y_23, f'Calib.\n+{calib_contribution:.0f} N\n({calib_ratio:.1f}×)', 
            ha='center', fontsize=8, color='#55A868', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                     edgecolor='#55A868', linewidth=1.2, alpha=0.9))
    
    # Add value labels on bars
    for i, val in enumerate(n_effs):
        y_pos = val * 1.35 if val > 10 else val * 1.5
        ax3.text(i, y_pos,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Effective Sample Size', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Bayesian Plasticity', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(scenarios, fontsize=9)
    ax3.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='gray', which='major')
    ax3.set_yscale('log')
    ax3.set_ylim([2, 700])
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Format y-axis for log scale readability
    from matplotlib.ticker import LogLocator, LogFormatter
    ax3.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax3.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax3.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    
    # Overall figure title - concise for publication
    fig.suptitle(
        'Calibration Convergence: Policy Shift Occurs During Calibration Phase',
        fontsize=13, fontweight='bold', y=0.99
    )
    
    # Add subtle background
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    # Add caption-style note at bottom
    fig.text(0.5, 0.01, 
            'Convergence occurs during calibration (1,121 samples), not during holdout evaluation (750 samples). '
            'γ-scaling enables plasticity; calibration data drives convergence.',
            ha='center', fontsize=8, style='italic', color='gray', wrap=True)
    
    # Save high-resolution version for publication
    plot_file = output_dir / "calibration_convergence_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"   ✅ Saved: {plot_file}")
    
    # Also save PDF for LaTeX (vector format)
    plot_file_pdf = output_dir / "calibration_convergence_comparison.pdf"
    plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"   ✅ Saved: {plot_file_pdf}")
    
    # Save EPS for some journals
    plot_file_eps = output_dir / "calibration_convergence_comparison.eps"
    plt.savefig(plot_file_eps, format='eps', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"   ✅ Saved: {plot_file_eps}")
    
    # Save metrics
    metrics_file = output_dir / "comparison_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'warmup_only': {
                'strong_pct': float(results_warmup['strong_pct']),
                'quality': float(results_warmup['avg_reward']),
                'effective_n': float(n_eff_warmup[strong_model])
            },
            'gamma_scaled': {
                'strong_pct': float(results_gamma['strong_pct']),
                'quality': float(results_gamma['avg_reward']),
                'effective_n': float(n_eff_gamma[strong_model]),
                'gamma': args.gamma
            },
            'fully_calibrated': {
                'strong_pct': float(results_calibrated['strong_pct']),
                'quality': float(results_calibrated['avg_reward']),
                'effective_n': float(n_eff_calibrated[strong_model]),
                'calibration_contribution': float(calib_contribution),
                'calibration_prior_ratio': float(calib_contribution / n_eff_gamma[strong_model])
            }
        }, f, indent=2)
    print(f"   ✅ Saved: {metrics_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERGENCE SUMMARY")
    print("="*80)
    
    print(f"\n🔄 The Convergence Journey:")
    print(f"\n1. WARMUP ONLY (Before Calibration):")
    print(f"   Strong usage: {results_warmup['strong_pct']:.1f}% (too high, GPT-4-turbo bias)")
    print(f"   Quality: {results_warmup['avg_reward']:.4f}")
    print(f"   Effective N: {n_eff_warmup[strong_model]:,.0f} (very strong prior)")
    
    print(f"\n2. AFTER GAMMA SCALING (γ={args.gamma}):")
    print(f"   Strong usage: {results_gamma['strong_pct']:.1f}% (more exploratory)")
    print(f"   Quality: {results_gamma['avg_reward']:.4f}")
    print(f"   Effective N: {n_eff_gamma[strong_model]:,.0f} (weakened prior)")
    print(f"   → Prior weakened {(1 - n_eff_gamma[strong_model]/n_eff_warmup[strong_model])*100:.1f}%")
    
    print(f"\n3. AFTER CALIBRATION (1,121 samples):")
    print(f"   Strong usage: {results_calibrated['strong_pct']:.1f}% (converged to GPT-4o)")
    print(f"   Quality: {results_calibrated['avg_reward']:.4f}")
    print(f"   Effective N: {n_eff_calibrated[strong_model]:,.0f}")
    print(f"   Calibration contribution: {calib_contribution:,.0f} effective samples")
    print(f"   Calibration/Prior ratio: {calib_contribution / n_eff_gamma[strong_model]:.3f}")
    
    print(f"\n🎯 Key Insight:")
    print(f"   Convergence change from warmup → calibrated: {abs(results_warmup['strong_pct'] - results_calibrated['strong_pct']):.1f}%")
    print(f"   This convergence happened DURING calibration (1,121 samples)")
    print(f"   NOT during holdout evaluation (750 samples)")
    
    print(f"\n💡 Why holdout evaluation showed minimal parameter change:")
    print(f"   The policy was already converged after calibration!")
    print(f"   Holdout evaluation is testing a frozen policy, not learning a new one.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


