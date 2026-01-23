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
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER, STRONG_MODEL_EQUIVALENTS


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


def get_effective_n(router: BanditRouter) -> Dict[str, float]:
    """Get effective sample size for each model in the router."""
    n_eff = {}
    for model in router.bandit.models:
        # Trace of A matrix / context_dim ≈ effective N
        n_eff[model] = np.trace(router.bandit.A[model]) / router.bandit.dim
    return n_eff


def main():
    parser = argparse.ArgumentParser(description="Compare before/after calibration convergence")
    parser.add_argument("--warmup-priors", type=str, 
                       default="../../../data/routellm/artifacts/priors_warmup_routellm_pca24.joblib")
    parser.add_argument("--calibrated-router", type=str, 
                       default="../data/canonical_router_calibrated.joblib")
    parser.add_argument("--holdout-data", type=str, 
                       default="../data/canonical_holdout_evaluation.jsonl")
    parser.add_argument("--pca", type=str, default="../../../artifacts/pca_23_routellm.joblib")
    parser.add_argument("--registry", type=str, 
                       default="../../../src/bandit_gpt/config/models.json",
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
    
    # Load holdout data
    with open(args.holdout_data) as f:
        holdout_data = [json.loads(line) for line in f]
    
    print(f"   ✅ Loaded {len(holdout_data)} holdout samples")
    
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
    
    n_eff_warmup = get_effective_n(router_warmup)
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
    
    n_eff_gamma = get_effective_n(router_gamma)
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
    
    n_eff_calibrated = get_effective_n(router_calibrated)
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scenarios = ['Warmup Only\n(Before Calibration)', 
                f'Warmup + γ={args.gamma}\n(No Calibration Data)',
                'Fully Calibrated\n(After 1,121 samples)']
    strong_pcts = [results_warmup['strong_pct'], 
                   results_gamma['strong_pct'], 
                   results_calibrated['strong_pct']]
    qualities = [results_warmup['avg_reward'], 
                results_gamma['avg_reward'], 
                results_calibrated['avg_reward']]
    n_effs = [n_eff_warmup[strong_model], 
              n_eff_gamma[strong_model], 
              n_eff_calibrated[strong_model]]
    
    # Plot 1: Strong Model Usage
    ax1 = axes[0]
    bars1 = ax1.bar(scenarios, strong_pcts, color=['#d62728', '#ff7f0e', '#2ca02c'], 
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax1.axhline(16.3, color='gold', linestyle='--', linewidth=2, label='Oracle Optimal (16.3%)')
    ax1.axhline(23.3, color='green', linestyle=':', linewidth=2, label='Target (23.3%)')
    
    # Add value labels on bars
    for bar, val in zip(bars1, strong_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Strong Model Usage (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Usage Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 50])
    
    # Plot 2: Quality Score
    ax2 = axes[1]
    bars2 = ax2.bar(scenarios, qualities, color=['#d62728', '#ff7f0e', '#2ca02c'], 
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax2.axhline(0.9853, color='gold', linestyle='--', linewidth=2, label='Oracle (0.9853)')
    ax2.axhline(0.8507, color='green', linestyle=':', linewidth=2, label='Target (0.8507)')
    
    for bar, val in zip(bars2, qualities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Quality Score', fontsize=13, fontweight='bold')
    ax2.set_title('Quality Maintenance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0.8, 1.0])
    
    # Plot 3: Effective N (showing calibration impact)
    ax3 = axes[2]
    bars3 = ax3.bar(scenarios, n_effs, color=['#d62728', '#ff7f0e', '#2ca02c'], 
                   edgecolor='black', linewidth=2, alpha=0.8)
    
    for bar, val in zip(bars3, n_effs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax3.set_ylabel('Effective Sample Size', fontsize=13, fontweight='bold')
    ax3.set_title('Prior Strength (Lower = More Adaptable)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_yscale('log')
    
    plt.suptitle(
        f'Calibration Convergence: The Impact of γ-Scaling + Calibration Data\n'
        f'Convergence happens DURING calibration (1,121 samples), not during holdout evaluation',
        fontsize=13, fontweight='bold', y=1.00
    )
    
    plt.tight_layout()
    plot_file = output_dir / "calibration_convergence_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {plot_file}")
    
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


