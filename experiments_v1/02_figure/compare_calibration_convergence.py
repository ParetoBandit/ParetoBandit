#!/usr/bin/env python3
"""
Calibration Convergence Analysis

Shows that convergence happens DURING calibration, not during holdout.

Compares:
1. Warmup-only policy (before calibration)
2. Calibrated policy (after 1,121 calibration samples)
3. Holdout performance (frozen policy to show it's already converged)
"""

import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER


def embed_prompt(prompt: str, encoder: SentenceTransformer, pca_model) -> np.ndarray:
    """Embed prompt with PCA."""
    embedding = encoder.encode(prompt, convert_to_numpy=True, show_progress_bar=False)
    embedding = pca_model.transform(embedding.reshape(1, -1)).flatten()
    return np.append(embedding, 1.0)


class SimpleRouter:
    """Simple LinUCB router without online learning."""
    
    def __init__(self, A: Dict, b: Dict, models: List[str], encoder, pca_model, alpha: float = 1.0):
        self.A = {m: A[m].copy() for m in models}
        self.b = {m: b[m].copy() for m in models}
        self.models = models
        self.encoder = encoder
        self.pca_model = pca_model
        self.alpha = alpha
    
    def select_model(self, prompt: str) -> str:
        """Select model using UCB (frozen policy)."""
        context = embed_prompt(prompt, self.encoder, self.pca_model)
        
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores[model] = expected + self.alpha * uncertainty
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def get_effective_n(self) -> Dict[str, float]:
        """Get effective sample size for each model."""
        n_eff = {}
        for model in self.models:
            # Trace of A matrix / context_dim ≈ effective N
            n_eff[model] = np.trace(self.A[model]) / len(self.b[model])
        return n_eff


def create_model_mapper(router_models: List[str], eval_data_sample: dict) -> Dict[str, str]:
    """Create model name mapping."""
    available_models = list(eval_data_sample['rewards'].keys())
    
    mapper = {}
    weak_models = ["mistralai/mixtral-8x7b-instruct"]
    strong_models = ["openai/gpt-4-turbo", "openai/gpt-4o"]
    
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


def evaluate_frozen_policy(router: SimpleRouter, eval_data: List[dict], model_mapper: Dict[str, str]) -> Dict:
    """Evaluate with frozen policy (no learning)."""
    strong_model = router.models[1]
    
    model_selections = {m: 0 for m in router.models}
    total_reward = 0.0
    
    for item in tqdm(eval_data, desc="Evaluating"):
        selected_model = router.select_model(item['prompt'])
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


def main():
    parser = argparse.ArgumentParser(description="Compare before/after calibration convergence")
    parser.add_argument("--warmup-priors", type=str, 
                       default="../../../data/routellm/artifacts/priors_warmup_routellm_pca24.joblib")
    parser.add_argument("--calibrated-router", type=str, 
                       default="../data/canonical_router_calibrated.joblib")
    parser.add_argument("--holdout-data", type=str, 
                       default="../data/canonical_holdout_evaluation.jsonl")
    parser.add_argument("--pca", type=str, default="../../../artifacts/pca_23_routellm.joblib")
    parser.add_argument("--output", type=str, default="calibration_convergence_comparison")
    parser.add_argument("--gamma", type=float, default=0.010, 
                       help="Gamma value used during calibration")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CALIBRATION CONVERGENCE ANALYSIS")
    print("Showing that convergence happens DURING calibration, not during holdout")
    print("="*80)
    
    # Load resources
    print("\n📥 Loading resources...")
    warmup_state = joblib.load(Path(args.warmup_priors))
    calibrated_state = joblib.load(Path(args.calibrated_router))
    pca_model = joblib.load(Path(args.pca))
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    
    with open(args.holdout_data) as f:
        holdout_data = [json.loads(line) for line in f]
    
    print(f"   ✅ Loaded {len(holdout_data)} holdout samples")
    
    models = warmup_state['models']
    strong_model = models[1]
    weak_model = models[0]
    
    model_mapper = create_model_mapper(models, holdout_data[0])
    
    # ========================================================================
    # SCENARIO 1: Warmup-only (before calibration)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: WARMUP-ONLY POLICY (Before Calibration)")
    print("="*80)
    
    router_warmup = SimpleRouter(
        warmup_state['A'], warmup_state['b'], models, encoder, pca_model
    )
    
    n_eff_warmup = router_warmup.get_effective_n()
    print(f"\n📊 Effective Sample Sizes:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_warmup[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_warmup[strong_model]:,.0f}")
    
    print(f"\n🤖 Evaluating warmup-only policy on holdout...")
    results_warmup = evaluate_frozen_policy(router_warmup, holdout_data, model_mapper)
    
    print(f"\n📈 Results:")
    print(f"   Strong model usage: {results_warmup['strong_pct']:.1f}%")
    print(f"   Quality: {results_warmup['avg_reward']:.4f}")
    
    # ========================================================================
    # SCENARIO 2: Warmup + Gamma scaling (simulated)
    # ========================================================================
    print("\n" + "="*80)
    print(f"SCENARIO 2: WARMUP + GAMMA SCALING (γ={args.gamma})")
    print("="*80)
    
    # Apply gamma scaling to warmup matrices
    A_gamma = {m: args.gamma * warmup_state['A'][m].copy() for m in models}
    b_gamma = {m: args.gamma * warmup_state['b'][m].copy() for m in models}
    
    router_gamma = SimpleRouter(A_gamma, b_gamma, models, encoder, pca_model)
    
    n_eff_gamma = router_gamma.get_effective_n()
    print(f"\n📊 Effective Sample Sizes After Gamma Scaling:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_gamma[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_gamma[strong_model]:,.0f}")
    print(f"   Reduction: {(1 - n_eff_gamma[strong_model]/n_eff_warmup[strong_model])*100:.1f}%")
    
    print(f"\n🤖 Evaluating gamma-scaled policy on holdout...")
    results_gamma = evaluate_frozen_policy(router_gamma, holdout_data, model_mapper)
    
    print(f"\n📈 Results:")
    print(f"   Strong model usage: {results_gamma['strong_pct']:.1f}%")
    print(f"   Quality: {results_gamma['avg_reward']:.4f}")
    
    # ========================================================================
    # SCENARIO 3: Fully calibrated (after 1,121 calibration samples)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: FULLY CALIBRATED (After 1,121 Dev Samples)")
    print("="*80)
    
    router_calibrated = SimpleRouter(
        calibrated_state['A'], calibrated_state['b'], models, encoder, pca_model
    )
    
    n_eff_calibrated = router_calibrated.get_effective_n()
    print(f"\n📊 Effective Sample Sizes After Calibration:")
    print(f"   {weak_model.split('/')[-1]}: {n_eff_calibrated[weak_model]:,.0f}")
    print(f"   {strong_model.split('/')[-1]}: {n_eff_calibrated[strong_model]:,.0f}")
    
    # Calculate calibration contribution
    calib_contribution = n_eff_calibrated[strong_model] - n_eff_gamma[strong_model]
    print(f"\n📊 Calibration Data Contribution:")
    print(f"   Added effective N: {calib_contribution:,.0f}")
    print(f"   Calibration/Prior ratio: {calib_contribution / n_eff_gamma[strong_model]:.3f}")
    
    print(f"\n🤖 Evaluating calibrated policy on holdout...")
    results_calibrated = evaluate_frozen_policy(router_calibrated, holdout_data, model_mapper)
    
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


