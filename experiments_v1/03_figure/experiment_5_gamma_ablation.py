#!/usr/bin/env python3
"""
Experiment 5: Gamma (γ) Ablation Study
=======================================

Validates the claim: "γ=0.05 prevents expert death"

Tests 5 gamma values to understand the effect of exploration mixing:
- γ=0.00: No mixing (pure exponential weighting)
- γ=0.01: Minimal mixing
- γ=0.05: Current design
- γ=0.10: Moderate mixing
- γ=0.20: High mixing

Measures:
- Cumulative regret
- Minimum expert weight over time (expert death indicator)
- Weight stability (variance)

Author: BanditGPT Team
Date: 2026-02-12
Status: NEW (addresses KDD reviewer Issue 5)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import logging
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import CorrallingRouter, CostAwareLinUCBRouter, CostAwareTabulaRasaRouter
from bandit_gpt.calibration import embed_prompt, apply_gamma_scaling
from sentence_transformers import SentenceTransformer
import joblib
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SEEDS = 5  # Gamma ablation with multiple seeds
LEARNING_RATE = 1.0
ALPHA_CONSTANT = 2.0  # From ablation study
GAMMA_VALUES = [0.001, 0.01, 0.05, 0.10, 0.20]  # Start from 0.001 to avoid singular matrix at 0.0
OUTPUT_DIR = Path(__file__).parent / "results" / "gamma_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_holdout_data():
    """Load holdout dataset."""
    import gzip
    
    entries = []
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))
    
    prompt_data = {}
    for entry in entries:
        prompt = entry['prompt']
        model_id = entry['model_id']
        score = entry.get('raw_score', 0.0)
        
        if prompt not in prompt_data:
            prompt_data[prompt] = {'prompt': prompt, 'scores': {}}
        
        prompt_data[prompt]['scores'][model_id] = score
    
    return list(prompt_data.values())

# ============================================================================
# SINGLE TRIAL
# ============================================================================
def run_single_trial(data: List[Dict], encoder, pca, warmup_priors, 
                     models: List[str], context_dim: int, seed: int, gamma: float) -> Dict:
    """Run a single trial with specific gamma value."""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Apply gamma scaling to priors
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=gamma)
    
    # Initialize experts
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors_scaled,
        alpha_start=ALPHA_CONSTANT,
        alpha_end=ALPHA_CONSTANT,
        cost_penalty=0.0,
        model_costs={m: {"normalized_cost": 0.0} for m in models}
    )
    
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha_start=ALPHA_CONSTANT,
        alpha_end=ALPHA_CONSTANT,
        cost_penalty=0.0,
        model_costs={m: {"normalized_cost": 0.0} for m in models}
    )
    
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=LEARNING_RATE,
        gamma=gamma
    )
    
    # Track metrics
    cumulative_regret = 0.0
    weight_history = []
    min_weights = []
    
    # Shuffle data
    indices = rng.permutation(len(data))
    
    for step, idx in enumerate(indices):
        sample = data[idx]
        prompt = sample['prompt']
        context = embed_prompt(prompt, encoder, pca)
        
        # Select model
        selected_model = router.select_model(context)
        
        # Get reward
        scores = sample.get('scores', {})
        if not scores:
            continue
        
        oracle_model = max(scores, key=scores.get)
        oracle_reward = scores[oracle_model]
        model_reward = scores.get(selected_model, 0.0)
        
        # Update
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        
        router.update(context, selected_model, model_reward)
        
        # Track weights
        weight_history.append({
            'step': step + 1,
            'warmup': float(router.weights[0]),
            'tabula_rasa': float(router.weights[1])
        })
        
        # Track minimum weight (for expert death detection)
        min_weights.append(min(router.weights))
    
    return {
        'gamma': gamma,
        'seed': seed,
        'final_regret': cumulative_regret,
        'weight_history': weight_history,
        'min_weights': min_weights,
        'final_min_weight': min(router.weights),
        'weight_variance': np.var([w['warmup'] for w in weight_history])
    }

# ============================================================================
# MULTI-GAMMA EXPERIMENT
# ============================================================================
def run_gamma_ablation():
    """Run ablation study across gamma values."""
    logger.info("="*80)
    logger.info("EXPERIMENT 5: GAMMA (γ) ABLATION STUDY")
    logger.info("="*80)
    
    # Load resources
    logger.info("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    logger.info(f"   ✅ Models: {len(models)}")
    logger.info(f"   ✅ Context Dim: {context_dim}")
    logger.info(f"   ✅ Testing γ values: {GAMMA_VALUES}")
    
    # Load data
    data = load_holdout_data()
    logger.info(f"   ✅ Data size: {len(data)}")
    
    # Run all gamma values
    logger.info(f"\n🔬 Running ablation: {len(GAMMA_VALUES)} γ values × {N_SEEDS} seeds...")
    
    all_results = {gamma: [] for gamma in GAMMA_VALUES}
    
    for gamma in GAMMA_VALUES:
        logger.info(f"\n  Testing γ={gamma}")
        
        for seed in tqdm(range(42, 42 + N_SEEDS), desc=f"  γ={gamma}", leave=False):
            result = run_single_trial(
                data=data,
                encoder=encoder,
                pca=pca,
                warmup_priors=warmup_priors,
                models=models,
                context_dim=context_dim,
                seed=seed,
                gamma=gamma
            )
            all_results[gamma].append(result)
        
        # Print summary
        regrets = [r['final_regret'] for r in all_results[gamma]]
        min_weights = [r['final_min_weight'] for r in all_results[gamma]]
        logger.info(f"    Regret: {np.mean(regrets):.1f} ± {np.std(regrets):.1f}")
        logger.info(f"    Min Weight: {np.mean(min_weights):.4f} ± {np.std(min_weights):.4f}")
    
    return all_results, data

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def analyze_and_visualize(results: Dict[float, List[Dict]], data_size: int):
    """Analyze gamma ablation and create visualization."""
    
    logger.info("\n" + "="*80)
    logger.info("📊 GAMMA ABLATION ANALYSIS")
    logger.info("="*80)
    
    # Compute statistics
    stats = {}
    for gamma in GAMMA_VALUES:
        regrets = [r['final_regret'] for r in results[gamma]]
        min_weights = [r['final_min_weight'] for r in results[gamma]]
        variances = [r['weight_variance'] for r in results[gamma]]
        
        # Check for expert death (weight < 1e-6)
        expert_death_count = sum(1 for w in min_weights if w < 1e-6)
        
        stats[gamma] = {
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'mean_min_weight': np.mean(min_weights),
            'std_min_weight': np.std(min_weights),
            'expert_death_rate': expert_death_count / len(results[gamma]),
            'mean_variance': np.mean(variances)
        }
    
    # Sort by regret
    sorted_gammas = sorted(stats.items(), key=lambda x: x[1]['mean_regret'])
    
    logger.info(f"\n📊 Results (sorted by regret):")
    logger.info("-" * 95)
    logger.info(f"{'γ':<8} {'Regret':<15} {'Min Weight':<20} {'Death Rate':<15} {'Variance':<15}")
    logger.info("-" * 95)
    
    for gamma, stat in sorted_gammas:
        logger.info(f"{gamma:<8.2f} "
                   f"{stat['mean_regret']:>10.1f} ± {stat['std_regret']:<4.1f} "
                   f"{stat['mean_min_weight']:>10.4f} ± {stat['std_min_weight']:<6.4f} "
                   f"{stat['expert_death_rate']:>10.1%}      "
                   f"{stat['mean_variance']:>10.4f}")
    
    # Key findings
    logger.info("\n🎯 Key Findings:")
    
    best_gamma = sorted_gammas[0][0]
    logger.info(f"   Best γ: {best_gamma} (lowest regret)")
    
    # Check if minimal γ has expert death
    min_gamma = min(GAMMA_VALUES)
    if stats[min_gamma]['expert_death_rate'] > 0:
        logger.info(f"   ✅ Expert death observed at γ={min_gamma} ({stats[min_gamma]['expert_death_rate']:.0%} of trials)")
    else:
        logger.info(f"   ⚠️  No expert death at γ={min_gamma} (min weight: {stats[min_gamma]['mean_min_weight']:.4f})")
    
    # Compare current γ=0.05 to best
    current_gamma = 0.05
    if best_gamma == current_gamma:
        logger.info(f"   ✅ Current γ={current_gamma} IS OPTIMAL")
    else:
        diff = stats[current_gamma]['mean_regret'] - stats[best_gamma]['mean_regret']
        pct = (diff / stats[best_gamma]['mean_regret']) * 100
        logger.info(f"   ⚠️  Best γ={best_gamma} is {diff:.1f} regret better ({pct:.1f}% improvement)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Regret vs gamma
    gammas = list(stats.keys())
    means = [stats[g]['mean_regret'] for g in gammas]
    stds = [stats[g]['std_regret'] for g in gammas]
    
    colors = ['#e74c3c' if g == 0.001 else ('#2ecc71' if g == 0.05 else '#95a5a6') for g in gammas]
    
    ax1.bar(range(len(gammas)), means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(gammas)))
    ax1.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax1.set_xlabel('Gamma (γ)', fontsize=11)
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('(A) Regret vs Gamma', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=stats[0.05]['mean_regret'], color='#2ecc71', linestyle='--', 
                alpha=0.5, label='γ=0.05 (current)')
    ax1.legend(fontsize=9)
    
    # Panel B: Minimum weight vs gamma (expert death indicator)
    min_means = [stats[g]['mean_min_weight'] for g in gammas]
    min_stds = [stats[g]['std_min_weight'] for g in gammas]
    
    ax2.bar(range(len(gammas)), min_means, yerr=min_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax2.set_xlabel('Gamma (γ)', fontsize=11)
    ax2.set_ylabel('Minimum Expert Weight', fontsize=11)
    ax2.set_title('(B) Expert Death Prevention', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, label='Death threshold (10⁻⁶)')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    
    # Panel C: Weight evolution for different gammas
    max_steps = len(results[GAMMA_VALUES[0]][0]['weight_history'])
    steps = np.arange(1, max_steps + 1)
    
    for gamma in [0.001, 0.05, 0.20]:
        min_weight_trajs = []
        for result in results[gamma]:
            min_weights = result['min_weights']
            min_weight_trajs.append(min_weights)
        
        min_weight_trajs = np.array(min_weight_trajs)
        mean_min = np.mean(min_weight_trajs, axis=0)
        
        label = f'γ={gamma}'
        if gamma == 0.001:
            label += ' (minimal mixing)'
        elif gamma == 0.05:
            label += ' (current)'
        elif gamma == 0.20:
            label += ' (high mixing)'
        
        ax3.plot(steps, mean_min, label=label, linewidth=2)
    
    ax3.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label='Death threshold')
    ax3.set_xlabel('Request Number', fontsize=11)
    ax3.set_ylabel('Minimum Expert Weight', fontsize=11)
    ax3.set_title('(C) Weight Evolution Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_yscale('log')
    ax3.set_ylim(1e-8, 1.0)
    
    # Panel D: Weight variance vs gamma (stability)
    variances = [stats[g]['mean_variance'] for g in gammas]
    
    ax4.bar(range(len(gammas)), variances, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(gammas)))
    ax4.set_xticklabels([f'{g:.2f}' for g in gammas])
    ax4.set_xlabel('Gamma (γ)', fontsize=11)
    ax4.set_ylabel('Weight Variance', fontsize=11)
    ax4.set_title('(D) Weight Stability', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "figure_gamma_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n💾 Saved figure: {output_path}")
    
    # Save statistics
    stats_output = {
        'gamma_values': {str(g): {
            'mean_regret': float(v['mean_regret']),
            'std_regret': float(v['std_regret']),
            'mean_min_weight': float(v['mean_min_weight']),
            'std_min_weight': float(v['std_min_weight']),
            'expert_death_rate': float(v['expert_death_rate']),
            'mean_variance': float(v['mean_variance'])
        } for g, v in stats.items()},
        'best_gamma': float(best_gamma),
        'current_gamma': current_gamma,
        'current_is_optimal': (best_gamma == current_gamma),
        'n_seeds': N_SEEDS,
        'learning_rate': LEARNING_RATE,
        'alpha': ALPHA_CONSTANT,
        'data_size': data_size
    }
    
    stats_path = OUTPUT_DIR / "gamma_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    logger.info(f"💾 Saved statistics: {stats_path}")
    
    return stats_output

# ============================================================================
# MAIN
# ============================================================================
def main():
    results, data = run_gamma_ablation()
    stats = analyze_and_visualize(results, len(data))
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT 5 COMPLETE")
    logger.info("="*80)
    logger.info(f"\nBest γ: {stats['best_gamma']}")
    logger.info(f"Current γ=0.05 optimal: {stats['current_is_optimal']}")

if __name__ == '__main__':
    main()
