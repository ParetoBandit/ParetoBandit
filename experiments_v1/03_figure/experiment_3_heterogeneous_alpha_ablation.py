#!/usr/bin/env python3
"""
Experiment 3: Heterogeneous Alpha Strategy Ablation Study
==========================================================

Validates the claim: "Heterogeneous alpha strategy is the core innovation"

Tests 4 configurations to isolate the effect of heterogeneous exploration:
1. BASELINE: Both experts with decaying alpha (homogeneous exploitation)
2. REVERSED: E1 constant, E2 decaying (inverted heterogeneity)
3. CURRENT: E1 decaying, E2 constant (our heterogeneous design)
4. CONSTANT: Both experts with constant alpha (homogeneous exploration)

This ablation answers:
- Is heterogeneity better than homogeneity?
- Does the specific assignment (which expert gets which alpha) matter?
- How much regret improvement comes from heterogeneous strategy?

Author: BanditGPT Team
Date: 2026-02-12
Status: NEW (addresses conference reviewer Issue 3)
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
N_SEEDS = 5  # Ablation with multiple seeds
LEARNING_RATE = 1.0
GAMMA = 0.05
OUTPUT_DIR = Path(__file__).parent / "results" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ablation configurations
CONFIGS = {
    'homogeneous_decay': {
        'name': 'Homogeneous Decay',
        'expert1_alpha_start': 1.0,
        'expert1_alpha_end': 0.01,
        'expert2_alpha_start': 1.0,
        'expert2_alpha_end': 0.01,
        'description': 'Both experts decay to pure exploitation'
    },
    'reversed_heterogeneous': {
        'name': 'Reversed Heterogeneous',
        'expert1_alpha_start': 2.0,
        'expert1_alpha_end': 2.0,  # Constant
        'expert2_alpha_start': 1.0,
        'expert2_alpha_end': 0.01,  # Decaying
        'description': 'Warmup constant, Tabula decaying (opposite of current)'
    },
    'current_heterogeneous': {
        'name': 'Current Heterogeneous',
        'expert1_alpha_start': 1.0,
        'expert1_alpha_end': 0.01,  # Decaying
        'expert2_alpha_start': 2.0,
        'expert2_alpha_end': 2.0,  # Constant
        'description': 'Warmup decaying, Tabula constant (our design)'
    },
    'homogeneous_constant': {
        'name': 'Homogeneous Constant',
        'expert1_alpha_start': 2.0,
        'expert1_alpha_end': 2.0,
        'expert2_alpha_start': 2.0,
        'expert2_alpha_end': 2.0,
        'description': 'Both experts maintain constant exploration'
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================
def load_holdout_data():
    """Load holdout dataset."""
    import gzip
    
    logger.info("📊 Loading holdout data...")
    
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
    
    data_list = list(prompt_data.values())
    logger.info(f"   ✅ Loaded {len(data_list)} unique prompts")
    return data_list

# ============================================================================
# SINGLE TRIAL
# ============================================================================
def run_single_trial(data: List[Dict], encoder, pca, warmup_priors, 
                     models: List[str], context_dim: int, seed: int, config_key: str) -> Dict:
    """Run a single trial with specific alpha configuration."""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    config = CONFIGS[config_key]
    
    # Initialize experts with config-specific alphas
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors,
        alpha_start=config['expert1_alpha_start'],
        alpha_end=config['expert1_alpha_end'],
        cost_penalty=0.0,
        model_costs={m: {"normalized_cost": 0.0} for m in models}
    )
    
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha_start=config['expert2_alpha_start'],
        alpha_end=config['expert2_alpha_end'],
        cost_penalty=0.0,
        model_costs={m: {"normalized_cost": 0.0} for m in models}
    )
    
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA
    )
    
    # Track metrics
    cumulative_regret = 0.0
    
    # Shuffle data
    indices = rng.permutation(len(data))
    
    for idx in indices:
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
    
    return {
        'config_key': config_key,
        'final_regret': cumulative_regret,
        'seed': seed
    }

# ============================================================================
# MULTI-CONFIG EXPERIMENT
# ============================================================================
def run_ablation_study():
    """Run ablation study across all configurations."""
    logger.info("="*80)
    logger.info("EXPERIMENT 3: HETEROGENEOUS ALPHA ABLATION STUDY")
    logger.info("="*80)
    
    # Load resources
    logger.info("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=GAMMA)
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    logger.info(f"   ✅ Models: {len(models)}")
    logger.info(f"   ✅ Context Dim: {context_dim}")
    
    # Load data
    data = load_holdout_data()
    
    # Run all configurations
    logger.info(f"\n🔬 Running ablation study: {len(CONFIGS)} configs × {N_SEEDS} seeds...")
    
    all_results = {key: [] for key in CONFIGS.keys()}
    
    for config_key in CONFIGS.keys():
        config = CONFIGS[config_key]
        logger.info(f"\n  Testing: {config['name']}")
        logger.info(f"  {config['description']}")
        
        for seed in range(42, 42 + N_SEEDS):
            result = run_single_trial(
                data=data,
                encoder=encoder,
                pca=pca,
                warmup_priors=warmup_priors_scaled,
                models=models,
                context_dim=context_dim,
                seed=seed,
                config_key=config_key
            )
            all_results[config_key].append(result)
            logger.info(f"    Seed {seed}: Regret = {result['final_regret']:.1f}")
    
    return all_results, data

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def analyze_and_visualize(results: Dict[str, List[Dict]], data_size: int):
    """Analyze ablation results and create visualization."""
    
    logger.info("\n" + "="*80)
    logger.info("📊 ABLATION ANALYSIS")
    logger.info("="*80)
    
    # Compute statistics
    stats = {}
    for config_key, trials in results.items():
        regrets = [t['final_regret'] for t in trials]
        stats[config_key] = {
            'name': CONFIGS[config_key]['name'],
            'description': CONFIGS[config_key]['description'],
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'min_regret': np.min(regrets),
            'max_regret': np.max(regrets),
            'regrets': regrets
        }
    
    # Sort by mean regret
    sorted_configs = sorted(stats.items(), key=lambda x: x[1]['mean_regret'])
    
    logger.info(f"\n📊 Results (sorted by regret):")
    logger.info("-" * 80)
    logger.info(f"{'Configuration':<30} {'Mean Regret':<15} {'Std':<10} {'Range':<15}")
    logger.info("-" * 80)
    
    for config_key, stat in sorted_configs:
        logger.info(f"{stat['name']:<30} "
                   f"{stat['mean_regret']:>10.1f}     "
                   f"{stat['std_regret']:>7.1f}   "
                   f"[{stat['min_regret']:.0f}, {stat['max_regret']:.0f}]")
    
    # Find best and compare to current
    best_config = sorted_configs[0]
    current_config = stats['current_heterogeneous']
    
    logger.info("\n🎯 Key Findings:")
    logger.info(f"   Best Configuration: {best_config[1]['name']}")
    logger.info(f"   Best Mean Regret: {best_config[1]['mean_regret']:.1f} ± {best_config[1]['std_regret']:.1f}")
    
    if best_config[0] == 'current_heterogeneous':
        logger.info(f"   ✅ Current design IS OPTIMAL")
    else:
        diff = current_config['mean_regret'] - best_config[1]['mean_regret']
        pct = (diff / best_config[1]['mean_regret']) * 100
        logger.info(f"   ⚠️  Current design is {diff:.1f} regret worse ({pct:.1f}% penalty)")
        logger.info(f"   Current Mean Regret: {current_config['mean_regret']:.1f} ± {current_config['std_regret']:.1f}")
    
    # Test heterogeneous vs homogeneous
    hetero_configs = ['current_heterogeneous', 'reversed_heterogeneous']
    homo_configs = ['homogeneous_decay', 'homogeneous_constant']
    
    hetero_regrets = [stats[k]['mean_regret'] for k in hetero_configs]
    homo_regrets = [stats[k]['mean_regret'] for k in homo_configs]
    
    logger.info(f"\n💡 Heterogeneity Analysis:")
    logger.info(f"   Heterogeneous avg: {np.mean(hetero_regrets):.1f}")
    logger.info(f"   Homogeneous avg: {np.mean(homo_regrets):.1f}")
    
    if np.mean(hetero_regrets) < np.mean(homo_regrets):
        improvement = ((np.mean(homo_regrets) - np.mean(hetero_regrets)) / np.mean(homo_regrets)) * 100
        logger.info(f"   ✅ Heterogeneity provides {improvement:.1f}% improvement")
    else:
        logger.info(f"   ❌ Heterogeneity does NOT help")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Bar chart with error bars
    config_keys = [k for k, _ in sorted_configs]
    means = [stats[k]['mean_regret'] for k in config_keys]
    stds = [stats[k]['std_regret'] for k in config_keys]
    names = [stats[k]['name'] for k in config_keys]
    
    colors = ['#e67e22' if k == 'current_heterogeneous' else '#95a5a6' for k in config_keys]
    
    ax1.bar(range(len(config_keys)), means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(config_keys)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('(A) Ablation Study: Alpha Strategy Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=current_config['mean_regret'], color='#e67e22', linestyle='--', 
                alpha=0.5, label='Current Design')
    ax1.legend(fontsize=10)
    
    # Panel B: Individual trial scatter
    for i, (config_key, stat) in enumerate(sorted_configs):
        regrets = stat['regrets']
        x = np.random.normal(i, 0.1, size=len(regrets))
        ax2.scatter(x, regrets, alpha=0.6, s=50, label=stat['name'])
    
    ax2.set_xticks(range(len(config_keys)))
    ax2.set_xticklabels([f"Config {i+1}" for i in range(len(config_keys))], fontsize=10)
    ax2.set_ylabel('Cumulative Regret (per seed)', fontsize=12)
    ax2.set_title('(B) Per-Seed Results', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "figure_alpha_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n💾 Saved figure: {output_path}")
    
    # Save statistics
    stats_output = {
        'configurations': {k: {
            'name': v['name'],
            'description': v['description'],
            'mean_regret': float(v['mean_regret']),
            'std_regret': float(v['std_regret']),
            'min_regret': float(v['min_regret']),
            'max_regret': float(v['max_regret'])
        } for k, v in stats.items()},
        'best_config': best_config[0],
        'current_is_optimal': (best_config[0] == 'current_heterogeneous'),
        'n_seeds': N_SEEDS,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'data_size': data_size
    }
    
    stats_path = OUTPUT_DIR / "ablation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    logger.info(f"💾 Saved statistics: {stats_path}")
    
    return stats_output

# ============================================================================
# MAIN
# ============================================================================
def main():
    results, data = run_ablation_study()
    stats = analyze_and_visualize(results, len(data))
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT 3 COMPLETE")
    logger.info("="*80)
    logger.info(f"\nBest Configuration: {stats['configurations'][stats['best_config']]['name']}")
    logger.info(f"Current Design Optimal: {stats['current_is_optimal']}")
    
    if stats['current_is_optimal']:
        logger.info("\n✅ CLAIM VALIDATED: Heterogeneous alpha strategy is effective")
    else:
        logger.info("\n⚠️  CLAIM QUESTIONED: Other configurations perform better")

if __name__ == '__main__':
    main()
