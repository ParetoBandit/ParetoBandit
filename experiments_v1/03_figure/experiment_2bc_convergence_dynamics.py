#!/usr/bin/env python3
"""
Experiment 2B+2C: Convergence Dynamics and Learning Speed
=========================================================

Validates two claims:
1. "2-3x faster regret reduction in first 1000 requests" (Claim 2C)
2. "~100-200 requests to detect shift and adapt weights" (Claim 2B)

This experiment tracks cumulative regret over time for:
- Corralling (with constant α=2.0)
- Warmup Only
- Tabula Rasa Only

Measures:
- Convergence rate (regret slope in early phase)
- Adaptation speed (how fast Corralling pivots from bad warmup)
- Break-even points (when each strategy catches up)

Author: BanditGPT Team
Date: 2026-02-12
Status: NEW (addresses conference reviewer Issues 2B & 2C)
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
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
N_SEEDS = 10  # Multiple seeds for robustness
LEARNING_RATE = 1.0
GAMMA = 0.05
ALPHA_CONSTANT = 2.0  # Using optimal from ablation study
OUTPUT_DIR = Path(__file__).parent / "results" / "convergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# SINGLE STRATEGY RUN
# ============================================================================
def run_strategy(data: List[Dict], encoder, pca, warmup_priors, 
                 models: List[str], context_dim: int, seed: int, 
                 strategy: str) -> Dict:
    """
    Run a single strategy and track regret at every step.
    
    Args:
        strategy: One of ['corralling', 'warmup_only', 'tabula_rasa_only']
    
    Returns:
        Dict with 'regret_history', 'weights_history' (for corralling), 'final_regret'
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Initialize based on strategy
    if strategy == 'warmup_only':
        router = CostAwareLinUCBRouter(
            models=models,
            warmup_priors=warmup_priors,
            alpha_start=ALPHA_CONSTANT,
            alpha_end=ALPHA_CONSTANT,
            cost_penalty=0.0,
            model_costs={m: {"normalized_cost": 0.0} for m in models}
        )
        
    elif strategy == 'tabula_rasa_only':
        router = CostAwareTabulaRasaRouter(
            models=models,
            context_dim=context_dim,
            alpha_start=ALPHA_CONSTANT,
            alpha_end=ALPHA_CONSTANT,
            cost_penalty=0.0,
            model_costs={m: {"normalized_cost": 0.0} for m in models}
        )
        
    elif strategy == 'corralling':
        warmup_expert = CostAwareLinUCBRouter(
            models=models,
            warmup_priors=warmup_priors,
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
            gamma=GAMMA
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Track metrics
    regret_history = []
    weights_history = [] if strategy == 'corralling' else None
    cumulative_regret = 0.0
    
    # Shuffle data
    indices = rng.permutation(len(data))
    
    for step, idx in enumerate(indices):
        sample = data[idx]
        prompt = sample['prompt']
        context = embed_prompt(prompt, encoder, pca)
        
        # Select model
        if strategy == 'corralling':
            selected_model = router.select_model(context)
        else:
            selected_model = router.select_model(context, total_steps=len(data))
        
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
        
        # Record state
        regret_history.append(cumulative_regret)
        
        if strategy == 'corralling':
            weights_history.append({
                'step': step + 1,
                'warmup': float(router.weights[0]),
                'tabula_rasa': float(router.weights[1])
            })
    
    return {
        'strategy': strategy,
        'seed': seed,
        'regret_history': regret_history,
        'weights_history': weights_history,
        'final_regret': cumulative_regret
    }

# ============================================================================
# MULTI-STRATEGY EXPERIMENT
# ============================================================================
def run_convergence_experiment():
    """Run all three strategies with multiple seeds."""
    logger.info("="*80)
    logger.info("EXPERIMENT 2B+2C: CONVERGENCE DYNAMICS")
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
    logger.info(f"   ✅ Alpha: {ALPHA_CONSTANT} (constant, from ablation)")
    
    # Load data
    data = load_holdout_data()
    
    # Run all strategies
    logger.info(f"\n🔬 Running 3 strategies × {N_SEEDS} seeds...")
    
    strategies = ['corralling', 'warmup_only', 'tabula_rasa_only']
    all_results = {s: [] for s in strategies}
    
    for strategy in strategies:
        logger.info(f"\n  Testing: {strategy}")
        
        for seed in tqdm(range(42, 42 + N_SEEDS), desc=f"  {strategy}", leave=False):
            result = run_strategy(
                data=data,
                encoder=encoder,
                pca=pca,
                warmup_priors=warmup_priors_scaled,
                models=models,
                context_dim=context_dim,
                seed=seed,
                strategy=strategy
            )
            all_results[strategy].append(result)
        
        # Print summary
        final_regrets = [r['final_regret'] for r in all_results[strategy]]
        logger.info(f"    Final Regret: {np.mean(final_regrets):.1f} ± {np.std(final_regrets):.1f}")
    
    return all_results, data

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def analyze_and_visualize(results: Dict[str, List[Dict]], data_size: int):
    """Analyze convergence dynamics and create visualization."""
    
    logger.info("\n" + "="*80)
    logger.info("📊 CONVERGENCE ANALYSIS")
    logger.info("="*80)
    
    # Extract regret trajectories
    max_steps = len(results['corralling'][0]['regret_history'])
    
    trajectories = {}
    for strategy in results.keys():
        strategy_trajs = np.array([r['regret_history'] for r in results[strategy]])
        trajectories[strategy] = {
            'mean': np.mean(strategy_trajs, axis=0),
            'std': np.std(strategy_trajs, axis=0),
            'all': strategy_trajs
        }
    
    steps = np.arange(1, max_steps + 1)
    
    # Compute convergence metrics
    early_window = slice(0, min(200, max_steps))
    mid_window = slice(200, min(400, max_steps))
    
    logger.info("\n📈 Convergence Metrics:")
    logger.info("-" * 80)
    logger.info(f"{'Strategy':<20} {'Early (0-200)':<20} {'Mid (200-400)':<20} {'Final':<15}")
    logger.info("-" * 80)
    
    for strategy in ['corralling', 'warmup_only', 'tabula_rasa_only']:
        early_regret = trajectories[strategy]['mean'][early_window][-1] if len(trajectories[strategy]['mean']) > 200 else trajectories[strategy]['mean'][-1]
        mid_regret = trajectories[strategy]['mean'][mid_window][-1] if len(trajectories[strategy]['mean']) > 400 else trajectories[strategy]['mean'][-1]
        final_regret = trajectories[strategy]['mean'][-1]
        
        logger.info(f"{strategy:<20} {early_regret:>15.1f}     {mid_regret:>15.1f}     {final_regret:>10.1f}")
    
    # Adaptation speed analysis (for corralling)
    logger.info("\n⚡ Adaptation Speed Analysis (Corralling):")
    
    # Extract weight trajectories
    all_weight_trajs = []
    for result in results['corralling']:
        warmup_weights = [w['warmup'] for w in result['weights_history']]
        all_weight_trajs.append(warmup_weights)
    
    weight_trajs = np.array(all_weight_trajs)
    mean_warmup_weight = np.mean(weight_trajs, axis=0)
    
    # Find when warmup weight drops below 0.4 (indicating shift to tabula rasa)
    adaptation_points = []
    for traj in weight_trajs:
        below_threshold = np.where(traj < 0.4)[0]
        if len(below_threshold) > 0:
            adaptation_points.append(below_threshold[0])
        else:
            adaptation_points.append(len(traj))  # Never adapted
    
    mean_adaptation = np.mean(adaptation_points)
    std_adaptation = np.std(adaptation_points)
    
    logger.info(f"   Mean adaptation point: {mean_adaptation:.0f} ± {std_adaptation:.0f} requests")
    logger.info(f"   (Defined as: warmup weight < 0.4)")
    
    # Convergence rate comparison
    logger.info("\n🏃 Convergence Rate (Early Phase 0-200):")
    
    early_corralling = trajectories['corralling']['mean'][early_window][-1] if len(trajectories['corralling']['mean']) > 200 else trajectories['corralling']['mean'][-1]
    early_warmup = trajectories['warmup_only']['mean'][early_window][-1] if len(trajectories['warmup_only']['mean']) > 200 else trajectories['warmup_only']['mean'][-1]
    early_tabula = trajectories['tabula_rasa_only']['mean'][early_window][-1] if len(trajectories['tabula_rasa_only']['mean']) > 200 else trajectories['tabula_rasa_only']['mean'][-1]
    
    speedup_vs_warmup = early_warmup / early_corralling if early_corralling > 0 else 1.0
    speedup_vs_tabula = early_tabula / early_corralling if early_corralling > 0 else 1.0
    
    logger.info(f"   Corralling vs Warmup: {speedup_vs_warmup:.2f}x (higher is better for Corralling)")
    logger.info(f"   Corralling vs Tabula Rasa: {speedup_vs_tabula:.2f}x")
    
    # Claim validation
    logger.info("\n🎯 Claim Validation:")
    
    claim_2b_range = (100, 200)
    if claim_2b_range[0] <= mean_adaptation <= claim_2b_range[1]:
        claim_2b = f"✅ VALIDATED: {mean_adaptation:.0f} is within claimed 100-200 requests"
    else:
        claim_2b = f"❌ CORRECTED: Actual {mean_adaptation:.0f} ± {std_adaptation:.0f}, not 100-200"
    logger.info(f"   Claim 2B (Adaptation): {claim_2b}")
    
    # For claim 2C, we need to check if Corralling is 2-3x faster
    claim_2c_range = (2.0, 3.0)
    if claim_2c_range[0] <= speedup_vs_tabula <= claim_2c_range[1]:
        claim_2c = f"✅ VALIDATED: {speedup_vs_tabula:.2f}x is within claimed 2-3x"
    else:
        claim_2c = f"❌ CORRECTED: Actual {speedup_vs_tabula:.2f}x, not 2-3x"
    logger.info(f"   Claim 2C (Speedup): {claim_2c}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel A: Regret trajectories
    ax1 = fig.add_subplot(gs[0])
    
    colors = {'corralling': '#3498db', 'warmup_only': '#2ecc71', 'tabula_rasa_only': '#e67e22'}
    labels = {'corralling': 'Corralling', 'warmup_only': 'Warmup Only', 'tabula_rasa_only': 'Tabula Rasa Only'}
    
    for strategy in ['corralling', 'warmup_only', 'tabula_rasa_only']:
        mean = trajectories[strategy]['mean']
        std = trajectories[strategy]['std']
        
        ax1.plot(steps, mean, label=labels[strategy], color=colors[strategy], linewidth=2)
        ax1.fill_between(steps, mean - std, mean + std, color=colors[strategy], alpha=0.2)
    
    ax1.set_xlabel('Request Number', fontsize=11)
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('(A) Convergence Dynamics', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max_steps)
    
    # Panel B: Early phase zoom (0-200)
    ax2 = fig.add_subplot(gs[1])
    
    early_steps = steps[early_window]
    for strategy in ['corralling', 'warmup_only', 'tabula_rasa_only']:
        mean = trajectories[strategy]['mean'][early_window]
        std = trajectories[strategy]['std'][early_window]
        
        ax2.plot(early_steps, mean, label=labels[strategy], color=colors[strategy], linewidth=2)
        ax2.fill_between(early_steps, mean - std, mean + std, color=colors[strategy], alpha=0.2)
    
    ax2.set_xlabel('Request Number', fontsize=11)
    ax2.set_ylabel('Cumulative Regret', fontsize=11)
    ax2.set_title('(B) Early Phase (0-200)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Panel C: Weight evolution (Corralling only)
    ax3 = fig.add_subplot(gs[2])
    
    mean_warmup = np.mean(weight_trajs, axis=0)
    std_warmup = np.std(weight_trajs, axis=0)
    mean_tabula = 1.0 - mean_warmup
    
    ax3.plot(steps, mean_warmup, label='Warmup Expert', color='#2ecc71', linewidth=2)
    ax3.fill_between(steps, mean_warmup - std_warmup, mean_warmup + std_warmup,
                     color='#2ecc71', alpha=0.2)
    
    ax3.plot(steps, mean_tabula, label='Tabula Rasa Expert', color='#e67e22', linewidth=2)
    
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal (0.5)')
    ax3.axhline(y=0.4, color='red', linestyle=':', alpha=0.5, label='Adaptation Threshold')
    
    # Mark adaptation point
    if mean_adaptation < len(steps):
        ax3.axvline(x=mean_adaptation, color='red', linestyle='--', alpha=0.7)
        ax3.text(mean_adaptation + 10, 0.9, f'Adapt ≈{mean_adaptation:.0f}', 
                fontsize=9, color='red')
    
    ax3.set_xlabel('Request Number', fontsize=11)
    ax3.set_ylabel('Expert Weight', fontsize=11)
    ax3.set_title('(C) Expert Weight Evolution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, max_steps)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "figure_convergence_dynamics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n💾 Saved figure: {output_path}")
    
    # Save statistics
    stats = {
        'n_seeds': N_SEEDS,
        'data_size': data_size,
        'alpha': ALPHA_CONSTANT,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'final_regrets': {
            strategy: {
                'mean': float(trajectories[strategy]['mean'][-1]),
                'std': float(trajectories[strategy]['std'][-1])
            } for strategy in results.keys()
        },
        'early_phase_regrets': {
            'corralling': float(early_corralling),
            'warmup_only': float(early_warmup),
            'tabula_rasa_only': float(early_tabula)
        },
        'convergence_speedup': {
            'vs_warmup': float(speedup_vs_warmup),
            'vs_tabula_rasa': float(speedup_vs_tabula)
        },
        'adaptation_analysis': {
            'mean_adaptation_point': float(mean_adaptation),
            'std_adaptation_point': float(std_adaptation),
            'threshold_used': 0.4
        },
        'claim_validation': {
            'claim_2b_adaptation_speed': claim_2b,
            'claim_2c_convergence_speedup': claim_2c
        }
    }
    
    stats_path = OUTPUT_DIR / "convergence_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"💾 Saved statistics: {stats_path}")
    
    return stats

# ============================================================================
# MAIN
# ============================================================================
def main():
    results, data = run_convergence_experiment()
    stats = analyze_and_visualize(results, len(data))
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT 2B+2C COMPLETE")
    logger.info("="*80)
    logger.info(f"\nClaim Validations:")
    logger.info(f"  2B: {stats['claim_validation']['claim_2b_adaptation_speed']}")
    logger.info(f"  2C: {stats['claim_validation']['claim_2c_convergence_speedup']}")

if __name__ == '__main__':
    main()
