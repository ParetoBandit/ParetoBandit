#!/usr/bin/env python3
"""
Experiment 2A: Expert Weight Evolution Over Time
=================================================

Validates the claim: "Trust shifts from Warmup (0.5→0.2) to Tabula Rasa under domain mismatch"

This experiment tracks expert weights at every timestep to show:
1. Initial weight allocation (should be ~0.5, 0.5)
2. Weight trajectory over time
3. Final weight allocation
4. Convergence dynamics

Methodology:
- Uses LMSYS Holdout data (N=750) with severe domain mismatch
- Runs Corralling with η=1.0 (optimal learning rate from Table 2)
- Tracks expert weights at every update
- Generates temporal visualization with confidence intervals (multi-seed)
- Validates or corrects the "0.5→0.2" claim with actual data

Author: BanditGPT Team
Date: 2026-02-12
Status: NEW (addresses conference reviewer Issue 2A)
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
N_SEEDS = 10  # Multiple seeds for confidence intervals
LEARNING_RATE = 1.0  # Optimal from Table 2
GAMMA = 0.05  # Mixing parameter
OUTPUT_DIR = Path(__file__).parent / "results" / "weight_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_holdout_data():
    """Load holdout dataset with domain mismatch."""
    import gzip
    
    logger.info("📊 Loading holdout data...")
    
    entries = []
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Group by prompt
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
                     models: List[str], context_dim: int, seed: int) -> Dict:
    """
    Run a single trial and track expert weights at every step.
    
    Returns:
        Dict with 'weights_history' (list of [w_warmup, w_tabula]) and 'regret'
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Initialize experts
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors,
        alpha_start=1.0,
        alpha_end=0.01,
        cost_penalty=0.0,
        model_costs={m: {"normalized_cost": 0.0} for m in models}
    )
    
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha_start=2.0,
        alpha_end=2.0,  # Constant exploration
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
    weights_history = []
    cumulative_regret = 0.0
    
    # Shuffle data
    indices = rng.permutation(len(data))
    
    for i, idx in enumerate(tqdm(indices, desc=f"Seed {seed}", leave=False)):
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
        
        # Record weights AFTER update
        weights_history.append({
            'step': i + 1,
            'warmup': float(router.weights[0]),
            'tabula_rasa': float(router.weights[1]),
            'regret': cumulative_regret
        })
    
    return {
        'weights_history': weights_history,
        'final_regret': cumulative_regret,
        'final_warmup_weight': float(router.weights[0]),
        'final_tabula_weight': float(router.weights[1])
    }

# ============================================================================
# MULTI-SEED EXPERIMENT
# ============================================================================
def run_multi_seed_experiment():
    """Run experiment with multiple seeds for statistical robustness."""
    logger.info("="*80)
    logger.info("EXPERIMENT 2A: EXPERT WEIGHT EVOLUTION")
    logger.info("="*80)
    
    # Load resources
    logger.info("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # Apply gamma scaling
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=GAMMA)
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    logger.info(f"   ✅ Models: {len(models)}")
    logger.info(f"   ✅ Context Dim: {context_dim}")
    
    # Load data
    data = load_holdout_data()
    
    # Run trials
    logger.info(f"\n🔬 Running {N_SEEDS} trials...")
    all_results = []
    
    for seed in range(42, 42 + N_SEEDS):
        result = run_single_trial(
            data=data,
            encoder=encoder,
            pca=pca,
            warmup_priors=warmup_priors_scaled,
            models=models,
            context_dim=context_dim,
            seed=seed
        )
        all_results.append(result)
        
        logger.info(f"   Seed {seed}: Final weights = [{result['final_warmup_weight']:.3f}, "
                   f"{result['final_tabula_weight']:.3f}], Regret = {result['final_regret']:.1f}")
    
    return all_results, data

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================
def analyze_and_visualize(results: List[Dict], data_size: int):
    """Analyze results and create publication-quality visualization."""
    
    logger.info("\n" + "="*80)
    logger.info("📊 ANALYSIS")
    logger.info("="*80)
    
    # Extract trajectories
    max_steps = max(len(r['weights_history']) for r in results)
    
    warmup_trajectories = []
    tabula_trajectories = []
    
    for result in results:
        warmup_traj = np.array([w['warmup'] for w in result['weights_history']])
        tabula_traj = np.array([w['tabula_rasa'] for w in result['weights_history']])
        
        # Pad if necessary (some trials might be shorter)
        if len(warmup_traj) < max_steps:
            warmup_traj = np.pad(warmup_traj, (0, max_steps - len(warmup_traj)), 
                                 constant_values=warmup_traj[-1])
            tabula_traj = np.pad(tabula_traj, (0, max_steps - len(tabula_traj)),
                                 constant_values=tabula_traj[-1])
        
        warmup_trajectories.append(warmup_traj)
        tabula_trajectories.append(tabula_traj)
    
    warmup_trajectories = np.array(warmup_trajectories)
    tabula_trajectories = np.array(tabula_trajectories)
    
    # Compute statistics
    warmup_mean = np.mean(warmup_trajectories, axis=0)
    warmup_std = np.std(warmup_trajectories, axis=0)
    tabula_mean = np.mean(tabula_trajectories, axis=0)
    tabula_std = np.std(tabula_trajectories, axis=0)
    
    steps = np.arange(1, max_steps + 1)
    
    # Summary statistics
    initial_warmup = warmup_mean[0]
    final_warmup = warmup_mean[-1]
    initial_tabula = tabula_mean[0]
    final_tabula = tabula_mean[-1]
    
    logger.info(f"\n📈 Weight Evolution:")
    logger.info(f"   Initial: Warmup={initial_warmup:.3f} ± {warmup_std[0]:.3f}, "
                f"Tabula={initial_tabula:.3f} ± {tabula_std[0]:.3f}")
    logger.info(f"   Final:   Warmup={final_warmup:.3f} ± {warmup_std[-1]:.3f}, "
                f"Tabula={final_tabula:.3f} ± {tabula_std[-1]:.3f}")
    logger.info(f"   Shift:   Warmup {initial_warmup:.3f}→{final_warmup:.3f} "
                f"({(final_warmup-initial_warmup)/initial_warmup*100:+.1f}%)")
    
    # Determine claim validation
    claim_range = (0.15, 0.25)  # Original claim: ~0.2
    if claim_range[0] <= final_warmup <= claim_range[1]:
        validation = "✅ VALIDATED"
    else:
        validation = f"❌ CORRECTED (actual: {final_warmup:.3f}, not ~0.2)"
    
    logger.info(f"\n🎯 Claim Validation: 'Warmup weight shifts to ~0.2'")
    logger.info(f"   Result: {validation}")
    
    # Create visualization
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Weight trajectories
    ax1.plot(steps, warmup_mean, label='Warmup Expert', color='#2ecc71', linewidth=2)
    ax1.fill_between(steps, warmup_mean - warmup_std, warmup_mean + warmup_std,
                     color='#2ecc71', alpha=0.2)
    
    ax1.plot(steps, tabula_mean, label='Tabula Rasa Expert', color='#e67e22', linewidth=2)
    ax1.fill_between(steps, tabula_mean - tabula_std, tabula_mean + tabula_std,
                     color='#e67e22', alpha=0.2)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal (0.5)')
    ax1.set_xlabel('Request Number', fontsize=12)
    ax1.set_ylabel('Expert Weight', fontsize=12)
    ax1.set_title('(A) Expert Weight Evolution Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max_steps)
    ax1.set_ylim(0, 1)
    
    # Panel B: Weight ratio over time
    ratio_mean = tabula_mean / (warmup_mean + 1e-8)
    ratio_std = ratio_mean * np.sqrt((tabula_std/tabula_mean)**2 + (warmup_std/warmup_mean)**2)
    
    ax2.plot(steps, ratio_mean, color='#3498db', linewidth=2)
    ax2.fill_between(steps, ratio_mean - ratio_std, ratio_mean + ratio_std,
                     color='#3498db', alpha=0.2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal Trust (ratio=1)')
    ax2.set_xlabel('Request Number', fontsize=12)
    ax2.set_ylabel('Tabula/Warmup Weight Ratio', fontsize=12)
    ax2.set_title('(B) Trust Ratio: Tabula Rasa / Warmup', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, max_steps)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "figure_weight_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n💾 Saved figure: {output_path}")
    
    # Save statistics
    stats = {
        'n_seeds': N_SEEDS,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'data_size': data_size,
        'initial_weights': {
            'warmup': float(initial_warmup),
            'tabula_rasa': float(initial_tabula)
        },
        'final_weights': {
            'warmup': float(final_warmup),
            'warmup_std': float(warmup_std[-1]),
            'tabula_rasa': float(final_tabula),
            'tabula_std': float(tabula_std[-1])
        },
        'claim_validation': validation,
        'average_final_regret': float(np.mean([r['final_regret'] for r in results])),
        'regret_std': float(np.std([r['final_regret'] for r in results]))
    }
    
    stats_path = OUTPUT_DIR / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"💾 Saved statistics: {stats_path}")
    
    return stats

# ============================================================================
# MAIN
# ============================================================================
def main():
    results, data = run_multi_seed_experiment()
    stats = analyze_and_visualize(results, len(data))
    
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT 2A COMPLETE")
    logger.info("="*80)
    logger.info(f"\nFinal Result: {stats['claim_validation']}")
    logger.info(f"Data: N={stats['data_size']} prompts, {N_SEEDS} seeds")
    logger.info(f"Final weights: Warmup={stats['final_weights']['warmup']:.3f} ± {stats['final_weights']['warmup_std']:.3f}, "
                f"Tabula={stats['final_weights']['tabula_rasa']:.3f} ± {stats['final_weights']['tabula_std']:.3f}")

if __name__ == '__main__':
    main()
