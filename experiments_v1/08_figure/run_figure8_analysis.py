"""
Figure 8: Regime-Stratified n_eff Sensitivity Analysis
=====================================================
Unified script that:
1. Runs experiments ONCE (or loads cached results)
2. Generates publication-quality figure
3. Generates formatted table (console + LaTeX)

Usage:
    python experiments_v1/08_figure/run_figure8_analysis.py
    python experiments_v1/08_figure/run_figure8_analysis.py --force-rerun
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
import logging
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS
)
from utils.aligned_evaluator import AlignedEvaluator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
NEIGHBOR_MODEL = "openai/gpt-4-turbo"
TOTAL_STEPS = 1000
RELEASE_STEP = 300
SEEDS = list(range(42, 72))  # 30 seeds for statistical power
N_EFF_VALUES = [1.0, 20.0]  # Test extremes

# ============================================================================
# DATA COLLECTION
# ============================================================================

def create_model_registry(models):
    """Create model registry with cost information."""
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5, "output_cost_per_m": 1.5
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0, "output_cost_per_m": 30.0
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0, "output_cost_per_m": 45.0
        }
    }
    return {k: v for k, v in all_models.items() if k in models}

def load_data():
    """Load and filter evaluation data."""
    required_models = WARMUP_MODELS + [NEW_MODEL]
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=required_models
    )
    filtered = [item for item in evaluator if all(m in item.rewards for m in required_models)]
    logger.info(f"✅ Loaded {len(filtered)} samples with complete coverage")
    return AlignedEvaluator(filtered)

def run_experiment(n_effective: float, seed: int) -> Dict:
    """
    Run single experiment with weight tracking.
    
    Returns:
        Dict with keys: rewards, warmup_weights, tabula_weights, regime, post_release_mean
    """
    np.random.seed(seed)
    
    evaluator = load_data()
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    router = BanditRouter.create(
        model_registry=create_model_registry(WARMUP_MODELS),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    rewards = []
    warmup_weights = []
    tabula_weights = []
    
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS:
            break
        
        # Model release with semantic transfer
        if t == RELEASE_STEP:
            A_neighbor = router.bandit.A[NEIGHBOR_MODEL].copy()
            b_neighbor = router.bandit.b[NEIGHBOR_MODEL].copy()
            theta_neighbor = router.bandit.A_inv[NEIGHBOR_MODEL] @ b_neighbor
            
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = n_effective * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = n_effective * theta_neighbor
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
            
            # Update Corralling experts
            if router.corralling_router:
                router.corralling_router.add_model(NEW_MODEL)
                for expert in router.corralling_router.experts:
                    if hasattr(expert, 'add_model'):
                        expert_type = type(expert).__name__
                        if 'TabulaRasa' in expert_type:
                            expert.add_model(NEW_MODEL, 0.5)
                        else:
                            transfer_A = n_effective * np.eye(router.bandit.dim)
                            transfer_b = n_effective * theta_neighbor
                            expert.add_model(NEW_MODEL, transfer_A, transfer_b, 0.5)
        
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        rewards.append(reward)
        
        if router.corralling_router:
            warmup_weights.append(router.corralling_router.weights[0])
            tabula_weights.append(router.corralling_router.weights[1])
    
    # Classify regime based on post-release average weight
    post_release_warmup = np.mean(warmup_weights[RELEASE_STEP:])
    regime = 'warmup' if post_release_warmup > 0.5 else 'tabula_rasa'
    
    return {
        'rewards': rewards,
        'warmup_weights': warmup_weights,
        'tabula_weights': tabula_weights,
        'regime': regime,
        'post_release_mean': np.mean(rewards[RELEASE_STEP:])
    }

def run_all_experiments(force_rerun: bool = False) -> Dict:
    """
    Run all experiments or load cached results.
    
    Returns:
        Dict mapping (n_eff, seed) -> experiment results
    """
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "figure8_unified_results.pkl"
    
    if not force_rerun and cache_file.exists():
        logger.info(f"📦 Loading cached results from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info("\n" + "="*60)
    logger.info("Running experiments (this takes ~5 minutes)...")
    logger.info("="*60 + "\n")
    
    results = {}
    total = len(N_EFF_VALUES) * len(SEEDS)
    count = 0
    
    for n_eff in N_EFF_VALUES:
        for seed in SEEDS:
            count += 1
            logger.info(f"[{count}/{total}] Running n_eff={n_eff}, seed={seed}...")
            results[(n_eff, seed)] = run_experiment(n_eff, seed)
    
    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"\n✅ Saved results to {cache_file}\n")
    
    return results

# ============================================================================
# ANALYSIS & CLASSIFICATION
# ============================================================================

def classify_regimes(results: Dict) -> Dict:
    """Classify seeds into warmup-dominant vs tabula rasa-dominant regimes."""
    
    # Use n_eff=1.0 results to classify (arbitrary choice, both should agree)
    regime_by_seed = {}
    for seed in SEEDS:
        result = results[(1.0, seed)]
        regime_by_seed[seed] = result['regime']
    
    warmup_seeds = [s for s, r in regime_by_seed.items() if r == 'warmup']
    tabula_seeds = [s for s, r in regime_by_seed.items() if r == 'tabula_rasa']
    
    return {
        'warmup_seeds': warmup_seeds,
        'tabula_seeds': tabula_seeds,
        'regime_by_seed': regime_by_seed
    }

def compute_statistics(results: Dict, regimes: Dict) -> Dict:
    """Compute performance statistics stratified by regime."""
    
    stats = {
        'warmup_regime': {},
        'tabula_regime': {},
        'overall': {}
    }
    
    for n_eff in N_EFF_VALUES:
        # Warmup regime
        warmup_means = [results[(n_eff, s)]['post_release_mean'] for s in regimes['warmup_seeds']]
        stats['warmup_regime'][n_eff] = {
            'mean': np.mean(warmup_means) if warmup_means else np.nan,
            'std': np.std(warmup_means) if warmup_means else np.nan,
            'n': len(warmup_means)
        }
        
        # Tabula rasa regime
        tabula_means = [results[(n_eff, s)]['post_release_mean'] for s in regimes['tabula_seeds']]
        stats['tabula_regime'][n_eff] = {
            'mean': np.mean(tabula_means) if tabula_means else np.nan,
            'std': np.std(tabula_means) if tabula_means else np.nan,
            'n': len(tabula_means)
        }
        
        # Overall
        all_means = warmup_means + tabula_means
        stats['overall'][n_eff] = {
            'mean': np.mean(all_means),
            'std': np.std(all_means),
            'n': len(all_means)
        }
    
    return stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_figure(results: Dict, regimes: Dict, stats: Dict) -> plt.Figure:
    """Create 2x2 regime-stratified figure."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 8: Regime-Dependent n_eff Sensitivity', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    colors = {'1.0': '#2E7D32', '20.0': '#C62828'}
    
    # Top row: Expert weight evolution
    for col, (regime_type, seeds) in enumerate([
        ('Warmup-Dominant', regimes['warmup_seeds']),
        ('Tabula Rasa-Dominant', regimes['tabula_seeds'])
    ]):
        ax = axes[0, col]
        
        if seeds:
            # Use first seed as representative
            seed = seeds[0]
            result = results[(1.0, seed)]  # n_eff doesn't matter for weights
            
            steps = np.arange(len(result['warmup_weights']))
            ax.plot(steps, result['warmup_weights'], 
                   color='#1976D2', linewidth=2, label='Warmup Expert')
            ax.plot(steps, result['tabula_weights'], 
                   color='#D32F2F', linewidth=2, label='Tabula Rasa Expert')
            ax.axvline(RELEASE_STEP, color='black', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Model Release')
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Expert Weight', fontsize=11)
        ax.set_title(f'{regime_type} Regime\n(Seeds: {seeds})', 
                    fontsize=12, fontweight='bold')
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    # Bottom row: Performance curves stratified by regime
    for col, (regime_type, seeds, regime_key) in enumerate([
        ('Warmup-Dominant', regimes['warmup_seeds'], 'warmup_regime'),
        ('Tabula Rasa-Dominant', regimes['tabula_seeds'], 'tabula_regime')
    ]):
        ax = axes[1, col]
        
        for n_eff in N_EFF_VALUES:
            n_eff_str = f"{n_eff:.1f}"
            
            if seeds:
                # Plot all seeds in this regime
                for seed in seeds:
                    result = results[(n_eff, seed)]
                    
                    # Smooth rewards
                    window = 60
                    rewards = np.array(result['rewards'])
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    steps = np.arange(len(smoothed))
                    
                    ax.plot(steps, smoothed, 
                           color=colors[n_eff_str], 
                           linewidth=1.5, 
                           alpha=0.5,
                           label=f'n_eff={n_eff_str}' if seed == seeds[0] else '')
        
        # Add performance annotations
        regime_stats = stats[regime_key]
        text_y = 0.95
        for n_eff in N_EFF_VALUES:
            n_eff_str = f"{n_eff:.1f}"
            mean = regime_stats[n_eff]['mean']
            if not np.isnan(mean):
                ax.text(0.98, text_y, f'n_eff={n_eff_str}: {mean:.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
                       facecolor=colors[n_eff_str], alpha=0.15))
                text_y -= 0.08
        
        ax.axvline(RELEASE_STEP, color='black', linestyle='--', 
                  linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Reward (60-step MA)', fontsize=11)
        ax.set_title(f'{regime_type} Performance', 
                    fontsize=12, fontweight='bold')
        ax.legend(framealpha=0.9, fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# TABLE GENERATION
# ============================================================================

def print_console_table(regimes: Dict, stats: Dict):
    """Print formatted table to console."""
    
    print("\n" + "="*80)
    print("REGIME-STRATIFIED PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\n--- REGIME CLASSIFICATION ---")
    print(f"Warmup-dominant seeds:      {regimes['warmup_seeds']} ({len(regimes['warmup_seeds'])}/{len(SEEDS)} = {100*len(regimes['warmup_seeds'])/len(SEEDS):.0f}%)")
    print(f"Tabula rasa-dominant seeds: {regimes['tabula_seeds']} ({len(regimes['tabula_seeds'])}/{len(SEEDS)} = {100*len(regimes['tabula_seeds'])/len(SEEDS):.0f}%)")
    
    print("\n--- PERFORMANCE BY REGIME ---")
    print(f"{'Configuration':<25} {'Mean Reward':<15} {'Std Dev':<12} {'N Seeds':<10}")
    print("-" * 80)
    
    for regime_name, regime_key in [
        ('WARMUP-DOMINANT REGIME', 'warmup_regime'),
        ('TABULA RASA-DOMINANT REGIME', 'tabula_regime'),
        ('OVERALL (ALL SEEDS)', 'overall')
    ]:
        print(f"\n{regime_name}:")
        regime_stats = stats[regime_key]
        
        for n_eff in N_EFF_VALUES:
            s = regime_stats[n_eff]
            print(f"  n_eff = {n_eff:<18.1f} {s['mean']:>8.4f}        {s['std']:>8.4f}     {s['n']:>5d}")
        
        # Compute effect size
        if len(N_EFF_VALUES) == 2:
            mean_low = regime_stats[N_EFF_VALUES[0]]['mean']
            mean_high = regime_stats[N_EFF_VALUES[1]]['mean']
            if not np.isnan(mean_low) and not np.isnan(mean_high):
                effect = 100 * (mean_low - mean_high) / mean_high
                print(f"  → Effect size: {effect:+.2f}%")
    
    print("\n" + "="*80 + "\n")

def generate_latex_table(regimes: Dict, stats: Dict) -> str:
    """Generate LaTeX table code."""
    
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Regime-Stratified n\\_eff Sensitivity Analysis}",
        "\\label{tab:neff-sensitivity}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "\\textbf{Regime} & \\textbf{n\\_eff} & \\textbf{Mean} & \\textbf{Std} & \\textbf{N} \\\\",
        "\\midrule",
    ]
    
    for regime_name, regime_key in [
        ('Warmup-Dominant', 'warmup_regime'),
        ('Tabula Rasa-Dominant', 'tabula_regime'),
    ]:
        regime_stats = stats[regime_key]
        
        for i, n_eff in enumerate(N_EFF_VALUES):
            s = regime_stats[n_eff]
            prefix = regime_name if i == 0 else ''
            lines.append(
                f"{prefix:<20} & {n_eff:.1f} & {s['mean']:.4f} & {s['std']:.4f} & {s['n']} \\\\"
            )
        
        if regime_key == 'warmup_regime':
            lines.append("\\midrule")
    
    # Add overall
    lines.append("\\midrule")
    overall_stats = stats['overall']
    for i, n_eff in enumerate(N_EFF_VALUES):
        s = overall_stats[n_eff]
        prefix = '\\textbf{Overall}' if i == 0 else ''
        lines.append(
            f"{prefix:<20} & {n_eff:.1f} & {s['mean']:.4f} & {s['std']:.4f} & {s['n']} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Figure 8 unified analysis')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force re-run experiments (ignore cache)')
    args = parser.parse_args()
    
    # Step 1: Run experiments (or load cache)
    results = run_all_experiments(force_rerun=args.force_rerun)
    
    # Step 2: Classify regimes
    regimes = classify_regimes(results)
    
    # Step 3: Compute statistics
    stats = compute_statistics(results, regimes)
    
    # Step 4: Generate figure
    logger.info("Generating figure...")
    fig = create_figure(results, regimes, stats)
    output_dir = Path(__file__).parent / "results"
    output_path = output_dir / "figure8_regime_stratified.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved figure to {output_path}")
    plt.close(fig)
    
    # Step 5: Print console table
    print_console_table(regimes, stats)
    
    # Step 6: Generate LaTeX table
    latex_table = generate_latex_table(regimes, stats)
    latex_path = output_dir / "appendixC_neff_sensitivity.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    logger.info(f"✅ Saved LaTeX table to {latex_path}")
    
    logger.info("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
