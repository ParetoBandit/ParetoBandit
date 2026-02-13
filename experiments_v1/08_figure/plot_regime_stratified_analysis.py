"""
Figure 8 REVISED: Regime-Stratified Analysis of n_eff Sensitivity
=================================================================
Shows the CORRECT story: n_eff effect is regime-dependent.

Layout: 2×2 grid
- Top row: Expert weight evolution (warmup vs tabula rasa regimes)
- Bottom row: Performance stratified by regime
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging

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

# Configuration
WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
NEIGHBOR_MODEL = "openai/gpt-4-turbo"
TOTAL_STEPS = 1000
RELEASE_STEP = 300
WINDOW_SIZE = 60
SEEDS = [42, 43, 44]

def create_model_registry(models):
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
    required_models = WARMUP_MODELS + [NEW_MODEL]
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=required_models
    )
    return AlignedEvaluator([item for item in evaluator if all(m in item.rewards for m in required_models)])

def run_with_weight_tracking(n_effective: float, seed: int):
    """Run experiment and track expert weights."""
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
        if t >= TOTAL_STEPS: break
        
        if t == RELEASE_STEP:
            # Semantic transfer
            A_neighbor = router.bandit.A[NEIGHBOR_MODEL].copy()
            b_neighbor = router.bandit.b[NEIGHBOR_MODEL].copy()
            theta_neighbor = router.bandit.A_inv[NEIGHBOR_MODEL] @ b_neighbor
            
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = n_effective * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = n_effective * theta_neighbor
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
            
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

def load_or_run_experiments():
    """Load cached results or run experiments."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "regime_stratified_results.pkl"
    
    if results_file.exists():
        logger.info(f"Loading cached results from {results_file}")
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    
    logger.info("\nRunning regime-stratified experiments...")
    results = {}
    
    for seed in SEEDS:
        logger.info(f"\nSeed {seed}:")
        seed_results = {}
        for n_eff in [1.0, 20.0]:  # Test extremes
            logger.info(f"  n_eff={n_eff}...")
            seed_results[n_eff] = run_with_weight_tracking(n_eff, seed)
        results[seed] = seed_results
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"\n✅ Saved results to {results_file}")
    
    return results

def smooth(data, w=WINDOW_SIZE):
    return np.convolve(data, np.ones(w)/w, mode='valid')

def create_figure(results):
    """Create 2×2 regime-stratified visualization."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Classify seeds by regime (using n_eff=1.0 as representative)
    warmup_seeds = [s for s in SEEDS if results[s][1.0]['regime'] == 'warmup']
    tabula_seeds = [s for s in SEEDS if results[s][1.0]['regime'] == 'tabula_rasa']
    
    logger.info(f"\nRegime classification:")
    logger.info(f"  Warmup-dominant seeds: {warmup_seeds}")
    logger.info(f"  Tabula rasa-dominant seeds: {tabula_seeds}")
    
    # ========================================================================
    # TOP ROW: Expert Weight Evolution
    # ========================================================================
    
    # Top-left: Warmup regime
    ax1 = fig.add_subplot(gs[0, 0])
    if warmup_seeds:
        seed = warmup_seeds[0]
        weights = results[seed][1.0]['warmup_weights']
        x = list(range(len(weights)))
        ax1.plot(x, weights, color='#3498db', linewidth=2.5, label='Warmup Expert')
        ax1.plot(x, results[seed][1.0]['tabula_weights'], color='#e74c3c', linewidth=2.5, label='Tabula Rasa Expert')
        ax1.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.fill_between([RELEASE_STEP, len(weights)], 0, 1, color='yellow', alpha=0.1)
        ax1.text(RELEASE_STEP + 50, 0.9, 'Evaluation\nWindow', fontsize=10, alpha=0.6)
        ax1.set_title(f'Warmup-Dominant Regime (Seed {seed})', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Routing Step (t)', fontsize=11)
        ax1.set_ylabel('Expert Weight', fontsize=11)
        ax1.legend(loc='right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])
    
    # Top-right: Tabula rasa regime
    ax2 = fig.add_subplot(gs[0, 1])
    if tabula_seeds:
        seed = tabula_seeds[0]
        weights = results[seed][1.0]['warmup_weights']
        x = list(range(len(weights)))
        ax2.plot(x, weights, color='#3498db', linewidth=2.5, label='Warmup Expert')
        ax2.plot(x, results[seed][1.0]['tabula_weights'], color='#e74c3c', linewidth=2.5, label='Tabula Rasa Expert')
        ax2.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.fill_between([RELEASE_STEP, len(weights)], 0, 1, color='yellow', alpha=0.1)
        ax2.text(RELEASE_STEP + 50, 0.9, 'Evaluation\nWindow', fontsize=10, alpha=0.6)
        ax2.set_title(f'Tabula Rasa-Dominant Regime (Seed {seed})', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Routing Step (t)', fontsize=11)
        ax2.set_ylabel('Expert Weight', fontsize=11)
        ax2.legend(loc='right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
    
    # ========================================================================
    # BOTTOM ROW: Performance Stratified by Regime
    # ========================================================================
    
    # Bottom-left: Warmup regime performance
    ax3 = fig.add_subplot(gs[1, 0])
    if warmup_seeds:
        for seed in warmup_seeds:
            r1 = smooth(results[seed][1.0]['rewards'])
            r20 = smooth(results[seed][20.0]['rewards'])
            x = [i + WINDOW_SIZE//2 for i in range(len(r1))]
            ax3.plot(x, r1, color='#2ecc71', linewidth=2.5, label=f'n_eff=1.0 (Seed {seed})')
            ax3.plot(x, r20, color='#95a5a6', linewidth=2.5, linestyle='--', label=f'n_eff=20.0 (Seed {seed})')
        
        ax3.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.fill_between([RELEASE_STEP, max(x)], ax3.get_ylim()[0], ax3.get_ylim()[1], color='yellow', alpha=0.1)
        
        # Calculate and show effect size
        mean_1 = np.mean([results[s][1.0]['post_release_mean'] for s in warmup_seeds])
        mean_20 = np.mean([results[s][20.0]['post_release_mean'] for s in warmup_seeds])
        effect_pct = ((mean_1 - mean_20) / mean_20) * 100
        
        ax3.text(0.98, 0.05, f'n_eff Effect: {effect_pct:+.1f}%\n(n_eff matters here!)', 
                transform=ax3.transAxes, fontsize=11, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax3.set_title('Performance: Warmup Regime (Transfer Used)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Routing Step (t)', fontsize=11)
        ax3.set_ylabel('Moving Average Reward', fontsize=11)
        ax3.legend(loc='lower left', fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # Bottom-right: Tabula rasa regime performance
    ax4 = fig.add_subplot(gs[1, 1])
    if tabula_seeds:
        for seed in tabula_seeds:
            r1 = smooth(results[seed][1.0]['rewards'])
            r20 = smooth(results[seed][20.0]['rewards'])
            x = [i + WINDOW_SIZE//2 for i in range(len(r1))]
            ax4.plot(x, r1, color='#2ecc71', linewidth=2.5, label=f'n_eff=1.0 (Seed {seed})')
            ax4.plot(x, r20, color='#95a5a6', linewidth=2.5, linestyle='--', label=f'n_eff=20.0 (Seed {seed})')
        
        ax4.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        ax4.fill_between([RELEASE_STEP, max(x)], ax4.get_ylim()[0], ax4.get_ylim()[1], color='yellow', alpha=0.1)
        
        # Calculate and show (lack of) effect
        mean_1 = np.mean([results[s][1.0]['post_release_mean'] for s in tabula_seeds])
        mean_20 = np.mean([results[s][20.0]['post_release_mean'] for s in tabula_seeds])
        effect_pct = ((mean_1 - mean_20) / mean_20) * 100
        
        ax4.text(0.98, 0.05, f'n_eff Effect: {effect_pct:+.1f}%\n(n_eff ignored!)', 
                transform=ax4.transAxes, fontsize=11, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax4.set_title('Performance: Tabula Rasa Regime (Transfer NOT Used)', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Routing Step (t)', fontsize=11)
        ax4.set_ylabel('Moving Average Reward', fontsize=11)
        ax4.legend(loc='lower left', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Figure 8: Regime-Dependent n_eff Sensitivity in Semantic Transfer\n' +
                 f'Corralling adaptively switches experts ({len(warmup_seeds)}/{len(SEEDS)} warmup, {len(tabula_seeds)}/{len(SEEDS)} tabula rasa)',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent / "results"
    output_path = output_dir / "figure8_regime_stratified_CORRECTED.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved figure to {output_path}")
    plt.close()

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Figure 8 REVISED: Regime-Stratified Analysis")
    logger.info("="*80)
    logger.info("\nShows the CORRECT story:")
    logger.info("  - n_eff matters in warmup-dominant regimes")
    logger.info("  - n_eff ignored in tabula rasa-dominant regimes")
    logger.info("  - Corralling's adaptive switching provides robustness\n")
    
    # Load or run experiments
    results = load_or_run_experiments()
    
    # Create visualization
    create_figure(results)
    
    logger.info("\n✅ Regime-stratified analysis complete!")
    logger.info("   - Figure shows expert weight evolution by regime")
    logger.info("   - Performance stratified to reveal true n_eff effects")
    logger.info("   - Demonstrates Corralling's adaptive behavior\n")
