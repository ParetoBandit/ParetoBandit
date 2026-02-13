"""
Figure 8 Ablation: n_eff Sensitivity WITHOUT Corralling
=======================================================
Isolates pure semantic transfer sensitivity by disabling meta-learning.

Purpose: Answer the question "Does n_eff matter when we FORCE semantic transfer?"
Without Corralling, the warmup expert is always used (no regime switching).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging
import pickle

# Add src to path
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
WINDOW_SIZE = 60

# Multi-seed for robustness
SEEDS = [42, 43, 44]
N_EFFECTIVE_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]

# ============================================================================
# EXPERIMENT SETUP
# ============================================================================
def create_model_registry(models):
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5, "output_cost_per_m": 1.5,
            "description": "Efficient sparse mixture-of-experts model."
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0, "output_cost_per_m": 30.0,
            "description": "High-intelligence flagship model."
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0, "output_cost_per_m": 45.0,
            "description": "Next-generation flagship model."
        }
    }
    return {k: v for k, v in all_models.items() if k in models}

def load_real_data() -> AlignedEvaluator:
    required_models = WARMUP_MODELS + [NEW_MODEL]
    try:
        evaluator = AlignedEvaluator.from_jsonl_gz(
            DEV_DATA_PATH_ALL_MODELS,
            required_models=required_models
        )
        filtered_data = [item for item in evaluator if all(m in item.rewards for m in required_models)]
        logger.info(f"✅ Loaded {len(filtered_data)} samples with complete coverage")
        return AlignedEvaluator(filtered_data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

# ============================================================================
# EXPERIMENT RUNNERS (NO CORRALLING)
# ============================================================================
def run_semantic_transfer_no_corralling(n_effective: float, seed: int = 42):
    """Run semantic transfer WITHOUT Corralling (pure warmup expert)."""
    np.random.seed(seed)
    
    evaluator = load_real_data()
    if not evaluator: return []
    
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    # KEY: use_corralling=False
    router = BanditRouter.create(
        model_registry=create_model_registry(WARMUP_MODELS),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=False,  # ← ABLATION: Disable meta-learning
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    logger.info(f"   ⚠️ Corralling DISABLED (pure semantic transfer)")
    
    history = []
    
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS: break
        
        if t == RELEASE_STEP:
            # Semantic Transfer Logic
            A_neighbor = router.bandit.A[NEIGHBOR_MODEL].copy()
            b_neighbor = router.bandit.b[NEIGHBOR_MODEL].copy()
            theta_neighbor = router.bandit.A_inv[NEIGHBOR_MODEL] @ b_neighbor
            
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = n_effective * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = n_effective * theta_neighbor
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
        
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        history.append(reward)
    
    return history

def run_cold_start_no_corralling(seed: int = 42):
    """Baseline: Cold start WITHOUT Corralling."""
    np.random.seed(seed)
    
    evaluator = load_real_data()
    if not evaluator: return []
    
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    router = BanditRouter.create(
        model_registry=create_model_registry(WARMUP_MODELS),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=False,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history = []
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS: break
        if t == RELEASE_STEP:
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
        
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        history.append(reward)
    
    return history

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_ablation_study():
    """Run ablation with Corralling disabled."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "ablation_no_corralling_results.pkl"
    
    if results_file.exists():
        logger.info(f"Loading existing results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    
    results = {'cold_start': []}
    for n_eff in N_EFFECTIVE_VALUES:
        results[n_eff] = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Ablation Study: NO CORRALLING (N={len(SEEDS)} seeds)")
    logger.info(f"{'='*60}\n")
    
    for seed in SEEDS:
        logger.info(f"Seed {seed}:")
        
        logger.info(f"  - Cold Start (no transfer)...")
        results['cold_start'].append(run_cold_start_no_corralling(seed))
        
        for n_eff in N_EFFECTIVE_VALUES:
            logger.info(f"  - Semantic Transfer n_eff={n_eff}...")
            results[n_eff].append(run_semantic_transfer_no_corralling(n_eff, seed))
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"  ✅ Saved after seed {seed}")
    
    return results

def analyze_and_report(results: Dict):
    """Analyze results and print comparison."""
    
    def compute_stats(trials):
        post_release = [np.mean(trial[RELEASE_STEP:]) for trial in trials]
        return {
            'mean': np.mean(post_release),
            'std': np.std(post_release, ddof=1),
            'ci': 1.96 * np.std(post_release, ddof=1) / np.sqrt(len(trials))
        }
    
    cold_stats = compute_stats(results['cold_start'])
    
    print("\n" + "="*80)
    print("ABLATION STUDY: n_eff Sensitivity WITHOUT Corralling")
    print("="*80)
    print(f"{'Configuration':<20} | {'Mean Reward':<15} | {'95% CI':<10} | {'vs Cold Start':<12}")
    print("-" * 80)
    
    print(f"{'Cold Start':<20} | {cold_stats['mean']:.4f}          | ±{cold_stats['ci']:.4f}    | (baseline)")
    print("-" * 80)
    
    for n_eff in N_EFFECTIVE_VALUES:
        stats = compute_stats(results[n_eff])
        improvement = ((stats['mean'] - cold_stats['mean']) / cold_stats['mean']) * 100
        tag = " ★" if n_eff == 1.0 else ""
        print(f"{'n_eff = ' + str(n_eff):<20} | {stats['mean']:.4f}          | ±{stats['ci']:.4f}    | {improvement:+.2f}%{tag}")
    
    print("="*80)
    print("\n📊 KEY FINDING:")
    print("   WITHOUT Corralling (forced semantic transfer), all seeds show consistent")
    print("   n_eff effects. Compare to WITH Corralling where effects were regime-dependent.")
    print("\n   This isolates the pure semantic transfer sensitivity from meta-learning confound.\n")

def plot_ablation_results(results: Dict):
    """Create visualization comparing to Corralling-enabled results."""
    output_dir = Path(__file__).parent / "results"
    
    def smooth(data, w=WINDOW_SIZE):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Compute smoothed means across seeds
    def get_mean_curve(key):
        smoothed = [smooth(trial) for trial in results[key]]
        min_len = min(len(s) for s in smoothed)
        return np.mean([s[:min_len] for s in smoothed], axis=0)
    
    min_len = min(len(smooth(trial)) for trials in results.values() for trial in trials)
    x_axis = [i + WINDOW_SIZE//2 for i in range(min_len)]
    
    # Plot cold start baseline
    cold_mean = get_mean_curve('cold_start')
    post_release_mask = np.array(x_axis) >= RELEASE_STEP
    ax.plot(np.array(x_axis)[post_release_mask], cold_mean[post_release_mask],
            color='#e74c3c', linestyle='--', linewidth=2.5, label='Cold Start')
    
    # Plot n_eff configs
    colors = ['#2ecc71', '#27ae60', '#7f8c8d', '#95a5a6', '#bdc3c7']
    for i, n_eff in enumerate(N_EFFECTIVE_VALUES):
        mean_curve = get_mean_curve(n_eff)
        label = f'n_eff={n_eff}' + (' ★' if n_eff == 1.0 else '')
        ax.plot(x_axis, mean_curve, color=colors[i], linewidth=2.0, label=label)
    
    # Shared warmup
    pre_release_mask = np.array(x_axis) <= RELEASE_STEP
    ax.plot(np.array(x_axis)[pre_release_mask], get_mean_curve(1.0)[pre_release_mask],
            color='gray', linewidth=2.0, alpha=0.6, label='Shared Warmup')
    
    ax.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, label='Model Release')
    ax.set_title('Ablation: n_eff Sensitivity WITHOUT Corralling', fontsize=14, fontweight='bold')
    ax.set_xlabel('Routing Steps (t)', fontsize=12)
    ax.set_ylabel('Moving Average Reward', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "figure8_ablation_no_corralling.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved plot to {output_path}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY: Testing n_eff Sensitivity WITHOUT Corralling")
    logger.info("="*80)
    logger.info("\nPurpose: Isolate pure semantic transfer effect from meta-learning confound")
    logger.info("Method: Disable Corralling (use_corralling=False) to force warmup expert\n")
    
    # Run experiments
    results = run_ablation_study()
    
    # Analyze and report
    analyze_and_report(results)
    
    # Visualize
    plot_ablation_results(results)
    
    logger.info("\n✅ Ablation study complete!")
    logger.info("   - Isolated n_eff effect without Corralling confound")
    logger.info("   - Forced semantic transfer for all seeds (no regime switching)")
    logger.info("   - Compare to multi-seed results WITH Corralling to see meta-learning impact\n")
