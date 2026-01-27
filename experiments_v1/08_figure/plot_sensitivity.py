"""
Figure 8: Sensitivity Analysis - Hybrid Visualization (Fixed Determinism)
=======================================================================
Visualizes robustness of Prior Strength (n_effective) with strict RNG control.

Changes:
1. FIXED: Added `np.random.seed(seed)` to runners to reset GLOBAL state.
   - Prevents CorrallingRouter divergence in pre-release phase.
   - Ensures curves overlap perfectly until t=300 (Scientific Validity).
2. VISUAL: Hybrid plot showing Optimal, Failure, and Robustness Band.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging

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

# Simulation Params
TOTAL_STEPS = 1000
RELEASE_STEP = 300
WINDOW_SIZE = 60

# Sensitivity Sweep Range
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
# EXPERIMENT RUNNERS (Fixed RNG)
# ============================================================================
def run_adaptation_experiment(n_effective: float, seed: int = 42):
    # [FIX] Reset GLOBAL seed for CorrallingRouter determinism
    np.random.seed(seed) 
    
    evaluator = load_real_data()
    if not evaluator: return []
    
    # Local RNG for data shuffling
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
    
    # [VERIFICATION] Confirm Corralling is active
    if router.corralling_router:
        # Log only if it's the first time seeing this router instance
        if not hasattr(router, '_logged_corralling'):
            logger.info("   ✅ Corralling Router ACTIVE with experts: " + 
                       str([type(e).__name__ for e in router.corralling_router.experts]))
            router._logged_corralling = True
    else:
        logger.error("   ❌ Corralling Router NOT ACTIVE!")
    
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
        history.append(reward)
        
    return history

def run_cold_start_baseline(seed: int = 42):
    # [FIX] Reset GLOBAL seed for CorrallingRouter determinism
    np.random.seed(seed)
    
    evaluator = load_real_data()
    if not evaluator: return []
    
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    # Cold Start: NO priors for NEW model, but WARMUP for OLD models
    # [FIX] Use WARMUP priors to simulate realistic production baseline
    # If we used "none" (Global Cold Start), the router would explore aggressively
    # and find the new model by accident. We want to test if it can break
    # the "Exploitation Trap" of the incumbent models.
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
    
    # [VERIFICATION] Confirm Corralling is active
    if router.corralling_router:
        # Log only if it's the first time seeing this router instance
        if not hasattr(router, '_logged_corralling'):
            logger.info("   ✅ Corralling Router ACTIVE with experts: " + 
                       str([type(e).__name__ for e in router.corralling_router.experts]))
            router._logged_corralling = True
    else:
        logger.error("   ❌ Corralling Router NOT ACTIVE!")
    
    history = []
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS: break
        if t == RELEASE_STEP:
            router.bandit.models.append(NEW_MODEL)
            # Cold Start: Identity Matrix
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            router.registry[NEW_MODEL] = create_model_registry([NEW_MODEL])[NEW_MODEL]
            
            if router.corralling_router:
                router.corralling_router.add_model(NEW_MODEL)
                cold_A = router.bandit.init_lambda * np.eye(router.bandit.dim)
                cold_b = np.zeros(router.bandit.dim)
                for expert in router.corralling_router.experts:
                    if hasattr(expert, 'add_model'):
                        expert_type = type(expert).__name__
                        if 'TabulaRasa' in expert_type:
                            expert.add_model(NEW_MODEL, 0.5)
                        else:
                            expert.add_model(NEW_MODEL, cold_A, cold_b, 0.5)

        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        history.append(reward)
    return history

# ============================================================================
# PLOTTING
# ============================================================================
def plot_hybrid_sensitivity(results: Dict[str, List[float]], cold_mean: float, best_n_eff: float):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def smooth(data, w=WINDOW_SIZE):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    plt.figure(figsize=(12, 7))
    
    smoothed_data = {}
    for k, v in results.items():
        smoothed_data[k] = smooth(v)
        
    min_len = min(len(v) for v in smoothed_data.values())
    x_axis = [i + WINDOW_SIZE//2 for i in range(min_len)]
    
    # 1. Robustness Band
    robust_keys = [2.0, 10.0, 20.0]
    robust_curves = [smoothed_data[k][:min_len] for k in robust_keys if k in smoothed_data]
    if robust_curves:
        robust_matrix = np.array(robust_curves)
        y_min = np.min(robust_matrix, axis=0)
        y_max = np.max(robust_matrix, axis=0)
        plt.fill_between(x_axis, y_min, y_max, 
                        color='#2ecc71', alpha=0.15,
                        label=f"Robust Range: $n_{{eff}} \\in [2, 20]$")
        for curve in robust_curves:
            plt.plot(x_axis, curve, color='#2ecc71', linewidth=0.5, alpha=0.3)

    # 2. Failure Mode
    if 1.0 in smoothed_data:
        n1_mean = np.mean(results[1.0][RELEASE_STEP:])
        n1_improvement = ((n1_mean - cold_mean) / cold_mean) * 100
        plt.plot(x_axis, smoothed_data[1.0][:min_len], 
                label=f"Weak Prior: $n_{{eff}}=1.0$ ({n1_improvement:+.1f}%)",
                color='#3498db', linestyle=':', linewidth=2.0, alpha=0.8)

    # 3. Optimal (Best performer)
    if best_n_eff in smoothed_data:
        best_mean_post = np.mean(results[best_n_eff][RELEASE_STEP:])
        imp = ((best_mean_post - cold_mean) / cold_mean) * 100
        plt.plot(x_axis, smoothed_data[best_n_eff][:min_len], 
                label=f"Best: $n_{{eff}}={best_n_eff:.1f}$ ({imp:+.1f}%)",
                color='#2ecc71', linewidth=3.0)

    # 4. Baseline (Plot ONLY after release to avoid confusion)
    # Pre-release, both strategies are identical (Warmup).
    # Post-release, this line shows what happens if the NEW model starts cold.
    if 'cold_start' in smoothed_data:
        post_release_mask = np.array(x_axis) >= RELEASE_STEP
        if np.any(post_release_mask):
            plt.plot(np.array(x_axis)[post_release_mask], 
                    np.array(smoothed_data['cold_start'][:min_len])[post_release_mask], 
                    color='#e74c3c', linestyle='--', linewidth=2.5,
                    label="Baseline: Cold Start (New Model)")

    # 5. Shared History (Pre-release)
    # Since all runs are identical before release, plot one representative line
    pre_release_mask = np.array(x_axis) <= RELEASE_STEP
    if np.any(pre_release_mask) and best_n_eff in smoothed_data:
        plt.plot(np.array(x_axis)[pre_release_mask], 
                np.array(smoothed_data[best_n_eff][:min_len])[pre_release_mask], 
                color='gray', linestyle='-', linewidth=2.0, alpha=0.6,
                label="Shared Warmup Phase")

    plt.axvline(x=RELEASE_STEP, color='black', linestyle=':', linewidth=1.5, label="Model Release")
    plt.title("Figure 8: Sensitivity Analysis - Prior Strength ($n_{eff}$)", fontsize=16, fontweight='bold')
    plt.xlabel("Routing Steps (t)", fontsize=13)
    plt.ylabel("Moving Average Reward", fontsize=13)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / "figure8_sensitivity_hybrid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved plot to {output_path}")

# ============================================================================
# MAIN
# ============================================================================
def run_sensitivity_sweep():
    results = {}
    
    logger.info("Running Cold Start Baseline...")
    results['cold_start'] = run_cold_start_baseline()
    
    for n_eff in N_EFFECTIVE_VALUES:
        logger.info(f"Running n_effective = {n_eff}...")
        results[n_eff] = run_adaptation_experiment(n_eff)
    
    cold_mean = np.mean(results['cold_start'][RELEASE_STEP:])
    best_n_eff = max(N_EFFECTIVE_VALUES, key=lambda n: np.mean(results[n][RELEASE_STEP:]))
    
    # Log results table
    print("\n" + "="*60)
    print(f"{'Configuration':<20} | {'Mean Reward':<12} | {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Cold Start':<20} | {cold_mean:.4f}       | 0.00%")
    for n_eff in N_EFFECTIVE_VALUES:
        n_mean = np.mean(results[n_eff][RELEASE_STEP:])
        diff = ((n_mean - cold_mean) / cold_mean) * 100
        tag = "★" if n_eff == best_n_eff else ""
        print(f"n_eff = {n_eff:<12} | {n_mean:.4f}       | {diff:+.2f}% {tag}")
    print("="*60 + "\n")

    plot_hybrid_sensitivity(results, cold_mean, best_n_eff)

if __name__ == "__main__":
    run_sensitivity_sweep()
