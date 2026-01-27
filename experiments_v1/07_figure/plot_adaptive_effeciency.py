"""
Figure 6: Adaptive Efficiency (Zero-Shot Readiness) - Production Router
========================================================================

Tests the ACTUAL production BanditRouter with Heterogeneous Experts Strategy.

The router automatically:
1. Uses Heterogeneous Corralling Architecture:
   - Expert 1 (Conservative): Decaying α 1.0→0.01 (efficient in stable periods)
   - Expert 2 (Adaptive): Constant α 2.0 (vigilant during distribution shifts)
   - Meta-Learner: Auto-switches based on which expert performs better
2. Does semantic transfer when adding GPT-5.1 via register_model()
3. Updates all experts with feedback (FIXED: corralling_router.update now called)
4. Adapts expert weights dynamically based on performance

Scenario:
1. Train on 2-model portfolio (Mixtral, GPT-4-turbo) for 300 steps
2. At t=300, dynamically add GPT-5.1 via router.register_model()
3. Router automatically finds semantic neighbor (GPT-4) and transfers knowledge
4. Conservative expert exploits transfer immediately (α=0.01)
5. Adaptive expert validates and explores alternatives (α=2.0)
6. Meta-learner maintains weight on Conservative if transfer is good

Expected Result: No performance dip at t=300 (zero-shot readiness).

This tests the REAL production system with all fixes applied.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging
import joblib
from tqdm import tqdm
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_TRIALS = 30
CONFIDENCE_LEVEL = 0.95
TOTAL_STEPS = 800
RELEASE_STEP = 300
WINDOW_SIZE = 60

# Models
WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"

# ============================================================================
# MODEL REGISTRY
# ============================================================================
def create_model_registry():
    """Create registry with cost and description metadata (NO HLE - testing pure semantic transfer)."""
    return {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
            "description": "Efficient sparse mixture-of-experts model, good for reasoning but cheaper."
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
            "description": "High-intelligence flagship model, excellent at complex reasoning, coding, and creative writing."
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0,
            "output_cost_per_m": 45.0,
            "description": "Next-generation high-intelligence flagship model, superior reasoning and multimodal capabilities."
        }
    }

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load evaluation data."""
    try:
        evaluator = AlignedEvaluator.from_jsonl_gz(
            DEV_DATA_PATH_ALL_MODELS,
            required_models=WARMUP_MODELS + [NEW_MODEL]
        )
        data = [item for item in evaluator if all(m in item.rewards for m in WARMUP_MODELS + [NEW_MODEL])]
        logger.info(f"✅ Loaded {len(data)} atomic samples")
        return data
    except Exception as e:
        logger.error(f"Data error: {e}")
        return []

# ============================================================================
# SINGLE TRIAL
# ============================================================================
def run_trial(seed: int, data: List, encoder, pca) -> Dict[str, List[float]]:
    """
    Run a single trial using production BanditRouter with Heterogeneous Experts.
    
    The router uses the Heterogeneous Strategy to handle both stable and
    non-stationary regimes automatically:
    
    Phase 1 (t=0-300): Stable Period
    - Conservative expert (α→0.01) exploits warmup priors efficiently
    - Adaptive expert (α=2.0) explores but likely gets downweighted
    - Meta-learner favors Conservative (low regret)
    
    Phase 2 (t=300): Distribution Shift (GPT-5.1 released)
    - Conservative expert receives transferred knowledge from GPT-4
    - With α=0.01, it immediately exploits the transfer (zero-shot)
    - Adaptive expert explores the new model as backup (α=2.0)
    - Meta-learner keeps weight on Conservative if transfer is good
    
    Expected: No performance dip (semantic transfer + heterogeneous adaptation)
    """
    rng = np.random.RandomState(seed)
    trial_data = data.copy()
    rng.shuffle(trial_data)
    
    # Create production router with initial 2 models
    registry = create_model_registry()
    initial_registry = {k: v for k, v in registry.items() if k in WARMUP_MODELS}
    
    # Create production router with warmup priors from 80k battles + Heterogeneous Corralling
    router = BanditRouter.create(
        model_registry=initial_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),  # Use actual warmup priors for Mixtral + GPT-4-turbo
        use_corralling=True,  # Enables Heterogeneous Experts Strategy:
                              # - Expert 1 (Conservative): Decaying α 1.0→0.01 for stable exploitation
                              # - Expert 2 (Adaptive): Constant α 2.0 for vigilant exploration
                              # - Meta-Learner: Auto-switches based on performance
        corralling_learning_rate=0.1,  # How fast meta-learner adapts weights
        corralling_gamma=0.05,  # Mixing parameter to prevent expert death
        pca_path=DEFAULT_PCA_PATH
    )
    
    # Track history
    history = []
    expert_weights_history = []
    
    for t, item in enumerate(trial_data):
        if t >= TOTAL_STEPS:
            break
        
        # --- RELEASE EVENT ---
        if t == RELEASE_STEP:
            logger.info(f"  [Trial {seed}] Registering {NEW_MODEL} at step {t}")
            # Router automatically does semantic transfer via register_model()!
            router.register_model(
                model_id=NEW_MODEL,
                cost_usd=registry[NEW_MODEL]["input_cost_per_m"],
                speed="balanced"  # Progressive registration - Tier B (T-shirt sizing)
            )
        
        # Route (BanditRouter.route() takes text prompt and returns model + log)
        selected, log_entry = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        
        # Update (FIXED: correct parameter order - model_id, context, reward)
        router.update(selected, item.prompt, reward)
        
        history.append(reward)
        
        # Track expert weights from Corralling Router
        if router.corralling_router:
            expert_weights_history.append({
                'conservative': router.corralling_router.weights[0],
                'adaptive': router.corralling_router.weights[1]
            })
    
    return {
        "history": history,
        "expert_weights": expert_weights_history
    }

# ============================================================================
# MULTI-TRIAL RUNNER
# ============================================================================
def run_rigorous_experiment():
    """Run N trials with production BanditRouter."""
    data = load_data()
    if not data:
        return None
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    all_histories = []
    all_expert_weights = []
    
    logger.info(f"\n🔬 Running {N_TRIALS} rigorous trials...")
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        result = run_trial(seed, data, encoder, pca)
        all_histories.append(result["history"])
        all_expert_weights.append(result["expert_weights"])
    
    return {
        "histories": np.array(all_histories),
        "expert_weights": all_expert_weights
    }

# ============================================================================
# PLOTTING
# ============================================================================
def plot_results(results):
    """Plot reward over time with 95% CI."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    histories = results["histories"]
    
    # Smooth each trial
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'),
        1, histories
    )
    
    mean = np.mean(smoothed, axis=0)
    sem = stats.sem(smoothed, axis=0)
    ci = sem * stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2., N_TRIALS - 1)
    
    x_axis = np.arange(len(mean)) + WINDOW_SIZE // 2
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Top: Reward over time
    ax1.plot(x_axis, mean, color="#2ecc71", linewidth=2.5, 
             label="Production Router (Heterogeneous Experts + Semantic Transfer)")
    ax1.fill_between(x_axis, mean - ci, mean + ci, color="#2ecc71", alpha=0.2,
                      label=f"{int(CONFIDENCE_LEVEL*100)}% Confidence Interval")
    
    ax1.axvline(x=RELEASE_STEP, color='black', alpha=0.5, linewidth=2, linestyle='--',
                label="GPT-5.1 Release (Semantic Transfer Applied)")
    
    ax1.set_title(f"Zero-Shot Readiness: Heterogeneous Experts Strategy (N={N_TRIALS}, 95% CI)", 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel("Routing Steps", fontsize=13)
    ax1.set_ylabel("Average Reward", fontsize=13)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Expert weights over time
    if results["expert_weights"] and len(results["expert_weights"][0]) > 0:
        # Convert list of trials (each containing list of dicts) to arrays
        conservative_weights = []
        adaptive_weights = []
        
        for trial_weights in results["expert_weights"]:
            trial_conservative = [w['conservative'] for w in trial_weights]
            trial_adaptive = [w['adaptive'] for w in trial_weights]
            conservative_weights.append(trial_conservative)
            adaptive_weights.append(trial_adaptive)
        
        # Average across trials
        conservative_mean = np.mean(conservative_weights, axis=0)
        adaptive_mean = np.mean(adaptive_weights, axis=0)
        
        x_weights = np.arange(len(conservative_mean))
        
        ax2.plot(x_weights, conservative_mean, color='#3498db', linewidth=2, 
                 label='Expert 1 (Conservative: α→0.01)')
        ax2.plot(x_weights, adaptive_mean, color='#e74c3c', linewidth=2,
                 label='Expert 2 (Adaptive: α=2.0)')
        
        ax2.axvline(x=RELEASE_STEP, color='black', alpha=0.3, linestyle='--',
                    label='GPT-5.1 Release')
        
        ax2.set_title("Expert Weights (Corralling Meta-Learning)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Routing Steps", fontsize=12)
        ax2.set_ylabel("Weight", fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    out_file = output_dir / "figure6_adaptive_efficiency.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved plot to {out_file}")
    
    # Print summary statistics
    idx_check = RELEASE_STEP + 100 - (WINDOW_SIZE // 2)
    if idx_check < len(mean):
        pre_release_mean = np.mean(histories[:, max(0, RELEASE_STEP-50):RELEASE_STEP], axis=1).mean()
        post_release_mean = np.mean(histories[:, RELEASE_STEP+50:RELEASE_STEP+150], axis=1).mean()
        
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   Pre-Release (t=250-300):  {pre_release_mean:.3f}")
        logger.info(f"   Post-Release (t=350-450): {post_release_mean:.3f}")
        logger.info(f"   Impact: {post_release_mean - pre_release_mean:+.3f}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    results = run_rigorous_experiment()
    if results:
        plot_results(results)
