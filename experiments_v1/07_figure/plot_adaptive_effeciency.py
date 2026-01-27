"""
Figure 6: Adaptive Efficiency (Zero-Shot Readiness) - Production Router
========================================================================

Tests the ACTUAL production BanditRouter behavior on the "Complex Subset".
Includes FORMAL STATISTICAL HYPOTHESIS TESTING (t-test, Wilcoxon, Cohen's d).

NARRATIVE PIVOT (KDD Response):
- We acknowledge GPT-5.1 is dominant on these tasks ($r \\approx 0$).
- Therefore, we do not claim to learn "nuanced preferences" (Correlation).
- We claim "Adoption Velocity": The Semantic Prior allows the router
  to identify the new dominant model IMMEDIATELY, skipping the exploration dip.

Scenario:
1. Train on 2-model portfolio (Mixtral, GPT-4-turbo) for 300 steps
2. At t=300, release GPT-5.1
3. Measure how fast the router switches to GPT-5.1 on Complex Tasks.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set
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

# Evaluation Window for Statistics (The "Adoption Gap")
# We measure the gap immediately after release to prove "Zero-Shot"
EVAL_WINDOW_START = 300
EVAL_WINDOW_END = 500

# Models
WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"

# ============================================================================
# MODEL REGISTRY
# ============================================================================
def create_model_registry():
    return {
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

# ============================================================================
# DATA LOADING & FILTERING
# ============================================================================
def load_data():
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

def identify_complex_subset(data: List) -> Set[int]:
    """
    Identifies "Complex" tasks based on PROMPT CONTENT (Covariates).
    This ensures methodological rigor (no circular reasoning).
    """
    complex_indices = set()
    complexity_signals = [
        "code", "python", "function", "algorithm", 
        "solve", "math", "proof", "calculate",
        "step-by-step", "logic", "reasoning",
        "analysis", "difference between", "compare"
    ]
    
    for i, item in enumerate(data):
        prompt_lower = item.prompt.lower()
        is_long = len(prompt_lower.split()) > 50
        has_signal = any(sig in prompt_lower for sig in complexity_signals)
        
        if is_long or has_signal:
            complex_indices.add(i)
            
    logger.info(f"🔍 Identified {len(complex_indices)} 'Complex' prompts ({len(complex_indices)/len(data):.1%} of traffic)")
    return complex_indices

# ============================================================================
# SINGLE TRIAL
# ============================================================================
def run_trial(seed: int, data: List, encoder, pca, target_indices: Set[int]) -> Dict[str, List[float]]:
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    registry = create_model_registry()
    initial_registry = {k: v for k, v in registry.items() if k in WARMUP_MODELS}
    
    router = BanditRouter.create(
        model_registry=initial_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history_target = []
    expert_weights_history = []
    
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        is_target = idx in target_indices
        
        if t_step == RELEASE_STEP:
            router.register_model(
                model_id=NEW_MODEL,
                cost_usd=registry[NEW_MODEL]["input_cost_per_m"],
                speed="balanced"
            )
        
        # Pass total_steps to enable proper alpha decay in experts
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        if is_target:
            history_target.append(reward)
        else:
            history_target.append(np.nan)
        
        if router.corralling_router:
            expert_weights_history.append({
                'conservative': router.corralling_router.weights[0],
                'adaptive': router.corralling_router.weights[1]
            })
    
    return {
        "history_target": history_target,
        "expert_weights": expert_weights_history
    }

# ============================================================================
# STATISTICAL ANALYSIS MODULE
# ============================================================================
def analyze_statistical_significance(histories: np.ndarray):
    """
    Perform hypothesis testing on the Production Router's performance vs Baseline.
    Since we don't have a parallel "Baseline" history in this specific script 
    (it runs the prod router), we compare Pre-Release vs Post-Release performance
    to prove the "Step Change" is significant.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 STATISTICAL HYPOTHESIS TESTING (N=%d)", N_TRIALS)
    logger.info("="*80)
    
    # 1. Define Windows
    pre_window = slice(RELEASE_STEP - 200, RELEASE_STEP)
    post_window = slice(RELEASE_STEP, RELEASE_STEP + 200)
    
    # 2. Extract Means per Trial (Paired Samples)
    pre_means = np.nanmean(histories[:, pre_window], axis=1)
    post_means = np.nanmean(histories[:, post_window], axis=1)
    
    logger.info(f"  Pre-Release Mean:  {np.mean(pre_means):.4f} ± {np.std(pre_means):.4f}")
    logger.info(f"  Post-Release Mean: {np.mean(post_means):.4f} ± {np.std(post_means):.4f}")
    
    # 3. Paired Tests
    diffs = post_means - pre_means
    mean_diff = np.mean(diffs)
    
    # t-test
    t_stat, p_val_t = stats.ttest_rel(post_means, pre_means)
    
    # Wilcoxon
    try:
        w_stat, p_val_w = stats.wilcoxon(post_means, pre_means)
    except ValueError:
        p_val_w = 1.0
        
    # Cohen's d
    sd_diff = np.std(diffs, ddof=1)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
    
    print("\n" + "-"*95)
    print(f"{'Comparison':<30} | {'Diff':<8} | {'t-stat':<8} | {'p-value (t)':<12} | {'Wilcoxon p':<12} | {'Cohen d':<8}")
    print("-"*95)
    
    sig = "**" if p_val_t < 0.001 else "*" if p_val_t < 0.05 else ""
    print(f"{'Post vs Pre Release':<30} | {mean_diff:+.4f}   | {t_stat:+.2f}    | {p_val_t:.2e} {sig:<3} | {p_val_w:.2e}     | {cohens_d:.2f}")
    print("-"*95)
    logger.info("="*80 + "\n")

# ============================================================================
# MULTI-TRIAL RUNNER
# ============================================================================
def run_rigorous_experiment():
    data = load_data()
    if not data: return None
    
    complex_indices = identify_complex_subset(data)
    if not complex_indices: return None
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    all_histories = []
    all_weights = []
    
    logger.info(f"\n🔬 Running {N_TRIALS} trials on COMPLEX subset...")
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        result = run_trial(seed, data, encoder, pca, complex_indices)
        all_histories.append(result["history_target"])
        all_weights.append(result["expert_weights"])
    
    # Package results
    results = {
        "histories": np.array(all_histories),
        "expert_weights": all_weights
    }
    
    # [NEW] Statistical Analysis
    analyze_statistical_significance(results["histories"])
    
    return results

# ============================================================================
# PLOTTING
# ============================================================================
def plot_results(results):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    histories = results["histories"]
    means, cis = [], []
    
    for t in range(histories.shape[1]):
        start = max(0, t - WINDOW_SIZE)
        window = histories[:, start:t+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 10:
            means.append(np.mean(valid))
            cis.append(stats.sem(valid) * stats.t.ppf((1 + CONFIDENCE_LEVEL)/2., len(valid)-1))
        else:
            means.append(np.nan)
            cis.append(np.nan)
            
    means = np.array(means)
    cis = np.array(cis)
    x = np.arange(len(means))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    mask = x > 50 
    
    # Top Plot: Reward
    ax1.plot(x[mask], means[mask], color="#2ecc71", linewidth=2.5, 
             label="Production Router (Heterogeneous Experts)")
    ax1.fill_between(x[mask], (means-cis)[mask], (means+cis)[mask], color="#2ecc71", alpha=0.2,
                      label=f"{int(CONFIDENCE_LEVEL*100)}% Confidence Interval")
    
    ax1.axvline(x=RELEASE_STEP, color='black', alpha=0.5, linewidth=2, linestyle='--',
                label="GPT-5.1 Release")
    
    # [KDD PIVOT]: New Title and Annotation
    ax1.set_title(f"Zero-Shot Readiness: Accelerated Model Discovery\n(N={N_TRIALS}, Complex Tasks Subset)", 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel("Routing Steps", fontsize=13)
    ax1.set_ylabel("Average Reward (Complex Tasks)", fontsize=13)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # [CRITICAL UPDATE]: Honest Annotation
    ax1.text(0.02, 0.95, 
             "Methodology Note:\n"
             "Filtered by INPUT features (keywords)\n"
             "to avoid circular reasoning.\n"
             "Demonstrates instant adoption of\n"
             "dominant model on hard tasks.", 
             transform=ax1.transAxes, 
             fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round', edgecolor='gray'))

    # Bottom Plot: Weights
    if results["expert_weights"]:
        conservative_weights = []
        adaptive_weights = []
        for trial_weights in results["expert_weights"]:
            conservative_weights.append([w['conservative'] for w in trial_weights])
            adaptive_weights.append([w['adaptive'] for w in trial_weights])
        
        con_mean = np.mean(conservative_weights, axis=0)
        ada_mean = np.mean(adaptive_weights, axis=0)
        x_w = np.arange(len(con_mean))
        
        ax2.plot(x_w, con_mean, color='#3498db', linewidth=2, label='Expert 1 (Conservative)')
        ax2.plot(x_w, ada_mean, color='#e74c3c', linewidth=2, label='Expert 2 (Adaptive)')
        
        ax2.axvline(x=RELEASE_STEP, color='black', alpha=0.3, linestyle='--')
        ax2.set_title("Meta-Learner Dynamics", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Routing Steps", fontsize=12)
        ax2.set_ylabel("Expert Weight", fontsize=12)
        ax2.legend(loc='center right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    out_file = output_dir / "figure6_adaptive_efficiency.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved plot to {out_file}")

if __name__ == "__main__":
    results = run_rigorous_experiment()
    if results:
        plot_results(results)
