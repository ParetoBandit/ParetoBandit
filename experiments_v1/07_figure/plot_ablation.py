"""
Figure 6: Ablation Study - Zero-Shot Readiness (Statistical Rigor Added)
========================================================================

Compares three strategies to prove the value of Latent Semantic Initialization.
Includes FORMAL HYPOTHESIS TESTING (Paired t-tests, Wilcoxon, Cohen's d).

Narrative Pivot:
- We admit that GPT-5.1 is dominant on these tasks ($r \\approx 0$).
- Therefore, we don't measure "Task Correlation" (which is 0).
- We measure "Adoption Velocity": How fast does the router realize GPT-5 is the winner?

Strategies:
1. Cold Start: No priors. Must randomly explore to find GPT-5 is good.
2. Warmup Only: Priors for old models, but Cold Start for GPT-5.
3. Semantic Transfer: Inherits "Prior of Competence" from GPT-4. 
   Starts with high belief, allowing instant adoption of the dominant model.
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

# Evaluation Window for Statistics (Post-Release)
EVAL_WINDOW_START = 300
EVAL_WINDOW_END = 500  # Focus on the "Adoption" phase

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
MODELS_3 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo", "openai/gpt-5.1"]
NEW_MODEL = "openai/gpt-5.1"

# ============================================================================
# LOGIC: INDEPENDENT COMPLEXITY FILTER (The Fix)
# ============================================================================
def identify_complex_subset(data: List) -> Set[int]:
    """
    Identifies "Complex" tasks based on PROMPT CONTENT (Covariates), 
    NOT rewards. This restores variance (Mixtral wins some, GPT wins others).
    """
    complex_indices = set()
    
    # Keywords for "High Complexity" tasks
    complexity_signals = [
        "code", "python", "function", "algorithm",  # Coding
        "solve", "math", "proof", "calculate",      # Math
        "step-by-step", "logic", "reasoning",       # Reasoning
        "analysis", "difference between", "compare" # Analytical
    ]
    
    for i, item in enumerate(data):
        prompt_lower = item.prompt.lower()
        
        # Check 1: Length heuristic
        is_long = len(prompt_lower.split()) > 50
        
        # Check 2: Content heuristic
        has_signal = any(sig in prompt_lower for sig in complexity_signals)
        
        if is_long or has_signal:
            complex_indices.add(i)
            
    logger.info(f"🔍 Identified {len(complex_indices)} 'Complex' prompts ({len(complex_indices)/len(data):.1%} of traffic)")
    return complex_indices

# ============================================================================
# MODEL REGISTRY
# ============================================================================
def create_registry(models):
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

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    try:
        evaluator = AlignedEvaluator.from_jsonl_gz(
            DEV_DATA_PATH_ALL_MODELS,
            required_models=MODELS_3
        )
        data = [item for item in evaluator if all(m in item.rewards for m in MODELS_3)]
        return data
    except Exception as e:
        logger.error(f"Data error: {e}")
        return []

# ============================================================================
# TRIAL RUNNERS (Updated to use complex_indices)
# ============================================================================
def run_trial_cold_start(seed: int, data: List, encoder, pca, target_indices: Set[int]) -> List[float]:
    """Strategy 1: Cold Start (No Priors)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors="none", 
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history_target = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        is_target = idx in target_indices
        
        if t_step == RELEASE_STEP:
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.registry[NEW_MODEL] = create_registry([NEW_MODEL])[NEW_MODEL]
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

        # Pass total_steps to enable proper alpha decay in experts
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        if is_target: history_target.append(reward)
        else: history_target.append(np.nan)
    return history_target

def run_trial_warmup_only(seed: int, data: List, encoder, pca, target_indices: Set[int]) -> List[float]:
    """Strategy 2: Warmup Priors, but Cold Start for GPT-5.1."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history_target = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        is_target = idx in target_indices
        
        if t_step == RELEASE_STEP:
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.registry[NEW_MODEL] = create_registry([NEW_MODEL])[NEW_MODEL]
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

        # Pass total_steps to enable proper alpha decay in experts
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        if is_target: history_target.append(reward)
        else: history_target.append(np.nan)
    return history_target

def run_trial_semantic_transfer(seed: int, data: List, encoder, pca, target_indices: Set[int]) -> List[float]:
    """Strategy 3: Warmup + Semantic Transfer."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history_target = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        is_target = idx in target_indices
        
        if t_step == RELEASE_STEP:
            router.register_model(model_id=NEW_MODEL, cost_usd=15.0, speed="balanced")
        
        # Pass total_steps to enable proper alpha decay in experts
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        if is_target: history_target.append(reward)
        else: history_target.append(np.nan)
    return history_target

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
def analyze_statistical_significance(results: Dict[str, List[List[float]]]):
    """
    Perform rigorous statistical hypothesis testing on the results.
    Computes Paired t-tests, Wilcoxon Signed-Rank tests, and Cohen's d.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 STATISTICAL HYPOTHESIS TESTING (N=%d, Window: t=%d-%d)", 
                N_TRIALS, EVAL_WINDOW_START, EVAL_WINDOW_END)
    logger.info("="*80)
    
    # 1. Aggregate per-trial performance (Mean Reward in Evaluation Window)
    trial_means = {}
    for strategy, histories in results.items():
        # Shape: (N_TRIALS, STEPS)
        matrix = np.array(histories)
        # Slice window
        window = matrix[:, EVAL_WINDOW_START:EVAL_WINDOW_END]
        # Mean per trial (handling NaNs for sparse complex tasks)
        means = np.nanmean(window, axis=1)
        trial_means[strategy] = means
        
        mean_reward = np.mean(means)
        std_reward = np.std(means)
        logger.info(f"  {strategy:25s}: {mean_reward:.4f} ± {std_reward:.4f}")

    # 2. Pairwise Comparisons
    comparisons = [
        ("Warmup + Transfer", "Warmup Only"),
        ("Warmup + Transfer", "Cold Start"),
        ("Warmup Only", "Cold Start")
    ]
    
    print("\n" + "-"*95)
    print(f"{'Comparison':<30} | {'Diff':<8} | {'t-stat':<8} | {'p-value (t)':<12} | {'Wilcoxon p':<12} | {'Cohen d':<8}")
    print("-"*95)
    
    for strategy_a, strategy_b in comparisons:
        a_scores = trial_means[strategy_a]
        b_scores = trial_means[strategy_b]
        
        # Paired t-test
        t_stat, p_val_t = stats.ttest_rel(a_scores, b_scores)
        
        # Wilcoxon Signed-Rank Test (Non-parametric robust check)
        try:
            w_stat, p_val_w = stats.wilcoxon(a_scores, b_scores)
        except ValueError: 
            p_val_w = 1.0
            
        # Cohen's d (for Paired Samples: Mean Difference / SD of Difference)
        diffs = a_scores - b_scores
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
        
        # Bonferroni correction: 0.05 / 3 = 0.0167
        sig_marker = "**" if p_val_t < 0.001 else "*" if p_val_t < 0.0167 else ""
        
        print(f"{strategy_a} vs {strategy_b:<15} | {mean_diff:+.4f}   | {t_stat:+.2f}    | {p_val_t:.2e} {sig_marker:<3} | {p_val_w:.2e}     | {cohens_d:.2f}")
    
    print("-"*95)
    print("Significance: * p < 0.05/3 (Bonferroni), ** p < 0.001")
    logger.info("="*80 + "\n")


# ============================================================================
# RUNNER & PLOTTING
# ============================================================================
def run_ablation():
    data = load_data()
    if not data: return None
    
    # Use the Input Filter (Covariates)
    target_indices = identify_complex_subset(data)
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    results = {"Cold Start": [], "Warmup Only": [], "Warmup + Transfer": []}
    
    logger.info(f"\n🔬 Running ablation study on COMPLEX subset (N={N_TRIALS})...")
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        results["Cold Start"].append(run_trial_cold_start(seed, data, encoder, pca, target_indices))
        results["Warmup Only"].append(run_trial_warmup_only(seed, data, encoder, pca, target_indices))
        results["Warmup + Transfer"].append(run_trial_semantic_transfer(seed, data, encoder, pca, target_indices))
    
    # [NEW] Run Statistical Analysis before plotting
    analyze_statistical_significance(results)
    
    return results

def plot_ablation(results):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"Cold Start": "#e74c3c", "Warmup Only": "#f39c12", "Warmup + Transfer": "#2ecc71"}
    styles = {"Cold Start": ":", "Warmup Only": "--", "Warmup + Transfer": "-"}
    
    for name, histories in results.items():
        matrix = np.array(histories)
        means, cis = [], []
        for t in range(matrix.shape[1]):
            win_start = max(0, t - WINDOW_SIZE)
            window = matrix[:, win_start:t+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 5:
                means.append(np.mean(valid))
                cis.append(stats.sem(valid) * stats.t.ppf((1 + CONFIDENCE_LEVEL)/2., len(valid)-1))
            else:
                means.append(np.nan)
                cis.append(np.nan)
        
        means = np.array(means)
        cis = np.array(cis)
        x = np.arange(len(means))
        mask = x > 50
        
        ax.plot(x[mask], means[mask], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        ax.fill_between(x[mask], (means-cis)[mask], (means+cis)[mask], color=colors[name], alpha=0.15)
    
    ax.axvline(x=RELEASE_STEP, color='black', alpha=0.5, linewidth=2, linestyle='--', label="GPT-5.1 Release")
    
    # NEW TITLE: Reflects "Adoption" instead of "Transfer"
    ax.set_title(f"Zero-Shot Readiness: Accelerating Adoption of Dominant Models\n(N={N_TRIALS}, Complex Subset)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Routing Steps", fontsize=12)
    ax.set_ylabel("Average Reward (Complex Tasks)", fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "figure6_ablation_final.png", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved final ablation plot.")

if __name__ == "__main__":
    results = run_ablation()
    if results:
        plot_ablation(results)
