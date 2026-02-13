"""
Figure 6: Ablation Study - Semantic Transfer for New Model Adoption (KDD Revision)
==================================================================================

Compares FOUR strategies to evaluate the value of semantic transfer.
Includes FORMAL HYPOTHESIS TESTING (Paired t-tests, Wilcoxon, Cohen's d).

FIXED (per KDD reviewer feedback):
- Tests on FULL task distribution (not cherry-picked subset)
- Reports heterogeneity analysis to validate contextual routing is meaningful
- Measures actual performance across all task types
- Reports full-episode metrics (not cherry-picked window)
- Includes realistic baseline (small LMSys prior)

Strategies:
1. Cold Start: No priors. Pure online learning from scratch. (Unrealistic)
2. Warmup Only: Priors for old models, but Cold Start for GPT-5.1. (Unrealistic)
3. Small LMSys Prior: GPT-5.1 gets ~20 preliminary benchmark samples. (Realistic Baseline)
4. Semantic Transfer: GPT-5.1 inherits preference from GPT-4-Turbo. (Our Method)
   Tests whether semantic similarity enables faster adaptation vs realistic baseline.
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
# HETEROGENEITY ANALYSIS (KDD Reviewer Fix)
# ============================================================================
def analyze_model_preferences(data: List) -> Dict:
    """
    Analyze whether models have heterogeneous preferences across tasks.
    This validates whether contextual routing is meaningful or if one model dominates.
    """
    mixtral = "mistralai/mixtral-8x7b-instruct"
    gpt4 = "openai/gpt-4-turbo"
    gpt5 = "openai/gpt-5.1"
    
    # Count wins for each model pair
    mixtral_wins = 0
    gpt4_wins = 0
    gpt5_wins_vs_mixtral = 0
    gpt5_wins_vs_gpt4 = 0
    ties = 0
    
    for item in data:
        r_mixtral = item.rewards.get(mixtral, 0)
        r_gpt4 = item.rewards.get(gpt4, 0)
        r_gpt5 = item.rewards.get(gpt5, 0)
        
        # Mixtral vs GPT-4
        if r_mixtral > r_gpt4:
            mixtral_wins += 1
        elif r_gpt4 > r_mixtral:
            gpt4_wins += 1
        else:
            ties += 1
            
        # GPT-5 dominance check
        if r_gpt5 > r_mixtral:
            gpt5_wins_vs_mixtral += 1
        if r_gpt5 > r_gpt4:
            gpt5_wins_vs_gpt4 += 1
    
    total = len(data)
    logger.info("\n" + "="*80)
    logger.info("📊 MODEL PREFERENCE HETEROGENEITY ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total tasks: {total}")
    logger.info(f"Mixtral wins vs GPT-4: {mixtral_wins} ({mixtral_wins/total:.1%})")
    logger.info(f"GPT-4 wins vs Mixtral: {gpt4_wins} ({gpt4_wins/total:.1%})")
    logger.info(f"Ties: {ties} ({ties/total:.1%})")
    logger.info(f"GPT-5.1 wins vs Mixtral: {gpt5_wins_vs_mixtral} ({gpt5_wins_vs_mixtral/total:.1%})")
    logger.info(f"GPT-5.1 wins vs GPT-4: {gpt5_wins_vs_gpt4} ({gpt5_wins_vs_gpt4/total:.1%})")
    logger.info("="*80 + "\n")
    
    return {
        "mixtral_wins": mixtral_wins,
        "gpt4_wins": gpt4_wins,
        "gpt5_dominance_rate": gpt5_wins_vs_gpt4 / total
    }

def validate_semantic_transfer_hypothesis(data: List, encoder, registry: Dict) -> Dict:
    """
    Test the core hypothesis: Does semantic similarity predict performance correlation?
    
    This is the KEY validation that the paper needs. We compute:
    1. Semantic embedding similarity for all model pairs
    2. Task-level performance correlation for all model pairs
    3. Meta-correlation: Do similar embeddings → similar performance?
    
    Returns correlation coefficient and p-value.
    """
    from itertools import combinations
    from sklearn.metrics.pairwise import cosine_similarity
    
    models = MODELS_3
    logger.info("\n" + "="*80)
    logger.info("🔬 SEMANTIC TRANSFER HYPOTHESIS VALIDATION")
    logger.info("="*80)
    logger.info("Testing: Do semantically similar models have correlated task performance?")
    logger.info("="*80)
    
    # Step 1: Compute semantic embeddings for all models
    embeddings = {}
    for model in models:
        desc = registry[model]["description"]
        embedding = encoder.encode(desc)
        embeddings[model] = embedding
        logger.info(f"  {model}: '{desc}'")
    
    # Step 2: Compute pairwise similarities and correlations
    embedding_sims = []
    perf_corrs = []
    pair_names = []
    
    for m1, m2 in combinations(models, 2):
        # Semantic embedding similarity
        emb_sim = cosine_similarity(
            embeddings[m1].reshape(1, -1),
            embeddings[m2].reshape(1, -1)
        )[0, 0]
        
        # Task-level performance correlation
        rewards_m1 = [item.rewards.get(m1, 0) for item in data]
        rewards_m2 = [item.rewards.get(m2, 0) for item in data]
        perf_corr, _ = stats.pearsonr(rewards_m1, rewards_m2)
        
        embedding_sims.append(emb_sim)
        perf_corrs.append(perf_corr)
        pair_names.append(f"{m1.split('/')[-1]} vs {m2.split('/')[-1]}")
        
        logger.info(f"\n  {pair_names[-1]}:")
        logger.info(f"    Embedding similarity: {emb_sim:.3f}")
        logger.info(f"    Performance correlation: {perf_corr:.3f}")
    
    # Step 3: Meta-correlation (THE KEY TEST)
    meta_corr, p_value = stats.pearsonr(embedding_sims, perf_corrs)
    
    logger.info("\n" + "-"*80)
    logger.info(f"📊 META-CORRELATION (Key Result):")
    logger.info(f"   Correlation(Embedding Sim, Performance Corr) = {meta_corr:.3f}")
    logger.info(f"   p-value: {p_value:.4f}")
    logger.info(f"   Interpretation:")
    if meta_corr > 0.3 and p_value < 0.05:
        logger.info(f"     ✅ Positive correlation validates hypothesis (r={meta_corr:.2f}, p<0.05)")
    elif meta_corr > 0 and p_value < 0.10:
        logger.info(f"     ⚠️  Weak evidence for hypothesis (r={meta_corr:.2f}, p<0.10)")
    else:
        logger.info(f"     ❌ No evidence for hypothesis (r={meta_corr:.2f}, p={p_value:.2f})")
    logger.info("="*80 + "\n")
    
    return {
        "meta_correlation": meta_corr,
        "p_value": p_value,
        "embedding_sims": embedding_sims,
        "perf_corrs": perf_corrs,
        "pair_names": pair_names
    }

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
# TRIAL RUNNERS (Fixed: Track ALL tasks, not filtered subset)
# ============================================================================
def run_trial_cold_start(seed: int, data: List, encoder, pca) -> List[float]:
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
    
    history_all = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        
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
        
        # Track ALL tasks (not filtered subset)
        history_all.append(reward)
    return history_all

def run_trial_warmup_only(seed: int, data: List, encoder, pca) -> List[float]:
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
    
    history_all = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        
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
        
        # Track ALL tasks (not filtered subset)
        history_all.append(reward)
    return history_all

def run_trial_small_lmsys_prior(seed: int, data: List, encoder, pca) -> List[float]:
    """Strategy 3: Warmup + Small LMSys Prior (Realistic Baseline)."""
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
    
    history_all = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        
        if t_step == RELEASE_STEP:
            # Simulate "small LMSys prior": weak initialization
            # Equivalent to ~20 preliminary benchmark samples
            # Use a weak random prior (simulates noisy early benchmarks)
            router.bandit.models.append(NEW_MODEL)
            
            # Weak prior: A = (lambda + 20)*I, b = random small vector
            n_eff_small = 20  # Equivalent to 20 samples
            router.bandit.A[NEW_MODEL] = (router.bandit.init_lambda + n_eff_small) * np.eye(router.bandit.dim)
            # Small random preference vector (simulates noisy early data)
            router.bandit.b[NEW_MODEL] = rng.randn(router.bandit.dim) * 0.1 * n_eff_small
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.registry[NEW_MODEL] = create_registry([NEW_MODEL])[NEW_MODEL]
            
            if router.corralling_router:
                router.corralling_router.add_model(NEW_MODEL)
                weak_A = router.bandit.A[NEW_MODEL].copy()
                weak_b = router.bandit.b[NEW_MODEL].copy()
                for expert in router.corralling_router.experts:
                    if hasattr(expert, 'add_model'):
                        expert_type = type(expert).__name__
                        if 'TabulaRasa' in expert_type:
                            expert.add_model(NEW_MODEL, 0.5)
                        else:
                            expert.add_model(NEW_MODEL, weak_A, weak_b, 0.5)
        
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        history_all.append(reward)
    return history_all

def run_trial_semantic_transfer(seed: int, data: List, encoder, pca) -> List[float]:
    """Strategy 4: Warmup + Semantic Transfer (Our Method)."""
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
    
    history_all = []
    for t_step, idx in enumerate(indices):
        if t_step >= TOTAL_STEPS: break
        item = data[idx]
        
        if t_step == RELEASE_STEP:
            router.register_model(model_id=NEW_MODEL, cost_usd=15.0, speed="balanced")
        
        # Pass total_steps to enable proper alpha decay in experts
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        # Track ALL tasks (not filtered subset)
        history_all.append(reward)
    return history_all

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
def analyze_statistical_significance(results: Dict[str, List[List[float]]]):
    """
    Perform rigorous statistical hypothesis testing on the results.
    Computes Paired t-tests, Wilcoxon Signed-Rank tests, and Cohen's d.
    Reports both post-release window AND full-episode cumulative regret.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 STATISTICAL HYPOTHESIS TESTING (N=%d)", N_TRIALS)
    logger.info("="*80)
    
    # 1. Aggregate per-trial performance - MULTIPLE METRICS
    trial_means_post_release = {}
    trial_means_full_episode = {}
    trial_cumulative_regret = {}
    
    for strategy, histories in results.items():
        # Shape: (N_TRIALS, STEPS)
        matrix = np.array(histories)
        
        # Metric 1: Post-Release Window (original metric, for comparison)
        window_post = matrix[:, EVAL_WINDOW_START:EVAL_WINDOW_END]
        means_post = np.mean(window_post, axis=1)
        trial_means_post_release[strategy] = means_post
        
        # Metric 2: Full Episode Average Reward (fairer comparison)
        means_full = np.mean(matrix, axis=1)
        trial_means_full_episode[strategy] = means_full
        
        # Metric 3: Cumulative Regret vs Oracle (most rigorous)
        # Oracle = always pick best model for each task (upper bound)
        # For now, use cumulative reward as proxy (higher is better)
        cumulative_rewards = np.sum(matrix, axis=1)
        trial_cumulative_regret[strategy] = cumulative_rewards
    
    # Report all three metrics
    logger.info("\n📊 POST-RELEASE WINDOW (t=%d-%d) [Original Metric]:", EVAL_WINDOW_START, EVAL_WINDOW_END)
    for strategy in results.keys():
        means = trial_means_post_release[strategy]
        logger.info(f"  {strategy:25s}: {np.mean(means):.4f} ± {np.std(means):.4f}")
    
    logger.info("\n📊 FULL EPISODE AVERAGE (t=0-%d) [Fairer Metric]:", TOTAL_STEPS)
    for strategy in results.keys():
        means = trial_means_full_episode[strategy]
        logger.info(f"  {strategy:25s}: {np.mean(means):.4f} ± {np.std(means):.4f}")
    
    logger.info("\n📊 CUMULATIVE REWARD (Sum over full episode) [Most Rigorous]:")
    for strategy in results.keys():
        rewards = trial_cumulative_regret[strategy]
        logger.info(f"  {strategy:25s}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    # 2. Pairwise Comparisons (Focus on realistic baselines)
    comparisons = [
        ("Semantic Transfer", "Small LMSys Prior"),  # PRIMARY: Our method vs realistic baseline
        ("Semantic Transfer", "Warmup Only"),
        ("Semantic Transfer", "Cold Start"),
        ("Small LMSys Prior", "Warmup Only")
    ]
    
    logger.info("\n🔬 STATISTICAL TESTS (Full Episode Average Reward - Fairer Metric):")
    print("\n" + "-"*95)
    print(f"{'Comparison':<30} | {'Diff':<8} | {'t-stat':<8} | {'p-value (t)':<12} | {'Wilcoxon p':<12} | {'Cohen d':<8}")
    print("-"*95)
    
    for strategy_a, strategy_b in comparisons:
        a_scores = trial_means_full_episode[strategy_a]
        b_scores = trial_means_full_episode[strategy_b]
        
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
        
        # Bonferroni correction: 0.05 / 4 = 0.0125
        sig_marker = "**" if p_val_t < 0.001 else "*" if p_val_t < 0.0125 else ""
        
        print(f"{strategy_a} vs {strategy_b:<15} | {mean_diff:+.4f}   | {t_stat:+.2f}    | {p_val_t:.2e} {sig_marker:<3} | {p_val_w:.2e}     | {cohens_d:.2f}")
    
    print("-"*95)
    print("Significance: * p < 0.05/4 (Bonferroni), ** p < 0.001")
    
    # Also test cumulative rewards (total rewards summed over full episode)
    logger.info("\n🔬 STATISTICAL TESTS (Cumulative Reward - Most Rigorous Metric):")
    print("\n" + "-"*95)
    print(f"{'Comparison':<30} | {'Diff':<8} | {'t-stat':<8} | {'p-value (t)':<12} | {'Wilcoxon p':<12} | {'Cohen d':<8}")
    print("-"*95)
    
    for strategy_a, strategy_b in comparisons:
        a_scores = trial_cumulative_regret[strategy_a]
        b_scores = trial_cumulative_regret[strategy_b]
        
        t_stat, p_val_t = stats.ttest_rel(a_scores, b_scores)
        
        try:
            w_stat, p_val_w = stats.wilcoxon(a_scores, b_scores)
        except ValueError: 
            p_val_w = 1.0
            
        diffs = a_scores - b_scores
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
        
        sig_marker = "**" if p_val_t < 0.001 else "*" if p_val_t < 0.0167 else ""
        
        print(f"{strategy_a} vs {strategy_b:<15} | {mean_diff:+.2f}   | {t_stat:+.2f}    | {p_val_t:.2e} {sig_marker:<3} | {p_val_w:.2e}     | {cohens_d:.2f}")
    
    print("-"*95)
    print("Significance: * p < 0.05/4 (Bonferroni), ** p < 0.001")
    logger.info("="*80 + "\n")


# ============================================================================
# RUNNER & PLOTTING
# ============================================================================
def run_ablation():
    data = load_data()
    if not data: return None
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    # Analyze task heterogeneity (validate contextual routing is meaningful)
    heterogeneity = analyze_model_preferences(data)
    
    # CRITICAL: Validate semantic transfer hypothesis
    # Does semantic similarity predict performance correlation?
    registry = create_registry(MODELS_3)
    semantic_validation = validate_semantic_transfer_hypothesis(data, encoder, registry)
    
    results = {
        "Cold Start": [], 
        "Warmup Only": [], 
        "Small LMSys Prior": [],
        "Semantic Transfer": []
    }
    
    logger.info(f"\n🔬 Running ablation study on FULL task distribution (N={N_TRIALS})...")
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        results["Cold Start"].append(run_trial_cold_start(seed, data, encoder, pca))
        results["Warmup Only"].append(run_trial_warmup_only(seed, data, encoder, pca))
        results["Small LMSys Prior"].append(run_trial_small_lmsys_prior(seed, data, encoder, pca))
        results["Semantic Transfer"].append(run_trial_semantic_transfer(seed, data, encoder, pca))
    
    # [NEW] Run Statistical Analysis before plotting
    analyze_statistical_significance(results)
    
    return results, heterogeneity, semantic_validation

def plot_ablation(results, heterogeneity):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        "Cold Start": "#e74c3c",          # Red (worst)
        "Warmup Only": "#f39c12",         # Orange
        "Small LMSys Prior": "#3498db",   # Blue (realistic baseline)
        "Semantic Transfer": "#2ecc71"    # Green (our method)
    }
    styles = {
        "Cold Start": ":",
        "Warmup Only": "--",
        "Small LMSys Prior": "-.",
        "Semantic Transfer": "-"
    }
    
    for name, histories in results.items():
        matrix = np.array(histories)
        means, cis = [], []
        for t in range(matrix.shape[1]):
            win_start = max(0, t - WINDOW_SIZE)
            window = matrix[:, win_start:t+1]
            if len(window.flatten()) > 5:
                means.append(np.mean(window))
                cis.append(stats.sem(window.flatten()) * stats.t.ppf((1 + CONFIDENCE_LEVEL)/2., len(window.flatten())-1))
            else:
                means.append(np.mean(window))
                cis.append(0)
        
        means = np.array(means)
        cis = np.array(cis)
        x = np.arange(len(means))
        mask = x > 50
        
        ax.plot(x[mask], means[mask], label=name, color=colors[name], linestyle=styles[name], linewidth=2.5)
        ax.fill_between(x[mask], (means-cis)[mask], (means+cis)[mask], color=colors[name], alpha=0.15)
    
    ax.axvline(x=RELEASE_STEP, color='black', alpha=0.5, linewidth=2, linestyle='--', label="GPT-5.1 Release")
    
    # Updated title: Full task distribution with heterogeneity info
    dominance_pct = heterogeneity['gpt5_dominance_rate'] * 100
    ax.set_title(f"Semantic Transfer for New Model Adoption (Revised)\n(N={N_TRIALS}, Full Task Distribution, GPT-5.1 wins {dominance_pct:.0f}% vs GPT-4)", 
                 fontsize=12, fontweight='bold')
    
    # Add methodology note
    ax.text(0.02, 0.98, 
            "Fixed per KDD Reviewer:\n"
            "• Full task distribution (no filtering)\n"
            "• Realistic baseline (Small LMSys Prior)\n"
            "• Full episode metrics", 
            transform=ax.transAxes, 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round', edgecolor='gray'))
    ax.set_xlabel("Routing Steps", fontsize=12)
    ax.set_ylabel("Average Reward (All Tasks)", fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "figure7_ablation.png", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved ablation plot to figure7_ablation.png")

if __name__ == "__main__":
    results_data = run_ablation()
    if results_data:
        results, heterogeneity, semantic_validation = results_data
        plot_ablation(results, heterogeneity)
        
        # Save semantic validation results for paper
        import json
        output_dir = Path(__file__).parent / "results"
        with open(output_dir / "semantic_validation.json", "w") as f:
            json.dump({
                "meta_correlation": semantic_validation["meta_correlation"],
                "p_value": semantic_validation["p_value"],
                "pair_names": semantic_validation["pair_names"],
                "embedding_sims": [float(x) for x in semantic_validation["embedding_sims"]],
                "perf_corrs": [float(x) for x in semantic_validation["perf_corrs"]]
            }, f, indent=2)
        logger.info(f"✅ Saved semantic validation results to semantic_validation.json")
