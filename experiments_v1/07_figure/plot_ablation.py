"""
Figure 6: Ablation Study - Semantic Transfer Value
==================================================

Compares three strategies to prove the value of semantic transfer.
ALL strategies use the same model availability timeline:
- t=0-300: Mixtral + GPT-4-turbo only
- t=300: GPT-5.1 added (release event)

Strategies differ in HOW they initialize:

1. Cold Start: No warmup priors, GPT-5.1 added cold at t=300
   - All models learn from scratch (identity matrix at t=0)
   - GPT-5.1 starts with identity matrix (no transfer)
   
2. Warmup Only: Warmup priors, GPT-5.1 cold at t=300
   - Mixtral + GPT-4-turbo benefit from 80k battles
   - GPT-5.1 starts with identity matrix (no transfer)

3. Warmup + Semantic Transfer: Warmup priors, GPT-5.1 transfers at t=300
   - Mixtral + GPT-4-turbo benefit from 80k battles  
   - GPT-5.1 inherits theta from GPT-4-turbo (semantic match)

This proves:
- Value of warmup priors: (2 vs 1)
- Value of semantic transfer: (3 vs 2)
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

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
MODELS_3 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo", "openai/gpt-5.1"]
NEW_MODEL = "openai/gpt-5.1"

# ============================================================================
# MODEL REGISTRY
# ============================================================================
def create_registry(models):
    """Create registry for specified models."""
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
            "description": "Efficient sparse mixture-of-experts model."
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
            "description": "High-intelligence flagship model."
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0,
            "output_cost_per_m": 45.0,
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
        logger.info(f"✅ Loaded {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Data error: {e}")
        return []

# ============================================================================
# TRIAL RUNNERS
# ============================================================================
def run_trial_cold_start(seed: int, data: List, encoder, pca) -> List[float]:
    """Strategy 1: 2 models from scratch, GPT-5.1 cold at t=300."""
    rng = np.random.RandomState(seed)
    trial_data = data.copy()
    rng.shuffle(trial_data)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors="none",  # No warmup - cold start
        use_corralling=True,  # Heterogeneous experts: α 1.0→0.01 (Conservative) and α 2.0 (Adaptive)
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history = []
    for t, item in enumerate(trial_data):
        if t >= TOTAL_STEPS:
            break
        
        # Add GPT-5.1 cold at t=300 (same as Warmup Only)
        if t == RELEASE_STEP:
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            
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
        
        selected, _ = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)  # Fixed: (model_id, context, reward)
        history.append(reward)
    
    return history

def run_trial_warmup_only(seed: int, data: List, encoder, pca) -> List[float]:
    """Strategy 2: 2 models with warmup, GPT-5.1 cold at t=300."""
    rng = np.random.RandomState(seed)
    trial_data = data.copy()
    rng.shuffle(trial_data)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,  # Heterogeneous experts: α 1.0→0.01 (Conservative) and α 2.0 (Adaptive)
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history = []
    for t, item in enumerate(trial_data):
        if t >= TOTAL_STEPS:
            break
        
        # Add GPT-5.1 WITHOUT semantic transfer (cold start)
        if t == RELEASE_STEP:
            # Manually add to bandit with identity matrix (bypass semantic transfer)
            router.bandit.models.append(NEW_MODEL)
            router.bandit.A[NEW_MODEL] = router.bandit.init_lambda * np.eye(router.bandit.dim)
            router.bandit.b[NEW_MODEL] = np.zeros(router.bandit.dim)
            router.bandit.A_inv[NEW_MODEL] = np.linalg.inv(router.bandit.A[NEW_MODEL])
            router.bandit.last_update[NEW_MODEL] = router.bandit.t
            
            # Add to registry
            router.registry[NEW_MODEL] = create_registry([NEW_MODEL])[NEW_MODEL]
            
            # Propagate to Corralling experts (cold start)
            if router.corralling_router:
                router.corralling_router.add_model(NEW_MODEL)
                cold_A = router.bandit.init_lambda * np.eye(router.bandit.dim)
                cold_b = np.zeros(router.bandit.dim)
                
                # Update each expert
                for expert in router.corralling_router.experts:
                    if hasattr(expert, 'add_model'):
                        # Check type by inspecting class name
                        expert_type = type(expert).__name__
                        if 'TabulaRasa' in expert_type:
                            expert.add_model(NEW_MODEL, 0.5)
                        else:  # CostAwareLinUCBRouter
                            expert.add_model(NEW_MODEL, cold_A, cold_b, 0.5)
        
        selected, _ = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)  # Fixed: (model_id, context, reward)
        history.append(reward)
    
    return history

def run_trial_semantic_transfer(seed: int, data: List, encoder, pca) -> List[float]:
    """Strategy 3: 2 models with warmup, GPT-5.1 with semantic transfer at t=300."""
    rng = np.random.RandomState(seed)
    trial_data = data.copy()
    rng.shuffle(trial_data)
    
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,  # Heterogeneous experts: α 1.0→0.01 (Conservative) and α 2.0 (Adaptive)
        corralling_learning_rate=0.1,
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    history = []
    for t, item in enumerate(trial_data):
        if t >= TOTAL_STEPS:
            break
        
        # Add GPT-5.1 WITH semantic transfer
        if t == RELEASE_STEP:
            router.register_model(
                model_id=NEW_MODEL,
                cost_usd=15.0,
                speed="balanced"
            )
        
        selected, _ = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)  # Fixed: (model_id, context, reward)
        history.append(reward)
    
    return history

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================
def run_ablation():
    data = load_data()
    if not data:
        return None
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    results = {
        "Cold Start (No Priors)": [],
        "Warmup Priors Only": [],
        "Warmup + Semantic Transfer": []
    }
    
    logger.info(f"\n🔬 Running ablation study with {N_TRIALS} trials...")
    
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        
        logger.info(f"\n  Trial {i+1}: Cold Start")
        results["Cold Start (No Priors)"].append(
            run_trial_cold_start(seed, data, encoder, pca)
        )
        
        logger.info(f"  Trial {i+1}: Warmup Only")
        results["Warmup Priors Only"].append(
            run_trial_warmup_only(seed, data, encoder, pca)
        )
        
        logger.info(f"  Trial {i+1}: Semantic Transfer")
        results["Warmup + Semantic Transfer"].append(
            run_trial_semantic_transfer(seed, data, encoder, pca)
        )
    
    return results

# ============================================================================
# PLOTTING
# ============================================================================
def plot_ablation(results):
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        "Cold Start (No Priors)": "#e74c3c",
        "Warmup Priors Only": "#f39c12",
        "Warmup + Semantic Transfer": "#2ecc71"
    }
    
    styles = {
        "Cold Start (No Priors)": ":",
        "Warmup Priors Only": "--",
        "Warmup + Semantic Transfer": "-"
    }
    
    for name, histories in results.items():
        matrix = np.array(histories)
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid'),
            1, matrix
        )
        
        mean = np.mean(smoothed, axis=0)
        sem = stats.sem(smoothed, axis=0)
        ci = sem * stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2., N_TRIALS - 1)
        
        x = np.arange(len(mean)) + WINDOW_SIZE // 2
        
        ax.plot(x, mean, label=name, color=colors[name], 
                linestyle=styles[name], linewidth=2.5)
        ax.fill_between(x, mean - ci, mean + ci, color=colors[name], alpha=0.15)
    
    ax.axvline(x=RELEASE_STEP, color='black', alpha=0.5, linewidth=2, 
               linestyle='--', label="GPT-5.1 Release")
    
    ax.set_title(f"Ablation Study: Value of Semantic Transfer (N={N_TRIALS}, 95% CI)",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Routing Steps", fontsize=13)
    ax.set_ylabel("Average Reward", fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_file = output_dir / "figure6_ablation_study.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved ablation plot to {out_file}")
    
    # Print summary
    logger.info(f"\n📊 Post-Release Performance (t=350-450):")
    for name, histories in results.items():
        matrix = np.array(histories)
        post = matrix[:, 350:450].mean()
        logger.info(f"   {name:30s}: {post:.3f}")

if __name__ == "__main__":
    results = run_ablation()
    if results:
        plot_ablation(results)

