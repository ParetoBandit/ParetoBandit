"""
Deep Analysis: Why is Cold Start winning?

This script analyzes the model selection patterns to understand
why Cold Start outperforms Warmup + Semantic Transfer.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import joblib
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
TOTAL_STEPS = 800
RELEASE_STEP = 300

def create_registry(models):
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

def load_data():
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=MODELS_2 + [NEW_MODEL]
    )
    data = [item for item in evaluator if all(m in item.rewards for m in MODELS_2 + [NEW_MODEL])]
    return data

def run_analysis_trial(seed: int, data, strategy: str):
    """Run one trial and track model selections."""
    rng = np.random.RandomState(seed)
    trial_data = data.copy()
    rng.shuffle(trial_data)
    
    warmup_priors_path = Path(__file__).parent.parent.parent / "src" / "artifacts" / "priors_warmup.joblib"
    
    # Create router based on strategy
    if strategy == "cold":
        router = BanditRouter.create(
            model_registry=create_registry(MODELS_2),
            context_model=DEFAULT_SENTENCE_TRANSFORMER,
            priors="none",
            use_corralling=True,
            alpha=0.5,
            pca_path=DEFAULT_PCA_PATH
        )
    elif strategy == "warmup_only":
        router = BanditRouter.create(
            model_registry=create_registry(MODELS_2),
            context_model=DEFAULT_SENTENCE_TRANSFORMER,
            priors=str(warmup_priors_path),
            use_corralling=True,
            alpha=0.5,
            pca_path=DEFAULT_PCA_PATH
        )
    else:  # semantic_transfer
        router = BanditRouter.create(
            model_registry=create_registry(MODELS_2),
            context_model=DEFAULT_SENTENCE_TRANSFORMER,
            priors=str(warmup_priors_path),
            use_corralling=True,
            alpha=0.5,
            pca_path=DEFAULT_PCA_PATH
        )
    
    # Track metrics
    rewards = []
    selections = []
    
    for t, item in enumerate(trial_data):
        if t >= TOTAL_STEPS:
            break
        
        # Release event
        if t == RELEASE_STEP:
            if strategy == "cold" or strategy == "warmup_only":
                # Add cold
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
            else:
                # Semantic transfer
                router.register_model(model_id=NEW_MODEL, cost_usd=15.0, speed="balanced")
        
        selected, _ = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(item.prompt, selected, reward)
        
        rewards.append(reward)
        selections.append(selected)
    
    return rewards, selections

def analyze():
    data = load_data()[:800]  # Use subset for speed
    logger.info(f"Loaded {len(data)} samples\n")
    
    seed = 42
    
    logger.info("="*60)
    logger.info("Running diagnostic trial for each strategy...")
    logger.info("="*60)
    
    strategies = {
        "Cold Start": "cold",
        "Warmup Only": "warmup_only",
        "Semantic Transfer": "semantic_transfer"
    }
    
    results = {}
    for name, strat in strategies.items():
        logger.info(f"\n📊 Analyzing: {name}")
        rewards, selections = run_analysis_trial(seed, data, strat)
        results[name] = {"rewards": rewards, "selections": selections}
        
        # Count model selections post-release (t=300-400)
        post_release = selections[300:400]
        from collections import Counter
        counts = Counter(post_release)
        
        logger.info(f"   Model Selection (t=300-400):")
        for model, count in counts.most_common():
            pct = 100 * count / len(post_release)
            logger.info(f"     {model:40s}: {count:3d} ({pct:5.1f}%)")
        
        # Average reward post-release
        avg_reward = np.mean(rewards[300:400])
        logger.info(f"   Avg Reward (t=300-400): {avg_reward:.3f}")
        
        # Check if GPT-5.1 has best ground truth reward
        logger.info(f"\n   Ground Truth Rewards (sample from t=305):")
        sample_item = data[305]
        for model in MODELS_2 + [NEW_MODEL]:
            gt_reward = sample_item.get_reward(model, default=0.0)
            logger.info(f"     {model:40s}: {gt_reward:.3f}")
    
    # Plot selection patterns
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, (name, strat) in enumerate(strategies.items()):
        ax = axes[idx]
        selections = results[name]["selections"]
        rewards = results[name]["rewards"]
        
        # Convert selections to numeric for plotting
        model_to_id = {m: i for i, m in enumerate(MODELS_2 + [NEW_MODEL])}
        selection_ids = [model_to_id[s] for s in selections]
        
        # Plot selections as scatter
        ax.scatter(range(len(selection_ids)), selection_ids, alpha=0.3, s=10, c=rewards, cmap='RdYlGn', vmin=0, vmax=5)
        ax.axvline(x=300, color='red', linestyle='--', alpha=0.7, label='GPT-5.1 Release')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Mixtral', 'GPT-4-turbo', 'GPT-5.1'])
        ax.set_title(f"{name} - Model Selection Pattern", fontweight='bold')
        ax.set_xlabel("Routing Step (t)")
        ax.set_ylabel("Selected Model")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = Path(__file__).parent / "results"
    plt.savefig(output_dir / "ablation_selection_analysis.png", dpi=300)
    logger.info(f"\n✅ Saved analysis plot to {output_dir}/ablation_selection_analysis.png")

if __name__ == "__main__":
    analyze()

