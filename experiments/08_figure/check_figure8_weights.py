"""
Diagnostic: Check if Figure 8 baseline's "75% warmup" claim hides regime switching
===========================================================================
Run Figure 8 baseline experiment with expert weight tracking to see if 75/25 is:
A) Stable within each seed (true blended weights)
B) Average across seeds with binary 0%/100% switching (Simpson's Paradox)
"""

import sys
import numpy as np
from pathlib import Path
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

# Figure 8 baseline configuration
WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
TOTAL_STEPS = 800
RELEASE_STEP = 300
SEEDS_TO_CHECK = [42, 43, 44, 45, 46]  # First 5 seeds from Figure 8 baseline's range (42-71)

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

def run_figure8_config(seed: int):
    """Run Figure 8 baseline experiment configuration with weight tracking."""
    np.random.seed(seed)
    
    evaluator = load_data()
    rng = np.random.RandomState(seed)
    indices = np.arange(len(evaluator.data))
    rng.shuffle(indices)
    shuffled_data = [evaluator.data[i] for i in indices]
    
    # Figure 8 baseline configuration: use_corralling=True, eta=0.1
    router = BanditRouter.create(
        model_registry=create_model_registry(WARMUP_MODELS),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,  # Figure 8 baseline uses conservative eta
        corralling_gamma=0.05,
        alpha=2.0,
        pca_path=DEFAULT_PCA_PATH
    )
    
    expert_weights = []
    rewards = []
    
    for t, item in enumerate(shuffled_data):
        if t >= TOTAL_STEPS: break
        
        if t == RELEASE_STEP:
            router.register_model(
                model_id=NEW_MODEL,
                cost_usd=create_model_registry([NEW_MODEL])[NEW_MODEL]["input_cost_per_m"],
                speed="balanced"
            )
        
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.get_reward(selected, default=0.0)
        router.update(selected, item.prompt, reward)
        
        rewards.append(reward)
        
        # Track expert weights
        if router.corralling_router:
            weights = router.corralling_router.weights.copy()
            expert_weights.append({
                'conservative': weights[0],  # Warmup expert
                'adaptive': weights[1]       # Tabula rasa expert
            })
    
    # Analyze post-release weights
    post_release_weights = expert_weights[RELEASE_STEP:]
    avg_conservative = np.mean([w['conservative'] for w in post_release_weights])
    avg_adaptive = np.mean([w['adaptive'] for w in post_release_weights])
    
    # Check if weights are stable or switching
    conservative_weights = [w['conservative'] for w in post_release_weights]
    is_stable = np.std(conservative_weights) < 0.1  # Stable if std < 10%
    
    return {
        'seed': seed,
        'avg_conservative': avg_conservative,
        'avg_adaptive': avg_adaptive,
        'is_stable': is_stable,
        'std_conservative': np.std(conservative_weights),
        'post_release_reward': np.mean(rewards[RELEASE_STEP:])
    }

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC: Figure 8 baseline Expert Weight Analysis")
    logger.info("="*80)
    logger.info("\nChecking if '~75% warmup' is:")
    logger.info("  A) Stable blended weights within each seed, OR")
    logger.info("  B) Average across seeds with binary switching (Simpson's Paradox)\n")
    
    results = []
    for seed in SEEDS_TO_CHECK:
        logger.info(f"Running seed {seed}...")
        result = run_figure8_config(seed)
        results.append(result)
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS: Post-Release Expert Weights (t>300)")
    logger.info("="*80)
    logger.info(f"{'Seed':<6} | {'Conservative':<12} | {'Adaptive':<12} | {'Stable?':<10} | {'Std':<8} | {'Reward':<8}")
    logger.info("-" * 80)
    
    for r in results:
        stable_str = "✓ Yes" if r['is_stable'] else "✗ No"
        logger.info(f"{r['seed']:<6} | {r['avg_conservative']:<12.3f} | {r['avg_adaptive']:<12.3f} | {stable_str:<10} | {r['std_conservative']:<8.3f} | {r['post_release_reward']:<8.3f}")
    
    logger.info("-" * 80)
    
    # Overall average
    avg_cons = np.mean([r['avg_conservative'] for r in results])
    avg_adap = np.mean([r['avg_adaptive'] for r in results])
    logger.info(f"{'AVG':<6} | {avg_cons:<12.3f} | {avg_adap:<12.3f} | ---        | ---      | ---")
    
    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION:")
    logger.info("="*80)
    
    # Check for regime switching
    binary_seeds = sum(1 for r in results if r['avg_conservative'] > 0.9 or r['avg_conservative'] < 0.1)
    blended_seeds = sum(1 for r in results if 0.3 < r['avg_conservative'] < 0.7)
    
    logger.info(f"\nSeeds with BINARY weights (>90% or <10%): {binary_seeds}/{len(results)}")
    logger.info(f"Seeds with BLENDED weights (30-70%): {blended_seeds}/{len(results)}")
    
    if binary_seeds > blended_seeds:
        logger.info("\n⚠️ WARNING: Figure 8 baseline's '~75% warmup' likely hides REGIME SWITCHING!")
        logger.info("   - Individual seeds show binary expert selection (0% or 100%)")
        logger.info("   - Average of ~75% is Simpson's Paradox (mixing incompatible regimes)")
        logger.info("   - Same confound as Figure 8!")
    else:
        logger.info("\n✓ Figure 8 baseline's '~75% warmup' represents TRUE BLENDED weights")
        logger.info("   - Most seeds show stable 75/25 split (not regime switching)")
        logger.info("   - Different behavior from Figure 8 (possible due to eta=0.1 vs eta=0.1)")
    
    logger.info("\n" + "="*80)
    logger.info("\n✅ Diagnostic complete. Check if Figure 8 baseline and 8 are consistent.\n")
