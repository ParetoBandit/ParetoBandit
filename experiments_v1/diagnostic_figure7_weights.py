"""
Diagnostic: Check Figure 7 Expert Weights
==========================================
Runs Figure 7 experiment and analyzes actual expert weights to see if they match
the claimed "~75% Conservative, ~25% Adaptive" or show binary regime switching.
"""
import sys
import numpy as np
from pathlib import Path
import logging
import joblib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# Configuration
N_TRIALS = 3  # Just check first 3 seeds like Figure 8
TOTAL_STEPS = 800
RELEASE_STEP = 300
WARMUP_MODELS = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"

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

def load_data():
    try:
        evaluator = AlignedEvaluator.from_jsonl_gz(
            DEV_DATA_PATH_ALL_MODELS,
            required_models=WARMUP_MODELS + [NEW_MODEL]
        )
        data = [item for item in evaluator if all(m in item.rewards for m in WARMUP_MODELS + [NEW_MODEL])]
        logger.info(f"✅ Loaded {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Data error: {e}")
        return []

def identify_complex_subset(data):
    """Identifies complex tasks (same logic as Figure 7)"""
    gap = np.array([
        item.rewards.get("openai/gpt-5.1", 0) - 
        max(item.rewards.get(m, 0) for m in WARMUP_MODELS)
        for item in data
    ])
    complex_indices = set(np.where(gap > 0.3)[0])
    logger.info(f"Complex subset: {len(complex_indices)}/{len(data)} samples")
    return complex_indices

def run_trial(seed, data, encoder, pca, complex_indices):
    """Run single trial and track expert weights"""
    np.random.seed(seed)
    registry = create_model_registry()
    initial_registry = {k: v for k, v in registry.items() if k in WARMUP_MODELS}
    
    # Phase 1: Train on 2 models (using correct API like Figure 7)
    router = BanditRouter.create(
        model_registry=initial_registry,
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=str(DEFAULT_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=0.1,  # Conservative
        corralling_gamma=0.05,
        pca_path=DEFAULT_PCA_PATH
    )
    
    expert_weights_history = []
    indices = sorted(list(complex_indices))
    
    # Pre-release + post-release phases
    for t in range(TOTAL_STEPS):
        idx = indices[t % len(indices)]
        item = data[idx]
        
        # Release new model at t=300
        if t == RELEASE_STEP:
            router.register_model(
                model_id=NEW_MODEL,
                cost_usd=registry[NEW_MODEL]["input_cost_per_m"],
                speed="balanced"
            )
        
        # Route and update
        selected, _ = router.route(item.prompt, profile="auto", total_steps=TOTAL_STEPS)
        reward = item.rewards[selected] if selected in item.rewards else 0.0
        router.update(selected, item.prompt, reward)
        
        # Track weights (same as Figure 7)
        if router.corralling_router:
            expert_weights_history.append([
                router.corralling_router.weights[0],  # conservative (warmup)
                router.corralling_router.weights[1]   # adaptive (tabula rasa)
            ])
    
    return np.array(expert_weights_history)

def analyze_weights(all_weights):
    """Analyze weight patterns across seeds"""
    print("\n" + "="*80)
    print("FIGURE 7 EXPERT WEIGHT ANALYSIS")
    print("="*80)
    
    for seed_idx in range(len(all_weights)):
        seed = 42 + seed_idx
        weights = all_weights[seed_idx]
        
        if len(weights) == 0:
            print(f"\nSeed {seed}: No weight data available")
            continue
        
        # Post-release weights (t=300-800)
        post_release_weights = weights[RELEASE_STEP:, :]
        
        # Calculate average weights
        avg_expert0 = np.mean(post_release_weights[:, 0])
        avg_expert1 = np.mean(post_release_weights[:, 1])
        
        # Calculate final weights (last 100 steps)
        final_weights = post_release_weights[-100:, :]
        final_expert0 = np.mean(final_weights[:, 0])
        final_expert1 = np.mean(final_weights[:, 1])
        
        # Detect regime
        if final_expert0 > 0.9:
            regime = "WARMUP-DOMINANT (100%)"
        elif final_expert1 > 0.9:
            regime = "TABULA RASA-DOMINANT (100%)"
        elif 0.6 < final_expert0 < 0.9:
            regime = "WARMUP-BIASED (~75%)"
        elif 0.6 < final_expert1 < 0.9:
            regime = "TABULA RASA-BIASED (~75%)"
        else:
            regime = "MIXED"
        
        print(f"\n--- SEED {seed} ---")
        print(f"  Post-release avg: Expert0={avg_expert0:.3f}, Expert1={avg_expert1:.3f}")
        print(f"  Final (t=700-800): Expert0={final_expert0:.3f}, Expert1={final_expert1:.3f}")
        print(f"  Regime: {regime}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print("\nFigure 7 CLAIMS (in documentation):")
    print("  'stable expert weights throughout (~75% Conservative, ~25% Adaptive)'")
    
    print("\nFigure 8 SHOWS:")
    print("  Binary regime switching (100% one expert OR 100% the other)")
    
    print("\nACTUAL Figure 7 RESULTS (above):")
    print("  [Analysis shown above for seeds 42-44]")
    
    print("\n" + "="*80)

def main():
    data = load_data()
    if not data:
        return
    
    complex_indices = identify_complex_subset(data)
    if not complex_indices:
        return
    
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    
    all_weights = []
    
    logger.info(f"\n🔬 Running {N_TRIALS} trials to check expert weights...")
    for i in tqdm(range(N_TRIALS), desc="Trials"):
        seed = 42 + i
        weights = run_trial(seed, data, encoder, pca, complex_indices)
        all_weights.append(weights)
    
    analyze_weights(all_weights)

if __name__ == "__main__":
    main()
