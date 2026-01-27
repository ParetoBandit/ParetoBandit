"""
Check: Are the routers actually selecting GPT-5.1 post-release?
"""
import sys
from pathlib import Path
import numpy as np
import joblib
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bandit_gpt.router import BanditRouter
from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS
)

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
TOTAL_STEPS = 800
RELEASE_STEP = 300

def create_registry(models):
    all_models = {
        "mistralai/mixtral-8x7b-instruct": {
            "input_cost_per_m": 0.5,
            "output_cost_per_m": 1.5,
        },
        "openai/gpt-4-turbo": {
            "input_cost_per_m": 10.0,
            "output_cost_per_m": 30.0,
        },
        "openai/gpt-5.1": {
            "input_cost_per_m": 15.0,
            "output_cost_per_m": 45.0,
        }
    }
    return {k: v for k, v in all_models.items() if k in models}

def run_diagnostic(strategy_name, priors, use_semantic_transfer):
    # Load data
    evaluator = AlignedEvaluator.from_jsonl_gz(
        DEV_DATA_PATH_ALL_MODELS,
        required_models=MODELS_2 + [NEW_MODEL]
    )
    data = [item for item in evaluator if all(m in item.rewards for m in MODELS_2 + [NEW_MODEL])]
    
    # Shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(data)
    data = data[:TOTAL_STEPS]
    
    # Create router
    router = BanditRouter.create(
        model_registry=create_registry(MODELS_2),
        context_model=DEFAULT_SENTENCE_TRANSFORMER,
        priors=priors,
        use_corralling=True,
        alpha=0.5,
        pca_path=DEFAULT_PCA_PATH
    )
    
    selections_pre = []
    selections_post = []
    rewards_pre = []
    rewards_post = []
    
    for t, item in enumerate(data):
        if t >= TOTAL_STEPS:
            break
        
        # Release event
        if t == RELEASE_STEP:
            if use_semantic_transfer:
                router.register_model(model_id=NEW_MODEL, cost_usd=15.0, speed="balanced")
            else:
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
        
        selected, _ = router.route(item.prompt, profile="auto")
        reward = item.get_reward(selected, default=0.0)
        router.update(item.prompt, selected, reward)
        
        if t < RELEASE_STEP:
            selections_pre.append(selected)
            rewards_pre.append(reward)
        else:
            selections_post.append(selected)
            rewards_post.append(reward)
    
    print(f"\n{'='*60}")
    print(f"{strategy_name}")
    print(f"{'='*60}")
    
    print(f"\nPre-Release (t=0-299):")
    counts_pre = Counter(selections_pre)
    for model, count in counts_pre.most_common():
        pct = 100 * count / len(selections_pre)
        avg_r = np.mean([r for s, r in zip(selections_pre, rewards_pre) if s == model])
        print(f"  {model:40s}: {count:3d} ({pct:5.1f}%) | Avg Reward: {avg_r:.3f}")
    print(f"  Overall Avg Reward: {np.mean(rewards_pre):.3f}")
    
    print(f"\nPost-Release (t=300-799):")
    counts_post = Counter(selections_post)
    for model, count in counts_post.most_common():
        pct = 100 * count / len(selections_post)
        avg_r = np.mean([r for s, r in zip(selections_post, rewards_post) if s == model])
        print(f"  {model:40s}: {count:3d} ({pct:5.1f}%) | Avg Reward: {avg_r:.3f}")
    print(f"  Overall Avg Reward: {np.mean(rewards_post):.3f}")
    
    # Check GPT-5.1 selection rate
    gpt5_count = counts_post.get(NEW_MODEL, 0)
    gpt5_pct = 100 * gpt5_count / len(selections_post)
    print(f"\n🎯 GPT-5.1 Selection Rate: {gpt5_pct:.1f}%")
    
    if gpt5_pct < 10:
        print(f"   ❌ PROBLEM: Router is NOT exploring GPT-5.1!")
    elif gpt5_pct < 50:
        print(f"   ⚠️  Router is under-utilizing GPT-5.1")
    else:
        print(f"   ✅ Router is using GPT-5.1 appropriately")

if __name__ == "__main__":
    warmup_priors_path = Path(__file__).parent.parent.parent / "src" / "artifacts" / "priors_warmup.joblib"
    
    run_diagnostic("Cold Start (No Priors)", "none", False)
    run_diagnostic("Warmup Only", str(warmup_priors_path), False)
    run_diagnostic("Warmup + Semantic Transfer", str(warmup_priors_path), True)
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("\nIf all strategies show low GPT-5.1 selection (<50%):")
    print("  → The bandit's exploration is too LOW (alpha/cost penalty too high)")
    print("  → Need to increase exploration or reduce cost penalty")
    print("\nOptimal: GPT-5.1 should be selected ~80-90% post-release")
    print("(Since it has +1.730 higher expected reward)")

