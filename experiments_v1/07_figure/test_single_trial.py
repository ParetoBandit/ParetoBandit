"""Quick diagnostic: Test single trial to verify alpha propagation."""
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.aligned_evaluator import AlignedEvaluator
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER, DEFAULT_PCA_PATH, DEV_DATA_PATH_ALL_MODELS
from bandit_gpt.router import BanditRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MODELS_2 = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
NEW_MODEL = "openai/gpt-5.1"
MODELS_3 = MODELS_2 + [NEW_MODEL]

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

# Load data
evaluator = AlignedEvaluator.from_jsonl_gz(DEV_DATA_PATH_ALL_MODELS, required_models=MODELS_3)
data = [item for item in evaluator if all(m in item.rewards for m in MODELS_3)][:400]
logger.info(f"Loaded {len(data)} samples")

warmup_priors_path = Path(__file__).parent.parent.parent / "src" / "artifacts" / "priors_warmup.joblib"

# Test Warmup + Semantic Transfer (should work)
logger.info("\n" + "="*80)
logger.info("Testing Warmup + Semantic Transfer")
logger.info("="*80)

router = BanditRouter.create(
    model_registry=create_registry(MODELS_2),
    context_model=DEFAULT_SENTENCE_TRANSFORMER,
    priors=str(warmup_priors_path),
    use_corralling=True,
    alpha=2.0,
    pca_path=DEFAULT_PCA_PATH
)

# Check expert alpha values
if router.corralling_router:
    for i, expert in enumerate(router.corralling_router.experts):
        logger.info(f"Expert {i} ({type(expert).__name__}): alpha_start={expert.alpha_start}, alpha_end={expert.alpha_end}")

selections_before = {m: 0 for m in MODELS_2}
selections_after = {m: 0 for m in MODELS_3}

for t, item in enumerate(data):
    if t == 300:
        logger.info(f"\n🚀 At t={t}, registering {NEW_MODEL} with semantic transfer...")
        router.register_model(
            model_id=NEW_MODEL,
            cost_usd=15.0,
            latency_s=2.0,
            speed="balanced"
        )
        logger.info(f"   Models in bandit: {router.bandit.models}")
        if router.corralling_router:
            logger.info(f"   Models in corralling: {router.corralling_router.models}")
            for i, expert in enumerate(router.corralling_router.experts):
                logger.info(f"   Expert {i} models: {expert.models}")
    
    selected, _ = router.route(item.prompt, profile="auto")
    reward = item.get_reward(selected, default=0.0)
    router.update(item.prompt, selected, reward)
    
    if t < 300:
        selections_before[selected] += 1
    else:
        selections_after[selected] += 1
    
    # Log first few selections after release
    if 300 <= t < 310:
        logger.info(f"   t={t}: selected={selected}, reward={reward:.3f}")

logger.info(f"\n📊 Pre-release (t=0-299): {selections_before}")
logger.info(f"📊 Post-release (t=300-399): {selections_after}")
gpt5_rate = selections_after.get(NEW_MODEL, 0) / sum(selections_after.values()) * 100
logger.info(f"🎯 GPT-5.1 selection rate: {gpt5_rate:.1f}%")

