#!/usr/bin/env python3
"""
Calibration Check Script - "Truth Serum"
-----------------------------------------
Verifies that the router's predictions align with actual rewards
across the entire dev set (not just a single "Simple greeting" prompt).

This helps detect systematic bias where the router consistently
under- or over-estimates a model's performance.

NOTE: This diagnostic tool intentionally uses CostAwareLinUCBRouter
directly (not BanditRouter) because it needs to:
  1. Train on ALL models per prompt (not just the selected one)
  2. Access internal θ parameters (A, b matrices) for prediction comparison
These operations are not supported by the production BanditRouter API,
which only trains the model it selects. This script is not used for any
claims in the paper — it is purely a development diagnostic.
"""

import sys
from pathlib import Path
import numpy as np
import joblib
import logging
from collections import defaultdict
import gzip
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import CostAwareLinUCBRouter
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH
)
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dev_data():
    """Load dev set for calibration check."""
    prompt_rewards = defaultdict(lambda: {})
    
    with gzip.open(CANONICAL_DEV_DATA_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                score = entry["raw_score"]
                prompt_rewards[prompt][model_id] = score
    
    # Convert to list
    data = []
    for prompt, rewards in prompt_rewards.items():
        if len(rewards) == 2:
            data.append({"prompt": prompt, "rewards": rewards})
    
    logger.info(f"✓ Loaded {len(data)} prompts from dev set")
    return data


def check_calibration(priors_path: Path, target_sample_size: float = 10.0):
    """
    Check if router predictions are calibrated with actual rewards.
    
    Args:
        priors_path: Path to sanitized priors
        target_sample_size: Prior strength to use (default: 10)
    """
    logger.info("="*70)
    logger.info("CALIBRATION CHECK - TRUTH SERUM TEST")
    logger.info("="*70)
    
    # 1. Load data and models
    train_data = load_dev_data()
    models = list(train_data[0]["rewards"].keys())
    
    # 2. Load encoder and PCA
    logger.info(f"\n📦 Loading models...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    priors = joblib.load(priors_path)
    logger.info(f"  ✓ Loaded sanitized priors from: {priors_path}")
    
    # 3. Normalize prior strength
    from generate_pareto_frontier import normalize_prior_strength
    normalized_priors = normalize_prior_strength(priors, target_sample_size)
    
    # 4. Initialize router (NO cost penalty for calibration check)
    logger.info(f"\n🔧 Initializing router with {target_sample_size} effective samples...")
    router = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=normalized_priors,
        model_costs={m: {"cost": 0, "normalized_cost": 0} for m in models},
        alpha_start=0.1,
        alpha_end=0.1,
        cost_penalty=0.0
    )
    
    # 5. Compute global normalization
    all_raw = [s for p in train_data for s in p["rewards"].values()]
    r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    
    logger.info(f"  ✓ Normalization: [{r_min:.3f}, {r_max:.3f}] → [0.0, 1.0]")
    
    # 6. Train on ALL data (force it to learn everything)
    logger.info(f"\n🎓 Training on {len(train_data)} samples...")
    for p in train_data:
        x = embed_prompt(p["prompt"], encoder, pca)
        # Update ALL models (not just selected one)
        for m in models:
            norm_r = (p["rewards"][m] - r_min) / r_range
            router.update(x, m, norm_r)
    
    logger.info(f"  ✓ Training complete")
    
    # 7. Compare predictions vs reality
    logger.info("\n" + "="*70)
    logger.info("CALIBRATION RESULTS")
    logger.info("="*70)
    logger.info(f"{'Model':<40} | {'True Avg':<10} | {'Pred Avg':<10} | {'Bias':<10}")
    logger.info("-"*70)
    
    for m in models:
        # Calculate true average
        true_rewards = [(p["rewards"][m] - r_min) / r_range for p in train_data]
        avg_true = np.mean(true_rewards)
        
        # Calculate predicted average
        preds = []
        for p in train_data:
            x = embed_prompt(p["prompt"], encoder, pca)
            theta = np.linalg.inv(router.A[m]) @ router.b[m]
            preds.append(theta @ x)
        avg_pred = np.mean(preds)
        
        bias = avg_pred - avg_true
        
        # Display
        model_name = m.split('/')[-1][:40]
        logger.info(f"{model_name:<40} | {avg_true:.4f}     | {avg_pred:.4f}     | {bias:+.4f}")
        
        # Diagnostic
        if abs(bias) > 0.05:
            logger.warning(f"  ⚠️  Bias > 0.05: Router systematically {'over' if bias > 0 else 'under'}-estimates this model")
        else:
            logger.info(f"  ✅ Well calibrated (bias < 0.05)")
    
    logger.info("="*70)


def main():
    priors_path = project_root / "src/artifacts/priors_warmup_normalized.joblib"
    
    if not priors_path.exists():
        logger.error(f"❌ Sanitized priors not found: {priors_path}")
        logger.error(f"   Run: python experiments/03_figure/sanitize_priors.py")
        return 1
    
    check_calibration(priors_path, target_sample_size=10.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

