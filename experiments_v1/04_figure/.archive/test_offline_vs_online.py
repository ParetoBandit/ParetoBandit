#!/usr/bin/env python3
"""
Test: Offline vs Online Evaluation
Compare frozen policy (offline) vs adaptive policy (online)
Uses LIBRARY routers (no custom code)
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
from typing import Dict, List
import logging
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import LIBRARY routers
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.calibration import SimpleLinUCBRouter, embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load data
def load_data():
    def load_split(filepath):
        prompt_rewards = defaultdict(lambda: {})
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    prompt = entry["prompt"]
                    model_id = entry["model_id"]
                    score = entry.get("raw_score", 0.0)
                    prompt_rewards[prompt][model_id] = score
        
        prompts_data = []
        for prompt, rewards in prompt_rewards.items():
            if len(rewards) == 2:
                prompts_data.append({"prompt": prompt, "rewards": rewards})
        return prompts_data
    
    train_data = load_split(CANONICAL_DEV_DATA_PATH)
    eval_data = load_split(CANONICAL_HOLDOUT_DATA_PATH)
    return train_data, eval_data

# Load model costs
def load_model_costs():
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    costs = {}
    for model in data["models"]:
        model_id = model["openrouter_id"]
        input_cost = model.get("price_1m_input", 0.0)
        output_cost = model.get("price_1m_output", 0.0)
        cost_per_request = (100 * input_cost + 400 * output_cost) / 1_000_000
        costs[model_id] = {
            "name": model.get("display_name", model_id),
            "cost": cost_per_request
        }
    return costs

# Tabula rasa router (for Corralling)
class TabulaRasaRouter:
    """Simple tabula rasa LinUCB (learns from scratch)."""
    def __init__(self, models, context_dim, alpha=1.0):
        self.models = models
        self.alpha = alpha
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
    
    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores[model] = expected + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()

def test_offline_evaluation(train_data, eval_data, encoder, pca, warmup_priors, model_costs):
    """
    OFFLINE: Train on dev, FREEZE, evaluate on holdout.
    This is what we're currently doing.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 1: OFFLINE EVALUATION (Train on Dev, Freeze, Eval on Holdout)")
    logger.info("="*70)
    
    models = list(train_data[0]["rewards"].keys())
    context_dim = warmup_priors["A"][list(warmup_priors["A"].keys())[0]].shape[0]
    
    # Create experts using LIBRARY routers
    warmup_expert = SimpleLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors,
        alpha=1.0
    )
    
    tabula_rasa_expert = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    # Use LIBRARY CorrallingRouter
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=1.0
    )
    
    # PHASE 1: Train on dev
    logger.info(f"\nPhase 1: Training on {len(train_data)} dev prompts...")
    for i, prompt_data in enumerate(train_data):
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        router.update(context, selected_model, reward)
        
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(train_data)} prompts")
    
    logger.info("  ✓ Training complete")
    
    # PHASE 2: Evaluate on holdout (FROZEN - no updates)
    logger.info(f"\nPhase 2: Evaluating on {len(eval_data)} holdout prompts (FROZEN)...")
    total_reward = 0.0
    total_cost = 0.0
    
    for prompt_data in eval_data:
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        # NO UPDATE - frozen policy
        total_reward += reward
        total_cost += model_costs[selected_model]["cost"]
    
    avg_reward = total_reward / len(eval_data)
    avg_cost = total_cost / len(eval_data)
    
    logger.info(f"\n✓ OFFLINE Results:")
    logger.info(f"  Reward: {avg_reward:.4f}")
    logger.info(f"  Cost: ${avg_cost:.6f}")
    
    return avg_reward, avg_cost

def test_online_evaluation(eval_data, encoder, pca, warmup_priors, model_costs):
    """
    ONLINE: Train AND evaluate on holdout (keep learning).
    This is standard for bandit algorithms.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 2: ONLINE EVALUATION (Train + Eval on Holdout, Keep Learning)")
    logger.info("="*70)
    
    models = list(eval_data[0]["rewards"].keys())
    context_dim = warmup_priors["A"][list(warmup_priors["A"].keys())[0]].shape[0]
    
    # Create fresh experts using LIBRARY routers
    warmup_expert = SimpleLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors,
        alpha=1.0
    )
    
    tabula_rasa_expert = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    # Use LIBRARY CorrallingRouter
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=1.0
    )
    
    # Train + Evaluate (ONLINE)
    logger.info(f"\nTraining + Evaluating on {len(eval_data)} holdout prompts (ONLINE)...")
    total_reward = 0.0
    total_cost = 0.0
    
    for i, prompt_data in enumerate(eval_data):
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        
        # UPDATE - keep learning!
        router.update(context, selected_model, reward)
        
        total_reward += reward
        total_cost += model_costs[selected_model]["cost"]
        
        if (i + 1) % 200 == 0:
            logger.info(f"  Processed {i+1}/{len(eval_data)} prompts")
    
    avg_reward = total_reward / len(eval_data)
    avg_cost = total_cost / len(eval_data)
    
    logger.info(f"\n✓ ONLINE Results:")
    logger.info(f"  Reward: {avg_reward:.4f}")
    logger.info(f"  Cost: ${avg_cost:.6f}")
    
    return avg_reward, avg_cost

def main():
    logger.info("="*70)
    logger.info("HYPOTHESIS TEST: Offline vs Online Evaluation")
    logger.info("="*70)
    logger.info("\nUsing LIBRARY routers:")
    logger.info("  - SimpleLinUCBRouter (warmup expert)")
    logger.info("  - TabulaRasaRouter (from scratch expert)")
    logger.info("  - CorrallingRouter (combines experts)")
    
    # Load
    logger.info("\nLoading data...")
    train_data, eval_data = load_data()
    model_costs = load_model_costs()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    logger.info(f"  Train (dev): {len(train_data)} prompts")
    logger.info(f"  Eval (holdout): {len(eval_data)} prompts")
    
    # Test 1: Offline
    offline_reward, offline_cost = test_offline_evaluation(
        train_data, eval_data, encoder, pca, warmup_priors, model_costs
    )
    
    # Test 2: Online
    online_reward, online_cost = test_online_evaluation(
        eval_data, encoder, pca, warmup_priors, model_costs
    )
    
    # Compare
    logger.info("\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    
    logger.info("\nBaselines:")
    logger.info("  Mixtral-only: 0.8227")
    logger.info("  GPT-4-only: 0.8120")
    logger.info("  Oracle: 0.9533")
    
    logger.info("\nbanditGPT Results:")
    logger.info(f"  OFFLINE (frozen): {offline_reward:.4f} @ ${offline_cost:.6f}")
    logger.info(f"  ONLINE (adaptive): {online_reward:.4f} @ ${online_cost:.6f}")
    
    improvement = online_reward - offline_reward
    logger.info(f"\nImprovement: {improvement:+.4f} ({improvement/offline_reward*100:+.1f}%)")
    
    logger.info("\n" + "="*70)
    logger.info("CONCLUSION")
    logger.info("="*70)
    
    if online_reward > offline_reward + 0.01:
        logger.info("\n✅ HYPOTHESIS CONFIRMED!")
        logger.info("   Online evaluation (keep learning) performs better.")
        logger.info("   This is why our offline results looked bad.")
    elif abs(online_reward - offline_reward) < 0.01:
        logger.info("\n⚠️  HYPOTHESIS REJECTED")
        logger.info("   No significant difference between offline and online.")
        logger.info("   The problem must be something else.")
    else:
        logger.info("\n❌ UNEXPECTED RESULT")
        logger.info("   Offline performed better than online?!")
        logger.info("   This suggests a bug in the online version.")
    
    logger.info("\n" + "="*70)

if __name__ == "__main__":
    main()

