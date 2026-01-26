#!/usr/bin/env python3
"""
Test banditGPT Hybrid with normalized costs
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import CorrallingRouter
from bandit_gpt.calibration import embed_prompt, apply_gamma_scaling
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

# Cost-aware routers
class CostAwareLinUCBRouter:
    def __init__(self, models, warmup_priors, model_costs, alpha=1.0, cost_penalty=0.0):
        self.models = models
        self.alpha = alpha
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.context_dim = warmup_priors['context_dim']
        self.A = {m: warmup_priors['A'][m].copy() for m in models}
        self.b = {m: warmup_priors['b'][m].copy() for m in models}
    
    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.model_costs[model]["normalized_cost"]
            ucb_scores[model] = expected_reward - self.cost_penalty * normalized_cost + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()

class CostAwareTabulaRasaRouter:
    def __init__(self, models, context_dim, model_costs, alpha=1.0, cost_penalty=0.0):
        self.models = models
        self.alpha = alpha
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
    
    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            normalized_cost = self.model_costs[model]["normalized_cost"]
            ucb_scores[model] = expected_reward - self.cost_penalty * normalized_cost + self.alpha * uncertainty
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()

def test_bandit(train_data, eval_data, encoder, pca, warmup_priors, normalized_costs, lambda_val):
    models = list(train_data[0]["rewards"].keys())
    
    # Tune hyperparameters for stability
    gamma = 0.1  # Less aggressive scaling (was 0.01)
    learning_rate = 0.1  # More stable updates (was 1.0)
    
    scaled_priors = apply_gamma_scaling(warmup_priors, gamma)
    context_dim = scaled_priors["A"][list(scaled_priors["A"].keys())[0]].shape[0]
    
    # Create experts
    warmup_expert = CostAwareLinUCBRouter(
        models=models,
        warmup_priors=scaled_priors,
        model_costs=normalized_costs,
        alpha=1.0,
        cost_penalty=lambda_val
    )
    
    tabula_rasa_expert = CostAwareTabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        model_costs=normalized_costs,
        alpha=1.0,
        cost_penalty=lambda_val
    )
    
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=learning_rate  # Use stable learning rate
    )
    
    # Train
    for prompt_data in train_data:
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        router.update(context, selected_model, reward)
    
    # Evaluate
    total_reward = 0.0
    total_cost = 0.0
    for prompt_data in eval_data:
        prompt = prompt_data["prompt"]
        rewards = prompt_data["rewards"]
        context = embed_prompt(prompt, encoder, pca)
        selected_model = router.select_model(context)
        reward = rewards[selected_model]
        total_reward += reward
        total_cost += normalized_costs[selected_model]["cost"]  # Use REAL cost
    
    return total_reward / len(eval_data), total_cost / len(eval_data)

def main():
    logger.info("="*70)
    logger.info("TEST: banditGPT Hybrid with Normalized Costs")
    logger.info("="*70)
    
    # Load
    logger.info("\nLoading...")
    model_costs = load_model_costs()
    train_data, eval_data = load_data()
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    logger.info(f"  Train: {len(train_data)} prompts")
    logger.info(f"  Eval: {len(eval_data)} prompts")
    
    # Normalize costs
    models = list(train_data[0]["rewards"].keys())
    max_cost = max(model_costs[m]["cost"] for m in models)
    min_cost = min(model_costs[m]["cost"] for m in models)
    cost_range = max_cost - min_cost
    
    logger.info(f"\nCost normalization:")
    logger.info(f"  Min: ${min_cost:.6f}")
    logger.info(f"  Max: ${max_cost:.6f}")
    logger.info(f"  Range: ${cost_range:.6f}")
    
    normalized_costs = {}
    for model_id in models:
        raw_cost = model_costs[model_id]["cost"]
        normalized = (raw_cost - min_cost) / cost_range if cost_range > 0 else 0.0
        normalized_costs[model_id] = {
            "cost": raw_cost,
            "normalized_cost": normalized
        }
        logger.info(f"  {model_costs[model_id]['name']}: ${raw_cost:.6f} → {normalized:.4f}")
    
    # Test different lambdas
    logger.info("\n" + "="*70)
    logger.info("Testing cost penalties (λ):")
    logger.info("="*70)
    
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    for i, lambda_val in enumerate(lambdas, 1):
        logger.info(f"\n[{i}/{len(lambdas)}] λ={lambda_val:.2f}...")
        reward, cost = test_bandit(train_data, eval_data, encoder, pca, warmup_priors, normalized_costs, lambda_val)
        results.append((lambda_val, cost, reward))
        logger.info(f"  Reward: {reward:.4f}, Cost: ${cost:.6f}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info("\nλ     | Cost      | Reward")
    logger.info("------|-----------|-------")
    for lambda_val, cost, reward in results:
        logger.info(f"{lambda_val:5.2f} | ${cost:.6f} | {reward:.4f}")
    
    logger.info("\n✅ Done!")

if __name__ == "__main__":
    main()

