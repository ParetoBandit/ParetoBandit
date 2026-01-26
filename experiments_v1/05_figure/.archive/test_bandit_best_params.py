#!/usr/bin/env python3
"""
Test banditGPT Hybrid with BEST parameters (Validation Check)
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
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
    CANONICAL_HOLDOUT_DATA_PATH
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# [Reusing CostAware Classes]
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

def load_resources():
    logger.info("Loading resources...")
    # Models and Costs
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    model_costs = {}
    for model in data["models"]:
        mid = model["openrouter_id"]
        ic = model.get("price_1m_input", 0.0)
        oc = model.get("price_1m_output", 0.0)
        cost = (100 * ic + 400 * oc) / 1_000_000
        model_costs[mid] = {"cost": cost}
    
    # Data (Holdout only for Online Eval)
    prompt_rewards = defaultdict(dict)
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt_rewards[entry["prompt"]][entry["model_id"]] = entry.get("raw_score", 0.0)
    eval_data = [{"prompt": p, "rewards": r} for p, r in prompt_rewards.items() if len(r) == 2]
    
    # Artifacts
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # Normalize costs
    models = list(eval_data[0]["rewards"].keys())
    max_c = max(model_costs[m]["cost"] for m in models)
    min_c = min(model_costs[m]["cost"] for m in models)
    norm_costs = {}
    for m in models:
        raw = model_costs[m]["cost"]
        norm_costs[m] = {"cost": raw, "normalized_cost": (raw - min_c) / (max_c - min_c)}
        
    return eval_data, encoder, pca, warmup_priors, norm_costs

def run_online_eval(eval_data, encoder, pca, priors, costs, gamma, lr, cost_penalty):
    models = list(eval_data[0]["rewards"].keys())
    
    # Apply gamma scaling
    scaled_priors = apply_gamma_scaling(priors, gamma)
    context_dim = scaled_priors["A"][models[0]].shape[0]
    
    # Initialize experts
    warmup = CostAwareLinUCBRouter(models, scaled_priors, costs, alpha=1.0, cost_penalty=cost_penalty)
    tabula = CostAwareTabulaRasaRouter(models, context_dim, costs, alpha=1.0, cost_penalty=cost_penalty)
    
    # Initialize router
    router = CorrallingRouter(experts=[warmup, tabula], models=models, learning_rate=lr)
    
    # Online Evaluation (Learn + Predict)
    rewards = []
    total_cost = 0.0
    model_counts = defaultdict(int)
    
    for pd in eval_data:
        context = embed_prompt(pd["prompt"], encoder, pca)
        
        # Select
        selected = router.select_model(context)
        model_counts[selected] += 1
        
        # Get reward/cost
        r = pd["rewards"][selected]
        c = costs[selected]["cost"]
        
        # Update (Online Learning)
        router.update(context, selected, r)
        
        rewards.append(r)
        total_cost += c
        
    return np.mean(rewards), total_cost / len(eval_data), model_counts

def main():
    # Load resources
    data, enc, pca, priors, costs = load_resources()
    
    # Configuration (Best Parameters)
    gamma = 0.01      # Weakens priors to allow learning
    lr = 0.1          # Stable updates
    penalties = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    logger.info("\n" + "="*70)
    logger.info(f"TEST: Online Evaluation (Gamma={gamma}, LR={lr})")
    logger.info("="*70)
    
    results = []
    for lambda_val in penalties:
        np.random.seed(42) # Reproducibility
        r, c, counts = run_online_eval(data, enc, pca, priors, costs, gamma, lr, lambda_val)
        
        # Calculate % expensive model usage (GPT-4)
        expensive_model = "openai/gpt-4-turbo"
        exp_pct = counts[expensive_model] / len(data) * 100
        
        logger.info(f"λ={lambda_val:5.1f}: Reward={r:.4f}, Cost=${c:.6f}, GPT-4={exp_pct:5.1f}%")
        results.append((lambda_val, r, c))
        
    logger.info("\n✅ Validation Complete")

if __name__ == "__main__":
    main()

