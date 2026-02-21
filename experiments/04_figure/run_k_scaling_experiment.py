#!/usr/bin/env python3
"""
K-Scaling Experiment: How Does Routing Performance Scale with Portfolio Size?
=============================================================================

This is the primary scaling experiment: demonstrates banditGPT's core
routing value proposition at K={2, 5, 10, 20, 41} using REAL multi-model
reward data from LMSYS Arena.

Key questions answered:
    1. Does UCB exploration benefit grow with K? (LinUCB vs ε-greedy)
    2. Does Thompson Sampling outperform UCB at K>2? (LinTS vs LinUCB)
    3. How does Corralling scale? (banditGPT-Hybrid vs single-expert)
    4. Do warmup priors help more at higher K?
    5. Where is the inflection point where sample budget limits learning?

Protocol:
    - Phase 1: Build warmup priors from PRIOR SPLIT of dev data (full-information)
    - Phase 2: Train on BURN-IN SPLIT of dev data (bandit feedback, disjoint from priors)
    - Phase 3: Evaluate on holdout (pure exploitation, no updates)
    - Seeds: 20 independent trials per (method, K) pair
    - Features: PCA(32) + bias = 33D (same as all other experiments)

    The dev data is split 50/50 into:
      - Prior split: used ONLY for building full-information warmup priors
      - Burn-in split: used ONLY for online bandit training (selected arm reward only)
    This prevents double-dipping — the prior-building and burn-in phases see
    entirely different prompts, analogous to how the K=2 paper uses RouteLLM
    Arena battle priors (Source A) vs. dev reward data (Source B).

Data:
    - Dev: 1121 prompts with all 41 core models → ~560 prior + ~561 burn-in
    - Holdout: 750 prompts with all 41 core models
    - Models span Llama-3.2-1b (0.356 reward) to GPT-5-chat (0.985 reward)

    Note: 2 models (openai/gpt-5, google/gemini-2.5-flash-preview-09-2025) are
    excluded from the maximum-K portfolio because they were evaluated on a
    separate batch of 233 dev prompts that contains ONLY those 2 models (no
    overlap with the 41-model prompts). Including them would reduce K=max dev
    data from 1121 → 888 prompts (−26%), harming statistical power for a
    marginal gain of +2 models. All 43 models have full holdout coverage (750).

Model portfolios designed for diversity across quality tiers and cost tiers.
"""

import sys
from pathlib import Path
import json
import gzip
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import (
    CorrallingRouter,
    CostAwareLinUCBRouter,
    CostAwareTabulaRasaRouter,
)
from bandit_gpt.baselines import (
    CostAwareLinTSRouter,
    CostAwareLearnedProjRouter,
)
from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from sentence_transformers import SentenceTransformer
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

N_TRIALS = 20
SEED_OFFSET = 42
ALPHA_START = 2.0
ALPHA_END = 0.1
TARGET_SAMPLE_SIZE = 10.0

# Realistic cost estimates ($/M tokens: input/output) from OpenRouter Feb 2026
# Cost per request assumes 100 input + 400 output tokens
MODEL_COSTS_PER_M = {
    # Frontier reasoning
    "openai/o3":                                {"input": 10.00, "output": 40.00},
    "openai/o1":                                {"input": 5.00,  "output": 20.00},
    "x-ai/grok-4":                              {"input": 5.00,  "output": 15.00},
    # Frontier chat
    "anthropic/claude-opus-4.5":                 {"input": 5.00,  "output": 25.00},
    "openai/gpt-5":                              {"input": 2.50,  "output": 10.00},
    "openai/gpt-5.1":                            {"input": 2.50,  "output": 10.00},
    "openai/gpt-5-chat":                         {"input": 1.25,  "output": 10.00},
    "anthropic/claude-sonnet-4.5":               {"input": 3.00,  "output": 15.00},
    "anthropic/claude-3.7-sonnet:thinking":      {"input": 3.00,  "output": 15.00},
    "anthropic/claude-sonnet-4":                 {"input": 3.00,  "output": 15.00},
    "google/gemini-3-pro-preview":               {"input": 1.25,  "output": 10.00},
    "google/gemini-2.5-pro-preview-06-05":       {"input": 1.25,  "output": 10.00},
    # Mid-tier
    "openai/gpt-4.1":                            {"input": 2.00,  "output": 8.00},
    "openai/gpt-4o":                             {"input": 1.25,  "output": 5.00},
    "openai/gpt-4-turbo":                        {"input": 10.00, "output": 30.00},
    "x-ai/grok-3":                               {"input": 3.00,  "output": 15.00},
    "x-ai/grok-3-mini":                          {"input": 0.30,  "output": 0.50},
    "anthropic/claude-3.5-sonnet":               {"input": 3.00,  "output": 15.00},
    "anthropic/claude-haiku-4.5":                {"input": 1.00,  "output": 5.00},
    "cohere/command-a-03-2025":                  {"input": 2.50,  "output": 10.00},
    "moonshotai/kimi-k2-0905":                   {"input": 0.60,  "output": 2.00},
    "deepseek/deepseek-chat-v3-0324":            {"input": 0.14,  "output": 0.28},
    "google/gemini-2.5-flash-preview-09-2025":   {"input": 0.15,  "output": 0.60},
    "google/gemini-2.5-flash-lite":              {"input": 0.02,  "output": 0.10},
    # Open-source large
    "openai/gpt-oss-120b":                       {"input": 1.00,  "output": 3.00},
    "openai/gpt-oss-20b":                        {"input": 0.15,  "output": 0.60},
    "meta-llama/llama-3.1-405b-instruct":        {"input": 0.80,  "output": 0.80},
    "meta-llama/llama-4-maverick":               {"input": 0.50,  "output": 0.70},
    "meta-llama/llama-4-scout":                  {"input": 0.15,  "output": 0.35},
    "meta-llama/llama-3.1-70b-instruct":         {"input": 0.10,  "output": 0.32},
    "amazon/nova-pro-v1":                        {"input": 0.80,  "output": 3.20},
    # Open-source small / budget
    "google/gemma-3-27b-it":                     {"input": 0.10,  "output": 0.20},
    "google/gemma-3-12b-it":                     {"input": 0.06,  "output": 0.12},
    "google/gemma-3-4b-it":                      {"input": 0.03,  "output": 0.06},
    "mistralai/mistral-small-3.2-24b-instruct":  {"input": 0.10,  "output": 0.30},
    "mistralai/mixtral-8x7b-instruct":           {"input": 0.06,  "output": 0.06},
    "mistralai/ministral-8b":                    {"input": 0.04,  "output": 0.04},
    "mistralai/ministral-3b":                    {"input": 0.02,  "output": 0.02},
    "microsoft/phi-4":                           {"input": 0.05,  "output": 0.10},
    "amazon/nova-micro-v1":                      {"input": 0.04,  "output": 0.02},
    "amazon/nova-lite-v1":                       {"input": 0.06,  "output": 0.02},
    "meta-llama/llama-3.1-8b-instruct":          {"input": 0.02,  "output": 0.02},
    "meta-llama/llama-3.2-1b-instruct":          {"input": 0.01,  "output": 0.01},
}


def compute_cost_per_request(model_id: str, input_tokens: int = 100,
                              output_tokens: int = 400) -> float:
    """Compute cost per request from per-M-token pricing."""
    if model_id not in MODEL_COSTS_PER_M:
        return 0.005  # Fallback: moderate cost
    c = MODEL_COSTS_PER_M[model_id]
    return (input_tokens * c["input"] + output_tokens * c["output"]) / 1_000_000


# =============================================================================
# MODEL PORTFOLIOS — designed for diversity across quality and cost tiers
# =============================================================================

PORTFOLIOS = {
    2: [
        "mistralai/mixtral-8x7b-instruct",      # Budget (0.823)
        "openai/gpt-4-turbo",                    # Expensive (0.812) — original pair
    ],
    5: [
        "meta-llama/llama-3.1-8b-instruct",     # Budget small (0.745)
        "mistralai/mixtral-8x7b-instruct",       # Budget mid (0.823)
        "google/gemma-3-27b-it",                 # Mid-tier open (0.951)
        "openai/gpt-4o",                         # Frontier mid (0.971)
        "openai/gpt-5-chat",                     # Frontier top (0.985)
    ],
    10: [
        "meta-llama/llama-3.2-1b-instruct",     # Tiny (0.356)
        "meta-llama/llama-3.1-8b-instruct",     # Small (0.745)
        "mistralai/mixtral-8x7b-instruct",       # Budget mid (0.823)
        "amazon/nova-micro-v1",                   # Budget (0.895)
        "google/gemma-3-12b-it",                 # Mid open (0.955)
        "anthropic/claude-haiku-4.5",            # Mid proprietary (0.951)
        "openai/gpt-4o",                         # Strong (0.971)
        "deepseek/deepseek-chat-v3-0324",        # Strong cheap (0.973)
        "anthropic/claude-sonnet-4",             # Frontier (0.975)
        "openai/gpt-5-chat",                     # Top (0.985)
    ],
    20: [
        "meta-llama/llama-3.2-1b-instruct",
        "mistralai/ministral-3b",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/ministral-8b",
        "microsoft/phi-4",
        "mistralai/mixtral-8x7b-instruct",
        "amazon/nova-lite-v1",
        "google/gemma-3-4b-it",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemma-3-12b-it",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.5-flash-lite",
        "deepseek/deepseek-chat-v3-0324",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "cohere/command-a-03-2025",
        "x-ai/grok-3",
        "openai/gpt-5-chat",
        "anthropic/claude-opus-4.5",
        "openai/o3",
    ],
}
# K=41: all models with complete dev+holdout coverage (computed dynamically).
# Two models are excluded because they were evaluated on a separate 233-prompt
# batch that contains ONLY those 2 models, reducing K=43 dev data to 888.
# Dropping them yields 1121 complete dev prompts (+26% training data).
EXCLUDED_MODELS = {
    "openai/gpt-5",
    "google/gemini-2.5-flash-preview-09-2025",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_models_data(filepath: Path, min_models: int = 2) -> List[Dict]:
    """Load multi-model reward data, keeping only prompts with ≥ min_models."""
    prompt_rewards = defaultdict(dict)

    with gzip.open(filepath, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]

    data = [
        {"prompt": p, "rewards": r}
        for p, r in prompt_rewards.items()
        if len(r) >= min_models
    ]
    return data


def filter_portfolio(data: List[Dict], models: List[str]) -> List[Dict]:
    """Keep only prompts that have rewards for ALL models in the portfolio."""
    filtered = []
    for item in data:
        if all(m in item["rewards"] for m in models):
            filtered.append({
                "prompt": item["prompt"],
                "rewards": {m: item["rewards"][m] for m in models},
            })
    return filtered


# =============================================================================
# WARMUP PRIOR GENERATION (full-information ridge regression)
# =============================================================================

def compute_reward_bounds(data: List[Dict], models: List[str]) -> Tuple[float, float]:
    """Compute reward normalization bounds from full-information data.

    Returns (r_min, r_max) across all models and prompts.  These bounds
    should be computed ONCE from the prior split (which observes all arms)
    and reused for both prior building and burn-in normalization, ensuring
    the reward scale is consistent across phases.
    """
    all_rewards = [
        data_item["rewards"][m]
        for data_item in data
        for m in models
        if m in data_item["rewards"]
    ]
    return min(all_rewards), max(all_rewards)


def build_warmup_priors(
    data: List[Dict],
    embeddings: List[np.ndarray],
    models: List[str],
    ridge_lambda: float = 1.0,
    reward_bounds: Optional[Tuple[float, float]] = None,
) -> Dict:
    """
    Build warmup priors for a set of models using full-information rewards.

    For each model m:
        A_m = λI + Σᵢ xᵢxᵢᵀ
        b_m = Σᵢ norm(rᵢ^(m)) xᵢ

    where rᵢ^(m) is the reward of model m on prompt i (observed for ALL models),
    normalized to [0, 1] using reward_bounds to match the online burn-in scale.

    Args:
        data: List of prompt dicts with "rewards" mapping model→reward
        embeddings: Pre-computed context vectors aligned with data
        models: List of model IDs to build priors for
        ridge_lambda: Regularization strength (default: 1.0)
        reward_bounds: Optional (r_min, r_max) for min-max normalization.
            If provided, rewards are normalized: r' = (r - r_min) / (r_max - r_min).
            This ensures prior theta is in the same scale as online updates.

    Returns:
        Dict with keys: A, b, models, context_dim, n_prompts, reward_bounds
    """
    dim = len(embeddings[0])
    A = {m: ridge_lambda * np.eye(dim) for m in models}
    b = {m: np.zeros(dim) for m in models}

    # Set up normalization
    if reward_bounds is not None:
        r_min, r_max = reward_bounds
        r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    else:
        r_min, r_range = 0.0, 1.0  # No normalization

    for i, item in enumerate(data):
        x = embeddings[i]
        xxT = np.outer(x, x)
        for m in models:
            if m in item["rewards"]:
                raw_reward = item["rewards"][m]
                reward = (raw_reward - r_min) / r_range  # Normalized to [0, 1]
                A[m] += xxT
                b[m] += reward * x

    return {
        "A": A,
        "b": b,
        "models": models,
        "context_dim": dim,
        "n_prompts": len(data),
        "reward_bounds": (r_min, r_min + r_range),  # Store for downstream use
    }


def normalize_prior_strength(priors: Dict, target_sample_size: float = 10.0) -> Dict:
    """Normalize prior strength to a specific effective sample size."""
    dim = priors["context_dim"]
    new_priors = copy.deepcopy(priors)

    for m in priors["A"]:
        current_mass = np.trace(priors["A"][m]) / dim
        scale = target_sample_size / max(current_mass, 1e-6)
        new_priors["A"][m] = priors["A"][m] * scale
        new_priors["b"][m] = priors["b"][m] * scale

    return new_priors


# =============================================================================
# ROUTING METHODS
# =============================================================================

def oracle_routing(data: List[Dict], models: List[str],
                   model_costs: Dict) -> Tuple[float, float]:
    """Oracle: select the best model for each prompt."""
    total_r, total_c = 0.0, 0.0
    for item in data:
        best_m = max(models, key=lambda m: item["rewards"].get(m, 0))
        total_r += item["rewards"][best_m]
        total_c += model_costs[best_m]
    n = len(data)
    return total_r / n, total_c / n


def run_bandit_method(
    method_name: str,
    router_factory,
    train_data, eval_data,
    train_emb, eval_emb,
    models, model_costs_norm,
    model_costs_raw,
    reward_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Generic runner for any router that follows select_model/update API.

    Args:
        reward_bounds: (r_min, r_max) for min-max normalization.  When provided,
            these bounds are used instead of computing from train_data.  This
            ensures burn-in uses the SAME reward scale as the warmup priors
            (both computed from the prior split).
    """
    router = router_factory()

    # Normalization bounds — use consistent bounds when provided
    if reward_bounds is not None:
        r_min, r_max = reward_bounds
    else:
        all_raw = [r for p in train_data for r in p["rewards"].values()]
        r_min, r_max = min(all_raw), max(all_raw)
    r_range = r_max - r_min if (r_max - r_min) > 1e-6 else 1.0
    burn_in = len(train_data)

    # Phase 1: Burn-in on dev set (bandit feedback only)
    for i, p in enumerate(train_data):
        x = train_emb[i]
        result = router.select_model(x, total_steps=burn_in)
        if isinstance(result, tuple):
            sel, token = result
        else:
            sel, token = result, None

        norm_r = (p["rewards"].get(sel, 0.0) - r_min) / r_range
        if token is not None and hasattr(router, "update") and "selection_token" in router.update.__code__.co_varnames:
            router.update(x, sel, norm_r, selection_token=token)
        else:
            router.update(x, sel, norm_r)

    # Switch ε-greedy to exploitation mode before evaluation
    if hasattr(router, "_training"):
        router._training = False

    # Phase 2: Evaluate on holdout (pure exploitation, no updates)
    total_r, total_c = 0.0, 0.0
    for i, p in enumerate(eval_data):
        x = eval_emb[i]
        result = router.select_model(x, total_steps=burn_in)
        sel = result[0] if isinstance(result, tuple) else result
        total_r += p["rewards"].get(sel, 0.0)
        total_c += model_costs_raw.get(sel, 0.005)

    n = len(eval_data)
    return total_r / n, total_c / n


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_k_experiment(
    K: int,
    models: List[str],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_priors: Dict,
    model_costs_raw: Dict,
    model_costs_norm: Dict,
    reward_bounds: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Run all methods for a given K and return results.

    Args:
        reward_bounds: (r_min, r_max) computed from the prior split.  Used for
            consistent reward normalization in both prior building and burn-in.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"K = {K} MODELS")
    logger.info(f"{'='*70}")
    logger.info(f"  Models: {models}")
    logger.info(f"  Burn-in: {len(train_data)} prompts, Eval: {len(eval_data)} prompts")
    logger.info(f"  Obs/arm (uniform): {len(train_data) // K}")

    dim = len(train_emb[0])
    results = {}

    # Oracle
    oracle_r, oracle_c = oracle_routing(eval_data, models, model_costs_raw)
    results["Oracle"] = {"reward": oracle_r, "cost": oracle_c}
    logger.info(f"  Oracle: R={oracle_r:.4f}, C=${oracle_c:.6f}")

    # Best static model
    best_static_r, best_static_m = 0.0, models[0]
    for m in models:
        r = np.mean([p["rewards"][m] for p in eval_data])
        if r > best_static_r:
            best_static_r, best_static_m = r, m
    results["Best Static"] = {
        "reward": best_static_r,
        "cost": model_costs_raw[best_static_m],
        "model": best_static_m,
    }
    logger.info(f"  Best Static: R={best_static_r:.4f} ({best_static_m.split('/')[-1]})")

    # Normalized priors for this portfolio
    priors_k = normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE)

    # Methods to test
    methods = {
        "Random": lambda: _RandomRouter(models, model_costs_norm),
        "ε-greedy (no priors)": lambda: _EpsGreedyWrapper(
            CostAwareTabulaRasaRouter(
                models=models, context_dim=dim, model_costs=model_costs_norm,
                alpha_start=ALPHA_START, alpha_end=ALPHA_END,
            ), epsilon=0.1, models=models,
        ),
        "LinUCB (no priors)": lambda: CostAwareTabulaRasaRouter(
            models=models, context_dim=dim, model_costs=model_costs_norm,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END,
        ),
        "LinUCB (w/ priors)": lambda: CostAwareLinUCBRouter(
            models=models, warmup_priors=priors_k, model_costs=model_costs_norm,
            alpha_start=ALPHA_START, alpha_end=ALPHA_END,
        ),
        "LinTS (no priors)": lambda: CostAwareLinTSRouter(
            models=models, context_dim=dim, model_costs=model_costs_norm,
            noise_variance=0.25,
        ),
        "LinTS (w/ priors)": lambda: CostAwareLinTSRouter(
            models=models, context_dim=dim, model_costs=model_costs_norm,
            noise_variance=0.25, warmup_priors=priors_k,
        ),
        "banditGPT-Hybrid": lambda: CorrallingRouter(
            experts=[
                CostAwareLinUCBRouter(
                    models=models, warmup_priors=normalize_prior_strength(warmup_priors, TARGET_SAMPLE_SIZE),
                    model_costs=model_costs_norm,
                    alpha_start=ALPHA_START, alpha_end=ALPHA_END,
                ),
                CostAwareTabulaRasaRouter(
                    models=models, context_dim=dim, model_costs=model_costs_norm,
                    alpha_start=ALPHA_START, alpha_end=ALPHA_END,
                ),
            ],
            models=models,
            learning_rate=1.0,
        ),
    }

    for method_name, factory in methods.items():
        trial_rewards = []
        trial_costs = []

        for trial in range(N_TRIALS):
            np.random.seed(SEED_OFFSET + trial)
            try:
                r, c = run_bandit_method(
                    method_name, factory,
                    train_data, eval_data, train_emb, eval_emb,
                    models, model_costs_norm, model_costs_raw,
                    reward_bounds=reward_bounds,
                )
                trial_rewards.append(r)
                trial_costs.append(c)
            except Exception as e:
                logger.warning(f"    {method_name} trial {trial} failed: {e}")

        if trial_rewards:
            avg_r = np.mean(trial_rewards)
            std_r = np.std(trial_rewards, ddof=1) if len(trial_rewards) > 1 else 0.0
            ci95_r = 1.96 * std_r / np.sqrt(len(trial_rewards))
            avg_c = np.mean(trial_costs)
            results[method_name] = {
                "reward": avg_r,
                "reward_std": std_r,
                "reward_ci95": ci95_r,
                "cost": avg_c,
                "n_trials": len(trial_rewards),
            }
            logger.info(
                f"  {method_name:<25} R={avg_r:.4f}±{ci95_r:.4f}  C=${avg_c:.6f}"
            )

    # ---- Derived metrics ----
    oracle_r = results["Oracle"]["reward"]
    static_r = results["Best Static"]["reward"]
    static_c = results["Best Static"]["cost"]
    random_r = results["Random"]["reward"] if "Random" in results else 0.0

    for method_name in methods:
        if method_name not in results:
            continue
        mr = results[method_name]["reward"]
        mc = results[method_name]["cost"]

        # 1. Simple Regret: Oracle_R - Method_R (lower is better)
        results[method_name]["regret"] = oracle_r - mr

        # 2. Normalized Learning Lift: (Method_R - Random_R) / (Oracle_R - Random_R)
        #    Fraction of the learnable gap (Oracle − Random) captured by learning.
        learnable_gap = oracle_r - random_r
        if learnable_gap > 1e-6:
            results[method_name]["learning_lift"] = (mr - random_r) / learnable_gap
        else:
            results[method_name]["learning_lift"] = 0.0

        # 3a. Cost Savings vs Best Static: 1 - Method_C / BestStatic_C
        if static_c > 1e-12:
            results[method_name]["cost_savings_vs_static"] = 1.0 - mc / static_c
        else:
            results[method_name]["cost_savings_vs_static"] = 0.0

    return results


# =============================================================================
# HELPER ROUTERS
# =============================================================================

class _RandomRouter:
    """Uniform random routing (no learning)."""
    def __init__(self, models, model_costs):
        self.models = models
    def select_model(self, context, total_steps=0):
        return np.random.choice(self.models)
    def update(self, context, model, reward, **kwargs):
        pass


class _EpsGreedyWrapper:
    """ε-greedy wrapper: random exploration + greedy exploitation.

    During training: with probability ε, select uniformly at random;
    otherwise, select the model with highest predicted reward (greedy,
    no UCB bonus — alpha forced to 0 on the inner router).

    During evaluation: pure greedy (ε=0).
    """
    def __init__(self, inner, epsilon, models):
        self.inner = inner
        self.epsilon = epsilon
        self.models = models
        self._training = True

    def select_model(self, context, total_steps=0):
        if self._training and np.random.random() < self.epsilon:
            return np.random.choice(self.models)
        # Greedy: pass total_steps=0 so alpha=alpha_end (minimal exploration)
        # This makes exploitation close to pure greedy, as ε handles exploration.
        return self.inner.select_model(context, total_steps=0)

    def update(self, context, model, reward, **kwargs):
        self.inner.update(context, model, reward)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("K-SCALING EXPERIMENT: PORTFOLIO SIZE vs ROUTING PERFORMANCE")
    logger.info("=" * 70)

    # Load data — use min_models=2 so we keep ALL usable prompts.
    # filter_portfolio() will handle per-K completeness downstream.
    # This matters because 1,121 prompts have all K≤20 models but only
    # 888 have all 43 models. Loading with min_models=43 would discard
    # 233 prompts that are perfectly usable for K≤20 portfolios.
    logger.info("\n--- Loading multi-model data ---")
    dev_data_raw = load_all_models_data(DEV_DATA_PATH_ALL_MODELS, min_models=2)
    holdout_data = load_all_models_data(HOLDOUT_DATA_PATH_ALL_MODELS, min_models=2)
    logger.info(f"  Dev (raw): {len(dev_data_raw)} prompts")
    logger.info(f"  Holdout (raw): {len(holdout_data)} prompts")

    # Discover all models from the broadest coverage prompts
    all_models = sorted(
        set().union(*(set(p["rewards"].keys()) for p in dev_data_raw))
    )
    logger.info(f"  Total unique models: {len(all_models)}")

    # Build K=41 portfolio: exclude 2 models with incomplete dev coverage.
    # These 2 models (gpt-5, gemini-2.5-flash-preview-09-2025) were evaluated
    # on a separate 233-prompt batch that only contains those 2 models.
    # Dropping them recovers 1121 complete dev prompts (vs 888 for K=43).
    core_models = sorted(m for m in all_models if m not in EXCLUDED_MODELS)
    PORTFOLIOS[41] = core_models
    logger.info(f"  Core models (K=41, excl {len(EXCLUDED_MODELS)} incomplete): {len(core_models)}")

    # Show per-K available data before splitting
    for K in sorted(PORTFOLIOS.keys()):
        models_k = PORTFOLIOS[K]
        n_dev_k = sum(1 for p in dev_data_raw if all(m in p["rewards"] for m in models_k))
        n_hold_k = sum(1 for p in holdout_data if all(m in p["rewards"] for m in models_k))
        logger.info(f"  K={K:>2}: dev={n_dev_k}, holdout={n_hold_k} (complete prompts)")

    # =========================================================================
    # SPLIT DEV DATA INTO DISJOINT PRIOR / BURN-IN HALVES
    # =========================================================================
    # This prevents double-dipping: the prior-building phase and burn-in phase
    # see entirely different prompts. Analogous to how the K=2 paper uses
    # RouteLLM Arena battles (different source) for priors vs dev rewards for
    # burn-in. Here both halves come from the same source but prompts are
    # disjoint, which is the same relationship as dev↔holdout.
    np.random.seed(SEED_OFFSET)
    n_dev = len(dev_data_raw)
    perm = np.random.permutation(n_dev)
    split_idx = n_dev // 2
    prior_indices = sorted(perm[:split_idx])
    burnin_indices = sorted(perm[split_idx:])

    prior_data = [dev_data_raw[i] for i in prior_indices]
    burnin_data = [dev_data_raw[i] for i in burnin_indices]

    # Sanity check: no prompt overlap
    prior_prompts = {p["prompt"] for p in prior_data}
    burnin_prompts = {p["prompt"] for p in burnin_data}
    assert len(prior_prompts & burnin_prompts) == 0, "Prior/burn-in prompt overlap!"
    logger.info(f"\n  Prior split: {len(prior_data)} prompts (for warmup priors)")
    logger.info(f"  Burn-in split: {len(burnin_data)} prompts (for bandit training)")

    # Load encoder and PCA
    logger.info("\n--- Loading encoder and PCA ---")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    logger.info(f"  Encoder: {DEFAULT_SENTENCE_TRANSFORMER}")
    logger.info(f"  PCA: {pca.n_components_} components → dim={pca.n_components_ + 1}")

    # Pre-compute embeddings for all three splits
    logger.info("\n--- Pre-computing embeddings ---")
    t0 = time.time()
    prior_emb = [embed_prompt(p["prompt"], encoder, pca) for p in prior_data]
    burnin_emb = [embed_prompt(p["prompt"], encoder, pca) for p in burnin_data]
    holdout_emb = [embed_prompt(p["prompt"], encoder, pca) for p in holdout_data]
    logger.info(
        f"  Encoded {len(prior_emb)} (prior) + {len(burnin_emb)} (burn-in) "
        f"+ {len(holdout_emb)} (holdout) prompts in {time.time()-t0:.1f}s"
    )

    # Compute raw costs
    raw_costs = {m: compute_cost_per_request(m) for m in all_models}

    # Build prompt→index maps once (used to align filtered data with embeddings)
    prior_prompt_idx = {p["prompt"]: i for i, p in enumerate(prior_data)}
    burnin_prompt_idx = {p["prompt"]: i for i, p in enumerate(burnin_data)}
    holdout_prompt_idx = {p["prompt"]: i for i, p in enumerate(holdout_data)}

    # Run experiments for each K
    all_k_results = {}
    t_start = time.time()

    for K in [2, 5, 10, 20, 41]:
        models_k = PORTFOLIOS[K]

        # Filter each split to this portfolio (keep only prompts with all K models)
        prior_k = filter_portfolio(prior_data, models_k)
        burnin_k = filter_portfolio(burnin_data, models_k)
        eval_k = filter_portfolio(holdout_data, models_k)

        if len(burnin_k) < 50 or len(eval_k) < 50:
            logger.warning(
                f"  K={K}: insufficient data "
                f"(prior={len(prior_k)}, burn-in={len(burnin_k)}, eval={len(eval_k)}). "
                f"Skipping."
            )
            continue

        # Align filtered prompts with their pre-computed embeddings
        prior_emb_k = [prior_emb[prior_prompt_idx[p["prompt"]]] for p in prior_k]
        burnin_emb_k = [burnin_emb[burnin_prompt_idx[p["prompt"]]] for p in burnin_k]
        eval_emb_k = [holdout_emb[holdout_prompt_idx[p["prompt"]]] for p in eval_k]

        # Compute reward normalization bounds from prior split (full information).
        # These bounds are used for BOTH prior building AND burn-in normalization,
        # ensuring the reward scale is identical across phases.
        r_bounds_k = compute_reward_bounds(prior_k, models_k)
        logger.info(f"\n  Reward bounds (prior split): [{r_bounds_k[0]:.4f}, {r_bounds_k[1]:.4f}]")

        # Build warmup priors from PRIOR SPLIT ONLY (disjoint from burn-in)
        logger.info(f"  Building warmup priors for K={K} from prior split ({len(prior_k)} prompts)...")
        warmup_k = build_warmup_priors(prior_k, prior_emb_k, models_k, reward_bounds=r_bounds_k)
        logger.info(
            f"  Priors built: {len(warmup_k['A'])} models, "
            f"n_eff={np.trace(list(warmup_k['A'].values())[0]) / warmup_k['context_dim']:.0f}"
        )

        # Normalize costs
        costs_k = {m: raw_costs[m] for m in models_k}
        max_c = max(costs_k.values())
        min_c = min(costs_k.values())
        c_range = max_c - min_c if max_c > min_c else 1.0
        norm_costs = {
            m: {"cost": costs_k[m], "normalized_cost": (costs_k[m] - min_c) / c_range}
            for m in models_k
        }

        # Run all methods: burn-in on BURN-IN SPLIT, evaluate on HOLDOUT
        k_results = run_k_experiment(
            K, models_k, burnin_k, eval_k, burnin_emb_k, eval_emb_k,
            warmup_k, costs_k, norm_costs,
            reward_bounds=r_bounds_k,
        )
        # Record data sizes for provenance
        k_results["_data"] = {
            "n_prior": len(prior_k),
            "n_burnin": len(burnin_k),
            "n_eval": len(eval_k),
        }
        all_k_results[K] = k_results

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL K-SCALING EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    logger.info(f"{'='*70}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "k_scaling_results.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    save_data = {
        "metadata": {
            "description": "K-scaling experiment: portfolio size vs routing performance",
            "protocol": (
                "Dev data split 50/50 into disjoint prior/burn-in halves. "
                "Priors built from prior split (full-information ridge regression). "
                "Burn-in training on burn-in split (bandit feedback only). "
                "Evaluation on holdout (no overlap with either dev half)."
            ),
            "n_dev_raw": len(dev_data_raw),
            "n_dev_prior_split": len(prior_data),
            "n_dev_burnin_split": len(burnin_data),
            "n_holdout": len(holdout_data),
            "n_trials": N_TRIALS,
            "alpha_schedule": f"{ALPHA_START}→{ALPHA_END}",
            "prior_neff": TARGET_SAMPLE_SIZE,
            "portfolios": {str(k): v for k, v in PORTFOLIOS.items()},
        },
        "results": {
            str(k): {
                method: {key: to_serializable(val) for key, val in mdata.items()}
                for method, mdata in kdata.items()
            }
            for k, kdata in all_k_results.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, default=to_serializable)
    logger.info(f"\nSaved: {output_file}")

    # =====================================================================
    # SUMMARY TABLES
    # =====================================================================

    sorted_ks = sorted(all_k_results.keys())
    bandit_methods = ["Random", "ε-greedy (no priors)",
                      "LinUCB (no priors)", "LinUCB (w/ priors)",
                      "LinTS (no priors)", "LinTS (w/ priors)",
                      "banditGPT-Hybrid"]
    all_methods = ["Oracle", "Best Static"] + bandit_methods

    def _print_table(title, methods, metric_key, fmt, higher_better=True,
                      suffix=""):
        """Print a summary table for a given metric across K values."""
        logger.info(f"\n{'='*90}")
        direction = "(higher is better)" if higher_better else "(lower is better)"
        logger.info(f"{title}  {direction}")
        logger.info(f"{'='*90}")
        header = f"{'Method':<25}"
        for K in sorted_ks:
            header += f" | K={K:<6}"
        logger.info(header)
        logger.info("-" * 90)
        for method in methods:
            row = f"{method:<25}"
            for K in sorted_ks:
                if method in all_k_results[K] and metric_key in all_k_results[K][method]:
                    val = all_k_results[K][method][metric_key]
                    row += f" | {val:{fmt}}{suffix}"
                else:
                    row += f" |    —   "
            logger.info(row)

    # Table 1: Reward
    _print_table("REWARD BY METHOD AND K", all_methods, "reward", ".4f")

    # Table 2: Simple Regret (Oracle_R - Method_R)
    _print_table("SIMPLE REGRET (Oracle - Method)", bandit_methods, "regret", ".4f",
                 higher_better=False)

    # Table 3: Normalized Learning Lift
    # Convert to percentage for display
    for K in sorted_ks:
        for m in bandit_methods:
            if m in all_k_results[K] and "learning_lift" in all_k_results[K][m]:
                all_k_results[K][m]["learning_lift_pct"] = (
                    all_k_results[K][m]["learning_lift"] * 100
                )
    _print_table(
        "NORMALIZED LEARNING LIFT (Method-Random)/(Oracle-Random)",
        bandit_methods, "learning_lift_pct", "5.1f", suffix="%",
    )

    # Table 4: Cost savings vs Best Static
    for K in sorted_ks:
        for m in bandit_methods:
            if m in all_k_results[K] and "cost_savings_vs_static" in all_k_results[K][m]:
                all_k_results[K][m]["cost_savings_pct"] = (
                    all_k_results[K][m]["cost_savings_vs_static"] * 100
                )
    _print_table(
        "COST SAVINGS vs BEST STATIC (1 - Method_C / Static_C)",
        bandit_methods, "cost_savings_pct", "+5.1f", suffix="%",
    )

    # Table 5: Cost-quality summary (compact)
    logger.info(f"\n{'='*90}")
    logger.info("COST-QUALITY SUMMARY PER K")
    logger.info(f"{'='*90}")
    for K in sorted_ks:
        kr = all_k_results[K]
        oracle_r = kr["Oracle"]["reward"]
        static_r = kr["Best Static"]["reward"]
        static_c = kr["Best Static"]["cost"]
        static_m = kr["Best Static"].get("model", "?").split("/")[-1]
        logger.info(
            f"\n  K={K}: Oracle R={oracle_r:.4f} | "
            f"Best Static R={static_r:.4f} ({static_m}, ${static_c:.6f}/req)"
        )
        for m in ["LinUCB (w/ priors)", "banditGPT-Hybrid"]:
            if m in kr:
                mr = kr[m]["reward"]
                mc = kr[m]["cost"]
                ll = kr[m].get("learning_lift", 0) * 100
                cs = kr[m].get("cost_savings_vs_static", 0) * 100
                logger.info(
                    f"    {m:<25} R={mr:.4f}  C=${mc:.6f}  "
                    f"LearningLift={ll:5.1f}%  CostSavings={cs:+5.1f}%"
                )

    logger.info(f"\n{'='*90}")


if __name__ == "__main__":
    main()
