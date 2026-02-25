"""
Shared utilities for multi-model experiments (K=5/10).

Provides model catalog, portfolio definitions, data loading, embedding,
cost normalization, and baseline implementations shared across
04_figure, 07_figure, and 08_figure.
"""

import sys
import math
import gzip
import json
import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CATALOG & PORTFOLIOS
# ============================================================================

def _req_cost(inp, out):
    return (100 * inp + 400 * out) / 1_000_000

MODEL_CATALOG = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        "input_cost_per_m": 0.05, "output_cost_per_m": 0.05,
        "cost": _req_cost(0.05, 0.05),
        "tier": "cheap", "provider": "meta",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        "input_cost_per_m": 0.54, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.54, 0.60),
        "tier": "cheap", "provider": "mistral",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        "input_cost_per_m": 0.10, "output_cost_per_m": 0.10,
        "cost": _req_cost(0.10, 0.10),
        "tier": "cheap", "provider": "google",
    },
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        "input_cost_per_m": 0.80, "output_cost_per_m": 4.00,
        "cost": _req_cost(0.80, 4.00),
        "tier": "mid", "provider": "anthropic",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        "input_cost_per_m": 0.27, "output_cost_per_m": 1.10,
        "cost": _req_cost(0.27, 1.10),
        "tier": "mid", "provider": "deepseek",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        "input_cost_per_m": 0.15, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.15, 0.60),
        "tier": "mid", "provider": "google",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        "input_cost_per_m": 0.20, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.20, 0.60),
        "tier": "mid", "provider": "meta",
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        "input_cost_per_m": 3.00, "output_cost_per_m": 15.00,
        "cost": _req_cost(3.00, 15.00),
        "tier": "expensive", "provider": "anthropic",
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        "input_cost_per_m": 10.00, "output_cost_per_m": 30.00,
        "cost": _req_cost(10.00, 30.00),
        "tier": "expensive", "provider": "openai",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        "input_cost_per_m": 2.00, "output_cost_per_m": 8.00,
        "cost": _req_cost(2.00, 8.00),
        "tier": "expensive", "provider": "openai",
    },
}

PORTFOLIO_K5 = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "google/gemini-2.5-flash-preview-09-2025",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]

PORTFOLIO_K10 = PORTFOLIO_K5 + [
    "meta-llama/llama-4-maverick",
    "google/gemma-3-27b-it",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4-turbo",
    "deepseek/deepseek-chat-v3-0324",
]

# Experiment parameters shared across experiments
N_TRIALS = 20
SEED_OFFSET = 42
TARGET_NEFF = 10.0
ALPHA_START = 2.0
CORRALLING_LR = 0.1
CORRALLING_GAMMA = 0.05

# ============================================================================
# COST NORMALIZATION
# ============================================================================

MARKET_COST_FLOOR = 0.0001
MARKET_COST_CEILING = 0.04
_LOG_FLOOR = math.log(MARKET_COST_FLOOR)
_LOG_RANGE = math.log(MARKET_COST_CEILING) - _LOG_FLOOR


def compute_normalized_cost(input_cost_per_m: float, output_cost_per_m: float) -> float:
    """Match BanditRouter._calculate_absolute_penalty for fair cost comparison."""
    avg_per_1k = ((input_cost_per_m + output_cost_per_m) / 2.0) / 1000.0
    safe = max(avg_per_1k, MARKET_COST_FLOOR)
    return max(0.0, min(1.0, (math.log(safe) - _LOG_FLOOR) / _LOG_RANGE))


def build_model_registry(models: List[str]) -> Dict[str, Dict]:
    """Build model registry in the format BanditRouter.__init__ expects."""
    return {
        m: {
            "input_cost_per_m": MODEL_CATALOG[m]["input_cost_per_m"],
            "output_cost_per_m": MODEL_CATALOG[m]["output_cost_per_m"],
        }
        for m in models
    }


def build_lints_costs(models: List[str]) -> Dict[str, Dict]:
    """Build normalized cost dict in the format CostAwareLinTSRouter expects."""
    return {
        m: {"normalized_cost": compute_normalized_cost(
            MODEL_CATALOG[m]["input_cost_per_m"],
            MODEL_CATALOG[m]["output_cost_per_m"],
        )}
        for m in models
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_rewards(data_path: Path, prompts: List[str], models: List[str]) -> List[Dict]:
    """Load rewards for specific prompts and models from gzipped JSONL."""
    prompt_set = set(prompts)
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            p = entry["prompt"]
            m = entry["model_id"]
            if p in prompt_set and m in model_set:
                rewards[p][m] = entry["raw_score"]

    data = []
    for p in prompts:
        if p in rewards and len(rewards[p]) == len(models):
            data.append({"prompt": p, "rewards": rewards[p]})
    return data


def load_holdout_rewards(models: List[str]) -> List[Dict]:
    """Load all holdout prompts for a given set of models."""
    K = len(models)
    model_set = set(models)
    holdout_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    with gzip.open(HOLDOUT_DATA_PATH_ALL_MODELS, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok") and entry["model_id"] in model_set:
                holdout_rewards[entry["prompt"]][entry["model_id"]] = entry["raw_score"]
    return [
        {"prompt": p, "rewards": r}
        for p, r in holdout_rewards.items()
        if len(r) == K
    ]


def load_warmup_priors(models: List[str], warmup_path=None):
    """Load warmup priors from joblib, subsetting to the given models."""
    path = warmup_path or MULTIMODEL_WARMUP_PRIORS_PATH
    raw = joblib.load(path)
    return {
        "A": {m: raw["A"][m].copy() for m in models if m in raw["A"]},
        "b": {m: raw["b"][m].copy() for m in models if m in raw["b"]},
        "context_dim": raw["context_dim"],
    }


def load_multimodel_data(models: List[str]):
    """
    Full data pipeline: load splits, rewards, embeddings.

    Returns:
        (train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max)
    """
    from sentence_transformers import SentenceTransformer

    with open(THREE_WAY_SPLITS_PATH) as f:
        splits = json.load(f)
    online_prompts = splits["online_learn_pool"]

    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)

    train_data = load_rewards(DEV_DATA_PATH_ALL_MODELS, online_prompts, models)
    eval_data = load_holdout_rewards(models)

    logger.info(f"  Train: {len(train_data)} | Eval: {len(eval_data)} | dim: {pca.n_components_}+1")

    train_emb = [embed_prompt(p["prompt"], encoder, pca) for p in train_data]
    eval_emb = [embed_prompt(p["prompt"], encoder, pca) for p in eval_data]

    costs = {m: MODEL_CATALOG[m]["cost"] for m in models}

    all_raw = [s for p in train_data for m in models for s in [p["rewards"][m]]]
    r_min, r_max = min(all_raw), max(all_raw)

    return train_data, eval_data, train_emb, eval_emb, costs, r_min, r_max


# ============================================================================
# BASELINES
# ============================================================================

def oracle_route(eval_data, models, costs):
    r = c = 0.0
    for p in eval_data:
        best = max(models, key=lambda m: p["rewards"][m])
        r += p["rewards"][best]; c += costs[best]
    n = len(eval_data)
    return r / n, c / n


def static_route(eval_data, model, costs):
    r = sum(p["rewards"][model] for p in eval_data)
    c = costs[model] * len(eval_data)
    n = len(eval_data)
    return r / n, c / n


def random_route(eval_data, models, costs, seed=42):
    rng = np.random.RandomState(seed)
    r = c = 0.0
    for p in eval_data:
        m = models[rng.randint(len(models))]
        r += p["rewards"][m]; c += costs[m]
    n = len(eval_data)
    return r / n, c / n


def epsilon_greedy_route(train_data, eval_data, models, costs, epsilon=0.1, seed=42):
    rng = np.random.RandomState(seed)
    means = {m: np.mean([p["rewards"][m] for p in train_data]) for m in models}
    best = max(means, key=means.get)
    r = c = 0.0
    for p in eval_data:
        m = models[rng.randint(len(models))] if rng.random() < epsilon else best
        r += p["rewards"][m]; c += costs[m]
    n = len(eval_data)
    return r / n, c / n
