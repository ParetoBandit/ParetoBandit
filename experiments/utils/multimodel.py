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

from bandit_gpt.calibration import embed_prompt  # noqa: F401 — used as fallback
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models, req_cost

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CATALOG & PORTFOLIOS
# ============================================================================

_PRICES = get_prices_for_models(
    [
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "anthropic/claude-haiku-4.5",
        "deepseek/deepseek-chat-v3-0324",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-2.5-pro-preview-06-05",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "moonshotai/kimi-k2-0905",
        "openai/gpt-4.1",
        "openai/gpt-5.1",
    ]
)

MODEL_CATALOG = {
    # ── Cheap tier ──────────────────────────────────────────────────────
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES["meta-llama/llama-3.1-8b-instruct"],
        "cost": req_cost(
            _PRICES["meta-llama/llama-3.1-8b-instruct"]["input_cost_per_m"],
            _PRICES["meta-llama/llama-3.1-8b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap", "provider": "meta",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        **_PRICES["mistralai/mixtral-8x7b-instruct"],
        "cost": req_cost(
            _PRICES["mistralai/mixtral-8x7b-instruct"]["input_cost_per_m"],
            _PRICES["mistralai/mixtral-8x7b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap", "provider": "mistral",
    },
    "google/gemma-3-12b-it": {
        "display": "Gemma-3-12B",
        **_PRICES["google/gemma-3-12b-it"],
        "cost": req_cost(
            _PRICES["google/gemma-3-12b-it"]["input_cost_per_m"],
            _PRICES["google/gemma-3-12b-it"]["output_cost_per_m"],
        ),
        "tier": "cheap", "provider": "google",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        **_PRICES["google/gemma-3-27b-it"],
        "cost": req_cost(
            _PRICES["google/gemma-3-27b-it"]["input_cost_per_m"],
            _PRICES["google/gemma-3-27b-it"]["output_cost_per_m"],
        ),
        "tier": "cheap", "provider": "google",
    },
    # ── Mid tier ────────────────────────────────────────────────────────
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        **_PRICES["anthropic/claude-haiku-4.5"],
        "cost": req_cost(
            _PRICES["anthropic/claude-haiku-4.5"]["input_cost_per_m"],
            _PRICES["anthropic/claude-haiku-4.5"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "anthropic",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        **_PRICES["deepseek/deepseek-chat-v3-0324"],
        "cost": req_cost(
            _PRICES["deepseek/deepseek-chat-v3-0324"]["input_cost_per_m"],
            _PRICES["deepseek/deepseek-chat-v3-0324"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "deepseek",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        **_PRICES["google/gemini-2.5-flash-preview-09-2025"],
        "cost": req_cost(
            _PRICES["google/gemini-2.5-flash-preview-09-2025"]["input_cost_per_m"],
            _PRICES["google/gemini-2.5-flash-preview-09-2025"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "google",
    },
    "google/gemini-2.5-pro-preview-06-05": {
        "display": "Gemini-2.5-Pro",
        **_PRICES["google/gemini-2.5-pro-preview-06-05"],
        "cost": req_cost(
            _PRICES["google/gemini-2.5-pro-preview-06-05"]["input_cost_per_m"],
            _PRICES["google/gemini-2.5-pro-preview-06-05"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "google",
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "display": "Llama-3.1-70B",
        **_PRICES["meta-llama/llama-3.1-70b-instruct"],
        "cost": req_cost(
            _PRICES["meta-llama/llama-3.1-70b-instruct"]["input_cost_per_m"],
            _PRICES["meta-llama/llama-3.1-70b-instruct"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "meta",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        **_PRICES["meta-llama/llama-4-maverick"],
        "cost": req_cost(
            _PRICES["meta-llama/llama-4-maverick"]["input_cost_per_m"],
            _PRICES["meta-llama/llama-4-maverick"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "meta",
    },
    "meta-llama/llama-4-scout": {
        "display": "Llama-4-Scout",
        **_PRICES["meta-llama/llama-4-scout"],
        "cost": req_cost(
            _PRICES["meta-llama/llama-4-scout"]["input_cost_per_m"],
            _PRICES["meta-llama/llama-4-scout"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "meta",
    },
    # ── Expensive tier ──────────────────────────────────────────────────
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        **_PRICES["anthropic/claude-sonnet-4"],
        "cost": req_cost(
            _PRICES["anthropic/claude-sonnet-4"]["input_cost_per_m"],
            _PRICES["anthropic/claude-sonnet-4"]["output_cost_per_m"],
        ),
        "tier": "expensive", "provider": "anthropic",
    },
    "anthropic/claude-sonnet-4.5": {
        "display": "Claude-Sonnet-4.5",
        **_PRICES["anthropic/claude-sonnet-4.5"],
        "cost": req_cost(
            _PRICES["anthropic/claude-sonnet-4.5"]["input_cost_per_m"],
            _PRICES["anthropic/claude-sonnet-4.5"]["output_cost_per_m"],
        ),
        "tier": "expensive", "provider": "anthropic",
    },
    "moonshotai/kimi-k2-0905": {
        "display": "Kimi-K2",
        **_PRICES["moonshotai/kimi-k2-0905"],
        "cost": req_cost(
            _PRICES["moonshotai/kimi-k2-0905"]["input_cost_per_m"],
            _PRICES["moonshotai/kimi-k2-0905"]["output_cost_per_m"],
        ),
        "tier": "mid", "provider": "moonshot",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES["openai/gpt-4.1"],
        "cost": req_cost(
            _PRICES["openai/gpt-4.1"]["input_cost_per_m"],
            _PRICES["openai/gpt-4.1"]["output_cost_per_m"],
        ),
        "tier": "expensive", "provider": "openai",
    },
    "openai/gpt-5.1": {
        "display": "GPT-5.1",
        **_PRICES["openai/gpt-5.1"],
        "cost": req_cost(
            _PRICES["openai/gpt-5.1"]["input_cost_per_m"],
            _PRICES["openai/gpt-5.1"]["output_cost_per_m"],
        ),
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
    "moonshotai/kimi-k2-0905",
    "deepseek/deepseek-chat-v3-0324",
]

# Experiment parameters shared across experiments
N_TRIALS = 20
SEED_OFFSET = 42
TARGET_NEFF = 10.0
ALPHA_START = 0.5
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
                rewards[p][m] = extract_reward(entry)

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
                holdout_rewards[entry["prompt"]][entry["model_id"]] = extract_reward(entry)
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

    from utils.embeddings import load_embedding_cache, embed_dataset_cached
    _cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    train_data = load_rewards(DEV_DATA_PATH_ALL_MODELS, online_prompts, models)
    eval_data = load_holdout_rewards(models)

    logger.info(f"  Train: {len(train_data)} | Eval: {len(eval_data)} | dim: {pca.n_components_}+1")

    train_emb = embed_dataset_cached(train_data, _cache, encoder, pca)
    eval_emb = embed_dataset_cached(eval_data, _cache, encoder, pca)

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
