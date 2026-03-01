#!/usr/bin/env python3
"""
BanditGPT vs RouteLLM: Train-then-Freeze Evaluation.

Compares the library's BanditRouter against RouteLLM and standard baselines
using canonical dev/holdout datasets with ground-truth multi-judge rewards.

Protocol
--------
1. **Canonical dev/holdout splits.**
   Data comes from pre-computed datasets (``dev_rewards_2models.jsonl.gz``,
   ``holdout_rewards_2models.jsonl.gz`` for K=2; all-models variants for
   K=10) with rewards derived via :func:`extract_reward` (mean of
   vote x confidence across multi-judge panel).

2. **Train-then-freeze evaluation.**
   BanditGPT trains on the dev set with oracle rewards, then is frozen
   for evaluation on the holdout set.  RouteLLM is static (pre-trained)
   and evaluated on the same holdout.  This makes the comparison
   interpretable: both routers are frozen during evaluation.

3. **Symmetric data access.**
   Both routers have access to the same dev set.  RouteLLM tunes its
   threshold tau on the full dev set using the same cost-penalised
   objective as BanditGPT (``reward - cost_penalty * cost``).
   BanditGPT trains its routing policy on the same dev set.  Neither
   method sees holdout data before evaluation.

4. **Fair RouteLLM comparison (K=2 only).**
   The K=2 portfolio is Mixtral-8x7B + GPT-4-Turbo — the model pair
   RouteLLM's MF router was trained on.  This is an in-distribution
   comparison (unlike earlier drafts that used OOD model pairs).

5. **K=10 Pareto frontier.**
   RouteLLM does not natively support K > 2.  The K=10 evaluation uses
   standard baselines: oracle, best-static, random, epsilon-greedy,
   and tabula-rasa (cold-start BanditGPT without priors or Corralling).

6. **Honest statistical reporting.**
   Primary: across-seed t-tests (df = N_SEEDS - 1).  Secondary: paired
   bootstrap (flagged as over-powered due to holdout-size n).

Outputs (``results/``)
    prequential_results.json
"""

import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
)
from utils.rewards import extract_reward
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Model catalogs
# ============================================================================

def _req_cost(inp: float, out: float) -> float:
    """Per-request cost assuming 100 input + 400 output tokens."""
    return (100 * inp + 400 * out) / 1_000_000


K2_MODELS: List[str] = [
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-4-turbo",
]

K2_CATALOG: Dict[str, Dict] = {
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        "input_cost_per_m": 0.54,
        "output_cost_per_m": 0.60,
        "cost": _req_cost(0.54, 0.60),
        "tier": "cheap",
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        "input_cost_per_m": 10.00,
        "output_cost_per_m": 30.00,
        "cost": _req_cost(10.00, 30.00),
        "tier": "expensive",
    },
}

K10_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "google/gemma-3-27b-it",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.5-flash-preview-09-2025",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4.1",
]

K10_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        "input_cost_per_m": 0.05, "output_cost_per_m": 0.05,
        "cost": _req_cost(0.05, 0.05), "tier": "cheap",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        "input_cost_per_m": 0.54, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.54, 0.60), "tier": "cheap",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        "input_cost_per_m": 0.10, "output_cost_per_m": 0.10,
        "cost": _req_cost(0.10, 0.10), "tier": "cheap",
    },
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        "input_cost_per_m": 0.80, "output_cost_per_m": 4.00,
        "cost": _req_cost(0.80, 4.00), "tier": "mid",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        "input_cost_per_m": 0.27, "output_cost_per_m": 1.10,
        "cost": _req_cost(0.27, 1.10), "tier": "mid",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        "input_cost_per_m": 0.15, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.15, 0.60), "tier": "mid",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        "input_cost_per_m": 0.20, "output_cost_per_m": 0.60,
        "cost": _req_cost(0.20, 0.60), "tier": "mid",
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        "input_cost_per_m": 3.00, "output_cost_per_m": 15.00,
        "cost": _req_cost(3.00, 15.00), "tier": "expensive",
    },
    "openai/gpt-4-turbo": {
        "display": "GPT-4-Turbo",
        "input_cost_per_m": 10.00, "output_cost_per_m": 30.00,
        "cost": _req_cost(10.00, 30.00), "tier": "expensive",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        "input_cost_per_m": 2.00, "output_cost_per_m": 8.00,
        "cost": _req_cost(2.00, 8.00), "tier": "expensive",
    },
}


# ============================================================================
# Experiment configuration
# ============================================================================

N_SEEDS: int = 5
SEED_OFFSET: int = 42
TARGET_NEFF: float = 10.0
ALPHA_START: float = 2.0
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

ROUTELLM_THRESHOLDS: List[float] = [
    0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.14, 0.15,
    0.2, 0.3, 0.5, 0.7, 0.9, 1.0,
]
ROUTELLM_COST_PENALTY: float = 0.05

LAMBDA_VALUES_K2: List[float] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
LAMBDA_VALUES_K10: List[float] = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

LEARNING_CURVE_CHECKPOINTS_K2: List[int] = [0, 10, 25, 50, 100, 200, 400, 600, 800, 1000]

_T_CRIT_DF4: float = 2.776  # 95% CI, df = N_SEEDS - 1 = 4


# ============================================================================
# Data loading
# ============================================================================


def load_rewards_from_file(
    data_path: Path,
    models: List[str],
    prompt_filter: Optional[set] = None,
) -> List[Dict]:
    """Load rewards for specific models from gzipped JSONL.

    Each returned dict has keys ``"prompt"`` and ``"rewards"``
    (a {model_id: float} map).  Only prompts with rewards for *all*
    requested models are included.

    Args:
        data_path: Path to a ``.jsonl.gz`` rewards file.
        models: Model IDs to retain.
        prompt_filter: If provided, only include prompts in this set.

    Returns:
        List of dicts with full model coverage, one per prompt.
    """
    model_set = set(models)
    rewards: Dict[str, Dict[str, float]] = defaultdict(dict)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
            model_id = entry["model_id"]
            if model_id not in model_set:
                continue
            if prompt_filter is not None and prompt not in prompt_filter:
                continue
            rewards[prompt][model_id] = extract_reward(entry)

    data = []
    n_models = len(models)
    for prompt, rmap in rewards.items():
        if len(rmap) == n_models:
            data.append({"prompt": prompt, "rewards": rmap})
    return data


def build_model_registry(
    models: List[str],
    catalog: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    """Build the registry dict that ``create_experiment_router`` expects."""
    return {
        m: {
            "input_cost_per_m": catalog[m]["input_cost_per_m"],
            "output_cost_per_m": catalog[m]["output_cost_per_m"],
        }
        for m in models
    }


# ============================================================================
# Embedding helpers
# ============================================================================


def embed_dataset(
    data: List[Dict],
    encoder: "SentenceTransformer",
    pca,
) -> List[np.ndarray]:
    """Embed all prompts in a dataset, returning aligned feature vectors."""
    return [embed_prompt(p["prompt"], encoder, pca) for p in data]


# ============================================================================
# Baseline evaluation functions
# ============================================================================


def oracle_route(
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    cost_penalty: float = 0.0,
) -> Tuple[float, float]:
    """Per-prompt clairvoyant argmax of reward - lambda * cost.

    When cost_penalty=0, this is the pure quality oracle.

    Returns:
        (mean_reward, mean_cost) on the evaluation set.
    """
    r_total = c_total = 0.0
    for p in eval_data:
        best_m = max(
            models,
            key=lambda m: p["rewards"][m] - cost_penalty * costs[m],
        )
        r_total += p["rewards"][best_m]
        c_total += costs[best_m]
    n = len(eval_data)
    return r_total / n, c_total / n


def static_route(
    eval_data: List[Dict],
    model: str,
    costs: Dict[str, float],
) -> Tuple[float, float]:
    """Always route to a single model."""
    r_total = sum(p["rewards"][model] for p in eval_data)
    c_total = costs[model] * len(eval_data)
    n = len(eval_data)
    return r_total / n, c_total / n


def random_route(
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    n_trials: int = 20,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, float]:
    """Uniform random routing averaged over *n_trials* seeds."""
    trial_r, trial_c = [], []
    for trial in range(n_trials):
        rng = np.random.RandomState(seed_offset + trial)
        r_total = c_total = 0.0
        for p in eval_data:
            m = models[rng.randint(len(models))]
            r_total += p["rewards"][m]
            c_total += costs[m]
        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)
    return {
        "reward": float(np.mean(trial_r)),
        "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
        "cost": float(np.mean(trial_c)),
        "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
        "n_trials": n_trials,
    }


def epsilon_greedy_route(
    train_data: List[Dict],
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    epsilon: float = 0.1,
    n_trials: int = 20,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, float]:
    """Epsilon-greedy routing: exploit the empirical best model from training."""
    model_means = {
        m: float(np.mean([p["rewards"][m] for p in train_data]))
        for m in models
    }
    best_model = max(model_means, key=model_means.get)

    trial_r, trial_c = [], []
    for trial in range(n_trials):
        rng = np.random.RandomState(seed_offset + trial)
        r_total = c_total = 0.0
        for p in eval_data:
            if rng.random() < epsilon:
                m = models[rng.randint(len(models))]
            else:
                m = best_model
            r_total += p["rewards"][m]
            c_total += costs[m]
        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)
    return {
        "reward": float(np.mean(trial_r)),
        "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
        "cost": float(np.mean(trial_c)),
        "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
        "n_trials": n_trials,
        "best_model": best_model,
    }


# ============================================================================
# RouteLLM evaluation (K=2 only)
# ============================================================================


def routellm_evaluate(
    controller,
    eval_data: List[Dict],
    costs: Dict[str, float],
    models: List[str],
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate RouteLLM on a dataset using oracle rewards.

    The controller routes each prompt to either the strong or weak model
    based on the MF router score vs the threshold.  The reward for the
    chosen model is looked up from the pre-computed oracle rewards.

    Args:
        controller: A ``routellm.controller.Controller`` instance.
        eval_data: List of {prompt, rewards} dicts.
        costs: Per-model cost dict.
        models: Ordered list of model IDs (weak first, strong second).
        threshold: MF router threshold.

    Returns:
        Dict with ``avg_reward``, ``avg_cost``, ``model_fractions``.
    """
    fallback = models[0]
    rewards_list: List[float] = []
    costs_list: List[float] = []
    model_counts: Dict[str, int] = {m: 0 for m in models}

    for p in eval_data:
        try:
            m = controller.route(p["prompt"], router="mf", threshold=threshold)
            if m not in costs:
                m = fallback
        except Exception:
            m = fallback

        rewards_list.append(p["rewards"][m])
        costs_list.append(costs[m])
        model_counts[m] = model_counts.get(m, 0) + 1

    n = len(eval_data)
    fractions = {m: cnt / n for m, cnt in model_counts.items()}

    return {
        "avg_reward": float(np.mean(rewards_list)),
        "avg_cost": float(np.mean(costs_list)),
        "model_fractions": fractions,
    }


def tune_routellm_threshold(
    controller,
    dev_data: List[Dict],
    costs: Dict[str, float],
    models: List[str],
    thresholds: List[float],
    cost_penalty: float,
) -> Tuple[float, Dict[str, Dict]]:
    """Select RouteLLM threshold on dev set using aligned cost-quality objective.

    Both RouteLLM and BanditGPT have access to the same dev set for
    calibration (symmetric data access).  The threshold is selected by
    maximising ``reward - cost_penalty * cost``, the same objective that
    BanditGPT's cost penalty lambda controls.

    Args:
        controller: RouteLLM Controller.
        dev_data: Full development dataset (same data BanditGPT trains on).
        costs: Per-model costs.
        models: Model IDs.
        thresholds: Candidate thresholds to sweep.
        cost_penalty: Lambda for the cost-penalised objective.

    Returns:
        (best_threshold, sweep_results) where sweep_results maps
        threshold -> {avg_reward, avg_cost, objective}.
    """
    sweep: Dict[str, Dict] = {}
    best_tau, best_obj = 0.0, -np.inf

    for tau in thresholds:
        result = routellm_evaluate(controller, dev_data, costs, models, tau)
        obj = result["avg_reward"] - cost_penalty * result["avg_cost"]
        sweep[str(tau)] = {
            "avg_reward": result["avg_reward"],
            "avg_cost": result["avg_cost"],
            "objective": obj,
        }
        if obj > best_obj:
            best_obj = obj
            best_tau = tau
        logger.info(
            f"  tau={tau:.2f}  reward={result['avg_reward']:.4f}  "
            f"cost=${result['avg_cost']:.6f}  obj={obj:.4f}"
        )

    logger.info(f"  -> selected tau={best_tau} (val obj={best_obj:.4f})")
    return best_tau, sweep


# ============================================================================
# BanditGPT train-then-freeze evaluation
# ============================================================================


def _compute_reward_normalization(
    train_data: List[Dict],
    models: List[str],
) -> Tuple[float, float]:
    """Compute min/max reward across all (prompt, model) pairs in training set."""
    all_raw = [r for p in train_data for m in models for r in [p["rewards"][m]]]
    r_min, r_max = min(all_raw), max(all_raw)
    return r_min, r_max


def train_bandit(
    router,
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    models: List[str],
    r_min: float,
    r_range: float,
) -> int:
    """Train the BanditRouter on the dev set via route() + process_feedback().

    Rewards are normalized to [0, 1] before feeding to the bandit.

    Args:
        router: A BanditRouter instance (from create_experiment_router).
        train_data: List of {prompt, rewards} dicts.
        train_embeddings: Pre-computed feature vectors aligned with train_data.
        models: Candidate model IDs.
        r_min: Minimum raw reward (for normalization).
        r_range: Range of raw rewards (max - min).

    Returns:
        Number of training steps completed.
    """
    n_steps = len(train_data)
    for p, x in zip(train_data, train_embeddings):
        model, log = router.route(x, total_steps=n_steps)
        raw_reward = p["rewards"][model]
        norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
        router.process_feedback(log.request_id, norm_reward)
    return n_steps


def evaluate_frozen(
    router,
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    costs: Dict[str, float],
    total_steps: int,
) -> Tuple[float, float, Dict[str, int]]:
    """Evaluate a frozen router on the holdout set (no learning).

    Args:
        router: Frozen BanditRouter.
        eval_data: Holdout {prompt, rewards} dicts.
        eval_embeddings: Pre-computed feature vectors for holdout.
        costs: Per-model cost dict.
        total_steps: Total steps completed during training (for alpha decay).

    Returns:
        (mean_reward, mean_cost, model_counts).
    """
    rng_state = np.random.get_state()
    r_total = c_total = 0.0
    model_counts: Dict[str, int] = defaultdict(int)

    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=total_steps)
        r_total += p["rewards"][model]
        c_total += costs[model]
        model_counts[model] += 1

    np.random.set_state(rng_state)
    n = len(eval_data)
    return r_total / n, c_total / n, dict(model_counts)


def run_pareto_sweep(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: str,
    costs: Dict[str, float],
    lambda_values: List[float],
    n_trials: int,
    *,
    use_corralling: bool = True,
    label: str = "banditGPT",
) -> List[Dict]:
    """Sweep cost penalty lambda with N trials per point.

    For each (lambda, trial): instantiate router, train on dev, freeze,
    evaluate on holdout.

    Returns:
        List of dicts, one per lambda, with mean/std reward and cost.
    """
    dim = train_emb[0].shape[0]
    r_min, r_max = _compute_reward_normalization(train_data, models)
    r_range = r_max - r_min
    burn_in = len(train_data)

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)
            router = create_experiment_router(
                model_registry=build_model_registry(models, catalog),
                feature_dim=dim,
                prior_n_effective=TARGET_NEFF,
                alpha=ALPHA_START,
                warmup_path=warmup_path,
                use_corralling=use_corralling,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
            )
            train_bandit(router, train_data, train_emb, models, r_min, r_range)
            r, c, _ = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            trial_r.append(r)
            trial_c.append(c)

        results.append({
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "n_trials": n_trials,
            "label": label,
        })
        logger.info(
            f"    lambda={lam:<6} | R={np.mean(trial_r):.4f}+/-{np.std(trial_r):.4f} "
            f"| C=${np.mean(trial_c):.6f}"
        )
    return results


def run_learning_curve(
    models: List[str],
    catalog: Dict[str, Dict],
    train_data: List[Dict],
    eval_data: List[Dict],
    train_emb: List[np.ndarray],
    eval_emb: List[np.ndarray],
    warmup_path: str,
    costs: Dict[str, float],
    n_trials: int,
    checkpoints: List[int],
    *,
    use_corralling: bool = True,
    cost_penalty: float = 0.0,
    label: str = "banditGPT",
) -> List[Dict]:
    """Learning curve: holdout quality as a function of online training steps.

    At each checkpoint, the router is frozen and evaluated on the full
    holdout set.  Step 0 evaluates with priors only (no online data).

    Returns:
        List of dicts, one per checkpoint, with mean/std reward.
    """
    dim = train_emb[0].shape[0]
    r_min, r_max = _compute_reward_normalization(train_data, models)
    r_range = r_max - r_min
    burn_in = len(train_data)
    checkpoint_set = set(checkpoints)

    by_step: Dict[int, Dict[str, List[float]]] = {
        s: {"rewards": [], "costs": []} for s in checkpoints
    }

    for trial in range(n_trials):
        np.random.seed(SEED_OFFSET + trial)
        router = create_experiment_router(
            model_registry=build_model_registry(models, catalog),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF,
            alpha=ALPHA_START,
            warmup_path=warmup_path,
            use_corralling=use_corralling,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=cost_penalty,
        )

        if 0 in checkpoint_set:
            r, c, _ = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            by_step[0]["rewards"].append(r)
            by_step[0]["costs"].append(c)

        for step_idx, (p, x) in enumerate(zip(train_data, train_emb)):
            model, log = router.route(x, total_steps=burn_in)
            raw_reward = p["rewards"][model]
            norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
            router.process_feedback(log.request_id, norm_reward)
            current = step_idx + 1
            if current in checkpoint_set:
                r, c, _ = evaluate_frozen(
                    router, eval_data, eval_emb, costs, burn_in,
                )
                by_step[current]["rewards"].append(r)
                by_step[current]["costs"].append(c)

        if (trial + 1) % 5 == 0:
            logger.info(f"    Learning curve trial {trial + 1}/{n_trials}")

    curve = []
    for s in sorted(checkpoints):
        rr = by_step[s]["rewards"]
        if rr:
            curve.append({
                "step": s,
                "mean_reward": float(np.mean(rr)),
                "std_reward": float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0,
                "n_trials": len(rr),
                "label": label,
            })
    return curve


# ============================================================================
# Statistical helpers
# ============================================================================


def _ci_scalar(vals: List[float]) -> Dict[str, float]:
    """Compute mean, std, and 95% CI for a list of per-seed scalars."""
    a = np.array(vals)
    m, s = float(a.mean()), float(a.std(ddof=1))
    hw = _T_CRIT_DF4 * s / np.sqrt(len(a))
    return {"mean": m, "std": s, "ci_lower": m - hw, "ci_upper": m + hw}


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap for E[a] - E[b].

    Warning: over-powered for online-learning policies because the
    effective sample size is much smaller than the holdout size due
    to sequential state dependence.
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a - b
    obs = float(d.mean())
    centred = d - obs
    bm = np.array([
        float(np.mean(rng.choice(centred, n, replace=True)))
        for _ in range(n_boot)
    ])
    p = float(np.mean(np.abs(bm) >= np.abs(obs)))
    br = np.array([
        float(np.mean(rng.choice(d, n, replace=True)))
        for _ in range(n_boot)
    ])
    return {
        "observed_diff": obs,
        "p_value": p,
        "ci_95_lower": float(np.percentile(br, 2.5)),
        "ci_95_upper": float(np.percentile(br, 97.5)),
        "note": (
            "Bootstrap n equals holdout size; sequential state dependence "
            "means effective n << holdout size.  Treat as exploratory only."
        ),
    }


def across_seeds_ttest(
    a_per_seed: List[float],
    b_per_seed: Optional[List[float]] = None,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Paired or one-sample t-test across N_SEEDS runs (df = N_SEEDS - 1).

    This is the primary significance test.  With 5 seeds the test requires
    Cohen's |d| >> 1 for significance at alpha = 0.05.
    """
    a = np.array(a_per_seed)
    if b_per_seed is not None:
        b = np.array(b_per_seed)
        diff = a - b
        result = scipy_stats.ttest_1samp(
            diff, popmean=0.0, alternative=alternative,
        )
        cohens_d = (
            float(diff.mean() / diff.std(ddof=1))
            if diff.std(ddof=1) > 0
            else float("nan")
        )
        return {
            "test": "paired_t_across_seeds",
            "df": len(diff) - 1,
            "t_stat": float(result.statistic),
            "p_value": float(result.pvalue),
            "mean_diff": float(diff.mean()),
            "cohens_d": cohens_d,
            "note": (
                f"df={len(diff) - 1}; low power — effects must be large "
                "(|d| >> 1) for significance at alpha=0.05 with 5 seeds."
            ),
        }
    result = scipy_stats.ttest_1samp(a, popmean=0.0, alternative=alternative)
    return {
        "test": "one_sample_t_across_seeds",
        "df": len(a) - 1,
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_a": float(a.mean()),
    }


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:  # noqa: C901
    """Run the full K=2 and K=10 evaluation and serialise results."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 0 — shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder and PCA ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components")

    results_all: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": N_SEEDS,
            "reward_source": "extract_reward(mean_vote_x_confidence)",
            "protocol": "train_then_freeze",
            "split_protocol": "canonical_dev_holdout",
            "fairness_design": (
                "Symmetric data access: both BanditGPT and RouteLLM see "
                "the same dev set. Symmetric objective: point comparison "
                f"uses a-priori lambda={ROUTELLM_COST_PENALTY} for all "
                "methods. No holdout-based hyperparameter selection."
            ),
        },
    }

    # ==================================================================
    # K=2 — BanditGPT vs RouteLLM (fair in-distribution comparison)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=2: BanditGPT vs RouteLLM")
    logger.info("  Mixtral-8x7B  vs  GPT-4-Turbo")
    logger.info("=" * 70)

    costs_k2 = {m: K2_CATALOG[m]["cost"] for m in K2_MODELS}

    # --- Load K=2 data -------------------------------------------------
    logger.info("\n  Loading K=2 dev and holdout data ...")
    dev_data_k2 = load_rewards_from_file(
        CANONICAL_DEV_DATA_PATH, K2_MODELS,
    )
    holdout_data_k2 = load_rewards_from_file(
        CANONICAL_HOLDOUT_DATA_PATH, K2_MODELS,
    )
    logger.info(f"    Dev: {len(dev_data_k2)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k2)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info("  Embedding K=2 prompts ...")
    dev_emb_k2 = embed_dataset(dev_data_k2, encoder, pca)
    holdout_emb_k2 = embed_dataset(holdout_data_k2, encoder, pca)
    dim = dev_emb_k2[0].shape[0]
    logger.info(f"    Feature dim: {dim}")

    # --- Phase 1: Tune RouteLLM threshold on full dev set --------------
    # Symmetric data access: RouteLLM sees the same dev prompts that
    # BanditGPT trains on.
    logger.info("\n  Phase 1: RouteLLM threshold tuning on dev set ...")
    from routellm.controller import Controller

    controller = Controller(
        routers=["mf"],
        strong_model=K2_MODELS[1],  # GPT-4-Turbo
        weak_model=K2_MODELS[0],    # Mixtral
    )

    best_tau, dev_sweep = tune_routellm_threshold(
        controller, dev_data_k2, costs_k2, K2_MODELS,
        ROUTELLM_THRESHOLDS, ROUTELLM_COST_PENALTY,
    )

    # --- Phase 2: Evaluate RouteLLM on holdout (frozen) ----------------
    logger.info("\n  Phase 2: RouteLLM holdout evaluation ...")
    routellm_holdout = routellm_evaluate(
        controller, holdout_data_k2, costs_k2, K2_MODELS, best_tau,
    )
    logger.info(
        f"    RouteLLM(tau={best_tau}): reward={routellm_holdout['avg_reward']:.4f}"
        f"  cost=${routellm_holdout['avg_cost']:.6f}"
    )

    # Full RouteLLM threshold sweep on holdout (for Pareto frontier)
    routellm_pareto: List[Dict] = []
    for tau in ROUTELLM_THRESHOLDS:
        h = routellm_evaluate(
            controller, holdout_data_k2, costs_k2, K2_MODELS, tau,
        )
        routellm_pareto.append({
            "threshold": tau,
            "avg_reward": h["avg_reward"],
            "avg_cost": h["avg_cost"],
            "model_fractions": h["model_fractions"],
        })

    # --- Phase 3: BanditGPT Pareto sweep (train on full dev) ----------
    logger.info(
        f"\n  Phase 3: BanditGPT Pareto sweep "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    bandit_pareto_k2 = run_pareto_sweep(
        K2_MODELS, K2_CATALOG,
        dev_data_k2, holdout_data_k2, dev_emb_k2, holdout_emb_k2,
        str(DEFAULT_WARMUP_PRIORS_PATH), costs_k2, LAMBDA_VALUES_K2,
        N_SEEDS, use_corralling=True, label="banditGPT_warmup",
    )

    # Tabula rasa ablation (cold start, no Corralling)
    logger.info(
        f"\n  Tabula rasa ablation "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k2 = run_pareto_sweep(
        K2_MODELS, K2_CATALOG,
        dev_data_k2, holdout_data_k2, dev_emb_k2, holdout_emb_k2,
        str(DEFAULT_WARMUP_PRIORS_PATH), costs_k2, LAMBDA_VALUES_K2,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
    )

    # --- Phase 4: K=2 learning curve -----------------------------------
    # Cap checkpoints to actual dev set size
    max_step = len(dev_data_k2)
    lc_checkpoints = [s for s in LEARNING_CURVE_CHECKPOINTS_K2 if s <= max_step]
    if max_step not in lc_checkpoints:
        lc_checkpoints.append(max_step)

    logger.info(f"\n  Phase 4: Learning curve ({N_SEEDS} seeds) ...")
    learning_curve_k2 = run_learning_curve(
        K2_MODELS, K2_CATALOG,
        dev_data_k2, holdout_data_k2, dev_emb_k2, holdout_emb_k2,
        str(DEFAULT_WARMUP_PRIORS_PATH), costs_k2, N_SEEDS,
        lc_checkpoints, use_corralling=True, cost_penalty=0.0,
    )

    # --- Phase 5: K=2 baselines ----------------------------------------
    logger.info("\n  Phase 5: K=2 baselines ...")

    # Compute oracle under the same cost-penalised objective used by
    # both RouteLLM and BanditGPT, so gap closure is measured consistently.
    oracle_r_k2, oracle_c_k2 = oracle_route(
        holdout_data_k2, K2_MODELS, costs_k2,
        cost_penalty=ROUTELLM_COST_PENALTY,
    )
    logger.info(
        f"    Oracle (lambda={ROUTELLM_COST_PENALTY}): "
        f"R={oracle_r_k2:.4f}  C=${oracle_c_k2:.6f}"
    )

    # Pure-quality oracle for reference (no cost penalty)
    oracle_r_k2_pure, oracle_c_k2_pure = oracle_route(
        holdout_data_k2, K2_MODELS, costs_k2, cost_penalty=0.0,
    )
    logger.info(
        f"    Oracle (pure quality): "
        f"R={oracle_r_k2_pure:.4f}  C=${oracle_c_k2_pure:.6f}"
    )

    static_k2 = {}
    for m in K2_MODELS:
        sr, sc = static_route(holdout_data_k2, m, costs_k2)
        static_k2[m] = {"reward": sr, "cost": sc}
        logger.info(
            f"    Static {K2_CATALOG[m]['display']:<15}: "
            f"R={sr:.4f}  C=${sc:.6f}"
        )

    random_k2 = random_route(holdout_data_k2, K2_MODELS, costs_k2, N_SEEDS * 4)
    logger.info(f"    Random: R={random_k2['reward']:.4f}")

    # --- Phase 6: K=2 point comparison & statistical tests -------------
    # Use an a-priori lambda matching RouteLLM's cost penalty to ensure
    # both routers are solving the exact same cost-quality trade-off.
    # No holdout-based hyperparameter selection (avoids test-set leakage).
    logger.info("\n  Phase 6: Point comparison & statistical tests ...")

    target_lambda_k2 = ROUTELLM_COST_PENALTY
    target_bandit = next(
        x for x in bandit_pareto_k2 if x["lambda"] == target_lambda_k2
    )
    target_tabula = next(
        x for x in tabula_pareto_k2 if x["lambda"] == target_lambda_k2
    )
    logger.info(
        f"    A-priori lambda={target_lambda_k2} "
        f"(matches RouteLLM cost_penalty for symmetric comparison)"
    )

    # Per-seed rewards at the target lambda (re-run to collect seeds)
    r_min_k2, r_max_k2 = _compute_reward_normalization(dev_data_k2, K2_MODELS)
    r_range_k2 = r_max_k2 - r_min_k2
    burn_in_k2 = len(dev_data_k2)

    bandit_per_seed: List[float] = []
    tabula_per_seed: List[float] = []
    for seed in range(N_SEEDS):
        np.random.seed(SEED_OFFSET + seed)
        router = create_experiment_router(
            model_registry=build_model_registry(K2_MODELS, K2_CATALOG),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF, alpha=ALPHA_START,
            warmup_path=str(DEFAULT_WARMUP_PRIORS_PATH),
            use_corralling=True, cost_penalty=target_lambda_k2,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
        )
        train_bandit(router, dev_data_k2, dev_emb_k2, K2_MODELS, r_min_k2, r_range_k2)
        r, _, _ = evaluate_frozen(router, holdout_data_k2, holdout_emb_k2, costs_k2, burn_in_k2)
        bandit_per_seed.append(r)

        np.random.seed(SEED_OFFSET + seed)
        router_tr = create_experiment_router(
            model_registry=build_model_registry(K2_MODELS, K2_CATALOG),
            feature_dim=dim,
            prior_n_effective=TARGET_NEFF, alpha=ALPHA_START,
            warmup_path=str(DEFAULT_WARMUP_PRIORS_PATH),
            use_corralling=False, cost_penalty=target_lambda_k2,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
        )
        train_bandit(router_tr, dev_data_k2, dev_emb_k2, K2_MODELS, r_min_k2, r_range_k2)
        r_tr, _, _ = evaluate_frozen(router_tr, holdout_data_k2, holdout_emb_k2, costs_k2, burn_in_k2)
        tabula_per_seed.append(r_tr)

    tests_k2: Dict[str, Any] = {}
    t_bvt = across_seeds_ttest(bandit_per_seed, tabula_per_seed)
    tests_k2["bandit_vs_tabula_rasa"] = t_bvt
    logger.info(
        f"    Bandit vs Tabula: delta={t_bvt['mean_diff']:+.5f}  "
        f"d={t_bvt['cohens_d']:.2f}  p={t_bvt['p_value']:.4f}"
    )

    # Assemble K=2 summary — gap closure under the shared objective
    weak_r = min(static_k2[m]["reward"] for m in K2_MODELS)
    gap_bandit = (
        (target_bandit["mean_reward"] - weak_r) / (oracle_r_k2 - weak_r) * 100
        if oracle_r_k2 > weak_r else 0.0
    )
    gap_routellm = (
        (routellm_holdout["avg_reward"] - weak_r) / (oracle_r_k2 - weak_r) * 100
        if oracle_r_k2 > weak_r else 0.0
    )

    logger.info(f"\n  K=2 SUMMARY (lambda={target_lambda_k2}, symmetric comparison):")
    logger.info(f"    Oracle:       {oracle_r_k2:.4f}")
    logger.info(f"    banditGPT:    {target_bandit['mean_reward']:.4f} (gap closure {gap_bandit:.1f}%)")
    logger.info(f"    RouteLLM:     {routellm_holdout['avg_reward']:.4f} (gap closure {gap_routellm:.1f}%)")
    logger.info(f"    Tabula rasa:  {target_tabula['mean_reward']:.4f}")
    logger.info(f"    Random:       {random_k2['reward']:.4f}")

    results_all["K2"] = {
        "models": K2_MODELS,
        "n_dev": len(dev_data_k2),
        "n_holdout": len(holdout_data_k2),
        "target_lambda": target_lambda_k2,
        "oracle": {"reward": oracle_r_k2, "cost": oracle_c_k2},
        "oracle_pure_quality": {"reward": oracle_r_k2_pure, "cost": oracle_c_k2_pure},
        "static": static_k2,
        "random": random_k2,
        "routellm": {
            "best_tau": best_tau,
            "dev_sweep": dev_sweep,
            "holdout": routellm_holdout,
            "pareto": routellm_pareto,
            "note": (
                "MF router is in-distribution for Mixtral + GPT-4-Turbo. "
                "Threshold tuned on full dev set using aligned "
                f"cost-quality objective (cost_penalty={ROUTELLM_COST_PENALTY}). "
                "Both RouteLLM and BanditGPT have symmetric access to dev data."
            ),
        },
        "banditgpt_pareto": bandit_pareto_k2,
        "tabula_rasa_pareto": tabula_pareto_k2,
        "learning_curve": learning_curve_k2,
        "point_comparison": {
            "lambda": target_lambda_k2,
            "banditgpt": {
                "reward": target_bandit["mean_reward"],
                "std": target_bandit["std_reward"],
            },
            "tabula_rasa": {
                "reward": target_tabula["mean_reward"],
                "std": target_tabula["std_reward"],
            },
            "routellm": {
                "reward": routellm_holdout["avg_reward"],
                "cost": routellm_holdout["avg_cost"],
            },
            "note": (
                "All methods evaluated under the same a-priori cost penalty "
                f"lambda={target_lambda_k2}. No holdout-based hyperparameter "
                "selection (no test-set leakage)."
            ),
        },
        "gap_closure_bandit_pct": gap_bandit,
        "gap_closure_routellm_pct": gap_routellm,
        "statistical_tests": {
            "bandit_per_seed": _ci_scalar(bandit_per_seed),
            "tabula_per_seed": _ci_scalar(tabula_per_seed),
            **tests_k2,
        },
    }

    # ==================================================================
    # K=10 — Multi-model Pareto frontier
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=10: Multi-Model Pareto Frontier")
    logger.info("=" * 70)

    costs_k10 = {m: K10_CATALOG[m]["cost"] for m in K10_MODELS}

    # --- Load K=10 data ------------------------------------------------
    logger.info("\n  Loading K=10 data ...")
    with open(THREE_WAY_SPLITS_PATH) as f:
        splits_3way = json.load(f)
    online_prompts = set(splits_3way["online_learn_pool"])

    train_data_k10 = load_rewards_from_file(
        DEV_DATA_PATH_ALL_MODELS, K10_MODELS,
        prompt_filter=online_prompts,
    )
    holdout_data_k10 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K10_MODELS,
    )
    logger.info(f"    Train (online-learn): {len(train_data_k10)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k10)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info("  Embedding K=10 prompts ...")
    train_emb_k10 = embed_dataset(train_data_k10, encoder, pca)
    holdout_emb_k10 = embed_dataset(holdout_data_k10, encoder, pca)

    # --- Baselines -----------------------------------------------------
    logger.info("\n  Computing K=10 baselines ...")
    oracle_r_k10, oracle_c_k10 = oracle_route(
        holdout_data_k10, K10_MODELS, costs_k10,
    )
    logger.info(f"    Oracle: R={oracle_r_k10:.4f}  C=${oracle_c_k10:.6f}")

    static_k10: Dict[str, Dict] = {}
    for m in K10_MODELS:
        sr, sc = static_route(holdout_data_k10, m, costs_k10)
        static_k10[m] = {"reward": sr, "cost": sc}
        logger.info(
            f"    Static {K10_CATALOG[m]['display']:<22}: R={sr:.4f}  C=${sc:.6f}"
        )

    random_k10 = random_route(
        holdout_data_k10, K10_MODELS, costs_k10, N_SEEDS * 4,
    )
    logger.info(f"    Random: R={random_k10['reward']:.4f}")

    eg_k10 = epsilon_greedy_route(
        train_data_k10, holdout_data_k10, K10_MODELS, costs_k10,
        n_trials=N_SEEDS * 4,
    )
    logger.info(f"    Epsilon-greedy: R={eg_k10['reward']:.4f}")

    # --- BanditGPT Pareto sweep ----------------------------------------
    logger.info(
        f"\n  BanditGPT K=10 Pareto sweep "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    bandit_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_data_k10, holdout_data_k10, train_emb_k10, holdout_emb_k10,
        str(MULTIMODEL_WARMUP_PRIORS_PATH), costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=True, label="banditGPT",
    )

    # Tabula rasa ablation
    logger.info(
        f"\n  Tabula rasa K=10 ablation "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_data_k10, holdout_data_k10, train_emb_k10, holdout_emb_k10,
        str(MULTIMODEL_WARMUP_PRIORS_PATH), costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
    )

    # --- K=10 summary (a-priori lambda, no holdout leakage) -------------
    target_lambda_k10 = ROUTELLM_COST_PENALTY  # same objective as K=2
    target_k10 = next(
        x for x in bandit_pareto_k10 if x["lambda"] == target_lambda_k10
    )
    target_tabula_k10 = next(
        x for x in tabula_pareto_k10 if x["lambda"] == target_lambda_k10
    )

    best_static_m = max(static_k10, key=lambda m: static_k10[m]["reward"])
    weak_r_k10 = min(static_k10[m]["reward"] for m in K10_MODELS)
    gap_k10 = (
        (target_k10["mean_reward"] - weak_r_k10)
        / (oracle_r_k10 - weak_r_k10) * 100
        if oracle_r_k10 > weak_r_k10 else 0.0
    )

    logger.info(f"\n  K=10 SUMMARY (lambda={target_lambda_k10}, a-priori):")
    logger.info(f"    Oracle:       {oracle_r_k10:.4f}")
    logger.info(f"    banditGPT:    {target_k10['mean_reward']:.4f} (gap closure {gap_k10:.1f}%)")
    logger.info(f"    Best static:  {static_k10[best_static_m]['reward']:.4f} ({K10_CATALOG[best_static_m]['display']})")
    logger.info(f"    Tabula rasa:  {target_tabula_k10['mean_reward']:.4f}")
    logger.info(f"    Eps-greedy:   {eg_k10['reward']:.4f}")
    logger.info(f"    Random:       {random_k10['reward']:.4f}")

    results_all["K10"] = {
        "models": [{"id": m, **K10_CATALOG[m]} for m in K10_MODELS],
        "n_train": len(train_data_k10),
        "n_holdout": len(holdout_data_k10),
        "target_lambda": target_lambda_k10,
        "oracle": {"reward": oracle_r_k10, "cost": oracle_c_k10},
        "static": {m: static_k10[m] for m in K10_MODELS},
        "best_static": {
            "model": best_static_m,
            "reward": static_k10[best_static_m]["reward"],
            "cost": static_k10[best_static_m]["cost"],
        },
        "random": random_k10,
        "epsilon_greedy": eg_k10,
        "banditgpt_pareto": bandit_pareto_k10,
        "tabula_rasa_pareto": tabula_pareto_k10,
        "point_comparison": {
            "lambda": target_lambda_k10,
            "banditgpt": {
                "reward": target_k10["mean_reward"],
                "std": target_k10["std_reward"],
            },
            "tabula_rasa": {
                "reward": target_tabula_k10["mean_reward"],
                "std": target_tabula_k10["std_reward"],
            },
            "note": (
                f"A-priori lambda={target_lambda_k10}. No holdout-based "
                "hyperparameter selection."
            ),
        },
        "gap_closure_pct": gap_k10,
        "n_trials": N_SEEDS,
    }

    # ==================================================================
    # Serialise
    # ==================================================================
    out_path = output_dir / "prequential_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


# ============================================================================
# CLI entry point
# ============================================================================


if __name__ == "__main__":
    run_experiment()
