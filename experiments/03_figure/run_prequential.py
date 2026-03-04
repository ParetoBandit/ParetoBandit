#!/usr/bin/env python3
"""
BanditGPT: Train-then-Freeze Evaluation.

Compares BanditRouter against supervised static baselines (KNN, SVM, MLP),
standard online baselines (UCB1, random, best-static), and ablations
(coldstart, tabula rasa) using canonical dev/holdout datasets with
ground-truth multi-judge rewards.

Protocol
--------
1. **Canonical dev/holdout splits.**
   Data comes from pre-computed datasets (``dev_rewards_2models.jsonl.gz``,
   ``holdout_rewards_2models.jsonl.gz`` for K=2; all-models variants for
   K=10) with rewards derived via :func:`extract_reward` (mean of
   vote x confidence across multi-judge panel).

2. **Train-then-freeze evaluation.**
   BanditGPT trains on the dev set with oracle rewards, then is frozen
   for evaluation on the holdout set.  Supervised baselines are trained
   on the same dev data and frozen identically.

3. **Symmetric data access.**
   All methods have access to the same dev set.  For any selection step
   (e.g., BanditGPT cost-penalty lambda), we use a dev-train/dev-val
   split: train/tune on dev-train, select on dev-val.  No method sees
   holdout data before evaluation.

4. **Dev-selected deployable Pareto frontier (primary metric).**
   The dev set is split into train (80%) and val (20%).  BanditGPT and
   ablations train on dev-train; dev metrics for frontier selection come
   from dev-val (eliminating train-set evaluation asymmetry).  The
   Pareto hull is built from (dev_val_cost, dev_val_reward); for those
   dev-optimal settings, holdout cost and holdout reward are
   extracted — no holdout or training data enters the hyperparameter
   selection step.  The area under this deployable frontier is the
   primary Pareto AUC metric.

   After selection, we optionally run a second pass that retrains the
   chosen router(s) on the **full dev set** and re-evaluates on holdout,
   keeping the dev-val selection fixed. This reduces estimator variance
   while maintaining a leakage-free selection protocol.

5. **K=10 Pareto frontier.**
   The K=10 evaluation uses baselines: oracle, best-static, random,
   best-static-plus-noise, UCB1 (non-contextual), supervised static
   routers (KNN, SVM, MLP — trained on the same features and objective
   as BanditGPT; ref: LLMRouter, UIUC 2025), and tabula-rasa
   (cold-start BanditGPT without priors or Corralling).

7. **Greedy frozen evaluation.**
   When evaluating a frozen BanditRouter, the UCB exploration bonus is
   set to zero (alpha=0) so that holdout scores reflect the learned
   policy under pure exploitation.

8. **Statistical reporting.**
   - *Primary hypothesis test*: paired bootstrap CI for the dev-selected
     Pareto AUC difference (1,000 resamples; dev indices fixed).
   - *Post-hoc point comparisons*: per-seed paired t-tests at three
     budget levels, restricted to dev-optimal hyperparameters, with
     Holm-Bonferroni correction across budget levels.
   - *Stability*: across-seed t-tests (df = N_SEEDS - 1).

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
    ARTIFACTS_DIR,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    K2_WARMUP_FROM_MULTIMODEL_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    THREE_WAY_SPLITS_PATH,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models
from utils.router_factory import create_experiment_router
from utils.supervised_baselines import run_supervised_baseline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Model catalogs
# ============================================================================

def _req_cost(inp: float, out: float) -> float:
    """Per-request cost assuming 100 input + 400 output tokens."""
    return (100 * inp + 400 * out) / 1_000_000


K2_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4.1",
]

_PRICES_K2 = get_prices_for_models(K2_MODELS)

K2_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES_K2["meta-llama/llama-3.1-8b-instruct"],
        "cost": _req_cost(
            _PRICES_K2["meta-llama/llama-3.1-8b-instruct"]["input_cost_per_m"],
            _PRICES_K2["meta-llama/llama-3.1-8b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES_K2["openai/gpt-4.1"],
        "cost": _req_cost(
            _PRICES_K2["openai/gpt-4.1"]["input_cost_per_m"],
            _PRICES_K2["openai/gpt-4.1"]["output_cost_per_m"],
        ),
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
    "moonshotai/kimi-k2-0905",
    "openai/gpt-4.1",
]

_PRICES_K10 = get_prices_for_models(K10_MODELS)

K10_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES_K10["meta-llama/llama-3.1-8b-instruct"],
        "cost": _req_cost(
            _PRICES_K10["meta-llama/llama-3.1-8b-instruct"]["input_cost_per_m"],
            _PRICES_K10["meta-llama/llama-3.1-8b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "mistralai/mixtral-8x7b-instruct": {
        "display": "Mixtral-8x7B",
        **_PRICES_K10["mistralai/mixtral-8x7b-instruct"],
        "cost": _req_cost(
            _PRICES_K10["mistralai/mixtral-8x7b-instruct"]["input_cost_per_m"],
            _PRICES_K10["mistralai/mixtral-8x7b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "google/gemma-3-27b-it": {
        "display": "Gemma-3-27B",
        **_PRICES_K10["google/gemma-3-27b-it"],
        "cost": _req_cost(
            _PRICES_K10["google/gemma-3-27b-it"]["input_cost_per_m"],
            _PRICES_K10["google/gemma-3-27b-it"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "anthropic/claude-haiku-4.5": {
        "display": "Claude-Haiku-4.5",
        **_PRICES_K10["anthropic/claude-haiku-4.5"],
        "cost": _req_cost(
            _PRICES_K10["anthropic/claude-haiku-4.5"]["input_cost_per_m"],
            _PRICES_K10["anthropic/claude-haiku-4.5"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "display": "DeepSeek-V3",
        **_PRICES_K10["deepseek/deepseek-chat-v3-0324"],
        "cost": _req_cost(
            _PRICES_K10["deepseek/deepseek-chat-v3-0324"]["input_cost_per_m"],
            _PRICES_K10["deepseek/deepseek-chat-v3-0324"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "google/gemini-2.5-flash-preview-09-2025": {
        "display": "Gemini-2.5-Flash",
        **_PRICES_K10["google/gemini-2.5-flash-preview-09-2025"],
        "cost": _req_cost(
            _PRICES_K10["google/gemini-2.5-flash-preview-09-2025"]["input_cost_per_m"],
            _PRICES_K10["google/gemini-2.5-flash-preview-09-2025"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "meta-llama/llama-4-maverick": {
        "display": "Llama-4-Maverick",
        **_PRICES_K10["meta-llama/llama-4-maverick"],
        "cost": _req_cost(
            _PRICES_K10["meta-llama/llama-4-maverick"]["input_cost_per_m"],
            _PRICES_K10["meta-llama/llama-4-maverick"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "anthropic/claude-sonnet-4": {
        "display": "Claude-Sonnet-4",
        **_PRICES_K10["anthropic/claude-sonnet-4"],
        "cost": _req_cost(
            _PRICES_K10["anthropic/claude-sonnet-4"]["input_cost_per_m"],
            _PRICES_K10["anthropic/claude-sonnet-4"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
    "moonshotai/kimi-k2-0905": {
        "display": "Kimi-K2",
        **_PRICES_K10["moonshotai/kimi-k2-0905"],
        "cost": _req_cost(
            _PRICES_K10["moonshotai/kimi-k2-0905"]["input_cost_per_m"],
            _PRICES_K10["moonshotai/kimi-k2-0905"]["output_cost_per_m"],
        ),
        "tier": "mid",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES_K10["openai/gpt-4.1"],
        "cost": _req_cost(
            _PRICES_K10["openai/gpt-4.1"]["input_cost_per_m"],
            _PRICES_K10["openai/gpt-4.1"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
}


# ============================================================================
# Experiment configuration
# ============================================================================

N_SEEDS: int = 20
SEED_OFFSET: int = 42
TARGET_NEFF: float = 5000.0
ALPHA_START: float = 1.0
CORRALLING_LR: float = 0.1
CORRALLING_GAMMA: float = 0.05

DEV_VAL_FRACTION: float = 0.2
DEV_VAL_SEED: int = 7

LAMBDA_VALUES_K2: List[float] = [
    # High-cost plateau (lambda barely affects cost here): 8 representative pts
    0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45,
    # Transition zone (cost drops steeply — densify for Pareto resolution)
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    # Low-cost floor (router converges to cheapest model)
    1.0, 1.5, 2.0, 3.0, 5.0,
]
LAMBDA_VALUES_K10: List[float] = [
    0.0, 0.01, 0.03, 0.05, 0.07, 0.08, 0.09, 0.095,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
    0.185, 0.19, 0.192, 0.195, 0.198, 0.2, 0.202, 0.205,
    0.208, 0.21, 0.215, 0.22, 0.25, 0.3, 0.5, 1.0,
]

LEARNING_CURVE_CHECKPOINTS_K2: List[int] = [
    0, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
]

_T_CRIT: float = float(scipy_stats.t.ppf(0.975, df=N_SEEDS - 1))


def _split_dev_train_val(
    data: List[Dict],
    emb: List[np.ndarray],
    val_fraction: float = DEV_VAL_FRACTION,
    seed: int = DEV_VAL_SEED,
) -> Tuple[List[Dict], List[np.ndarray], List[Dict], List[np.ndarray]]:
    """Deterministically split (data, emb) into train and val portions.

    The split is stratified-random with a fixed seed for reproducibility.
    The val portion is used for unbiased dev-set metrics (dev_mean_cost,
    dev_mean_reward) that drive hyperparameter selection, avoiding the
    train-set evaluation asymmetry between online and static methods.

    Args:
        data: Full dev dataset.
        emb: Aligned embeddings.
        val_fraction: Fraction reserved for validation.
        seed: RNG seed for the split.

    Returns:
        (train_data, train_emb, val_data, val_emb).
    """
    n = len(data)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_d = [data[i] for i in range(n) if i not in val_idx]
    train_e = [emb[i] for i in range(n) if i not in val_idx]
    val_d = [data[i] for i in range(n) if i in val_idx]
    val_e = [emb[i] for i in range(n) if i in val_idx]
    return train_d, train_e, val_d, val_e


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


def best_static_noisy_route(
    train_data: List[Dict],
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    epsilon: float = 0.1,
    n_trials: int = 20,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, float]:
    """Best-static-plus-noise baseline: route to the empirical best model
    (computed from full training set means) with probability 1-epsilon,
    and uniformly at random otherwise.

    This is NOT an online epsilon-greedy bandit — the "best arm" is
    identified via full hindsight over the training data, making this
    an epsilon-perturbed best-in-hindsight static policy.
    """
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


def ucb1_online_route(
    train_data: List[Dict],
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    cost_penalty: float = 0.0,
    n_trials: int = 20,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, float]:
    """Non-contextual UCB1 train-then-freeze baseline.

    Online phase (train): for each training prompt (shuffled per trial),
    select the arm with the highest UCB1 score::

        score(a) = mean_reward(a) - cost_penalty * cost(a) + sqrt(2 ln(t) / n(a))

    Observe the reward and update the running average.

    Frozen phase (eval): act greedily using the learned arm means
    (no exploration bonus, analogous to alpha=0 for BanditGPT).

    This ablates the value of contextual features: UCB1 cannot
    personalise routing to prompt characteristics.

    Args:
        train_data: Dev set prompts with ``rewards`` dict.
        eval_data: Holdout prompts.
        models: Model IDs (arms).
        costs: Per-model cost dict.
        cost_penalty: Lambda for cost-penalised UCB.
        n_trials: Number of random-permutation seeds.
        seed_offset: Base seed.

    Returns:
        Dict with mean/std reward/cost across trials.
    """
    K = len(models)
    trial_r, trial_c = [], []

    for trial in range(n_trials):
        rng = np.random.RandomState(seed_offset + trial)
        order = rng.permutation(len(train_data))

        sum_reward = np.zeros(K)
        counts = np.zeros(K)
        t = 0

        for idx in order:
            p = train_data[idx]
            t += 1

            if t <= K:
                arm = t - 1
            else:
                means = sum_reward / np.maximum(counts, 1)
                bonus = np.sqrt(2.0 * np.log(t) / np.maximum(counts, 1))
                cost_arr = np.array([costs[models[a]] for a in range(K)])
                scores = means - cost_penalty * cost_arr + bonus
                arm = int(np.argmax(scores))

            reward = p["rewards"][models[arm]]
            sum_reward[arm] += reward
            counts[arm] += 1

        # Frozen evaluation: greedy on learned means
        means = sum_reward / np.maximum(counts, 1)
        cost_arr = np.array([costs[models[a]] for a in range(K)])
        greedy_scores = means - cost_penalty * cost_arr
        greedy_arm = int(np.argmax(greedy_scores))

        r_total = c_total = 0.0
        for p in eval_data:
            r_total += p["rewards"][models[greedy_arm]]
            c_total += costs[models[greedy_arm]]
        n = len(eval_data)
        trial_r.append(r_total / n)
        trial_c.append(c_total / n)

    return {
        "reward": float(np.mean(trial_r)),
        "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
        "cost": float(np.mean(trial_c)),
        "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
        "n_trials": n_trials,
        "greedy_arm": models[greedy_arm],
    }


# ============================================================================
# BanditGPT train-then-freeze evaluation
# ============================================================================


REWARD_THEORETICAL_MIN: float = 0.0
REWARD_THEORETICAL_MAX: float = 1.0


def _compute_reward_normalization(
    train_data: List[Dict],
    models: List[str],
) -> Tuple[float, float]:
    """Return the theoretical reward bounds for normalisation.

    ``extract_reward()`` returns mean(vote × confidence) where
    vote ∈ {0, 1} and confidence ∈ [0, 1], yielding rewards in [0, 1].
    Using theoretical bounds avoids any a-priori information leakage
    from the counterfactual reward matrix, which would violate the
    strict online sequential setting.
    """
    return REWARD_THEORETICAL_MIN, REWARD_THEORETICAL_MAX


def train_bandit(
    router,
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    models: List[str],
    r_min: float,
    r_range: float,
    *,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
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
        shuffle: If True, present training data in random order.
            Presentation order affects online learning, so different
            seeds yield meaningfully different policies.
        rng: Explicit numpy Generator for the training permutation.
            If None, falls back to ``np.random.permutation`` (global state).

    Returns:
        Number of training steps completed.
    """
    n_steps = len(train_data)
    if shuffle:
        order = rng.permutation(n_steps) if rng is not None else np.random.permutation(n_steps)
    else:
        order = np.arange(n_steps)
    for idx in order:
        p, x = train_data[idx], train_embeddings[idx]
        model, log = router.route(x, total_steps=n_steps)
        raw_reward = p["rewards"][model]
        norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
        router.process_feedback(log.request_id, norm_reward)
    return n_steps


def _set_exploit_mode(router, *, enable: bool) -> Dict[str, Any]:
    """Temporarily switch to greedy exploitation on a frozen router.

    Two levels of greediness are applied:
      1. **Expert-level**: alpha=0 on all LinUCB experts so arm selection
         is pure ``argmax(θ^T x)`` (no UCB bonus).
      2. **Meta-level**: ``exploit_mode=True`` on the CorrallingRouter so
         the highest-weight expert is selected deterministically
         (``argmax(weights)``), not sampled.

    This is standard practice for offline / frozen policy evaluation
    (Dudik et al., 2014; Swaminathan & Joachims, 2015): the learned
    policy is executed greedily so that holdout scores reflect the
    learned routing policy, not residual exploration.

    Args:
        router: A BanditRouter instance.
        enable: If True, enable exploit mode and return saved state.
            If False, this is a no-op (returns empty dict).

    Returns:
        Saved state dict to pass to ``_restore_exploit_mode``.
    """
    if not enable:
        return {}
    saved: Dict[str, Any] = {"expert_alphas": [], "meta_exploit": False}
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts"):
        for expert in cr.experts:
            saved["expert_alphas"].append((expert.alpha_start, expert.alpha_end))
            expert.alpha_start = 0.0
            expert.alpha_end = 0.0
        saved["meta_exploit"] = cr.exploit_mode
        cr.exploit_mode = True
    return saved


def _restore_exploit_mode(
    router, saved: Dict[str, Any],
) -> None:
    """Restore expert alpha values and meta exploit mode after evaluation."""
    if not saved:
        return
    cr = getattr(router, "corralling_router", None)
    if cr is not None and hasattr(cr, "experts") and saved.get("expert_alphas"):
        for expert, (a_s, a_e) in zip(cr.experts, saved["expert_alphas"]):
            expert.alpha_start = a_s
            expert.alpha_end = a_e
        cr.exploit_mode = saved.get("meta_exploit", False)


def evaluate_frozen(
    router,
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    costs: Dict[str, float],
    total_steps: int,
    *,
    per_prompt: bool = False,
) -> Tuple[float, float, Dict[str, int], Optional[List[float]], Optional[List[float]]]:
    """Evaluate a frozen router on the holdout set (no learning).

    Uses greedy exploitation (alpha=0) during evaluation so that
    holdout scores reflect the learned policy, not optimistic UCB
    exploration bonuses.

    Args:
        router: Frozen BanditRouter.
        eval_data: Holdout {prompt, rewards} dicts.
        eval_embeddings: Pre-computed feature vectors for holdout.
        costs: Per-model cost dict.
        total_steps: Total steps completed during training (for alpha decay).
        per_prompt: If True, also return per-prompt rewards and costs
            for paired statistical tests and bootstrap.

    Returns:
        (mean_reward, mean_cost, model_counts, per_prompt_rewards,
        per_prompt_costs).  The last two are None unless ``per_prompt=True``.
    """
    saved = _set_exploit_mode(router, enable=True)
    rng_state = np.random.get_state()

    r_total = c_total = 0.0
    model_counts: Dict[str, int] = defaultdict(int)
    prompt_rewards: Optional[List[float]] = [] if per_prompt else None
    prompt_costs: Optional[List[float]] = [] if per_prompt else None

    for p, x in zip(eval_data, eval_embeddings):
        model, _log = router.route(x, total_steps=total_steps)
        reward = p["rewards"][model]
        cost = costs[model]
        r_total += reward
        c_total += cost
        model_counts[model] += 1
        if prompt_rewards is not None:
            prompt_rewards.append(reward)
            prompt_costs.append(cost)

    np.random.set_state(rng_state)
    _restore_exploit_mode(router, saved)

    n = len(eval_data)
    return r_total / n, c_total / n, dict(model_counts), prompt_rewards, prompt_costs


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
    dev_val_data: Optional[List[Dict]] = None,
    dev_val_emb: Optional[List[np.ndarray]] = None,
    alpha: float = ALPHA_START,
    prior_n_effective: float = TARGET_NEFF,
    forgetting_factor: float = 1.0,
) -> List[Dict]:
    """Sweep cost penalty lambda with N trials per point.

    For each (lambda, trial): instantiate router, train on *train_data*
    (the learn split of the dev set), freeze, and evaluate on the
    holdout set.

    Dev metrics (``dev_mean_cost``, ``dev_mean_reward``) are computed
    on an internal **validation split** of the dev set that was *not*
    used for training, eliminating the train-set evaluation asymmetry
    between online and static methods.  If *dev_val_data* is not
    provided, the full *train_data* is used for dev metrics (fallback
    for backward compatibility, but not recommended).

    Returns:
        List of dicts, one per lambda, with mean/std holdout reward
        and cost, plus ``dev_mean_cost`` and ``dev_mean_reward`` for
        constructing the dev-selected Pareto frontier.
    """
    dim = train_emb[0].shape[0]
    r_min, r_max = _compute_reward_normalization(train_data, models)
    r_range = r_max - r_min
    burn_in = len(train_data)

    val_d = dev_val_data if dev_val_data is not None else train_data
    val_e = dev_val_emb if dev_val_emb is not None else train_emb

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        trial_dev_c: List[float] = []
        trial_dev_r: List[float] = []
        all_per_prompt: List[List[float]] = []
        all_per_prompt_costs: List[List[float]] = []
        for trial in range(n_trials):
            seed = SEED_OFFSET + trial
            np.random.seed(seed)  # router init may consume global RNG
            trial_rng = np.random.default_rng(seed)
            router = create_experiment_router(
                model_registry=build_model_registry(models, catalog),
                feature_dim=dim,
                prior_n_effective=prior_n_effective,
                alpha=alpha,
                warmup_path=warmup_path,
                use_corralling=use_corralling,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
                forgetting_factor=forgetting_factor,
            )
            train_bandit(router, train_data, train_emb, models, r_min, r_range,
                         rng=trial_rng)
            r, c, _, pp, pp_c = evaluate_frozen(
                router, eval_data, eval_emb, costs, burn_in, per_prompt=True,
            )
            trial_r.append(r)
            trial_c.append(c)
            all_per_prompt.append(pp)
            all_per_prompt_costs.append(pp_c)
            dev_r, dev_c, _, _, _ = evaluate_frozen(
                router, val_d, val_e, costs, burn_in,
            )
            trial_dev_c.append(dev_c)
            trial_dev_r.append(dev_r)

        results.append({
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "dev_mean_cost": float(np.mean(trial_dev_c)),
            "dev_mean_reward": float(np.mean(trial_dev_r)),
            "per_seed_rewards": [float(x) for x in trial_r],
            "per_seed_costs": [float(x) for x in trial_c],
            "per_seed_per_prompt_rewards": all_per_prompt,
            "per_seed_per_prompt_costs": all_per_prompt_costs,
            "n_trials": n_trials,
            "label": label,
        })
        logger.info(
            f"    lambda={lam:<6} | R={np.mean(trial_r):.4f}+/-{np.std(trial_r):.4f} "
            f"| C=${np.mean(trial_c):.6f}"
        )
    return results


def _recompute_holdout_metrics_with_full_dev_training(
    *,
    models: List[str],
    catalog: Dict[str, Dict],
    full_dev_data: List[Dict],
    full_dev_emb: List[np.ndarray],
    holdout_data: List[Dict],
    holdout_emb: List[np.ndarray],
    warmup_path: str | None,
    costs: Dict[str, float],
    lambda_values: List[float],
    n_trials: int,
    alpha: float,
    prior_n_effective: float,
    forgetting_factor: float,
    use_corralling: bool,
    label: str,
) -> Dict[float, Dict[str, Any]]:
    """Second-pass evaluation: train on full dev, evaluate on holdout.

    This intentionally **does not** compute any dev metrics, because the
    dev-val split is used for selection and must remain "unseen" by the
    selection metric. We only update holdout metrics after selection.

    Returns
    -------
    dict[float, dict]
        Mapping lambda -> holdout metrics dict compatible with the keys used
        in `run_pareto_sweep()` entries (except dev_* keys).
    """
    dim = full_dev_emb[0].shape[0]
    r_min, r_max = _compute_reward_normalization(full_dev_data, models)
    r_range = r_max - r_min
    burn_in = len(full_dev_data)

    out: Dict[float, Dict[str, Any]] = {}
    for lam in lambda_values:
        trial_r: List[float] = []
        trial_c: List[float] = []
        all_per_prompt: List[List[float]] = []
        all_per_prompt_costs: List[List[float]] = []
        for trial in range(n_trials):
            seed = SEED_OFFSET + trial
            np.random.seed(seed)  # router init may consume global RNG
            trial_rng = np.random.default_rng(seed)
            router = create_experiment_router(
                model_registry=build_model_registry(models, catalog),
                feature_dim=dim,
                prior_n_effective=prior_n_effective,
                alpha=alpha,
                warmup_path=warmup_path,
                use_corralling=use_corralling,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
                forgetting_factor=forgetting_factor,
            )
            train_bandit(router, full_dev_data, full_dev_emb, models, r_min, r_range,
                         rng=trial_rng)
            r, c, _, pp, pp_c = evaluate_frozen(
                router, holdout_data, holdout_emb, costs, burn_in, per_prompt=True,
            )
            trial_r.append(r)
            trial_c.append(c)
            all_per_prompt.append(pp)
            all_per_prompt_costs.append(pp_c)

        out[lam] = {
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "per_seed_rewards": [float(x) for x in trial_r],
            "per_seed_costs": [float(x) for x in trial_c],
            "per_seed_per_prompt_rewards": all_per_prompt,
            "per_seed_per_prompt_costs": all_per_prompt_costs,
            "n_trials": n_trials,
            "label": label,
            "trained_on": "full_dev",
        }
    return out


def _patch_pareto_entries_holdout_metrics_inplace(
    entries: List[Dict[str, Any]],
    *,
    recomputed: Dict[float, Dict[str, Any]],
) -> None:
    """In-place update of holdout metrics while preserving dev_* metrics."""
    for e in entries:
        lam = float(e["lambda"])
        if lam not in recomputed:
            continue
        new = recomputed[lam]
        # Preserve dev_mean_* (selection metrics). Replace holdout-related keys.
        for k in [
            "mean_reward",
            "std_reward",
            "mean_cost",
            "std_cost",
            "per_seed_rewards",
            "per_seed_costs",
            "per_seed_per_prompt_rewards",
            "per_seed_per_prompt_costs",
            "n_trials",
            "label",
        ]:
            e[k] = new[k]
        e["trained_on"] = new.get("trained_on", "full_dev")


def run_coldstart_sweep(
    models: List[str],
    catalog: Dict[str, Dict],
    eval_data: List[Dict],
    eval_emb: List[np.ndarray],
    warmup_path: str,
    costs: Dict[str, float],
    lambda_values: List[float],
    n_trials: int,
    feature_dim: int,
    *,
    use_corralling: bool = True,
    label: str = "coldstart",
    dev_data: Optional[List[Dict]] = None,
    dev_emb: Optional[List[np.ndarray]] = None,
    alpha: float = ALPHA_START,
    prior_n_effective: float = TARGET_NEFF,
    forgetting_factor: float = 1.0,
) -> List[Dict]:
    """Cold-start Pareto sweep: priors only, zero online training steps.

    Creates the router with warmup priors and evaluates immediately on the
    holdout set without any ``train_bandit()`` call.  This isolates the
    value of offline priors from online adaptation.

    If *dev_data* and *dev_emb* are provided, the cold-start router is
    also evaluated on the dev set to produce ``dev_mean_cost`` and
    ``dev_mean_reward``, allowing the cold-start baseline to participate
    in the dev-selected Pareto frontier pipeline.

    Returns:
        List of dicts, one per lambda, with mean/std reward and cost,
        plus dev metrics if *dev_data* is provided.
    """
    burn_in = 0

    results = []
    for lam in lambda_values:
        trial_r, trial_c = [], []
        trial_dev_c: List[float] = []
        trial_dev_r: List[float] = []
        for trial in range(n_trials):
            np.random.seed(SEED_OFFSET + trial)  # router init reproducibility
            router = create_experiment_router(
                model_registry=build_model_registry(models, catalog),
                feature_dim=feature_dim,
                prior_n_effective=prior_n_effective,
                alpha=alpha,
                warmup_path=warmup_path,
                use_corralling=use_corralling,
                corralling_learning_rate=CORRALLING_LR,
                corralling_gamma=CORRALLING_GAMMA,
                cost_penalty=lam,
                forgetting_factor=forgetting_factor,
            )
            r, c, _, _, _ = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            trial_r.append(r)
            trial_c.append(c)
            if dev_data is not None and dev_emb is not None:
                dr, dc, _, _, _ = evaluate_frozen(
                    router, dev_data, dev_emb, costs, burn_in,
                )
                trial_dev_c.append(dc)
                trial_dev_r.append(dr)

        entry: Dict[str, Any] = {
            "lambda": lam,
            "mean_reward": float(np.mean(trial_r)),
            "std_reward": float(np.std(trial_r, ddof=1)) if n_trials > 1 else 0.0,
            "mean_cost": float(np.mean(trial_c)),
            "std_cost": float(np.std(trial_c, ddof=1)) if n_trials > 1 else 0.0,
            "per_seed_rewards": [float(x) for x in trial_r],
            "per_seed_costs": [float(x) for x in trial_c],
            "n_trials": n_trials,
            "label": label,
        }
        if trial_dev_c:
            entry["dev_mean_cost"] = float(np.mean(trial_dev_c))
            entry["dev_mean_reward"] = float(np.mean(trial_dev_r))
        results.append(entry)
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
    alpha: float = ALPHA_START,
    label: str = "banditGPT",
    prior_n_effective: float = TARGET_NEFF,
    forgetting_factor: float = 1.0,
) -> List[Dict]:
    """Learning curve: holdout quality as a function of online training steps.

    At each checkpoint, the router is frozen and evaluated on the full
    holdout set.  Step 0 evaluates with priors only (no online data).

    Args:
        models: Candidate model IDs.
        catalog: Model metadata catalog.
        train_data: Dev-set prompts with rewards.
        eval_data: Holdout-set prompts with rewards.
        train_emb: Pre-computed feature vectors for dev set.
        eval_emb: Pre-computed feature vectors for holdout set.
        warmup_path: Path to warmup priors file.
        costs: Per-model cost dict.
        n_trials: Number of random seeds.
        checkpoints: Training steps at which to evaluate.
        use_corralling: Whether to use Corralling meta-learner.
        cost_penalty: Lambda for cost-quality trade-off.
        alpha: Exploration coefficient for LinUCB experts.
        label: Label for the curve in output data.

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
        seed = SEED_OFFSET + trial
        np.random.seed(seed)  # router init may consume global RNG
        trial_rng = np.random.default_rng(seed)
        router = create_experiment_router(
            model_registry=build_model_registry(models, catalog),
            feature_dim=dim,
            prior_n_effective=prior_n_effective,
            alpha=alpha,
            warmup_path=warmup_path,
            use_corralling=use_corralling,
            corralling_learning_rate=CORRALLING_LR,
            corralling_gamma=CORRALLING_GAMMA,
            cost_penalty=cost_penalty,
            forgetting_factor=forgetting_factor,
        )

        if 0 in checkpoint_set:
            r, c, _, _, _ = evaluate_frozen(router, eval_data, eval_emb, costs, burn_in)
            by_step[0]["rewards"].append(r)
            by_step[0]["costs"].append(c)

        order = trial_rng.permutation(len(train_data))
        for step_idx, idx in enumerate(order):
            p, x = train_data[idx], train_emb[idx]
            model, log = router.route(x, total_steps=burn_in)
            raw_reward = p["rewards"][model]
            norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
            router.process_feedback(log.request_id, norm_reward)
            current = step_idx + 1
            if current in checkpoint_set:
                r, c, _, _, _ = evaluate_frozen(
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
# Isocost comparison & Pareto AUC
# ============================================================================


def _pareto_hull(
    costs: List[float], rewards: List[float],
) -> Tuple[List[float], List[float]]:
    """Monotone upper envelope sorted by ascending cost."""
    pairs = sorted(zip(costs, rewards), key=lambda x: (x[0], -x[1]))
    hull_c, hull_r = [], []
    best_r = -np.inf
    for c, r in pairs:
        if r > best_r:
            hull_c.append(c)
            hull_r.append(r)
            best_r = r
    return hull_c, hull_r


def pareto_auc(
    costs: List[float],
    rewards: List[float],
    cost_lo: float,
    cost_hi: float,
) -> float:
    """Area under the Pareto frontier (trapezoidal) over [cost_lo, cost_hi].

    Normalised by the cost range so the result is in reward units.
    Points outside the range are clipped.  Returns 0 if the frontier
    has no points inside the range.
    """
    hull_c, hull_r = _pareto_hull(costs, rewards)
    if len(hull_c) < 1:
        return 0.0
    hc, hr = np.array(hull_c), np.array(hull_r)
    mask = (hc >= cost_lo) & (hc <= cost_hi)
    if mask.sum() < 2:
        return float(hr[mask].mean()) if mask.sum() == 1 else 0.0
    return float(np.trapz(hr[mask], hc[mask]) / (cost_hi - cost_lo))


def interpolate_pareto_reward(
    hull_c: List[float],
    hull_r: List[float],
    target_cost: float,
) -> Optional[float]:
    """Linearly interpolate quality on the Pareto hull at a target cost.

    Returns None if the target cost is outside the hull's cost range.
    """
    if not hull_c or target_cost < hull_c[0] or target_cost > hull_c[-1]:
        return None
    return float(np.interp(target_cost, hull_c, hull_r))


def find_closest_pareto_point(
    pareto: List[Dict],
    target_cost: float,
    cost_key: str = "mean_cost",
) -> Dict:
    """Find the Pareto sweep point whose cost is closest to *target_cost*."""
    return min(pareto, key=lambda p: abs(p[cost_key] - target_cost))


def _dev_pareto_indices(
    sweep_results: List[Dict],
    dev_cost_key: str = "dev_mean_cost",
    dev_reward_key: str = "dev_mean_reward",
) -> List[int]:
    """Identify sweep indices that lie on the dev-set Pareto frontier.

    The hull is built strictly from dev-set metrics (dev_cost,
    dev_reward) — no holdout information is used.  This selects the
    hyperparameters a practitioner would consider optimal based
    solely on historical (dev) data.

    Returns:
        Sorted list of indices into *sweep_results* that form the
        dev-optimal Pareto frontier.
    """
    n = len(sweep_results)
    pairs = [
        (sweep_results[i][dev_cost_key], sweep_results[i][dev_reward_key], i)
        for i in range(n)
    ]
    pairs.sort(key=lambda x: (x[0], -x[1]))
    hull_idx: List[int] = []
    best_r = -np.inf
    for _, r, idx in pairs:
        if r > best_r:
            hull_idx.append(idx)
            best_r = r
    return hull_idx


def dev_selected_pareto_auc(
    sweep_results: List[Dict],
    cost_lo: float,
    cost_hi: float,
    *,
    dev_cost_key: str = "dev_mean_cost",
    dev_reward_key: str = "dev_mean_reward",
    holdout_cost_key: str = "mean_cost",
    holdout_reward_key: str = "mean_reward",
) -> Tuple[float, List[float], List[float], List[int]]:
    """Pareto AUC of the dev-selected deployable frontier.

    **Hyperparameter selection is strictly partitioned from holdout
    evaluation.** The procedure:

    1. Build the Pareto hull from ``(dev_cost, dev_reward)`` to
       identify which hyperparameter settings a practitioner would
       consider optimal using *only* dev-set information.
    2. For those dev-optimal settings, extract the corresponding
       ``(holdout_cost, holdout_reward)`` — the performance the
       practitioner would actually observe after deployment.
    3. Take the Pareto hull of the resulting holdout points (since
       dev-optimal points may not be monotone on the holdout set)
       and compute AUC.

    Args:
        sweep_results: List of dicts from ``run_pareto_sweep``,
            each with dev and holdout metrics.
        cost_lo: Lower bound of the shared cost range for AUC.
        cost_hi: Upper bound of the shared cost range for AUC.
        dev_cost_key: Dict key for dev-set cost.
        dev_reward_key: Dict key for dev-set reward.
        holdout_cost_key: Dict key for holdout cost.
        holdout_reward_key: Dict key for holdout reward.

    Returns:
        (auc, hull_holdout_costs, hull_holdout_rewards, dev_hull_indices).
    """
    dev_idx = _dev_pareto_indices(
        sweep_results, dev_cost_key, dev_reward_key,
    )
    holdout_costs = [sweep_results[i][holdout_cost_key] for i in dev_idx]
    holdout_rewards = [sweep_results[i][holdout_reward_key] for i in dev_idx]
    hull_c, hull_r = _pareto_hull(holdout_costs, holdout_rewards)
    auc = pareto_auc(hull_c, hull_r, cost_lo, cost_hi)
    return auc, hull_c, hull_r, dev_idx


def bootstrap_pareto_auc_difference(
    bg_pp_rewards: List[np.ndarray],
    bg_pp_costs: List[np.ndarray],
    bl_pp_rewards: List[np.ndarray],
    bl_pp_costs: List[np.ndarray],
    cost_lo: float,
    cost_hi: float,
    n_holdout: int,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap CI for the difference in dev-selected Pareto AUC.

    **Caller must pre-filter to dev-Pareto-optimal points.** This
    function receives per-prompt reward *and cost* arrays for the
    hyperparameters identified as optimal on the dev set.  Both axes
    of the Pareto frontier are resampled jointly, correctly capturing
    variance in both reward and cost.

    Args:
        bg_pp_rewards: Per-prompt holdout rewards for each dev-optimal
            BanditGPT setting (list of 1-D arrays, length n_holdout).
        bg_pp_costs: Per-prompt holdout costs (same structure).
        bl_pp_rewards: Per-prompt holdout rewards for each dev-optimal
            baseline setting.
        bl_pp_costs: Per-prompt holdout costs (same structure).
        cost_lo: Lower bound of shared cost range.
        cost_hi: Upper bound of shared cost range.
        n_holdout: Number of holdout prompts.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with observed AUC difference, 95% CI, and bootstrap p-value.
    """
    rng = np.random.default_rng(seed)

    def _auc_for_resample(
        idx: np.ndarray,
        pp_cost_arrays: List[np.ndarray],
        pp_reward_arrays: List[np.ndarray],
    ) -> float:
        costs = [float(np.mean(c[idx])) for c in pp_cost_arrays]
        rewards = [float(np.mean(r[idx])) for r in pp_reward_arrays]
        hull_c, hull_r = _pareto_hull(costs, rewards)
        return pareto_auc(hull_c, hull_r, cost_lo, cost_hi)

    all_idx = np.arange(n_holdout)
    obs_bg = _auc_for_resample(all_idx, bg_pp_costs, bg_pp_rewards)
    obs_bl = _auc_for_resample(all_idx, bl_pp_costs, bl_pp_rewards)
    obs_diff = obs_bg - obs_bl

    boot_diffs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_holdout, size=n_holdout, replace=True)
        bg_auc = _auc_for_resample(idx, bg_pp_costs, bg_pp_rewards)
        bl_auc = _auc_for_resample(idx, bl_pp_costs, bl_pp_rewards)
        boot_diffs.append(bg_auc - bl_auc)

    boot_arr = np.array(boot_diffs)
    centred = boot_arr - obs_diff
    p_value = float(np.mean(np.abs(centred) >= np.abs(obs_diff)))

    return {
        "observed_diff": obs_diff,
        "bg_auc": obs_bg,
        "baseline_auc": obs_bl,
        "ci_95_lower": float(np.percentile(boot_arr, 2.5)),
        "ci_95_upper": float(np.percentile(boot_arr, 97.5)),
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "n_bg_dev_optimal": len(bg_pp_rewards),
        "n_bl_dev_optimal": len(bl_pp_rewards),
        "note": (
            "Paired bootstrap over holdout prompts.  Dev-Pareto-optimal "
            "indices fixed before bootstrapping.  Both costs and rewards "
            "are resampled jointly."
        ),
    }


def _extract_dev_optimal_per_prompt(
    sweep: List[Dict],
    dev_idx: List[int],
    per_prompt_reward_map: Dict,
    per_prompt_cost_map: Dict,
    hparam_key: str = "lambda",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract seed-averaged per-prompt reward and cost arrays for dev-optimal points.

    Args:
        sweep: Full sweep results.
        dev_idx: Indices of dev-Pareto-optimal points.
        per_prompt_reward_map: Maps hyperparameter value -> per-prompt
            reward array (shape (n_seeds, n_holdout) or (n_holdout,)).
        per_prompt_cost_map: Maps hyperparameter value -> per-prompt
            cost array (same shapes as reward map).
        hparam_key: Key to extract the hyperparameter value from sweep.

    Returns:
        (per_prompt_reward_arrays, per_prompt_cost_arrays) for the
        dev-optimal subset, each seed-averaged to shape (n_holdout,).
    """
    pp_r_arrays: List[np.ndarray] = []
    pp_c_arrays: List[np.ndarray] = []
    for i in dev_idx:
        hval = sweep[i][hparam_key]
        r_arr = per_prompt_reward_map[hval]
        c_arr = per_prompt_cost_map[hval]
        pp_r_arrays.append(np.mean(r_arr, axis=0) if r_arr.ndim == 2 else r_arr)
        pp_c_arrays.append(np.mean(c_arr, axis=0) if c_arr.ndim == 2 else c_arr)
    return pp_r_arrays, pp_c_arrays


# ============================================================================
# Statistical helpers
# ============================================================================


def _ci_scalar(vals: List[float]) -> Dict[str, float]:
    """Compute mean, std, and 95% CI for a list of per-seed scalars."""
    a = np.array(vals)
    m, s = float(a.mean()), float(a.std(ddof=1))
    hw = _T_CRIT * s / np.sqrt(len(a))
    return {"mean": m, "std": s, "ci_lower": m - hw, "ci_upper": m + hw}


def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap for E[a] - E[b].

    For train-then-freeze evaluation, holdout predictions are
    conditionally independent given the frozen model.  The paired
    bootstrap over holdout instances is therefore valid.  Treat
    this as a complementary nonparametric check alongside the
    parametric paired t-test.
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

    **Secondary** significance test.  Variance across seeds measures
    algorithmic stability (sensitivity to training permutation), not
    generalization across prompts.  The primary test is the paired
    t-test over holdout prompts (see ``isocost_comparison``).
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
                f"df={len(diff) - 1}; n={len(diff)} seeds — adequate power "
                "for moderate effects (|d| ~ 0.5) at alpha=0.05."
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
    logger.info(f"  PCA: {pca.n_components_} components (unified for K=2 and K=10)")

    results_all: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": N_SEEDS,
            "reward_source": "extract_reward(mean_vote_x_confidence)",
            "protocol": "train_then_freeze",
            "split_protocol": "canonical_dev_holdout",
            "dev_val_fraction": DEV_VAL_FRACTION,
            "dev_val_seed": DEV_VAL_SEED,
            "fairness_design": (
                "Symmetric data access: all methods train/tune "
                "on the same dev-train split.  Dev metrics for Pareto frontier "
                "selection come from a held-out dev-val split "
                f"({DEV_VAL_FRACTION:.0%} of dev), eliminating train-set "
                "evaluation asymmetry between online and static methods. "
                "After selection, we retrain BanditGPT on full dev and "
                "re-evaluate on holdout (dev-val selection fixed). "
                "No holdout data enters hyperparameter selection."
            ),
            "final_training_pass": (
                "Two-pass protocol: (1) dev-train/dev-val split for selection "
                "and dev metrics; (2) retrain on full dev and re-evaluate on "
                "holdout, while keeping dev-val selection fixed."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Phase 0 — optional dev-val-selected hyperparameters (Appendix H)
    # ------------------------------------------------------------------
    hparams_dir = (
        Path(__file__).resolve().parent.parent / "appendix"
        / "H_alpha_neff_ablation" / "results"
    )
    hparams_k2_path = hparams_dir / "best_hparams_k2.json"
    hparams_k10_path = hparams_dir / "best_hparams_k10.json"

    def _load_hparams(path: Path, key: str) -> Optional[Dict[str, float]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cfg = data.get(key, {})
            return {
                "alpha": float(cfg["alpha"]),
                "prior_n_effective": float(cfg["n_eff"]),
                "forgetting_factor": float(cfg["gamma"]),
            }
        except Exception as exc:
            logger.warning(f"Failed to load hparams from {path}: {exc}")
            return None

    tuned_k2 = _load_hparams(hparams_k2_path, "K2")
    tuned_k10 = _load_hparams(hparams_k10_path, "K10")

    if tuned_k2 is not None:
        logger.info(
            f"Loaded K=2 tuned hparams (dev-val-selected): "
            f"alpha={tuned_k2['alpha']} n_eff={tuned_k2['prior_n_effective']} "
            f"forgetting_factor={tuned_k2['forgetting_factor']} "
            f"from {hparams_k2_path}"
        )
    else:
        logger.warning(
            f"Appendix H results not found at {hparams_k2_path}. "
            f"Falling back to module-level defaults (alpha={ALPHA_START}, "
            f"n_eff={TARGET_NEFF}). Run "
            f"experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py "
            f"first for tuned hyperparameters."
        )
    if tuned_k10 is not None:
        logger.info(
            f"Loaded K=10 tuned hparams (dev-val-selected): "
            f"alpha={tuned_k10['alpha']} n_eff={tuned_k10['prior_n_effective']} "
            f"forgetting_factor={tuned_k10['forgetting_factor']} "
            f"from {hparams_k10_path}"
        )
    else:
        logger.warning(
            f"Appendix H results not found at {hparams_k10_path}. "
            f"Falling back to module-level defaults (alpha={ALPHA_START}, "
            f"n_eff={TARGET_NEFF}). Run "
            f"experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py "
            f"first for tuned hyperparameters."
        )

    # ==================================================================
    # K=2 — BanditGPT vs supervised baselines & ablations
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=2: BanditGPT vs Supervised Baselines")
    logger.info(f"  {K2_CATALOG[K2_MODELS[0]]['display']}  vs  {K2_CATALOG[K2_MODELS[1]]['display']}")
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

    # --- Dev train/val split -----------------------------------------------
    # BanditGPT and supervised baselines both train on dev_train.
    # Dev metrics for Pareto frontier selection come from dev_val (unseen).
    logger.info("\n  Splitting dev into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    dev_train_k2, dev_train_emb_k2, dev_val_k2, dev_val_emb_k2 = (
        _split_dev_train_val(dev_data_k2, dev_emb_k2)
    )
    logger.info(f"    Dev-train: {len(dev_train_k2)}  Dev-val: {len(dev_val_k2)}")

    # --- Supervised static baselines (LLMRouter-style) ------------------
    # Same features (bge-m3 PCA), same objective (argmax reward), same data.
    # Isolates BanditGPT's online adaptation advantage over supervised routing.
    logger.info("\n  Phase 2b: Supervised static baselines (KNN/SVM/MLP) ...")
    supervised_k2: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        res = run_supervised_baseline(
            kind, K2_MODELS, costs_k2,
            dev_train_k2, dev_train_emb_k2,
            holdout_data_k2, holdout_emb_k2,
            n_trials=N_SEEDS, per_prompt=True,
        )
        supervised_k2[kind] = res
        logger.info(
            f"    {kind.upper():<4}: R={res['reward']:.4f} "
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f}"
        )

    # --- Phase 3: BanditGPT Pareto sweep (train on dev-train) ----------
    logger.info(
        f"\n  Phase 3: BanditGPT Pareto sweep "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    k2_alpha = tuned_k2["alpha"] if tuned_k2 is not None else ALPHA_START
    k2_neff = tuned_k2["prior_n_effective"] if tuned_k2 is not None else TARGET_NEFF
    k2_forgetting = tuned_k2["forgetting_factor"] if tuned_k2 is not None else 1.0
    # Use K=2 warmup priors filtered from the multi-model artifact when available.
    k2_warmup_path = (
        str(K2_WARMUP_FROM_MULTIMODEL_PATH)
        if K2_WARMUP_FROM_MULTIMODEL_PATH.exists()
        else str(DEFAULT_WARMUP_PRIORS_PATH)
    )

    bandit_pareto_k2 = run_pareto_sweep(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_data_k2, dev_train_emb_k2, holdout_emb_k2,
        k2_warmup_path, costs_k2, LAMBDA_VALUES_K2,
        N_SEEDS, use_corralling=True, label="banditGPT_warmup",
        dev_val_data=dev_val_k2, dev_val_emb=dev_val_emb_k2,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
    )

    # Tabula rasa ablation (no priors, no Corralling — genuine blank slate)
    logger.info(
        f"\n  Tabula rasa ablation "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k2 = run_pareto_sweep(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_data_k2, dev_train_emb_k2, holdout_emb_k2,
        None, costs_k2, LAMBDA_VALUES_K2,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
        dev_val_data=dev_val_k2, dev_val_emb=dev_val_emb_k2,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
    )

    # --- Phase 3b: Second pass — retrain on FULL dev (dev-selected points only) ----
    bg_dev_opt_idx_k2 = _dev_pareto_indices(bandit_pareto_k2)
    tr_dev_opt_idx_k2 = _dev_pareto_indices(tabula_pareto_k2)
    bg_dev_opt_lams_k2 = sorted({float(bandit_pareto_k2[i]["lambda"]) for i in bg_dev_opt_idx_k2})
    tr_dev_opt_lams_k2 = sorted({float(tabula_pareto_k2[i]["lambda"]) for i in tr_dev_opt_idx_k2})
    logger.info(
        "\n  Phase 3b: Full-dev retrain pass (holdout metrics only; dev-optimal lambdas) ..."
        f"\n    BanditGPT: {len(bg_dev_opt_lams_k2)}/{len(LAMBDA_VALUES_K2)} lambdas"
        f"\n    Tabula:    {len(tr_dev_opt_lams_k2)}/{len(LAMBDA_VALUES_K2)} lambdas"
    )
    bandit_full_k2 = _recompute_holdout_metrics_with_full_dev_training(
        models=K2_MODELS,
        catalog=K2_CATALOG,
        full_dev_data=dev_data_k2,
        full_dev_emb=dev_emb_k2,
        holdout_data=holdout_data_k2,
        holdout_emb=holdout_emb_k2,
        warmup_path=k2_warmup_path,
        costs=costs_k2,
        lambda_values=bg_dev_opt_lams_k2,
        n_trials=N_SEEDS,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
        use_corralling=True,
        label="banditGPT_warmup_full_dev",
    )
    tabula_full_k2 = _recompute_holdout_metrics_with_full_dev_training(
        models=K2_MODELS,
        catalog=K2_CATALOG,
        full_dev_data=dev_data_k2,
        full_dev_emb=dev_emb_k2,
        holdout_data=holdout_data_k2,
        holdout_emb=holdout_emb_k2,
        warmup_path=None,
        costs=costs_k2,
        lambda_values=tr_dev_opt_lams_k2,
        n_trials=N_SEEDS,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
        use_corralling=False,
        label="tabula_rasa_full_dev",
    )
    _patch_pareto_entries_holdout_metrics_inplace(bandit_pareto_k2, recomputed=bandit_full_k2)
    _patch_pareto_entries_holdout_metrics_inplace(tabula_pareto_k2, recomputed=tabula_full_k2)

    # Cold-start BanditGPT (priors only, 0 online training steps)
    logger.info(
        f"\n  Cold-start sweep (priors only, 0 training steps) "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    coldstart_pareto_k2 = run_coldstart_sweep(
        K2_MODELS, K2_CATALOG,
        holdout_data_k2, holdout_emb_k2,
        k2_warmup_path, costs_k2, LAMBDA_VALUES_K2,
        N_SEEDS, feature_dim=dim, use_corralling=True, label="coldstart",
        dev_data=dev_val_k2, dev_emb=dev_val_emb_k2,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
    )

    # --- Phase 4: K=2 learning curve -----------------------------------
    # Train on dev-train (same split as Pareto sweep) so the learning
    # curve and Pareto frontier use symmetric data.
    max_step = len(dev_train_k2)
    lc_checkpoints = [s for s in LEARNING_CURVE_CHECKPOINTS_K2 if s <= max_step]
    if max_step not in lc_checkpoints:
        lc_checkpoints.append(max_step)

    logger.info(f"\n  Phase 4: Learning curve ({N_SEEDS} seeds) ...")
    learning_curve_k2 = run_learning_curve(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_data_k2, dev_train_emb_k2, holdout_emb_k2,
        k2_warmup_path, costs_k2, N_SEEDS,
        lc_checkpoints, use_corralling=True, cost_penalty=0.0,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
    )

    # --- Phase 5: K=2 baselines ----------------------------------------
    logger.info("\n  Phase 5: K=2 baselines ...")

    oracle_r_k2, oracle_c_k2 = oracle_route(
        holdout_data_k2, K2_MODELS, costs_k2, cost_penalty=0.0,
    )
    oracle_r_k2_pure, oracle_c_k2_pure = oracle_r_k2, oracle_c_k2
    logger.info(
        f"    Oracle (pure quality): R={oracle_r_k2:.4f}  C=${oracle_c_k2:.6f}"
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

    ucb1_k2 = ucb1_online_route(
        dev_train_k2, holdout_data_k2, K2_MODELS, costs_k2,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(
        f"    UCB1 (non-contextual): R={ucb1_k2['reward']:.4f} "
        f"+/-{ucb1_k2['std_reward']:.4f}  "
        f"(greedy arm: {K2_CATALOG[ucb1_k2['greedy_arm']]['display']})"
    )

    # --- Phase 6: Dev-selected Pareto AUC & point comparisons -----------
    # PRIMARY metric: dev-selected Pareto AUC — hull built from
    # (dev_cost, dev_reward), then deployed points evaluated at
    # (holdout_cost, holdout_reward).  No holdout in selection step.
    # Mirrors the K=10 structure: BanditGPT vs coldstart/tabula rasa.
    # Supervised baselines (KNN/SVM/MLP) are single-point references.
    logger.info("\n  Phase 6: Dev-selected Pareto AUC & point comparisons ...")

    # Overlapping cost range (BanditGPT vs coldstart)
    bg_dev_costs_k2 = [p["dev_mean_cost"] for p in bandit_pareto_k2]
    cs_dev_costs_k2 = [p["dev_mean_cost"] for p in coldstart_pareto_k2]
    tr_dev_costs_k2 = [p["dev_mean_cost"] for p in tabula_pareto_k2]
    cost_lo_k2 = max(min(bg_dev_costs_k2), min(cs_dev_costs_k2), min(tr_dev_costs_k2))
    cost_hi_k2 = min(max(bg_dev_costs_k2), max(cs_dev_costs_k2), max(tr_dev_costs_k2))

    # Dev-selected Pareto AUC (primary)
    bg_ds_auc_k2, _, _, bg_dev_idx_k2 = dev_selected_pareto_auc(
        bandit_pareto_k2, cost_lo_k2, cost_hi_k2,
    )
    cs_ds_auc_k2, _, _, cs_dev_idx_k2 = dev_selected_pareto_auc(
        coldstart_pareto_k2, cost_lo_k2, cost_hi_k2,
    )
    tr_ds_auc_k2, _, _, tr_dev_idx_k2 = dev_selected_pareto_auc(
        tabula_pareto_k2, cost_lo_k2, cost_hi_k2,
    )

    # Oracle envelope AUC (reference — holdout-selected hyperparameters)
    bg_costs_k2 = [p["mean_cost"] for p in bandit_pareto_k2]
    cs_costs_k2 = [p["mean_cost"] for p in coldstart_pareto_k2]
    tr_costs_k2 = [p["mean_cost"] for p in tabula_pareto_k2]
    oracle_cost_lo = max(min(bg_costs_k2), min(cs_costs_k2), min(tr_costs_k2))
    oracle_cost_hi = min(max(bg_costs_k2), max(cs_costs_k2), max(tr_costs_k2))
    bg_oracle_auc_k2 = pareto_auc(
        bg_costs_k2,
        [p["mean_reward"] for p in bandit_pareto_k2],
        oracle_cost_lo, oracle_cost_hi,
    )
    cs_oracle_auc_k2 = pareto_auc(
        cs_costs_k2,
        [p["mean_reward"] for p in coldstart_pareto_k2],
        oracle_cost_lo, oracle_cost_hi,
    )
    tr_oracle_auc_k2 = pareto_auc(
        tr_costs_k2,
        [p["mean_reward"] for p in tabula_pareto_k2],
        oracle_cost_lo, oracle_cost_hi,
    )

    logger.info(
        f"    Dev-selected Pareto AUC (cost [{cost_lo_k2:.6f}, {cost_hi_k2:.6f}]):"
    )
    logger.info(f"      BanditGPT:  {bg_ds_auc_k2:.4f} ({len(bg_dev_idx_k2)} dev-optimal pts)")
    logger.info(f"      Coldstart:  {cs_ds_auc_k2:.4f} ({len(cs_dev_idx_k2)} dev-optimal pts)")
    logger.info(f"      Tabula rasa:{tr_ds_auc_k2:.4f} ({len(tr_dev_idx_k2)} dev-optimal pts)")
    logger.info(f"      BG vs CS:   {bg_ds_auc_k2 - cs_ds_auc_k2:+.4f}")
    logger.info(f"      BG vs TR:   {bg_ds_auc_k2 - tr_ds_auc_k2:+.4f}")
    logger.info(
        f"    Oracle envelope AUC (ref): BanditGPT={bg_oracle_auc_k2:.4f} "
        f"Coldstart={cs_oracle_auc_k2:.4f} Tabula={tr_oracle_auc_k2:.4f}"
    )

    # Paired bootstrap CI for dev-selected AUC difference (BanditGPT vs coldstart)
    logger.info("    Computing bootstrap CI for Pareto AUC difference ...")
    bg_pp_r_by_lam: Dict[float, np.ndarray] = {}
    bg_pp_c_by_lam: Dict[float, np.ndarray] = {}
    for p in bandit_pareto_k2:
        if p.get("per_seed_per_prompt_rewards") is not None:
            bg_pp_r_by_lam[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            bg_pp_c_by_lam[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )
    cs_pp_r_by_lam: Dict[float, np.ndarray] = {}
    cs_pp_c_by_lam: Dict[float, np.ndarray] = {}
    for p in coldstart_pareto_k2:
        if p.get("per_seed_per_prompt_rewards") is not None:
            cs_pp_r_by_lam[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            cs_pp_c_by_lam[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )

    bg_boot_pp_r, bg_boot_pp_c = _extract_dev_optimal_per_prompt(
        bandit_pareto_k2, bg_dev_idx_k2,
        bg_pp_r_by_lam, bg_pp_c_by_lam, "lambda",
    )
    cs_boot_pp_r, cs_boot_pp_c = _extract_dev_optimal_per_prompt(
        coldstart_pareto_k2, cs_dev_idx_k2,
        cs_pp_r_by_lam, cs_pp_c_by_lam, "lambda",
    )
    bootstrap_k2 = bootstrap_pareto_auc_difference(
        bg_boot_pp_r, bg_boot_pp_c,
        cs_boot_pp_r, cs_boot_pp_c,
        cost_lo=cost_lo_k2, cost_hi=cost_hi_k2,
        n_holdout=len(holdout_data_k2), n_bootstrap=1_000,
    )
    logger.info(
        f"    Bootstrap AUC diff (BG vs CS): {bootstrap_k2['observed_diff']:+.4f} "
        f"95% CI [{bootstrap_k2['ci_95_lower']:+.4f}, "
        f"{bootstrap_k2['ci_95_upper']:+.4f}] "
        f"p={bootstrap_k2['p_value']:.4g}"
    )

    # Point comparisons: BanditGPT dev-selected best vs supervised baselines
    bg_dev_optimal_k2 = [bandit_pareto_k2[i] for i in bg_dev_idx_k2]
    bg_best_dev_k2 = max(bg_dev_optimal_k2, key=lambda p: p["dev_mean_reward"])

    supervised_point_tests_k2: Dict[str, Dict] = {}
    logger.info("\n    Supervised baseline point comparisons:")
    for kind, sv_result in supervised_k2.items():
        bg_pp = np.array(bg_best_dev_k2["per_seed_per_prompt_rewards"])
        bg_ensemble_pp = bg_pp.mean(axis=0)
        sv_pp = np.array(sv_result["per_seed_per_prompt_rewards"])
        sv_ensemble_pp = sv_pp.mean(axis=0)
        t_res = scipy_stats.ttest_rel(bg_ensemble_pp, sv_ensemble_pp)
        supervised_point_tests_k2[kind] = {
            "banditgpt_reward": float(bg_ensemble_pp.mean()),
            "supervised_reward": sv_result["reward"],
            "delta": float(bg_ensemble_pp.mean()) - sv_result["reward"],
            "t_stat": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
            "df": len(bg_ensemble_pp) - 1,
        }
        sig = "**" if t_res.pvalue < 0.05 else ""
        logger.info(
            f"      {kind.upper():<4}: BG={float(bg_ensemble_pp.mean()):.4f} "
            f"vs {kind.upper()}={sv_result['reward']:.4f} "
            f"delta={float(bg_ensemble_pp.mean()) - sv_result['reward']:+.4f} "
            f"p={t_res.pvalue:.4g}{sig}"
        )

    # Assemble K=2 summary
    best_sv_kind = max(supervised_k2, key=lambda k: supervised_k2[k]["reward"])
    best_sv = supervised_k2[best_sv_kind]

    logger.info(f"\n  K=2 SUMMARY (dev-selected Pareto AUC primary):")
    logger.info(f"    Oracle (pure quality): {oracle_r_k2_pure:.4f}")
    logger.info(
        f"    Dev-selected Pareto AUC: BanditGPT={bg_ds_auc_k2:.4f} "
        f"Coldstart={cs_ds_auc_k2:.4f} Tabula={tr_ds_auc_k2:.4f}"
    )
    logger.info(
        f"    Bootstrap 95% CI (BG vs CS): [{bootstrap_k2['ci_95_lower']:+.4f}, "
        f"{bootstrap_k2['ci_95_upper']:+.4f}] p={bootstrap_k2['p_value']:.4g}"
    )
    logger.info(
        f"    Best supervised: {best_sv_kind.upper()} "
        f"R={best_sv['reward']:.4f} +/-{best_sv['std_reward']:.4f}"
    )
    logger.info(f"    Random:       {random_k2['reward']:.4f}")
    logger.info(f"    UCB1 (non-ctx): {ucb1_k2['reward']:.4f}")

    results_all["K2"] = {
        "models": K2_MODELS,
        "n_dev": len(dev_data_k2),
        "n_holdout": len(holdout_data_k2),
        "oracle": {"reward": oracle_r_k2, "cost": oracle_c_k2},
        "oracle_pure_quality": {"reward": oracle_r_k2_pure, "cost": oracle_c_k2_pure},
        "static": static_k2,
        "random": random_k2,
        "ucb1": ucb1_k2,
        "supervised": supervised_k2,
        "supervised_point_tests": supervised_point_tests_k2,
        "banditgpt_pareto": bandit_pareto_k2,
        "coldstart_pareto": coldstart_pareto_k2,
        "tabula_rasa_pareto": tabula_pareto_k2,
        "learning_curve": learning_curve_k2,
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k2, cost_hi_k2],
            "banditgpt": bg_ds_auc_k2,
            "coldstart": cs_ds_auc_k2,
            "tabula_rasa": tr_ds_auc_k2,
            "advantage_vs_coldstart": bg_ds_auc_k2 - cs_ds_auc_k2,
            "advantage_vs_tabula": bg_ds_auc_k2 - tr_ds_auc_k2,
            "bootstrap_ci": bootstrap_k2,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward) — no holdout data in selection.  Deployed "
                "points are (holdout_cost, holdout_reward) of dev-optimal "
                "hyperparameters.  Bootstrap CI over 1,000 holdout resamples "
                "with fixed dev indices."
            ),
        },
        "pareto_auc_oracle_envelope": {
            "cost_range": [oracle_cost_lo, oracle_cost_hi],
            "banditgpt": bg_oracle_auc_k2,
            "coldstart": cs_oracle_auc_k2,
            "tabula_rasa": tr_oracle_auc_k2,
            "note": (
                "Oracle envelope AUC: holdout-selected hyperparameters.  "
                "Upper bound — not deployable.  Retained for reference only."
            ),
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
    logger.info(f"  Embedding K=10 prompts (PCA={pca.n_components_} comp) ...")
    train_emb_k10 = embed_dataset(train_data_k10, encoder, pca)
    holdout_emb_k10 = embed_dataset(holdout_data_k10, encoder, pca)

    # --- Dev train/val split (K=10) ------------------------------------
    logger.info(f"  Splitting K=10 train into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    train_train_k10, train_train_emb_k10, train_val_k10, train_val_emb_k10 = (
        _split_dev_train_val(train_data_k10, train_emb_k10)
    )
    logger.info(
        f"    Train-train: {len(train_train_k10)}  "
        f"Train-val: {len(train_val_k10)}"
    )

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

    eg_k10 = best_static_noisy_route(
        train_train_k10, holdout_data_k10, K10_MODELS, costs_k10,
        n_trials=N_SEEDS * 4,
    )
    logger.info(f"    Best-static+noise: R={eg_k10['reward']:.4f}")

    # UCB1 online baseline (non-contextual) — ablates value of features
    ucb1_k10 = ucb1_online_route(
        train_train_k10, holdout_data_k10, K10_MODELS, costs_k10,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(
        f"    UCB1 (non-contextual): R={ucb1_k10['reward']:.4f} "
        f"+/-{ucb1_k10['std_reward']:.4f}"
    )

    # Supervised static baselines (LLMRouter-style) — same features, same objective
    logger.info("\n  Supervised static baselines (KNN/SVM/MLP) ...")
    supervised_k10: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        res = run_supervised_baseline(
            kind, K10_MODELS, costs_k10,
            train_train_k10, train_train_emb_k10,
            holdout_data_k10, holdout_emb_k10,
            n_trials=N_SEEDS, per_prompt=True,
        )
        supervised_k10[kind] = res
        logger.info(
            f"    {kind.upper():<4}: R={res['reward']:.4f} "
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f}"
        )

    # --- BanditGPT Pareto sweep ----------------------------------------
    k10_warmup_path = str(MULTIMODEL_WARMUP_PRIORS_PATH)
    logger.info(f"  Using warmup priors: {MULTIMODEL_WARMUP_PRIORS_PATH.name}")

    logger.info(
        f"\n  BanditGPT K=10 Pareto sweep "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    k10_alpha = tuned_k10["alpha"] if tuned_k10 is not None else ALPHA_START
    k10_neff = tuned_k10["prior_n_effective"] if tuned_k10 is not None else TARGET_NEFF
    k10_forgetting = tuned_k10["forgetting_factor"] if tuned_k10 is not None else 1.0
    bandit_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_data_k10, train_train_emb_k10, holdout_emb_k10,
        k10_warmup_path, costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=True, label="banditGPT",
        dev_val_data=train_val_k10, dev_val_emb=train_val_emb_k10,
        alpha=k10_alpha,
        prior_n_effective=k10_neff,
        forgetting_factor=k10_forgetting,
    )

    # Tabula rasa ablation (no priors, no Corralling)
    logger.info(
        f"\n  Tabula rasa K=10 ablation "
        f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ..."
    )
    tabula_pareto_k10 = run_pareto_sweep(
        K10_MODELS, K10_CATALOG,
        train_train_k10, holdout_data_k10, train_train_emb_k10, holdout_emb_k10,
        None, costs_k10, LAMBDA_VALUES_K10,
        N_SEEDS, use_corralling=False, label="tabula_rasa",
        dev_val_data=train_val_k10, dev_val_emb=train_val_emb_k10,
        alpha=k10_alpha,
        prior_n_effective=k10_neff,
        forgetting_factor=k10_forgetting,
    )

    # --- Second pass — retrain on FULL dev (dev-selected points only), eval on holdout ---
    bg_dev_opt_idx_k10 = _dev_pareto_indices(bandit_pareto_k10)
    tr_dev_opt_idx_k10 = _dev_pareto_indices(tabula_pareto_k10)
    bg_dev_opt_lams_k10 = sorted({float(bandit_pareto_k10[i]["lambda"]) for i in bg_dev_opt_idx_k10})
    tr_dev_opt_lams_k10 = sorted({float(tabula_pareto_k10[i]["lambda"]) for i in tr_dev_opt_idx_k10})
    logger.info(
        "\n  Full-dev retrain pass for K=10 (holdout metrics only; dev-optimal lambdas) ..."
        f"\n    BanditGPT: {len(bg_dev_opt_lams_k10)}/{len(LAMBDA_VALUES_K10)} lambdas"
        f"\n    Tabula:    {len(tr_dev_opt_lams_k10)}/{len(LAMBDA_VALUES_K10)} lambdas"
    )
    bandit_full_k10 = _recompute_holdout_metrics_with_full_dev_training(
        models=K10_MODELS,
        catalog=K10_CATALOG,
        full_dev_data=train_data_k10,
        full_dev_emb=train_emb_k10,
        holdout_data=holdout_data_k10,
        holdout_emb=holdout_emb_k10,
        warmup_path=k10_warmup_path,
        costs=costs_k10,
        lambda_values=bg_dev_opt_lams_k10,
        n_trials=N_SEEDS,
        alpha=k10_alpha,
        prior_n_effective=k10_neff,
        forgetting_factor=k10_forgetting,
        use_corralling=True,
        label="banditGPT_full_dev",
    )
    tabula_full_k10 = _recompute_holdout_metrics_with_full_dev_training(
        models=K10_MODELS,
        catalog=K10_CATALOG,
        full_dev_data=train_data_k10,
        full_dev_emb=train_emb_k10,
        holdout_data=holdout_data_k10,
        holdout_emb=holdout_emb_k10,
        warmup_path=None,
        costs=costs_k10,
        lambda_values=tr_dev_opt_lams_k10,
        n_trials=N_SEEDS,
        alpha=k10_alpha,
        prior_n_effective=k10_neff,
        forgetting_factor=k10_forgetting,
        use_corralling=False,
        label="tabula_rasa_full_dev",
    )
    _patch_pareto_entries_holdout_metrics_inplace(bandit_pareto_k10, recomputed=bandit_full_k10)
    _patch_pareto_entries_holdout_metrics_inplace(tabula_pareto_k10, recomputed=tabula_full_k10)

    # --- K=10 summary: Dev-selected Pareto AUC --------------------------
    best_static_m = max(static_k10, key=lambda m: static_k10[m]["reward"])

    # Dev-selected Pareto AUC (primary)
    bg_dev_costs_k10 = [p["dev_mean_cost"] for p in bandit_pareto_k10]
    tr_dev_costs_k10 = [p["dev_mean_cost"] for p in tabula_pareto_k10]
    cost_lo_k10 = max(min(bg_dev_costs_k10), min(tr_dev_costs_k10))
    cost_hi_k10 = min(max(bg_dev_costs_k10), max(tr_dev_costs_k10))

    bg_ds_auc_k10, _, _, bg_dev_idx_k10 = dev_selected_pareto_auc(
        bandit_pareto_k10, cost_lo_k10, cost_hi_k10,
    )
    tr_ds_auc_k10, _, _, tr_dev_idx_k10 = dev_selected_pareto_auc(
        tabula_pareto_k10, cost_lo_k10, cost_hi_k10,
    )

    # Oracle envelope AUC (reference)
    bg_costs_k10 = [p["mean_cost"] for p in bandit_pareto_k10]
    tr_costs_k10 = [p["mean_cost"] for p in tabula_pareto_k10]
    oracle_cost_lo_k10 = max(min(bg_costs_k10), min(tr_costs_k10))
    oracle_cost_hi_k10 = min(max(bg_costs_k10), max(tr_costs_k10))
    bg_oracle_auc_k10 = pareto_auc(
        bg_costs_k10,
        [p["mean_reward"] for p in bandit_pareto_k10],
        oracle_cost_lo_k10, oracle_cost_hi_k10,
    )
    tr_oracle_auc_k10 = pareto_auc(
        tr_costs_k10,
        [p["mean_reward"] for p in tabula_pareto_k10],
        oracle_cost_lo_k10, oracle_cost_hi_k10,
    )

    # Paired bootstrap CI for K=10 AUC difference
    logger.info("  Computing K=10 bootstrap CI ...")
    bg_pp_r_k10: Dict[float, np.ndarray] = {}
    bg_pp_c_k10: Dict[float, np.ndarray] = {}
    for p in bandit_pareto_k10:
        if p.get("per_seed_per_prompt_rewards") is not None:
            bg_pp_r_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            bg_pp_c_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )
    tr_pp_r_k10: Dict[float, np.ndarray] = {}
    tr_pp_c_k10: Dict[float, np.ndarray] = {}
    for p in tabula_pareto_k10:
        if p.get("per_seed_per_prompt_rewards") is not None:
            tr_pp_r_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_rewards"],
            )
            tr_pp_c_k10[p["lambda"]] = np.array(
                p["per_seed_per_prompt_costs"],
            )

    bg_boot_pp_r_k10, bg_boot_pp_c_k10 = _extract_dev_optimal_per_prompt(
        bandit_pareto_k10, bg_dev_idx_k10,
        bg_pp_r_k10, bg_pp_c_k10, "lambda",
    )
    tr_boot_pp_r_k10, tr_boot_pp_c_k10 = _extract_dev_optimal_per_prompt(
        tabula_pareto_k10, tr_dev_idx_k10,
        tr_pp_r_k10, tr_pp_c_k10, "lambda",
    )
    bootstrap_k10 = bootstrap_pareto_auc_difference(
        bg_boot_pp_r_k10, bg_boot_pp_c_k10,
        tr_boot_pp_r_k10, tr_boot_pp_c_k10,
        cost_lo=cost_lo_k10, cost_hi=cost_hi_k10,
        n_holdout=len(holdout_data_k10), n_bootstrap=1_000,
    )

    logger.info(f"\n  K=10 SUMMARY (dev-selected Pareto AUC primary):")
    logger.info(f"    Oracle:       {oracle_r_k10:.4f}")
    logger.info(
        f"    Dev-selected AUC: BanditGPT={bg_ds_auc_k10:.4f} vs "
        f"Tabula rasa={tr_ds_auc_k10:.4f} "
        f"(adv: {bg_ds_auc_k10 - tr_ds_auc_k10:+.4f})"
    )
    logger.info(
        f"    Bootstrap 95% CI: [{bootstrap_k10['ci_95_lower']:+.4f}, "
        f"{bootstrap_k10['ci_95_upper']:+.4f}] "
        f"p={bootstrap_k10['p_value']:.4g}"
    )
    logger.info(
        f"    Oracle envelope (ref): BanditGPT={bg_oracle_auc_k10:.4f} vs "
        f"Tabula rasa={tr_oracle_auc_k10:.4f}"
    )
    logger.info(
        f"    Best static:  {static_k10[best_static_m]['reward']:.4f} "
        f"({K10_CATALOG[best_static_m]['display']})"
    )
    logger.info(f"    Best-static+noise: {eg_k10['reward']:.4f}")
    logger.info(f"    UCB1 (non-ctx):    {ucb1_k10['reward']:.4f}")
    for kind in ("knn", "svm", "mlp"):
        s = supervised_k10[kind]
        logger.info(
            f"    {kind.upper():<4} (supervised): {s['reward']:.4f} "
            f"+/-{s['std_reward']:.4f}"
        )
    logger.info(f"    Random:            {random_k10['reward']:.4f}")

    results_all["K10"] = {
        "models": [{"id": m, **K10_CATALOG[m]} for m in K10_MODELS],
        "n_train": len(train_data_k10),
        "n_holdout": len(holdout_data_k10),
        "oracle": {"reward": oracle_r_k10, "cost": oracle_c_k10},
        "static": {m: static_k10[m] for m in K10_MODELS},
        "best_static": {
            "model": best_static_m,
            "reward": static_k10[best_static_m]["reward"],
            "cost": static_k10[best_static_m]["cost"],
        },
        "random": random_k10,
        "best_static_noisy": eg_k10,
        "ucb1": ucb1_k10,
        "supervised": supervised_k10,
        "banditgpt_pareto": bandit_pareto_k10,
        "tabula_rasa_pareto": tabula_pareto_k10,
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k10, cost_hi_k10],
            "banditgpt": bg_ds_auc_k10,
            "tabula_rasa": tr_ds_auc_k10,
            "advantage": bg_ds_auc_k10 - tr_ds_auc_k10,
            "bootstrap_ci": bootstrap_k10,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward).  Deployed = holdout performance of dev-optimal "
                "hyperparameters.  Bootstrap CI with fixed dev indices."
            ),
        },
        "pareto_auc_oracle_envelope": {
            "cost_range": [oracle_cost_lo_k10, oracle_cost_hi_k10],
            "banditgpt": bg_oracle_auc_k10,
            "tabula_rasa": tr_oracle_auc_k10,
            "advantage": bg_oracle_auc_k10 - tr_oracle_auc_k10,
            "note": "Oracle envelope — holdout-selected hyperparameters (reference only).",
        },
        "n_trials": N_SEEDS,
    }

    # ==================================================================
    # Serialise (strip bulky per-prompt arrays to keep JSON manageable)
    # ==================================================================
    def _strip_per_prompt(obj: Any) -> Any:
        """Recursively drop per_prompt_rewards / per_seed_per_prompt_rewards."""
        if isinstance(obj, dict):
            return {
                k: _strip_per_prompt(v) for k, v in obj.items()
                if k not in (
                    "per_prompt_rewards", "per_prompt_costs",
                    "per_seed_per_prompt_rewards", "per_seed_per_prompt_costs",
                )
            }
        if isinstance(obj, list):
            return [_strip_per_prompt(v) for v in obj]
        return obj

    out_path = output_dir / "prequential_results.json"
    with open(out_path, "w") as f:
        json.dump(_strip_per_prompt(results_all), f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nResults -> {out_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


# ============================================================================
# CLI entry point
# ============================================================================


if __name__ == "__main__":
    run_experiment()
