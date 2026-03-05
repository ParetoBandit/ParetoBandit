#!/usr/bin/env python3
"""
BanditGPT: Train-then-Freeze Evaluation vs Peer-Reviewed Baselines.

Compares BanditGPT against LLMRouter supervised static baselines
(KNN, SVM, MLP; ref: UIUC 2025) and standard online baselines (UCB1,
random, best-static) using literature-standard metrics: Pareto frontier,
PerfGain, CostSave, Gap@Oracle, and learning curves.

Protocol
--------
1. **Canonical train/val/holdout splits.**
   Data comes from ``dev_rewards_complete_all_models.jsonl.gz`` (2,854
   prompts; split into prior-train and online-learn via
   ``splits_three_way.json``) and
   ``holdout_rewards_complete_all_models.jsonl.gz`` (1,500 prompts).
   Rewards are derived via :func:`extract_reward` (mean of
   vote × confidence across multi-judge panel).

2. **Train-then-freeze evaluation.**
   BanditGPT trains on the dev set with oracle rewards, then is frozen
   for evaluation on the holdout set.  Supervised baselines are trained
   on the same dev data and frozen identically.

3. **Symmetric data access.**
   All methods have access to the same dev set.  For any selection step
   (e.g., BanditGPT cost-penalty lambda), we use a dev-train/dev-val
   split: train/tune on dev-train, select on dev-val.  No method sees
   holdout data before evaluation.

   **Note on warmup priors:** BanditGPT is initialised with warmup
   priors derived from a separate prior-training pool (43-model offline
   data).  The warmup priors are an architectural feature of BanditGPT
   (transfer learning from a broader model registry), not an unfair
   data advantage.

4. **Dev-selected deployable Pareto frontier (primary metric).**
   The dev set is split into train (80%) and val (20%).  BanditGPT
   trains on dev-train; dev metrics for frontier selection come from
   dev-val (eliminating train-set evaluation asymmetry).  The Pareto
   hull is built from (dev_val_cost, dev_val_reward); for those
   dev-optimal settings, holdout cost and holdout reward are
   extracted.  No holdout or training data enters the selection step.

5. **Comparison metrics (LLMRouterBench conventions).**
   For each LLMRouter supervised baseline, we compute:
   - **PerfGain**: BanditGPT interpolated reward minus baseline reward
     at the baseline's cost (isocost comparison).
   - **CostSave**: baseline cost minus BanditGPT interpolated cost at
     the baseline's reward (iso-quality comparison).
   - **Gap@Oracle**: remaining reward gap to instance-wise optimal.

6. **Greedy frozen evaluation.**
   When evaluating a frozen BanditRouter, the UCB exploration bonus is
   set to zero (alpha=0) so that holdout scores reflect the learned
   policy under pure exploitation.

7. **Statistical reporting.**
   - *Post-hoc point comparisons*: paired t-tests of dev-selected
     BanditGPT vs each supervised baseline (KNN, SVM, MLP), with
     Holm-Bonferroni correction across the baselines.
   - *Stability*: across-seed t-tests (df = N_SEEDS - 1).

Outputs (``results/``)
    prequential_results.json
"""

import copy
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
from statsmodels.stats.multitest import multipletests
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.calibration import embed_prompt
from bandit_gpt.config import (
    ARTIFACTS_DIR,
    DEFAULT_PCA_PATH,
    DEFAULT_SENTENCE_TRANSFORMER,
    K2_WARMUP_PRIORS_PATH,
    K3_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
    K3_MODELS_PATH,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models, load_model_catalog, req_cost
from utils.router_factory import create_experiment_router
from utils.supervised_baselines import (
    run_supervised_baseline,
    run_supervised_learning_curve,
    tune_supervised_hparams,
)
from utils.pareto import (
    interpolate_pareto_reward,
    interpolate_pareto_cost,
    dev_selected_pareto_auc,
)
from utils.metrics import perfgain, costsave, gap_at_oracle
from utils.embeddings import load_embedding_cache, embed_dataset_cached

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Formatting helpers
# ============================================================================


def _fmt(val: Optional[float], suffix: str = "") -> str:
    """Format an optional float for log messages."""
    if val is None:
        return "N/A"
    return f"{val:+.4f}{suffix}"


# ============================================================================
# Model catalogs
# ============================================================================

K2_MODELS: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4.1",
]

_PRICES_K2 = get_prices_for_models(K2_MODELS)

K2_CATALOG: Dict[str, Dict] = {
    "meta-llama/llama-3.1-8b-instruct": {
        "display": "Llama-3.1-8B",
        **_PRICES_K2["meta-llama/llama-3.1-8b-instruct"],
        "cost": req_cost(
            _PRICES_K2["meta-llama/llama-3.1-8b-instruct"]["input_cost_per_m"],
            _PRICES_K2["meta-llama/llama-3.1-8b-instruct"]["output_cost_per_m"],
        ),
        "tier": "cheap",
    },
    "openai/gpt-4.1": {
        "display": "GPT-4.1",
        **_PRICES_K2["openai/gpt-4.1"],
        "cost": req_cost(
            _PRICES_K2["openai/gpt-4.1"]["input_cost_per_m"],
            _PRICES_K2["openai/gpt-4.1"]["output_cost_per_m"],
        ),
        "tier": "expensive",
    },
}

K3_MODELS, K3_CATALOG = load_model_catalog(K3_MODELS_PATH)


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
LAMBDA_VALUES_K3: List[float] = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.0, 1.5, 2.0, 3.0, 5.0,
]

def _make_learning_curve_checkpoints(n_train: int) -> List[int]:
    """Build learning-curve checkpoint list adapted to the training set size.

    Dense at the start (where learning is fastest), sparser later.
    Always includes 0 and n_train as endpoints.
    """
    candidates = [0, 10, 25, 50, 100, 150, 200, 300, 400, 500,
                  600, 700, 800, 900, 1000, 1200, 1500, 2000]
    checkpoints = [s for s in candidates if s <= n_train]
    if n_train not in checkpoints:
        checkpoints.append(n_train)
    return checkpoints

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


EARLY_STOP_EVAL_INTERVAL: int = 50
EARLY_STOP_PATIENCE: int = 5
EARLY_STOP_MIN_STEPS: int = 50


def train_bandit_with_early_stopping(
    router,
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    val_data: List[Dict],
    val_embeddings: List[np.ndarray],
    models: List[str],
    costs: Dict[str, float],
    r_min: float,
    r_range: float,
    *,
    eval_interval: int = EARLY_STOP_EVAL_INTERVAL,
    patience: int = EARLY_STOP_PATIENCE,
    min_steps: int = EARLY_STOP_MIN_STEPS,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, List[Dict]]:
    """Train a BanditRouter with early stopping on validation reward.

    Periodically evaluates the frozen router on the dev-val set.
    When the validation reward fails to improve for *patience*
    consecutive evaluations, training stops and the router is
    restored to the checkpoint with the best validation reward.

    This is analogous to early stopping in supervised learning
    (e.g., MLP's ``early_stopping=True``) and prevents Corralling
    weight oscillation from degrading a good policy.

    Args:
        router: A BanditRouter instance.
        train_data: Dev-train prompts with reward dicts.
        train_embeddings: Pre-computed feature vectors for dev-train.
        val_data: Dev-val prompts with reward dicts.
        val_embeddings: Pre-computed feature vectors for dev-val.
        models: Candidate model IDs.
        costs: Per-model cost dict.
        r_min: Minimum raw reward (for normalization).
        r_range: Range of raw rewards (max - min).
        eval_interval: Evaluate on val every *eval_interval* training steps.
        patience: Stop after this many consecutive non-improving evaluations.
        min_steps: Minimum training steps before the first evaluation.
        shuffle: If True, present training data in random order.
        rng: Explicit numpy Generator for the training permutation.

    Returns:
        Tuple of (best_step, eval_history) where *best_step* is the
        training step at which the best validation reward was observed,
        and *eval_history* is a list of {step, val_reward, val_cost}
        dicts for each evaluation checkpoint.
    """
    n_total = len(train_data)
    if shuffle:
        order = (
            rng.permutation(n_total)
            if rng is not None
            else np.random.permutation(n_total)
        )
    else:
        order = np.arange(n_total)

    best_val_reward = -np.inf
    best_step = 0
    best_router_state: Optional[Any] = None
    evals_without_improvement = 0
    eval_history: List[Dict] = []

    for step_idx, idx in enumerate(order):
        p, x = train_data[idx], train_embeddings[idx]
        model, log = router.route(x, total_steps=n_total)
        raw_reward = p["rewards"][model]
        norm_reward = (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
        router.process_feedback(log.request_id, norm_reward)
        current_step = step_idx + 1

        if current_step >= min_steps and current_step % eval_interval == 0:
            val_r, val_c, _, _, _ = evaluate_frozen(
                router, val_data, val_embeddings, costs, n_total,
            )
            eval_history.append({
                "step": current_step,
                "val_reward": val_r,
                "val_cost": val_c,
            })
            if val_r > best_val_reward:
                best_val_reward = val_r
                best_step = current_step
                best_router_state = copy.deepcopy(router)
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if evals_without_improvement >= patience:
                break

    if best_router_state is None:
        best_step = n_total
        return best_step, eval_history

    router.__dict__.update(best_router_state.__dict__)
    return best_step, eval_history


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
    rng_state = np.random.get_state()

    r_total = c_total = 0.0
    model_counts: Dict[str, int] = defaultdict(int)
    prompt_rewards: Optional[List[float]] = [] if per_prompt else None
    prompt_costs: Optional[List[float]] = [] if per_prompt else None

    with router.exploit():
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
    tabula_rasa_alpha: Optional[float] = None,
    tabula_rasa_forgetting_factor: Optional[float] = None,
) -> List[Dict]:
    """Sweep cost penalty lambda with N trials per point.

    For each (lambda, trial): instantiate router, train on *train_data*
    with early stopping on *dev_val_data*, freeze, and evaluate on the
    holdout set.

    Training uses early stopping on dev-val reward (patience-based,
    best-checkpoint restoration) to prevent overfitting.  Dev metrics
    (``dev_mean_cost``, ``dev_mean_reward``) are computed on the same
    dev-val split, eliminating the train-set evaluation asymmetry
    between online and static methods.

    Args:
        tabula_rasa_alpha: Per-expert alpha for the tabula-rasa expert
            inside Corralling.  ``None`` uses the legacy schedule.
        tabula_rasa_forgetting_factor: Per-expert forgetting factor for
            the tabula-rasa expert.  ``None`` inherits ``forgetting_factor``.

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
        trial_best_steps: List[int] = []
        all_per_prompt: List[List[float]] = []
        all_per_prompt_costs: List[List[float]] = []
        all_model_counts: List[Dict[str, int]] = []
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
                tabula_rasa_alpha=tabula_rasa_alpha,
                tabula_rasa_forgetting_factor=tabula_rasa_forgetting_factor,
            )
            best_step, _hist = train_bandit_with_early_stopping(
                router, train_data, train_emb,
                val_d, val_e, models, costs,
                r_min, r_range, rng=trial_rng,
            )
            trial_best_steps.append(best_step)
            r, c, mc, pp, pp_c = evaluate_frozen(
                router, eval_data, eval_emb, costs, burn_in, per_prompt=True,
            )
            trial_r.append(r)
            trial_c.append(c)
            all_model_counts.append(mc)
            all_per_prompt.append(pp)
            all_per_prompt_costs.append(pp_c)
            dev_r, dev_c, _, _, _ = evaluate_frozen(
                router, val_d, val_e, costs, burn_in,
            )
            trial_dev_c.append(dev_c)
            trial_dev_r.append(dev_r)

        mean_best_step = float(np.mean(trial_best_steps))

        agg_counts: Dict[str, float] = {m: 0.0 for m in models}
        for mc in all_model_counts:
            for m in models:
                agg_counts[m] += mc.get(m, 0)
        total_routed = sum(agg_counts.values()) or 1.0
        routing_fractions = {
            m: agg_counts[m] / total_routed for m in models
        }

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
            "mean_best_step": mean_best_step,
            "model_counts": {m: int(agg_counts[m]) for m in models},
            "routing_fractions": routing_fractions,
        })
        most_routed = max(routing_fractions, key=routing_fractions.get)
        most_routed_short = most_routed.split("/")[-1]
        logger.info(
            f"    lambda={lam:<6} | R={np.mean(trial_r):.4f}+/-{np.std(trial_r):.4f} "
            f"| C=${np.mean(trial_c):.6f} | stop@{mean_best_step:.0f} "
            f"| top={most_routed_short} ({routing_fractions[most_routed]:.0%})"
        )
    return results


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
    tabula_rasa_alpha: Optional[float] = None,
    tabula_rasa_forgetting_factor: Optional[float] = None,
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
        all_per_prompt: List[List[float]] = []
        all_per_prompt_costs: List[List[float]] = []
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
                tabula_rasa_alpha=tabula_rasa_alpha,
                tabula_rasa_forgetting_factor=tabula_rasa_forgetting_factor,
            )
            r, c, _, pp, pp_c = evaluate_frozen(
                router, eval_data, eval_emb, costs, burn_in, per_prompt=True,
            )
            trial_r.append(r)
            trial_c.append(c)
            all_per_prompt.append(pp)
            all_per_prompt_costs.append(pp_c)
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
            "per_seed_per_prompt_rewards": all_per_prompt,
            "per_seed_per_prompt_costs": all_per_prompt_costs,
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
    tabula_rasa_alpha: Optional[float] = None,
    tabula_rasa_forgetting_factor: Optional[float] = None,
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
        tabula_rasa_alpha: Per-expert alpha for the tabula-rasa expert.
        tabula_rasa_forgetting_factor: Per-expert forgetting factor for
            the tabula-rasa expert.

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
            tabula_rasa_alpha=tabula_rasa_alpha,
            tabula_rasa_forgetting_factor=tabula_rasa_forgetting_factor,
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
    """Run the full K=2 and K=3 evaluation and serialise results."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 0 — shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components (unified for K=2 and K=3)")

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

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
                "evaluation asymmetry between online and static methods.  "
                "No holdout data enters hyperparameter selection."
            ),
            "early_stopping": {
                "enabled": True,
                "metric": "dev_val_reward",
                "eval_interval": EARLY_STOP_EVAL_INTERVAL,
                "patience": EARLY_STOP_PATIENCE,
                "min_steps": EARLY_STOP_MIN_STEPS,
                "checkpoint_restore": True,
                "note": (
                    "BanditGPT and tabula rasa training use early stopping "
                    "on dev-val reward, with best-checkpoint restoration via "
                    "deepcopy.  Analogous to MLP early_stopping=True."
                ),
            },
        },
    }

    # ------------------------------------------------------------------
    # Phase 0 — dev-val-selected hyperparameters from Appendix H (required)
    # ------------------------------------------------------------------
    hparams_dir = (
        Path(__file__).resolve().parent.parent / "appendix"
        / "H_alpha_neff_ablation" / "results"
    )
    hparams_k2_path = hparams_dir / "best_hparams_k2.json"
    hparams_k3_path = hparams_dir / "best_hparams_k3.json"
    hparams_k2_tr_path = hparams_dir / "best_hparams_k2_tabula_rasa.json"
    hparams_k3_tr_path = hparams_dir / "best_hparams_k3_tabula_rasa.json"

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
    tuned_k3 = _load_hparams(hparams_k3_path, "K3")
    tuned_k2_tr = _load_hparams(hparams_k2_tr_path, "K2")
    tuned_k3_tr = _load_hparams(hparams_k3_tr_path, "K3")

    _ablation_script = (
        "experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py"
    )
    _missing_hparams: List[str] = []
    for label, tuned, path in [
        ("K=2 BanditGPT", tuned_k2, hparams_k2_path),
        ("K=3 BanditGPT", tuned_k3, hparams_k3_path),
        ("K=2 Tabula Rasa", tuned_k2_tr, hparams_k2_tr_path),
        ("K=3 Tabula Rasa", tuned_k3_tr, hparams_k3_tr_path),
    ]:
        if tuned is not None:
            logger.info(
                f"Loaded {label} tuned hparams: "
                f"alpha={tuned['alpha']} n_eff={tuned['prior_n_effective']} "
                f"gamma={tuned['forgetting_factor']} from {path.name}"
            )
        else:
            _missing_hparams.append(f"  {label}: {path}")
    if _missing_hparams:
        raise FileNotFoundError(
            "Appendix H hyperparameter files are required but missing:\n"
            + "\n".join(_missing_hparams)
            + f"\nRun `python {_ablation_script}` first to generate all 4 files."
        )

    results_all["metadata"]["per_expert_hparams"] = {
        "note": (
            "Corralling uses per-expert hyperparameters from Appendix H: "
            "warmup expert uses the BanditGPT-tuned (alpha, n_eff, gamma) "
            "and the tabula-rasa expert uses the independently-tuned "
            "(alpha, gamma).  This avoids giving both experts the same "
            "hyperparameters when their optimal settings differ."
        ),
        "K2_warmup": tuned_k2,
        "K2_tabula_rasa": tuned_k2_tr,
        "K3_warmup": tuned_k3,
        "K3_tabula_rasa": tuned_k3_tr,
    }

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
        DEV_DATA_PATH_ALL_MODELS, K2_MODELS,
    )
    holdout_data_k2 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K2_MODELS,
    )
    logger.info(f"    Dev: {len(dev_data_k2)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k2)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info("  Embedding K=2 prompts ...")
    dev_emb_k2 = embed_dataset_cached(dev_data_k2, embedding_cache, encoder, pca)
    holdout_emb_k2 = embed_dataset_cached(holdout_data_k2, embedding_cache, encoder, pca)
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
    logger.info("    Tuning hyperparameters on dev-val ...")
    supervised_tuning_k2: Dict[str, Dict] = {}
    supervised_k2: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        tuning = tune_supervised_hparams(
            kind, dev_train_k2, dev_train_emb_k2,
            dev_val_k2, dev_val_emb_k2,
            K2_MODELS, costs_k2,
        )
        supervised_tuning_k2[kind] = tuning
        res = run_supervised_baseline(
            kind, K2_MODELS, costs_k2,
            dev_train_k2, dev_train_emb_k2,
            holdout_data_k2, holdout_emb_k2,
            n_trials=N_SEEDS, per_prompt=True,
            hparams=tuning["best_hparams"],
        )
        supervised_k2[kind] = res
        logger.info(
            f"    {kind.upper():<4}: R={res['reward']:.4f} "
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f} "
            f"(tuned: {tuning['best_hparams']})"
        )

    # --- Phase 3: BanditGPT Pareto sweep (train on dev-train) ----------
    logger.info(
        f"\n  Phase 3: BanditGPT Pareto sweep "
        f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ..."
    )
    k2_alpha = tuned_k2["alpha"]
    k2_neff = tuned_k2["prior_n_effective"]
    k2_forgetting = tuned_k2["forgetting_factor"]
    k2_tr_alpha_for_corral = tuned_k2_tr["alpha"]
    k2_tr_ff_for_corral = tuned_k2_tr["forgetting_factor"]
    if not K2_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=2 warmup priors not found: {K2_WARMUP_PRIORS_PATH}\n"
            "Generate with: python scripts/extract_warmup_from_multimodel.py "
            "--input data_collection/warmup_priors/priors_warmup_k10_15comp.joblib "
            f"--output {K2_WARMUP_PRIORS_PATH} "
            "--models meta-llama/llama-3.1-8b-instruct,openai/gpt-4.1"
        )
    k2_warmup_path = str(K2_WARMUP_PRIORS_PATH)
    logger.info(f"  Using warmup priors: {K2_WARMUP_PRIORS_PATH.name}")

    logger.info(
        f"    Corralling per-expert hparams: "
        f"warmup(alpha={k2_alpha}, gamma={k2_forgetting}) | "
        f"tabula_rasa(alpha={k2_tr_alpha_for_corral}, gamma={k2_tr_ff_for_corral})"
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
        tabula_rasa_alpha=k2_tr_alpha_for_corral,
        tabula_rasa_forgetting_factor=k2_tr_ff_for_corral,
    )

    # --- Phase 4: K=2 learning curve -----------------------------------
    # Train on dev-train (same split as Pareto sweep) so the learning
    # curve and Pareto frontier use symmetric data.
    lc_checkpoints = _make_learning_curve_checkpoints(len(dev_train_k2))

    logger.info(f"\n  Phase 4: Learning curve ({N_SEEDS} seeds) ...")
    learning_curve_k2 = run_learning_curve(
        K2_MODELS, K2_CATALOG,
        dev_train_k2, holdout_data_k2, dev_train_emb_k2, holdout_emb_k2,
        k2_warmup_path, costs_k2, N_SEEDS,
        lc_checkpoints, use_corralling=True, cost_penalty=0.0,
        alpha=k2_alpha,
        prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
        tabula_rasa_alpha=k2_tr_alpha_for_corral,
        tabula_rasa_forgetting_factor=k2_tr_ff_for_corral,
    )

    # --- Phase 4b: Supervised learning curve ----------------------------
    best_sv_kind_k2 = max(supervised_k2, key=lambda k: supervised_k2[k]["reward"])
    logger.info(
        f"\n  Phase 4b: Supervised learning curve "
        f"(best={best_sv_kind_k2.upper()}, checkpoints={len(lc_checkpoints)}) ..."
    )
    supervised_lc_k2 = run_supervised_learning_curve(
        best_sv_kind_k2, K2_MODELS, costs_k2,
        dev_train_k2, dev_train_emb_k2,
        holdout_data_k2, holdout_emb_k2,
        lc_checkpoints,
        n_trials=N_SEEDS,
        hparams=supervised_tuning_k2[best_sv_kind_k2]["best_hparams"],
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

    # --- Phase 6: BanditGPT vs LLMRouter supervised baselines -----------
    # Dev-selected Pareto AUC for BanditGPT, plus literature-standard
    # metrics (PerfGain, CostSave, Gap@Oracle) against each supervised
    # baseline.  No coldstart / tabula rasa comparisons.
    logger.info("\n  Phase 6: BanditGPT Pareto AUC & comparison vs supervised baselines ...")

    # BanditGPT dev-selected Pareto AUC
    bg_dev_costs_k2 = [p["dev_mean_cost"] for p in bandit_pareto_k2]
    cost_lo_k2 = min(bg_dev_costs_k2)
    cost_hi_k2 = max(bg_dev_costs_k2)

    bg_ds_auc_k2, bg_hull_c_k2, bg_hull_r_k2, bg_dev_idx_k2 = (
        dev_selected_pareto_auc(bandit_pareto_k2, cost_lo_k2, cost_hi_k2)
    )

    logger.info(
        f"    Dev-selected Pareto AUC (cost [{cost_lo_k2:.6f}, {cost_hi_k2:.6f}]):"
    )
    logger.info(
        f"      BanditGPT: {bg_ds_auc_k2:.4f} "
        f"({len(bg_dev_idx_k2)} dev-optimal pts)"
    )

    # Comparison vs supervised baselines (PerfGain / CostSave / Gap@Oracle)
    oracle_reward_k2 = oracle_r_k2_pure
    comparison_vs_supervised_k2: Dict[str, Dict[str, Any]] = {}
    logger.info("\n    BanditGPT vs LLMRouter supervised baselines (K=2):")
    for kind, sv_result in supervised_k2.items():
        sv_cost = sv_result["cost"]
        sv_reward = sv_result["reward"]

        bg_reward_at_sv_cost = interpolate_pareto_reward(
            bg_hull_c_k2, bg_hull_r_k2, sv_cost,
        )
        pg = (
            perfgain(bg_reward_at_sv_cost, sv_reward)
            if bg_reward_at_sv_cost is not None else None
        )

        bg_cost_at_sv_reward = interpolate_pareto_cost(
            bg_hull_c_k2, bg_hull_r_k2, sv_reward,
        )
        if bg_cost_at_sv_reward is not None:
            cs_abs, cs_pct = costsave(bg_cost_at_sv_reward, sv_cost)
        else:
            cs_abs, cs_pct = None, None

        gap_sv_abs, gap_sv_pct = gap_at_oracle(oracle_reward_k2, sv_reward)
        if bg_reward_at_sv_cost is not None:
            gap_bg_abs, gap_bg_pct = gap_at_oracle(
                oracle_reward_k2, bg_reward_at_sv_cost,
            )
        else:
            gap_bg_abs, gap_bg_pct = None, None

        sv_mc = sv_result.get("model_counts", {})
        sv_total = sum(sv_mc.values()) or 1
        sv_routing_fractions = {m: sv_mc.get(m, 0) / sv_total for m in K2_MODELS}

        comparison_vs_supervised_k2[kind] = {
            "supervised_cost": sv_cost,
            "supervised_reward": sv_reward,
            "supervised_routing_fractions": sv_routing_fractions,
            "banditgpt_reward_at_sv_cost": bg_reward_at_sv_cost,
            "perfgain": pg,
            "banditgpt_cost_at_sv_reward": bg_cost_at_sv_reward,
            "costsave_abs": cs_abs,
            "costsave_pct": cs_pct,
            "gap_oracle_supervised_abs": gap_sv_abs,
            "gap_oracle_supervised_pct": gap_sv_pct,
            "gap_oracle_banditgpt_abs": gap_bg_abs,
            "gap_oracle_banditgpt_pct": gap_bg_pct,
        }
        strong_frac = sv_routing_fractions.get("openai/gpt-4.1", 0.0)
        logger.info(
            f"      vs {kind.upper()}: PerfGain={_fmt(pg)} "
            f"CostSave={_fmt(cs_pct, '%')} "
            f"Gap@Oracle(SV)={gap_sv_abs:.4f}({gap_sv_pct:.1f}%) "
            f"%→GPT-4.1={strong_frac:.1%}"
        )

    # Point comparisons: BanditGPT dev-selected best vs supervised baselines
    bg_dev_optimal_k2 = [bandit_pareto_k2[i] for i in bg_dev_idx_k2]
    bg_best_dev_k2 = max(bg_dev_optimal_k2, key=lambda p: p["dev_mean_reward"])

    supervised_point_tests_k2: Dict[str, Dict] = {}
    logger.info("\n    Paired t-tests (BanditGPT dev-best vs supervised):")
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
            "p_value_raw": float(t_res.pvalue),
            "df": len(bg_ensemble_pp) - 1,
        }

    # Holm-Bonferroni correction across supervised baselines
    test_kinds = list(supervised_point_tests_k2.keys())
    raw_pvals = [supervised_point_tests_k2[k]["p_value_raw"] for k in test_kinds]
    if len(raw_pvals) > 1:
        reject, corrected, _, _ = multipletests(raw_pvals, method="holm")
        for k, p_corr, rej in zip(test_kinds, corrected, reject):
            supervised_point_tests_k2[k]["p_value_holm"] = float(p_corr)
            supervised_point_tests_k2[k]["reject_holm_05"] = bool(rej)
    else:
        for k in test_kinds:
            supervised_point_tests_k2[k]["p_value_holm"] = (
                supervised_point_tests_k2[k]["p_value_raw"]
            )
            supervised_point_tests_k2[k]["reject_holm_05"] = (
                supervised_point_tests_k2[k]["p_value_raw"] < 0.05
            )

    for kind in test_kinds:
        d = supervised_point_tests_k2[kind]
        sig = "**" if d["reject_holm_05"] else ""
        logger.info(
            f"      {kind.upper():<4}: BG={d['banditgpt_reward']:.4f} "
            f"vs {kind.upper()}={d['supervised_reward']:.4f} "
            f"delta={d['delta']:+.4f} "
            f"p_raw={d['p_value_raw']:.4g} p_holm={d['p_value_holm']:.4g}{sig}"
        )

    # BanditGPT dev-optimal Gap@Oracle
    bg_best_dev_reward_k2 = bg_best_dev_k2["mean_reward"]
    bg_gap_oracle_k2_abs, bg_gap_oracle_k2_pct = gap_at_oracle(
        oracle_reward_k2, bg_best_dev_reward_k2,
    )

    # Assemble K=2 summary
    best_sv_kind = max(supervised_k2, key=lambda k: supervised_k2[k]["reward"])
    best_sv = supervised_k2[best_sv_kind]

    logger.info(f"\n  K=2 SUMMARY:")
    logger.info(f"    Oracle (pure quality): {oracle_r_k2_pure:.4f}")
    logger.info(
        f"    Dev-selected Pareto AUC: BanditGPT={bg_ds_auc_k2:.4f}"
    )
    logger.info(
        f"    BanditGPT dev-best: R={bg_best_dev_reward_k2:.4f} "
        f"Gap@Oracle={bg_gap_oracle_k2_abs:.4f} ({bg_gap_oracle_k2_pct:.1f}%)"
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
        "supervised_tuning": {
            k: {"best_hparams": v["best_hparams"], "best_val_reward": v["best_val_reward"]}
            for k, v in supervised_tuning_k2.items()
        },
        "supervised_note": (
            "Supervised baselines are tuned via grid search on dev-val, "
            "mirroring BanditGPT's hyperparameter selection protocol. "
            "KNN is deterministic (std=0 reflects zero initialization "
            "variance, not zero statistical uncertainty). SVM/MLP use "
            "multi-seed trials for initialization variance. All three "
            "mirror the LLMRouter protocol."
        ),
        "supervised_point_tests": supervised_point_tests_k2,
        "banditgpt_pareto": bandit_pareto_k2,
        "learning_curve": learning_curve_k2,
        "supervised_learning_curve": {
            "kind": best_sv_kind_k2,
            "curve": supervised_lc_k2,
        },
        "learning_curve_note": (
            "BanditGPT starts with warmup priors (step 0 reflects prior "
            "quality, not zero knowledge)."
        ),
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k2, cost_hi_k2],
            "banditgpt": bg_ds_auc_k2,
            "banditgpt_hull_costs": bg_hull_c_k2,
            "banditgpt_hull_rewards": bg_hull_r_k2,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward) — no holdout data in selection.  Deployed "
                "points are (holdout_cost, holdout_reward) of dev-optimal "
                "hyperparameters."
            ),
        },
        "comparison_vs_supervised": comparison_vs_supervised_k2,
        "gap_at_oracle_banditgpt": {
            "abs": bg_gap_oracle_k2_abs,
            "pct": bg_gap_oracle_k2_pct,
            "banditgpt_dev_best_reward": bg_best_dev_reward_k2,
        },
    }

    # Lambda=0 entry (quality-maximizing, apples-to-apples with supervised)
    lam0_entries = [p for p in bandit_pareto_k2 if p["lambda"] == 0.0]
    if lam0_entries:
        lam0 = lam0_entries[0]
        results_all["K2"]["banditgpt_lambda0"] = {
            "reward": lam0["mean_reward"],
            "std_reward": lam0["std_reward"],
            "cost": lam0["mean_cost"],
            "std_cost": lam0["std_cost"],
        }

    # ==================================================================
    # K=3 — Multi-model Pareto frontier
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=3: Multi-Model Pareto Frontier")
    logger.info("=" * 70)

    costs_k3 = {m: K3_CATALOG[m]["cost"] for m in K3_MODELS}

    # --- Load K=3 data ------------------------------------------------
    # Use all K3-complete dev prompts, excluding those reserved for
    # prior training (to avoid data leakage from warmup priors).
    logger.info("\n  Loading K=3 data ...")
    prior_train_prompts: set = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(f"    Excluding {len(prior_train_prompts)} prior-train prompts")

    all_dev_k3 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K3_MODELS)
    train_data_k3 = [
        d for d in all_dev_k3 if d["prompt"] not in prior_train_prompts
    ]
    holdout_data_k3 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K3_MODELS,
    )
    logger.info(f"    Dev (all K3-complete): {len(all_dev_k3)} prompts")
    logger.info(f"    Train (excl. prior-train): {len(train_data_k3)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k3)} prompts")

    # --- Embeddings ----------------------------------------------------
    logger.info(f"  Embedding K=3 prompts (PCA={pca.n_components_} comp) ...")
    train_emb_k3 = embed_dataset_cached(train_data_k3, embedding_cache, encoder, pca)
    holdout_emb_k3 = embed_dataset_cached(holdout_data_k3, embedding_cache, encoder, pca)

    # --- Dev train/val split (K=3) ------------------------------------
    logger.info(f"  Splitting K=3 train into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    train_train_k3, train_train_emb_k3, train_val_k3, train_val_emb_k3 = (
        _split_dev_train_val(train_data_k3, train_emb_k3)
    )
    logger.info(
        f"    Train-train: {len(train_train_k3)}  "
        f"Train-val: {len(train_val_k3)}"
    )

    # --- Baselines -----------------------------------------------------
    logger.info("\n  Computing K=3 baselines ...")
    oracle_r_k3, oracle_c_k3 = oracle_route(
        holdout_data_k3, K3_MODELS, costs_k3,
    )
    logger.info(f"    Oracle: R={oracle_r_k3:.4f}  C=${oracle_c_k3:.6f}")

    static_k3: Dict[str, Dict] = {}
    for m in K3_MODELS:
        sr, sc = static_route(holdout_data_k3, m, costs_k3)
        static_k3[m] = {"reward": sr, "cost": sc}
        logger.info(
            f"    Static {K3_CATALOG[m]['display']:<22}: R={sr:.4f}  C=${sc:.6f}"
        )

    random_k3 = random_route(
        holdout_data_k3, K3_MODELS, costs_k3, N_SEEDS * 4,
    )
    logger.info(f"    Random: R={random_k3['reward']:.4f}")

    eg_k3 = best_static_noisy_route(
        train_train_k3, holdout_data_k3, K3_MODELS, costs_k3,
        n_trials=N_SEEDS * 4,
    )
    logger.info(f"    Best-static+noise: R={eg_k3['reward']:.4f}")

    ucb1_k3 = ucb1_online_route(
        train_train_k3, holdout_data_k3, K3_MODELS, costs_k3,
        cost_penalty=0.0, n_trials=N_SEEDS,
    )
    logger.info(
        f"    UCB1 (non-contextual): R={ucb1_k3['reward']:.4f} "
        f"+/-{ucb1_k3['std_reward']:.4f}"
    )

    logger.info("\n  Supervised static baselines (KNN/SVM/MLP) ...")
    logger.info("    Tuning hyperparameters on dev-val ...")
    supervised_tuning_k3: Dict[str, Dict] = {}
    supervised_k3: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        tuning = tune_supervised_hparams(
            kind, train_train_k3, train_train_emb_k3,
            train_val_k3, train_val_emb_k3,
            K3_MODELS, costs_k3,
        )
        supervised_tuning_k3[kind] = tuning
        res = run_supervised_baseline(
            kind, K3_MODELS, costs_k3,
            train_train_k3, train_train_emb_k3,
            holdout_data_k3, holdout_emb_k3,
            n_trials=N_SEEDS, per_prompt=True,
            hparams=tuning["best_hparams"],
        )
        supervised_k3[kind] = res
        logger.info(
            f"    {kind.upper():<4}: R={res['reward']:.4f} "
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f} "
            f"(tuned: {tuning['best_hparams']})"
        )

    # --- BanditGPT Pareto sweep ----------------------------------------
    if not K3_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(
            f"K=3 warmup priors not found: {K3_WARMUP_PRIORS_PATH}\n"
            "Generate with: python scripts/extract_warmup_from_multimodel.py "
            "--input data_collection/warmup_priors/priors_warmup_k10_15comp.joblib "
            f"--output {K3_WARMUP_PRIORS_PATH} "
            "--model-config data_collection/config/models_k3.json"
        )
    k3_warmup_path = str(K3_WARMUP_PRIORS_PATH)
    logger.info(f"  Using warmup priors: {K3_WARMUP_PRIORS_PATH.name}")

    logger.info(
        f"\n  BanditGPT K=3 Pareto sweep "
        f"({len(LAMBDA_VALUES_K3)} lambda x {N_SEEDS} seeds) ..."
    )
    k3_alpha = tuned_k3["alpha"]
    k3_neff = tuned_k3["prior_n_effective"]
    k3_forgetting = tuned_k3["forgetting_factor"]
    k3_tr_alpha_for_corral = tuned_k3_tr["alpha"]
    k3_tr_ff_for_corral = tuned_k3_tr["forgetting_factor"]
    logger.info(
        f"    Corralling per-expert hparams: "
        f"warmup(alpha={k3_alpha}, gamma={k3_forgetting}) | "
        f"tabula_rasa(alpha={k3_tr_alpha_for_corral}, gamma={k3_tr_ff_for_corral})"
    )
    bandit_pareto_k3 = run_pareto_sweep(
        K3_MODELS, K3_CATALOG,
        train_train_k3, holdout_data_k3, train_train_emb_k3, holdout_emb_k3,
        k3_warmup_path, costs_k3, LAMBDA_VALUES_K3,
        N_SEEDS, use_corralling=True, label="banditGPT",
        dev_val_data=train_val_k3, dev_val_emb=train_val_emb_k3,
        alpha=k3_alpha,
        prior_n_effective=k3_neff,
        forgetting_factor=k3_forgetting,
        tabula_rasa_alpha=k3_tr_alpha_for_corral,
        tabula_rasa_forgetting_factor=k3_tr_ff_for_corral,
    )

    # --- K=3 summary: BanditGPT Pareto AUC & comparison vs supervised --
    best_static_m = max(static_k3, key=lambda m: static_k3[m]["reward"])

    bg_dev_costs_k3 = [p["dev_mean_cost"] for p in bandit_pareto_k3]
    cost_lo_k3 = min(bg_dev_costs_k3)
    cost_hi_k3 = max(bg_dev_costs_k3)

    bg_ds_auc_k3, bg_hull_c_k3, bg_hull_r_k3, bg_dev_idx_k3 = (
        dev_selected_pareto_auc(bandit_pareto_k3, cost_lo_k3, cost_hi_k3)
    )

    logger.info(
        f"    Dev-selected Pareto AUC (cost [{cost_lo_k3:.6f}, {cost_hi_k3:.6f}]):"
    )
    logger.info(
        f"      BanditGPT: {bg_ds_auc_k3:.4f} "
        f"({len(bg_dev_idx_k3)} dev-optimal pts)"
    )

    oracle_reward_k3 = oracle_r_k3
    comparison_vs_supervised_k3: Dict[str, Dict[str, Any]] = {}
    logger.info("\n    BanditGPT vs supervised multi-class baselines (reference):")
    for kind, sv_result in supervised_k3.items():
        sv_cost = sv_result["cost"]
        sv_reward = sv_result["reward"]

        bg_reward_at_sv_cost = interpolate_pareto_reward(
            bg_hull_c_k3, bg_hull_r_k3, sv_cost,
        )
        pg = (
            perfgain(bg_reward_at_sv_cost, sv_reward)
            if bg_reward_at_sv_cost is not None else None
        )

        bg_cost_at_sv_reward = interpolate_pareto_cost(
            bg_hull_c_k3, bg_hull_r_k3, sv_reward,
        )
        if bg_cost_at_sv_reward is not None:
            cs_abs, cs_pct = costsave(bg_cost_at_sv_reward, sv_cost)
        else:
            cs_abs, cs_pct = None, None

        gap_sv_abs, gap_sv_pct = gap_at_oracle(oracle_reward_k3, sv_reward)
        if bg_reward_at_sv_cost is not None:
            gap_bg_abs, gap_bg_pct = gap_at_oracle(
                oracle_reward_k3, bg_reward_at_sv_cost,
            )
        else:
            gap_bg_abs, gap_bg_pct = None, None

        sv_mc = sv_result.get("model_counts", {})
        sv_total = sum(sv_mc.values()) or 1
        sv_routing_fractions = {m: sv_mc.get(m, 0) / sv_total for m in K3_MODELS}
        sv_top_model = max(sv_routing_fractions, key=sv_routing_fractions.get)

        comparison_vs_supervised_k3[kind] = {
            "supervised_cost": sv_cost,
            "supervised_reward": sv_reward,
            "supervised_routing_fractions": sv_routing_fractions,
            "banditgpt_reward_at_sv_cost": bg_reward_at_sv_cost,
            "perfgain": pg,
            "banditgpt_cost_at_sv_reward": bg_cost_at_sv_reward,
            "costsave_abs": cs_abs,
            "costsave_pct": cs_pct,
            "gap_oracle_supervised_abs": gap_sv_abs,
            "gap_oracle_supervised_pct": gap_sv_pct,
            "gap_oracle_banditgpt_abs": gap_bg_abs,
            "gap_oracle_banditgpt_pct": gap_bg_pct,
        }
        top_short = sv_top_model.split("/")[-1]
        logger.info(
            f"      vs {kind.upper()}: PerfGain={_fmt(pg)} "
            f"CostSave={_fmt(cs_pct, '%')} "
            f"Gap@Oracle(SV)={gap_sv_abs:.4f}({gap_sv_pct:.1f}%) "
            f"top={top_short} ({sv_routing_fractions[sv_top_model]:.0%})"
        )

    bg_dev_optimal_k3 = [bandit_pareto_k3[i] for i in bg_dev_idx_k3]
    bg_best_dev_k3 = max(bg_dev_optimal_k3, key=lambda p: p["dev_mean_reward"])
    bg_best_dev_reward_k3 = bg_best_dev_k3["mean_reward"]
    bg_gap_oracle_k3_abs, bg_gap_oracle_k3_pct = gap_at_oracle(
        oracle_reward_k3, bg_best_dev_reward_k3,
    )

    logger.info(f"\n  K=3 SUMMARY:")
    logger.info(f"    Oracle:       {oracle_r_k3:.4f}")
    logger.info(
        f"    Dev-selected Pareto AUC: BanditGPT={bg_ds_auc_k3:.4f}"
    )
    logger.info(
        f"    BanditGPT dev-best: R={bg_best_dev_reward_k3:.4f} "
        f"Gap@Oracle={bg_gap_oracle_k3_abs:.4f} ({bg_gap_oracle_k3_pct:.1f}%)"
    )
    logger.info(
        f"    Best static:  {static_k3[best_static_m]['reward']:.4f} "
        f"({K3_CATALOG[best_static_m]['display']})"
    )
    logger.info(f"    Best-static+noise: {eg_k3['reward']:.4f}")
    logger.info(f"    UCB1 (non-ctx):    {ucb1_k3['reward']:.4f}")
    for kind in ("knn", "svm", "mlp"):
        s = supervised_k3[kind]
        logger.info(
            f"    {kind.upper():<4} (supervised): {s['reward']:.4f} "
            f"+/-{s['std_reward']:.4f}"
        )
    logger.info(f"    Random:            {random_k3['reward']:.4f}")

    results_all["K3"] = {
        "models": [{"id": m, **K3_CATALOG[m]} for m in K3_MODELS],
        "n_train": len(train_data_k3),
        "n_holdout": len(holdout_data_k3),
        "oracle": {"reward": oracle_r_k3, "cost": oracle_c_k3},
        "static": {m: static_k3[m] for m in K3_MODELS},
        "best_static": {
            "model": best_static_m,
            "reward": static_k3[best_static_m]["reward"],
            "cost": static_k3[best_static_m]["cost"],
        },
        "random": random_k3,
        "best_static_noisy": eg_k3,
        "ucb1": ucb1_k3,
        "supervised": supervised_k3,
        "supervised_tuning": {
            k: {"best_hparams": v["best_hparams"], "best_val_reward": v["best_val_reward"]}
            for k, v in supervised_tuning_k3.items()
        },
        "supervised_note": (
            "K=3 supervised baselines are a multi-class extension of the "
            "LLMRouter binary classifiers (KNN/SVM/MLP) and are presented "
            "as reference points, not a direct comparison to the published "
            "LLMRouter system (which supports K=2 only). Tuned via grid "
            "search on dev-val, mirroring BanditGPT's hparam protocol."
        ),
        "banditgpt_pareto": bandit_pareto_k3,
        "pareto_auc_dev_selected": {
            "cost_range": [cost_lo_k3, cost_hi_k3],
            "banditgpt": bg_ds_auc_k3,
            "banditgpt_hull_costs": bg_hull_c_k3,
            "banditgpt_hull_rewards": bg_hull_r_k3,
            "note": (
                "Dev-selected Pareto AUC: hull built from (dev_cost, "
                "dev_reward).  Deployed = holdout performance of dev-optimal "
                "hyperparameters."
            ),
        },
        "comparison_vs_supervised": comparison_vs_supervised_k3,
        "gap_at_oracle_banditgpt": {
            "abs": bg_gap_oracle_k3_abs,
            "pct": bg_gap_oracle_k3_pct,
            "banditgpt_dev_best_reward": bg_best_dev_reward_k3,
        },
        "n_trials": N_SEEDS,
    }

    lam0_entries_k3 = [p for p in bandit_pareto_k3 if p["lambda"] == 0.0]
    if lam0_entries_k3:
        lam0_k3 = lam0_entries_k3[0]
        results_all["K3"]["banditgpt_lambda0"] = {
            "reward": lam0_k3["mean_reward"],
            "std_reward": lam0_k3["std_reward"],
            "cost": lam0_k3["mean_cost"],
            "std_cost": lam0_k3["std_cost"],
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
