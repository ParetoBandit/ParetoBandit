#!/usr/bin/env python3
"""
BanditGPT v1 vs v2: Head-to-Head Router Comparison.

Extends the ``run_prequential.py`` protocol to run **both** router
implementations (``bandit_gpt.router`` and ``bandit_gpt.router_v2``)
under identical conditions and compare their performance.

Protocol
--------
Same as ``run_prequential.py`` (train-then-freeze, symmetric data access,
dev-selected Pareto frontier), but each BanditGPT evaluation phase is
run twice — once with v1 and once with v2 — so the comparison is
apples-to-apples.

Outputs (``results/``)
    prequential_results_v2.json   — Full results including both routers.
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
    K10_WARMUP_PRIORS_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
    K10_MODELS_PATH,
)
from utils.rewards import extract_reward
from utils.model_pricing import get_prices_for_models, load_model_catalog, req_cost
from utils.router_factory import (
    create_experiment_router,
    create_experiment_router_v2,
)
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

K10_MODELS, K10_CATALOG = load_model_catalog(K10_MODELS_PATH)


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
    0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    1.0, 1.5, 2.0, 3.0, 5.0,
]
LAMBDA_VALUES_K10: List[float] = [
    0.0, 0.01, 0.03, 0.05, 0.07, 0.08, 0.09, 0.095,
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
    0.185, 0.19, 0.192, 0.195, 0.198, 0.2, 0.202, 0.205,
    0.208, 0.21, 0.215, 0.22, 0.25, 0.3, 0.5, 1.0,
]

def _make_learning_curve_checkpoints(n_train: int) -> List[int]:
    """Build learning-curve checkpoint list adapted to the training set size."""
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
    """Deterministically split (data, emb) into train and val portions."""
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
    """Load rewards for specific models from gzipped JSONL."""
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
    """Build the registry dict that router factory functions expect."""
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
    """Per-prompt clairvoyant argmax of reward - lambda * cost."""
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


def ucb1_online_route(
    train_data: List[Dict],
    eval_data: List[Dict],
    models: List[str],
    costs: Dict[str, float],
    cost_penalty: float = 0.0,
    n_trials: int = 20,
    seed_offset: int = SEED_OFFSET,
) -> Dict[str, float]:
    """Non-contextual UCB1 train-then-freeze baseline."""
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
    """Return the theoretical reward bounds for normalisation."""
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
    """Train a BanditRouter on the dev set via route() + process_feedback()."""
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
    """Train a BanditRouter with early stopping on validation reward."""
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
    """Evaluate a frozen router on the holdout set (no learning)."""
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


# ============================================================================
# Generic Pareto sweep (works with both v1 and v2 via factory callable)
# ============================================================================


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
    router_factory_fn=create_experiment_router,
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

    Accepts ``router_factory_fn`` so the same sweep logic can run
    with either ``create_experiment_router`` (v1) or
    ``create_experiment_router_v2`` (v2).

    Returns:
        List of dicts, one per lambda, with mean/std holdout reward
        and cost, plus dev metrics for Pareto frontier selection.
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
        for trial in range(n_trials):
            seed = SEED_OFFSET + trial
            np.random.seed(seed)
            trial_rng = np.random.default_rng(seed)
            router = router_factory_fn(
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

        mean_best_step = float(np.mean(trial_best_steps))
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
        })
        logger.info(
            f"    lambda={lam:<6} | R={np.mean(trial_r):.4f}+/-{np.std(trial_r):.4f} "
            f"| C=${np.mean(trial_c):.6f} | stop@{mean_best_step:.0f}"
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
    router_factory_fn=create_experiment_router,
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

    Accepts ``router_factory_fn`` for v1/v2 comparison.
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
        np.random.seed(seed)
        trial_rng = np.random.default_rng(seed)
        router = router_factory_fn(
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


def across_seeds_ttest(
    a_per_seed: List[float],
    b_per_seed: Optional[List[float]] = None,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Paired or one-sample t-test across N_SEEDS runs."""
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
# v1 vs v2 head-to-head analysis
# ============================================================================


def _compare_pareto_sweeps(
    v1_pareto: List[Dict],
    v2_pareto: List[Dict],
    oracle_reward: float,
    label_prefix: str,
) -> Dict[str, Any]:
    """Head-to-head statistical comparison between v1 and v2 Pareto sweeps.

    For each shared lambda value, computes paired t-tests across seeds
    and reports aggregate statistics.

    Args:
        v1_pareto: Pareto sweep results from router v1.
        v2_pareto: Pareto sweep results from router v2.
        oracle_reward: Instance-wise optimal reward.
        label_prefix: K-condition label (e.g., "K2" or "K10").

    Returns:
        Dict with per-lambda comparisons and aggregate summary.
    """
    v1_by_lam = {p["lambda"]: p for p in v1_pareto}
    v2_by_lam = {p["lambda"]: p for p in v2_pareto}
    shared_lambdas = sorted(set(v1_by_lam) & set(v2_by_lam))

    per_lambda: List[Dict] = []
    v1_wins = v2_wins = ties = 0
    reward_deltas: List[float] = []

    for lam in shared_lambdas:
        v1 = v1_by_lam[lam]
        v2 = v2_by_lam[lam]

        v1_seeds = np.array(v1["per_seed_rewards"])
        v2_seeds = np.array(v2["per_seed_rewards"])
        diff = v2_seeds - v1_seeds
        mean_diff = float(diff.mean())
        reward_deltas.append(mean_diff)

        t_result = scipy_stats.ttest_rel(v2_seeds, v1_seeds)
        significant = float(t_result.pvalue) < 0.05

        if significant:
            if mean_diff > 0:
                v2_wins += 1
            else:
                v1_wins += 1
        else:
            ties += 1

        per_lambda.append({
            "lambda": lam,
            "v1_reward": v1["mean_reward"],
            "v1_std": v1["std_reward"],
            "v1_cost": v1["mean_cost"],
            "v2_reward": v2["mean_reward"],
            "v2_std": v2["std_reward"],
            "v2_cost": v2["mean_cost"],
            "reward_delta_v2_minus_v1": mean_diff,
            "t_stat": float(t_result.statistic),
            "p_value": float(t_result.pvalue),
            "significant_05": significant,
            "v1_mean_stop": v1.get("mean_best_step"),
            "v2_mean_stop": v2.get("mean_best_step"),
        })

    # Aggregate: signed-rank test on reward deltas across all lambdas
    deltas_arr = np.array(reward_deltas)
    if len(deltas_arr) > 1:
        wilcoxon_res = scipy_stats.wilcoxon(deltas_arr, alternative="two-sided")
        aggregate_test = {
            "test": "wilcoxon_signed_rank",
            "statistic": float(wilcoxon_res.statistic),
            "p_value": float(wilcoxon_res.pvalue),
            "mean_delta": float(deltas_arr.mean()),
            "median_delta": float(np.median(deltas_arr)),
        }
    else:
        aggregate_test = {"note": "Insufficient shared lambdas for aggregate test"}

    return {
        "per_lambda": per_lambda,
        "summary": {
            "n_shared_lambdas": len(shared_lambdas),
            "v1_sig_wins": v1_wins,
            "v2_sig_wins": v2_wins,
            "ties": ties,
            "mean_reward_delta": float(deltas_arr.mean()) if len(deltas_arr) > 0 else 0.0,
        },
        "aggregate_test": aggregate_test,
    }


def _compare_learning_curves(
    v1_curve: List[Dict],
    v2_curve: List[Dict],
) -> Dict[str, Any]:
    """Compare v1 and v2 learning curves at matched checkpoints."""
    v1_by_step = {p["step"]: p for p in v1_curve}
    v2_by_step = {p["step"]: p for p in v2_curve}
    shared_steps = sorted(set(v1_by_step) & set(v2_by_step))

    comparisons = []
    for step in shared_steps:
        v1 = v1_by_step[step]
        v2 = v2_by_step[step]
        delta = v2["mean_reward"] - v1["mean_reward"]
        comparisons.append({
            "step": step,
            "v1_reward": v1["mean_reward"],
            "v1_std": v1["std_reward"],
            "v2_reward": v2["mean_reward"],
            "v2_std": v2["std_reward"],
            "delta_v2_minus_v1": delta,
        })

    final_v1 = v1_curve[-1] if v1_curve else None
    final_v2 = v2_curve[-1] if v2_curve else None
    final_delta = (
        (final_v2["mean_reward"] - final_v1["mean_reward"])
        if final_v1 and final_v2 else None
    )

    return {
        "per_step": comparisons,
        "final_delta_v2_minus_v1": final_delta,
    }


# ============================================================================
# Main experiment
# ============================================================================


def run_experiment() -> None:  # noqa: C901
    """Run v1 vs v2 comparison for K=2 and K=10."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 0 — shared resources
    # ------------------------------------------------------------------
    logger.info("Loading encoder, PCA, and embedding cache ...")
    pca = joblib.load(DEFAULT_PCA_PATH)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    logger.info(f"  PCA: {pca.n_components_} components (unified for K=2 and K=10)")

    embedding_cache = load_embedding_cache(
        expected_encoder=DEFAULT_SENTENCE_TRANSFORMER,
        expected_pca_components=pca.n_components_,
    )

    results_all: Dict[str, Any] = {
        "metadata": {
            "experiment": "router_v1_vs_v2_comparison",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_seeds": N_SEEDS,
            "protocol": "train_then_freeze",
            "description": (
                "Head-to-head comparison of bandit_gpt.router (v1) vs "
                "bandit_gpt.router_v2 (v2) under identical experimental "
                "conditions.  Both routers receive the same data, "
                "hyperparameters, warmup priors, and evaluation protocol."
            ),
        },
    }

    # ------------------------------------------------------------------
    # Phase 0 — hyperparameters from Appendix H
    # ------------------------------------------------------------------
    hparams_dir = (
        Path(__file__).resolve().parent.parent / "appendix"
        / "H_alpha_neff_ablation" / "results"
    )
    hparams_k2_path = hparams_dir / "best_hparams_k2.json"
    hparams_k10_path = hparams_dir / "best_hparams_k10.json"
    hparams_k2_tr_path = hparams_dir / "best_hparams_k2_tabula_rasa.json"
    hparams_k10_tr_path = hparams_dir / "best_hparams_k10_tabula_rasa.json"

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
    tuned_k2_tr = _load_hparams(hparams_k2_tr_path, "K2")
    tuned_k10_tr = _load_hparams(hparams_k10_tr_path, "K10")

    _ablation_script = (
        "experiments/appendix/H_alpha_neff_ablation/run_3d_grid_ablation.py"
    )
    _missing_hparams: List[str] = []
    for lbl, tuned, path in [
        ("K=2 BanditGPT", tuned_k2, hparams_k2_path),
        ("K=10 BanditGPT", tuned_k10, hparams_k10_path),
        ("K=2 Tabula Rasa", tuned_k2_tr, hparams_k2_tr_path),
        ("K=10 Tabula Rasa", tuned_k10_tr, hparams_k10_tr_path),
    ]:
        if tuned is not None:
            logger.info(
                f"Loaded {lbl} tuned hparams: "
                f"alpha={tuned['alpha']} n_eff={tuned['prior_n_effective']} "
                f"gamma={tuned['forgetting_factor']} from {path.name}"
            )
        else:
            _missing_hparams.append(f"  {lbl}: {path}")
    if _missing_hparams:
        raise FileNotFoundError(
            "Appendix H hyperparameter files are required but missing:\n"
            + "\n".join(_missing_hparams)
            + f"\nRun `python {_ablation_script}` first to generate all 4 files."
        )

    results_all["metadata"]["per_expert_hparams"] = {
        "K2_warmup": tuned_k2,
        "K2_tabula_rasa": tuned_k2_tr,
        "K10_warmup": tuned_k10,
        "K10_tabula_rasa": tuned_k10_tr,
    }

    # ==================================================================
    # K=2
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=2: Router v1 vs v2 Head-to-Head")
    logger.info("=" * 70)

    costs_k2 = {m: K2_CATALOG[m]["cost"] for m in K2_MODELS}

    logger.info("\n  Loading K=2 dev and holdout data ...")
    dev_data_k2 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K2_MODELS)
    holdout_data_k2 = load_rewards_from_file(HOLDOUT_DATA_PATH_ALL_MODELS, K2_MODELS)
    logger.info(f"    Dev: {len(dev_data_k2)} prompts")
    logger.info(f"    Holdout: {len(holdout_data_k2)} prompts")

    logger.info("  Embedding K=2 prompts ...")
    dev_emb_k2 = embed_dataset_cached(dev_data_k2, embedding_cache, encoder, pca)
    holdout_emb_k2 = embed_dataset_cached(holdout_data_k2, embedding_cache, encoder, pca)
    dim = dev_emb_k2[0].shape[0]

    logger.info(f"  Splitting dev into train/val "
                f"({1 - DEV_VAL_FRACTION:.0%}/{DEV_VAL_FRACTION:.0%}) ...")
    dev_train_k2, dev_train_emb_k2, dev_val_k2, dev_val_emb_k2 = (
        _split_dev_train_val(dev_data_k2, dev_emb_k2)
    )
    logger.info(f"    Dev-train: {len(dev_train_k2)}  Dev-val: {len(dev_val_k2)}")

    # --- Oracle and baselines (shared between v1 and v2) ----------------
    oracle_r_k2, oracle_c_k2 = oracle_route(holdout_data_k2, K2_MODELS, costs_k2)
    logger.info(f"    Oracle: R={oracle_r_k2:.4f}  C=${oracle_c_k2:.6f}")

    # --- Supervised baselines (run once — same for both) ----------------
    logger.info("\n  Supervised baselines (shared between v1 and v2) ...")
    supervised_k2: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        tuning = tune_supervised_hparams(
            kind, dev_train_k2, dev_train_emb_k2,
            dev_val_k2, dev_val_emb_k2,
            K2_MODELS, costs_k2,
        )
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
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f}"
        )

    # --- K=2 shared hparams ---
    k2_alpha = tuned_k2["alpha"]
    k2_neff = tuned_k2["prior_n_effective"]
    k2_forgetting = tuned_k2["forgetting_factor"]
    k2_tr_alpha = tuned_k2_tr["alpha"]
    k2_tr_ff = tuned_k2_tr["forgetting_factor"]
    if not K2_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(f"K=2 warmup priors not found: {K2_WARMUP_PRIORS_PATH}")
    k2_warmup_path = str(K2_WARMUP_PRIORS_PATH)

    common_sweep_kwargs_k2 = dict(
        models=K2_MODELS, catalog=K2_CATALOG,
        train_data=dev_train_k2, eval_data=holdout_data_k2,
        train_emb=dev_train_emb_k2, eval_emb=holdout_emb_k2,
        warmup_path=k2_warmup_path, costs=costs_k2,
        lambda_values=LAMBDA_VALUES_K2, n_trials=N_SEEDS,
        use_corralling=True,
        dev_val_data=dev_val_k2, dev_val_emb=dev_val_emb_k2,
        alpha=k2_alpha, prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
        tabula_rasa_alpha=k2_tr_alpha,
        tabula_rasa_forgetting_factor=k2_tr_ff,
    )

    # --- V1 Pareto sweep ------------------------------------------------
    logger.info(f"\n  [V1] BanditGPT Pareto sweep "
                f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ...")
    bandit_pareto_k2_v1 = run_pareto_sweep(
        **common_sweep_kwargs_k2,
        router_factory_fn=create_experiment_router,
        label="banditGPT_v1",
    )

    # --- V2 Pareto sweep ------------------------------------------------
    logger.info(f"\n  [V2] BanditGPT Pareto sweep "
                f"({len(LAMBDA_VALUES_K2)} lambda x {N_SEEDS} seeds) ...")
    bandit_pareto_k2_v2 = run_pareto_sweep(
        **common_sweep_kwargs_k2,
        router_factory_fn=create_experiment_router_v2,
        label="banditGPT_v2",
    )

    # --- K=2 learning curves -------------------------------------------
    lc_checkpoints_k2 = _make_learning_curve_checkpoints(len(dev_train_k2))
    common_lc_kwargs_k2 = dict(
        models=K2_MODELS, catalog=K2_CATALOG,
        train_data=dev_train_k2, eval_data=holdout_data_k2,
        train_emb=dev_train_emb_k2, eval_emb=holdout_emb_k2,
        warmup_path=k2_warmup_path, costs=costs_k2,
        n_trials=N_SEEDS, checkpoints=lc_checkpoints_k2,
        use_corralling=True, cost_penalty=0.0,
        alpha=k2_alpha, prior_n_effective=k2_neff,
        forgetting_factor=k2_forgetting,
        tabula_rasa_alpha=k2_tr_alpha,
        tabula_rasa_forgetting_factor=k2_tr_ff,
    )

    logger.info(f"\n  [V1] Learning curve ({N_SEEDS} seeds) ...")
    lc_k2_v1 = run_learning_curve(
        **common_lc_kwargs_k2,
        router_factory_fn=create_experiment_router,
        label="banditGPT_v1",
    )

    logger.info(f"\n  [V2] Learning curve ({N_SEEDS} seeds) ...")
    lc_k2_v2 = run_learning_curve(
        **common_lc_kwargs_k2,
        router_factory_fn=create_experiment_router_v2,
        label="banditGPT_v2",
    )

    # --- K=2 head-to-head analysis -------------------------------------
    logger.info("\n  K=2 Head-to-Head Analysis ...")
    k2_comparison = _compare_pareto_sweeps(
        bandit_pareto_k2_v1, bandit_pareto_k2_v2,
        oracle_r_k2, "K2",
    )
    k2_lc_comparison = _compare_learning_curves(lc_k2_v1, lc_k2_v2)

    logger.info(
        f"    Pareto: v1 sig-wins={k2_comparison['summary']['v1_sig_wins']} "
        f"v2 sig-wins={k2_comparison['summary']['v2_sig_wins']} "
        f"ties={k2_comparison['summary']['ties']} "
        f"mean-delta={k2_comparison['summary']['mean_reward_delta']:+.4f}"
    )
    if k2_lc_comparison["final_delta_v2_minus_v1"] is not None:
        logger.info(
            f"    Learning curve final delta (v2-v1): "
            f"{k2_lc_comparison['final_delta_v2_minus_v1']:+.4f}"
        )

    # --- K=2 Pareto AUC for both versions ------------------------------
    bg_dev_costs_k2_v1 = [p["dev_mean_cost"] for p in bandit_pareto_k2_v1]
    bg_dev_costs_k2_v2 = [p["dev_mean_cost"] for p in bandit_pareto_k2_v2]
    cost_lo_k2 = min(min(bg_dev_costs_k2_v1), min(bg_dev_costs_k2_v2))
    cost_hi_k2 = max(max(bg_dev_costs_k2_v1), max(bg_dev_costs_k2_v2))

    v1_auc_k2, v1_hull_c_k2, v1_hull_r_k2, _ = (
        dev_selected_pareto_auc(bandit_pareto_k2_v1, cost_lo_k2, cost_hi_k2)
    )
    v2_auc_k2, v2_hull_c_k2, v2_hull_r_k2, _ = (
        dev_selected_pareto_auc(bandit_pareto_k2_v2, cost_lo_k2, cost_hi_k2)
    )
    logger.info(
        f"    Pareto AUC: v1={v1_auc_k2:.4f}  v2={v2_auc_k2:.4f}  "
        f"delta={v2_auc_k2 - v1_auc_k2:+.4f}"
    )

    results_all["K2"] = {
        "models": K2_MODELS,
        "n_dev": len(dev_data_k2),
        "n_holdout": len(holdout_data_k2),
        "oracle": {"reward": oracle_r_k2, "cost": oracle_c_k2},
        "supervised": supervised_k2,
        "v1_pareto": bandit_pareto_k2_v1,
        "v2_pareto": bandit_pareto_k2_v2,
        "v1_learning_curve": lc_k2_v1,
        "v2_learning_curve": lc_k2_v2,
        "pareto_auc": {
            "cost_range": [cost_lo_k2, cost_hi_k2],
            "v1": v1_auc_k2,
            "v2": v2_auc_k2,
            "delta_v2_minus_v1": v2_auc_k2 - v1_auc_k2,
        },
        "head_to_head_pareto": k2_comparison,
        "head_to_head_learning_curve": k2_lc_comparison,
    }

    # Lambda=0 for both
    for tag, pareto in [("v1", bandit_pareto_k2_v1), ("v2", bandit_pareto_k2_v2)]:
        lam0 = [p for p in pareto if p["lambda"] == 0.0]
        if lam0:
            results_all["K2"][f"{tag}_lambda0"] = {
                "reward": lam0[0]["mean_reward"],
                "std_reward": lam0[0]["std_reward"],
                "cost": lam0[0]["mean_cost"],
                "std_cost": lam0[0]["std_cost"],
            }

    # ==================================================================
    # K=10
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("K=10: Router v1 vs v2 Head-to-Head")
    logger.info("=" * 70)

    costs_k10 = {m: K10_CATALOG[m]["cost"] for m in K10_MODELS}

    logger.info("\n  Loading K=10 data ...")
    prior_train_prompts: set = set()
    if THREE_WAY_SPLITS_PATH.exists():
        with open(THREE_WAY_SPLITS_PATH) as f:
            splits_3way = json.load(f)
        prior_train_prompts = set(splits_3way.get("prior_train_pool", []))
        logger.info(f"    Excluding {len(prior_train_prompts)} prior-train prompts")

    all_dev_k10 = load_rewards_from_file(DEV_DATA_PATH_ALL_MODELS, K10_MODELS)
    train_data_k10 = [
        d for d in all_dev_k10 if d["prompt"] not in prior_train_prompts
    ]
    holdout_data_k10 = load_rewards_from_file(
        HOLDOUT_DATA_PATH_ALL_MODELS, K10_MODELS,
    )
    logger.info(f"    Train: {len(train_data_k10)} prompts  Holdout: {len(holdout_data_k10)} prompts")

    logger.info(f"  Embedding K=10 prompts (PCA={pca.n_components_} comp) ...")
    train_emb_k10 = embed_dataset_cached(train_data_k10, embedding_cache, encoder, pca)
    holdout_emb_k10 = embed_dataset_cached(holdout_data_k10, embedding_cache, encoder, pca)

    logger.info(f"  Splitting K=10 train into train/val ...")
    train_train_k10, train_train_emb_k10, train_val_k10, train_val_emb_k10 = (
        _split_dev_train_val(train_data_k10, train_emb_k10)
    )
    logger.info(
        f"    Train-train: {len(train_train_k10)}  "
        f"Train-val: {len(train_val_k10)}"
    )

    oracle_r_k10, oracle_c_k10 = oracle_route(holdout_data_k10, K10_MODELS, costs_k10)
    logger.info(f"    Oracle: R={oracle_r_k10:.4f}  C=${oracle_c_k10:.6f}")

    # Supervised baselines (shared between v1 and v2)
    logger.info("\n  Supervised baselines ...")
    supervised_k10: Dict[str, Dict] = {}
    for kind in ("knn", "svm", "mlp"):
        tuning = tune_supervised_hparams(
            kind, train_train_k10, train_train_emb_k10,
            train_val_k10, train_val_emb_k10,
            K10_MODELS, costs_k10,
        )
        res = run_supervised_baseline(
            kind, K10_MODELS, costs_k10,
            train_train_k10, train_train_emb_k10,
            holdout_data_k10, holdout_emb_k10,
            n_trials=N_SEEDS, per_prompt=True,
            hparams=tuning["best_hparams"],
        )
        supervised_k10[kind] = res
        logger.info(
            f"    {kind.upper():<4}: R={res['reward']:.4f} "
            f"+/-{res['std_reward']:.4f}  C=${res['cost']:.6f}"
        )

    # --- K=10 hparams ---
    if not K10_WARMUP_PRIORS_PATH.exists():
        raise FileNotFoundError(f"K=10 warmup priors not found: {K10_WARMUP_PRIORS_PATH}")
    k10_warmup_path = str(K10_WARMUP_PRIORS_PATH)
    k10_alpha = tuned_k10["alpha"]
    k10_neff = tuned_k10["prior_n_effective"]
    k10_forgetting = tuned_k10["forgetting_factor"]
    k10_tr_alpha = tuned_k10_tr["alpha"]
    k10_tr_ff = tuned_k10_tr["forgetting_factor"]

    common_sweep_kwargs_k10 = dict(
        models=K10_MODELS, catalog=K10_CATALOG,
        train_data=train_train_k10, eval_data=holdout_data_k10,
        train_emb=train_train_emb_k10, eval_emb=holdout_emb_k10,
        warmup_path=k10_warmup_path, costs=costs_k10,
        lambda_values=LAMBDA_VALUES_K10, n_trials=N_SEEDS,
        use_corralling=True,
        dev_val_data=train_val_k10, dev_val_emb=train_val_emb_k10,
        alpha=k10_alpha, prior_n_effective=k10_neff,
        forgetting_factor=k10_forgetting,
        tabula_rasa_alpha=k10_tr_alpha,
        tabula_rasa_forgetting_factor=k10_tr_ff,
    )

    logger.info(f"\n  [V1] K=10 Pareto sweep "
                f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ...")
    bandit_pareto_k10_v1 = run_pareto_sweep(
        **common_sweep_kwargs_k10,
        router_factory_fn=create_experiment_router,
        label="banditGPT_v1",
    )

    logger.info(f"\n  [V2] K=10 Pareto sweep "
                f"({len(LAMBDA_VALUES_K10)} lambda x {N_SEEDS} seeds) ...")
    bandit_pareto_k10_v2 = run_pareto_sweep(
        **common_sweep_kwargs_k10,
        router_factory_fn=create_experiment_router_v2,
        label="banditGPT_v2",
    )

    # --- K=10 head-to-head analysis ------------------------------------
    logger.info("\n  K=10 Head-to-Head Analysis ...")
    k10_comparison = _compare_pareto_sweeps(
        bandit_pareto_k10_v1, bandit_pareto_k10_v2,
        oracle_r_k10, "K10",
    )

    logger.info(
        f"    Pareto: v1 sig-wins={k10_comparison['summary']['v1_sig_wins']} "
        f"v2 sig-wins={k10_comparison['summary']['v2_sig_wins']} "
        f"ties={k10_comparison['summary']['ties']} "
        f"mean-delta={k10_comparison['summary']['mean_reward_delta']:+.4f}"
    )

    # K=10 Pareto AUC
    bg_dev_costs_k10_v1 = [p["dev_mean_cost"] for p in bandit_pareto_k10_v1]
    bg_dev_costs_k10_v2 = [p["dev_mean_cost"] for p in bandit_pareto_k10_v2]
    cost_lo_k10 = min(min(bg_dev_costs_k10_v1), min(bg_dev_costs_k10_v2))
    cost_hi_k10 = max(max(bg_dev_costs_k10_v1), max(bg_dev_costs_k10_v2))

    v1_auc_k10, _, _, _ = dev_selected_pareto_auc(
        bandit_pareto_k10_v1, cost_lo_k10, cost_hi_k10,
    )
    v2_auc_k10, _, _, _ = dev_selected_pareto_auc(
        bandit_pareto_k10_v2, cost_lo_k10, cost_hi_k10,
    )
    logger.info(
        f"    Pareto AUC: v1={v1_auc_k10:.4f}  v2={v2_auc_k10:.4f}  "
        f"delta={v2_auc_k10 - v1_auc_k10:+.4f}"
    )

    results_all["K10"] = {
        "models": [{"id": m, **K10_CATALOG[m]} for m in K10_MODELS],
        "n_train": len(train_data_k10),
        "n_holdout": len(holdout_data_k10),
        "oracle": {"reward": oracle_r_k10, "cost": oracle_c_k10},
        "supervised": supervised_k10,
        "v1_pareto": bandit_pareto_k10_v1,
        "v2_pareto": bandit_pareto_k10_v2,
        "pareto_auc": {
            "cost_range": [cost_lo_k10, cost_hi_k10],
            "v1": v1_auc_k10,
            "v2": v2_auc_k10,
            "delta_v2_minus_v1": v2_auc_k10 - v1_auc_k10,
        },
        "head_to_head_pareto": k10_comparison,
    }

    for tag, pareto in [("v1", bandit_pareto_k10_v1), ("v2", bandit_pareto_k10_v2)]:
        lam0 = [p for p in pareto if p["lambda"] == 0.0]
        if lam0:
            results_all["K10"][f"{tag}_lambda0"] = {
                "reward": lam0[0]["mean_reward"],
                "std_reward": lam0[0]["std_reward"],
                "cost": lam0[0]["mean_cost"],
                "std_cost": lam0[0]["std_cost"],
            }

    # ==================================================================
    # Final summary
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL v1 vs v2 SUMMARY")
    logger.info("=" * 70)

    for k_label, k_key in [("K=2", "K2"), ("K=10", "K10")]:
        auc = results_all[k_key]["pareto_auc"]
        h2h = results_all[k_key]["head_to_head_pareto"]["summary"]
        logger.info(
            f"  {k_label}: "
            f"AUC v1={auc['v1']:.4f} v2={auc['v2']:.4f} "
            f"(delta={auc['delta_v2_minus_v1']:+.4f}) | "
            f"v1-wins={h2h['v1_sig_wins']} v2-wins={h2h['v2_sig_wins']} "
            f"ties={h2h['ties']}"
        )

    # ==================================================================
    # Serialise
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

    out_path = output_dir / "prequential_results_v2.json"
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
