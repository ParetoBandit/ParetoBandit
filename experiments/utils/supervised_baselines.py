"""
Supervised static routing baselines (KNN, SVM, MLP).

Implements the three core supervised baselines from the LLMRouter framework
(UIUC, Dec 2025) using the same underlying algorithms but integrated with
BanditGPT's data pipeline and evaluation protocol for KDD-fair comparison.

All baselines follow the same pattern:
  1. Tune hyperparameters via grid search on dev-val (``tune_supervised_hparams``),
     mirroring BanditGPT's dev-val selection protocol.
  2. Train on dev-train: label = argmax_m reward[m] per prompt (same objective
     as BanditGPT's ``extract_reward``).
  3. Freeze: no online adaptation.
  4. Evaluate on holdout: predict best model for each prompt, look up reward.

This isolates the exact claim BanditGPT makes: online adaptation > supervised
static routing, holding features, objective, and data constant.

Reference:
    LLMRouter (github.com/ulab-uiuc/LLMRouter) — KNNRouter, SVMRouter, MLPRouter.
    LLMRouterBench (arXiv:2601.07206).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)

HPARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "knn": {"n_neighbors": [3, 5, 11, 21]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]},
    "mlp": {"hidden_layer_sizes": [(64,), (128, 64), (256, 128)]},
}


def _prepare_labels(
    data: List[Dict],
    models: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Derive best-model classification labels from the reward matrix.

    Args:
        data: List of dicts with ``rewards`` mapping model_id -> float.
        models: Ordered list of candidate model IDs.

    Returns:
        Tuple of (labels, label_names) where labels[i] is the index
        into ``models`` of the best model for prompt i.
    """
    labels = np.empty(len(data), dtype=int)
    for i, d in enumerate(data):
        best_idx = 0
        best_r = -np.inf
        for j, m in enumerate(models):
            r = d["rewards"].get(m, -np.inf)
            if r > best_r:
                best_r = r
                best_idx = j
        labels[i] = best_idx
    return labels, models


def _embeddings_to_matrix(emb: List[np.ndarray]) -> np.ndarray:
    """Stack a list of 1-D embedding vectors into a (N, D) matrix."""
    return np.vstack(emb)


def train_supervised_router(
    kind: str,
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    models: List[str],
    *,
    seed: int = 42,
    hparams: Optional[Dict[str, Any]] = None,
) -> Any:
    """Train a supervised static router.

    Args:
        kind: One of ``"knn"``, ``"svm"``, ``"mlp"``.
        train_data: Training prompts with reward dicts.
        train_embeddings: Pre-computed PCA feature vectors (aligned with train_data).
        models: Ordered list of candidate model IDs.
        seed: Random seed for reproducibility.
        hparams: Optional hyperparameters override dict passed to the classifier.

    Returns:
        Fitted sklearn classifier.
    """
    X = _embeddings_to_matrix(train_embeddings)
    y, _ = _prepare_labels(train_data, models)

    hp = hparams or {}

    if kind == "knn":
        defaults = {"n_neighbors": 5, "metric": "cosine", "weights": "distance"}
        defaults.update(hp)
        clf = KNeighborsClassifier(**defaults)
    elif kind == "svm":
        defaults = {
            "kernel": "rbf", "C": 1.0, "gamma": "scale",
            "probability": False, "random_state": seed,
        }
        defaults.update(hp)
        clf = SVC(**defaults)
    elif kind == "mlp":
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "max_iter": 500,
            "random_state": seed,
        }
        defaults.update(hp)
        clf = MLPClassifier(**defaults)
    else:
        raise ValueError(f"Unknown router kind: {kind!r}. Use 'knn', 'svm', or 'mlp'.")

    clf.fit(X, y)
    return clf


def evaluate_supervised_router(
    clf: Any,
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    models: List[str],
    costs: Dict[str, float],
    *,
    per_prompt: bool = False,
) -> Dict[str, Any]:
    """Evaluate a frozen supervised router on a holdout set.

    Args:
        clf: Fitted sklearn classifier (from ``train_supervised_router``).
        eval_data: Holdout prompts with reward dicts.
        eval_embeddings: Pre-computed PCA feature vectors for holdout.
        models: Ordered list of candidate model IDs.
        costs: Per-model cost dict.
        per_prompt: If True, also return per-prompt rewards and costs.

    Returns:
        Dict with ``avg_reward``, ``avg_cost``, ``model_counts``, and
        optionally ``per_prompt_rewards`` / ``per_prompt_costs``.
    """
    X = _embeddings_to_matrix(eval_embeddings)
    preds = clf.predict(X)

    r_total = 0.0
    c_total = 0.0
    counts: Dict[str, int] = {m: 0 for m in models}
    pp_r: Optional[List[float]] = [] if per_prompt else None
    pp_c: Optional[List[float]] = [] if per_prompt else None

    for i, (d, pred_idx) in enumerate(zip(eval_data, preds)):
        chosen = models[pred_idx]
        reward = d["rewards"].get(chosen, 0.0)
        cost = costs.get(chosen, 0.0)
        r_total += reward
        c_total += cost
        counts[chosen] = counts.get(chosen, 0) + 1
        if pp_r is not None:
            pp_r.append(reward)
            pp_c.append(cost)

    n = len(eval_data)
    result: Dict[str, Any] = {
        "avg_reward": r_total / n,
        "avg_cost": c_total / n,
        "model_counts": counts,
    }
    if per_prompt:
        result["per_prompt_rewards"] = pp_r
        result["per_prompt_costs"] = pp_c
    return result


def tune_supervised_hparams(
    kind: str,
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    val_data: List[Dict],
    val_embeddings: List[np.ndarray],
    models: List[str],
    costs: Dict[str, float],
    *,
    grid: Optional[Dict[str, List[Any]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Select supervised baseline hyperparameters via grid search on dev-val.

    Trains each configuration on *train_data*, evaluates on *val_data*,
    and returns the hyperparameter dict that maximises dev-val reward.
    This mirrors the dev-val selection protocol used for BanditGPT's
    hyperparameters (Appendix H), ensuring symmetric tuning effort.

    Args:
        kind: One of ``"knn"``, ``"svm"``, ``"mlp"``.
        train_data: Dev-train prompts with reward dicts.
        train_embeddings: PCA feature vectors for dev-train.
        val_data: Dev-val prompts with reward dicts.
        val_embeddings: PCA feature vectors for dev-val.
        models: Ordered list of candidate model IDs.
        costs: Per-model cost dict.
        grid: Hyperparameter grid (param_name -> list of values).
            Defaults to ``HPARAM_GRIDS[kind]``.
        seed: Random seed for stochastic classifiers.

    Returns:
        Dict with ``best_hparams``, ``best_val_reward``, and
        ``all_configs`` (list of tried configurations with their
        val rewards).
    """
    search_grid = grid if grid is not None else HPARAM_GRIDS.get(kind, {})
    if not search_grid:
        return {"best_hparams": {}, "best_val_reward": 0.0, "all_configs": []}

    param_names = list(search_grid.keys())
    param_values = [search_grid[k] for k in param_names]

    best_reward = -np.inf
    best_hp: Dict[str, Any] = {}
    all_configs: List[Dict[str, Any]] = []

    for combo in itertools.product(*param_values):
        hp = dict(zip(param_names, combo))
        clf = train_supervised_router(
            kind, train_data, train_embeddings, models,
            seed=seed, hparams=hp,
        )
        res = evaluate_supervised_router(
            clf, val_data, val_embeddings, models, costs,
        )
        val_r = res["avg_reward"]
        all_configs.append({**hp, "val_reward": val_r})
        if val_r > best_reward:
            best_reward = val_r
            best_hp = hp

    logger.info(
        f"    {kind.upper()} tuning: {len(all_configs)} configs, "
        f"best val_reward={best_reward:.4f} with {best_hp}"
    )
    return {
        "best_hparams": best_hp,
        "best_val_reward": float(best_reward),
        "all_configs": all_configs,
    }


def run_supervised_baseline(
    kind: str,
    models: List[str],
    costs: Dict[str, float],
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    *,
    n_trials: int = 5,
    seed_start: int = 42,
    per_prompt: bool = False,
    hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train-then-freeze evaluation of a supervised router across multiple seeds.

    For KNN (deterministic given data), a single trial suffices. For SVM and
    MLP (stochastic initialisation), we average over ``n_trials`` seeds to
    estimate variance.

    Args:
        kind: One of ``"knn"``, ``"svm"``, ``"mlp"``.
        models: Ordered list of candidate model IDs.
        costs: Per-model cost dict.
        train_data: Dev-train prompts with reward dicts.
        train_embeddings: PCA feature vectors for dev-train.
        eval_data: Holdout prompts with reward dicts.
        eval_embeddings: PCA feature vectors for holdout.
        n_trials: Number of random seed trials (default 5).
        seed_start: Starting seed (default 42).
        per_prompt: If True, collect per-prompt arrays across seeds.
        hparams: Optional hyperparameter overrides.

    Returns:
        Dict with ``reward``, ``cost``, ``std_reward``, ``std_cost``,
        ``n_trials``, ``kind``, and optionally seed-averaged per-prompt arrays.
    """
    effective_trials = 1 if kind == "knn" else n_trials

    rewards_per_trial: List[float] = []
    costs_per_trial: List[float] = []
    all_pp_rewards: Optional[List[List[float]]] = [] if per_prompt else None
    all_pp_costs: Optional[List[List[float]]] = [] if per_prompt else None

    for trial in range(effective_trials):
        seed = seed_start + trial
        clf = train_supervised_router(
            kind, train_data, train_embeddings, models,
            seed=seed, hparams=hparams,
        )
        res = evaluate_supervised_router(
            clf, eval_data, eval_embeddings, models, costs,
            per_prompt=per_prompt,
        )
        rewards_per_trial.append(res["avg_reward"])
        costs_per_trial.append(res["avg_cost"])
        if all_pp_rewards is not None:
            all_pp_rewards.append(res["per_prompt_rewards"])
            all_pp_costs.append(res["per_prompt_costs"])

    out: Dict[str, Any] = {
        "kind": kind,
        "reward": float(np.mean(rewards_per_trial)),
        "cost": float(np.mean(costs_per_trial)),
        "std_reward": float(np.std(rewards_per_trial, ddof=1)) if effective_trials > 1 else 0.0,
        "std_cost": float(np.std(costs_per_trial, ddof=1)) if effective_trials > 1 else 0.0,
        "n_trials": effective_trials,
        "model_counts": res["model_counts"],
    }
    if all_pp_rewards is not None:
        out["per_seed_per_prompt_rewards"] = [
            list(s) for s in all_pp_rewards
        ]
        out["per_seed_per_prompt_costs"] = [
            list(s) for s in all_pp_costs
        ]
    return out


def run_supervised_learning_curve(
    kind: str,
    models: List[str],
    costs: Dict[str, float],
    train_data: List[Dict],
    train_embeddings: List[np.ndarray],
    eval_data: List[Dict],
    eval_embeddings: List[np.ndarray],
    checkpoints: List[int],
    *,
    n_trials: int = 5,
    seed_start: int = 42,
    hparams: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Retrain a supervised baseline at each checkpoint and evaluate on holdout.

    Mirrors BanditGPT's learning curve by training on increasing subsets of
    ``train_data`` and evaluating frozen performance on the full holdout set
    at each checkpoint.

    Args:
        kind: One of ``"knn"``, ``"svm"``, ``"mlp"``.
        models: Ordered list of candidate model IDs.
        costs: Per-model cost dict.
        train_data: Dev-train prompts with reward dicts.
        train_embeddings: PCA feature vectors for dev-train.
        eval_data: Holdout prompts with reward dicts.
        eval_embeddings: PCA feature vectors for holdout.
        checkpoints: Sorted list of training-set sizes at which to evaluate.
        n_trials: Number of random seed trials for stochastic methods.
        seed_start: Starting seed.
        hparams: Tuned hyperparameter overrides.

    Returns:
        List of dicts with ``step``, ``mean_reward``, ``std_reward``,
        ``mean_cost``, ``std_cost`` — same schema as BanditGPT's
        learning curve for easy overlay plotting.
    """
    hp = hparams or {}
    n_neighbors = hp.get("n_neighbors", 5) if kind == "knn" else 0
    effective_trials = 1 if kind == "knn" else n_trials

    curve: List[Dict[str, Any]] = []
    for step in checkpoints:
        if step == 0:
            continue
        if kind == "knn" and step < n_neighbors:
            continue

        subset_data = train_data[:step]
        subset_emb = train_embeddings[:step]

        trial_rewards: List[float] = []
        trial_costs: List[float] = []
        for trial in range(effective_trials):
            seed = seed_start + trial
            clf = train_supervised_router(
                kind, subset_data, subset_emb, models,
                seed=seed, hparams=hp,
            )
            res = evaluate_supervised_router(
                clf, eval_data, eval_embeddings, models, costs,
            )
            trial_rewards.append(res["avg_reward"])
            trial_costs.append(res["avg_cost"])

        mean_r = float(np.mean(trial_rewards))
        std_r = (
            float(np.std(trial_rewards, ddof=1))
            if effective_trials > 1
            else 0.0
        )
        mean_c = float(np.mean(trial_costs))
        std_c = (
            float(np.std(trial_costs, ddof=1))
            if effective_trials > 1
            else 0.0
        )
        curve.append({
            "step": step,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "mean_cost": mean_c,
            "std_cost": std_c,
        })

    return curve
