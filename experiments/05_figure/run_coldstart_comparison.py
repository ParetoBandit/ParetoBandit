#!/usr/bin/env python3
"""
Cold-Start Experiment: Hybrid vs Disjoint LinUCB Without Warmup Priors
======================================================================

Tests whether family parameter sharing provides an advantage when the
router starts from scratch (no offline warmup priors).

Hypothesis
----------
When strong warmup priors exist, per-arm estimates are already better
than what family pooling can provide from limited online data, making
Hybrid redundant.  In a cold-start scenario every arm starts with
identity covariance and zero bias, so family sharing could bootstrap
faster convergence by pooling observations across related models.

Design
------
- **No warmup priors**: ``warmup_path=None`` in ``create_experiment_router``.
- **Still uses Corralling**: both experts start cold (no priors loaded).
- **Alpha values**: {0.5, 1.0, 2.0} — cold-start needs more exploration.
- **n_eff irrelevant**: with no priors loaded, prior scaling has no effect.
- **Portfolios**: ``K_CONFIGS_FAMILIES`` (K=5 and K=10) — same portfolios
  with strong within-provider Pearson correlations (r >= 0.5).
- **Learning curves**: holdout reward recorded at ``EVAL_INTERVAL=25``
  checkpoints plus the final step.
- **20 seeds, paired t-test** for statistical rigor.

Output
------
- ``results/coldstart_results.json``
- ``results/coldstart_figure.png``
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.multimodel import (
    build_model_registry,
    load_multimodel_data,
    N_TRIALS,
    SEED_OFFSET,
    CORRALLING_LR,
    CORRALLING_GAMMA,
)
from utils.router_factory import create_experiment_router

from run_k_scaling_experiment import (
    K_CONFIGS_FAMILIES,
    CORRELATION_METHOD,
    CORRELATION_THRESHOLD,
    COST_PENALTY,
    compute_provider_correlations,
    build_family_map_data_driven,
    compute_oracle,
    evaluate_holdout,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EVAL_INTERVAL: int = 25
ALPHA_VALUES: List[float] = [0.5, 1.0, 2.0]
NEFF_DEFAULT: float = 10.0


def _set_global_seeds(seed: int) -> None:
    """Set all global RNG seeds for strict reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def run_coldstart_trial(
    models: List[str],
    family_map: Optional[Dict[str, str]],
    train_data: Sequence[Dict[str, Any]],
    train_emb: Sequence[np.ndarray],
    holdout_data: Sequence[Dict[str, Any]],
    holdout_emb: Sequence[np.ndarray],
    r_min: float,
    r_range: float,
    seed: int,
    total_steps: int,
    alpha: float,
) -> Tuple[Dict[int, float], List[float], Counter]:
    """Run one cold-start trial (no warmup priors).

    Parameters
    ----------
    models : list[str]
        Model IDs for the router.
    family_map : dict[str, str] or None
        Hybrid family map, or ``None`` for Disjoint.
    train_data, train_emb :
        Training data and pre-computed embeddings.
    holdout_data, holdout_emb :
        Holdout data and pre-computed embeddings.
    r_min : float
        Minimum raw reward for [0, 1] normalization.
    r_range : float
        Reward range for normalization.
    seed : int
        Random seed for this trial.
    total_steps : int
        Number of training steps.
    alpha : float
        Warmup expert exploration coefficient.

    Returns
    -------
    tuple[dict[int, float], list[float], Counter]
        ``(holdout_curve, online_rewards, arm_pull_counts)``
    """
    _set_global_seeds(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_data))

    policy = "hybrid" if family_map is not None else "disjoint"
    router = create_experiment_router(
        model_registry=build_model_registry(models),
        feature_dim=train_emb[0].shape[0],
        prior_n_effective=NEFF_DEFAULT,
        alpha=alpha,
        warmup_path=None,
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=COST_PENALTY,
        policy=policy,
        family_map=family_map,
    )

    checkpoints = set(range(0, total_steps + 1, EVAL_INTERVAL))
    checkpoints.add(total_steps)

    holdout_curve: Dict[int, float] = {}
    holdout_curve[0] = evaluate_holdout(
        router, holdout_data, holdout_emb, models
    )

    online_rewards: List[float] = []
    arm_pulls: Counter = Counter()

    for step_i in range(total_steps):
        i = idx[step_i]
        emb = train_emb[i]
        model, log = router.route(emb, total_steps=total_steps)
        raw_reward = train_data[i]["rewards"].get(model, 0.0)
        norm_reward = (
            (raw_reward - r_min) / r_range if r_range > 1e-6 else 0.5
        )
        router.process_feedback(log.request_id, norm_reward)
        online_rewards.append(raw_reward)
        arm_pulls[model] += 1

        step = step_i + 1
        if step in checkpoints:
            holdout_curve[step] = evaluate_holdout(
                router, holdout_data, holdout_emb, models
            )

    return holdout_curve, online_rewards, arm_pulls


def run_experiment() -> Dict[str, Any]:
    """Execute the full cold-start comparison experiment.

    Returns
    -------
    dict
        Complete results keyed by ``(K, alpha)`` label.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("COLD-START EXPERIMENT: Hybrid vs Disjoint (No Warmup Priors)")
    logger.info("=" * 70)

    max_k = max(K_CONFIGS_FAMILIES)
    all_models = K_CONFIGS_FAMILIES[max_k]
    logger.info(f"Loading data for {len(all_models)} models...")
    train_data, holdout_data, train_emb, holdout_emb, costs, r_min, r_max = (
        load_multimodel_data(all_models)
    )
    r_range = r_max - r_min
    total_steps = len(train_data)

    logger.info(
        f"  Train: {len(train_data)} | Holdout: {len(holdout_data)} "
        f"| r_range: [{r_min:.3f}, {r_max:.3f}]"
    )

    _, train_reward_vecs = compute_provider_correlations(train_data, all_models)

    all_results: Dict[str, Any] = {
        "_meta": {
            "train_prompts": len(train_data),
            "holdout_prompts": len(holdout_data),
            "alpha_values": ALPHA_VALUES,
            "n_trials": N_TRIALS,
            "warmup": "none (cold-start)",
            "eval_interval": EVAL_INTERVAL,
        }
    }

    for K, models in sorted(K_CONFIGS_FAMILIES.items()):
        fmap = build_family_map_data_driven(train_reward_vecs, models)
        families: Dict[str, List[str]] = defaultdict(list)
        for m, f in fmap.items():
            families[f].append(m)
        n_shared = sum(1 for ms in families.values() if len(ms) > 1)
        oracle = compute_oracle(holdout_data, models)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"K = {K}  (oracle={oracle:.4f}, {n_shared} shared families)")
        logger.info(f"{'=' * 70}")

        for alpha in ALPHA_VALUES:
            label = f"K{K}_a{alpha}"
            logger.info(f"\n  alpha={alpha}")

            hybrid_curves: List[Dict[int, float]] = []
            disjoint_curves: List[Dict[int, float]] = []
            hybrid_online: List[List[float]] = []
            disjoint_online: List[List[float]] = []

            for trial in range(N_TRIALS):
                seed = SEED_OFFSET + trial
                logger.info(f"    Trial {trial + 1}/{N_TRIALS} (seed={seed})...")

                hc, h_on, _ = run_coldstart_trial(
                    models, fmap, train_data, train_emb,
                    holdout_data, holdout_emb, r_min, r_range,
                    seed, total_steps, alpha,
                )
                dc, d_on, _ = run_coldstart_trial(
                    models, None, train_data, train_emb,
                    holdout_data, holdout_emb, r_min, r_range,
                    seed, total_steps, alpha,
                )
                hybrid_curves.append(hc)
                disjoint_curves.append(dc)
                hybrid_online.append(h_on)
                disjoint_online.append(d_on)

            all_steps = sorted(hybrid_curves[0].keys())
            h_mat = np.array(
                [[c[s] for s in all_steps] for c in hybrid_curves]
            )
            d_mat = np.array(
                [[c[s] for s in all_steps] for c in disjoint_curves]
            )

            h_final = h_mat[:, -1]
            d_final = d_mat[:, -1]
            t_stat, p_val = sp_stats.ttest_rel(h_final, d_final)

            # Time-to-best: step at which each policy first reaches
            # within 0.5 pp of its own final holdout reward.
            def _time_to_best(mat: np.ndarray, steps: List[int]) -> float:
                per_trial: List[int] = []
                for row in mat:
                    final = row[-1]
                    threshold = final - 0.005
                    for j, s in enumerate(steps):
                        if row[j] >= threshold:
                            per_trial.append(s)
                            break
                    else:
                        per_trial.append(steps[-1])
                return float(np.mean(per_trial))

            h_ttb = _time_to_best(h_mat, all_steps)
            d_ttb = _time_to_best(d_mat, all_steps)

            logger.info(
                f"    Hybrid  final: {np.mean(h_final):.4f} "
                f"+/- {1.96 * np.std(h_final) / np.sqrt(N_TRIALS):.4f}  "
                f"ttb={h_ttb:.0f}"
            )
            logger.info(
                f"    Disjoint final: {np.mean(d_final):.4f} "
                f"+/- {1.96 * np.std(d_final) / np.sqrt(N_TRIALS):.4f}  "
                f"ttb={d_ttb:.0f}"
            )
            logger.info(
                f"    t={t_stat:.3f}, p={p_val:.6f}"
            )

            all_results[label] = {
                "K": K,
                "alpha": alpha,
                "models": models,
                "oracle": oracle,
                "n_shared_families": n_shared,
                "eval_steps": all_steps,
                "hybrid_holdout_mean": np.mean(h_mat, axis=0).tolist(),
                "hybrid_holdout_ci95": (
                    1.96 * np.std(h_mat, axis=0) / np.sqrt(N_TRIALS)
                ).tolist(),
                "disjoint_holdout_mean": np.mean(d_mat, axis=0).tolist(),
                "disjoint_holdout_ci95": (
                    1.96 * np.std(d_mat, axis=0) / np.sqrt(N_TRIALS)
                ).tolist(),
                "hybrid_final": float(np.mean(h_final)),
                "hybrid_final_ci": float(
                    1.96 * np.std(h_final) / np.sqrt(N_TRIALS)
                ),
                "disjoint_final": float(np.mean(d_final)),
                "disjoint_final_ci": float(
                    1.96 * np.std(d_final) / np.sqrt(N_TRIALS)
                ),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "hybrid_time_to_best": h_ttb,
                "disjoint_time_to_best": d_ttb,
            }

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    results_path = out_dir / "coldstart_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    generate_figure(all_results, out_dir)
    return all_results


def generate_figure(
    results: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Generate learning-curve figure for cold-start Hybrid vs Disjoint.

    Layout: rows = K values, columns = alpha values.  Each panel shows
    holdout reward convergence with 95% CI bands.

    Parameters
    ----------
    results : dict
        Full results from :func:`run_experiment`.
    out_dir : Path
        Directory to write the figure.
    """
    config_keys = sorted(k for k in results if k != "_meta")
    k_values = sorted(set(results[k]["K"] for k in config_keys))
    alpha_values = sorted(set(results[k]["alpha"] for k in config_keys))

    n_rows = len(k_values)
    n_cols = len(alpha_values)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for row, K in enumerate(k_values):
        for col, alpha in enumerate(alpha_values):
            label = f"K{K}_a{alpha}"
            if label not in results:
                axes[row, col].set_visible(False)
                continue

            r = results[label]
            ax = axes[row, col]
            steps = np.array(r["eval_steps"])
            h_m = np.array(r["hybrid_holdout_mean"])
            h_c = np.array(r["hybrid_holdout_ci95"])
            d_m = np.array(r["disjoint_holdout_mean"])
            d_c = np.array(r["disjoint_holdout_ci95"])

            ax.fill_between(steps, h_m - h_c, h_m + h_c, alpha=0.15, color="C0")
            ax.fill_between(steps, d_m - d_c, d_m + d_c, alpha=0.15, color="C1")
            ax.plot(steps, h_m, "C0-", lw=1.5, label="Hybrid")
            ax.plot(steps, d_m, "C1--", lw=1.5, label="Disjoint")
            ax.axhline(r["oracle"], color="gray", ls=":", lw=1,
                       label=f"Oracle ({r['oracle']:.3f})")

            p = r["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            gap = r["hybrid_final"] - r["disjoint_final"]
            ax.set_title(
                f"K={K}, alpha={alpha} (cold-start)\n"
                f"gap={gap:+.4f} ({sig}, p={p:.3f})",
                fontsize=9,
            )
            ax.set_xlabel("Training steps")
            if col == 0:
                ax.set_ylabel("Holdout reward")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Cold-Start: Hybrid vs Disjoint (No Warmup Priors)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = out_dir / "coldstart_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_experiment()
