#!/usr/bin/env python3
"""
K-Scaling Experiment: DisjointLinUCB Sample Efficiency
======================================================

Measures how the production BanditRouter (Corralling + warmup priors,
Disjoint LinUCB policy) scales as the model portfolio grows from K=5
to K=10.

Design
------
- Two portfolio sizes: K=5, K=10
- Both portfolios include multi-member providers
- Uses the **full production router**: Corralling meta-learner with
  warmup priors (Expert 1) and tabula rasa (Expert 2)
- Router is exercised via ``route()`` + ``process_feedback()`` — the
  same code path production traffic uses
- Common dataset across all K (intersection of K=10 model superset)
- Metrics: holdout reward at checkpoints, cumulative online reward,
  empirical arm-pull distribution
- 20 seeds per condition

Data
----
Uses the three-way split (prior-train / online-learn / holdout) from
the shared multimodel utilities, matching other experiments (04, 07, 08).
The holdout set is used exclusively for evaluation, preventing target
leakage.

Output
------
- results/k_scaling_results.json
- results/k_scaling_figure.png (1-row x 2-col figure)
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.router import (
    BanditRouter,
    compute_correlation_families,
    tetrachoric_corr,
)
from utils.multimodel import (
    MODEL_CATALOG,
    MULTIMODEL_WARMUP_PRIORS_PATH,
    build_model_registry,
    load_multimodel_data,
    N_TRIALS,
    SEED_OFFSET,
    CORRALLING_LR,
    CORRALLING_GAMMA,
)

# Optimal (alpha, n_eff) from the 2D joint ablation (Appendix H).
# Overrides the shared defaults (alpha=0.5, n_eff=10) which were
# effectively cold-start due to the n_warmup=20000 fallback.
ALPHA_START: float = 0.25
TARGET_NEFF: float = 1000.0
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Experiment parameters ─────────────────────────────────────────────
EVAL_INTERVAL: int = 25
ROLLING_WINDOW: int = 50
CORRELATION_METHOD: str = "pearson"
CORRELATION_THRESHOLD: float = 0.5
COST_PENALTY: float = 0.0  # quality-only evaluation

# ── Portfolio design ───────────────────────────────────────────────────
# Models are chosen to maximise within-provider Pearson correlation on
# continuous rewards, giving Hybrid LinUCB the best conditions for
# family-based parameter sharing.  A separate "no families" control
# (one model per provider) verifies that Hybrid degenerates to Disjoint
# when every arm is a singleton.
#
# "families" portfolios — high Pearson within-provider pairs:
#   Anthropic: claude-sonnet-4 + claude-sonnet-4.5   (r=0.638, MAD=0.041)
#   Google:    gemma-3-12b     + gemma-3-27b          (r=0.611, MAD=0.054)
#   Google:    gemini-2.5-flash + gemini-2.5-pro      (r=0.600, MAD=0.049)
#   Meta:      llama-3.1-70b   + llama-4-scout        (r=0.534, MAD=0.134)
#
# "no_families" portfolio — one model per provider (sanity check):
#   Hybrid with all singletons must exactly match Disjoint.

K_CONFIGS_FAMILIES: Dict[int, List[str]] = {
    5: [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "openai/gpt-4.1",
    ],
    10: [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "google/gemini-2.5-flash-preview-09-2025",
        "google/gemini-2.5-pro-preview-06-05",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-4-scout",
        "openai/gpt-4.1",
        "openai/gpt-5.1",
    ],
}

K_CONFIGS_NO_FAMILIES: Dict[int, List[str]] = {
    5: [
        "anthropic/claude-sonnet-4",
        "google/gemma-3-27b-it",
        "meta-llama/llama-4-maverick",
        "mistralai/mixtral-8x7b-instruct",
        "openai/gpt-4.1",
    ],
}

# Combined for backward compatibility — grid search iterates both.
K_CONFIGS: Dict[int, List[str]] = K_CONFIGS_FAMILIES


def _set_global_seeds(seed: int) -> None:
    """Set all global RNG seeds for strict reproducibility.

    The router's ``_argmax_random_tiebreak`` uses ``np.random.randint``
    from the global NumPy RNG.  Without this, tie-breaking varies across
    runs even when the per-trial local RNG is seeded.
    """
    np.random.seed(seed)
    random.seed(seed)


# ── Within-provider reward correlations ───────────────────────────────

def compute_provider_correlations(
    data: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, np.ndarray]]:
    """Compute pairwise tetrachoric and phi correlations within each provider.

    Parameters
    ----------
    data : sequence of dicts
        Prompt-centric records with ``"rewards"`` dicts.
    models : sequence of str
        Model IDs to include.

    Returns
    -------
    tuple[dict, dict]
        ``(provider_results, reward_vectors)`` where *provider_results*
        maps provider name to a list of pairwise correlation dicts, and
        *reward_vectors* maps model ID to its reward array.
    """
    reward_vecs: Dict[str, np.ndarray] = {
        m: np.array([p["rewards"][m] for p in data]) for m in models
    }

    providers: Dict[str, List[str]] = defaultdict(list)
    for m in sorted(models):
        prov = m.split("/")[0] if "/" in m else "__none__"
        providers[prov].append(m)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for prov, members in sorted(providers.items()):
        if len(members) < 2:
            continue
        pair_results: List[Dict[str, Any]] = []
        for m1, m2 in combinations(members, 2):
            phi, p_val = sp_stats.pearsonr(reward_vecs[m1], reward_vecs[m2])
            r_tet = tetrachoric_corr(reward_vecs[m1], reward_vecs[m2])
            pair_results.append({
                "pair": (m1, m2),
                "phi": float(phi),
                "r_tet": float(r_tet),
                "p": float(p_val),
            })
        results[prov] = pair_results
    return results, reward_vecs


def build_family_map_data_driven(
    reward_vecs: Dict[str, np.ndarray],
    models: Sequence[str],
    threshold: float = CORRELATION_THRESHOLD,
    method: str = CORRELATION_METHOD,
) -> Dict[str, str]:
    """Build a data-driven family map using within-provider correlations.

    Only models present in *models* are included; *reward_vecs* may
    contain a superset.

    Parameters
    ----------
    reward_vecs : dict[str, np.ndarray]
        Per-model continuous reward vectors (computed on the **train** set).
    models : sequence of str
        Subset of model IDs to include.
    threshold : float
        Minimum correlation for family membership.
    method : str
        Correlation method: ``"pearson"`` (continuous) or
        ``"tetrachoric"`` (binary).

    Returns
    -------
    dict[str, str]
        Mapping from model ID to family label.
    """
    subset = {m: reward_vecs[m] for m in models if m in reward_vecs}
    return compute_correlation_families(subset, threshold=threshold, method=method)


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_holdout(
    router: BanditRouter,
    holdout_data: Sequence[Dict[str, Any]],
    holdout_emb: Sequence[np.ndarray],
    models: List[str],
) -> float:
    """Frozen holdout evaluation via the production ``route()`` API.

    Calls ``router.route()`` without ``process_feedback()`` so the
    router state is not modified.  Uses the default ``total_steps=1``
    to immediately yield ``alpha_end`` (near-greedy selection).

    Parameters
    ----------
    router : BanditRouter
        Trained (or partially trained) router.
    holdout_data : sequence of dicts
        Holdout prompt records with ``"rewards"`` dicts.
    holdout_emb : sequence of np.ndarray
        Pre-computed embeddings aligned with *holdout_data*.
    models : list[str]
        Model IDs (for looking up rewards; selection is done by router).

    Returns
    -------
    float
        Mean reward over the holdout set.
    """
    rewards: List[float] = []
    for i, p in enumerate(holdout_data):
        model, _log = router.route(holdout_emb[i])
        rewards.append(p["rewards"].get(model, 0.0))
    return float(np.mean(rewards))


def compute_oracle(
    holdout_data: Sequence[Dict[str, Any]],
    models: Sequence[str],
) -> float:
    """Compute the per-prompt best-model upper bound (oracle reward).

    Parameters
    ----------
    holdout_data : sequence of dicts
        Holdout prompt records with ``"rewards"`` dicts.
    models : sequence of str
        Model IDs to consider.

    Returns
    -------
    float
        Mean of per-prompt maximum rewards.
    """
    return float(np.mean([
        max(p["rewards"][m] for m in models) for p in holdout_data
    ]))


# ── Single trial ──────────────────────────────────────────────────────

def run_trial(
    models: List[str],
    train_data: Sequence[Dict[str, Any]],
    train_emb: Sequence[np.ndarray],
    holdout_data: Sequence[Dict[str, Any]],
    holdout_emb: Sequence[np.ndarray],
    r_min: float,
    r_range: float,
    seed: int,
    total_steps: int,
) -> Tuple[Dict[int, float], List[float], Counter]:
    """Run one trial using the full production BanditRouter.

    Creates a ``BanditRouter`` via ``create_experiment_router`` with
    Corralling and warmup priors (Disjoint LinUCB), then trains via
    ``route()`` / ``process_feedback()`` — the same API production
    traffic uses.

    Parameters
    ----------
    models : list[str]
        Model IDs for the router.
    train_data : sequence of dicts
        Training prompt records with ``"rewards"`` dicts.
    train_emb : sequence of np.ndarray
        Pre-computed training embeddings.
    holdout_data : sequence of dicts
        Holdout prompt records (evaluation only — no feedback).
    holdout_emb : sequence of np.ndarray
        Pre-computed holdout embeddings.
    r_min : float
        Minimum raw reward (for [0, 1] normalization).
    r_range : float
        Reward range (max - min) for normalization.
    seed : int
        Random seed for this trial (controls shuffle + global RNG).
    total_steps : int
        Number of training steps.

    Returns
    -------
    tuple[dict[int, float], list[float], Counter]
        ``(holdout_curve, online_rewards, arm_pull_counts)`` where
        *holdout_curve* maps training step to holdout reward,
        *online_rewards* is the per-step reward sequence, and
        *arm_pull_counts* tracks how many times each model was selected.
    """
    _set_global_seeds(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_data))

    router = create_experiment_router(
        model_registry=build_model_registry(models),
        feature_dim=train_emb[0].shape[0],
        prior_n_effective=TARGET_NEFF,
        alpha=ALPHA_START,
        warmup_path=str(MULTIMODEL_WARMUP_PRIORS_PATH),
        use_corralling=True,
        corralling_learning_rate=CORRALLING_LR,
        corralling_gamma=CORRALLING_GAMMA,
        cost_penalty=COST_PENALTY,
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


def _summarize_arm_pulls(
    all_pulls: List[Counter],
    models: List[str],
) -> Dict[str, Any]:
    """Aggregate arm-pull distributions across trials.

    Parameters
    ----------
    all_pulls : list[Counter]
        One Counter per trial mapping model ID to pull count.
    models : list[str]
        Full model list (some may have zero pulls in some trials).

    Returns
    -------
    dict
        Summary statistics: per-model mean/std/min/max, and overall
        min/median/max of per-model means across the portfolio.
    """
    per_model_counts: Dict[str, List[int]] = {m: [] for m in models}
    for pulls in all_pulls:
        for m in models:
            per_model_counts[m].append(pulls.get(m, 0))

    per_model_summary: Dict[str, Dict[str, float]] = {}
    all_means: List[float] = []
    for m in models:
        arr = np.array(per_model_counts[m])
        mean_val = float(np.mean(arr))
        per_model_summary[m] = {
            "mean": mean_val,
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
        }
        all_means.append(mean_val)

    all_means_arr = np.array(all_means)
    return {
        "per_model": per_model_summary,
        "across_models_min": float(np.min(all_means_arr)),
        "across_models_median": float(np.median(all_means_arr)),
        "across_models_max": float(np.max(all_means_arr)),
    }


# ── Main experiment loop ──────────────────────────────────────────────

def run_experiment() -> Dict[str | int, Any]:
    """Execute the full K-scaling experiment through the production router.

    Returns
    -------
    dict
        Complete results keyed by K value, plus ``"_meta"`` metadata.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("K-SCALING EXPERIMENT: Disjoint LinUCB (Production Router)")
    logger.info("=" * 70)

    max_k = max(K_CONFIGS)
    all_models = K_CONFIGS[max_k]
    logger.info(f"Loading data for {len(all_models)} models (K={max_k} superset)...")
    train_data, holdout_data, train_emb, holdout_emb, costs, r_min, r_max = (
        load_multimodel_data(all_models)
    )
    r_range = r_max - r_min
    feature_dim = train_emb[0].shape[0]
    total_steps = len(train_data)

    logger.info(
        f"  Train: {len(train_data)} | Holdout: {len(holdout_data)} "
        f"| dim: {feature_dim} | r_range: [{r_min:.3f}, {r_max:.3f}]"
    )

    # Within-provider reward correlations (for analytics/logging only)
    logger.info(
        "\n── Within-provider reward correlations (TRAIN set, tetrachoric) ──"
    )
    prov_corr, train_reward_vecs = compute_provider_correlations(
        train_data, all_models
    )
    for prov, pairs in prov_corr.items():
        for pc in pairs:
            m1, m2 = pc["pair"]
            logger.info(
                f"  {prov}: {m1.split('/')[-1]} vs {m2.split('/')[-1]}  "
                f"phi={pc['phi']:.3f}  r_tet={pc['r_tet']:.3f}"
            )

    all_results: Dict[str | int, Any] = {
        "_meta": {
            "train_prompts": len(train_data),
            "holdout_prompts": len(holdout_data),
            "feature_dim": feature_dim,
            "correlation_method": CORRELATION_METHOD,
            "correlation_threshold": CORRELATION_THRESHOLD,
            "correlation_source": "train",
            "cost_penalty": COST_PENALTY,
            "router_type": "BanditRouter (Corralling + warmup + tabula_rasa, Disjoint)",
            "provider_correlations": {
                prov: [
                    {
                        "pair": pc["pair"],
                        "phi": pc["phi"],
                        "r_tet": pc["r_tet"],
                        "p": pc["p"],
                    }
                    for pc in pairs
                ]
                for prov, pairs in prov_corr.items()
            },
        }
    }

    for K, models in sorted(K_CONFIGS.items()):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"K = {K}")
        logger.info(f"{'=' * 70}")

        # Family map computed for analytics/logging only
        fmap = build_family_map_data_driven(train_reward_vecs, models)
        families: Dict[str, List[str]] = defaultdict(list)
        for m, f in fmap.items():
            families[f].append(m)
        n_shared = sum(1 for ms in families.values() if len(ms) > 1)

        oracle = compute_oracle(holdout_data, models)
        logger.info(f"  Models: {K}")
        logger.info(
            f"  Train prompts: {total_steps}, Holdout: {len(holdout_data)}"
        )
        logger.info(f"  Oracle reward: {oracle:.4f}")
        logger.info(f"  Avg obs/model (uniform): {total_steps / K:.0f}")

        trial_holdout: List[Dict[int, float]] = []
        trial_online: List[List[float]] = []
        trial_pulls: List[Counter] = []

        for trial in range(N_TRIALS):
            seed = SEED_OFFSET + trial
            logger.info(f"  Trial {trial + 1}/{N_TRIALS} (seed={seed})...")

            hc, on_r, ap = run_trial(
                models, train_data, train_emb,
                holdout_data, holdout_emb, r_min, r_range,
                seed, total_steps,
            )
            trial_holdout.append(hc)
            trial_online.append(on_r)
            trial_pulls.append(ap)

        # ── Holdout aggregation ───────────────────────────────────────
        all_steps = sorted(trial_holdout[0].keys())
        holdout_mat = np.array(
            [[c[s] for s in all_steps] for c in trial_holdout]
        )
        final_rewards = holdout_mat[:, -1]

        # ── Online reward aggregation ─────────────────────────────────
        online_mat = np.array(trial_online)

        def rolling_mean(
            mat: np.ndarray, w: int = ROLLING_WINDOW
        ) -> np.ndarray:
            kernel = np.ones(w) / w
            return np.array([
                np.convolve(row, kernel, mode="valid") for row in mat
            ])

        rolling = rolling_mean(online_mat)
        rolling_steps = np.arange(ROLLING_WINDOW, total_steps + 1)

        cum_final = np.sum(online_mat, axis=1)

        # ── Arm-pull distribution ─────────────────────────────────────
        pull_summary = _summarize_arm_pulls(trial_pulls, models)

        logger.info(f"\n  Results for K={K}:")
        logger.info(f"    Oracle:               {oracle:.4f}")
        logger.info(
            f"    Holdout final: {np.mean(final_rewards):.4f} "
            f"+/- {1.96 * np.std(final_rewards) / np.sqrt(N_TRIALS):.4f}"
        )
        logger.info(f"    Cumulative online reward:")
        logger.info(
            f"      Mean: {np.mean(cum_final):.1f} "
            f"+/- {1.96 * np.std(cum_final) / np.sqrt(N_TRIALS):.1f}"
        )
        logger.info(
            f"    Arm-pull distribution (mean obs/model across trials):"
        )
        logger.info(
            f"      min={pull_summary['across_models_min']:.0f}  "
            f"median={pull_summary['across_models_median']:.0f}  "
            f"max={pull_summary['across_models_max']:.0f}"
        )

        all_results[K] = {
            "models": models,
            "family_map": fmap,
            "families": {f: ms for f, ms in families.items()},
            "n_families": len(families),
            "n_shared_families": n_shared,
            "train_prompts": total_steps,
            "holdout_prompts": len(holdout_data),
            "obs_per_model_avg": total_steps / K,
            "oracle": oracle,
            "eval_steps": all_steps,
            "holdout_mean": np.mean(holdout_mat, axis=0).tolist(),
            "holdout_ci95": (
                1.96 * np.std(holdout_mat, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "final_reward": float(np.mean(final_rewards)),
            "final_reward_ci": float(
                1.96 * np.std(final_rewards) / np.sqrt(N_TRIALS)
            ),
            "rolling_steps": rolling_steps.tolist(),
            "rolling_mean": np.mean(rolling, axis=0).tolist(),
            "rolling_ci95": (
                1.96 * np.std(rolling, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "cum_mean": float(np.mean(cum_final)),
            "cum_ci": float(
                1.96 * np.std(cum_final) / np.sqrt(N_TRIALS)
            ),
            "arm_pulls": pull_summary,
        }

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "k_scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_dir / 'k_scaling_results.json'}")

    generate_figure(all_results, out_dir)
    return all_results


# ── Figure generation ─────────────────────────────────────────────────

def generate_figure(
    results: Dict[str | int, Any],
    out_dir: Path,
) -> None:
    """Generate a 1-row x N-col holdout reward convergence figure.

    Parameters
    ----------
    results : dict
        Full experiment results from :func:`run_experiment`.
    out_dir : Path
        Directory to write the figure PNG.
    """
    k_keys = sorted(k for k in results if k != "_meta")
    fig, axes = plt.subplots(
        1, len(k_keys), figsize=(5 * len(k_keys), 3.5)
    )
    if len(k_keys) == 1:
        axes = [axes]

    for col, K in enumerate(k_keys):
        r = results[K]
        ax = axes[col]
        steps = np.array(r["eval_steps"])
        h_mean = np.array(r["holdout_mean"])
        h_ci = np.array(r["holdout_ci95"])

        ax.fill_between(
            steps, h_mean - h_ci, h_mean + h_ci, alpha=0.15, color="C0"
        )
        ax.plot(
            steps, h_mean, "C0-", lw=1.5,
            label=f"Disjoint ({K} arms)",
        )
        ax.axhline(
            r["oracle"], color="gray", ls=":", lw=1,
            label=f"Oracle ({r['oracle']:.3f})",
        )

        ax.set_title(
            f"K={K}  final={r['final_reward']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Training steps")
        if col == 0:
            ax.set_ylabel("Holdout reward (greedy)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "k_scaling_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    run_experiment()
