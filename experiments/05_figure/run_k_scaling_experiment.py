#!/usr/bin/env python3
"""
K-Scaling Experiment: Hybrid vs. Disjoint LinUCB Sample Efficiency
===================================================================

Tests whether family parameter sharing in the **production** BanditRouter
(Corralling + warmup priors) improves convergence as the model portfolio
grows from K=5 to K=10.

Hypothesis
----------
With K models and N training prompts, each model sees ~N/K observations
on average (actual allocation is non-uniform due to bandit exploitation).
In Hybrid LinUCB, the family-level beta is updated by ALL family members'
observations, pooling data across similar models.  As K grows, per-model
observations shrink while family-level observations remain substantial,
making the shared component increasingly valuable.

Design
------
- Two portfolio sizes: K=5, K=10
- Both portfolios include multi-member providers so that shared families
  can potentially form at every K value (controls family density across
  the comparison)
- Condition A (Hybrid): ``policy="hybrid"`` with data-driven family
  assignments via tetrachoric correlation on **train** rewards
  (within-provider, threshold r_tet >= 0.6)
- Condition B (Disjoint): ``policy="disjoint"`` (all independent arms)
- Both use the **full production router**: Corralling meta-learner with
  warmup priors (Expert 1) and tabula rasa (Expert 2)
- Router is exercised via ``route()`` + ``process_feedback()`` — the
  same code path production traffic uses
- Common dataset across all K (intersection of K=10 model superset)
- Metrics: holdout reward at checkpoints, cumulative online reward,
  empirical arm-pull distribution
- 20 seeds per condition, paired t-test

Data
----
Uses the three-way split (prior-train / online-learn / holdout) from
the shared multimodel utilities, matching other experiments (04, 07, 08).
Family assignments are computed on the **training** set only.  The holdout
set is used exclusively for evaluation, preventing target leakage.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    TARGET_NEFF,
    ALPHA_START,
    CORRALLING_LR,
    CORRALLING_GAMMA,
)
from utils.router_factory import create_experiment_router

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Experiment parameters ─────────────────────────────────────────────
EVAL_INTERVAL: int = 25
ROLLING_WINDOW: int = 50
TETRACHORIC_THRESHOLD: float = 0.6
COST_PENALTY: float = 0.0  # quality-only evaluation

# Both portfolios draw from MODEL_CATALOG — models with full cost data,
# warmup priors, and production support.  Each K includes multi-member
# providers (OpenAI, Meta, Anthropic, Google) so shared families can
# form at every portfolio size, avoiding the confound of family density
# varying with K.
K_CONFIGS: Dict[int, List[str]] = {
    5: [
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-4-maverick",
        "anthropic/claude-sonnet-4",
    ],
    10: [
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-4-maverick",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-haiku-4.5",
        "google/gemma-3-27b-it",
        "google/gemini-2.5-flash-preview-09-2025",
        "mistralai/mixtral-8x7b-instruct",
        "deepseek/deepseek-chat-v3-0324",
    ],
}


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
    threshold: float = TETRACHORIC_THRESHOLD,
) -> Dict[str, str]:
    """Build a data-driven family map using tetrachoric correlation.

    Only models present in *models* are included; *reward_vecs* may
    contain a superset.

    Parameters
    ----------
    reward_vecs : dict[str, np.ndarray]
        Per-model reward vectors (computed on the **train** set).
    models : sequence of str
        Subset of model IDs to include.
    threshold : float
        Minimum tetrachoric correlation for family membership.

    Returns
    -------
    dict[str, str]
        Mapping from model ID to family label.
    """
    subset = {m: reward_vecs[m] for m in models if m in reward_vecs}
    return compute_correlation_families(subset, threshold=threshold)


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
    family_map: Optional[Dict[str, str]],
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
    Corralling, warmup priors, and the given policy mode, then trains
    via ``route()`` / ``process_feedback()`` — the same API production
    traffic uses.

    Parameters
    ----------
    models : list[str]
        Model IDs for the router.
    family_map : dict[str, str] or None
        Hybrid family map (``policy="hybrid"``), or ``None`` for
        Disjoint (``policy="disjoint"``).
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

    policy = "hybrid" if family_map is not None else "disjoint"
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
    logger.info("K-SCALING EXPERIMENT: Hybrid vs Disjoint (Production Router)")
    logger.info("=" * 70)

    # Load data for the K=10 superset via shared multimodel pipeline
    # (three-way split: prior-train / online-learn / holdout)
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

    # Family correlations computed on the TRAIN set only (no holdout leakage)
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
            "tetrachoric_threshold": TETRACHORIC_THRESHOLD,
            "correlation_source": "train",
            "cost_penalty": COST_PENALTY,
            "router_type": "BanditRouter (Corralling + warmup + tabula_rasa)",
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

        fmap = build_family_map_data_driven(train_reward_vecs, models)
        families: Dict[str, List[str]] = defaultdict(list)
        for m, f in fmap.items():
            families[f].append(m)
        n_shared = sum(1 for ms in families.values() if len(ms) > 1)
        members_in_shared = sum(
            len(ms) for ms in families.values() if len(ms) > 1
        )

        oracle = compute_oracle(holdout_data, models)
        logger.info(f"  Models: {K}")
        logger.info(
            f"  Families: {len(families)} ({n_shared} shared, "
            f"{len(families) - n_shared} singletons)"
        )
        for fam, members in sorted(families.items()):
            if len(members) > 1:
                short = [m.split("/")[-1] for m in members]
                logger.info(f"    {fam}: {short}")
        logger.info(f"  Models in shared families: {members_in_shared}/{K}")
        logger.info(
            f"  Train prompts: {total_steps}, Holdout: {len(holdout_data)}"
        )
        logger.info(f"  Oracle reward: {oracle:.4f}")
        logger.info(f"  Avg obs/model (uniform): {total_steps / K:.0f}")

        hybrid_holdout: List[Dict[int, float]] = []
        disjoint_holdout: List[Dict[int, float]] = []
        hybrid_online: List[List[float]] = []
        disjoint_online: List[List[float]] = []
        hybrid_pulls: List[Counter] = []
        disjoint_pulls: List[Counter] = []

        for trial in range(N_TRIALS):
            seed = SEED_OFFSET + trial
            logger.info(f"  Trial {trial + 1}/{N_TRIALS} (seed={seed})...")

            hc, h_on, h_ap = run_trial(
                models, fmap, train_data, train_emb,
                holdout_data, holdout_emb, r_min, r_range,
                seed, total_steps,
            )
            dc, d_on, d_ap = run_trial(
                models, None, train_data, train_emb,
                holdout_data, holdout_emb, r_min, r_range,
                seed, total_steps,
            )
            hybrid_holdout.append(hc)
            disjoint_holdout.append(dc)
            hybrid_online.append(h_on)
            disjoint_online.append(d_on)
            hybrid_pulls.append(h_ap)
            disjoint_pulls.append(d_ap)

        # ── Holdout aggregation ───────────────────────────────────────
        all_steps = sorted(hybrid_holdout[0].keys())
        hybrid_hmat = np.array(
            [[c[s] for s in all_steps] for c in hybrid_holdout]
        )
        disjoint_hmat = np.array(
            [[c[s] for s in all_steps] for c in disjoint_holdout]
        )
        hybrid_final = hybrid_hmat[:, -1]
        disjoint_final = disjoint_hmat[:, -1]
        t_final, p_final = sp_stats.ttest_rel(hybrid_final, disjoint_final)

        # ── Online reward aggregation ─────────────────────────────────
        h_online_mat = np.array(hybrid_online)
        d_online_mat = np.array(disjoint_online)

        def rolling_mean(
            mat: np.ndarray, w: int = ROLLING_WINDOW
        ) -> np.ndarray:
            kernel = np.ones(w) / w
            return np.array([
                np.convolve(row, kernel, mode="valid") for row in mat
            ])

        h_rolling = rolling_mean(h_online_mat)
        d_rolling = rolling_mean(d_online_mat)
        rolling_steps = np.arange(ROLLING_WINDOW, total_steps + 1)

        h_cum_final = np.sum(h_online_mat, axis=1)
        d_cum_final = np.sum(d_online_mat, axis=1)
        t_cum, p_cum = sp_stats.ttest_rel(h_cum_final, d_cum_final)
        cum_diff = h_cum_final - d_cum_final
        cohens_d_cum = float(
            np.mean(cum_diff) / (np.std(cum_diff, ddof=1) + 1e-12)
        )

        # ── Arm-pull distribution ─────────────────────────────────────
        h_pull_summary = _summarize_arm_pulls(hybrid_pulls, models)
        d_pull_summary = _summarize_arm_pulls(disjoint_pulls, models)

        logger.info(f"\n  Results for K={K}:")
        logger.info(f"    Oracle:               {oracle:.4f}")
        logger.info(
            f"    Hybrid holdout final: {np.mean(hybrid_final):.4f} "
            f"+/- {1.96 * np.std(hybrid_final) / np.sqrt(N_TRIALS):.4f}"
        )
        logger.info(
            f"    Disjoint holdout final: {np.mean(disjoint_final):.4f} "
            f"+/- {1.96 * np.std(disjoint_final) / np.sqrt(N_TRIALS):.4f}"
        )
        logger.info(
            f"    Holdout t-test:  t={t_final:.3f}, p={p_final:.6f}"
        )
        logger.info(f"    Cumulative online reward:")
        logger.info(
            f"      Hybrid:  {np.mean(h_cum_final):.1f} "
            f"+/- {1.96 * np.std(h_cum_final) / np.sqrt(N_TRIALS):.1f}"
        )
        logger.info(
            f"      Disjoint: {np.mean(d_cum_final):.1f} "
            f"+/- {1.96 * np.std(d_cum_final) / np.sqrt(N_TRIALS):.1f}"
        )
        logger.info(
            f"      t={t_cum:.3f}, p={p_cum:.6f}, d={cohens_d_cum:.3f}"
        )
        logger.info(
            f"    Arm-pull distribution (mean obs/model across trials):"
        )
        logger.info(
            f"      Hybrid:  min={h_pull_summary['across_models_min']:.0f}  "
            f"median={h_pull_summary['across_models_median']:.0f}  "
            f"max={h_pull_summary['across_models_max']:.0f}"
        )
        logger.info(
            f"      Disjoint: min={d_pull_summary['across_models_min']:.0f}  "
            f"median={d_pull_summary['across_models_median']:.0f}  "
            f"max={d_pull_summary['across_models_max']:.0f}"
        )

        all_results[K] = {
            "models": models,
            "family_map": fmap,
            "families": {f: ms for f, ms in families.items()},
            "n_families": len(families),
            "n_shared_families": n_shared,
            "members_in_shared": members_in_shared,
            "train_prompts": total_steps,
            "holdout_prompts": len(holdout_data),
            "obs_per_model_avg": total_steps / K,
            "oracle": oracle,
            "eval_steps": all_steps,
            "hybrid_holdout_mean": np.mean(hybrid_hmat, axis=0).tolist(),
            "hybrid_holdout_ci95": (
                1.96 * np.std(hybrid_hmat, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "disjoint_holdout_mean": np.mean(
                disjoint_hmat, axis=0
            ).tolist(),
            "disjoint_holdout_ci95": (
                1.96 * np.std(disjoint_hmat, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "hybrid_final": float(np.mean(hybrid_final)),
            "hybrid_final_ci": float(
                1.96 * np.std(hybrid_final) / np.sqrt(N_TRIALS)
            ),
            "disjoint_final": float(np.mean(disjoint_final)),
            "disjoint_final_ci": float(
                1.96 * np.std(disjoint_final) / np.sqrt(N_TRIALS)
            ),
            "final_t_stat": float(t_final),
            "final_p_value": float(p_final),
            "rolling_steps": rolling_steps.tolist(),
            "hybrid_rolling_mean": np.mean(h_rolling, axis=0).tolist(),
            "hybrid_rolling_ci95": (
                1.96 * np.std(h_rolling, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "disjoint_rolling_mean": np.mean(d_rolling, axis=0).tolist(),
            "disjoint_rolling_ci95": (
                1.96 * np.std(d_rolling, axis=0) / np.sqrt(N_TRIALS)
            ).tolist(),
            "hybrid_cum_mean": float(np.mean(h_cum_final)),
            "hybrid_cum_ci": float(
                1.96 * np.std(h_cum_final) / np.sqrt(N_TRIALS)
            ),
            "disjoint_cum_mean": float(np.mean(d_cum_final)),
            "disjoint_cum_ci": float(
                1.96 * np.std(d_cum_final) / np.sqrt(N_TRIALS)
            ),
            "cum_t_stat": float(t_cum),
            "cum_p_value": float(p_cum),
            "cum_cohens_d": cohens_d_cum,
            "hybrid_arm_pulls": h_pull_summary,
            "disjoint_arm_pulls": d_pull_summary,
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
        n_shared = r["n_shared_families"]
        n_fam = r["n_families"]

        ax = axes[col]
        steps = np.array(r["eval_steps"])
        h_hm = np.array(r["hybrid_holdout_mean"])
        h_hc = np.array(r["hybrid_holdout_ci95"])
        d_hm = np.array(r["disjoint_holdout_mean"])
        d_hc = np.array(r["disjoint_holdout_ci95"])

        ax.fill_between(
            steps, h_hm - h_hc, h_hm + h_hc, alpha=0.15, color="C0"
        )
        ax.fill_between(
            steps, d_hm - d_hc, d_hm + d_hc, alpha=0.15, color="C1"
        )
        ax.plot(
            steps, h_hm, "C0-", lw=1.5,
            label=f"Hybrid ({n_shared} shared fam.)",
        )
        ax.plot(
            steps, d_hm, "C1--", lw=1.5,
            label=f"Disjoint ({K} indep. arms)",
        )
        ax.axhline(
            r["oracle"], color="gray", ls=":", lw=1,
            label=f"Oracle ({r['oracle']:.3f})",
        )

        p_f = r["final_p_value"]
        sig_f = (
            "***" if p_f < 0.001
            else "**" if p_f < 0.01
            else "*" if p_f < 0.05
            else "ns"
        )
        gap_f = r["hybrid_final"] - r["disjoint_final"]
        ax.set_title(
            f"K={K}  ({n_fam} fam, {n_shared} shared)\n"
            f"Holdout gap: {gap_f:+.4f} ({sig_f})",
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
