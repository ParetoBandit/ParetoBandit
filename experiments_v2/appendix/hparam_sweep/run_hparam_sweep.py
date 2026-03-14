#!/usr/bin/env python3
"""Appendix: Alpha x PCA Component Sweep (Pareto AUC).

Joint grid search over exploration parameter (alpha) and PCA
dimensionality for **two variants** — BanditGPT (warmup priors) and
Tabula Rasa (cold start) — to find the configuration maximizing
Pareto AUC on the K=3 portfolio.

Fixed across both variants: n_eff=5000 (warmup only), gamma=1.0,
disjoint LinUCB.

For each (variant, alpha, pca_dim) triple, the router is trained on
the val split across 7 cost_penalty values x 10 seeds, then evaluated
on the holdout split.  The 7 (mean_cost, mean_reward) holdout points
define a Pareto frontier whose AUC is the selection metric.  A single
global cost range is used for all configs to ensure comparable AUC.

PCA truncation note
-------------------
The shipped ``pca_32.joblib`` was fitted on ~47K LMSYS prompts via
``all-MiniLM-L6-v2`` (384-dim, ``whiten=False``).  The pre-built
``pca_N.joblib`` files for N in {15, 20, 25, 30} share identical
``mean_``, ``components_[:N]``, and ``explained_variance_[:N]`` with
``pca_32`` — they are simple truncations.  (The ``pca_6``, ``pca_8``,
``pca_10`` files are from a *different* encoder — ``BAAI/bge-m3``,
1024-dim — and are **not** used here.)

``project_embeddings()`` applies manual whitening
(``/ sqrt(explained_variance)``) when ``pca.whiten=False``, so the
coordinate system is consistent across all truncation levels.

Usage::

    python experiments_v2/appendix/hparam_sweep/run_hparam_sweep.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    FULL_PCA_PATH,
    K3_ARM_ORDER,
    K3_WARMUP_PRIORS_PATH,
    VAL_DATA_PATH,
    HOLDOUT_DATA_PATH,
)
from bandit_gpt.router import BanditRouter
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.storage import EphemeralContextStore
from utils.embeddings import project_embeddings
from utils.pareto import pareto_auc
from utils.simulation import build_model_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Sweep Grid
# ======================================================================

ALPHA_VALUES: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
PCA_COMPONENTS: List[int] = [6, 8, 10, 15, 20, 25, 30]
COST_PENALTIES: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
VARIANTS: List[str] = ["banditgpt", "tabula_rasa"]

N_EFF: float = 5000.0
GAMMA: float = 1.0
N_SEEDS: int = 10
SEED_OFFSET: int = 7000

ARM_ORDER: List[str] = K3_ARM_ORDER
RESULTS_DIR = Path(__file__).parent / "results"


# ======================================================================
# PCA / Prior Truncation
# ======================================================================


def truncate_pca(pca_full: PCA, n_components: int) -> PCA:
    """Return a truncated copy keeping the first *n_components* axes.

    If *n_components* >= the full model's dimensionality, the original
    PCA object is returned unchanged.
    """
    if n_components >= pca_full.n_components_:
        return pca_full
    pca = PCA(n_components=n_components, whiten=pca_full.whiten)
    pca.components_ = pca_full.components_[:n_components]
    pca.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca.singular_values_ = pca_full.singular_values_[:n_components]
    pca.mean_ = pca_full.mean_
    pca.n_components_ = n_components
    pca.n_features_in_ = pca_full.n_features_in_
    pca.n_samples_ = pca_full.n_samples_
    pca.noise_variance_ = pca_full.noise_variance_
    return pca


def resize_warmup_priors(
    priors: Dict[str, Any],
    n_components: int,
) -> Dict[str, Any]:
    """Resize sufficient statistics (A, b) to match *n_components*.

    Truncates when *n_components* < original, pads with identity/zeros
    when *n_components* > original.  The bias row/column is always
    preserved in the last position.
    """
    old_pca = priors["context_dim"] - 1
    if n_components == old_pca:
        return copy.deepcopy(priors)

    old_dim = priors["context_dim"]
    new_dim = n_components + 1

    new_A: Dict[str, np.ndarray] = {}
    new_b: Dict[str, np.ndarray] = {}

    if n_components < old_pca:
        keep_idx = list(range(n_components)) + [old_dim - 1]
        for m in priors["models"]:
            new_A[m] = priors["A"][m][np.ix_(keep_idx, keep_idx)]
            new_b[m] = priors["b"][m][keep_idx]
    else:
        for m in priors["models"]:
            A_old = priors["A"][m]
            b_old = priors["b"][m]
            A_new = np.eye(new_dim, dtype=np.float64)
            b_new = np.zeros(new_dim, dtype=np.float64)
            A_new[:old_pca, :old_pca] = A_old[:old_pca, :old_pca]
            b_new[:old_pca] = b_old[:old_pca]
            A_new[-1, :old_pca] = A_old[-1, :old_pca]
            A_new[:old_pca, -1] = A_old[:old_pca, -1]
            A_new[-1, -1] = A_old[-1, -1]
            b_new[-1] = b_old[-1]
            new_A[m] = A_new
            new_b[m] = b_new

    out = copy.deepcopy(priors)
    out["A"] = new_A
    out["b"] = new_b
    out["context_dim"] = new_dim
    return out


# ======================================================================
# Data Loading
# ======================================================================


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file returning a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _parse_records(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract prompts, rewards, and costs from JSONL records.

    Returns a dict with ``prompts``, ``rewards``, ``costs``, ``n``
    (no embeddings yet — those are added per PCA setting).
    """
    prompts = [r["prompt"] for r in records]
    rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    for r in records:
        for arm_id in ARM_ORDER:
            info = r["arms"][arm_id]
            rewards[arm_id].append(info["reward"])
            costs[arm_id].append(info["cost"])
    return {
        "prompts": prompts,
        "rewards": {a: np.array(v) for a, v in rewards.items()},
        "costs": {a: np.array(v) for a, v in costs.items()},
        "n": len(prompts),
    }


def _attach_embeddings(
    parsed: Dict[str, Any],
    raw_emb: np.ndarray,
    pca_model: PCA,
) -> Dict[str, Any]:
    """Project raw embeddings through *pca_model* and attach to parsed data.

    Returns a new dict with ``embeddings`` added.
    """
    emb_list = project_embeddings(raw_emb, pca_model)
    embeddings = np.vstack(emb_list)
    return {**parsed, "embeddings": embeddings}


# ======================================================================
# Trial Runner
# ======================================================================


def _run_trial(
    *,
    val_data: Dict[str, Any],
    holdout_data: Dict[str, Any],
    registry: Dict[str, Any],
    feature_dim: int,
    warmup_path: Optional[str],
    alpha: float,
    cost_penalty: float,
    seed: int,
) -> Dict[str, float]:
    """Train on val, evaluate on holdout.

    Args:
        warmup_path: Path to warmup priors.  ``None`` for tabula rasa.

    Returns:
        Dict with ``mean_reward`` and ``mean_cost`` on holdout.
    """
    rng = np.random.default_rng(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()

    use_warmup = warmup_path is not None
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_warmup else "none",
        warmup_path=warmup_path if use_warmup else None,
        prior_n_effective=N_EFF if use_warmup else 1.0,
        alpha=alpha,
        use_corralling=False,
        cost_penalty=cost_penalty,
        forgetting_factor=GAMMA,
        drift_threshold=0.0,
        policy="disjoint",
        adaptive_gamma=False,
        budget_pacer=None,
    )

    val_order = rng.permutation(val_data["n"])
    for i in val_order:
        model, log = router.route(val_data["embeddings"][i])
        reward = float(val_data["rewards"][model][i])
        log.cost_usd = float(val_data["costs"][model][i])
        router.process_feedback(log.request_id, reward=reward)

    holdout_order = rng.permutation(holdout_data["n"])
    holdout_rewards: List[float] = []
    holdout_costs: List[float] = []

    for i in holdout_order:
        model, log = router.route(holdout_data["embeddings"][i])
        reward = float(holdout_data["rewards"][model][i])
        cost = float(holdout_data["costs"][model][i])
        holdout_rewards.append(reward)
        holdout_costs.append(cost)
        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

    return {
        "mean_reward": float(np.mean(holdout_rewards)),
        "mean_cost": float(np.mean(holdout_costs)),
    }


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading 32-component PCA ...")
    pca_32: PCA = joblib.load(FULL_PCA_PATH)

    logger.info("Loading K=3 warmup priors (25-comp) ...")
    priors_25: Dict[str, Any] = joblib.load(K3_WARMUP_PRIORS_PATH)
    logger.info(
        "  context_dim=%d, models=%s",
        priors_25["context_dim"],
        priors_25["models"],
    )

    logger.info("Loading data records ...")
    val_records = _load_jsonl(VAL_DATA_PATH)
    holdout_records = _load_jsonl(HOLDOUT_DATA_PATH)
    logger.info("  val=%d  holdout=%d", len(val_records), len(holdout_records))

    val_parsed = _parse_records(val_records)
    holdout_parsed = _parse_records(holdout_records)

    logger.info("Encoding raw embeddings (SentenceTransformer) ...")
    encoder = FeatureService()
    raw_val = encoder.encode_prompts_batch(val_parsed["prompts"])
    raw_holdout = encoder.encode_prompts_batch(holdout_parsed["prompts"])
    logger.info("  raw val: %s, raw holdout: %s", raw_val.shape, raw_holdout.shape)

    registry = build_model_registry(ARM_ORDER)

    total_configs = len(VARIANTS) * len(ALPHA_VALUES) * len(PCA_COMPONENTS)
    total_trials = total_configs * len(COST_PENALTIES) * N_SEEDS
    logger.info(
        "Sweep: %d variants x %d alpha x %d PCA = %d configs, "
        "%d cost_penalty x %d seeds = %d total trials",
        len(VARIANTS),
        len(ALPHA_VALUES),
        len(PCA_COMPONENTS),
        total_configs,
        len(COST_PENALTIES),
        N_SEEDS,
        total_trials,
    )

    # Phase 1: collect (cost, reward) curves for every config.
    all_results: List[Dict[str, Any]] = []

    for pca_dim in PCA_COMPONENTS:
        logger.info("\n========== PCA dim = %d ==========", pca_dim)

        pca_trunc = truncate_pca(pca_32, pca_dim)
        priors_trunc = resize_warmup_priors(priors_25, pca_dim)
        feature_dim = pca_dim + 1

        tmp_priors_path = RESULTS_DIR / f"_tmp_priors_{pca_dim}comp.joblib"
        joblib.dump(priors_trunc, tmp_priors_path)

        logger.info("  Projecting embeddings (PCA %d) ...", pca_dim)
        val_data = _attach_embeddings(val_parsed, raw_val, pca_trunc)
        holdout_data = _attach_embeddings(holdout_parsed, raw_holdout, pca_trunc)

        for variant in VARIANTS:
            use_warmup = variant == "banditgpt"
            wp = str(tmp_priors_path) if use_warmup else None
            logger.info("  --- %s ---", variant)

            for alpha in ALPHA_VALUES:
                cp_costs: List[float] = []
                cp_rewards: List[float] = []

                for cp in COST_PENALTIES:
                    seed_rewards: List[float] = []
                    seed_costs: List[float] = []

                    for s in range(N_SEEDS):
                        seed = SEED_OFFSET + s
                        result = _run_trial(
                            val_data=val_data,
                            holdout_data=holdout_data,
                            registry=registry,
                            feature_dim=feature_dim,
                            warmup_path=wp,
                            alpha=alpha,
                            cost_penalty=cp,
                            seed=seed,
                        )
                        seed_rewards.append(result["mean_reward"])
                        seed_costs.append(result["mean_cost"])

                    cp_costs.append(float(np.mean(seed_costs)))
                    cp_rewards.append(float(np.mean(seed_rewards)))

                all_results.append({
                    "variant": variant,
                    "alpha": alpha,
                    "pca_dim": pca_dim,
                    "cp_costs": [round(c, 6) for c in cp_costs],
                    "cp_rewards": [round(r, 6) for r in cp_rewards],
                })

                logger.info(
                    "    alpha=%.2f  pca=%d  %s  (done)",
                    alpha, pca_dim, variant,
                )

        tmp_priors_path.unlink(missing_ok=True)

    # Phase 2: compute Pareto AUC with a single global cost range.
    global_costs = [c for r in all_results for c in r["cp_costs"]]
    global_lo = min(global_costs)
    global_hi = max(global_costs)
    logger.info(
        "\nGlobal cost range for Pareto AUC: [%.6f, %.6f]",
        global_lo, global_hi,
    )

    for entry in all_results:
        auc = pareto_auc(
            entry["cp_costs"], entry["cp_rewards"], global_lo, global_hi,
        )
        entry["pareto_auc"] = round(auc, 6)
        entry["cost_lo"] = round(global_lo, 6)
        entry["cost_hi"] = round(global_hi, 6)

    # Phase 3: find best per variant.
    per_variant_best: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_results = [r for r in all_results if r["variant"] == variant]
        best = max(variant_results, key=lambda x: x["pareto_auc"])
        per_variant_best[variant] = {
            "alpha": best["alpha"],
            "pca_dim": best["pca_dim"],
            "pareto_auc": best["pareto_auc"],
        }
        logger.info(
            "BEST %s: alpha=%.2f, pca=%d, Pareto AUC=%.6f",
            variant, best["alpha"], best["pca_dim"], best["pareto_auc"],
        )

    # Log ranked top-10 per variant.
    for variant in VARIANTS:
        ranked = sorted(
            [r for r in all_results if r["variant"] == variant],
            key=lambda x: x["pareto_auc"],
            reverse=True,
        )
        logger.info("\nTop 10 (%s):", variant)
        for i, r in enumerate(ranked[:10]):
            logger.info(
                "  %2d. alpha=%.2f  pca=%2d  AUC=%.6f",
                i + 1, r["alpha"], r["pca_dim"], r["pareto_auc"],
            )

    output = {
        "experiment": "appendix_hparam_sweep",
        "grid": {
            "variants": VARIANTS,
            "alpha_values": ALPHA_VALUES,
            "pca_components": PCA_COMPONENTS,
            "cost_penalties": COST_PENALTIES,
            "n_eff": N_EFF,
            "gamma": GAMMA,
            "n_seeds": N_SEEDS,
            "seed_offset": SEED_OFFSET,
        },
        "global_cost_range": [round(global_lo, 6), round(global_hi, 6)],
        "best_per_variant": per_variant_best,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "hparam_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)

    best_path = RESULTS_DIR / "best_hparams.json"
    with open(best_path, "w") as f:
        json.dump(per_variant_best, f, indent=2)
    logger.info("Best hparams written to %s", best_path)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
