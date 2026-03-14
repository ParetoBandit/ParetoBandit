#!/usr/bin/env python3
"""Appendix: Alpha Sweep (Pareto AUC) at PCA-25.

Grid search over exploration parameter (alpha) for **two variants** —
BanditGPT (warmup priors) and Tabula Rasa (cold start) — to find the
configuration maximizing Pareto AUC on the K=3 portfolio.

We fix PCA dimensionality to d=25 (~28.5% cumulative variance) to
retain a broad semantic representation of prompts while keeping the
feature space manageable for LinUCB.  A separate PCA ablation
(Appendix I) shows the Pareto AUC surface is flat across d in [6, 25],
confirming that the choice of d=25 does not sacrifice performance.

Fixed: PCA=25, n_eff=5000 (warmup only), gamma=1.0, disjoint LinUCB.

For each (variant, alpha) pair, the router is trained on the val split
across 7 cost_penalty values x 10 seeds, then evaluated on the holdout
split.  The 7 (mean_cost, mean_reward) holdout points define a Pareto
frontier whose AUC is the selection metric.  A single global cost
range is used across all configs.

Usage::

    python experiments_v2/appendix/hparam_sweep/run_hparam_sweep.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
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

ALPHA_VALUES: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
COST_PENALTIES: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
VARIANTS: List[str] = ["banditgpt", "tabula_rasa"]

PCA_DIM: int = 25
N_EFF: float = 5000.0
GAMMA: float = 1.0
N_SEEDS: int = 10
SEED_OFFSET: int = 7000

ARM_ORDER: List[str] = K3_ARM_ORDER
RESULTS_DIR = Path(__file__).parent / "results"


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


def _parse_and_embed(
    records: List[Dict[str, Any]],
    fs: FeatureService,
) -> Dict[str, Any]:
    """Extract prompts, rewards, costs, and embed via FeatureService.

    Args:
        records: JSONL records with ``prompt`` and ``arms`` fields.
        fs: Feature service configured with the target PCA.

    Returns:
        Dict with ``prompts``, ``rewards``, ``costs``, ``embeddings``, ``n``.
    """
    prompts = [r["prompt"] for r in records]
    rewards: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    costs: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    for r in records:
        for arm_id in ARM_ORDER:
            info = r["arms"][arm_id]
            rewards[arm_id].append(info["reward"])
            costs[arm_id].append(info["cost"])

    embeddings = fs.extract_features_batch(prompts)

    return {
        "prompts": prompts,
        "rewards": {a: np.array(v) for a, v in rewards.items()},
        "costs": {a: np.array(v) for a, v in costs.items()},
        "embeddings": embeddings,
        "n": len(prompts),
    }


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

    logger.info("Loading data records ...")
    val_records = _load_jsonl(VAL_DATA_PATH)
    holdout_records = _load_jsonl(HOLDOUT_DATA_PATH)
    logger.info("  val=%d  holdout=%d", len(val_records), len(holdout_records))

    logger.info("Initializing FeatureService (PCA-%d) ...", PCA_DIM)
    fs = FeatureService(pca_components=PCA_DIM)
    feature_dim = fs.dimension
    logger.info("  feature_dim=%d (PCA-%d + bias)", feature_dim, PCA_DIM)

    logger.info("Encoding and embedding prompts ...")
    val_data = _parse_and_embed(val_records, fs)
    holdout_data = _parse_and_embed(holdout_records, fs)

    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K3_WARMUP_PRIORS_PATH)

    total_configs = len(VARIANTS) * len(ALPHA_VALUES)
    total_trials = total_configs * len(COST_PENALTIES) * N_SEEDS
    logger.info(
        "Sweep: %d variants x %d alpha = %d configs, "
        "%d cost_penalty x %d seeds = %d total trials",
        len(VARIANTS),
        len(ALPHA_VALUES),
        total_configs,
        len(COST_PENALTIES),
        N_SEEDS,
        total_trials,
    )

    # Phase 1: collect (cost, reward) curves.
    all_results: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        use_warmup = variant == "banditgpt"
        wp = warmup_path if use_warmup else None
        logger.info("\n--- %s ---", variant)

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
                "pca_dim": PCA_DIM,
                "cp_costs": [round(c, 6) for c in cp_costs],
                "cp_rewards": [round(r, 6) for r in cp_rewards],
            })

            logger.info("  alpha=%.3f  %s  (done)", alpha, variant)

    # Phase 2: Pareto AUC with global cost range.
    global_costs = [c for r in all_results for c in r["cp_costs"]]
    global_lo = min(global_costs)
    global_hi = max(global_costs)
    logger.info(
        "\nGlobal cost range: [%.6f, %.6f]", global_lo, global_hi,
    )

    for entry in all_results:
        auc = pareto_auc(
            entry["cp_costs"], entry["cp_rewards"], global_lo, global_hi,
        )
        entry["pareto_auc"] = round(auc, 6)
        entry["cost_lo"] = round(global_lo, 6)
        entry["cost_hi"] = round(global_hi, 6)

    # Phase 3: best per variant.
    per_variant_best: Dict[str, Dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_results = [r for r in all_results if r["variant"] == variant]
        best = max(variant_results, key=lambda x: x["pareto_auc"])
        per_variant_best[variant] = {
            "alpha": best["alpha"],
            "pca_dim": PCA_DIM,
            "pareto_auc": best["pareto_auc"],
        }

    for variant in VARIANTS:
        b = per_variant_best[variant]
        logger.info(
            "BEST %s: alpha=%.3f, Pareto AUC=%.6f",
            variant, b["alpha"], b["pareto_auc"],
        )
        ranked = sorted(
            [r for r in all_results if r["variant"] == variant],
            key=lambda x: x["pareto_auc"],
            reverse=True,
        )
        for i, r in enumerate(ranked):
            logger.info(
                "  %2d. alpha=%.3f  AUC=%.6f", i + 1, r["alpha"], r["pareto_auc"],
            )

    output = {
        "experiment": "appendix_alpha_sweep",
        "grid": {
            "variants": VARIANTS,
            "alpha_values": ALPHA_VALUES,
            "pca_dim": PCA_DIM,
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
