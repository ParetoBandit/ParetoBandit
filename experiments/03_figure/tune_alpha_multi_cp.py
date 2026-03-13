#!/usr/bin/env python3
"""Tune alpha independently at each cost_penalty for the warmup ablation.

For each cost_penalty in {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}, sweeps
alpha and n_eff on the val split (20 seeds) and selects the best
hyperparameters per (condition, cost_penalty).

Protocol follows PILOT (EMNLP 2025): tune alpha to maximise reward at
each specific operating point.

Usage:
    python experiments/03_figure/tune_alpha_multi_cp.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import (
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ARM_ORDER = K2_ARM_ORDER
ALPHA_GRID = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
N_EFF_GRID_WARMUP = [5000.0]
COST_PENALTIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
N_SEEDS = 20
SEED_OFFSET = 500


def _simulate_val_reward(
    train: SplitData,
    val: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    alpha: float,
    prior_n_effective: float,
    cost_penalty: float,
    warmup_path: str,
    use_priors: bool,
    seed: int,
) -> float:
    """Run train-then-val and return mean val reward."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="warmup" if use_priors else "none",
        warmup_path=warmup_path if use_priors else None,
        prior_n_effective=prior_n_effective,
        alpha=alpha,
        use_corralling=False,
        cost_penalty=cost_penalty,
        forgetting_factor=1.0,
        policy="disjoint",
    )

    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    val_rewards = []
    for i in rng.permutation(val.n):
        model, log = router.route(val.embeddings[i])
        reward = float(val.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)
        val_rewards.append(reward)

    return float(np.mean(val_rewards))


def main() -> None:
    t0 = time.time()
    logger.info("Loading data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    train = load_split(TRAIN_DATA_PATH, fs, ARM_ORDER)
    val = load_split(VAL_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    logger.info("  Train=%d  Val=%d  dim=%d", train.n, val.n, feature_dim)

    all_results: Dict[str, Any] = {}
    best_hparams: Dict[str, Dict[str, Any]] = {}

    for cp in COST_PENALTIES:
        logger.info("\n=== cost_penalty=%.2f ===", cp)
        cp_key = f"{cp:.2f}"
        cp_results: Dict[str, List[Dict[str, Any]]] = {}

        for cond_name, use_priors in [("warmup", True), ("tabula_rasa", False)]:
            neff_grid = N_EFF_GRID_WARMUP if use_priors else [1.0]
            entries: List[Dict[str, Any]] = []

            for n_eff in neff_grid:
                for alpha in ALPHA_GRID:
                    seed_rewards = []
                    for s in range(N_SEEDS):
                        r = _simulate_val_reward(
                            train, val, registry, feature_dim,
                            alpha=alpha,
                            prior_n_effective=n_eff,
                            cost_penalty=cp,
                            warmup_path=warmup_path,
                            use_priors=use_priors,
                            seed=SEED_OFFSET + s,
                        )
                        seed_rewards.append(r)
                    mean_r = float(np.mean(seed_rewards))
                    se_r = float(np.std(seed_rewards, ddof=1) / np.sqrt(N_SEEDS))
                    entries.append({
                        "alpha": alpha, "n_eff": n_eff,
                        "mean_val_reward": mean_r, "se_val_reward": se_r,
                    })

            best = max(entries, key=lambda e: e["mean_val_reward"])
            cp_results[cond_name] = entries
            best_hparams.setdefault(cp_key, {})[cond_name] = {
                "alpha": best["alpha"],
                "n_eff": best["n_eff"],
                "val_reward": best["mean_val_reward"],
                "val_se": best["se_val_reward"],
            }
            logger.info(
                "  [%s] cp=%.2f BEST: alpha=%.2f n_eff=%.0f val=%.4f",
                cond_name, cp, best["alpha"], best["n_eff"], best["mean_val_reward"],
            )

        all_results[cp_key] = cp_results

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MULTI-CP ALPHA TUNING SUMMARY")
    print("=" * 80)
    print(f"  {'cp':>5s}  {'cond':>12s}  {'alpha':>6s}  {'n_eff':>6s}  {'val_reward':>12s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*12}")
    for cp in COST_PENALTIES:
        cp_key = f"{cp:.2f}"
        for cond in ["warmup", "tabula_rasa"]:
            b = best_hparams[cp_key][cond]
            print(
                f"  {cp:5.2f}  {cond:>12s}  {b['alpha']:6.2f}  {b['n_eff']:6.0f}  "
                f"{b['val_reward']:.4f} ± {b['val_se']:.4f}"
            )

    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed:.1f}s")
    print("=" * 80)

    out_path = Path(__file__).parent / "results" / "alpha_tuning_multi_cp_neff5000.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_seeds": N_SEEDS,
            "cost_penalties": COST_PENALTIES,
            "best_hparams": best_hparams,
            "full_results": all_results,
        }, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
