#!/usr/bin/env python3
"""Tune alpha at cost_penalty=0.20 for the warmup ablation experiment.

Sweeps the exploration parameter alpha for both conditions (warmup and
tabula rasa) at a fixed cost_penalty=0.20 on the val split, following the
PILOT (EMNLP 2025) protocol of tuning alpha to maximise reward at the
specific operating point being reported.

Protocol: train on train split, evaluate on val split, pick the alpha
that maximises mean val reward (averaged over 3 seeds).

Usage:
    python experiments/03_figure/tune_alpha_cp020.py
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
N_EFF_GRID_WARMUP = [10.0, 50.0, 200.0]
COST_PENALTY = 0.20
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

    train_idx = rng.permutation(train.n)
    for i in train_idx:
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    val_rewards = []
    val_idx = rng.permutation(val.n)
    for i in val_idx:
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

    logger.info("  Train=%d  Val=%d  feature_dim=%d", train.n, val.n, feature_dim)

    conditions = [
        ("warmup", True),
        ("tabula_rasa", False),
    ]

    results: Dict[str, List[Dict[str, Any]]] = {}

    for cond_name, use_priors in conditions:
        neff_grid = N_EFF_GRID_WARMUP if use_priors else [1.0]
        cond_results: List[Dict[str, Any]] = []

        for n_eff in neff_grid:
            for alpha in ALPHA_GRID:
                seed_rewards = []
                for s in range(N_SEEDS):
                    r = _simulate_val_reward(
                        train, val, registry, feature_dim,
                        alpha=alpha,
                        prior_n_effective=n_eff,
                        cost_penalty=COST_PENALTY,
                        warmup_path=warmup_path,
                        use_priors=use_priors,
                        seed=SEED_OFFSET + s,
                    )
                    seed_rewards.append(r)
                mean_r = float(np.mean(seed_rewards))
                se_r = float(np.std(seed_rewards, ddof=1) / np.sqrt(N_SEEDS))
                entry = {
                    "alpha": alpha,
                    "n_eff": n_eff,
                    "mean_val_reward": mean_r,
                    "se_val_reward": se_r,
                }
                cond_results.append(entry)
                logger.info(
                    "  [%s] alpha=%.2f n_eff=%.0f  val_reward=%.4f ± %.4f",
                    cond_name, alpha, n_eff, mean_r, se_r,
                )

        results[cond_name] = cond_results

    # ── Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"ALPHA TUNING AT cost_penalty={COST_PENALTY}")
    print("=" * 72)

    for cond_name in ["warmup", "tabula_rasa"]:
        cond_results = results[cond_name]
        best = max(cond_results, key=lambda x: x["mean_val_reward"])
        print(f"\n  {cond_name}:")
        print(f"    {'alpha':>6s}  {'n_eff':>6s}  {'val_reward':>12s}")
        print(f"    {'-'*6}  {'-'*6}  {'-'*12}")
        for entry in sorted(cond_results, key=lambda x: (-x["n_eff"], x["alpha"])):
            marker = " ***" if entry is best else ""
            print(
                f"    {entry['alpha']:6.2f}  {entry['n_eff']:6.0f}  "
                f"{entry['mean_val_reward']:.4f} ± {entry['se_val_reward']:.4f}{marker}"
            )
        print(f"\n    BEST: alpha={best['alpha']}, n_eff={best['n_eff']}, "
              f"val_reward={best['mean_val_reward']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed:.1f}s")
    print("=" * 72)

    out_path = Path(__file__).parent / "results" / "alpha_tuning_cp020.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"cost_penalty": COST_PENALTY, "n_seeds": N_SEEDS, "results": results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
