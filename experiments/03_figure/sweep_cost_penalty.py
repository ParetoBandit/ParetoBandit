#!/usr/bin/env python3
"""Sweep cost_penalty to find the operating point with maximum warmup
regret reduction.

For each cost_penalty, runs both warmup and tabula rasa conditions (3
seeds each) and reports cumulative regret + regret reduction. Uses the
cp=0.20-tuned alpha values as a starting point — the relative ranking
across cost_penalties is informative even if alpha isn't re-tuned at
each point.

Usage:
    python experiments/03_figure/sweep_cost_penalty.py
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
    HOLDOUT_DATA_PATH,
    K2_ARM_ORDER,
    K2_WARMUP_PRIORS_PATH,
    TRAIN_DATA_PATH,
)
from bandit_gpt.feature_service import FeatureService
from bandit_gpt.router import BanditRouter
from bandit_gpt.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
for _noisy in ("bandit_gpt.router", "bandit_gpt.router_v2", "bandit_gpt.feature_service"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ARM_ORDER = K2_ARM_ORDER
COST_PENALTIES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
N_SEEDS = 5
SEED_OFFSET = 2000

WARMUP_HPARAMS = {
    "alpha": 0.01, "prior_n_effective": 50.0, "forgetting_factor": 1.0,
}
TABULA_RASA_HPARAMS = {
    "alpha": 0.30, "prior_n_effective": 1.0, "forgetting_factor": 1.0,
}


def _simulate(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    alpha: float,
    prior_n_effective: float,
    forgetting_factor: float,
    cost_penalty: float,
    warmup_path: str,
    use_priors: bool,
    seed: int,
) -> Dict[str, float]:
    """Run train-then-test, return test reward and cumulative regret."""
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
        forgetting_factor=forgetting_factor,
        policy="disjoint",
    )

    # Train phase
    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # Test phase
    cum_reward = 0.0
    cum_oracle = 0.0
    test_idx = rng.permutation(test.n)
    for i in test_idx:
        model, log = router.route(test.embeddings[i])
        reward = float(test.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)
        cum_reward += reward
        oracle = max(float(test.rewards[a][i]) for a in ARM_ORDER)
        cum_oracle += oracle

    return {
        "test_reward": cum_reward / test.n,
        "cumulative_regret": cum_oracle - cum_reward,
    }


def main() -> None:
    t0 = time.time()
    logger.info("Loading data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    train = load_split(TRAIN_DATA_PATH, fs, ARM_ORDER)
    test = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    logger.info("  Train=%d  Test=%d  dim=%d", train.n, test.n, feature_dim)

    rows: List[Dict[str, Any]] = []

    for cp in COST_PENALTIES:
        for cond_name, hparams, use_priors in [
            ("warmup", WARMUP_HPARAMS, True),
            ("tabula_rasa", TABULA_RASA_HPARAMS, False),
        ]:
            seed_results = []
            for s in range(N_SEEDS):
                r = _simulate(
                    train, test, registry, feature_dim,
                    alpha=hparams["alpha"],
                    prior_n_effective=hparams["prior_n_effective"],
                    forgetting_factor=hparams["forgetting_factor"],
                    cost_penalty=cp,
                    warmup_path=warmup_path,
                    use_priors=use_priors,
                    seed=SEED_OFFSET + s,
                )
                seed_results.append(r)

            mean_regret = float(np.mean([r["cumulative_regret"] for r in seed_results]))
            se_regret = float(np.std([r["cumulative_regret"] for r in seed_results], ddof=1) / np.sqrt(N_SEEDS))
            mean_reward = float(np.mean([r["test_reward"] for r in seed_results]))

            rows.append({
                "cost_penalty": cp,
                "condition": cond_name,
                "mean_regret": mean_regret,
                "se_regret": se_regret,
                "mean_reward": mean_reward,
            })
            logger.info(
                "  cp=%.2f  %s  regret=%.1f±%.1f  reward=%.4f",
                cp, cond_name, mean_regret, se_regret, mean_reward,
            )

    # Report
    print("\n" + "=" * 72)
    print(f"COST PENALTY SWEEP — Regret Reduction ({N_SEEDS} seeds)")
    print("=" * 72)
    print(f"  {'cp':>5s}  {'warmup_regret':>14s}  {'TR_regret':>14s}  {'reduction':>10s}  {'warmup_r':>10s}  {'TR_r':>10s}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")

    for cp in COST_PENALTIES:
        w = next(r for r in rows if r["cost_penalty"] == cp and r["condition"] == "warmup")
        t = next(r for r in rows if r["cost_penalty"] == cp and r["condition"] == "tabula_rasa")
        reduction = (1.0 - w["mean_regret"] / t["mean_regret"]) * 100 if t["mean_regret"] > 0 else 0
        print(
            f"  {cp:5.2f}  {w['mean_regret']:7.1f}±{w['se_regret']:4.1f}  "
            f"{t['mean_regret']:7.1f}±{t['se_regret']:4.1f}  "
            f"{reduction:9.1f}%  {w['mean_reward']:.4f}  {t['mean_reward']:.4f}"
        )

    print("=" * 72)
    elapsed = time.time() - t0
    print(f"  Wall time: {elapsed:.1f}s")

    out_path = Path(__file__).parent / "results" / "cost_penalty_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_seeds": N_SEEDS, "rows": rows}, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
