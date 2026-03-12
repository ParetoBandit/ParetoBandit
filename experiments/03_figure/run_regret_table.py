#!/usr/bin/env python3
"""Regret reduction table across cost_penalties with per-cp tuned alphas.

Loads the best hparams from ``tune_alpha_multi_cp.py``, runs both warmup
and tabula rasa conditions (20 seeds each) at each cost_penalty on the
held-out test split, and produces a regret reduction table.

Usage:
    python experiments/03_figure/run_regret_table.py
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
N_SEEDS = 20
SEED_OFFSET = 1000
RESULTS_DIR = Path(__file__).parent / "results"


def _simulate_regret(
    train: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    alpha: float,
    prior_n_effective: float,
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
        forgetting_factor=1.0,
        policy="disjoint",
    )

    for i in rng.permutation(train.n):
        model, log = router.route(train.embeddings[i])
        reward = float(train.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)

    cum_reward = 0.0
    cum_oracle = 0.0
    for i in rng.permutation(test.n):
        model, log = router.route(test.embeddings[i])
        reward = float(test.rewards[model][i])
        router.process_feedback(log.request_id, reward=reward)
        cum_reward += reward
        cum_oracle += max(float(test.rewards[a][i]) for a in ARM_ORDER)

    return {
        "test_reward": cum_reward / test.n,
        "cumulative_regret": cum_oracle - cum_reward,
    }


def main() -> None:
    t0 = time.time()

    tuning_path = RESULTS_DIR / "alpha_tuning_multi_cp.json"
    if not tuning_path.exists():
        raise FileNotFoundError(
            f"Run tune_alpha_multi_cp.py first: {tuning_path}"
        )
    with open(tuning_path) as f:
        tuning = json.load(f)
    best_hparams = tuning["best_hparams"]
    cost_penalties = tuning["cost_penalties"]

    logger.info("Loading data ...")
    fs = FeatureService()
    feature_dim = fs.dimension
    train = load_split(TRAIN_DATA_PATH, fs, ARM_ORDER)
    test = load_split(HOLDOUT_DATA_PATH, fs, ARM_ORDER)
    registry = build_model_registry(ARM_ORDER)
    warmup_path = str(K2_WARMUP_PRIORS_PATH)
    logger.info("  Train=%d  Test=%d  dim=%d", train.n, test.n, feature_dim)

    table_rows: List[Dict[str, Any]] = []

    for cp in cost_penalties:
        cp_key = f"{cp:.2f}"
        for cond_name, use_priors in [("warmup", True), ("tabula_rasa", False)]:
            hp = best_hparams[cp_key][cond_name]
            seed_results = []
            for s in range(N_SEEDS):
                r = _simulate_regret(
                    train, test, registry, feature_dim,
                    alpha=hp["alpha"],
                    prior_n_effective=hp["n_eff"],
                    cost_penalty=cp,
                    warmup_path=warmup_path,
                    use_priors=use_priors,
                    seed=SEED_OFFSET + s,
                )
                seed_results.append(r)

            regrets = [r["cumulative_regret"] for r in seed_results]
            rewards = [r["test_reward"] for r in seed_results]
            table_rows.append({
                "cost_penalty": cp,
                "condition": cond_name,
                "alpha": hp["alpha"],
                "n_eff": hp["n_eff"],
                "mean_regret": float(np.mean(regrets)),
                "se_regret": float(np.std(regrets, ddof=1) / np.sqrt(N_SEEDS)),
                "mean_reward": float(np.mean(rewards)),
                "se_reward": float(np.std(rewards, ddof=1) / np.sqrt(N_SEEDS)),
            })
            logger.info(
                "  cp=%.2f %s (a=%.2f n=%.0f) regret=%.1f±%.1f reward=%.4f",
                cp, cond_name, hp["alpha"], hp["n_eff"],
                table_rows[-1]["mean_regret"], table_rows[-1]["se_regret"],
                table_rows[-1]["mean_reward"],
            )

    # ── Print table ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"REGRET REDUCTION TABLE (K=2, {N_SEEDS} seeds, per-cp tuned alphas)")
    print("=" * 90)
    print(
        f"  {'cp':>5s}  {'W_alpha':>7s}  {'W_regret':>12s}  "
        f"{'TR_alpha':>8s}  {'TR_regret':>12s}  {'reduction':>10s}  "
        f"{'W_reward':>10s}  {'TR_reward':>10s}"
    )
    print(f"  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for cp in cost_penalties:
        w = next(r for r in table_rows if r["cost_penalty"] == cp and r["condition"] == "warmup")
        t = next(r for r in table_rows if r["cost_penalty"] == cp and r["condition"] == "tabula_rasa")
        reduction = (1.0 - w["mean_regret"] / t["mean_regret"]) * 100 if t["mean_regret"] > 0 else 0
        print(
            f"  {cp:5.2f}  {w['alpha']:7.2f}  "
            f"{w['mean_regret']:6.1f}±{w['se_regret']:4.1f}  "
            f"{t['alpha']:8.2f}  "
            f"{t['mean_regret']:6.1f}±{t['se_regret']:4.1f}  "
            f"{reduction:9.1f}%  "
            f"{w['mean_reward']:.4f}  {t['mean_reward']:.4f}"
        )

    print("=" * 90)

    # ── LaTeX table ────────────────────────────────────────────────────
    print("\n% LaTeX table")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\toprule")
    print(
        r"$\lambda_c$ & $\alpha_W$ & Regret$_W$ & "
        r"$\alpha_{TR}$ & Regret$_{TR}$ & Reduction (\%) \\"
    )
    print(r"\midrule")
    for cp in cost_penalties:
        w = next(r for r in table_rows if r["cost_penalty"] == cp and r["condition"] == "warmup")
        t = next(r for r in table_rows if r["cost_penalty"] == cp and r["condition"] == "tabula_rasa")
        reduction = (1.0 - w["mean_regret"] / t["mean_regret"]) * 100 if t["mean_regret"] > 0 else 0
        print(
            f"  {cp:.2f} & {w['alpha']:.2f} & "
            f"${w['mean_regret']:.1f} \\pm {w['se_regret']:.1f}$ & "
            f"{t['alpha']:.2f} & "
            f"${t['mean_regret']:.1f} \\pm {t['se_regret']:.1f}$ & "
            f"{reduction:.1f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")

    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed:.1f}s")

    out_path = RESULTS_DIR / "regret_reduction_table.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_seeds": N_SEEDS,
            "cost_penalties": cost_penalties,
            "rows": table_rows,
        }, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
