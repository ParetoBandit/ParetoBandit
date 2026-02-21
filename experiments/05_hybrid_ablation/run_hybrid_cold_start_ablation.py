#!/usr/bin/env python3
"""
Hybrid vs Disjoint LinUCB: Cold-Start Transfer Ablation
========================================================

Measures the cold-start benefit of family-shared parameters when a new
model joins an existing family.

Protocol
--------
1. Train both policies on a set of "established" models using dev data.
2. Add a new model from the SAME family (e.g. GPT-5.2 after training on
   GPT-5.1) and a new model from a DIFFERENT family.
3. Measure the **first-N-prompt reward** for the newly added models.

The hybrid policy should show higher early reward for the same-family
model because its shared beta already captures the family's preferences.

Usage
-----
    python experiments/05_hybrid_ablation/run_hybrid_cold_start_ablation.py

Output is written to experiments/05_hybrid_ablation/results/ as JSON.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.router import (
    DisjointLinUCBPolicy,
    HybridLinUCBPolicy,
    infer_model_family,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---- Experiment parameters ----
DIM = 33
ALPHA = 0.1
INIT_LAMBDA = 1.0
N_PRETRAIN = 500          # observations per established arm during training
N_EVAL = 200              # observations for newly added arm during evaluation
N_SEEDS = 20
SEED_OFFSET = 42

# Model setup: two families, two established + two newcomers
ESTABLISHED = ["openai/gpt-5.1", "anthropic/claude-3-sonnet"]
NEWCOMER_SAME_FAMILY = "openai/gpt-5.2"       # same family as gpt-5.1
NEWCOMER_DIFF_FAMILY = "google/gemini-2-flash"  # different family


def _make_context(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0)


def _true_reward(model: str, x: np.ndarray, rng: np.random.Generator) -> float:
    """Synthetic reward function per model.  Models in the same family
    share a large common component with small per-model noise."""
    family = infer_model_family(model)

    # Family-level reward (shared signal)
    family_seeds = {"openai/gpt-5": 100, "anthropic/claude-3": 200, "google/gemini-2": 300}
    fam_rng = np.random.default_rng(family_seeds.get(family, hash(family) % 10000))
    theta_family = fam_rng.standard_normal(DIM) * 0.3

    # Arm-specific perturbation (small)
    arm_rng = np.random.default_rng(hash(model) % 10000)
    theta_arm = arm_rng.standard_normal(DIM) * 0.05

    theta = theta_family + theta_arm
    mean = float(x @ theta)
    return float(np.clip(mean + rng.normal(0, 0.1), 0.0, 1.0))


def run_trial(seed: int) -> Dict:
    """Run a single trial comparing Disjoint vs Hybrid."""
    rng = np.random.default_rng(seed)

    family_map = {m: infer_model_family(m) for m in ESTABLISHED}
    disjoint = DisjointLinUCBPolicy(ESTABLISHED, dim=DIM, alpha=ALPHA, init_lambda=INIT_LAMBDA)
    hybrid = HybridLinUCBPolicy(ESTABLISHED, dim=DIM, alpha=ALPHA, init_lambda=INIT_LAMBDA,
                                 family_map=family_map)

    # ---- Phase 1: Pre-train on established models ----
    for _ in range(N_PRETRAIN):
        x = _make_context(rng)
        for model in ESTABLISHED:
            r = _true_reward(model, x, rng)
            disjoint.update(model, x, r)
            hybrid.update(model, x, r)

    # ---- Phase 2: Add newcomers ----
    for newcomer in [NEWCOMER_SAME_FAMILY, NEWCOMER_DIFF_FAMILY]:
        disjoint.add_arm(newcomer)
        fam = infer_model_family(newcomer)
        hybrid.add_arm(newcomer, family=fam)

    # ---- Phase 3: Evaluate newcomers (first-N-prompt reward) ----
    results: Dict[str, Dict[str, List[float]]] = {
        "disjoint": {NEWCOMER_SAME_FAMILY: [], NEWCOMER_DIFF_FAMILY: []},
        "hybrid": {NEWCOMER_SAME_FAMILY: [], NEWCOMER_DIFF_FAMILY: []},
    }

    for i in range(N_EVAL):
        x = _make_context(rng)
        for newcomer in [NEWCOMER_SAME_FAMILY, NEWCOMER_DIFF_FAMILY]:
            r = _true_reward(newcomer, x, rng)

            # Record the *predicted mean* before updating (measures prior quality)
            theta_d = disjoint.A_inv[newcomer] @ disjoint.b[newcomer]
            pred_d = float(x @ theta_d)

            F = hybrid.family_map.get(newcomer, newcomer)
            beta_h = hybrid.A0_inv[F] @ hybrid.b0[F] if F in hybrid.A0_inv else np.zeros(DIM)
            theta_h = hybrid.A_inv[newcomer] @ hybrid.b[newcomer]
            pred_h = float(x @ (beta_h + theta_h))

            results["disjoint"][newcomer].append(abs(r - pred_d))
            results["hybrid"][newcomer].append(abs(r - pred_h))

            disjoint.update(newcomer, x, r)
            hybrid.update(newcomer, x, r)

    return results


def main():
    logger.info("=" * 70)
    logger.info("HYBRID vs DISJOINT LinUCB: Cold-Start Transfer Ablation")
    logger.info("=" * 70)
    logger.info(
        f"\nEstablished: {ESTABLISHED}"
        f"\nSame-family newcomer: {NEWCOMER_SAME_FAMILY}"
        f"\nDiff-family newcomer: {NEWCOMER_DIFF_FAMILY}"
        f"\nPretrain={N_PRETRAIN}, Eval={N_EVAL}, Seeds={N_SEEDS}"
    )

    all_results: Dict[str, Dict[str, List[List[float]]]] = {
        "disjoint": {NEWCOMER_SAME_FAMILY: [], NEWCOMER_DIFF_FAMILY: []},
        "hybrid": {NEWCOMER_SAME_FAMILY: [], NEWCOMER_DIFF_FAMILY: []},
    }

    for trial in range(N_SEEDS):
        seed = SEED_OFFSET + trial
        r = run_trial(seed)
        for policy in ["disjoint", "hybrid"]:
            for newcomer in [NEWCOMER_SAME_FAMILY, NEWCOMER_DIFF_FAMILY]:
                all_results[policy][newcomer].append(r[policy][newcomer])

    # ---- Aggregate ----
    logger.info("\n--- Results: Mean Absolute Prediction Error (first 200 prompts) ---\n")
    summary = {}
    for newcomer in [NEWCOMER_SAME_FAMILY, NEWCOMER_DIFF_FAMILY]:
        family_label = "same-family" if newcomer == NEWCOMER_SAME_FAMILY else "diff-family"
        summary[newcomer] = {}
        for policy in ["disjoint", "hybrid"]:
            # Average across seeds, then compute mean/std
            per_seed_means = [np.mean(trial) for trial in all_results[policy][newcomer]]
            avg = np.mean(per_seed_means)
            std = np.std(per_seed_means, ddof=1) if N_SEEDS > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(N_SEEDS)

            # Early reward (first 20 prompts)
            early_means = [np.mean(trial[:20]) for trial in all_results[policy][newcomer]]
            early_avg = np.mean(early_means)
            early_ci = 1.96 * np.std(early_means, ddof=1) / np.sqrt(N_SEEDS) if N_SEEDS > 1 else 0.0

            summary[newcomer][policy] = {
                "mean_abs_error": round(float(avg), 4),
                "ci95": round(float(ci95), 4),
                "early_20_error": round(float(early_avg), 4),
                "early_20_ci95": round(float(early_ci), 4),
            }

            logger.info(
                f"  {policy:10s} | {family_label:12s} | "
                f"Full MAE={avg:.4f}+/-{ci95:.4f}  "
                f"Early-20 MAE={early_avg:.4f}+/-{early_ci:.4f}"
            )

    # ---- Save ----
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hybrid_cold_start_ablation.json"
    with open(out_path, "w") as f:
        json.dump(
            {"params": {"dim": DIM, "alpha": ALPHA, "n_pretrain": N_PRETRAIN,
                        "n_eval": N_EVAL, "n_seeds": N_SEEDS},
             "summary": summary},
            f, indent=2,
        )
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
