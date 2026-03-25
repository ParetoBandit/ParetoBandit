#!/usr/bin/env python3
"""Cross-judge regret comparison: cold-start LinUCB under R1 vs GPT-4.1-mini.

Runs the bandit on the 2K judge-robustness subset (stratified val/test split)
with **no warmup priors** (Tabula Rasa only) to avoid any data-leakage
concern.  This is deliberately the harder test: if the bandit converges
under cold-start, warm-start robustness follows.

Protocol (mirrors Exp 01):
    1. Burn-in on val split (online learning, no metrics).
    2. Evaluate on test split (cumulative regret from step 1).

Conditions:
    - 2 judges (R1, GPT-4.1-mini)
    - 4 budget regimes (unconstrained + tight / moderate / loose)
    - 2 methods per condition (Tabula Rasa, Random)
    - 20 seeds each

Outputs ``results/cross_judge_regret_results.json`` consumed by
``generate_cross_judge_figure.py``.

Usage
-----
    python experiments/appendix/judge_robustness/run_cross_judge_regret.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import random as stdlib_random

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.budget_pacer import BudgetPacer, PacingMode
from pareto_bandit.config import (
    BEST_K3_TABULA_RASA_HPARAMS,
    DEFAULT_PACER_LAMBDA_MAX,
    DEFAULT_PACER_LR,
    K3_ARM_ORDER,
    K3_BUDGET_LABELS,
    K3_BUDGET_TARGETS,
    N_SEEDS,
)
from pareto_bandit.feature_service import FeatureService
from pareto_bandit.router import BanditRouter
from pareto_bandit.storage import EphemeralContextStore
from utils.simulation import SplitData, build_model_registry, load_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ======================================================================
# Constants
# ======================================================================

ARM_ORDER: List[str] = K3_ARM_ORDER
SEED_OFFSET: int = 11000
TABULA_RASA_HPARAMS: Dict[str, Any] = BEST_K3_TABULA_RASA_HPARAMS

RESULTS_DIR = Path(__file__).resolve().parent / "results"

JUDGES: Dict[str, Dict[str, Path]] = {
    "R1": {
        "val": RESULTS_DIR / "cross_judge_r1_val.jsonl",
        "test": RESULTS_DIR / "cross_judge_r1_test.jsonl",
    },
    "GPT-4.1-mini": {
        "val": RESULTS_DIR / "cross_judge_gpt_mini_val.jsonl",
        "test": RESULTS_DIR / "cross_judge_gpt_mini_test.jsonl",
    },
}

BUDGET_REGIMES: List[Dict[str, Any]] = [
    {"label": "unconstrained", "target": None},
] + [
    {"label": lbl, "target": tgt}
    for lbl, tgt in zip(K3_BUDGET_LABELS, K3_BUDGET_TARGETS)
]


# ======================================================================
# Dataclasses
# ======================================================================


@dataclass
class TrialResult:
    """Aggregate metrics for one (judge, budget, method, seed) trial."""

    judge: str
    budget_label: str
    method: str
    seed: int
    mean_reward: float
    mean_cost: float
    cumulative_regret: float
    n_test: int
    per_step_regret: List[float] = field(default_factory=list)
    model_fractions: Dict[str, float] = field(default_factory=dict)


# ======================================================================
# Router creation
# ======================================================================


def _create_tabula_rasa_router(
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    budget_pacer: Optional[BudgetPacer] = None,
) -> BanditRouter:
    """Build a cold-start K=3 router (no warmup priors)."""
    fs = FeatureService.for_precomputed(feature_dim)
    store = EphemeralContextStore()
    return BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        context_store=store,
        priors="none",
        prior_n_effective=TABULA_RASA_HPARAMS["prior_n_effective"],
        alpha=TABULA_RASA_HPARAMS["alpha"],
        cost_penalty=0.0,
        forgetting_factor=TABULA_RASA_HPARAMS["forgetting_factor"],
        budget_pacer=budget_pacer,
    )


# ======================================================================
# Trial runner
# ======================================================================


def _run_tabula_rasa_trial(
    val: SplitData,
    test: SplitData,
    registry: Dict[str, Any],
    feature_dim: int,
    *,
    judge: str,
    budget_label: str,
    budget_pacer: Optional[BudgetPacer] = None,
    seed: int,
    record_per_step: bool = False,
) -> TrialResult:
    """Run one Tabula Rasa trial: burn-in on val, evaluate on test.

    Parameters
    ----------
    val:
        Validation split for online learning (burn-in, no metrics).
    test:
        Holdout split for evaluation.
    registry:
        Model registry with pricing info.
    feature_dim:
        Embedding dimensionality.
    judge:
        Judge label (for bookkeeping).
    budget_label:
        Budget regime label.
    budget_pacer:
        Optional BudgetPacer instance (None for unconstrained).
    seed:
        Random seed for reproducibility.
    record_per_step:
        If True, store per-step cumulative regret for plotting.

    Returns
    -------
    TrialResult
        Aggregate and per-step metrics.
    """
    rng = np.random.default_rng(seed)

    if budget_pacer is not None:
        budget_pacer.reset()

    router = _create_tabula_rasa_router(
        registry, feature_dim, budget_pacer=budget_pacer,
    )
    router.bandit._rng = np.random.default_rng(seed + 1_000_000)

    # --- Burn-in on val (no metrics) ---
    val_order = rng.permutation(val.n)
    for i in val_order:
        model, log = router.route(val.embeddings[i])
        reward = float(val.rewards[model][i])
        log.cost_usd = float(val.costs[model][i])
        router.process_feedback(log.request_id, reward=reward)

    # --- Evaluate on test ---
    test_order = rng.permutation(test.n)
    cumulative_regret = 0.0
    per_step_regret: List[float] = []
    model_counts: Dict[str, int] = {m: 0 for m in ARM_ORDER}
    total_reward = 0.0
    total_cost = 0.0

    for i in test_order:
        model, log = router.route(test.embeddings[i])
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        oracle = max(float(test.rewards[a][i]) for a in ARM_ORDER)

        log.cost_usd = cost
        router.process_feedback(log.request_id, reward=reward)

        step_regret = oracle - reward
        cumulative_regret += step_regret
        total_reward += reward
        total_cost += cost
        model_counts[model] += 1

        if record_per_step:
            per_step_regret.append(cumulative_regret)

    n = test.n
    model_fractions = {m: cnt / n for m, cnt in model_counts.items()}

    return TrialResult(
        judge=judge,
        budget_label=budget_label,
        method="tabula_rasa",
        seed=seed,
        mean_reward=total_reward / n,
        mean_cost=total_cost / n,
        cumulative_regret=cumulative_regret,
        n_test=n,
        per_step_regret=per_step_regret,
        model_fractions=model_fractions,
    )


def _run_random_trial(
    test: SplitData,
    *,
    judge: str,
    budget_label: str,
    seed: int,
    record_per_step: bool = False,
) -> TrialResult:
    """Run one Random baseline trial (uniform 1/K, no learning).

    Follows the same pattern as Exp 01: only model selection is randomised;
    prompts are iterated in fixed file order so that the only RNG
    consumption is ``rng.choice``.  This makes the baseline fully
    deterministic given (seed, test data).

    Parameters
    ----------
    test:
        Test split to evaluate on.
    judge:
        Judge label.
    budget_label:
        Budget regime label.
    seed:
        Random seed.
    record_per_step:
        Store per-step cumulative regret.

    Returns
    -------
    TrialResult
        Aggregate and per-step metrics.
    """
    rng = np.random.default_rng(seed)
    n = test.n
    choices = rng.choice(ARM_ORDER, size=n)

    cumulative_regret = 0.0
    per_step_regret: List[float] = []
    total_reward = 0.0
    total_cost = 0.0

    for i in range(n):
        model = choices[i]
        reward = float(test.rewards[model][i])
        cost = float(test.costs[model][i])
        oracle = max(float(test.rewards[a][i]) for a in ARM_ORDER)

        step_regret = oracle - reward
        cumulative_regret += step_regret
        total_reward += reward
        total_cost += cost

        if record_per_step:
            per_step_regret.append(cumulative_regret)

    return TrialResult(
        judge=judge,
        budget_label=budget_label,
        method="random",
        seed=seed,
        mean_reward=total_reward / n,
        mean_cost=total_cost / n,
        cumulative_regret=cumulative_regret,
        n_test=n,
        per_step_regret=per_step_regret,
        model_fractions={m: 1.0 / len(ARM_ORDER) for m in ARM_ORDER},
    )


# ======================================================================
# Main
# ======================================================================


EMBEDDING_SEED: int = 2026


def _seed_all(seed: int) -> None:
    """Pin all sources of randomness for deterministic embedding computation.

    Must be called before ``FeatureService`` / ``SentenceTransformer``
    encoding so that MPS/CUDA non-determinism is eliminated.
    """
    stdlib_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Cross-Judge Regret Comparison (Cold-Start)")
    logger.info("=" * 70)

    _seed_all(EMBEDDING_SEED)

    registry = build_model_registry(ARM_ORDER)
    fs = FeatureService()
    feature_dim = fs.dimension

    logger.info("Models: %s", ARM_ORDER)
    logger.info("Seeds: %d  Offset: %d", N_SEEDS, SEED_OFFSET)
    logger.info("Budget regimes: %s", [b["label"] for b in BUDGET_REGIMES])
    logger.info("Hparams: %s", TABULA_RASA_HPARAMS)

    # Both judges share the same prompts (and thus the same embeddings);
    # only the rewards differ.  load_split is called once per file;
    # GPT-mini splits reuse R1's embeddings since prompt order matches.
    logger.info("Loading R1 splits ...")
    r1_val = load_split(JUDGES["R1"]["val"], fs, ARM_ORDER)
    r1_test = load_split(JUDGES["R1"]["test"], fs, ARM_ORDER)
    logger.info("  R1: val=%d  test=%d", r1_val.n, r1_test.n)

    logger.info("Loading GPT-4.1-mini splits (reusing R1 embeddings) ...")
    gpt_val = load_split(JUDGES["GPT-4.1-mini"]["val"], fs, ARM_ORDER)
    gpt_test = load_split(JUDGES["GPT-4.1-mini"]["test"], fs, ARM_ORDER)

    assert gpt_val.prompts == r1_val.prompts, (
        "Prompt order mismatch in val — re-run build_cross_judge_splits.py"
    )
    assert gpt_test.prompts == r1_test.prompts, (
        "Prompt order mismatch in test — re-run build_cross_judge_splits.py"
    )

    gpt_val = SplitData(
        prompts=gpt_val.prompts, rewards=gpt_val.rewards,
        costs=gpt_val.costs, embeddings=r1_val.embeddings,
    )
    gpt_test = SplitData(
        prompts=gpt_test.prompts, rewards=gpt_test.rewards,
        costs=gpt_test.costs, embeddings=r1_test.embeddings,
    )
    logger.info("  GPT-4.1-mini: val=%d  test=%d", gpt_val.n, gpt_test.n)

    splits: Dict[str, Dict[str, SplitData]] = {
        "R1": {"val": r1_val, "test": r1_test},
        "GPT-4.1-mini": {"val": gpt_val, "test": gpt_test},
    }

    all_trials: List[TrialResult] = []

    # Pre-compute Random baselines once per (judge, seed) — results are
    # independent of budget regime since Random has no budget pacer.
    random_trials: Dict[str, List[TrialResult]] = {}
    for judge_name in JUDGES:
        test = splits[judge_name]["test"]
        judge_random: List[TrialResult] = []
        for s in range(N_SEEDS):
            seed = SEED_OFFSET + s
            rnd = _run_random_trial(
                test,
                judge=judge_name,
                budget_label="unconstrained",
                seed=seed,
                record_per_step=True,
            )
            judge_random.append(rnd)
        random_trials[judge_name] = judge_random
        rnd_regrets = [t.cumulative_regret for t in judge_random]
        logger.info(
            "  Random [%s]: regret=%.2f±%.2f",
            judge_name,
            np.mean(rnd_regrets),
            np.std(rnd_regrets, ddof=1) / np.sqrt(len(rnd_regrets)),
        )

    for judge_name in JUDGES:
        val = splits[judge_name]["val"]
        test = splits[judge_name]["test"]

        for regime in BUDGET_REGIMES:
            budget_label = regime["label"]
            budget_target = regime["target"]
            is_unconstrained = budget_target is None

            logger.info(
                "\n--- Judge=%s  Budget=%s (target=%s) ---",
                judge_name, budget_label,
                f"${budget_target:.6f}" if budget_target else "None",
            )

            tr_regrets: List[float] = []
            for s in range(N_SEEDS):
                seed = SEED_OFFSET + s

                pacer: Optional[BudgetPacer] = None
                if budget_target is not None:
                    pacer = BudgetPacer(
                        target_avg_spend_usd=budget_target,
                        mode=PacingMode.ADAPTIVE,
                        lr=DEFAULT_PACER_LR,
                        lambda_max=DEFAULT_PACER_LAMBDA_MAX,
                    )

                tr = _run_tabula_rasa_trial(
                    val, test, registry, feature_dim,
                    judge=judge_name,
                    budget_label=budget_label,
                    budget_pacer=pacer,
                    seed=seed,
                    record_per_step=is_unconstrained,
                )
                all_trials.append(tr)
                tr_regrets.append(tr.cumulative_regret)

            # Attach Random trials (reuse pre-computed, tag with this budget)
            for rnd in random_trials[judge_name]:
                tagged = TrialResult(
                    judge=rnd.judge,
                    budget_label=budget_label,
                    method=rnd.method,
                    seed=rnd.seed,
                    mean_reward=rnd.mean_reward,
                    mean_cost=rnd.mean_cost,
                    cumulative_regret=rnd.cumulative_regret,
                    n_test=rnd.n_test,
                    per_step_regret=rnd.per_step_regret if is_unconstrained else [],
                    model_fractions=rnd.model_fractions,
                )
                all_trials.append(tagged)

            rnd_regrets = [t.cumulative_regret for t in random_trials[judge_name]]
            logger.info(
                "  TabRasa: regret=%.2f±%.2f  Random: regret=%.2f±%.2f",
                np.mean(tr_regrets),
                np.std(tr_regrets, ddof=1) / np.sqrt(len(tr_regrets)),
                np.mean(rnd_regrets),
                np.std(rnd_regrets, ddof=1) / np.sqrt(len(rnd_regrets)),
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_judge_regret_results.json"

    def _trial_to_dict(t: TrialResult) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "judge": t.judge,
            "budget_label": t.budget_label,
            "method": t.method,
            "seed": t.seed,
            "mean_reward": t.mean_reward,
            "mean_cost": t.mean_cost,
            "cumulative_regret": t.cumulative_regret,
            "n_test": t.n_test,
            "model_fractions": t.model_fractions,
        }
        if t.per_step_regret:
            d["per_step_regret"] = t.per_step_regret
        return d

    output = {
        "experiment": "cross_judge_regret",
        "arm_order": ARM_ORDER,
        "n_seeds": N_SEEDS,
        "seed_offset": SEED_OFFSET,
        "tabula_rasa_hparams": TABULA_RASA_HPARAMS,
        "budget_regimes": BUDGET_REGIMES,
        "judges": list(JUDGES.keys()),
        "trials": [_trial_to_dict(t) for t in all_trials],
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("\nSaved %d trials to %s", len(all_trials), out_path)
    logger.info("Wall time: %.1fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
