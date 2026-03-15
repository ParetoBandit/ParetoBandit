#!/usr/bin/env python3
"""Diagnostic: would the paper's conclusions change with a different judge?

Computes the quantities a KDD reviewer would demand:

1. Expected reward ordering — do all judges agree on E[r_A] > E[r_B] for
   every model pair?  If yes, the bandit converges to the same policy.
2. Per-prompt oracle regret under each judge — the theoretical ceiling
   the bandit aims for.
3. Effect-size ratio — how large are the paper's reported effects relative
   to judge-induced noise?
4. Simulated routing swap — if the bandit trained on R1 but was evaluated
   by another judge, how much reward would it lose?

Usage
-----
    python experiments_v2/appendix/judge_robustness/diagnostic_conclusion_sensitivity.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CALIBRATION_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "calibration"
PARETO_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
SUBSET_PROMPTS_PATH = CALIBRATION_DIR / "judge_robustness_prompts.jsonl"
SUPPLEMENTARY_REWARDS_PATH = CALIBRATION_DIR / "judge_robustness_rewards.jsonl"

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]
MODEL_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
}


def load_all_scores() -> Dict[str, Dict[Tuple[str, str], float]]:
    """Load R1 + supplementary scores keyed by judge name."""
    prompts: Set[str] = set()
    with open(SUBSET_PROMPTS_PATH) as f:
        for line in f:
            prompts.add(json.loads(line)["prompt"])

    r1: Dict[Tuple[str, str], float] = {}
    with open(PARETO_REWARDS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok") or rec["prompt"] not in prompts:
                continue
            r1[(rec["prompt"], rec["model_id"])] = rec["raw_score"]

    supp: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
    with open(SUPPLEMENTARY_REWARDS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            key = (rec["prompt"], rec["model_id"])
            for jd in rec.get("judge_details", []):
                if "gpt-4.1-mini" in jd["judge"]:
                    supp["GPT-4.1-mini"][key] = jd["reward"]
                elif "claude-3.7-sonnet" in jd["judge"]:
                    supp["Claude-3.7-Sonnet"][key] = jd["reward"]

    return {"R1": r1, **dict(supp)}


def build_prompt_matrices(
    all_scores: Dict[str, Dict[Tuple[str, str], float]],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build {judge: [n_prompts × n_models]} matrices on common keys."""
    all_keys = set.intersection(
        *[set(s.keys()) for s in all_scores.values()]
    )
    prompts_with_all = set()
    for key in all_keys:
        prompt, _ = key
        if all((prompt, m) in all_keys for m in MODELS):
            prompts_with_all.add(prompt)

    prompt_list = sorted(prompts_with_all)
    matrices: Dict[str, np.ndarray] = {}
    for judge, scores in all_scores.items():
        matrices[judge] = np.array([
            [scores[(p, m)] for m in MODELS]
            for p in prompt_list
        ])
    return prompt_list, matrices


def analyze() -> None:
    """Run conclusion-sensitivity diagnostics."""
    all_scores = load_all_scores()
    _, matrices = build_prompt_matrices(all_scores)
    n = len(next(iter(matrices.values())))
    judges = list(matrices.keys())

    print(f"n = {n} prompts, K = {len(MODELS)} models, {len(judges)} judges")
    print()

    # ── 1. Expected reward ordering ──────────────────────────────────────
    print("=" * 70)
    print("1. EXPECTED REWARD ORDERING")
    print("   Do all judges agree on E[r_A] > E[r_B] for every model pair?")
    print("   If yes → bandit converges to the same policy regardless of judge.")
    print("=" * 70)
    for j in judges:
        mat = matrices[j]
        means = mat.mean(axis=0)
        ranking = np.argsort(-means)
        print(f"  {j:20s}: ", end="")
        for idx in ranking:
            print(f"{MODEL_SHORT[MODELS[idx]]}={means[idx]:.4f}  ", end="")
        print()

    print()
    print("  Pairwise expected reward sign (A > B under each judge):")
    for i, mi in enumerate(MODELS):
        for k, mk in enumerate(MODELS):
            if i >= k:
                continue
            signs = {}
            for j in judges:
                mat = matrices[j]
                diff = mat[:, i].mean() - mat[:, k].mean()
                signs[j] = ">" if diff > 0 else "<"
                pval = sp_stats.ttest_rel(mat[:, i], mat[:, k]).pvalue
                print(f"    {MODEL_SHORT[mi]:14s} vs {MODEL_SHORT[mk]:14s} "
                      f"under {j:20s}: diff={diff:+.4f}  "
                      f"p={pval:.2e}  sign={signs[j]}")
            all_same = len(set(signs.values())) == 1
            print(f"    → All judges agree: {'YES' if all_same else 'NO'}")
            print()

    # ── 2. Per-prompt oracle reward under each judge ─────────────────────
    print("=" * 70)
    print("2. ORACLE PERFORMANCE UNDER EACH JUDGE")
    print("   What reward does the oracle (best-model) policy achieve?")
    print("=" * 70)
    for j in judges:
        mat = matrices[j]
        oracle_reward = np.max(mat, axis=1).mean()
        random_reward = mat.mean()
        best_fixed = max(mat[:, i].mean() for i in range(len(MODELS)))
        print(f"  {j:20s}: oracle={oracle_reward:.4f}  "
              f"best_fixed={best_fixed:.4f}  "
              f"random={random_reward:.4f}  "
              f"oracle_lift={oracle_reward - best_fixed:.4f}")
    print()

    # ── 3. Cross-judge oracle evaluation (expanded) ──────────────────────
    print("=" * 70)
    print("3. CROSS-JUDGE ORACLE: FOLLOW JUDGE A's ROUTING, EVAL BY JUDGE B")
    print("   The reward you actually get (under eval judge) by trusting")
    print("   the oracle of the training judge.")
    print("=" * 70)
    # Compute absolute rewards, not just regret
    for oracle_j in judges:
        oracle_picks = np.argmax(matrices[oracle_j], axis=1)
        for eval_j in judges:
            eval_mat = matrices[eval_j]
            achieved = eval_mat[np.arange(n), oracle_picks].mean()
            best_possible = np.max(eval_mat, axis=1).mean()
            regret = best_possible - achieved
            frac_of_oracle = achieved / best_possible
            print(f"  train={oracle_j:20s} eval={eval_j:20s}: "
                  f"reward={achieved:.4f}  "
                  f"oracle={best_possible:.4f}  "
                  f"regret={regret:.4f}  "
                  f"frac_oracle={frac_of_oracle:.3f}")
        print()

    # ── 4. Effect size vs judge noise ────────────────────────────────────
    print("=" * 70)
    print("4. EFFECT SIZE vs JUDGE NOISE")
    print("   Judge noise = std of (judge_A - judge_B) on same responses")
    print("   This bounds how much 'wobble' judge choice introduces.")
    print("=" * 70)
    for j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
        mat_r1 = matrices["R1"]
        mat_j = matrices[j]
        diffs = mat_j - mat_r1  # per-response
        print(f"  R1 vs {j}:")
        print(f"    Mean diff:     {np.mean(diffs):+.4f}")
        print(f"    Std diff:      {np.std(diffs):.4f}")
        print(f"    |diff| > 0.10: {np.mean(np.abs(diffs) > 0.10):.1%}")
        print(f"    |diff| > 0.20: {np.mean(np.abs(diffs) > 0.20):.1%}")

        # Per-prompt oracle regret difference
        r1_oracle = np.max(mat_r1, axis=1) - mat_r1[
            np.arange(n), np.argmax(mat_r1, axis=1)
        ]
        j_oracle = np.max(mat_r1, axis=1) - mat_r1[
            np.arange(n), np.argmax(mat_j, axis=1)
        ]
        regret_from_wrong_judge = j_oracle.mean()
        print(f"    Mean regret from following {j}'s oracle "
              f"(eval by R1): {regret_from_wrong_judge:.4f}")
        print()

    # ── 5. Bootstrap: confidence interval on cross-regret ────────────────
    print("=" * 70)
    print("5. BOOTSTRAP 95% CI ON CROSS-REGRET")
    print("   How precisely can we bound the cost of judge choice?")
    print("=" * 70)
    rng = np.random.default_rng(42)
    n_boot = 10_000
    for oracle_j in judges:
        for eval_j in judges:
            if oracle_j == eval_j:
                continue
            oracle_picks = np.argmax(matrices[oracle_j], axis=1)
            eval_mat = matrices[eval_j]
            achieved = eval_mat[np.arange(n), oracle_picks]
            best_possible = np.max(eval_mat, axis=1)
            per_prompt_regret = best_possible - achieved

            boot_means = np.array([
                per_prompt_regret[rng.integers(0, n, size=n)].mean()
                for _ in range(n_boot)
            ])
            lo, hi = np.percentile(boot_means, [2.5, 97.5])
            print(f"  oracle={oracle_j:20s} eval={eval_j:20s}: "
                  f"mean={per_prompt_regret.mean():.4f}  "
                  f"95% CI=[{lo:.4f}, {hi:.4f}]")
    print()

    # ── 6. Summary for paper ─────────────────────────────────────────────
    print("=" * 70)
    print("6. KEY NUMBERS FOR THE PAPER")
    print("=" * 70)

    # All judges agree on expected reward ordering?
    orderings = []
    for j in judges:
        means = matrices[j].mean(axis=0)
        orderings.append(tuple(np.argsort(-means)))
    all_same_order = len(set(orderings)) == 1
    print(f"  All judges agree on E[reward] ranking: {all_same_order}")
    print(f"  Rankings: {dict(zip(judges, orderings))}")

    # Cross-regret summary
    r1_oracle = np.argmax(matrices["R1"], axis=1)
    for eval_j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
        eval_mat = matrices[eval_j]
        achieved = eval_mat[np.arange(n), r1_oracle].mean()
        oracle_possible = np.max(eval_mat, axis=1).mean()
        print(f"  R1-oracle reward under {eval_j}: {achieved:.4f} "
              f"(oracle: {oracle_possible:.4f}, "
              f"captures {achieved/oracle_possible:.1%})")

    # Population routing distribution L1 distance
    for j in judges:
        picks = np.argmax(matrices[j], axis=1)
        freqs = np.array([np.mean(picks == i) for i in range(len(MODELS))])
        r1_picks = np.argmax(matrices["R1"], axis=1)
        r1_freqs = np.array([np.mean(r1_picks == i) for i in range(len(MODELS))])
        l1 = np.sum(np.abs(freqs - r1_freqs))
        print(f"  Routing distribution L1(R1, {j}): {l1:.3f}")


if __name__ == "__main__":
    analyze()
