#!/usr/bin/env python3
"""Diagnostic: does the data actually support using R1 alone?

Computes routing-relevant agreement metrics that go beyond correlation:

1. Best-model agreement — how often judges pick the same winner
2. Routing regret — reward lost by following one judge's oracle vs another's
3. Gap-conditioned agreement — does disagreement matter where gaps are large?
4. Rank-order concordance — full Kendall W across all 3 judges per prompt
5. Gap stability — is R1's larger gap signal or noise?

Usage
-----
    python experiments/appendix/judge_robustness/diagnostic_agreement.py
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from judge_robustness_utils import (
    MODELS,
    MODEL_SHORT,
    build_prompt_matrices,
    load_all_scores,
)


def analyze() -> None:
    """Run all diagnostic analyses and print results."""
    all_scores = load_all_scores()
    prompt_list, matrices = build_prompt_matrices(all_scores)
    n_prompts = len(prompt_list)
    judge_names = list(matrices.keys())

    print(f"Prompts with scores from all 3 judges × all 3 models: {n_prompts}")
    print(f"Judges: {judge_names}")
    print()

    # ── 1. Best-model agreement ──────────────────────────────────────────
    print("=" * 70)
    print("1. BEST-MODEL AGREEMENT")
    print("=" * 70)
    best_models = {j: np.argmax(m, axis=1) for j, m in matrices.items()}

    for j1 in judge_names:
        for j2 in judge_names:
            if j1 >= j2:
                continue
            agree = np.mean(best_models[j1] == best_models[j2])
            print(f"  {j1} vs {j2}: {agree:.1%}")

    all_three_agree = np.mean(
        (best_models["R1"] == best_models["GPT-4.1-mini"])
        & (best_models["R1"] == best_models["Claude-3.7-Sonnet"])
    )
    print(f"  All three agree: {all_three_agree:.1%}")
    print(f"  Random baseline (K=3): {1/3:.1%}")
    print()

    # ── 2. Gap-conditioned best-model agreement ──────────────────────────
    print("=" * 70)
    print("2. BEST-MODEL AGREEMENT CONDITIONED ON R1 GAP SIZE")
    print("=" * 70)
    r1_mat = matrices["R1"]
    r1_gaps = np.max(r1_mat, axis=1) - np.min(r1_mat, axis=1)
    thresholds = [0.0, 0.05, 0.10, 0.20, 0.30]

    for lo, hi in zip(thresholds, thresholds[1:] + [1.0]):
        mask = (r1_gaps >= lo) & (r1_gaps < hi)
        n = mask.sum()
        if n == 0:
            continue
        for j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
            agree = np.mean(best_models["R1"][mask] == best_models[j][mask])
            print(f"  Gap [{lo:.2f}, {hi:.2f}) n={n:4d}: "
                  f"R1 vs {j:20s} agree={agree:.1%}")
        print()

    # ── 2b. Consensus-gap conditioned agreement ─────────────────────────
    # Addresses circularity concern: conditioning on R1's gap bins when
    # validating R1 can bias results.  Here we condition on the panel
    # median gap (median across all 3 judges) as a judge-neutral proxy.
    print("=" * 70)
    print("2b. BEST-MODEL AGREEMENT CONDITIONED ON CONSENSUS (MEDIAN) GAP")
    print("    Avoids circularity of conditioning on R1's own gap bins.")
    print("=" * 70)
    all_gaps = np.stack([
        np.max(matrices[j], axis=1) - np.min(matrices[j], axis=1)
        for j in judge_names
    ], axis=0)
    consensus_gaps = np.median(all_gaps, axis=0)

    for lo, hi in zip(thresholds, thresholds[1:] + [1.0]):
        mask = (consensus_gaps >= lo) & (consensus_gaps < hi)
        n = mask.sum()
        if n == 0:
            continue
        for j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
            agree = np.mean(best_models["R1"][mask] == best_models[j][mask])
            print(f"  Consensus gap [{lo:.2f}, {hi:.2f}) n={n:4d}: "
                  f"R1 vs {j:20s} agree={agree:.1%}")
        print()

    # ── 3. Routing regret ────────────────────────────────────────────────
    print("=" * 70)
    print("3. ROUTING REGRET (reward lost by following judge X's oracle")
    print("   evaluated by judge Y's scores)")
    print("=" * 70)
    for oracle_j in judge_names:
        oracle_picks = np.argmax(matrices[oracle_j], axis=1)
        for eval_j in judge_names:
            eval_mat = matrices[eval_j]
            oracle_reward = eval_mat[np.arange(n_prompts), oracle_picks]
            best_reward = np.max(eval_mat, axis=1)
            regret = best_reward - oracle_reward
            print(f"  Oracle={oracle_j:20s}  Eval={eval_j:20s}  "
                  f"mean_regret={np.mean(regret):.4f}  "
                  f"median={np.median(regret):.4f}")
    print()

    # ── 4. Kendall's W (multi-rater concordance) ────────────────────────
    print("=" * 70)
    print("4. KENDALL'S W (concordance across all 3 judges)")
    print("=" * 70)
    # Per prompt: rank the 3 models by each judge, compute W
    w_values = []
    for i in range(n_prompts):
        ranks = np.array([
            stats.rankdata(matrices[j][i]) for j in judge_names
        ])
        n_raters, n_items = ranks.shape
        rank_sums = ranks.sum(axis=0)
        mean_rank_sum = np.mean(rank_sums)
        ss = np.sum((rank_sums - mean_rank_sum) ** 2)
        w = 12.0 * ss / (n_raters ** 2 * (n_items ** 3 - n_items))
        w_values.append(w)

    w_arr = np.array(w_values)
    print(f"  Mean W:   {np.mean(w_arr):.3f}")
    print(f"  Median W: {np.median(w_arr):.3f}")
    print(f"  W > 0.5:  {np.mean(w_arr > 0.5):.1%} of prompts")
    print(f"  W > 0.7:  {np.mean(w_arr > 0.7):.1%} of prompts")
    print(f"  W = 1.0:  {np.mean(w_arr == 1.0):.1%} of prompts (perfect concordance)")
    print()

    # Kendall W conditioned on gap size
    print("  Kendall W by R1 gap size:")
    for lo, hi in zip(thresholds, thresholds[1:] + [1.0]):
        mask = (r1_gaps >= lo) & (r1_gaps < hi)
        n = mask.sum()
        if n > 0:
            print(f"    Gap [{lo:.2f}, {hi:.2f}) n={n:4d}: "
                  f"mean W={np.mean(w_arr[mask]):.3f}")
    print()

    print("  Kendall W by CONSENSUS (median) gap size:")
    for lo, hi in zip(thresholds, thresholds[1:] + [1.0]):
        mask = (consensus_gaps >= lo) & (consensus_gaps < hi)
        n = mask.sum()
        if n > 0:
            print(f"    Consensus gap [{lo:.2f}, {hi:.2f}) n={n:4d}: "
                  f"mean W={np.mean(w_arr[mask]):.3f}")
    print()

    # ── 5. Is R1's larger gap signal or noise? ───────────────────────────
    print("=" * 70)
    print("5. GAP STABILITY: IS R1's LARGER GAP SIGNAL OR NOISE?")
    print("=" * 70)
    # If R1's larger gaps are signal, then prompts where R1 sees a large
    # gap should also have large gaps under the other judges.
    for j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
        j_mat = matrices[j]
        j_gaps = np.max(j_mat, axis=1) - np.min(j_mat, axis=1)
        r, p = stats.pearsonr(r1_gaps, j_gaps)
        tau, taup = stats.kendalltau(r1_gaps, j_gaps)
        print(f"  R1 gap vs {j:20s} gap: "
              f"Pearson r={r:.3f} (p={p:.2e})  "
              f"Kendall tau={tau:.3f} (p={taup:.2e})")

    print()
    # Rank-order: when R1 says gap is large, do others agree on the ranking?
    print("  Best-model agreement on prompts where R1 gap > median vs <= median:")
    median_gap = np.median(r1_gaps)
    for j in ["GPT-4.1-mini", "Claude-3.7-Sonnet"]:
        hi_mask = r1_gaps > median_gap
        lo_mask = ~hi_mask
        agree_hi = np.mean(best_models["R1"][hi_mask] == best_models[j][hi_mask])
        agree_lo = np.mean(best_models["R1"][lo_mask] == best_models[j][lo_mask])
        print(f"  {j:20s}: above_median={agree_hi:.1%}  "
              f"below_median={agree_lo:.1%}  "
              f"delta={agree_hi - agree_lo:+.1%}")
    print()

    # ── 6. Model selection frequency ─────────────────────────────────────
    print("=" * 70)
    print("6. MODEL SELECTION FREQUENCY (which model each judge picks)")
    print("=" * 70)
    for j in judge_names:
        picks = best_models[j]
        print(f"  {j:20s}: ", end="")
        for mi, m in enumerate(MODELS):
            frac = np.mean(picks == mi)
            print(f"{MODEL_SHORT[m]}={frac:.1%}  ", end="")
        print()
    print()

    # ── 7. Summary assessment ────────────────────────────────────────────
    print("=" * 70)
    print("7. SUMMARY")
    print("=" * 70)
    print(f"  Best-model agreement (R1 vs Claude):     "
          f"{np.mean(best_models['R1'] == best_models['Claude-3.7-Sonnet']):.1%}")
    print(f"  Best-model agreement (R1 vs GPT-mini):   "
          f"{np.mean(best_models['R1'] == best_models['GPT-4.1-mini']):.1%}")
    print(f"  All-three agreement:                     {all_three_agree:.1%}")
    print(f"  Mean Kendall W:                          {np.mean(w_arr):.3f}")
    print(f"  R1 gap-GPT gap Pearson r:                "
          f"{stats.pearsonr(r1_gaps, np.max(matrices['GPT-4.1-mini'], axis=1) - np.min(matrices['GPT-4.1-mini'], axis=1))[0]:.3f}")
    print(f"  R1 gap-Claude gap Pearson r:             "
          f"{stats.pearsonr(r1_gaps, np.max(matrices['Claude-3.7-Sonnet'], axis=1) - np.min(matrices['Claude-3.7-Sonnet'], axis=1))[0]:.3f}")


if __name__ == "__main__":
    analyze()
