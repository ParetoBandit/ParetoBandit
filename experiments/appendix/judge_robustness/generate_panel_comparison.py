#!/usr/bin/env python3
"""Generate figure: R1 alone vs three-judge panel average.

Demonstrates concretely why averaging judges compresses the routing
signal available to the bandit:

1. **Gap distribution**: overlaid histogram of per-prompt reward gaps
   under R1 vs the panel average, with annotations.
2. **Actionable-prompt fraction**: cumulative plot showing what fraction
   of prompts have gap >= threshold, for each scoring approach.
3. **Summary statistics**: printed to stdout for inclusion in the tex.

Usage
-----
    python experiments/appendix/judge_robustness/generate_panel_comparison.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from bandit_gpt.config import CALIBRATION_DIR, PARETO_REWARDS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SUBSET_PROMPTS_PATH = CALIBRATION_DIR / "judge_robustness_prompts.jsonl"
SUPPLEMENTARY_REWARDS_PATH = CALIBRATION_DIR / "judge_robustness_rewards.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#7B2D8E"
CB_GRAY = "#999999"


def _setup_matplotlib() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════


def load_all_scores() -> Dict[str, Dict[Tuple[str, str], float]]:
    """Load R1 + supplementary scores keyed by judge name.

    Returns
    -------
    Dict[str, Dict[Tuple[str, str], float]]
        {judge_name: {(prompt, model_id): score}}.
    """
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


def build_matrices(
    all_scores: Dict[str, Dict[Tuple[str, str], float]],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Build [n_prompts × n_models] matrices on common keys.

    Returns
    -------
    Tuple[List[str], Dict[str, np.ndarray]]
        (prompt_list, {judge_name: score_matrix}).
    """
    common_keys = set.intersection(
        *[set(s.keys()) for s in all_scores.values()]
    )
    prompts_with_all = sorted({
        p for p, _ in common_keys
        if all((p, m) in common_keys for m in MODELS)
    })

    matrices: Dict[str, np.ndarray] = {}
    for judge, scores in all_scores.items():
        matrices[judge] = np.array([
            [scores[(p, m)] for m in MODELS]
            for p in prompts_with_all
        ])
    return prompts_with_all, matrices


# ══════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════


def compute_panel_average(
    matrices: Dict[str, np.ndarray],
) -> np.ndarray:
    """Element-wise mean across all judges.

    Parameters
    ----------
    matrices:
        {judge_name: [n_prompts × n_models]} arrays.

    Returns
    -------
    np.ndarray
        Panel-averaged score matrix [n_prompts × n_models].
    """
    return np.mean(list(matrices.values()), axis=0)


def per_prompt_gaps(mat: np.ndarray) -> np.ndarray:
    """Best-model score minus worst-model score, per prompt.

    Parameters
    ----------
    mat:
        Score matrix [n_prompts × n_models].

    Returns
    -------
    np.ndarray
        Gap array of length n_prompts.
    """
    return np.max(mat, axis=1) - np.min(mat, axis=1)


def best_vs_second_gaps(mat: np.ndarray) -> np.ndarray:
    """Gap between best and second-best model, per prompt.

    This is the margin the bandit needs to detect in order to make
    the correct routing decision.

    Parameters
    ----------
    mat:
        Score matrix [n_prompts × n_models].

    Returns
    -------
    np.ndarray
        Margin array of length n_prompts.
    """
    sorted_scores = np.sort(mat, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


# ══════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════


def plot_signal_compression(
    r1_mat: np.ndarray,
    panel_mat: np.ndarray,
) -> Path:
    """Two-panel figure: gap distributions and actionable-prompt fraction.

    Parameters
    ----------
    r1_mat:
        R1-only score matrix.
    panel_mat:
        Panel-averaged score matrix.

    Returns
    -------
    Path
        Path to saved PDF.
    """
    _setup_matplotlib()

    r1_gaps = per_prompt_gaps(r1_mat)
    panel_gaps = per_prompt_gaps(panel_mat)
    r1_margins = best_vs_second_gaps(r1_mat)
    panel_margins = best_vs_second_gaps(panel_mat)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.5, 3.0), constrained_layout=True,
    )

    # ── Panel A: gap distribution overlay ──
    bins = np.linspace(0, 0.8, 50)
    ax1.hist(
        r1_gaps, bins=bins, alpha=0.5, color=CB_BLUE, density=True,
        label=f"R1 alone ($\\mu$={np.mean(r1_gaps):.3f})",
    )
    ax1.hist(
        panel_gaps, bins=bins, alpha=0.5, color=CB_PURPLE, density=True,
        label=f"3-judge avg ($\\mu$={np.mean(panel_gaps):.3f})",
    )

    ax1.axvline(
        np.mean(r1_gaps), color=CB_BLUE, ls="--", lw=1.0, alpha=0.7,
    )
    ax1.axvline(
        np.mean(panel_gaps), color=CB_PURPLE, ls="--", lw=1.0, alpha=0.7,
    )

    compression = 1.0 - np.mean(panel_gaps) / np.mean(r1_gaps)
    ax1.text(
        0.96, 0.96,
        f"Gap compression:\n{compression:.0%} smaller",
        transform=ax1.transAxes, fontsize=8,
        va="top", ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#fff3cd",
            edgecolor="#ffc107", alpha=0.9,
        ),
    )

    ax1.set_xlabel("Per-Prompt Reward Gap (best $-$ worst)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Signal Compression from Averaging", fontweight="bold")
    ax1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.99, 0.82))
    ax1.grid(True, alpha=0.12, ls="--", lw=0.5)

    # ── Panel B: actionable prompt fraction ──
    thresholds = np.linspace(0, 0.5, 200)
    r1_frac = np.array([np.mean(r1_margins >= t) for t in thresholds])
    panel_frac = np.array([np.mean(panel_margins >= t) for t in thresholds])

    ax2.plot(
        thresholds, r1_frac * 100,
        color=CB_BLUE, lw=1.8,
        label="R1 alone",
    )
    ax2.plot(
        thresholds, panel_frac * 100,
        color=CB_PURPLE, lw=1.8, ls="--",
        label="3-judge avg",
    )
    ax2.fill_between(
        thresholds,
        panel_frac * 100,
        r1_frac * 100,
        alpha=0.15,
        color=CB_RED,
        label="Lost signal",
    )

    ref_threshold = 0.05
    r1_at_ref = np.mean(r1_margins >= ref_threshold) * 100
    panel_at_ref = np.mean(panel_margins >= ref_threshold) * 100
    ax2.annotate(
        f"At margin $\\geq$ {ref_threshold}:\n"
        f"R1: {r1_at_ref:.0f}%\n"
        f"Avg: {panel_at_ref:.0f}%",
        xy=(ref_threshold, panel_at_ref),
        xytext=(0.20, panel_at_ref + 15),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color=CB_GRAY, lw=0.8),
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white",
            edgecolor="#cccccc", alpha=0.9,
        ),
    )

    ax2.set_xlabel("Routing Margin Threshold (best $-$ 2nd best)")
    ax2.set_ylabel("% of Prompts Above Threshold")
    ax2.set_title("(b) Actionable Prompts for Routing", fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.12, ls="--", lw=0.5)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 100)

    out = RESULTS_DIR / "panel_vs_r1_signal.pdf"
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    logger.info("Saved panel comparison: %s", out)
    return out


def plot_per_prompt_gap_scatter(
    r1_mat: np.ndarray,
    panel_mat: np.ndarray,
) -> Path:
    """Scatter: per-prompt gap under R1 vs panel average.

    Points below the identity line are prompts where averaging
    compressed the signal.

    Parameters
    ----------
    r1_mat:
        R1-only score matrix.
    panel_mat:
        Panel-averaged score matrix.

    Returns
    -------
    Path
        Path to saved PDF.
    """
    _setup_matplotlib()

    r1_gaps = per_prompt_gaps(r1_mat)
    panel_gaps = per_prompt_gaps(panel_mat)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)

    ax.scatter(
        r1_gaps, panel_gaps,
        s=4, alpha=0.15, c=CB_BLUE,
        edgecolors="none", rasterized=True,
    )

    lims = [0, max(r1_gaps.max(), panel_gaps.max()) + 0.02]
    ax.plot(lims, lims, "--", color=CB_GRAY, lw=0.8, alpha=0.5, zorder=0)

    below = np.mean(panel_gaps < r1_gaps)
    ax.text(
        0.96, 0.04,
        f"{below:.0%} of prompts\nhave smaller gap\nunder averaging",
        transform=ax.transAxes, fontsize=7.5,
        va="bottom", ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="#fff3cd",
            edgecolor="#ffc107", alpha=0.9,
        ),
    )

    ax.set_xlabel("R1 Per-Prompt Gap")
    ax.set_ylabel("Panel Average Per-Prompt Gap")
    ax.set_title("Gap Compression Per Prompt", fontweight="bold")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.12, ls="--", lw=0.5)

    out = RESULTS_DIR / "gap_compression_scatter.pdf"
    fig.savefig(out, facecolor="white")
    fig.savefig(out.with_suffix(".png"), facecolor="white")
    plt.close(fig)
    logger.info("Saved gap scatter: %s", out)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════


def print_summary(
    matrices: Dict[str, np.ndarray],
    panel_mat: np.ndarray,
) -> Dict:
    """Print and return key comparison statistics.

    Parameters
    ----------
    matrices:
        Per-judge score matrices.
    panel_mat:
        Panel-averaged score matrix.

    Returns
    -------
    Dict
        Summary statistics for JSON export.
    """
    r1_mat = matrices["R1"]
    n = len(r1_mat)

    r1_gaps = per_prompt_gaps(r1_mat)
    panel_gaps = per_prompt_gaps(panel_mat)
    r1_margins = best_vs_second_gaps(r1_mat)
    panel_margins = best_vs_second_gaps(panel_mat)

    r1_best = np.argmax(r1_mat, axis=1)
    panel_best = np.argmax(panel_mat, axis=1)

    print("=" * 70)
    print("R1 ALONE vs 3-JUDGE PANEL AVERAGE")
    print("=" * 70)
    print()

    print("Gap statistics (best - worst model):")
    print(f"  R1 mean gap:    {np.mean(r1_gaps):.4f}")
    print(f"  Panel mean gap: {np.mean(panel_gaps):.4f}")
    compression = 1.0 - np.mean(panel_gaps) / np.mean(r1_gaps)
    print(f"  Compression:    {compression:.1%}")
    print(f"  R1 median gap:    {np.median(r1_gaps):.4f}")
    print(f"  Panel median gap: {np.median(panel_gaps):.4f}")
    print()

    print("Routing margin statistics (best - 2nd best):")
    print(f"  R1 mean margin:    {np.mean(r1_margins):.4f}")
    print(f"  Panel mean margin: {np.mean(panel_margins):.4f}")
    margin_compression = 1.0 - np.mean(panel_margins) / np.mean(r1_margins)
    print(f"  Compression:       {margin_compression:.1%}")
    print()

    print("Fraction of prompts with clear routing signal:")
    for thresh in [0.02, 0.05, 0.10, 0.20]:
        r1_frac = np.mean(r1_margins >= thresh)
        panel_frac = np.mean(panel_margins >= thresh)
        lost = r1_frac - panel_frac
        print(f"  margin >= {thresh:.2f}:  R1={r1_frac:.1%}  "
              f"Panel={panel_frac:.1%}  Lost={lost:+.1%}")
    print()

    print("Prompts where averaging compresses the gap:")
    print(f"  panel_gap < R1_gap:  {np.mean(panel_gaps < r1_gaps):.1%}")
    print(f"  panel_gap = R1_gap:  {np.mean(panel_gaps == r1_gaps):.1%}")
    print(f"  panel_gap > R1_gap:  {np.mean(panel_gaps > r1_gaps):.1%}")
    print()

    print("Best-model agreement (R1 vs panel):")
    agree = np.mean(r1_best == panel_best)
    print(f"  {agree:.1%}")
    print()

    print("Oracle reward comparison:")
    r1_oracle = np.max(r1_mat, axis=1).mean()
    panel_oracle_by_r1 = r1_mat[
        np.arange(n), np.argmax(panel_mat, axis=1)
    ].mean()
    print(f"  R1 oracle (eval by R1):        {r1_oracle:.4f}")
    print(f"  Panel oracle (eval by R1):     {panel_oracle_by_r1:.4f}")
    print(f"  Regret from using panel picks: {r1_oracle - panel_oracle_by_r1:.4f}")
    print()

    print("Model selection frequencies:")
    for label, picks in [("R1", r1_best), ("Panel avg", panel_best)]:
        print(f"  {label:12s}: ", end="")
        for mi, m in enumerate(MODELS):
            print(f"{MODEL_SHORT[m]}={np.mean(picks == mi):.1%}  ", end="")
        print()
    print()

    # SNR-based convergence estimate
    r1_snr = np.mean(r1_margins) / np.std(r1_margins)
    panel_snr = np.mean(panel_margins) / np.std(panel_margins)
    print("Signal-to-noise ratio (margin mean / margin std):")
    print(f"  R1:    {r1_snr:.3f}")
    print(f"  Panel: {panel_snr:.3f}")
    print(f"  Ratio: {r1_snr / panel_snr:.2f}x")
    print()

    summary = {
        "r1_mean_gap": round(float(np.mean(r1_gaps)), 4),
        "panel_mean_gap": round(float(np.mean(panel_gaps)), 4),
        "gap_compression_pct": round(compression * 100, 1),
        "r1_mean_margin": round(float(np.mean(r1_margins)), 4),
        "panel_mean_margin": round(float(np.mean(panel_margins)), 4),
        "margin_compression_pct": round(margin_compression * 100, 1),
        "frac_compressed": round(float(np.mean(panel_gaps < r1_gaps)) * 100, 1),
        "r1_panel_best_model_agreement": round(float(agree) * 100, 1),
        "r1_snr": round(r1_snr, 3),
        "panel_snr": round(panel_snr, 3),
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Generate all panel-comparison outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    all_scores = load_all_scores()
    prompt_list, matrices = build_matrices(all_scores)
    logger.info("Prompts: %d, Models: %d, Judges: %d",
                len(prompt_list), len(MODELS), len(matrices))

    panel_mat = compute_panel_average(matrices)

    logger.info("Computing statistics...")
    summary = print_summary(matrices, panel_mat)

    logger.info("Generating figures...")
    plot_signal_compression(matrices["R1"], panel_mat)
    plot_per_prompt_gap_scatter(matrices["R1"], panel_mat)

    out = RESULTS_DIR / "panel_comparison_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Exported summary: %s", out)

    logger.info("Done.")


if __name__ == "__main__":
    main()
