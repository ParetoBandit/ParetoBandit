#!/usr/bin/env python3
"""Generate figures for Experiment 02: Non-Stationary K=3 Adaptation.

Reads ``reward_shift_results.json`` and produces publication-ready figures
for the reward shift experiment (Llama/Mistral reward swap).

The primary figure shows cumulative regret for four conditions:

  1. **Fixed Policy (offline)** — frozen warmup priors, no online learning.
  2. **Naive Bandit (γ=1.0)** — online LinUCB, infinite memory.
  3. **SW-UCB (W=200)** — Sliding-Window LinUCB, no priors.
  4. **BanditGPT (γ=0.995)** — warmup priors + geometric forgetting.

A secondary figure shows BanditGPT's arm selection dynamics.

Usage:
    python -m experiments.02_nonstationary_k3_drift.generate_figure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from bandit_gpt.config import BEST_K3_HPARAMS
from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"

# ======================================================================
# Visual encoding — maximally distinct for 4 conditions
# ======================================================================

BANDITGPT_LABEL: str = f"BanditGPT (\u03b3={BEST_K3_HPARAMS['forgetting_factor']})"

CONDITION_ORDER: List[str] = [
    "Fixed Policy (offline)",
    "Naive Bandit (\u03b3=1.0)",
    "SW-UCB (W=200)",
    BANDITGPT_LABEL,
]

CONDITION_COLORS: Dict[str, str] = {
    "Fixed Policy (offline)": "#888888",
    "Naive Bandit (\u03b3=1.0)": "#D55E00",
    "SW-UCB (W=200)": "#CC79A7",
    BANDITGPT_LABEL: "#0072B2",
}

CONDITION_STYLES: Dict[str, str] = {
    "Fixed Policy (offline)": "--",
    "Naive Bandit (\u03b3=1.0)": "--",
    "SW-UCB (W=200)": "-.",
    BANDITGPT_LABEL: "-",
}

CONDITION_LINEWIDTHS: Dict[str, float] = {
    "Fixed Policy (offline)": 2.0,
    "Naive Bandit (\u03b3=1.0)": 2.0,
    "SW-UCB (W=200)": 2.0,
    BANDITGPT_LABEL: 2.8,
}

ARM_COLORS: Dict[str, str] = {
    "Llama-8B": "#56B4E9",
    "Mistral-Large": "#D55E00",
    "Gemini-Pro": "#009E73",
}


def _load_results(filename: str) -> Dict[str, Any]:
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def _add_phase_boundary(
    ax: plt.Axes,
    boundary: int,
    y_frac: float = 0.95,
    label_left: str = "Phase 1",
    label_right: str = "Phase 2",
) -> None:
    """Draw a vertical dashed line at the phase boundary with labels."""
    ax.axvline(
        boundary, color="black", linestyle="--",
        linewidth=1.2, alpha=0.5, zorder=1,
    )
    y_lo, y_hi = ax.get_ylim()
    y_text = y_lo + y_frac * (y_hi - y_lo)
    ax.text(
        boundary - 15, y_text,
        label_left,
        ha="right", va="top", fontsize=8, fontstyle="italic",
        color="#555555",
    )
    ax.text(
        boundary + 15, y_text,
        label_right,
        ha="left", va="top", fontsize=8, fontstyle="italic",
        color="#555555",
    )


# ======================================================================
# Figure: Cumulative Regret (primary figure)
# ======================================================================


def plot_cumulative_regret(data: Dict[str, Any]) -> plt.Figure:
    """5-condition cumulative regret under reward swap.

    Parameters
    ----------
    data : dict
        Parsed ``reward_shift_results.json``.

    Returns
    -------
    plt.Figure
    """
    conditions = data["conditions"]
    phase_boundary = None

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label in CONDITION_ORDER:
        if label not in conditions:
            continue
        curve = conditions[label]
        steps = [c["step"] for c in curve]
        regrets = [c["mean_cumulative_regret"] for c in curve]

        if phase_boundary is None:
            phase_boundary = curve[0]["phase_boundary"]

        color = CONDITION_COLORS[label]
        ls = CONDITION_STYLES[label]
        lw = CONDITION_LINEWIDTHS[label]

        has_per_seed = "per_seed_cumulative_regret" in curve[0]
        if has_per_seed:
            matrix = np.array([c["per_seed_cumulative_regret"] for c in curve])
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            se_regrets = [
                c["std_cumulative_regret"] / np.sqrt(c["n_seeds"])
                for c in curve
            ]
            ci_lo = [r - s for r, s in zip(regrets, se_regrets)]
            ci_hi = [r + s for r, s in zip(regrets, se_regrets)]

        ax.plot(
            steps, regrets,
            color=color, linestyle=ls, linewidth=lw,
            label=label, zorder=4,
        )
        ax.fill_between(
            steps, ci_lo, ci_hi,
            alpha=0.12, color=color, zorder=2,
        )

        ax.annotate(
            f"{regrets[-1]:.0f}",
            xy=(steps[-1], regrets[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=color, va="center", fontweight="bold",
        )

    if phase_boundary is not None:
        _add_phase_boundary(
            ax, phase_boundary, y_frac=0.25,
            label_left="Phase 1\n(normal)",
            label_right="Phase 2\n(Llama\u2194Mistral swap)",
        )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(
        "Cumulative Regret Under Reward Swap (K=3, 40 seeds, 95% bootstrap CI)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


# ======================================================================
# Figure: Arm Fraction Dynamics (BanditGPT only)
# ======================================================================


def plot_arm_fractions(data: Dict[str, Any]) -> plt.Figure:
    """Per-arm selection fractions for BanditGPT, showing routing adaptation.

    Parameters
    ----------
    data : dict
        Parsed ``reward_shift_results.json``.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If no BanditGPT condition is found in the results.
    """
    conditions = data["conditions"]
    arm_short_names = list(data["arm_short"].values())

    if BANDITGPT_LABEL not in conditions:
        raise ValueError(
            f"Expected condition {BANDITGPT_LABEL!r} not found in results. "
            f"Available: {list(conditions.keys())}"
        )

    curve = conditions[BANDITGPT_LABEL]
    steps = [c["step"] for c in curve]
    phase_boundary = curve[0]["phase_boundary"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    has_per_seed = "per_seed_arm_fractions" in curve[0]

    for arm in arm_short_names:
        means = [c["arm_fractions"].get(arm, 0.0) for c in curve]
        color = ARM_COLORS[arm]

        if has_per_seed:
            matrix = np.array([
                c["per_seed_arm_fractions"][arm] for c in curve
            ])
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            stds = [c["arm_fractions_std"].get(arm, 0.0) for c in curve]
            ci_lo = [m - s for m, s in zip(means, stds)]
            ci_hi = [m + s for m, s in zip(means, stds)]

        ax.plot(
            steps, means,
            color=color, linewidth=2.4, label=arm, zorder=4,
        )
        ax.fill_between(
            steps, ci_lo, ci_hi,
            alpha=0.15, color=color, zorder=2,
        )

    _add_phase_boundary(
        ax, phase_boundary, y_frac=0.95,
        label_left="Phase 1\n(normal)",
        label_right="Phase 2\n(Llama\u2194Mistral swap)",
    )

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Arm Selection Fraction", fontsize=13)
    ax.set_title(
        "Model Selection Dynamics Under Reward Swap (BanditGPT, 95% bootstrap CI)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, loc="center right", framealpha=0.9)

    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filepath = RESULTS_DIR / "reward_shift_results.json"
    if not filepath.exists():
        print(f"  Skipping: {filepath.name} not found")
        return

    data = _load_results("reward_shift_results.json")

    figures: Dict[str, plt.Figure] = {
        "cumulative_regret": plot_cumulative_regret(data),
        "reward_shift_arm_fractions": plot_arm_fractions(data),
    }

    for name, fig in figures.items():
        for fmt in ("pdf", "png"):
            out = RESULTS_DIR / f"{name}.{fmt}"
            fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}.{{pdf,png}}")

    print(f"\nAll figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
