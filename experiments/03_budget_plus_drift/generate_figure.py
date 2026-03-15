#!/usr/bin/env python3
"""Generate figures for Experiment 03: Budget Pacing Under Cost Drift.

Reads ``results/budget_cost_drift_results.json`` and produces:

``adaptation_dynamics.pdf/.png``:
  1x3 panel showing BanditGPT-only adaptation mechanics:
    (a) dual variable λ_t,  (b) Gemini-Pro selection fraction,
    (c) running average cost per request, each at three budget levels.

Usage:
    python experiments/03_budget_plus_drift/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "budget_cost_drift_results.json"

# ======================================================================
# Visual encoding
# ======================================================================

BUDGET_COLORS: Dict[str, str] = {
    "tight": "#D55E00",
    "moderate": "#0072B2",
    "loose": "#CC79A7",
}

BUDGET_NICE_LABELS: Dict[str, str] = {
    "tight": r"Tight ($B{=}\$2.3{\times}10^{-4}$)",
    "moderate": r"Moderate ($B{=}\$6.6{\times}10^{-4}$)",
    "loose": r"Loose ($B{=}\$1.9{\times}10^{-3}$)",
}

BUDGET_PANEL_TITLES: Dict[str, str] = {
    "tight": "(a) Tight Budget",
    "moderate": "(b) Moderate Budget",
    "loose": "(c) Loose Budget",
}

UNCONSTRAINED_COLOR = "#009E73"


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def _find_condition_key(
    conditions: Dict[str, Any],
    prefix: str,
    budget_label: str,
) -> Optional[str]:
    """Find condition key matching a prefix and budget label."""
    for key in conditions:
        if key.startswith(prefix) and f"({budget_label})" in key:
            return key
    return None


def _add_phase_boundary(
    ax: plt.Axes,
    boundary: int,
    label: bool = True,
) -> None:
    ax.axvline(
        boundary, color="black", linestyle="--",
        linewidth=1.0, alpha=0.4, zorder=1,
    )
    if label:
        y_lo, y_hi = ax.get_ylim()
        y_text = y_lo + 0.25 * (y_hi - y_lo)
        ax.text(
            boundary - 10, y_text, "P1",
            ha="right", va="top", fontsize=7, fontstyle="italic",
            color="#555555",
        )
        ax.text(
            boundary + 10, y_text, "P2",
            ha="left", va="top", fontsize=7, fontstyle="italic",
            color="#555555",
        )


# ======================================================================
# Adaptation Dynamics (1x3)
# ======================================================================


def plot_adaptation_dynamics(data: Dict[str, Any]) -> plt.Figure:
    """1x3 figure showing BanditGPT adaptation mechanics across budgets.

    Parameters
    ----------
    data : dict
        Parsed results JSON.

    Returns
    -------
    plt.Figure
    """
    budget_labels = data["budget_labels"]
    budget_targets = data["budget_targets"]
    conditions = data["conditions"]
    phase_boundary = data["phase1_n"]
    n_seeds = data["n_seeds"]
    sqrt_n = np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # ---- Panel (a): Lambda trajectory ----
    ax_lam = axes[0]
    for blabel, _btarget in zip(budget_labels, budget_targets):
        cond_key = _find_condition_key(conditions, "BanditGPT", blabel)
        if cond_key is None:
            continue
        curve = conditions[cond_key]["curves"]
        steps = [c["step"] for c in curve]
        lambdas = [c["mean_lambda"] for c in curve]
        se_lambdas = [c["std_lambda"] / sqrt_n for c in curve]
        color = BUDGET_COLORS[blabel]

        ax_lam.plot(
            steps, lambdas,
            color=color, linewidth=2.2, label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_lam.fill_between(
            steps,
            [m - s for m, s in zip(lambdas, se_lambdas)],
            [m + s for m, s in zip(lambdas, se_lambdas)],
            alpha=0.18, color=color, zorder=2,
        )

    _add_phase_boundary(ax_lam, phase_boundary, label=False)
    ax_lam.set_title(
        r"(a) Dual Variable $\lambda_t$",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_lam.set_xlabel("Step", fontsize=10)
    ax_lam.set_ylabel(r"$\lambda_t$", fontsize=11)
    ax_lam.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_lam.grid(True, alpha=0.2, linewidth=0.5)
    ax_lam.tick_params(labelsize=9)

    # ---- Panel (b): Gemini-Pro selection fraction ----
    ax_mix = axes[1]
    for blabel in budget_labels:
        cond_key = _find_condition_key(conditions, "BanditGPT", blabel)
        if cond_key is None:
            continue
        curve = conditions[cond_key]["curves"]
        steps = [c["step"] for c in curve]
        fracs = [c["arm_fractions"].get("Gemini-Pro", 0.0) for c in curve]
        ses = [c["arm_fractions_std"].get("Gemini-Pro", 0.0) / sqrt_n for c in curve]
        color = BUDGET_COLORS[blabel]

        ax_mix.plot(
            steps, fracs,
            color=color, linewidth=2.2, label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_mix.fill_between(
            steps,
            [m - s for m, s in zip(fracs, ses)],
            [m + s for m, s in zip(fracs, ses)],
            alpha=0.18, color=color, zorder=2,
        )

    if "Unconstrained" in conditions:
        uc_curve = conditions["Unconstrained"]["curves"]
        uc_steps = [c["step"] for c in uc_curve]
        uc_fracs = [c["arm_fractions"].get("Gemini-Pro", 0.0) for c in uc_curve]
        uc_ses = [c["arm_fractions_std"].get("Gemini-Pro", 0.0) / sqrt_n for c in uc_curve]
        ax_mix.plot(
            uc_steps, uc_fracs,
            color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
            label="Unconstrained", zorder=3,
        )
        ax_mix.fill_between(
            uc_steps,
            [m - s for m, s in zip(uc_fracs, uc_ses)],
            [m + s for m, s in zip(uc_fracs, uc_ses)],
            alpha=0.12, color=UNCONSTRAINED_COLOR, zorder=2,
        )

    _add_phase_boundary(ax_mix, phase_boundary, label=False)
    ax_mix.set_title(
        "(b) Gemini-Pro Selection Fraction",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_mix.set_xlabel("Step", fontsize=10)
    ax_mix.set_ylabel("Fraction", fontsize=11)
    ax_mix.set_ylim(-0.02, 1.02)
    ax_mix.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax_mix.grid(True, alpha=0.2, linewidth=0.5)
    ax_mix.tick_params(labelsize=9)

    # ---- Panel (c): Running average cost per request ----
    ax_cost = axes[2]
    for blabel, btarget in zip(budget_labels, budget_targets):
        cond_key = _find_condition_key(conditions, "BanditGPT", blabel)
        if cond_key is None:
            continue
        curve = conditions[cond_key]["curves"]
        steps = [c["step"] for c in curve]
        avg_costs = [c["mean_avg_cost"] for c in curve]
        se_costs = [c["std_avg_cost"] / sqrt_n for c in curve]
        color = BUDGET_COLORS[blabel]

        ax_cost.plot(
            steps, avg_costs,
            color=color, linewidth=2.2, label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_cost.fill_between(
            steps,
            [m - s for m, s in zip(avg_costs, se_costs)],
            [m + s for m, s in zip(avg_costs, se_costs)],
            alpha=0.18, color=color, zorder=2,
        )
        ax_cost.axhline(
            btarget, color=color, linestyle=":", linewidth=1.2,
            alpha=0.5, zorder=1,
        )

    if "Unconstrained" in conditions:
        uc_curve = conditions["Unconstrained"]["curves"]
        uc_steps = [c["step"] for c in uc_curve]
        uc_costs = [c["mean_avg_cost"] for c in uc_curve]
        uc_ses = [c["std_avg_cost"] / sqrt_n for c in uc_curve]
        ax_cost.plot(
            uc_steps, uc_costs,
            color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
            label="Unconstrained", zorder=3,
        )
        ax_cost.fill_between(
            uc_steps,
            [m - s for m, s in zip(uc_costs, uc_ses)],
            [m + s for m, s in zip(uc_costs, uc_ses)],
            alpha=0.12, color=UNCONSTRAINED_COLOR, zorder=2,
        )

    _add_phase_boundary(ax_cost, phase_boundary, label=False)
    ax_cost.set_title(
        "(c) Running Avg Cost / Request",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_cost.set_xlabel("Step", fontsize=10)
    ax_cost.set_ylabel("$/request", fontsize=11)
    ax_cost.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax_cost.grid(True, alpha=0.2, linewidth=0.5)
    ax_cost.tick_params(labelsize=9)

    fig.suptitle(
        r"BanditGPT Adaptation Dynamics Under Cost Drift ($K{=}3$, "
        rf"Pacer + $\gamma{{=}}0.995$, {n_seeds} seeds, $\pm$1 SE)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()

    dynamics_fig = plot_adaptation_dynamics(data)
    for fmt in ("pdf", "png"):
        dynamics_fig.savefig(
            RESULTS_DIR / f"adaptation_dynamics.{fmt}",
            bbox_inches="tight", dpi=300,
        )
    plt.close(dynamics_fig)
    print("Saved adaptation_dynamics.{pdf,png}")


if __name__ == "__main__":
    main()
