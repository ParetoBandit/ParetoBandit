#!/usr/bin/env python3
"""Generate figures for Experiment 03: Budget Pacing Under Cost Drift.

Reads ``results/budget_cost_drift_results.json`` and produces:

``adaptation_dynamics.pdf/.png``:
  1x3 panel showing BanditGPT-only adaptation mechanics
  (λ_t, Gemini fraction, running cost) across three budget levels.

Usage:
    python experiments/03_budget_plus_drift/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

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


def _extract_curve_with_ci(
    conditions: Dict[str, Any],
    prefix: str,
    budget_label: str,
    mean_field: str,
    per_seed_field: Optional[str] = None,
    std_field: Optional[str] = None,
    sqrt_n: float = 1.0,
) -> Optional[Tuple[List[int], List[float], np.ndarray, np.ndarray]]:
    """Extract (steps, means, ci_lo, ci_hi) for a condition's checkpoint curve.

    When *per_seed_field* is present in the JSON, bootstrap CIs are computed.
    Otherwise falls back to ``mean ± std / sqrt(n)`` (SE-based bands).
    """
    key = _find_condition_key(conditions, prefix, budget_label)
    if key is None:
        return None
    curve = conditions[key]["curves"]
    steps = [c["step"] for c in curve]
    means = [c[mean_field] for c in curve]

    has_per_seed = per_seed_field is not None and per_seed_field in curve[0]
    if has_per_seed:
        matrix = np.array([c[per_seed_field] for c in curve])
        ci_lo, ci_hi = bootstrap_ci_series(matrix)
    elif std_field is not None:
        ses = [c[std_field] / sqrt_n for c in curve]
        ci_lo = np.array([m - s for m, s in zip(means, ses)])
        ci_hi = np.array([m + s for m, s in zip(means, ses)])
    else:
        ci_lo = np.array(means)
        ci_hi = np.array(means)

    return steps, means, ci_lo, ci_hi


def plot_adaptation_dynamics(data: Dict[str, Any]) -> plt.Figure:
    """1x3 figure showing BanditGPT adaptation dynamics under cost drift.

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

    # ------------------------------------------------------------------
    # (a) Dual variable λ_t
    # ------------------------------------------------------------------
    ax_lam = axes[0]
    for blabel, _btarget in zip(budget_labels, budget_targets):
        result = _extract_curve_with_ci(
            conditions, "BanditGPT", blabel,
            mean_field="mean_lambda",
            per_seed_field="per_seed_lambda",
            std_field="std_lambda",
            sqrt_n=sqrt_n,
        )
        if result is None:
            continue
        steps, lambdas, ci_lo, ci_hi = result
        color = BUDGET_COLORS[blabel]

        ax_lam.plot(
            steps, lambdas,
            color=color, linewidth=2.2, label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_lam.fill_between(
            steps, ci_lo, ci_hi,
            alpha=0.18, color=color, zorder=2,
        )

    _add_phase_boundary(ax_lam, phase_boundary, label=False)
    ax_lam.set_title(
        r"(a) Dual Variable $\lambda_t$",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_lam.set_xlabel("Step", fontsize=10)
    ax_lam.set_ylabel(r"$\lambda_t$", fontsize=11)
    ax_lam.grid(True, alpha=0.2, linewidth=0.5)
    ax_lam.tick_params(labelsize=9)

    # ------------------------------------------------------------------
    # (b) Gemini-Pro selection fraction
    # ------------------------------------------------------------------
    ax_mix = axes[1]

    def _plot_gemini_fraction(
        ax: plt.Axes,
        cond_key: str,
        color: str,
        label: str,
        linestyle: str = "-",
        linewidth: float = 2.2,
        alpha_fill: float = 0.18,
        zorder_line: int = 4,
        zorder_fill: int = 2,
    ) -> None:
        curve = conditions[cond_key]["curves"]
        steps = [c["step"] for c in curve]
        fracs = [c["arm_fractions"].get("Gemini-Pro", 0.0) for c in curve]

        has_per_seed = "per_seed_arm_fractions" in curve[0]
        if has_per_seed:
            matrix = np.array([
                c["per_seed_arm_fractions"]["Gemini-Pro"] for c in curve
            ])
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            ses = [
                c["arm_fractions_std"].get("Gemini-Pro", 0.0) / sqrt_n
                for c in curve
            ]
            ci_lo = [m - s for m, s in zip(fracs, ses)]
            ci_hi = [m + s for m, s in zip(fracs, ses)]

        ax.plot(
            steps, fracs,
            color=color, linestyle=linestyle, linewidth=linewidth,
            label=label, zorder=zorder_line,
        )
        ax.fill_between(
            steps, ci_lo, ci_hi,
            alpha=alpha_fill, color=color, zorder=zorder_fill,
        )

    for blabel in budget_labels:
        cond_key = _find_condition_key(conditions, "BanditGPT", blabel)
        if cond_key is None:
            continue
        _plot_gemini_fraction(
            ax_mix, cond_key, BUDGET_COLORS[blabel],
            label=BUDGET_NICE_LABELS[blabel],
        )

    if "Unconstrained" in conditions:
        _plot_gemini_fraction(
            ax_mix, "Unconstrained", UNCONSTRAINED_COLOR,
            label="Unconstrained",
            linestyle="-.", linewidth=2.0,
            alpha_fill=0.12, zorder_line=3,
        )

    _add_phase_boundary(ax_mix, phase_boundary, label=False)
    ax_mix.set_title(
        "(b) Gemini-Pro Selection Fraction",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_mix.set_xlabel("Step", fontsize=10)
    ax_mix.set_ylabel("Fraction", fontsize=11)
    ax_mix.set_ylim(-0.02, 1.02)
    ax_mix.grid(True, alpha=0.2, linewidth=0.5)
    ax_mix.tick_params(labelsize=9)

    # ------------------------------------------------------------------
    # (c) Running average cost per request
    # ------------------------------------------------------------------
    ax_cost = axes[2]
    for blabel, btarget in zip(budget_labels, budget_targets):
        result = _extract_curve_with_ci(
            conditions, "BanditGPT", blabel,
            mean_field="mean_avg_cost",
            per_seed_field="per_seed_avg_cost",
            std_field="std_avg_cost",
            sqrt_n=sqrt_n,
        )
        if result is None:
            continue
        steps, avg_costs, ci_lo, ci_hi = result
        color = BUDGET_COLORS[blabel]

        ax_cost.plot(
            steps, avg_costs,
            color=color, linewidth=2.2, label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_cost.fill_between(
            steps, ci_lo, ci_hi,
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

        has_per_seed = "per_seed_avg_cost" in uc_curve[0]
        if has_per_seed:
            matrix = np.array([c["per_seed_avg_cost"] for c in uc_curve])
            uc_ci_lo, uc_ci_hi = bootstrap_ci_series(matrix)
        else:
            uc_ses = [c["std_avg_cost"] / sqrt_n for c in uc_curve]
            uc_ci_lo = [m - s for m, s in zip(uc_costs, uc_ses)]
            uc_ci_hi = [m + s for m, s in zip(uc_costs, uc_ses)]

        ax_cost.plot(
            uc_steps, uc_costs,
            color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
            label="Unconstrained", zorder=3,
        )
        ax_cost.fill_between(
            uc_steps, uc_ci_lo, uc_ci_hi,
            alpha=0.12, color=UNCONSTRAINED_COLOR, zorder=2,
        )

    _add_phase_boundary(ax_cost, phase_boundary, label=False)
    ax_cost.set_title(
        "(c) Running Avg Cost / Request",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_cost.set_xlabel("Step", fontsize=10)
    ax_cost.set_ylabel("$/request", fontsize=11)
    ax_cost.grid(True, alpha=0.2, linewidth=0.5)
    ax_cost.tick_params(labelsize=9)

    fig.suptitle(
        r"Budget Pacing Under Cost Drift ($K{=}3$, "
        rf"{n_seeds} seeds, 95% bootstrap CI)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    handles, labels = ax_mix.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.18)

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
