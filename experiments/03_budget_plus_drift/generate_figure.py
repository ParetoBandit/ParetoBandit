#!/usr/bin/env python3
"""Generate figures for Experiment 03: Budget Pacing Under Cost Drift (Three-Phase).

Reads ``results/budget_cost_drift_results.json`` and produces:

``adaptation_dynamics.pdf/.png``:
  1x3 panel showing adaptation dynamics across three phases
  (normal → price drop → price restored) for ParetoBandit at each budget level.

  (a) Dual Variable λ_t — shows whether the pacer re-raises λ in Phase 3.
  (b) Gemini-Pro Selection Fraction — shows whether Gemini usage drops back.
  (c) Running Avg Cost / Request — shows budget compliance recovery.

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
from matplotlib.transforms import blended_transform_factory

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
    "tight": r"Tight ($B{=}\$3.0{\times}10^{-4}$)",
    "moderate": r"Moderate ($B{=}\$6.6{\times}10^{-4}$)",
    "loose": r"Loose ($B{=}\$1.9{\times}10^{-3}$)",
}

UNCONSTRAINED_COLOR = "#009E73"

_PHASE_LABELS: List[str] = ["P1: Normal", "P2: Price Drop", "P3: Restored"]
_PHASE2_SHADE_COLOR = "#0072B2"


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


def _add_phase_shading(
    ax: plt.Axes,
    boundaries: List[int],
) -> None:
    """Shade Phase 2 (price-drop perturbation) and label all three phases.

    Uses a blended transform (data-x, axes-y) so labels sit at a
    consistent vertical position regardless of y-axis scale.
    """
    ax.axvspan(
        boundaries[0], boundaries[1],
        alpha=0.07, color=_PHASE2_SHADE_COLOR, zorder=0,
    )
    for b in boundaries[:-1]:
        ax.axvline(
            b, color="black", linestyle="--",
            linewidth=1.2, alpha=0.5, zorder=1,
        )
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    midpoints = [
        boundaries[0] / 2,
        (boundaries[0] + boundaries[1]) / 2,
        (boundaries[1] + boundaries[2]) / 2,
    ]
    for mid, label in zip(midpoints, _PHASE_LABELS):
        ax.text(
            mid, 0.97, label,
            transform=trans, ha="center", va="top",
            fontsize=8, fontstyle="italic", color="#555555",
        )


# ======================================================================
# Curve extraction with bootstrap CI
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
    """Extract (steps, means, ci_lo, ci_hi) for a condition's checkpoint curve."""
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


# ======================================================================
# Main figure: 1x3 adaptation dynamics
# ======================================================================


def plot_adaptation_dynamics(data: Dict[str, Any]) -> plt.Figure:
    """1x3 figure showing ParetoBandit adaptation across 3 cost phases.

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
    phase_boundaries = data["phase_boundaries"]
    n_seeds = data["n_seeds"]
    sqrt_n = np.sqrt(n_seeds)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # ------------------------------------------------------------------
    # (a) Dual variable λ_t
    # ------------------------------------------------------------------
    ax_lam = axes[0]
    for blabel in budget_labels:
        result = _extract_curve_with_ci(
            conditions,
            "ParetoBandit",
            blabel,
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
            steps,
            lambdas,
            color=color,
            linewidth=2.2,
            label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_lam.fill_between(
            steps, ci_lo, ci_hi, alpha=0.15, color=color, zorder=2
        )

    _add_phase_shading(ax_lam, phase_boundaries)

    ax_lam.set_title(
        r"(a) Dual Variable $\lambda_t$",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_lam.set_xlabel("Prompts Routed", fontsize=11)
    ax_lam.set_ylabel(r"$\lambda_t$", fontsize=12)
    ax_lam.grid(True, alpha=0.2, linewidth=0.5)
    ax_lam.tick_params(labelsize=10)

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
        alpha_fill: float = 0.15,
        zorder_line: int = 4,
        zorder_fill: int = 2,
    ) -> None:
        curve = conditions[cond_key]["curves"]
        steps = [c["step"] for c in curve]
        fracs = [c["arm_fractions"].get("Gemini-Pro", 0.0) for c in curve]

        has_per_seed = "per_seed_arm_fractions" in curve[0]
        if has_per_seed:
            matrix = np.array(
                [c["per_seed_arm_fractions"]["Gemini-Pro"] for c in curve]
            )
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            ses = [
                c["arm_fractions_std"].get("Gemini-Pro", 0.0) / sqrt_n
                for c in curve
            ]
            ci_lo = [m - s for m, s in zip(fracs, ses)]
            ci_hi = [m + s for m, s in zip(fracs, ses)]

        ax.plot(
            steps,
            fracs,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=zorder_line,
        )
        ax.fill_between(
            steps, ci_lo, ci_hi, alpha=alpha_fill, color=color, zorder=zorder_fill
        )

    for blabel in budget_labels:
        cond_key = _find_condition_key(conditions, "ParetoBandit", blabel)
        if cond_key is None:
            continue
        _plot_gemini_fraction(
            ax_mix,
            cond_key,
            BUDGET_COLORS[blabel],
            label=BUDGET_NICE_LABELS[blabel],
        )

    if "Unconstrained" in conditions:
        _plot_gemini_fraction(
            ax_mix,
            "Unconstrained",
            UNCONSTRAINED_COLOR,
            label=r"Unconstrained ($\lambda_s{=}0$)",
            linestyle="-.",
            linewidth=2.0,
            alpha_fill=0.10,
            zorder_line=3,
        )

    _add_phase_shading(ax_mix, phase_boundaries)

    ax_mix.set_title(
        "(b) Gemini-Pro Selection Fraction",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_mix.set_xlabel("Prompts Routed", fontsize=11)
    ax_mix.set_ylabel("Fraction", fontsize=12)
    ax_mix.set_ylim(-0.02, 1.02)
    ax_mix.grid(True, alpha=0.2, linewidth=0.5)
    ax_mix.tick_params(labelsize=10)

    # ------------------------------------------------------------------
    # (c) Trailing-window average cost per request (last 50 steps)
    # ------------------------------------------------------------------
    ax_cost = axes[2]
    for blabel, btarget in zip(budget_labels, budget_targets):
        result = _extract_curve_with_ci(
            conditions,
            "ParetoBandit",
            blabel,
            mean_field="mean_window_cost",
            per_seed_field="per_seed_window_cost",
            std_field="std_window_cost",
            sqrt_n=sqrt_n,
        )
        if result is None:
            continue
        steps, avg_costs, ci_lo, ci_hi = result
        color = BUDGET_COLORS[blabel]

        ax_cost.plot(
            steps,
            avg_costs,
            color=color,
            linewidth=2.2,
            label=BUDGET_NICE_LABELS[blabel],
            zorder=4,
        )
        ax_cost.fill_between(
            steps, ci_lo, ci_hi, alpha=0.15, color=color, zorder=2
        )
        ax_cost.axhline(
            btarget,
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.6,
            zorder=1,
        )
        ax_cost.annotate(
            f"  {blabel} target",
            xy=(steps[-1], btarget),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="center",
            ha="left",
            fontweight="bold",
        )

    if "Unconstrained" in conditions:
        uc_curve = conditions["Unconstrained"]["curves"]
        uc_steps = [c["step"] for c in uc_curve]
        uc_costs = [c["mean_window_cost"] for c in uc_curve]

        has_per_seed = "per_seed_window_cost" in uc_curve[0]
        if has_per_seed:
            matrix = np.array([c["per_seed_window_cost"] for c in uc_curve])
            uc_ci_lo, uc_ci_hi = bootstrap_ci_series(matrix)
        else:
            uc_ses = [c["std_window_cost"] / sqrt_n for c in uc_curve]
            uc_ci_lo = [m - s for m, s in zip(uc_costs, uc_ses)]
            uc_ci_hi = [m + s for m, s in zip(uc_costs, uc_ses)]

        ax_cost.plot(
            uc_steps,
            uc_costs,
            color=UNCONSTRAINED_COLOR,
            linestyle="-.",
            linewidth=2.0,
            label=r"Unconstrained ($\lambda_s{=}0$)",
            zorder=3,
        )
        ax_cost.fill_between(
            uc_steps,
            uc_ci_lo,
            uc_ci_hi,
            alpha=0.10,
            color=UNCONSTRAINED_COLOR,
            zorder=2,
        )

    _add_phase_shading(ax_cost, phase_boundaries)

    ax_cost.set_title(
        "(c) Windowed Avg Cost / Request",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_cost.set_xlabel("Prompts Routed", fontsize=11)
    ax_cost.set_ylabel("$/request", fontsize=12)
    ax_cost.grid(True, alpha=0.2, linewidth=0.5)
    ax_cost.tick_params(labelsize=10)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.suptitle(
        r"Cost Correction: Normal $\to$ Price Drop $\to$ Restored "
        rf"($K{{=}}3$, {n_seeds} seeds, 95% bootstrap CI)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    handles, labels = ax_mix.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    return fig


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()

    fig = plot_adaptation_dynamics(data)
    for fmt in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"adaptation_dynamics.{fmt}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)
    print("Saved adaptation_dynamics.{pdf,png}")


if __name__ == "__main__":
    main()
