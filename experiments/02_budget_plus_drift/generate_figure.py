#!/usr/bin/env python3
"""Generate figures for Experiment 02: Budget Pacing Under Cost Drift (Three-Phase).

Reads ``results/budget_cost_drift_results.json`` and produces:

``adaptation_dynamics.pdf/.png``:
  3x1 stacked panel showing adaptation dynamics across three phases
  (normal → price drop → price restored) for ParetoBandit at each budget level.

  (a) Gemini-Pro Selection Fraction — shows how the pacer exploits the price drop.
  (b) Windowed Mean Reward — quality outcome through the cost drift.
  (c) Running Avg Cost / Request — shows budget compliance recovery.

Usage:
    python experiments/02_budget_plus_drift/generate_figure.py
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

_PHASE_LABELS: List[str] = ["Normal", "Price Drop", "Restored"]
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
            fontsize=10, fontweight="bold", color="#333333",
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
# Direct curve extraction (for conditions without prefix/label lookup)
# ======================================================================


def _extract_curve_with_ci_direct(
    curve: List[Dict[str, Any]],
    mean_field: str,
    per_seed_field: Optional[str] = None,
    std_field: Optional[str] = None,
    sqrt_n: float = 1.0,
) -> Optional[Tuple[List[int], List[float], np.ndarray, np.ndarray]]:
    """Extract (steps, means, ci_lo, ci_hi) from a raw curve list."""
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


def _extract_arm_fraction_with_ci(
    conditions: Dict[str, Any],
    cond_key: str,
    arm_short_name: str,
    sqrt_n: float,
) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    """Extract arm selection fraction with bootstrap CI."""
    curve = conditions[cond_key]["curves"]
    steps = [c["step"] for c in curve]
    fracs = [c["arm_fractions"].get(arm_short_name, 0.0) for c in curve]

    has_per_seed = "per_seed_arm_fractions" in curve[0]
    if has_per_seed:
        matrix = np.array(
            [c["per_seed_arm_fractions"][arm_short_name] for c in curve]
        )
        ci_lo, ci_hi = bootstrap_ci_series(matrix)
    else:
        ses = [
            c["arm_fractions_std"].get(arm_short_name, 0.0) / sqrt_n
            for c in curve
        ]
        ci_lo = np.array([m - s for m, s in zip(fracs, ses)])
        ci_hi = np.array([m + s for m, s in zip(fracs, ses)])

    return steps, fracs, ci_lo, ci_hi


# ======================================================================
# Main figure: 3x1 stacked adaptation dynamics
# ======================================================================

GEMINI_ARM_SHORT = "Gemini-Pro"


def plot_adaptation_dynamics(data: Dict[str, Any]) -> plt.Figure:
    """3x1 stacked figure: (a) Gemini-Pro fraction, (b) windowed reward, (c) cost/request.

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

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # ------------------------------------------------------------------
    # (a) Gemini-Pro selection fraction
    # ------------------------------------------------------------------
    ax_gem = axes[0]

    for blabel in budget_labels:
        cond_key = _find_condition_key(conditions, "ParetoBandit", blabel)
        if cond_key is None:
            continue
        steps, fracs, ci_lo, ci_hi = _extract_arm_fraction_with_ci(
            conditions, cond_key, GEMINI_ARM_SHORT, sqrt_n,
        )
        color = BUDGET_COLORS[blabel]
        ax_gem.plot(
            steps, fracs,
            color=color, linewidth=2.2,
            label=BUDGET_NICE_LABELS[blabel], zorder=4,
        )
        ax_gem.fill_between(
            steps, ci_lo, ci_hi, alpha=0.15, color=color, zorder=2,
        )

    if "Unconstrained" in conditions:
        steps, fracs, ci_lo, ci_hi = _extract_arm_fraction_with_ci(
            conditions, "Unconstrained", GEMINI_ARM_SHORT, sqrt_n,
        )
        ax_gem.plot(
            steps, fracs,
            color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
            label=r"Unconstrained ($\lambda_s{=}0$)", zorder=3,
        )
        ax_gem.fill_between(
            steps, ci_lo, ci_hi,
            alpha=0.10, color=UNCONSTRAINED_COLOR, zorder=2,
        )

    _add_phase_shading(ax_gem, phase_boundaries)

    ax_gem.set_title(
        "(a) Gemini-Pro Selection Fraction",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_gem.set_ylabel("Fraction", fontsize=12)
    ax_gem.set_ylim(-0.02, 1.02)
    ax_gem.grid(True, alpha=0.2, linewidth=0.5)
    ax_gem.tick_params(labelsize=10)

    # ------------------------------------------------------------------
    # (b) Windowed mean reward (trailing 50 steps)
    # ------------------------------------------------------------------
    ax_rwd = axes[1]

    for blabel in budget_labels:
        result = _extract_curve_with_ci(
            conditions, "ParetoBandit", blabel,
            mean_field="mean_window_reward",
            per_seed_field="per_seed_window_reward",
            std_field="std_window_reward",
            sqrt_n=sqrt_n,
        )
        if result is None:
            continue
        steps, rewards, ci_lo, ci_hi = result
        color = BUDGET_COLORS[blabel]
        ax_rwd.plot(
            steps, rewards,
            color=color, linewidth=2.2,
            label=BUDGET_NICE_LABELS[blabel], zorder=4,
        )
        ax_rwd.fill_between(
            steps, ci_lo, ci_hi, alpha=0.15, color=color, zorder=2,
        )

    if "Unconstrained" in conditions:
        uc_result = _extract_curve_with_ci_direct(
            conditions["Unconstrained"]["curves"],
            mean_field="mean_window_reward",
            per_seed_field="per_seed_window_reward",
            std_field="std_window_reward",
            sqrt_n=sqrt_n,
        )
        if uc_result is not None:
            steps, rewards, ci_lo, ci_hi = uc_result
            ax_rwd.plot(
                steps, rewards,
                color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
                label=r"Unconstrained ($\lambda_s{=}0$)", zorder=3,
            )
            ax_rwd.fill_between(
                steps, ci_lo, ci_hi,
                alpha=0.10, color=UNCONSTRAINED_COLOR, zorder=2,
            )

    _add_phase_shading(ax_rwd, phase_boundaries)

    ax_rwd.set_title(
        "(b) Windowed Mean Reward",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_rwd.set_ylabel("Mean Reward", fontsize=12)
    ax_rwd.grid(True, alpha=0.2, linewidth=0.5)
    ax_rwd.tick_params(labelsize=10)

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
        ax_cost.text(
            1.01, btarget, f"{blabel} target",
            transform=blended_transform_factory(ax_cost.transAxes, ax_cost.transData),
            fontsize=7.5, color=color, va="center", ha="left",
            fontweight="bold", clip_on=False,
        )

    if "Unconstrained" in conditions:
        uc_result = _extract_curve_with_ci_direct(
            conditions["Unconstrained"]["curves"],
            mean_field="mean_window_cost",
            per_seed_field="per_seed_window_cost",
            std_field="std_window_cost",
            sqrt_n=sqrt_n,
        )
        if uc_result is not None:
            uc_steps, uc_costs, uc_ci_lo, uc_ci_hi = uc_result
            ax_cost.plot(
                uc_steps, uc_costs,
                color=UNCONSTRAINED_COLOR, linestyle="-.", linewidth=2.0,
                label=r"Unconstrained ($\lambda_s{=}0$)", zorder=3,
            )
            ax_cost.fill_between(
                uc_steps, uc_ci_lo, uc_ci_hi,
                alpha=0.10, color=UNCONSTRAINED_COLOR, zorder=2,
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
    ax_cost.set_xlabel("Prompts Routed", fontsize=11)

    handles, labels = ax_gem.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        fontsize=9.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.subplots_adjust(hspace=0.15)

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
