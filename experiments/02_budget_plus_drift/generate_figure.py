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
NAIVE_COLOR = "#999999"
FORGET_COLOR = "#E69F00"
BASELINE_BUDGET_LABEL = "moderate"

_PHASE_LABELS: List[str] = ["Normal", "Price Drop", "Restored"]
_PHASE2_SHADE_COLOR = "#D55E00"


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
    font_scale: float = 1.0,
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
            fontsize=10 * font_scale, fontweight="bold", color="#333333",
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
# Budget-target label placement with collision avoidance
# ======================================================================


def _place_budget_target_labels(
    ax: plt.Axes,
    labels: List[Tuple[float, str, str]],
    font_scale: float = 1.0,
    label_fontsize_base: float = 9.0,
    min_sep_frac: float = 0.045,
) -> None:
    """Place budget-target labels to the right of the axes, spreading overlapping ones.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to place the labels.
    labels : list of (y_data, text_label, color)
        Each entry gives the y-position in data coords, the label string,
        and the colour for the label.
    font_scale : float
        Multiplicative font scaling factor.
    label_fontsize_base : float
        Base font size before scaling (default 9.0).
    min_sep_frac : float
        Minimum vertical separation between label centres, expressed as a
        fraction of the y-axis range.
    """
    if not labels:
        return

    fs = font_scale
    sorted_labels = sorted(labels, key=lambda x: x[0])
    y_lo, y_hi = ax.get_ylim()
    min_sep = min_sep_frac * (y_hi - y_lo)

    adj_y = [entry[0] for entry in sorted_labels]
    for i in range(1, len(adj_y)):
        if adj_y[i] - adj_y[i - 1] < min_sep:
            mid = (adj_y[i] + adj_y[i - 1]) / 2
            adj_y[i - 1] = mid - min_sep / 2
            adj_y[i] = mid + min_sep / 2

    for (_, blabel, color), y_pos in zip(sorted_labels, adj_y):
        ax.text(
            1.01,
            y_pos,
            f"{blabel} target",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            fontsize=label_fontsize_base * fs,
            color=color,
            va="center",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )


# ======================================================================
# Main figure: 3x1 stacked adaptation dynamics
# ======================================================================

GEMINI_ARM_SHORT = "Gemini-Pro"


def plot_adaptation_dynamics(
    data: Dict[str, Any],
    figsize: Tuple[float, float] = (8, 12),
    font_scale: float = 1.0,
) -> plt.Figure:
    """3x1 stacked figure: (a) Gemini-Pro fraction, (b) windowed reward, (c) cost/request.

    Parameters
    ----------
    data : dict
        Parsed results JSON.
    figsize : tuple of float
        Figure width and height in inches.
    font_scale : float
        Multiplicative factor applied to all explicit font sizes (default 1.0).

    Returns
    -------
    plt.Figure
    """
    fs = font_scale
    budget_labels = data["budget_labels"]
    budget_targets = data["budget_targets"]
    conditions = data["conditions"]
    phase_boundaries = data["phase_boundaries"]
    n_seeds = data["n_seeds"]
    sqrt_n = np.sqrt(n_seeds)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

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

    naive_gem_key = _find_condition_key(conditions, "Naive Bandit", BASELINE_BUDGET_LABEL)
    if naive_gem_key is not None:
        steps_ng, fracs_ng, ci_lo_ng, ci_hi_ng = _extract_arm_fraction_with_ci(
            conditions, naive_gem_key, GEMINI_ARM_SHORT, sqrt_n,
        )
        ax_gem.plot(
            steps_ng, fracs_ng,
            color=NAIVE_COLOR, linestyle="--", linewidth=1.8,
            label=f"Naive Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_gem.fill_between(
            steps_ng, ci_lo_ng, ci_hi_ng,
            alpha=0.10, color=NAIVE_COLOR, zorder=2,
        )

    forget_gem_key = _find_condition_key(conditions, "Forgetting Bandit", BASELINE_BUDGET_LABEL)
    if forget_gem_key is not None:
        steps_fg, fracs_fg, ci_lo_fg, ci_hi_fg = _extract_arm_fraction_with_ci(
            conditions, forget_gem_key, GEMINI_ARM_SHORT, sqrt_n,
        )
        ax_gem.plot(
            steps_fg, fracs_fg,
            color=FORGET_COLOR, linestyle="--", linewidth=1.8,
            label=f"Forgetting Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_gem.fill_between(
            steps_fg, ci_lo_fg, ci_hi_fg,
            alpha=0.10, color=FORGET_COLOR, zorder=2,
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

    _add_phase_shading(ax_gem, phase_boundaries, font_scale=fs)

    ax_gem.set_title(
        "(a) Gemini-Pro Selection Fraction",
        fontsize=12 * fs, fontweight="bold", pad=10,
    )
    ax_gem.set_ylabel("Fraction", fontsize=12 * fs)
    ax_gem.set_ylim(-0.02, 1.02)
    ax_gem.grid(True, alpha=0.2, linewidth=0.5)
    ax_gem.tick_params(labelsize=10 * fs, labelbottom=False)

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

    naive_rwd_result = _extract_curve_with_ci(
        conditions, "Naive Bandit", BASELINE_BUDGET_LABEL,
        mean_field="mean_window_reward",
        per_seed_field="per_seed_window_reward",
        std_field="std_window_reward",
        sqrt_n=sqrt_n,
    )
    if naive_rwd_result is not None:
        steps_n, rewards_n, ci_lo_n, ci_hi_n = naive_rwd_result
        ax_rwd.plot(
            steps_n, rewards_n,
            color=NAIVE_COLOR, linestyle="--", linewidth=1.8,
            label=f"Naive Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_rwd.fill_between(
            steps_n, ci_lo_n, ci_hi_n,
            alpha=0.10, color=NAIVE_COLOR, zorder=2,
        )

    forget_rwd_result = _extract_curve_with_ci(
        conditions, "Forgetting Bandit", BASELINE_BUDGET_LABEL,
        mean_field="mean_window_reward",
        per_seed_field="per_seed_window_reward",
        std_field="std_window_reward",
        sqrt_n=sqrt_n,
    )
    if forget_rwd_result is not None:
        steps_f, rewards_f, ci_lo_f, ci_hi_f = forget_rwd_result
        ax_rwd.plot(
            steps_f, rewards_f,
            color=FORGET_COLOR, linestyle="--", linewidth=1.8,
            label=f"Forgetting Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_rwd.fill_between(
            steps_f, ci_lo_f, ci_hi_f,
            alpha=0.10, color=FORGET_COLOR, zorder=2,
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

    _add_phase_shading(ax_rwd, phase_boundaries, font_scale=fs)

    ax_rwd.set_title(
        "(b) Windowed Mean Reward",
        fontsize=12 * fs, fontweight="bold", pad=10,
    )
    ax_rwd.set_ylabel("Mean Reward", fontsize=12 * fs)
    ax_rwd.grid(True, alpha=0.2, linewidth=0.5)
    ax_rwd.tick_params(labelsize=10 * fs, labelbottom=False)

    # ------------------------------------------------------------------
    # (c) Trailing-window average cost per request (last 50 steps)
    # ------------------------------------------------------------------
    ax_cost = axes[2]
    _target_labels: List[Tuple[float, str, str]] = []
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
        _target_labels.append((btarget, blabel, color))

    naive_cost_result = _extract_curve_with_ci(
        conditions, "Naive Bandit", BASELINE_BUDGET_LABEL,
        mean_field="mean_window_cost",
        per_seed_field="per_seed_window_cost",
        std_field="std_window_cost",
        sqrt_n=sqrt_n,
    )
    if naive_cost_result is not None:
        steps_nc, costs_nc, ci_lo_nc, ci_hi_nc = naive_cost_result
        ax_cost.plot(
            steps_nc, costs_nc,
            color=NAIVE_COLOR, linestyle="--", linewidth=1.8,
            label=f"Naive Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_cost.fill_between(
            steps_nc, ci_lo_nc, ci_hi_nc,
            alpha=0.10, color=NAIVE_COLOR, zorder=2,
        )

    forget_cost_result = _extract_curve_with_ci(
        conditions, "Forgetting Bandit", BASELINE_BUDGET_LABEL,
        mean_field="mean_window_cost",
        per_seed_field="per_seed_window_cost",
        std_field="std_window_cost",
        sqrt_n=sqrt_n,
    )
    if forget_cost_result is not None:
        steps_fc, costs_fc, ci_lo_fc, ci_hi_fc = forget_cost_result
        ax_cost.plot(
            steps_fc, costs_fc,
            color=FORGET_COLOR, linestyle="--", linewidth=1.8,
            label=f"Forgetting Bandit ({BASELINE_BUDGET_LABEL})", zorder=3,
        )
        ax_cost.fill_between(
            steps_fc, ci_lo_fc, ci_hi_fc,
            alpha=0.10, color=FORGET_COLOR, zorder=2,
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

    _add_phase_shading(ax_cost, phase_boundaries, font_scale=fs)

    _place_budget_target_labels(ax_cost, _target_labels, font_scale=fs)

    ax_cost.set_title(
        "(c) Windowed Avg Cost / Request",
        fontsize=12 * fs,
        fontweight="bold",
        pad=10,
    )
    ax_cost.set_xlabel("Prompts Routed", fontsize=11 * fs)
    ax_cost.set_ylabel("$/request", fontsize=12 * fs)
    ax_cost.grid(True, alpha=0.2, linewidth=0.5)
    ax_cost.tick_params(labelsize=10 * fs)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    ax_cost.set_xlabel("Prompts Routed", fontsize=11 * fs)

    handles, labels = ax_gem.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        fontsize=10.0 * fs,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1.0])
    fig.subplots_adjust(hspace=0.15)

    return fig


# ======================================================================
# Slide-ready single-panel figures (16:9 widescreen)
# ======================================================================

_SLIDE_FS = 1.8
_SLIDE_LW = 3.0
_SLIDE_CI_ALPHA = 0.20
_SLIDE_FIGSIZE = (13, 5.5)


def _slide_legend(ax: plt.Axes, fs: float) -> None:
    """Place legend inside the axes at upper-right."""
    ax.legend(fontsize=9 * fs, framealpha=0.9, loc="upper right")


def plot_slide_gemini_fraction(data: Dict[str, Any]) -> plt.Figure:
    """Slide panel: Gemini-Pro selection fraction over time."""
    fs = _SLIDE_FS
    conditions = data["conditions"]
    phase_boundaries = data["phase_boundaries"]
    budget_labels = data["budget_labels"]
    sqrt_n = np.sqrt(data["n_seeds"])

    fig, ax = plt.subplots(figsize=_SLIDE_FIGSIZE)

    for blabel in budget_labels:
        cond_key = _find_condition_key(conditions, "ParetoBandit", blabel)
        if cond_key is None:
            continue
        steps, fracs, ci_lo, ci_hi = _extract_arm_fraction_with_ci(
            conditions, cond_key, GEMINI_ARM_SHORT, sqrt_n,
        )
        color = BUDGET_COLORS[blabel]
        ax.plot(steps, fracs, color=color, linewidth=_SLIDE_LW,
                label=BUDGET_NICE_LABELS[blabel], zorder=4)
        ax.fill_between(steps, ci_lo, ci_hi,
                        alpha=_SLIDE_CI_ALPHA, color=color, zorder=2)

    if "Unconstrained" in conditions:
        steps, fracs, ci_lo, ci_hi = _extract_arm_fraction_with_ci(
            conditions, "Unconstrained", GEMINI_ARM_SHORT, sqrt_n,
        )
        ax.plot(steps, fracs, color=UNCONSTRAINED_COLOR, linestyle="-.",
                linewidth=_SLIDE_LW, label=r"Unconstrained ($\lambda_s{=}0$)",
                zorder=3)
        ax.fill_between(steps, ci_lo, ci_hi,
                        alpha=_SLIDE_CI_ALPHA * 0.6, color=UNCONSTRAINED_COLOR,
                        zorder=2)

    _add_phase_shading(ax, phase_boundaries, font_scale=fs)
    ax.set_title("Gemini-Pro Selection Fraction",
                 fontsize=14 * fs, fontweight="bold", pad=12)
    ax.set_ylabel("Fraction", fontsize=12 * fs)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10 * fs)
    fig.tight_layout()
    return fig


def plot_slide_reward(data: Dict[str, Any]) -> plt.Figure:
    """Slide panel: windowed mean reward over time."""
    fs = _SLIDE_FS
    conditions = data["conditions"]
    phase_boundaries = data["phase_boundaries"]
    budget_labels = data["budget_labels"]
    sqrt_n = np.sqrt(data["n_seeds"])

    fig, ax = plt.subplots(figsize=_SLIDE_FIGSIZE)

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
        ax.plot(steps, rewards, color=color, linewidth=_SLIDE_LW,
                label=BUDGET_NICE_LABELS[blabel], zorder=4)
        ax.fill_between(steps, ci_lo, ci_hi,
                        alpha=_SLIDE_CI_ALPHA, color=color, zorder=2)

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
            ax.plot(steps, rewards, color=UNCONSTRAINED_COLOR, linestyle="-.",
                    linewidth=_SLIDE_LW,
                    label=r"Unconstrained ($\lambda_s{=}0$)", zorder=3)
            ax.fill_between(steps, ci_lo, ci_hi,
                            alpha=_SLIDE_CI_ALPHA * 0.6,
                            color=UNCONSTRAINED_COLOR, zorder=2)

    _add_phase_shading(ax, phase_boundaries, font_scale=fs)
    ax.set_title("Windowed Mean Reward",
                 fontsize=14 * fs, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Reward", fontsize=12 * fs)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10 * fs)
    fig.tight_layout()
    return fig


def plot_slide_cost(data: Dict[str, Any]) -> plt.Figure:
    """Slide panel: windowed average cost per request over time."""
    fs = _SLIDE_FS
    conditions = data["conditions"]
    phase_boundaries = data["phase_boundaries"]
    budget_labels = data["budget_labels"]
    budget_targets = data["budget_targets"]
    sqrt_n = np.sqrt(data["n_seeds"])

    fig, ax = plt.subplots(figsize=_SLIDE_FIGSIZE)

    _slide_target_labels: List[Tuple[float, str, str]] = []
    for blabel, btarget in zip(budget_labels, budget_targets):
        result = _extract_curve_with_ci(
            conditions, "ParetoBandit", blabel,
            mean_field="mean_window_cost",
            per_seed_field="per_seed_window_cost",
            std_field="std_window_cost",
            sqrt_n=sqrt_n,
        )
        if result is None:
            continue
        steps, avg_costs, ci_lo, ci_hi = result
        color = BUDGET_COLORS[blabel]
        ax.plot(steps, avg_costs, color=color, linewidth=_SLIDE_LW,
                label=BUDGET_NICE_LABELS[blabel], zorder=4)
        ax.fill_between(steps, ci_lo, ci_hi,
                        alpha=_SLIDE_CI_ALPHA, color=color, zorder=2)
        ax.axhline(btarget, color=color, linestyle=":", linewidth=2.0,
                   alpha=0.6, zorder=1)
        _slide_target_labels.append((btarget, blabel, color))

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
            ax.plot(uc_steps, uc_costs, color=UNCONSTRAINED_COLOR,
                    linestyle="-.", linewidth=_SLIDE_LW,
                    label=r"Unconstrained ($\lambda_s{=}0$)", zorder=3)
            ax.fill_between(uc_steps, uc_ci_lo, uc_ci_hi,
                            alpha=_SLIDE_CI_ALPHA * 0.6,
                            color=UNCONSTRAINED_COLOR, zorder=2)

    _add_phase_shading(ax, phase_boundaries, font_scale=fs)
    _place_budget_target_labels(
        ax, _slide_target_labels, font_scale=fs, label_fontsize_base=9.0,
    )
    ax.set_title("Windowed Avg Cost / Request",
                 fontsize=14 * fs, fontweight="bold", pad=12)
    ax.set_ylabel("$/request", fontsize=12 * fs)
    ax.set_xlabel("Prompts Routed", fontsize=12 * fs)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10 * fs)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        fontsize=9 * fs,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1.0])
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
