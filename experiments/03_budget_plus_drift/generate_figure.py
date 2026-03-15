#!/usr/bin/env python3
"""Generate figures for Experiment 03: Budget Pacing Under Cost Drift.

Reads ``results/budget_cost_drift_results.json`` and produces:

``adaptation_dynamics.pdf/.png``:
  Top row (1x3): BanditGPT-only adaptation mechanics
  (λ_t, Gemini fraction, running cost).  Bottom row: single-panel
  summary table with cost/ratio/reward/Gemini% for all conditions
  across all budget levels (tight, moderate, loose).

Usage:
    python experiments/03_budget_plus_drift/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "tight": "(d) Tight Budget",
    "moderate": "(e) Moderate Budget",
    "loose": "(f) Loose Budget",
}

UNCONSTRAINED_COLOR = "#009E73"

CONDITION_STYLES: Dict[str, Dict[str, Any]] = {
    "Fixed Policy": {
        "color": "#888888", "linestyle": ":", "linewidth": 1.8, "zorder": 2,
    },
    "Naive Bandit": {
        "color": "#E69F00", "linestyle": "--", "linewidth": 1.8, "zorder": 3,
    },
    "Recalibrated": {
        "color": "#56B4E9", "linestyle": "-.", "linewidth": 2.0, "zorder": 4,
    },
    "BanditGPT": {
        "color": "#D55E00", "linestyle": "-", "linewidth": 2.5, "zorder": 5,
    },
    "Unconstrained": {
        "color": UNCONSTRAINED_COLOR, "linestyle": "-.", "linewidth": 1.5,
        "zorder": 2,
    },
}

CONDITION_ORDER: List[str] = [
    "Fixed Policy", "Naive Bandit", "Recalibrated", "BanditGPT",
    "Unconstrained",
]


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


def _extract_curve(
    conditions: Dict[str, Any],
    prefix: str,
    budget_label: str,
    field: str,
    sqrt_n: float,
    std_field: Optional[str] = None,
) -> Optional[Tuple[List[int], List[float], List[float]]]:
    """Extract (steps, means, SEs) for a condition's checkpoint curve."""
    key = _find_condition_key(conditions, prefix, budget_label)
    if key is None:
        return None
    curve = conditions[key]["curves"]
    steps = [c["step"] for c in curve]
    means = [c[field] for c in curve]
    if std_field is not None:
        ses = [c[std_field] / sqrt_n for c in curve]
    else:
        ses = [0.0] * len(means)
    return steps, means, ses


_BUDGET_TABLE_LABELS: Dict[str, str] = {
    "tight": r"Tight ($2.3\times10^{-4}$)",
    "moderate": r"Moderate ($6.6\times10^{-4}$)",
    "loose": r"Loose ($1.9\times10^{-3}$)",
}

_TABLE_CONDITIONS: List[str] = [
    "Fixed Policy", "Naive Bandit", "Recalibrated", "BanditGPT",
]

_COMPLIANCE_COLOR = "#D5ECD4"
_HEADER_COLOR = "#E8E8E8"
_BANDITGPT_ROW_COLOR = "#FFF3E0"


def _fmt_cost(cost: float) -> str:
    """Format a dollar cost in compact scientific notation."""
    return f"${cost:.2e}"


def _fmt_ratio(ratio: float) -> str:
    return f"{ratio:.2f}x"


def _fmt_reward(reward: float) -> str:
    return f"{reward:.3f}"


def _fmt_pct(frac: float) -> str:
    return f"{frac:.0%}"


def _build_summary_table(
    data: Dict[str, Any],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Build cell text and cell colour arrays for the summary table.

    Returns ``(cell_text, cell_colours)`` where each is a list-of-rows
    and each row is a list of column strings / hex colours.
    """
    budget_labels = data["budget_labels"]
    budget_targets = data["budget_targets"]
    conditions = data["conditions"]

    n_cols = 10
    white = "#FFFFFF"
    cell_text: List[List[str]] = []
    cell_colours: List[List[str]] = []

    for blabel, btarget in zip(budget_labels, budget_targets):
        for i, cond_prefix in enumerate(_TABLE_CONDITIONS):
            key = _find_condition_key(conditions, cond_prefix, blabel)
            if key is None:
                cell_text.append([""] * n_cols)
                cell_colours.append([white] * n_cols)
                continue

            cond = conditions[key]
            p1 = cond["phase1_summary"]
            p2 = cond["phase2_summary"]

            p1_ratio = p1["mean_cost"] / btarget
            p2_ratio = p2["mean_cost"] / btarget
            p1_gem = p1["arm_fractions"].get("Gemini-Pro", 0.0)
            p2_gem = p2["arm_fractions"].get("Gemini-Pro", 0.0)

            budget_cell = _BUDGET_TABLE_LABELS[blabel] if i == 0 else ""

            row = [
                budget_cell,
                cond_prefix,
                _fmt_cost(p1["mean_cost"]),
                _fmt_ratio(p1_ratio),
                _fmt_cost(p2["mean_cost"]),
                _fmt_ratio(p2_ratio),
                _fmt_reward(p1["mean_reward"]),
                _fmt_reward(p2["mean_reward"]),
                _fmt_pct(p1_gem),
                _fmt_pct(p2_gem),
            ]

            is_bandit = cond_prefix == "BanditGPT"
            base = _BANDITGPT_ROW_COLOR if is_bandit else white
            colours = [base] * n_cols

            if abs(p1_ratio - 1.0) <= 0.05:
                colours[3] = _COMPLIANCE_COLOR
            if abs(p2_ratio - 1.0) <= 0.05:
                colours[5] = _COMPLIANCE_COLOR

            cell_text.append(row)
            cell_colours.append(colours)

    return cell_text, cell_colours


def plot_adaptation_dynamics(data: Dict[str, Any]) -> plt.Figure:
    """Top row: 1x3 BanditGPT dynamics.  Bottom: summary table.

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

    gs = plt.GridSpec(
        2, 3,
        height_ratios=[1.0, 0.7],
        hspace=0.35,
    )
    fig = plt.figure(figsize=(17, 9.0))

    # ==================================================================
    # Top row: BanditGPT-only dynamics
    # ==================================================================

    ax_lam = fig.add_subplot(gs[0, 0])
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

    ax_mix = fig.add_subplot(gs[0, 1])
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

    ax_cost = fig.add_subplot(gs[0, 2])
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

    # ==================================================================
    # Bottom row: Summary table spanning all 3 columns
    # ==================================================================

    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    col_labels = [
        "Budget", "Condition",
        "P1 Cost", "P1 Ratio", "P2 Cost", "P2 Ratio",
        "P1 Reward", "P2 Reward",
        "P1 Gemini%", "P2 Gemini%",
    ]

    cell_text, cell_colours = _build_summary_table(data)

    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        colColours=[_HEADER_COLOR] * len(col_labels),
        cellLoc="center",
        loc="upper center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=9)
        else:
            data_row = row - 1
            if data_row < len(cell_text):
                cond_name = cell_text[data_row][1]
                if cond_name == "BanditGPT":
                    cell.set_text_props(fontweight="bold")

    col_widths = [0.14, 0.11, 0.09, 0.08, 0.09, 0.08, 0.09, 0.09, 0.09, 0.09]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(len(cell_text) + 1):
            tbl[(row_idx, col_idx)].set_width(w)

    ax_table.text(
        0.5, 1.02,
        "(d) Budget Compliance Summary — All Conditions",
        transform=ax_table.transAxes,
        fontsize=11, fontweight="bold", ha="center", va="bottom",
    )

    fig.suptitle(
        r"Budget Pacing Under Cost Drift ($K{=}3$, "
        rf"{n_seeds} seeds, $\pm$1 SE)",
        fontsize=13, fontweight="bold", y=1.01,
    )
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
