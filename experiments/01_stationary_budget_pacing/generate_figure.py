#!/usr/bin/env python3
"""Generate figures for Experiment 01: Stationary Budget Pacing.

Reads ``results/budget_pacing_results.json`` and produces two
publication-ready figures:

1. **Pareto frontier** (``pareto_frontier.{pdf,png}``):
   Quality (mean reward) vs. cost, comparing the BudgetPacer adaptive
   curve against the static ``cost_penalty`` baseline.

2. **Model-mix dynamics** (``model_mix.{pdf,png}``):
   Grouped bar chart of model selection fractions as a function of
   budget target / cost-penalty setting.

Usage:
    python experiments/01_stationary_budget_pacing/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci

# ======================================================================
# Paths
# ======================================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "budget_pacing_results.json"

# ======================================================================
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ======================================================================

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_RED = "#D55E00"
CB_GREEN = "#009E73"
CB_PURPLE = "#CC79A7"
CB_TEAL = "#56B4E9"
CB_GRAY = "#999999"

MODEL_COLORS: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": CB_TEAL,
    "mistralai/mistral-large-2512": CB_ORANGE,
    "google/gemini-2.5-pro": CB_BLUE,
}
MODEL_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-2.5-Pro",
}

# ======================================================================
# Helpers
# ======================================================================


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_FILE) as f:
        return json.load(f)


def _pareto_front(
    costs: List[float], rewards: List[float],
) -> tuple[List[float], List[float]]:
    """Return the upper-left Pareto frontier from (cost, reward) points.

    Sorts by cost ascending, then greedily keeps points whose reward
    exceeds the running maximum.
    """
    pairs = sorted(zip(costs, rewards), key=lambda p: p[0])
    front_c, front_r = [], []
    best = -np.inf
    for c, r in pairs:
        if r >= best - 1e-12:
            front_c.append(c)
            front_r.append(r)
            best = r
    return front_c, front_r


def _dollar_fmt(x: float, _pos: Any = None) -> str:
    """Format x as dollar string for tick labels."""
    if x >= 0.01:
        return f"${x:.3f}"
    if x >= 0.001:
        return f"${x:.4f}"
    return f"${x:.5f}"


# ======================================================================
# Figure 1: Pareto Frontier
# ======================================================================


def _ci_errorbars(
    rows: List[Dict[str, Any]],
    field: str,
) -> Tuple[List[float], List[float]]:
    """Compute asymmetric error bar half-widths from bootstrap CI or SE.

    Returns (lo_err, hi_err) suitable for ``ax.errorbar(..., yerr=...)``.
    """
    means = [r[f"mean_{field}"] for r in rows]
    per_seed_key = f"per_seed_{field}s"
    se_key = f"se_{field}"

    lo_errs: List[float] = []
    hi_errs: List[float] = []
    for i, r in enumerate(rows):
        if per_seed_key in r:
            ci_lo, ci_hi = bootstrap_ci(np.array(r[per_seed_key]))
            lo_errs.append(means[i] - ci_lo)
            hi_errs.append(ci_hi - means[i])
        else:
            se = r[se_key]
            lo_errs.append(se)
            hi_errs.append(se)
    return lo_errs, hi_errs


def plot_pareto(data: Dict[str, Any]) -> plt.Figure:
    """Quality-cost Pareto frontier: BudgetPacer vs. static baseline."""
    results = data["results"]
    arm_order = data["arm_order"]

    static = [r for r in results if r["method"] == "static"]
    pacer = [r for r in results if r["method"] == "pacer"]

    s_costs = [r["mean_cost"] for r in static]
    s_rewards = [r["mean_reward"] for r in static]
    s_err_r_lo, s_err_r_hi = _ci_errorbars(static, "reward")
    s_err_c_lo, s_err_c_hi = _ci_errorbars(static, "cost")

    p_costs = [r["mean_cost"] for r in pacer]
    p_rewards = [r["mean_reward"] for r in pacer]
    p_err_r_lo, p_err_r_hi = _ci_errorbars(pacer, "reward")
    p_err_c_lo, p_err_c_hi = _ci_errorbars(pacer, "cost")

    fig, ax = plt.subplots(figsize=(8, 6))

    sf_c, sf_r = _pareto_front(s_costs, s_rewards)
    pf_c, pf_r = _pareto_front(p_costs, p_rewards)

    ax.plot(
        sf_c, sf_r,
        color=CB_GRAY, linestyle="--", linewidth=2.0,
        marker="s", markersize=6, markerfacecolor="white",
        markeredgecolor=CB_GRAY, markeredgewidth=1.5,
        label="Static cost penalty", zorder=4,
    )
    ax.errorbar(
        s_costs, s_rewards,
        xerr=[s_err_c_lo, s_err_c_hi],
        yerr=[s_err_r_lo, s_err_r_hi],
        fmt="none", ecolor=CB_GRAY, alpha=0.4, capsize=3, zorder=3,
    )

    ax.plot(
        pf_c, pf_r,
        color=CB_BLUE, linestyle="-", linewidth=2.5,
        marker="o", markersize=7, markerfacecolor="white",
        markeredgecolor=CB_BLUE, markeredgewidth=2.0,
        label="BudgetPacer (adaptive)", zorder=6,
    )
    ax.errorbar(
        p_costs, p_rewards,
        xerr=[p_err_c_lo, p_err_c_hi],
        yerr=[p_err_r_lo, p_err_r_hi],
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    for r in pacer:
        util = r.get("budget_utilization", 0)
        if 0.95 <= util <= 1.05:
            marker_color = CB_GREEN
        elif util < 0.95:
            marker_color = CB_ORANGE
        else:
            marker_color = CB_RED
        ax.plot(
            r["mean_cost"], r["mean_reward"],
            "o", markersize=7, markerfacecolor=marker_color,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=7,
        )

    # Shade the dominance region between the two Pareto frontiers.
    # Interpolate both curves onto a shared log-spaced grid so
    # fill_between can highlight where adaptive > static.
    overlap_lo = max(min(sf_c), min(pf_c))
    overlap_hi = min(max(sf_c), max(pf_c))
    if overlap_lo < overlap_hi:
        x_shared = np.geomspace(overlap_lo, overlap_hi, 200)
        static_interp = np.interp(
            np.log(x_shared), np.log(sf_c), sf_r,
        )
        pacer_interp = np.interp(
            np.log(x_shared), np.log(pf_c), pf_r,
        )
        ax.fill_between(
            x_shared, static_interp, pacer_interp,
            where=pacer_interp >= static_interp,
            color=CB_BLUE, alpha=0.10, zorder=2,
            label="_nolegend_",
        )

    ax.set_xlabel("Mean Cost per Request (USD)", fontsize=13)
    ax.set_ylabel("Mean Reward", fontsize=13)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=CB_GRAY, linestyle="--", marker="s",
               markerfacecolor="white", markeredgecolor=CB_GRAY,
               markersize=6, linewidth=2,
               label=r"Static $\lambda$"),
        Line2D([0], [0], color=CB_BLUE, linestyle="-", marker="o",
               markerfacecolor="white", markeredgecolor=CB_BLUE,
               markersize=7, linewidth=2.5,
               label=r"Adaptive $\lambda$ (BudgetPacer)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_GREEN,
               markeredgecolor=CB_BLUE, markersize=7,
               label="On-target (0.95–1.05×)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_ORANGE,
               markeredgecolor=CB_BLUE, markersize=7,
               label="Under-target (<0.95×)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right",
              framealpha=0.9)

    ax.set_title(
        "Stationary Budget Pacing — Pareto Frontier (K=3)",
        fontsize=15, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    return fig


# ======================================================================
# Figure 2: Model-Mix Dynamics
# ======================================================================


def plot_model_mix(data: Dict[str, Any]) -> plt.Figure:
    """Grouped bar chart of model selection fractions."""
    results = data["results"]
    arm_order = data["arm_order"]

    static = [r for r in results if r["method"] == "static"]
    pacer = [r for r in results if r["method"] == "pacer"]

    all_rows = static + pacer
    n = len(all_rows)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5.5))

    x = np.arange(n)
    bar_width = 0.7
    bottom = np.zeros(n)

    for model in arm_order:
        fracs = [r["model_fractions"].get(model, 0) for r in all_rows]
        color = MODEL_COLORS.get(model, CB_GRAY)
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        ax.bar(x, fracs, bar_width, bottom=bottom, label=short,
               color=color, edgecolor="white", linewidth=0.5)
        bottom += np.array(fracs)

    labels = []
    for r in all_rows:
        if r["method"] == "static":
            labels.append(f"cp={r['cost_penalty']:.2f}")
        else:
            t = r["target_spend"]
            labels.append(f"${t:.4f}" if t >= 0.001 else f"${t:.1e}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Selection Fraction", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10)

    divider_x = len(static) - 0.5
    ax.axvline(divider_x, color="black", linestyle="--", linewidth=1.0, alpha=0.4)
    y_text = 1.02
    ax.text(divider_x / 2, y_text, "Static cost penalty",
            ha="center", fontsize=9, fontstyle="italic", color=CB_GRAY,
            transform=ax.get_xaxis_transform())
    ax.text((divider_x + n) / 2, y_text, "BudgetPacer (adaptive)",
            ha="center", fontsize=9, fontstyle="italic", color=CB_BLUE,
            transform=ax.get_xaxis_transform())

    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    cost_ax = ax.twiny()
    cost_ax.set_xlim(ax.get_xlim())
    cost_ax.set_xticks(x)
    cost_labels = [f"${r['mean_cost']:.4f}" if r['mean_cost'] >= 0.001
                   else f"${r['mean_cost']:.1e}" for r in all_rows]
    cost_ax.set_xticklabels(cost_labels, rotation=45, ha="left", fontsize=7,
                            color=CB_GRAY)
    cost_ax.set_xlabel("Realized Mean Cost / Request", fontsize=9, color=CB_GRAY)
    cost_ax.tick_params(labelsize=7, colors=CB_GRAY)

    ax.set_title(
        "Model Selection Mix Under Budget Constraints (K=3)",
        fontsize=14, fontweight="bold", pad=45,
    )

    fig.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    data = _load_results()

    figures = {
        "pareto_frontier": plot_pareto(data),
        "model_mix": plot_model_mix(data),
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
