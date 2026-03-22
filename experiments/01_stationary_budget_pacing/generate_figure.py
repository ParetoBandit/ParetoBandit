#!/usr/bin/env python3
"""Generate figures for Experiment 01: Stationary Budget Pacing.

Reads ``results/budget_pacing_results.json`` and produces a publication-ready
three-panel figure (``budget_pacing.{pdf,png}``) communicating the router's
core value proposition:

    **Panel A — "What you get for your money"**
    Quality (mean reward) as a function of budget target, with fixed
    single-model baselines as horizontal reference lines.  Shows that the
    router smoothly interpolates between cheap/low-quality and
    expensive/high-quality models.

    **Panel B — "The budget actually works"**
    Realized cost vs. budget target (45-degree = perfect compliance).
    Demonstrates that the pacer respects the operator's budget.

    **Panel C — "How it allocates"**
    Stacked area showing the model mix at each budget level.

Usage:
    python experiments/01_stationary_budget_pacing/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "budget_pacing_results.json"

# Colorblind-safe palette (Wong, Nature Methods 2011)
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


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_FILE) as f:
        return json.load(f)


def _dollar_fmt(x: float, _pos: Any = None) -> str:
    if x >= 0.01:
        return f"${x:.3f}"
    if x >= 0.001:
        return f"${x:.4f}"
    return f"${x:.5f}"


def _bootstrap_errorbars(
    rows: List[Dict[str, Any]], field: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute asymmetric bootstrap CI half-widths for errorbar plotting."""
    means = np.array([r[f"mean_{field}"] for r in rows])
    lo_errs = np.zeros(len(rows))
    hi_errs = np.zeros(len(rows))
    per_seed_key = f"per_seed_{field}s"
    for i, r in enumerate(rows):
        if per_seed_key in r:
            ci_lo, ci_hi = bootstrap_ci(np.array(r[per_seed_key]))
            lo_errs[i] = means[i] - ci_lo
            hi_errs[i] = ci_hi - means[i]
        else:
            se = r.get(f"se_{field}", 0.0)
            lo_errs[i] = se
            hi_errs[i] = se
    return lo_errs, hi_errs


def plot_budget_pacing(data: Dict[str, Any]) -> plt.Figure:
    """Three-panel budget-paced routing figure.

    Panel A: Quality vs budget target with fixed-model references.
    Panel B: Budget compliance (realized cost vs target).
    Panel C: Model mix (stacked bars) across budget levels.
    """
    results = data["results"]
    arm_order = data["arm_order"]

    fixed = [r for r in results if r["method"] == "fixed_model"]
    pacer = sorted(
        [r for r in results if r["method"] == "pacer"],
        key=lambda r: r["target_spend"],
    )

    targets = np.array([r["target_spend"] for r in pacer])
    rewards = np.array([r["mean_reward"] for r in pacer])
    costs = np.array([r["mean_cost"] for r in pacer])

    r_err_lo, r_err_hi = _bootstrap_errorbars(pacer, "reward")
    c_err_lo, c_err_hi = _bootstrap_errorbars(pacer, "cost")

    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.25, 1.0, 1.0],
        wspace=0.40, left=0.05, right=0.97, top=0.88, bottom=0.15,
    )
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ══════════════════════════════════════════════════════════════════
    # Panel A: Quality vs Budget Target
    # ══════════════════════════════════════════════════════════════════

    ax_a.plot(
        costs, rewards,
        color=CB_BLUE, linewidth=2.5, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=CB_BLUE,
        markeredgewidth=2.0, zorder=6,
    )
    ax_a.errorbar(
        costs, rewards,
        yerr=[r_err_lo, r_err_hi],
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    for r in pacer:
        util = r.get("budget_utilization", 0.0)
        if 0.95 <= util <= 1.05:
            mc = CB_GREEN
        elif util < 0.95:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_a.plot(
            r["mean_cost"], r["mean_reward"],
            "o", markersize=7, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=7,
        )

    for r in fixed:
        mid = r.get("model_id", "")
        short = MODEL_SHORT.get(mid, mid.split("/")[-1])
        color = MODEL_COLORS.get(mid, CB_GRAY)
        ax_a.plot(
            r["mean_cost"], r["mean_reward"],
            marker="*", markersize=14, markerfacecolor=color,
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=10, linestyle="none",
        )
        x_off, y_off = 8, 0
        va = "center"
        if "gemini" in mid.lower():
            x_off, y_off = -8, 6
        elif "mistral" in mid.lower():
            x_off, y_off = -10, 0
        elif "llama" in mid.lower():
            x_off, y_off = 8, -2
        ax_a.annotate(
            short,
            xy=(r["mean_cost"], r["mean_reward"]),
            xytext=(x_off, y_off), textcoords="offset points",
            fontsize=10, color=color, fontweight="bold",
            fontstyle="italic", va=va, ha="left" if x_off > 0 else "right",
        )

    gemini = next(r for r in fixed if "gemini" in r.get("model_id", ""))
    target_q = 0.90 * gemini["mean_reward"]
    annot_r = next(
        (r for r in pacer if r["mean_reward"] >= target_q), pacer[-1],
    )
    quality_pct = annot_r["mean_reward"] / gemini["mean_reward"] * 100
    cost_pct = annot_r["mean_cost"] / gemini["mean_cost"] * 100
    ax_a.annotate(
        f"{quality_pct:.0f}% of Gemini quality\nat {cost_pct:.0f}% of its cost",
        xy=(annot_r["mean_cost"], annot_r["mean_reward"]),
        xytext=(85, -30), textcoords="offset points",
        fontsize=11, fontweight="bold", fontstyle="italic",
        color="#1a1a1a", ha="center",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.9),
    )

    y_lo = min(r["mean_reward"] for r in fixed) - 0.015
    y_hi = max(r["mean_reward"] for r in fixed) + 0.015
    ax_a.set_ylim(y_lo, y_hi)
    ax_a.set_xlabel("Cost per Request (USD)", fontsize=13)
    ax_a.set_ylabel("Mean Quality", fontsize=13)
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax_a.grid(True, alpha=0.15, linewidth=0.5)
    ax_a.tick_params(labelsize=11)
    ax_a.set_title("(a)  Quality vs. Budget", fontsize=14, fontweight="bold",
                    pad=8)

    from matplotlib.lines import Line2D
    legend_a = [
        Line2D([0], [0], color=CB_BLUE, linewidth=2.5, marker="o",
               markerfacecolor="white", markeredgecolor=CB_BLUE,
               markersize=7, label="ParetoBandit"),
        Line2D([0], [0], color="none", marker="*", markerfacecolor=CB_GRAY,
               markeredgecolor="black", markersize=12,
               label="Fixed single-model"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_GREEN,
               markeredgecolor=CB_BLUE, markersize=7,
               label="On-budget (0.95–1.05×)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor=CB_ORANGE,
               markeredgecolor=CB_BLUE, markersize=7,
               label="Under-budget (<0.95×)"),
    ]
    ax_a.legend(handles=legend_a, fontsize=9, loc="lower right",
                framealpha=0.9, ncol=1)

    # ══════════════════════════════════════════════════════════════════
    # Panel B: Budget Compliance
    # ══════════════════════════════════════════════════════════════════

    diag_range = np.geomspace(targets.min() * 0.5, targets.max() * 2.0, 100)
    ax_b.plot(
        diag_range, diag_range,
        color=CB_GRAY, linestyle="--", linewidth=1.0, alpha=0.6,
        label="Perfect compliance", zorder=3,
    )
    ax_b.fill_between(
        diag_range, diag_range * 0.95, diag_range * 1.05,
        color=CB_GREEN, alpha=0.35, label="±5% band", zorder=2,
    )

    for r in pacer:
        util = r.get("budget_utilization", 0.0)
        if 0.95 <= util <= 1.05:
            mc = CB_GREEN
        elif util < 0.95:
            mc = CB_ORANGE
        else:
            mc = CB_RED
        ax_b.plot(
            r["target_spend"], r["mean_cost"],
            "o", markersize=9, markerfacecolor=mc,
            markeredgecolor=CB_BLUE, markeredgewidth=1.5, zorder=6,
        )
        ha = "left"
        x_off, y_off = 7, 0
        ax_b.annotate(
            f"{util:.2f}×",
            xy=(r["target_spend"], r["mean_cost"]),
            xytext=(x_off, y_off), textcoords="offset points",
            fontsize=11, color="0.3", ha=ha, va="center",
        )

    ax_b.errorbar(
        targets, costs,
        yerr=[c_err_lo, c_err_hi],
        fmt="none", ecolor=CB_BLUE, alpha=0.4, capsize=3, zorder=5,
    )

    ax_b.set_xlabel("Budget Target ($/request)", fontsize=13)
    ax_b.set_ylabel("Realized Cost ($/request)", fontsize=13)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.xaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    ax_b.yaxis.set_major_formatter(mticker.FuncFormatter(_dollar_fmt))
    shared_lim = (targets.min() * 0.6, targets.max() * 3.5)
    ax_b.set_xlim(shared_lim)
    ax_b.set_ylim(shared_lim)
    ax_b.grid(True, alpha=0.15, linewidth=0.5)
    ax_b.tick_params(labelsize=11)
    ax_b.set_title("(b)  Budget Compliance", fontsize=14,
                    fontweight="bold", pad=8)
    ax_b.legend(fontsize=9, loc="upper left", framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════
    # Panel C: Model Mix (stacked bars)
    # ══════════════════════════════════════════════════════════════════

    x_pos = np.arange(len(pacer))
    bottom = np.zeros(len(pacer))
    bar_width = 0.65

    for model in arm_order:
        fracs = np.array([
            r["model_fractions"].get(model, 0.0) for r in pacer
        ])
        color = MODEL_COLORS.get(model, CB_GRAY)
        short = MODEL_SHORT.get(model, model.split("/")[-1])
        ax_c.bar(
            x_pos, fracs, bar_width, bottom=bottom,
            label=short, color=color, edgecolor="white", linewidth=0.5,
        )
        bottom += fracs

    budget_labels = []
    for r in pacer:
        t = r["target_spend"]
        if t >= 0.001:
            budget_labels.append(f"${t:.4f}")
        else:
            budget_labels.append(f"${t:.1e}")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(budget_labels, rotation=45, ha="right", fontsize=9.5)
    ax_c.set_xlabel("Budget Target ($/request)", fontsize=13)
    ax_c.set_ylabel("Selection Fraction", fontsize=13)
    ax_c.set_ylim(0, 1.05)
    ax_c.grid(axis="y", alpha=0.15, linewidth=0.5)
    ax_c.tick_params(labelsize=11)
    ax_c.set_title("(c)  Model Allocation", fontsize=14,
                    fontweight="bold", pad=8)
    ax_c.legend(fontsize=10, loc="center left", framealpha=0.9,
                bbox_to_anchor=(0.0, 0.5))

    fig.suptitle(
        "Budget-Paced LLM Routing (K=3)",
        fontsize=17, fontweight="bold",
    )

    return fig


def main() -> None:
    data = _load_results()

    fig = plot_budget_pacing(data)

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"budget_pacing.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved budget_pacing.{{pdf,png}}")
    print(f"\nFigures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
