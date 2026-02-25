#!/usr/bin/env python3
"""
Generate Figure: Hard Constraint Impact
========================================

Three-panel figure:
  (a) Reward vs. cost ceiling — graceful degradation as budget tightens
  (b) Reward vs. latency ceiling — latency-reward trade-off
  (c) Constrained Pareto frontiers — constraints + λ interaction

Plus a standalone production scenario table (printed to stdout for LaTeX).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results" / "constraint_impact_results.json"
OUTPUT_DIR = Path(__file__).parent / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

COLORS = {
    "cost": "#2166ac",
    "latency": "#b2182b",
    "tight": "#d6604d",
    "moderate": "#4393c3",
    "unconstrained": "#1a1a1a",
}

YLIM = (0.870, 0.910)


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def panel_cost_sweep(ax, data):
    """Panel (a): Reward vs. cost ceiling."""
    pts = data["cost_sweep"]
    xs, ys, errs, ks = [], [], [], []
    for p in pts:
        mc = p["max_cost"]
        xs.append(mc if mc is not None else pts[-2]["max_cost"] * 3)
        ys.append(p["mean_reward"])
        errs.append(p["ci95_reward"])
        ks.append(p["eligible_K"])

    ax.errorbar(xs[:-1], ys[:-1], yerr=errs[:-1],
                fmt="o-", color=COLORS["cost"], capsize=3, markersize=5, linewidth=1.5)
    ax.axhline(ys[-1], color=COLORS["unconstrained"], linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between([xs[0]*0.5, xs[-1]*1.5],
                    ys[-1] - errs[-1], ys[-1] + errs[-1],
                    color=COLORS["unconstrained"], alpha=0.08)
    ax.text(xs[-2]*1.5, ys[-1] + 0.002, f"Unconstrained (K={ks[-1]})",
            fontsize=7, color=COLORS["unconstrained"], ha="right")

    for i, (x, y, k) in enumerate(zip(xs[:-1], ys[:-1], ks[:-1])):
        ax.annotate(f"K'={k}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=6.5,
                    color=COLORS["cost"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Cost ceiling ($/request)")
    ax.set_title("(a) Cost constraint sweep")
    ax.set_xlim(xs[0] * 0.5, xs[-2] * 2)


def panel_latency_sweep(ax, data):
    """Panel (b): Reward vs. latency ceiling."""
    pts = data["latency_sweep"]
    xs, ys, errs, ks = [], [], [], []
    for p in pts:
        ml = p["max_latency"]
        xs.append(ml if ml is not None else pts[-2]["max_latency"] * 1.3)
        ys.append(p["mean_reward"])
        errs.append(p["ci95_reward"])
        ks.append(p["eligible_K"])

    ax.errorbar(xs[:-1], ys[:-1], yerr=errs[:-1],
                fmt="s-", color=COLORS["latency"], capsize=3, markersize=5, linewidth=1.5)
    ax.axhline(ys[-1], color=COLORS["unconstrained"], linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between([xs[0]*0.8, xs[-1]*1.2],
                    ys[-1] - errs[-1], ys[-1] + errs[-1],
                    color=COLORS["unconstrained"], alpha=0.08)
    ax.text(xs[-2]*1.15, ys[-1] + 0.002, f"Unconstrained (K={ks[-1]})",
            fontsize=7, color=COLORS["unconstrained"], ha="right")

    for i, (x, y, k) in enumerate(zip(xs[:-1], ys[:-1], ks[:-1])):
        ax.annotate(f"K'={k}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=6.5,
                    color=COLORS["latency"], fontweight="bold")

    ax.set_xlabel("Latency ceiling (seconds)")
    ax.set_title("(b) Latency constraint sweep")


def panel_constrained_pareto(ax, data):
    """Panel (c): Pareto frontiers under different constraint regimes."""
    pareto = data["constrained_pareto"]
    style_map = [
        (COLORS["tight"], "v", "-", "Tight"),
        (COLORS["moderate"], "D", "-", "Moderate"),
        (COLORS["unconstrained"], "o", "--", "Unconstrained"),
    ]

    for (regime_name, regime_data), (color, marker, ls, short_label) in zip(
        pareto.items(), style_map
    ):
        pts = regime_data["points"]
        xs = [p["mean_cost"] for p in pts]
        ys = [p["mean_reward"] for p in pts]
        errs = [p["ci95_reward"] for p in pts]
        k = regime_data["eligible_K"]
        ax.errorbar(xs, ys, yerr=errs,
                    fmt=f"{marker}{ls}", color=color, capsize=2,
                    markersize=4, linewidth=1.3,
                    label=f"{short_label} (K'={k})")

    ax.set_xscale("log")
    ax.set_xlabel("Realized cost ($/request)")
    ax.set_title("(c) Constrained Pareto frontiers")
    ax.legend(loc="lower right", framealpha=0.9)


def print_scenario_table(data):
    """Print production scenario results as a LaTeX-ready summary."""
    print("\n" + "=" * 80)
    print("PRODUCTION SCENARIO TABLE (for LaTeX)")
    print("=" * 80)
    fmt = "{:<18} {:>8} {:>8} {:>4} {:>10} {:>10} {:>12}"
    print(fmt.format("Scenario", "MaxCost", "MaxLat", "K'",
                      "Reward", "Cost", "Violations"))
    print("-" * 80)
    for s in data["production_scenarios"]:
        mc = f"${s['max_cost']}" if s["max_cost"] else "None"
        ml = f"{s['max_latency']}s" if s["max_latency"] else "None"
        q = f"{s['mean_reward']:.3f}\u00b1{s['std_reward']:.3f}"
        c = f"${s['mean_cost']:.6f}"
        v = f"{s['violation_rate']:.4f}"
        print(fmt.format(s["label"], mc, ml, str(s["eligible_K"]), q, c, v))
    print("=" * 80)


def main():
    data = load_results()
    print_scenario_table(data)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.06, right=0.97, top=0.90, bottom=0.15)

    panel_cost_sweep(axes[0], data)
    panel_latency_sweep(axes[1], data)
    panel_constrained_pareto(axes[2], data)

    axes[0].set_ylim(*YLIM)
    axes[0].set_ylabel("Reward")

    fig.suptitle(
        "Impact of Hard Per-Request Constraints on Routing Quality (K=10)",
        fontsize=11, fontweight="bold", y=0.98,
    )

    out_png = OUTPUT_DIR / "figure_constraint_impact.png"
    out_pdf = OUTPUT_DIR / "figure_constraint_impact.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nFigure saved to {out_png}")
    print(f"Figure saved to {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
