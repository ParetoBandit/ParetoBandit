#!/usr/bin/env python3
"""Generate cost heuristic validation figures for the appendix.

Produces two panels:
- (a) Per-model violin plots of actual per-request costs (log scale) with
      heuristic c_tilde ordering annotated.
- (b) Pairwise ranking preservation bar chart showing what fraction of
      prompts maintain the heuristic cost ordering.

Cost arrays are read from the precomputed JSON produced by
``run_cost_heuristic_validation.py`` (key ``_costs_by_model``), so the
figure is guaranteed to be consistent with the reported statistics.

Usage:
    python experiments/appendix/cost_heuristic_validation/generate_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "cost_heuristic_validation.json"


def _arm_order(stats: Dict[str, Any]) -> List[str]:
    """Return arm names sorted by heuristic c_tilde (ascending)."""
    return sorted(stats.keys(), key=lambda n: stats[n]["heuristic_c_tilde"])


def plot_portfolio(
    ax_violin: plt.Axes,
    ax_rank: plt.Axes,
    results: Dict[str, Any],
    colors: List[str],
) -> None:
    """Plot violin + ranking bar for one portfolio."""
    stats = results["model_stats"]
    ranking = results["ranking"]
    arm_names = _arm_order(stats)

    costs_by_model = results["_costs_by_model"]
    cost_arrays = [np.array(costs_by_model[n]) for n in arm_names]

    parts = ax_violin.violinplot(
        cost_arrays,
        positions=range(len(arm_names)),
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_alpha(0.7)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.0)
    parts["cmedians"].set_linestyle("--")

    ax_violin.set_yscale("log")
    ax_violin.set_xticks(range(len(arm_names)))
    labels = []
    for n in arm_names:
        ct = stats[n]["heuristic_c_tilde"]
        labels.append(f"{n}\n($\\tilde{{c}}$={ct:.3f})")
    ax_violin.set_xticklabels(labels, fontsize=8)
    ax_violin.set_ylabel("Per-Request Cost (USD, log scale)", fontsize=9)
    ax_violin.grid(axis="y", alpha=0.3, which="both")
    ax_violin.set_axisbelow(True)

    for i, n in enumerate(arm_names):
        mean_c = stats[n]["mean_cost"]
        ax_violin.annotate(
            f"${mean_c:.6f}",
            xy=(i, mean_c),
            xytext=(8, -5),
            textcoords="offset points",
            fontsize=6.5,
            color=colors[i % len(colors)],
            fontweight="bold",
        )

    title = f"K={results['n_arms']} ({results['n_prompts']} prompts)"
    ax_violin.set_title(title, fontsize=10, fontweight="bold")

    pairwise = ranking["pairwise"]
    pair_names = list(pairwise.keys())
    pair_fracs = [pairwise[k]["frac"] for k in pair_names]

    y_pos = range(len(pair_names))
    bar_colors = [
        "#4CAF50" if f >= 0.95
        else "#FF9800" if f >= 0.80
        else "#F44336"
        for f in pair_fracs
    ]
    bars = ax_rank.barh(
        y_pos, pair_fracs, color=bar_colors, edgecolor="white", height=0.6,
    )

    for bar, frac in zip(bars, pair_fracs):
        ax_rank.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{frac:.1%}",
            va="center", fontsize=7, fontweight="bold",
        )

    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(pair_names, fontsize=7.5)
    ax_rank.set_xlim(0, 1.15)
    ax_rank.axvline(x=0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_rank.set_xlabel("Fraction Correct", fontsize=9)
    ax_rank.set_title("Pairwise Ranking", fontsize=10, fontweight="bold")
    ax_rank.grid(axis="x", alpha=0.3)
    ax_rank.set_axisbelow(True)

    fm = ranking["full_ordering_match"]
    ax_rank.text(
        0.5, -0.08,
        f"Full ordering match: {fm['frac']:.1%}",
        transform=ax_rank.transAxes,
        ha="center", fontsize=8, fontstyle="italic",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    colors_k3 = ["#2196F3", "#FF9800", "#4CAF50"]
    colors_k4 = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    has_k4 = "k4" in data
    n_rows = 2 if has_k4 else 1
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(11, 4.5 * n_rows),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    if n_rows == 1:
        axes = [axes]

    plot_portfolio(axes[0][0], axes[0][1], data["k3"], colors_k3)

    if has_k4:
        plot_portfolio(axes[1][0], axes[1][1], data["k4"], colors_k4)

    fig.tight_layout(h_pad=2.5)

    for fmt in ("png", "pdf"):
        out = RESULTS_DIR / f"cost_heuristic_validation.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
