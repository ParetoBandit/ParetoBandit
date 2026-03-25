#!/usr/bin/env python3
"""Generate cost heuristic validation figures for the appendix.

Produces two single-topic figures so they remain readable in single-column
layout:
- ``cost_heuristic_validation_distributions``: panels (a) and (b), showing
  per-model per-request cost distributions for K=3 and K=4.
- ``cost_heuristic_validation_ranking``: panels (c) and (d), showing pairwise
  ranking preservation for K=3 and K=4.

Cost arrays are read from the precomputed JSON produced by
``run_cost_heuristic_validation.py`` (key ``_costs_by_model``), so the
figures are guaranteed to be consistent with the reported statistics.

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


def plot_distribution_panel(
    ax: plt.Axes,
    results: Dict[str, Any],
    colors: List[str],
    panel_label: str,
) -> None:
    """Plot one per-portfolio cost-distribution panel."""
    stats = results["model_stats"]
    arm_names = _arm_order(stats)

    costs_by_model = results["_costs_by_model"]
    cost_arrays = [np.array(costs_by_model[n]) for n in arm_names]

    parts = ax.violinplot(
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

    ax.set_yscale("log")
    ax.set_xticks(range(len(arm_names)))
    labels = []
    for n in arm_names:
        ct = stats[n]["heuristic_c_tilde"]
        labels.append(f"{n}\n($\\tilde{{c}}$={ct:.3f})")
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Per-Request Cost (USD, log scale)", fontsize=20)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.set_axisbelow(True)

    title = f"K={results['n_arms']} ({results['n_prompts']} prompts)"
    ax.set_title(f"{panel_label} {title}", fontsize=22, fontweight="bold", pad=18)
    ax.tick_params(axis="y", labelsize=16)


def plot_ranking_panel(
    ax: plt.Axes,
    results: Dict[str, Any],
    panel_label: str,
) -> None:
    """Plot one per-portfolio pairwise ranking panel."""
    ranking = results["ranking"]
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
    bars = ax.barh(
        y_pos, pair_fracs, color=bar_colors, edgecolor="white", height=0.6,
    )

    for bar, frac in zip(bars, pair_fracs):
        text_x = max(0.06, frac - 0.03)
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{frac:.1%}",
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_names, fontsize=14)
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Fraction Correct", fontsize=18)
    ax.set_title(f"{panel_label} Pairwise Ranking", fontsize=22, fontweight="bold", pad=18)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=14)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    colors_k3 = ["#2196F3", "#FF9800", "#4CAF50"]
    colors_k4 = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    has_k4 = "k4" in data
    distribution_rows = 2 if has_k4 else 1
    distribution_fig, distribution_axes = plt.subplots(
        distribution_rows,
        1,
        figsize=(7.4, 5.2 * distribution_rows),
    )
    if distribution_rows == 1:
        distribution_axes = [distribution_axes]

    plot_distribution_panel(distribution_axes[0], data["k3"], colors_k3, "(a)")
    if has_k4:
        plot_distribution_panel(distribution_axes[1], data["k4"], colors_k4, "(b)")

    distribution_fig.tight_layout(h_pad=4.0)

    ranking_rows = 2 if has_k4 else 1
    ranking_fig, ranking_axes = plt.subplots(
        ranking_rows,
        1,
        figsize=(7.4, 4.6 * ranking_rows),
    )
    if ranking_rows == 1:
        ranking_axes = [ranking_axes]

    plot_ranking_panel(ranking_axes[0], data["k3"], "(a)")
    if has_k4:
        plot_ranking_panel(ranking_axes[1], data["k4"], "(b)")

    ranking_fig.tight_layout(h_pad=3.0)

    outputs = [
        (distribution_fig, "cost_heuristic_validation_distributions"),
        (ranking_fig, "cost_heuristic_validation_ranking"),
    ]
    for fig, stem in outputs:
        for fmt in ("png", "pdf"):
            out = RESULTS_DIR / f"{stem}.{fmt}"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Saved {out}")

    plt.close(distribution_fig)
    plt.close(ranking_fig)


if __name__ == "__main__":
    main()
