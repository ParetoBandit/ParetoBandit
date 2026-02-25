#!/usr/bin/env python3
"""
Generate Figure 7: LinTS baseline comparison Pareto frontiers for K=5 and K=10.

Two-panel figure (side by side): one panel per portfolio.
Each panel shows:
  - banditGPT Pareto curve (solid, markers)
  - LinTS (warmup) Pareto curve (dashed)
  - LinTS (no priors) Pareto curve (dotted)
  - Horizontal reference lines for Oracle, Best Static, ε-greedy, Random
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "lints_comparison_results.json"

COLORS = {
    "banditGPT":       "#2171B5",
    "LinTS":           "#E6550D",
    "LinTS (no priors)": "#E6550D",
}

STYLES = {
    "banditGPT":       dict(linestyle="-",  linewidth=2.2, marker="o", markersize=5, zorder=5),
    "LinTS":           dict(linestyle="--", linewidth=2.0, marker="s", markersize=4, zorder=4),
    "LinTS (no priors)": dict(linestyle=":", linewidth=1.8, marker="^", markersize=4, zorder=3, alpha=0.7),
}


def pareto_front(points):
    """Return subset of (cost, reward) points on the upper-left Pareto frontier."""
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_r = -np.inf
    for c, r in pts:
        if r >= best_r:
            front.append((c, r))
            best_r = r
    return front


def plot_panel(ax, data, title):
    series_map = {
        "pareto_banditgpt":    ("banditGPT",       COLORS["banditGPT"]),
        "pareto_lints":        ("LinTS",            COLORS["LinTS"]),
        "pareto_lints_tabula": ("LinTS (no priors)", COLORS["LinTS (no priors)"]),
    }

    for key, (label, color) in series_map.items():
        pts = data[key]
        costs = [p["mean_cost"] for p in pts]
        rewards = [p["mean_reward"] for p in pts]
        stds = [p["std_reward"] for p in pts]

        front = pareto_front(list(zip(costs, rewards)))
        fc, fr = zip(*front) if front else (costs, rewards)

        ax.plot(fc, fr, label=label, color=color, **STYLES[label])
        ax.fill_between(
            costs,
            [r - s for r, s in zip(rewards, stds)],
            [r + s for r, s in zip(rewards, stds)],
            color=color, alpha=0.08,
        )

    oracle_r = data["oracle"]["reward"]
    best_static = data["best_static"]
    rand_r = data["random"]["reward"]
    eg_r = data["epsilon_greedy"]["reward"]

    ax.axhline(oracle_r, color="#2CA02C", linestyle="-", linewidth=1, alpha=0.6, label="Oracle")
    ax.axhline(best_static["reward"], color="#9467BD", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Best Static ({best_static['model'].split('/')[-1]})")
    ax.axhline(eg_r, color="#8C564B", linestyle="-.", linewidth=1, alpha=0.6, label="ε-greedy")
    ax.axhline(rand_r, color="#7F7F7F", linestyle=":", linewidth=1, alpha=0.4, label="Random")

    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Cost per Request ($)", fontsize=17)
    ax.set_ylabel("Mean Reward", fontsize=17)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=14)


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax, (portfolio, title) in zip(axes, [
        ("K5", "K=5 Portfolio"),
        ("K10", "K=10 Portfolio"),
    ]):
        plot_panel(ax, results[portfolio], title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=14,
               bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=True)

    fig.suptitle("Figure 7: banditGPT vs. Linear Thompson Sampling — Pareto Frontiers",
                 fontsize=22, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"figure7_lints_comparison.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
