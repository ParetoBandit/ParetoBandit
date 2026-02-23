#!/usr/bin/env python3
"""
Generate Figure 9: Cumulative Regret Curves for K=5 and K=10.

Two-panel figure (side by side): one panel per portfolio.
Each panel plots cumulative regret over online learning steps for:
  banditGPT, LinTS (warmup), LinTS (no priors), ε-greedy, Random.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "cumulative_regret_results.json"

METHOD_STYLE = {
    "banditGPT":         dict(color="#2171B5", linestyle="-",  linewidth=2.2, zorder=5),
    "LinTS":             dict(color="#E6550D", linestyle="--", linewidth=2.0, zorder=4),
    "LinTS (no priors)": dict(color="#E6550D", linestyle=":",  linewidth=1.8, zorder=3, alpha=0.7),
    "ε-greedy":          dict(color="#8C564B", linestyle="-.", linewidth=1.5, zorder=2),
    "Random":            dict(color="#7F7F7F", linestyle=":",  linewidth=1.5, zorder=1, alpha=0.6),
}


def plot_panel(ax, data, title):
    methods = data["methods"]
    for name, style in METHOD_STYLE.items():
        if name not in methods:
            continue
        m = methods[name]
        steps = np.array(m["steps"])
        mean = np.array(m["mean"])
        std = np.array(m["std"])

        ax.plot(steps, mean, label=name, **style)
        ax.fill_between(steps, mean - std, mean + std,
                        color=style["color"], alpha=0.08)

    n = data["n_steps"]
    x_sqrt = np.linspace(1, n, 200)
    scale = methods["banditGPT"]["mean"][-1] / np.sqrt(n)
    ax.plot(x_sqrt, scale * np.sqrt(x_sqrt), color="#AAAAAA",
            linestyle="--", linewidth=1, alpha=0.5, label=r"$O(\sqrt{T})$ reference")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Online Learning Step", fontsize=10)
    ax.set_ylabel("Cumulative Regret", fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (portfolio, title) in zip(axes, [
        ("K5", "K=5 Portfolio"),
        ("K10", "K=10 Portfolio"),
    ]):
        plot_panel(ax, results[portfolio], title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True, fancybox=True)

    fig.suptitle("Figure 9: Cumulative Regret — Online Learning Phase",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"figure9_cumulative_regret.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close()


if __name__ == "__main__":
    main()
