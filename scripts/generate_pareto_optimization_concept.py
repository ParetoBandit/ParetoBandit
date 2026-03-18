"""Recreate the Pareto Optimization concept diagram without a title.

Faithfully reproduces the visual from blog/pareto_optimization_concept.png
(teal frontier curve, scattered sub-optimal points, regret arrow, budget
constraint, bandit target dot) but omits the top-level title text.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK_NAVY = "#1B2A4A"
TEAL = "#2A9D8F"
WHITE = "#FFFFFF"


def create_pareto_concept(output_path: str = "blog/pareto_optimization_concept.png") -> Path:
    """Render the Pareto-optimization concept diagram (no title) and save as PNG.

    Args:
        output_path: Destination file path.  Parent directories are created
            automatically if they do not exist.

    Returns:
        Resolved ``Path`` to the written image file.
    """
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    # Pareto frontier curve (concave, diminishing returns).
    t = np.linspace(0.02, 0.95, 300)
    x_curve = t
    y_curve = 1.0 - (1.0 - t) ** 2.5

    ax.plot(x_curve, y_curve, color=TEAL, lw=3.5, zorder=2)

    # Budget constraint vertical dashed line at ~0.65.
    budget_x = 0.65
    ax.axvline(x=budget_x, color=DARK_NAVY, linestyle="--", lw=2, zorder=1)
    ax.text(budget_x, 1.03, "Budget Constraint", fontsize=18, fontweight="bold",
            ha="center", va="bottom", color=DARK_NAVY)

    # Bandit target dot (on the frontier at the budget line).
    budget_y = float(np.interp(budget_x, x_curve, y_curve))
    ax.plot(budget_x, budget_y, "o", color=TEAL, markersize=20, zorder=4,
            markeredgecolor=DARK_NAVY, markeredgewidth=2.5)
    ax.text(budget_x + 0.03, budget_y - 0.01, "Bandit Target:\nMaximize Reward,\nMinimize Regret",
            fontsize=17, fontweight="bold", color=DARK_NAVY, va="top")

    # Label for the frontier curve.
    ax.text(0.18, 0.88, "Pareto Frontier\n(Optimal Routing)",
            fontsize=19, fontweight="bold", color=TEAL, va="center")

    # Scattered sub-optimal routing points (spread to avoid label overlap).
    sub_pts = np.array([
        [0.18, 0.28],
        [0.26, 0.13],
        [0.34, 0.42],
        [0.40, 0.32],
        [0.50, 0.38],
        [0.56, 0.30],
        [0.63, 0.36],
    ])
    ax.scatter(sub_pts[:, 0], sub_pts[:, 1], s=160, color=TEAL, zorder=3,
               edgecolors=DARK_NAVY, linewidths=0.8)
    ax.text(0.30, 0.13, "Sub-optimal\nRouting", fontsize=17, fontweight="bold",
            color=DARK_NAVY, va="center", ha="left")

    # Regret arrow: anchored to the (0.50, 0.38) sub-optimal point.
    arrow_x = 0.50
    arrow_bot = 0.38
    arrow_top = float(np.interp(arrow_x, x_curve, y_curve)) - 0.03
    ax.annotate(
        "",
        xy=(arrow_x, arrow_top),
        xytext=(arrow_x, arrow_bot),
        arrowprops=dict(arrowstyle="-|>", color=DARK_NAVY, lw=2.5, mutation_scale=20),
        zorder=3,
    )
    ax.text(arrow_x - 0.03, (arrow_bot + arrow_top) / 2, "Regret\n(Quality lost)",
            fontsize=17, fontweight="bold", color=DARK_NAVY, va="center", ha="right")

    # Axis labels.
    ax.set_xlabel("Cost ($/request)", fontsize=22, fontweight="bold", color=DARK_NAVY, labelpad=12)
    ax.set_ylabel("Quality (Reward)", fontsize=22, fontweight="bold", color=DARK_NAVY, labelpad=12)

    # Strip tick labels but keep axis lines.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_color(DARK_NAVY)
    ax.spines["bottom"].set_color(DARK_NAVY)

    # Arrow tips on axes.
    ax.plot(0, 1.1, "^", color=DARK_NAVY, markersize=10, clip_on=False, transform=ax.transData)
    ax.plot(1.0, 0, ">", color=DARK_NAVY, markersize=10, clip_on=False, transform=ax.transData)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_pareto_concept()
    print(f"Saved: {path}")
