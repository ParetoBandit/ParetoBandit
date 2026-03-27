#!/usr/bin/env python3
"""Generate the conceptual 'cost-quality spectrum' diagram for the blog post.

Produces a clean figure showing three LLM models as points on a
cost-quality plane, with an arrow illustrating the gap that an
adaptive router can fill. Uses the same Wong colorblind-safe palette
as the demo figures for visual consistency.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_TEAL = "#56B4E9"
CB_GREEN = "#009E73"
CB_GRAY = "#999999"

MODELS = [
    {"name": "Llama-8B\n(Budget)", "cost": 0.00003, "quality": 0.793, "color": CB_TEAL},
    {"name": "Mistral-Large\n(Mid-tier)", "cost": 0.00053, "quality": 0.923, "color": CB_ORANGE},
    {"name": "Gemini-Pro\n(Frontier)", "cost": 0.015, "quality": 0.932, "color": CB_BLUE},
]

fig, ax = plt.subplots(figsize=(9, 5.5))

costs = [m["cost"] for m in MODELS]
qualities = [m["quality"] for m in MODELS]
colors = [m["color"] for m in MODELS]

for m in MODELS:
    ax.scatter(
        m["cost"], m["quality"],
        s=260, color=m["color"], edgecolors="black",
        linewidths=1.2, zorder=10, marker="*",
    )
    x_off = 12 if m["cost"] < 0.001 else -14
    ha = "left" if m["cost"] < 0.001 else "right"
    ax.annotate(
        m["name"],
        xy=(m["cost"], m["quality"]),
        xytext=(x_off, -8),
        textcoords="offset points",
        fontsize=11, fontweight="bold", color=m["color"],
        ha=ha, va="top",
    )

curve_costs = np.geomspace(costs[0], costs[-1], 80)
curve_qualities = np.interp(
    np.log10(curve_costs),
    [np.log10(c) for c in costs],
    qualities,
)
ax.plot(
    curve_costs, curve_qualities,
    color=CB_GREEN, linewidth=2.8, linestyle="-", alpha=0.7, zorder=5,
)
ax.fill_between(
    curve_costs, curve_qualities - 0.008, curve_qualities + 0.008,
    color=CB_GREEN, alpha=0.12, zorder=4,
)

mid_idx = len(curve_costs) // 3
ax.annotate(
    "ParetoBandit\ncontinuous frontier",
    xy=(curve_costs[mid_idx], curve_qualities[mid_idx]),
    xytext=(40, -40),
    textcoords="offset points",
    fontsize=11, fontweight="bold", color=CB_GREEN,
    ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color=CB_GREEN, lw=1.8),
)

ax.annotate(
    "",
    xy=(0.00005, 0.77),
    xytext=(0.012, 0.77),
    arrowprops=dict(arrowstyle="<->", color=CB_GRAY, lw=1.5, linestyle="--"),
)
ax.text(
    0.0008, 0.762,
    "530x cost gap between models",
    ha="center", va="top", fontsize=9.5, color="#555555",
    fontstyle="italic",
)

ax.set_xscale("log")
ax.set_xlabel("Cost per Request (USD, log scale)", fontsize=12, labelpad=8)
ax.set_ylabel("Response Quality", fontsize=12, labelpad=8)
ax.set_title(
    "The LLM Routing Problem: Three Models, One Budget",
    fontsize=14, fontweight="bold", pad=14,
)

ax.set_ylim(0.75, 0.96)
ax.set_xlim(1.5e-5, 0.03)

from matplotlib.ticker import FuncFormatter
def dollar_fmt(x, _):
    if x >= 0.01:
        return f"${x:.3f}"
    if x >= 0.001:
        return f"${x:.4f}"
    return f"${x:.5f}"

ax.xaxis.set_major_formatter(FuncFormatter(dollar_fmt))
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out_path = Path(__file__).parent / "figures" / "concept_cost_quality_spectrum.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
