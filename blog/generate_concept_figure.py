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

# Approximate the actual ParetoBandit Pareto frontier from Figure 1
# of the paper.  The curve passes BELOW the fixed-model stars because
# at any single budget the router is mixing models — it doesn't
# replicate a dedicated single-model's quality at that model's cost.
frontier_points = np.array([
    [3.0e-5,  0.812],   # near Llama cost, slightly above Llama quality
    [5.0e-5,  0.832],
    [8.0e-5,  0.852],
    [1.3e-4,  0.868],
    [2.3e-4,  0.885],
    [4.0e-4,  0.905],   # near Mistral cost, but below Mistral quality
    [7.0e-4,  0.916],
    [1.2e-3,  0.921],
    [3.0e-3,  0.928],
    [7.0e-3,  0.931],
    [1.5e-2,  0.932],   # converges near Gemini at loose budgets
])
from scipy.interpolate import PchipInterpolator
log_c = np.log10(frontier_points[:, 0])
q = frontier_points[:, 1]
interp = PchipInterpolator(log_c, q)

curve_costs = np.geomspace(frontier_points[0, 0], frontier_points[-1, 0], 120)
curve_qualities = interp(np.log10(curve_costs))

ax.plot(
    curve_costs, curve_qualities,
    color=CB_BLUE, linewidth=2.8, linestyle="-", alpha=0.8, zorder=5,
    marker="o", markersize=0,
)
ax.fill_between(
    curve_costs, curve_qualities - 0.006, curve_qualities + 0.006,
    color=CB_BLUE, alpha=0.08, zorder=4,
)

mid_idx = len(curve_costs) // 4
ax.annotate(
    "ParetoBandit\ncontinuous frontier",
    xy=(curve_costs[mid_idx], curve_qualities[mid_idx]),
    xytext=(40, -40),
    textcoords="offset points",
    fontsize=11, fontweight="bold", color=CB_BLUE,
    ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color=CB_BLUE, lw=1.8),
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
