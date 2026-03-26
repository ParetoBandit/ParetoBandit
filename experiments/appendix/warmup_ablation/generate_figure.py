#!/usr/bin/env python3
"""Generate figure for Appendix: Cold-Start vs Warmup Prior Regret.

Reads ``results/warmup_ablation_results.json`` and produces a
violin + strip plot (``warmup_ablation.pdf/.png``) showing per-seed
total regret distributions for warmup, tabula rasa, and the
matched-γ mechanistic control across all budget regimes.

Usage:
    python experiments/appendix/warmup_ablation/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_TEAL = "#009E73"
CB_GRAY = "#999999"

GroupSpec = Tuple[str, str, str]

BUDGET_GROUPS: List[GroupSpec] = [
    ("ParetoBandit (warmup)", "Tabula Rasa", "Unconstrained"),
    ("Warmup (tight budget)", "Tabula Rasa (tight budget)", "Tight"),
    ("Warmup (moderate budget)", "Tabula Rasa (moderate budget)", "Moderate"),
    ("Warmup (loose budget)", "Tabula Rasa (loose budget)", "Loose"),
]


def _halfviolin(
    ax: plt.Axes,
    data: np.ndarray,
    center: float,
    side: str,
    color: str,
    width: float = 0.35,
) -> None:
    """Draw a half-violin (KDE density) on one side of *center*.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data : 1-D array
        Sample values.
    center : float
        X-position of the violin centre line.
    side : {"left", "right"}
        Which side to draw on.
    color : str
        Fill colour.
    width : float
        Maximum half-width of the violin in data-coordinate units.
    """
    from scipy.stats import gaussian_kde

    if len(np.unique(data)) < 2:
        ax.plot([center, center], [data.min(), data.max()], color=color, lw=2)
        return

    kde = gaussian_kde(data, bw_method=0.4)
    y_grid = np.linspace(data.min() - 5, data.max() + 5, 200)
    density = kde(y_grid)
    density = density / density.max() * width

    if side == "left":
        ax.fill_betweenx(y_grid, center - density, center, alpha=0.25, color=color)
        ax.plot(center - density, y_grid, color=color, lw=0.8, alpha=0.6)
    else:
        ax.fill_betweenx(y_grid, center, center + density, alpha=0.25, color=color)
        ax.plot(center + density, y_grid, color=color, lw=0.8, alpha=0.6)


def _format_p(p: float) -> str:
    """Format a p-value for figure annotation."""
    if p < 1e-4:
        return "< 10⁻⁴"
    if p < 0.001:
        return f"{p:.4f}"
    if p < 0.05:
        return f"{p:.3f}"
    return f"{p:.2f}"


def main() -> None:
    with open(RESULTS_DIR / "warmup_ablation_results.json") as f:
        data = json.load(f)

    conditions = data["conditions"]
    available = [
        g for g in BUDGET_GROUPS
        if g[0] in conditions and g[1] in conditions
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    x_positions = np.arange(len(available))
    jitter_w = 0.05
    violin_w = 0.22
    rng = np.random.default_rng(42)

    offsets = {"warmup": -0.20, "tabula": 0.20}

    random_regrets: np.ndarray | None = None
    if "Random" in conditions:
        random_regrets = np.array(conditions["Random"]["per_seed_regret"])

    for i, (warmup_key, tabula_key, budget_label) in enumerate(available):
        cx = x_positions[i]

        strip_specs: List[Tuple[str, str, float, str, str | None]] = [
            (warmup_key, CB_BLUE, offsets["warmup"], "left",
             "Warmup" if i == 0 else None),
            (tabula_key, CB_ORANGE, offsets["tabula"], "right",
             "Tabula Rasa" if i == 0 else None),
        ]

        for cond_key, color, x_off, side, label in strip_specs:
            seeds = np.array(conditions[cond_key]["per_seed_regret"])
            pos = cx + x_off

            _halfviolin(ax, seeds, pos, side, color, width=violin_w)

            jitter = rng.uniform(-jitter_w, jitter_w, size=len(seeds))
            ax.scatter(
                pos + jitter, seeds,
                color=color, s=26, alpha=0.7, zorder=5,
                edgecolors="white", linewidths=0.3, label=label,
            )

            med = np.median(seeds)
            ax.hlines(
                med, pos - 0.07, pos + 0.07,
                color=color, linewidth=2.0, zorder=6,
            )

    if random_regrets is not None:
        unconstrained_x = x_positions[0]
        ax.scatter(
            [unconstrained_x], [np.median(random_regrets)],
            marker="*", s=260, color=CB_GRAY, zorder=7,
            edgecolors="black", linewidths=0.5,
            label="Random (unconstrained)",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([g[2] for g in available], fontsize=17)
    ax.set_xlabel("Budget Regime", fontsize=20, labelpad=18)
    ax.set_ylabel("Total Regret (per seed)", fontsize=20)
    ax.set_title(
        "Cold-Start Ablation",
        fontsize=17, fontweight="bold",
    )
    ax.legend(
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        borderaxespad=0.0,
        framealpha=0.9,
    )
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=16)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"warmup_ablation.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved warmup_ablation.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
