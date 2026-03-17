#!/usr/bin/env python3
"""Generate figure for Appendix: Cold-Start vs Warmup Prior Regret.

Reads ``results/warmup_ablation_results.json`` and produces a
three-panel cumulative regret figure (``warmup_ablation.pdf/.png``)
showing unconstrained, tight-budget, and moderate-budget comparisons.

Usage:
    python experiments/appendix/warmup_ablation/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"

PANELS: List[Tuple[str, str, str, str]] = [
    (
        "ParetoBandit (warmup)",
        "Tabula Rasa",
        "Random",
        "Unconstrained",
    ),
    (
        "Warmup (tight budget)",
        "Tabula Rasa (tight budget)",
        "Random",
        "Tight budget",
    ),
    (
        "Warmup (moderate budget)",
        "Tabula Rasa (moderate budget)",
        "Random",
        "Moderate budget",
    ),
]


def _plot_condition(
    ax: plt.Axes,
    curves: List[Dict[str, Any]],
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot one condition's cumulative regret with bootstrap CI."""
    steps = [c["step"] for c in curves]
    mean_reg = [c["mean_cumulative_regret"] for c in curves]

    if "per_seed_cumulative_regret" in curves[0]:
        matrix = np.array([c["per_seed_cumulative_regret"] for c in curves])
        ci_lo, ci_hi = bootstrap_ci_series(matrix)
    else:
        se_reg = [c["se_cumulative_regret"] for c in curves]
        ci_lo = [m - s for m, s in zip(mean_reg, se_reg)]
        ci_hi = [m + s for m, s in zip(mean_reg, se_reg)]

    ax.plot(
        steps, mean_reg,
        color=color, linestyle=linestyle, linewidth=1.8, label=label,
    )
    ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)


def main() -> None:
    with open(RESULTS_DIR / "warmup_ablation_results.json") as f:
        data = json.load(f)

    conditions = data["conditions"]
    early_step = data["early_step"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

    for ax, (warmup_key, tabula_key, random_key, title) in zip(axes, PANELS):
        _plot_condition(
            ax, conditions[warmup_key]["curves"],
            color=CB_BLUE, linestyle="-", label="Warmup",
        )
        _plot_condition(
            ax, conditions[tabula_key]["curves"],
            color=CB_ORANGE, linestyle="-", label="Tabula Rasa",
        )
        _plot_condition(
            ax, conditions[random_key]["curves"],
            color=CB_GRAY, linestyle="--", label="Random",
        )

        ax.axvline(
            x=early_step, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
        )
        ax.set_xlabel("Step")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative Regret")
    axes[0].legend(fontsize=9, loc="upper left")

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
