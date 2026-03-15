#!/usr/bin/env python3
"""Generate figure for Appendix: Cold-Start vs Warmup Prior Regret.

Reads ``results/warmup_ablation_results.json`` and produces a
single-panel cumulative regret figure (``warmup_ablation.pdf/.png``).

Usage:
    python experiments/appendix/warmup_ablation/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"

CONDITION_STYLES: Dict[str, Dict[str, Any]] = {
    "BanditGPT (warmup)": {
        "color": CB_BLUE,
        "linestyle": "-",
        "label": "BanditGPT (warmup)",
    },
    "Tabula Rasa": {
        "color": CB_ORANGE,
        "linestyle": "-",
        "label": "Tabula Rasa (cold start)",
    },
    "Random": {
        "color": CB_GRAY,
        "linestyle": "--",
        "label": "Random",
    },
}

PLOT_ORDER = ["BanditGPT (warmup)", "Tabula Rasa", "Random"]


def main() -> None:
    with open(RESULTS_DIR / "warmup_ablation_results.json") as f:
        data = json.load(f)

    conditions = data["conditions"]
    early_step = data["early_step"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    for cond_key in PLOT_ORDER:
        cond = conditions[cond_key]
        curves = cond["curves"]
        style = CONDITION_STYLES[cond_key]

        steps = [c["step"] for c in curves]
        mean_reg = [c["mean_cumulative_regret"] for c in curves]
        se_reg = [c["se_cumulative_regret"] for c in curves]

        ax.plot(
            steps,
            mean_reg,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )
        lower = [m - s for m, s in zip(mean_reg, se_reg)]
        upper = [m + s for m, s in zip(mean_reg, se_reg)]
        ax.fill_between(steps, lower, upper, color=style["color"], alpha=0.15)

    ax.axvline(
        x=early_step,
        color="black",
        linestyle=":",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.text(
        early_step + 15,
        ax.get_ylim()[1] * 0.55,
        f"step {early_step}",
        fontsize=8,
        alpha=0.6,
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("K=3 Stationary: Warmup Priors vs Cold Start")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

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
