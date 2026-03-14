#!/usr/bin/env python3
"""Generate alpha-sweep figure comparing BanditGPT vs Tabula Rasa.

Reads ``results/hparam_sweep_results.json`` and produces:
  - ``results/hparam_sweep_alpha.{pdf,png}``

Usage::

    python experiments_v2/appendix/hparam_sweep/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"

VARIANT_STYLE: Dict[str, Dict] = {
    "banditgpt": {"color": CB_BLUE, "label": "BanditGPT (warmup)"},
    "tabula_rasa": {"color": CB_ORANGE, "label": "Tabula Rasa (cold start)"},
}


def main() -> None:
    with open(RESULTS_DIR / "hparam_sweep_results.json") as f:
        data = json.load(f)

    alpha_values: List[float] = data["grid"]["alpha_values"]
    variants: List[str] = data["grid"]["variants"]
    best_per_variant = data["best_per_variant"]

    auc_by_variant: Dict[str, List[float]] = {v: [] for v in variants}
    for alpha in alpha_values:
        for variant in variants:
            entry = next(
                r for r in data["results"]
                if r["variant"] == variant and r["alpha"] == alpha
            )
            auc_by_variant[variant].append(entry["pareto_auc"])

    x = np.arange(len(alpha_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, variant in enumerate(variants):
        style = VARIANT_STYLE[variant]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            auc_by_variant[variant],
            width,
            label=style["label"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
        )

        best_alpha = best_per_variant[variant]["alpha"]
        best_idx = alpha_values.index(best_alpha)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.0)

    ax.set_xlabel(r"$\alpha$ (exploration parameter)", fontsize=11)
    ax.set_ylabel("Pareto AUC (holdout)", fontsize=11)
    ax.set_title(
        r"Alpha Sweep — K=3, PCA-25, $\gamma$=1.0",
        fontsize=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}" for a in alpha_values])

    all_aucs = [a for v in variants for a in auc_by_variant[v]]
    y_lo = min(all_aucs) - 0.005
    y_hi = max(all_aucs) + 0.005
    ax.set_ylim(y_lo, y_hi)

    for i, variant in enumerate(variants):
        offset = (i - 0.5) * width
        for j, auc in enumerate(auc_by_variant[variant]):
            ax.text(
                x[j] + offset,
                auc + 0.0005,
                f"{auc:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"hparam_sweep_alpha.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved hparam_sweep_alpha.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
