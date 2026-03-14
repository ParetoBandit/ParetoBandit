#!/usr/bin/env python3
"""Generate side-by-side heatmaps of Pareto AUC for BanditGPT vs Tabula Rasa.

Reads ``results/hparam_sweep_results.json`` and produces:
  - ``results/hparam_sweep_heatmap.{pdf,png}``

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

VARIANT_LABELS: Dict[str, str] = {
    "banditgpt": "BanditGPT (warmup priors)",
    "tabula_rasa": "Tabula Rasa (cold start)",
}


def _build_heatmap(
    results: List[Dict],
    variant: str,
    alpha_values: List[float],
    pca_components: List[int],
) -> np.ndarray:
    """Build a 2-D (n_alpha, n_pca) heatmap for one variant."""
    n_alpha = len(alpha_values)
    n_pca = len(pca_components)
    heatmap = np.full((n_alpha, n_pca), np.nan)
    for entry in results:
        if entry["variant"] != variant:
            continue
        ai = alpha_values.index(entry["alpha"])
        pi = pca_components.index(entry["pca_dim"])
        heatmap[ai, pi] = entry["pareto_auc"]
    return heatmap


def main() -> None:
    with open(RESULTS_DIR / "hparam_sweep_results.json") as f:
        data = json.load(f)

    alpha_values: List[float] = data["grid"]["alpha_values"]
    pca_components: List[int] = data["grid"]["pca_components"]
    variants: List[str] = data["grid"]["variants"]
    best_per_variant = data["best_per_variant"]

    all_aucs = [e["pareto_auc"] for e in data["results"]]
    vmin = min(all_aucs) - 0.002
    vmax = max(all_aucs) + 0.002

    n_alpha = len(alpha_values)
    n_pca = len(pca_components)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.5), sharey=True)

    for ax_idx, variant in enumerate(variants):
        ax = axes[ax_idx]
        heatmap = _build_heatmap(
            data["results"], variant, alpha_values, pca_components,
        )

        im = ax.imshow(
            heatmap,
            aspect="auto",
            cmap="YlOrRd_r",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks(range(n_pca))
        ax.set_xticklabels([str(d) for d in pca_components])
        ax.set_xlabel("PCA Components", fontsize=11)

        if ax_idx == 0:
            ax.set_yticks(range(n_alpha))
            ax.set_yticklabels([f"{a:.2f}" for a in alpha_values])
            ax.set_ylabel(r"$\alpha$ (exploration)", fontsize=11)

        ax.set_title(VARIANT_LABELS[variant], fontsize=12)

        mid = (vmin + vmax) / 2
        for ai in range(n_alpha):
            for pi in range(n_pca):
                val = heatmap[ai, pi]
                if np.isnan(val):
                    continue
                text_color = "white" if val < mid else "black"
                ax.text(
                    pi, ai, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold",
                )

        best = best_per_variant[variant]
        best_ai = alpha_values.index(best["alpha"])
        best_pi = pca_components.index(best["pca_dim"])
        ax.add_patch(plt.Rectangle(
            (best_pi - 0.5, best_ai - 0.5), 1, 1,
            fill=False, edgecolor="blue", linewidth=2.5,
        ))

        x_off = min(best_pi + 2.5, n_pca - 1)
        y_off = min(best_ai + 1.5, n_alpha - 0.5)
        ax.annotate(
            f"Best: α={best['alpha']}, d={best['pca_dim']}",
            xy=(best_pi, best_ai),
            xytext=(x_off, y_off),
            fontsize=8,
            fontweight="bold",
            color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        )

    fig.subplots_adjust(right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Pareto AUC", fontsize=10)

    fig.suptitle(
        r"Alpha × PCA Sweep — Pareto AUC (K=3, $\gamma$=1.0)",
        fontsize=13, y=0.98,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"hparam_sweep_heatmap.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved hparam_sweep_heatmap.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
