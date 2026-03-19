#!/usr/bin/env python3
"""Generate alpha-sweep figure comparing ParetoBandit, Tabula Rasa, and SW-UCB.

Reads ``results/hparam_sweep_results.json`` and produces:
  - ``results/hparam_sweep_alpha.{pdf,png}``

Usage::

    python experiments/05_hparam_optimization/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import (
    BEST_K3_HPARAMS,
    BEST_K3_SW_UCB_HPARAMS,
    BEST_K3_TABULA_RASA_HPARAMS,
)

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"

VARIANT_STYLE: Dict[str, Dict[str, Any]] = {
    "paretobandit": {"color": CB_BLUE, "label": "ParetoBandit (warmup)"},
    "tabula_rasa": {"color": CB_ORANGE, "label": "Tabula Rasa (cold start)"},
    "sw_ucb": {"color": CB_GREEN, "label": f"SW-UCB (W={BEST_K3_SW_UCB_HPARAMS['window_size']})"},
}


def _find_entry(
    val_results: List[Dict[str, Any]],
    variant: str,
    alpha: float,
    *,
    gamma: float = 1.0,
    n_eff: float = 1.0,
    window_size: int = 0,
) -> Dict[str, Any]:
    """Look up a single sweep entry by its hyperparameter coordinates."""
    return next(
        r for r in val_results
        if r["variant"] == variant
        and r["alpha"] == alpha
        and r["gamma"] == gamma
        and r["n_eff"] == n_eff
        and r["window_size"] == window_size
    )


def main() -> None:
    with open(RESULTS_DIR / "hparam_sweep_results.json") as f:
        data = json.load(f)

    alpha_values: List[float] = data["grid"]["alpha_values"]
    plot_variants: List[str] = ["paretobandit", "tabula_rasa", "sw_ucb"]
    best_per_variant = data["best_per_variant"]
    val_results = data["val_budget_paced_full"]

    BEST_NEFF: Dict[str, float] = {
        "paretobandit": BEST_K3_HPARAMS["prior_n_effective"],
        "tabula_rasa": BEST_K3_TABULA_RASA_HPARAMS["prior_n_effective"],
    }
    PLOT_GAMMA = 1.0
    BEST_SW_WINDOW: int = BEST_K3_SW_UCB_HPARAMS["window_size"]

    auc_by_variant: Dict[str, List[float]] = {v: [] for v in plot_variants}
    std_by_variant: Dict[str, List[float]] = {v: [] for v in plot_variants}
    for alpha in alpha_values:
        for variant in plot_variants:
            if variant == "sw_ucb":
                entry = _find_entry(
                    val_results, variant, alpha,
                    gamma=1.0, n_eff=1.0, window_size=BEST_SW_WINDOW,
                )
            else:
                entry = _find_entry(
                    val_results, variant, alpha,
                    gamma=PLOT_GAMMA, n_eff=BEST_NEFF[variant],
                )
            auc_by_variant[variant].append(entry["val_pareto_auc"])
            std_by_variant[variant].append(entry["val_pareto_auc_std"])

    n_variants = len(plot_variants)
    x = np.arange(len(alpha_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, variant in enumerate(plot_variants):
        style = VARIANT_STYLE[variant]
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            auc_by_variant[variant],
            width,
            yerr=std_by_variant[variant],
            capsize=3,
            label=style["label"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "alpha": 0.6},
        )

        best_alpha = best_per_variant[variant]["alpha"]
        if best_alpha in alpha_values:
            best_idx = alpha_values.index(best_alpha)
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2.0)

    ax.set_xlabel(r"$\alpha$ (exploration parameter)", fontsize=11)
    ax.set_ylabel(f"Val Pareto AUC (per-seed, {data['grid']['n_seeds']} seeds)", fontsize=11)
    pca_d = BEST_K3_HPARAMS["pca_components"]
    neff_w = int(BEST_NEFF["paretobandit"])
    neff_c = int(BEST_NEFF["tabula_rasa"])
    ax.set_title(
        rf"Alpha Sweep — K=3, PCA-{pca_d}, "
        rf"$\gamma$={PLOT_GAMMA}, "
        rf"$n_{{\mathrm{{eff}}}}$={neff_w}/{neff_c} (warmup/cold), "
        rf"W={BEST_SW_WINDOW} (SW-UCB)",
        fontsize=11,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}" for a in alpha_values])

    all_aucs = [a for v in plot_variants for a in auc_by_variant[v]]
    y_lo = min(all_aucs) - 0.005
    y_hi = max(all_aucs) + 0.005
    ax.set_ylim(y_lo, y_hi)

    for i, variant in enumerate(plot_variants):
        offset = (i - (n_variants - 1) / 2) * width
        for j, auc in enumerate(auc_by_variant[variant]):
            ax.text(
                x[j] + offset,
                auc + std_by_variant[variant][j] + 0.0003,
                f"{auc:.4f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
            )

    test_per_variant = data.get("test_per_variant", {})
    if test_per_variant:
        note_parts = []
        for variant in plot_variants:
            t = test_per_variant.get(variant, {})
            if t:
                label = VARIANT_STYLE[variant]["label"].split(" (")[0]
                note_parts.append(
                    f"{label}: test AUC={t['test_pareto_auc']:.4f}"
                )
        if note_parts:
            note = "Holdout: " + ", ".join(note_parts)
            ax.annotate(
                note, xy=(0.5, 0.02), xycoords="axes fraction",
                fontsize=7, ha="center", fontstyle="italic", color="0.4",
            )

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
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
