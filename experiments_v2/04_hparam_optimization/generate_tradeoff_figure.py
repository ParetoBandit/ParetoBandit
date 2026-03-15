#!/usr/bin/env python3
"""Generate the efficiency-vs-adaptability tradeoff figure.

Reads ``results/hparam_sweep_results.json`` and produces:
  - ``results/epsilon_tradeoff.{pdf,png}``

The figure shows each (alpha, n_eff, gamma) configuration as a
point in (Budget-Paced AUC, Phase-2 Regret) space, coloured by
gamma.  The epsilon-feasible region is shaded, and the selected
configurations are highlighted.

Usage::

    python experiments_v2/04_hparam_optimization/generate_tradeoff_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

GAMMA_COLORS: Dict[float, str] = {
    0.995: "#D55E00",
    0.997: "#CC79A7",
    0.999: "#009E73",
    1.0:   "#0072B2",
}
GAMMA_LABELS: Dict[float, str] = {
    0.995: r"$\gamma=0.995$",
    0.997: r"$\gamma=0.997$",
    0.999: r"$\gamma=0.999$",
    1.0:   r"$\gamma=1.0$ (no forgetting)",
}

VARIANT_TITLES: Dict[str, str] = {
    "banditgpt": "BanditGPT (warmup priors)",
    "tabula_rasa": "Tabula Rasa (cold start)",
}

EPSILON = 0.05


def _load_data() -> Dict[str, Any]:
    with open(RESULTS_DIR / "hparam_sweep_results.json") as f:
        return json.load(f)


def _build_merged(
    stat_results: List[Dict[str, Any]],
    nonstat_results: List[Dict[str, Any]],
    variant: str,
) -> List[Dict[str, Any]]:
    """Merge stationary AUC and non-stationary regret for a variant."""
    var_stat = [r for r in stat_results if r["variant"] == variant]
    var_ns = [r for r in nonstat_results if r["variant"] == variant]

    merged: List[Dict[str, Any]] = []
    for s in var_stat:
        ns_match = [
            r for r in var_ns
            if r["alpha"] == s["alpha"]
            and r["n_eff"] == s["n_eff"]
            and r["gamma"] == s["gamma"]
        ]
        if not ns_match:
            continue
        merged.append({
            "alpha": s["alpha"],
            "n_eff": s["n_eff"],
            "gamma": s["gamma"],
            "auc": s["val_pareto_auc"],
            "p2_regret": ns_match[0]["phase2_regret"],
            "p2_std": ns_match[0]["phase2_regret_std"],
        })
    return merged


def main() -> None:
    data = _load_data()
    stat_results = data["val_budget_paced"]
    ns_results = data["val_nonstationary"]
    best_per_variant = data["best_per_variant"]
    auc_only_best = data["auc_only_best"]
    variants = data["grid"]["variants"]

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.2), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for ax, variant in zip(axes, variants):
        merged = _build_merged(stat_results, ns_results, variant)
        best_auc = max(m["auc"] for m in merged)
        threshold = best_auc * (1.0 - EPSILON)

        ax.axvspan(
            threshold, best_auc + 0.001,
            color="#E8F5E9", alpha=0.7, zorder=0,
            label=r"$\epsilon$-feasible ($\geq$95\% of max)",
        )
        ax.axvline(threshold, color="#66BB6A", ls="--", lw=0.8, alpha=0.7)

        for gamma_val in sorted(GAMMA_COLORS.keys()):
            pts = [m for m in merged if m["gamma"] == gamma_val]
            if not pts:
                continue
            xs = [p["auc"] for p in pts]
            ys = [p["p2_regret"] for p in pts]
            ax.scatter(
                xs, ys,
                c=GAMMA_COLORS[gamma_val],
                s=28, alpha=0.6, edgecolors="none",
                label=GAMMA_LABELS[gamma_val],
                zorder=2,
            )

        sel = best_per_variant[variant]
        sel_pt = [
            m for m in merged
            if m["alpha"] == sel["alpha"]
            and m["n_eff"] == sel["n_eff"]
            and m["gamma"] == sel["gamma"]
        ]
        if sel_pt:
            ax.scatter(
                [sel_pt[0]["auc"]], [sel_pt[0]["p2_regret"]],
                marker="*", s=220, c="black", zorder=4,
                label=r"$\epsilon$-constraint selected",
            )

        auc_b = auc_only_best[variant]
        auc_pt = [
            m for m in merged
            if m["alpha"] == auc_b["alpha"]
            and m["gamma"] == auc_b["gamma"]
            and (variant == "tabula_rasa" or m["n_eff"] == auc_b["n_eff"])
        ]
        if auc_pt:
            ax.scatter(
                [auc_pt[0]["auc"]], [auc_pt[0]["p2_regret"]],
                marker="X", s=140, c="red", zorder=4, edgecolors="darkred",
                linewidths=0.5,
                label="AUC-only selected",
            )

        ax.set_title(VARIANT_TITLES[variant], fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget-Paced Pareto AUC (val)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2, ls="--")

    axes[0].set_ylabel("Phase-2 Cumulative Regret\n(lower = better adaptation)", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc="lower center", ncol=3, fontsize=8.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        r"Efficiency--Adaptability Tradeoff ($K{=}3$, PCA-25, 10 seeds)",
        fontsize=12, y=1.0,
    )
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.08)

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"epsilon_tradeoff.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved epsilon_tradeoff.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
