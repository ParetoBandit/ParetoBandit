#!/usr/bin/env python3
"""Generate the epsilon-constraint tradeoff figure (ParetoBandit only).

Reads ``results/hparam_sweep_results.json`` and produces:
  - ``results/epsilon_tradeoff.{pdf,png}``

Single-panel scatter of (Budget-Paced AUC, Phase-2 Regret) for every
ParetoBandit configuration, coloured by gamma.  The epsilon-feasible
region is shaded, and the epsilon-constraint winner and AUC-only
winner are highlighted with annotations.

Usage::

    python experiments/05_hparam_optimization/generate_tradeoff_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

EPSILON = 0.0025


def _load_data() -> Dict[str, Any]:
    with open(RESULTS_DIR / "hparam_sweep_results.json") as f:
        return json.load(f)


def _build_merged(
    stat_results: List[Dict[str, Any]],
    nonstat_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge stationary AUC and non-stationary regret for ParetoBandit."""
    var_stat = [r for r in stat_results if r["variant"] == "paretobandit"]
    var_ns = [r for r in nonstat_results if r["variant"] == "paretobandit"]

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
    best = data["best_per_variant"]["paretobandit"]
    auc_best = data["auc_only_best"]["paretobandit"]

    merged = _build_merged(stat_results, ns_results)
    best_auc = max(m["auc"] for m in merged)
    threshold = best_auc * (1.0 - EPSILON)

    all_aucs = [m["auc"] for m in merged]
    all_regs = [m["p2_regret"] for m in merged]
    x_lo = min(all_aucs) - 0.001
    x_hi = max(all_aucs) + 0.002

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # --- Epsilon-feasible region (clipped to visible x-range) ---
    ax.axvspan(
        max(threshold, x_lo), x_hi,
        color="#E8F5E9", alpha=0.6, zorder=0,
    )
    if threshold >= x_lo:
        ax.axvline(threshold, color="#66BB6A", ls="--", lw=0.9, alpha=0.8)
        ax.text(
            threshold - 0.0003, max(all_regs) + 0.3,
            r"$\epsilon$-feasible" "\n" r"($\geq$99.75% best AUC)",
            fontsize=7.5, color="#388E3C", ha="right", va="top",
        )

    # --- Scatter by gamma ---
    for gamma_val in sorted(GAMMA_COLORS.keys()):
        pts = [m for m in merged if m["gamma"] == gamma_val]
        if not pts:
            continue
        xs = [p["auc"] for p in pts]
        ys = [p["p2_regret"] for p in pts]
        ax.scatter(
            xs, ys,
            c=GAMMA_COLORS[gamma_val],
            s=36, alpha=0.65, edgecolors="white", linewidths=0.3,
            label=GAMMA_LABELS[gamma_val],
            zorder=2,
        )

    # --- Epsilon-constraint winner ---
    sel_pt = next(
        m for m in merged
        if m["alpha"] == best["alpha"]
        and m["n_eff"] == best["n_eff"]
        and m["gamma"] == best["gamma"]
    )
    ax.scatter(
        [sel_pt["auc"]], [sel_pt["p2_regret"]],
        marker="*", s=280, c="black", zorder=5,
        label=r"$\epsilon$-constraint selected",
    )
    ax.annotate(
        f"Selected: $\\gamma={sel_pt['gamma']}$, regret={sel_pt['p2_regret']:.0f}",
        xy=(sel_pt["auc"], sel_pt["p2_regret"]),
        xytext=(-40, -22), textcoords="offset points",
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # --- AUC-only winner ---
    auc_pt = next(
        m for m in merged
        if m["alpha"] == auc_best["alpha"]
        and m["n_eff"] == auc_best["n_eff"]
        and m["gamma"] == auc_best["gamma"]
    )
    ax.scatter(
        [auc_pt["auc"]], [auc_pt["p2_regret"]],
        marker="X", s=160, c="red", zorder=5,
        edgecolors="darkred", linewidths=0.5,
        label="AUC-only selected",
    )
    ax.annotate(
        f"AUC-only: $\\gamma={auc_pt['gamma']}$, regret={auc_pt['p2_regret']:.0f}",
        xy=(auc_pt["auc"], auc_pt["p2_regret"]),
        xytext=(30, 18), textcoords="offset points",
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # --- Axis labels and formatting ---
    ax.set_xlim(x_lo, x_hi)
    y_pad = (max(all_regs) - min(all_regs)) * 0.12
    ax.set_ylim(min(all_regs) - y_pad, max(all_regs) + y_pad)

    ax.set_xlabel("Budget-Paced Pareto AUC (val)", fontsize=11)
    ax.set_ylabel(
        "Phase-2 Cumulative Regret\n(lower = better adaptation)",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.2, ls="--")

    ax.legend(
        loc="upper left", fontsize=8, framealpha=0.9,
        borderpad=0.6, handletextpad=0.4,
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"epsilon_tradeoff.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved epsilon_tradeoff.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
