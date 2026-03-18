#!/usr/bin/env python3
"""Generate figures for the Prior Mismatch Sensitivity Analysis.

Reads ``results/prior_mismatch_results.json`` and produces two figures:

  1. **Cumulative regret curves** (``prior_mismatch_regret.pdf/.png``)
     — one panel per ``n_eff`` value showing all prior-quality levels.

  2. **Summary heatmap** (``prior_mismatch_heatmap.pdf/.png``)
     — total regret as a function of prior quality × n_eff, with the
     tabula-rasa baseline drawn as a horizontal reference.

Usage:
    python experiments/appendix/prior_mismatch/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "Well-calibrated": "#0072B2",
    "Random-1680": "#56B4E9",
    "MMLU-only": "#009E73",
    "GSM8K-only": "#E69F00",
    "Inverted": "#D55E00",
    "Tabula Rasa": "#999999",
}
QUALITY_ORDER = [
    "Well-calibrated", "Random-1680", "MMLU-only", "GSM8K-only", "Inverted",
]
N_EFF_VALUES = [10, 100, 1000]


def _plot_condition(
    ax: plt.Axes,
    curves: List[Dict[str, Any]],
    *,
    color: str,
    linestyle: str = "-",
    label: str,
    linewidth: float = 1.8,
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
        color=color, linestyle=linestyle, linewidth=linewidth, label=label,
    )
    ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.12)


# ======================================================================
# Figure 1: Cumulative Regret Curves (one panel per n_eff)
# ======================================================================


def generate_regret_figure(data: Dict[str, Any]) -> None:
    """Three-panel cumulative regret figure, one per n_eff value."""
    conditions = data["conditions"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)

    for ax, n_eff in zip(axes, N_EFF_VALUES):
        # Tabula Rasa baseline (same in all panels)
        tr_key = "Tabula Rasa"
        if tr_key in conditions:
            _plot_condition(
                ax, conditions[tr_key]["curves"],
                color=COLORS["Tabula Rasa"], linestyle="--",
                label="Tabula Rasa",
            )

        for quality in QUALITY_ORDER:
            key = f"{quality} (n_eff={n_eff})"
            if key not in conditions:
                continue
            _plot_condition(
                ax, conditions[key]["curves"],
                color=COLORS[quality], label=quality,
            )

        early_step = data.get("early_step", 200)
        ax.axvline(
            x=early_step, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
        )
        ax.set_xlabel("Step")
        ax.set_title(f"n_eff = {n_eff}", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Cumulative Regret")
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Prior Mismatch Sensitivity: Cumulative Regret by Prior Quality and n_eff",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"prior_mismatch_regret.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved prior_mismatch_regret.pdf/.png to {RESULTS_DIR}")


# ======================================================================
# Figure 2: Summary Heatmap (prior quality × n_eff)
# ======================================================================


def generate_heatmap_figure(data: Dict[str, Any]) -> None:
    """Heatmap of total regret: prior quality (rows) × n_eff (columns).

    Annotates each cell with mean ± SE.  A horizontal dashed line marks
    the tabula-rasa baseline, and cells that exceed it are highlighted.
    """
    conditions = data["conditions"]
    early_step = data.get("early_step", 200)

    # Build the regret matrix
    n_qualities = len(QUALITY_ORDER)
    n_neffs = len(N_EFF_VALUES)
    regret_matrix = np.full((n_qualities, n_neffs), np.nan)
    early_matrix = np.full((n_qualities, n_neffs), np.nan)
    se_matrix = np.full((n_qualities, n_neffs), np.nan)
    early_se_matrix = np.full((n_qualities, n_neffs), np.nan)

    for qi, quality in enumerate(QUALITY_ORDER):
        for ni, n_eff in enumerate(N_EFF_VALUES):
            key = f"{quality} (n_eff={n_eff})"
            if key in conditions:
                regret_matrix[qi, ni] = conditions[key]["total_regret"]["mean"]
                se_matrix[qi, ni] = conditions[key]["total_regret"]["se"]
                rk = f"regret_at_{early_step}"
                early_matrix[qi, ni] = conditions[key][rk]["mean"]
                early_se_matrix[qi, ni] = conditions[key][rk]["se"]

    tr_regret = conditions.get("Tabula Rasa", {}).get(
        "total_regret", {},
    ).get("mean", np.nan)
    tr_early = conditions.get("Tabula Rasa", {}).get(
        f"regret_at_{early_step}", {},
    ).get("mean", np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    for ax, matrix, se_mat, tr_val, title in [
        (ax1, regret_matrix, se_matrix, tr_regret, "Total Regret"),
        (ax2, early_matrix, early_se_matrix, tr_early, f"R@{early_step} (Early Regret)"),
    ]:
        vmin = min(np.nanmin(matrix), tr_val) * 0.95
        vmax = max(np.nanmax(matrix), tr_val) * 1.05

        im = ax.imshow(
            matrix, aspect="auto", cmap="RdYlGn_r",
            vmin=vmin, vmax=vmax,
        )

        ax.set_xticks(range(n_neffs))
        ax.set_xticklabels([str(int(v)) for v in N_EFF_VALUES])
        ax.set_yticks(range(n_qualities))
        ax.set_yticklabels(QUALITY_ORDER)
        ax.set_xlabel("n_eff")
        ax.set_title(title, fontsize=11)

        for qi in range(n_qualities):
            for ni in range(n_neffs):
                val = matrix[qi, ni]
                se = se_mat[qi, ni]
                if np.isnan(val):
                    continue
                exceeds_tr = val > tr_val
                fontweight = "bold" if exceeds_tr else "normal"
                color = "white" if val > (vmin + vmax) / 2 else "black"
                ax.text(
                    ni, qi, f"{val:.1f}\n±{se:.1f}",
                    ha="center", va="center", fontsize=8,
                    fontweight=fontweight, color=color,
                )

        # Tabula Rasa reference annotation
        ax.text(
            n_neffs - 0.5, -0.6,
            f"TR = {tr_val:.1f}",
            ha="right", va="center", fontsize=9,
            fontstyle="italic", color=COLORS["Tabula Rasa"],
        )

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    fig.suptitle(
        "Prior Mismatch × n_eff: When Do Warmup Priors Hurt?",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"prior_mismatch_heatmap.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved prior_mismatch_heatmap.pdf/.png to {RESULTS_DIR}")


# ======================================================================
# Figure 3: Bar chart — total regret by condition
# ======================================================================


def generate_bar_figure(data: Dict[str, Any]) -> None:
    """Grouped bar chart: total regret for each (prior quality, n_eff)."""
    conditions = data["conditions"]
    early_step = data.get("early_step", 200)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    tr_regret = conditions["Tabula Rasa"]["total_regret"]["mean"]
    tr_regret_se = conditions["Tabula Rasa"]["total_regret"]["se"]
    tr_early = conditions["Tabula Rasa"][f"regret_at_{early_step}"]["mean"]

    bar_width = 0.18
    x = np.arange(len(QUALITY_ORDER))

    for offset_idx, n_eff in enumerate(N_EFF_VALUES):
        means = []
        ses = []
        early_means = []
        early_ses = []

        for quality in QUALITY_ORDER:
            key = f"{quality} (n_eff={n_eff})"
            cond = conditions.get(key, {})
            means.append(cond.get("total_regret", {}).get("mean", np.nan))
            ses.append(cond.get("total_regret", {}).get("se", 0))
            rk = f"regret_at_{early_step}"
            early_means.append(cond.get(rk, {}).get("mean", np.nan))
            early_ses.append(cond.get(rk, {}).get("se", 0))

        offset = (offset_idx - 1) * bar_width
        ax1.bar(
            x + offset, means, bar_width,
            yerr=ses, capsize=3,
            label=f"n_eff={int(n_eff)}", alpha=0.85,
        )
        ax2.bar(
            x + offset, early_means, bar_width,
            yerr=early_ses, capsize=3,
            label=f"n_eff={int(n_eff)}", alpha=0.85,
        )

    ax1.axhline(
        y=tr_regret, color=COLORS["Tabula Rasa"],
        linestyle="--", linewidth=1.5, label=f"Tabula Rasa ({tr_regret:.1f})",
    )
    ax2.axhline(
        y=tr_early, color=COLORS["Tabula Rasa"],
        linestyle="--", linewidth=1.5, label=f"Tabula Rasa ({tr_early:.1f})",
    )

    ax1.set_ylabel("Total Regret")
    ax1.set_title("Total Cumulative Regret by Prior Quality and n_eff")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.set_ylabel(f"R@{early_step}")
    ax2.set_title(f"Early Regret (R@{early_step}) by Prior Quality and n_eff")
    ax2.set_xticks(x)
    ax2.set_xticklabels(QUALITY_ORDER)
    ax2.set_xlabel("Prior Quality")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"prior_mismatch_bars.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved prior_mismatch_bars.pdf/.png to {RESULTS_DIR}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    results_path = RESULTS_DIR / "prior_mismatch_results.json"
    with open(results_path) as f:
        data = json.load(f)

    generate_regret_figure(data)
    generate_heatmap_figure(data)
    generate_bar_figure(data)


if __name__ == "__main__":
    main()
