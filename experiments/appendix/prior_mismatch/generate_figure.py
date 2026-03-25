#!/usr/bin/env python3
"""Generate figures for the Prior Mismatch Sensitivity Analysis.

Reads ``results/prior_mismatch_results.json`` and produces two figures:

  1. **Summary heatmap** (``prior_mismatch_heatmap.pdf/.png``)
     — total regret as a function of prior quality × n_eff, with the
     tabula-rasa baseline drawn as a horizontal reference.

  2. **Per-seed distribution** (``prior_mismatch_distribution.pdf/.png``)
     — violin + strip plot revealing Tabula Rasa's bimodal per-seed
     regret vs the tight, unimodal warmup distributions.

Usage:
    python experiments/appendix/prior_mismatch/generate_figure.py
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

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

RESULTS_DIR = Path(__file__).parent / "results"

COLORS = {
    "Well-calibrated": "#0072B2",
    "Random-1680": "#56B4E9",
    "MMLU-only": "#009E73",
    "GSM8K-only": "#E69F00",
    "Inverted": "#D55E00",
    "Tabula Rasa": "#999999",
    "Tabula Rasa (γ-matched)": "#555555",
}
QUALITY_ORDER = [
    "Well-calibrated", "Random-1680", "MMLU-only", "GSM8K-only", "Inverted",
]
N_EFF_VALUES = [10, 100, 1000]


# ======================================================================
# Figure 1: Summary Heatmap (prior quality × n_eff)
# ======================================================================


def generate_heatmap_figure(data: Dict[str, Any]) -> None:
    """Heatmap of total regret: prior quality (rows) × n_eff (columns).

    Annotates each cell with mean ± SE.  Cells that exceed the
    tabula-rasa baseline are bolded.
    """
    conditions = data["conditions"]
    early_step = data.get("early_step", 200)

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

    gm_key = "Tabula Rasa (γ-matched)"
    gm_regret = conditions.get(gm_key, {}).get(
        "total_regret", {},
    ).get("mean", np.nan)
    gm_early = conditions.get(gm_key, {}).get(
        f"regret_at_{early_step}", {},
    ).get("mean", np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

    for ax, matrix, se_mat, tr_val, gm_val, title in [
        (ax1, regret_matrix, se_matrix, tr_regret, gm_regret, "Total Regret"),
        (ax2, early_matrix, early_se_matrix, tr_early, gm_early,
         f"R@{early_step} (Early Regret)"),
    ]:
        ref_vals = [v for v in [tr_val, gm_val] if not np.isnan(v)]
        vmin = min(np.nanmin(matrix), *ref_vals) * 0.95
        vmax = max(np.nanmax(matrix), *ref_vals) * 1.05

        im = ax.imshow(
            matrix, aspect="auto", cmap="RdYlGn_r",
            vmin=vmin, vmax=vmax,
        )

        ax.set_xticks(range(n_neffs))
        ax.set_xticklabels([str(int(v)) for v in N_EFF_VALUES])
        ax.set_yticks(range(n_qualities))
        ax.set_yticklabels(QUALITY_ORDER)
        ax.set_xlabel("n_eff")
        ax.set_title(title, fontsize=17)

        for qi in range(n_qualities):
            for ni in range(n_neffs):
                val = matrix[qi, ni]
                se = se_mat[qi, ni]
                if np.isnan(val):
                    continue
                exceeds_gm = val > gm_val if not np.isnan(gm_val) else val > tr_val
                fontweight = "bold" if exceeds_gm else "normal"
                ax.text(
                    ni, qi, f"{val:.1f}\n±{se:.1f}",
                    ha="center", va="center", fontsize=14,
                    fontweight=fontweight, color="black",
                )

        ref_y = n_qualities - 0.5 + 0.55
        ax.text(
            n_neffs - 0.5, ref_y,
            f"TR (γ=0.995) = {tr_val:.1f}",
            ha="right", va="center", fontsize=12,
            fontstyle="italic", color=COLORS["Tabula Rasa"],
        )
        if not np.isnan(gm_val):
            ax.text(
                n_neffs - 0.5, ref_y + 0.35,
                f"TR (γ-matched) = {gm_val:.1f}",
                ha="right", va="center", fontsize=12,
                fontstyle="italic", color=COLORS[gm_key],
            )

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    fig.suptitle(
        "Prior Mismatch × n_eff: When Do Warmup Priors Hurt?\n"
        "(Unconstrained regime, K=3 stationary, 20 seeds)",
        fontsize=18, y=1.04,
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
# Figure 2: Per-seed regret distribution (violin + swarm)
# ======================================================================


def generate_distribution_figure(data: Dict[str, Any]) -> None:
    """Violin + strip plot of per-seed total regret for each condition.

    Reveals the bimodal / heavy-tailed nature of Tabula Rasa's per-seed
    regret vs the tight, unimodal distributions of warmup conditions.
    One panel per ``n_eff`` value plus the shared Tabula Rasa baseline.
    """
    conditions = data["conditions"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0), sharey=True)

    for ax, n_eff in zip(axes, N_EFF_VALUES):
        labels: List[str] = []
        all_vals: List[np.ndarray] = []
        colors_list: List[str] = []

        for bl_key, bl_short in [
            ("Tabula Rasa", "TR\n(γ=.995)"),
            ("Tabula Rasa (γ-matched)", "TR\n(γ-match)"),
        ]:
            if bl_key in conditions and "per_seed_regret" in conditions[bl_key]:
                vals = np.array(conditions[bl_key]["per_seed_regret"])
                labels.append(bl_short)
                all_vals.append(vals)
                colors_list.append(COLORS.get(bl_key, "#999999"))

        for quality in QUALITY_ORDER:
            key = f"{quality} (n_eff={n_eff})"
            if key not in conditions:
                continue
            cond = conditions[key]
            if "per_seed_regret" not in cond:
                continue
            vals = np.array(cond["per_seed_regret"])
            short = quality.replace("-", "-\n") if len(quality) > 10 else quality
            labels.append(short)
            all_vals.append(vals)
            colors_list.append(COLORS[quality])

        if not all_vals:
            continue

        positions = list(range(len(all_vals)))

        parts = ax.violinplot(
            all_vals, positions=positions, showmeans=False,
            showmedians=False, showextrema=False, widths=0.7,
        )
        for pc, color in zip(parts["bodies"], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.25)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.8)

        rng = np.random.default_rng(42)
        for pos, vals, color in zip(positions, all_vals, colors_list):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                color=color, s=18, alpha=0.7, edgecolors="white",
                linewidths=0.3, zorder=5,
            )
            ax.scatter(
                [pos], [np.mean(vals)],
                color=color, s=60, marker="D", edgecolors="black",
                linewidths=0.8, zorder=6,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=12, rotation=30, ha="right")
        ax.set_title(f"n_eff = {n_eff}", fontsize=17)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Total Regret (per seed)")

    fig.suptitle(
        "Per-Seed Regret Distributions: Warmup Stability vs Cold-Start Variance\n"
        "(Unconstrained regime, K=3 stationary, 20 seeds)",
        fontsize=18, y=1.04,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"prior_mismatch_distribution.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved prior_mismatch_distribution.pdf/.png to {RESULTS_DIR}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    results_path = RESULTS_DIR / "prior_mismatch_results.json"
    with open(results_path) as f:
        data = json.load(f)

    generate_heatmap_figure(data)
    generate_distribution_figure(data)


if __name__ == "__main__":
    main()
