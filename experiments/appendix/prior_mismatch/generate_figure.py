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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci_mean, bootstrap_ci_median

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


def _median_ci(per_seed: List[float]) -> tuple[float, float, float]:
    """Return (median, ci_lo, ci_hi) from per-seed data."""
    arr = np.asarray(per_seed, dtype=np.float64)
    med = float(np.median(arr))
    lo, hi = bootstrap_ci_median(arr)
    return med, lo, hi


def generate_heatmap_figure(data: Dict[str, Any]) -> None:
    """Heatmap of total regret: prior quality (rows) x n_eff (columns).

    Annotates each cell with median [CI_lo, CI_hi].  Cells that exceed
    the gamma-matched baseline median are bolded.
    """
    conditions = data["conditions"]
    early_step = data.get("early_step", 200)

    n_qualities = len(QUALITY_ORDER)
    n_neffs = len(N_EFF_VALUES)
    median_matrix = np.full((n_qualities, n_neffs), np.nan)
    ci_lo_matrix = np.full((n_qualities, n_neffs), np.nan)
    ci_hi_matrix = np.full((n_qualities, n_neffs), np.nan)
    early_median_matrix = np.full((n_qualities, n_neffs), np.nan)
    early_ci_lo_matrix = np.full((n_qualities, n_neffs), np.nan)
    early_ci_hi_matrix = np.full((n_qualities, n_neffs), np.nan)

    for qi, quality in enumerate(QUALITY_ORDER):
        for ni, n_eff in enumerate(N_EFF_VALUES):
            key = f"{quality} (n_eff={n_eff})"
            cond = conditions.get(key)
            if cond is None:
                continue
            ps = cond.get("per_seed_regret")
            if ps is not None:
                med, lo, hi = _median_ci(ps)
                median_matrix[qi, ni] = med
                ci_lo_matrix[qi, ni] = lo
                ci_hi_matrix[qi, ni] = hi

            rk = f"per_seed_regret_at_{early_step}"
            ps_early = cond.get(rk)
            if ps_early is not None:
                med, lo, hi = _median_ci(ps_early)
                early_median_matrix[qi, ni] = med
                early_ci_lo_matrix[qi, ni] = lo
                early_ci_hi_matrix[qi, ni] = hi

    def _bl_median_ci(bl_key: str, per_seed_key: str) -> tuple[float, float, float]:
        cond = conditions.get(bl_key, {})
        ps = cond.get(per_seed_key)
        if ps is not None:
            return _median_ci(ps)
        return np.nan, np.nan, np.nan

    tr_med, tr_lo, tr_hi = _bl_median_ci("Tabula Rasa", "per_seed_regret")
    tr_early_med, _, _ = _bl_median_ci(
        "Tabula Rasa", f"per_seed_regret_at_{early_step}",
    )

    gm_key = "Tabula Rasa (γ-matched)"
    gm_med, gm_lo, gm_hi = _bl_median_ci(gm_key, "per_seed_regret")
    gm_early_med, _, _ = _bl_median_ci(
        gm_key, f"per_seed_regret_at_{early_step}",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

    for ax, matrix, ci_lo, ci_hi, tr_val, gm_val, title in [
        (ax1, median_matrix, ci_lo_matrix, ci_hi_matrix,
         tr_med, gm_med, "Total Regret (median)"),
        (ax2, early_median_matrix, early_ci_lo_matrix, early_ci_hi_matrix,
         tr_early_med, gm_early_med, f"R@{early_step} (median)"),
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
                lo_v = ci_lo[qi, ni]
                hi_v = ci_hi[qi, ni]
                if np.isnan(val):
                    continue
                exceeds_gm = (
                    val > gm_val if not np.isnan(gm_val) else val > tr_val
                )
                fontweight = "bold" if exceeds_gm else "normal"
                ax.text(
                    ni, qi,
                    f"{val:.1f}\n[{lo_v:.1f}, {hi_v:.1f}]",
                    ha="center", va="center", fontsize=12,
                    fontweight=fontweight, color="black",
                )

        ref_y = n_qualities - 0.5 + 0.55
        tr_gamma = data["hparams"]["tabula_rasa"]["forgetting_factor"]
        ax.text(
            n_neffs - 0.5, ref_y,
            f"TR (γ={tr_gamma:.3f}) median = {tr_val:.1f}",
            ha="right", va="center", fontsize=12,
            fontstyle="italic", color=COLORS["Tabula Rasa"],
        )
        if not np.isnan(gm_val):
            ax.text(
                n_neffs - 0.5, ref_y + 0.35,
                f"TR (γ-matched) median = {gm_val:.1f} [{gm_lo:.1f}, {gm_hi:.1f}]",
                ha="right", va="center", fontsize=12,
                fontstyle="italic", color=COLORS[gm_key],
            )

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)

    n_seeds = data.get("n_seeds", 20)
    fig.suptitle(
        "Prior Mismatch × n_eff: When Do Warmup Priors Hurt?\n"
        f"(Unconstrained regime, K=3 stationary, {n_seeds} seeds,"
        " 95% bootstrap CI)",
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
    Median shown as diamond with bootstrap 95% CI whiskers.
    """
    conditions = data["conditions"]
    n_seeds = data.get("n_seeds", 20)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0), sharey=True)

    for ax, n_eff in zip(axes, N_EFF_VALUES):
        labels: List[str] = []
        all_vals: List[np.ndarray] = []
        colors_list: List[str] = []

        tr_gamma = data["hparams"]["tabula_rasa"]["forgetting_factor"]
        for bl_key, bl_short in [
            ("Tabula Rasa", f"TR\n(γ={tr_gamma:.3f})"),
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
            med = float(np.median(vals))
            ci_lo, ci_hi = bootstrap_ci_median(vals)
            ax.errorbar(
                pos, med,
                yerr=[[med - ci_lo], [ci_hi - med]],
                fmt="D", color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.8,
                ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2,
                zorder=6,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=14, rotation=30, ha="right")
        ax.set_title(f"n_eff = {n_eff}", fontsize=19)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Total Regret (per seed)", fontsize=17)

    fig.suptitle(
        "Per-Seed Regret Distributions: Warmup Stability vs Cold-Start Variance\n"
        f"(Unconstrained regime, K=3 stationary, {n_seeds} seeds,"
        " ◆ = median with 95% bootstrap CI)",
        fontsize=20, y=1.04,
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
