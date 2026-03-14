#!/usr/bin/env python3
"""Generate figures for Appendix: Adaptive Gamma Sensitivity.

Produces two publication-ready figures:

1. **EMA heatmap** — alpha_s x alpha_l grid showing total cumulative regret.
   Demonstrates that regret is robust across a wide range of EMA time
   constants, with the production default marked.

2. **One-at-a-time bar chart** — burn-in length and noise-margin multiplier
   sweeps shown as grouped bars with error bars, confirming insensitivity
   to these secondary parameters.

Usage::

    python experiments_v2/appendix/adaptive_gamma_sensitivity/generate_figure.py
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


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_DIR / "adaptive_gamma_sensitivity_results.json") as f:
        return json.load(f)


def plot_ema_heatmap(data: Dict[str, Any]) -> plt.Figure:
    """Heatmap of total regret over the alpha_s x alpha_l grid.

    Parameters
    ----------
    data : dict
        Full results dict from the sensitivity experiment.

    Returns
    -------
    plt.Figure
    """
    alpha_s_vals = sorted(data["sweep_grids"]["alpha_s_values"])
    alpha_l_vals = sorted(data["sweep_grids"]["alpha_l_values"])
    defaults = data["defaults"]

    ema_results = [r for r in data["results"] if r["sweep_group"] == "ema_grid"]

    regret_grid = np.full((len(alpha_l_vals), len(alpha_s_vals)), np.nan)
    for r in ema_results:
        i = alpha_l_vals.index(r["aw_alpha_long"])
        j = alpha_s_vals.index(r["aw_alpha_short"])
        regret_grid[i, j] = r["mean_regret"]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    vmin = np.nanmin(regret_grid)
    vmax = np.nanmax(regret_grid)
    margin = (vmax - vmin) * 0.05
    im = ax.imshow(
        regret_grid,
        cmap="RdYlGn_r",
        aspect="auto",
        vmin=vmin - margin,
        vmax=vmax + margin,
        origin="lower",
    )

    for i in range(len(alpha_l_vals)):
        for j in range(len(alpha_s_vals)):
            val = regret_grid[i, j]
            if not np.isnan(val):
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="white" if val > (vmin + vmax) / 2 else "black",
                )

    default_j = alpha_s_vals.index(defaults["aw_alpha_short"])
    default_i = alpha_l_vals.index(defaults["aw_alpha_long"])
    rect = plt.Rectangle(
        (default_j - 0.5, default_i - 0.5), 1, 1,
        linewidth=3, edgecolor="black", facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_xticks(range(len(alpha_s_vals)))
    ax.set_xticklabels([f"{v}" for v in alpha_s_vals], fontsize=11)
    ax.set_yticks(range(len(alpha_l_vals)))
    ax.set_yticklabels([f"{v}" for v in alpha_l_vals], fontsize=11)
    ax.set_xlabel(r"$\alpha_s$ (short-horizon EMA)", fontsize=13)
    ax.set_ylabel(r"$\alpha_l$ (long-horizon EMA)", fontsize=13)
    ax.set_title(
        "Cumulative Regret vs. Adaptive-γ EMA Parameters",
        fontsize=14, fontweight="bold", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Cumulative Regret", fontsize=11)

    fig.tight_layout()
    return fig


def plot_oat_bars(data: Dict[str, Any]) -> plt.Figure:
    """Bar chart for burn-in and noise-margin one-at-a-time sweeps.

    Parameters
    ----------
    data : dict
        Full results dict from the sensitivity experiment.

    Returns
    -------
    plt.Figure
    """
    defaults = data["defaults"]

    burn_in_results = sorted(
        [r for r in data["results"] if r["sweep_group"] == "burn_in"],
        key=lambda r: r["aw_burn_in_steps"],
    )
    noise_results = sorted(
        [r for r in data["results"] if r["sweep_group"] == "noise_margin"],
        key=lambda r: r["aw_noise_margin_k"],
    )

    # Retrieve the default config from the EMA grid (it appears there too)
    default_result = next(
        (r for r in data["results"]
         if r["aw_alpha_short"] == defaults["aw_alpha_short"]
         and r["aw_alpha_long"] == defaults["aw_alpha_long"]
         and r["aw_burn_in_steps"] == defaults["aw_burn_in_steps"]
         and r["aw_noise_margin_k"] == defaults["aw_noise_margin_k"]),
        None,
    )

    # Merge default into sweeps if missing
    for sweep_list, key_field, default_val in [
        (burn_in_results, "aw_burn_in_steps", defaults["aw_burn_in_steps"]),
        (noise_results, "aw_noise_margin_k", defaults["aw_noise_margin_k"]),
    ]:
        if not any(r[key_field] == default_val for r in sweep_list):
            if default_result is not None:
                sweep_list.append({**default_result, "sweep_group": "default"})
                sweep_list.sort(key=lambda r: r[key_field])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel a: Burn-in sweep
    ax = axes[0]
    labels = [str(r["aw_burn_in_steps"]) for r in burn_in_results]
    means = [r["mean_regret"] for r in burn_in_results]
    ses = [r["se_regret"] for r in burn_in_results]
    x = np.arange(len(labels))
    colors = [
        "#0072B2" if r["aw_burn_in_steps"] == defaults["aw_burn_in_steps"]
        else "#56B4E9"
        for r in burn_in_results
    ]
    ax.bar(x, means, yerr=ses, capsize=5, color=colors, edgecolor="black",
           linewidth=0.8, width=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Burn-in Steps", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("a) Burn-in Length", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    # Panel b: Noise-margin sweep
    ax = axes[1]
    labels = [f"{r['aw_noise_margin_k']:.0f}σ" for r in noise_results]
    means = [r["mean_regret"] for r in noise_results]
    ses = [r["se_regret"] for r in noise_results]
    x = np.arange(len(labels))
    colors = [
        "#0072B2" if r["aw_noise_margin_k"] == defaults["aw_noise_margin_k"]
        else "#56B4E9"
        for r in noise_results
    ]
    ax.bar(x, means, yerr=ses, capsize=5, color=colors, edgecolor="black",
           linewidth=0.8, width=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Noise-Margin Multiplier", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title("b) Noise Margin", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    fig.suptitle(
        "Adaptive-γ Sensitivity: One-at-a-Time Sweeps",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    data = _load_results()

    fig_heatmap = plot_ema_heatmap(data)
    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"ema_sensitivity_heatmap.{fmt}"
        fig_heatmap.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_heatmap)
    print("  Saved ema_sensitivity_heatmap.{pdf,png}")

    fig_bars = plot_oat_bars(data)
    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"oat_sensitivity_bars.{fmt}"
        fig_bars.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig_bars)
    print("  Saved oat_sensitivity_bars.{pdf,png}")

    print(f"\nAll figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
