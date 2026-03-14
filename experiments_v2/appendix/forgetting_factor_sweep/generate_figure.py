#!/usr/bin/env python3
"""Generate figures for Appendix: Forgetting Factor Sweep.

Produces a single publication-ready figure showing cumulative regret as a
function of the forgetting factor γ, with:

- Total regret on the primary y-axis (line + error bars)
- Phase 1 and Phase 2 regret as stacked bars
- The adaptive-γ result shown as a horizontal reference band
- Effective half-life annotated on the secondary x-axis

Usage::

    python experiments_v2/appendix/forgetting_factor_sweep/generate_figure.py
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
    with open(RESULTS_DIR / "forgetting_factor_sweep_results.json") as f:
        return json.load(f)


def plot_gamma_sweep(data: Dict[str, Any]) -> plt.Figure:
    """Regret vs. forgetting factor with phase decomposition.

    Parameters
    ----------
    data : dict
        Full results dict from the sweep experiment.

    Returns
    -------
    plt.Figure
    """
    fixed_results = [r for r in data["results"] if not r["adaptive_gamma"]]
    adaptive_result = next(
        (r for r in data["results"] if r["adaptive_gamma"]),
        None,
    )

    fixed_results.sort(key=lambda r: r["forgetting_factor"])

    gammas = [r["forgetting_factor"] for r in fixed_results]
    labels = [r["label"] for r in fixed_results]
    totals = [r["mean_regret"] for r in fixed_results]
    total_ses = [r["se_regret"] for r in fixed_results]
    p1s = [r["mean_phase1_regret"] for r in fixed_results]
    p1_ses = [r["se_phase1_regret"] for r in fixed_results]
    p2s = [r["mean_phase2_regret"] for r in fixed_results]
    p2_ses = [r["se_phase2_regret"] for r in fixed_results]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(gammas))
    width = 0.35

    bars_p1 = ax.bar(
        x, p1s, width, yerr=p1_ses, capsize=4,
        label="Phase 1 (pre-shift)",
        color="#56B4E9", edgecolor="black", linewidth=0.5, zorder=3,
    )
    bars_p2 = ax.bar(
        x, p2s, width, bottom=p1s, yerr=p2_ses, capsize=4,
        label="Phase 2 (post-shift)",
        color="#D55E00", edgecolor="black", linewidth=0.5, zorder=3,
    )

    ax.plot(
        x, totals, "ko-", linewidth=2.2, markersize=7,
        label="Total regret", zorder=5,
    )

    for i, (tot, se) in enumerate(zip(totals, total_ses)):
        ax.fill_between(
            [i - 0.15, i + 0.15],
            [tot - se, tot - se],
            [tot + se, tot + se],
            alpha=0.15, color="black", zorder=4,
        )

    if adaptive_result is not None:
        adaptive_y = adaptive_result["mean_regret"]
        adaptive_se = adaptive_result["se_regret"]
        ax.axhline(
            adaptive_y, color="#CC79A7", linestyle="--",
            linewidth=2.2, label=f"Adaptive γ ({adaptive_y:.1f}±{adaptive_se:.1f})",
            zorder=4,
        )
        ax.axhspan(
            adaptive_y - adaptive_se, adaptive_y + adaptive_se,
            alpha=0.12, color="#CC79A7", zorder=2,
        )

    for i, tot in enumerate(totals):
        ax.annotate(
            f"{tot:.1f}",
            xy=(i, tot),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Forgetting Factor (γ)", fontsize=13)
    ax.set_ylabel("Cumulative Regret", fontsize=13)
    ax.set_title(
        "Forgetting Factor Sweep — Reward Shift (K=3)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    half_lives = []
    for r in fixed_results:
        hl = r["effective_half_life"]
        if np.isfinite(hl):
            half_lives.append(f"~{hl:.0f}")
        else:
            half_lives.append("∞")
    ax2.set_xticklabels(half_lives, fontsize=9, color="#555555")
    ax2.set_xlabel("Effective Half-life (steps)", fontsize=10, color="#555555")

    fig.tight_layout()
    return fig


def main() -> None:
    data = _load_results()

    fig = plot_gamma_sweep(data)
    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"forgetting_factor_sweep.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved forgetting_factor_sweep.{pdf,png}")

    print(f"\nAll figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
