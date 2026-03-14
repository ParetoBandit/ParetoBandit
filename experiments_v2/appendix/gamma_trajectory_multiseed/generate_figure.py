#!/usr/bin/env python3
"""Generate figures for Appendix: Multi-Seed Gamma Trajectories.

Produces a 1x2 panel figure (reward shift | cost shift) showing:

- Median gamma trajectory (solid line)
- 25th--75th percentile band (dark shading)
- 10th--90th percentile band (light shading)
- 5 representative individual seed traces (thin lines)
- Phase boundary and reference lines for γ=1.0 and γ=0.999

This complements the main text's mean ± std figure by showing the
full distributional shape across seeds.

Usage::

    python experiments_v2/appendix/gamma_trajectory_multiseed/generate_figure.py
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

ADAPTIVE_COLOR = "#CC79A7"
TRACE_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#D55E00", "#0072B2"]


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_DIR / "gamma_trajectory_multiseed_results.json") as f:
        return json.load(f)


def plot_multiseed_gamma(data: Dict[str, Any]) -> plt.Figure:
    """Side-by-side gamma trajectories with quartile bands.

    Parameters
    ----------
    data : dict
        Full results dict from the trajectory recording experiment.

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    panels = [
        ("reward_shift", axes[0], "a) Reward Shift",
         "Phase 1\n(normal)", "Phase 2\n(Llama↔Mistral swap)"),
        ("cost_shift", axes[1], "b) Cost Shift",
         "Phase 1\n(normal pricing)", "Phase 2\n(Gemini price drop)"),
    ]

    n_traces = 5

    for exp_key, ax, panel_title, p1_label, p2_label in panels:
        exp = data[exp_key]
        checkpoints = exp["checkpoints"]
        per_seed = np.array(exp["per_seed_gamma"])
        phase_boundary = exp["phase_boundary"]

        steps = np.array(checkpoints)
        n_seeds = per_seed.shape[0]

        median = np.median(per_seed, axis=0)
        q25 = np.percentile(per_seed, 25, axis=0)
        q75 = np.percentile(per_seed, 75, axis=0)
        p10 = np.percentile(per_seed, 10, axis=0)
        p90 = np.percentile(per_seed, 90, axis=0)

        ax.fill_between(
            steps, p10, p90,
            alpha=0.10, color=ADAPTIVE_COLOR, zorder=2,
            label="10th–90th percentile",
        )
        ax.fill_between(
            steps, q25, q75,
            alpha=0.25, color=ADAPTIVE_COLOR, zorder=3,
            label="25th–75th percentile",
        )
        ax.plot(
            steps, median,
            color=ADAPTIVE_COLOR, linewidth=2.5,
            label="Median", zorder=5,
        )

        rng_trace = np.random.default_rng(42)
        trace_indices = rng_trace.choice(n_seeds, size=n_traces, replace=False)
        for idx, color in zip(trace_indices, TRACE_COLORS):
            ax.plot(
                steps, per_seed[idx],
                color=color, linewidth=0.4, alpha=0.3, zorder=4,
            )
        ax.plot([], [], color="#999999", linewidth=0.6, alpha=0.4,
                label=f"Individual seeds (n={n_traces})")

        ax.axhline(
            1.0, color="#D55E00", linestyle="--",
            linewidth=1.5, alpha=0.6, zorder=1,
            label="γ = 1.0",
        )
        ax.axhline(
            0.999, color="#0072B2", linestyle=":",
            linewidth=1.5, alpha=0.6, zorder=1,
            label="γ = 0.999",
        )

        ax.axvline(
            phase_boundary, color="black", linestyle="--",
            linewidth=1.2, alpha=0.5, zorder=1,
        )
        y_lo, y_hi = 0.9985, 1.0005
        y_text = y_lo + 0.93 * (y_hi - y_lo)
        ax.text(
            phase_boundary - 15, y_text,
            p1_label.replace("\n", " "),
            ha="right", va="top", fontsize=8,
            fontstyle="italic", color="#555555",
        )
        ax.text(
            phase_boundary + 15, y_text,
            p2_label.replace("\n", " "),
            ha="left", va="top", fontsize=8,
            fontstyle="italic", color="#555555",
        )

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_title(panel_title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("Forgetting Factor (γ)", fontsize=12)
    axes[0].set_ylim(0.9985, 1.0005)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=4, fontsize=9.5, framealpha=0.9,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main() -> None:
    data = _load_results()

    fig = plot_multiseed_gamma(data)
    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"gamma_trajectory_multiseed.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved gamma_trajectory_multiseed.{pdf,png}")

    print(f"\nAll figures written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
