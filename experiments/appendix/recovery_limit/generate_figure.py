#!/usr/bin/env python3
"""Generate figures for Appendix: Recovery Limit under Quality Degradation.

Reads ``results/recovery_limit_results.json`` and produces a two-panel
figure:

  (a) Recovery Envelope — P3/P1 reward ratio vs degradation severity for
      standard (608-step) and extended (1800-step) Phase 3 horizons.
  (b) Extended Recovery Dynamics — windowed mean reward time-series for
      selected degradation levels at the extended horizon.

Usage::

    python experiments/appendix/recovery_limit/generate_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "recovery_limit_results.json"

FULL_RECOVERY_THRESHOLD = 0.97

STANDARD_COLOR = "#0072B2"
EXTENDED_COLOR = "#D55E00"
THRESHOLD_COLOR = "#999999"

DEGRADATION_COLORS: Dict[float, str] = {
    0.30: "#D55E00",
    0.50: "#0072B2",
    0.60: "#CC79A7",
    0.70: "#009E73",
}


def _load_results() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def plot_recovery_limit(
    data: Dict[str, Any],
    figsize: Tuple[float, float] = (12, 5),
    font_scale: float = 1.0,
) -> plt.Figure:
    """Two-panel figure: recovery envelope + extended dynamics.

    Parameters
    ----------
    data : dict
        Parsed results JSON.
    figsize : tuple
        Figure size in inches.
    font_scale : float
        Multiplicative factor for font sizes.

    Returns
    -------
    plt.Figure
    """
    fs = font_scale
    standard = data["standard_results"]
    extended = data["extended_results"]
    phase_n = data["phase_n"]
    ext_n = data["extended_phase3_n"]

    fig, (ax_env, ax_dyn) = plt.subplots(1, 2, figsize=figsize)

    # ------------------------------------------------------------------
    # (a) Recovery envelope
    # ------------------------------------------------------------------
    std_deg = [r["degradation_pct"] for r in standard]
    std_ratio = [r["p3_p1_ratio"] * 100 for r in standard]

    std_ratio_matrix = []
    for r in standard:
        p1_seeds = r["phases"]["phase1"]["per_seed_reward"]
        p3_seeds = r["phases"]["phase3"]["per_seed_reward"]
        ratios = [
            p3 / p1 * 100 if p1 > 0 else 0
            for p1, p3 in zip(p1_seeds, p3_seeds)
        ]
        std_ratio_matrix.append(ratios)
    std_ratio_matrix_np = np.array(std_ratio_matrix)
    std_ci_lo, std_ci_hi = bootstrap_ci_series(std_ratio_matrix_np)

    ax_env.plot(
        std_deg, std_ratio,
        color=STANDARD_COLOR, linewidth=2.2, marker="o", markersize=5,
        label=f"{phase_n} prompts", zorder=4,
    )
    ax_env.fill_between(
        std_deg, std_ci_lo, std_ci_hi,
        alpha=0.15, color=STANDARD_COLOR, zorder=2,
    )

    ext_deg = [r["degradation_pct"] for r in extended]
    ext_ratio = [r["p3_p1_ratio"] * 100 for r in extended]

    ext_ratio_matrix = []
    for r in extended:
        p1_seeds = r["phases"]["phase1"]["per_seed_reward"]
        p3_seeds = r["phases"]["phase3"]["per_seed_reward"]
        ratios = [
            p3 / p1 * 100 if p1 > 0 else 0
            for p1, p3 in zip(p1_seeds, p3_seeds)
        ]
        ext_ratio_matrix.append(ratios)
    ext_ratio_matrix_np = np.array(ext_ratio_matrix)
    ext_ci_lo, ext_ci_hi = bootstrap_ci_series(ext_ratio_matrix_np)

    ax_env.plot(
        ext_deg, ext_ratio,
        color=EXTENDED_COLOR, linewidth=2.2, marker="s", markersize=5,
        linestyle="--",
        label=f"{ext_n} prompts", zorder=4,
    )
    ax_env.fill_between(
        ext_deg, ext_ci_lo, ext_ci_hi,
        alpha=0.15, color=EXTENDED_COLOR, zorder=2,
    )

    ax_env.axhline(
        FULL_RECOVERY_THRESHOLD * 100,
        color=THRESHOLD_COLOR, linestyle=":", linewidth=1.5,
        label="97% recovery threshold", zorder=1,
    )

    ax_env.set_xlabel("Degradation Severity (%)", fontsize=11 * fs)
    ax_env.set_ylabel("P3 / P1 Reward Ratio (%)", fontsize=11 * fs)
    ax_env.set_title(
        "(a) Recovery Envelope",
        fontsize=12 * fs, fontweight="bold", pad=10,
    )
    ax_env.set_xlim(0, 100)
    ax_env.set_ylim(85, 105)
    ax_env.invert_xaxis()
    ax_env.legend(fontsize=9 * fs, loc="lower left")
    ax_env.grid(True, alpha=0.2, linewidth=0.5)
    ax_env.tick_params(labelsize=10 * fs)

    # ------------------------------------------------------------------
    # (b) Extended recovery dynamics
    # ------------------------------------------------------------------
    phase2_start = phase_n
    phase3_start = 2 * phase_n

    for r in extended:
        fr = r["failure_reward"]
        if fr not in DEGRADATION_COLORS:
            continue
        deg = r["degradation_pct"]
        color = DEGRADATION_COLORS[fr]
        curves = r["curves"]

        p3_curves = [c for c in curves if c["step"] > phase2_start + phase_n]
        if not p3_curves:
            continue

        steps = [c["step"] - phase3_start for c in p3_curves]
        rewards = [c["mean_window_reward"] for c in p3_curves]

        has_per_seed = "per_seed_window_reward" in p3_curves[0]
        if has_per_seed:
            matrix = np.array(
                [c["per_seed_window_reward"] for c in p3_curves],
            )
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            stds = [c.get("std_window_reward", 0) for c in p3_curves]
            sqrt_n = np.sqrt(data["n_seeds"])
            ci_lo = np.array([m - s / sqrt_n for m, s in zip(rewards, stds)])
            ci_hi = np.array([m + s / sqrt_n for m, s in zip(rewards, stds)])

        ax_dyn.plot(
            steps, rewards,
            color=color, linewidth=2.0,
            label=f"{deg:.0f}% degradation (r={fr})",
            zorder=4,
        )
        ax_dyn.fill_between(
            steps, ci_lo, ci_hi,
            alpha=0.12, color=color, zorder=2,
        )

    # Phase 1 reference line (from the first extended result)
    if extended:
        p1_reward = extended[0]["phases"]["phase1"]["mean_reward"]
        ax_dyn.axhline(
            p1_reward, color=THRESHOLD_COLOR, linestyle=":",
            linewidth=1.5, zorder=1,
        )
        ax_dyn.text(
            0.98, p1_reward, "Phase 1 baseline",
            transform=blended_transform_factory(
                ax_dyn.transAxes, ax_dyn.transData,
            ),
            fontsize=8 * fs, color=THRESHOLD_COLOR,
            va="bottom", ha="right",
        )

    ax_dyn.axvline(
        phase_n, color="black", linestyle="--",
        linewidth=1.0, alpha=0.5, zorder=1,
    )
    ax_dyn.text(
        phase_n + 10, 0.97, f"{phase_n}\nprompts",
        transform=blended_transform_factory(
            ax_dyn.transData, ax_dyn.transAxes,
        ),
        fontsize=8 * fs, color="#333333", va="top",
    )

    ax_dyn.set_xlabel("Prompts Routed (Phase 3)", fontsize=11 * fs)
    ax_dyn.set_ylabel("Windowed Mean Reward", fontsize=11 * fs)
    ax_dyn.set_title(
        f"(b) Extended Recovery Dynamics ({ext_n} prompts)",
        fontsize=12 * fs, fontweight="bold", pad=10,
    )
    ax_dyn.legend(fontsize=8.5 * fs, loc="lower right")
    ax_dyn.grid(True, alpha=0.2, linewidth=0.5)
    ax_dyn.tick_params(labelsize=10 * fs)

    budget_label = data.get("budget_label", "moderate")
    budget_target = data.get("budget_target", 6.62e-4)
    fig.suptitle(
        f"Recovery Limit Study — {budget_label.capitalize()} Budget"
        f" (${budget_target:.2e}/prompt)",
        fontsize=13 * fs, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()

    fig = plot_recovery_limit(data)
    for fmt in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"recovery_limit.{fmt}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)
    print(f"Saved recovery_limit.{{pdf,png}} to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
