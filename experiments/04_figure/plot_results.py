#!/usr/bin/env python3
"""Generate Figure 4: Adaptive Drift Detection Under Distribution Shift.

Three-panel figure showing how the router detects prior miscalibration
and self-adapts (K=2, Llama-3.1-8B vs Gemini-2.5-Pro) under a controlled
synthetic shift (2.0σ embedding perturbation + reward boost):

  (a) Cumulative cost-adjusted regret for all 4 conditions.
  (b) Drift detection signal — EMA chi-squared score, baseline, and
      trigger threshold over training steps (adaptive condition only).
  (c) Llama routing fraction over time, showing when each condition
      discovers the newly-competitive cheap arm.

Usage::

    python experiments/04_figure/plot_results.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent / "results"

# ============================================================================
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ============================================================================
ORANGE = "#E69F00"
BLUE = "#0072B2"
GREEN = "#009E73"
GRAY = "#999999"
RED = "#D55E00"

CONDITION_STYLE: Dict[str, Dict[str, Any]] = {
    "Warmup-only": {
        "color": ORANGE, "linestyle": "--", "linewidth": 2.0, "zorder": 2,
    },
    "Oracle ff=0.999": {
        "color": BLUE, "linestyle": "-", "linewidth": 2.0, "zorder": 3,
    },
    "Adaptive (Reset)": {
        "color": GREEN, "linestyle": "-", "linewidth": 2.5, "zorder": 4,
    },
    "Tabula Rasa": {
        "color": GRAY, "linestyle": ":", "linewidth": 1.5, "zorder": 1,
    },
}


def _ci95(std: float, n: int) -> float:
    """95% t-CI half-width."""
    if n < 2 or std == 0:
        return 0.0
    return sp_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


def _get_style(label: str) -> Dict[str, Any]:
    """Look up condition style, fallback to gray."""
    return CONDITION_STYLE.get(
        label, {"color": GRAY, "linestyle": "-", "linewidth": 1.5, "zorder": 1},
    )


def plot_figure4(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate three-panel Figure 4."""
    meta = results["metadata"]
    cross = results["cross_dist"]
    conditions = cross["conditions"]
    headline_lam = meta["headline_lambda"]
    oracle_llama_frac = meta.get("oracle_llama_frac", 0)
    phase1_n = cross.get("phase1_n", meta.get("phase1_n_pareto", 0))
    shift_mag = meta.get("shift_magnitude", 0)
    drift_threshold_sigma = meta.get("drift_threshold", 2.0)

    fig, (ax_regret, ax_drift, ax_arms) = plt.subplots(
        1, 3, figsize=(17, 5), gridspec_kw={"wspace": 0.32},
    )

    # ================================================================
    # Panel (a): Cumulative regret curves
    # ================================================================
    for label, lc in conditions.items():
        if not lc:
            continue
        style = _get_style(label)
        steps = [d["step"] for d in lc]
        regrets = [d["mean_cumulative_regret"] for d in lc]
        cis = [_ci95(d["std_cumulative_regret"], d["n_seeds"]) for d in lc]

        ax_regret.plot(
            steps, regrets, style["linestyle"],
            color=style["color"], linewidth=style["linewidth"],
            label=label, zorder=style["zorder"],
        )
        ax_regret.fill_between(
            steps,
            [r - c for r, c in zip(regrets, cis)],
            [r + c for r, c in zip(regrets, cis)],
            color=style["color"], alpha=0.10, zorder=style["zorder"] - 0.5,
        )

    # Regret savings annotation
    wo_lc = conditions.get("Warmup-only", [])
    adaptive_lc = conditions.get("Adaptive (Reset)", [])
    if wo_lc and adaptive_lc:
        wo_final = wo_lc[-1]["mean_cumulative_regret"]
        ad_final = adaptive_lc[-1]["mean_cumulative_regret"]
        if wo_final > 0 and ad_final < wo_final:
            delta = wo_final - ad_final
            pct = delta / wo_final * 100
            final_step = wo_lc[-1]["step"]
            mid_y = (wo_final + ad_final) / 2
            arrow_x = final_step * 0.97
            label_x = final_step * 0.75
            ax_regret.annotate(
                "",
                xy=(arrow_x, ad_final),
                xytext=(arrow_x, wo_final),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
            )
            label_bbox = dict(
                boxstyle="round,pad=0.3", facecolor="lightyellow",
                alpha=0.9, edgecolor="gray",
            )
            ax_regret.annotate(
                f"−{delta:.0f}\n({pct:.0f}%)",
                xy=(arrow_x, mid_y),
                xytext=(label_x, mid_y),
                fontsize=8, fontweight="bold", ha="center", va="center",
                bbox=label_bbox,
                arrowprops=dict(
                    arrowstyle="-", color="gray", lw=1.0,
                    connectionstyle="arc3,rad=0",
                ),
            )

    if phase1_n > 0:
        ax_regret.axvline(
            phase1_n, color="black", linestyle=":", alpha=0.4, linewidth=1.0,
        )
        ylo, yhi = ax_regret.get_ylim()
        ax_regret.text(
            phase1_n + 20, ylo + (yhi - ylo) * 0.02,
            "← in-dist | shifted →",
            fontsize=7, alpha=0.5, va="bottom",
        )

    ax_regret.set_xlabel("Online step", fontsize=11)
    ax_regret.set_ylabel("Cumulative cost-adjusted regret", fontsize=11)
    ax_regret.set_title(
        f"(a) Regret (λ={headline_lam}, {shift_mag:.0f}σ shift)",
        fontsize=12, fontweight="bold",
    )
    ax_regret.legend(fontsize=8, loc="upper left")
    ax_regret.grid(True, alpha=0.3)

    # ================================================================
    # Panel (b): Drift detection signal (chi-squared)
    # ================================================================
    adaptive_key = "Adaptive (Reset)"
    adaptive_lc = conditions.get(adaptive_key, [])

    drift_steps: List[int] = []
    ema_chi2_vals: List[float] = []
    baseline_vals: List[float] = []
    baseline_std_vals: List[float] = []
    n_resets_vals: List[float] = []

    for d in adaptive_lc:
        ds = d.get("drift_state")
        if ds is None:
            continue
        drift_steps.append(d["step"])
        ema_chi2_vals.append(ds.get("mean_ema_chi2", 0))
        baseline_vals.append(ds.get("mean_baseline", 0))
        baseline_std_vals.append(ds.get("mean_baseline_std", 0))
        n_resets_vals.append(d.get("mean_n_resets", 0))

    if drift_steps:
        ax_drift.plot(
            drift_steps, ema_chi2_vals, "-", color=GREEN, linewidth=2.0,
            label="EMA χ²", zorder=3,
        )
        ax_drift.plot(
            drift_steps, baseline_vals, "--", color=GRAY, linewidth=1.5,
            label="Baseline", zorder=2,
        )

        # Threshold line = baseline + drift_threshold_sigma * baseline_std
        threshold_line = []
        for b, bs in zip(baseline_vals, baseline_std_vals):
            if b > 0 and bs > 0:
                threshold_line.append(b + drift_threshold_sigma * bs)
            else:
                threshold_line.append(np.nan)
        ax_drift.plot(
            drift_steps, threshold_line, ":", color=RED,
            linewidth=1.5, label=f"Threshold ({drift_threshold_sigma:.0f}σ)",
            zorder=2,
        )

        # Mark reset point(s)
        reset_step = None
        for i in range(1, len(n_resets_vals)):
            if n_resets_vals[i] > n_resets_vals[i - 1]:
                reset_step = drift_steps[i]
                break
        if reset_step is not None:
            ax_drift.axvline(
                reset_step, color=GREEN, linestyle="-.", alpha=0.6,
                linewidth=1.5,
            )
            yhi = ax_drift.get_ylim()[1]
            ax_drift.text(
                reset_step + 80, yhi * 0.98,
                f"Reset\n(step {reset_step})",
                fontsize=7, color=GREEN, fontweight="bold", va="top",
            )

    if phase1_n > 0:
        ax_drift.axvline(
            phase1_n, color="black", linestyle=":", alpha=0.4, linewidth=1.0,
        )

    ax_drift.set_xlabel("Online step", fontsize=11)
    ax_drift.set_ylabel("Chi-squared score", fontsize=11)
    ax_drift.set_title(
        "(b) Covariate Shift Detector",
        fontsize=12, fontweight="bold",
    )
    ax_drift.legend(fontsize=8, loc="lower right")
    ax_drift.grid(True, alpha=0.3)

    # ================================================================
    # Panel (c): Llama routing fraction over time
    # ================================================================
    for label, lc in conditions.items():
        if not lc:
            continue
        style = _get_style(label)
        steps = [d["step"] for d in lc]
        llama_fracs = [d["arm_fractions"].get("Llama-8B", 0) for d in lc]
        llama_stds = [
            d.get("arm_fractions_std", {}).get("Llama-8B", 0) for d in lc
        ]
        cis = [_ci95(s, d["n_seeds"]) for s, d in zip(llama_stds, lc)]

        ax_arms.plot(
            steps, llama_fracs, style["linestyle"],
            color=style["color"], linewidth=style["linewidth"],
            label=label, zorder=style["zorder"],
        )
        if any(c > 0 for c in cis):
            ax_arms.fill_between(
                steps,
                [f - c for f, c in zip(llama_fracs, cis)],
                [f + c for f, c in zip(llama_fracs, cis)],
                color=style["color"], alpha=0.10,
                zorder=style["zorder"] - 0.5,
            )

    if oracle_llama_frac > 0:
        ax_arms.axhline(
            oracle_llama_frac, color=RED, linestyle="-.",
            linewidth=1.5, alpha=0.7,
            label=f"Cost-adj. optimum ({oracle_llama_frac:.0%})",
        )

    if phase1_n > 0:
        ax_arms.axvline(
            phase1_n, color="black", linestyle=":", alpha=0.4, linewidth=1.0,
        )

    ax_arms.set_xlabel("Online step", fontsize=11)
    ax_arms.set_ylabel("Llama-8B routing fraction", fontsize=11)
    ax_arms.set_title(
        "(c) Cheap-Arm Discovery",
        fontsize=12, fontweight="bold",
    )
    ax_arms.legend(fontsize=8, loc="lower right")
    ax_arms.grid(True, alpha=0.3)
    ax_arms.set_ylim(0, 1.0)

    # ================================================================
    # Save
    # ================================================================
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = output_dir / f"figure4_adaptive_drift.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    path = RESULTS_DIR / "distribution_shift_results.json"
    with open(path) as f:
        data = json.load(f)
    plot_figure4(data, RESULTS_DIR)
