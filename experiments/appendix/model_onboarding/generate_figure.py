#!/usr/bin/env python3
"""Generate the model onboarding appendix figure (K=3 → K=4).

Reads ``results/model_onboarding_results.json`` and produces a
publication-ready 1×2 panel figure:

  **(a)** Full arm composition for the moderate-budget condition.
  **(b)** Cost compliance: ParetoBandit stays within budget despite onboarding.

Usage:
    python experiments/appendix/model_onboarding/generate_figure.py
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
from matplotlib.transforms import blended_transform_factory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "model_onboarding_results.json"

# ======================================================================
# Visual constants
# ======================================================================

FLASH_ID = "google/gemini-2.5-flash"

ARM_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-flash": "Flash",
    "google/gemini-2.5-pro": "Gemini-Pro",
}

ARM_COLORS: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "#56B4E9",
    "mistralai/mistral-large-2512": "#D55E00",
    "google/gemini-2.5-flash": "#E69F00",
    "google/gemini-2.5-pro": "#009E73",
}

BUDGET_COLORS: Dict[str, str] = {
    "tight": "#D55E00",
    "moderate": "#0072B2",
    "loose": "#CC79A7",
}

BUDGET_NICE: Dict[str, str] = {
    "tight": r"Tight ($B{=}\$3.0{\times}10^{-4}$)",
    "moderate": r"Moderate ($B{=}\$6.6{\times}10^{-4}$)",
    "loose": r"Loose ($B{=}\$1.9{\times}10^{-3}$)",
}

K4_ARMS: List[str] = list(ARM_SHORT.keys())

_MILLI: float = 1_000.0


def _load() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def _add_phase_shading(
    ax: plt.Axes,
    boundary: int,
    *,
    label_left: str = "Phase 1\n(K=3)",
    label_right: str = "Phase 2\n(K=4, +Flash)",
) -> None:
    """Shade the Phase 1 region gray and label the phase boundary.

    Uses a blended transform (data-x, axes-y) so labels sit at a
    consistent vertical position regardless of y-axis scale or limits.
    """
    ax.axvspan(0, boundary, alpha=0.06, color="#000000", zorder=0)
    ax.axvline(
        boundary,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        zorder=1,
    )
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        boundary - 30,
        0.96,
        label_left,
        transform=trans,
        ha="right",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
    )
    ax.text(
        boundary + 30,
        0.96,
        label_right,
        transform=trans,
        ha="left",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
    )


# ======================================================================
# Panel (a): Full arm composition (moderate budget)
# ======================================================================


def _panel_arm_composition(
    ax: plt.Axes,
    data: Dict[str, Any],
    phase_boundary: int,
    budget_label: str = "moderate",
) -> None:
    """All four arms' windowed mix for one budget tier, Flash emphasized."""
    key = f"paretobandit_transfer_{budget_label}"
    trace = data["checkpoint_traces"][key]
    steps = [c["step"] for c in trace]
    has_per_seed = "per_seed_windowed_mix" in trace[0]

    for arm_id in K4_ARMS:
        means = [c["windowed_mix_mean"].get(arm_id, 0.0) for c in trace]
        color = ARM_COLORS[arm_id]
        is_flash = arm_id == FLASH_ID
        lw = 2.8 if is_flash else 1.5
        alpha_ci = 0.14 if is_flash else 0.06

        if has_per_seed:
            matrix = np.array(
                [
                    c["per_seed_windowed_mix"].get(arm_id, [0.0])
                    for c in trace
                ]
            )
            ci_lo, ci_hi = bootstrap_ci_series(matrix)
        else:
            ses = [c["windowed_mix_se"].get(arm_id, 0.0) for c in trace]
            ci_lo = [m - s for m, s in zip(means, ses)]
            ci_hi = [m + s for m, s in zip(means, ses)]

        ax.plot(
            steps,
            means,
            color=color,
            linewidth=lw,
            label=ARM_SHORT[arm_id],
            zorder=5 if is_flash else 3,
        )
        ax.fill_between(
            steps, ci_lo, ci_hi, alpha=alpha_ci, color=color, zorder=2
        )

    ax.set_ylim(-0.02, 1.02)
    _add_phase_shading(ax, phase_boundary)

    ax.set_title(
        f"(a) Arm Composition — {budget_label.title()} Budget",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Prompts Routed", fontsize=12)
    ax.set_ylabel("Windowed Fraction", fontsize=12)
    ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10)


# ======================================================================
# Panel (b): Cost compliance (milli-dollar linear scale)
# ======================================================================


def _panel_cost_compliance(
    ax: plt.Axes,
    data: Dict[str, Any],
    phase_boundary: int,
) -> None:
    """Running average cost on a linear milli-dollar scale with annotated
    budget targets, showing onboarding does not disrupt compliance."""
    traces = data["checkpoint_traces"]
    budget_targets = data["budget_targets"]

    for blabel in ["tight", "moderate", "loose"]:
        key = f"paretobandit_transfer_{blabel}"
        if key not in traces:
            continue
        trace = traces[key]
        steps = [c["step"] for c in trace]
        costs_m = [c["cumulative_cost"] * _MILLI for c in trace]
        color = BUDGET_COLORS[blabel]

        ax.plot(
            steps,
            costs_m,
            color=color,
            linewidth=2.2,
            label=BUDGET_NICE[blabel],
            zorder=4,
        )

        target_m = budget_targets[blabel] * _MILLI
        ax.axhline(
            target_m,
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.7,
            zorder=1,
        )

    _add_phase_shading(ax, phase_boundary)

    ax.set_title(
        "(b) Cost Compliance", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xlabel("Prompts Routed", fontsize=12)
    ax.set_ylabel(
        r"Avg Cost / Request ($\times 10^{-3}$ USD)", fontsize=12
    )
    ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=10)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    phase_boundary = data["phase1_n"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

    _panel_arm_composition(axes[0], data, phase_boundary)
    _panel_cost_compliance(axes[1], data, phase_boundary)

    fig.suptitle(
        r"Model Onboarding: K=3 $\to$ K=4 (Gemini Flash)"
        r" — 20 seeds, 95% bootstrap CI",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"model_onboarding.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model_onboarding.{{pdf,png}} to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
