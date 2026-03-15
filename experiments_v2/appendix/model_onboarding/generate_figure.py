#!/usr/bin/env python3
"""Generate figures for the model onboarding appendix (K=3 → K=4).

Reads ``results/model_onboarding_results.json`` and produces a
publication-ready 1×3 panel figure:

  **(a)** Flash adoption trajectory across budget tiers.
  **(b)** Full arm composition for the moderate-budget condition.
  **(c)** Running average cost: BanditGPT vs. Fixed Policy.

Usage:
    python experiments_v2/appendix/model_onboarding/generate_figure.py
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
    "unconstrained": "#009E73",
}

BUDGET_NICE: Dict[str, str] = {
    "tight": r"Tight ($B{=}\$2.3{\times}10^{-4}$)",
    "moderate": r"Moderate ($B{=}\$6.6{\times}10^{-4}$)",
    "loose": r"Loose ($B{=}\$1.9{\times}10^{-3}$)",
    "unconstrained": "Unconstrained",
}

BUDGET_ORDER: List[str] = ["tight", "moderate", "loose", "unconstrained"]


def _load() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def _onboarding_line(ax: plt.Axes, boundary: int) -> None:
    """Draw a vertical dashed line at the onboarding boundary."""
    ax.axvline(
        boundary, color="black", linestyle="--",
        linewidth=1.0, alpha=0.4, zorder=1,
    )


# ======================================================================
# Panel (a): Flash adoption trajectory
# ======================================================================


def _panel_flash_adoption(
    ax: plt.Axes,
    data: Dict[str, Any],
    phase_boundary: int,
) -> None:
    """Windowed Flash selection fraction across budget tiers."""
    traces = data["checkpoint_traces"]

    for blabel in BUDGET_ORDER:
        key = f"banditgpt_transfer_{blabel}"
        if key not in traces:
            continue
        trace = traces[key]
        steps = [c["step"] for c in trace]
        flash_mean = [
            c["windowed_mix_mean"].get(FLASH_ID, 0.0) for c in trace
        ]
        flash_se = [
            c["windowed_mix_se"].get(FLASH_ID, 0.0) for c in trace
        ]
        color = BUDGET_COLORS[blabel]
        ls = "-." if blabel == "unconstrained" else "-"
        lw = 1.8 if blabel == "unconstrained" else 2.2

        ax.plot(
            steps, flash_mean,
            color=color, linestyle=ls, linewidth=lw,
            label=BUDGET_NICE[blabel], zorder=4,
        )
        ax.fill_between(
            steps,
            [m - s for m, s in zip(flash_mean, flash_se)],
            [m + s for m, s in zip(flash_mean, flash_se)],
            alpha=0.10, color=color, zorder=2,
        )

    ax.axhline(0.25, color="#888888", linestyle=":", linewidth=1.0,
               alpha=0.5, zorder=1, label="Uniform 1/4")

    _onboarding_line(ax, phase_boundary)
    y_lo, y_hi = ax.get_ylim()
    ax.text(
        phase_boundary + 20, 0.02,
        "onboard\nFlash", ha="left", va="bottom",
        fontsize=7, fontstyle="italic", color="#555555",
    )

    ax.set_title("(a) Flash Adoption", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Flash Windowed Fraction", fontsize=10)
    ax.set_ylim(-0.02, 0.52)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)


# ======================================================================
# Panel (b): Full arm composition (moderate budget)
# ======================================================================


def _panel_arm_composition(
    ax: plt.Axes,
    data: Dict[str, Any],
    phase_boundary: int,
    budget_label: str = "moderate",
) -> None:
    """All four arms' windowed mix for one budget tier."""
    key = f"banditgpt_transfer_{budget_label}"
    trace = data["checkpoint_traces"][key]
    arms = list(ARM_SHORT.keys())

    steps = [c["step"] for c in trace]

    for arm_id in arms:
        means = [
            c["windowed_mix_mean"].get(arm_id, 0.0) for c in trace
        ]
        ses = [c["windowed_mix_se"].get(arm_id, 0.0) for c in trace]
        color = ARM_COLORS[arm_id]

        ax.plot(
            steps, means,
            color=color, linewidth=2.2, label=ARM_SHORT[arm_id], zorder=4,
        )
        ax.fill_between(
            steps,
            [m - s for m, s in zip(means, ses)],
            [m + s for m, s in zip(means, ses)],
            alpha=0.12, color=color, zorder=2,
        )

    _onboarding_line(ax, phase_boundary)

    ax.set_title(
        f"(b) Arm Mix — {BUDGET_NICE[budget_label]}",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Windowed Fraction", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)


# ======================================================================
# Panel (c): Cost compliance
# ======================================================================


def _panel_cost_compliance(
    ax: plt.Axes,
    data: Dict[str, Any],
    phase_boundary: int,
) -> None:
    """Running average cost for BanditGPT (all budgets) vs Fixed Policy."""
    traces = data["checkpoint_traces"]
    budget_targets = data["budget_targets"]

    # BanditGPT traces
    for blabel in ["tight", "moderate", "loose"]:
        key = f"banditgpt_transfer_{blabel}"
        if key not in traces:
            continue
        trace = traces[key]
        steps = [c["step"] for c in trace]
        costs = [c["cumulative_cost"] for c in trace]
        color = BUDGET_COLORS[blabel]

        ax.plot(
            steps, costs,
            color=color, linewidth=2.2, label=f"BanditGPT ({blabel})",
            zorder=4,
        )

        target = budget_targets[blabel]
        ax.axhline(
            target, color=color, linestyle=":", linewidth=1.0,
            alpha=0.5, zorder=1,
        )

    # Fixed Policy trace (same for all budgets — pick one)
    fp_key = "fixed_uniform_moderate"
    if fp_key in traces:
        fp_trace = traces[fp_key]
        fp_steps = [c["step"] for c in fp_trace]
        fp_costs = [c["cumulative_cost"] for c in fp_trace]
        ax.plot(
            fp_steps, fp_costs,
            color="#888888", linestyle="--", linewidth=2.0,
            label="Fixed Policy (1/4)", zorder=3,
        )

    _onboarding_line(ax, phase_boundary)

    ax.set_title(
        "(c) Cost Compliance",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Avg Cost / Request ($)", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, loc="center right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    phase_boundary = data["phase1_n"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    _panel_flash_adoption(axes[0], data, phase_boundary)
    _panel_arm_composition(axes[1], data, phase_boundary)
    _panel_cost_compliance(axes[2], data, phase_boundary)

    fig.suptitle(
        r"Model Onboarding: K=3 $\to$ K=4 (Gemini Flash)"
        r" — 20 seeds, $\pm$1 SE",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"model_onboarding.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model_onboarding.{{pdf,png}} to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
