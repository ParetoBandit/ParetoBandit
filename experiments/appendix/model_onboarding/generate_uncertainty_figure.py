#!/usr/bin/env python3
"""Generate the uncertainty-evolution figure for the model onboarding appendix.

Plots tr(A_inv) per arm over the Phase 1 → Phase 2 trajectory, showing
how cold-started Flash starts at maximum uncertainty (tr(A_inv) = d/λ)
and converges toward the warm-started incumbents.

Reads ``results/model_onboarding_results.json`` (which must include the
``trace_A_inv`` fields produced by the updated run script).

Usage:
    python experiments/appendix/model_onboarding/generate_uncertainty_figure.py
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

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "model_onboarding_results.json"

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

K4_ARMS: List[str] = list(ARM_SHORT.keys())

BUDGET_LABEL = "moderate"


def _load() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    phase_boundary = data["phase1_n"]

    key = f"paretobandit_transfer_{BUDGET_LABEL}"
    trace = data["checkpoint_traces"][key]

    steps = [c["step"] for c in trace]

    has_per_seed = "per_seed_trace_A_inv" in trace[0]
    if not has_per_seed:
        print("ERROR: Results JSON does not contain per_seed_trace_A_inv. "
              "Re-run run_model_onboarding.py first.")
        sys.exit(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    for arm_id in K4_ARMS:
        means = [c["trace_A_inv_mean"].get(arm_id, 0.0) for c in trace]
        color = ARM_COLORS[arm_id]
        label = ARM_SHORT[arm_id]

        matrix = np.array([
            c["per_seed_trace_A_inv"].get(arm_id, [0.0])
            for c in trace
        ])

        nonzero_mask = np.array(means) > 0
        plot_steps = [s for s, m in zip(steps, nonzero_mask) if m]
        plot_means = [v for v, m in zip(means, nonzero_mask) if m]
        plot_matrix = matrix[nonzero_mask]

        if len(plot_steps) == 0:
            continue

        ci_lo, ci_hi = bootstrap_ci_series(plot_matrix)

        lw = 2.8 if arm_id == "google/gemini-2.5-flash" else 1.8
        ls = "-"

        ax.plot(
            plot_steps, plot_means,
            color=color, linewidth=lw, linestyle=ls,
            label=label, zorder=4,
        )
        ax.fill_between(
            plot_steps, ci_lo, ci_hi,
            alpha=0.12, color=color, zorder=2,
        )

    ax.axvline(
        phase_boundary, color="black", linestyle="--",
        linewidth=1.0, alpha=0.4, zorder=1,
    )
    ax.text(
        phase_boundary + 20, 70,
        "onboard\nFlash", ha="left", va="top",
        fontsize=8, fontstyle="italic", color="#555555",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel(r"$\mathrm{tr}(A_a^{-1})$  (total uncertainty)", fontsize=11)
    ax.set_title(
        r"Uncertainty Evolution: Warm-Started Arms vs. Cold-Started Flash"
        f"\n({BUDGET_LABEL.title()} budget, 20 seeds, 95% bootstrap CI)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5, which="both")
    ax.tick_params(labelsize=9)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"uncertainty_evolution.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved uncertainty_evolution.{{pdf,png}} to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
