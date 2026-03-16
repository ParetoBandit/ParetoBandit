#!/usr/bin/env python3
"""Generate the uncertainty-evolution figure for the warmup ablation appendix.

Compares tr(A_inv) per arm between the warmup-prior and tabula-rasa
conditions, showing how warmup priors give the router an immediate
information advantage that the cold-start condition must earn online.

Reads ``results/warmup_ablation_results.json`` (which must include
``uncertainty_curves``).

Usage:
    python experiments/appendix/warmup_ablation/generate_uncertainty_figure.py
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

ARM_COLORS: Dict[str, str] = {
    "Llama-8B": "#56B4E9",
    "Mistral-Large": "#D55E00",
    "Gemini-Pro": "#009E73",
}

ARM_ORDER: List[str] = ["Llama-8B", "Mistral-Large", "Gemini-Pro"]


def _load() -> Dict[str, Any]:
    with open(RESULTS_DIR / "warmup_ablation_results.json") as f:
        return json.load(f)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()

    warmup = data["conditions"]["ParetoBandit (warmup)"]
    tabula = data["conditions"]["Tabula Rasa"]

    w_curves = warmup["uncertainty_curves"]
    t_curves = tabula["uncertainty_curves"]

    if not w_curves or not t_curves:
        print("ERROR: No uncertainty_curves in results. Re-run run_warmup_ablation.py.")
        sys.exit(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    for arm in ARM_ORDER:
        color = ARM_COLORS[arm]

        # Warmup (solid)
        w_steps = [c["step"] for c in w_curves]
        w_means = [c["trace_A_inv_mean"][arm] for c in w_curves]
        w_matrix = np.array([c["per_seed_trace_A_inv"][arm] for c in w_curves])
        w_lo, w_hi = bootstrap_ci_series(w_matrix)

        ax.plot(
            w_steps, w_means,
            color=color, linewidth=2.0, linestyle="-",
            label=f"{arm} (warmup)", zorder=4,
        )
        ax.fill_between(w_steps, w_lo, w_hi, alpha=0.10, color=color, zorder=2)

        # Tabula rasa (dashed)
        t_steps = [c["step"] for c in t_curves]
        t_means = [c["trace_A_inv_mean"][arm] for c in t_curves]
        t_matrix = np.array([c["per_seed_trace_A_inv"][arm] for c in t_curves])
        t_lo, t_hi = bootstrap_ci_series(t_matrix)

        ax.plot(
            t_steps, t_means,
            color=color, linewidth=2.0, linestyle="--",
            label=f"{arm} (cold start)", zorder=4,
        )
        ax.fill_between(t_steps, t_lo, t_hi, alpha=0.08, color=color, zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel(r"$\mathrm{tr}(A_a^{-1})$  (total uncertainty)", fontsize=11)
    ax.set_title(
        "Uncertainty Evolution: Warmup Priors vs. Cold Start\n"
        "(K=3 stationary, 20 seeds, 95% bootstrap CI)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.5, which="both")
    ax.tick_params(labelsize=9)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"warmup_uncertainty.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved warmup_uncertainty.{{pdf,png}} to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
