#!/usr/bin/env python3
"""Generate the uncertainty-evolution figure for the warmup ablation appendix.

Compares tr(A_inv) per arm between the warmup-prior, tabula-rasa, and
matched-γ tabula-rasa conditions.  The matched-γ control isolates the
contribution of priors vs. the forgetting-factor difference.

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


_CONDITION_STYLES: Dict[str, tuple] = {
    "ParetoBandit (warmup)": ("-", "warmup"),
    "Tabula Rasa": ("--", "cold start"),
    "Tabula Rasa (matched-γ)": (":", "cold start γ-matched"),
}


def _plot_uncertainty_curves(
    ax: plt.Axes,
    curves: List[Dict[str, Any]],
    arm: str,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Plot one condition's uncertainty curve for a single arm."""
    steps = [c["step"] for c in curves]
    means = [c["trace_A_inv_mean"][arm] for c in curves]
    matrix = np.array([c["per_seed_trace_A_inv"][arm] for c in curves])
    lo, hi = bootstrap_ci_series(matrix)

    ax.plot(
        steps, means,
        color=color, linewidth=2.0, linestyle=linestyle,
        label=f"{arm} ({label})", zorder=4,
    )
    ax.fill_between(steps, lo, hi, alpha=0.08, color=color, zorder=2)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()

    cond_data = {
        key: data["conditions"][key]
        for key in _CONDITION_STYLES
        if key in data["conditions"]
    }

    has_curves = all(
        cond.get("uncertainty_curves") for cond in cond_data.values()
    )
    if not has_curves:
        print(
            "ERROR: Missing uncertainty_curves. "
            "Re-run run_warmup_ablation.py."
        )
        sys.exit(1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    for arm in ARM_ORDER:
        color = ARM_COLORS[arm]
        for cond_key, (ls, ls_label) in _CONDITION_STYLES.items():
            if cond_key not in cond_data:
                continue
            curves = cond_data[cond_key]["uncertainty_curves"]
            _plot_uncertainty_curves(ax, curves, arm, color, ls, ls_label)

    ax.set_yscale("log")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel(r"$\mathrm{tr}(A_a^{-1})$  (total uncertainty)", fontsize=11)

    n_seeds = data.get("n_seeds", 20)
    ax.set_title(
        "Uncertainty Evolution: Warmup vs. Cold Start vs. γ-Matched\n"
        f"(K=3 stationary, {n_seeds} seeds, 95% bootstrap CI)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", framealpha=0.9)
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
