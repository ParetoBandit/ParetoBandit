#!/usr/bin/env python3
"""Generate the model onboarding appendix figure (K=3 → K=4).

Reads ``results/model_onboarding_results.json`` and produces a
publication-ready 1×3 panel figure showing Flash adoption across
three onboarding scenarios (good_cheap, bad_cheap, good_expensive),
with budget tiers overlaid in each panel.

Usage:
    python experiments/04_model_onboarding/generate_figure.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.bootstrap import bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "model_onboarding_results.json"

# ======================================================================
# Visual constants
# ======================================================================

FLASH_ID = "google/gemini-2.5-flash"

BUDGET_STYLES: Dict[str, Dict[str, Any]] = {
    "tight": {"color": "#D55E00", "linestyle": "-", "label": "Tight"},
    "moderate": {"color": "#0072B2", "linestyle": "-", "label": "Moderate"},
    "loose": {"color": "#CC79A7", "linestyle": "-", "label": "Loose"},
    "unconstrained": {
        "color": "#444444",
        "linestyle": "--",
        "label": "Unconstrained",
    },
}

BUDGET_ORDER: List[str] = ["tight", "moderate", "loose", "unconstrained"]

SCENARIO_TITLES: Dict[str, str] = {
    "good_cheap": "(a) Good & Cheap — adopted",
    "good_expensive": "(b) Good & Expensive — suppressed",
    "bad_cheap": "(c) Bad & Cheap — rejected",
}

SCENARIO_ORDER: List[str] = ["good_cheap", "good_expensive", "bad_cheap"]


def _load() -> Dict[str, Any]:
    with open(RESULTS_DIR / RESULTS_FILE) as f:
        return json.load(f)


def _add_phase_shading(
    ax: plt.Axes,
    boundary: int,
    burnin_boundary: int | None = None,
    *,
    label_left: str = "Phase 1\n(K=3)",
    label_right: str = "Phase 2\n(K=4, +Flash)",
) -> None:
    """Shade the Phase 1 region and annotate phase / burn-in boundaries."""
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
        boundary * 0.5,
        0.96,
        label_left,
        transform=trans,
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color="#333333",
    )
    xlim = ax.get_xlim()
    right_center = boundary + (xlim[1] - boundary) * 0.5
    ax.text(
        right_center,
        0.96,
        label_right,
        transform=trans,
        ha="center",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color="#333333",
    )

    if burnin_boundary is not None:
        ax.axvline(
            burnin_boundary,
            color="#E69F00",
            linestyle=":",
            linewidth=1.4,
            alpha=0.7,
            zorder=1,
        )


# ======================================================================
# Panel: Flash adoption across budget tiers for a single scenario
# ======================================================================


def _panel_flash_by_budget(
    ax: plt.Axes,
    scenario_data: Dict[str, Any],
    phase_boundary: int,
    burnin_pulls: int,
    scenario_name: str,
) -> None:
    """Flash windowed mix for all budget tiers in one scenario panel."""
    traces = scenario_data["checkpoint_traces"]

    for blabel in BUDGET_ORDER:
        key = f"paretobandit_transfer_{blabel}"
        if key not in traces:
            continue
        trace = traces[key]
        steps = [c["step"] for c in trace]
        style = BUDGET_STYLES[blabel]
        has_per_seed = "per_seed_windowed_mix" in trace[0]

        means = [
            c["windowed_mix_mean"].get(FLASH_ID, 0.0) for c in trace
        ]

        if has_per_seed:
            matrix = np.array(
                [
                    c["per_seed_windowed_mix"].get(FLASH_ID, [0.0])
                    for c in trace
                ]
            )
            ci_lo, ci_hi = bootstrap_ci_series(matrix, ci_level=0.95)
        else:
            z95 = 1.96
            ses = [
                c["windowed_mix_se"].get(FLASH_ID, 0.0) for c in trace
            ]
            ci_lo = [m - z95 * s for m, s in zip(means, ses)]
            ci_hi = [m + z95 * s for m, s in zip(means, ses)]

        ax.plot(
            steps,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            label=style["label"],
            zorder=4,
        )
        ax.fill_between(
            steps,
            ci_lo,
            ci_hi,
            alpha=0.10,
            color=style["color"],
            zorder=2,
        )

    ax.set_ylim(-0.02, 0.52)
    burnin_boundary = (
        phase_boundary + burnin_pulls if burnin_pulls > 0 else None
    )
    _add_phase_shading(ax, phase_boundary, burnin_boundary)

    title = SCENARIO_TITLES.get(scenario_name, scenario_name)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Prompts Routed", fontsize=13)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=11)



# ======================================================================
# Main
# ======================================================================


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    phase_boundary = data["phase1_n"]
    burnin_pulls = data.get("burnin_pulls", 0)
    n_seeds = data.get("n_seeds", 20)

    scenarios = data.get("scenarios", {})
    if not scenarios:
        print("No scenario data found in results JSON. Exiting.")
        return

    available = [s for s in SCENARIO_ORDER if s in scenarios]
    n_scenarios = len(available)
    if n_scenarios == 0:
        print("No matching scenarios found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Figure 1: 1 x N_scenarios — Flash adoption by budget tier
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, n_scenarios, figsize=(5.5 * n_scenarios, 5.0), squeeze=False,
    )
    axes_row = axes[0]

    mid_col = n_scenarios // 2
    for col, scenario_name in enumerate(available):
        ax = axes_row[col]
        _panel_flash_by_budget(
            ax,
            scenarios[scenario_name],
            phase_boundary,
            burnin_pulls,
            scenario_name,
        )
        if col == 0:
            ax.set_ylabel("Flash Windowed Fraction", fontsize=13)
        if col == mid_col:
            ax.legend(
                fontsize=11,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=4,
                framealpha=0.9,
            )

    fig.suptitle(
        r"Model Onboarding: K=3 $\to$ K=4 (Gemini Flash)"
        f" — {n_seeds} seeds, 95% bootstrap CI",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.97])

    for fmt in ("pdf", "png"):
        out = RESULTS_DIR / f"model_onboarding.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model_onboarding.{{pdf,png}} to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
