#!/usr/bin/env python3
"""Generate figures for Appendix: Validation Burn-In Ablation.

Reads ``results/val_burnin_ablation_results.json`` and produces figures:

1. **val_burnin_test_regret** — Cumulative regret on the held-out test
   split under varying burn-in fractions (unconstrained).

2. **val_burnin_budget_summary** — Budget-stratified 2×2 factorial bar
   chart showing that the burn-in finding holds under tight and moderate
   budget constraints, with cost compliance panel.

Usage:
    python experiments/appendix/val_burnin_ablation/generate_figure.py
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

from utils.bootstrap import bootstrap_ci, bootstrap_ci_series

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_GRAY = "#999999"
CB_TEAL = "#56B4E9"

BURNIN_COLORS: Dict[float, str] = {
    0.0: CB_RED,
    0.25: CB_ORANGE,
    0.50: CB_PURPLE,
    0.75: CB_TEAL,
    1.0: CB_BLUE,
}

BURNIN_LABELS: Dict[float, str] = {
    0.0: "0% burn-in (priors only)",
    0.25: "25% burn-in (446 steps)",
    0.50: "50% burn-in (893 steps)",
    0.75: "75% burn-in (1339 steps)",
    1.0: "100% burn-in (1785 steps)",
}


def _plot_condition(
    ax: plt.Axes,
    curves: List[Dict[str, Any]],
    *,
    color: str,
    linestyle: str,
    label: str,
    linewidth: float = 1.8,
) -> None:
    """Plot one condition's cumulative regret with bootstrap CI."""
    steps = [c["step"] for c in curves]
    mean_reg = [c["mean_cumulative_regret"] for c in curves]

    if "per_seed_cumulative_regret" in curves[0]:
        matrix = np.array([c["per_seed_cumulative_regret"] for c in curves])
        ci_lo, ci_hi = bootstrap_ci_series(matrix)
    else:
        se_reg = [c["se_cumulative_regret"] for c in curves]
        ci_lo = [m - 1.96 * s for m, s in zip(mean_reg, se_reg)]
        ci_hi = [m + 1.96 * s for m, s in zip(mean_reg, se_reg)]

    ax.plot(
        steps, mean_reg,
        color=color, linestyle=linestyle, linewidth=linewidth, label=label,
    )
    ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.12)


def _figure_test_regret(data: Dict[str, Any]) -> None:
    """Figure 1: Test-split cumulative regret by burn-in fraction."""
    conditions = data["conditions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_full = axes[0]
    ax_zoom = axes[1]

    for frac in data["burnin_fractions"]:
        label_key = f"Warmup ({int(frac * 100)}% burn-in)"
        cond = conditions[label_key]
        curves = cond["test_metrics"]["curves"]

        for ax in (ax_full, ax_zoom):
            _plot_condition(
                ax, curves,
                color=BURNIN_COLORS[frac],
                linestyle="-",
                label=BURNIN_LABELS[frac],
            )

    for tabula_key, tabula_label, tabula_ls in [
        ("Tabula Rasa (no burn-in)", "Tabula Rasa (no priors)", "--"),
        ("Tabula Rasa (100% burn-in)", "Tabula Rasa + 100% burn-in", ":"),
    ]:
        if tabula_key not in conditions:
            continue
        curves = conditions[tabula_key]["test_metrics"]["curves"]
        for ax in (ax_full, ax_zoom):
            _plot_condition(
                ax, curves,
                color=CB_GRAY,
                linestyle=tabula_ls,
                label=tabula_label,
            )

    for ax in (ax_full, ax_zoom):
        ax.axvline(
            x=data["early_step"], color="black", linestyle=":",
            linewidth=0.8, alpha=0.5,
        )
        ax.set_xlabel("Test Step", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    ax_full.set_ylabel("Cumulative Regret (test split)", fontsize=13)
    ax_full.set_title("(a) Full test trajectory", fontsize=13)
    ax_full.legend(fontsize=10, loc="upper left")

    ax_zoom.set_xlim(0, 400)
    ax_zoom.set_title("(b) First 400 test steps (zoom)", fontsize=13)

    fig.suptitle(
        "Effect of validation burn-in on held-out test regret",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"val_burnin_test_regret.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved val_burnin_test_regret.pdf/.png to {RESULTS_DIR}")


DISCUSSED_REGIMES = {"unconstrained", "tight", "moderate"}
"""Budget regimes discussed in results_discussion.tex.

Only these regimes appear in the figures.  If results exist for
additional regimes (e.g. 'loose') they are still in the JSON but
are excluded from the plots to avoid presenting undiscussed data.
"""


def _figure_budget_summary(data: Dict[str, Any]) -> None:
    """Budget-stratified 2×2 factorial with cost compliance.

    Two panels:
      (a) Total test regret across budget regimes.
      (b) Cost compliance (mean cost / target) for budget-constrained
          regimes, confirming that spend is consistent across conditions.

    Shared legend placed below the panels.
    """
    conditions = data["conditions"]
    budget_regimes = data.get("budget_regimes", {"unconstrained": None})

    regime_labels: List[str] = []
    group_specs: List[Dict[str, str]] = []
    regime_targets: List[float | None] = []

    for regime_name, target in budget_regimes.items():
        if regime_name not in DISCUSSED_REGIMES:
            continue
        regime_labels.append(regime_name.capitalize())
        regime_targets.append(target)
        if regime_name == "unconstrained":
            group_specs.append({
                "w0": "Warmup (0% burn-in)",
                "w100": "Warmup (100% burn-in)",
                "t0": "Tabula Rasa (no burn-in)",
                "t100": "Tabula Rasa (100% burn-in)",
            })
        else:
            group_specs.append({
                "w0": f"Warmup (0% burn-in, {regime_name})",
                "w100": f"Warmup (100% burn-in, {regime_name})",
                "t0": f"Tabula Rasa (0% burn-in, {regime_name})",
                "t100": f"Tabula Rasa (100% burn-in, {regime_name})",
            })

    bar_defs = [
        ("w0", "Warmup, 0% burn-in", CB_BLUE, 0.85),
        ("w100", "Warmup, 100% burn-in", CB_TEAL, 0.85),
        ("t100", "Tabula Rasa, 100% burn-in", CB_GRAY, 0.85),
        ("t0", "Tabula Rasa, 0% burn-in", CB_ORANGE, 0.85),
    ]

    n_regimes = len(regime_labels)
    n_bars = len(bar_defs)

    constrained_idxs = [
        i for i, t in enumerate(regime_targets) if t is not None
    ]
    has_compliance = len(constrained_idxs) > 0

    if has_compliance:
        fig, (ax_reg, ax_cost) = plt.subplots(
            1, 2, figsize=(max(14, 4 * n_regimes), 6.0),
        )
    else:
        fig, ax_reg = plt.subplots(
            1, 1, figsize=(max(9, 3 * n_regimes), 6.0),
        )
        ax_cost = None

    # -- Panel (a): Regret bars --
    x = np.arange(n_regimes)
    bar_width = 0.18
    offsets = np.arange(n_bars) - (n_bars - 1) / 2

    legend_handles = []
    for i, (key, label, color, alpha) in enumerate(bar_defs):
        means: List[float] = []
        errs_lo: List[float] = []
        errs_hi: List[float] = []

        for g in group_specs:
            cond_key = g[key]
            if cond_key not in conditions:
                means.append(0.0)
                errs_lo.append(0.0)
                errs_hi.append(0.0)
                continue
            tm = conditions[cond_key]["test_metrics"]
            regrets = tm["per_seed_test_regret"]
            lo, hi = bootstrap_ci(np.array(regrets))
            m = tm["test_regret"]["mean"]
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)

        pos = x + offsets[i] * bar_width
        bars = ax_reg.bar(
            pos, means, bar_width,
            color=color, alpha=alpha, edgecolor="white", label=label,
        )
        legend_handles.append(bars)
        ax_reg.errorbar(
            pos, means,
            yerr=[errs_lo, errs_hi],
            fmt="none", ecolor="black", capsize=3, linewidth=1.0,
        )

    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels(regime_labels, fontsize=13)
    ax_reg.set_xlabel("Budget Regime", fontsize=13)
    ax_reg.set_ylabel("Total Test Regret", fontsize=13)
    ax_reg.tick_params(axis="both", labelsize=12)
    panel_lbl = "(a) " if has_compliance else ""
    ax_reg.set_title(
        f"{panel_lbl}Val burn-in effect across budget regimes\n"
        "(2×2 factorial: priors × burn-in; 95% bootstrap CI)",
        fontsize=13, fontweight="bold",
    )
    ax_reg.grid(axis="y", alpha=0.3)

    # -- Panel (b): Cost compliance (budget-constrained regimes only) --
    if has_compliance and ax_cost is not None:
        c_labels = [regime_labels[i] for i in constrained_idxs]
        c_specs = [group_specs[i] for i in constrained_idxs]
        c_targets = [regime_targets[i] for i in constrained_idxs]
        n_c = len(c_labels)
        xc = np.arange(n_c)

        for i, (key, label, color, alpha) in enumerate(bar_defs):
            ratios: List[float] = []
            for gi, g in enumerate(c_specs):
                cond_key = g[key]
                if cond_key not in conditions:
                    ratios.append(0.0)
                    continue
                bc = conditions[cond_key].get("budget_compliance")
                if bc is not None:
                    ratios.append(bc["mean_cost_target_ratio"])
                else:
                    tm = conditions[cond_key]["test_metrics"]
                    mean_cost = tm["test_mean_cost_usd"]["mean"]
                    tgt = c_targets[gi] if c_targets[gi] else 1.0
                    ratios.append(mean_cost / tgt)

            pos = xc + offsets[i] * bar_width
            ax_cost.bar(
                pos, ratios, bar_width,
                color=color, alpha=alpha, edgecolor="white", label=label,
            )
            for p, r in zip(pos, ratios):
                if r > 0:
                    ax_cost.text(
                        p, r + 0.01, f"{r:.2f}",
                        ha="center", va="bottom", fontsize=11,
                    )
        ax_cost.set_xticks(xc)
        ax_cost.set_xticklabels(c_labels, fontsize=13)
        ax_cost.set_xlabel("Budget Regime", fontsize=13)
        ax_cost.set_ylabel("Mean Cost / Target", fontsize=13)
        ax_cost.tick_params(axis="both", labelsize=12)
        ax_cost.set_title(
            "(b) Budget compliance\n"
            "(1.0 = exactly at target; <1.0 = under budget)",
            fontsize=13, fontweight="bold",
        )
        ax_cost.grid(axis="y", alpha=0.3)

    # -- Shared legend below panels --
    legend_labels = [bd[1] for bd in bar_defs]
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(w_pad=4.0)
    fig.subplots_adjust(bottom=0.15)

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"val_burnin_budget_summary.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved val_burnin_budget_summary.pdf/.png to {RESULTS_DIR}")


def main() -> None:
    results_path = RESULTS_DIR / "val_burnin_ablation_results.json"
    with open(results_path) as f:
        data = json.load(f)

    _figure_test_regret(data)

    if "budget_regimes" in data and len(data["budget_regimes"]) > 1:
        _figure_budget_summary(data)


if __name__ == "__main__":
    main()
