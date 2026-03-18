#!/usr/bin/env python3
"""Generate figures for Appendix: Validation Burn-In Ablation.

Reads ``results/val_burnin_ablation_results.json`` and produces two
figures:

1. **val_burnin_test_regret.pdf/.png** — Cumulative regret on the
   held-out test split under varying burn-in fractions (0–100% of val).
   Two panels: (a) full test trajectory, (b) zoom on first 400 steps.

2. **val_burnin_combined.pdf/.png** — Combined val+test cumulative
   regret for the 0% and 100% burn-in conditions, with a vertical line
   marking the val→test boundary.  This is the "no free lunch" view
   where all learning costs are visible.

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

from utils.bootstrap import bootstrap_ci_series

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

    # Panel (a): Full test trajectory
    ax_full = axes[0]
    # Panel (b): Zoom on first 400 steps
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
        ax.set_xlabel("Test Step")
        ax.grid(True, alpha=0.3)

    ax_full.set_ylabel("Cumulative Regret (test split)")
    ax_full.set_title("(a) Full test trajectory", fontsize=11)
    ax_full.legend(fontsize=8, loc="upper left")

    ax_zoom.set_xlim(0, 400)
    ax_zoom.set_title("(b) First 400 test steps (zoom)", fontsize=11)

    fig.suptitle(
        "Effect of validation burn-in on held-out test regret",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"val_burnin_test_regret.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved val_burnin_test_regret.pdf/.png to {RESULTS_DIR}")


def _figure_combined_trajectory(data: Dict[str, Any]) -> None:
    """Figure 2: Combined val+test trajectory (no-free-lunch view).

    Two panels with a shared y-axis:
      (a) Full 100% burn-in trajectory (val+test) with the val→test
          boundary marked, showing where val regret ends and test
          regret begins.
      (b) Aligned test-phase comparison: test-only regret for the
          100% burn-in condition overlaid with the 0% condition and
          Tabula Rasa, so the reader can compare test-phase learning
          rates on the same x-axis scale.
    """
    conditions = data["conditions"]

    full_burnin_key = "Warmup (100% burn-in)"
    no_burnin_key = "Warmup (0% burn-in)"
    tabula_no_key = "Tabula Rasa (no burn-in)"
    tabula_full_key = "Tabula Rasa (100% burn-in)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_combined = axes[0]
    ax_test = axes[1]

    # -- Panel (a): Full combined trajectory for 100% burn-in --
    if full_burnin_key in conditions:
        comb = conditions[full_burnin_key]["combined_trajectory"]
        n_val = comb["n_val_burnin"]

        _plot_condition(
            ax_combined, comb["curves"],
            color=CB_BLUE,
            linestyle="-",
            label="Warmup + 100% burn-in (val → test)",
            linewidth=2.0,
        )

    if tabula_full_key in conditions:
        comb = conditions[tabula_full_key]["combined_trajectory"]
        _plot_condition(
            ax_combined, comb["curves"],
            color=CB_ORANGE,
            linestyle="-",
            label="Tabula Rasa + 100% burn-in (val → test)",
            linewidth=1.8,
        )

    if full_burnin_key in conditions:
        n_val = conditions[full_burnin_key]["combined_trajectory"]["n_val_burnin"]
        ax_combined.axvline(
            x=n_val, color="black", linestyle="--",
            linewidth=1.2, alpha=0.7,
        )
        ymax = ax_combined.get_ylim()[1]
        ax_combined.annotate(
            "val → test boundary",
            xy=(n_val, ymax * 0.5),
            xytext=(n_val + 100, ymax * 0.65),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    ax_combined.set_xlabel("Global Step (val + test)")
    ax_combined.set_ylabel("Cumulative Regret")
    ax_combined.set_title(
        "(a) Full trajectory: all learning costs visible",
        fontsize=11,
    )
    ax_combined.legend(fontsize=8, loc="upper left")
    ax_combined.grid(True, alpha=0.3)

    # -- Panel (b): Test-phase regret, aligned at step 0 --
    for label_key, color, ls, lbl in [
        (full_burnin_key, CB_BLUE, "-", "Warmup + 100% burn-in"),
        (no_burnin_key, CB_RED, "-", "Warmup + 0% burn-in (priors only)"),
        (tabula_full_key, CB_ORANGE, "--", "Tabula Rasa + 100% burn-in"),
        (tabula_no_key, CB_GRAY, "--", "Tabula Rasa (no priors)"),
    ]:
        if label_key not in conditions:
            continue
        curves = conditions[label_key]["test_metrics"]["curves"]
        _plot_condition(
            ax_test, curves,
            color=color, linestyle=ls, label=lbl, linewidth=1.8,
        )

    ax_test.set_xlabel("Test Step")
    ax_test.set_ylabel("Cumulative Regret (test split)")
    ax_test.set_title(
        "(b) Test-phase regret (aligned comparison)",
        fontsize=11,
    )
    ax_test.legend(fontsize=8, loc="upper left")
    ax_test.grid(True, alpha=0.3)

    fig.suptitle(
        "No-free-lunch view: val learning costs + test performance",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"val_burnin_combined.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved val_burnin_combined.pdf/.png to {RESULTS_DIR}")


def _figure_summary_bar(data: Dict[str, Any]) -> None:
    """Figure 3: Grouped bar chart — warmup vs tabula rasa at key burn-in levels."""
    from utils.bootstrap import bootstrap_ci

    conditions = data["conditions"]

    group_labels = ["0%\nburn-in", "100%\nburn-in"]
    warmup_keys = ["Warmup (0% burn-in)", "Warmup (100% burn-in)"]
    tabula_keys = ["Tabula Rasa (no burn-in)", "Tabula Rasa (100% burn-in)"]

    warmup_means: List[float] = []
    warmup_ci_lo: List[float] = []
    warmup_ci_hi: List[float] = []
    tabula_means: List[float] = []
    tabula_ci_lo: List[float] = []
    tabula_ci_hi: List[float] = []

    for wk, tk in zip(warmup_keys, tabula_keys):
        for key_list, means_l, lo_l, hi_l in [
            ([wk], warmup_means, warmup_ci_lo, warmup_ci_hi),
            ([tk], tabula_means, tabula_ci_lo, tabula_ci_hi),
        ]:
            k = key_list[0]
            if k not in conditions:
                means_l.append(0.0)
                lo_l.append(0.0)
                hi_l.append(0.0)
                continue
            tm = conditions[k]["test_metrics"]
            regrets = tm["per_seed_test_regret"]
            lo, hi = bootstrap_ci(np.array(regrets))
            mean = tm["test_regret"]["mean"]
            means_l.append(mean)
            lo_l.append(mean - lo)
            hi_l.append(hi - mean)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(len(group_labels))
    width = 0.35

    bars_w = ax.bar(
        x - width / 2, warmup_means, width,
        color=CB_BLUE, alpha=0.85, edgecolor="white", label="Warmup priors",
    )
    ax.errorbar(
        x - width / 2, warmup_means,
        yerr=[warmup_ci_lo, warmup_ci_hi],
        fmt="none", ecolor="black", capsize=4, linewidth=1.2,
    )

    bars_t = ax.bar(
        x + width / 2, tabula_means, width,
        color=CB_GRAY, alpha=0.85, edgecolor="white", label="Tabula Rasa",
    )
    ax.errorbar(
        x + width / 2, tabula_means,
        yerr=[tabula_ci_lo, tabula_ci_hi],
        fmt="none", ecolor="black", capsize=4, linewidth=1.2,
    )

    for bars, means in [(bars_w, warmup_means), (bars_t, tabula_means)]:
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_xlabel("Val Burn-In Fraction", fontsize=11)
    ax.set_ylabel("Total Test Regret", fontsize=11)
    ax.set_title(
        "2×2 factorial: priors × burn-in (95% bootstrap CI)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"val_burnin_summary.{ext}",
            dpi=200, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved val_burnin_summary.pdf/.png to {RESULTS_DIR}")


def main() -> None:
    results_path = RESULTS_DIR / "val_burnin_ablation_results.json"
    with open(results_path) as f:
        data = json.load(f)

    _figure_test_regret(data)
    _figure_combined_trajectory(data)
    _figure_summary_bar(data)


if __name__ == "__main__":
    main()
