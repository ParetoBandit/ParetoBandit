#!/usr/bin/env python3
"""Generate figure for Appendix: Cold-Start vs Warmup Prior Regret.

Reads ``results/warmup_ablation_results.json`` and produces a
violin + strip plot (``warmup_ablation.pdf/.png``) showing per-seed
total regret distributions for warmup vs tabula rasa across all budget
regimes.  The violin shape reveals the bimodal failure mode of
cold-start routing under budget constraints.

Usage:
    python experiments/appendix/warmup_ablation/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact

RESULTS_DIR = Path(__file__).parent / "results"

CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GRAY = "#999999"

GroupSpec = Tuple[str, str, str]

BUDGET_GROUPS: List[GroupSpec] = [
    ("ParetoBandit (warmup)", "Tabula Rasa", "Unconstrained"),
    ("Warmup (tight budget)", "Tabula Rasa (tight budget)", "Tight"),
    ("Warmup (moderate budget)", "Tabula Rasa (moderate budget)", "Moderate"),
    ("Warmup (loose budget)", "Tabula Rasa (loose budget)", "Loose"),
]


def _halfviolin(
    ax: plt.Axes,
    data: np.ndarray,
    center: float,
    side: str,
    color: str,
    width: float = 0.35,
) -> None:
    """Draw a half-violin (KDE density) on one side of *center*.

    Parameters
    ----------
    ax : Axes
        Target axes.
    data : 1-D array
        Sample values.
    center : float
        X-position of the violin centre line.
    side : {"left", "right"}
        Which side to draw on.
    color : str
        Fill colour.
    width : float
        Maximum half-width of the violin in data-coordinate units.
    """
    from scipy.stats import gaussian_kde

    if len(np.unique(data)) < 2:
        ax.plot([center, center], [data.min(), data.max()], color=color, lw=2)
        return

    kde = gaussian_kde(data, bw_method=0.4)
    y_grid = np.linspace(data.min() - 5, data.max() + 5, 200)
    density = kde(y_grid)
    density = density / density.max() * width

    if side == "left":
        ax.fill_betweenx(y_grid, center - density, center, alpha=0.25, color=color)
        ax.plot(center - density, y_grid, color=color, lw=0.8, alpha=0.6)
    else:
        ax.fill_betweenx(y_grid, center, center + density, alpha=0.25, color=color)
        ax.plot(center + density, y_grid, color=color, lw=0.8, alpha=0.6)


def main() -> None:
    with open(RESULTS_DIR / "warmup_ablation_results.json") as f:
        data = json.load(f)

    conditions = data["conditions"]
    paired_tests = {
        (t["warmup"], t["baseline"]): t
        for t in data.get("paired_tests", [])
    }

    available = [
        g for g in BUDGET_GROUPS
        if g[0] in conditions and g[1] in conditions
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x_positions = np.arange(len(available))
    jitter_w = 0.06
    violin_w = 0.30
    offset = 0.18
    rng = np.random.default_rng(42)

    random_regrets: np.ndarray | None = None
    if "Random" in conditions:
        random_regrets = np.array(conditions["Random"]["per_seed_regret"])

    for i, (warmup_key, tabula_key, budget_label) in enumerate(available):
        cx = x_positions[i]

        for cond_key, color, x_off, side, label in [
            (warmup_key, CB_BLUE, -offset, "left", "Warmup" if i == 0 else None),
            (tabula_key, CB_ORANGE, offset, "right", "Tabula Rasa" if i == 0 else None),
        ]:
            seeds = np.array(conditions[cond_key]["per_seed_regret"])
            pos = cx + x_off

            _halfviolin(ax, seeds, pos, side, color, width=violin_w)

            jitter = rng.uniform(-jitter_w, jitter_w, size=len(seeds))
            ax.scatter(
                pos + jitter, seeds,
                color=color, s=18, alpha=0.7, zorder=5,
                edgecolors="white", linewidths=0.3, label=label,
            )

            med = np.median(seeds)
            ax.hlines(
                med, pos - 0.08, pos + 0.08,
                color=color, linewidth=2.0, zorder=6,
            )

        test = paired_tests.get((warmup_key, tabula_key))
        if test is not None:
            wins = test.get("n_warmup_wins", 0)
            n_eff = test.get("n_effective", 0)
            p_sign = test.get("sign_test_p_value", 1.0)

            if p_sign < 1e-4:
                sign_str = "p_sign < 10⁻⁴"
            elif p_sign < 0.001:
                sign_str = f"p_sign = {p_sign:.4f}"
            elif p_sign < 0.05:
                sign_str = f"p_sign = {p_sign:.3f}"
            else:
                sign_str = f"p_sign = {p_sign:.2f}"

            w_cat = test.get("warmup_catastrophic_count", 0)
            b_cat = test.get("baseline_catastrophic_count", 0)
            n_seeds = len(conditions[warmup_key]["per_seed_regret"])

            fisher_table = [[w_cat, n_seeds - w_cat],
                            [b_cat, n_seeds - b_cat]]
            _, p_fisher = fisher_exact(fisher_table, alternative="less")

            if p_fisher < 1e-4:
                fisher_str = "p_Fisher < 10⁻⁴"
            elif p_fisher < 0.001:
                fisher_str = f"p_Fisher = {p_fisher:.4f}"
            elif p_fisher < 0.05:
                fisher_str = f"p_Fisher = {p_fisher:.3f}"
            else:
                fisher_str = f"p_Fisher = {p_fisher:.2f}"

            ax.annotate(
                f"Sign: {wins}/{n_eff} wins, {sign_str}\n"
                f"Cat: {w_cat}/{n_seeds} vs {b_cat}/{n_seeds}, {fisher_str}",
                xy=(cx, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -22), textcoords="offset points",
                ha="center", va="top", fontsize=6.0, color="0.35",
                style="italic", annotation_clip=False,
                linespacing=1.4,
            )

    if random_regrets is not None:
        unconstrained_x = x_positions[0]
        ax.scatter(
            [unconstrained_x], [np.median(random_regrets)],
            marker="*", s=200, color=CB_GRAY, zorder=7,
            edgecolors="black", linewidths=0.5,
            label="Random (unconstrained)",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([g[2] for g in available], fontsize=10)
    ax.set_xlabel("Budget Regime", fontsize=11, labelpad=55)
    ax.set_ylabel("Total Regret (per seed)", fontsize=11)
    ax.set_title(
        "Cold-Start Instability: Warmup Priors vs. Tabula Rasa\n"
        f"(K=3, {data['n_seeds']} seeds, test split n={data['n_prompts']})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            RESULTS_DIR / f"warmup_ablation.{ext}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Saved warmup_ablation.pdf/.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
