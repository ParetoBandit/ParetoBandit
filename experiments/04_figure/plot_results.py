#!/usr/bin/env python3
"""
Generate Figure 4: The Value of Warmup Priors.

Two-panel figure:
  (a) Pareto frontier — BanditGPT (warmup) vs Tabula Rasa, with
      supervised baselines as reference points.
  (b) Learning curve — holdout reward vs online training steps,
      showing that warmup priors provide immediate quality.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.pareto import pareto_hull, dev_pareto_indices

RESULTS_DIR = Path(__file__).parent / "results"

BLUE = "#0072B2"
RED = "#D55E00"
GREEN = "#009E73"
GRAY = "#999999"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
TEAL = "#56B4E9"

SV_MARKERS: Dict[str, Tuple[str, str, str]] = {
    "knn": ("^", RED, "KNN"),
    "svm": ("s", GREEN, "SVM"),
    "mlp": ("P", PURPLE, "MLP"),
}


def load_results() -> dict:
    """Load warmup ablation results."""
    path = RESULTS_DIR / "warmup_ablation_results.json"
    with open(path) as f:
        return json.load(f)


def _ci95(mean: float, std: float, n: int) -> float:
    """95% t-CI half-width."""
    if n < 2 or std == 0:
        return 0.0
    return sp_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)


def _get_hull(pareto_data: list) -> Tuple[List[float], List[float]]:
    """Build dev-selected deployed Pareto hull."""
    idx = dev_pareto_indices(pareto_data, "dev_mean_cost", "dev_mean_reward")
    hc = [pareto_data[i]["mean_cost"] for i in idx]
    hr = [pareto_data[i]["mean_reward"] for i in idx]
    return pareto_hull(hc, hr)


def plot_figure4(results: dict, output_dir: Path) -> None:
    """Generate two-panel Figure 4."""
    k3 = results["K3"]

    fig, (ax_pareto, ax_lc) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.30},
    )

    # ================================================================
    # Panel (a): Pareto frontiers
    # ================================================================
    warmup_pareto = k3["warmup_pareto"]
    tr_pareto = k3["tabula_rasa_pareto"]

    # BanditGPT (warmup) — bold line
    wh_c, wh_r = _get_hull(warmup_pareto)
    ax_pareto.plot(
        wh_c, wh_r, "-o", color=BLUE, linewidth=2.5, markersize=4,
        label="BanditGPT (warmup priors)", zorder=5,
    )
    for pt in warmup_pareto:
        ci = _ci95(pt["mean_reward"], pt["std_reward"], pt["n_trials"])
        ax_pareto.errorbar(
            pt["mean_cost"], pt["mean_reward"], yerr=ci,
            fmt="none", color=BLUE, alpha=0.3, capsize=2,
        )

    # Tabula rasa — dashed
    th_c, th_r = _get_hull(tr_pareto)
    ax_pareto.plot(
        th_c, th_r, "--s", color=ORANGE, linewidth=2, markersize=4,
        label="Tabula rasa (no priors)", zorder=4,
    )
    for pt in tr_pareto:
        ci = _ci95(pt["mean_reward"], pt["std_reward"], pt["n_trials"])
        ax_pareto.errorbar(
            pt["mean_cost"], pt["mean_reward"], yerr=ci,
            fmt="none", color=ORANGE, alpha=0.3, capsize=2,
        )

    # Supervised baselines
    for kind, (marker, color, label) in SV_MARKERS.items():
        sv = k3["supervised"].get(kind)
        if sv:
            ax_pareto.scatter(
                sv["cost"], sv["reward"], marker=marker, color=color,
                s=100, zorder=6, edgecolors="k", linewidths=0.5,
                label=f"{label} (supervised)",
            )

    # Oracle
    oracle = k3["oracle"]
    ax_pareto.scatter(
        oracle["cost"], oracle["reward"], marker="*", color="gold",
        s=200, zorder=7, edgecolors="k", linewidths=0.5, label="Oracle",
    )

    # UCB1
    ucb1 = k3["ucb1"]
    ax_pareto.scatter(
        ucb1.get("cost", ucb1.get("mean_cost", 0.0034)),
        ucb1["reward"],
        marker="D", color=GRAY, s=80, zorder=6,
        edgecolors="k", linewidths=0.5, label="UCB1",
    )

    # Pareto AUC annotation
    auc_info = k3["pareto_auc"]
    bootstrap = auc_info["bootstrap_ci"]
    ax_pareto.text(
        0.03, 0.03,
        (
            f"Pareto AUC: {auc_info['warmup']:.3f} vs "
            f"{auc_info['tabula_rasa']:.3f}\n"
            f"Advantage: {auc_info['advantage']:+.3f} "
            f"(95% CI [{bootstrap['ci_95_lower']:+.3f}, "
            f"{bootstrap['ci_95_upper']:+.3f}])"
        ),
        transform=ax_pareto.transAxes,
        fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax_pareto.set_xlabel("Normalized cost ($/request)", fontsize=11)
    ax_pareto.set_ylabel("Holdout reward", fontsize=11)
    ax_pareto.set_title("(a) Pareto Frontier: Warmup vs Tabula Rasa", fontsize=12)
    ax_pareto.legend(fontsize=8, loc="lower right")
    ax_pareto.grid(True, alpha=0.3)

    # ================================================================
    # Panel (b): Learning curves
    # ================================================================
    lc_warmup = k3["warmup_learning_curve"]
    lc_tr = k3["tabula_rasa_learning_curve"]

    w_steps = [d["step"] for d in lc_warmup]
    w_means = [d["mean_reward"] for d in lc_warmup]
    w_cis = [
        _ci95(d["mean_reward"], d["std_reward"], d["n_trials"])
        for d in lc_warmup
    ]

    t_steps = [d["step"] for d in lc_tr]
    t_means = [d["mean_reward"] for d in lc_tr]
    t_cis = [
        _ci95(d["mean_reward"], d["std_reward"], d["n_trials"])
        for d in lc_tr
    ]

    ax_lc.plot(w_steps, w_means, "-o", color=BLUE, linewidth=2, markersize=4,
               label="BanditGPT (warmup priors)")
    ax_lc.fill_between(
        w_steps,
        [m - c for m, c in zip(w_means, w_cis)],
        [m + c for m, c in zip(w_means, w_cis)],
        color=BLUE, alpha=0.15,
    )

    ax_lc.plot(t_steps, t_means, "--s", color=ORANGE, linewidth=2, markersize=4,
               label="Tabula rasa (no priors)")
    ax_lc.fill_between(
        t_steps,
        [m - c for m, c in zip(t_means, t_cis)],
        [m + c for m, c in zip(t_means, t_cis)],
        color=ORANGE, alpha=0.15,
    )

    # Supervised learning curve
    sv_lc_info = k3.get("supervised_learning_curve", {})
    sv_lc_data = sv_lc_info.get("curve", [])
    if sv_lc_data:
        sv_steps = [d["step"] for d in sv_lc_data]
        sv_means = [d["mean_reward"] for d in sv_lc_data]
        ax_lc.plot(
            sv_steps, sv_means, ":", color=GREEN, linewidth=1.5,
            label=f"{sv_lc_info.get('kind', 'SVM').upper()} (supervised)",
        )

    # Reference lines for supervised baselines
    x_max = max(w_steps[-1], t_steps[-1])
    for kind, (_, color, label) in SV_MARKERS.items():
        sv = k3["supervised"].get(kind)
        if sv:
            ax_lc.axhline(
                sv["reward"], color=color, linestyle=":", alpha=0.6,
                linewidth=1,
            )
            ax_lc.text(
                x_max * 1.01, sv["reward"], f"{label}",
                fontsize=7, color=color, va="center",
            )

    # Oracle reference
    ax_lc.axhline(
        oracle["reward"], color="gold", linestyle="--", alpha=0.5,
        linewidth=1,
    )
    ax_lc.text(
        x_max * 1.01, oracle["reward"], "Oracle",
        fontsize=7, color="goldenrod", va="center",
    )

    # Sample efficiency annotation
    se = k3.get("sample_efficiency", {})
    step0_adv = (
        (se.get("warmup_step0", 0) or 0) - (se.get("tabula_rasa_step0", 0) or 0)
    )
    if step0_adv > 0:
        ax_lc.annotate(
            f"Step-0 advantage:\n+{step0_adv:.3f}",
            xy=(0, se.get("warmup_step0", 0)),
            xytext=(x_max * 0.15, se.get("warmup_step0", 0) - 0.02),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    ax_lc.set_xlabel("Online training steps", fontsize=11)
    ax_lc.set_ylabel("Holdout reward", fontsize=11)
    ax_lc.set_title("(b) Learning Curve: Sample Efficiency", fontsize=12)
    ax_lc.legend(fontsize=8, loc="lower right")
    ax_lc.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "figure4_warmup_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    data = load_results()
    plot_figure4(data, RESULTS_DIR)
