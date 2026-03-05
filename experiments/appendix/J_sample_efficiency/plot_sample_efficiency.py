#!/usr/bin/env python3
"""
Plot Appendix J: Sample Efficiency of Warmup Priors.

Two-panel figure (K=2 left, K=10 right) overlaying the BanditGPT
(warmup) and Tabula Rasa learning curves with 95% CI bands.

Annotations:
  - Steps-to-threshold markers for each variant.
  - Horizontal reference lines for oracle and best/weak static models.
  - Convergence region where both curves reach similar reward.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

BLUE = "#0072B2"
RED = "#D55E00"
GRAY = "#999999"
ORANGE = "#E69F00"
GREEN = "#009E73"


def plot_sample_efficiency(res: Dict[str, Any], out: Path) -> None:
    """Generate the two-panel sample efficiency figure.

    Args:
        res: Full results dict from ``run_sample_efficiency.py``.
        out: Output directory for the figure.
    """
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = float(sp_stats.t.ppf(0.975, df=n_seeds - 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    for ax, k_label, k_key in zip(axes, ["K=2", "K=10"], ["K2", "K10"]):
        if k_key not in res:
            continue
        k_data = res[k_key]

        warmup_curve = k_data["warmup_curve"]
        tr_curve = k_data["tabula_rasa_curve"]
        metrics = k_data["metrics"]
        oracle = k_data["oracle_per_prompt"]
        weak_static = k_data.get("weak_static", 0.0)

        _plot_panel(
            ax, warmup_curve, tr_curve, metrics,
            oracle, weak_static, n_seeds, t_crit, k_label,
        )

    fig.suptitle(
        "Sample Efficiency: Warmup Priors vs Tabula Rasa",
        fontsize=14, fontweight="bold", y=1.02,
    )

    path = out / "figure_sample_efficiency.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def _plot_panel(
    ax: plt.Axes,
    warmup_curve: List[Dict],
    tr_curve: List[Dict],
    metrics: Dict[str, Any],
    oracle: float,
    weak_static: float,
    n_seeds: int,
    t_crit: float,
    k_label: str,
) -> None:
    """Render a single panel of the sample-efficiency figure."""
    se = np.sqrt(n_seeds)

    # -- BanditGPT (warmup) --
    w_steps = [d["step"] for d in warmup_curve]
    w_rewards = [d["mean_reward"] for d in warmup_curve]
    w_ci = [t_crit * d["std_reward"] / se for d in warmup_curve]
    w_upper = [r + c for r, c in zip(w_rewards, w_ci)]
    w_lower = [r - c for r, c in zip(w_rewards, w_ci)]

    ax.plot(
        w_steps, w_rewards, "D-", color=BLUE, lw=2.5, ms=4, zorder=5,
        label=f"BanditGPT (warmup, n={n_seeds})",
    )
    ax.fill_between(w_steps, w_lower, w_upper, color=BLUE, alpha=0.12, zorder=2)

    # -- Tabula Rasa --
    t_steps = [d["step"] for d in tr_curve]
    t_rewards = [d["mean_reward"] for d in tr_curve]
    t_ci = [t_crit * d["std_reward"] / se for d in tr_curve]
    t_upper = [r + c for r, c in zip(t_rewards, t_ci)]
    t_lower = [r - c for r, c in zip(t_rewards, t_ci)]

    ax.plot(
        t_steps, t_rewards, "s--", color=RED, lw=2, ms=3.5, zorder=4,
        alpha=0.85, label=f"Tabula Rasa (no priors, n={n_seeds})",
    )
    ax.fill_between(t_steps, t_lower, t_upper, color=RED, alpha=0.08, zorder=1)

    # -- Reference lines --
    ax.axhline(
        y=oracle, color=GREEN, ls=":", lw=1.5, alpha=0.5, zorder=3,
        label=f"Oracle (per-prompt best: {oracle:.3f})",
    )
    if weak_static > 0:
        ax.axhline(
            y=weak_static, color=GRAY, ls=":", lw=1.5, alpha=0.5, zorder=3,
            label=f"Weak static ({weak_static:.3f})",
        )

    # -- Target threshold --
    target = metrics.get("target_reward")
    if target is not None:
        ax.axhline(
            y=target, color=ORANGE, ls="--", lw=1.2, alpha=0.6, zorder=3,
            label=f"90% oracle ({target:.3f})",
        )

    # -- Steps-to-threshold annotations --
    w_stt = metrics.get("warmup", {}).get("steps_to_threshold")
    t_stt = metrics.get("tabula_rasa", {}).get("steps_to_threshold")

    if w_stt is not None and target is not None:
        ax.axvline(x=w_stt, color=BLUE, ls=":", lw=1.2, alpha=0.4)
        ax.annotate(
            f"Warmup @ {w_stt}",
            xy=(w_stt, target),
            xytext=(w_stt + 30, target + 0.015),
            fontsize=8, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
            zorder=10,
        )

    if t_stt is not None and target is not None:
        ax.axvline(x=t_stt, color=RED, ls=":", lw=1.2, alpha=0.4)
        y_offset = -0.015 if w_stt is not None else 0.015
        ax.annotate(
            f"Tabula Rasa @ {t_stt}",
            xy=(t_stt, target),
            xytext=(t_stt + 30, target + y_offset),
            fontsize=8, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            zorder=10,
        )

    # -- Speedup annotation --
    speedup = metrics.get("speedup", "N/A")
    if speedup != "N/A" and w_stt is not None and t_stt is not None:
        mid_step = (w_stt + t_stt) / 2
        ax.annotate(
            f"{speedup} faster",
            xy=(mid_step, target),
            xytext=(mid_step, target - 0.035),
            fontsize=9, color=ORANGE, fontweight="bold",
            ha="center",
            arrowprops=dict(
                arrowstyle="<->", color=ORANGE, lw=1.8,
                connectionstyle="arc3,rad=0",
            ),
            zorder=10,
        )

    ax.set_xlabel("Online Learning Steps", fontsize=11, fontweight="bold")
    ax.set_ylabel("Holdout Reward (Quality)", fontsize=11, fontweight="bold")
    ax.set_title(f"Sample Efficiency — {k_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls="--")

    max_step = max(
        max(w_steps) if w_steps else 0,
        max(t_steps) if t_steps else 0,
    )
    ax.set_xlim(-30, max_step + 80)

    all_rewards = w_rewards + t_rewards
    if all_rewards:
        r_min = min(all_rewards)
        r_max = max(all_rewards)
        margin = (r_max - r_min) * 0.15 if r_max > r_min else 0.05
        ax.set_ylim(r_min - margin, min(r_max + margin, oracle + margin))

    ax.tick_params(labelsize=10)


def main() -> None:
    """Load results JSON and generate figure."""
    results_dir = Path(__file__).parent / "results"
    results_path = results_dir / "sample_efficiency_results.json"
    if not results_path.exists():
        print(f"Results not found at {results_path}. Run run_sample_efficiency.py first.")
        return
    with open(results_path) as f:
        res = json.load(f)
    plot_sample_efficiency(res, results_dir)


if __name__ == "__main__":
    main()
