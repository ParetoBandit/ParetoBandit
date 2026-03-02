#!/usr/bin/env python3
"""
Generate Figure 3 from prequential_results.json.

Figure 3 (K=2) — two-panel:
    (a) Pareto frontier: warm-start, cold-start, RouteLLM, tabula rasa
    (b) Learning curve: BanditGPT quality vs training steps, RouteLLM peak ref

Figure 4 (K=10) has been moved to ``experiments/04_figure/plot_results.py``.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

RESULTS_DIR = Path(__file__).parent / "results"

BLUE = "#0072B2"
RED = "#D55E00"
GREEN = "#009E73"
GRAY = "#999999"
ORANGE = "#E69F00"


def load_results() -> dict:
    with open(RESULTS_DIR / "prequential_results.json") as f:
        return json.load(f)


def _pareto_hull(costs, rewards):
    """Monotone upper envelope sorted by ascending cost."""
    pairs = sorted(zip(costs, rewards), key=lambda x: (x[0], -x[1]))
    hull_c, hull_r = [], []
    best_r = -np.inf
    for c, r in pairs:
        if r > best_r:
            hull_c.append(c)
            hull_r.append(r)
            best_r = r
    return hull_c, hull_r


def _dev_pareto_indices(sweep, dev_cost_key, dev_reward_key):
    """Identify indices on the dev-set Pareto hull (no holdout leakage)."""
    n = len(sweep)
    pairs = [(sweep[i][dev_cost_key], sweep[i][dev_reward_key], i)
             for i in range(n)]
    pairs.sort(key=lambda x: (x[0], -x[1]))
    idx = []
    best_r = -np.inf
    for _, r, i in pairs:
        if r > best_r:
            idx.append(i)
            best_r = r
    return idx


def _dev_selected_deployed_hull(sweep, dev_cost_key, dev_reward_key,
                                holdout_cost_key, holdout_reward_key):
    """Dev-selected deployable frontier.

    Identifies Pareto-optimal points using dev metrics, then returns
    the holdout performance of those points (with a Pareto hull over
    the holdout values since dev-optimal points may not be monotone).
    """
    idx = _dev_pareto_indices(sweep, dev_cost_key, dev_reward_key)
    hc = [sweep[i][holdout_cost_key] for i in idx]
    hr = [sweep[i][holdout_reward_key] for i in idx]
    return _pareto_hull(hc, hr)


def plot_figure3(res: dict, out: Path) -> None:
    k2 = res["K2"]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    constrained_layout=True)

    # --- Panel A: Pareto Frontier ---
    # Bold lines = dev-selected frontier (deployable)
    # Shaded = oracle envelope (holdout-selected, upper bound)

    bp = k2["banditgpt_pareto"]
    bg_c = [p["mean_cost"] for p in bp]
    bg_r = [p["mean_reward"] for p in bp]
    bg_r_err = [t_crit * p["std_reward"] / np.sqrt(n_seeds) for p in bp]
    rl = k2["routellm"]["pareto"]

    # Oracle envelopes (shaded background)
    oracle_hull_c, oracle_hull_r = _pareto_hull(bg_c, bg_r)
    ax1.fill_between(oracle_hull_c, 0, oracle_hull_r, color=BLUE,
                     alpha=0.06, zorder=1, label="_nolegend_")
    rl_pairs = sorted(
        [(p["avg_cost"], p["avg_reward"]) for p in rl], key=lambda x: x[0],
    )
    rl_all_c = [c for c, _ in rl_pairs]
    rl_all_r = [r for _, r in rl_pairs]
    rl_oracle_hull_c, rl_oracle_hull_r = _pareto_hull(rl_all_c, rl_all_r)
    ax1.fill_between(rl_oracle_hull_c, 0, rl_oracle_hull_r, color=RED,
                     alpha=0.06, zorder=1, label="_nolegend_")

    # Scatter: all sweep points
    ax1.scatter(bg_c, bg_r, marker="D", s=15, color=BLUE,
                alpha=0.3, zorder=3)
    ax1.errorbar(bg_c, bg_r, yerr=bg_r_err, fmt="none", ecolor=BLUE,
                 alpha=0.2, capsize=2, zorder=3)
    ax1.scatter(rl_all_c, rl_all_r, marker="o", s=15, color=RED,
                alpha=0.3, zorder=3)

    # Dev-selected deployable frontiers (bold primary lines)
    # Hull is built from (dev_cost, dev_reward); plotted points are
    # (holdout_cost, holdout_reward) of the dev-optimal hyperparameters.
    has_dev = "dev_mean_cost" in bp[0] and "dev_mean_reward" in bp[0]
    if has_dev:
        ds_bg_c, ds_bg_r = _dev_selected_deployed_hull(
            bp, "dev_mean_cost", "dev_mean_reward",
            "mean_cost", "mean_reward")
        ax1.plot(ds_bg_c, ds_bg_r, "D-", color=BLUE, lw=2.5, ms=5, zorder=5,
                 label="BanditGPT (dev-selected)")
        ds_rl_c, ds_rl_r = _dev_selected_deployed_hull(
            rl, "dev_mean_cost", "dev_mean_reward",
            "avg_cost", "avg_reward")
        ax1.plot(ds_rl_c, ds_rl_r, "o-", color=RED, lw=2.5, ms=5, zorder=4,
                 label="RouteLLM-MF (dev-selected)")
    else:
        ax1.plot(oracle_hull_c, oracle_hull_r, "D-", color=BLUE, lw=2.5,
                 ms=5, zorder=5, label="BanditGPT (hybrid)")
        ax1.plot(rl_oracle_hull_c, rl_oracle_hull_r, "o-", color=RED,
                 lw=2.5, ms=5, zorder=4, label="RouteLLM-MF")

    # Static single-model baselines
    for m, s in k2["static"].items():
        ax1.scatter(s["cost"], s["reward"], marker="^", s=70, color=ORANGE,
                    zorder=6, alpha=0.8)
    ax1.scatter([], [], marker="^", s=70, color=ORANGE, label="Static (single model)")

    # Oracle
    ax1.axhline(y=k2["oracle"]["reward"], color=GREEN, ls="--", lw=1.5,
                alpha=0.5, zorder=1, label=f"Oracle ({k2['oracle']['reward']:.3f})")

    # Shaded background legend entry
    ax1.fill_between([], [], [], color=GRAY, alpha=0.1,
                     label="Oracle envelope (upper bound)")

    ax1.set_xlabel("Average Cost per Request ($)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Reward (Quality)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Cost–Quality Frontier — K=2", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax1.grid(True, alpha=0.15, ls="--")
    ax1.set_xlim(left=-0.0003)
    ax1.set_ylim(0.70, 0.83)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.4f}"))
    ax1.tick_params(labelsize=9)

    # --- Panel B: Learning Curve ---

    lc = k2["learning_curve"]
    steps = [d["step"] for d in lc]
    rewards = [d["mean_reward"] for d in lc]
    stds = [d["std_reward"] for d in lc]

    ci_upper = [r + t_crit * s / np.sqrt(n_seeds) for r, s in zip(rewards, stds)]
    ci_lower = [r - t_crit * s / np.sqrt(n_seeds) for r, s in zip(rewards, stds)]

    ax2.plot(steps, rewards, "D-", color=BLUE, lw=2.5, ms=4, zorder=5,
             label=f"BanditGPT (online, n={n_seeds} seeds)")
    ax2.fill_between(steps, ci_lower, ci_upper, color=BLUE, alpha=0.12, zorder=2)

    rl_peak = max(p["avg_reward"] for p in k2["routellm"]["pareto"])
    ax2.axhline(y=rl_peak, color=RED, ls="--", lw=2, alpha=0.8, zorder=3,
                label=f"RouteLLM peak ({rl_peak:.3f}, ~100k pre-trained)")

    weak_r = min(s["reward"] for s in k2["static"].values())
    ax2.axhline(y=weak_r, color=GRAY, ls=":", lw=1.5, alpha=0.6, zorder=3,
                label=f"Weak model static ({weak_r:.3f})")

    # Persistent crossover: earliest step from which the CI lower bound
    # stays above RouteLLM's peak for ALL subsequent checkpoints.
    crossover_step = None
    crossover_reward = None
    n_pts = len(ci_lower)
    for i in range(n_pts):
        if all(ci_lower[j] >= rl_peak for j in range(i, n_pts)):
            crossover_step = steps[i]
            crossover_reward = rewards[i]
            break

    if crossover_step is not None:
        ax2.axvline(x=crossover_step, color=ORANGE, ls=":", lw=1.5, alpha=0.6)
        ax2.annotate(
            f"Crossover @ step {crossover_step}",
            xy=(crossover_step, crossover_reward),
            xytext=(crossover_step + 80, crossover_reward - 0.015),
            fontsize=8, color=ORANGE, fontweight="bold",
            ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
            zorder=10,
        )
    else:
        ax2.text(
            0.98, 0.15, "No persistent\ncrossover detected",
            transform=ax2.transAxes, fontsize=9, color=ORANGE,
            ha="right", va="bottom", fontstyle="italic", alpha=0.8,
        )

    ax2.set_xlabel("Online Learning Steps (dev prompts seen)", fontsize=11,
                   fontweight="bold")
    ax2.set_ylabel("Average Reward (Quality)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Online Adaptation Value — K=2", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax2.grid(True, alpha=0.15, ls="--")
    ax2.set_xlim(-30, max(steps) + 50)
    ax2.set_ylim(0.70, 0.83)
    ax2.tick_params(labelsize=9)

    path = out / "figure3_k2.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    res = load_results()
    plot_figure3(res, RESULTS_DIR)
