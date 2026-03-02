#!/usr/bin/env python3
"""
Generate Figure 4 (K=10 Multi-Model Pareto Frontier) from results JSON.

Single-panel figure showing the dev-selected deployable Pareto frontier
for BanditGPT vs tabula rasa, with oracle, best-static, best-static+noise,
UCB1, and random baselines.

Can load results from either:
  - ``results/multimodel_pareto_results.json`` (produced by this folder's
    ``run_multimodel_pareto.py``)
  - ``../03_figure/results/prequential_results.json`` (legacy combined results)
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
PURPLE = "#CC79A7"


def load_results() -> dict:
    """Load K=10 results from local or legacy path."""
    local_path = RESULTS_DIR / "multimodel_pareto_results.json"
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)
    legacy_path = Path(__file__).parent.parent / "03_figure" / "results" / "prequential_results.json"
    if legacy_path.exists():
        with open(legacy_path) as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No results file found at {local_path} or {legacy_path}. "
        "Run run_multimodel_pareto.py first."
    )


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


def plot_figure4(res: dict, out: Path) -> None:
    """Generate the K=10 Pareto frontier figure.

    Args:
        res: Results dict containing ``K10`` and ``metadata`` keys.
        out: Directory to write the output PNG.
    """
    k10 = res["K10"]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    bp = k10["banditgpt_pareto"]
    bg_c = [p["mean_cost"] for p in bp]
    bg_r = [p["mean_reward"] for p in bp]
    bg_std = {(p["mean_cost"], p["mean_reward"]): p["std_reward"] for p in bp}
    tr = k10["tabula_rasa_pareto"]
    tr_c = [p["mean_cost"] for p in tr]
    tr_r = [p["mean_reward"] for p in tr]

    oracle_hull_c, oracle_hull_r = _pareto_hull(bg_c, bg_r)
    ax.fill_between(oracle_hull_c, 0, oracle_hull_r, color=BLUE,
                    alpha=0.06, zorder=1)
    tr_oracle_c, tr_oracle_r = _pareto_hull(tr_c, tr_r)
    ax.fill_between(tr_oracle_c, 0, tr_oracle_r, color=GRAY,
                    alpha=0.06, zorder=1)

    has_dev = "dev_mean_cost" in bp[0] and "dev_mean_reward" in bp[0]
    if has_dev:
        ds_bg_c, ds_bg_r = _dev_selected_deployed_hull(
            bp, "dev_mean_cost", "dev_mean_reward",
            "mean_cost", "mean_reward")
        bg_std_by_holdout = {
            (p["mean_cost"], p["mean_reward"]): p["std_reward"] for p in bp
        }
        hull_err = [
            t_crit * bg_std_by_holdout.get((c, r), 0.0) / np.sqrt(n_seeds)
            for c, r in zip(ds_bg_c, ds_bg_r)
        ]
        ax.errorbar(ds_bg_c, ds_bg_r, yerr=hull_err, fmt="D-", color=BLUE,
                    lw=2.5, ms=5, capsize=3, zorder=5,
                    label="BanditGPT (dev-selected)")
        ds_tr_c, ds_tr_r = _dev_selected_deployed_hull(
            tr, "dev_mean_cost", "dev_mean_reward",
            "mean_cost", "mean_reward")
        ax.plot(ds_tr_c, ds_tr_r, "s:", color=GRAY, lw=1.5, ms=4, zorder=3,
                label="Tabula rasa (dev-selected)")
    else:
        hull_err = [
            t_crit * bg_std.get((c, r), 0.0) / np.sqrt(n_seeds)
            for c, r in zip(oracle_hull_c, oracle_hull_r)
        ]
        ax.errorbar(oracle_hull_c, oracle_hull_r, yerr=hull_err, fmt="D-",
                    color=BLUE, lw=2.5, ms=5, capsize=3, zorder=5,
                    label="BanditGPT (Corralling + warmup)")
        ax.plot(tr_oracle_c, tr_oracle_r, "s:", color=GRAY, lw=1.5, ms=4,
                zorder=3, label="Tabula rasa (no priors, no Corralling)")

    ax.fill_between([], [], [], color=GRAY, alpha=0.1,
                    label="Oracle envelope (upper bound)")

    ax.scatter(k10["oracle"]["cost"], k10["oracle"]["reward"],
               marker="*", s=200, color=GREEN, zorder=6,
               label=f"Oracle ({k10['oracle']['reward']:.3f})")

    bs = k10["best_static"]
    ax.scatter(bs["cost"], bs["reward"], marker="P", s=100, color=ORANGE,
               zorder=6, label=f"Best static ({bs['model'].split('/')[-1]})")

    eg = k10.get("best_static_noisy", k10.get("epsilon_greedy", {}))
    ax.scatter(eg["cost"], eg["reward"], marker="X", s=100, color=PURPLE,
               zorder=6, label=f"Best-static+noise ({eg['best_model'].split('/')[-1]})")

    ucb1 = k10.get("ucb1")
    if ucb1 is not None:
        ax.scatter(ucb1["cost"], ucb1["reward"], marker="d", s=80,
                   color="#E91E63", zorder=6,
                   label=f"UCB1 non-contextual ({ucb1['greedy_arm'].split('/')[-1]})")

    rnd = k10["random"]
    ax.scatter(rnd["cost"], rnd["reward"], marker="o", s=80, color=GRAY,
               zorder=6, alpha=0.7, label="Random")

    ax.set_xlabel("Average Cost per Request ($)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Reward (Quality)", fontsize=12, fontweight="bold")
    ax.set_title("K=10 Cost\u2013Quality Pareto Frontier", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls="--")
    ax.set_xlim(left=-0.0003)
    ax.set_ylim(0.80, None)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.4f}"))
    ax.tick_params(labelsize=9)

    path = out / "figure4_k10.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    res = load_results()
    plot_figure4(res, RESULTS_DIR)
