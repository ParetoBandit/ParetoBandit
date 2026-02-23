#!/usr/bin/env python3
"""
Figure 6: Multi-Model Cost-Quality Pareto Frontier (K=5 and K=10)
=================================================================

Four-panel publication figure:
  (a) K=5 Pareto frontier (cost vs quality)
  (b) K=10 Pareto frontier (cost vs quality)
  (c) Learning curve (reward vs training steps) for K=5 and K=10
  (d) Traffic allocation heatmap (model × K)

Reads: results/multimodel_pareto_results.json
Saves: results/figure6_multimodel_pareto.pdf and .png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "multimodel_pareto_results.json"

# Visual style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
})

BANDIT_COLOR = "#2196F3"
TABULA_COLOR = "#FF9800"
STATIC_COLOR = "#4CAF50"
ORACLE_COLOR = "#E91E63"
RANDOM_COLOR = "#9E9E9E"
EPSGREEDY_COLOR = "#9C27B0"


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _static_pareto_ids(data):
    """Return set of model IDs on the static cost-reward Pareto frontier."""
    items = [(m_id, s["cost"], s["reward"]) for m_id, s in data["static"].items()]
    items.sort(key=lambda x: x[1])  # sort by cost ascending
    frontier = []
    best_reward = -1
    for m_id, cost, reward in items:
        if reward > best_reward:
            frontier.append(m_id)
            best_reward = reward
    return set(frontier)


def plot_pareto_panel(ax, data, title):
    """Plot a single Pareto frontier panel (cost vs quality)."""
    pareto_ids = _static_pareto_ids(data)
    static_items = list(data["static"].items())
    for m_id, s in static_items:
        cat = next(mc for mc in data["models"] if mc["id"] == m_id)
        on_frontier = m_id in pareto_ids
        ax.scatter(
            s["cost"] * 1000, s["reward"],
            marker="^",
            s=50 if on_frontier else 22,
            color=STATIC_COLOR,
            alpha=1.0 if on_frontier else 0.35,
            zorder=5,
            edgecolors="white", linewidth=0.5,
        )
        if on_frontier:
            offset = (5, -8) if s["reward"] > 0.96 else (5, 4)
            ax.annotate(
                cat["display"], (s["cost"] * 1000, s["reward"]),
                textcoords="offset points", xytext=offset,
                fontsize=5.5, color=STATIC_COLOR, alpha=0.8,
            )

    # Oracle
    ax.axhline(data["oracle"]["reward"], color=ORACLE_COLOR, ls="--",
               lw=0.8, alpha=0.6, label="Oracle")

    # Random
    ax.scatter(
        data["random"]["cost"] * 1000, data["random"]["reward"],
        marker="x", s=50, color=RANDOM_COLOR, zorder=5, label="Random",
    )

    # ε-Greedy
    ax.scatter(
        data["epsilon_greedy"]["cost"] * 1000, data["epsilon_greedy"]["reward"],
        marker="D", s=40, color=EPSGREEDY_COLOR, zorder=5,
        edgecolors="white", linewidth=0.5, label="ε-Greedy",
    )

    # No-Corralling ablation (warmup only, no meta-learner)
    pareto_nc = data["pareto_tabula_rasa"]
    nc_costs = [p["mean_cost"] * 1000 for p in pareto_nc]
    nc_rewards = [p["mean_reward"] for p in pareto_nc]
    ax.plot(nc_costs, nc_rewards, "s--", color=TABULA_COLOR, lw=1.0, ms=3,
            alpha=0.7, label="No Corralling", zorder=8)

    # banditGPT Pareto curve (full system)
    pareto = data["pareto_banditgpt"]
    costs = [p["mean_cost"] * 1000 for p in pareto]
    rewards = [p["mean_reward"] for p in pareto]
    stds = [p["std_reward"] for p in pareto]
    ax.plot(costs, rewards, "o-", color=BANDIT_COLOR, lw=1.8, ms=5,
            label="banditGPT (Full)", zorder=10)
    ax.fill_between(
        costs,
        [r - s for r, s in zip(rewards, stds)],
        [r + s for r, s in zip(rewards, stds)],
        alpha=0.15, color=BANDIT_COLOR,
    )

    # Annotate select λ values (only non-overlapping)
    annotated_lambdas = {0.0: (6, 8), 0.5: (-20, -12), 5.0: (-8, 8)}
    for p in pareto:
        if p["lambda"] in annotated_lambdas:
            ox, oy = annotated_lambdas[p["lambda"]]
            ax.annotate(
                f'λ={p["lambda"]}',
                (p["mean_cost"] * 1000, p["mean_reward"]),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=5.5, color=BANDIT_COLOR, alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Cost per request ($×10⁻³)")
    ax.set_ylabel("Mean Reward (Holdout)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.2)


def plot_learning_curve(ax, data):
    """Overlay K=5 and K=10 learning curves."""
    for k_name, color, marker in [("K5", BANDIT_COLOR, "o"), ("K10", TABULA_COLOR, "s")]:
        d = data[k_name]
        curve = d["learning_curve"]
        steps = [c["step"] for c in curve]
        rewards = [c["mean_reward"] for c in curve]
        stds = [c["std_reward"] for c in curve]
        ax.plot(steps, rewards, f"{marker}-", color=color, lw=1.5, ms=4,
                label=f'K={d["K"]}')
        ax.fill_between(
            steps,
            [r - s for r, s in zip(rewards, stds)],
            [r + s for r, s in zip(rewards, stds)],
            alpha=0.15, color=color,
        )
        # Best-static horizontal line
        bsr = d["best_static"]["reward"]
        ax.axhline(bsr, color=color, ls=":", lw=0.8, alpha=0.5)
        ax.annotate(
            f'Best static ({d["best_static"]["model"].split("/")[-1]})',
            (steps[-1], bsr), fontsize=6, color=color, alpha=0.6,
            ha="right", va="bottom",
        )

    ax.axhline(1.0, color=ORACLE_COLOR, ls="--", lw=0.8, alpha=0.5,
               label="Oracle")
    ax.set_xlabel("Online Learning Steps")
    ax.set_ylabel("Mean Reward (Holdout)")
    ax.set_title("(c) Convergence: K=5 vs K=10", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)


def plot_traffic_heatmap(ax, data):
    """Traffic allocation heatmap for K=5 and K=10."""
    k10_models = [m["id"] for m in data["K10"]["models"]]
    display_names = [data["K10"]["models"][i]["display"] for i in range(len(k10_models))]

    matrix = np.zeros((2, len(k10_models)))
    for col_idx, m_id in enumerate(k10_models):
        if m_id in data["K5"]["traffic_allocation"]:
            matrix[0, col_idx] = data["K5"]["traffic_allocation"][m_id]["mean_frac"] * 100
        if m_id in data["K10"]["traffic_allocation"]:
            matrix[1, col_idx] = data["K10"]["traffic_allocation"][m_id]["mean_frac"] * 100

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(k10_models)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["K=5", "K=10"])
    ax.set_title("(d) Traffic Allocation (%)", fontweight="bold")

    for i in range(2):
        for j in range(len(k10_models)):
            val = matrix[i, j]
            if val > 0.5:
                color = "white" if val > 20 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
            elif i == 0 and k10_models[j] not in [m["id"] for m in data["K5"]["models"]]:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color="#999")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Traffic %")


def main():
    data = load_results()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.28,
                           left=0.07, right=0.95, top=0.94, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_pareto_panel(ax_a, data["K5"], f'(a) Pareto Frontier: K=5')
    plot_pareto_panel(ax_b, data["K10"], f'(b) Pareto Frontier: K=10')
    plot_learning_curve(ax_c, data)
    plot_traffic_heatmap(ax_d, data)

    fig.suptitle(
        "Figure 6: Multi-Model Routing — banditGPT at K=5 and K=10",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ext in ("pdf", "png"):
        path = RESULTS_DIR / f"figure6_multimodel_pareto.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    # Print summary statistics
    for k_name in ["K5", "K10"]:
        d = data[k_name]
        print(f"\n{k_name} Summary:")
        print(f"  Oracle:         {d['oracle']['reward']:.4f}")
        print(f"  banditGPT peak: {d['peak_bandit_reward']:.4f}")
        print(f"  Best static:    {d['best_static']['reward']:.4f} "
              f"({d['best_static']['model'].split('/')[-1]})")
        print(f"  Gap closure:    {d['gap_closure_pct']:.1f}%")
        print(f"  Warmup (step=0):{d['learning_curve'][0]['mean_reward']:.4f}")


if __name__ == "__main__":
    main()
