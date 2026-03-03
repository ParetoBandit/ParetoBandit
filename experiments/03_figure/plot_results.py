#!/usr/bin/env python3
"""
Generate Figure 3 (learning curve) and Table 1 (isocost) from
prequential_results.json.

Figure 3 — standalone learning curve:
    BanditGPT holdout reward vs online training steps, with RouteLLM
    peak as reference.

Table 1 — interpolated isocost comparison:
    BanditGPT vs RouteLLM at matched budgets, using Pareto hull
    interpolation (sweep-density invariant).

Figure 4 (K=10) lives in ``experiments/04_figure/plot_results.py``.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _pareto_hull(
    costs: List[float], rewards: List[float],
) -> Tuple[List[float], List[float]]:
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


def _dev_pareto_indices(
    sweep: List[Dict], dev_cost_key: str, dev_reward_key: str,
) -> List[int]:
    """Identify indices on the dev-set Pareto hull (no holdout leakage)."""
    n = len(sweep)
    pairs = [(sweep[i][dev_cost_key], sweep[i][dev_reward_key], i)
             for i in range(n)]
    pairs.sort(key=lambda x: (x[0], -x[1]))
    idx: List[int] = []
    best_r = -np.inf
    for _, r, i in pairs:
        if r > best_r:
            idx.append(i)
            best_r = r
    return idx


def _interpolate(
    hull_c: List[float], hull_r: List[float], target: float,
) -> Optional[float]:
    """Linearly interpolate on the Pareto hull; None if out of range."""
    if not hull_c or target < hull_c[0] or target > hull_c[-1]:
        return None
    return float(np.interp(target, hull_c, hull_r))


# =========================================================================
# Figure 3: Standalone Learning Curve
# =========================================================================


def plot_learning_curve(res: dict, out: Path) -> None:
    """Generate a standalone learning curve figure (formerly panel b)."""
    k2 = res["K2"]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)

    lc = k2["learning_curve"]
    steps = [d["step"] for d in lc]
    rewards = [d["mean_reward"] for d in lc]
    stds = [d["std_reward"] for d in lc]

    ci_upper = [r + t_crit * s / np.sqrt(n_seeds)
                for r, s in zip(rewards, stds)]
    ci_lower = [r - t_crit * s / np.sqrt(n_seeds)
                for r, s in zip(rewards, stds)]

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax.plot(steps, rewards, "D-", color=BLUE, lw=2.5, ms=4, zorder=5,
            label=f"BanditGPT (online, n={n_seeds} seeds)")
    ax.fill_between(steps, ci_lower, ci_upper, color=BLUE,
                    alpha=0.12, zorder=2)

    rl_peak = max(p["avg_reward"] for p in k2["routellm"]["pareto"])
    ax.axhline(y=rl_peak, color=RED, ls="--", lw=2, alpha=0.8, zorder=3,
               label=f"RouteLLM peak ({rl_peak:.3f}, ~100k pre-trained)")

    weak_r = min(s["reward"] for s in k2["static"].values())
    ax.axhline(y=weak_r, color=GRAY, ls=":", lw=1.5, alpha=0.6, zorder=3,
               label=f"Weak model static ({weak_r:.3f})")

    crossover_step = None
    crossover_reward = None
    n_pts = len(ci_lower)
    for i in range(n_pts):
        if all(ci_lower[j] >= rl_peak for j in range(i, n_pts)):
            crossover_step = steps[i]
            crossover_reward = rewards[i]
            break

    if crossover_step is not None:
        ax.axvline(x=crossover_step, color=ORANGE, ls=":", lw=1.5, alpha=0.6)
        ax.annotate(
            f"Crossover @ step {crossover_step}",
            xy=(crossover_step, crossover_reward),
            xytext=(crossover_step + 80, crossover_reward - 0.015),
            fontsize=9, color=ORANGE, fontweight="bold",
            ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
            zorder=10,
        )

    ax.set_xlabel("Online Learning Steps (dev prompts seen)", fontsize=12,
                  fontweight="bold")
    ax.set_ylabel("Average Reward (Quality)", fontsize=12, fontweight="bold")
    ax.set_title("Online Adaptation vs Static Pre-Training — K=2",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls="--")
    ax.set_xlim(-30, max(steps) + 50)
    ax.set_ylim(0.70, 0.83)
    ax.tick_params(labelsize=10)

    path = out / "figure3_learning_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Table: Interpolated Isocost Comparison
# =========================================================================

ISOCOST_BUDGETS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.011]


def _build_isocost_table_data(
    res: dict,
    budgets: List[float] = ISOCOST_BUDGETS,
) -> List[Dict]:
    """Build table rows from the results JSON.

    For each target budget, interpolates reward on the dev-selected
    Pareto hull of both BanditGPT and RouteLLM.  Returns a list of
    row dicts suitable for rendering as LaTeX or markdown.
    """
    k2 = res["K2"]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=n_seeds - 1)

    bp = k2["banditgpt_pareto"]
    rl = k2["routellm"]["pareto"]

    bg_idx = _dev_pareto_indices(bp, "dev_mean_cost", "dev_mean_reward")
    bg_hc_raw = [bp[i]["mean_cost"] for i in bg_idx]
    bg_hr_raw = [bp[i]["mean_reward"] for i in bg_idx]
    bg_hull_c, bg_hull_r = _pareto_hull(bg_hc_raw, bg_hr_raw)

    rl_idx = _dev_pareto_indices(rl, "dev_mean_cost", "dev_mean_reward")
    rl_hc_raw = [rl[i]["avg_cost"] for i in rl_idx]
    rl_hr_raw = [rl[i]["avg_reward"] for i in rl_idx]
    rl_hull_c, rl_hull_r = _pareto_hull(rl_hc_raw, rl_hr_raw)

    # Per-seed std for BanditGPT (closest hull point)
    bg_hull_set = set(bg_hull_c)
    bg_hull_points = [bp[i] for i in bg_idx
                      if bp[i]["mean_cost"] in bg_hull_set]

    rows: List[Dict] = []
    for budget in budgets:
        bg_r = _interpolate(bg_hull_c, bg_hull_r, budget)
        rl_r = _interpolate(rl_hull_c, rl_hull_r, budget)

        bg_std = None
        if bg_hull_points:
            closest = min(bg_hull_points,
                          key=lambda p: abs(p["mean_cost"] - budget))
            bg_std = closest["std_reward"]

        bg_ci_hw = (t_crit * bg_std / np.sqrt(n_seeds)
                    if bg_std is not None else None)

        rows.append({
            "budget": budget,
            "bg_reward": bg_r,
            "bg_ci_hw": bg_ci_hw,
            "rl_reward": rl_r,
            "delta": (bg_r - rl_r) if bg_r is not None and rl_r is not None else None,
        })

    return rows


def generate_latex_table(res: dict, out: Path) -> None:
    """Write an isocost comparison LaTeX table to disk."""
    rows = _build_isocost_table_data(res)
    k2 = res["K2"]

    # Static baselines
    static = k2["static"]
    mixtral_r = static["mistralai/mixtral-8x7b-instruct"]["reward"]
    mixtral_c = static["mistralai/mixtral-8x7b-instruct"]["cost"]
    gpt4_r = static["openai/gpt-4-turbo"]["reward"]
    gpt4_c = static["openai/gpt-4-turbo"]["cost"]
    oracle_r = k2["oracle_pure_quality"]["reward"]
    rl_peak = max(p["avg_reward"] for p in k2["routellm"]["pareto"])

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Interpolated isocost comparison: BanditGPT vs "
                 r"RouteLLM-MF on the $K{=}2$ holdout set ($n{=}750$). "
                 r"Rewards are read off each method's dev-selected Pareto hull "
                 r"at exact target costs via linear interpolation, "
                 r"eliminating sensitivity to sweep density "
                 r"(24 BanditGPT $\lambda$ values vs 101 RouteLLM thresholds). "
                 r"``---'' indicates the method's dev-selected frontier does not "
                 r"extend to that budget. "
                 r"95\% CIs from the nearest on-hull BanditGPT sweep point "
                 r"(20 seeds).}")
    lines.append(r"\label{tab:isocost_k2}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Budget (\$/req) & BanditGPT & RouteLLM-MF & "
                 r"$\Delta$ & Winner \\")
    lines.append(r"\midrule")

    for row in rows:
        b = row["budget"]
        budget_str = f"\\${b:.3f}"

        if row["bg_reward"] is not None:
            hw = row["bg_ci_hw"]
            if hw is not None:
                bg_str = f"{row['bg_reward']:.3f}$\\pm${hw:.3f}"
            else:
                bg_str = f"{row['bg_reward']:.3f}"
        else:
            bg_str = "---"

        rl_str = f"{row['rl_reward']:.3f}" if row["rl_reward"] is not None else "---"

        if row["delta"] is not None:
            d = row["delta"]
            delta_str = f"{d:+.3f}"
            if d > 0.005:
                winner = r"\textbf{BanditGPT}"
            elif d < -0.005:
                winner = r"\textbf{RouteLLM}"
            else:
                winner = "Tie"
        elif row["bg_reward"] is not None and row["rl_reward"] is None:
            delta_str = "---"
            winner = r"BanditGPT only"
        else:
            delta_str = "---"
            winner = "---"

        lines.append(f"{budget_str} & {bg_str} & {rl_str} & "
                     f"{delta_str} & {winner} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Reference baselines}} \\")
    lines.append(f"Static Mixtral (\\${mixtral_c:.4f}) & "
                 f"{mixtral_r:.3f} & --- & --- & --- \\\\")
    lines.append(f"Static GPT-4-Turbo (\\${gpt4_c:.3f}) & "
                 f"{gpt4_r:.3f} & --- & --- & --- \\\\")
    lines.append(f"RouteLLM peak & --- & {rl_peak:.3f} & --- & --- \\\\")
    lines.append(f"Oracle (per-prompt best) & {oracle_r:.3f} & "
                 f"--- & --- & --- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = out / "table_isocost_k2.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")

    # Also print a readable markdown version
    print("\n--- Isocost Table (markdown preview) ---")
    print(f"| Budget ($/req) | BanditGPT | RouteLLM-MF | Delta | Winner |")
    print(f"|---|---|---|---|---|")
    for row in rows:
        b = row["budget"]
        bg = f"{row['bg_reward']:.3f}" if row["bg_reward"] is not None else "---"
        rl = f"{row['rl_reward']:.3f}" if row["rl_reward"] is not None else "---"
        d = f"{row['delta']:+.3f}" if row["delta"] is not None else "---"
        if row["delta"] is not None:
            w = "BanditGPT" if row["delta"] > 0.005 else ("RouteLLM" if row["delta"] < -0.005 else "Tie")
        elif row["bg_reward"] is not None:
            w = "BanditGPT only"
        else:
            w = "---"
        print(f"| ${b:.3f} | {bg} | {rl} | {d} | {w} |")
    print(f"| Static Mixtral | {mixtral_r:.3f} | --- | --- | --- |")
    print(f"| Static GPT-4-Turbo | {gpt4_r:.3f} | --- | --- | --- |")
    print(f"| RouteLLM peak | --- | {rl_peak:.3f} | --- | --- |")
    print(f"| Oracle | {oracle_r:.3f} | --- | --- | --- |")


if __name__ == "__main__":
    res = load_results()
    plot_learning_curve(res, RESULTS_DIR)
    generate_latex_table(res, RESULTS_DIR)
