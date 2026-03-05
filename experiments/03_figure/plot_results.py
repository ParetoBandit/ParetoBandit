#!/usr/bin/env python3
"""
Generate figures and tables comparing BanditGPT against LLMRouter
supervised baselines (KNN, SVM, MLP) and other reference methods.

Outputs from ``prequential_results.json``:

- **Pareto frontier** (``figure3_pareto_k{K}.png``):
  BanditGPT dev-selected Pareto curve with LLMRouter baselines,
  annotated with PerfGain / CostSave.
- **Learning curve** (``figure3_learning_curve.png``):
  BanditGPT holdout reward vs online training steps.
- **Summary table** (``table_summary_k{K}.tex``):
  BaRP-style comparison with PerfGain, CostSave, Gap@Oracle.
- **Isocost table** (``table_isocost_k{K}.tex``):
  BanditGPT vs supervised baselines at matched budgets.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.pareto import (
    pareto_hull,
    dev_pareto_indices,
    interpolate_pareto_reward,
    interpolate_pareto_cost,
)
from utils.metrics import perfgain, costsave, gap_at_oracle

RESULTS_DIR = Path(__file__).parent / "results"

BLUE = "#0072B2"
RED = "#D55E00"
GREEN = "#009E73"
GRAY = "#999999"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
TEAL = "#56B4E9"

SV_MARKERS: Dict[str, Tuple[str, str]] = {
    "knn": ("^", RED),
    "svm": ("s", GREEN),
    "mlp": ("P", PURPLE),
}


def load_results() -> dict:
    with open(RESULTS_DIR / "prequential_results.json") as f:
        return json.load(f)


def _best_supervised_peak(kdata: dict) -> Tuple[str, float]:
    """Return (kind, holdout_reward) for the best supervised baseline."""
    supervised = kdata.get("supervised", {})
    if not supervised:
        return ("none", 0.0)
    best_kind = max(supervised, key=lambda k: supervised[k]["reward"])
    return best_kind, supervised[best_kind]["reward"]


def _get_bg_hull(kdata: dict) -> Tuple[List[float], List[float]]:
    """Build BanditGPT dev-selected deployed Pareto hull from sweep data."""
    bp = kdata["banditgpt_pareto"]
    idx = dev_pareto_indices(bp, "dev_mean_cost", "dev_mean_reward")
    hc = [bp[i]["mean_cost"] for i in idx]
    hr = [bp[i]["mean_reward"] for i in idx]
    return pareto_hull(hc, hr)


# =========================================================================
# Pareto Frontier Plot
# =========================================================================


def plot_pareto_frontier(
    res: dict,
    out: Path,
    k_label: str = "K2",
) -> None:
    """Generate a cost-quality Pareto frontier figure.

    Shows BanditGPT's dev-selected Pareto curve against LLMRouter
    supervised baselines (KNN/SVM/MLP) and reference methods, with
    PerfGain and CostSave annotations.

    Args:
        res: Full results dict.
        out: Output directory for the figure PNG.
        k_label: Which K condition to plot (``"K2"`` or ``"K10"``).
    """
    kdata = res[k_label]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=max(n_seeds - 1, 1))

    bg_hull_c, bg_hull_r = _get_bg_hull(kdata)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.plot(
        bg_hull_c, bg_hull_r, "o-",
        color=BLUE, lw=2.5, ms=5, zorder=6,
        label="BanditGPT (Pareto frontier)",
    )

    # CI band from nearest sweep points
    bp = kdata["banditgpt_pareto"]
    idx = dev_pareto_indices(bp, "dev_mean_cost", "dev_mean_reward")
    hull_points = [bp[i] for i in idx]
    hull_set = set(zip(bg_hull_c, bg_hull_r))
    for hp in hull_points:
        c, r = hp["mean_cost"], hp["mean_reward"]
        if (c, r) not in hull_set:
            continue
        std = hp.get("std_reward", 0)
        hw = t_crit * std / np.sqrt(n_seeds)
        ax.errorbar(
            c, r, yerr=hw, fmt="none",
            ecolor=BLUE, elinewidth=1.2, capsize=3, alpha=0.5, zorder=5,
        )

    # Supervised baselines with PerfGain / CostSave annotations
    supervised = kdata.get("supervised", {})
    comparison = kdata.get("comparison_vs_supervised", {})
    for kind, sv in supervised.items():
        marker, color = SV_MARKERS.get(kind, ("x", GRAY))
        ax.scatter(
            sv["cost"], sv["reward"],
            marker=marker, c=color, s=100, zorder=8, edgecolors="white",
            linewidths=0.5, label=f"{kind.upper()} (LLMRouter)",
        )
        cmp = comparison.get(kind, {})
        pg = cmp.get("perfgain")
        if pg is not None:
            ax.annotate(
                "", xy=(sv["cost"], sv["reward"] + pg),
                xytext=(sv["cost"], sv["reward"]),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.5, ls="--",
                ),
                zorder=7,
            )
            ax.annotate(
                f"PG={pg:+.3f}",
                xy=(sv["cost"], sv["reward"] + pg / 2),
                fontsize=7, color=color, ha="left",
                xytext=(5, 0), textcoords="offset points",
            )

    # Reference baselines
    oracle = kdata.get("oracle_pure_quality", kdata.get("oracle", {}))
    if oracle:
        ax.scatter(
            oracle["cost"], oracle["reward"],
            marker="*", c=ORANGE, s=180, zorder=9,
            edgecolors="white", linewidths=0.5,
            label=f"Oracle ({oracle['reward']:.3f})",
        )

    random_b = kdata.get("random", {})
    if random_b:
        ax.scatter(
            random_b["cost"], random_b["reward"],
            marker="x", c=GRAY, s=80, zorder=8,
            label=f"Random ({random_b['reward']:.3f})",
        )

    ucb1 = kdata.get("ucb1", {})
    if ucb1:
        ax.scatter(
            ucb1["cost"], ucb1["reward"],
            marker="D", c=TEAL, s=60, zorder=8,
            edgecolors="white", linewidths=0.5,
            label=f"UCB1 ({ucb1['reward']:.3f})",
        )

    static = kdata.get("static", {})
    if static:
        best_m = max(static, key=lambda m: static[m]["reward"])
        ax.scatter(
            static[best_m]["cost"], static[best_m]["reward"],
            marker="v", c=GRAY, s=70, zorder=8, alpha=0.7,
            label=f"Best static ({static[best_m]['reward']:.3f})",
        )

    ax.set_xlabel("Normalized Cost ($/request)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Reward (Quality)", fontsize=12, fontweight="bold")
    k_num = k_label.replace("K", "")
    ax.set_title(
        f"BanditGPT vs LLMRouter Baselines — K={k_num}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.15, ls="--")
    ax.tick_params(labelsize=10)

    path = out / f"figure3_pareto_{k_label.lower()}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# =========================================================================
# Learning Curve
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

    sv_kind, sv_peak = _best_supervised_peak(k2)
    sv_lc = k2.get("supervised_learning_curve")

    if sv_lc and sv_lc.get("curve"):
        sv_curve = sv_lc["curve"]
        sv_kind_lc = sv_lc["kind"]
        sv_steps = [d["step"] for d in sv_curve]
        sv_rewards = [d["mean_reward"] for d in sv_curve]
        sv_stds = [d["std_reward"] for d in sv_curve]
        sv_ci_upper = [r + t_crit * s / np.sqrt(n_seeds)
                       for r, s in zip(sv_rewards, sv_stds)]
        sv_ci_lower = [r - t_crit * s / np.sqrt(n_seeds)
                       for r, s in zip(sv_rewards, sv_stds)]
        ax.plot(sv_steps, sv_rewards, "s--", color=RED, lw=2, ms=3.5,
                zorder=4, alpha=0.85,
                label=f"Best supervised ({sv_kind_lc.upper()}, retrained)")
        ax.fill_between(sv_steps, sv_ci_lower, sv_ci_upper, color=RED,
                        alpha=0.08, zorder=1)
        sv_peak = sv_rewards[-1]
    elif sv_peak > 0:
        ax.axhline(y=sv_peak, color=RED, ls="--", lw=2, alpha=0.8, zorder=3,
                   label=f"Best supervised ({sv_kind.upper()}: {sv_peak:.3f})")

    weak_r = min(s["reward"] for s in k2["static"].values())
    ax.axhline(y=weak_r, color=GRAY, ls=":", lw=1.5, alpha=0.6, zorder=3,
               label=f"Weak model static ({weak_r:.3f})")

    crossover_step = None
    crossover_reward = None
    n_pts = len(ci_lower)
    if sv_peak > 0:
        for i in range(n_pts):
            if all(ci_lower[j] >= sv_peak for j in range(i, n_pts)):
                crossover_step = steps[i]
                crossover_reward = rewards[i]
                break

    if crossover_step is not None:
        ax.axvline(x=crossover_step, color=ORANGE, ls=":", lw=1.5, alpha=0.6)
        ax.annotate(
            f"Crossover @ step {crossover_step}",
            xy=(crossover_step, crossover_reward),
            xytext=(crossover_step + 80, crossover_reward + 0.025),
            fontsize=9, color=ORANGE, fontweight="bold",
            ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5),
            zorder=10,
        )

    ax.set_xlabel("Online Learning Steps (dev prompts seen)", fontsize=12,
                  fontweight="bold")
    ax.set_ylabel("Average Reward (Quality)", fontsize=12, fontweight="bold")
    ax.set_title("Online Adaptation vs Supervised Static — K=2",
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
# Summary Table (BaRP-style)
# =========================================================================


def _fmt_val(v: Optional[float], fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "---"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:+.1f}\\%" if v is not None else "---"


def _fmt_pct_md(v: Optional[float]) -> str:
    return f"{v:+.1f}%" if v is not None else "---"


def generate_summary_table(
    res: dict,
    out: Path,
    k_label: str = "K2",
) -> None:
    """Generate a BaRP-style summary comparison table.

    Columns: Method | Reward | Cost | PerfGain | CostSave(%) | Gap@Oracle

    Output as both LaTeX and markdown.

    Args:
        res: Full results dict.
        out: Output directory for the table files.
        k_label: Which K condition (``"K2"`` or ``"K10"``).
    """
    kdata = res[k_label]
    oracle = kdata.get("oracle_pure_quality", kdata.get("oracle", {}))
    oracle_r = oracle["reward"]
    oracle_c = oracle["cost"]

    bg_hull_c, bg_hull_r = _get_bg_hull(kdata)
    comparison = kdata.get("comparison_vs_supervised", {})
    supervised = kdata.get("supervised", {})
    gap_bg_info = kdata.get("gap_at_oracle_banditgpt", {})

    bg_dev_best_r = gap_bg_info.get("banditgpt_dev_best_reward")
    bg_gap_abs = gap_bg_info.get("abs")
    bg_gap_pct = gap_bg_info.get("pct")

    bg_idx = dev_pareto_indices(
        kdata["banditgpt_pareto"], "dev_mean_cost", "dev_mean_reward",
    )
    bg_dev_best = max(
        [kdata["banditgpt_pareto"][i] for i in bg_idx],
        key=lambda p: p["dev_mean_reward"],
    )
    bg_cost = bg_dev_best["mean_cost"]

    rows: List[Dict[str, Any]] = []

    rows.append({
        "method": "Oracle",
        "reward": oracle_r,
        "cost": oracle_c,
        "perfgain": None,
        "costsave_pct": None,
        "gap_oracle": 0.0,
    })

    rows.append({
        "method": "BanditGPT (dev-opt)",
        "reward": bg_dev_best_r,
        "cost": bg_cost,
        "perfgain": None,
        "costsave_pct": None,
        "gap_oracle": bg_gap_abs,
    })

    lam0 = kdata.get("banditgpt_lambda0")
    if lam0:
        lam0_gap_abs, _ = gap_at_oracle(oracle_r, lam0["reward"])
        rows.append({
            "method": r"BanditGPT ($\lambda{=}0$)",
            "reward": lam0["reward"],
            "std_reward": lam0.get("std_reward", 0.0),
            "cost": lam0["cost"],
            "perfgain": None,
            "costsave_pct": None,
            "gap_oracle": lam0_gap_abs,
        })

    for kind in ("knn", "svm", "mlp"):
        if kind not in supervised:
            continue
        sv = supervised[kind]
        cmp = comparison.get(kind, {})
        pg = cmp.get("perfgain")
        cs_pct = cmp.get("costsave_pct")
        gap_sv_abs = cmp.get("gap_oracle_supervised_abs")
        label = f"{kind.upper()} (LLMRouter)"
        if kind == "knn":
            label += r"$^\dagger$"
        rows.append({
            "method": label,
            "reward": sv["reward"],
            "std_reward": sv.get("std_reward", 0.0),
            "cost": sv["cost"],
            "perfgain": pg,
            "costsave_pct": cs_pct,
            "gap_oracle": gap_sv_abs,
        })

    ucb1 = kdata.get("ucb1", {})
    if ucb1:
        ucb1_gap, _ = gap_at_oracle(oracle_r, ucb1["reward"])
        rows.append({
            "method": "UCB1",
            "reward": ucb1["reward"],
            "cost": ucb1["cost"],
            "perfgain": None,
            "costsave_pct": None,
            "gap_oracle": ucb1_gap,
        })

    random_b = kdata.get("random", {})
    if random_b:
        rand_gap, _ = gap_at_oracle(oracle_r, random_b["reward"])
        rows.append({
            "method": "Random",
            "reward": random_b["reward"],
            "cost": random_b["cost"],
            "perfgain": None,
            "costsave_pct": None,
            "gap_oracle": rand_gap,
        })

    # --- LaTeX output ---
    k_num = k_label.replace("K", "")
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{BanditGPT vs peer-reviewed baselines ($K{=}" + k_num + r"$). "
        r"PerfGain and CostSave are measured relative to each LLMRouter "
        r"supervised baseline. Gap@Oracle = oracle reward $-$ method reward. "
        r"$^\dagger$KNN is deterministic; reported $\pm$0 reflects zero "
        r"initialization variance, not zero generalization variance.}"
    )
    lines.append(r"\label{tab:summary_" + k_label.lower() + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Reward & Cost (\$/req) & PerfGain & CostSave (\%) "
        r"& Gap@Oracle \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        method = row["method"]
        r_val = _fmt_val(row["reward"], ".4f")
        std_r = row.get("std_reward")
        r_str = f"{r_val}$\\pm${std_r:.4f}" if std_r is not None else r_val
        c_str = _fmt_val(row["cost"], ".6f")
        pg_str = _fmt_val(row["perfgain"], "+.4f") if row["perfgain"] is not None else "---"
        cs_str = _fmt_pct(row["costsave_pct"]) if row["costsave_pct"] is not None else "---"
        gap_str = _fmt_val(row["gap_oracle"], ".4f") if row["gap_oracle"] is not None else "---"
        lines.append(f"{method} & {r_str} & {c_str} & {pg_str} & {cs_str} & {gap_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = out / f"table_summary_{k_label.lower()}.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {tex_path}")

    # --- Markdown output ---
    md_lines: List[str] = []
    md_lines.append(f"\n--- Summary Table {k_label} (markdown) ---")
    md_lines.append(
        "| Method | Reward | Cost ($/req) | PerfGain | CostSave (%) | Gap@Oracle |"
    )
    md_lines.append("|---|---|---|---|---|---|")
    for row in rows:
        r_val = _fmt_val(row["reward"], ".4f")
        std_r = row.get("std_reward")
        r_str = f"{r_val}+/-{std_r:.4f}" if std_r is not None else r_val
        c_str = _fmt_val(row["cost"], ".6f")
        pg_str = _fmt_val(row["perfgain"], "+.4f") if row["perfgain"] is not None else "---"
        cs_str = _fmt_pct_md(row["costsave_pct"]) if row["costsave_pct"] is not None else "---"
        gap_str = _fmt_val(row["gap_oracle"], ".4f") if row["gap_oracle"] is not None else "---"
        md_method = row["method"].replace(r"$^\dagger$", "*")
        md_lines.append(
            f"| {md_method} | {r_str} | {c_str} | {pg_str} | {cs_str} | {gap_str} |"
        )
    md_lines.append("")
    md_lines.append(
        "*KNN is deterministic; +/-0 reflects zero initialization variance, "
        "not zero generalization variance."
    )
    print("\n".join(md_lines))


# =========================================================================
# Isocost / Iso-quality Table
# =========================================================================

ISOCOST_BUDGETS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.011]


def _build_isocost_table_data(
    res: dict,
    k_label: str = "K2",
    budgets: Optional[List[float]] = None,
) -> List[Dict]:
    """Build isocost table rows comparing BanditGPT vs supervised baselines.

    For each target budget, interpolates reward on BanditGPT's
    dev-selected Pareto hull and reports the supervised baselines'
    rewards if they happen to operate near that budget.

    Args:
        res: Full results dict.
        k_label: Which K condition.
        budgets: Target cost budgets.  Defaults to :data:`ISOCOST_BUDGETS`.

    Returns:
        List of row dicts for table rendering.
    """
    if budgets is None:
        budgets = ISOCOST_BUDGETS
    kdata = res[k_label]
    n_seeds = res["metadata"]["n_seeds"]
    t_crit = sp_stats.t.ppf(0.975, df=max(n_seeds - 1, 1))

    bg_hull_c, bg_hull_r = _get_bg_hull(kdata)

    bp = kdata["banditgpt_pareto"]
    idx = dev_pareto_indices(bp, "dev_mean_cost", "dev_mean_reward")
    hull_set = set(zip(bg_hull_c, bg_hull_r))
    bg_hull_points = [bp[i] for i in idx if (bp[i]["mean_cost"], bp[i]["mean_reward"]) in hull_set]

    supervised = kdata.get("supervised", {})

    rows: List[Dict] = []
    for budget in budgets:
        bg_r = interpolate_pareto_reward(bg_hull_c, bg_hull_r, budget)

        bg_std = None
        if bg_hull_points:
            closest = min(
                bg_hull_points, key=lambda p: abs(p["mean_cost"] - budget),
            )
            bg_std = closest.get("std_reward")

        bg_ci_hw = (
            t_crit * bg_std / np.sqrt(n_seeds)
            if bg_std is not None else None
        )

        row: Dict[str, Any] = {
            "budget": budget,
            "bg_reward": bg_r,
            "bg_ci_hw": bg_ci_hw,
        }

        for kind in ("knn", "svm", "mlp"):
            sv = supervised.get(kind, {})
            sv_r = sv.get("reward")
            row[f"{kind}_reward"] = sv_r
            if bg_r is not None and sv_r is not None:
                row[f"pg_{kind}"] = perfgain(bg_r, sv_r)
            else:
                row[f"pg_{kind}"] = None

        rows.append(row)

    return rows


def generate_isocost_table(
    res: dict,
    out: Path,
    k_label: str = "K2",
) -> None:
    """Write an isocost comparison LaTeX/markdown table.

    Compares BanditGPT (interpolated on Pareto hull) vs each
    LLMRouter supervised baseline at matched budgets.

    Args:
        res: Full results dict.
        out: Output directory.
        k_label: Which K condition.
    """
    rows = _build_isocost_table_data(res, k_label)
    kdata = res[k_label]

    static = kdata.get("static", {})
    models = kdata.get("models", [])
    oracle = kdata.get("oracle_pure_quality", kdata.get("oracle", {}))
    oracle_r = oracle.get("reward", 0)
    supervised = kdata.get("supervised", {})

    k_num = k_label.replace("K", "")

    # LaTeX
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    n_holdout = kdata.get("n_holdout", "?")
    lines.append(
        r"\caption{Isocost comparison: BanditGPT vs LLMRouter supervised "
        r"baselines ($K{=}" + k_num + r"$, $n{=}" + str(n_holdout) + r"$). "
        r"BanditGPT rewards interpolated on dev-selected Pareto hull. "
        r"PG = PerfGain at each budget level.}"
    )
    lines.append(r"\label{tab:isocost_" + k_label.lower() + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Budget (\$/req) & BanditGPT & KNN & SVM & MLP "
        r"& PG$_{\mathrm{KNN}}$ & PG$_{\mathrm{SVM}}$ "
        r"& PG$_{\mathrm{MLP}}$ \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        b = row["budget"]
        budget_str = f"\\${b:.3f}"

        if row["bg_reward"] is not None:
            hw = row.get("bg_ci_hw")
            bg_str = (
                f"{row['bg_reward']:.3f}$\\pm${hw:.3f}"
                if hw is not None else f"{row['bg_reward']:.3f}"
            )
        else:
            bg_str = "---"

        cols = [budget_str, bg_str]
        for kind in ("knn", "svm", "mlp"):
            sv_r = row.get(f"{kind}_reward")
            cols.append(f"{sv_r:.3f}" if sv_r is not None else "---")
        for kind in ("knn", "svm", "mlp"):
            pg = row.get(f"pg_{kind}")
            cols.append(f"{pg:+.3f}" if pg is not None else "---")

        lines.append(" & ".join(cols) + " \\\\")

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{8}{l}{\textit{Reference baselines}} \\"
    )
    if oracle_r:
        lines.append(f"Oracle & {oracle_r:.3f} & --- & --- & --- & --- & --- & --- \\\\")
    for kind in ("knn", "svm", "mlp"):
        sv = supervised.get(kind, {})
        if sv:
            lines.append(
                f"{kind.upper()} (all data) & --- & {sv['reward']:.3f} "
                f"& --- & --- & --- & --- & --- \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = out / f"table_isocost_{k_label.lower()}.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")

    # Markdown preview
    print(f"\n--- Isocost Table {k_label} (markdown) ---")
    print(
        "| Budget ($/req) | BanditGPT | KNN | SVM | MLP | PG_KNN | PG_SVM | PG_MLP |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        b = row["budget"]
        bg = f"{row['bg_reward']:.3f}" if row["bg_reward"] is not None else "---"
        cols = [f"${b:.3f}", bg]
        for kind in ("knn", "svm", "mlp"):
            sv_r = row.get(f"{kind}_reward")
            cols.append(f"{sv_r:.3f}" if sv_r is not None else "---")
        for kind in ("knn", "svm", "mlp"):
            pg = row.get(f"pg_{kind}")
            cols.append(f"{pg:+.3f}" if pg is not None else "---")
        print("| " + " | ".join(cols) + " |")


# =========================================================================
# Iso-quality Table (CostSave)
# =========================================================================


def generate_isoquality_table(
    res: dict,
    out: Path,
    k_label: str = "K2",
) -> None:
    """Write an iso-quality comparison table (CostSave at each baseline's reward).

    For each supervised baseline, reports BanditGPT's interpolated cost
    at the baseline's reward level and the resulting CostSave.

    Args:
        res: Full results dict.
        out: Output directory.
        k_label: Which K condition.
    """
    kdata = res[k_label]
    bg_hull_c, bg_hull_r = _get_bg_hull(kdata)
    supervised = kdata.get("supervised", {})
    comparison = kdata.get("comparison_vs_supervised", {})

    k_num = k_label.replace("K", "")

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Iso-quality comparison: BanditGPT cost at each "
        r"LLMRouter baseline's reward level ($K{=}" + k_num + r"$). "
        r"Positive CostSave means BanditGPT is cheaper.}"
    )
    lines.append(r"\label{tab:isoquality_" + k_label.lower() + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Baseline & Baseline Cost & BanditGPT Cost & "
        r"CostSave (\$/req) & CostSave (\%) \\"
    )
    lines.append(r"\midrule")

    md_lines: List[str] = []
    md_lines.append(f"\n--- Iso-quality Table {k_label} (markdown) ---")
    md_lines.append(
        "| Baseline | Baseline Cost | BanditGPT Cost | CostSave ($/req) | CostSave (%) |"
    )
    md_lines.append("|---|---|---|---|---|")

    for kind in ("knn", "svm", "mlp"):
        sv = supervised.get(kind)
        if sv is None:
            continue
        cmp = comparison.get(kind, {})
        bg_cost = cmp.get("banditgpt_cost_at_sv_reward")
        cs_abs = cmp.get("costsave_abs")
        cs_pct = cmp.get("costsave_pct")

        sv_c_str = f"\\${sv['cost']:.6f}"
        bg_c_str = f"\\${bg_cost:.6f}" if bg_cost is not None else "---"
        cs_abs_str = f"{cs_abs:+.6f}" if cs_abs is not None else "---"
        cs_pct_str = _fmt_pct(cs_pct) if cs_pct is not None else "---"
        lines.append(
            f"{kind.upper()} & {sv_c_str} & {bg_c_str} & "
            f"{cs_abs_str} & {cs_pct_str} \\\\"
        )

        sv_c_md = f"${sv['cost']:.6f}"
        bg_c_md = f"${bg_cost:.6f}" if bg_cost is not None else "---"
        cs_abs_md = f"{cs_abs:+.6f}" if cs_abs is not None else "---"
        cs_pct_md = _fmt_pct_md(cs_pct) if cs_pct is not None else "---"
        md_lines.append(
            f"| {kind.upper()} | {sv_c_md} | {bg_c_md} | {cs_abs_md} | {cs_pct_md} |"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = out / f"table_isoquality_{k_label.lower()}.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved {path}")
    print("\n".join(md_lines))


# =========================================================================
# Entry point
# =========================================================================


if __name__ == "__main__":
    res = load_results()

    for k_label in ("K2", "K10"):
        if k_label in res:
            plot_pareto_frontier(res, RESULTS_DIR, k_label)
            generate_summary_table(res, RESULTS_DIR, k_label)
            generate_isocost_table(res, RESULTS_DIR, k_label)
            generate_isoquality_table(res, RESULTS_DIR, k_label)

    plot_learning_curve(res, RESULTS_DIR)
