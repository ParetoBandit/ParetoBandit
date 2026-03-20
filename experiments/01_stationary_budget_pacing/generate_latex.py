"""Generate LaTeX commands and tables from budget pacing experiment results.

Reads results/budget_pacing_results.json and emits:
- _autogen.tex: \\newcommand definitions for the paper
- table_budget_compliance.tex: budget compliance diagnostics table
- table_routing_metrics.tex: standard LLM routing evaluation metrics

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_cost_dollar,
    fmt_cost_sci,
    fmt_num,
    fmt_pct,
    fmt_ratio,
    fmt_reward,
)
from utils.pareto import (
    interpolate_pareto_cost,
    interpolate_pareto_reward,
    pareto_aucpc_normalized,
    pareto_hull,
)


# -----------------------------------------------------------------------------
# Index names for pacer targets (0 = tightest, 6 = loosest)
# -----------------------------------------------------------------------------

PACER_INDEX_NAMES = [
    "Tightest",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Loosest",
]

REQUESTS_PER_DAY = 100_000
BINDING_UTIL_LOW = 0.95
BINDING_UTIL_HIGH = 1.05

QUALITY_FRACTIONS = {"90": 0.90, "95": 0.95}
COST_FRACTIONS = {"25": 0.25, "50": 0.50}


def load_results(json_path: Path) -> Dict[str, Any]:
    """Load budget pacing results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def get_static_by_penalty(
    results: List[Dict[str, Any]], cost_penalty: float
) -> Optional[Dict[str, Any]]:
    """Return the static result row for the given cost penalty."""
    for r in results:
        if r.get("method") == "static" and r.get("cost_penalty") == cost_penalty:
            return r
    return None


def get_pacer_by_target(
    results: List[Dict[str, Any]], target_spend: float, tol: float = 1e-12
) -> Optional[Dict[str, Any]]:
    """Return the pacer result row for the given target spend."""
    for r in results:
        if r.get("method") != "pacer":
            continue
        t = r.get("target_spend")
        if t is not None and abs(t - target_spend) < tol:
            return r
    return None


def add_static_commands(cs: CommandSet, results: List[Dict[str, Any]]) -> None:
    """Add \\bp commands for static baseline points."""
    # λ_s = 0.0
    s0 = get_static_by_penalty(results, 0.0)
    if s0:
        cs.reward("StaticZeroReward", s0["mean_reward"], s0.get("se_reward"))
        cs.cost_sci("StaticZeroCost", s0["mean_cost"])

    # λ_s = 1.0
    s1 = get_static_by_penalty(results, 1.0)
    if s1:
        cs.reward("StaticOneReward", s1["mean_reward"], s1.get("se_reward"))
        cs.add("StaticOneRewardSE", fmt_reward(s1["mean_reward"], s1.get("se_reward")))
        cs.cost_sci("StaticOneCost", s1["mean_cost"])

    # λ_s = 0.5
    s05 = get_static_by_penalty(results, 0.5)
    if s05:
        cs.reward("StaticHalfReward", s05["mean_reward"], s05.get("se_reward"))
        cs.cost_sci("StaticHalfCost", s05["mean_cost"])

    # λ_s = 2.0
    s2 = get_static_by_penalty(results, 2.0)
    if s2:
        cs.reward("StaticTwoReward", s2["mean_reward"], s2.get("se_reward"))
        cs.cost_sci("StaticTwoCost", s2["mean_cost"])

    # λ_s = 5.0
    s5 = get_static_by_penalty(results, 5.0)
    if s5:
        cs.reward("StaticFiveReward", s5["mean_reward"], s5.get("se_reward"))
        cs.cost_sci("StaticFiveCost", s5["mean_cost"])


def add_pacer_commands(
    cs: CommandSet,
    results: List[Dict[str, Any]],
    budget_targets: List[float],
) -> None:
    """Add \\bp commands for each pacer target."""
    for idx, target in enumerate(budget_targets):
        p = get_pacer_by_target(results, target)
        if not p:
            continue
        name = PACER_INDEX_NAMES[idx]
        cs.reward(f"Pacer{name}Reward", p["mean_reward"], p.get("se_reward"))
        cs.add(f"Pacer{name}RewardSE", fmt_reward(p["mean_reward"], p.get("se_reward")))
        cs.cost_sci(f"Pacer{name}Cost", p["mean_cost"])
        cs.ratio(f"Pacer{name}Util", p.get("budget_utilization", 0.0))
        cs.num(f"Pacer{name}FinalLambda", p.get("final_lambda", 0.0), digits=2)
        lq = p.get("lambda_quartiles") or {}
        cs.num(f"Pacer{name}LambdaQ50", lq.get("q50", 0.0), digits=2)


def add_derived_commands(
    cs: CommandSet,
    results: List[Dict[str, Any]],
    budget_targets: List[float],
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Add \\bp commands for derived quantities."""
    # Cheapest and most expensive model costs (budget targets)
    if budget_targets:
        cs.cost_sci("CheapestModelCost", budget_targets[0])
        cs.cost_sci("ExpensiveModelCost", budget_targets[-1])

    # Static λ=1.0 cost ratio to pacer second-tightest cost
    s1 = get_static_by_penalty(results, 1.0)
    p1 = get_pacer_by_target(results, budget_targets[1]) if len(budget_targets) > 1 else None
    if s1 and p1 and p1.get("mean_cost", 0) > 0:
        ratio_val = s1["mean_cost"] / p1["mean_cost"]
        cs.num("StaticOneCostRatio", ratio_val, digits=1)

    # Unconstrained (static λ=0)
    s0 = get_static_by_penalty(results, 0.0)
    if s0:
        cs.reward("UnconstrainedReward", s0["mean_reward"], s0.get("se_reward"))
        cs.cost_sci("UnconstrainedCost", s0["mean_cost"])

    # Per-prompt oracle
    if data and "oracle_mean_reward" in data:
        cs.reward("OracleReward", data["oracle_mean_reward"])
        cs.cost_sci("OracleCost", data["oracle_mean_cost"])

    # Utilisation range for binding targets (0.95 <= util <= 1.05)
    binding_utils: List[float] = []
    for target in budget_targets:
        p = get_pacer_by_target(results, target)
        if p:
            u = p.get("budget_utilization")
            if u is not None and BINDING_UTIL_LOW <= u <= BINDING_UTIL_HIGH:
                binding_utils.append(u)
    if binding_utils:
        cs.ratio("UtilBindingLow", min(binding_utils))
        cs.ratio("UtilBindingHigh", max(binding_utils))

    # Loosest target utilisation
    if budget_targets:
        p_loose = get_pacer_by_target(results, budget_targets[-1])
        if p_loose and "budget_utilization" in p_loose:
            cs.ratio("LoosestUtil", p_loose["budget_utilization"])

    # Binding threshold (fifth target, B <= ~$0.0019/req)
    if len(budget_targets) > 4:
        cs.cost_sci("BindingThresholdCost", budget_targets[4])

    # Savings at fifth pacer target vs unconstrained (100K req/day)
    s0 = get_static_by_penalty(results, 0.0)
    p5 = (
        get_pacer_by_target(results, budget_targets[4])
        if len(budget_targets) > 4
        else None
    )
    if s0 and p5:
        savings_per_req = s0["mean_cost"] - p5["mean_cost"]
        savings_per_day = REQUESTS_PER_DAY * savings_per_req
        savings_per_year = savings_per_day * 365
        cs.cost_dollar("SavingsPerDay", savings_per_day)
        cs.cost_dollar("SavingsPerYear", savings_per_year)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full CommandSet from JSON data."""
    cs = CommandSet(prefix="bp")
    results = data.get("results", [])
    budget_targets = data.get("budget_targets", [])

    add_static_commands(cs, results)
    add_pacer_commands(cs, results, budget_targets)
    add_derived_commands(cs, results, budget_targets, data=data)

    dt = data.get("dominance_test")
    if dt is not None:
        p = dt["p_value"]
        if p >= 1e-10:
            cs.raw("DominancePValue", f"{p:.1e}")
        else:
            cs.raw("DominancePValue", r"< 10^{-10}")
        cs.num("PacerAUCMean", dt["pacer_auc_mean"], digits=4)
        cs.num("StaticAUCMean", dt["static_auc_mean"], digits=4)
        cs.num("AUCDiffMean", dt["auc_diff_mean"], digits=4)

    return cs


def format_static_row(r: Dict[str, Any], is_last: bool = False) -> str:
    """Format a single static baseline row for the table."""
    lam = r["cost_penalty"]
    reward_str = fmt_reward(r["mean_reward"], r.get("se_reward"))
    cost_str = fmt_cost_dollar(r["mean_cost"])
    qgap = fmt_num(r.get("mean_quality_gap", 0), digits=1)
    line_end = r"  \\[4pt]" if is_last else r"  \\"
    return (
        f"Static & $\\lambda_s{{=}}{lam:.2f}$ & ${reward_str}$ & ${cost_str}$ & ${qgap}$\n"
        f"  & ---\n  & ---\n  & ---\n  {line_end}"
    )


def format_pacer_row(
    r: Dict[str, Any],
    target: float,
    bold_util: bool = False,
) -> str:
    """Format a single pacer row for the table."""
    budget_str = fmt_cost_sci(target)
    reward_str = fmt_reward(r["mean_reward"], r.get("se_reward"))
    cost_str = fmt_cost_dollar(r["mean_cost"])
    qgap = fmt_num(r.get("mean_quality_gap", 0), digits=1)
    util_str = fmt_ratio(r.get("budget_utilization", 0))
    lq = r.get("lambda_quartiles") or {}
    lambda_q50 = fmt_num(lq.get("q50", 0), digits=2)
    lambda_final = fmt_num(r.get("final_lambda", 0), digits=2)
    util_part = f"$\\mathbf{{{util_str}}}$" if bold_util else f"${util_str}$"
    return (
        f"Pacer & $B{{=}}{budget_str}$ & ${reward_str}$ & ${cost_str}$ & ${qgap}$\n"
        f"  & {util_part}\n"
        f"  & ${lambda_q50}$\n"
        f"  & ${lambda_final}$\n"
        "  \\\\"
    )


def generate_table(data: Dict[str, Any]) -> str:
    """Generate the full budget compliance table LaTeX."""
    results = data.get("results", [])
    budget_targets = data.get("budget_targets", [])
    static_penalties = data.get("static_cost_penalties", [])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Budget compliance diagnostics for the BudgetPacer (adaptive mode)",
        r"  and the static cost-penalty baseline ($K{=}3$, 20~seeds).",
        r"  %",
        r"  \emph{Utilisation} = mean realised cost / target $B$;",
        r"  values near $1.0\times$ indicate the pacer is hitting the target.",
        r"  $\lambda_{q50}$: median dual variable over the test phase;",
        r"  $\lambda_{\mathrm{final}}$: value at the last test step.",
        r"  Reward values report mean $\pm$ 1\,SE across seeds.",
        r"  %",
        r"  The static baseline has no target; its cost is determined implicitly",
        r"  by the penalty weight $\lambda_s$.",
        r"  %",
        r"  \textbf{Key finding:}  For all binding targets",
        r"  ($B \leq \bpBindingThresholdCost$/req), utilisation is "
        r"\bpUtilBindingLow--\bpUtilBindingHigh.",
        r"  At the loosest target the router under-spends because the dual variable",
        r"  decays to zero---the pacer constrains from above but never inflates",
        r"  cost.",
        r"}",
        r"\label{tab:budget_compliance}",
        r"\small",
        r"\begin{tabular}{@{}llrrrcrrr@{}}",
        r"\toprule",
        r"\textbf{Method}",
        r"  & \textbf{Setting}",
        r"  & \textbf{Reward}",
        r"  & \textbf{Cost (\$/req)}",
        r"  & \textbf{QGap}",
        r"  & \textbf{Util.}",
        r"  & \textbf{$\lambda_{q50}$}",
        r"  & \textbf{$\lambda_{\mathrm{f}}$}",
        r"  \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\textit{Static cost penalty (no budget target)}} \\[2pt]",
    ]

    for i, lam in enumerate(static_penalties):
        s = get_static_by_penalty(results, lam)
        if s:
            lines.append(format_static_row(s, is_last=(i == len(static_penalties) - 1)))

    lines.extend([
        r"\multicolumn{8}{@{}l}{\textit{BudgetPacer (adaptive, "
        r"$\eta{=}0.05$, $\alpha_{\mathrm{ema}}{=}0.05$, $\bar\lambda{=}5$)}} \\[2pt]",
    ])

    # Binding threshold: B <= ~0.0019 (fifth target is ~0.00187)
    fifth_target = budget_targets[4] if len(budget_targets) > 4 else 0.002
    for idx, target in enumerate(budget_targets):
        p = get_pacer_by_target(results, target)
        if p:
            u = p.get("budget_utilization")
            bold_util = (
                u is not None
                and BINDING_UTIL_LOW <= u <= BINDING_UTIL_HIGH
                and target <= fifth_target
            )
            lines.append(format_pacer_row(p, target, bold_util=bold_util))

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =========================================================================
# Routing evaluation metrics (cost@Q, quality@C, AUCPC, Pareto AUC)
# =========================================================================


def _aggregate_seeds(
    values: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and SE from per-seed metric values.

    Args:
        values: Per-seed scalars (empty entries already filtered out).

    Returns:
        ``(mean, se)`` or ``(None, None)`` if *values* is empty.
    """
    if not values:
        return None, None
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def compute_routing_metrics(
    data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Compute standard LLM routing evaluation metrics for each method.

    For each method builds per-seed Pareto hulls from the sweep and
    computes:

    - **Pareto AUC** (reused from the dominance test).
    - **AUCPC** (normalised to [0, 1] over the model-cost range).
    - **cost@Q%**: cost on the Pareto hull at Q% of per-prompt oracle
      quality.
    - **quality@C%**: quality at C% of per-prompt oracle cost.
    - **CostSave@95%**: ``1 - cost@95% / oracle_cost``.

    The **per-prompt oracle** always selects the highest-quality model
    for each prompt.  It serves as the quality ceiling (reward 1.0 ≡
    oracle quality) and the cost reference (oracle cost = average cost
    of the best-per-prompt model).

    Two sets of anchors are used:

    - **AUCPC** uses the model-mean cost endpoints
      (``budget_targets[0]`` / ``budget_targets[-1]``) so that both
      methods are evaluated over the identical cost range with all
      sweep points included.  Quality anchors: cheapest-model quality
      (pacer at tightest budget) and per-prompt oracle quality.
    - **cost@Q / Q@C / Save** use the per-prompt oracle operating
      point as the reference for quality and cost thresholds.

    Args:
        data: Full JSON dict from ``budget_pacing_results.json``.
              Must contain ``oracle_mean_reward`` and
              ``oracle_mean_cost`` (per-prompt oracle stats).

    Returns:
        Dict keyed by ``"static"``, ``"pacer"``, ``"_anchors"``.
    """
    results = data["results"]
    n_seeds = data["n_seeds"]
    budget_targets = data["budget_targets"]

    static_rows = [r for r in results if r["method"] == "static"]
    pacer_rows = [r for r in results if r["method"] == "pacer"]

    s0 = get_static_by_penalty(results, 0.0)
    s1 = get_static_by_penalty(results, 1.0)
    assert s0 is not None and s1 is not None, (
        "Static lambda=0 and lambda=1 rows required for anchor points"
    )

    oracle_reward = data["oracle_mean_reward"]
    oracle_cost = data["oracle_mean_cost"]

    # Reference for cost@Q and Q@C: per-prompt oracle
    ref_reward = oracle_reward
    ref_cost = oracle_cost

    # AUCPC cost anchors: true model-mean costs (symmetric for both methods)
    aucpc_cheap_cost = budget_targets[0]
    aucpc_frontier_cost = budget_targets[-1]

    # AUCPC quality anchors: cheapest-model quality and per-prompt oracle
    p_tightest = get_pacer_by_target(results, budget_targets[0])
    aucpc_cheap_reward = (
        p_tightest["mean_reward"] if p_tightest else s1["mean_reward"]
    )
    aucpc_frontier_reward = oracle_reward

    dt = data.get("dominance_test", {})
    out: Dict[str, Dict[str, Any]] = {}

    for method_label, rows, dt_auc_key in [
        ("static", static_rows, "per_seed_static_auc"),
        ("pacer", pacer_rows, "per_seed_pacer_auc"),
    ]:
        per_seed_auc: List[float] = dt.get(dt_auc_key, [])
        per_seed_aucpc: List[float] = []
        per_seed_cost_at: Dict[str, List[float]] = {
            k: [] for k in QUALITY_FRACTIONS
        }
        per_seed_quality_at: Dict[str, List[float]] = {
            k: [] for k in COST_FRACTIONS
        }

        for s in range(n_seeds):
            s_costs = [r["per_seed_costs"][s] for r in rows]
            s_rewards = [r["per_seed_rewards"][s] for r in rows]

            aucpc = pareto_aucpc_normalized(
                s_costs,
                s_rewards,
                cheap_cost=aucpc_cheap_cost,
                frontier_cost=aucpc_frontier_cost,
                cheap_reward=aucpc_cheap_reward,
                frontier_reward=aucpc_frontier_reward,
            )
            per_seed_aucpc.append(aucpc)

            hull_c, hull_r = pareto_hull(s_costs, s_rewards)

            for label, q_frac in QUALITY_FRACTIONS.items():
                target_q = q_frac * ref_reward
                c_val = interpolate_pareto_cost(hull_c, hull_r, target_q)
                if c_val is not None:
                    per_seed_cost_at[label].append(c_val)

            for label, c_frac in COST_FRACTIONS.items():
                target_c = c_frac * ref_cost
                q_val = interpolate_pareto_reward(hull_c, hull_r, target_c)
                if q_val is not None:
                    per_seed_quality_at[label].append(q_val)

        metrics: Dict[str, Any] = {}

        auc_m, auc_se = _aggregate_seeds(per_seed_auc)
        metrics["pareto_auc_mean"] = auc_m
        metrics["pareto_auc_se"] = auc_se

        aucpc_m, aucpc_se = _aggregate_seeds(per_seed_aucpc)
        metrics["aucpc_mean"] = aucpc_m
        metrics["aucpc_se"] = aucpc_se

        for label in QUALITY_FRACTIONS:
            m, se = _aggregate_seeds(per_seed_cost_at[label])
            metrics[f"cost_at_{label}_mean"] = m
            metrics[f"cost_at_{label}_se"] = se

        for label in COST_FRACTIONS:
            m, se = _aggregate_seeds(per_seed_quality_at[label])
            metrics[f"quality_at_{label}_mean"] = m
            metrics[f"quality_at_{label}_se"] = se

        cs95_vals = [
            1.0 - c / ref_cost for c in per_seed_cost_at["95"]
        ]
        cs95_m, cs95_se = _aggregate_seeds(cs95_vals)
        metrics["cost_save_95_mean"] = cs95_m
        metrics["cost_save_95_se"] = cs95_se

        out[method_label] = metrics

    out["_anchors"] = {
        "ref_reward": ref_reward,
        "ref_cost": ref_cost,
        "aucpc_cheap_cost": aucpc_cheap_cost,
        "aucpc_frontier_cost": aucpc_frontier_cost,
        "aucpc_cheap_reward": aucpc_cheap_reward,
        "aucpc_frontier_reward": aucpc_frontier_reward,
        "n_seeds": n_seeds,
    }

    return out


def add_routing_metric_commands(
    cs: CommandSet,
    routing_metrics: Dict[str, Dict[str, Any]],
) -> None:
    r"""Add ``\bp`` commands for routing evaluation metrics.

    Args:
        cs: The command-set accumulator.
        routing_metrics: Output of :func:`compute_routing_metrics`.
    """
    for method, pfx in [("static", "RmStatic"), ("pacer", "RmPacer")]:
        m = routing_metrics[method]
        if m["pareto_auc_mean"] is not None:
            cs.num(f"{pfx}AUC", m["pareto_auc_mean"], digits=4)
        if m["aucpc_mean"] is not None:
            cs.num(f"{pfx}AUCPC", m["aucpc_mean"], digits=3)
        if m["cost_at_90_mean"] is not None:
            cs.cost_sci(f"{pfx}CostAt90", m["cost_at_90_mean"])
        if m["cost_at_95_mean"] is not None:
            cs.cost_sci(f"{pfx}CostAt95", m["cost_at_95_mean"])
        if m["quality_at_50_mean"] is not None:
            cs.reward(f"{pfx}QAt50", m["quality_at_50_mean"])
        if m["quality_at_25_mean"] is not None:
            cs.reward(f"{pfx}QAt25", m["quality_at_25_mean"])
        if m.get("cost_save_95_mean") is not None:
            cs.pct(f"{pfx}Save95", m["cost_save_95_mean"], digits=1)


def _fmt_metric_cell(
    val: Optional[float],
    fmt_fn: Any,
    bold: bool = False,
) -> str:
    """Format a single table cell, optionally bolded.

    Args:
        val: Metric value, or ``None`` for a missing entry.
        fmt_fn: Callable that maps ``float -> str`` (LaTeX math body).
        bold: Wrap the entire cell in ``\\textbf``.

    Returns:
        LaTeX snippet ready for insertion into a tabular row.
    """
    if val is None:
        return "---"
    s = fmt_fn(val)
    cell = f"${s}$"
    if bold:
        cell = f"\\textbf{{{cell}}}"
    return cell


def generate_routing_table(
    routing_metrics: Dict[str, Dict[str, Any]],
    dominance_p: float,
) -> str:
    """Generate the LLM routing evaluation metrics table.

    Produces a compact two-row table comparing the static cost-penalty
    baseline and the BudgetPacer on standard metrics from the LLM
    routing literature (cost@Q, quality@C, AUCPC, Pareto AUC).

    Args:
        routing_metrics: Output of :func:`compute_routing_metrics`.
        dominance_p: Wilcoxon *p*-value for Pareto AUC dominance.

    Returns:
        Complete LaTeX table string.
    """
    anchors = routing_metrics["_anchors"]
    sm = routing_metrics["static"]
    pm = routing_metrics["pacer"]
    n_seeds = int(anchors["n_seeds"])

    def _pick_bold(
        s_val: Optional[float],
        p_val: Optional[float],
        higher_is_better: bool,
    ) -> Tuple[bool, bool]:
        if s_val is None or p_val is None:
            return False, False
        if higher_is_better:
            return (s_val > p_val, p_val > s_val)
        return (s_val < p_val, p_val < s_val)

    col_specs: List[Tuple[str, Any, bool]] = [
        ("pareto_auc_mean", lambda v: fmt_num(v, digits=4), True),
        ("aucpc_mean", lambda v: fmt_num(v, digits=3), True),
        ("cost_at_90_mean", fmt_cost_dollar, False),
        ("cost_at_95_mean", fmt_cost_dollar, False),
        ("quality_at_50_mean", lambda v: fmt_reward(v), True),
        ("quality_at_25_mean", lambda v: fmt_reward(v), True),
        ("cost_save_95_mean", lambda v: f"{fmt_pct(v, digits=1)}\\%", True),
    ]

    s_cells: List[str] = []
    p_cells: List[str] = []
    for key, fmt_fn, higher in col_specs:
        s_bold, p_bold = _pick_bold(sm.get(key), pm.get(key), higher)
        s_cells.append(_fmt_metric_cell(sm.get(key), fmt_fn, bold=s_bold))
        p_cells.append(_fmt_metric_cell(pm.get(key), fmt_fn, bold=p_bold))

    p_str = f"{dominance_p:.1e}" if dominance_p >= 1e-10 else r"< 10^{-10}"
    ref_q = fmt_reward(anchors["ref_reward"])
    ref_c = fmt_cost_dollar(anchors["ref_cost"])
    aucpc_lo = fmt_cost_sci(anchors["aucpc_cheap_cost"])
    aucpc_hi = fmt_cost_sci(anchors["aucpc_frontier_cost"])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LLM routing evaluation metrics comparing the static",
        f"  cost-penalty baseline and the BudgetPacer ($K{{=}}3$, {n_seeds}~seeds).",
        r"  %",
        r"  \textbf{cost@$Q$\%\,qual}: cost on the Pareto frontier at $Q$\% of",
        f"  per-prompt oracle quality (${ref_q}$).",
        r"  \textbf{qual@$C$\%\,cost}: quality at $C$\% of per-prompt oracle cost",
        f"  (${ref_c}$/req).",
        r"  \textbf{Save@95\%}: fractional cost reduction at 95\% of oracle quality",
        r"  relative to the oracle cost.",
        r"  \textbf{AUCPC}: normalised area under the cost--performance",
        r"  curve, integrated over the model-cost range",
        f"  $[{aucpc_lo},\\, {aucpc_hi}]$.",
        r"  \textbf{Bold} marks the better value in each column.",
        f"  Pareto AUC dominance: Wilcoxon $p = {p_str}$.",
        r"}",
        r"\label{tab:routing_metrics}",
        r"\small",
        r"\begin{tabular}{@{}l ccccccc@{}}",
        r"\toprule",
        r"\textbf{Method}",
        r"  & \textbf{Pareto AUC\,$\uparrow$}",
        r"  & \textbf{AUCPC\,$\uparrow$}",
        r"  & \textbf{cost@90\%\,qual\,$\downarrow$}",
        r"  & \textbf{cost@95\%\,qual\,$\downarrow$}",
        r"  & \textbf{qual@50\%\,cost\,$\uparrow$}",
        r"  & \textbf{qual@25\%\,cost\,$\uparrow$}",
        r"  & \textbf{Save@95\%\,$\uparrow$}",
        r"  \\",
        r"\midrule",
        "Static & " + " & ".join(s_cells) + r"  \\",
        "Pacer  & " + " & ".join(p_cells) + r"  \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main() -> None:
    """Load JSON, emit _autogen.tex, table_budget_compliance.tex, and table_routing_metrics.tex."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "budget_pacing_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_results(json_path)
    routing_metrics = compute_routing_metrics(data)

    cs = build_command_set(data)
    add_routing_metric_commands(cs, routing_metrics)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 01: stationary budget pacing")

    table_path = exp_dir / "table_budget_compliance.tex"
    table_content = generate_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table → {table_path}")

    dt = data.get("dominance_test", {})
    routing_table_path = exp_dir / "table_routing_metrics.tex"
    routing_table_content = generate_routing_table(
        routing_metrics, dt.get("p_value", 1.0),
    )
    routing_table_path.write_text(routing_table_content)
    print(f"  Wrote table → {routing_table_path}")


if __name__ == "__main__":
    main()
