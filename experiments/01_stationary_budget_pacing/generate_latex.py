"""Generate LaTeX commands and tables from budget pacing experiment results.

Reads results/budget_pacing_results.json and emits:
- _autogen.tex: \\newcommand definitions for the paper
- table_budget_compliance.tex: budget compliance diagnostics table

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_cost_dollar,
    fmt_cost_sci,
    fmt_num,
    fmt_ratio,
    fmt_reward,
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
    add_derived_commands(cs, results, budget_targets)

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


def main() -> None:
    """Load JSON, emit _autogen.tex and table_budget_compliance.tex."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "budget_pacing_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_results(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 01: stationary budget pacing")

    table_path = exp_dir / "table_budget_compliance.tex"
    table_content = generate_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table → {table_path}")


if __name__ == "__main__":
    main()
