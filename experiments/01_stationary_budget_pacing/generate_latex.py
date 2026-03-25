"""Generate LaTeX commands and tables from budget pacing experiment results.

Reads results/budget_pacing_results.json and emits:
- _autogen.tex: \\newcommand definitions for the paper
- table_budget_compliance.tex: budget compliance diagnostics table

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import (
    BINDING_RATIO_HIGH,
    BINDING_RATIO_LOW,
    CommandSet,
    ci_from_seeds_or_normal,
    fmt_cost_sci,
    fmt_num,
    fmt_pct,
    fmt_ratio,
    fmt_reward,
    load_json,
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
BINDING_UTIL_LOW = BINDING_RATIO_LOW
BINDING_UTIL_HIGH = BINDING_RATIO_HIGH


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


def _get_fixed_by_substring(
    results: List[Dict[str, Any]], substr: str,
) -> Optional[Dict[str, Any]]:
    """Return the fixed_model result whose model_id contains *substr*."""
    for r in results:
        if r.get("method") == "fixed_model" and substr in r.get("model_id", ""):
            return r
    return None


def add_fixed_model_commands(
    cs: CommandSet, results: List[Dict[str, Any]],
) -> None:
    """Add ``\\bp`` commands for each fixed single-model baseline."""
    model_map = {
        "Llama": "llama",
        "Mistral": "mistral",
        "Gemini": "gemini",
    }
    for label, substr in model_map.items():
        r = _get_fixed_by_substring(results, substr)
        if r is None:
            continue
        se = r.get("se_reward")
        if se is not None and se < 1e-6:
            se = None
        cs.reward(f"Fixed{label}Reward", r["mean_reward"], se)
        cs.cost_sci(f"Fixed{label}Cost", r["mean_cost"])

        per_seed_r = r.get("per_seed_rewards")
        if per_seed_r and np.std(per_seed_r) > 1e-8:
            lo, hi = bootstrap_ci(np.array(per_seed_r))
            cs.ci_bounds(f"Fixed{label}Reward", lo, hi, digits=3)


def add_annotation_commands(
    cs: CommandSet,
    results: List[Dict[str, Any]],
    budget_targets: List[float],
) -> None:
    """Add ``\\bp`` commands for the figure annotation callout.

    Finds the first pacer point that reaches >= 90% of Gemini quality
    and emits macros for the quality/cost percentages, the absolute
    cost, and the model-mix fractions at that point.  Also emits
    per-pacer-target quality-vs-Gemini and savings-vs-Gemini macros.
    """
    gemini = _get_fixed_by_substring(results, "gemini")
    if gemini is None:
        return

    pacer = sorted(
        [r for r in results if r.get("method") == "pacer"],
        key=lambda r: r["target_spend"],
    )
    if not pacer:
        return

    target_q = 0.90 * gemini["mean_reward"]
    annot_r = next(
        (r for r in pacer if r["mean_reward"] >= target_q), pacer[-1],
    )

    qual_pct = annot_r["mean_reward"] / gemini["mean_reward"] * 100
    cost_pct = annot_r["mean_cost"] / gemini["mean_cost"] * 100
    saving_pct = 100.0 - cost_pct

    cs.num("AnnotQualPct", qual_pct, digits=0)
    cs.num("AnnotCostPct", cost_pct, digits=0)
    cs.cost_sci("AnnotCost", annot_r["mean_cost"])
    cs.cost_sci("AnnotBudget", annot_r["target_spend"])
    cs.num("AnnotSavingPctGemini", saving_pct, digits=0)

    gemini_r_seeds = np.array(gemini.get("per_seed_rewards", []))
    gemini_c_seeds = np.array(gemini.get("per_seed_costs", []))
    annot_r_seeds = np.array(annot_r.get("per_seed_rewards", []))
    annot_c_seeds = np.array(annot_r.get("per_seed_costs", []))

    if len(annot_r_seeds) >= 2 and len(gemini_r_seeds) >= 2:
        q_pct_seeds = annot_r_seeds / gemini_r_seeds * 100
        lo, hi = bootstrap_ci(q_pct_seeds)
        cs.ci_bounds("AnnotQualPct", lo, hi, digits=0)

    if len(annot_c_seeds) >= 2 and len(gemini_c_seeds) >= 2:
        saving_seeds = (1.0 - annot_c_seeds / gemini_c_seeds) * 100
        lo, hi = bootstrap_ci(saving_seeds)
        cs.ci_bounds("AnnotSavingPctGemini", lo, hi, digits=0)

    fracs = annot_r.get("model_fractions", {})
    for mid, frac in fracs.items():
        short = mid.split("/")[-1]
        if "llama" in mid.lower():
            cs.num("AnnotLlamaFrac", frac * 100, digits=0)
        elif "mistral" in mid.lower():
            cs.num("AnnotMistralFrac", frac * 100, digits=0)
        elif "gemini" in mid.lower():
            cs.num("AnnotGeminiFrac", frac * 100, digits=1)

    for idx, target in enumerate(budget_targets):
        p = next(
            (r for r in pacer if abs(r["target_spend"] - target) < 1e-12),
            None,
        )
        if p is None:
            continue
        name = PACER_INDEX_NAMES[idx]
        q_pct = p["mean_reward"] / gemini["mean_reward"] * 100
        c_saving = (1.0 - p["mean_cost"] / gemini["mean_cost"]) * 100
        cs.num(f"Pacer{name}QualPctGemini", q_pct, digits=1)
        cs.num(f"Pacer{name}SavingPctGemini", c_saving, digits=1)

        p_r_seeds = np.array(p.get("per_seed_rewards", []))
        p_c_seeds = np.array(p.get("per_seed_costs", []))
        if len(p_r_seeds) >= 2 and len(gemini_r_seeds) >= 2:
            pq = p_r_seeds / gemini_r_seeds * 100
            lo, hi = bootstrap_ci(pq)
            cs.ci_bounds(f"Pacer{name}QualPctGemini", lo, hi, digits=1)
        if len(p_c_seeds) >= 2 and len(gemini_c_seeds) >= 2:
            ps = (1.0 - p_c_seeds / gemini_c_seeds) * 100
            lo, hi = bootstrap_ci(ps)
            cs.ci_bounds(f"Pacer{name}SavingPctGemini", lo, hi, digits=1)


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
        cs.num(f"Pacer{name}LambdaMedian", lq.get("q50", 0.0), digits=2)

        per_seed_r = p.get("per_seed_rewards")
        if per_seed_r:
            arr = np.array(per_seed_r)
            lo, hi = bootstrap_ci(arr)
            cs.ci_bounds(f"Pacer{name}Reward", lo, hi, digits=3)

        per_seed_c = p.get("per_seed_costs")
        if per_seed_c and target > 0:
            arr_c = np.array(per_seed_c)
            lo_c, hi_c = bootstrap_ci(arr_c)
            cs.ci_bounds(f"Pacer{name}Cost", lo_c, hi_c, digits=6)
            util_arr = arr_c / target
            lo_u, hi_u = bootstrap_ci(util_arr)
            cs.ci_bounds(f"Pacer{name}Util", lo_u, hi_u, digits=2)


def add_derived_commands(
    cs: CommandSet,
    results: List[Dict[str, Any]],
    budget_targets: List[float],
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Add ``\\bp`` commands for derived quantities."""
    if budget_targets:
        cs.cost_sci("CheapestModelCost", budget_targets[0])
        cs.cost_sci("ExpensiveModelCost", budget_targets[-1])

    fixed_models = [r for r in results if r.get("method") == "fixed_model"]
    if fixed_models:
        cheapest_fm = min(fixed_models, key=lambda r: r["mean_cost"])
        expensive_fm = max(fixed_models, key=lambda r: r["mean_cost"])
        if cheapest_fm["mean_cost"] > 0:
            price_range = expensive_fm["mean_cost"] / cheapest_fm["mean_cost"]
            cs.num("PriceRangeX", price_range, digits=0)

    if data and "oracle_mean_reward" in data:
        loosest_pacer = None
        if budget_targets:
            loosest_pacer = get_pacer_by_target(results, budget_targets[-1])
        if loosest_pacer and data["oracle_mean_reward"] > 0:
            ceiling_pct = (
                loosest_pacer["mean_reward"] / data["oracle_mean_reward"] * 100
            )
            cs.num("CeilingOraclePct", ceiling_pct, digits=1)
            lp_seeds = loosest_pacer.get("per_seed_rewards")
            if lp_seeds:
                ceil_seeds = np.array(lp_seeds) / data["oracle_mean_reward"] * 100
                lo, hi = bootstrap_ci(ceil_seeds)
                cs.ci_bounds("CeilingOraclePct", lo, hi, digits=1)

    if data and "oracle_mean_reward" in data:
        cs.reward("OracleReward", data["oracle_mean_reward"])
        cs.cost_sci("OracleCost", data["oracle_mean_cost"])

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

    if budget_targets:
        p_loose = get_pacer_by_target(results, budget_targets[-1])
        if p_loose and "budget_utilization" in p_loose:
            cs.ratio("LoosestUtil", p_loose["budget_utilization"])


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full CommandSet from JSON data."""
    cs = CommandSet(prefix="bp")
    results = data.get("results", [])
    budget_targets = data.get("budget_targets", [])

    hp = data.get("warmup_hparams", {})
    if hp:
        cs.num("Alpha", hp.get("alpha", 0.01), digits=2)
        cs.num("Neff", hp.get("prior_n_effective", 0.0), digits=1)
        cs.num("Gamma", hp.get("forgetting_factor", 0.0), digits=3)

    add_fixed_model_commands(cs, results)
    add_pacer_commands(cs, results, budget_targets)
    add_derived_commands(cs, results, budget_targets, data=data)
    add_annotation_commands(cs, results, budget_targets)

    return cs


def format_fixed_row(
    r: Dict[str, Any],
    gemini_reward: float,
    gemini_cost: float,
) -> str:
    """Format a fixed single-model reference row for the table.

    The model name occupies the Budget column (first column).
    Util. and lambda columns show dashes.

    Args:
        r: Fixed-model result dict.
        gemini_reward: Gemini's mean reward (for % quality).
        gemini_cost: Gemini's mean cost (for % savings).

    Returns:
        LaTeX table row string.
    """
    mid = r.get("model_id", "")
    if "llama" in mid.lower():
        short = "Llama-3.1-8B"
    elif "mistral" in mid.lower():
        short = "Mistral-Large"
    elif "gemini" in mid.lower():
        short = "Gemini-2.5-Pro"
    else:
        short = mid.split("/")[-1]

    reward_str = fmt_reward(r["mean_reward"])
    cost_str = fmt_cost_sci(r["mean_cost"])
    q_pct = r["mean_reward"] / gemini_reward * 100
    saving = (1.0 - r["mean_cost"] / gemini_cost) * 100
    q_pct_str = fmt_num(q_pct, digits=1)
    saving_str = f"${fmt_pct(saving / 100, digits=1)}\\%$"
    return (
        f"{short} & ${reward_str}$ & ${cost_str}$"
        f" & ${q_pct_str}\\%$ & {saving_str}"
        f" & --- & ---"
        "  \\\\"
    )


def format_pacer_row(
    r: Dict[str, Any],
    target: float,
    gemini_reward: float,
    gemini_cost: float,
    bold_util: bool = False,
) -> str:
    """Format a single pacer row for the table.

    Args:
        r: Pacer result dict.
        target: Budget target in $/req.
        gemini_reward: Gemini's mean reward (for % quality).
        gemini_cost: Gemini's mean cost (for % savings).
        bold_util: Whether to bold the utilisation cell.

    Returns:
        LaTeX table row string.
    """
    budget_str = fmt_cost_sci(target)
    mean_r = r["mean_reward"]
    reward_str = fmt_reward(mean_r)
    per_seed_r = r.get("per_seed_rewards")
    if per_seed_r and len(per_seed_r) >= 2:
        lo, hi = bootstrap_ci(np.array(per_seed_r))
        reward_str += f"\\,[{lo:.3f},\\,{hi:.3f}]"
    cost_str = fmt_cost_sci(r["mean_cost"])
    q_pct = mean_r / gemini_reward * 100
    saving = (1.0 - r["mean_cost"] / gemini_cost) * 100
    q_pct_str = fmt_num(q_pct, digits=1)
    saving_str = fmt_pct(saving / 100, digits=1)
    util_str = fmt_ratio(r.get("budget_utilization", 0))
    lq = r.get("lambda_quartiles") or {}
    lambda_q50 = fmt_num(lq.get("q50", 0), digits=2)
    util_part = f"$\\mathbf{{{util_str}}}$" if bold_util else f"${util_str}$"
    return (
        f"$B{{=}}{budget_str}$ & ${reward_str}$ & ${cost_str}$"
        f" & ${q_pct_str}\\%$ & ${saving_str}\\%$"
        f" & {util_part} & ${lambda_q50}$"
        "  \\\\"
    )


def generate_table(data: Dict[str, Any]) -> str:
    """Generate the budget compliance table LaTeX.

    Columns: Budget | Reward | Cost | %Gemini Qual | Savings vs Gemini |
    Util. | lambda_q50.

    Includes fixed single-model reference rows (with dashes for
    budget, utilisation, and lambda) so the reader can compare the
    router's operating points to what a single model provides.
    """
    results = data.get("results", [])
    budget_targets = data.get("budget_targets", [])

    gemini = _get_fixed_by_substring(results, "gemini")
    gemini_reward = gemini["mean_reward"] if gemini else 1.0
    gemini_cost = gemini["mean_cost"] if gemini else 1.0

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Budget-paced routing results ($K{=}3$, 20~seeds).",
        r"  %",
        r"  \emph{\% Gemini Qual.}: mean reward as a percentage of",
        r"  Gemini-2.5-Pro's quality.",
        r"  \emph{Savings}: cost reduction relative to always calling",
        r"  Gemini-2.5-Pro.",
        r"  \emph{Util.}:\ mean realised cost / target $B$;",
        r"  bold values indicate binding targets",
        r"  (\bpUtilBindingLow--\bpUtilBindingHigh).",
        r"  $\lambda_{q50}$: median dual variable over the test phase.",
        r"  Fixed models are single-model baselines (no routing).",
        r"}",
        r"\label{tab:budget_compliance}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Budget}",
        r"  & \textbf{Reward}",
        r"  & \textbf{Cost}",
        r"  & \textbf{\% Gemini}",
        r"  & \textbf{Savings}",
        r"  & \textbf{Util.}",
        r"  & \textbf{$\lambda_{q50}$}",
        r"  \\",
        r"  &",
        r"  &",
        r"  & \textbf{Qual.}",
        r"  & \textbf{vs Gemini}",
        r"  &",
        r"  &",
        r"  \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textit{Fixed single-model baselines}} \\[2pt]",
    ]

    fixed_order = ["llama", "mistral", "gemini"]
    for substr in fixed_order:
        fm = _get_fixed_by_substring(results, substr)
        if fm:
            lines.append(format_fixed_row(fm, gemini_reward, gemini_cost))

    lines.append(r"\\[-4pt]")
    lines.append(
        r"\multicolumn{7}{@{}l}{\textit{BudgetPacer (adaptive)}} \\[2pt]",
    )

    for idx, target in enumerate(budget_targets):
        p = get_pacer_by_target(results, target)
        if p:
            u = p.get("budget_utilization")
            bold_util = (
                u is not None
                and BINDING_UTIL_LOW <= u <= BINDING_UTIL_HIGH
            )
            lines.append(format_pacer_row(
                p, target, gemini_reward, gemini_cost,
                bold_util=bold_util,
            ))

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

    data = load_json(json_path)

    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 01: stationary budget pacing")

    table_path = exp_dir / "table_budget_compliance.tex"
    table_content = generate_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table → {table_path}")


if __name__ == "__main__":
    main()
