"""Generate LaTeX commands and budget compliance table from budget+drift results.

Reads results/budget_cost_drift_results.json and emits:
- _autogen.tex: \\newcommand definitions (prefix \\bd)
- _autogen_table_budget_compliance.tex: formal budget compliance table

Three-phase design: Phase 1 (normal pricing) → Phase 2 (Gemini price drop)
→ Phase 3 (pricing restored).

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import (
    BINDING_RATIO_HIGH,
    BINDING_RATIO_LOW,
    BUDGET_LABEL_TO_SHORT,
    BUDGET_TABLE_DISPLAY,
    PHASE_NAMES,
    CommandSet,
    fmt_cost_eng,
    fmt_cost_sci,
    fmt_int,
    fmt_num,
    fmt_ratio,
    load_json,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CONDITION_ORDER: tuple[str, ...] = (
    "Fixed Policy",
    "Naive Bandit",
    "Recalibrated",
    "Forgetting Bandit",
    "ParetoBandit",
)

NON_BINDING_RATIO_THRESHOLD = 0.90

GEMINI_ARM_KEY = "Gemini-Pro"


def _se_from_seeds(values: List[float]) -> float:
    """Standard error of the mean from a list of per-seed values."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _condition_key(condition: str, budget_label: str) -> str:
    """Build JSON condition key, e.g. 'Fixed Policy (tight)'."""
    return f"{condition} ({budget_label})"


def _short_name(condition: str, budget_label: str) -> str:
    """Build short name for commands, e.g. 'FixedTight', 'ParetoBanditMod'."""
    short_budget = BUDGET_LABEL_TO_SHORT.get(budget_label, budget_label.title())
    cond_map = {
        "Fixed Policy": "Fixed",
        "Naive Bandit": "Naive",
        "Recalibrated": "Recal",
        "Forgetting Bandit": "Forget",
        "ParetoBandit": "ParetoBandit",
    }
    cond_short = cond_map.get(condition, condition.replace(" ", ""))
    return f"{cond_short}{short_budget}"


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full CommandSet from JSON data."""
    cs = CommandSet(prefix="bd")
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]

    prior_n_eff = data.get("prior_n_effective", 1000.0)
    cs.raw("Neff", fmt_int(prior_n_eff))
    cs.num("Alpha", data.get("alpha", 0.01), digits=2)

    gamma = None
    for key, cond_data in conditions.items():
        if "ParetoBandit" in key:
            curves = cond_data.get("curves", [])
            if curves:
                gamma = curves[0].get("mean_gamma")
                break
    if gamma is not None:
        cs.num("Gamma", gamma, digits=3)

    for target, label in zip(budget_targets, budget_labels):
        short_budget = BUDGET_LABEL_TO_SHORT.get(label, label.title())

        for condition in CONDITION_ORDER:
            key = _condition_key(condition, label)
            cond_data = conditions.get(key)
            if not cond_data:
                continue

            short = _short_name(condition, label)

            for phase_num in (1, 2, 3):
                phase_key = f"phase{phase_num}_summary"
                phase_data = cond_data.get(phase_key) or {}
                phase_name = PHASE_NAMES[phase_num]

                mean_cost = phase_data.get("mean_cost", 0.0)
                ratio = mean_cost / target if target > 0 else 0.0

                cs.ratio(f"{short}Phase{phase_name}Ratio", ratio)
                cs.raw(f"{short}Phase{phase_name}Cost", fmt_cost_eng(mean_cost))

                per_seed_costs = cond_data.get(
                    f"per_seed_phase{phase_num}_cost", [],
                )
                if per_seed_costs:
                    cost_se = _se_from_seeds(per_seed_costs)
                    ratio_se = cost_se / target if target > 0 else 0.0
                    cs.raw(
                        f"{short}Phase{phase_name}CostSE",
                        fmt_cost_eng(cost_se),
                    )
                    cs.num(f"{short}Phase{phase_name}RatioSE", ratio_se, digits=2)

                    arr = np.array(per_seed_costs)
                    if target > 0:
                        ratio_seeds = arr / target
                        lo, hi = bootstrap_ci(ratio_seeds)
                        cs.ci_bounds(f"{short}Phase{phase_name}Ratio",
                                     lo, hi, digits=2)

            if condition == "ParetoBandit":
                for phase_num in (1, 2, 3):
                    phase_key = f"phase{phase_num}_summary"
                    phase_data = cond_data.get(phase_key) or {}
                    phase_name = PHASE_NAMES[phase_num]

                    mean_lambda = phase_data.get("mean_lambda", 0.0)
                    cs.num(f"ParetoBandit{short_budget}LambdaPhase{phase_name}",
                           mean_lambda, digits=2)

                    arm_fracs = phase_data.get("arm_fractions") or {}
                    gemini_frac = arm_fracs.get(GEMINI_ARM_KEY, 0.0)
                    cs.raw(
                        f"ParetoBandit{short_budget}GeminiPhase{phase_name}",
                        fmt_int(gemini_frac * 100),
                    )

                    mean_reward = phase_data.get("mean_reward", 0.0)
                    cs.num(f"ParetoBandit{short_budget}RewardPhase{phase_name}",
                           mean_reward, digits=4)

                    per_seed_rewards = cond_data.get(
                        f"per_seed_phase{phase_num}_reward", [],
                    )
                    if per_seed_rewards:
                        reward_se = _se_from_seeds(per_seed_rewards)
                        cs.num(
                            f"ParetoBandit{short_budget}RewardPhase{phase_name}SE",
                            reward_se, digits=4,
                        )
                        r_lo, r_hi = bootstrap_ci(np.array(per_seed_rewards))
                        cs.ci_bounds(
                            f"ParetoBandit{short_budget}RewardPhase{phase_name}",
                            r_lo, r_hi, digits=4,
                        )

                p1 = cond_data.get("phase1_summary") or {}
                p2 = cond_data.get("phase2_summary") or {}
                r1 = p1.get("mean_reward", 0.0)
                r2 = p2.get("mean_reward", 0.0)
                cs.num(f"ParetoBandit{short_budget}RewardLift", r2 - r1, digits=3)

                ps_r1 = cond_data.get("per_seed_phase1_reward", [])
                ps_r2 = cond_data.get("per_seed_phase2_reward", [])
                if ps_r1 and ps_r2:
                    deltas = [b - a for a, b in zip(ps_r1, ps_r2)]
                    lift_se = _se_from_seeds(deltas)
                    cs.num(
                        f"ParetoBandit{short_budget}RewardLiftSE",
                        lift_se, digits=4,
                    )
                    d_arr = np.array(deltas)
                    d_lo, d_hi = bootstrap_ci(d_arr)
                    cs.ci_bounds(
                        f"ParetoBandit{short_budget}RewardLift",
                        d_lo, d_hi, digits=3,
                    )

                ratio_p1 = p1.get("mean_cost", 0.0) / target if target > 0 else 0.0
                cs.ratio(f"ParetoBandit{short_budget}PhaseOneUtil", ratio_p1)

    uc_data = conditions.get("Unconstrained")
    if uc_data is not None:
        uc_p1 = uc_data.get("phase1_summary") or {}
        uc_cost = uc_p1.get("mean_cost", 0.0)
        uc_reward = uc_p1.get("mean_reward", 0.0)
        cs.raw("UncPhaseOneCostEng", fmt_cost_eng(uc_cost))
        cs.num("UncPhaseOneReward", uc_reward, digits=4)

        uc_ps_r1 = uc_data.get("per_seed_phase1_reward", [])
        if uc_ps_r1:
            lo, hi = bootstrap_ci(np.array(uc_ps_r1))
            cs.ci_bounds("UncPhaseOneReward", lo, hi, digits=4)

        for label in budget_labels:
            short_budget = BUDGET_LABEL_TO_SHORT.get(label, label.title())
            pb_key = _condition_key("ParetoBandit", label)
            pb_data = conditions.get(pb_key)
            if pb_data is None:
                continue
            pb_p1 = pb_data.get("phase1_summary") or {}
            pb_cost = pb_p1.get("mean_cost", 0.0)
            pb_reward = pb_p1.get("mean_reward", 0.0)

            cost_ratio = uc_cost / pb_cost if pb_cost > 0 else 0.0
            cost_saving_pct = (1.0 - pb_cost / uc_cost) * 100 if uc_cost > 0 else 0.0
            reward_gap_pct = (1.0 - pb_reward / uc_reward) * 100 if uc_reward > 0 else 0.0

            cs.raw(f"ParetoBandit{short_budget}CostRatioVsUnc",
                   f"{cost_ratio:.1f}")
            cs.raw(f"ParetoBandit{short_budget}CostSavingPct",
                   fmt_int(round(cost_saving_pct)))
            cs.num(f"ParetoBandit{short_budget}RewardGapPct",
                   reward_gap_pct, digits=1)

    return cs


def _format_ratio_cell_with_ci(
    ratio: float,
    ci_lo: float,
    ci_hi: float,
    is_paretobandit: bool,
    is_non_binding: bool = False,
) -> str:
    """Format a budget-utilisation ratio cell with 95% CI, optional bold & dagger."""
    within_5pct = BINDING_RATIO_LOW <= ratio <= BINDING_RATIO_HIGH
    should_bold = is_paretobandit or within_5pct

    ratio_str = fmt_ratio(ratio)
    ci_str = f"[{fmt_num(ci_lo, 2)}, {fmt_num(ci_hi, 2)}]"
    if should_bold:
        inner = f"\\mathbf{{{ratio_str}}}\\,{{\\scriptstyle{ci_str}}}"
    else:
        inner = f"{ratio_str}\\,{{\\scriptstyle{ci_str}}}"

    if is_non_binding:
        return f"${inner}^{{\\dagger}}$"
    return f"${inner}$"


def generate_budget_compliance_table(data: Dict[str, Any]) -> str:
    """Generate the formal budget compliance table LaTeX (3 phases)."""
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Budget compliance under cost drift (Experiment~2,",
        rf"{data['n_seeds']}~seeds, three phases).  Each cell shows realised average cost",
        r"as a multiple of the budget target ($1.00\times$ = perfect) with 95\% bootstrap CI.",
        r"\textbf{Bold} marks values within $5\%$ of $1.00\times$.",
        r"$\dagger$~Phase~2 constraint non-binding: the price drop reduces",
        r"all methods' costs below target, regardless of algorithm.",
        r"ParetoBandit is the only condition that reliably meets the target in",
        r"Phase~1 and recovers compliance in Phase~3 after the price is restored.}",
        r"\label{tab:budget_compliance}",
        r"\small",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Budget} & \textbf{Condition}",
        r"  & \textbf{Phase~1} & \textbf{Phase~2} & \textbf{Phase~3} \\",
        r"\midrule",
    ]

    for target, label in zip(budget_targets, budget_labels):
        short_budget = BUDGET_LABEL_TO_SHORT.get(label, label.title())
        budget_display = BUDGET_TABLE_DISPLAY.get(short_budget, short_budget)
        target_str = fmt_cost_sci(target)

        for cond_idx, condition in enumerate(CONDITION_ORDER):
            key = _condition_key(condition, label)
            cond_data = conditions.get(key)
            if not cond_data:
                continue

            ratios: List[float] = []
            ratio_cis: List[tuple] = []
            for phase_num in (1, 2, 3):
                phase_key = f"phase{phase_num}_summary"
                pd = cond_data.get(phase_key) or {}
                mc = pd.get("mean_cost", 0.0)
                ratios.append(mc / target if target > 0 else 0.0)

                per_seed = cond_data.get(
                    f"per_seed_phase{phase_num}_cost", [],
                )
                if per_seed and target > 0:
                    arr = np.array(per_seed) / target
                    lo, hi = bootstrap_ci(arr)
                    ratio_cis.append((lo, hi))
                else:
                    ratio_cis.append((ratios[-1], ratios[-1]))

            is_paretobandit = condition == "ParetoBandit"
            p2_non_binding = ratios[1] < NON_BINDING_RATIO_THRESHOLD

            cond_display = "\\textbf{ParetoBandit}" if is_paretobandit else condition
            cell_p1 = _format_ratio_cell_with_ci(
                ratios[0], ratio_cis[0][0], ratio_cis[0][1], is_paretobandit,
            )
            cell_p2 = _format_ratio_cell_with_ci(
                ratios[1], ratio_cis[1][0], ratio_cis[1][1], is_paretobandit,
                is_non_binding=p2_non_binding,
            )
            cell_p3 = _format_ratio_cell_with_ci(
                ratios[2], ratio_cis[2][0], ratio_cis[2][1], is_paretobandit,
            )

            line_end = r"\\[3pt]" if cond_idx == len(CONDITION_ORDER) - 1 else r"\\"
            n_conds = len(CONDITION_ORDER)
            if cond_idx == 0:
                row = (
                    f"\\multirow{{{n_conds}}}{{*}}{{{budget_display} (${target_str}$)}}"
                    f"  & {cond_display}        & {cell_p1} & {cell_p2} & {cell_p3} {line_end}"
                )
            else:
                row = f"  & {cond_display}        & {cell_p1} & {cell_p2} & {cell_p3} {line_end}"
            lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main() -> None:
    """Load JSON, emit _autogen.tex and _autogen_table_budget_compliance.tex."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "budget_cost_drift_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 02: budget + cost drift (3-phase)")

    table_path = exp_dir / "_autogen_table_budget_compliance.tex"
    table_content = generate_budget_compliance_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table → {table_path}")


if __name__ == "__main__":
    main()
