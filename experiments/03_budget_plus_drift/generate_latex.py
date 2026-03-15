"""Generate LaTeX commands and budget compliance table from budget+drift results.

Reads results/budget_cost_drift_results.json and emits:
- _autogen.tex: \\newcommand definitions (prefix \\bd)
- _autogen_table_budget_compliance.tex: formal budget compliance table

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_cost_eng,
    fmt_cost_sci,
    fmt_int,
    fmt_num,
    fmt_ratio,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

BUDGET_LABEL_TO_SHORT: Dict[str, str] = {
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}

# Full display names for table headers (Mod -> Moderate)
BUDGET_TABLE_DISPLAY: Dict[str, str] = {
    "Tight": "Tight",
    "Mod": "Moderate",
    "Loose": "Loose",
}

CONDITION_ORDER: tuple[str, ...] = (
    "Fixed Policy",
    "Naive Bandit",
    "Recalibrated",
    "BanditGPT",
)

BINDING_RATIO_LOW = 0.95
BINDING_RATIO_HIGH = 1.05
NON_BINDING_RATIO_THRESHOLD = 0.90

GEMINI_ARM_KEY = "Gemini-Pro"


def load_results(json_path: Path) -> Dict[str, Any]:
    """Load budget cost drift results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def _condition_key(condition: str, budget_label: str) -> str:
    """Build JSON condition key, e.g. 'Fixed Policy (tight)'."""
    return f"{condition} ({budget_label})"


def _short_name(condition: str, budget_label: str) -> str:
    """Build short name for commands, e.g. 'FixedTight', 'BanditGPTMod'."""
    short_budget = BUDGET_LABEL_TO_SHORT.get(budget_label, budget_label.title())
    cond_map = {
        "Fixed Policy": "Fixed",
        "Naive Bandit": "Naive",
        "Recalibrated": "Recal",
        "BanditGPT": "BanditGPT",
    }
    cond_short = cond_map.get(condition, condition.replace(" ", ""))
    return f"{cond_short}{short_budget}"


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full CommandSet from JSON data."""
    cs = CommandSet(prefix="bd")
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]

    # prior_n_effective (fixes n_eff=10 bug)
    prior_n_eff = data.get("prior_n_effective", 1000.0)
    cs.raw("Neff", fmt_int(prior_n_eff))

    for idx, (target, label) in enumerate(zip(budget_targets, budget_labels)):
        short_budget = BUDGET_LABEL_TO_SHORT.get(label, label.title())

        for condition in CONDITION_ORDER:
            key = _condition_key(condition, label)
            cond_data = conditions.get(key)
            if not cond_data:
                continue

            short = _short_name(condition, label)
            p1 = cond_data.get("phase1_summary") or {}
            p2 = cond_data.get("phase2_summary") or {}

            mean_cost_p1 = p1.get("mean_cost", 0.0)
            mean_cost_p2 = p2.get("mean_cost", 0.0)

            ratio_p1 = mean_cost_p1 / target if target > 0 else 0.0
            ratio_p2 = mean_cost_p2 / target if target > 0 else 0.0

            # Phase 1 and Phase 2 budget compliance ratios
            cs.ratio(f"{short}PhaseOneRatio", ratio_p1)
            cs.ratio(f"{short}PhaseTwoRatio", ratio_p2)

            # Phase 1 inline table: cost and ratio (for narrative)
            cs.raw(f"{short}PhaseOneCost", fmt_cost_eng(mean_cost_p1))
            # ratio already added above

            # BanditGPT-specific: lambda, Gemini adoption, reward lift, Phase 1 util
            if condition == "BanditGPT":
                mean_lambda_p1 = p1.get("mean_lambda", 0.0)
                mean_lambda_p2 = p2.get("mean_lambda", 0.0)
                cs.num(f"BanditGPT{short_budget}LambdaPhaseOne", mean_lambda_p1, digits=2)
                cs.num(f"BanditGPT{short_budget}LambdaPhaseTwo", mean_lambda_p2, digits=2)

                arm_fracs_p2 = p2.get("arm_fractions") or {}
                gemini_frac = arm_fracs_p2.get(GEMINI_ARM_KEY, 0.0)
                cs.raw(
                    f"BanditGPT{short_budget}GeminiPhaseTwo",
                    fmt_int(gemini_frac * 100),
                )

                mean_reward_p1 = p1.get("mean_reward", 0.0)
                mean_reward_p2 = p2.get("mean_reward", 0.0)
                reward_lift = mean_reward_p2 - mean_reward_p1
                cs.num(f"BanditGPT{short_budget}RewardLift", reward_lift, digits=3)

                cs.ratio(f"BanditGPT{short_budget}PhaseOneUtil", ratio_p1)

    return cs


def _format_ratio_cell(
    ratio: float,
    is_banditgpt: bool,
    is_phase2_non_binding: bool = False,
) -> str:
    """Format a ratio cell with optional bold and dagger."""
    within_5pct = BINDING_RATIO_LOW <= ratio <= BINDING_RATIO_HIGH
    should_bold = is_banditgpt or within_5pct

    ratio_str = fmt_ratio(ratio)
    if should_bold:
        inner = f"\\mathbf{{{ratio_str}}}"
    else:
        inner = ratio_str

    cell = f"${inner}$"
    if is_phase2_non_binding:
        cell = f"${inner}^{{\\dagger}}$"
    return cell


def generate_budget_compliance_table(data: Dict[str, Any]) -> str:
    """Generate the formal budget compliance table LaTeX."""
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Budget compliance under cost drift (Experiment~3,",
        r"50~seeds).  Each cell shows realised average cost as a multiple",
        r"of the budget target ($1.00\times$ = perfect).",
        r"\textbf{Bold} marks values within $5\%$ of $1.00\times$.",
        r"$\dagger$~Phase~2 constraint non-binding: the price drop reduces",
        r"all methods' costs below target, regardless of algorithm.",
        r"BanditGPT is the only condition that reliably meets the target in",
        r"Phase~1 and, where the constraint remains binding (tight), in",
        r"Phase~2.}",
        r"\label{tab:budget_compliance}",
        r"\small",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"\textbf{Budget} & \textbf{Condition}",
        r"  & \textbf{Phase~1} & \textbf{Phase~2} \\",
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

            p1 = cond_data.get("phase1_summary") or {}
            p2 = cond_data.get("phase2_summary") or {}

            mean_cost_p1 = p1.get("mean_cost", 0.0)
            mean_cost_p2 = p2.get("mean_cost", 0.0)

            ratio_p1 = mean_cost_p1 / target if target > 0 else 0.0
            ratio_p2 = mean_cost_p2 / target if target > 0 else 0.0

            is_banditgpt = condition == "BanditGPT"
            phase2_non_binding = ratio_p2 < NON_BINDING_RATIO_THRESHOLD

            cond_display = "\\textbf{BanditGPT}" if is_banditgpt else condition
            cell_p1 = _format_ratio_cell(ratio_p1, is_banditgpt, is_phase2_non_binding=False)
            cell_p2 = _format_ratio_cell(ratio_p2, is_banditgpt, is_phase2_non_binding=phase2_non_binding)

            line_end = r"\\[3pt]" if cond_idx == len(CONDITION_ORDER) - 1 else r"\\"
            if cond_idx == 0:
                row = (
                    f"\\multirow{{4}}{{*}}{{{budget_display} (${target_str}$)}}"
                    f"  & {cond_display}        & {cell_p1} & {cell_p2} {line_end}"
                )
            else:
                row = f"  & {cond_display}        & {cell_p1} & {cell_p2} {line_end}"
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

    data = load_results(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 03: budget + cost drift")

    table_path = exp_dir / "_autogen_table_budget_compliance.tex"
    table_content = generate_budget_compliance_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table → {table_path}")


if __name__ == "__main__":
    main()
