#!/usr/bin/env python3
"""Generate LaTeX commands and failure-response table from catastrophic failure results.

Reads ``results/catastrophic_failure_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\cf``).
- ``_autogen_table_failure_response.tex``: Formal table comparing algorithms'
  response to catastrophic failure across budget levels.

Three-phase design: Phase 1 (normal) -> Phase 2 (Mistral failure) ->
Phase 3 (recovered).

Usage::

    python experiments/03_catastrophic_failure/generate_latex.py
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
    fmt_cost_eng,
    fmt_cost_sci,
    fmt_int,
    fmt_num,
    fmt_pct,
    fmt_ratio,
    fmt_reward,
)

# ======================================================================
# Constants
# ======================================================================

BUDGET_LABEL_TO_SHORT: Dict[str, str] = {
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}

BUDGET_TABLE_DISPLAY: Dict[str, str] = {
    "Tight": "Tight",
    "Mod": "Moderate",
    "Loose": "Loose",
}

CONDITION_ORDER: tuple[str, ...] = (
    "Fixed Policy",
    "Naive Bandit",
    "ParetoBandit",
)

PHASE_NAMES = {1: "One", 2: "Two", 3: "Three"}


# ======================================================================
# Helpers
# ======================================================================


def load_results(json_path: Path) -> Dict[str, Any]:
    """Load catastrophic failure results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def _condition_key(condition: str, budget_label: str) -> str:
    """Build JSON condition key, e.g. ``'Fixed Policy (tight)'``."""
    return f"{condition} ({budget_label})"


def _short_name(condition: str, budget_label: str) -> str:
    """Build short name for LaTeX commands, e.g. ``FixedTight``."""
    short_budget = BUDGET_LABEL_TO_SHORT.get(budget_label, budget_label.title())
    cond_map = {
        "Fixed Policy": "Fixed",
        "Naive Bandit": "Naive",
        "ParetoBandit": "ParetoBandit",
    }
    cond_short = cond_map.get(condition, condition.replace(" ", ""))
    return f"{cond_short}{short_budget}"


# ======================================================================
# Command Set Builder
# ======================================================================


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data.

    Parameters
    ----------
    data : dict
        Parsed experiment JSON.

    Returns
    -------
    CommandSet
        Accumulator with ``\\cf``-prefixed LaTeX commands.
    """
    cs = CommandSet(prefix="cf")
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]

    cs.raw("FailureArm", data["failure_arm_short"])
    cs.num("FailureReward", data["failure_reward"], digits=2)
    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("PhaseN", fmt_int(data["phase_n"]))
    cs.num("Gamma", data["forgetting_factor"], digits=3)

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
                mean_reward = phase_data.get("mean_reward", 0.0)
                mean_lambda = phase_data.get("mean_lambda", 0.0)
                ratio = mean_cost / target if target > 0 else 0.0

                cs.ratio(f"{short}Phase{phase_name}Ratio", ratio)
                cs.raw(f"{short}Phase{phase_name}Cost", fmt_cost_eng(mean_cost))
                cs.num(f"{short}Phase{phase_name}Reward", mean_reward, digits=4)

                if condition == "ParetoBandit":
                    cs.num(
                        f"ParetoBandit{short_budget}LambdaPhase{phase_name}",
                        mean_lambda, digits=2,
                    )

                    arm_fracs = phase_data.get("arm_fractions") or {}
                    failure_arm_short = data["failure_arm_short"]
                    failure_frac = arm_fracs.get(failure_arm_short, 0.0)
                    cs.raw(
                        f"ParetoBandit{short_budget}MistralPhase{phase_name}",
                        fmt_int(failure_frac * 100),
                    )

            if condition == "ParetoBandit":
                p1 = cond_data.get("phase1_summary") or {}
                p2 = cond_data.get("phase2_summary") or {}
                p3 = cond_data.get("phase3_summary") or {}
                r1 = p1.get("mean_reward", 0.0)
                r2 = p2.get("mean_reward", 0.0)
                r3 = p3.get("mean_reward", 0.0)
                cs.num(f"ParetoBandit{short_budget}RewardDrop", r1 - r2, digits=3)
                cs.num(f"ParetoBandit{short_budget}RecoveryGap", r1 - r3, digits=3)

    # ------------------------------------------------------------------
    # Phase 2 adaptation dynamics and cost-quality tradeoff
    # ------------------------------------------------------------------
    phase_boundaries = data["phase_boundaries"]
    p2_start = phase_boundaries[0]
    p3_start = phase_boundaries[1]

    for target, label in zip(budget_targets, budget_labels):
        short_budget = BUDGET_LABEL_TO_SHORT.get(label, label.title())
        key = _condition_key("ParetoBandit", label)
        cond = conditions.get(key)
        if not cond or "curves" not in cond:
            continue
        curves = cond["curves"]

        p1_rewards = [c["mean_window_reward"] for c in curves
                      if p2_start - 120 <= c["step"] <= p2_start]
        p1_baseline = float(np.mean(p1_rewards[-5:])) if len(p1_rewards) >= 5 else p1_rewards[-1]

        p1_costs = [c["mean_window_cost"] for c in curves
                    if p2_start - 120 <= c["step"] <= p2_start]
        p1_cost_baseline = float(np.mean(p1_costs[-5:])) if len(p1_costs) >= 5 else p1_costs[-1]

        p2_entries = [(c["step"], c["mean_window_reward"], c["mean_window_cost"])
                      for c in curves if p2_start < c["step"] <= p3_start]
        p2_rewards_only = [r for _, r, _ in p2_entries]
        p2_steady_reward = float(np.mean(p2_rewards_only[-5:]))
        p2_steady_cost = float(np.mean([c for _, _, c in p2_entries[-5:]]))

        trough_step, trough_reward, _ = min(p2_entries, key=lambda x: x[1])
        trough_offset = trough_step - p2_start

        adapt_step: Optional[int] = None
        past_trough = False
        for s, r, _ in p2_entries:
            if s >= trough_step:
                past_trough = True
            if past_trough and r >= p2_steady_reward * 0.95:
                adapt_step = s - p2_start
                break

        reward_drop_pct = (p1_baseline - p2_steady_reward) / p1_baseline * 100
        cost_change_pct = (p2_steady_cost - p1_cost_baseline) / p1_cost_baseline * 100

        pfx = f"ParetoBandit{short_budget}"
        cs.raw(f"{pfx}TroughOffset", fmt_int(trough_offset))
        cs.num(f"{pfx}TroughReward", trough_reward, digits=3)
        cs.num(f"{pfx}TroughDropPct", (p1_baseline - trough_reward) / p1_baseline * 100, digits=1)
        if adapt_step is not None:
            cs.raw(f"{pfx}AdaptSteps", fmt_int(adapt_step))
        cs.num(f"{pfx}P2SteadyReward", p2_steady_reward, digits=3)
        cs.num(f"{pfx}P2RewardGapPct", reward_drop_pct, digits=1)
        cs.num(f"{pfx}P2CostChangePct", cost_change_pct, digits=1)

    # Unconstrained cost-quality tradeoff
    if "Unconstrained" in conditions:
        uc = conditions["Unconstrained"]
        for phase_num in (1, 2, 3):
            phase_key = f"phase{phase_num}_summary"
            phase_data = uc.get(phase_key) or {}
            phase_name = PHASE_NAMES[phase_num]
            cs.num(
                f"UncPhase{phase_name}Reward",
                phase_data.get("mean_reward", 0.0), digits=4,
            )
            cs.raw(
                f"UncPhase{phase_name}Cost",
                fmt_cost_eng(phase_data.get("mean_cost", 0.0)),
            )

        if "curves" in uc:
            uc_curves = uc["curves"]
            uc_p1_costs = [c["mean_window_cost"] for c in uc_curves
                           if p2_start - 120 <= c["step"] <= p2_start]
            uc_p1_cost = float(np.mean(uc_p1_costs[-5:])) if len(uc_p1_costs) >= 5 else uc_p1_costs[-1]

            uc_p2_entries = [(c["step"], c["mean_window_reward"], c["mean_window_cost"])
                            for c in uc_curves if p2_start < c["step"] <= p3_start]
            uc_p2_steady_cost = float(np.mean([c for _, _, c in uc_p2_entries[-5:]]))
            uc_p2_steady_reward = float(np.mean([r for _, r, _ in uc_p2_entries[-5:]]))
            uc_cost_spike_pct = (uc_p2_steady_cost - uc_p1_cost) / uc_p1_cost * 100

            cs.raw("UncP1Cost", fmt_cost_eng(uc_p1_cost))
            cs.raw("UncP2SteadyCost", fmt_cost_eng(uc_p2_steady_cost))
            cs.num("UncCostSpikePct", uc_cost_spike_pct, digits=1)
            cs.num("UncP2SteadyRewardShort", uc_p2_steady_reward, digits=3)

    return cs


# ======================================================================
# Failure Response Table
# ======================================================================

BINDING_RATIO_LOW = 0.95
BINDING_RATIO_HIGH = 1.05


def _format_ratio_cell(
    ratio: float,
    is_paretobandit: bool,
) -> str:
    """Format a ratio cell with optional bold."""
    within_5pct = BINDING_RATIO_LOW <= ratio <= BINDING_RATIO_HIGH
    should_bold = is_paretobandit or within_5pct

    ratio_str = fmt_ratio(ratio)
    inner = f"\\mathbf{{{ratio_str}}}" if should_bold else ratio_str
    return f"${inner}$"


def generate_failure_response_table(data: Dict[str, Any]) -> str:
    """Generate the failure response table.

    Each row shows one algorithm at one budget level.  Columns show
    Phase 1 mean reward, Phase 2 mean reward (during failure), Phase 3
    mean reward (after recovery), and the Mistral selection fraction
    during Phase 2.

    Parameters
    ----------
    data : dict
        Parsed experiment JSON.

    Returns
    -------
    str
        LaTeX table source.
    """
    conditions = data.get("conditions", {})
    budget_targets = data["budget_targets"]
    budget_labels = data["budget_labels"]
    failure_arm_short = data["failure_arm_short"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Failure response across algorithm variants (Experiment~3,",
        rf"{data['n_seeds']}~seeds).  Phase~1: normal operation; Phase~2:",
        rf"{failure_arm_short} catastrophic failure (reward $\to$ {data['failure_reward']:.2f},",
        r"cost $\to$ \$0); Phase~3: model restored.  \textbf{Bold} marks",
        r"budget compliance within 5\% of target.",
        r"ParetoBandit detects the failure, redistributes traffic, and maintains",
        r"budget compliance; Fixed Policy keeps routing to the dead model.",
        r"Phase~2 budget ratio may fall below $1.00\times$ because the failed",
        r"model's cost drops to \$0.}",
        r"\label{tab:failure_response}",
        r"\small",
        r"\begin{tabular}{@{}ll ccc c ccc @{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Mean Reward}} & "
        rf"& \multicolumn{{3}}{{c}}{{\textbf{{Budget Ratio ($\times$ target)}}}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}",
        r"\textbf{Budget} & \textbf{Algorithm}",
        r"  & \textbf{P1} & \textbf{P2} & \textbf{P3}",
        r"  & \textbf{P2 " + failure_arm_short + r"\%}",
        r"  & \textbf{P1} & \textbf{P2} & \textbf{P3} \\",
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

            rewards, ratios = [], []
            for phase_num in (1, 2, 3):
                phase_key = f"phase{phase_num}_summary"
                pd = cond_data.get(phase_key) or {}
                rewards.append(pd.get("mean_reward", 0.0))
                mc = pd.get("mean_cost", 0.0)
                ratios.append(mc / target if target > 0 else 0.0)

            p2_data = cond_data.get("phase2_summary") or {}
            arm_fracs = p2_data.get("arm_fractions") or {}
            p2_mistral_pct = arm_fracs.get(failure_arm_short, 0.0) * 100

            is_pb = condition == "ParetoBandit"
            cond_display = "\\textbf{ParetoBandit}" if is_pb else condition

            r_cells = [fmt_num(r, digits=3) for r in rewards]
            ratio_cells = [_format_ratio_cell(rat, is_pb) for rat in ratios]

            line_end = r"\\[3pt]" if cond_idx == len(CONDITION_ORDER) - 1 else r"\\"
            if cond_idx == 0:
                budget_col = (
                    f"\\multirow{{{len(CONDITION_ORDER)}}}{{*}}"
                    f"{{{budget_display} (${target_str}$)}}"
                )
            else:
                budget_col = ""

            row = (
                f"{budget_col}  & {cond_display}"
                f"  & {r_cells[0]} & {r_cells[1]} & {r_cells[2]}"
                f"  & {p2_mistral_pct:.0f}\\%"
                f"  & {ratio_cells[0]} & {ratio_cells[1]} & {ratio_cells[2]}"
                f" {line_end}"
            )
            lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    """Load JSON, emit ``_autogen.tex`` and ``_autogen_table_failure_response.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "catastrophic_failure_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.  Run run_catastrophic_failure.py first.")
        sys.exit(1)

    data = load_results(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 02: catastrophic failure (3-phase)")

    table_path = exp_dir / "_autogen_table_failure_response.tex"
    table_content = generate_failure_response_table(data)
    table_path.write_text(table_content)
    print(f"  Wrote table -> {table_path}")


if __name__ == "__main__":
    main()
