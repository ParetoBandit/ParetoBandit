#!/usr/bin/env python3
"""Generate LaTeX commands from model onboarding experiment results.

Reads ``results/model_onboarding_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\mo``).

Covers all three onboarding scenarios (good_cheap, good_expensive,
bad_cheap) across budget tiers and strategies.

Usage::

    python experiments/04_model_onboarding/generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_cost_eng,
    fmt_int,
    fmt_num,
    fmt_pct,
    fmt_reward,
    load_json,
)

BUDGET_SHORT: Dict[str, str] = {
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
    "unconstrained": "Unc",
}

SCENARIO_SHORT: Dict[str, str] = {
    "good_cheap": "GoodCheap",
    "good_expensive": "GoodExp",
    "bad_cheap": "BadCheap",
}

STRATEGY_SHORT: Dict[str, str] = {
    "fixed_uniform": "Fixed",
    "paretobandit_transfer": "PB",
}


def _summary_key(strategy: str, budget_label: str) -> str:
    """Build JSON summaries key, e.g. ``'paretobandit_transfer_tight'``."""
    return f"{strategy}_{budget_label}"


def _add_summary_commands(
    cs: CommandSet,
    summaries: Dict[str, Any],
    strategy: str,
    budget_label: str,
    scenario_prefix: str = "",
) -> None:
    """Emit commands for a single (strategy, budget) summary row."""
    key = _summary_key(strategy, budget_label)
    s = summaries.get(key)
    if s is None:
        return

    strat = STRATEGY_SHORT.get(strategy, strategy)
    budget = BUDGET_SHORT.get(budget_label, budget_label.title())
    pfx = f"{scenario_prefix}{strat}{budget}"

    cs.reward(f"{pfx}PhaseTwoReward", s["phase2_reward"]["mean"])
    cs.raw(f"{pfx}PhaseTwoCost", fmt_cost_eng(s["phase2_cost"]["mean"]))
    cs.reward(f"{pfx}OverallReward", s["overall_reward"]["mean"])
    cs.raw(f"{pfx}OverallCost", fmt_cost_eng(s["overall_cost"]["mean"]))

    fracs = s.get("phase2_model_fractions", {})
    for mid, frac_data in fracs.items():
        if "flash" in mid.lower():
            cs.num(f"{pfx}FlashPct", frac_data["mean"] * 100, digits=1)

    adoption = s.get("flash_adoption", {})
    if adoption:
        cs.raw(f"{pfx}NSustained", fmt_int(adoption.get("n_sustained", 0)))
        sustained_step = adoption.get("mean_sustained_step")
        if sustained_step is not None:
            cs.raw(f"{pfx}SustainedStep", fmt_int(sustained_step))
        final_share = adoption.get("flash_final_share", {})
        if final_share:
            cs.num(f"{pfx}FlashFinalPct", final_share["mean"] * 100, digits=1)

    compliance = s.get("budget_compliance")
    if compliance and budget_label != "unconstrained":
        cs.num(f"{pfx}CostRatio", compliance.get("mean_cost_target_ratio", 0.0), digits=2)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="mo")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("PhaseOneN", fmt_int(data["phase1_n"]))
    cs.raw("PhaseTwoN", fmt_int(data["phase2_n"]))
    cs.raw("BurninPulls", fmt_int(data["burnin_pulls"]))

    hp = data.get("hparams", {})
    cs.num("Alpha", hp.get("alpha", 0.01), digits=2)
    cs.num("Neff", hp.get("prior_n_effective", 0.0), digits=1)
    cs.num("Gamma", hp.get("forgetting_factor", 0.0), digits=3)

    budget_targets = data.get("budget_targets", {})
    for label, target in budget_targets.items():
        short = BUDGET_SHORT.get(label, label.title())
        cs.raw(f"Budget{short}", fmt_cost_eng(target))

    for scenario_name, scenario_data in data.get("scenarios", {}).items():
        sc_pfx = SCENARIO_SHORT.get(scenario_name, scenario_name)
        sc_summaries = scenario_data.get("summaries", {})

        for strategy in STRATEGY_SHORT:
            for budget_label in list(budget_targets.keys()) + ["unconstrained"]:
                _add_summary_commands(
                    cs, sc_summaries, strategy, budget_label,
                    scenario_prefix=sc_pfx,
                )

    _add_scenario_comparison_commands(cs, data)

    return cs


def _add_scenario_comparison_commands(
    cs: CommandSet, data: Dict[str, Any],
) -> None:
    """Add derived commands comparing scenarios for the discussion text."""
    scenarios = data.get("scenarios", {})

    gc = scenarios.get("good_cheap", {}).get("summaries", {})
    ge = scenarios.get("good_expensive", {}).get("summaries", {})
    bc = scenarios.get("bad_cheap", {}).get("summaries", {})

    fp_any = gc.get("fixed_uniform_tight") or gc.get("fixed_uniform_moderate")
    if fp_any:
        cs.raw("FixedAnyCost", fmt_cost_eng(fp_any["phase2_cost"]["mean"]))
        cs.reward("FixedAnyReward", fp_any["phase2_reward"]["mean"])
        fp_fracs = fp_any.get("phase2_model_fractions", {})
        for mid, frac_data in fp_fracs.items():
            if "flash" in mid.lower():
                cs.num("FixedAnyFlashPct", frac_data["mean"] * 100, digits=1)

    for budget_label in ("tight", "moderate", "loose"):
        budget = BUDGET_SHORT.get(budget_label, budget_label.title())
        bt = data.get("budget_targets", {}).get(budget_label, 0.0)

        fp_key = _summary_key("fixed_uniform", budget_label)
        fp = gc.get(fp_key)
        if fp and bt > 0:
            cost_over_budget = fp["phase2_cost"]["mean"] / bt
            cs.num(f"FixedOverBudget{budget}X", cost_over_budget, digits=1)

    ge_loose = ge.get("paretobandit_transfer_loose")
    if ge_loose:
        adoption = ge_loose.get("flash_adoption", {})
        final_share = adoption.get("flash_final_share", {})
        if final_share:
            cs.num("GoodExpLooseFlashFinalPct", final_share["mean"] * 100, digits=1)
        cs.raw("GoodExpLooseNSustained", fmt_int(adoption.get("n_sustained", 0)))

    ge_unc = ge.get("paretobandit_transfer_unconstrained")
    if ge_unc:
        adoption = ge_unc.get("flash_adoption", {})
        final_share = adoption.get("flash_final_share", {})
        if final_share:
            cs.num("GoodExpUncFlashFinalPct", final_share["mean"] * 100, digits=1)

    bc_unc = bc.get("paretobandit_transfer_unconstrained")
    if bc_unc:
        cs.reward("BadCheapUncOverallReward", bc_unc["overall_reward"]["mean"])

    fp_bc = bc.get("fixed_uniform_tight") or bc.get("fixed_uniform_moderate")
    if fp_bc:
        cs.reward("FixedBadCheapReward", fp_bc["phase2_reward"]["mean"])

    fp_ge = ge.get("fixed_uniform_tight") or ge.get("fixed_uniform_moderate")
    if fp_ge:
        cs.raw("FixedGoodExpCost", fmt_cost_eng(fp_ge["phase2_cost"]["mean"]))


def main() -> None:
    """Load JSON and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "model_onboarding_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 04: model onboarding (K=3 → K=4)")


if __name__ == "__main__":
    main()
