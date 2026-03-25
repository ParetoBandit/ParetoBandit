#!/usr/bin/env python3
"""Generate LaTeX commands from validation burn-in ablation results.

Reads ``results/val_burnin_ablation_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\vb``).

Covers unconstrained burn-in fractions (0–100%) and the 2×2
budget-constrained factorial (priors × burn-in × budget tier).

Usage::

    python experiments/appendix/val_burnin_ablation/generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_int,
    fmt_num,
    load_json,
)

BURNIN_NAMES: Dict[float, str] = {
    0.0: "Zero",
    0.25: "TwentyFive",
    0.5: "Fifty",
    0.75: "SeventyFive",
    1.0: "Hundred",
}

BUDGET_SHORT: Dict[str, str] = {
    "unconstrained": "Unc",
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}


def _add_condition_commands(
    cs: CommandSet,
    cond: Dict[str, Any],
    pfx: str,
) -> None:
    """Emit commands for a single condition."""
    tm = cond["test_metrics"]

    regret = tm["test_regret"]
    cs.num(f"{pfx}Regret", regret["mean"], digits=1)
    cs.num(f"{pfx}RegretSE", regret["se"], digits=1)

    reward = tm["test_reward"]
    cs.num(f"{pfx}Reward", reward["mean"], digits=3)

    r200 = tm["test_regret_at_200"]
    cs.num(f"{pfx}RAtTwoHundred", r200["mean"], digits=1)

    compliance = cond.get("budget_compliance")
    if compliance:
        cs.num(f"{pfx}CostRatio", compliance["mean_cost_target_ratio"], digits=2)


def _add_stat_test_commands(
    cs: CommandSet,
    test: Dict[str, Any],
    pfx: str,
) -> None:
    """Emit commands for a single statistical test."""
    cs.num(f"{pfx}DeltaRegret", test["delta_regret"], digits=1)
    cs.num(f"{pfx}DeltaPct", test["delta_pct"], digits=1)

    p = test["p_value"]
    if p < 1e-4:
        cs.raw(f"{pfx}PAdj", f"{{<}}10^{{-{int(-np.floor(np.log10(p)))}}}")
    else:
        cs.num(f"{pfx}PAdj", p, digits=3)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="vb")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("NVal", fmt_int(data["n_val"]))
    cs.raw("NTest", fmt_int(data["n_test"]))

    hp_warmup = data["hparams"]["warmup"]
    cs.num("Alpha", hp_warmup["alpha"], digits=2)
    cs.num("Neff", hp_warmup["prior_n_effective"], digits=1)
    cs.num("Gamma", hp_warmup["forgetting_factor"], digits=3)

    hp_tr = data["hparams"]["tabula_rasa"]
    cs.num("TRGamma", hp_tr["forgetting_factor"], digits=3)

    fa = data.get("forgetting_analysis", {})
    cs.num("EffMemSteps", fa.get("effective_memory_steps", 333.3), digits=0)

    conditions = data["conditions"]

    for frac, frac_name in BURNIN_NAMES.items():
        key = f"Warmup ({int(frac*100)}% burn-in)"
        cond = conditions.get(key)
        if cond:
            _add_condition_commands(cs, cond, f"Warmup{frac_name}")

    for frac_pct_str in ("no", "100%"):
        key = f"Tabula Rasa ({frac_pct_str} burn-in)"
        cond = conditions.get(key)
        if cond:
            frac_tag = "Zero" if frac_pct_str == "no" else "Hundred"
            _add_condition_commands(cs, cond, f"TR{frac_tag}")

    warmup_ref = conditions.get("Warmup (100% burn-in)")
    warmup_zero = conditions.get("Warmup (0% burn-in)")
    if warmup_ref and warmup_zero:
        ref_reg = warmup_ref["test_metrics"]["test_regret"]["mean"]
        zero_reg = warmup_zero["test_metrics"]["test_regret"]["mean"]
        cs.num("WarmupZeroDeltaPct", (zero_reg - ref_reg) / ref_reg * 100, digits=1)

    if warmup_ref:
        ref_reg = warmup_ref["test_metrics"]["test_regret"]["mean"]
        for frac_pct_str, frac_tag in (("no", "Zero"), ("100%", "Hundred")):
            tr_key = f"Tabula Rasa ({frac_pct_str} burn-in)"
            tr_cond = conditions.get(tr_key)
            if tr_cond:
                tr_reg = tr_cond["test_metrics"]["test_regret"]["mean"]
                cs.num(
                    f"TR{frac_tag}GroupDeltaPct",
                    (tr_reg - ref_reg) / ref_reg * 100,
                    digits=1,
                )

    for budget_label in ("tight", "moderate", "loose"):
        budget = BUDGET_SHORT[budget_label]
        for warmup in (True, False):
            for frac_pct in ("0%", "100%"):
                if warmup:
                    key = f"Warmup ({frac_pct} burn-in, {budget_label})"
                    frac_tag = "Zero" if frac_pct == "0%" else "Hundred"
                    pfx = f"Warmup{frac_tag}{budget}"
                else:
                    key = f"Tabula Rasa ({frac_pct} burn-in, {budget_label})"
                    frac_tag = "Zero" if frac_pct == "0%" else "Hundred"
                    pfx = f"TR{frac_tag}{budget}"
                cond = conditions.get(key)
                if cond:
                    _add_condition_commands(cs, cond, pfx)

    for budget_label in ("tight", "moderate", "loose"):
        budget = BUDGET_SHORT[budget_label]
        ref_key = f"Warmup (100% burn-in, {budget_label})"
        ref_cond = conditions.get(ref_key)
        if ref_cond is None:
            continue
        ref_reg = ref_cond["test_metrics"]["test_regret"]["mean"]
        for frac_pct, frac_tag in (("0%", "Zero"), ("100%", "Hundred")):
            tr_key = f"Tabula Rasa ({frac_pct} burn-in, {budget_label})"
            tr_cond = conditions.get(tr_key)
            if tr_cond:
                tr_reg = tr_cond["test_metrics"]["test_regret"]["mean"]
                cs.num(
                    f"TR{frac_tag}{budget}GroupDeltaPct",
                    (tr_reg - ref_reg) / ref_reg * 100,
                    digits=1,
                )

    for test_key, test_data in data.get("statistical_tests", {}).items():
        cond = conditions.get(test_key)
        if cond is None:
            continue
        budget_label = cond.get("budget_label", "unconstrained")
        budget = BUDGET_SHORT.get(budget_label, "Unc")

        if "Warmup" in test_key:
            frac = cond.get("burnin_fraction", 0.0)
            frac_name = BURNIN_NAMES.get(frac, f"{int(frac*100)}")
            if budget_label == "unconstrained":
                pfx = f"StatWarmup{frac_name}"
            else:
                pfx = f"StatWarmup{frac_name}{budget}"
        else:
            frac_tag = "Hundred" if "100%" in test_key else "Zero"
            if budget_label == "unconstrained":
                pfx = f"StatTR{frac_tag}"
            else:
                pfx = f"StatTR{frac_tag}{budget}"

        _add_stat_test_commands(cs, test_data, pfx)

    return cs


def main() -> None:
    """Load JSON and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "val_burnin_ablation_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: validation burn-in ablation")


if __name__ == "__main__":
    main()
