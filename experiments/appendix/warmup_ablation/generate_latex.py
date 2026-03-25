#!/usr/bin/env python3
"""Generate LaTeX commands from warmup ablation experiment results.

Reads ``results/warmup_ablation_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\wa``).

Covers all conditions (warmup, tabula rasa, matched-gamma, random)
across budget regimes, including paired test statistics and
catastrophic failure counts.

Usage::

    python experiments/appendix/warmup_ablation/generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import (
    CommandSet,
    fmt_int,
    fmt_num,
    fmt_pct,
    fmt_reward,
    load_json,
)

BUDGET_SHORT: Dict[str, str] = {
    "unconstrained": "Unc",
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}

CONDITION_MAP: Dict[str, str] = {
    "ParetoBandit (warmup)": "Warmup",
    "Tabula Rasa": "TR",
    "Random": "Random",
    "Tabula Rasa (matched-γ)": "TRMatched",
}

BUDGET_CONDITION_MAP: Dict[str, str] = {
    "Warmup (tight budget)": ("Warmup", "Tight"),
    "Tabula Rasa (tight budget)": ("TR", "Tight"),
    "TR matched-γ (tight budget)": ("TRMatched", "Tight"),
    "Warmup (moderate budget)": ("Warmup", "Mod"),
    "Tabula Rasa (moderate budget)": ("TR", "Mod"),
    "TR matched-γ (moderate budget)": ("TRMatched", "Mod"),
    "Warmup (loose budget)": ("Warmup", "Loose"),
    "Tabula Rasa (loose budget)": ("TR", "Loose"),
    "TR matched-γ (loose budget)": ("TRMatched", "Loose"),
}


def _add_condition_commands(
    cs: CommandSet,
    cond_data: Dict[str, Any],
    name_prefix: str,
) -> None:
    """Emit regret, reward, R@200, and std commands for one condition."""
    tr = cond_data["total_regret"]
    cs.num(f"{name_prefix}Regret", tr["mean"], digits=1)
    cs.num(f"{name_prefix}RegretStd", tr["std"], digits=1)
    cs.num(f"{name_prefix}RegretSE", tr["se"], digits=1)

    rw = cond_data["mean_reward"]
    cs.reward(f"{name_prefix}Reward", rw["mean"])

    r200 = cond_data["regret_at_200"]
    cs.num(f"{name_prefix}RAtTwoHundred", r200["mean"], digits=1)

    ci_lo = tr["mean"] - 1.96 * tr["se"]
    ci_hi = tr["mean"] + 1.96 * tr["se"]
    cs.num(f"{name_prefix}RegretCILo", ci_lo, digits=1)
    cs.num(f"{name_prefix}RegretCIHi", ci_hi, digits=1)


def _add_paired_test_commands(
    cs: CommandSet,
    test_data: Dict[str, Any],
    budget_label: str,
    baseline_label: str,
) -> None:
    """Emit commands for a single paired test."""
    budget = BUDGET_SHORT.get(budget_label, budget_label.title())
    if "matched" in baseline_label.lower():
        bl = "TRMatched"
    elif "tabula" in baseline_label.lower():
        bl = "TR"
    else:
        bl = baseline_label

    pfx = f"{budget}Vs{bl}"

    cs.raw(f"{pfx}WarmupWins", fmt_int(test_data["n_warmup_wins"]))
    cs.raw(f"{pfx}BaselineWins", fmt_int(test_data["n_baseline_wins"]))
    cs.raw(f"{pfx}Ties", fmt_int(test_data["n_ties"]))

    p_sign = test_data.get("sign_test_p_value_holm", test_data.get("sign_test_p_value", 1.0))
    if p_sign < 1e-4:
        cs.raw(f"{pfx}SignP", f"{{<}}10^{{-{int(-np.floor(np.log10(p_sign)))}}}")
    else:
        cs.num(f"{pfx}SignP", p_sign, digits=3)

    p_fisher = test_data.get("fisher_exact_p_value_holm", test_data.get("fisher_exact_p_value", 1.0))
    if p_fisher < 1e-4:
        cs.raw(f"{pfx}FisherP", f"{{<}}10^{{-{int(-np.floor(np.log10(p_fisher)))}}}")
    else:
        cs.num(f"{pfx}FisherP", p_fisher, digits=3)

    cs.raw(f"{pfx}WarmupCat", fmt_int(test_data["warmup_catastrophic_count"]))
    cs.raw(f"{pfx}BaselineCat", fmt_int(test_data["baseline_catastrophic_count"]))
    cs.num(f"{pfx}BaselineCatRate", test_data["baseline_catastrophic_rate"] * 100, digits=0)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="wa")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("NPrompts", fmt_int(data["n_prompts"]))
    cs.raw("EarlyStep", fmt_int(data["early_step"]))

    hp_warmup = data["hparams"]["warmup"]
    cs.num("Alpha", hp_warmup["alpha"], digits=2)
    cs.num("Neff", hp_warmup["prior_n_effective"], digits=1)
    cs.num("Gamma", hp_warmup["forgetting_factor"], digits=3)

    hp_tr = data["hparams"]["tabula_rasa"]
    cs.num("TRGamma", hp_tr["forgetting_factor"], digits=3)

    conditions = data["conditions"]

    for cond_label, cond_short in CONDITION_MAP.items():
        cond = conditions.get(cond_label)
        if cond is None:
            continue
        _add_condition_commands(cs, cond, f"Unc{cond_short}")

    warmup_unc = conditions.get("ParetoBandit (warmup)")
    tr_unc = conditions.get("Tabula Rasa")
    if warmup_unc and tr_unc:
        w_reg = warmup_unc["total_regret"]["mean"]
        t_reg = tr_unc["total_regret"]["mean"]
        cs.num("UncRegretReductionPct", (1 - w_reg / t_reg) * 100, digits=0)

    for cond_label, (cond_short, budget_short) in BUDGET_CONDITION_MAP.items():
        cond = conditions.get(cond_label)
        if cond is None:
            continue
        _add_condition_commands(cs, cond, f"{budget_short}{cond_short}")

    for budget_label in ("tight", "moderate", "loose"):
        budget = BUDGET_SHORT[budget_label]
        warmup_key = f"Warmup ({budget_label} budget)"
        tr_key = f"Tabula Rasa ({budget_label} budget)"
        w = conditions.get(warmup_key)
        t = conditions.get(tr_key)
        if w and t:
            w_reg = w["total_regret"]["mean"]
            t_reg = t["total_regret"]["mean"]
            cs.num(f"{budget}RegretReductionPct", (1 - w_reg / t_reg) * 100, digits=0)

    for test in data.get("paired_tests", []):
        _add_paired_test_commands(
            cs, test, test["budget"], test["baseline"],
        )

    return cs


def main() -> None:
    """Load JSON and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "warmup_ablation_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: cold-start vs warmup priors")


if __name__ == "__main__":
    main()
