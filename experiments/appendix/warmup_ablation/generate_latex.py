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

import inspect

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import (
    CommandSet,
    fmt_int,
    fmt_num,
    fmt_pct,
    fmt_reward,
    load_json,
)

_BOOTSTRAP_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(bootstrap_ci).parameters.items()
    if v.default is not inspect.Parameter.empty
}
_CI_LEVEL: float = _BOOTSTRAP_DEFAULTS["ci_level"]
_N_BOOTSTRAP: int = _BOOTSTRAP_DEFAULTS["n_bootstrap"]
_Z_SCORE: float = float(sp_stats.norm.ppf(1.0 - (1.0 - _CI_LEVEL) / 2.0))

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
    early_step: int,
) -> None:
    """Emit regret, reward, R@early_step, std, and bootstrap CI commands."""
    tr = cond_data["total_regret"]
    cs.num(f"{name_prefix}Regret", tr["mean"], digits=1)
    cs.num(f"{name_prefix}RegretStd", tr["std"], digits=1)
    cs.num(f"{name_prefix}RegretSE", tr["se"], digits=1)

    rw = cond_data["mean_reward"]
    cs.reward(f"{name_prefix}Reward", rw["mean"])

    r_early_key = f"regret_at_{early_step}"
    r_early = cond_data[r_early_key]
    cs.num(f"{name_prefix}RAtTwoHundred", r_early["mean"], digits=1)

    per_seed = cond_data.get("per_seed_regret")
    if per_seed is not None:
        ci_lo, ci_hi = bootstrap_ci(np.array(per_seed))
    else:
        ci_lo = tr["mean"] - _Z_SCORE * tr["se"]
        ci_hi = tr["mean"] + _Z_SCORE * tr["se"]
    cs.num(f"{name_prefix}RegretCILo", ci_lo, digits=1)
    cs.num(f"{name_prefix}RegretCIHi", ci_hi, digits=1)

    per_seed_r_early_key = f"per_seed_regret_at_{early_step}"
    per_seed_r200 = cond_data.get(per_seed_r_early_key)
    if per_seed_r200 is not None:
        r200_ci_lo, r200_ci_hi = bootstrap_ci(np.array(per_seed_r200))
    else:
        r200_ci_lo = r_early["mean"] - _Z_SCORE * r_early["se"]
        r200_ci_hi = r_early["mean"] + _Z_SCORE * r_early["se"]
    cs.num(f"{name_prefix}RAtTwoHundredCILo", r200_ci_lo, digits=1)
    cs.num(f"{name_prefix}RAtTwoHundredCIHi", r200_ci_hi, digits=1)


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


_DIFF_PAIRS: List[tuple] = [
    ("ParetoBandit (warmup)", "Tabula Rasa", "Unc", "TR"),
    ("ParetoBandit (warmup)", "Tabula Rasa (matched-γ)", "Unc", "TRMatched"),
]
for _bl, _bs in BUDGET_SHORT.items():
    if _bl == "unconstrained":
        continue
    _DIFF_PAIRS.append(
        (f"Warmup ({_bl} budget)", f"Tabula Rasa ({_bl} budget)", _bs, "TR")
    )
    _DIFF_PAIRS.append(
        (f"Warmup ({_bl} budget)", f"TR matched-γ ({_bl} budget)", _bs, "TRMatched")
    )


_TR_VS_MATCHED_PAIRS: List[tuple] = [
    ("Tabula Rasa", "Tabula Rasa (matched-γ)", "Unc"),
]
for _bl, _bs in BUDGET_SHORT.items():
    if _bl == "unconstrained":
        continue
    _TR_VS_MATCHED_PAIRS.append(
        (f"Tabula Rasa ({_bl} budget)", f"TR matched-γ ({_bl} budget)", _bs)
    )


def _add_diff_ci_commands(
    cs: CommandSet,
    conditions: Dict[str, Any],
    early_step: int,
) -> None:
    """Emit bootstrap CI commands for paired regret differences.

    For each warmup-vs-baseline pair, emits total-regret and R@early_step
    paired differences with bootstrap CIs.
    """
    per_seed_r_early_key = f"per_seed_regret_at_{early_step}"

    for warmup_key, baseline_key, budget_pfx, bl_pfx in _DIFF_PAIRS:
        w = conditions.get(warmup_key)
        b = conditions.get(baseline_key)
        if w is None or b is None:
            continue

        # Total regret difference
        w_seeds = np.array(w["per_seed_regret"])
        b_seeds = np.array(b["per_seed_regret"])
        diff = b_seeds - w_seeds  # positive = warmup better
        mean_diff = float(np.mean(diff))
        ci_lo, ci_hi = bootstrap_ci(diff)
        pfx = f"{budget_pfx}Vs{bl_pfx}"
        cs.num(f"{pfx}DiffMean", mean_diff, digits=1)
        cs.num(f"{pfx}DiffCILo", ci_lo, digits=1)
        cs.num(f"{pfx}DiffCIHi", ci_hi, digits=1)

        # R@early_step difference
        w_r200 = w.get(per_seed_r_early_key)
        b_r200 = b.get(per_seed_r_early_key)
        if w_r200 is not None and b_r200 is not None:
            diff_r200 = np.array(b_r200) - np.array(w_r200)
            mean_r200 = float(np.mean(diff_r200))
            lo_r200, hi_r200 = bootstrap_ci(diff_r200)
            cs.num(f"{pfx}RAtTwoHundredDiffMean", mean_r200, digits=1)
            cs.num(f"{pfx}RAtTwoHundredDiffCILo", lo_r200, digits=1)
            cs.num(f"{pfx}RAtTwoHundredDiffCIHi", hi_r200, digits=1)

    # TR vs matched-γ TR: positive = matched-γ has lower regret
    for tr_key, mg_key, budget_pfx in _TR_VS_MATCHED_PAIRS:
        tr = conditions.get(tr_key)
        mg = conditions.get(mg_key)
        if tr is None or mg is None:
            continue
        tr_seeds = np.array(tr["per_seed_regret"])
        mg_seeds = np.array(mg["per_seed_regret"])
        diff = mg_seeds - tr_seeds  # positive = matched-γ worse
        mean_diff = float(np.mean(diff))
        ci_lo, ci_hi = bootstrap_ci(diff)
        pfx = f"{budget_pfx}TRMatchedVsTR"
        cs.num(f"{pfx}DiffMean", mean_diff, digits=1)
        cs.num(f"{pfx}DiffCILo", ci_lo, digits=1)
        cs.num(f"{pfx}DiffCIHi", ci_hi, digits=1)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="wa")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("NPrompts", fmt_int(data["n_prompts"]))
    cs.raw("EarlyStep", fmt_int(data["early_step"]))
    cs.raw("K", fmt_int(len(data["arms"])))

    cat_mult = data["catastrophic_multiplier"]
    if cat_mult == int(cat_mult):
        cs.raw("CatMultiplier", fmt_int(int(cat_mult)))
    else:
        cs.num("CatMultiplier", cat_mult, digits=1)

    cs.raw("NTests", fmt_int(len(data.get("paired_tests", []))))
    cs.raw("BootstrapResamples", f"{_N_BOOTSTRAP:,}".replace(",", r"{,}"))
    cs.raw("CILevel", fmt_int(int(_CI_LEVEL * 100)))

    hp_warmup = data["hparams"]["warmup"]
    cs.num("Alpha", hp_warmup["alpha"], digits=2)
    cs.num("Neff", hp_warmup["prior_n_effective"], digits=1)
    cs.num("Gamma", hp_warmup["forgetting_factor"], digits=3)

    hp_tr = data["hparams"]["tabula_rasa"]
    cs.num("TRAlpha", hp_tr["alpha"], digits=2)
    cs.num("TRGamma", hp_tr["forgetting_factor"], digits=3)

    if hp_warmup["alpha"] == hp_tr["alpha"]:
        cs.num("SharedAlpha", hp_warmup["alpha"], digits=2)

    for blabel, btarget in data.get("budget_targets", {}).items():
        bshort = BUDGET_SHORT.get(blabel, blabel.title())
        exp = int(np.floor(np.log10(btarget)))
        coeff = btarget / 10**exp
        cs.raw(f"{bshort}Budget", f"{coeff:.2g}{{\\times}}10^{{{exp}}}")

    early_step = data["early_step"]
    conditions = data["conditions"]

    for cond_label, cond_short in CONDITION_MAP.items():
        cond = conditions.get(cond_label)
        if cond is None:
            continue
        _add_condition_commands(cs, cond, f"Unc{cond_short}", early_step)

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
        _add_condition_commands(cs, cond, f"{budget_short}{cond_short}", early_step)

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

    _add_diff_ci_commands(cs, conditions, early_step)

    # Derived quantities that appear in prose — autogenerate to prevent drift
    warmup_gamma = hp_warmup["forgetting_factor"]
    tr_gamma = hp_tr["forgetting_factor"]
    cs.raw("WarmupEffMem", fmt_int(round(1.0 / (1.0 - warmup_gamma))))
    cs.raw("TREffMem", fmt_int(round(1.0 / (1.0 - tr_gamma))))

    # Std-deviation ratio (tight budget, warmup vs TR)
    w_tight = conditions.get("Warmup (tight budget)")
    t_tight = conditions.get("Tabula Rasa (tight budget)")
    if w_tight and t_tight:
        w_std = w_tight["total_regret"]["std"]
        t_std = t_tight["total_regret"]["std"]
        if w_std > 0:
            cs.num("TightStdRatio", t_std / w_std, digits=0)

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
