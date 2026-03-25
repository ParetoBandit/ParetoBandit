#!/usr/bin/env python3
"""Generate LaTeX commands from prior mismatch sensitivity results.

Reads ``results/prior_mismatch_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\prm``).

Covers the 5×3 prior-quality × n_eff grid plus two Tabula Rasa
baselines, including pairwise test statistics.

Usage::

    python experiments/appendix/prior_mismatch/generate_latex.py
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

QUALITY_SHORT: Dict[str, str] = {
    "Well-calibrated": "WellCal",
    "Random-1680": "Rand",
    "MMLU-only": "MMLU",
    "GSM8K-only": "GSM",
    "Inverted": "Inv",
}

NEFF_SHORT: Dict[int, str] = {
    10: "Ten",
    100: "Hundred",
    1000: "Thousand",
}


def _add_condition_commands(
    cs: CommandSet,
    cond: Dict[str, Any],
    pfx: str,
) -> None:
    """Emit regret stats for a single condition."""
    tr = cond["total_regret"]
    cs.num(f"{pfx}RegretMean", tr["mean"], digits=1)
    cs.num(f"{pfx}RegretMedian", tr["median"], digits=1)
    cs.num(f"{pfx}RegretStd", tr["std"], digits=1)
    cs.num(f"{pfx}RegretSE", tr["se"], digits=1)

    rw = cond["mean_reward"]
    cs.num(f"{pfx}Reward", rw["mean"], digits=3)

    r200 = cond["regret_at_200"]
    cs.num(f"{pfx}RAtTwoHundred", r200["mean"], digits=1)


def _add_pairwise_commands(
    cs: CommandSet,
    test: Dict[str, Any],
    pfx: str,
) -> None:
    """Emit pairwise test commands."""
    cs.raw(f"{pfx}CondWins", fmt_int(test["n_condition_wins"]))
    cs.raw(f"{pfx}BaseWins", fmt_int(test["n_baseline_wins"]))
    cs.num(f"{pfx}DeltaMean", test["delta_mean"], digits=1)
    cs.num(f"{pfx}DeltaMedian", test["delta_median"], digits=1)
    cs.num(f"{pfx}CondMedian", test["condition_median"], digits=1)

    p_sign = test.get("sign_test_p_holm", test.get("sign_test_p", 1.0))
    if p_sign < 1e-4:
        cs.raw(f"{pfx}SignPHolm", f"$<10^{{-{int(-np.floor(np.log10(p_sign)))}}}$")
    else:
        cs.num(f"{pfx}SignPHolm", p_sign, digits=4)

    p_fisher = test.get("fisher_exact_p_holm", test.get("fisher_exact_p", 1.0))
    cs.num(f"{pfx}FisherPHolm", p_fisher, digits=2)

    cs.raw(f"{pfx}CondCat", fmt_int(test["condition_catastrophic_count"]))
    cs.raw(f"{pfx}BaseCat", fmt_int(test["baseline_catastrophic_count"]))


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="prm")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("NPrompts", fmt_int(data["n_prompts"]))
    cs.num("CatThreshold", data["catastrophic_threshold"], digits=1)
    cs.num("CatRefMedian", data["catastrophic_ref_median"], digits=1)

    hp = data["hparams"]["warmup"]
    cs.num("Alpha", hp["alpha"], digits=2)
    cs.num("Gamma", hp["forgetting_factor"], digits=3)

    hp_tr = data["hparams"]["tabula_rasa"]
    cs.num("TRGamma", hp_tr["forgetting_factor"], digits=3)

    conditions = data["conditions"]

    for bl in ("Tabula Rasa", "Tabula Rasa (γ-matched)"):
        cond = conditions.get(bl)
        if cond is None:
            continue
        short = "TR" if bl == "Tabula Rasa" else "TRMatched"
        _add_condition_commands(cs, cond, short)

    for quality_label, q_short in QUALITY_SHORT.items():
        for neff, n_short in NEFF_SHORT.items():
            key = f"{quality_label} (n_eff={neff})"
            cond = conditions.get(key)
            if cond is None:
                continue
            _add_condition_commands(cs, cond, f"{q_short}{n_short}")

    tr_cond = conditions.get("Tabula Rasa (γ-matched)")
    if tr_cond:
        tr_median = tr_cond["total_regret"]["median"]
        for quality_label, q_short in QUALITY_SHORT.items():
            for neff, n_short in NEFF_SHORT.items():
                key = f"{quality_label} (n_eff={neff})"
                cond = conditions.get(key)
                if cond is None:
                    continue
                c_median = cond["total_regret"]["median"]
                reduction_pct = (1 - c_median / tr_median) * 100
                cs.num(f"{q_short}{n_short}ReductionPct", reduction_pct, digits=1)

    inv_1000 = conditions.get("Inverted (n_eff=1000)")
    if inv_1000 and tr_cond:
        tr_median = tr_cond["total_regret"]["median"]
        inv_median = inv_1000["total_regret"]["median"]
        cs.num("InvThousandIncreasePct", (inv_median / tr_median - 1) * 100, digits=0)

    for test_key, test_data in data.get("pairwise_tests_vs_tabula_rasa", {}).items():
        q_label = test_key.rsplit(" (n_eff=", 1)[0]
        neff_str = test_key.rsplit("=", 1)[-1].rstrip(")")
        q_short = QUALITY_SHORT.get(q_label, q_label.replace("-", ""))
        n_short = NEFF_SHORT.get(int(neff_str), neff_str)
        _add_pairwise_commands(cs, test_data, f"Test{q_short}{n_short}")

    gamma_test = data.get("baseline_comparison_gamma_effect")
    if gamma_test:
        cs.raw("GammaCondWins", fmt_int(gamma_test["n_condition_wins"]))
        cs.raw("GammaBaseWins", fmt_int(gamma_test["n_baseline_wins"]))
        cs.raw("GammaTies", fmt_int(gamma_test["n_ties"]))
        cs.num("GammaSignP", gamma_test.get("sign_test_p", 1.0), digits=2)
        cs.num("GammaFisherP", gamma_test.get("fisher_exact_p", 1.0), digits=2)

    for test_key, test_data in data.get("pairwise_tests_vs_gamma_matched", {}).items():
        q_label = test_key.rsplit(" (n_eff=", 1)[0]
        neff_str = test_key.rsplit("=", 1)[-1].rstrip(")")
        q_short = QUALITY_SHORT.get(q_label, q_label.replace("-", ""))
        n_short = NEFF_SHORT.get(int(neff_str), neff_str)
        _add_pairwise_commands(cs, test_data, f"GM{q_short}{n_short}")

    return cs


def main() -> None:
    """Load JSON and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "prior_mismatch_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: prior mismatch sensitivity")


if __name__ == "__main__":
    main()
