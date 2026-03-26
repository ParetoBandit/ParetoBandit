#!/usr/bin/env python3
"""Generate LaTeX commands from prior mismatch sensitivity results.

Reads ``results/prior_mismatch_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\prm``).

Covers the 5×3 prior-quality × n_eff grid plus two Tabula Rasa
baselines, including pairwise test statistics, design parameters,
prior diagnostics, and derived summary statistics.

Usage::

    python experiments/appendix/prior_mismatch/generate_latex.py
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import (
    CommandSet,
    ci_from_seeds_or_normal,
    fmt_int,
    fmt_num,
    load_json,
    median_ci_from_seeds,
)

_BOOTSTRAP_DEFAULTS = {
    k: v.default
    for k, v in inspect.signature(bootstrap_ci).parameters.items()
    if v.default is not inspect.Parameter.empty
}
_CI_LEVEL: float = _BOOTSTRAP_DEFAULTS["ci_level"]
_N_BOOTSTRAP: int = _BOOTSTRAP_DEFAULTS["n_bootstrap"]


def _fmt_comma_int(val: int) -> str:
    """Format an integer with LaTeX-safe thousands separators: ``8{,}373``."""
    s = f"{val:,}"
    return s.replace(",", "{,}")

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
    """Emit regret stats (mean, median, std, SE) with bootstrap CIs."""
    tr = cond["total_regret"]
    cs.num(f"{pfx}RegretMean", tr["mean"], digits=1)
    cs.num(f"{pfx}RegretMedian", tr["median"], digits=1)
    cs.num(f"{pfx}RegretStd", tr["std"], digits=1)
    cs.num(f"{pfx}RegretSE", tr["se"], digits=1)

    per_seed_regret = cond.get("per_seed_regret")
    lo, hi = ci_from_seeds_or_normal(per_seed_regret, tr["mean"], tr["se"])
    cs.ci_bounds(f"{pfx}RegretMean", lo, hi, digits=1)

    med_ci = median_ci_from_seeds(per_seed_regret)
    if med_ci is not None:
        cs.ci_bounds(f"{pfx}RegretMedian", med_ci[0], med_ci[1], digits=1)

    rw = cond["mean_reward"]
    cs.num(f"{pfx}Reward", rw["mean"], digits=3)
    per_seed_reward = cond.get("per_seed_reward")
    rw_se = rw.get("se", rw.get("std", 0.0) / 20**0.5)
    lo, hi = ci_from_seeds_or_normal(per_seed_reward, rw["mean"], rw_se)
    cs.ci_bounds(f"{pfx}Reward", lo, hi, digits=3)

    r200 = cond["regret_at_200"]
    cs.num(f"{pfx}RAtTwoHundred", r200["mean"], digits=1)
    per_seed_r200 = cond.get("per_seed_regret_at_200")
    r200_se = r200.get("se", r200.get("std", 0.0) / 20**0.5)
    lo, hi = ci_from_seeds_or_normal(per_seed_r200, r200["mean"], r200_se)
    cs.ci_bounds(f"{pfx}RAtTwoHundred", lo, hi, digits=1)


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
        cs.raw(f"{pfx}SignPHolm", f"{{<}}10^{{-{int(-np.floor(np.log10(p_sign)))}}}")
    else:
        cs.num(f"{pfx}SignPHolm", p_sign, digits=4)

    p_fisher = test.get("fisher_exact_p_holm", test.get("fisher_exact_p", 1.0))
    cs.num(f"{pfx}FisherPHolm", p_fisher, digits=2)

    cs.raw(f"{pfx}CondCat", fmt_int(test["condition_catastrophic_count"]))
    cs.raw(f"{pfx}BaseCat", fmt_int(test["baseline_catastrophic_count"]))


def _add_design_commands(cs: CommandSet, data: Dict[str, Any]) -> None:
    """Emit design-parameter macros (experiment setup, prior diagnostics)."""
    cs.raw("SeedOffset", _fmt_comma_int(data["seed_offset"]))

    n_qualities = len(data["prior_quality_levels"])
    n_neff = len(data["n_eff_values"])
    cs.raw("NQualities", fmt_int(n_qualities))
    cs.raw("NNeff", fmt_int(n_neff))
    cs.raw("NConditions", fmt_int(n_qualities * n_neff + 2))
    cs.raw("NPairwise", fmt_int(n_qualities * n_neff))

    cat_mult = data["catastrophic_threshold"] / data["catastrophic_ref_median"]
    cs.raw("CatMultiplier", fmt_int(round(cat_mult)))

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from pareto_bandit.config import BEST_K3_HPARAMS

    pca_components = data["hparams"].get(
        "pca_components", BEST_K3_HPARAMS["pca_components"]
    )
    feature_dim = pca_components + 1
    cs.raw("FeatureDim", fmt_int(feature_dim))

    diag = data.get("prior_diagnostics", {})
    for quality_label, q_short in QUALITY_SHORT.items():
        info = diag.get(quality_label, {})
        n_prior_prompts = info.get("n_prompts")
        if n_prior_prompts is not None:
            cs.raw(f"{q_short}NPrompts", _fmt_comma_int(n_prior_prompts))

    gsm_info = diag.get("GSM8K-only", {})
    gsm_models = gsm_info.get("models", {})
    if gsm_models:
        min_pred = min(m["bias_pred"] for m in gsm_models.values())
        cs.num("GSMMinBiasPred", min_pred, digits=2)


def _add_effective_memory_commands(cs: CommandSet, data: Dict[str, Any]) -> None:
    """Emit effective-memory and forgetting-related macros."""
    gamma_w = data["hparams"]["warmup"]["forgetting_factor"]
    gamma_tr = data["hparams"]["tabula_rasa"]["forgetting_factor"]

    eff_mem_w = 1.0 / (1.0 - gamma_w)
    eff_mem_tr = 1.0 / (1.0 - gamma_tr)
    pct_shorter = (1.0 - eff_mem_tr / eff_mem_w) * 100
    cs.raw("EffMemWarmup", fmt_int(round(eff_mem_w)))
    cs.raw("EffMemTR", fmt_int(round(eff_mem_tr)))
    cs.raw("EffMemDiffPct", fmt_int(round(pct_shorter)))


def _add_summary_commands(
    cs: CommandSet,
    data: Dict[str, Any],
) -> None:
    """Emit derived summary macros (std ranges, excess regret, etc.)."""
    conditions = data["conditions"]
    tr_cond = conditions.get("Tabula Rasa (γ-matched)")
    if tr_cond is None:
        return
    tr_median = tr_cond["total_regret"]["median"]

    non_inv_stds: List[float] = []
    for quality_label, q_short in QUALITY_SHORT.items():
        if quality_label == "Inverted":
            continue
        for neff in NEFF_SHORT:
            key = f"{quality_label} (n_eff={neff})"
            cond = conditions.get(key)
            if cond is not None:
                non_inv_stds.append(cond["total_regret"]["std"])

    if non_inv_stds:
        cs.num("NonInvStdMin", min(non_inv_stds), digits=1)
        cs.num("NonInvStdMax", max(non_inv_stds), digits=1)

    inv_1000 = conditions.get("Inverted (n_eff=1000)")
    if inv_1000:
        excess = inv_1000["total_regret"]["median"] - tr_median
        cs.raw("InvThousandExcessRegret", fmt_int(round(excess)))

        inv_ps = inv_1000.get("per_seed_regret")
        bl_ps = tr_cond.get("per_seed_regret") if tr_cond else None
        if inv_ps is not None and bl_ps is not None:
            inv_arr = np.asarray(inv_ps, dtype=np.float64)
            bl_arr = np.asarray(bl_ps, dtype=np.float64)
            n = len(inv_arr)
            rng = np.random.default_rng(42)
            indices = rng.integers(0, n, size=(_N_BOOTSTRAP, n))
            boot_excess = (
                np.median(inv_arr[indices], axis=1)
                - np.median(bl_arr[indices], axis=1)
            )
            alpha = 1.0 - _CI_LEVEL
            lo = float(np.percentile(boot_excess, 100.0 * alpha / 2.0))
            hi = float(np.percentile(boot_excess, 100.0 * (1.0 - alpha / 2.0)))
            cs.ci_bounds("InvThousandExcessRegret", lo, hi, digits=1)

    inv_10 = conditions.get("Inverted (n_eff=10)")
    if inv_10:
        damage_pct = (inv_10["total_regret"]["median"] / tr_median - 1) * 100
        cs.raw("InvTenDamagePct", fmt_int(round(damage_pct)))

    domain_labels = ("MMLU-only", "GSM8K-only")
    domain_p_values: List[float] = []
    for q_label in domain_labels:
        for neff in NEFF_SHORT:
            key = f"{q_label} (n_eff={neff})"
            tests = data.get("pairwise_tests_vs_tabula_rasa", {})
            test = tests.get(key, {})
            p = test.get("sign_test_p_holm", test.get("sign_test_p"))
            if p is not None:
                domain_p_values.append(p)
    if domain_p_values:
        min_p = min(domain_p_values)
        if min_p < 1e-4:
            exp = int(-np.floor(np.log10(min_p)))
            cs.raw("DomainMinPHolm", f"{{<}}10^{{-{exp}}}")
        else:
            cs.num("DomainMinPHolm", min_p, digits=4)

    gamma_test = data.get("baseline_comparison_gamma_effect")
    if gamma_test:
        n_seeds = data["n_seeds"]
        cond_rate = gamma_test["condition_catastrophic_count"] / n_seeds
        n_warmup_conds = len(QUALITY_SHORT) * len(NEFF_SHORT)
        if cond_rate > 0:
            log_prob = n_warmup_conds * n_seeds * math.log10(1 - cond_rate)
            cs.raw("ZeroCatProbExp", fmt_int(round(-log_prob)))


def _add_warmup_ablation_cross_refs(cs: CommandSet) -> None:
    """Load warmup-ablation results for cross-experiment references."""
    wa_json = (
        PROJECT_ROOT
        / "experiments"
        / "appendix"
        / "warmup_ablation"
        / "results"
        / "warmup_ablation_results.json"
    )
    if not wa_json.exists():
        return
    wa_data = load_json(wa_json)

    tests: List[Dict[str, Any]] = wa_data.get("paired_tests", [])
    budget_cat_count = 0
    budget_seeds = 0
    budget_warmup_cat = 0
    for t in tests:
        budget = t.get("budget", "")
        baseline = t.get("baseline", "")
        if budget in ("moderate", "loose") and "Tabula Rasa" in baseline:
            budget_cat_count += t.get("baseline_catastrophic_count", 0)
            budget_warmup_cat += t.get("warmup_catastrophic_count", 0)
            budget_seeds += t.get("n_effective", 0)

    if budget_seeds > 0:
        cat_pct = budget_cat_count / budget_seeds * 100
        cs.raw("WaBudgetCatPct", fmt_int(round(cat_pct)))
        cs.raw("WaBudgetCatCount", fmt_int(budget_cat_count))
        cs.raw("WaBudgetCatSeeds", fmt_int(budget_seeds))

        from scipy.stats import fisher_exact

        table = np.array([
            [budget_warmup_cat, budget_seeds - budget_warmup_cat],
            [budget_cat_count, budget_seeds - budget_cat_count],
        ])
        _, p = fisher_exact(table, alternative="two-sided")
        cs.num("WaBudgetFisherP", p, digits=3)


def _add_reduction_ci(
    cs: CommandSet,
    pfx: str,
    cond: Dict[str, Any],
    baseline_per_seed: np.ndarray,
    baseline_median: float,
) -> None:
    """Emit bootstrap CI for a median-reduction percentage.

    The reduction for a single bootstrap replicate is
    ``(1 - median(cond_resample) / median(baseline_resample)) * 100``.
    """
    cond_per_seed = cond.get("per_seed_regret")
    if cond_per_seed is None:
        return
    cond_arr = np.asarray(cond_per_seed, dtype=np.float64)
    bl_arr = np.asarray(baseline_per_seed, dtype=np.float64)
    n = len(cond_arr)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n, size=(_N_BOOTSTRAP, n))
    boot_reduction = (
        1.0 - np.median(cond_arr[indices], axis=1) / np.median(bl_arr[indices], axis=1)
    ) * 100.0
    alpha = 1.0 - _CI_LEVEL
    lo = float(np.percentile(boot_reduction, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_reduction, 100.0 * (1.0 - alpha / 2.0)))
    cs.ci_bounds(f"{pfx}ReductionPct", lo, hi, digits=1)


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="prm")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("NPrompts", fmt_int(data["n_prompts"]))
    cs.num("CatThreshold", data["catastrophic_threshold"], digits=1)
    cs.num("CatRefMedian", data["catastrophic_ref_median"], digits=1)
    cs.raw("BootstrapResamples", f"{_N_BOOTSTRAP:,}".replace(",", r"{,}"))
    cs.raw("CILevel", fmt_int(int(_CI_LEVEL * 100)))

    hp = data["hparams"]["warmup"]
    cs.num("Alpha", hp["alpha"], digits=2)
    cs.num("Gamma", hp["forgetting_factor"], digits=3)

    hp_tr = data["hparams"]["tabula_rasa"]
    cs.num("TRGamma", hp_tr["forgetting_factor"], digits=3)

    _add_design_commands(cs, data)
    _add_effective_memory_commands(cs, data)

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
        tr_per_seed = np.asarray(
            tr_cond.get("per_seed_regret", []), dtype=np.float64
        )
        for quality_label, q_short in QUALITY_SHORT.items():
            for neff, n_short in NEFF_SHORT.items():
                key = f"{quality_label} (n_eff={neff})"
                cond = conditions.get(key)
                if cond is None:
                    continue
                c_median = cond["total_regret"]["median"]
                reduction_pct = (1 - c_median / tr_median) * 100
                cs.num(f"{q_short}{n_short}ReductionPct", reduction_pct, digits=1)
                if len(tr_per_seed) >= 2:
                    _add_reduction_ci(
                        cs, f"{q_short}{n_short}", cond, tr_per_seed, tr_median
                    )

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
        cs.raw("GammaCondCatCount", fmt_int(gamma_test["condition_catastrophic_count"]))
        cs.raw("GammaBaseCatCount", fmt_int(gamma_test["baseline_catastrophic_count"]))
        n_seeds = data["n_seeds"]
        cond_rate = gamma_test["condition_catastrophic_count"] / n_seeds * 100
        cs.raw("GammaCondCatPct", fmt_int(round(cond_rate)))

    for test_key, test_data in data.get("pairwise_tests_vs_gamma_matched", {}).items():
        q_label = test_key.rsplit(" (n_eff=", 1)[0]
        neff_str = test_key.rsplit("=", 1)[-1].rstrip(")")
        q_short = QUALITY_SHORT.get(q_label, q_label.replace("-", ""))
        n_short = NEFF_SHORT.get(int(neff_str), neff_str)
        _add_pairwise_commands(cs, test_data, f"GM{q_short}{n_short}")

    _add_summary_commands(cs, data)
    _add_warmup_ablation_cross_refs(cs)

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
