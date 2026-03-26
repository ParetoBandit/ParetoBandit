#!/usr/bin/env python3
"""Generate LaTeX commands from judge robustness experiment results.

Reads three JSON result files and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\jr``).

Covers:
  - Cross-judge regret (cold-start bandit under R1 vs GPT-4.1-mini)
  - Inter-judge agreement metrics (CCC, MAD, bias)
  - Panel-comparison statistics (margin compression, SNR)
  - Per-judge expected reward ordering and gap statistics

Usage::

    python experiments/appendix/judge_robustness/generate_latex.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import (
    CommandSet,
    fmt_int,
    fmt_num,
    load_json,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

JUDGE_SHORT: Dict[str, str] = {
    "R1": "ROne",
    "GPT-4.1-mini": "GPT",
    "Claude-3.7-Sonnet": "Claude",
}

BUDGET_SHORT: Dict[str, str] = {
    "unconstrained": "Unc",
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}

METHOD_SHORT: Dict[str, str] = {
    "tabula_rasa": "TR",
    "random": "Rand",
}

MODEL_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "mistralai/mistral-large-2512": "Mistral",
    "google/gemini-2.5-pro": "Gemini",
}

AGREEMENT_JUDGE_SHORT: Dict[str, str] = {
    "openai/gpt-4.1-mini": "GPT",
    "anthropic/claude-3.7-sonnet": "Claude",
}


def _aggregate_trials(
    trials: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Group trials by (judge, budget, method) and compute summary stats."""
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for t in trials:
        key = (t["judge"], t["budget_label"], t["method"])
        groups[key].append(t["cumulative_regret"])

    stats: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for key, vals in groups.items():
        arr = np.array(vals)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        ci_lo, ci_hi = bootstrap_ci(arr)
        stats[key] = {
            "mean": mean,
            "se": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)),
        }
    return stats


def _add_regret_commands(
    cs: CommandSet,
    regret_data: Dict[str, Any],
) -> None:
    """Emit per-(judge, budget, method) regret macros."""
    trials = regret_data["trials"]
    stats = _aggregate_trials(trials)

    cs.raw("Nseeds", fmt_int(regret_data["n_seeds"]))
    n_test = trials[0]["n_test"] if trials else 0
    cs.raw("NTest", fmt_int(n_test))

    hp = regret_data["tabula_rasa_hparams"]
    cs.num("TRAlpha", hp["alpha"], digits=2)
    cs.num("TRGamma", hp["forgetting_factor"], digits=3)

    for (judge, budget, method), s in stats.items():
        j = JUDGE_SHORT.get(judge, judge)
        b = BUDGET_SHORT.get(budget, budget)
        m = METHOD_SHORT.get(method, method)
        pfx = f"{j}{b}{m}"

        cs.num(f"{pfx}Regret", s["mean"], digits=1)
        cs.num(f"{pfx}RegretSE", s["se"], digits=1)
        cs.num(f"{pfx}RegretCILo", s["ci_lo"], digits=1)
        cs.num(f"{pfx}RegretCIHi", s["ci_hi"], digits=1)

    groups_raw: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for t in trials:
        key = (t["judge"], t["budget_label"], t["method"])
        groups_raw[key].append(t["cumulative_regret"])

    for judge in regret_data["judges"]:
        j = JUDGE_SHORT.get(judge, judge)
        for budget in ["unconstrained", "tight", "moderate", "loose"]:
            b = BUDGET_SHORT.get(budget, budget)
            tr_key = (judge, budget, "tabula_rasa")
            rand_key = (judge, budget, "random")
            if tr_key in stats and rand_key in stats:
                tr_mean = stats[tr_key]["mean"]
                rand_mean = stats[rand_key]["mean"]
                reduction = (1 - tr_mean / rand_mean) * 100
                cs.num(f"{j}{b}ReductionPct", reduction, digits=0)

                tr_seeds = groups_raw.get(tr_key)
                rand_seeds = groups_raw.get(rand_key)
                if tr_seeds and rand_seeds:
                    red_seeds = (1 - np.array(tr_seeds) / np.array(rand_seeds)) * 100
                    lo, hi = bootstrap_ci(red_seeds)
                    cs.ci_bounds(f"{j}{b}ReductionPct", lo, hi, digits=0)

    r1_rand_unc = stats.get(("R1", "unconstrained", "random"))
    supp_judges = [j for j in regret_data.get("judges", []) if j != "R1"]
    for supp_judge in supp_judges:
        j = JUDGE_SHORT.get(supp_judge, supp_judge)
        supp_rand_unc = stats.get((supp_judge, "unconstrained", "random"))
        if supp_rand_unc and r1_rand_unc and r1_rand_unc["mean"] > 0:
            compression = (1 - supp_rand_unc["mean"] / r1_rand_unc["mean"]) * 100
            cs.num(f"{j}RandRegretCompressionPct", compression, digits=0)

    # Per-budget Tabula Rasa ratio (supp / R1) for each supplementary judge.
    for supp_judge in supp_judges:
        j = JUDGE_SHORT.get(supp_judge, supp_judge)
        for budget in ["unconstrained", "tight", "moderate", "loose"]:
            b = BUDGET_SHORT.get(budget, budget)
            supp_key = (supp_judge, budget, "tabula_rasa")
            r1_key = ("R1", budget, "tabula_rasa")
            if supp_key in stats and r1_key in stats:
                ratio = stats[supp_key]["mean"] / stats[r1_key]["mean"]
                cs.num(f"{b}{j}ROneRatio", ratio, digits=2)

                supp_seeds = groups_raw.get(supp_key)
                r1_seeds = groups_raw.get(r1_key)
                if supp_seeds and r1_seeds:
                    ratio_seeds = np.array(supp_seeds) / np.array(r1_seeds)
                    lo, hi = bootstrap_ci(ratio_seeds)
                    cs.ci_bounds(f"{b}{j}ROneRatio", lo, hi, digits=2)

    # Legacy alias: keep \jrRandRegretCompressionPct for GPT-4.1-mini.
    gpt_rand_unc = stats.get(("GPT-4.1-mini", "unconstrained", "random"))
    if gpt_rand_unc and r1_rand_unc and r1_rand_unc["mean"] > 0:
        compression = (1 - gpt_rand_unc["mean"] / r1_rand_unc["mean"]) * 100
        cs.num("RandRegretCompressionPct", compression, digits=0)


def _add_agreement_commands(
    cs: CommandSet,
    summary: Dict[str, Any],
) -> None:
    """Emit inter-judge agreement and gap statistics macros."""
    cs.raw("NSubset", fmt_int(summary["n_subset_prompts"]))

    for judge_id, j_short in AGREEMENT_JUDGE_SHORT.items():
        metrics = summary.get("agreement_metrics", {}).get(judge_id)
        if metrics is None:
            continue
        cs.num(f"{j_short}Spearman", metrics["spearman_rho"], digits=3)
        cs.num(f"{j_short}Kendall", metrics["kendall_tau_b"], digits=3)
        cs.num(f"{j_short}MAD", metrics["mad"], digits=3)
        cs.num(f"{j_short}Bias", metrics["mean_bias"], digits=3)
        cs.num(f"{j_short}BALo", metrics.get("empirical_loa_lower", metrics["bland_altman_lower"]), digits=2)
        cs.num(f"{j_short}BAHi", metrics.get("empirical_loa_upper", metrics["bland_altman_upper"]), digits=2)

    corr = summary.get("correlations", {})
    for judge_id, j_short in AGREEMENT_JUDGE_SHORT.items():
        judge_corr = corr.get(judge_id, {})
        per_model = judge_corr.get("per_model", {})
        for model_id, m_short in MODEL_SHORT.items():
            model_data = per_model.get(model_id)
            if model_data is None:
                continue
            cs.num(f"{j_short}{m_short}Mean", model_data["supp_mean"], digits=3)
        cs.num(f"{j_short}OverallMean", judge_corr.get("supp_mean", 0), digits=3)

    r1_corr_data = list(corr.values())
    if r1_corr_data:
        r1_mean = r1_corr_data[0].get("r1_mean", 0)
        cs.num("ROneMean", r1_mean, digits=3)

    for model_id, m_short in MODEL_SHORT.items():
        for judge_id in corr:
            r1_model = corr[judge_id].get("per_model", {}).get(model_id)
            if r1_model:
                cs.num(f"ROne{m_short}Mean", r1_model["r1_mean"], digits=3)
                break

    gaps = summary.get("gap_statistics", {})
    for label, key in [("ROne", "deepseek-r1"), ("GPT", "GPT-4.1-mini"), ("Claude", "Claude-3.7-Sonnet")]:
        gap_data = gaps.get(key)
        if gap_data:
            cs.num(f"{label}MeanGap", gap_data["mean_gap"], digits=3)
            cs.num(f"{label}MedianGap", gap_data["median_gap"], digits=3)


def _add_ordering_bootstrap_commands(
    cs: CommandSet,
    summary: Dict[str, Any],
) -> None:
    """Emit pairwise ordering bootstrap CI macros.

    For each judge, emits the worst-case (narrowest) CI lower bound
    across all model pairs so the tex can assert "all pairwise 95% CIs
    exclude zero."
    """
    pairwise = summary.get("pairwise_ordering_bootstrap", {})
    if not pairwise:
        return

    all_sig = True
    worst_bonf_lo = float("inf")
    for judge_label, pairs in pairwise.items():
        for pair in pairs:
            if not pair["significant"]:
                all_sig = False
            bonf_lo = pair.get("bonf_ci_lo", pair["ci_lo"])
            if bonf_lo < worst_bonf_lo:
                worst_bonf_lo = bonf_lo

    cs.raw("OrderingAllSig", "true" if all_sig else "false")
    cs.num("OrderingWorstCILo", worst_bonf_lo, digits=4)

    judge_label_map = {
        "deepseek-r1": "ROne",
        "GPT-4.1-mini": "GPT",
        "Claude-3.7-Sonnet": "Claude",
    }
    for judge_key, pairs in pairwise.items():
        j = judge_label_map.get(judge_key, judge_key)
        for pair in pairs:
            ma = MODEL_SHORT.get(pair["model_a"], pair["model_a"])
            mb = MODEL_SHORT.get(pair["model_b"], pair["model_b"])
            pfx = f"{j}{ma}{mb}"
            cs.num(f"{pfx}Diff", pair["mean_diff"], digits=3)
            cs.num(f"{pfx}CILo", pair.get("bonf_ci_lo", pair["ci_lo"]), digits=3)
            cs.num(f"{pfx}CIHi", pair.get("bonf_ci_hi", pair["ci_hi"]), digits=3)


CROSS_JUDGE_SHORT: Dict[str, str] = {
    "R1": "ROne",
    "GPT-4.1-mini": "GPT",
    "Claude-3.7-Sonnet": "Claude",
}


def _add_cross_oracle_commands(
    cs: CommandSet,
    summary: Dict[str, Any],
) -> None:
    """Emit cross-oracle matrix, oracle lift, and derived macros."""
    cross = summary.get("cross_oracle_matrix")
    if not cross:
        return

    judges = list(cross.keys())

    for j_train in judges:
        jt = CROSS_JUDGE_SHORT.get(j_train, j_train)
        for j_eval in judges:
            je = CROSS_JUDGE_SHORT.get(j_eval, j_eval)
            entry = cross[j_train][j_eval]
            cs.num(f"Cross{jt}{je}Mean", entry["mean"], digits=3)
            if "capture_pct" in entry:
                cs.num(
                    f"Cross{jt}{je}Capture",
                    entry["capture_pct"],
                    digits=1,
                )

    lifts = summary.get("oracle_lifts", {})
    for j_name, j_short in CROSS_JUDGE_SHORT.items():
        if j_name in lifts:
            cs.num(f"{j_short}OracleLift", lifts[j_name], digits=3)

    r1_captures = [
        cross["R1"][j]["capture_pct"]
        for j in judges if j != "R1" and "capture_pct" in cross["R1"][j]
    ]
    if r1_captures:
        cs.num("ROneMinCapture", min(r1_captures), digits=1)
        cs.num("ROneMaxCapture", max(r1_captures), digits=1)

    other_on_r1 = [
        cross[j]["R1"]["capture_pct"]
        for j in judges
        if j != "R1" and "capture_pct" in cross[j]["R1"]
    ]
    if other_on_r1:
        cs.num("OtherOnROneMinCapture", min(other_on_r1), digits=1)
        cs.num("OtherOnROneMaxCapture", max(other_on_r1), digits=1)

    all_off_diag = [
        cross[jt][je]["capture_pct"]
        for jt in judges for je in judges
        if jt != je and "capture_pct" in cross[jt][je]
    ]
    if all_off_diag:
        cs.num("MinOffDiagCapture", min(all_off_diag), digits=0)

    margins_pp: List[float] = []
    for j_eval in judges:
        if j_eval == "R1":
            continue
        r1_cap = cross["R1"][j_eval].get("capture_pct", 0.0)
        other_caps = [
            cross[jt][j_eval].get("capture_pct", 0.0)
            for jt in judges if jt != "R1" and jt != j_eval
        ]
        if other_caps:
            margins_pp.append(r1_cap - max(other_caps))
    if margins_pp:
        cs.num("CaptureMarginMinPP", min(margins_pp), digits=1)
        cs.num("CaptureMarginMaxPP", max(margins_pp), digits=1)

    max_half_width = 0.0
    for j_train in judges:
        for j_eval in judges:
            entry = cross[j_train][j_eval]
            hw = (entry["ci_hi"] - entry["ci_lo"]) / 2.0
            max_half_width = max(max_half_width, hw)
    cs.num("CellCIHalfWidthPP", max_half_width * 100, digits=1)

    r1_lift = lifts.get("R1", 0.0)
    if r1_lift > 0:
        supp_lifts = [lifts[j] for j in lifts if j != "R1"]
        if supp_lifts:
            max_comp = (1 - min(supp_lifts) / r1_lift) * 100
            cs.num("OracleLiftMaxCompressionPct", max_comp, digits=0)


def _add_panel_commands(
    cs: CommandSet,
    panel: Dict[str, Any],
) -> None:
    """Emit panel-comparison macros (margin compression, SNR)."""
    cs.num("ROneMargin", panel["r1_mean_margin"], digits=4)
    cs.num("PanelMargin", panel["panel_mean_margin"], digits=4)
    cs.num("MarginCompressionPct", panel["margin_compression_pct"], digits=0)
    cs.num("FracCompressed", panel["frac_compressed"], digits=0)
    cs.num("ROneSNR", panel["r1_snr"], digits=3)
    cs.num("PanelSNR", panel["panel_snr"], digits=3)


def build_command_set() -> CommandSet:
    """Build the full ``CommandSet`` from all judge-robustness JSON files."""
    cs = CommandSet(prefix="jr")

    regret_path = RESULTS_DIR / "cross_judge_regret_results.json"
    if regret_path.exists():
        _add_regret_commands(cs, load_json(regret_path))

    summary_path = RESULTS_DIR / "judge_robustness_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        _add_agreement_commands(cs, summary)
        _add_ordering_bootstrap_commands(cs, summary)
        _add_cross_oracle_commands(cs, summary)

    panel_path = RESULTS_DIR / "panel_comparison_summary.json"
    if panel_path.exists():
        _add_panel_commands(cs, load_json(panel_path))

    return cs


def main() -> None:
    """Load JSON files and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    cs = build_command_set()

    if len(cs) == 0:
        print("Error: no result files found in results/")
        sys.exit(1)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: judge robustness (cross-judge regret)")


if __name__ == "__main__":
    main()
