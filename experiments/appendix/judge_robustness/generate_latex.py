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

    for budget in ["unconstrained", "tight", "moderate", "loose"]:
        b = BUDGET_SHORT.get(budget, budget)
        gpt_key = ("GPT-4.1-mini", budget, "tabula_rasa")
        r1_key = ("R1", budget, "tabula_rasa")
        if gpt_key in stats and r1_key in stats:
            ratio = stats[gpt_key]["mean"] / stats[r1_key]["mean"]
            cs.num(f"{b}GPTROneRatio", ratio, digits=2)

            gpt_seeds = groups_raw.get(gpt_key)
            r1_seeds = groups_raw.get(r1_key)
            if gpt_seeds and r1_seeds:
                ratio_seeds = np.array(gpt_seeds) / np.array(r1_seeds)
                lo, hi = bootstrap_ci(ratio_seeds)
                cs.ci_bounds(f"{b}GPTROneRatio", lo, hi, digits=2)


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
        cs.num(f"{j_short}CCC", metrics["lins_ccc"], digits=3)
        cs.num(f"{j_short}MAD", metrics["mad"], digits=3)
        cs.num(f"{j_short}Bias", metrics["mean_bias"], digits=3)
        cs.num(f"{j_short}Pearson", metrics["pearson_r"], digits=3)
        cs.num(f"{j_short}BALo", metrics["bland_altman_lower"], digits=2)
        cs.num(f"{j_short}BAHi", metrics["bland_altman_upper"], digits=2)

    for judge_id, j_short in AGREEMENT_JUDGE_SHORT.items():
        routing = summary.get("routing_agreement", {}).get(judge_id)
        if routing is None:
            continue
        cs.num(f"{j_short}BestModelAgr", routing["best_model_agreement"] * 100, digits=1)
        cs.num(f"{j_short}PairwiseAgr", routing["pairwise_sign_agreement"] * 100, digits=1)

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
        _add_agreement_commands(cs, load_json(summary_path))

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
