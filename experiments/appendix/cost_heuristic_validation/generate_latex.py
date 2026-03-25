#!/usr/bin/env python3
"""Generate LaTeX commands for the cost heuristic validation appendix.

Reads results/cost_heuristic_validation.json and emits:
- results/_autogen.tex: \\newcommand definitions (prefix \\ch)

All narrative numbers in results_discussion.tex reference these \\ch*
commands so they stay in sync when the experiment is re-run.

Usage:
    python experiments/appendix/cost_heuristic_validation/generate_latex.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from utils.latex_gen import CommandSet

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "cost_heuristic_validation.json"


def main() -> None:
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    cs = CommandSet("ch")

    # ------------------------------------------------------------------
    # K=3 results
    # ------------------------------------------------------------------
    k3 = data["k3"]
    cs.raw("NPromptsK", str(k3["n_prompts"]))
    cs.raw("NDroppedK", str(k3["n_dropped"]))

    # Ranking
    k3_ranking = k3["ranking"]
    k3_fm = k3_ranking["full_ordering_match"]
    cs.pct("FullOrderK", k3_fm["frac"], digits=1)
    cs.pct("FullOrderKCILo", k3_fm["ci_95_lo"], digits=1)
    cs.pct("FullOrderKCIHi", k3_fm["ci_95_hi"], digits=1)

    k3_pw = k3_ranking["pairwise"]
    pw_fracs_k3 = [v["frac"] for v in k3_pw.values()]
    cs.pct("MinPairwiseK", min(pw_fracs_k3), digits=1)

    # CV range
    k3_stats = k3["model_stats"]
    cv_vals_k3 = [s["cv"] for s in k3_stats.values()]
    cs.raw("CVRangeK", f"{min(cv_vals_k3):.2f}--{max(cv_vals_k3):.2f}")

    # Log-cost separation
    k3_lcs = k3["log_cost_separation"]
    std_fracs_k3 = [
        v * 100 for v in k3_lcs["within_std_as_frac_of_range"].values()
    ]
    cs.raw("LogStdFracRange", f"{min(std_fracs_k3):.0f}--{max(std_fracs_k3):.0f}")

    k3_adj = k3_lcs["adjacent_separation"]
    d_vals_k3 = [v["cohens_d"] for v in k3_adj.values()]
    cs.num("MinCohensD", min(d_vals_k3), digits=2)
    cs.num("MaxCohensD", max(d_vals_k3), digits=2)

    # Prompt-cost correlations
    corrs = k3["prompt_cost_correlation"]
    rho_vals = [s["spearman_rho"] for s in corrs.values() if "note" not in s]
    if rho_vals:
        cs.raw("CorrRange", f"{min(rho_vals):.2f}--{max(rho_vals):.2f}")

    # Cross-model correlations
    xc = k3["cross_model_correlation"]
    xc_vals = [v for v in xc.values() if v != 0.0]
    if xc_vals:
        cs.raw("CrossCorrRange", f"{min(xc_vals):.2f}--{max(xc_vals):.2f}")

    # ------------------------------------------------------------------
    # K=4 results (if available)
    # ------------------------------------------------------------------
    if "k4" in data:
        k4 = data["k4"]
        cs.raw("NPromptsKFour", str(k4["n_prompts"]))
        cs.raw("NDroppedKFour", str(k4["n_dropped"]))
        cs.raw(
            "KFourMissing",
            str(k3.get("n_excluded_for_alignment", 0)),
        )

        k4_ranking = k4["ranking"]
        k4_fm = k4_ranking["full_ordering_match"]
        cs.pct("FullOrderKFour", k4_fm["frac"], digits=1)
        cs.pct("FullOrderKFourCILo", k4_fm["ci_95_lo"], digits=1)
        cs.pct("FullOrderKFourCIHi", k4_fm["ci_95_hi"], digits=1)

        k4_pw = k4_ranking["pairwise"]
        mistral_flash_key = [k for k in k4_pw if "Mistral" in k and "Flash" in k]
        if mistral_flash_key:
            mf = k4_pw[mistral_flash_key[0]]
            cs.pct("MistralFlashPW", mf["frac"], digits=1)
            cs.pct("MistralFlashPWCILo", mf["ci_95_lo"], digits=1)
            cs.pct("MistralFlashPWCIHi", mf["ci_95_hi"], digits=1)
            cs.raw("MistralFlashTies", str(mf["ties"]))

        k4_stats = k4["model_stats"]
        cs.num("FlashCtilde", k4_stats["Gemini-Flash"]["heuristic_c_tilde"], digits=3)
        cs.num("MistralCtilde", k4_stats["Mistral-Large"]["heuristic_c_tilde"], digits=3)
        cs.num("ProCtilde", k4_stats["Gemini-Pro"]["heuristic_c_tilde"], digits=3)
        cs.num("FlashCV", k4_stats["Gemini-Flash"]["cv"], digits=2)

        # Mistral-Flash gap in c_tilde space (deterministic function of pricing)
        k4_ct_vals = {
            name: s["heuristic_c_tilde"] for name, s in k4_stats.items()
        }
        mf_gap = k4_ct_vals["Gemini-Flash"] - k4_ct_vals["Mistral-Large"]
        cs.num("MistralFlashGap", mf_gap, digits=3)

        # CV range across all K=4 arms
        cv_vals_k4 = [s["cv"] for s in k4_stats.values()]
        cs.raw("CVRangeKFour", f"{min(cv_vals_k4):.2f}--{max(cv_vals_k4):.2f}")

        # Cohen's d for the Mistral->Flash pair (weakest separation)
        k4_adj = k4["log_cost_separation"]["adjacent_separation"]
        mf_adj_key = [k for k in k4_adj if "Mistral" in k and "Flash" in k]
        if mf_adj_key:
            cs.num("MFCohensD", k4_adj[mf_adj_key[0]]["cohens_d"], digits=2)

    cs.write(RESULTS_DIR / "_autogen.tex", "Cost Heuristic Validation")


if __name__ == "__main__":
    main()
