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

    # Ranking
    k3_ranking = k3["ranking"]
    cs.pct("FullOrderK", k3_ranking["full_ordering_match"]["frac"], digits=1)

    k3_pw = k3_ranking["pairwise"]
    pw_fracs_k3 = [v["frac"] for v in k3_pw.values()]
    cs.pct("MinPairwiseK", min(pw_fracs_k3), digits=1)

    # c_tilde space
    k3_ct = k3["c_tilde_space"]["per_model"]
    cs.num(
        "LlamaStdFrac",
        k3_ct["Llama-8B"]["std_as_fraction_of_total_gap"] * 100,
        digits=0,
    )
    cs.num(
        "MistralStdFrac",
        k3_ct["Mistral-Large"]["std_as_fraction_of_total_gap"] * 100,
        digits=0,
    )
    cs.num(
        "ProStdFrac",
        k3_ct["Gemini-Pro"]["std_as_fraction_of_total_gap"] * 100,
        digits=0,
    )
    std_fracs_k3 = [
        v["std_as_fraction_of_total_gap"] * 100
        for v in k3_ct.values()
    ]
    cs.raw(
        "StdFracRange",
        f"{min(std_fracs_k3):.0f}--{max(std_fracs_k3):.0f}",
    )

    # CV range
    k3_stats = k3["model_stats"]
    cv_vals_k3 = [s["cv"] for s in k3_stats.values()]
    cs.raw("CVRangeK", f"{min(cv_vals_k3):.2f}--{max(cv_vals_k3):.2f}")

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
        k4_ranking = k4["ranking"]
        cs.pct("FullOrderKFour", k4_ranking["full_ordering_match"]["frac"], digits=1)

        k4_pw = k4_ranking["pairwise"]
        mistral_flash_key = [k for k in k4_pw if "Mistral" in k and "Flash" in k]
        if mistral_flash_key:
            cs.pct("MistralFlashPW", k4_pw[mistral_flash_key[0]]["frac"], digits=1)

        k4_stats = k4["model_stats"]
        cs.num("FlashCtilde", k4_stats["Gemini-Flash"]["heuristic_c_tilde"], digits=3)
        cs.num("MistralCtilde", k4_stats["Mistral-Large"]["heuristic_c_tilde"], digits=3)
        cs.num("ProCtilde", k4_stats["Gemini-Pro"]["heuristic_c_tilde"], digits=3)

        cs.num("FlashCV", k4_stats["Gemini-Flash"]["cv"], digits=2)

        # Mistral-Flash gap in c_tilde space
        k4_gaps = k4["c_tilde_space"]["inter_model_gaps"]
        mf_gap_key = [k for k in k4_gaps if "Mistral" in k and "Flash" in k]
        if mf_gap_key:
            cs.num("MistralFlashGap", k4_gaps[mf_gap_key[0]], digits=3)

        # CV range across all K=4 arms
        cv_vals_k4 = [s["cv"] for s in k4_stats.values()]
        cs.raw("CVRangeKFour", f"{min(cv_vals_k4):.2f}--{max(cv_vals_k4):.2f}")

    cs.write(RESULTS_DIR / "_autogen.tex", "Cost Heuristic Validation")


if __name__ == "__main__":
    main()
