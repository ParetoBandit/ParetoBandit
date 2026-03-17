#!/usr/bin/env python3
"""Generate LaTeX commands for the cost heuristic validation appendix.

Reads results/cost_heuristic_validation.json and emits:
- results/_autogen.tex: \\newcommand definitions (prefix \\ch)
- Updates results_discussion.tex by replacing \\VAL_* placeholders

Run from the experiment directory: python generate_latex.py
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
DISCUSSION_TEX = Path(__file__).parent / "results_discussion.tex"


def main() -> None:
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    cs = CommandSet("ch")

    # K=3 results
    k3 = data["k3"]
    k3_ranking = k3["ranking"]
    cs.pct("FullOrderK", k3_ranking["full_ordering_match"]["frac"], digits=1)

    k3_ct = k3["c_tilde_space"]["per_model"]
    cs.num("LlamaStdFrac", k3_ct["Llama-8B"]["std_as_fraction_of_total_gap"] * 100, digits=0)
    cs.num("MistralStdFrac", k3_ct["Mistral-Large"]["std_as_fraction_of_total_gap"] * 100, digits=0)
    cs.num("ProStdFrac", k3_ct["Gemini-Pro"]["std_as_fraction_of_total_gap"] * 100, digits=0)

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

    # K=4 results (if available)
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

    cs.write(RESULTS_DIR / "_autogen.tex", "Cost Heuristic Validation")

    # Update discussion tex with actual values
    replacements = {
        "\\VAL_FULL_ORDERING_K3": f"{k3_ranking['full_ordering_match']['frac']:.1%}",
        "\\VAL_LLAMA_STD_FRAC": f"{k3_ct['Llama-8B']['std_as_fraction_of_total_gap']*100:.0f}\\%",
        "\\VAL_MISTRAL_STD_FRAC": f"{k3_ct['Mistral-Large']['std_as_fraction_of_total_gap']*100:.0f}\\%",
    }

    if "k4" in data:
        k4 = data["k4"]
        k4_ranking = k4["ranking"]
        k4_stats = k4["model_stats"]
        replacements["\\VAL_FULL_ORDERING_K4"] = f"{k4_ranking['full_ordering_match']['frac']:.1%}"
        replacements["\\VAL_FLASH_CTILDE"] = f"{k4_stats['Gemini-Flash']['heuristic_c_tilde']:.3f}"
        replacements["\\VAL_MISTRAL_CTILDE"] = f"{k4_stats['Mistral-Large']['heuristic_c_tilde']:.3f}"
        replacements["\\VAL_PRO_CTILDE"] = f"{k4_stats['Gemini-Pro']['heuristic_c_tilde']:.3f}"
        mistral_flash_key = [k for k in k4_ranking["pairwise"] if "Mistral" in k and "Flash" in k]
        if mistral_flash_key:
            replacements["\\VAL_MISTRAL_FLASH_PAIRWISE"] = (
                f"{k4_ranking['pairwise'][mistral_flash_key[0]]['frac']:.1%}"
            )

    corrs_k3 = k3["prompt_cost_correlation"]
    rho_vals = [s["spearman_rho"] for s in corrs_k3.values() if "note" not in s]
    if rho_vals:
        replacements["\\VAL_CORR_RANGE"] = f"{min(rho_vals):.2f}--{max(rho_vals):.2f}"

    xc_k3 = k3["cross_model_correlation"]
    xc_vals = [v for v in xc_k3.values() if v != 0.0]
    if xc_vals:
        replacements["\\VAL_CROSS_CORR_RANGE"] = f"{min(xc_vals):.2f}--{max(xc_vals):.2f}"

    tex_content = DISCUSSION_TEX.read_text()
    for placeholder, value in replacements.items():
        tex_content = tex_content.replace(placeholder, value)
    DISCUSSION_TEX.write_text(tex_content)
    print(f"  Updated {DISCUSSION_TEX}")


if __name__ == "__main__":
    main()
