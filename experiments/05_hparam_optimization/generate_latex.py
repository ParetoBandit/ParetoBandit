"""Generate LaTeX commands from hyperparameter optimization results.

Reads results/hparam_sweep_results.json and results/best_hparams.json,
emits _autogen.tex with \\newcommand definitions for the paper.

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import CommandSet, fmt_int, fmt_num, load_json


def format_alpha(val: float) -> str:
    """Format alpha for LaTeX: 0.1 -> '0.10', 0.01 -> '0.01'."""
    if val >= 0.1:
        return f"{val:.2f}"
    return f"{val:.2f}"


def format_gamma(val: float) -> str:
    """Format gamma for LaTeX: 1.0 -> '1.0', 0.995 -> '0.995'."""
    if val == 1.0:
        return "1.0"
    return f"{val:.3f}"


def build_command_set(
    sweep_data: Dict[str, Any],
    best_data: Dict[str, Any],
) -> CommandSet:
    """Build the full CommandSet from JSON data."""
    cs = CommandSet(prefix="hp")

    grid = sweep_data.get("grid", {})
    best_val = best_data.get("best_per_variant_val", {})
    auc_only = best_data.get("auc_only_best", {})
    test_per = best_data.get("test_per_variant", {})

    # -------------------------------------------------------------------------
    # Grid dimensions
    # -------------------------------------------------------------------------
    alpha_vals = grid.get("alpha_values", [])
    n_eff_vals = grid.get("n_eff_values", [])
    gamma_vals = grid.get("gamma_values", [])
    epsilon = grid.get("epsilon", 0.05)

    cs.raw("GridAlpha", str(len(alpha_vals)))
    cs.raw("GridNeff", str(len(n_eff_vals)))
    cs.raw("GridGamma", str(len(gamma_vals)))
    cs.raw("Epsilon", str(epsilon))

    # -------------------------------------------------------------------------
    # ParetoBandit selected config (best_per_variant_val)
    # -------------------------------------------------------------------------
    bg_best = best_val.get("paretobandit", {})
    if bg_best:
        cs.raw("ParetoBanditAlpha", format_alpha(bg_best["alpha"]))
        cs.raw("ParetoBanditNeff", fmt_int(bg_best["n_eff"]))
        cs.raw("ParetoBanditGamma", format_gamma(bg_best["gamma"]))
        cs.num("ParetoBanditAUC", bg_best["val_pareto_auc"], digits=3)
        cs.num("ParetoBanditRegret", bg_best["val_phase2_regret"], digits=1)
    bg_test = test_per.get("paretobandit", {})
    if bg_test:
        cs.num("ParetoBanditTestAUC", bg_test["test_pareto_auc"], digits=4)
        cs.num("ParetoBanditTestStd", bg_test["test_pareto_auc_std"], digits=4)
        cs.num("ParetoBanditTestDelta", bg_test["test_delta_pct"], digits=2)

    tr_test = test_per.get("tabula_rasa", {})
    if tr_test:
        cs.num("TabulaTestAUC", tr_test["test_pareto_auc"], digits=4)
        cs.num("TabulaTestStd", tr_test["test_pareto_auc_std"], digits=4)
        cs.num("TabulaTestDelta", tr_test["test_delta_pct"], digits=2)

    if bg_test:
        cs.num("FixedTestAUC", bg_test["test_fixed_auc"], digits=4)

    # -------------------------------------------------------------------------
    # ParetoBandit AUC-only config
    # -------------------------------------------------------------------------
    bg_auc = auc_only.get("paretobandit", {})
    if bg_auc:
        cs.raw("AUCOnlyAlpha", format_alpha(bg_auc["alpha"]))
        cs.raw("AUCOnlyNeff", fmt_int(bg_auc["n_eff"]))
        cs.raw("AUCOnlyGamma", format_gamma(bg_auc["gamma"]))
        cs.num("AUCOnlyAUC", bg_auc["val_pareto_auc"], digits=3)

    # -------------------------------------------------------------------------
    # AUC sacrifice: (auc_only - selected) / auc_only * 100
    # -------------------------------------------------------------------------
    if bg_auc and bg_best:
        auc_only_val = bg_auc["val_pareto_auc"]
        selected_val = bg_best["val_pareto_auc"]
        if auc_only_val > 0:
            sacrifice_pct = (auc_only_val - selected_val) / auc_only_val * 100
            cs.num("AUCSacrifice", sacrifice_pct, digits=2)

    # -------------------------------------------------------------------------
    # Tabula Rasa selected config
    # -------------------------------------------------------------------------
    tr_best = best_val.get("tabula_rasa", {})
    if tr_best:
        cs.raw("TabulaAlpha", format_alpha(tr_best["alpha"]))
        cs.raw("TabulaGamma", format_gamma(tr_best["gamma"]))
        cs.num("TabulaAUC", tr_best["val_pareto_auc"], digits=3)
        cs.num("TabulaRegret", tr_best["val_phase2_regret"], digits=1)

    # -------------------------------------------------------------------------
    # Tabula Rasa AUC-only config
    # -------------------------------------------------------------------------
    tr_auc = auc_only.get("tabula_rasa", {})
    if tr_auc:
        cs.raw("TabulaAUCOnlyAlpha", format_alpha(tr_auc["alpha"]))
        cs.raw("TabulaAUCOnlyGamma", format_gamma(tr_auc["gamma"]))
        cs.num("TabulaAUCOnlyAUC", tr_auc["val_pareto_auc"], digits=3)

    return cs


def main() -> None:
    """Load JSON files, emit _autogen.tex."""
    exp_dir = Path(__file__).resolve().parent
    sweep_path = exp_dir / "results" / "hparam_sweep_results.json"
    best_path = exp_dir / "results" / "best_hparams.json"

    if not sweep_path.exists():
        print(f"Error: {sweep_path} not found.")
        sys.exit(1)
    if not best_path.exists():
        print(f"Error: {best_path} not found.")
        sys.exit(1)

    sweep_data = load_json(sweep_path)
    best_data = load_json(best_path)

    cs = build_command_set(sweep_data, best_data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 05: hyperparameter optimization")


if __name__ == "__main__":
    main()
