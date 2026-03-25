"""Generate LaTeX commands from hyperparameter optimization results.

Reads results/hparam_sweep_results.json and results/best_hparams.json,
emits _autogen.tex with \\newcommand definitions for the paper.

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import CommandSet, fmt_int, fmt_num, load_json


def format_alpha(val: float) -> str:
    """Format alpha for LaTeX: 0.1 -> '0.10', 0.01 -> '0.01'."""
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
    cross_arm = best_data.get("cross_arm_validation", {})

    # -------------------------------------------------------------------------
    # Grid dimensions and T_adapt
    # -------------------------------------------------------------------------
    alpha_vals = grid.get("alpha_values", [])
    gamma_vals = grid.get("gamma_values", [])
    t_adapt = grid.get("t_adapt", best_data.get("t_adapt", 500))

    cs.raw("GridAlpha", str(len(alpha_vals)))
    cs.raw("GridGamma", str(len(gamma_vals)))
    cs.raw("TAdapt", str(t_adapt))
    cs.raw("SelectionMethod", "pareto\\_knee\\_point")

    # -------------------------------------------------------------------------
    # ParetoBandit selected config (knee-point)
    # -------------------------------------------------------------------------
    pb_best = best_val.get("paretobandit", {})
    if pb_best:
        cs.raw("ParetoBanditAlpha", format_alpha(pb_best["alpha"]))
        cs.raw("ParetoBanditNeff", fmt_int(pb_best["n_eff"]))
        cs.raw("ParetoBanditGamma", format_gamma(pb_best["gamma"]))
        cs.num("ParetoBanditAUC", pb_best["val_pareto_auc"], digits=3)
        cs.num("ParetoBanditPTwoReward", pb_best["val_phase2_reward"], digits=4)

    pb_test = test_per.get("paretobandit", {})
    if pb_test:
        cs.num("ParetoBanditTestAUC", pb_test["test_pareto_auc"], digits=4)
        cs.num("ParetoBanditTestStd", pb_test["test_pareto_auc_std"], digits=4)
        cs.num("ParetoBanditTestDelta", pb_test["test_delta_pct"], digits=2)

    # -------------------------------------------------------------------------
    # Tabula Rasa selected config (knee-point)
    # -------------------------------------------------------------------------
    tr_best = best_val.get("tabula_rasa", {})
    if tr_best:
        cs.raw("TabulaAlpha", format_alpha(tr_best["alpha"]))
        cs.raw("TabulaGamma", format_gamma(tr_best["gamma"]))
        cs.num("TabulaAUC", tr_best["val_pareto_auc"], digits=3)
        cs.num("TabulaPTwoReward", tr_best["val_phase2_reward"], digits=4)

    tr_test = test_per.get("tabula_rasa", {})
    if tr_test:
        cs.num("TabulaTestAUC", tr_test["test_pareto_auc"], digits=4)
        cs.num("TabulaTestStd", tr_test["test_pareto_auc_std"], digits=4)
        cs.num("TabulaTestDelta", tr_test["test_delta_pct"], digits=2)

    # -------------------------------------------------------------------------
    # Fixed-model baseline (test)
    # -------------------------------------------------------------------------
    if pb_test:
        cs.num("FixedTestAUC", pb_test["test_fixed_auc"], digits=4)

    # -------------------------------------------------------------------------
    # ParetoBandit AUC-only config
    # -------------------------------------------------------------------------
    pb_auc = auc_only.get("paretobandit", {})
    if pb_auc:
        cs.raw("AUCOnlyAlpha", format_alpha(pb_auc["alpha"]))
        cs.raw("AUCOnlyNeff", fmt_int(pb_auc["n_eff"]))
        cs.raw("AUCOnlyGamma", format_gamma(pb_auc["gamma"]))
        cs.num("AUCOnlyAUC", pb_auc["val_pareto_auc"], digits=3)

    # -------------------------------------------------------------------------
    # Tabula Rasa AUC-only config
    # -------------------------------------------------------------------------
    tr_auc = auc_only.get("tabula_rasa", {})
    if tr_auc:
        cs.raw("TabulaAUCOnlyAlpha", format_alpha(tr_auc["alpha"]))
        cs.raw("TabulaAUCOnlyGamma", format_gamma(tr_auc["gamma"]))
        cs.num("TabulaAUCOnlyAUC", tr_auc["val_pareto_auc"], digits=3)

    # -------------------------------------------------------------------------
    # AUC sacrifice: (auc_only - selected) / auc_only * 100
    # -------------------------------------------------------------------------
    if pb_auc and pb_best:
        auc_only_val = pb_auc["val_pareto_auc"]
        selected_val = pb_best["val_pareto_auc"]
        if auc_only_val > 0:
            sacrifice_pct = (auc_only_val - selected_val) / auc_only_val * 100
            cs.num("AUCSacrifice", sacrifice_pct, digits=2)

    # -------------------------------------------------------------------------
    # Cross-arm validation — val split (Phase-2 reward per failed arm)
    # -------------------------------------------------------------------------
    pb_cross = cross_arm.get("paretobandit", {})
    for short_name, key_suffix in [
        ("Llama-8B", "Llama"),
        ("Mistral-Large", "Mistral"),
        ("Gemini-Pro", "Gemini"),
    ]:
        arm_data = pb_cross.get(short_name, {})
        if arm_data:
            cs.num(
                f"CrossArm{key_suffix}PTwo",
                arm_data["phase2_reward"],
                digits=4,
            )
            cs.num(
                f"CrossArm{key_suffix}Std",
                arm_data["phase2_reward_std"],
                digits=4,
            )

    # -------------------------------------------------------------------------
    # Cross-arm validation — held-out TEST split
    # -------------------------------------------------------------------------
    cross_arm_test = best_data.get("cross_arm_validation_test", {})
    pb_cross_test = cross_arm_test.get("paretobandit", {})
    for short_name, key_suffix in [
        ("Llama-8B", "Llama"),
        ("Mistral-Large", "Mistral"),
        ("Gemini-Pro", "Gemini"),
    ]:
        arm_data = pb_cross_test.get(short_name, {})
        if arm_data:
            cs.num(
                f"TestCrossArm{key_suffix}PTwo",
                arm_data["phase2_reward"],
                digits=4,
            )
            cs.num(
                f"TestCrossArm{key_suffix}Std",
                arm_data["phase2_reward_std"],
                digits=4,
            )

    # -------------------------------------------------------------------------
    # Bootstrap knee-point stability
    # -------------------------------------------------------------------------
    boot_data = best_data.get("bootstrap_knee_stability", {})
    pb_boot = boot_data.get("paretobandit", {})
    if pb_boot:
        cs.num("BootKneeFreq", pb_boot["original_knee_frequency"] * 100, digits=1)
        cs.num("BootNeighborFreq", pb_boot["neighborhood_frequency"] * 100, digits=1)
        cs.raw("BootNUnique", str(pb_boot["n_unique_selections"]))
        cs.raw("BootNIter", str(pb_boot["n_bootstrap"]))

    tr_boot = boot_data.get("tabula_rasa", {})
    if tr_boot:
        cs.num("TabulaBootKneeFreq", tr_boot["original_knee_frequency"] * 100, digits=1)
        cs.num("TabulaBootNeighborFreq", tr_boot["neighborhood_frequency"] * 100, digits=1)

    # -------------------------------------------------------------------------
    # T_adapt sensitivity (if results exist)
    # -------------------------------------------------------------------------
    _T_ADAPT_TAG: Dict[str, str] = {"250": "Lo", "500": "Mid", "1000": "Hi"}
    sensitivity_path = Path(__file__).resolve().parent / "results" / "t_adapt_sensitivity.json"
    if sensitivity_path.exists():
        sens_data = load_json(sensitivity_path)
        per_t = sens_data.get("per_t_adapt", {})
        for t_val_str, t_result in per_t.items():
            t_tag = _T_ADAPT_TAG.get(t_val_str.replace(".", ""), t_val_str.replace(".", ""))
            cs.raw(f"Sens{t_tag}Alpha", format_alpha(t_result["alpha"]))
            cs.raw(f"Sens{t_tag}Gamma", format_gamma(t_result["gamma"]))
            cs.raw(f"Sens{t_tag}Neff", fmt_int(t_result["n_eff"]))
            cs.num(f"Sens{t_tag}AUC", t_result["val_pareto_auc"], digits=4)
            cs.num(f"Sens{t_tag}PTwo", t_result["val_phase2_reward"], digits=4)
        if "stable" in sens_data:
            cs.raw("SensStable", "yes" if sens_data["stable"] else "no")

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
