"""Generate LaTeX commands from recovery limit experiment results.

Reads ``results/recovery_limit_results.json`` and emits:

- ``_autogen.tex``: ``\\newcommand`` definitions (prefix ``\\rl``).

Usage::

    python experiments/appendix/recovery_limit/generate_latex.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.bootstrap import bootstrap_ci
from utils.latex_gen import CommandSet, ci_from_seeds_or_normal, fmt_cost_sci, fmt_int, fmt_num, load_json

FULL_RECOVERY_THRESHOLD = 0.97

def _find_recovery_boundary(
    results: List[Dict[str, Any]],
    threshold: float = FULL_RECOVERY_THRESHOLD,
) -> Optional[float]:
    """Find the degradation % at which P3/P1 crosses below *threshold*.

    Uses linear interpolation between the two adjacent sweep points.
    Results are sorted by degradation_pct ascending (mild to severe).
    """
    sorted_res = sorted(results, key=lambda e: e["degradation_pct"])
    for i in range(len(sorted_res) - 1):
        r_curr = sorted_res[i]["p3_p1_ratio"]
        r_next = sorted_res[i + 1]["p3_p1_ratio"]
        if r_curr >= threshold >= r_next:
            d_curr = sorted_res[i]["degradation_pct"]
            d_next = sorted_res[i + 1]["degradation_pct"]
            frac = (r_curr - threshold) / (r_curr - r_next) if r_curr != r_next else 0.0
            return d_curr + frac * (d_next - d_curr)
    return None


def _find_floor(results: List[Dict[str, Any]]) -> float:
    """Return the minimum P3/P1 ratio across all severities."""
    return min(e["p3_p1_ratio"] for e in results)


def _find_best_recovery(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Return (degradation_pct, p3_p1_ratio) for the best recovery (mildest degradation)."""
    mildest = min(results, key=lambda e: e["degradation_pct"])
    return mildest["degradation_pct"], mildest["p3_p1_ratio"]


def build_command_set(data: Dict[str, Any]) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="rl")

    cs.raw("Nseeds", fmt_int(data["n_seeds"]))
    cs.raw("PhaseN", fmt_int(data["phase_n"]))
    cs.raw("ExtPhaseThreeN", fmt_int(data.get("extended_phase3_n", data["phase_n"] * 2)))
    cs.raw("BudgetTarget", fmt_cost_sci(data["budget_target"]))
    cs.num("Gamma", data["forgetting_factor"], digits=3)

    gamma = data["forgetting_factor"]
    if gamma < 1.0:
        half_life = round(math.log(2) / (1.0 - gamma))
        cs.raw("HalfLife", str(half_life))

    std = data.get("standard_results", [])
    ext = data.get("extended_results", [])

    exp03_fr = data.get("mistral_normal_reward", 0.89)
    cs.num("MistralNormalReward", exp03_fr, digits=2)

    if std:
        boundary_std = _find_recovery_boundary(std)
        if boundary_std is not None:
            cs.raw("StdRecoveryThresholdPct", fmt_int(round(boundary_std)))

        floor_std = _find_floor(std)
        cs.raw("StdFloorPct", fmt_int(round(floor_std * 100)))

        best_deg, best_ratio = _find_best_recovery(std)
        cs.num("MildDegPct", best_deg, digits=1)
        cs.num("MildRecoveryRatio", best_ratio * 100, digits=1)

        for entry in sorted(std, key=lambda e: e["degradation_pct"]):
            deg_int = round(entry["degradation_pct"])
            ratio_pct = entry["p3_p1_ratio"] * 100
            cs.num(f"StdDeg{deg_int}Recovery", ratio_pct, digits=1)

            phases = entry.get("phases", {})
            p1_seeds = phases.get("phase1", {}).get("per_seed_reward", [])
            p3_seeds = phases.get("phase3", {}).get("per_seed_reward", [])
            ci_lo_pct, ci_hi_pct = None, None
            if p1_seeds and p3_seeds and len(p1_seeds) == len(p3_seeds):
                a1 = np.array(p1_seeds)
                a3 = np.array(p3_seeds)
                ratio_seeds = a3 / np.where(a1 > 0, a1, 1e-12) * 100
                ci_lo_pct, ci_hi_pct = bootstrap_ci(ratio_seeds)
                cs.ci_bounds(f"StdDeg{deg_int}Recovery",
                             ci_lo_pct, ci_hi_pct, digits=1)

            fr = entry.get("failure_reward")
            if fr is not None:
                fr_int = round(fr * 100)
                cs.num(f"StdFR{fr_int}Recovery", ratio_pct, digits=1)
                cs.num(f"StdFR{fr_int}DegPct", entry["degradation_pct"], digits=0)
                if ci_lo_pct is not None:
                    cs.ci_bounds(f"StdFR{fr_int}Recovery",
                                 ci_lo_pct, ci_hi_pct, digits=1)

    if ext:
        boundary_ext = _find_recovery_boundary(ext)
        if boundary_ext is not None:
            cs.raw("ExtRecoveryThresholdPct", fmt_int(round(boundary_ext)))

        floor_ext = _find_floor(ext)
        cs.raw("ExtFloorPct", fmt_int(round(floor_ext * 100)))

        for entry in sorted(ext, key=lambda e: e["degradation_pct"]):
            deg_int = round(entry["degradation_pct"])
            ratio_pct = entry["p3_p1_ratio"] * 100
            cs.num(f"ExtDeg{deg_int}Recovery", ratio_pct, digits=1)

            phases = entry.get("phases", {})
            p1_seeds = phases.get("phase1", {}).get("per_seed_reward", [])
            p3_seeds = phases.get("phase3", {}).get("per_seed_reward", [])
            ci_lo_pct, ci_hi_pct = None, None
            if p1_seeds and p3_seeds and len(p1_seeds) == len(p3_seeds):
                a1 = np.array(p1_seeds)
                a3 = np.array(p3_seeds)
                ratio_seeds = a3 / np.where(a1 > 0, a1, 1e-12) * 100
                ci_lo_pct, ci_hi_pct = bootstrap_ci(ratio_seeds)
                cs.ci_bounds(f"ExtDeg{deg_int}Recovery",
                             ci_lo_pct, ci_hi_pct, digits=1)

            fr = entry.get("failure_reward")
            if fr is not None:
                fr_int = round(fr * 100)
                cs.num(f"ExtFR{fr_int}Recovery", ratio_pct, digits=1)
                if ci_lo_pct is not None:
                    cs.ci_bounds(f"ExtFR{fr_int}Recovery",
                                 ci_lo_pct, ci_hi_pct, digits=1)

    return cs


def main() -> None:
    """Load JSON and emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "recovery_limit_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_json(json_path)
    cs = build_command_set(data)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: recovery limit")


if __name__ == "__main__":
    main()
