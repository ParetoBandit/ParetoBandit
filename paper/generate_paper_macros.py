"""Generate paper-level derived macros from experiment JSON results.

Reads experiment JSON files and emits ``paper_macros_autogen.tex``
with ``\\pp``-prefixed commands for headline numbers used in
abstract.tex, introduction.tex, conclusion.tex, evaluation.tex,
and system_design.tex.

Usage::

    python paper/generate_paper_macros.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict

PAPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import CommandSet, fmt_cost_sci, fmt_int, fmt_num, load_json


def build_command_set() -> CommandSet:
    """Build paper-level derived macros from all experiment results."""
    cs = CommandSet(prefix="pp")

    # ------------------------------------------------------------------
    # Budget pacing (Exp 01): price range, compliance
    # ------------------------------------------------------------------
    bp_path = PROJECT_ROOT / "experiments" / "01_stationary_budget_pacing" / "results" / "budget_pacing_results.json"
    if bp_path.exists():
        bp = load_json(bp_path)
        results = bp.get("results", [])

        fixed = [r for r in results if r.get("method") == "fixed_model"]
        if fixed:
            cheapest = min(fixed, key=lambda r: r["mean_cost"])
            dearest = max(fixed, key=lambda r: r["mean_cost"])
            if cheapest["mean_cost"] > 0:
                price_range = dearest["mean_cost"] / cheapest["mean_cost"]
                cs.raw("PriceRangeX", fmt_int(round(price_range)))

        hp = bp.get("warmup_hparams", {})
        if hp:
            gamma = hp.get("forgetting_factor", 0.996)
            cs.num("Gamma", gamma, digits=3)
            cs.num("Alpha", hp.get("alpha", 0.01), digits=2)
            neff = hp.get("prior_n_effective", 0.0)
            cs.raw("Neff", fmt_int(neff))
            if gamma < 1.0:
                eff_mem = round(1.0 / (1.0 - gamma))
                half_life = round(math.log(2) / (1.0 - gamma))
                cs.raw("EffMemSteps", str(eff_mem))
                cs.raw("HalfLife", str(half_life))
                efold_1000 = gamma ** 1000
                cs.num("GammaDecayOneK", efold_1000, digits=2)

        budget_targets = bp.get("budget_targets", [])
        pacer_results = [r for r in results if r.get("method") == "pacer"]
        max_overshoot_pct = 0.0
        for pr in pacer_results:
            u = pr.get("budget_utilization")
            if u is not None and u > 1.0:
                overshoot = (u - 1.0) * 100
                if overshoot > max_overshoot_pct:
                    max_overshoot_pct = overshoot
        cs.num("MaxBudgetOvershootPct", max_overshoot_pct, digits=1)

    # ------------------------------------------------------------------
    # Budget + drift (Exp 02): quality lift
    # ------------------------------------------------------------------
    bd_path = PROJECT_ROOT / "experiments" / "02_budget_plus_drift" / "results" / "budget_cost_drift_results.json"
    if bd_path.exists():
        bd = load_json(bd_path)
        conditions = bd.get("conditions", {})
        budget_labels = bd.get("budget_labels", [])

        max_lift = 0.0
        for label in budget_labels:
            key = f"ParetoBandit ({label})"
            cond = conditions.get(key, {})
            p1 = cond.get("phase1_summary", {})
            p2 = cond.get("phase2_summary", {})
            lift = p2.get("mean_reward", 0) - p1.get("mean_reward", 0)
            if lift > max_lift:
                max_lift = lift
        cs.num("MaxQualityLift", max_lift, digits=3)

    # ------------------------------------------------------------------
    # Catastrophic failure (Exp 03): overshoot
    # ------------------------------------------------------------------
    cf_path = PROJECT_ROOT / "experiments" / "03_catastrophic_failure" / "results" / "catastrophic_failure_results.json"
    if cf_path.exists():
        cf = load_json(cf_path)
        conditions = cf.get("conditions", {})
        budget_targets = cf.get("budget_targets", [])
        budget_labels = cf.get("budget_labels", [])

        max_overshoot = 0.0
        for target, label in zip(budget_targets, budget_labels):
            for cond_name in ("Forgetting Bandit", "Naive Bandit"):
                key = f"{cond_name} ({label})"
                cond = conditions.get(key, {})
                for phase_num in (1, 2, 3):
                    pdata = cond.get(f"phase{phase_num}_summary", {})
                    mc = pdata.get("mean_cost", 0)
                    if target > 0:
                        ratio = mc / target
                        if ratio > max_overshoot:
                            max_overshoot = ratio
        cs.num("MaxOvershootX", max_overshoot, digits=1)

    # ------------------------------------------------------------------
    # Model onboarding (Exp 04): adoption steps
    # ------------------------------------------------------------------
    mo_path = PROJECT_ROOT / "experiments" / "04_model_onboarding" / "results" / "model_onboarding_results.json"
    if mo_path.exists():
        mo = load_json(mo_path)
        scenarios = mo.get("scenarios", {})
        gc = scenarios.get("good_cheap", {}).get("summaries", {})

        max_sustained_step = 0
        for skey, sdata in gc.items():
            if "paretobandit" in skey:
                adoption = sdata.get("flash_adoption", {})
                step = adoption.get("mean_sustained_step", 0)
                if step > max_sustained_step:
                    max_sustained_step = step
        if max_sustained_step > 0:
            cs.raw("AdoptionSteps", fmt_int(max_sustained_step))

    # ------------------------------------------------------------------
    # Latency (E2E)
    # ------------------------------------------------------------------
    e2e_path = PROJECT_ROOT / "experiments" / "appendix" / "latency_benchmark" / "results" / "e2e_latency_results.json"
    if e2e_path.exists():
        e2e = load_json(e2e_path)
        stages = e2e.get("stages", {})
        cs.num("EteTotalMedianMs", stages.get("total_p50_ms", 0), digits=1)
        cs.num("EteRoutePctOfTotal", e2e.get("fractions", {}).get("route_pct_of_total_p50", 0), digits=0)

    inf_path = PROJECT_ROOT / "experiments" / "appendix" / "latency_benchmark" / "results" / "inference_latency_results.json"
    if e2e_path.exists() and inf_path.exists():
        inf = load_json(inf_path)
        e2e_total = stages.get("total_p50_ms", 0)
        if e2e_total > 0:
            max_pct = 0.0
            for model_results in inf.get("results", {}).values():
                for length_results in model_results.values():
                    total_inf = length_results.get("total_ms", {}).get("mean", 0)
                    if total_inf > 0:
                        pct = e2e_total / total_inf * 100
                        if pct > max_pct:
                            max_pct = pct
            if max_pct > 0:
                cs.num("EteMaxPctOfInference", math.ceil(max_pct * 10) / 10, digits=1)

    bench_path = PROJECT_ROOT / "experiments" / "appendix" / "latency_benchmark" / "results" / "latency_benchmark_results.json"
    if bench_path.exists():
        bench = load_json(bench_path)
        for r in bench.get("results", []):
            name = r.get("name", r.get("label", ""))
            if name == "ParetoBandit (d=26)":
                cs.num("PBRouteMedianUs", r["route_p50_us"], digits=1)
                cs.raw("PBTotalMedianUs", fmt_int(round(r["total_p50_us"])))
                cs.raw("PBThroughput", f"{r['throughput_rps'] / 1000:,.0f}{{,}}000")
        pb26_thpt = None
        cached385_thpt = None
        for r2 in bench.get("results", []):
            n2 = r2.get("name", r2.get("label", ""))
            if n2 == "ParetoBandit (d=26)":
                pb26_thpt = r2["throughput_rps"]
            elif n2 == "Cached Inv. (d=385)":
                cached385_thpt = r2["throughput_rps"]
        if pb26_thpt and cached385_thpt and cached385_thpt > 0:
            pca_speedup = pb26_thpt / cached385_thpt
            cs.raw("PcaThroughputGainX", fmt_int(round(pca_speedup)))

    # ------------------------------------------------------------------
    # Recovery limit / exp03 degradation severity
    # ------------------------------------------------------------------
    cf_path2 = PROJECT_ROOT / "experiments" / "03_catastrophic_failure" / "results" / "catastrophic_failure_results.json"
    rl_path = PROJECT_ROOT / "experiments" / "appendix" / "recovery_limit" / "results" / "recovery_limit_results.json"
    if cf_path2.exists() and rl_path.exists():
        cf2 = load_json(cf_path2)
        rl = load_json(rl_path)
        normal_reward = rl.get("mistral_normal_reward", 0.918)
        failure_reward = cf2.get("failure_reward", 0.75)
        if normal_reward > 0:
            exp03_deg_pct = (1.0 - failure_reward / normal_reward) * 100
            cs.raw("CfDegPct", fmt_int(round(exp03_deg_pct)))

    return cs


def main() -> None:
    """Emit ``paper_macros_autogen.tex``."""
    cs = build_command_set()
    autogen_path = PAPER_DIR / "paper_macros_autogen.tex"
    cs.write(autogen_path, header="Paper-level derived macros")


if __name__ == "__main__":
    main()
