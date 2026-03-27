#!/usr/bin/env python3
"""Reproduce all experiments, autogen LaTeX, and figures.

Executes every experiment in dependency order, then regenerates the
LaTeX macros and publication figures that the paper consumes.

Dependency graph
----------------
The hparam sweep must run first — its output (``best_hparams.json``)
feeds ``src/pareto_bandit/config/__init__.py``, which all downstream
experiments import.  The remaining experiments are independent of one
another and can conceptually run in any order.

Pipeline stages (per experiment)
--------------------------------
1. ``run_*.py``          — run simulation, write results JSON
2. ``generate_latex.py`` — read JSON, emit ``_autogen.tex``
3. ``generate_figure.py``— read JSON, emit PNG/PDF figures

Usage
-----
Full reproduction (all stages)::

    python experiments/reproduce.py

Regenerate LaTeX + figures only (skip expensive simulations)::

    python experiments/reproduce.py --skip-run

Single experiment::

    python experiments/reproduce.py --only 01_stationary_budget_pacing

List available experiments::

    python experiments/reproduce.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass
class Experiment:
    """One reproducible experiment with its pipeline scripts."""

    key: str
    directory: Path
    run_scripts: List[str] = field(default_factory=list)
    generate_latex: Optional[str] = None
    generate_figures: List[str] = field(default_factory=list)
    description: str = ""


EXPERIMENTS: List[Experiment] = [
    Experiment(
        key="hparam_optimization",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "hparam_optimization",
        run_scripts=["run_hparam_sweep.py"],
        generate_latex="generate_latex.py",
        description="Hyperparameter sweep (Pareto knee-point selection)",
    ),
    Experiment(
        key="cost_heuristic_validation",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "cost_heuristic_validation",
        run_scripts=["run_cost_heuristic_validation.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Cost heuristic validation (Appendix)",
    ),
    Experiment(
        key="01_stationary_budget_pacing",
        directory=PROJECT_ROOT / "experiments" / "01_stationary_budget_pacing",
        run_scripts=["run_budget_pacing.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Exp 1: Stationary budget pacing",
    ),
    Experiment(
        key="02_budget_plus_drift",
        directory=PROJECT_ROOT / "experiments" / "02_budget_plus_drift",
        run_scripts=["run_budget_cost_drift.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Exp 2: Budget pacing under cost drift",
    ),
    Experiment(
        key="03_catastrophic_failure",
        directory=PROJECT_ROOT / "experiments" / "03_catastrophic_failure",
        run_scripts=["run_catastrophic_failure.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Exp 3: Catastrophic failure response",
    ),
    Experiment(
        key="04_model_onboarding",
        directory=PROJECT_ROOT / "experiments" / "04_model_onboarding",
        run_scripts=["run_model_onboarding.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Exp 4: Model onboarding (K=3 -> K=4)",
    ),
    Experiment(
        key="warmup_ablation",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "warmup_ablation",
        run_scripts=["run_warmup_ablation.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Appendix: Warmup vs cold-start ablation",
    ),
    Experiment(
        key="prior_mismatch",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "prior_mismatch",
        run_scripts=["run_prior_mismatch.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Appendix: Prior mismatch sensitivity",
    ),
    Experiment(
        key="judge_robustness",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "judge_robustness",
        run_scripts=["run_cross_judge_regret.py", "generate_figure.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_cross_judge_figure.py"],
        description="Appendix: Cross-judge regret comparison",
    ),
    Experiment(
        key="recovery_limit",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "recovery_limit",
        run_scripts=["run_recovery_limit.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Appendix: Recovery limit analysis",
    ),
    Experiment(
        key="latency_benchmark",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "latency_benchmark",
        # run_inference_latency_benchmark.py is excluded: it makes live
        # OpenRouter API calls (requires keys + costs money).  Its output
        # (inference_latency_results.json) is consumed by generate_latex.py
        # when present but gracefully skipped otherwise.
        run_scripts=["run_latency_benchmark.py", "run_e2e_latency_benchmark.py"],
        generate_latex="generate_latex.py",
        generate_figures=["generate_figure.py"],
        description="Appendix: Latency microbenchmark",
    ),
    Experiment(
        key="t_adapt_sensitivity",
        directory=PROJECT_ROOT / "experiments" / "appendix" / "hparam_optimization",
        run_scripts=["run_t_adapt_sensitivity.py"],
        description="Appendix: T_adapt sensitivity (no separate autogen)",
    ),
]


def _run(cmd: List[str], cwd: Path, label: str) -> bool:
    """Run a subprocess, return True on success."""
    print(f"  [{label}] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        return False
    print(f"  OK ({elapsed:.1f}s)")
    return True


def run_experiment(exp: Experiment, *, skip_run: bool = False) -> bool:
    """Execute one experiment's full pipeline.

    Args:
        exp: Experiment descriptor.
        skip_run: If True, skip the simulation and only regenerate
            LaTeX/figures from existing result JSONs.

    Returns:
        True if all stages succeeded.
    """
    print(f"\n{'=' * 60}")
    print(f"  {exp.key}: {exp.description}")
    print(f"  {exp.directory}")
    print(f"{'=' * 60}")

    ok = True

    if not skip_run:
        for script in exp.run_scripts:
            path = exp.directory / script
            if not path.exists():
                print(f"  SKIP (not found): {script}")
                continue
            if not _run([PYTHON, script], exp.directory, "run"):
                ok = False
                return ok

    if exp.generate_latex:
        path = exp.directory / exp.generate_latex
        if path.exists():
            if not _run([PYTHON, exp.generate_latex], exp.directory, "latex"):
                ok = False

    for fig_script in exp.generate_figures:
        path = exp.directory / fig_script
        if path.exists():
            if not _run([PYTHON, fig_script], exp.directory, "figure"):
                ok = False

    return ok


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reproduce all experiments and regenerate paper artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip simulations; only regenerate LaTeX and figures from existing JSONs.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the experiment matching this key (use --list to see keys).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'Key':<35s} Description")
        print("-" * 70)
        for exp in EXPERIMENTS:
            print(f"  {exp.key:<33s} {exp.description}")
        return

    targets = EXPERIMENTS
    if args.only:
        targets = [e for e in EXPERIMENTS if e.key == args.only]
        if not targets:
            print(f"Error: unknown experiment key '{args.only}'")
            print(f"Available keys: {', '.join(e.key for e in EXPERIMENTS)}")
            sys.exit(1)

    t_start = time.time()
    results = []
    for exp in targets:
        ok = run_experiment(exp, skip_run=args.skip_run)
        results.append((exp.key, ok))

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    for key, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  {key:<35s} {status}")

    n_failed = sum(1 for _, ok in results if not ok)
    if n_failed:
        print(f"\n{n_failed} experiment(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} experiments succeeded.")


if __name__ == "__main__":
    main()
