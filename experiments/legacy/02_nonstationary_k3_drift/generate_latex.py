"""Generate LaTeX commands and figure caption from reward-shift experiment results.

Reads results/reward_shift_results.json and emits:
- _autogen.tex: \\newcommand definitions for the paper (prefix \\rs)
- figure_regret_caption.tex: figure environment with caption using the commands

Run from the experiment directory: python generate_latex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import CommandSet, fmt_int, fmt_num

# Condition name mapping: JSON key -> short prefix for commands
CONDITION_MAP = {
    "Fixed Policy (offline)": "Fixed",
    "Naive Bandit (γ=1.0)": "Naive",
    "SW-UCB (W=200)": "SWUCB",
    "ParetoBandit (γ=0.995)": "ParetoBandit",
}

MISTRAL_ARM = "Mistral-Large"


def load_results(json_path: Path) -> Dict[str, Any]:
    """Load reward-shift results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def find_checkpoint(checkpoints: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    """Return the checkpoint dict for the given step, or None if not found."""
    for cp in checkpoints:
        if cp.get("step") == step:
            return cp
    return None


def compute_phase_regrets(
    checkpoints: List[Dict[str, Any]],
    phase_boundary: int,
) -> tuple[float, float]:
    """Compute Phase 1 and Phase 2 regret from exact checkpoint at the phase boundary.

    Requires that ``phase_boundary`` is an explicit checkpoint in the JSON
    (i.e. ``run_reward_shift.py`` includes ``n_p1`` in its checkpoint set).

    Phase 1 = regret at step ``phase_boundary``.
    Phase 2 = terminal regret minus Phase 1.
    """
    cp_boundary = find_checkpoint(checkpoints, phase_boundary)
    if cp_boundary is None:
        raise ValueError(
            f"No checkpoint at phase boundary step {phase_boundary}. "
            f"Re-run run_reward_shift.py to generate a checkpoint at the "
            f"exact phase boundary (n_p1 is now included in the checkpoint set)."
        )
    phase1 = cp_boundary["mean_cumulative_regret"]

    terminal_step = max(cp["step"] for cp in checkpoints)
    cp_term = find_checkpoint(checkpoints, terminal_step)
    assert cp_term is not None
    total = cp_term["mean_cumulative_regret"]
    phase2 = total - phase1
    return phase1, phase2


def extract_condition_values(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract terminal and phase-split values for each condition.

    Returns a dict keyed by short name (Fixed, Naive, SWUCB, ParetoBandit) with:
    - total: mean cumulative regret at step 1785
    - std: std_cumulative_regret at step 1785
    - phase1, phase2: interpolated phase regrets
    - arm_fractions: dict at terminal (for Fixed Policy)
    """
    conditions = data.get("conditions", {})
    phase_boundary = data.get("phase1_n")
    if phase_boundary is None:
        raise ValueError("JSON missing 'phase1_n' key — cannot determine phase boundary")
    result: Dict[str, Dict[str, Any]] = {}

    for json_name, short_name in CONDITION_MAP.items():
        checkpoints = conditions.get(json_name)
        if not checkpoints:
            continue

        terminal_step = max(cp["step"] for cp in checkpoints)
        cp_term = find_checkpoint(checkpoints, terminal_step)
        if cp_term is None:
            continue

        phase1, phase2 = compute_phase_regrets(checkpoints, phase_boundary)

        result[short_name] = {
            "total": cp_term["mean_cumulative_regret"],
            "std": cp_term.get("std_cumulative_regret", 0.0),
            "phase1": phase1,
            "phase2": phase2,
            "arm_fractions": cp_term.get("arm_fractions", {}),
        }

    return result


def build_command_set(data: Dict[str, Any], vals: Dict[str, Dict[str, Any]]) -> CommandSet:
    """Build the full CommandSet from extracted values."""
    cs = CommandSet(prefix="rs")

    # Per-condition terminal values
    for short in ["Fixed", "Naive", "SWUCB", "ParetoBandit"]:
        if short not in vals:
            continue
        v = vals[short]
        cs.num(f"{short}Total", v["total"], digits=1)
        cs.num(f"{short}Std", v["std"], digits=1)

    # Per-condition phase-split regret
    for short in ["Fixed", "Naive", "SWUCB", "ParetoBandit"]:
        if short not in vals:
            continue
        v = vals[short]
        cs.num(f"{short}PhaseOne", v["phase1"], digits=1)
        cs.num(f"{short}PhaseTwo", v["phase2"], digits=1)

    # Derived quantities
    fixed_total = vals.get("Fixed", {}).get("total")
    naive_total = vals.get("Naive", {}).get("total")
    swucb_total = vals.get("SWUCB", {}).get("total")
    bg_total = vals.get("ParetoBandit", {}).get("total")
    naive_p2 = vals.get("Naive", {}).get("phase2")
    bg_p2 = vals.get("ParetoBandit", {}).get("phase2")
    swucb_p1 = vals.get("SWUCB", {}).get("phase1")
    swucb_std = vals.get("SWUCB", {}).get("std")
    bg_std = vals.get("ParetoBandit", {}).get("std")

    if fixed_total is not None and naive_total is not None and fixed_total > 0:
        reduction = (fixed_total - naive_total) / fixed_total * 100
        cs.raw("ReductionFixedNaive", fmt_int(reduction))

    if naive_total is not None and bg_total is not None and naive_total > 0:
        reduction = (naive_total - bg_total) / naive_total * 100
        cs.raw("ReductionNaiveParetoBandit", fmt_int(reduction))

    if naive_p2 is not None and bg_p2 is not None and bg_p2 > 0:
        excess = (naive_p2 - bg_p2) / bg_p2 * 100
        cs.raw("NaivePhaseTwoExcess", fmt_int(excess))

    if swucb_p1 is not None:
        prior_p1_values = [
            v for v in [
                vals.get("Fixed", {}).get("phase1"),
                vals.get("Naive", {}).get("phase1"),
                vals.get("ParetoBandit", {}).get("phase1"),
            ] if v is not None and v > 0
        ]
        if prior_p1_values:
            min_p1 = min(prior_p1_values)
            factor = swucb_p1 / min_p1
            cs.num("SWUCBPhaseOneFactor", factor, digits=1)

    if swucb_std is not None and bg_std is not None and bg_std > 0:
        var_factor = swucb_std / bg_std
        cs.num("SWUCBVarFactor", var_factor, digits=1)

    if naive_p2 is not None and bg_p2 is not None and naive_p2 > 0:
        reduction_p2 = (naive_p2 - bg_p2) / naive_p2 * 100
        cs.raw("ParetoBanditPhaseTwoReduction", fmt_int(reduction_p2))

    # Fixed Policy arm fractions at terminal
    fixed_af = vals.get("Fixed", {}).get("arm_fractions", {})
    mistral_frac = fixed_af.get(MISTRAL_ARM, 0.0)
    cs.raw("FixedMistralPct", fmt_int(mistral_frac * 100))

    return cs


def generate_figure_caption() -> str:
    """Generate the figure_regret_caption.tex content.

    Uses \\rs commands; the caption must be written after _autogen.tex is
    included so the commands are defined. The caption references all four
    conditions and the auto-generated numbers.
    """
    return r"""\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\linewidth]{../experiments/02_nonstationary_k3_drift/results/cumulative_regret.pdf}
    \caption{Cumulative cost-adjusted regret under model quality shift
    ($K{=}3$; 40~seeds, 95\% bootstrap CI shading).
    %
    Four conditions of increasing sophistication are compared:
    \textbf{Fixed Policy (offline)} deploys warmup priors without
    online learning.
    \textbf{Naive Bandit ($\gamma{=}1.0$)} adds online LinUCB
    with infinite memory.
    \textbf{SW-UCB ($W{=}200$)} uses sliding-window LinUCB without
    warmup priors.
    \textbf{ParetoBandit ($\gamma{=}0.995$)} adds geometric
    forgetting with a tuned effective memory of ${\sim}200$ steps.
    %
    Total regret: Fixed~\rsFixedTotal{},
    Naive~\rsNaiveTotal{} ($-$\rsReductionFixedNaive\%),
    SW-UCB~\rsSWUCBTotal{},
    ParetoBandit~\rsParetoBanditTotal{} ($-$\rsReductionNaiveParetoBandit\%
    vs.\ Naive).
    All pairwise comparisons are significant after
    Holm--Bonferroni correction (paired $t$-tests, 39~d.f.).
    }
    \label{fig:cumulative_regret}
\end{figure*}
"""


def main() -> None:
    """Load JSON, emit _autogen.tex and figure_regret_caption.tex."""
    exp_dir = Path(__file__).resolve().parent
    json_path = exp_dir / "results" / "reward_shift_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        sys.exit(1)

    data = load_results(json_path)
    vals = extract_condition_values(data)
    cs = build_command_set(data, vals)

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Exp 02: reward shift (K=3 drift)")

    caption_path = exp_dir / "figure_regret_caption.tex"
    caption_content = generate_figure_caption()
    caption_path.write_text(caption_content)
    print(f"  Wrote figure caption → {caption_path}")


if __name__ == "__main__":
    main()
