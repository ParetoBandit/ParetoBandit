#!/usr/bin/env python3
"""Appendix: Cost Heuristic Validation.

Empirically validates the static log-normalized cost heuristic c_tilde_a
used in the selection utility (Eq. 1) against actual per-request costs
from the offline dataset.

Analyses
--------
1. **Per-model cost distributions**: mean, std, CV, quantiles for each arm.
2. **Cost ranking preservation**: fraction of prompts where the heuristic's
   model-cost ordering holds.  Evaluated pairwise and as a full ordering.
3. **Log-normalized c_tilde space**: computes per-request c_tilde values
   and compares within-model std to inter-model gaps.
4. **Prompt-cost correlation**: Spearman rho between prompt word count
   and per-request cost (tests whether prompt features predict cost).
5. **Cross-model cost correlation**: Spearman rho between models'
   per-request costs (shared output-length factor).

Runs on K=3 (val split) and K=4 (val_k4 split) if available.

Usage:
    python experiments/appendix/cost_heuristic_validation/run_cost_heuristic_validation.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pareto_bandit.config import VAL_DATA_PATH, OFFLINE_DATASET_DIR
from pareto_bandit.costs import log_normalize_cost
from pareto_bandit.types import RouterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

K3_ARMS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]
K4_ARMS = K3_ARMS + ["google/gemini-2.5-flash"]

PRICING: Dict[str, Dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"in": 0.1, "out": 0.1},
    "mistralai/mistral-large-2512": {"in": 0.5, "out": 1.5},
    "google/gemini-2.5-pro": {"in": 1.25, "out": 10.0},
    "google/gemini-2.5-flash": {"in": 0.3, "out": 2.5},
}

ARM_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
    "mistralai/mistral-large-2512": "Mistral-Large",
    "google/gemini-2.5-pro": "Gemini-Pro",
    "google/gemini-2.5-flash": "Gemini-Flash",
}


def _heuristic_c_tilde(arm_id: str) -> float:
    """Compute the static heuristic c_tilde for a model."""
    cfg = RouterConfig()
    p = PRICING[arm_id]
    blended_per_1k = (p["in"] + p["out"]) / 2.0 / 1000.0
    return log_normalize_cost(blended_per_1k, cfg.market_cost_floor, cfg.market_cost_ceiling)


def _per_request_c_tilde(cost_usd: float, arm_id: str) -> float:
    """Convert a per-request cost (USD) to log-normalized c_tilde space.

    Uses the model's actual per-request cost relative to its mean to derive
    an effective per-token rate, then normalizes identically to the heuristic.
    """
    cfg = RouterConfig()
    p = PRICING[arm_id]
    blended_per_1k = (p["in"] + p["out"]) / 2.0 / 1000.0
    if blended_per_1k <= 0:
        return 0.0
    mean_cost_at_500_tokens = blended_per_1k / 1000.0 * 500.0
    if mean_cost_at_500_tokens <= 0:
        return 0.0
    effective_rate = cost_usd / mean_cost_at_500_tokens * blended_per_1k
    return log_normalize_cost(effective_rate, cfg.market_cost_floor, cfg.market_cost_ceiling)


def load_split(path: Path, arms: List[str]) -> Dict[str, Any]:
    """Load a JSONL split and extract per-model costs and prompt texts."""
    prompts: List[str] = []
    costs: Dict[str, List[float]] = {a: [] for a in arms}
    rewards: Dict[str, List[float]] = {a: [] for a in arms}

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])
            for a in arms:
                costs[a].append(rec["arms"][a]["cost"])
                rewards[a].append(rec["arms"][a]["reward"])

    return {
        "prompts": prompts,
        "costs": {a: np.array(v) for a, v in costs.items()},
        "rewards": {a: np.array(v) for a, v in rewards.items()},
        "n": len(prompts),
    }


def analyze_portfolio(
    data: Dict[str, Any],
    arms: List[str],
    label: str,
) -> Dict[str, Any]:
    """Run the full validation analysis for one portfolio configuration."""
    n = data["n"]
    costs = data["costs"]
    prompts = data["prompts"]

    results: Dict[str, Any] = {"label": label, "n_prompts": n, "n_arms": len(arms)}

    # --- 1. Per-model cost distributions ---
    model_stats: Dict[str, Dict[str, float]] = {}
    for a in arms:
        c = costs[a]
        heuristic = _heuristic_c_tilde(a)
        model_stats[ARM_SHORT[a]] = {
            "mean_cost": float(c.mean()),
            "std_cost": float(c.std()),
            "cv": float(c.std() / c.mean()) if c.mean() > 0 else 0.0,
            "min_cost": float(c.min()),
            "p25_cost": float(np.percentile(c, 25)),
            "p50_cost": float(np.percentile(c, 50)),
            "p75_cost": float(np.percentile(c, 75)),
            "max_cost": float(c.max()),
            "heuristic_c_tilde": float(heuristic),
            "blended_rate_per_m": float((PRICING[a]["in"] + PRICING[a]["out"]) / 2.0),
        }
    results["model_stats"] = model_stats

    # --- 2. Cost ranking preservation ---
    heuristic_order = sorted(arms, key=lambda a: _heuristic_c_tilde(a))
    heuristic_ranking = {a: i for i, a in enumerate(heuristic_order)}

    full_match = 0
    pairwise: Dict[str, Dict[str, int]] = {}

    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            ai, aj = heuristic_order[i], heuristic_order[j]
            key = f"{ARM_SHORT[ai]} < {ARM_SHORT[aj]}"
            correct = int(np.sum(costs[ai] < costs[aj]))
            pairwise[key] = {"correct": correct, "total": n, "frac": correct / n}

    for idx in range(n):
        actual_order = sorted(arms, key=lambda a: costs[a][idx])
        if actual_order == heuristic_order:
            full_match += 1

    results["ranking"] = {
        "heuristic_order": [ARM_SHORT[a] for a in heuristic_order],
        "full_ordering_match": {"correct": full_match, "total": n, "frac": full_match / n},
        "pairwise": pairwise,
    }

    # --- 3. c_tilde space analysis ---
    c_tilde_stats: Dict[str, Dict[str, float]] = {}
    c_tilde_heuristics = {a: _heuristic_c_tilde(a) for a in arms}
    c_tilde_per_request: Dict[str, np.ndarray] = {}

    for a in arms:
        ct_values = np.array([_per_request_c_tilde(c, a) for c in costs[a]])
        c_tilde_per_request[a] = ct_values
        c_tilde_stats[ARM_SHORT[a]] = {
            "heuristic": float(c_tilde_heuristics[a]),
            "per_request_mean": float(ct_values.mean()),
            "per_request_std": float(ct_values.std()),
            "per_request_min": float(ct_values.min()),
            "per_request_max": float(ct_values.max()),
        }

    sorted_arms = sorted(arms, key=lambda a: c_tilde_heuristics[a])
    total_gap = c_tilde_heuristics[sorted_arms[-1]] - c_tilde_heuristics[sorted_arms[0]]

    inter_model_gaps: Dict[str, float] = {}
    for i in range(len(sorted_arms) - 1):
        ai, aj = sorted_arms[i], sorted_arms[i + 1]
        gap = c_tilde_heuristics[aj] - c_tilde_heuristics[ai]
        inter_model_gaps[f"{ARM_SHORT[ai]} -> {ARM_SHORT[aj]}"] = float(gap)
    inter_model_gaps["total"] = float(total_gap)

    for a in arms:
        std = c_tilde_per_request[a].std()
        frac = float(std / total_gap) if total_gap > 0 else 0.0
        c_tilde_stats[ARM_SHORT[a]]["std_as_fraction_of_total_gap"] = frac

    results["c_tilde_space"] = {
        "per_model": c_tilde_stats,
        "inter_model_gaps": inter_model_gaps,
    }

    # --- 4. Prompt-cost correlation ---
    word_counts = np.array([len(p.split()) for p in prompts])
    prompt_corr: Dict[str, Dict[str, float]] = {}
    for a in arms:
        if costs[a].std() == 0:
            prompt_corr[ARM_SHORT[a]] = {"spearman_rho": 0.0, "p_value": 1.0, "note": "constant cost"}
        else:
            rho, pval = stats.spearmanr(word_counts, costs[a])
            prompt_corr[ARM_SHORT[a]] = {"spearman_rho": float(rho), "p_value": float(pval)}
    results["prompt_cost_correlation"] = prompt_corr

    # --- 5. Cross-model cost correlation ---
    cross_corr: Dict[str, float] = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            if costs[arms[i]].std() == 0 or costs[arms[j]].std() == 0:
                rho = 0.0
            else:
                rho, _ = stats.spearmanr(costs[arms[i]], costs[arms[j]])
            key = f"{ARM_SHORT[arms[i]]} vs {ARM_SHORT[arms[j]]}"
            cross_corr[key] = float(rho)
    results["cross_model_correlation"] = cross_corr

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of the validation results."""
    print(f"\n{'=' * 70}")
    print(f"  {results['label']}  ({results['n_prompts']} prompts, "
          f"{results['n_arms']} arms)")
    print(f"{'=' * 70}")

    print("\n  Per-Model Cost Distributions:")
    for name, s in results["model_stats"].items():
        print(f"    {name:<16s}  mean=${s['mean_cost']:.8f}  "
              f"std=${s['std_cost']:.8f}  CV={s['cv']:.2f}  "
              f"c_tilde={s['heuristic_c_tilde']:.4f}")

    r = results["ranking"]
    print(f"\n  Cost Ranking Preservation:")
    print(f"    Heuristic order: {' < '.join(r['heuristic_order'])}")
    fm = r["full_ordering_match"]
    print(f"    Full match: {fm['correct']}/{fm['total']} ({fm['frac']:.1%})")
    for key, pw in r["pairwise"].items():
        print(f"    {key}: {pw['correct']}/{pw['total']} ({pw['frac']:.1%})")

    ct = results["c_tilde_space"]
    print(f"\n  c_tilde Space (Log-Normalized):")
    for name, s in ct["per_model"].items():
        print(f"    {name:<16s}  heuristic={s['heuristic']:.4f}  "
              f"actual_mean={s['per_request_mean']:.4f}  "
              f"std={s['per_request_std']:.4f}  "
              f"std/gap={s['std_as_fraction_of_total_gap']:.3f}")
    print(f"    Inter-model gaps: {ct['inter_model_gaps']}")

    print(f"\n  Prompt-Cost Correlation (Spearman):")
    for name, s in results["prompt_cost_correlation"].items():
        print(f"    {name:<16s}  rho={s['spearman_rho']:.3f}  p={s['p_value']:.2e}")

    print(f"\n  Cross-Model Cost Correlation:")
    for key, rho in results["cross_model_correlation"].items():
        print(f"    {key}: rho={rho:.3f}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    # K=3 analysis (always available)
    logger.info("Analyzing K=3 portfolio (val split)...")
    k3_data = load_split(VAL_DATA_PATH, K3_ARMS)
    k3_results = analyze_portfolio(k3_data, K3_ARMS, "K=3 Portfolio (val, 1785 prompts)")
    all_results["k3"] = k3_results
    print_results(k3_results)

    # K=4 analysis (if available)
    k4_path = OFFLINE_DATASET_DIR / "val_k4.jsonl"
    if k4_path.exists():
        logger.info("Analyzing K=4 portfolio (val_k4 split)...")
        k4_data = load_split(k4_path, K4_ARMS)

        flash_costs = k4_data["costs"]["google/gemini-2.5-flash"]
        n_unique = len(set(float(c) for c in flash_costs))
        if n_unique <= 1:
            logger.warning(
                "Flash costs are constant ($%.8f) — K=4 analysis will use "
                "synthetic data. Run collect_flash_token_counts.py + "
                "merge_flash_into_splits.py first for actual costs.",
                flash_costs[0],
            )

        k4_results = analyze_portfolio(k4_data, K4_ARMS, "K=4 Portfolio (val_k4, with Flash)")
        all_results["k4"] = k4_results
        print_results(k4_results)
    else:
        logger.info("K=4 val split not found, skipping K=4 analysis.")

    # Save results
    output_path = RESULTS_DIR / "cost_heuristic_validation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
