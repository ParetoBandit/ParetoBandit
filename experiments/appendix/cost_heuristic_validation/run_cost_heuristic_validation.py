#!/usr/bin/env python3
"""Appendix: Cost Heuristic Validation.

Empirically validates the static log-normalized cost heuristic c_tilde_a
used in the selection utility (Eq. 1) against actual per-request costs
from the offline dataset.

Analyses
--------
1. **Per-model cost distributions**: mean, std, CV, quantiles for each arm.
2. **Cost ranking preservation**: fraction of prompts where the heuristic's
   model-cost ordering holds.  Evaluated pairwise (strict ``<``) and as a
   full ordering, with Wilson 95 % confidence intervals.
3. **Log-cost separation**: within-model vs. inter-model spread in
   log(cost_USD) space.  Quantifies how cleanly separated the cost tiers
   are without referencing the heuristic itself (avoids circularity).
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

from pareto_bandit.config import (
    K3_ARM_ORDER,
    K3_ARM_SHORT,
    OFFLINE_DATASET_DIR,
    VAL_DATA_PATH,
)
from pareto_bandit.costs import log_normalize_cost
from pareto_bandit.types import RouterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

K3_ARMS: List[str] = K3_ARM_ORDER
K4_ARMS: List[str] = K3_ARMS + ["google/gemini-2.5-flash"]

PRICING: Dict[str, Dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"in": 0.1, "out": 0.1},
    "mistralai/mistral-large-2512": {"in": 0.5, "out": 1.5},
    "google/gemini-2.5-pro": {"in": 1.25, "out": 10.0},
    "google/gemini-2.5-flash": {"in": 0.3, "out": 2.5},
}

ARM_SHORT: Dict[str, str] = {
    **K3_ARM_SHORT,
    "google/gemini-2.5-flash": "Gemini-Flash",
}


def _heuristic_c_tilde(arm_id: str) -> float:
    """Compute the static heuristic c_tilde for a model."""
    cfg = RouterConfig()
    p = PRICING[arm_id]
    blended_per_1k = (p["in"] + p["out"]) / 2.0 / 1000.0
    return log_normalize_cost(blended_per_1k, cfg.market_cost_floor, cfg.market_cost_ceiling)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95 % confidence interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Z-score (default 1.96 for 95 % CI).

    Returns:
        ``(lower, upper)`` bounds of the interval.
    """
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
        / denom
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_split(path: Path, arms: List[str]) -> Dict[str, Any]:
    """Load a JSONL split and extract per-model costs and prompt texts.

    Rows where *any* arm has a zero or negative cost are dropped (these
    indicate API refusals or empty responses that would produce spurious
    data points in the cost analysis).

    Returns:
        Dictionary with keys ``prompts``, ``costs``, ``rewards``, ``n``,
        ``n_raw``, ``n_dropped``.
    """
    raw_prompts: List[str] = []
    raw_costs: Dict[str, List[float]] = {a: [] for a in arms}
    raw_rewards: Dict[str, List[float]] = {a: [] for a in arms}

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            raw_prompts.append(rec["prompt"])
            for a in arms:
                raw_costs[a].append(rec["arms"][a]["cost"])
                raw_rewards[a].append(rec["arms"][a]["reward"])

    n_raw = len(raw_prompts)
    keep = np.ones(n_raw, dtype=bool)
    for a in arms:
        arr = np.array(raw_costs[a])
        keep &= arr > 0

    n_dropped = int(n_raw - keep.sum())
    if n_dropped > 0:
        logger.warning(
            "Dropped %d/%d rows with zero/negative cost in %s",
            n_dropped, n_raw, path.name,
        )

    indices = np.where(keep)[0]
    prompts = [raw_prompts[i] for i in indices]
    costs = {a: np.array(raw_costs[a])[indices] for a in arms}
    rewards = {a: np.array(raw_rewards[a])[indices] for a in arms}

    return {
        "prompts": prompts,
        "costs": costs,
        "rewards": rewards,
        "n": len(prompts),
        "n_raw": n_raw,
        "n_dropped": n_dropped,
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

    results: Dict[str, Any] = {
        "label": label,
        "n_prompts": n,
        "n_raw": data["n_raw"],
        "n_dropped": data["n_dropped"],
        "n_arms": len(arms),
    }

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

    # --- 2. Cost ranking preservation (strict <, with ties and CIs) ---
    heuristic_order = sorted(arms, key=lambda a: _heuristic_c_tilde(a))

    full_match = 0
    pairwise: Dict[str, Dict[str, Any]] = {}

    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            ai, aj = heuristic_order[i], heuristic_order[j]
            key = f"{ARM_SHORT[ai]} < {ARM_SHORT[aj]}"
            strict = int(np.sum(costs[ai] < costs[aj]))
            ties = int(np.sum(costs[ai] == costs[aj]))
            ci_lo, ci_hi = _wilson_ci(strict, n)
            pairwise[key] = {
                "correct": strict,
                "ties": ties,
                "total": n,
                "frac": strict / n,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            }

    for idx in range(n):
        actual_order = sorted(arms, key=lambda a: costs[a][idx])
        if actual_order == heuristic_order:
            full_match += 1

    full_ci_lo, full_ci_hi = _wilson_ci(full_match, n)
    results["ranking"] = {
        "heuristic_order": [ARM_SHORT[a] for a in heuristic_order],
        "full_ordering_match": {
            "correct": full_match,
            "total": n,
            "frac": full_match / n,
            "ci_95_lo": full_ci_lo,
            "ci_95_hi": full_ci_hi,
        },
        "pairwise": pairwise,
    }

    # --- 3. Log-cost separation analysis ---
    # Measures within-model vs inter-model spread in log(USD) space,
    # independent of the heuristic itself.
    log_costs = {a: np.log(costs[a]) for a in arms}
    log_stats: Dict[str, Dict[str, float]] = {}
    for a in arms:
        lc = log_costs[a]
        log_stats[ARM_SHORT[a]] = {
            "log_mean": float(lc.mean()),
            "log_std": float(lc.std()),
            "heuristic_c_tilde": float(_heuristic_c_tilde(a)),
        }

    sorted_arms = sorted(arms, key=lambda a: _heuristic_c_tilde(a))

    adjacent_separation: Dict[str, Dict[str, float]] = {}
    for i in range(len(sorted_arms) - 1):
        ai, aj = sorted_arms[i], sorted_arms[i + 1]
        lc_i, lc_j = log_costs[ai], log_costs[aj]
        gap = float(lc_j.mean() - lc_i.mean())
        pooled_std = float(
            np.sqrt((lc_i.std() ** 2 + lc_j.std() ** 2) / 2)
        )
        cohens_d = gap / pooled_std if pooled_std > 0 else float("inf")
        key = f"{ARM_SHORT[ai]} -> {ARM_SHORT[aj]}"
        adjacent_separation[key] = {
            "log_mean_gap": gap,
            "pooled_log_std": pooled_std,
            "cohens_d": cohens_d,
        }

    total_log_range = float(
        log_costs[sorted_arms[-1]].mean() - log_costs[sorted_arms[0]].mean()
    )
    within_std_fracs: Dict[str, float] = {}
    for a in arms:
        frac = (
            float(log_costs[a].std() / total_log_range)
            if total_log_range > 0
            else 0.0
        )
        within_std_fracs[ARM_SHORT[a]] = frac

    results["log_cost_separation"] = {
        "per_model": log_stats,
        "adjacent_separation": adjacent_separation,
        "total_log_range": total_log_range,
        "within_std_as_frac_of_range": within_std_fracs,
    }

    # --- 4. Prompt-cost correlation ---
    word_counts = np.array([len(p.split()) for p in prompts])
    prompt_corr: Dict[str, Dict[str, float]] = {}
    for a in arms:
        if costs[a].std() == 0:
            prompt_corr[ARM_SHORT[a]] = {
                "spearman_rho": 0.0, "p_value": 1.0, "note": "constant cost",
            }
        else:
            rho, pval = stats.spearmanr(word_counts, costs[a])
            prompt_corr[ARM_SHORT[a]] = {
                "spearman_rho": float(rho), "p_value": float(pval),
            }
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

    # Per-model cost arrays for figure generation (avoids re-reading JSONL)
    results["_costs_by_model"] = {
        ARM_SHORT[a]: costs[a].tolist() for a in arms
    }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of the validation results."""
    print(f"\n{'=' * 70}")
    print(
        f"  {results['label']}  ({results['n_prompts']} prompts, "
        f"{results['n_arms']} arms; {results['n_dropped']} rows dropped "
        f"from {results['n_raw']} raw)"
    )
    print(f"{'=' * 70}")

    print("\n  Per-Model Cost Distributions:")
    for name, s in results["model_stats"].items():
        print(
            f"    {name:<16s}  mean=${s['mean_cost']:.8f}  "
            f"std=${s['std_cost']:.8f}  CV={s['cv']:.2f}  "
            f"c_tilde={s['heuristic_c_tilde']:.4f}"
        )

    r = results["ranking"]
    print("\n  Cost Ranking Preservation (strict <):")
    print(f"    Heuristic order: {' < '.join(r['heuristic_order'])}")
    fm = r["full_ordering_match"]
    print(
        f"    Full match: {fm['correct']}/{fm['total']} ({fm['frac']:.1%}) "
        f"[95% CI: {fm['ci_95_lo']:.1%}\u2013{fm['ci_95_hi']:.1%}]"
    )
    for key, pw in r["pairwise"].items():
        print(
            f"    {key}: {pw['correct']}/{pw['total']} ({pw['frac']:.1%}) "
            f"[CI: {pw['ci_95_lo']:.1%}\u2013{pw['ci_95_hi']:.1%}] "
            f"ties={pw['ties']}"
        )

    lcs = results["log_cost_separation"]
    print("\n  Log-Cost Separation:")
    for name, s in lcs["per_model"].items():
        frac = lcs["within_std_as_frac_of_range"][name]
        print(
            f"    {name:<16s}  log_mean={s['log_mean']:.3f}  "
            f"log_std={s['log_std']:.3f}  "
            f"std/range={frac:.3f}"
        )
    for key, sep in lcs["adjacent_separation"].items():
        print(
            f"    {key}: gap={sep['log_mean_gap']:.3f}  "
            f"pooled_std={sep['pooled_log_std']:.3f}  "
            f"Cohen's d={sep['cohens_d']:.2f}"
        )

    print("\n  Prompt-Cost Correlation (Spearman):")
    for name, s in results["prompt_cost_correlation"].items():
        print(f"    {name:<16s}  rho={s['spearman_rho']:.3f}  p={s['p_value']:.2e}")

    print("\n  Cross-Model Cost Correlation:")
    for key, rho in results["cross_model_correlation"].items():
        print(f"    {key}: rho={rho:.3f}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    # K=3 analysis (always available)
    logger.info("Analyzing K=3 portfolio (val split)...")
    k3_data = load_split(VAL_DATA_PATH, K3_ARMS)
    k3_results = analyze_portfolio(
        k3_data, K3_ARMS,
        f"K=3 Portfolio (val, {k3_data['n']} prompts)",
    )
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

        k4_results = analyze_portfolio(
            k4_data, K4_ARMS,
            "K=4 Portfolio (val_k4, with Flash)",
        )
        all_results["k4"] = k4_results
        print_results(k4_results)
    else:
        logger.info("K=4 val split not found, skipping K=4 analysis.")

    output_path = RESULTS_DIR / "cost_heuristic_validation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
