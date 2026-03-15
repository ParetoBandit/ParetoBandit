#!/usr/bin/env python3
"""
Pilot: Flash vs Pro on 100 hard-reasoning prompts.

Tests two hypotheses simultaneously:

1. **Model differentiation**: Is there a meaningful reward gap between
   Gemini-2.5-Flash and Gemini-2.5-Pro on hard reasoning prompts?
   Measured via tetrachoric correlation of binarized rewards.
   Sweet spot for semantic transfer: r_tet ∈ [0.5, 0.85].

2. **CoT rubric discriminativeness**: Does the new three-factor
   Discriminative Router Judge rubric (Logic 50%, Constraint 30%,
   Utility 20%) produce more informative reward signals than a
   binary pass/fail?  Measured via per-factor disagreement rates
   and reward variance.

Pipeline
--------
1. Sample 100 prompts from the augmented ``diverse_5k.jsonl``:
   50 MATH (L4+5), 25 GPQA, 25 TheoremQA.
2. Collect responses from Flash and Pro via OpenRouter.
3. Judge each (prompt, response) pair with the CoT multi-judge panel.
4. Compute tetrachoric correlation, per-factor stats, and summary.

Usage
-----
    python data_collection/scripts/pilot_flash_vs_pro.py
    python data_collection/scripts/pilot_flash_vs_pro.py --workers 32
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data_collection" / "scripts"))

from bandit_gpt.config import PROMPTS_DIR
from rejudge_cot import CoTRewardGenerator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Pilot models ──────────────────────────────────────────────────────
FLASH = "google/gemini-2.5-flash"
PRO = "google/gemini-2.5-pro"
PILOT_MODELS = [FLASH, PRO]

# ── Sampling quotas from hard-reasoning sources ──────────────────────
SAMPLE_QUOTAS = {
    "math": 50,
    "gpqa": 25,
    "theoremqa": 25,
}
TOTAL_PILOT = sum(SAMPLE_QUOTAS.values())

# ── Output paths ─────────────────────────────────────────────────────
PILOT_DIR = PROJECT_ROOT / "data_collection" / "pilot_flash_vs_pro"


# =====================================================================
#  Sampling
# =====================================================================

def sample_hard_prompts(
    diverse_path: Path,
    quotas: Dict[str, int],
    seed: int = 42,
) -> List[Dict]:
    """Sample prompts from hard-reasoning sources, stratified by source.

    Prefers hard/very_hard tiers when available, falls back to any tier.
    """
    rng = np.random.RandomState(seed)
    all_prompts: List[Dict] = []
    with open(diverse_path) as f:
        for line in f:
            all_prompts.append(json.loads(line))

    sampled: List[Dict] = []
    for source, n in quotas.items():
        pool = [p for p in all_prompts if p["source"] == source]
        hard_pool = [
            p for p in pool if p["tier"] in ("hard", "very_hard")
        ]
        # Prefer hard/very_hard; fall back to full pool if insufficient
        candidates = hard_pool if len(hard_pool) >= n else pool
        if len(candidates) < n:
            logger.warning(
                f"  [{source}] Only {len(candidates)} available, "
                f"taking all (requested {n})"
            )
            sampled.extend(candidates)
        else:
            idx = rng.choice(len(candidates), size=n, replace=False)
            sampled.extend([candidates[i] for i in idx])

    rng.shuffle(sampled)
    return sampled


# =====================================================================
#  Tetrachoric correlation (copied from router.py for self-containment)
# =====================================================================

def tetrachoric_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Tetrachoric correlation for two binary (0/1) vectors."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        return np.nan

    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    d = np.sum((x == 0) & (y == 0))

    if a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
        return np.nan

    h = norm.ppf((a + b) / n)
    k = norm.ppf((a + c) / n)

    from scipy.stats import multivariate_normal

    def _objective(r: float) -> float:
        p11 = multivariate_normal.cdf(
            [h, k], mean=[0, 0], cov=[[1, r], [r, 1]],
        )
        return p11 - a / n

    try:
        return float(brentq(_objective, -0.999, 0.999, xtol=1e-8))
    except ValueError:
        return np.nan


# =====================================================================
#  Analysis
# =====================================================================

def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute all pilot metrics from collected reward data.

    Returns a dict with tetrachoric correlation, per-factor stats,
    rubric discriminativeness metrics, and sample-level details.
    """
    # Group by prompt
    by_prompt: Dict[str, Dict[str, Dict]] = {}
    for r in results:
        if not r.get("ok"):
            continue
        prompt = r["prompt"]
        model = r["model_id"]
        if prompt not in by_prompt:
            by_prompt[prompt] = {}
        by_prompt[prompt][model] = r

    # Build aligned reward vectors
    flash_rewards: List[float] = []
    pro_rewards: List[float] = []
    flash_logic: List[int] = []
    pro_logic: List[int] = []
    flash_constraint: List[int] = []
    pro_constraint: List[int] = []
    flash_utility: List[float] = []
    pro_utility: List[float] = []
    prompt_details: List[Dict] = []

    for prompt, models in by_prompt.items():
        if FLASH not in models or PRO not in models:
            continue
        f_data = models[FLASH]
        p_data = models[PRO]

        f_reward = f_data["raw_score"]
        p_reward = p_data["raw_score"]
        if np.isnan(f_reward) or np.isnan(p_reward):
            continue

        flash_rewards.append(f_reward)
        pro_rewards.append(p_reward)

        # Per-factor averages across judges
        f_details = f_data.get("judge_details", [])
        p_details = p_data.get("judge_details", [])

        f_logic = np.mean([j["logic"] for j in f_details]) if f_details else 0.5
        p_logic = np.mean([j["logic"] for j in p_details]) if p_details else 0.5
        f_const = np.mean([j["constraint"] for j in f_details]) if f_details else 0.5
        p_const = np.mean([j["constraint"] for j in p_details]) if p_details else 0.5
        f_util = np.mean([j["utility"] for j in f_details]) if f_details else 0.5
        p_util = np.mean([j["utility"] for j in p_details]) if p_details else 0.5

        flash_logic.append(int(f_logic >= 0.5))
        pro_logic.append(int(p_logic >= 0.5))
        flash_constraint.append(int(f_const >= 0.5))
        pro_constraint.append(int(p_const >= 0.5))
        flash_utility.append(f_util)
        pro_utility.append(p_util)

        prompt_details.append({
            "prompt": prompt[:100],
            "flash_reward": round(f_reward, 4),
            "pro_reward": round(p_reward, 4),
            "delta": round(p_reward - f_reward, 4),
            "flash_logic": round(f_logic, 2),
            "pro_logic": round(p_logic, 2),
        })

    n = len(flash_rewards)
    if n == 0:
        return {"error": "No valid paired results"}

    flash_arr = np.array(flash_rewards)
    pro_arr = np.array(pro_rewards)

    # Binarize at median for tetrachoric (standard approach)
    median_threshold = np.median(np.concatenate([flash_arr, pro_arr]))
    flash_bin = (flash_arr >= median_threshold).astype(int)
    pro_bin = (pro_arr >= median_threshold).astype(int)

    r_tet = tetrachoric_corr(flash_bin, pro_bin)

    # Also compute Pearson on continuous rewards
    r_pearson = float(np.corrcoef(flash_arr, pro_arr)[0, 1])

    # Agreement rate (both pass or both fail)
    agree = np.mean(flash_bin == pro_bin)

    # Per-factor disagreement
    logic_disagree = np.mean(
        np.array(flash_logic) != np.array(pro_logic)
    )
    constraint_disagree = np.mean(
        np.array(flash_constraint) != np.array(pro_constraint)
    )
    utility_gap = np.mean(
        np.abs(np.array(flash_utility) - np.array(pro_utility))
    )

    # Pro advantage
    pro_wins = np.sum(pro_arr > flash_arr + 0.05)
    flash_wins = np.sum(flash_arr > pro_arr + 0.05)
    ties = n - pro_wins - flash_wins

    # Rubric discriminativeness: variance of composite reward
    all_rewards = np.concatenate([flash_arr, pro_arr])
    reward_variance = float(np.var(all_rewards))
    # Compare to binary: if all rewards were 0 or 1
    binary_rewards = (all_rewards >= median_threshold).astype(float)
    binary_variance = float(np.var(binary_rewards))

    # Sort by delta to find biggest differentiators
    prompt_details.sort(key=lambda x: x["delta"], reverse=True)

    return {
        "n_paired": n,
        "tetrachoric_correlation": round(r_tet, 4) if not np.isnan(r_tet) else None,
        "pearson_correlation": round(r_pearson, 4),
        "agreement_rate": round(agree, 4),
        "binarization_threshold": round(float(median_threshold), 4),
        "flash_mean_reward": round(float(flash_arr.mean()), 4),
        "pro_mean_reward": round(float(pro_arr.mean()), 4),
        "reward_delta_mean": round(float((pro_arr - flash_arr).mean()), 4),
        "pro_wins": int(pro_wins),
        "flash_wins": int(flash_wins),
        "ties": int(ties),
        "per_factor": {
            "logic_disagreement_rate": round(float(logic_disagree), 4),
            "constraint_disagreement_rate": round(float(constraint_disagree), 4),
            "utility_mean_abs_gap": round(float(utility_gap), 4),
        },
        "rubric_discriminativeness": {
            "composite_reward_variance": round(reward_variance, 6),
            "binary_reward_variance": round(binary_variance, 6),
            "variance_ratio": round(reward_variance / max(binary_variance, 1e-12), 2),
        },
        "top_5_pro_advantage": prompt_details[:5],
        "top_5_flash_advantage": prompt_details[-5:][::-1],
        "transfer_assessment": _assess_transfer(r_tet, agree, float((pro_arr - flash_arr).mean())),
    }


def _assess_transfer(
    r_tet: float, agreement: float, delta_mean: float,
) -> Dict[str, str]:
    """Produce a human-readable transfer feasibility verdict."""
    if np.isnan(r_tet) or r_tet is None:
        return {"verdict": "INCONCLUSIVE", "reason": "Could not compute tetrachoric correlation"}

    if r_tet > 0.90:
        verdict = "TOO_SIMILAR"
        reason = (
            f"r_tet={r_tet:.3f} > 0.90: Flash and Pro are nearly identical on these prompts. "
            f"Transfer would be trivially positive but the router has no reason to route to Pro. "
            f"Consider a different 4th model (e.g. openai/o3-mini)."
        )
    elif r_tet > 0.70:
        verdict = "IDEAL"
        reason = (
            f"r_tet={r_tet:.3f} ∈ [0.70, 0.90]: High enough for positive transfer, "
            f"low enough for θ_a to learn a meaningful differential. "
            f"Pro mean advantage: {delta_mean:+.4f}. Proceed with Gemini-2.5-Pro."
        )
    elif r_tet > 0.50:
        verdict = "GOOD"
        reason = (
            f"r_tet={r_tet:.3f} ∈ [0.50, 0.70]: Moderate correlation. Transfer will help "
            f"but the prior won't be highly accurate. Still a viable choice. "
            f"Pro mean advantage: {delta_mean:+.4f}."
        )
    elif r_tet > 0.30:
        verdict = "MARGINAL"
        reason = (
            f"r_tet={r_tet:.3f} ∈ [0.30, 0.50]: Weak correlation. Transfer may not help much. "
            f"The experiment will show limited cold-start acceleration. "
            f"Consider whether the routing value justifies the cost."
        )
    else:
        verdict = "TOO_DISSIMILAR"
        reason = (
            f"r_tet={r_tet:.3f} < 0.30: Flash and Pro have very different reward patterns. "
            f"Transfer prior would be misleading. This is unexpected for same-provider models."
        )

    return {"verdict": verdict, "reason": reason}


# =====================================================================
#  Main
# =====================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pilot: Flash vs Pro on 100 hard-reasoning prompts.",
    )
    parser.add_argument(
        "--diverse-prompts", type=str,
        default=str(PROMPTS_DIR / "diverse_5k.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path = PILOT_DIR / "pilot_prompts.jsonl"
    results_path = PILOT_DIR / "pilot_results.jsonl"
    analysis_path = PILOT_DIR / "pilot_analysis.json"
    summary_path = PILOT_DIR / "pilot_summary.txt"

    logger.info("=" * 65)
    logger.info("PILOT: Flash vs Pro — Hard Reasoning (100 prompts)")
    logger.info("=" * 65)

    # ── 1. Sample prompts ─────────────────────────────────────────────
    logger.info("\n1. Sampling hard-reasoning prompts ...")
    sampled = sample_hard_prompts(
        Path(args.diverse_prompts), SAMPLE_QUOTAS, args.seed,
    )
    logger.info(f"   Sampled {len(sampled)} prompts:")
    from collections import Counter
    src_dist = Counter(p["source"] for p in sampled)
    tier_dist = Counter(p["tier"] for p in sampled)
    for s, c in src_dist.most_common():
        logger.info(f"     {s}: {c}")
    logger.info(f"   Tier distribution: {dict(tier_dist)}")

    # Write pilot prompts
    with open(prompts_path, "w") as f:
        for p in sampled:
            f.write(json.dumps({"prompt": p["prompt"], "source": p["source"], "tier": p["tier"]}) + "\n")
    logger.info(f"   Wrote {len(sampled)} prompts to {prompts_path}")

    # ── 2. Collect responses + judge ──────────────────────────────────
    logger.info(f"\n2. Collecting responses and judging ...")
    logger.info(f"   Models: {FLASH}, {PRO}")
    logger.info(f"   Workers: {args.workers}")

    gen = CoTRewardGenerator(max_workers=args.workers)

    # Check for resume
    completed: set[tuple[str, str]] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        completed.add((entry["prompt"], entry["model_id"]))
                except json.JSONDecodeError:
                    continue
        if completed:
            logger.info(f"   Resuming: {len(completed)} tasks already done")

    tasks = []
    for p in sampled:
        for model in PILOT_MODELS:
            if (p["prompt"], model) not in completed:
                tasks.append((p["prompt"], model))

    total_tasks = len(sampled) * len(PILOT_MODELS)
    logger.info(f"   Tasks: {len(tasks)} remaining / {total_tasks} total")

    if tasks:
        t0 = time.perf_counter()
        done = 0
        with open(results_path, "a") as outfile:
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(gen.process_task, t): t for t in tasks
                }
                for fut in _as_completed(futures):
                    res = fut.result()
                    outfile.write(json.dumps(res) + "\n")
                    outfile.flush()
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        elapsed = time.perf_counter() - t0
                        logger.info(
                            f"   [{done}/{len(tasks)}] "
                            f"({elapsed:.0f}s elapsed)"
                        )

        elapsed = time.perf_counter() - t0
        logger.info(f"   Done in {elapsed:.0f}s")
    else:
        logger.info("   All tasks already completed.")

    # ── 3. Analyze ────────────────────────────────────────────────────
    logger.info(f"\n3. Analyzing results ...")
    all_results: List[Dict] = []
    with open(results_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    analysis = analyze_results(all_results)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # ── 4. Print summary ──────────────────────────────────────────────
    summary_lines = [
        "=" * 65,
        "PILOT RESULTS: Flash vs Pro on Hard Reasoning",
        "=" * 65,
        "",
        f"Paired prompts:     {analysis.get('n_paired', 0)}",
        "",
        "── Model Quality ──",
        f"Flash mean reward:  {analysis.get('flash_mean_reward', '?')}",
        f"Pro mean reward:    {analysis.get('pro_mean_reward', '?')}",
        f"Delta (Pro - Flash):{analysis.get('reward_delta_mean', '?'):+.4f}" if isinstance(analysis.get('reward_delta_mean'), float) else f"Delta: {analysis.get('reward_delta_mean', '?')}",
        f"Pro wins / Flash wins / Ties: {analysis.get('pro_wins', '?')} / {analysis.get('flash_wins', '?')} / {analysis.get('ties', '?')}",
        "",
        "── Transfer Feasibility ──",
        f"Tetrachoric r:      {analysis.get('tetrachoric_correlation', '?')}",
        f"Pearson r:          {analysis.get('pearson_correlation', '?')}",
        f"Agreement rate:     {analysis.get('agreement_rate', '?')}",
    ]

    ta = analysis.get("transfer_assessment", {})
    summary_lines.extend([
        f"Verdict:            {ta.get('verdict', '?')}",
        f"Reason:             {ta.get('reason', '?')}",
        "",
        "── CoT Rubric Discriminativeness ──",
    ])

    pf = analysis.get("per_factor", {})
    rd = analysis.get("rubric_discriminativeness", {})
    summary_lines.extend([
        f"Logic disagreement: {pf.get('logic_disagreement_rate', '?')}",
        f"Constraint disagr.: {pf.get('constraint_disagreement_rate', '?')}",
        f"Utility gap (abs):  {pf.get('utility_mean_abs_gap', '?')}",
        f"Composite variance: {rd.get('composite_reward_variance', '?')}",
        f"Binary variance:    {rd.get('binary_reward_variance', '?')}",
        f"Variance ratio:     {rd.get('variance_ratio', '?')}x",
        "",
        "── Top 5 Pro Advantages ──",
    ])
    for item in analysis.get("top_5_pro_advantage", []):
        summary_lines.append(
            f"  Δ={item['delta']:+.4f}  Flash={item['flash_reward']:.2f}  "
            f"Pro={item['pro_reward']:.2f}  {item['prompt'][:70]}..."
        )
    summary_lines.append("")
    summary_lines.append("── Top 5 Flash Advantages ──")
    for item in analysis.get("top_5_flash_advantage", []):
        summary_lines.append(
            f"  Δ={item['delta']:+.4f}  Flash={item['flash_reward']:.2f}  "
            f"Pro={item['pro_reward']:.2f}  {item['prompt'][:70]}..."
        )

    summary_text = "\n".join(summary_lines)
    logger.info("\n" + summary_text)

    with open(summary_path, "w") as f:
        f.write(summary_text)

    logger.info(f"\nAnalysis -> {analysis_path}")
    logger.info(f"Summary  -> {summary_path}")


if __name__ == "__main__":
    main()
