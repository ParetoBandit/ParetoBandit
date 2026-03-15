#!/usr/bin/env python3
"""
Pilot: GPT-4.1 vs GPT-4.1-mini vs GPT-4.1-nano on 100 hard-reasoning prompts.

Evaluates which OpenAI model is the best 4th-model candidate for the
semantic transfer experiment.  GPT-4.1 is the transfer neighbor (already
in K=3), so we need the tetrachoric correlation between GPT-4.1 and each
candidate to fall in [0.5, 0.85].

Reuses the same 100 prompts from the Flash-vs-Pro pilot for comparability.

Usage
-----
    python data_collection/scripts/pilot_openai_family.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "data_collection" / "scripts"))

from bandit_gpt.config import PROMPTS_DIR
from rejudge_cot import CoTRewardGenerator
from pilot_flash_vs_pro import (
    tetrachoric_corr,
    sample_hard_prompts,
    SAMPLE_QUOTAS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Models to test ────────────────────────────────────────────────────
GPT41 = "openai/gpt-4.1"
MINI = "openai/gpt-4.1-mini"
NANO = "openai/gpt-4.1-nano"
PILOT_MODELS = [GPT41, MINI, NANO]

PILOT_DIR = PROJECT_ROOT / "data_collection" / "pilot_openai_family"


def analyze_openai_family(results: List[Dict]) -> Dict[str, Any]:
    """Compute tetrachoric correlations and per-judge stats for the OpenAI family."""
    by_prompt: Dict[str, Dict[str, Dict]] = {}
    for r in results:
        if not r.get("ok"):
            continue
        prompt = r["prompt"]
        model = r["model_id"]
        if prompt not in by_prompt:
            by_prompt[prompt] = {}
        by_prompt[prompt][model] = r

    # Build aligned reward vectors for prompts where all 3 models have results
    gpt_rewards, mini_rewards, nano_rewards = [], [], []
    gpt_logic, mini_logic, nano_logic = [], [], []
    prompt_details: List[Dict] = []

    for prompt, models in by_prompt.items():
        if not all(m in models for m in PILOT_MODELS):
            continue
        g = models[GPT41]["raw_score"]
        m = models[MINI]["raw_score"]
        n = models[NANO]["raw_score"]
        if any(np.isnan(x) for x in [g, m, n]):
            continue

        gpt_rewards.append(g)
        mini_rewards.append(m)
        nano_rewards.append(n)

        # Per-factor (majority vote across judges)
        for model_id, logic_list in [
            (GPT41, gpt_logic), (MINI, mini_logic), (NANO, nano_logic),
        ]:
            details = models[model_id].get("judge_details", [])
            avg_logic = np.mean([j["logic"] for j in details]) if details else 0.5
            logic_list.append(int(avg_logic >= 0.5))

        prompt_details.append({
            "prompt": prompt[:100],
            "gpt41": round(g, 4),
            "mini": round(m, 4),
            "nano": round(n, 4),
            "mini_delta": round(m - g, 4),
            "nano_delta": round(n - g, 4),
        })

    n = len(gpt_rewards)
    if n == 0:
        return {"error": "No valid triple results"}

    gpt_arr = np.array(gpt_rewards)
    mini_arr = np.array(mini_rewards)
    nano_arr = np.array(nano_rewards)

    # Binarize at pooled median
    all_rewards = np.concatenate([gpt_arr, mini_arr, nano_arr])
    threshold = np.median(all_rewards)

    gpt_bin = (gpt_arr >= threshold).astype(int)
    mini_bin = (mini_arr >= threshold).astype(int)
    nano_bin = (nano_arr >= threshold).astype(int)

    r_tet_mini = tetrachoric_corr(gpt_bin, mini_bin)
    r_tet_nano = tetrachoric_corr(gpt_bin, nano_bin)
    r_tet_mini_nano = tetrachoric_corr(mini_bin, nano_bin)

    r_pearson_mini = float(np.corrcoef(gpt_arr, mini_arr)[0, 1])
    r_pearson_nano = float(np.corrcoef(gpt_arr, nano_arr)[0, 1])

    agree_mini = np.mean(gpt_bin == mini_bin)
    agree_nano = np.mean(gpt_bin == nano_bin)

    # Logic agreement
    logic_agree_mini = np.mean(np.array(gpt_logic) == np.array(mini_logic))
    logic_agree_nano = np.mean(np.array(gpt_logic) == np.array(nano_logic))

    # Win rates
    def _wins(a, b, margin=0.05):
        return int(np.sum(a > b + margin)), int(np.sum(b > a + margin)), int(np.sum(np.abs(a - b) <= margin))

    mini_w, gpt_w_vs_mini, ties_mini = _wins(mini_arr, gpt_arr)
    nano_w, gpt_w_vs_nano, ties_nano = _wins(nano_arr, gpt_arr)

    prompt_details.sort(key=lambda x: x["mini_delta"], reverse=True)

    def _verdict(r_tet, label):
        if np.isnan(r_tet):
            return "INCONCLUSIVE"
        if r_tet > 0.90:
            return "TOO_SIMILAR"
        if r_tet > 0.70:
            return "IDEAL"
        if r_tet > 0.50:
            return "GOOD"
        if r_tet > 0.30:
            return "MARGINAL"
        return "TOO_DISSIMILAR"

    return {
        "n_paired": n,
        "binarization_threshold": round(float(threshold), 4),
        "gpt41": {
            "mean_reward": round(float(gpt_arr.mean()), 4),
            "std_reward": round(float(gpt_arr.std()), 4),
            "logic_pass_rate": round(float(np.mean(gpt_logic)), 4),
        },
        "mini": {
            "mean_reward": round(float(mini_arr.mean()), 4),
            "std_reward": round(float(mini_arr.std()), 4),
            "logic_pass_rate": round(float(np.mean(mini_logic)), 4),
            "delta_vs_gpt41": round(float((mini_arr - gpt_arr).mean()), 4),
            "wins_vs_gpt41": mini_w,
            "losses_vs_gpt41": gpt_w_vs_mini,
            "ties_vs_gpt41": ties_mini,
            "tetrachoric_vs_gpt41": round(r_tet_mini, 4) if not np.isnan(r_tet_mini) else None,
            "pearson_vs_gpt41": round(r_pearson_mini, 4),
            "agreement_vs_gpt41": round(float(agree_mini), 4),
            "logic_agreement_vs_gpt41": round(float(logic_agree_mini), 4),
            "transfer_verdict": _verdict(r_tet_mini, "mini"),
        },
        "nano": {
            "mean_reward": round(float(nano_arr.mean()), 4),
            "std_reward": round(float(nano_arr.std()), 4),
            "logic_pass_rate": round(float(np.mean(nano_logic)), 4),
            "delta_vs_gpt41": round(float((nano_arr - gpt_arr).mean()), 4),
            "wins_vs_gpt41": nano_w,
            "losses_vs_gpt41": gpt_w_vs_nano,
            "ties_vs_gpt41": ties_nano,
            "tetrachoric_vs_gpt41": round(r_tet_nano, 4) if not np.isnan(r_tet_nano) else None,
            "pearson_vs_gpt41": round(r_pearson_nano, 4),
            "agreement_vs_gpt41": round(float(agree_nano), 4),
            "logic_agreement_vs_gpt41": round(float(logic_agree_nano), 4),
            "transfer_verdict": _verdict(r_tet_nano, "nano"),
        },
        "mini_vs_nano_tetrachoric": round(r_tet_mini_nano, 4) if not np.isnan(r_tet_mini_nano) else None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pilot: GPT-4.1 family on 100 hard-reasoning prompts.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path = PILOT_DIR / "pilot_prompts.jsonl"
    results_path = PILOT_DIR / "pilot_results.jsonl"
    analysis_path = PILOT_DIR / "pilot_analysis.json"

    logger.info("=" * 65)
    logger.info("PILOT: GPT-4.1 vs Mini vs Nano — Hard Reasoning (100 prompts)")
    logger.info("=" * 65)

    # ── 1. Reuse the same 100 prompts ─────────────────────────────────
    prev_prompts = PROJECT_ROOT / "data_collection" / "pilot_flash_vs_pro" / "pilot_prompts.jsonl"
    if prev_prompts.exists():
        logger.info("\n1. Reusing prompts from Flash-vs-Pro pilot ...")
        sampled = []
        with open(prev_prompts) as f:
            for line in f:
                sampled.append(json.loads(line))
    else:
        logger.info("\n1. Sampling hard-reasoning prompts ...")
        diverse_path = PROMPTS_DIR / "diverse_5k.jsonl"
        sampled = sample_hard_prompts(diverse_path, SAMPLE_QUOTAS, args.seed)

    logger.info(f"   {len(sampled)} prompts")
    src_dist = Counter(p["source"] for p in sampled)
    for s, c in src_dist.most_common():
        logger.info(f"     {s}: {c}")

    with open(prompts_path, "w") as f:
        for p in sampled:
            f.write(json.dumps({"prompt": p["prompt"], "source": p["source"], "tier": p["tier"]}) + "\n")

    # ── 2. Collect responses + judge ──────────────────────────────────
    logger.info(f"\n2. Collecting responses and judging ...")
    logger.info(f"   Models: {', '.join(PILOT_MODELS)}")
    logger.info(f"   Workers: {args.workers}")

    gen = CoTRewardGenerator(max_workers=args.workers)

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
                futures = {pool.submit(gen.process_task, t): t for t in tasks}
                for fut in _as_completed(futures):
                    res = fut.result()
                    outfile.write(json.dumps(res) + "\n")
                    outfile.flush()
                    done += 1
                    if done % 10 == 0 or done == len(tasks):
                        elapsed = time.perf_counter() - t0
                        logger.info(f"   [{done}/{len(tasks)}] ({elapsed:.0f}s)")

        logger.info(f"   Done in {time.perf_counter() - t0:.0f}s")
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

    analysis = analyze_openai_family(all_results)

    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # ── 4. Print summary ──────────────────────────────────────────────
    logger.info(f"\n{'=' * 65}")
    logger.info("PILOT RESULTS: GPT-4.1 Family on Hard Reasoning")
    logger.info("=" * 65)
    logger.info(f"Paired prompts: {analysis.get('n_paired', 0)}")
    logger.info(f"Binarization threshold: {analysis.get('binarization_threshold', '?')}")

    logger.info(f"\n{'─' * 65}")
    logger.info(f"{'Metric':<35s} {'GPT-4.1':>10s} {'Mini':>10s} {'Nano':>10s}")
    logger.info(f"{'─' * 65}")

    g = analysis.get("gpt41", {})
    m = analysis.get("mini", {})
    na = analysis.get("nano", {})

    logger.info(f"{'Mean reward':<35s} {g.get('mean_reward','?'):>10} {m.get('mean_reward','?'):>10} {na.get('mean_reward','?'):>10}")
    logger.info(f"{'Logic pass rate':<35s} {g.get('logic_pass_rate','?'):>10} {m.get('logic_pass_rate','?'):>10} {na.get('logic_pass_rate','?'):>10}")
    logger.info(f"{'Δ vs GPT-4.1':<35s} {'—':>10} {m.get('delta_vs_gpt41','?'):>+10} {na.get('delta_vs_gpt41','?'):>+10}")
    logger.info(f"{'Wins vs GPT-4.1':<35s} {'—':>10} {m.get('wins_vs_gpt41','?'):>10} {na.get('wins_vs_gpt41','?'):>10}")
    logger.info(f"{'Losses vs GPT-4.1':<35s} {'—':>10} {m.get('losses_vs_gpt41','?'):>10} {na.get('losses_vs_gpt41','?'):>10}")
    logger.info(f"{'Ties vs GPT-4.1':<35s} {'—':>10} {m.get('ties_vs_gpt41','?'):>10} {na.get('ties_vs_gpt41','?'):>10}")

    logger.info(f"\n{'─' * 65}")
    logger.info("TRANSFER FEASIBILITY (vs GPT-4.1)")
    logger.info(f"{'─' * 65}")
    logger.info(f"{'Tetrachoric r':<35s} {'—':>10} {m.get('tetrachoric_vs_gpt41','?'):>10} {na.get('tetrachoric_vs_gpt41','?'):>10}")
    logger.info(f"{'Pearson r':<35s} {'—':>10} {m.get('pearson_vs_gpt41','?'):>10} {na.get('pearson_vs_gpt41','?'):>10}")
    logger.info(f"{'Agreement rate':<35s} {'—':>10} {m.get('agreement_vs_gpt41','?'):>10} {na.get('agreement_vs_gpt41','?'):>10}")
    logger.info(f"{'Logic agreement':<35s} {'—':>10} {m.get('logic_agreement_vs_gpt41','?'):>10} {na.get('logic_agreement_vs_gpt41','?'):>10}")
    logger.info(f"{'VERDICT':<35s} {'—':>10} {m.get('transfer_verdict','?'):>10} {na.get('transfer_verdict','?'):>10}")

    logger.info(f"\nMini vs Nano tetrachoric: {analysis.get('mini_vs_nano_tetrachoric', '?')}")

    logger.info(f"\nAnalysis -> {analysis_path}")


if __name__ == "__main__":
    main()
