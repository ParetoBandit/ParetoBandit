#!/usr/bin/env python3
"""Judge comparison test: re-judge existing responses with o3 and
Claude Opus 4.6 to evaluate discrimination between model tiers.

Samples prompts where all 4 candidate models have existing responses,
then judges each (prompt, response) pair with both o3 and Opus 4.6
using the v4b "Senior Technical Auditor" rubric.  Compares per-judge
discrimination against the baseline R1 scores.

Usage
-----
    python data_collection/scripts/judge_comparison_test.py
    python data_collection/scripts/judge_comparison_test.py --n-prompts 50
    python data_collection/scripts/judge_comparison_test.py --summary-only

Requirements
------------
    export OPENROUTER_API_KEY=...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────

PARETO_REWARDS = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
MIDTIER_DIR = PROJECT_ROOT / "data_collection" / "midtier_candidates"
V4_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v4_metacognitive"
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "judge_comparison"

CANDIDATE_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

JUDGE_MODELS = [
    "openai/o3",
    "anthropic/claude-opus-4.6",
]

SEED = 42

# ── API config ──────────────────────────────────────────────────────────

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        API_KEY = os.getenv("OPENROUTER_API_KEY")
    except ImportError:
        pass
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set")

BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# ── V4b Rubric: Senior Technical Auditor ────────────────────────────────

V4B_SYSTEM_PROMPT = (
    "You are a Senior Technical Auditor. Your goal is to distinguish 'Standard Correctness' "
    "from 'Frontier Intelligence.'\n\n"

    "## AUDIT PROTOCOL\n"
    "1. REASONING TRACE: Before grading, reconstruct the model's logic. If you find a 'jump' "
    "where the model assumes a result without showing the work, cap Reasoning at 0.7.\n"
    "2. THE 'FLASH' TRAP: Flash-tier models are often 'correct' but 'shallow.' Look for "
    "repetitive sentence structures or a lack of specific, technical detail.\n"
    "3. FRONTIER REWARD: Only give >0.92 if the model provides 'Hidden Value'—this means "
    "addressing an unstated but relevant edge case or providing a more elegant solution than requested.\n\n"

    "## SCORING CRITERIA\n"
    "### Factual Correctness (50%)\n"
    "- 1.0: Zero errors, high precision.\n"
    "- 0.8: Correct but uses vague language to avoid being wrong.\n"
    "- 0.4: Correct final answer, but used a hallucinated fact to get there.\n\n"

    "### Reasoning Depth (35%)\n"
    "- 0.95+: 'System 2' thinking. Multi-step, non-linear, anticipates failure modes.\n"
    "- 0.80-0.90: 'Standard RLHF' style. Clean, helpful, but follows a generic template.\n"
    "- 0.50-0.79: 'Pattern Matching.' Linear logic that breaks if the problem changes.\n\n"

    "### Completeness & Nuance (15%)\n"
    "- 1.0: Addresses the prompt AND its implications.\n"
    "- 0.8: Addresses all explicit tokens in the prompt.\n\n"

    "## OUTPUT MANDATE\n"
    "You must output a 'Tier Label' [Low | Mid | Frontier] before the scores. "
    "If you label a model 'Frontier,' you must explain exactly one 'Deep Insight' it provided "
    "that a standard model would miss."
)

W_CORRECTNESS = 0.50
W_REASONING = 0.35
W_COMPLETENESS = 0.15

# Judge-specific timeouts and token limits.
JUDGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai/o3": {"max_tokens": 4096, "timeout": 240.0},
    "anthropic/claude-opus-4.6": {"max_tokens": 4096, "timeout": 300.0},
}

# ── Score parsing ───────────────────────────────────────────────────────


def _parse_score(content: str, heading: str, *, default: float = 0.5) -> float:
    """Extract a continuous 0.0-1.0 score from various markdown formats."""
    patterns = [
        r"#{1,3}\s*" + heading + r"\s*[:\-]?\s*(\d+\.?\d*)",
        r"\*\*" + heading + r"[:\*]*\s*\[?\s*(\d+\.?\d*)",
        r"\d+\.\s*\*\*" + heading + r"[:\*]*\s*\[?\s*(\d+\.?\d*)",
        heading + r"\s*[:\-]\s*\[?\s*(\d+\.?\d*)",
    ]
    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            return max(0.0, min(1.0, val))
    return default


def _parse_tier(content: str) -> str:
    """Extract tier label from judge output."""
    m = re.search(r"\b(Low|Mid|Frontier)\b", content)
    return m.group(1).capitalize() if m else "Unknown"


# ── Response loading ────────────────────────────────────────────────────


def load_responses() -> Dict[Tuple[str, str], str]:
    """Load all available responses from existing data sources.

    Returns:
        Mapping of (model_id, prompt) → response text.
    """
    cache: Dict[Tuple[str, str], str] = {}
    target_models = set(CANDIDATE_MODELS)

    if PARETO_REWARDS.exists():
        with open(PARETO_REWARDS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok") and r.get("response") and r["model_id"] in target_models:
                        cache[(r["model_id"], r["prompt"])] = r["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("  Loaded %d from pareto_rewards.jsonl", len(cache))

    for jsonl_path in sorted(MIDTIER_DIR.glob("*.jsonl")):
        count_before = len(cache)
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok") and r.get("response") and r["model_id"] in target_models:
                        key = (r["model_id"], r["prompt"])
                        if key not in cache:
                            cache[key] = r["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        added = len(cache) - count_before
        if added:
            logger.info("  Loaded %d from %s", added, jsonl_path.name)

    # Flash responses from v4 collection.
    flash_path = V4_DIR / "google_gemini-2.5-flash_v4.jsonl"
    if flash_path.exists():
        count_before = len(cache)
        with open(flash_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok") and r.get("response"):
                        key = (r["model_id"], r["prompt"])
                        if key not in cache:
                            cache[key] = r["response"]
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("  Loaded %d Flash responses from v4", len(cache) - count_before)

    return cache


def find_overlap_prompts(
    cache: Dict[Tuple[str, str], str],
    models: List[str],
) -> List[str]:
    """Find prompts where all models have responses."""
    prompt_models: Dict[str, Set[str]] = defaultdict(set)
    for (mid, prompt) in cache:
        if mid in models:
            prompt_models[prompt].add(mid)
    model_set = set(models)
    return [p for p, ms in prompt_models.items() if ms >= model_set]


# ── Judging ─────────────────────────────────────────────────────────────


def judge_single(
    judge_model: str,
    prompt: str,
    response: str,
) -> Optional[Dict[str, Any]]:
    """Judge a (prompt, response) pair with a single judge model.

    Returns:
        Dict with scores and metadata, or None on failure.
    """
    config = JUDGE_CONFIG.get(judge_model, {"max_tokens": 4096, "timeout": 240.0})
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
    }
    payload: Dict[str, Any] = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": V4B_SYSTEM_PROMPT},
            {"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}"},
        ],
        "temperature": 0.0,
        "max_tokens": config["max_tokens"],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config["timeout"],
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("API returned null content")
            content = content.strip()

            correctness = _parse_score(content, r"Factual\s+Correctness")
            reasoning = _parse_score(content, r"Reasoning\s+Depth")
            completeness = _parse_score(content, r"Completeness\s*(?:&|and)?\s*Nuance")

            composite = (
                correctness * W_CORRECTNESS
                + reasoning * W_REASONING
                + completeness * W_COMPLETENESS
            )

            tier = _parse_tier(content)

            return {
                "correctness_score": round(correctness, 4),
                "reasoning_score": round(reasoning, 4),
                "completeness_score": round(completeness, 4),
                "raw_score": round(composite, 4),
                "tier_classification": tier,
                "judge_raw": content,
            }

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
            else:
                logger.warning("Judge %s HTTP %d: %s", judge_model, status, e)
                return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
            else:
                logger.warning("Judge %s failed: %s", judge_model, e)
                return None

    return None


# ── Main pipeline ───────────────────────────────────────────────────────


def load_r1_baseline() -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load R1 scores from the v4 collection for comparison."""
    baseline: Dict[Tuple[str, str], Dict[str, float]] = {}
    for model_id in CANDIDATE_MODELS:
        slug = model_id.replace("/", "_")
        path = V4_DIR / f"{slug}_v4.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        baseline[(r["model_id"], r["prompt"])] = {
                            "correctness": r.get("correctness_score", 0),
                            "reasoning": r.get("reasoning_score", 0),
                            "completeness": r.get("completeness_score", 0),
                            "composite": r.get("raw_score", 0),
                        }
                except (json.JSONDecodeError, KeyError):
                    continue
    return baseline


def run_comparison(
    n_prompts: int = 200,
    workers: int = 6,
) -> None:
    """Run the judge comparison test."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load responses.
    logger.info("Loading response cache...")
    cache = load_responses()
    logger.info("Total cached responses: %d", len(cache))

    overlap = find_overlap_prompts(cache, CANDIDATE_MODELS)
    logger.info("Prompts with all 4 models: %d", len(overlap))

    rng = random.Random(SEED)
    rng.shuffle(overlap)
    sample = overlap[:n_prompts]
    logger.info("Sampled %d prompts for test", len(sample))

    # 2. Build task list: (judge, model, prompt, response).
    tasks: List[Tuple[str, str, str, str]] = []
    for prompt in sample:
        for model_id in CANDIDATE_MODELS:
            response = cache[(model_id, prompt)]
            for judge in JUDGE_MODELS:
                tasks.append((judge, model_id, prompt, response))

    rng.shuffle(tasks)
    logger.info("Total judge tasks: %d", len(tasks))

    # 3. Load completed tasks for resume.
    completed: Set[Tuple[str, str, str]] = set()  # (judge, model, prompt)
    for judge in JUDGE_MODELS:
        slug = judge.replace("/", "_")
        out_path = OUTPUT_DIR / f"{slug}_comparison.jsonl"
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if r.get("ok"):
                            completed.add((r["judge_model"], r["model_id"], r["prompt"]))
                    except (json.JSONDecodeError, KeyError):
                        continue

    remaining = [t for t in tasks if (t[0], t[1], t[2]) not in completed]
    logger.info("Remaining tasks: %d (skipping %d completed)", len(remaining), len(tasks) - len(remaining))

    if not remaining:
        logger.info("All tasks completed.")
    else:
        # 4. Open output files.
        out_files: Dict[str, Any] = {}
        for judge in JUDGE_MODELS:
            slug = judge.replace("/", "_")
            out_files[judge] = open(OUTPUT_DIR / f"{slug}_comparison.jsonl", "a")

        lock = threading.Lock()

        # 5. Run.
        def process(task: Tuple[str, str, str, str]) -> Dict[str, Any]:
            judge, model_id, prompt, response = task
            result = judge_single(judge, prompt, response)
            if result is None:
                return {
                    "judge_model": judge,
                    "model_id": model_id,
                    "prompt": prompt,
                    "ok": False,
                    "ts": time.time(),
                }
            return {
                "judge_model": judge,
                "model_id": model_id,
                "prompt": prompt,
                "ok": True,
                "correctness_score": result["correctness_score"],
                "reasoning_score": result["reasoning_score"],
                "completeness_score": result["completeness_score"],
                "raw_score": result["raw_score"],
                "tier_classification": result["tier_classification"],
                "judge_raw": result["judge_raw"],
                "ts": time.time(),
            }

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process, t): t for t in remaining}
                with tqdm(total=len(remaining), desc="Judge Comparison") as pbar:
                    for fut in as_completed(futures):
                        res = fut.result()
                        judge = res["judge_model"]
                        with lock:
                            f = out_files.get(judge)
                            if f:
                                f.write(json.dumps(res) + "\n")
                                f.flush()
                        pbar.update(1)
        finally:
            for f in out_files.values():
                f.close()

    # 6. Analysis.
    print_analysis(sample)


def print_analysis(sample_prompts: Optional[List[str]] = None) -> None:
    """Print discrimination analysis comparing all judges."""
    # Load all judge results.
    all_judges = ["deepseek/deepseek-r1"] + JUDGE_MODELS
    judge_scores: Dict[str, Dict[str, List[float]]] = {
        j: {m: [] for m in CANDIDATE_MODELS} for j in all_judges
    }
    judge_tiers: Dict[str, Dict[str, Dict[str, int]]] = {
        j: {m: defaultdict(int) for m in CANDIDATE_MODELS} for j in all_judges
    }

    # R1 baseline from v4 collection.
    r1_data = load_r1_baseline()
    prompt_filter = set(sample_prompts) if sample_prompts else None

    for (mid, prompt), scores in r1_data.items():
        if prompt_filter and prompt not in prompt_filter:
            continue
        if mid in CANDIDATE_MODELS:
            judge_scores["deepseek/deepseek-r1"][mid].append(scores["composite"])

    # o3 and Opus from comparison files.
    for judge in JUDGE_MODELS:
        slug = judge.replace("/", "_")
        path = OUTPUT_DIR / f"{slug}_comparison.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("ok"):
                        continue
                    mid = r["model_id"]
                    if mid in CANDIDATE_MODELS:
                        judge_scores[judge][mid].append(r["raw_score"])
                        judge_tiers[judge][mid][r.get("tier_classification", "Unknown")] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    # Print comparison table.
    print("\n" + "=" * 90)
    print("JUDGE DISCRIMINATION COMPARISON")
    print("=" * 90)

    for judge in all_judges:
        short_judge = judge.split("/")[-1]
        scores = judge_scores[judge]
        means = {}
        print(f"\n{'─' * 40} {short_judge} {'─' * 40}")
        print(f"  {'Model':<35s} {'N':>5s} {'Mean':>7s} {'Std':>7s} {'Median':>7s}")
        print(f"  {'─' * 65}")

        for mid in CANDIDATE_MODELS:
            arr = np.array(scores[mid])
            if len(arr) == 0:
                print(f"  {mid.split('/')[-1]:<35s} {'[no data]':>5s}")
                continue
            means[mid] = float(arr.mean())
            print(
                f"  {mid.split('/')[-1]:<35s} {len(arr):5d} "
                f"{arr.mean():7.3f} {arr.std():7.3f} {np.median(arr):7.3f}"
            )

        if len(means) >= 2:
            ordered = [(m, means[m]) for m in CANDIDATE_MODELS if m in means]
            ordered.sort(key=lambda x: x[1])
            print(f"\n  Gaps:")
            for i in range(len(ordered) - 1):
                a_name = ordered[i][0].split("/")[-1]
                b_name = ordered[i + 1][0].split("/")[-1]
                delta = ordered[i + 1][1] - ordered[i][1]
                print(f"    {a_name} → {b_name}: {delta:+.4f}")
            total = ordered[-1][1] - ordered[0][1]
            print(f"    TOTAL SPREAD: {total:.4f}")

        # Tier distribution for non-R1 judges.
        if judge in JUDGE_MODELS:
            tiers = judge_tiers[judge]
            print(f"\n  Tier distribution:")
            print(f"  {'Model':<35s} {'Low':>6s} {'Mid':>6s} {'Front':>6s} {'Unk':>6s}")
            print(f"  {'─' * 62}")
            for mid in CANDIDATE_MODELS:
                t = tiers[mid]
                total_t = sum(t.values())
                if total_t == 0:
                    continue
                print(
                    f"  {mid.split('/')[-1]:<35s} "
                    f"{t.get('Low', 0):6d} {t.get('Mid', 0):6d} "
                    f"{t.get('Frontier', 0):6d} {t.get('Unknown', 0):6d}"
                )

    # Head-to-head summary.
    print(f"\n{'=' * 90}")
    print("DISCRIMINATION SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'Judge':<25s} {'3B→8B':>8s} {'8B→Flash':>9s} {'Flash→Pro':>10s} {'Total':>8s}")
    print(f"  {'─' * 62}")

    for judge in all_judges:
        scores = judge_scores[judge]
        means = {}
        for mid in CANDIDATE_MODELS:
            arr = np.array(scores[mid])
            if len(arr) > 0:
                means[mid] = float(arr.mean())

        if len(means) < 4:
            short = judge.split("/")[-1]
            print(f"  {short:<25s} [incomplete data]")
            continue

        g1 = means[CANDIDATE_MODELS[1]] - means[CANDIDATE_MODELS[0]]
        g2 = means[CANDIDATE_MODELS[2]] - means[CANDIDATE_MODELS[1]]
        g3 = means[CANDIDATE_MODELS[3]] - means[CANDIDATE_MODELS[2]]
        total = means[CANDIDATE_MODELS[3]] - means[CANDIDATE_MODELS[0]]

        short = judge.split("/")[-1]
        print(f"  {short:<25s} {g1:+8.4f} {g2:+9.4f} {g3:+10.4f} {total:8.4f}")

    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-prompts", type=int, default=200,
        help="Number of prompts to sample (default: 200).",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Parallel workers (default: 6).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip judging, just print analysis of existing results.",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_analysis()
        return

    run_comparison(n_prompts=args.n_prompts, workers=args.workers)


if __name__ == "__main__":
    main()
