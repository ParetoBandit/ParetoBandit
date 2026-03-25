#!/usr/bin/env python3
"""Build a stratified subset of the 12K pareto dataset and re-judge with
GPT-4.1-mini and Claude-3.7-Sonnet for judge-robustness analysis.

This script serves two purposes:

1. **Sample**: Draw a stratified subset of prompts from the canonical 12K
   pareto dataset, preserving the source and difficulty distributions.
2. **Re-judge**: Send existing (prompt, response) pairs to two additional
   judges (GPT-4.1-mini, Claude-3.7-Sonnet) via the ``CoTRewardGenerator``
   infrastructure.  No new model responses are generated — only judge
   inference calls.

The output is a JSONL file in the same format as the main rewards file,
but with ``judge_details`` containing scores from the two supplementary
judges.  A companion analysis script (or notebook) then compares these
scores against the primary DeepSeek-R1 scores to validate that routing
decisions are robust to judge choice.

Usage
-----
    # Phase 1: Sample a 2K stratified subset (no API calls)
    python data_collection/scripts/judge_robustness_subset.py --sample-only

    # Phase 2: Re-judge the subset with supplementary judges
    python data_collection/scripts/judge_robustness_subset.py

    # Smaller subset for testing
    python data_collection/scripts/judge_robustness_subset.py --n-prompts 100

    # Resume after interruption
    python data_collection/scripts/judge_robustness_subset.py --resume

Requirements
------------
    export OPENROUTER_API_KEY=...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_collection.scripts.rejudge_cot import CoTRewardGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
PARETO_REWARDS_PATH = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
PARETO_CLASSIFIED_PATH = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_classified.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "calibration"
SUBSET_PROMPTS_PATH = OUTPUT_DIR / "judge_robustness_prompts.jsonl"
SUBSET_REWARDS_PATH = OUTPUT_DIR / "judge_robustness_rewards.jsonl"

# Supplementary judges (DeepSeek-R1 is the primary; these two are additive).
SUPPLEMENTARY_JUDGES: List[str] = [
    "openai/gpt-4.1-mini",
    "anthropic/claude-3.7-sonnet",
]

DEFAULT_N_PROMPTS: int = 2000
RNG_SEED: int = 2026


# =========================================================================
# Phase 1: Stratified sampling
# =========================================================================


def load_classified_prompts(
    classified_path: Path = PARETO_CLASSIFIED_PATH,
) -> List[Dict[str, Any]]:
    """Load post-hoc classified prompt records.

    Parameters
    ----------
    classified_path:
        Path to ``pareto_classified.jsonl``.

    Returns
    -------
    list[dict]
        Records with ``prompt``, ``source``, ``difficulty``, etc.
    """
    records: List[Dict[str, Any]] = []
    with open(classified_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d classified prompts from %s", len(records), classified_path.name)
    return records


def stratified_sample(
    records: List[Dict[str, Any]],
    n_prompts: int,
    seed: int = RNG_SEED,
) -> List[Dict[str, Any]]:
    """Draw a stratified subset preserving source distribution.

    Stratification is by top-level ``source`` (benchmark family, e.g.
    ``"mmlu"`` or ``"bbh"`` — all BBH sub-tasks are grouped together).
    Within each stratum, samples proportionally.  If a stratum has fewer
    prompts than its proportional allocation, all prompts are included and
    the surplus is redistributed to larger strata.

    Parameters
    ----------
    records:
        Full classified prompt list.
    n_prompts:
        Target subset size.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Stratified subset of prompt records.
    """
    rng = np.random.default_rng(seed)

    def _source_family(source: str) -> str:
        return source.split("/")[0] if "/" in source else source

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_family[_source_family(r.get("source", "unknown"))].append(r)

    total = len(records)
    selected: List[Dict[str, Any]] = []
    remaining_budget = n_prompts

    # Two passes: first take all from strata smaller than their allocation,
    # then proportionally sample from the rest.
    allocated: Dict[str, int] = {}
    for family, pool in by_family.items():
        proportional = max(1, int(round(len(pool) / total * n_prompts)))
        allocated[family] = min(proportional, len(pool))

    # If total allocated < target, scale up large strata; if > target, scale down.
    while sum(allocated.values()) != n_prompts:
        diff = n_prompts - sum(allocated.values())
        adjustable = [
            f for f in allocated
            if (diff > 0 and allocated[f] < len(by_family[f]))
            or (diff < 0 and allocated[f] > 1)
        ]
        if not adjustable:
            break
        step = 1 if diff > 0 else -1
        for f in sorted(adjustable, key=lambda x: len(by_family[x]), reverse=(diff > 0)):
            allocated[f] += step
            diff -= step
            if diff == 0:
                break

    for family, pool in by_family.items():
        n = allocated[family]
        if n >= len(pool):
            selected.extend(pool)
        else:
            indices = rng.choice(len(pool), size=n, replace=False)
            selected.extend(pool[i] for i in indices)

    rng.shuffle(selected)

    source_counts = defaultdict(int)
    diff_counts = defaultdict(int)
    for r in selected:
        source_counts[r.get("source", "unknown")] += 1
        diff_counts[r.get("difficulty", "unknown")] += 1

    logger.info("Stratified sample: %d prompts (target: %d)", len(selected), n_prompts)
    logger.info("  Sources: %s", dict(sorted(source_counts.items(), key=lambda x: -x[1])))
    logger.info("  Difficulty: %s", dict(sorted(diff_counts.items(), key=lambda x: -x[1])))
    return selected


def save_subset_prompts(
    subset: List[Dict[str, Any]],
    output_path: Path = SUBSET_PROMPTS_PATH,
) -> None:
    """Write the subset prompt list to JSONL.

    Parameters
    ----------
    subset:
        Stratified subset of classified records.
    output_path:
        Destination file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d subset prompts to %s", len(subset), output_path)


# =========================================================================
# Phase 2: Re-judge with supplementary panel
# =========================================================================


def load_existing_responses(
    rewards_path: Path = PARETO_REWARDS_PATH,
    prompt_set: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], str]:
    """Load existing (prompt, model_id) → response mappings.

    Parameters
    ----------
    rewards_path:
        Path to ``pareto_rewards.jsonl``.
    prompt_set:
        If provided, only load responses for prompts in this set.

    Returns
    -------
    dict
        ``{(model_id, prompt): response_text}`` mapping.
    """
    responses: Dict[Tuple[str, str], str] = {}
    n_loaded = 0
    with open(rewards_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            prompt = rec["prompt"]
            if prompt_set is not None and prompt not in prompt_set:
                continue
            key = (rec["model_id"], prompt)
            if key not in responses and rec.get("response"):
                responses[key] = rec["response"]
                n_loaded += 1

    logger.info(
        "Loaded %d (model, prompt) → response pairs from %s",
        n_loaded, rewards_path.name,
    )
    return responses


def build_rejudge_tasks(
    subset: List[Dict[str, Any]],
    responses: Dict[Tuple[str, str], str],
) -> List[Tuple[str, str, str]]:
    """Build (prompt, response, model_id) triples for re-judging.

    Parameters
    ----------
    subset:
        Stratified prompt subset.
    responses:
        Existing response cache.

    Returns
    -------
    list[tuple]
        Triples of (prompt_text, response_text, model_id).
    """
    tasks: List[Tuple[str, str, str]] = []
    missing = 0

    for rec in subset:
        prompt = rec["prompt"]
        for model_id in rec.get("arms", {}).keys():
            key = (model_id, prompt)
            if key in responses:
                tasks.append((prompt, responses[key], model_id))
            else:
                missing += 1

    if missing > 0:
        logger.warning("  %d (prompt, model) pairs missing responses — skipped", missing)
    logger.info("Built %d re-judge tasks", len(tasks))
    return tasks


def run_rejudging(
    tasks: List[Tuple[str, str, str]],
    output_path: Path = SUBSET_REWARDS_PATH,
    judges: List[str] = SUPPLEMENTARY_JUDGES,
    workers: int = 10,
) -> int:
    """Re-judge all tasks with the supplementary judge panel.

    Uses ``CoTRewardGenerator`` with the judge panel overridden to contain
    only the supplementary judges.  Tasks are processed in parallel via a
    thread pool (each task internally also parallelises across judges).
    Results are written incrementally to ``output_path`` with resume support.

    Parameters
    ----------
    tasks:
        (prompt, response, model_id) triples.
    output_path:
        Destination JSONL.
    judges:
        Judge model IDs.
    workers:
        Concurrent tasks (each task fans out to ``len(judges)`` API calls).

    Returns
    -------
    int
        Number of new records written.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gen = CoTRewardGenerator(max_workers=workers)
    gen.judge_panel = list(judges)
    logger.info("Judge panel: %s", gen.judge_panel)

    completed: Set[Tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        completed.add((rec["prompt"], rec["model_id"]))
                except json.JSONDecodeError:
                    continue
        if completed:
            logger.info("Resume: %d tasks already completed", len(completed))

    remaining = [
        t for t in tasks if (t[0], t[2]) not in completed
    ]
    logger.info("Tasks to run: %d (skipped %d)", len(remaining), len(tasks) - len(remaining))

    if not remaining:
        logger.info("Nothing to do.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    write_lock = threading.Lock()

    with open(output_path, "a") as outfile:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(gen._process_rejudge_task, task): task
                for task in remaining
            }
            with tqdm(total=len(remaining), desc="Re-judging") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    with write_lock:
                        outfile.write(
                            json.dumps(result, ensure_ascii=False) + "\n",
                        )
                        outfile.flush()
                        written += 1
                    pbar.update(1)

    diag_path = output_path.with_suffix(".judge_diagnostics.json")
    diag_payload = {
        "judges": judges,
        "n_tasks": len(tasks),
        "n_written": written,
        "summary": gen.diagnostics.per_judge_summary(),
        "bias_matrix": gen.diagnostics.bias_matrix(),
    }
    with open(diag_path, "w") as df:
        json.dump(diag_payload, df, indent=2)
    logger.info("Judge diagnostics: %s", diag_path)
    logger.info("Wrote %d new records to %s", written, output_path)
    return written


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stratified subset of the 12K pareto dataset and "
            "re-judge with supplementary judges for robustness analysis."
        ),
    )
    parser.add_argument(
        "--n-prompts", type=int, default=DEFAULT_N_PROMPTS,
        help=f"Number of prompts to sample (default: {DEFAULT_N_PROMPTS}).",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Only build the stratified subset; skip re-judging.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous re-judging run.",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel workers for judge API calls (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED,
        help=f"Random seed for sampling (default: {RNG_SEED}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    subset_prompts_path = output_dir / "judge_robustness_prompts.jsonl"
    subset_rewards_path = output_dir / "judge_robustness_rewards.jsonl"

    logger.info("=" * 70)
    logger.info("Judge Robustness Subset Builder")
    logger.info("=" * 70)
    logger.info("  Target prompts : %d", args.n_prompts)
    logger.info("  Judges         : %s", SUPPLEMENTARY_JUDGES)
    logger.info("  Output dir     : %s", output_dir)
    logger.info("  Seed           : %d", args.seed)

    # ── Phase 1: Stratified sampling ──────────────────────────────────
    if args.resume and subset_prompts_path.exists():
        logger.info("\n--- Phase 1: Loading existing subset (--resume) ---")
        subset: List[Dict[str, Any]] = []
        with open(subset_prompts_path) as f:
            for line in f:
                subset.append(json.loads(line))
        logger.info("Loaded %d subset prompts", len(subset))
    else:
        logger.info("\n--- Phase 1: Stratified sampling ---")
        classified = load_classified_prompts()
        subset = stratified_sample(classified, args.n_prompts, seed=args.seed)
        save_subset_prompts(subset, subset_prompts_path)

    if args.sample_only:
        logger.info("\n--sample-only: skipping re-judging.")
        _print_cost_estimate(subset)
        logger.info("Done.")
        return

    # ── Phase 2: Re-judge with supplementary panel ────────────────────
    logger.info("\n--- Phase 2: Load existing responses ---")
    prompt_set = {r["prompt"] for r in subset}
    responses = load_existing_responses(prompt_set=prompt_set)

    logger.info("\n--- Phase 3: Re-judge with supplementary panel ---")
    tasks = build_rejudge_tasks(subset, responses)
    _print_cost_estimate(subset, tasks)
    written = run_rejudging(
        tasks,
        output_path=subset_rewards_path,
        workers=args.workers,
    )

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Subset prompts : %s", subset_prompts_path)
    logger.info("  Subset rewards : %s", subset_rewards_path)
    logger.info("  Records written: %d", written)
    logger.info("\nTo analyze, compare R1 scores from pareto_rewards.jsonl")
    logger.info("against supplementary judge scores in %s", subset_rewards_path.name)
    logger.info("Done.")


def _print_cost_estimate(
    subset: List[Dict[str, Any]],
    tasks: Optional[List[Tuple[str, str, str]]] = None,
) -> None:
    """Print an estimated API cost for the re-judging run.

    Parameters
    ----------
    subset:
        Sampled prompt records.
    tasks:
        If available, use the actual task count; otherwise estimate from
        the subset size.
    """
    n_prompts = len(subset)
    if tasks is not None:
        n_pairs = len(tasks)
    else:
        n_models = len(set(
            m for r in subset for m in r.get("arms", {}).keys()
        ))
        n_pairs = n_prompts * n_models

    n_judges = len(SUPPLEMENTARY_JUDGES)
    n_calls = n_pairs * n_judges
    avg_input_tokens = 913  # ~413 response + ~500 rubric/system
    avg_output_tokens = 300

    gpt_calls = n_pairs
    claude_calls = n_pairs
    gpt_cost = (gpt_calls * avg_input_tokens * 0.40 + gpt_calls * avg_output_tokens * 1.60) / 1_000_000
    claude_cost = (claude_calls * avg_input_tokens * 0.80 + claude_calls * avg_output_tokens * 4.00) / 1_000_000

    logger.info("\n--- Cost Estimate ---")
    logger.info("  Prompts: %d, (prompt,model) pairs: %d", n_prompts, n_pairs)
    logger.info("  Judge calls: %d (%d per judge)", n_calls, n_pairs)
    logger.info("  GPT-4.1-mini      : ~$%.2f", gpt_cost)
    logger.info("  Claude-3.7-Sonnet : ~$%.2f", claude_cost)
    logger.info("  Total estimate: ~$%.2f", gpt_cost + claude_cost)


if __name__ == "__main__":
    main()
