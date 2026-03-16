#!/usr/bin/env python3
"""
Replace one judge in the canonical rewards with a new judge model.

Performs a *selective* re-judge: for every record in ``rewards.jsonl``,
calls the new judge on the existing ``(prompt, response)`` pair, replaces
the old judge's entry in ``judge_details``, and recomputes ``raw_score``.
Existing scores from the other two judges are kept as-is.

This is dramatically cheaper than a full re-judge (1 API call per record
instead of 3).

Default swap: ``qwen/qwen-2.5-72b-instruct`` -> ``openai/gpt-4.1-mini``.

Usage::

    # Run the selective re-judge (writes rewards_new_panel.jsonl)
    python data_collection/scripts/swap_judge.py run

    # Finalize: replace rewards.jsonl with the new-panel version
    python data_collection/scripts/swap_judge.py finalize

    # Then re-split:
    python data_collection/scripts/merge_and_split_rewards.py \\
        --input data_collection/rewards/rewards.jsonl \\
        --models-file data_collection/config/models_k4.json \\
        --output-dir data_collection/rewards \\
        --prefix "" --no-panel-filter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from pareto_bandit.config import REWARDS_PATH  # noqa: E402

logger = logging.getLogger(__name__)

OLD_JUDGE = "qwen/qwen-2.5-72b-instruct"
NEW_JUDGE = "openai/gpt-4.1-mini"

OUTPUT_PATH = REWARDS_PATH.parent / "rewards_new_panel.jsonl"

_W_REASONING: float = 0.40
_W_INSTRUCTION: float = 0.30
_W_COMMUNICATION: float = 0.30

_MAX_RETRIES: int = 4
_RETRY_BACKOFF_BASE: float = 2.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = (
    "You are a Discriminative Router Judge. Your goal is to evaluate "
    "how well an LLM response addresses the given prompt.\n\n"
    "Score on three continuous dimensions (0.0–1.0). Use the FULL "
    "range; do NOT default to 0 or 1.\n\n"
    "1. **Reasoning Quality (40 %)** — How sound is the reasoning?\n"
    "   0.9–1.0 Flawless; every step correct and clearly justified.\n"
    "   0.7–0.8 Sound overall; minor inefficiency or a trivial error "
    "that does not change the conclusion.\n"
    "   0.5–0.6 Partially correct; approach is reasonable but "
    "important steps are wrong or missing.\n"
    "   0.3–0.4 Weak; only fragments of correct logic.\n"
    "   0.0–0.2 No coherent reasoning, or completely wrong approach.\n"
    "   If the prompt needs no multi-step reasoning, score factual "
    "accuracy and depth of explanation.\n\n"
    "2. **Instruction Following (30 %)** — Were all explicit and "
    "implicit constraints satisfied?\n"
    "   0.9–1.0 Every constraint followed precisely.\n"
    "   0.7–0.8 All major constraints met; one minor instruction "
    "partially missed.\n"
    "   0.5–0.6 Some important instructions missed or only partially "
    "addressed.\n"
    "   0.3–0.4 Multiple instructions ignored or misinterpreted.\n"
    "   0.0–0.2 Response largely ignores the prompt's requirements.\n\n"
    "3. **Communication Quality (30 %)** — How clear, well-structured, "
    "and useful is the response?\n"
    "   0.9–1.0 Exceptionally clear, well-organized, appropriate "
    "detail.\n"
    "   0.7–0.8 Clear and competent; minor improvements possible.\n"
    "   0.5–0.6 Adequate but noticeably unclear, verbose, or poorly "
    "organized.\n"
    "   0.3–0.4 Hard to follow; significant clarity issues.\n"
    "   0.0–0.2 Unintelligible, unhelpful, or inappropriate tone.\n\n"
    "Format your response EXACTLY as follows:\n\n"
    "## Reasoning\n"
    "<Concise chain-of-thought analysis>\n\n"
    "## Reasoning Quality\n"
    "<0.0 to 1.0>\n\n"
    "## Instruction Following\n"
    "<0.0 to 1.0>\n\n"
    "## Communication Quality\n"
    "<0.0 to 1.0>"
)


def _parse_continuous_score(
    content: str, heading: str, *, default: float = 0.5,
) -> float:
    """Extract a continuous 0.0-1.0 score from a markdown heading."""
    pattern = r"##\s*" + heading + r"\s*[:\-]?\s*(\d+\.?\d*)"
    m = re.search(pattern, content, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    return default


def _call_judge(
    api_key: str,
    prompt: str,
    response: str,
    judge_model: str = NEW_JUDGE,
    max_tokens: int = 4000,
    timeout: float = 90.0,
) -> Dict[str, Any] | None:
    """Call a single judge model and parse the rubric response.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff on
    transient failures (timeouts, 429, 500, 502-504).

    Returns:
        Parsed rubric dict, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
    }
    user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            if raw_content is None:
                raise ValueError("API returned null content")
            content = raw_content.strip()

            reasoning_quality = _parse_continuous_score(
                content, r"Reasoning\s+Quality",
            )
            instruction_following = _parse_continuous_score(
                content, r"Instruction\s+Following",
            )
            communication_quality = _parse_continuous_score(
                content, r"Communication\s+Quality",
            )
            reward = (
                reasoning_quality * _W_REASONING
                + instruction_following * _W_INSTRUCTION
                + communication_quality * _W_COMMUNICATION
            )

            reasoning = content
            rm = re.search(
                r"##\s*Reasoning\s*(.*?)(\n##|$)",
                content, re.DOTALL | re.IGNORECASE,
            )
            if rm:
                reasoning = rm.group(1).strip()

            return {
                "judge": judge_model,
                "reasoning_quality": round(reasoning_quality, 4),
                "instruction_following": round(instruction_following, 4),
                "communication_quality": round(communication_quality, 4),
                "reward": round(reward, 4),
                "reasoning": reasoning,
                "logic": round(reasoning_quality, 4),
                "constraint": round(instruction_following, 4),
                "utility": round(communication_quality, 4),
            }

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            ValueError,
            KeyError,
            IndexError,
        ) as e:
            last_exc = e
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else 0
            if status in _RETRYABLE_STATUS_CODES:
                pass
            else:
                logger.warning(
                    "Judge %s non-retryable HTTP %d", judge_model, status,
                )
                return None
        except Exception as e:
            logger.warning("Judge %s unexpected error: %s", judge_model, e)
            return None

        backoff = _RETRY_BACKOFF_BASE ** attempt
        logger.debug(
            "Judge %s attempt %d/%d failed (%s), retrying in %.1fs",
            judge_model, attempt, _MAX_RETRIES, last_exc, backoff,
        )
        time.sleep(backoff)

    logger.warning(
        "Judge %s failed after %d attempts: %s",
        judge_model, _MAX_RETRIES, last_exc,
    )
    return None


def _recompute_raw_score(judge_details: List[Dict[str, Any]]) -> float:
    """Recompute raw_score as the mean of per-judge composite rewards."""
    rewards = [
        jd["reward"]
        for jd in judge_details
        if np.isfinite(jd.get("reward", float("nan")))
    ]
    if not rewards:
        return float("nan")
    return float(np.mean(rewards))


def _logit_transform(score: float) -> float:
    if np.isnan(score):
        return float("nan")
    score = np.clip(score, 0.01, 0.99)
    return float(np.log(score / (1 - score)))


def _load_completed_keys(output_path: Path) -> Set[Tuple[str, str]]:
    """Return set of (prompt, model_id) already completed in the output."""
    done: Set[Tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec.get("prompt", ""), rec.get("model_id", "")))
            except json.JSONDecodeError:
                continue
    return done


def _process_single(
    api_key: str,
    rec: Dict[str, Any],
    old_judge: str,
    new_judge: str,
) -> Dict[str, Any]:
    """Replace old_judge's entry in one record with new_judge's scoring."""
    prompt = rec.get("prompt", "")
    response = rec.get("response", "")

    new_jd = _call_judge(api_key, prompt, response, judge_model=new_judge)

    old_details = rec.get("judge_details", [])
    kept = [jd for jd in old_details if jd.get("judge") != old_judge]

    if new_jd is not None:
        kept.append(new_jd)

    new_raw_score = _recompute_raw_score(kept)

    result = dict(rec)
    result["judge_details"] = kept
    result["raw_score"] = round(new_raw_score, 4)
    result["reward_logit"] = round(_logit_transform(new_raw_score), 4)
    result["ts"] = time.time()
    return result


def cmd_run(args: argparse.Namespace) -> None:
    """Run the selective re-judge on all records."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv(_ROOT / ".env")
            api_key = os.getenv("OPENROUTER_API_KEY")
        except Exception:
            pass
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found")
        sys.exit(1)

    rewards_path = Path(args.input) if args.input else REWARDS_PATH
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    records: List[Dict[str, Any]] = []
    with open(rewards_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ok"):
                records.append(rec)

    logger.info("Loaded %d records from %s", len(records), rewards_path)

    already_done = _load_completed_keys(output_path)
    remaining = [
        r for r in records
        if (r.get("prompt", ""), r.get("model_id", "")) not in already_done
    ]
    logger.info(
        "Remaining after resume: %d (skipping %d already completed)",
        len(remaining), len(records) - len(remaining),
    )

    if not remaining:
        logger.info("All records already processed.")
        return

    old_judge = args.old_judge
    new_judge = args.new_judge
    logger.info("Swapping judge: %s -> %s", old_judge, new_judge)

    lock = threading.Lock()
    completed = 0
    failed = 0

    with open(output_path, "a") as outfile:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _process_single, api_key, rec, old_judge, new_judge,
                ): rec
                for rec in remaining
            }
            with tqdm(total=len(remaining), desc="Swap judge") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    n_judges = len(result.get("judge_details", []))
                    with lock:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
                        completed += 1
                        if n_judges < 3:
                            failed += 1
                    pbar.update(1)

    logger.info(
        "Done. Processed %d records. Incomplete panels: %d",
        completed, failed,
    )
    if failed:
        logger.warning(
            "%d records have <3 judges (new judge failed). "
            "Re-run to retry those.",
            failed,
        )


def cmd_finalize(args: argparse.Namespace) -> None:
    """Replace rewards.jsonl with the new-panel version."""
    output_path = Path(args.output) if args.output else OUTPUT_PATH
    rewards_path = Path(args.input) if args.input else REWARDS_PATH

    if not output_path.exists():
        logger.error("New-panel file not found at %s. Run 'run' first.", output_path)
        sys.exit(1)

    n_new = 0
    with open(output_path) as f:
        for _ in f:
            n_new += 1

    n_orig = 0
    with open(rewards_path) as f:
        for _ in f:
            n_orig += 1

    if n_new != n_orig:
        logger.error(
            "Record count mismatch: original=%d, new=%d. "
            "Ensure the 'run' step completed fully before finalizing.",
            n_orig, n_new,
        )
        sys.exit(1)

    backup_path = rewards_path.with_suffix(".jsonl.pre_swap.bak")
    if backup_path.exists():
        logger.error("Backup already exists at %s. Remove it first.", backup_path)
        sys.exit(1)

    rewards_path.rename(backup_path)
    output_path.rename(rewards_path)
    logger.info("Finalized: %s -> %s (backup at %s)", output_path, rewards_path, backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run selective re-judge.")
    p_run.add_argument(
        "--input", type=str, default=None,
        help=f"Input rewards JSONL (default: {REWARDS_PATH}).",
    )
    p_run.add_argument(
        "--output", type=str, default=None,
        help=f"Output JSONL (default: {OUTPUT_PATH}).",
    )
    p_run.add_argument(
        "--old-judge", type=str, default=OLD_JUDGE,
        help=f"Judge to replace (default: {OLD_JUDGE}).",
    )
    p_run.add_argument(
        "--new-judge", type=str, default=NEW_JUDGE,
        help=f"Replacement judge (default: {NEW_JUDGE}).",
    )
    p_run.add_argument(
        "--workers", type=int, default=15,
        help="Max parallel workers (default: 15).",
    )
    p_run.set_defaults(func=cmd_run)

    p_fin = sub.add_parser(
        "finalize",
        help="Replace rewards.jsonl with the new-panel version.",
    )
    p_fin.add_argument("--input", type=str, default=None)
    p_fin.add_argument("--output", type=str, default=None)
    p_fin.set_defaults(func=cmd_finalize)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
