"""Replace GLM-5 judge scores with Claude-3.5-Haiku in k4_rewards_v3.jsonl.

This script performs two passes:

1. **Swap pass** — For every record that contains a ``z-ai/glm-5`` judge
   entry, strip that entry, query ``anthropic/claude-3.5-haiku`` with the
   stored (prompt, response), and insert the new Haiku score.  The panel
   aggregate (``raw_score``, ``reward_logit``) is recomputed.

2. **Fill pass** — Records still missing from the file (relative to the
   source v2 file) are fully judged with the canonical 3-judge panel
   (DeepSeek-R1, Qwen-2.5-72B, Claude-3.5-Haiku).

Resume-safe: records that already contain a Haiku entry (and no GLM-5
entry) are left untouched.  Partial progress from the swap pass is
checkpointed every ``--checkpoint-every`` records.

Usage
-----
::

    python data_collection/scripts/swap_glm5_to_haiku.py \
        --rewards-file data_collection/rewards/k4_rewards_v3.jsonl \
        --source-file  data_collection/rewards/archive/v2_binary_rubric/k4_rewards_v2_clean.jsonl \
        --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

OLD_JUDGE = "z-ai/glm-5"
NEW_JUDGE = "anthropic/claude-3.5-haiku"

CANONICAL_PANEL: List[str] = [
    "deepseek/deepseek-r1",
    "qwen/qwen-2.5-72b-instruct",
    NEW_JUDGE,
]

_W_REASONING: float = 0.40
_W_INSTRUCTION: float = 0.30
_W_COMMUNICATION: float = 0.30


def _build_system_prompt() -> str:
    """Return the v3 judge system prompt (identical to rejudge_cot.py)."""
    return (
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


# ── API helpers ──────────────────────────────────────────────────────────

import re
import requests

_MAX_RETRIES: int = 3
_RETRY_BACKOFF_BASE: float = 2.0
_TIMEOUT: float = 90.0


def _parse_score(content: str, heading: str, *, default: float = 0.5) -> float:
    pattern = r"##\s*" + heading + r"\s*[:\-]?\s*(\d+\.?\d*)"
    m = re.search(pattern, content, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))
    return default


def _query_haiku(
    api_key: str,
    prompt: str,
    response: str,
) -> Dict[str, Any] | None:
    """Call Claude-3.5-Haiku as a judge for one (prompt, response) pair.

    Returns a judge-detail dict on success, ``None`` on unrecoverable
    failure.
    """
    system_prompt = _build_system_prompt()
    user_content = f"PROMPT: {prompt}\n\nRESPONSE: {response}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
    }
    payload = {
        "model": NEW_JUDGE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 4000,
    }

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            if raw_content is None:
                raise ValueError("API returned null content")
            content = raw_content.strip()

            rq = _parse_score(content, r"Reasoning\s+Quality")
            if_ = _parse_score(content, r"Instruction\s+Following")
            cq = _parse_score(content, r"Communication\s+Quality")
            reward = rq * _W_REASONING + if_ * _W_INSTRUCTION + cq * _W_COMMUNICATION

            reasoning = content
            rm = re.search(
                r"##\s*Reasoning\s*(.*?)(\n##|$)",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if rm:
                reasoning = rm.group(1).strip()

            return {
                "judge": NEW_JUDGE,
                "reasoning_quality": round(rq, 4),
                "instruction_following": round(if_, 4),
                "communication_quality": round(cq, 4),
                "reward": round(reward, 4),
                "reasoning": reasoning,
                "logic": round(rq, 4),
                "constraint": round(if_, 4),
                "utility": round(cq, 4),
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
            if status not in (429, 502, 503, 504):
                logger.warning("Haiku non-retryable HTTP %d", status)
                return None
        except Exception as e:
            logger.warning("Haiku unexpected error: %s", e)
            return None

        backoff = _RETRY_BACKOFF_BASE ** attempt
        logger.debug(
            "Haiku attempt %d/%d failed (%s), retrying in %.1fs",
            attempt, _MAX_RETRIES, last_exc, backoff,
        )
        time.sleep(backoff)

    logger.warning("Haiku failed after %d attempts: %s", _MAX_RETRIES, last_exc)
    return None


# ── Aggregate helpers ────────────────────────────────────────────────────

def _logit(score: float) -> float:
    if not np.isfinite(score):
        return float("nan")
    score = np.clip(score, 0.01, 0.99)
    return float(np.log(score / (1 - score)))


def _recompute_aggregate(judge_details: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Return (raw_score, reward_logit) from the mean of per-judge rewards."""
    rewards = [jd["reward"] for jd in judge_details if np.isfinite(jd.get("reward", float("nan")))]
    if not rewards:
        return float("nan"), float("nan")
    raw = float(np.mean(rewards))
    return raw, _logit(raw)


# ── Swap pass ────────────────────────────────────────────────────────────

def _swap_single(
    rec: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any] | None:
    """Remove GLM-5 entry, query Haiku, return updated record.

    Returns ``None`` if the Haiku call fails (record left untouched for
    a later retry).
    """
    haiku_result = _query_haiku(api_key, rec["prompt"], rec["response"])
    if haiku_result is None:
        return None

    new_details = [jd for jd in rec["judge_details"] if jd["judge"] != OLD_JUDGE]
    new_details.append(haiku_result)

    raw_score, reward_logit = _recompute_aggregate(new_details)
    rec["judge_details"] = new_details
    rec["raw_score"] = round(raw_score, 6)
    rec["reward_logit"] = round(reward_logit, 6)
    rec["ts"] = time.time()
    return rec


# ── Full-panel judging (for missing records) ─────────────────────────────

def _judge_full_panel(
    prompt: str,
    response: str,
    model_id: str,
    api_key: str,
) -> Dict[str, Any]:
    """Judge a (prompt, response) with the full 3-judge panel.

    Imports and delegates to the ``CoTRewardGenerator`` from
    ``rejudge_cot.py`` for the DeepSeek-R1 and Qwen judges.  Uses
    ``_query_haiku`` for the Haiku judge to keep retry logic consistent.
    """
    from rejudge_cot import CoTRewardGenerator

    gen = CoTRewardGenerator(api_key=api_key, max_workers=3)
    gen.judge_panel = list(CANONICAL_PANEL)

    final_score, judge_details = gen.judge_with_panel_cot(prompt, response, model_id)
    reward_logit = gen.logit_transform(final_score)

    return {
        "model_id": model_id,
        "prompt": prompt,
        "response": response,
        "ok": True,
        "teacher_used": False,
        "judge_details": judge_details,
        "reward_logit": round(reward_logit, 6) if np.isfinite(reward_logit) else None,
        "raw_score": round(final_score, 6) if np.isfinite(final_score) else None,
        "ts": time.time(),
    }


# ── Main driver ──────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        key = os.getenv("OPENROUTER_API_KEY")
    except Exception:
        pass
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace GLM-5 judge scores with Claude-3.5-Haiku in an "
            "existing rewards JSONL, then fill any missing records from "
            "a source file."
        ),
    )
    parser.add_argument(
        "--rewards-file",
        type=str,
        required=True,
        help="Path to the v3 rewards JSONL to patch in-place.",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default=None,
        help=(
            "Path to the original v2 rewards JSONL (with stored responses) "
            "for filling missing records.  If omitted, only the swap pass "
            "runs."
        ),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write a checkpoint after this many records are swapped.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = _get_api_key()
    rewards_path = Path(args.rewards_file)

    # ── Load existing records ────────────────────────────────────────────
    records: List[Dict[str, Any]] = []
    with open(rewards_path) as f:
        for line in f:
            records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), rewards_path)

    # Partition into needs-swap vs already-done.
    needs_swap: List[int] = []
    already_ok: List[int] = []
    for i, rec in enumerate(records):
        judges = {jd["judge"] for jd in rec.get("judge_details", [])}
        if OLD_JUDGE in judges:
            needs_swap.append(i)
        else:
            already_ok.append(i)

    logger.info(
        "%d records need GLM-5→Haiku swap, %d already OK",
        len(needs_swap),
        len(already_ok),
    )

    # ── Pass 1: Swap GLM-5 → Haiku ──────────────────────────────────────
    if needs_swap:
        lock = threading.Lock()
        swapped = 0
        failed_indices: List[int] = []

        def _do_swap(idx: int) -> Tuple[int, bool]:
            result = _swap_single(records[idx], api_key)
            if result is None:
                return idx, False
            with lock:
                records[idx] = result
            return idx, True

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_do_swap, i): i for i in needs_swap}
            with tqdm(total=len(needs_swap), desc="Swapping GLM-5→Haiku") as pbar:
                for fut in as_completed(futures):
                    idx, ok = fut.result()
                    if ok:
                        swapped += 1
                    else:
                        failed_indices.append(idx)
                    pbar.update(1)

                    if swapped > 0 and swapped % args.checkpoint_every == 0:
                        _write_checkpoint(rewards_path, records)
                        logger.info("Checkpoint at %d swapped", swapped)

        logger.info(
            "Swap pass complete: %d swapped, %d failed",
            swapped,
            len(failed_indices),
        )
        _write_checkpoint(rewards_path, records)

    # ── Pass 2: Fill missing records from source ─────────────────────────
    if args.source_file:
        source_path = Path(args.source_file)
        existing_keys = {
            (rec["prompt"], rec["model_id"]) for rec in records
        }

        source_tasks: List[Tuple[str, str, str]] = []
        with open(source_path) as f:
            for line in f:
                src = json.loads(line)
                if not src.get("ok") or not src.get("response"):
                    continue
                key = (src["prompt"], src["model_id"])
                if key not in existing_keys:
                    source_tasks.append((src["prompt"], src["response"], src["model_id"]))

        logger.info(
            "%d records missing from rewards file, queuing for full-panel judging",
            len(source_tasks),
        )

        if source_tasks:
            lock = threading.Lock()

            def _do_fill(task: Tuple[str, str, str]) -> Dict[str, Any]:
                return _judge_full_panel(task[0], task[1], task[2], api_key)

            new_records: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_do_fill, t): t for t in source_tasks}
                with tqdm(total=len(source_tasks), desc="Filling missing records") as pbar:
                    for fut in as_completed(futures):
                        res = fut.result()
                        with lock:
                            new_records.append(res)
                        pbar.update(1)

            records.extend(new_records)
            logger.info("Added %d new records", len(new_records))
            _write_checkpoint(rewards_path, records)

    # ── Summary ──────────────────────────────────────────────────────────
    judges_counter: Dict[str, int] = {}
    for rec in records:
        for jd in rec.get("judge_details", []):
            j = jd["judge"]
            judges_counter[j] = judges_counter.get(j, 0) + 1

    logger.info("Final record count: %d", len(records))
    logger.info("Judge breakdown:")
    for j, n in sorted(judges_counter.items()):
        logger.info("  %s: %d", j, n)


def _write_checkpoint(path: Path, records: List[Dict[str, Any]]) -> None:
    """Atomically overwrite the rewards file via a temp-file swap."""
    fd, tmp = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem,
    )
    try:
        with os.fdopen(fd, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        shutil.move(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


if __name__ == "__main__":
    main()
