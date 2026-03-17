#!/usr/bin/env python3
"""Collect actual token counts for Gemini-2.5-Flash via OpenRouter.

The canonical Flash rewards (``flash_canonical/gemini_flash_v3.jsonl``)
were collected without token tracking, so the K=4 splits use a constant
cost fallback.  This script re-queries Flash for each prompt to obtain
exact ``prompt_tokens`` and ``completion_tokens`` from the OpenRouter
``usage`` response field.

The resulting token counts are used by ``merge_flash_into_splits.py``
to replace the synthetic constant cost with actual per-request costs.

Usage
-----
    # Collect token counts for all val+test prompts
    python data_collection/scripts/collect_flash_token_counts.py

    # Quick test (5 prompts)
    python data_collection/scripts/collect_flash_token_counts.py --limit 5

    # Resume interrupted run
    python data_collection/scripts/collect_flash_token_counts.py --resume

    # Print summary of collected data
    python data_collection/scripts/collect_flash_token_counts.py --summary-only

Requirements
------------
    export OPENROUTER_API_KEY=...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import VAL_DATA_PATH, HOLDOUT_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FLASH_ID = "google/gemini-2.5-flash"
FLASH_INPUT_COST_PER_M = 0.3
FLASH_OUTPUT_COST_PER_M = 2.5

OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "flash_canonical"
OUTPUT_FILE = OUTPUT_DIR / "flash_token_counts.jsonl"

API_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
MAX_TOKENS = 16_000
TIMEOUT = 300.0


def _get_api_key() -> str:
    """Resolve OpenRouter API key from environment or .env file."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.getenv("OPENROUTER_API_KEY")
    except ImportError:
        pass
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. "
            "Set it via environment variable or .env file."
        )
    return key


def load_prompts(splits: Tuple[str, ...] = ("val", "test")) -> List[str]:
    """Load unique prompts from canonical K=3 splits."""
    split_paths = {"val": VAL_DATA_PATH, "test": HOLDOUT_DATA_PATH}
    seen: Set[str] = set()
    prompts: List[str] = []
    for name in splits:
        with open(split_paths[name]) as f:
            for line in f:
                p = json.loads(line)["prompt"]
                if p not in seen:
                    prompts.append(p)
                    seen.add(p)
    logger.info("Loaded %d unique prompts from %s", len(prompts), splits)
    return prompts


def load_completed() -> Set[str]:
    """Load prompts already processed in the output file."""
    completed: Set[str] = set()
    if not OUTPUT_FILE.exists():
        return completed
    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    completed.add(rec["prompt"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def query_flash(
    prompt: str, api_key: str
) -> Dict[str, Any]:
    """Query OpenRouter for a Flash response and extract token usage.

    Returns a dict with prompt, ok, input_tokens, output_tokens, cost_usd,
    and diagnostic fields.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
    }
    payload = {
        "model": FLASH_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

            if input_tokens is None or output_tokens is None:
                return {
                    "prompt": prompt,
                    "ok": False,
                    "error": "missing usage field in response",
                    "ts": time.time(),
                }

            cost_usd = (
                input_tokens * FLASH_INPUT_COST_PER_M
                + output_tokens * FLASH_OUTPUT_COST_PER_M
            ) / 1_000_000

            return {
                "prompt": prompt,
                "ok": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost_usd, 10),
                "ts": time.time(),
            }

        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return {
                "prompt": prompt,
                "ok": False,
                "error": f"HTTP {status}",
                "ts": time.time(),
            }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            time.sleep(2 ** attempt)
            continue
        except Exception as e:
            return {
                "prompt": prompt,
                "ok": False,
                "error": str(e),
                "ts": time.time(),
            }

    return {
        "prompt": prompt,
        "ok": False,
        "error": f"failed after {MAX_RETRIES} attempts: {last_exc}",
        "ts": time.time(),
    }


def print_summary() -> None:
    """Print summary statistics for collected token counts."""
    if not OUTPUT_FILE.exists():
        print("No data collected yet.")
        return

    input_tokens: List[int] = []
    output_tokens: List[int] = []
    costs: List[float] = []
    n_failed = 0

    with open(OUTPUT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    input_tokens.append(rec["input_tokens"])
                    output_tokens.append(rec["output_tokens"])
                    costs.append(rec["cost_usd"])
                else:
                    n_failed += 1
            except (json.JSONDecodeError, KeyError):
                continue

    if not costs:
        print("No successful records.")
        return

    it = np.array(input_tokens)
    ot = np.array(output_tokens)
    c = np.array(costs)

    print("\n" + "=" * 70)
    print("FLASH TOKEN COUNT COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Successful: {len(c)}  |  Failed: {n_failed}")
    print()
    print(f"  Input tokens:  mean={it.mean():.0f}  std={it.std():.0f}  "
          f"min={it.min()}  max={it.max()}")
    print(f"  Output tokens: mean={ot.mean():.0f}  std={ot.std():.0f}  "
          f"min={ot.min()}  max={ot.max()}")
    print()
    print(f"  Cost per request:")
    print(f"    mean=${c.mean():.8f}  std=${c.std():.8f}  CV={c.std()/c.mean():.2f}")
    print(f"    min=${c.min():.8f}  p50=${np.median(c):.8f}  max=${c.max():.8f}")
    print(f"    (compare: old constant = $0.00070000)")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of prompts (for testing).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel workers (default: 10).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip collection, just print summary.",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return

    api_key = _get_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    completed = load_completed() if args.resume else set()
    if completed:
        logger.info("Resuming: %d prompts already completed", len(completed))

    pending = [p for p in prompts if p not in completed]
    logger.info("Pending: %d prompts to collect", len(pending))

    if not pending:
        logger.info("Nothing to do — all prompts already collected.")
        print_summary()
        return

    n_done = 0
    n_ok = 0
    t0 = time.time()

    with open(OUTPUT_FILE, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(query_flash, p, api_key): p
                for p in pending
            }
            with tqdm(total=len(pending), desc="Flash token counts") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    n_done += 1

                    if result.get("ok"):
                        n_ok += 1

                    pbar.update(1)

                    if n_done % 200 == 0:
                        elapsed = time.time() - t0
                        rate = n_done / elapsed * 60
                        logger.info(
                            "[%d/%d] %.0f req/min | ok=%d | "
                            "elapsed=%.1f min",
                            n_done, len(pending), rate, n_ok,
                            elapsed / 60,
                        )

    elapsed = time.time() - t0
    logger.info(
        "Collection complete: %d/%d ok in %.1f min (%.1f req/min)",
        n_ok, n_done, elapsed / 60, n_done / max(elapsed, 1) * 60,
    )
    print_summary()


if __name__ == "__main__":
    main()
