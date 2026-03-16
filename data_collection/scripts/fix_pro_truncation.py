#!/usr/bin/env python3
"""
Fix truncated Gemini-2.5-Pro responses in the canonical rewards dataset.

The original data-collection pipeline used ``max_tokens=4000`` for all models.
Gemini-2.5-Pro produces verbose chain-of-thought on math/STEM prompts that
can exceed this limit, causing ~38 % of its responses to be truncated
mid-sentence.  Truncated responses receive artificially low judge scores,
making Pro appear Pareto-dominated by Mistral and collapsing the K=3
cost-quality staircase.

This script:

1. **detect**  — Reads ``rewards.jsonl`` and identifies Pro responses that
   end mid-sentence (no terminal punctuation).
2. **fix**     — Re-generates those responses via OpenRouter with a higher
   ``max_tokens`` (default 16 000) and re-judges them through the canonical
   PoLL panel (DeepSeek-R1 + Qwen-72B + Claude-3.5-Haiku).
3. **merge**   — Replaces the old truncated records in ``rewards.jsonl``
   with the fixed ones.

Usage::

    # Step 1+2: detect & fix (writes pro_truncation_fix.jsonl)
    python data_collection/scripts/fix_pro_truncation.py fix

    # Step 3: merge back into canonical rewards
    python data_collection/scripts/fix_pro_truncation.py merge

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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "data_collection" / "scripts"))

from pareto_bandit.config import REWARDS_PATH  # noqa: E402
from rejudge_cot import CoTRewardGenerator  # noqa: E402

logger = logging.getLogger(__name__)

PRO_MODEL_ID = "google/gemini-2.5-pro"
DEFAULT_MAX_TOKENS = 16_000
DEFAULT_TIMEOUT = 600.0

FIX_OUTPUT_PATH = (
    REWARDS_PATH.parent / "pro_truncation_fix.jsonl"
)

_TERMINAL_SUFFIXES = (
    ".", "!", "?", "```", "\n", ")", "]",
    "**", "*", "|", "`", "'",
)


def is_truncated(response: str) -> bool:
    """Return True if *response* appears to have been cut off mid-sentence.

    A response is considered truncated if its stripped text does not end
    with any common sentence-terminal, block-closing, or markdown-formatting
    character.  Endings like ``**`` (bold), ``*`` (italic), ``|`` (table),
    and backtick (inline code) are treated as complete because empirical
    reward distributions confirm they match non-truncated baselines.
    """
    if not response or not response.strip():
        return True
    return not response.rstrip().endswith(_TERMINAL_SUFFIXES)


def detect_truncated(
    rewards_path: Path = REWARDS_PATH,
) -> Tuple[List[Dict[str, Any]], int]:
    """Read canonical rewards and return truncated Pro records.

    Returns:
        Tuple of (list of truncated record dicts, total Pro record count).
    """
    truncated: List[Dict[str, Any]] = []
    total_pro = 0

    with open(rewards_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("model_id") != PRO_MODEL_ID:
                continue
            total_pro += 1
            if not rec.get("ok"):
                truncated.append(rec)
                continue
            if is_truncated(rec.get("response", "")):
                truncated.append(rec)

    return truncated, total_pro


class ProFixGenerator(CoTRewardGenerator):
    """CoTRewardGenerator subclass with higher max_tokens for Pro.

    Overrides ``get_model_response`` to use a configurable
    ``response_max_tokens`` (default 16 000) and longer timeout, while
    keeping the rest of the pipeline (judge panel, retry logic, etc.)
    identical.
    """

    def __init__(
        self,
        response_max_tokens: int = DEFAULT_MAX_TOKENS,
        response_timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.response_max_tokens = response_max_tokens
        self.response_timeout = response_timeout

    def get_model_response(self, model_id: str, prompt: str) -> str | None:
        """Generate a model response with the higher token limit."""
        import requests

        if (model_id, prompt) in self.response_cache:
            return self.response_cache[(model_id, prompt)]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": self.response_max_tokens,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.response_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as exc:
            logger.warning("Response generation failed for %s: %s", model_id, exc)
            return None


def _load_completed(output_path: Path) -> Set[str]:
    """Return set of prompt texts already completed in the fix file."""
    done: Set[str] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok") and rec.get("prompt"):
                    done.add(rec["prompt"])
            except json.JSONDecodeError:
                continue
    return done


def cmd_fix(args: argparse.Namespace) -> None:
    """Detect truncated Pro responses and regenerate + rejudge them."""
    truncated_recs, total_pro = detect_truncated()
    n_trunc = len(truncated_recs)
    logger.info(
        "Detected %d / %d truncated Pro responses (%.1f%%)",
        n_trunc, total_pro, 100 * n_trunc / max(total_pro, 1),
    )

    if n_trunc == 0:
        logger.info("Nothing to fix.")
        return

    already_done = _load_completed(FIX_OUTPUT_PATH)
    prompts_to_fix = [
        rec["prompt"]
        for rec in truncated_recs
        if rec.get("prompt") and rec["prompt"] not in already_done
    ]
    logger.info(
        "Remaining after resume: %d (skipping %d already completed)",
        len(prompts_to_fix), n_trunc - len(prompts_to_fix),
    )

    if not prompts_to_fix:
        logger.info("All truncated responses already fixed.")
        return

    gen = ProFixGenerator(
        response_max_tokens=args.max_tokens,
        response_timeout=args.timeout,
        max_workers=args.workers,
    )

    still_truncated = 0
    completed = 0

    with open(FIX_OUTPUT_PATH, "a") as outfile:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(gen.process_task, (prompt, PRO_MODEL_ID)): prompt
                for prompt in prompts_to_fix
            }
            with tqdm(total=len(prompts_to_fix), desc="Fixing Pro") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    resp_text = result.get("response", "")
                    if is_truncated(resp_text):
                        still_truncated += 1

                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()
                    completed += 1
                    pbar.update(1)

    logger.info(
        "Done. Fixed %d responses. Still truncated after retry: %d",
        completed, still_truncated,
    )
    if still_truncated:
        logger.warning(
            "%d responses remain truncated even with max_tokens=%d. "
            "These may be due to safety filters or model-side limits.",
            still_truncated, args.max_tokens,
        )


def cmd_merge(args: argparse.Namespace) -> None:
    """Replace truncated Pro records in rewards.jsonl with fixed ones."""
    if not FIX_OUTPUT_PATH.exists():
        logger.error("Fix file not found at %s. Run 'fix' first.", FIX_OUTPUT_PATH)
        sys.exit(1)

    # Load fixed records keyed by prompt
    fixed: Dict[str, Dict[str, Any]] = {}
    with open(FIX_OUTPUT_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("ok") and rec.get("prompt"):
                fixed[rec["prompt"]] = rec

    logger.info("Loaded %d fixed Pro records from %s", len(fixed), FIX_OUTPUT_PATH)

    # Read canonical rewards and replace truncated Pro records
    backup_path = REWARDS_PATH.with_suffix(".jsonl.bak")
    REWARDS_PATH.rename(backup_path)
    logger.info("Backed up original to %s", backup_path)

    replaced = 0
    total = 0
    with open(backup_path) as infile, open(REWARDS_PATH, "w") as outfile:
        for line in infile:
            rec = json.loads(line)
            total += 1

            if (
                rec.get("model_id") == PRO_MODEL_ID
                and rec.get("prompt") in fixed
            ):
                outfile.write(json.dumps(fixed[rec["prompt"]]) + "\n")
                replaced += 1
            else:
                outfile.write(line)

    logger.info(
        "Merged %d fixed records into %s (%d total records)",
        replaced, REWARDS_PATH, total,
    )
    logger.info(
        "Original backed up at %s. Delete it once verified.",
        backup_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fix ---------------------------------------------------------------
    p_fix = sub.add_parser(
        "fix",
        help="Detect truncated Pro responses and regenerate + rejudge.",
    )
    p_fix.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens for Pro response generation (default: {DEFAULT_MAX_TOKENS}).",
    )
    p_fix.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds for response generation (default: {DEFAULT_TIMEOUT}).",
    )
    p_fix.add_argument(
        "--workers", type=int, default=10,
        help="Max parallel workers (default: 10).",
    )
    p_fix.set_defaults(func=cmd_fix)

    # -- merge -------------------------------------------------------------
    p_merge = sub.add_parser(
        "merge",
        help="Replace truncated Pro records in rewards.jsonl with fixed ones.",
    )
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
