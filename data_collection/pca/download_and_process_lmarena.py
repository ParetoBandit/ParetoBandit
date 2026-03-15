#!/usr/bin/env python3
"""
Download LMArena 140k Human Preference Data and Create De-duplicated Battles Dataset.

Downloads ``lmarena-ai/arena-human-preference-140k`` from HuggingFace, filters to
English ``eval_order=1`` (independent evaluations only), de-duplicates by prompt
text (keeping all battles per unique prompt), normalizes model names, and exports
a clean JSONL file.

Output format (one JSON object per line)::

    {
        "prompt_id": "<sha256 of normalized prompt text>",
        "prompt": "...",
        "model_a": "openai/gpt-4o-2024-08-06",
        "model_b": "anthropic/claude-3.5-sonnet",
        "winner": "model_a",
        "category": "coding",
        "timestamp": "2025-05-12T14:23:00"
    }

Usage::

    python scripts/download_and_process_lmarena.py
    python scripts/download_and_process_lmarena.py --output data/lmarena_battles.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text for de-duplication.

    Strips leading/trailing whitespace, collapses internal whitespace runs,
    and applies Unicode NFKC normalization so that visually identical prompts
    hash identically.

    Args:
        text: Raw prompt string.

    Returns:
        Canonicalized prompt string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text.strip())
    return text


def prompt_id_from_text(text: str) -> str:
    """Deterministic SHA-256 prompt ID from normalized text.

    Args:
        text: Already-normalized prompt string.

    Returns:
        Hex-encoded SHA-256 hash (64 characters).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model-name normalisation helpers
# ---------------------------------------------------------------------------

# LMArena uses short names like "gpt-4o-2024-08-06".  We keep them close to
# their canonical form with a ``provider/`` prefix for consistency with the
# rest of the banditGPT codebase.

_MODEL_CANON: Dict[str, str] = {
    # OpenAI
    "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
    "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09": "openai/gpt-4-turbo-2024-04-09",
    "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
    "gpt-4-0125-preview": "openai/gpt-4-0125-preview",
    "gpt-4.1-2025-04-14": "openai/gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14": "openai/gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14": "openai/gpt-4.1-nano-2025-04-14",
    "o1-2024-12-17": "openai/o1-2024-12-17",
    "o1-mini-2024-09-12": "openai/o1-mini-2024-09-12",
    "o1-preview-2024-09-12": "openai/o1-preview-2024-09-12",
    "o3-mini-2025-01-31": "openai/o3-mini-2025-01-31",
    "o4-mini-2025-04-16": "openai/o4-mini-2025-04-16",
    "chatgpt-4o-latest-2025-03-26": "openai/chatgpt-4o-latest-2025-03-26",
    # Anthropic
    "claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620": "anthropic/claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku-20241022",
    "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
    "claude-3-haiku-20240307": "anthropic/claude-3-haiku-20240307",
    "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet-20240229",
    "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514",
    # Google
    "gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001": "google/gemini-2.0-flash-lite-001",
    "gemini-1.5-pro-002": "google/gemini-1.5-pro-002",
    "gemini-1.5-flash-002": "google/gemini-1.5-flash-002",
    "gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17": "google/gemini-2.5-flash-preview-04-17",
    "gemma-2-27b-it": "google/gemma-2-27b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    # Meta
    "llama-3.1-405b-instruct-fp8": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "llama-4-maverick-17b-128e-instruct": "meta-llama/llama-4-maverick",
    "llama-4-scout-17b-16e-instruct": "meta-llama/llama-4-scout",
    # Mistral
    "mistral-large-2411": "mistralai/mistral-large-2411",
    "mistral-small-2503": "mistralai/mistral-small-2503",
    # DeepSeek
    "deepseek-v3-0324": "deepseek/deepseek-chat-v3-0324",
    "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "deepseek-r1": "deepseek/deepseek-r1",
    # Qwen
    "qwen2.5-72b-instruct": "qwen/qwen2.5-72b-instruct",
    "qwen2.5-plus-1220": "qwen/qwen2.5-plus-1220",
    "qwen3-235b-a22b": "qwen/qwen3-235b-a22b",
    "qwq-32b": "qwen/qwq-32b",
    # Amazon
    "nova-pro-v1": "amazon/nova-pro-v1",
    "nova-lite-v1": "amazon/nova-lite-v1",
    # Cohere
    "command-a-03-2025": "cohere/command-a-03-2025",
    "command-r-plus-08-2024": "cohere/command-r-plus-08-2024",
    # xAI
    "grok-2-1212": "xai/grok-2-1212",
    "grok-3-mini-beta": "xai/grok-3-mini-beta",
    # Yi / 01-ai
    "yi-lightning": "01-ai/yi-lightning",
    # Reka
    "reka-flash-20240904": "reka/reka-flash-20240904",
    # Microsoft
    "phi-4": "microsoft/phi-4",
}


def normalize_model_name(raw_name: str) -> str:
    """Map LMArena's short model names to ``provider/model`` form.

    Falls through to ``raw_name`` unchanged when no canonical mapping exists.

    Args:
        raw_name: Model identifier as it appears in LMArena.

    Returns:
        Canonicalized ``provider/model-name`` string.
    """
    stripped = raw_name.strip()
    if stripped in _MODEL_CANON:
        return _MODEL_CANON[stripped]
    return stripped


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------


def extract_conversation_prompt(conversation: Any) -> Optional[str]:
    """Pull the first user turn out of a conversation field."""
    if isinstance(conversation, list):
        for turn in conversation:
            if isinstance(turn, dict) and turn.get("role") == "user":
                content = turn.get("content", "")
                if isinstance(content, list):
                    # In newer LMArena datasets, content is a list of blocks
                    text_blocks = [
                        block.get("text", "") 
                        for block in content 
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    content = "".join(text_blocks)
                
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one HuggingFace row into the output schema.

    Returns *None* for rows that should be skipped (non-English, multi-turn,
    invalid winner, etc.).

    Args:
        row: A single record from the HuggingFace dataset.

    Returns:
        Processed battle dict or *None*.
    """
    # Only independent evaluations (first in each pair)
    # Only independent evaluations (first in each pair)
    if row.get("evaluation_order") != 1:
        return None

    # English only
    lang = (row.get("language") or "").strip().lower()
    if lang not in ("en", "english"):
        return None

    # Extract prompt from conversation_a
    prompt_raw = extract_conversation_prompt(row.get("conversation_a"))
    if prompt_raw is None:
        return None

    # Basic quality filters
    if len(prompt_raw) < 10 or len(prompt_raw) > 20_000:
        return None

    norm_text = normalize_prompt_text(prompt_raw)
    pid = prompt_id_from_text(norm_text)

    # Winner
    winner = (row.get("winner") or "").strip()
    if winner == "model_a":
        winner_field = "model_a"
    elif winner == "model_b":
        winner_field = "model_b"
    elif winner in ("tie", "tie (bothbad)"):
        winner_field = "tie"
    else:
        return None

    model_a = normalize_model_name(row.get("model_a", ""))
    model_b = normalize_model_name(row.get("model_b", ""))
    if not model_a or not model_b:
        return None

    category_raw = row.get("category_tag")
    if isinstance(category_raw, dict):
        category = "unknown" # Could try to parse tags if needed
    else:
        category = str(category_raw or "unknown").strip().lower()
    
    timestamp = str(row.get("timestamp", ""))

    return {
        "prompt_id": pid,
        "prompt": norm_text,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner_field,
        "category": category,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def download_and_process(args: argparse.Namespace) -> Path:
    """Download, filter, de-dupe, and export LMArena battles.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the output JSONL file.
    """
    from datasets import load_dataset

    output_path = Path(args.output)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    print("=" * 80)
    print("DOWNLOAD & PROCESS LMARENA 140K BATTLES")
    print("=" * 80)
    print(f"  Output       : {output_path}")
    print(f"  HF token     : {'found' if hf_token else 'not found (public access)'}")

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    print("\n[1/4] Downloading lmarena-ai/arena-human-preference-140k ...")
    ds = load_dataset(
        "lmarena-ai/arena-human-preference-140k",
        split="train",
        token=hf_token,
        streaming=False,
    )
    print(f"  Downloaded {len(ds):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Filter & normalise
    # ------------------------------------------------------------------
    print("\n[2/4] Filtering (English, eval_order=1) and normalising ...")
    battles: List[Dict[str, Any]] = []
    skipped = 0
    for row in tqdm(ds, desc="  Processing", unit="row"):
        rec = process_row(row)
        if rec is None:
            skipped += 1
            continue
        battles.append(rec)

    print(f"  Kept   : {len(battles):,} battles")
    print(f"  Skipped: {skipped:,} rows")

    # ------------------------------------------------------------------
    # Step 3: De-duplicate by prompt_id (keep all battles per prompt)
    # ------------------------------------------------------------------
    print("\n[3/4] De-duplicating by prompt text ...")
    prompt_id_to_text: Dict[str, str] = {}
    prompt_id_battles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    duplicate_texts_found = 0

    for b in battles:
        pid = b["prompt_id"]
        if pid not in prompt_id_to_text:
            prompt_id_to_text[pid] = b["prompt"]
        elif prompt_id_to_text[pid] != b["prompt"]:
            # Hash collision (astronomically unlikely with SHA-256)
            duplicate_texts_found += 1
        prompt_id_battles[pid].append(b)

    n_unique_prompts = len(prompt_id_to_text)
    n_total_battles = sum(len(v) for v in prompt_id_battles.values())
    print(f"  Unique prompts : {n_unique_prompts:,}")
    print(f"  Total battles  : {n_total_battles:,}")
    if duplicate_texts_found:
        print(f"  Hash collisions: {duplicate_texts_found} (inspect manually)")

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------
    print(f"\n[4/4] Saving to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for pid in sorted(prompt_id_battles.keys()):
            for battle in prompt_id_battles[pid]:
                fh.write(json.dumps(battle, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    model_counts: Counter = Counter()
    category_counts: Counter = Counter()
    winner_counts: Counter = Counter()
    for b in battles:
        model_counts[b["model_a"]] += 1
        model_counts[b["model_b"]] += 1
        category_counts[b["category"]] += 1
        winner_counts[b["winner"]] += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Unique prompts : {n_unique_prompts:,}")
    print(f"  Total battles  : {n_total_battles:,}")
    print(f"  Unique models  : {len(model_counts):,}")
    print(f"  Winner dist    : {dict(winner_counts)}")

    print(f"\n  Top 15 models (by battle participation):")
    for model, cnt in model_counts.most_common(15):
        print(f"    {model:55s} {cnt:>7,}")

    print(f"\n  Categories:")
    for cat, cnt in category_counts.most_common():
        print(f"    {cat:30s} {cnt:>7,}")

    print(f"\n  Output: {output_path}")
    print("=" * 80)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download and process LMArena 140k human preference data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_output = str(
        PROJECT_ROOT / "src" / "bandit_gpt" / "data" / "offline_dataset"
        / "lmarena_battles_en.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Output JSONL path (default: %(default)s)",
    )
    args = parser.parse_args()
    download_and_process(args)


if __name__ == "__main__":
    main()
