#!/usr/bin/env python3
"""
Score prompt difficulty via attention-entropy of a small probe model.

Uses GPT-2 (124M) as a lightweight probe.  For each prompt the script
computes the mean Shannon entropy of every attention head across all
layers, normalised by the maximum possible entropy (log₂ of the
sequence length).  This yields a score in [0, 1] where:

    0  →  The probe attends to a single token everywhere (trivial input).
    1  →  Uniform attention across the full context (maximally uncertain).

High scores indicate prompts where even a strong model is likely to
"spread its bets", which empirically correlates with inter-model
divergence — exactly the prompts a routing bandit should prioritise.

Usage:
    # Score all canonical reward prompts and print summary statistics:
    python data_collection/scripts/score_prompt_difficulty.py

    # Save per-prompt scores to JSONL:
    python data_collection/scripts/score_prompt_difficulty.py \
        --output data_collection/prompts/prompt_difficulty_scores.jsonl

    # Custom data source:
    python data_collection/scripts/score_prompt_difficulty.py \
        --input data_collection/prompts/my_prompts.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_gpt.config import (
    TRAIN_DATA_PATH_ALL_MODELS,
    VAL_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROBE_MODEL_NAME = "gpt2"
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Core: attention-entropy difficulty scoring
# ---------------------------------------------------------------------------

def compute_attention_entropy(attentions: tuple[Tensor, ...]) -> float:
    """Compute mean normalised Shannon entropy across all heads and layers.

    Parameters
    ----------
    attentions:
        Tuple of tensors, one per layer, each of shape
        ``(batch=1, n_heads, seq_len, seq_len)``.

    Returns
    -------
    float
        Mean entropy normalised by ``log₂(seq_len)``, in [0, 1].
    """
    entropies: list[float] = []
    for layer_attn in attentions:
        # layer_attn: (1, n_heads, seq_len, seq_len)
        attn = layer_attn.squeeze(0)  # (n_heads, seq_len, seq_len)
        # Clamp to avoid log(0)
        attn = attn.clamp(min=1e-12)
        # Shannon entropy per (head, query_position)
        ent = -(attn * attn.log2()).sum(dim=-1)  # (n_heads, seq_len)
        entropies.append(ent.mean().item())

    seq_len = attentions[0].shape[-1]
    max_entropy = np.log2(max(seq_len, 2))
    return float(np.mean(entropies) / max_entropy)


@torch.no_grad()
def score_prompt(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Return a difficulty score in [0, 1] for a single prompt.

    Parameters
    ----------
    prompt:
        Raw prompt text.
    model:
        Probe language model (GPT-2).
    tokenizer:
        Corresponding tokenizer.
    device:
        Torch device to run inference on.

    Returns
    -------
    float
        Normalised attention entropy (higher = harder).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    outputs = model(**inputs, output_attentions=True)
    return compute_attention_entropy(outputs.attentions)


@torch.no_grad()
def score_prompts_batched(
    prompts: Sequence[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """Score a list of prompts with batched inference.

    Parameters
    ----------
    prompts:
        Raw prompt texts.
    model:
        Probe language model.
    tokenizer:
        Corresponding tokenizer.
    device:
        Torch device.
    batch_size:
        Number of prompts per forward pass.

    Returns
    -------
    list[float]
        Per-prompt difficulty scores in [0, 1].
    """
    scores: list[float] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_texts = prompts[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(device)

        outputs = model(**inputs, output_attentions=True)

        # Process each item in the batch individually, masking out padding
        for i in range(len(batch_texts)):
            attn_mask = inputs["attention_mask"][i]  # (seq_len,)
            real_len = attn_mask.sum().item()

            item_entropies: list[float] = []
            for layer_attn in outputs.attentions:
                # layer_attn: (batch, n_heads, seq_len, seq_len)
                attn = layer_attn[i]  # (n_heads, seq_len, seq_len)
                # Restrict to real (non-padding) tokens
                attn = attn[:, :real_len, :real_len]
                attn = attn.clamp(min=1e-12)
                ent = -(attn * attn.log2()).sum(dim=-1)  # (n_heads, real_len)
                item_entropies.append(ent.mean().item())

            max_entropy = np.log2(max(real_len, 2))
            scores.append(float(np.mean(item_entropies) / max_entropy))

        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            logger.info(
                f"  Scored {end}/{len(prompts)} prompts "
                f"({100 * end / len(prompts):.0f}%)"
            )

    return scores


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_reward_prompts() -> list[str]:
    """Load unique prompts from all three canonical reward splits."""
    seen: set[str] = set()
    prompts: list[str] = []
    for gz_path in [
        TRAIN_DATA_PATH_ALL_MODELS,
        VAL_DATA_PATH_ALL_MODELS,
        HOLDOUT_DATA_PATH_ALL_MODELS,
    ]:
        if not gz_path.exists():
            logger.warning(f"  Skipping {gz_path.name} (not found)")
            continue
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    p = entry["prompt"]
                    if p not in seen:
                        seen.add(p)
                        prompts.append(p)
    return prompts


def load_jsonl_prompts(path: Path) -> list[str]:
    """Load prompts from a plain JSONL file (one ``{"prompt": ...}`` per line)."""
    prompts: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[arg-type]
        for line in f:
            entry = json.loads(line)
            text = entry.get("prompt", "")
            if text:
                prompts.append(text)
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score prompt difficulty via probe-model attention entropy."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a JSONL (or .jsonl.gz) of prompts. "
        "If omitted, loads all canonical reward-set prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write per-prompt JSONL scores. "
        "If omitted, only prints summary statistics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/mps/cpu).",
    )
    args = parser.parse_args()

    # --- Device selection ------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("Prompt Difficulty Scorer  (attention-entropy probe)")
    logger.info("=" * 60)
    logger.info(f"  Probe model : {PROBE_MODEL_NAME}")
    logger.info(f"  Device      : {device}")
    logger.info(f"  Max tokens  : {MAX_LENGTH}")

    # --- Load prompts ----------------------------------------------------
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        logger.info(f"\n1. Loading prompts from {input_path.name} ...")
        prompts = load_jsonl_prompts(input_path)
    else:
        logger.info("\n1. Loading canonical reward-set prompts ...")
        prompts = load_reward_prompts()
    logger.info(f"   Loaded {len(prompts)} unique prompts")

    # --- Load probe model ------------------------------------------------
    logger.info(f"\n2. Loading probe model ({PROBE_MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(PROBE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PROBE_MODEL_NAME, attn_implementation="eager",
    ).to(device)
    model.eval()

    # --- Score -----------------------------------------------------------
    logger.info(f"\n3. Scoring {len(prompts)} prompts (batch_size={args.batch_size}) ...")
    t0 = time.perf_counter()
    scores = score_prompts_batched(
        prompts, model, tokenizer, device, batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"   Finished in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} prompts/s)")

    # --- Summary statistics ----------------------------------------------
    arr = np.array(scores)
    logger.info("\n4. Difficulty score distribution:")
    logger.info(f"   Mean   : {arr.mean():.4f}")
    logger.info(f"   Std    : {arr.std():.4f}")
    logger.info(f"   Median : {np.median(arr):.4f}")
    logger.info(f"   Min    : {arr.min():.4f}")
    logger.info(f"   Max    : {arr.max():.4f}")
    for pct in [10, 25, 75, 90, 95]:
        logger.info(f"   P{pct:<3d}  : {np.percentile(arr, pct):.4f}")

    # --- Quintile breakdown ----------------------------------------------
    logger.info("\n   Quintile breakdown:")
    for q, label in enumerate(["Easy", "Med-Easy", "Medium", "Med-Hard", "Hard"]):
        lo = np.percentile(arr, q * 20)
        hi = np.percentile(arr, (q + 1) * 20)
        mask = (arr >= lo) & (arr <= hi) if q == 4 else (arr >= lo) & (arr < hi)
        count = mask.sum()
        logger.info(f"     {label:>9s}  [{lo:.3f}, {hi:.3f})  n={count}")

    # --- Write output ----------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for prompt, score in zip(prompts, scores):
                f.write(json.dumps({"prompt": prompt, "difficulty": round(score, 6)}) + "\n")
        logger.info(f"\n5. Wrote {len(scores)} scored prompts to {output_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
