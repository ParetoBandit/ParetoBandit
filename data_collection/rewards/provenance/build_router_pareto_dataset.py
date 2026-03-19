#!/usr/bin/env python3
"""
Build a public, reproducible full-information prompt dataset for a 3-arm
contextual router (budget / mid-cost / high-cost).

Pipeline
--------
1. **Source**: Load diverse prompts from public HuggingFace benchmarks
   (TruthfulQA, GSM8K, MBPP, ARC-Challenge, OpenBookQA, WinoGrande,
   MMLU, HellaSwag, BIG-Bench Hard).
2. **Filter**: Quality gates (length, ASCII ratio) + semantic deduplication
   against the existing canonical prompt set.
3. **Evaluate**: For each prompt × arm, generate a response via OpenRouter and
   judge it with DeepSeek-R1 as a single LLM judge using the v3 continuous
   rubric.
4. **Cost**: Compute per-arm cost from the K=3 model config pricing.
5. **Classify**: Label each prompt as ``trivial``, ``pareto_interesting``, or
   ``degenerate`` based on reward spread and Pareto dominance.
6. **Save**: Full-information JSONL compatible with ``merge_and_split_rewards.py``.

The reward pipeline reuses ``CoTRewardGenerator`` from ``rejudge_cot.py``
(with the judge panel narrowed to DeepSeek-R1 only) to guarantee identical
rubric and parsing behaviour.

Usage
-----
    # Full pipeline: source prompts → evaluate → classify → save
    python data_collection/scripts/build_router_pareto_dataset.py

    # Prompts-only: just build and save the prompt set (skip reward collection)
    python data_collection/scripts/build_router_pareto_dataset.py --prompts-only

    # Resume a previous reward-collection run:
    python data_collection/scripts/build_router_pareto_dataset.py --resume

    # Limit for testing:
    python data_collection/scripts/build_router_pareto_dataset.py --limit 50

Requirements
------------
    pip install datasets sentence-transformers scikit-learn requests tqdm numpy
    export OPENROUTER_API_KEY=...
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit.config import (
    CANONICAL_PROMPTS_PATH,
    DATA_COLLECTION_DIR,
    K3_MODELS_PATH,
    OFFLINE_DATASET_DIR,
    PROMPTS_DIR,
    REWARDS_PATH,
)

import requests as _requests

from data_collection.scripts.rejudge_cot import CoTRewardGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =========================================================================
# Cost-aware subclass of CoTRewardGenerator
# =========================================================================


class CostAwareRewardGenerator(CoTRewardGenerator):
    """Extends ``CoTRewardGenerator`` to capture per-request token usage and cost.

    The base class discards the ``usage`` field returned by the OpenRouter API.
    This subclass preserves it so that each reward record includes actual
    ``input_tokens``, ``output_tokens``, and ``cost_usd`` for the candidate
    model response (not the judge call).

    Parameters
    ----------
    arm_pricing:
        Mapping ``{model_id: ArmConfig}`` used to convert token counts
        into USD costs.
    api_key:
        OpenRouter API key (falls back to ``OPENROUTER_API_KEY`` env var).
    max_workers:
        Thread pool size for parallel API calls.
    """

    _RESPONSE_MAX_TOKENS: Dict[str, int] = {
        "google/gemini-2.5-pro": 16_000,
        "google/gemini-2.5-flash": 16_000,
    }
    _RESPONSE_TIMEOUT: Dict[str, float] = {
        "google/gemini-2.5-pro": 600.0,
        "google/gemini-2.5-flash": 300.0,
    }

    def __init__(
        self,
        arm_pricing: Dict[str, "ArmConfig"],
        api_key: Optional[str] = None,
        max_workers: int = 10,
    ) -> None:
        super().__init__(api_key=api_key, max_workers=max_workers)
        self.arm_pricing = arm_pricing

    def get_model_response_with_usage(
        self,
        model_id: str,
        prompt: str,
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """Get model response and return ``(content, input_tokens, output_tokens)``.

        Falls back to the parent's cache when available.  For cached responses
        token counts are unavailable and returned as ``None``.

        Parameters
        ----------
        model_id:
            OpenRouter model identifier.
        prompt:
            User prompt text.

        Returns
        -------
        tuple[str | None, int | None, int | None]
            ``(response_text, prompt_tokens, completion_tokens)``.
        """
        if (model_id, prompt) in self.response_cache:
            return self.response_cache[(model_id, prompt)], None, None

        max_tokens = self._RESPONSE_MAX_TOKENS.get(model_id, 4000)
        timeout = self._RESPONSE_TIMEOUT.get(model_id, 300)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = _requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                return content, prompt_tokens, completion_tokens
            except _requests.exceptions.HTTPError as e:
                last_exc = e
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("Response %s HTTP %d (non-retryable)", model_id, status)
                return None, None, None
            except (
                _requests.exceptions.Timeout,
                _requests.exceptions.ConnectionError,
            ) as e:
                last_exc = e
                time.sleep(2 ** attempt)
                continue
            except Exception:
                return None, None, None

        logger.warning(
            "Response %s failed after 3 attempts: %s", model_id, last_exc,
        )
        return None, None, None

    def process_task(self, task: Tuple[str, str]) -> Dict[str, Any]:
        """Process a single (prompt, model_id) pair with cost tracking.

        Overrides the base ``process_task`` to capture token usage from the
        OpenRouter API response and compute the actual USD cost based on
        the arm's published pricing.

        Parameters
        ----------
        task:
            ``(prompt_text, model_id)`` tuple.

        Returns
        -------
        dict
            Reward record with additional ``input_tokens``,
            ``output_tokens``, and ``cost_usd`` fields.
        """
        prompt_text, model_id = task

        response, input_tokens, output_tokens = self.get_model_response_with_usage(
            model_id, prompt_text,
        )

        if not response:
            return {"model_id": model_id, "ok": False, "ts": time.time()}

        final_score, judge_details = self.judge_with_panel_cot(
            prompt_text, response, model_id,
        )
        reward_logit = self.logit_transform(final_score)

        arm = self.arm_pricing.get(model_id)
        if arm and input_tokens is not None and output_tokens is not None:
            cost_usd = (
                input_tokens * arm.input_cost_per_m
                + output_tokens * arm.output_cost_per_m
            ) / 1_000_000
        elif arm:
            cost_usd = arm.request_cost()
        else:
            cost_usd = 0.0

        return {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "teacher_used": (model_id, prompt_text) in self.response_cache,
            "judge_details": judge_details,
            "reward_logit": reward_logit,
            "raw_score": final_score,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 8),
            "ts": time.time(),
        }

# =========================================================================
# Configuration
# =========================================================================

RNG_SEED: int = 2026
TARGET_PROMPTS: int = 12_000

MIN_PROMPT_LEN: int = 20
MAX_PROMPT_LEN: int = 5_000
MIN_ASCII_RATIO: float = 0.5
DEDUP_COSINE_THRESHOLD: float = 0.85

OUTPUT_DIR: Path = DATA_COLLECTION_DIR / "pareto_dataset"
PROMPTS_OUTPUT_PATH: Path = OUTPUT_DIR / "pareto_prompts_12k.jsonl"
REWARDS_OUTPUT_PATH: Path = OUTPUT_DIR / "pareto_rewards_12k.jsonl"
CLASSIFIED_OUTPUT_PATH: Path = OUTPUT_DIR / "pareto_classified_12k.jsonl"


@dataclass
class ArmConfig:
    """Configuration for a single router arm (candidate model)."""

    model_id: str
    display: str
    input_cost_per_m: float
    output_cost_per_m: float

    def request_cost(
        self,
        input_tokens: int = 100,
        output_tokens: int = 400,
    ) -> float:
        """Compute cost in USD for a single request.

        Parameters
        ----------
        input_tokens:
            Number of input tokens (default: 100, RouteLLM median).
        output_tokens:
            Number of output tokens (default: 400, RouteLLM median).

        Returns
        -------
        float
            Cost in USD.
        """
        return (
            input_tokens * self.input_cost_per_m
            + output_tokens * self.output_cost_per_m
        ) / 1_000_000


@dataclass
class SourceQuota:
    """Per-source prompt loading configuration."""

    name: str
    loader: str
    max_prompts: int
    loaded: int = 0


# =========================================================================
# 1. Load K=3 arm configuration
# =========================================================================


def load_arm_configs(models_path: Path = K3_MODELS_PATH) -> List[ArmConfig]:
    """Load arm configurations from the canonical K=3 models JSON.

    Parameters
    ----------
    models_path:
        Path to the models JSON file.

    Returns
    -------
    list[ArmConfig]
        One ``ArmConfig`` per model in the portfolio.
    """
    with open(models_path) as f:
        registry = json.load(f)

    arms: List[ArmConfig] = []
    for m in registry["models"]:
        arms.append(
            ArmConfig(
                model_id=m["model_id"],
                display=m["display"],
                input_cost_per_m=m["input_cost_per_m"],
                output_cost_per_m=m["output_cost_per_m"],
            )
        )
    logger.info("Loaded %d arms from %s", len(arms), models_path.name)
    for arm in arms:
        logger.info(
            "  %-40s  in=$%.2f/M  out=$%.2f/M  req=$%.6f",
            arm.model_id,
            arm.input_cost_per_m,
            arm.output_cost_per_m,
            arm.request_cost(),
        )
    return arms


# =========================================================================
# 2. Load public benchmark prompts from HuggingFace
# =========================================================================


def _load_truthfulqa(limit: int = 1000) -> List[Dict[str, str]]:
    """Load TruthfulQA generation questions."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        prompts.append({"prompt": row["question"], "source": "truthful_qa"})
    return prompts


def _load_gsm8k(limit: int = 2500) -> List[Dict[str, str]]:
    """Load GSM8K math word problems (train split)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        prompts.append({"prompt": row["question"], "source": "gsm8k"})
    return prompts


def _load_mbpp(limit: int = 500) -> List[Dict[str, str]]:
    """Load MBPP coding task descriptions."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        prompts.append({"prompt": row["text"], "source": "mbpp"})
    return prompts


def _load_arc_challenge(limit: int = 1200) -> List[Dict[str, str]]:
    """Load ARC-Challenge science reasoning questions."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        choices_text = ", ".join(
            f"({lbl}) {txt}"
            for lbl, txt in zip(row["choices"]["label"], row["choices"]["text"])
        )
        prompt = (
            f"Question: {row['question']}\n"
            f"Choices: {choices_text}\n"
            f"Answer with the correct option and explain your reasoning."
        )
        prompts.append({"prompt": prompt, "source": "arc_challenge"})
    return prompts


def _load_openbookqa(limit: int = 1200) -> List[Dict[str, str]]:
    """Load OpenBookQA general science questions."""
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        choices_text = ", ".join(
            f"({lbl}) {txt}"
            for lbl, txt in zip(row["choices"]["label"], row["choices"]["text"])
        )
        prompt = (
            f"Question: {row['question_stem']}\n"
            f"Choices: {choices_text}\n"
            f"Answer with the correct option."
        )
        prompts.append({"prompt": prompt, "source": "openbookqa"})
    return prompts


def _load_winogrande(limit: int = 1200) -> List[Dict[str, str]]:
    """Load WinoGrande commonsense reasoning prompts."""
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        prompt = (
            f"Complete the sentence with the correct option:\n"
            f"{row['sentence']}\n"
            f"Option 1: {row['option1']}\n"
            f"Option 2: {row['option2']}\n"
            f"Which option correctly fills the blank? Explain your reasoning."
        )
        prompts.append({"prompt": prompt, "source": "winogrande"})
    return prompts


def _load_mmlu(limit: int = 3000) -> List[Dict[str, str]]:
    """Load a diverse sample from MMLU (Massive Multitask Language Understanding).

    Samples across all available subjects to ensure broad topic coverage.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        choices = ["A", "B", "C", "D"]
        choices_text = "\n".join(
            f"  ({choices[i]}) {row['choices'][i]}"
            for i in range(len(row["choices"]))
        )
        prompt = (
            f"[{row['subject'].replace('_', ' ').title()}]\n"
            f"{row['question']}\n"
            f"{choices_text}\n"
            f"Answer with the correct letter and explain why."
        )
        prompts.append({"prompt": prompt, "source": "mmlu"})
    return prompts


def _load_hellaswag(limit: int = 1500) -> List[Dict[str, str]]:
    """Load HellaSwag commonsense NLI / story completion prompts."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="train")
    prompts = []
    for row in ds:
        if len(prompts) >= limit:
            break
        endings = "\n".join(
            f"  ({i+1}) {e}" for i, e in enumerate(row["endings"])
        )
        prompt = (
            f"Read the following context and choose the most plausible "
            f"continuation:\n\n"
            f"Context: {row['ctx']}\n\n"
            f"Options:\n{endings}\n\n"
            f"Which option is the best continuation? Explain briefly."
        )
        prompts.append({"prompt": prompt, "source": "hellaswag"})
    return prompts


def _load_bbh() -> List[Dict[str, str]]:
    """Load BIG-Bench Hard reasoning tasks (all 27 sub-tasks)."""
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names("lukaemon/bbh")
    prompts = []
    for cfg in configs:
        try:
            ds = load_dataset("lukaemon/bbh", cfg, split="test")
            for row in ds:
                q = row.get("input", "").strip()
                if q:
                    prompts.append({"prompt": q, "source": f"bbh/{cfg}"})
        except Exception:
            continue
    return prompts


# Source loader registry: (name, loader_func, target_count).
_SOURCE_LOADERS: List[Tuple[str, Any, int]] = [
    ("truthful_qa", _load_truthfulqa, 800),
    ("gsm8k", _load_gsm8k, 2500),
    ("mbpp", _load_mbpp, 500),
    ("arc_challenge", _load_arc_challenge, 1200),
    ("openbookqa", _load_openbookqa, 1200),
    ("winogrande", _load_winogrande, 1200),
    ("mmlu", _load_mmlu, 3000),
    ("hellaswag", _load_hellaswag, 1500),
    ("bbh", _load_bbh, 0),  # 0 = load all available
]


def load_public_prompt_sources() -> List[Dict[str, str]]:
    """Load a mix of public datasets covering QA, reasoning, math, and coding.

    Mirrors RouterBench / RouterEval diversity while staying fully open-source.
    Each returned dict has keys ``prompt`` and ``source``.

    Returns
    -------
    list[dict]
        Prompt records with ``prompt`` and ``source`` fields.
    """
    all_prompts: List[Dict[str, str]] = []

    for name, loader, target in _SOURCE_LOADERS:
        logger.info("  Loading %s (target: %s) ...", name, target or "all")
        try:
            if target > 0:
                loaded = loader(limit=target)
            else:
                loaded = loader()
            logger.info("    Loaded %d prompts from %s", len(loaded), name)
            all_prompts.extend(loaded)
        except Exception as exc:
            logger.warning("    Failed to load %s: %s", name, exc)

    logger.info("  Total raw prompts: %d", len(all_prompts))
    return all_prompts


# =========================================================================
# 3. Quality filtering and deduplication
# =========================================================================


def _ascii_ratio(text: str) -> float:
    """Fraction of characters that are ASCII."""
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def quality_filter(prompts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply length and ASCII-ratio quality gates.

    Parameters
    ----------
    prompts:
        Raw prompt records.

    Returns
    -------
    list[dict]
        Filtered prompt records.
    """
    filtered = [
        p
        for p in prompts
        if MIN_PROMPT_LEN <= len(p["prompt"]) <= MAX_PROMPT_LEN
        and _ascii_ratio(p["prompt"]) >= MIN_ASCII_RATIO
    ]
    logger.info(
        "  Quality filter: %d → %d (dropped %d)",
        len(prompts),
        len(filtered),
        len(prompts) - len(filtered),
    )
    return filtered


def load_existing_prompts() -> Set[str]:
    """Return all prompts already in the canonical reward dataset.

    These are excluded to avoid overlap with the existing 4K prompt set.

    Returns
    -------
    set[str]
        Existing prompt texts.
    """
    existing: Set[str] = set()

    if CANONICAL_PROMPTS_PATH.exists():
        with open(CANONICAL_PROMPTS_PATH) as f:
            for line in f:
                entry = json.loads(line)
                existing.add(entry.get("prompt", "").strip())

    if REWARDS_PATH.exists():
        with open(REWARDS_PATH) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        existing.add(entry.get("prompt", "").strip())
                except json.JSONDecodeError:
                    continue

    for gz_path in OFFLINE_DATASET_DIR.glob("*.jsonl.gz"):
        try:
            with gzip.open(gz_path, "rt") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("ok"):
                        existing.add(entry.get("prompt", "").strip())
        except Exception:
            continue

    logger.info("  Loaded %d existing prompts to exclude", len(existing))
    return existing


def deduplicate_prompts(
    prompts: List[Dict[str, str]],
    existing: Set[str],
) -> List[Dict[str, str]]:
    """Remove exact duplicates and prompts already in the existing dataset.

    Parameters
    ----------
    prompts:
        Candidate prompt records.
    existing:
        Set of prompt texts to exclude (from the canonical dataset).

    Returns
    -------
    list[dict]
        Deduplicated prompt records.
    """
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []

    for p in prompts:
        text = p["prompt"].strip()
        if text in seen or text in existing:
            continue
        seen.add(text)
        unique.append(p)

    logger.info(
        "  Exact dedup: %d → %d (removed %d exact + %d existing overlap)",
        len(prompts),
        len(unique),
        len(prompts) - len(unique) - len(seen & existing),
        len(seen & existing),
    )
    return unique


def semantic_dedup(
    prompts: List[Dict[str, str]],
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> List[Dict[str, str]]:
    """Remove semantically near-duplicate prompts via embedding similarity.

    Uses ``all-MiniLM-L6-v2`` for fast lightweight embeddings (not the
    heavier ``BAAI/bge-m3`` used for bandit features — we only need rough
    dedup here).

    Parameters
    ----------
    prompts:
        Prompt records after exact dedup.
    threshold:
        Cosine similarity above which a prompt is considered a duplicate.

    Returns
    -------
    list[dict]
        Semantically deduplicated prompt records.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(prompts) < 2:
        return prompts

    logger.info("  Encoding %d prompts for semantic dedup ...", len(prompts))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [p["prompt"] for p in prompts]
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=256)

    keep_indices: List[int] = []
    covered = np.zeros(len(prompts), dtype=bool)

    for i in range(len(prompts)):
        if not covered[i]:
            keep_indices.append(i)
            sims = cosine_similarity([embeddings[i]], embeddings)[0]
            covered[sims > threshold] = True

    unique = [prompts[i] for i in keep_indices]
    logger.info(
        "  Semantic dedup (threshold=%.2f): %d → %d",
        threshold,
        len(prompts),
        len(unique),
    )
    return unique


# =========================================================================
# 4. Full-information reward collection
# =========================================================================


def build_full_information_log(
    prompts: List[Dict[str, str]],
    arms: List[ArmConfig],
    generator: CoTRewardGenerator,
    output_path: Path,
    *,
    max_prompts: int = TARGET_PROMPTS,
    workers: int = 10,
) -> int:
    """Generate responses from all arms and judge with DeepSeek-R1.

    For each (prompt, arm) pair, the existing ``CoTRewardGenerator`` handles:
    - Response generation via OpenRouter
    - Single-judge R1 evaluation using the v3 continuous rubric
    - Logit transform

    Results are written incrementally to ``output_path`` in the canonical
    reward JSONL format. Supports resume: already-completed (prompt, model_id)
    pairs are skipped.

    Parameters
    ----------
    prompts:
        Prompt records (at most ``max_prompts`` will be processed).
    arms:
        Arm configurations.
    generator:
        Configured ``CoTRewardGenerator`` instance.
    output_path:
        Destination JSONL for reward records.
    max_prompts:
        Cap on prompts to process.
    workers:
        Parallel worker threads.

    Returns
    -------
    int
        Number of new records written.
    """
    prompt_texts = [p["prompt"] for p in prompts[:max_prompts]]
    prompt_sources = {
        p["prompt"]: p.get("source", "unknown") for p in prompts[:max_prompts]
    }

    tasks: List[Tuple[str, str]] = []
    for text in prompt_texts:
        for arm in arms:
            tasks.append((text, arm.model_id))

    logger.info(
        "Full-information log: %d prompts × %d arms = %d tasks",
        len(prompt_texts),
        len(arms),
        len(tasks),
    )

    completed: Set[Tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    key = (entry.get("prompt", ""), entry.get("model_id", ""))
                    completed.add(key)
                except json.JSONDecodeError:
                    continue
        if completed:
            logger.info("  Resume: %d tasks already completed", len(completed))

    remaining = [t for t in tasks if t not in completed]
    logger.info("  Tasks to run: %d (skipped %d)", len(remaining), len(tasks) - len(remaining))

    if not remaining:
        logger.info("  Nothing to do.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    lock = threading.Lock()

    def _process_task(task: Tuple[str, str]) -> Dict[str, Any]:
        """Process a single (prompt, model_id) pair."""
        return generator.process_task(task)

    diag_interval = max(100, len(remaining) // 10)

    with open(output_path, "a") as outfile:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_task, t): t for t in remaining
            }
            with tqdm(total=len(remaining), desc="Reward collection") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    source = prompt_sources.get(result.get("prompt", ""), "unknown")
                    result["benchmark_source"] = source

                    with lock:
                        outfile.write(json.dumps(result) + "\n")
                        outfile.flush()
                        written += 1
                    pbar.update(1)

                    if written % diag_interval == 0:
                        generator.diagnostics.log_report(min_samples=30)

    logger.info("Wrote %d new reward records to %s", written, output_path)
    return written


# =========================================================================
# 5. Cost computation
# =========================================================================


def compute_arm_costs(
    rewards_path: Path,
    arms: List[ArmConfig],
) -> Dict[str, Dict[str, float]]:
    """Extract per-arm costs from reward records.

    Each record written by ``CostAwareRewardGenerator`` includes a
    ``cost_usd`` field computed from real token counts.  If that field
    is missing (e.g. from a cached response), falls back to an estimate
    using the arm's published pricing and RouteLLM median token counts.

    Parameters
    ----------
    rewards_path:
        Path to the full-information rewards JSONL.
    arms:
        Arm configurations with pricing.

    Returns
    -------
    dict
        Mapping ``{prompt: {model_id: cost_usd}}``.
    """
    arm_lookup = {a.model_id: a for a in arms}
    costs: Dict[str, Dict[str, float]] = defaultdict(dict)

    with open(rewards_path) as f:
        for line in f:
            rec = json.loads(line)
            prompt = rec.get("prompt", "")
            mid = rec.get("model_id", "")
            if mid not in arm_lookup:
                continue

            if rec.get("cost_usd") is not None:
                costs[prompt][mid] = rec["cost_usd"]
            else:
                arm = arm_lookup[mid]
                in_tok = rec.get("input_tokens")
                out_tok = rec.get("output_tokens")
                if in_tok is not None and out_tok is not None:
                    costs[prompt][mid] = (
                        in_tok * arm.input_cost_per_m
                        + out_tok * arm.output_cost_per_m
                    ) / 1_000_000
                else:
                    costs[prompt][mid] = arm.request_cost()

    return dict(costs)


# =========================================================================
# 6. Pareto-interesting classification
# =========================================================================

SPREAD_TRIVIAL: float = 0.05
SPREAD_INTERESTING: float = 0.10
NEAR_BEST_TOL: float = 0.05


def classify_prompt_record(
    prompt: str,
    arm_rewards: Dict[str, float],
    arm_costs: Dict[str, float],
    arms: List[ArmConfig],
) -> Dict[str, Any]:
    """Classify a single prompt based on reward-cost tradeoffs.

    A prompt is **Pareto-interesting** when a cheaper arm achieves
    near-best quality, creating a non-trivial routing decision.

    Parameters
    ----------
    prompt:
        The prompt text.
    arm_rewards:
        ``{model_id: reward}`` for each arm.
    arm_costs:
        ``{model_id: cost}`` for each arm.
    arms:
        Ordered arm configs (assumed budget → high-cost ordering).

    Returns
    -------
    dict
        Classification record with reward stats and difficulty label.
    """
    rewards = {a.model_id: arm_rewards.get(a.model_id, float("nan")) for a in arms}
    costs = {a.model_id: arm_costs.get(a.model_id, 0.0) for a in arms}

    finite_rewards = {k: v for k, v in rewards.items() if np.isfinite(v)}
    if not finite_rewards:
        return {
            "prompt": prompt,
            "difficulty": "degenerate",
            "best_arm": None,
            "reward_spread": 0.0,
            "arms": {},
        }

    best_arm = max(finite_rewards, key=finite_rewards.get)  # type: ignore[arg-type]
    worst_arm = min(finite_rewards, key=finite_rewards.get)  # type: ignore[arg-type]
    best_reward = finite_rewards[best_arm]
    worst_reward = finite_rewards[worst_arm]
    spread = best_reward - worst_reward

    def _is_near_best(model_id: str) -> bool:
        r = finite_rewards.get(model_id, float("-inf"))
        return r >= best_reward - NEAR_BEST_TOL

    cheapest_near_best = any(
        _is_near_best(a.model_id) and costs.get(a.model_id, 0) < costs.get(best_arm, 0)
        for a in arms
    )

    if spread <= SPREAD_TRIVIAL:
        difficulty = "trivial"
    elif spread >= SPREAD_INTERESTING and cheapest_near_best:
        difficulty = "pareto_interesting"
    elif spread >= SPREAD_INTERESTING:
        difficulty = "hard_but_dominated"
    else:
        difficulty = "moderate"

    arm_details = {}
    for a in arms:
        arm_details[a.model_id] = {
            "reward": round(rewards.get(a.model_id, float("nan")), 4),
            "cost": round(costs.get(a.model_id, 0.0), 8),
            "near_best": _is_near_best(a.model_id),
        }

    return {
        "prompt": prompt,
        "difficulty": difficulty,
        "best_arm": best_arm,
        "best_reward": round(best_reward, 4),
        "worst_reward": round(worst_reward, 4),
        "reward_spread": round(spread, 4),
        "arms": arm_details,
    }


def classify_all_prompts(
    rewards_path: Path,
    arms: List[ArmConfig],
) -> List[Dict[str, Any]]:
    """Classify all prompts in a reward file.

    Groups reward records by prompt, computes per-arm reward/cost,
    and classifies each prompt.

    Parameters
    ----------
    rewards_path:
        Path to the full-information rewards JSONL.
    arms:
        Arm configurations.

    Returns
    -------
    list[dict]
        One classification record per prompt.
    """
    arm_ids = {a.model_id for a in arms}
    prompt_rewards: Dict[str, Dict[str, float]] = defaultdict(dict)
    prompt_sources: Dict[str, str] = {}

    with open(rewards_path) as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            mid = rec.get("model_id", "")
            if mid not in arm_ids:
                continue
            prompt = rec["prompt"]
            raw_score = rec.get("raw_score")
            if raw_score is not None and np.isfinite(raw_score):
                prompt_rewards[prompt][mid] = raw_score
            if "benchmark_source" in rec:
                prompt_sources[prompt] = rec["benchmark_source"]

    costs = compute_arm_costs(rewards_path, arms)

    classified: List[Dict[str, Any]] = []
    for prompt, arm_r in prompt_rewards.items():
        if len(arm_r) < len(arms):
            continue
        record = classify_prompt_record(prompt, arm_r, costs.get(prompt, {}), arms)
        record["source"] = prompt_sources.get(prompt, "unknown")
        classified.append(record)

    bins = defaultdict(int)
    for r in classified:
        bins[r["difficulty"]] += 1

    logger.info("Classification results (%d prompts):", len(classified))
    for label, count in sorted(bins.items()):
        logger.info("  %-25s %5d  (%.1f%%)", label, count, 100 * count / max(len(classified), 1))

    return classified


def filter_balanced_set(
    classified: List[Dict[str, Any]],
    seed: int = RNG_SEED,
) -> List[Dict[str, Any]]:
    """Build a balanced subset emphasizing Pareto-interesting prompts.

    Keeps all Pareto-interesting prompts and samples from other bins
    to form a roughly balanced dataset.

    Parameters
    ----------
    classified:
        Full list of classified prompt records.
    seed:
        Random seed for sampling.

    Returns
    -------
    list[dict]
        Balanced subset of classified records.
    """
    rng = np.random.RandomState(seed)
    bins: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in classified:
        bins[r["difficulty"]].append(r)

    pareto = bins.get("pareto_interesting", [])
    target_per_bin = max(len(pareto), 100)

    selected: List[Dict[str, Any]] = list(pareto)
    for label in ["trivial", "moderate", "hard_but_dominated", "degenerate"]:
        pool = bins.get(label, [])
        n = min(len(pool), target_per_bin)
        if n > 0:
            indices = rng.choice(len(pool), size=n, replace=False)
            selected.extend(pool[i] for i in indices)

    rng.shuffle(selected)  # type: ignore[arg-type]
    logger.info(
        "Balanced set: %d prompts (%d pareto_interesting, %d other)",
        len(selected),
        len(pareto),
        len(selected) - len(pareto),
    )
    return selected


# =========================================================================
# 7. Save outputs
# =========================================================================


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records as JSONL.

    Parameters
    ----------
    records:
        Dicts to serialize.
    path:
        Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# =========================================================================
# 8. Main
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a public, reproducible full-information prompt dataset "
            "for a 3-arm contextual router."
        ),
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_PROMPTS,
        help=f"Target number of prompts (default: {TARGET_PROMPTS}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit total prompts to process (for testing).",
    )
    parser.add_argument(
        "--prompts-only", action="store_true",
        help="Only build the prompt set; skip reward collection.",
    )
    parser.add_argument(
        "--classify-only", action="store_true",
        help="Only classify existing rewards (skip prompt sourcing and collection).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous reward-collection run.",
    )
    parser.add_argument(
        "--no-semantic-dedup", action="store_true",
        help="Skip semantic deduplication (faster, slightly noisier).",
    )
    parser.add_argument(
        "--models-file", type=str, default=None,
        help="Path to models JSON config (default: K=3 canonical).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Parallel workers for API calls (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=RNG_SEED,
        help=f"Random seed (default: {RNG_SEED}).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    models_path = Path(args.models_file) if args.models_file else K3_MODELS_PATH
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    prompts_path = output_dir / "pareto_prompts.jsonl"
    rewards_path = output_dir / "pareto_rewards.jsonl"
    classified_path = output_dir / "pareto_classified.jsonl"

    logger.info("=" * 70)
    logger.info("Build Router Pareto Dataset")
    logger.info("=" * 70)
    logger.info("  Target prompts : %d", args.target)
    logger.info("  Models config  : %s", models_path)
    logger.info("  Output dir     : %s", output_dir)
    logger.info("  Seed           : %d", args.seed)

    arms = load_arm_configs(models_path)

    # ------------------------------------------------------------------
    # Phase 1: Prompt sourcing (skip if --classify-only or --resume)
    # ------------------------------------------------------------------
    if not args.classify_only:
        if args.resume and prompts_path.exists():
            logger.info("\n--- Phase 1: Loading existing prompts (--resume) ---")
            prompts: List[Dict[str, str]] = []
            with open(prompts_path) as f:
                for line in f:
                    prompts.append(json.loads(line))
            logger.info("  Loaded %d prompts from %s", len(prompts), prompts_path)
        else:
            logger.info("\n--- Phase 1: Source public benchmark prompts ---")
            existing = load_existing_prompts()
            raw = load_public_prompt_sources()
            filtered = quality_filter(raw)
            unique = deduplicate_prompts(filtered, existing)

            if not args.no_semantic_dedup:
                unique = semantic_dedup(unique, threshold=DEDUP_COSINE_THRESHOLD)

            if args.limit:
                unique = unique[: args.limit]
            elif len(unique) > args.target:
                rng = np.random.RandomState(args.seed)
                indices = rng.choice(len(unique), size=args.target, replace=False)
                unique = [unique[i] for i in sorted(indices)]
                logger.info("  Subsampled to %d prompts", len(unique))

            prompts = unique
            save_jsonl(prompts, prompts_path)

        logger.info("  Final prompt count: %d", len(prompts))

        # Source distribution summary.
        source_counts: Dict[str, int] = defaultdict(int)
        for p in prompts:
            source_counts[p.get("source", "unknown")] += 1
        logger.info("  Source distribution:")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            logger.info("    %-20s %5d", src, cnt)

        if args.prompts_only:
            logger.info("\n--prompts-only: skipping reward collection.")
            logger.info("Done.")
            return

        # ------------------------------------------------------------------
        # Phase 2: Reward collection (single R1 judge, real cost tracking)
        # ------------------------------------------------------------------
        logger.info("\n--- Phase 2: Full-information reward collection (R1 judge) ---")
        arm_pricing = {a.model_id: a for a in arms}
        generator = CostAwareRewardGenerator(
            arm_pricing=arm_pricing, max_workers=args.workers,
        )
        generator.judge_panel = ["deepseek/deepseek-r1"]
        logger.info("  Judge panel: %s", generator.judge_panel)
        build_full_information_log(
            prompts,
            arms,
            generator,
            rewards_path,
            max_prompts=args.limit or args.target,
            workers=args.workers,
        )

        diag_path = rewards_path.with_suffix(".judge_diagnostics.json")
        diag_payload = {
            "summary": generator.diagnostics.per_judge_summary(),
            "bias_matrix": generator.diagnostics.bias_matrix(),
            "weights_equal": generator.diagnostics.compute_weights("equal"),
            "weights_inverse_variance": generator.diagnostics.compute_weights(
                "inverse_variance"
            ),
            "weights_inverse_bias": generator.diagnostics.compute_weights(
                "inverse_bias"
            ),
        }
        with open(diag_path, "w") as df:
            json.dump(diag_payload, df, indent=2)
        logger.info("Judge diagnostics written to %s", diag_path)

    # ------------------------------------------------------------------
    # Phase 3: Pareto classification
    # ------------------------------------------------------------------
    logger.info("\n--- Phase 3: Pareto-interesting classification ---")

    if not rewards_path.exists():
        logger.error("Rewards file not found: %s", rewards_path)
        logger.error("Run without --classify-only first.")
        sys.exit(1)

    classified = classify_all_prompts(rewards_path, arms)
    save_jsonl(classified, classified_path)

    balanced = filter_balanced_set(classified, seed=args.seed)
    balanced_path = output_dir / "pareto_balanced.jsonl"
    save_jsonl(balanced, balanced_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("  Prompts file   : %s", prompts_path)
    logger.info("  Rewards file   : %s", rewards_path)
    logger.info("  Classified     : %s", classified_path)
    logger.info("  Balanced set   : %s", balanced_path)
    logger.info(
        "  Total classified prompts: %d  |  Balanced: %d",
        len(classified),
        len(balanced),
    )

    if classified:
        all_spreads = [r["reward_spread"] for r in classified]
        logger.info(
            "  Reward spread — mean=%.3f  std=%.3f  median=%.3f",
            np.mean(all_spreads),
            np.std(all_spreads),
            np.median(all_spreads),
        )

    logger.info("\nTo merge into the canonical dataset, run:")
    logger.info(
        "  python data_collection/scripts/merge_and_split_rewards.py \\\n"
        "    --input %s %s \\\n"
        "    --models-file %s \\\n"
        "    --output-dir data_collection/rewards \\\n"
        "    --prefix k3_merged \\\n"
        "    --no-panel-filter",
        REWARDS_PATH,
        rewards_path,
        models_path,
    )
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
