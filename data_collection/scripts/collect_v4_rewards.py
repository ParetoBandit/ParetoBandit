#!/usr/bin/env python3
"""Collect v4 rewards using the Meta-Cognitive Verifier rubric.

Unified script that handles both re-judging (existing responses) and
fresh collection (generate response + judge) for any set of models.
Pre-loads a response cache from all existing data sources, so API
credits are only spent on generation when responses are genuinely
missing.

Rubric v4 (50/35/15)
---------------------
1. Factual Integrity & Grounding (50 %) — correctness, hallucination
   detection, "Confident Bullshit" penalization.
2. Logic & Structural Depth (35 %) — adversarial reasoning check,
   System-2 thinking, counter-dependency analysis.
3. Edge Case & Nuance Recall (15 %) — implicit complexity, non-obvious
   limitations, "it depends" factors.

Judge: DeepSeek-R1 only (single judge, matching original pareto setup).
All tasks are shuffled across models to eliminate ordering effects.

Usage
-----
    # Default: 4-model K=3+onboarding collection
    python data_collection/scripts/collect_v4_rewards.py

    # Single model only
    python data_collection/scripts/collect_v4_rewards.py \\
        --models meta-llama/llama-3.1-8b-instruct

    # Quick test (50 prompts)
    python data_collection/scripts/collect_v4_rewards.py --limit 50

    # More workers
    python data_collection/scripts/collect_v4_rewards.py --workers 15

    # Summary only (no collection)
    python data_collection/scripts/collect_v4_rewards.py --summary-only

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

PARETO_PROMPTS = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_prompts.jsonl"
)
PARETO_REWARDS = (
    PROJECT_ROOT / "data_collection" / "pareto_dataset" / "pareto_rewards.jsonl"
)
MIDTIER_DIR = PROJECT_ROOT / "data_collection" / "midtier_candidates"
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v4_metacognitive"

DEFAULT_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemini-2.5-flash",
]

SEED = 42

# ── V4 Rubric ───────────────────────────────────────────────────────────

V4_SYSTEM_PROMPT = (
    "## Role\n"
    "You are an Expert Meta-Cognitive Verifier. Your task is to rigorously "
    "audit an LLM response. You must distinguish between \"surface-level "
    "fluency\" and \"structural correctness.\"\n\n"
    "## Internal Protocol (Pre-Scoring)\n"
    "Before providing scores, use your internal thinking space to:\n"
    "1. **Adversarial Deconstruction:** Actively try to find a scenario "
    "where the model's logic fails.\n"
    "2. **Template Detection:** Check if the model is just repeating a "
    "standard training-data pattern or actually reasoning through THIS "
    "specific prompt.\n"
    "3. **Complexity Check:** Does the response address the hardest 10% "
    "of the prompt, or just the easy 90%?\n\n"
    "---\n\n"
    "## Evaluation Criteria\n\n"
    "### 1. Factual Integrity & Grounding (50%)\n"
    "* **Frontier (0.9-1.0):** Zero hallucinations. Identifies if the "
    "prompt contains false premises.\n"
    "* **Mid (0.6-0.8):** Correct on main facts but might miss technical "
    "nuances or specific data points.\n"
    "* **Low (0.0-0.5):** Contains \"Confident Bullshit\"—authoritative "
    "tone but objectively false claims.\n\n"
    "### 2. Logic & Structural Depth (35%)\n"
    "* **Frontier (0.9-1.0):** **Counter-dependency check.** If Step A "
    "changes, does the model correctly update Step B? Exhibits \"System 2\" "
    "thinking.\n"
    "* **Mid (0.5-0.8):** Correct linear logic but \"fragile.\" Fails if "
    "the problem is slightly permuted.\n"
    "* **Low (0.0-0.4):** Circular reasoning, logical leaps, or "
    "\"stochastic parroting\" of the prompt.\n\n"
    "### 3. Edge Case & Nuance Recall (15%)\n"
    "* **Frontier (0.9-1.0):** Mentions at least one non-obvious "
    "limitation, edge case, or \"it depends\" factor.\n"
    "* **Mid (0.5-0.8):** Addresses all explicit parts of the prompt but "
    "ignores implicit complexity.\n"
    "* **Low (0.0-0.4):** Generic, one-size-fits-all response.\n\n"
    "---\n\n"
    "## Calibration for the Judge (Model Tiers)\n"
    "* **Detecting Low-Tier:** High verbosity, low info-density. Look for "
    "\"In conclusion,\" \"It is important to note,\" etc., used to pad a "
    "thin answer.\n"
    "* **Detecting Mid-Tier:** Accurate but \"Safe.\" It looks like a "
    "high-quality Wikipedia summary.\n"
    "* **Detecting Frontier:** Concise or deeply technical where needed. "
    "It might challenge the user's prompt or provide a \"Step 0\" "
    "(clarifying assumptions) that others missed.\n\n"
    "---\n\n"
    "## Output Format\n"
    "1. **Thought Trace:** [Briefly summarize your internal verification "
    "of their logic]\n"
    "2. **Correctness Score:** [0.0 - 1.0]\n"
    "3. **Reasoning Score:** [0.0 - 1.0]\n"
    "4. **Completeness Score:** [0.0 - 1.0]\n"
    "5. **Model Tier Classification:** [Low | Mid | Frontier] + "
    "1-sentence justification."
)

W_CORRECTNESS: float = 0.50
W_REASONING: float = 0.35
W_COMPLETENESS: float = 0.15

JUDGE_MODEL = "deepseek/deepseek-r1"
JUDGE_MAX_TOKENS = 2048
JUDGE_TIMEOUT = 180.0
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# ── Score parsing ───────────────────────────────────────────────────────


def _parse_score(content: str, heading: str, *, default: float = 0.5) -> float:
    """Extract a continuous 0.0-1.0 score from various markdown formats.

    Handles ``## Heading: 0.8``, ``**Heading:** 0.8``,
    ``2. **Heading:** [0.8]``, and plain ``Heading: 0.8``.

    Args:
        content: Full judge response text.
        heading: Regex pattern for the heading name.
        default: Fallback if no match is found.

    Returns:
        Score clamped to [0.0, 1.0].
    """
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
    """Extract the Model Tier Classification from judge output."""
    m = re.search(
        r"Model\s+Tier\s+Classification[:\*]*\s*\[?\s*(Low|Mid|Frontier)",
        content,
        re.IGNORECASE,
    )
    return m.group(1).capitalize() if m else "Unknown"


# ── V4 Reward Generator ────────────────────────────────────────────────


class V4RewardGenerator:
    """Collect rewards using the Meta-Cognitive Verifier rubric (v4).

    Manages a response cache so existing responses are reused.
    Generates fresh responses only when the cache misses.
    Judges every task with DeepSeek-R1 using the v4 system prompt.

    Args:
        max_workers: Parallel workers for API calls.
        api_key: OpenRouter API key (falls back to env).
    """

    def __init__(self, max_workers: int = 10, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv(PROJECT_ROOT / ".env")
                self.api_key = os.getenv("OPENROUTER_API_KEY")
            except ImportError:
                pass
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.base_url = "https://openrouter.ai/api/v1"
        self.max_workers = max_workers
        self.lock = threading.Lock()

        self.response_cache: Dict[Tuple[str, str], str] = {}

        self._stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"n": 0, "sum": 0.0, "sum_sq": 0.0}
        )

    # ── Cache loading ───────────────────────────────────────────────

    def load_response_cache(self, path: Path, *, model_filter: Optional[Set[str]] = None) -> int:
        """Load responses from an existing JSONL rewards file.

        Only loads records with ``ok=True`` and a non-empty ``response``.

        Args:
            path: Path to a JSONL file with ``model_id``, ``prompt``,
                ``response`` fields.
            model_filter: If provided, only load responses for these
                model IDs.

        Returns:
            Number of responses loaded.
        """
        if not path.exists():
            logger.warning("Cache file not found: %s", path)
            return 0

        count = 0
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not rec.get("ok") or not rec.get("response"):
                        continue
                    mid = rec["model_id"]
                    if model_filter and mid not in model_filter:
                        continue
                    key = (mid, rec["prompt"])
                    if key not in self.response_cache:
                        self.response_cache[key] = rec["response"]
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info("  Loaded %d responses from %s", count, path.name)
        return count

    def load_all_caches(self, models: List[str]) -> None:
        """Load response caches from all known data sources.

        Args:
            models: List of model IDs to collect for.
        """
        model_set = set(models)
        logger.info("Loading response caches for %d models...", len(model_set))

        total = 0
        if PARETO_REWARDS.exists():
            total += self.load_response_cache(PARETO_REWARDS, model_filter=model_set)

        if MIDTIER_DIR.is_dir():
            for jsonl_path in sorted(MIDTIER_DIR.glob("*.jsonl")):
                total += self.load_response_cache(jsonl_path, model_filter=model_set)

        logger.info("Total cached responses: %d", total)

    # ── Response generation ─────────────────────────────────────────

    def _get_response(self, model_id: str, prompt: str) -> Optional[str]:
        """Return a cached response or generate a fresh one via OpenRouter.

        Args:
            model_id: OpenRouter model identifier.
            prompt: User prompt text.

        Returns:
            Response text, or None on failure.
        """
        cached = self.response_cache.get((model_id, prompt))
        if cached is not None:
            return cached

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4000,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                if text:
                    with self.lock:
                        self.response_cache[(model_id, prompt)] = text
                    return text
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                else:
                    logger.warning(
                        "Response generation failed for %s after %d attempts: %s",
                        model_id, MAX_RETRIES, e,
                    )
        return None

    # ── Judging ─────────────────────────────────────────────────────

    def _judge(self, prompt: str, response: str) -> Optional[Dict[str, Any]]:
        """Judge a single (prompt, response) pair with DeepSeek-R1.

        Args:
            prompt: User prompt text.
            response: Model response to evaluate.

        Returns:
            Dict with per-dimension scores, composite, tier, and
            raw judge output. None on failure.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }

        effective_prompt = (
            V4_SYSTEM_PROMPT
            + "\n\nIMPORTANT: Keep the Thought Trace to 3-5 sentences. "
            "Be direct — identify errors or confirm correctness, "
            "then move to scoring."
        )

        payload = {
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": effective_prompt},
                {
                    "role": "user",
                    "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": JUDGE_MAX_TOKENS,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=JUDGE_TIMEOUT,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("API returned null content")
                content = content.strip()

                correctness = _parse_score(content, r"Correctness\s+Score")
                reasoning = _parse_score(content, r"Reasoning\s+Score")
                completeness = _parse_score(content, r"Completeness\s+Score")

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
                    "composite": round(composite, 4),
                    "tier_classification": tier,
                    "judge_raw": content,
                }

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                else:
                    logger.warning("Judge failed after %d attempts: %s", MAX_RETRIES, e)
                    return None
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                else:
                    logger.warning("Judge HTTP %d: %s", status, e)
                    return None
            except Exception as e:
                logger.warning("Judge unexpected error: %s", e)
                return None

        return None

    # ── Process a single task ───────────────────────────────────────

    def process_task(self, task: Tuple[str, str]) -> Dict[str, Any]:
        """Process a single (prompt, model_id) task.

        Retrieves the response (from cache or fresh generation),
        then judges with the v4 rubric.

        Args:
            task: Tuple of (prompt_text, model_id).

        Returns:
            Result dict suitable for JSONL serialization.
        """
        prompt_text, model_id = task

        response = self._get_response(model_id, prompt_text)
        if not response:
            return {
                "model_id": model_id,
                "prompt": prompt_text,
                "ok": False,
                "ts": time.time(),
            }

        cached = (model_id, prompt_text) in self.response_cache
        judge_result = self._judge(prompt_text, response)

        if judge_result is None:
            return {
                "model_id": model_id,
                "prompt": prompt_text,
                "response": response,
                "ok": False,
                "response_cached": cached,
                "ts": time.time(),
            }

        score = np.clip(judge_result["composite"], 0.01, 0.99)
        reward_logit = float(np.log(score / (1 - score)))

        result = {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "response_cached": cached,
            "rubric_version": "v4_metacognitive",
            "judge_model": JUDGE_MODEL,
            "correctness_score": judge_result["correctness_score"],
            "reasoning_score": judge_result["reasoning_score"],
            "completeness_score": judge_result["completeness_score"],
            "raw_score": judge_result["composite"],
            "reward_logit": reward_logit,
            "tier_classification": judge_result["tier_classification"],
            "judge_details": [
                {
                    "judge": JUDGE_MODEL,
                    "correctness_score": judge_result["correctness_score"],
                    "reasoning_score": judge_result["reasoning_score"],
                    "completeness_score": judge_result["completeness_score"],
                    "reward": judge_result["composite"],
                    "reasoning": judge_result["judge_raw"],
                }
            ],
            "ts": time.time(),
        }

        with self.lock:
            s = self._stats[model_id]
            s["n"] += 1
            s["sum"] += judge_result["composite"]
            s["sum_sq"] += judge_result["composite"] ** 2

        return result

    # ── Stats ───────────────────────────────────────────────────────

    def get_running_stats(self) -> Dict[str, Dict[str, float]]:
        """Return running mean and std per model."""
        out = {}
        for mid, s in self._stats.items():
            n = s["n"]
            if n < 1:
                continue
            mean = s["sum"] / n
            var = max(0.0, s["sum_sq"] / n - mean ** 2)
            out[mid] = {"n": n, "mean": round(mean, 4), "std": round(var ** 0.5, 4)}
        return out


# ── Data loading ────────────────────────────────────────────────────────


def load_prompts(path: Path = PARETO_PROMPTS, *, limit: Optional[int] = None) -> List[str]:
    """Load prompt texts from the pareto prompts file.

    Args:
        path: Path to pareto_prompts.jsonl.
        limit: If set, only return the first *limit* prompts.

    Returns:
        List of unique prompt strings.
    """
    prompts: List[str] = []
    seen: Set[str] = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            p = rec["prompt"]
            if p not in seen:
                prompts.append(p)
                seen.add(p)
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def load_completed(output_dir: Path, models: List[str]) -> Dict[str, Set[str]]:
    """Load already-completed (prompt) sets per model from output files.

    Args:
        output_dir: Directory containing per-model JSONL files.
        models: Model IDs to check.

    Returns:
        Mapping of model_id → set of completed prompt strings.
    """
    completed: Dict[str, Set[str]] = {m: set() for m in models}
    if not output_dir.exists():
        return completed

    for model_id in models:
        slug = model_id.replace("/", "_")
        path = output_dir / f"{slug}_v4.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok") and rec.get("model_id") == model_id:
                        completed[model_id].add(rec["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ── Output helpers ──────────────────────────────────────────────────────


def model_output_path(output_dir: Path, model_id: str) -> Path:
    """Return the per-model output JSONL path."""
    slug = model_id.replace("/", "_")
    return output_dir / f"{slug}_v4.jsonl"


def write_manifest(output_dir: Path, models: List[str], n_prompts: int) -> None:
    """Write a collection manifest with run metadata."""
    manifest = {
        "rubric_version": "v4_metacognitive",
        "judge_model": JUDGE_MODEL,
        "judge_panel_size": 1,
        "weights": {
            "correctness": W_CORRECTNESS,
            "reasoning": W_REASONING,
            "completeness": W_COMPLETENESS,
        },
        "models": models,
        "n_prompts": n_prompts,
        "seed": SEED,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = output_dir / "collection_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", path)


# ── Summary ─────────────────────────────────────────────────────────────


def print_summary(output_dir: Path, models: List[str]) -> None:
    """Print a summary table of collected v4 rewards."""
    print("\n" + "=" * 80)
    print("V4 META-COGNITIVE VERIFIER REWARD SUMMARY")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []
    for model_id in models:
        path = model_output_path(output_dir, model_id)
        if not path.exists():
            rows.append({"model": model_id, "n": 0})
            continue

        scores = []
        correctness_scores = []
        reasoning_scores = []
        completeness_scores = []
        tiers: Dict[str, int] = defaultdict(int)
        cached = 0

        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not rec.get("ok"):
                        continue
                    scores.append(rec["raw_score"])
                    correctness_scores.append(rec.get("correctness_score", 0))
                    reasoning_scores.append(rec.get("reasoning_score", 0))
                    completeness_scores.append(rec.get("completeness_score", 0))
                    tiers[rec.get("tier_classification", "Unknown")] += 1
                    if rec.get("response_cached"):
                        cached += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        if not scores:
            rows.append({"model": model_id, "n": 0})
            continue

        arr = np.array(scores)
        rows.append({
            "model": model_id,
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "corr": float(np.mean(correctness_scores)),
            "reas": float(np.mean(reasoning_scores)),
            "comp": float(np.mean(completeness_scores)),
            "cached": cached,
            "tiers": dict(tiers),
        })

    print(
        f"\n  {'Model':<42s} {'N':>6s} {'Mean':>7s} {'Std':>6s} "
        f"{'Corr':>6s} {'Reas':>6s} {'Comp':>6s} {'Cache%':>7s}"
    )
    print(f"  {'-' * 82}")

    for r in rows:
        if r["n"] == 0:
            print(f"  {r['model']:<42s} {'[no data]':>6s}")
            continue
        cache_pct = r["cached"] / r["n"] * 100
        print(
            f"  {r['model']:<42s} {r['n']:6d} {r['mean']:7.3f} "
            f"{r['std']:6.3f} {r['corr']:6.3f} {r['reas']:6.3f} "
            f"{r['comp']:6.3f} {cache_pct:6.1f}%"
        )
        tier_str = ", ".join(
            f"{k}={v}" for k, v in sorted(r["tiers"].items())
        )
        print(f"  {'':42s} Tiers: {tier_str}")

    # Pairwise gaps.
    scored = [r for r in rows if r["n"] > 0]
    if len(scored) >= 2:
        print(f"\n  Pairwise Δ (composite mean):")
        scored.sort(key=lambda r: r["mean"])
        for i in range(len(scored) - 1):
            a, b = scored[i], scored[i + 1]
            delta = b["mean"] - a["mean"]
            short_a = a["model"].split("/")[-1]
            short_b = b["model"].split("/")[-1]
            print(f"    {short_a} → {short_b}: {delta:+.4f}")

    print("=" * 80 + "\n")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model IDs to collect (default: 4-model K=3+onboarding set).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of prompts (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel workers for API calls (default: 10).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip collection, just print summary of existing results.",
    )
    args = parser.parse_args()

    models = args.models if args.models else DEFAULT_MODELS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_summary(OUTPUT_DIR, models)
        return

    # 1. Load prompts.
    logger.info("Loading prompts from %s", PARETO_PROMPTS)
    prompts = load_prompts(limit=args.limit)
    logger.info("  %d unique prompts loaded", len(prompts))

    # 2. Initialize generator and load response caches.
    gen = V4RewardGenerator(max_workers=args.workers)
    gen.load_all_caches(models)

    cache_report = []
    for mid in models:
        n_cached = sum(
            1 for p in prompts if (mid, p) in gen.response_cache
        )
        cache_report.append(f"  {mid}: {n_cached}/{len(prompts)} cached")
    logger.info("Response cache coverage:\n%s", "\n".join(cache_report))

    # 3. Load completed tasks (resume support).
    completed = load_completed(OUTPUT_DIR, models)
    total_completed = sum(len(v) for v in completed.values())
    if total_completed > 0:
        logger.info("Resuming: %d tasks already completed", total_completed)

    # 4. Build and shuffle task list.
    tasks: List[Tuple[str, str]] = []
    for prompt in prompts:
        for model_id in models:
            if prompt not in completed[model_id]:
                tasks.append((prompt, model_id))

    rng = random.Random(SEED)
    rng.shuffle(tasks)

    total_tasks = len(tasks)
    logger.info(
        "Tasks: %d remaining (%d prompts × %d models - %d completed)",
        total_tasks,
        len(prompts),
        len(models),
        total_completed,
    )

    if total_tasks == 0:
        logger.info("Nothing to do — all tasks completed.")
        print_summary(OUTPUT_DIR, models)
        return

    # 5. Open per-model output files.
    out_files: Dict[str, Any] = {}
    for model_id in models:
        path = model_output_path(OUTPUT_DIR, model_id)
        out_files[model_id] = open(path, "a")

    # 6. Write manifest.
    write_manifest(OUTPUT_DIR, models, len(prompts))

    # 7. Run parallel collection.
    logger.info("Starting collection with %d workers...", args.workers)
    diag_interval = max(100, total_tasks // 20)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(gen.process_task, t): t for t in tasks
            }
            done_count = 0
            with tqdm(total=total_tasks, desc="V4 Collection") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    model_id = result["model_id"]

                    with gen.lock:
                        f = out_files.get(model_id)
                        if f:
                            f.write(json.dumps(result) + "\n")
                            f.flush()
                        done_count += 1

                    pbar.update(1)

                    if done_count % diag_interval == 0:
                        stats = gen.get_running_stats()
                        parts = []
                        for mid in models:
                            s = stats.get(mid)
                            if s:
                                short = mid.split("/")[-1]
                                parts.append(
                                    f"{short}: {s['mean']:.3f}±{s['std']:.3f} "
                                    f"(n={s['n']})"
                                )
                        if parts:
                            logger.info("Progress — %s", " | ".join(parts))
    finally:
        for f in out_files.values():
            f.close()

    # 8. Final summary.
    print_summary(OUTPUT_DIR, models)


if __name__ == "__main__":
    main()
