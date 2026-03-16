#!/usr/bin/env python3
"""Collect v5 rewards using a two-judge panel: R1 (correctness) + Opus 4.6 (reasoning, completeness).

Each (prompt, response) pair is independently judged by both DeepSeek-R1 and
Claude Opus 4.6 using the "Senior Technical Auditor" rubric.  The final
composite cherry-picks the strongest pillar from each judge:

    composite = 0.50 * R1_correctness + 0.35 * Opus_reasoning + 0.15 * Opus_completeness

This maximises discrimination across all four model tiers (min Cohen's d = 0.33)
by leveraging R1's strength at budget-tier separation and Opus's strength at
frontier-tier separation.

Usage
-----
    # Full run (30 workers by default)
    python data_collection/scripts/collect_v5_dual_judge.py

    # Quick test
    python data_collection/scripts/collect_v5_dual_judge.py --limit 5

    # Custom workers
    python data_collection/scripts/collect_v5_dual_judge.py --workers 20

    # Summary only
    python data_collection/scripts/collect_v5_dual_judge.py --summary-only

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
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
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
V4_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v4_metacognitive"
OUTPUT_DIR = PROJECT_ROOT / "data_collection" / "rewards" / "v5_dual_judge"

DEFAULT_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
]

SEED = 42

# ── V5 Rubric: Senior Technical Auditor ─────────────────────────────────

V5_SYSTEM_PROMPT = (
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

# Cherry-pick weights: R1 correctness + Opus reasoning & completeness.
W_CORRECTNESS: float = 0.50
W_REASONING: float = 0.35
W_COMPLETENESS: float = 0.15

# Judge models.
JUDGE_R1 = "deepseek/deepseek-r1"
JUDGE_OPUS = "anthropic/claude-opus-4.6"

# Per-judge config.
JUDGE_CONFIG: Dict[str, Dict[str, Any]] = {
    JUDGE_R1: {
        "max_tokens": 2048,
        "timeout": 180.0,
        "system_suffix": (
            "\n\nIMPORTANT: Keep the Reasoning Trace to 3-5 sentences. "
            "Be direct — identify errors or confirm correctness, then move to scoring."
        ),
    },
    JUDGE_OPUS: {
        "max_tokens": 4096,
        "timeout": 300.0,
        "system_suffix": "",
    },
}

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# ── Adaptive throttle ──────────────────────────────────────────────────

THROTTLE_WINDOW = 100
THROTTLE_THRESHOLD = 0.05
THROTTLE_SLEEP = 2.0


class AdaptiveThrottle:
    """Track per-judge 429 rate and inject backoff when needed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=THROTTLE_WINDOW))
        self._totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "429s": 0})

    def record(self, judge: str, was_429: bool) -> None:
        with self._lock:
            self._windows[judge].append(1 if was_429 else 0)
            self._totals[judge]["calls"] += 1
            if was_429:
                self._totals[judge]["429s"] += 1

    def should_throttle(self, judge: str) -> bool:
        with self._lock:
            w = self._windows[judge]
            if len(w) < 10:
                return False
            return sum(w) / len(w) > THROTTLE_THRESHOLD

    def wait_if_needed(self, judge: str) -> None:
        if self.should_throttle(judge):
            time.sleep(THROTTLE_SLEEP)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            out = {}
            for j, t in self._totals.items():
                rate = t["429s"] / max(t["calls"], 1)
                out[j.split("/")[-1]] = {
                    "calls": t["calls"],
                    "429s": t["429s"],
                    "rate": f"{rate:.1%}",
                }
            return out


# Global throttle instance.
throttle = AdaptiveThrottle()

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
        r"#{1,3}\s*" + heading + r"\s*[:\-]?\s*\[?\s*(\d+\.?\d*)",
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
    """Extract a Tier Label from judge output."""
    m = re.search(r"\bTier\s*(?:Label)?[:\s]*\[?\s*(Low|Mid|Frontier)\b", content, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m = re.search(r"\b(Low|Mid|Frontier)\b", content)
    return m.group(1).capitalize() if m else "Unknown"


# ── Dual-Judge Reward Generator ─────────────────────────────────────────


class V5DualJudgeGenerator:
    """Collect rewards using a two-judge panel (R1 + Opus 4.6).

    For each (prompt, response) pair:
      1. R1 judges and we extract ``Factual Correctness`` (weight 0.50).
      2. Opus judges and we extract ``Reasoning Depth`` (0.35) and
         ``Completeness & Nuance`` (0.15).
      3. Composite = cherry-pick sum.

    Args:
        max_workers: Parallel task workers.
        api_key: OpenRouter API key (falls back to env / .env).
    """

    def __init__(self, max_workers: int = 30, api_key: Optional[str] = None) -> None:
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
            lambda: {"n": 0, "sum": 0.0, "sum_sq": 0.0,
                     "r1_corr_sum": 0.0, "opus_reas_sum": 0.0, "opus_comp_sum": 0.0}
        )

    # ── Cache loading ───────────────────────────────────────────────

    def load_response_cache(
        self, path: Path, *, model_filter: Optional[Set[str]] = None,
    ) -> int:
        """Load responses from an existing JSONL file into the cache.

        Args:
            path: JSONL with ``model_id``, ``prompt``, ``response`` fields.
            model_filter: Only load these model IDs if set.

        Returns:
            Number of new responses loaded.
        """
        if not path.exists():
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
        if count:
            logger.info("  Loaded %d responses from %s", count, path.name)
        return count

    def load_all_caches(self, models: List[str]) -> None:
        """Load response caches from all known data sources."""
        model_set = set(models)
        logger.info("Loading response caches for %d models...", len(model_set))
        total = 0

        if PARETO_REWARDS.exists():
            total += self.load_response_cache(PARETO_REWARDS, model_filter=model_set)

        if MIDTIER_DIR.is_dir():
            for p in sorted(MIDTIER_DIR.glob("*.jsonl")):
                total += self.load_response_cache(p, model_filter=model_set)

        if V4_DIR.is_dir():
            for p in sorted(V4_DIR.glob("*.jsonl")):
                total += self.load_response_cache(p, model_filter=model_set)

        logger.info("Total cached responses: %d", total)

    # ── Response generation ─────────────────────────────────────────

    def _get_response(self, model_id: str, prompt: str) -> Optional[str]:
        """Return a cached response or generate a fresh one via OpenRouter."""
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
                    logger.warning("Response gen failed for %s: %s", model_id, e)
        return None

    # ── Single judge call ───────────────────────────────────────────

    def _call_judge(
        self, judge_model: str, prompt: str, response: str,
    ) -> Optional[Dict[str, Any]]:
        """Send (prompt, response) to a single judge and parse scores.

        Args:
            judge_model: OpenRouter judge model identifier.
            prompt: Original user prompt.
            response: Model response to evaluate.

        Returns:
            Dict with parsed scores and raw output, or None on failure.
        """
        cfg = JUDGE_CONFIG[judge_model]
        system_prompt = V5_SYSTEM_PROMPT + cfg["system_suffix"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
        }
        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}"},
            ],
            "temperature": 0.0,
            "max_tokens": cfg["max_tokens"],
        }

        for attempt in range(1, MAX_RETRIES + 1):
            throttle.wait_if_needed(judge_model)
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=cfg["timeout"],
                )
                resp.raise_for_status()
                throttle.record(judge_model, False)

                content = resp.json()["choices"][0]["message"]["content"]
                if content is None:
                    raise ValueError("Null content from API")
                content = content.strip()

                correctness = _parse_score(content, r"Factual\s+Correctness")
                reasoning = _parse_score(content, r"Reasoning\s+Depth")
                completeness = _parse_score(content, r"Completeness\s*(?:&|and)?\s*Nuance")
                tier = _parse_tier(content)

                return {
                    "correctness": round(correctness, 4),
                    "reasoning": round(reasoning, 4),
                    "completeness": round(completeness, 4),
                    "tier": tier,
                    "raw": content,
                }

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    throttle.record(judge_model, True)
                    time.sleep(RETRY_BACKOFF_BASE ** attempt + random.random())
                elif status in (500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** attempt)
                else:
                    logger.warning("Judge %s HTTP %d: %s", judge_model, status, e)
                    return None
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
                    logger.warning("Judge %s failed: %s", judge_model, e)
                    return None
            except Exception as e:
                logger.warning("Judge %s unexpected: %s", judge_model, e)
                return None

        return None

    # ── Process a single task ───────────────────────────────────────

    def process_task(self, task: Tuple[str, str]) -> Dict[str, Any]:
        """Process a single (prompt, model_id) task with both judges.

        Steps:
          1. Get response (cache or generate).
          2. Call R1 → extract correctness.
          3. Call Opus → extract reasoning + completeness.
          4. Compute cherry-pick composite.

        Args:
            task: (prompt_text, model_id).

        Returns:
            Result dict for JSONL serialization.
        """
        prompt_text, model_id = task

        response = self._get_response(model_id, prompt_text)
        if not response:
            return {
                "model_id": model_id,
                "prompt": prompt_text,
                "ok": False,
                "error": "response_generation_failed",
                "ts": time.time(),
            }

        was_cached = (model_id, prompt_text) in self.response_cache

        r1_result = self._call_judge(JUDGE_R1, prompt_text, response)
        opus_result = self._call_judge(JUDGE_OPUS, prompt_text, response)

        if r1_result is None or opus_result is None:
            failed_judge = "r1" if r1_result is None else "opus"
            return {
                "model_id": model_id,
                "prompt": prompt_text,
                "response": response,
                "ok": False,
                "error": f"judge_failed_{failed_judge}",
                "response_cached": was_cached,
                "ts": time.time(),
            }

        r1_corr = r1_result["correctness"]
        opus_reas = opus_result["reasoning"]
        opus_comp = opus_result["completeness"]

        composite = (
            r1_corr * W_CORRECTNESS
            + opus_reas * W_REASONING
            + opus_comp * W_COMPLETENESS
        )
        composite = round(composite, 4)

        score_clipped = np.clip(composite, 0.01, 0.99)
        reward_logit = float(np.log(score_clipped / (1.0 - score_clipped)))

        result = {
            "model_id": model_id,
            "prompt": prompt_text,
            "response": response,
            "ok": True,
            "response_cached": was_cached,
            "rubric_version": "v5_dual_judge",
            "r1_correctness": r1_corr,
            "opus_reasoning": opus_reas,
            "opus_completeness": opus_comp,
            "raw_score": composite,
            "reward_logit": round(reward_logit, 4),
            "r1_tier": r1_result["tier"],
            "opus_tier": opus_result["tier"],
            "judge_details": {
                "r1": {
                    "judge": JUDGE_R1,
                    "correctness": r1_result["correctness"],
                    "reasoning": r1_result["reasoning"],
                    "completeness": r1_result["completeness"],
                    "tier": r1_result["tier"],
                    "raw": r1_result["raw"],
                },
                "opus": {
                    "judge": JUDGE_OPUS,
                    "correctness": opus_result["correctness"],
                    "reasoning": opus_result["reasoning"],
                    "completeness": opus_result["completeness"],
                    "tier": opus_result["tier"],
                    "raw": opus_result["raw"],
                },
            },
            "ts": time.time(),
        }

        with self.lock:
            s = self._stats[model_id]
            s["n"] += 1
            s["sum"] += composite
            s["sum_sq"] += composite ** 2
            s["r1_corr_sum"] += r1_corr
            s["opus_reas_sum"] += opus_reas
            s["opus_comp_sum"] += opus_comp

        return result

    # ── Running stats ───────────────────────────────────────────────

    def get_running_stats(self) -> Dict[str, Dict[str, float]]:
        """Return running mean, std, and per-pillar means per model."""
        out = {}
        for mid, s in self._stats.items():
            n = s["n"]
            if n < 1:
                continue
            mean = s["sum"] / n
            var = max(0.0, s["sum_sq"] / n - mean ** 2)
            out[mid] = {
                "n": n,
                "mean": round(mean, 4),
                "std": round(var ** 0.5, 4),
                "r1_corr": round(s["r1_corr_sum"] / n, 4),
                "opus_reas": round(s["opus_reas_sum"] / n, 4),
                "opus_comp": round(s["opus_comp_sum"] / n, 4),
            }
        return out


# ── Data loading ────────────────────────────────────────────────────────


def load_prompts(
    path: Path = PARETO_PROMPTS, *, limit: Optional[int] = None,
) -> List[str]:
    """Load unique prompt texts from the pareto prompts file."""
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
    """Load already-completed (prompt) sets per model from v5 output files."""
    completed: Dict[str, Set[str]] = {m: set() for m in models}
    if not output_dir.exists():
        return completed
    for model_id in models:
        slug = model_id.replace("/", "_")
        path = output_dir / f"{slug}_v5.jsonl"
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


def model_output_path(output_dir: Path, model_id: str) -> Path:
    slug = model_id.replace("/", "_")
    return output_dir / f"{slug}_v5.jsonl"


def write_manifest(output_dir: Path, models: List[str], n_prompts: int) -> None:
    """Write collection manifest with run metadata."""
    manifest = {
        "rubric_version": "v5_dual_judge",
        "rubric_name": "Senior Technical Auditor",
        "judges": {
            "r1": {"model": JUDGE_R1, "pillar": "correctness", "weight": W_CORRECTNESS},
            "opus": {"model": JUDGE_OPUS, "pillars": ["reasoning", "completeness"],
                     "weights": {"reasoning": W_REASONING, "completeness": W_COMPLETENESS}},
        },
        "composite_formula": "0.50 * R1_correctness + 0.35 * Opus_reasoning + 0.15 * Opus_completeness",
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
    """Print a summary table of collected v5 rewards."""
    print("\n" + "=" * 90)
    print("V5 DUAL-JUDGE REWARD SUMMARY (R1 correctness + Opus reasoning/completeness)")
    print("=" * 90)

    rows: List[Dict[str, Any]] = []
    for model_id in models:
        path = model_output_path(output_dir, model_id)
        if not path.exists():
            rows.append({"model": model_id, "n": 0})
            continue

        composites: List[float] = []
        r1_corrs: List[float] = []
        opus_reass: List[float] = []
        opus_comps: List[float] = []
        r1_tiers: Dict[str, int] = defaultdict(int)
        opus_tiers: Dict[str, int] = defaultdict(int)
        cached = 0

        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if not rec.get("ok"):
                        continue
                    composites.append(rec["raw_score"])
                    r1_corrs.append(rec.get("r1_correctness", 0))
                    opus_reass.append(rec.get("opus_reasoning", 0))
                    opus_comps.append(rec.get("opus_completeness", 0))
                    r1_tiers[rec.get("r1_tier", "Unknown")] += 1
                    opus_tiers[rec.get("opus_tier", "Unknown")] += 1
                    if rec.get("response_cached"):
                        cached += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        if not composites:
            rows.append({"model": model_id, "n": 0})
            continue

        arr = np.array(composites)
        rows.append({
            "model": model_id,
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "r1_corr": float(np.mean(r1_corrs)),
            "opus_reas": float(np.mean(opus_reass)),
            "opus_comp": float(np.mean(opus_comps)),
            "cached": cached,
            "r1_tiers": dict(r1_tiers),
            "opus_tiers": dict(opus_tiers),
        })

    print(
        f"\n  {'Model':<38s} {'N':>6s} {'Comp':>7s} {'Std':>6s} "
        f"{'R1_Cr':>6s} {'Op_Rs':>6s} {'Op_Cm':>6s} {'Cache%':>7s}"
    )
    print(f"  {'-' * 84}")

    for r in rows:
        if r["n"] == 0:
            print(f"  {r['model']:<38s} {'[no data]':>6s}")
            continue
        cache_pct = r["cached"] / r["n"] * 100
        print(
            f"  {r['model']:<38s} {r['n']:6d} {r['mean']:7.3f} "
            f"{r['std']:6.3f} {r['r1_corr']:6.3f} {r['opus_reas']:6.3f} "
            f"{r['opus_comp']:6.3f} {cache_pct:6.1f}%"
        )
        r1_tier_str = ", ".join(f"{k}={v}" for k, v in sorted(r["r1_tiers"].items()))
        opus_tier_str = ", ".join(f"{k}={v}" for k, v in sorted(r["opus_tiers"].items()))
        print(f"  {'':38s} R1 tiers: {r1_tier_str}")
        print(f"  {'':38s} Opus tiers: {opus_tier_str}")

    scored = [r for r in rows if r["n"] > 0]
    if len(scored) >= 2:
        print(f"\n  Pairwise gaps (composite):")
        scored.sort(key=lambda r: r["mean"])
        for i in range(len(scored) - 1):
            a, b = scored[i], scored[i + 1]
            delta = b["mean"] - a["mean"]
            short_a = a["model"].split("/")[-1]
            short_b = b["model"].split("/")[-1]
            print(f"    {short_a} -> {short_b}: {delta:+.4f}")
        total = scored[-1]["mean"] - scored[0]["mean"]
        print(f"    TOTAL SPREAD: {total:.4f}")

    print("=" * 90 + "\n")


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Model IDs to collect (default: 4-model set).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of prompts (for testing).",
    )
    parser.add_argument(
        "--workers", type=int, default=30,
        help="Parallel workers (default: 30).",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip collection, just print summary.",
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
    gen = V5DualJudgeGenerator(max_workers=args.workers)
    gen.load_all_caches(models)

    cache_report = []
    for mid in models:
        n_cached = sum(1 for p in prompts if (mid, p) in gen.response_cache)
        cache_report.append(f"  {mid}: {n_cached}/{len(prompts)} cached")
    logger.info("Response cache coverage:\n%s", "\n".join(cache_report))

    # 3. Resume support.
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
        "Tasks: %d remaining (%d prompts x %d models - %d completed)",
        total_tasks, len(prompts), len(models), total_completed,
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
    logger.info(
        "Starting dual-judge collection with %d workers "
        "(R1 correctness + Opus reasoning/completeness)...",
        args.workers,
    )
    diag_interval = max(50, total_tasks // 40)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(gen.process_task, t): t for t in tasks}
            done_count = 0
            t_start = time.time()

            with tqdm(total=total_tasks, desc="V5 Dual-Judge") as pbar:
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
                        elapsed = time.time() - t_start
                        rate = done_count / elapsed * 60
                        stats = gen.get_running_stats()
                        parts = []
                        for mid in models:
                            s = stats.get(mid)
                            if s:
                                short = mid.split("/")[-1]
                                parts.append(
                                    f"{short}: {s['mean']:.3f}±{s['std']:.3f} (n={s['n']})"
                                )
                        throttle_info = throttle.summary()
                        t_parts = [
                            f"{k}: {v['429s']}/{v['calls']} 429s"
                            for k, v in throttle_info.items()
                        ]
                        logger.info(
                            "[%.0f tasks/min] %s | Throttle: %s",
                            rate, " | ".join(parts), " | ".join(t_parts),
                        )
    finally:
        for f in out_files.values():
            f.close()

    # 8. Final summary.
    print_summary(OUTPUT_DIR, models)


if __name__ == "__main__":
    main()
