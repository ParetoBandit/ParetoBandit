#!/usr/bin/env python3
"""
Warmup priors (no benchmarks): grade real model outputs offline.

What it does:
  - Calls each OpenRouter model in `data/models_cache.json` on a small prompt set.
  - Grades each (prompt, response) with TieredGrader:
      - Soft grader always runs (local).
      - Teacher verifier runs only for "hard" prompts (truth check).
  - Aggregates per-model prior scores and writes:
      data/priors/openrouter_priors_<timestamp>.json

These priors are then consumed by the router to avoid cold-start "pick cheapest" behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from banditgpt.async_bandit.demo_quality_grader import call_openrouter
from banditgpt.async_bandit.quality_cost_predictor import QualityCostPredictor
from banditgpt.async_bandit.tiered_grader import TieredGrader, OpenRouterTeacherVerifier
from banditgpt._resources import get_models_cache_path, get_priors_path, get_quality_predictor_path


def _load_models(cache_path: Path) -> List[str]:
    d = json.loads(cache_path.read_text())
    models = d.get("models", [])
    out: List[str] = []
    for m in models:
        oid = (m or {}).get("openrouter_id")
        if isinstance(oid, str) and oid.strip():
            out.append(oid.strip())
    # de-dup, preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def default_warmup_prompts() -> List[Tuple[str, str]]:
    """
    Small prompt set for priors. Tags are for analysis only.
    We include a mix of soft (style) and hard (truth/constraints) prompts.
    """
    return [
        ("Write a haiku about winter.", "soft_creative"),
        ("Summarize in 1 sentence: The router predicts quality; cost is a lookup at decision time.", "soft_summarize"),
        ("Calculate 2 + 2. Answer with just the number.", "hard_math_easy"),
        ("Calculate the pH of a $10^{-8}$ M solution of HCl.", "hard_chem_ph"),
        ("Solve: If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "hard_reasoning"),
        ("Return valid JSON with keys exactly: {\"a\": 1, \"b\": 2}. Output JSON only.", "hard_json"),
        ("Write Python code: define a function add(a,b) that returns a+b. Output code only.", "hard_code"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Warmup priors for all models in cache (no benchmarks).")
    ap.add_argument("--cache", type=str, default=str(get_models_cache_path()))
    ap.add_argument("--out-dir", type=str, default=str(get_priors_path("").parent))
    ap.add_argument("--grader", type=str, default=str(get_quality_predictor_path()))
    ap.add_argument("--max-models", type=int, default=0, help="0 = all models, otherwise limit for quick runs")
    ap.add_argument("--teacher-model", type=str, default="openai/gpt-4o")
    ap.add_argument("--teacher-max-tokens", type=int, default=64)
    ap.add_argument("--max-tokens-default", type=int, default=800)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    load_dotenv()  # Load from current dir or parents

    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grader_path = Path(args.grader)

    model_ids = _load_models(cache_path)
    if int(args.max_models) > 0:
        model_ids = model_ids[: int(args.max_models)]
    prompts = default_warmup_prompts()

    if not grader_path.exists():
        raise FileNotFoundError(f"Soft grader checkpoint not found: {grader_path}")

    soft = QualityCostPredictor.load(grader_path)
    soft.eval()
    teacher = OpenRouterTeacherVerifier(model_id=str(args.teacher_model), max_tokens=int(args.teacher_max_tokens))
    grader = TieredGrader(soft_grader=soft, teacher_verifier=teacher)

    # Token overrides to avoid empty responses for known models
    # Gemini 3.0+ and reasoning models often need higher limits to generate responses
    overrides = {
        # Gemini 3.0 models - require ~4000 tokens to generate responses
        "google/gemini-3-pro-preview": 4000,
        # Gemini 2.5 models - also benefit from higher limits
        "google/gemini-2.5-pro": 4000,
        "google/gemini-2.5-pro-preview-06-05": 4000,
        "google/gemini-2.5-flash-preview-09-2025": 4000,
        "google/gemini-2.5-flash-lite": 2000,
        # Reasoning models
        "deepseek/deepseek-r1-0528-qwen3-8b": 2000,
        # Other frontier models
        "anthropic/claude-opus-4.5": 1200,
        "x-ai/grok-4": 1200,
        "openai/gpt-5.2": 1200,
    }

    per_model_scores: Dict[str, List[float]] = defaultdict(list)
    per_model_counts: Dict[str, int] = defaultdict(int)
    per_model_meta: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"ok": 0, "errors": 0, "teacher_used": 0})

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_json = out_dir / f"openrouter_priors_{run_id}.json"

    for i, mid in enumerate(model_ids, start=1):
        print(f"[{i}/{len(model_ids)}] {mid}")
        for prompt, tag in prompts:
            max_tokens = int(overrides.get(mid, int(args.max_tokens_default)))
            resp = call_openrouter(mid, prompt, max_tokens=max_tokens, timeout_s=60.0)
            if resp is not None and len(str(resp).strip()) == 0:
                resp = call_openrouter(mid, prompt, max_tokens=max(1200, max_tokens * 2), timeout_s=60.0)
            if isinstance(resp, str) and resp.startswith("[ERROR"):
                per_model_meta[mid]["errors"] += 1
                continue

            prod = grader.predict_production(prompt, resp, reward_normalizer=None)
            # Use stationary stretched KPI as prior (logit of clipped p_correct).
            score = float(prod.get("reward_logit", 0.0))
            per_model_scores[mid].append(score)
            per_model_counts[mid] += 1
            per_model_meta[mid]["ok"] += 1
            if bool(prod.get("tiered_used_teacher")):
                per_model_meta[mid]["teacher_used"] += 1

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

    priors: Dict[str, Any] = {
        "run_id": run_id,
        "teacher_model": str(args.teacher_model),
        "teacher_max_tokens": int(args.teacher_max_tokens),
        "prompt_set": [{"tag": t, "prompt": p} for p, t in prompts],
        "priors": {},
    }

    for mid in model_ids:
        xs = per_model_scores.get(mid, [])
        if not xs:
            priors["priors"][mid] = {"prior": 0.0, "n": 0, **per_model_meta[mid]}
            continue
        arr = np.asarray(xs, dtype=np.float64)
        priors["priors"][mid] = {
            "prior": float(np.mean(arr)),
            "n": int(arr.size),
            "std": float(np.std(arr) + 1e-9),
            **per_model_meta[mid],
        }

    out_json.write_text(json.dumps(priors, indent=2))
    print(f"\nWrote priors: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

