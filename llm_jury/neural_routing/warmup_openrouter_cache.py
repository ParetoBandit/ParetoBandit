#!/usr/bin/env python3
"""
Warmup / priming job: score *all* OpenRouter models in the local cache.

What this produces:
  - A JSONL log of (model_id, prompt_id, reward_raw, reward_logit, reward_z_online, ...)
  - A per-model summary (JSON) aggregating average rewards and error rates.

Why:
  - Initializes a historical baseline ("priming") so production starts warm.
  - Provides a stable, absolute-ish stretched KPI: reward_logit (safe-logit).

Usage:
  python -m llm_jury.neural_routing.warmup_openrouter_cache --max-models 10
  python -m llm_jury.neural_routing.warmup_openrouter_cache --all
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_model_cache(path: Path) -> List[Dict[str, Any]]:
    d = json.loads(path.read_text())
    models = d.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models_cache.json: expected key 'models' to be a list")
    return [m for m in models if isinstance(m, dict)]


def _extract_openrouter_ids(models: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for m in models:
        or_id = m.get("openrouter_id")
        if isinstance(or_id, str) and or_id.strip():
            ids.append(or_id.strip())
    # de-dup, preserve order
    seen = set()
    out: List[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def call_openrouter(model_id: str, prompt: str, *, max_tokens: int) -> Tuple[str, Optional[str]]:
    """
    Returns (content, error_string).
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "", "OPENROUTER_API_KEY not found in environment (.env)"

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    reasoning_patterns = ["/o1", "/o3", "/gpt-5", "/deepseek-r1"]
    is_reasoning = any(p in model_id.lower() for p in reasoning_patterns)

    try:
        if is_reasoning:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=int(max_tokens),
            )
        else:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(max_tokens),
            )
        content = (resp.choices[0].message.content or "").strip()
        return content, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def default_warmup_prompts() -> List[Tuple[str, str]]:
    """
    Small, diverse prompt set intended for priming / baseline measurement.
    Each entry is (prompt, tag).
    """
    return [
        ("What is the capital of France? Answer with one word.", "factoid_short"),
        ("A farmer has 17 sheep. All but 9 run away. How many are left? Explain.", "logic_riddle"),
        (
            "Write a Python function `is_palindrome(s)` that ignores non-alphanumerics and case. "
            "Include a docstring and 2 examples.",
            "coding_basic",
        ),
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets? "
            "Explain carefully.",
            "reasoning_word_problem",
        ),
        (
            "Explain why the sky appears blue during the day but red/orange at sunset. Be concise and accurate.",
            "science_explain",
        ),
        (
            "Summarize this in 1 sentence: The bandit learns quality while cost is handled at decision time.",
            "summarize_short",
        ),
    ]


@dataclass
class WarmupRow:
    ts_s: float
    model_id: str
    prompt_tag: str
    prompt: str
    max_tokens: int
    ok: bool
    error: Optional[str]
    response_len: int
    # Grader outputs
    p_correct_raw: Optional[float]
    reward_raw: Optional[float]
    reward_logit: Optional[float]
    reward_z_online: Optional[float]


def main() -> None:
    parser = argparse.ArgumentParser(description="Warmup scores for all cached OpenRouter models")
    parser.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "data" / "models_cache.json"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "data" / "warmup"))
    parser.add_argument("--grader", type=str, default=str(PROJECT_ROOT / "data" / "quality_predictor" / "best_quality_predictor.pt"))
    parser.add_argument("--max-models", type=int, default=10, help="Safety limit; use --all to score everything")
    parser.add_argument("--all", action="store_true", help="Score all cached models (can be expensive)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (rate-limit friendly)")
    parser.add_argument("--max-tokens-default", type=int, default=600)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)
    grader_path = Path(args.grader)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    if not grader_path.exists():
        raise FileNotFoundError(f"Grader checkpoint not found: {grader_path}")

    from llm_jury.neural_routing.quality_cost_predictor import QualityCostPredictor, RunningZScoreNormalizer

    print(f"Loading cache: {cache_path}")
    cached_models = _load_model_cache(cache_path)
    model_ids = _extract_openrouter_ids(cached_models)
    print(f"Found {len(model_ids)} unique OpenRouter ids in cache.")

    if not args.all:
        model_ids = model_ids[: max(0, int(args.max_models))]
        print(f"Scoring first {len(model_ids)} models (use --all for full cache).")
    else:
        print(f"Scoring ALL {len(model_ids)} models (this may take a long time and incur cost).")

    # Token overrides for known "empty-response at low budget" models.
    max_tokens_overrides = {
        "google/gemini-3-pro-preview": 4000,
        "deepseek/deepseek-r1-0528-qwen3-8b": 2000,
        "anthropic/claude-opus-4.5": 1200,
        "x-ai/grok-4": 1200,
        "openai/gpt-5.2": 1200,
    }

    prompts = default_warmup_prompts()

    print(f"Loading grader: {grader_path}")
    grader = QualityCostPredictor.load(grader_path)
    grader.eval()

    # Online z-score only (order-dependent): useful for sanity-checks and optional bandit priming.
    norm = RunningZScoreNormalizer(
        mean_init=0.65,
        std_init=0.05,
        alpha=0.01,
        clamp=3.0,
        auto_init_from_first_sample=True,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_jsonl = out_dir / f"openrouter_warmup_{run_id}.jsonl"
    out_summary = out_dir / f"openrouter_warmup_{run_id}.summary.json"
    out_norm = out_dir / f"openrouter_warmup_{run_id}.normalizer.json"

    per_model: Dict[str, List[WarmupRow]] = defaultdict(list)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, model_id in enumerate(model_ids):
            print(f"[{i+1}/{len(model_ids)}] {model_id}")
            for prompt, tag in prompts:
                max_tokens = int(max_tokens_overrides.get(model_id, args.max_tokens_default))
                content, err = call_openrouter(model_id, prompt, max_tokens=max_tokens)

                if (content is not None) and len(str(content).strip()) == 0 and err is None:
                    # Retry once with larger budget
                    retry_tokens = max(max_tokens * 2, 1200)
                    content, err = call_openrouter(model_id, prompt, max_tokens=retry_tokens)
                    max_tokens = retry_tokens

                ok = err is None
                prod = None
                if ok:
                    prod = grader.predict_production(prompt, content, reward_normalizer=norm)
                row = WarmupRow(
                    ts_s=time.time(),
                    model_id=model_id,
                    prompt_tag=tag,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    ok=ok,
                    error=err,
                    response_len=len(content or ""),
                    p_correct_raw=None if not ok else float(prod["p_correct_raw"]),
                    reward_raw=None if not ok else float(prod["reward_raw"]),
                    reward_logit=None if not ok else float(prod["reward_logit"]),
                    reward_z_online=None if not ok else (None if prod.get("reward_z") is None else float(prod["reward_z"])),
                )
                per_model[model_id].append(row)
                f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                f.flush()

                if args.sleep > 0:
                    time.sleep(float(args.sleep))

    # Aggregate summary
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "n_models": len(model_ids),
        "n_prompts": len(prompts),
        "rows": len(model_ids) * len(prompts),
        "models": {},
    }

    for model_id, rows in per_model.items():
        ok_rows = [r for r in rows if r.ok]
        err_rows = [r for r in rows if not r.ok]
        r_raw = np.array([r.reward_raw for r in ok_rows if r.reward_raw is not None], dtype=np.float64)
        r_logit = np.array([r.reward_logit for r in ok_rows if r.reward_logit is not None], dtype=np.float64)
        summary["models"][model_id] = {
            "n": len(rows),
            "n_ok": len(ok_rows),
            "n_err": len(err_rows),
            "err_rate": (len(err_rows) / len(rows)) if rows else None,
            "avg_reward_raw": float(np.mean(r_raw)) if r_raw.size else None,
            "avg_reward_logit": float(np.mean(r_logit)) if r_logit.size else None,
            "min_reward_logit": float(np.min(r_logit)) if r_logit.size else None,
            "max_reward_logit": float(np.max(r_logit)) if r_logit.size else None,
        }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_norm.write_text(json.dumps(norm.state_dict(), indent=2), encoding="utf-8")

    print(f"\nWrote JSONL: {out_jsonl}")
    print(f"Wrote summary: {out_summary}")
    print(f"Wrote normalizer: {out_norm}")


if __name__ == "__main__":
    main()

