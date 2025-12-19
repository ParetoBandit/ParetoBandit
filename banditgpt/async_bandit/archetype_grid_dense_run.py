#!/usr/bin/env python3
"""
Archetype Grid dense run: run all models on K golden prompts and export shippable priors.

This is step B+C of the "Archetype Grid" strategy:
  - Read `data/priors/archetype_grid_prompts.jsonl` (K prompts).
  - Run ALL cached OpenRouter models on each prompt (dense K x M matrix).
  - Grade with TieredGrader (teacher only for "hard" prompts).
  - Update a SharedCovarianceLinUCBPolicy offline.
  - Write `data/priors/shippable_priors.npz`.

Supports --resume by appending JSONL logs and skipping already-scored (prompt_id, model_id) pairs.
Supports --workers for parallel API calls (default 10).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from banditgpt.async_bandit.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    SharedCovarianceLinUCBPolicy,
    build_registry_from_models_cache,
    l2_normalize,
)
from banditgpt.async_bandit.demo_quality_grader import call_openrouter
from banditgpt.async_bandit.quality_cost_predictor import QualityCostPredictor
from banditgpt.async_bandit.tiered_grader import TieredGrader, OpenRouterTeacherVerifier
from banditgpt._resources import get_models_cache_path, get_priors_path, get_quality_predictor_path


def _load_models(cache_path: Path) -> List[str]:
    d = json.loads(cache_path.read_text())
    out: List[str] = []
    for m in d.get("models", []):
        oid = (m or {}).get("openrouter_id")
        if isinstance(oid, str) and oid.strip():
            out.append(oid.strip())
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _load_archetype_prompts(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = int(d.get("cluster_id", len(rows)))
            p = str(d.get("prompt", "") or "").strip()
            if p:
                rows.append((cid, p))
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Dense archetype grid run -> shippable priors")
    ap.add_argument("--cache", type=str, default=str(get_models_cache_path()))
    ap.add_argument("--grid", type=str, default=str(get_priors_path("archetype_grid_prompts.jsonl")))
    ap.add_argument("--grader", type=str, default=str(get_quality_predictor_path()))
    ap.add_argument("--out", type=str, default=str(get_priors_path("shippable_priors.npz")))
    ap.add_argument("--log", type=str, default=str(get_priors_path("archetype_grid_dense_run.jsonl")))
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--context-model", type=str, default=DEFAULT_CONTEXT_MODEL)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--reward-mode", type=str, default="logit", choices=["logit"])  # keep stationary KPI for priors

    ap.add_argument("--use-teacher", action="store_true")
    ap.add_argument("--teacher-model", type=str, default="openai/gpt-4o")
    ap.add_argument("--teacher-max-tokens", type=int, default=64)

    ap.add_argument("--max-tokens-default", type=int, default=800)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit-prompts", type=int, default=0)
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10, help="Number of parallel API workers")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_dotenv()  # Load from current dir or parents

    cache_path = Path(args.cache)
    grid_path = Path(args.grid)
    log_path = Path(args.log)
    out_path = Path(args.out)

    models = _load_models(cache_path)
    if int(args.limit_models) > 0:
        models = models[: int(args.limit_models)]
    prompts = _load_archetype_prompts(grid_path)
    if int(args.limit_prompts) > 0:
        prompts = prompts[: int(args.limit_prompts)]

    print(f"Loaded {len(models)} models, {len(prompts)} prompts", flush=True)
    print(f"Using {args.workers} parallel workers", flush=True)

    # Build encoder (same as router)
    from sentence_transformers import SentenceTransformer  # type: ignore

    print("Loading embedding model...", flush=True)
    enc = SentenceTransformer(str(args.context_model))

    # Load graders
    print("Loading graders...", flush=True)
    soft_path = Path(args.grader)
    soft = QualityCostPredictor.load(soft_path)
    soft.eval()
    teacher = OpenRouterTeacherVerifier(model_id=str(args.teacher_model), max_tokens=int(args.teacher_max_tokens)) if bool(args.use_teacher) else None
    grader = TieredGrader(soft_grader=soft, teacher_verifier=teacher)

    # Shared-covariance bandit prior builder
    policy = SharedCovarianceLinUCBPolicy(models, dim=384, alpha=float(args.alpha), ridge_lambda=1.0, recompute_inv_every=50)

    # Max token overrides for models that need larger context windows
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

    done: set = set()
    if bool(args.resume) and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add((int(d["cluster_id"]), str(d["model_id"])))
                except Exception:
                    continue
        print(f"Resume enabled: found {len(done)} completed pairs in {log_path}", flush=True)

    # Thread-safe file writer
    log_lock = threading.Lock()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = log_path.open("a", encoding="utf-8")

    # Grader lock (TieredGrader may not be fully thread-safe with PyTorch)
    grader_lock = threading.Lock()

    def process_one(cid: int, mid: str, prompt: str, x: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process a single (prompt, model) pair. Returns result dict or None if skipped."""
        key = (int(cid), str(mid))
        if key in done:
            return None  # Already done

        max_tokens = int(overrides.get(mid, int(args.max_tokens_default)))
        resp = call_openrouter(mid, prompt, max_tokens=max_tokens, timeout_s=90.0)
        if resp is not None and len(str(resp).strip()) == 0:
            resp = call_openrouter(mid, prompt, max_tokens=max(1200, max_tokens * 2), timeout_s=90.0)
        ok = not (isinstance(resp, str) and resp.startswith("[ERROR"))

        reward_logit = None
        teacher_used = False
        if ok:
            with grader_lock:
                prod = grader.predict_production(prompt, resp, reward_normalizer=None)
            reward_logit = float(prod.get("reward_logit", 0.0))
            teacher_used = bool(prod.get("tiered_used_teacher", False))

        result = {
            "cluster_id": int(cid),
            "model_id": str(mid),
            "ok": bool(ok),
            "teacher_used": bool(teacher_used),
            "reward_logit": reward_logit,
            "ts": time.time(),
            "_x": x,  # pass context for policy update
        }
        return result

    n_total = len(prompts) * len(models)
    n_done = len(done)
    t0 = time.time()

    print(f"Starting dense run: {n_total} total pairs, {n_done} already done", flush=True)

    try:
        for (cid, prompt) in prompts:
            x = enc.encode(prompt, normalize_embeddings=True)
            x = l2_normalize(np.asarray(x, dtype=np.float64))

            # Filter models we still need to do for this prompt
            models_todo = [mid for mid in models if (int(cid), str(mid)) not in done]
            if not models_todo:
                # All models done for this prompt
                n_done += len(models)
                continue

            # Submit all models for this prompt in parallel
            results: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = {
                    executor.submit(process_one, cid, mid, prompt, x): mid
                    for mid in models_todo
                }
                for future in as_completed(futures):
                    mid = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"  [ERROR] {mid}: {e}", flush=True)

            # Write results and update policy (sequential for thread safety)
            for result in results:
                mid = result["model_id"]
                x_ctx = result.pop("_x")

                # Write to log
                with log_lock:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()

                # Update policy
                if result["ok"] and result["reward_logit"] is not None:
                    policy.update(mid, x_ctx, float(result["reward_logit"]))

                done.add((int(result["cluster_id"]), str(mid)))
                n_done += 1

            # Progress per prompt
            dt = max(time.time() - t0, 1e-6)
            rate = n_done / dt
            eta_s = (n_total - n_done) / max(rate, 0.001)
            eta_h = eta_s / 3600
            print(f"[cluster {cid}] {n_done}/{n_total} ({100*n_done/n_total:.1f}%) rate={rate:.2f}/s ETA={eta_h:.1f}h", flush=True)

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

    finally:
        fout.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    policy.to_shippable_priors_npz(out_path, dtype=np.float16)
    print(f"Wrote shippable priors: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
