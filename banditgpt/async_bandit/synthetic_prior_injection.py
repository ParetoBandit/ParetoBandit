#!/usr/bin/env python3
"""
Synthetic Prior Injection (offline bandit pre-warm).

Goal:
  Avoid cold-start "all models equal => pick cheapest" without using benchmarks.

Method:
  1) Load a public proxy dataset of prompts (LMSYS / Magpie / HelpSteer2 style).
  2) Sample N prompts (e.g., 500).
  3) For each prompt, run a set of models (all cached models or a subset).
  4) Grade each (prompt, response) with TieredGrader:
       - soft grader always runs (local)
       - teacher verifier runs only on "hard" prompts (truth/constraints)
  5) Update the disjoint LinUCB policy offline:
       bandit.update(model_id, x(prompt), reward)
  6) Save router state (bandit + normalizer + registry) to JSON for deployment.

Notes:
  - This script can be expensive if you run many models * many prompts.
  - Start small: --max-models 10 --max-prompts 50, then scale.
  - To get priors for *all* models without 81x cost per prompt, use
    `--models-per-prompt K` to evaluate only K models per prompt but ensure
    full coverage via round-robin scheduling.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from banditgpt.async_bandit.bandit_router import BanditRouter, build_registry_from_models_cache, l2_normalize
from banditgpt.async_bandit.demo_quality_grader import call_openrouter
from banditgpt.async_bandit.quality_cost_predictor import QualityCostPredictor
from banditgpt.async_bandit.tiered_grader import TieredGrader, OpenRouterTeacherVerifier


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _load_model_ids(cache_path: Path) -> List[str]:
    d = json.loads(cache_path.read_text())
    models = d.get("models", [])
    ids: List[str] = []
    for m in models:
        oid = (m or {}).get("openrouter_id")
        if isinstance(oid, str) and oid.strip():
            ids.append(oid.strip())
    # de-dup, preserve order
    seen = set()
    out: List[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_prompt(ex: Any) -> Optional[str]:
    """
    Best-effort prompt extraction across common dataset schemas.
    """
    if not isinstance(ex, dict):
        return None

    # 1) direct fields
    for k in ("prompt", "question", "instruction", "input", "query", "text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 2) messages: list[dict(role, content)]
    # LMSYS Chatbot Arena: conversation_a / conversation_b
    msgs = (
        ex.get("messages")
        or ex.get("conversation")
        or ex.get("conversations")
        or ex.get("conversation_a")
        or ex.get("conversation_b")
    )
    if isinstance(msgs, list) and msgs:
        # messages as dicts
        if all(isinstance(m, dict) for m in msgs):
            for m in msgs:
                role = str(m.get("role", "") or m.get("from", "") or "").lower()
                content = m.get("content") or m.get("value") or m.get("text")
                if role in ("user", "human") and isinstance(content, str) and content.strip():
                    return content.strip()
            # fallback: first content-ish string
            for m in msgs:
                content = m.get("content") or m.get("value") or m.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        # messages as strings
        if all(isinstance(m, str) for m in msgs):
            s = str(msgs[0]).strip()
            return s if s else None
    return None


def _load_proxy_prompts(
    *,
    dataset_name: str,
    split: str,
    max_prompts: int,
    seed: int,
) -> List[str]:
    """
    Load prompts from a HF dataset via `datasets`.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: datasets. Install with: pip install datasets") from e

    ds = load_dataset(dataset_name, split=split)
    # Shuffle deterministically
    rng = random.Random(int(seed))
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    out: List[str] = []
    for i in idxs:
        ex = ds[int(i)]
        p = _extract_prompt(ex)
        if p:
            out.append(p)
        if len(out) >= int(max_prompts):
            break
    if not out:
        raise ValueError(f"No prompts extracted from dataset={dataset_name} split={split}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline synthetic prior injection for BanditRouter")
    ap.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "data" / "models_cache.json"))
    ap.add_argument("--out-state", type=str, default=str(PROJECT_ROOT / "data" / "router_state_synthetic.json"))
    ap.add_argument("--grader", type=str, default=str(PROJECT_ROOT / "data" / "quality_predictor" / "best_quality_predictor.pt"))

    ap.add_argument("--dataset", type=str, default="lmsys/chatbot_arena_conversations", help="HF dataset name")
    ap.add_argument("--split", type=str, default="train", help="HF dataset split")
    ap.add_argument("--max-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-models", type=int, default=0, help="0 = all models, otherwise limit for cost control")
    ap.add_argument("--models", type=str, default="", help="Optional comma-separated OpenRouter ids to use")
    ap.add_argument(
        "--models-per-prompt",
        type=int,
        default=0,
        help="If >0, only evaluate K models per prompt (round-robin across all selected models)",
    )

    ap.add_argument("--reward-mode", type=str, default="logit", choices=["logit", "z"])
    ap.add_argument("--alpha", type=float, default=0.5, help="LinUCB alpha (uncertainty weight)")

    # Teacher verifier (recommended for hard prompts)
    ap.add_argument("--use-teacher", action="store_true")
    ap.add_argument("--teacher-model", type=str, default="openai/gpt-4o")
    ap.add_argument("--teacher-max-tokens", type=int, default=64)

    # OpenRouter call settings
    ap.add_argument("--max-tokens-default", type=int, default=800)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=25, help="Save state every N prompts (0 = only at end)")
    args = ap.parse_args()

    # Reduce tokenizer fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    load_dotenv(PROJECT_ROOT / ".env")

    cache_path = Path(args.cache)
    registry = build_registry_from_models_cache(cache_path)
    all_model_ids = _load_model_ids(cache_path)

    chosen: List[str]
    if str(args.models).strip():
        chosen = [m.strip() for m in str(args.models).split(",") if m.strip()]
    else:
        chosen = list(all_model_ids)
    if int(args.max_models) > 0:
        chosen = chosen[: int(args.max_models)]
    # Ensure models are in registry
    chosen = [m for m in chosen if m in registry]
    if not chosen:
        raise ValueError("No models selected (check --models or models_cache.json).")

    prompts = _load_proxy_prompts(
        dataset_name=str(args.dataset),
        split=str(args.split),
        max_prompts=int(args.max_prompts),
        seed=int(args.seed),
    )
    print(f"Loaded proxy prompts: {len(prompts)} from {args.dataset}:{args.split}", flush=True)

    grader_path = Path(args.grader)
    if not grader_path.exists():
        raise FileNotFoundError(f"Soft grader checkpoint not found: {grader_path}")

    soft = QualityCostPredictor.load(grader_path)
    soft.eval()
    print(f"Loaded soft grader: {grader_path}", flush=True)

    teacher = None
    if bool(args.use_teacher):
        teacher = OpenRouterTeacherVerifier(model_id=str(args.teacher_model), max_tokens=int(args.teacher_max_tokens))
    grader = TieredGrader(soft_grader=soft, teacher_verifier=teacher)

    print("Loading router embedding model (SentenceTransformer)...", flush=True)
    router = BanditRouter(model_registry=registry, reward_mode=str(args.reward_mode), alpha=float(args.alpha))
    print("Router initialized.", flush=True)
    k = int(args.models_per_prompt)
    effective_k = len(chosen) if (k <= 0 or k >= len(chosen)) else k
    print(
        f"Selected models: {len(chosen)} (models_per_prompt={effective_k}) | reward_mode={args.reward_mode} | alpha={args.alpha} | teacher={bool(args.use_teacher)}",
        flush=True,
    )

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

    # Offline update loop
    n_calls = 0
    n_scored = 0
    rr = 0  # round-robin pointer
    for pi, prompt in enumerate(prompts, start=1):
        if pi == 1:
            print("Starting offline updates (first prompt)...", flush=True)
        x = router.encoder.encode(prompt)
        x = l2_normalize(np.asarray(x, dtype=np.float64))

        # Choose which models to evaluate for this prompt.
        # If k <= 0 or k >= #models, evaluate all.
        if k <= 0 or k >= len(chosen):
            to_eval = chosen
        else:
            to_eval = []
            for _ in range(k):
                to_eval.append(chosen[rr % len(chosen)])
                rr += 1

        for mi, mid in enumerate(to_eval, start=1):
            max_tokens = int(overrides.get(mid, int(args.max_tokens_default)))
            resp = call_openrouter(mid, prompt, max_tokens=max_tokens, timeout_s=60.0)
            n_calls += 1
            if resp is not None and len(str(resp).strip()) == 0:
                resp = call_openrouter(mid, prompt, max_tokens=max(1200, max_tokens * 2), timeout_s=60.0)
                n_calls += 1

            if isinstance(resp, str) and resp.startswith("[ERROR"):
                if float(args.sleep) > 0:
                    time.sleep(float(args.sleep))
                continue

            prod = grader.predict_production(prompt, resp, reward_normalizer=router.normalizer if args.reward_mode == "z" else None)
            if str(args.reward_mode) == "logit":
                r = float(prod.get("reward_logit", 0.0))
            else:
                rz = prod.get("reward_z")
                r = float(rz) if rz is not None else float(prod.get("reward_raw", 0.0))

            router.bandit.update(mid, x, float(r))
            n_scored += 1

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

        if pi % 10 == 0:
            print(f"[prompts {pi}/{len(prompts)}] calls={n_calls} scored={n_scored} (k={k if k>0 else len(chosen)})")

        if int(args.save_every) > 0 and (pi % int(args.save_every) == 0):
            out_state = Path(args.out_state)
            router.save_state(out_state)
            print(f"  ✓ checkpoint saved: {out_state}")

    out_state = Path(args.out_state)
    router.save_state(out_state)
    print(f"\nSaved warmed router state: {out_state}")
    print(f"Total calls: {n_calls} | Scored: {n_scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

