#!/usr/bin/env python3
"""
CLI: given a prompt, print recommended models from the async-bandit router.

This is the "hot path" only:
  - No LLM calls
  - No grading
  - Just: prompt -> context embedding -> utility ranking across models

Prior Loading Priority (when --priors-mode=auto):
  1. USER priors (~/.llm_jury/priors/user_priors.npz) - if exists
  2. BUNDLED priors (<package>/data/priors/shippable_priors.npz) - if exists
  3. COLD START (no priors)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from llm_jury.async_bandit.bandit_router import (
    BanditRouter,
    build_registry_from_models_cache,
    build_cost_proportional_priors,
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BUNDLED_PRIORS = PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"
DEFAULT_USER_PRIORS = Path.home() / ".llm_jury" / "priors" / "user_priors.npz"


def main() -> int:
    p = argparse.ArgumentParser(description="Recommend OpenRouter models for a prompt (async bandit hot path)")
    p.add_argument("--prompt", type=str, required=True, help="User prompt text")
    p.add_argument("--top-k", type=int, default=10, help="Number of models to return")
    p.add_argument("--cache", type=str, default=str(PROJECT_ROOT / "data" / "models_cache.json"))
    p.add_argument("--state", type=str, default="", help="Optional router state JSON (BanditRouter.save_state)")
    p.add_argument("--priors", type=str, default="", help="Optional priors JSON (model_id -> prior score)")
    p.add_argument("--cost-prior-gamma", type=float, default=0.0, help="If >0, use cost-proportional priors: gamma*log(cost)")
    p.add_argument("--shippable-priors", type=str, default="", help="Explicit priors NPZ path (overrides --priors-mode)")
    p.add_argument("--priors-mode", type=str, default="merged", choices=["merged", "auto", "user", "bundled", "none"],
                   help="Prior loading: merged (bundled+user, recommended), auto (user->bundled), user, bundled, none")

    # Utility knobs
    p.add_argument("--lambda-cost", type=float, default=50.0, help="Cost penalty weight")
    p.add_argument("--lambda-latency", type=float, default=0.05, help="Latency penalty weight (per second)")
    p.add_argument("--max-latency-s", type=float, default=0.0, help="Optional hard latency cap (0 = disabled)")
    p.add_argument("--reward-mode", type=str, default="logit", choices=["logit", "z"])

    # Token assumptions for cost/latency estimation
    p.add_argument("--output-tokens", type=int, default=600, help="Expected output tokens for cost/latency math")
    p.add_argument("--input-tokens", type=int, default=0, help="Override input tokens (0 = estimate from prompt)")

    # Complexity gating (recommended)
    p.add_argument("--use-complexity-gating", action="store_true", help="Gate candidates using prompt complexity model")

    p.add_argument("--json", action="store_true", help="Output JSON instead of a table")

    args = p.parse_args()

    cache_path = Path(args.cache)
    registry = build_registry_from_models_cache(cache_path)

    # Load optional per-model priors (separate from shippable priors)
    model_priors_path = Path(args.priors) if str(args.priors).strip() else None
    model_priors = None
    if model_priors_path is not None and model_priors_path.exists():
        model_priors = json.loads(model_priors_path.read_text())
    elif float(args.cost_prior_gamma) > 0:
        model_priors = build_cost_proportional_priors(registry, gamma=float(args.cost_prior_gamma))

    # Determine which shippable priors to load
    explicit_shippable = Path(args.shippable_priors) if str(args.shippable_priors).strip() else None
    state_path = Path(args.state) if str(args.state).strip() else None

    # Priority: explicit state > explicit shippable > auto-detection
    priors_source = "none"
    if state_path is not None and state_path.exists():
        router = BanditRouter.load_state(state_path)
        router.registry = dict(registry)
        router.reward_mode = str(args.reward_mode)
        if model_priors is not None:
            router.model_priors = {str(k): float(v) for k, v in dict(model_priors).items()}
        priors_source = f"state:{state_path}"
    elif explicit_shippable is not None and explicit_shippable.exists():
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=explicit_shippable,
            model_registry=registry,
            reward_mode=str(args.reward_mode),
            alpha=0.5,
        )
        if model_priors is not None:
            router.model_priors = {str(k): float(v) for k, v in dict(model_priors).items()}
        priors_source = f"explicit:{explicit_shippable}"
    else:
        # Auto-detection based on --priors-mode
        router = BanditRouter.create(
            model_registry=registry,
            reward_mode=str(args.reward_mode),
            alpha=0.5,
            priors=str(args.priors_mode),
            user_priors_path=DEFAULT_USER_PRIORS,
            bundled_priors_path=DEFAULT_BUNDLED_PRIORS,
        )
        if model_priors is not None:
            router.model_priors = {str(k): float(v) for k, v in dict(model_priors).items()}
        priors_source = router.priors_source
        if router.priors_path:
            priors_source = f"{router.priors_source}:{router.priors_path}"

    max_latency = None if float(args.max_latency_s) <= 0 else float(args.max_latency_s)
    in_tok = None if int(args.input_tokens) <= 0 else int(args.input_tokens)

    rows = router.rank_prompt(
        args.prompt,
        top_k=int(args.top_k),
        lambda_cost=float(args.lambda_cost),
        lambda_latency=float(args.lambda_latency),
        max_latency_s=max_latency,
        input_tokens=in_tok,
        output_tokens=int(args.output_tokens),
        use_complexity_gating=bool(args.use_complexity_gating),
    )

    if bool(args.json):
        output = {"priors_source": priors_source, "recommendations": rows}
        print(json.dumps(output, indent=2))
        return 0

    # Pretty table
    print(f"\nPrompt: {args.prompt}")
    print(f"Priors: {priors_source}\n")
    print(f"{'rank':>4}  {'model_id':<45}  {'U':>8}  {'q_hat':>8}  {'prior':>8}  {'cost($)':>10}  {'lat(s)':>8}  display")
    print("-" * 115)
    for i, r in enumerate(rows, start=1):
        print(
            f"{i:>4}  "
            f"{str(r.get('model_id','')):<45}  "
            f"{float(r.get('utility', 0.0)):>8.3f}  "
            f"{float(r.get('quality_hat', 0.0)):>8.3f}  "
            f"{float(r.get('prior', 0.0)):>8.3f}  "
            f"{float(r.get('cost_usd', 0.0)):>10.6f}  "
            f"{float(r.get('latency_s', 0.0)):>8.2f}  "
            f"{str(r.get('display_name',''))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

