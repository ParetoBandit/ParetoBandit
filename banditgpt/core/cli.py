#!/usr/bin/env python3
"""
Unified CLI for the async bandit router.

Commands:
    recommend           - Get model recommendations for a prompt
    compress-priors     - Compress warmed state to shippable priors
    archetype-cluster   - Cluster prompts into archetypes
    archetype-dense-run - Run all models on archetype prompts
    warmup              - Generate per-model priors on curated prompts
    synthetic-inject    - Pre-warm bandit on a proxy dataset

Usage:
    python -m banditgpt.core.cli recommend --prompt "Hello world"
    python -m banditgpt.core.cli compress-priors --state router_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from banditgpt._resources import (
    get_bundled_priors_path,
    get_user_priors_path,
    get_models_cache_path,
    get_package_data_dir,
    get_priors_path,
)
from banditgpt.core.prior_manifest import (
    PriorIntegrityError,
    load_priors_manifest,
    verify_bundled_prior,
)

DEFAULT_BUNDLED_PRIORS = get_bundled_priors_path()
DEFAULT_USER_PRIORS = get_user_priors_path()


# ---------------------------------------------------------------------------
# Command: recommend
# ---------------------------------------------------------------------------


def cmd_recommend(args: argparse.Namespace) -> int:
    """Get model recommendations for a prompt."""
    from banditgpt.core.bandit_router import (
        BanditRouter,
        build_cost_proportional_priors,
        build_registry_from_models_cache,
    )

    cache_path = Path(args.cache)
    registry = build_registry_from_models_cache(cache_path)

    # Load optional per-model priors
    model_priors = None
    if args.priors and Path(args.priors).exists():
        model_priors = json.loads(Path(args.priors).read_text())
    elif args.cost_prior_gamma > 0:
        model_priors = build_cost_proportional_priors(registry, gamma=args.cost_prior_gamma)

    # Load router
    explicit_shippable = Path(args.shippable_priors) if args.shippable_priors else None
    state_path = Path(args.state) if args.state else None

    if state_path and state_path.exists():
        router = BanditRouter.load_state(state_path)
        router.registry = dict(registry)
        router.reward_mode = args.reward_mode
        if model_priors:
            router.model_priors = {str(k): float(v) for k, v in model_priors.items()}
        priors_source = f"state:{state_path}"
    elif explicit_shippable and explicit_shippable.exists():
        router = BanditRouter.load_from_shippable_priors(
            priors_npz=explicit_shippable,
            model_registry=registry,
            reward_mode=args.reward_mode,
            alpha=0.5,
            prior_strength=50.0,  # Expert distillation confidence boost
        )
        if model_priors:
            router.model_priors = {str(k): float(v) for k, v in model_priors.items()}
        priors_source = f"explicit:{explicit_shippable}"
    else:
        router = BanditRouter.create(
            model_registry=registry,
            reward_mode=args.reward_mode,
            alpha=0.5,
            priors=args.priors_mode,
            user_priors_path=DEFAULT_USER_PRIORS,
            bundled_priors_path=DEFAULT_BUNDLED_PRIORS,
        )
        if model_priors:
            router.model_priors = {str(k): float(v) for k, v in model_priors.items()}
        priors_source = router.priors_source
        if router.priors_path:
            priors_source = f"{router.priors_source}:{router.priors_path}"

    max_latency = args.max_latency_s if args.max_latency_s > 0 else None
    in_tok = args.input_tokens if args.input_tokens > 0 else None

    rows = router.rank_prompt(
        args.prompt,
        top_k=args.top_k,
        profile=args.profile,
        exploration=args.exploration,
        lambda_cost=args.lambda_cost,
        lambda_latency=args.lambda_latency,
        max_latency_s=max_latency,
        input_tokens=in_tok,
        output_tokens=args.output_tokens,
        use_complexity_gating=args.use_complexity_gating,
    )

    if args.json:
        print(json.dumps({"priors_source": priors_source, "recommendations": rows}, indent=2))
        return 0

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


def add_recommend_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the recommend command."""
    parser.add_argument("--prompt", type=str, required=True, help="User prompt text")
    parser.add_argument("--top-k", type=int, default=10, help="Number of models to return")
    parser.add_argument("--cache", type=str, default=str(get_models_cache_path()))
    parser.add_argument("--state", type=str, default="", help="Optional router state JSON")
    parser.add_argument("--priors", type=str, default="", help="Optional priors JSON")
    parser.add_argument("--cost-prior-gamma", type=float, default=0.0)
    parser.add_argument("--shippable-priors", type=str, default="")
    parser.add_argument(
        "--priors-mode",
        type=str,
        default="merged",
        choices=["merged", "auto", "user", "bundled", "none"],
    )
    # Optimization profile (user-friendly)
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["quality_first", "balanced", "cost_saver", "low_latency"],
        help="Optimization preset: quality_first, balanced, cost_saver, low_latency",
    )
    # Exploration rate (controls risk appetite)
    parser.add_argument(
        "--exploration",
        type=str,
        default=None,
        help="Exploration rate: static (0), safe (0.1), balanced (0.5), aggressive (2.0)",
    )
    # Raw weights (power users)
    parser.add_argument("--lambda-cost", type=float, default=None, help="Cost penalty weight (overrides profile)")
    parser.add_argument("--lambda-latency", type=float, default=None, help="Latency penalty weight (overrides profile)")
    parser.add_argument("--max-latency-s", type=float, default=0.0)
    parser.add_argument("--reward-mode", type=str, default="logit", choices=["logit", "z"])
    parser.add_argument("--output-tokens", type=int, default=600)
    parser.add_argument("--input-tokens", type=int, default=0)
    parser.add_argument("--use-complexity-gating", action="store_true")
    parser.add_argument("--json", action="store_true")


# ---------------------------------------------------------------------------
# Command: compress-priors
# ---------------------------------------------------------------------------


def cmd_compress_priors(args: argparse.Namespace) -> int:
    """Compress warmed state to shippable priors."""
    from banditgpt.core.bandit_router import BanditRouter, build_registry_from_models_cache

    state_path = Path(args.state)
    cache_path = Path(args.cache)
    out_path = Path(args.out)

    registry = build_registry_from_models_cache(cache_path)
    router = BanditRouter.load_state(state_path)
    router.registry = dict(registry)

    router.save_shippable_priors(out_path, dtype=np.float16)
    print(f"Wrote shippable priors: {out_path}")
    return 0


def add_compress_priors_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the compress-priors command."""
    parser.add_argument("--state", type=str, required=True, help="Path to router_state_*.json")
    parser.add_argument("--cache", type=str, default=str(get_models_cache_path()))
    parser.add_argument("--out", type=str, default=str(get_priors_path("shippable_priors.npz")))


# ---------------------------------------------------------------------------
# Command: archetype-cluster
# ---------------------------------------------------------------------------


def cmd_archetype_cluster(args: argparse.Namespace) -> int:
    """Cluster prompts from a dataset into representative archetypes."""
    # Import here to avoid heavy dependencies at module load
    from banditgpt.core.archetype_grid import main as archetype_main

    # Re-parse args for the original script
    sys.argv = [
        "archetype_grid",
        "--dataset", args.dataset,
        "--split", args.split,
        "--max-prompts", str(args.max_prompts),
        "--k", str(args.k),
        "--out", args.out,
    ]
    return archetype_main()


def add_archetype_cluster_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for archetype-cluster command."""
    parser.add_argument("--dataset", type=str, default="lmsys/chatbot_arena_conversations")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-prompts", type=int, default=50000)
    parser.add_argument("--k", type=int, default=500, help="Number of clusters")
    parser.add_argument("--out", type=str, default=str(get_priors_path("archetype_grid_prompts.jsonl")))


# ---------------------------------------------------------------------------
# Command: verify-priors
# ---------------------------------------------------------------------------


def cmd_verify_priors(args: argparse.Namespace) -> int:
    """Validate bundled priors (and optional user priors) with checksums."""
    manifest = load_priors_manifest()
    ok = True

    for entry in manifest.files:
        try:
            verify_bundled_prior(entry.bundled_path, manifest=manifest)
            print(f"[ok] {entry.name}")
        except PriorIntegrityError as exc:
            ok = False
            print(f"[fail] {entry.name}: {exc}")

    if args.check_user:
        user_path = Path(args.user_priors)
        if user_path.exists():
            try:
                np.load(user_path)
                print(f"[ok] user priors: {user_path}")
            except Exception as exc:  # pragma: no cover - corruption path
                ok = False
                print(f"[fail] user priors: {user_path} is not readable ({exc})")
        else:
            print(f"[warn] user priors not found at {user_path}")

    if ok:
        print("Priors verified successfully.")
        return 0

    print("One or more priors failed integrity checks.")
    return 1


def add_verify_priors_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--check-user",
        action="store_true",
        help="Also verify user priors if present (loadable NPZ).",
    )
    parser.add_argument(
        "--user-priors",
        type=str,
        default=str(DEFAULT_USER_PRIORS),
        help="Path to user priors (default: ~/.banditgpt/priors/user_priors.npz)",
    )


# ---------------------------------------------------------------------------
# Command: archetype-dense-run
# ---------------------------------------------------------------------------


def cmd_archetype_dense_run(args: argparse.Namespace) -> int:
    """Run all models on archetype prompts and build priors."""
    from banditgpt.core.archetype_grid_dense_run import main as dense_run_main

    argv = [
        "archetype_grid_dense_run",
        "--grid", args.grid,
        "--out", args.out,
        "--log", args.log,
        "--workers", str(args.workers),
    ]
    if args.use_teacher:
        argv.append("--use-teacher")
        argv.extend(["--teacher-model", args.teacher_model])
    if args.resume:
        argv.append("--resume")

    sys.argv = argv
    return dense_run_main()


def add_archetype_dense_run_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for archetype-dense-run command."""
    parser.add_argument("--grid", type=str, default=str(get_priors_path("archetype_grid_prompts.jsonl")))
    parser.add_argument("--out", type=str, default=str(get_priors_path("shippable_priors.npz")))
    parser.add_argument("--log", type=str, default=str(get_priors_path("archetype_grid_dense_run.jsonl")))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--use-teacher", action="store_true")
    parser.add_argument("--teacher-model", type=str, default="openai/gpt-4o")
    parser.add_argument("--resume", action="store_true")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="banditgpt.core.cli",
        description="Async bandit router CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # recommend
    recommend_parser = subparsers.add_parser("recommend", help="Get model recommendations")
    add_recommend_args(recommend_parser)
    recommend_parser.set_defaults(func=cmd_recommend)

    # compress-priors
    compress_parser = subparsers.add_parser("compress-priors", help="Compress state to priors")
    add_compress_priors_args(compress_parser)
    compress_parser.set_defaults(func=cmd_compress_priors)

    # archetype-cluster
    cluster_parser = subparsers.add_parser("archetype-cluster", help="Cluster prompts")
    add_archetype_cluster_args(cluster_parser)
    cluster_parser.set_defaults(func=cmd_archetype_cluster)

    # archetype-dense-run
    dense_parser = subparsers.add_parser("archetype-dense-run", help="Dense run on archetypes")
    add_archetype_dense_run_args(dense_parser)
    dense_parser.set_defaults(func=cmd_archetype_dense_run)

    # verify-priors
    verify_parser = subparsers.add_parser("verify-priors", help="Validate bundled priors checksums")
    add_verify_priors_args(verify_parser)
    verify_parser.set_defaults(func=cmd_verify_priors)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
