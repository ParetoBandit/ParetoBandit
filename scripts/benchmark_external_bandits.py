#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bandit_gpt import BanditRouter, FeatureService


def synthetic_registry() -> dict[str, dict[str, Any]]:
    return {
        "cheap-fast/model-a": {
            "openrouter_id": "cheap-fast/model-a",
            "input_cost_per_m": 0.08,
            "output_cost_per_m": 0.18,
            "time_to_first_token_seconds": 0.15,
            "hle": 0.42,
        },
        "cheap-fast/model-b": {
            "openrouter_id": "cheap-fast/model-b",
            "input_cost_per_m": 0.12,
            "output_cost_per_m": 0.25,
            "time_to_first_token_seconds": 0.20,
            "hle": 0.51,
        },
        "mid/model-c": {
            "openrouter_id": "mid/model-c",
            "input_cost_per_m": 1.50,
            "output_cost_per_m": 3.00,
            "time_to_first_token_seconds": 0.45,
            "hle": 0.67,
        },
        "premium/model-d": {
            "openrouter_id": "premium/model-d",
            "input_cost_per_m": 7.00,
            "output_cost_per_m": 14.00,
            "time_to_first_token_seconds": 0.90,
            "hle": 0.80,
        },
    }


def make_context(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(size=dim - 1)
    v = v / (np.linalg.norm(v) + 1e-12)
    return np.append(v, 1.0).astype(np.float64)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Metrics:
    backend: str
    route_p50_ms: float
    route_p95_ms: float
    update_p50_ms: float
    update_p95_ms: float
    throughput_rps: float
    rounds_measured: int
    notes: str = ""


class BanditGPTAdapter:
    def __init__(self, arms: list[str], dim: int):
        self.arms = arms
        self.router = BanditRouter.create(
            model_registry=synthetic_registry(),
            feature_service=FeatureService.for_precomputed(dim),
            priors="none",
            use_corralling=False,
        )

    def route(self, x: np.ndarray) -> str:
        model, _ = self.router.route(x, profile="auto")
        return model

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        self.router.update(arm, x, reward)


class MabwiserAdapter:
    def __init__(self, arms: list[str], dim: int, seed: int):
        from mabwiser.mab import LearningPolicy, MAB

        self.arms = arms
        self.mab = MAB(
            arms=arms,
            learning_policy=LearningPolicy.LinUCB(alpha=1.0),
            seed=seed,
        )
        init_contexts = np.eye(len(arms), dim, dtype=np.float64)
        init_rewards = np.linspace(0.2, 0.8, len(arms))
        self.mab.fit(arms, init_rewards, contexts=init_contexts)

    def route(self, x: np.ndarray) -> str:
        return self.mab.predict(contexts=[x])[0]

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        self.mab.partial_fit([arm], [reward], contexts=[x])


class ContextualBanditsAdapter:
    def __init__(self, arms: list[str], dim: int, seed: int):
        from contextualbandits.online import LinUCB

        self.arms = arms
        self.arm_to_idx = {arm: i for i, arm in enumerate(arms)}
        self.idx_to_arm = {i: arm for i, arm in enumerate(arms)}

        try:
            self.model = LinUCB(
                nchoices=len(arms),
                alpha=1.0,
                random_state=seed,
            )
        except TypeError:
            self.model = LinUCB(
                nchoices=len(arms),
                random_state=seed,
            )

        init_x = np.eye(len(arms), dim, dtype=np.float64)
        init_a = np.arange(len(arms))
        init_r = np.linspace(0.2, 0.8, len(arms))
        self.model.fit(init_x, init_a, init_r)

    def route(self, x: np.ndarray) -> str:
        idx = int(self.model.predict(x.reshape(1, -1))[0])
        return self.idx_to_arm[idx]

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        a = np.array([self.arm_to_idx[arm]])
        r = np.array([reward], dtype=np.float64)
        self.model.partial_fit(x.reshape(1, -1), a, r)


def percentile_ms(values_ns: list[int], q: float) -> float:
    if not values_ns:
        return 0.0
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), q) / 1e6)


def benchmark_backend(
    name: str,
    adapter: Any,
    rounds: int,
    warmup_rounds: int,
    contexts: list[np.ndarray],
    rewards: list[dict[str, float]],
) -> Metrics:
    route_ns: list[int] = []
    update_ns: list[int] = []
    start_total = time.perf_counter_ns()

    for i in range(rounds):
        x = contexts[i]

        t0 = time.perf_counter_ns()
        arm = adapter.route(x)
        t1 = time.perf_counter_ns()
        adapter.update(arm, x, rewards[i][arm])
        t2 = time.perf_counter_ns()

        if i >= warmup_rounds:
            route_ns.append(t1 - t0)
            update_ns.append(t2 - t1)

    end_total = time.perf_counter_ns()
    measured = max(1, rounds - warmup_rounds)
    total_s = (end_total - start_total) / 1e9
    throughput = measured / total_s
    return Metrics(
        backend=name,
        route_p50_ms=percentile_ms(route_ns, 50),
        route_p95_ms=percentile_ms(route_ns, 95),
        update_p50_ms=percentile_ms(update_ns, 50),
        update_p95_ms=percentile_ms(update_ns, 95),
        throughput_rps=throughput,
        rounds_measured=measured,
    )


def write_markdown(
    output_path: Path,
    results: list[Metrics],
    tolerance_ratio: float,
    criterion_pass: bool | None,
) -> None:
    lines = [
        "# External Bandit Router Benchmark",
        "",
        f"- Tolerance ratio: `{tolerance_ratio:.2f}x`",
        "",
        "| Backend | Route p50 (ms) | Route p95 (ms) | Update p50 (ms) | Update p95 (ms) | Throughput (req/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.backend} | {row.route_p50_ms:.3f} | {row.route_p95_ms:.3f} | "
            f"{row.update_p50_ms:.3f} | {row.update_p95_ms:.3f} | {row.throughput_rps:.1f} |"
        )

    lines.extend(["", "## Gate Result"])
    if criterion_pass is None:
        lines.append("- Not evaluated (insufficient external backends).")
    else:
        lines.append(f"- Pass: `{criterion_pass}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BanditGPT against external routers.")
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--warmup-rounds", type=int, default=300)
    parser.add_argument("--dimension", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance-ratio", type=float, default=1.25)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when either external backend is unavailable.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmark-results/external_bandits.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("benchmark-results/external_bandits.md"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    registry = synthetic_registry()
    arms = list(registry.keys())
    dim = args.dimension

    # Fixed synthetic workload for apples-to-apples comparison.
    contexts = [make_context(rng, dim) for _ in range(args.rounds)]
    latent_thetas = {arm: rng.normal(size=dim) for arm in arms}
    rewards: list[dict[str, float]] = []
    for x in contexts:
        row = {}
        for arm in arms:
            score = float(np.dot(latent_thetas[arm], x))
            noisy = sigmoid(score) + float(rng.normal(0.0, 0.03))
            row[arm] = float(np.clip(noisy, 0.0, 1.0))
        rewards.append(row)

    adapters: list[tuple[str, Any]] = [("banditgpt", BanditGPTAdapter(arms, dim))]
    unavailable: dict[str, str] = {}

    for name, cls in [
        ("mabwiser_linucb", MabwiserAdapter),
        ("contextualbandits_linucb", ContextualBanditsAdapter),
    ]:
        try:
            adapters.append((name, cls(arms, dim, args.seed)))
        except Exception as exc:  # pragma: no cover - environment-dependent
            unavailable[name] = str(exc)

    if args.strict and unavailable:
        print(json.dumps({"status": "error", "unavailable_backends": unavailable}, indent=2))
        return 2

    metrics: list[Metrics] = []
    for name, adapter in adapters:
        metrics.append(
            benchmark_backend(
                name=name,
                adapter=adapter,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                contexts=contexts,
                rewards=rewards,
            )
        )

    by_name = {m.backend: m for m in metrics}
    bandit = by_name["banditgpt"]
    external = [m for m in metrics if m.backend != "banditgpt"]

    criterion_pass: bool | None
    if not external:
        criterion_pass = None
    else:
        ext_route_p95 = float(np.mean([m.route_p95_ms for m in external]))
        ext_update_p95 = float(np.mean([m.update_p95_ms for m in external]))
        criterion_pass = (
            bandit.route_p95_ms <= ext_route_p95 * args.tolerance_ratio
            and bandit.update_p95_ms <= ext_update_p95 * args.tolerance_ratio
        )

    payload = {
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "dimension": dim,
        "tolerance_ratio": args.tolerance_ratio,
        "criterion_pass": criterion_pass,
        "unavailable_backends": unavailable,
        "results": [m.__dict__ for m in metrics],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, metrics, args.tolerance_ratio, criterion_pass)

    print(f"Wrote JSON report: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_md}")
    if unavailable:
        print(f"Unavailable backends: {json.dumps(unavailable, indent=2)}")
    if criterion_pass is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
