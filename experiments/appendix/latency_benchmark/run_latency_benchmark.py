#!/usr/bin/env python3
"""Appendix: Routing Latency Microbenchmark.

Measures wall-clock latency of route() and update() operations across eight
configurations designed to isolate:

  (a) **Sherman-Morrison vs. full inversion** — controlled comparison at the
      same level of abstraction (Bare SM vs. Cached Inv., sharing identical
      ``route()`` code via :class:`_CachedRouteLinUCBBase`).
  (b) **Production overhead** — ParetoBandit (locks, budget pacing, forgetting,
      etc.) vs. Bare SM (pure algorithm).
  (c) **PCA dimensionality reduction** — d=26 vs. d=385 within each group.
  (d) **Worst-case caching strategy** — Per-Route Inv. (never caches inverse).

Configurations
--------------
Production:
  1. ParetoBandit (d=26)        — Full production router (SM + locks + pacing).
  2. ParetoBandit (d=385)       — Same at raw dimension.

Algorithmic isolation (same base class, identical ``route()``):
  3. Bare SM (d=26)             — O(d^2) Sherman-Morrison update.
  4. Bare SM (d=385)            — Same at raw dimension.
  5. Cached Inv. (d=26)         — O(d^3) full ``np.linalg.inv`` on update.
  6. Cached Inv. (d=385)        — Same at raw dimension.

Worst-case baseline:
  7. Per-Route Inv. (d=26)      — O(K d^3) full inv on every route call.
  8. Per-Route Inv. (d=385)     — Same at raw dimension.

Each configuration is benchmarked over ``--rounds`` route+update cycles
(default 5,000) with a ``--warmup`` prefix excluded from timing (default 500).
Context vectors are pre-generated with a fixed seed for reproducibility.

Usage:
    python experiments/appendix/latency_benchmark/run_latency_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit import BanditRouter
from pareto_bandit.feature_service import FeatureService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in ("pareto_bandit.router", "pareto_bandit.feature_service", "pareto_bandit.policy"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).parent / "results"


# ======================================================================
# Synthetic workload
# ======================================================================

def _synthetic_registry() -> Dict[str, Dict[str, Any]]:
    """K=3 registry matching the paper's portfolio cost tiers."""
    return {
        "budget-model": {
            "model_id": "budget-model",
            "input_cost_per_m": 0.06,
            "output_cost_per_m": 0.06,
        },
        "mid-model": {
            "model_id": "mid-model",
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0,
        },
        "frontier-model": {
            "model_id": "frontier-model",
            "input_cost_per_m": 7.0,
            "output_cost_per_m": 21.0,
        },
    }


def _make_contexts(
    n: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate ``n`` whitened unit-norm context vectors with bias term.

    Returns:
        Array of shape ``(n, dim)`` where ``dim`` includes the bias.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim - 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    raw /= np.maximum(norms, 1e-12)
    bias = np.ones((n, 1), dtype=np.float64)
    return np.hstack([raw, bias])


def _make_rewards(n: int, seed: int = 42) -> np.ndarray:
    """Uniform rewards in [0.3, 0.9] for benchmarking."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.3, 0.9, size=n)


# ======================================================================
# Bare-bones LinUCB baselines (same abstraction level for fair comparison)
# ======================================================================

class _CachedRouteLinUCBBase:
    """Shared base for LinUCB routers that cache ``A_inv`` and ``theta``.

    All subclasses share **identical** ``__init__`` and ``route()``
    implementations; only ``update()`` varies.  This guarantees that any
    route-latency difference between subclasses is attributable to
    measurement noise, not code differences.
    """

    def __init__(self, models: List[str], dim: int, alpha: float = 0.1):
        self.models = models
        self.dim = dim
        self.alpha = alpha
        init_lambda = 1.0
        self.A: Dict[str, np.ndarray] = {
            m: np.eye(dim) * init_lambda for m in models
        }
        self.A_inv: Dict[str, np.ndarray] = {
            m: np.eye(dim) / init_lambda for m in models
        }
        self.b: Dict[str, np.ndarray] = {
            m: np.zeros(dim, dtype=np.float64) for m in models
        }
        self.theta: Dict[str, np.ndarray] = {
            m: np.zeros(dim, dtype=np.float64) for m in models
        }

    def route(self, x: np.ndarray) -> str:
        """UCB arm selection using cached ``theta`` and ``A_inv``."""
        best_model, best_score = "", -np.inf
        for m in self.models:
            mean = float(self.theta[m].dot(x))
            var = float(x.dot(self.A_inv[m]).dot(x))
            ucb = mean + self.alpha * np.sqrt(max(var, 0.0))
            if ucb > best_score:
                best_score = ucb
                best_model = m
        return best_model

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        raise NotImplementedError


class BareSMRouter(_CachedRouteLinUCBBase):
    """Bare-bones LinUCB with O(d^2) Sherman-Morrison rank-1 update.

    Structurally identical to :class:`NaiveCachedInvRouter` except that
    ``update()`` applies the rank-1 SM correction instead of a full
    ``np.linalg.inv``.  Comparing this against ``NaiveCachedInvRouter``
    isolates the algorithmic benefit of SM without production overhead
    confounds (locks, budget pacing, forgetting, etc.).
    """

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x
        A_inv_x = self.A_inv[model] @ x
        denom = 1.0 + float(x.dot(A_inv_x))
        self.A_inv[model] -= np.outer(A_inv_x, A_inv_x) / denom
        self.theta[model] = self.A_inv[model] @ self.b[model]


class NaiveCachedInvRouter(_CachedRouteLinUCBBase):
    """Bare-bones LinUCB with O(d^3) full inversion on each update.

    The natural "naive but reasonable" baseline: the inverse is cached
    after each update so ``route()`` uses the same O(d^2) dot-product
    scoring as :class:`BareSMRouter`.  Only ``update()`` differs.
    """

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x
        self.A_inv[model] = np.linalg.inv(self.A[model])
        self.theta[model] = self.A_inv[model] @ self.b[model]


class NaivePerRouteInvRouter:
    """LinUCB that recomputes ``np.linalg.inv(A_a)`` on every route call.

    Worst-case baseline: the inverse is never cached, so ``route()`` pays
    K full O(d^3) inversions while ``update()`` only accumulates the
    sufficient statistics.  Included to bound worst-case behaviour.
    """

    def __init__(self, models: List[str], dim: int, alpha: float = 0.1):
        self.models = models
        self.dim = dim
        self.alpha = alpha
        init_lambda = 1.0
        self.A: Dict[str, np.ndarray] = {
            m: np.eye(dim) * init_lambda for m in models
        }
        self.b: Dict[str, np.ndarray] = {
            m: np.zeros(dim, dtype=np.float64) for m in models
        }

    def route(self, x: np.ndarray) -> str:
        best_model, best_score = "", -np.inf
        for m in self.models:
            A_inv = np.linalg.inv(self.A[m])
            theta = A_inv @ self.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(A_inv).dot(x))
            ucb = mean + self.alpha * np.sqrt(max(var, 0.0))
            if ucb > best_score:
                best_score = ucb
                best_model = m
        return best_model

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x


# ======================================================================
# Benchmark harness
# ======================================================================

@dataclass
class LatencyResult:
    """Timing results for a single configuration."""

    name: str
    dimension: int
    route_p50_us: float
    route_p95_us: float
    route_p99_us: float
    update_p50_us: float
    update_p95_us: float
    update_p99_us: float
    total_p50_us: float
    total_p95_us: float
    throughput_rps: float
    rounds_measured: int


def _percentile_us(values_ns: List[int], q: float) -> float:
    """Convert nanosecond timings to microsecond percentile."""
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), q) / 1e3)


def benchmark_paretobandit(
    dim: int,
    rounds: int,
    warmup: int,
    contexts: np.ndarray,
    rewards: np.ndarray,
) -> LatencyResult:
    """Benchmark ParetoBandit (Sherman-Morrison) at a given dimension."""
    registry = _synthetic_registry()
    fs = FeatureService.for_precomputed(dim)
    router = BanditRouter.create(
        model_registry=registry,
        feature_service=fs,
        priors="none",
        use_corralling=False,
    )

    route_ns: List[int] = []
    update_ns: List[int] = []

    for i in range(rounds):
        x = contexts[i]
        t0 = time.perf_counter_ns()
        model_id, _ = router.route(x)
        t1 = time.perf_counter_ns()
        router.update(model_id, x, reward=float(rewards[i]))
        t2 = time.perf_counter_ns()
        if i >= warmup:
            route_ns.append(t1 - t0)
            update_ns.append(t2 - t1)

    total_ns = [r + u for r, u in zip(route_ns, update_ns)]
    measured = rounds - warmup
    wall_s = sum(total_ns) / 1e9
    return LatencyResult(
        name=f"ParetoBandit (d={dim})",
        dimension=dim,
        route_p50_us=_percentile_us(route_ns, 50),
        route_p95_us=_percentile_us(route_ns, 95),
        route_p99_us=_percentile_us(route_ns, 99),
        update_p50_us=_percentile_us(update_ns, 50),
        update_p95_us=_percentile_us(update_ns, 95),
        update_p99_us=_percentile_us(update_ns, 99),
        total_p50_us=_percentile_us(total_ns, 50),
        total_p95_us=_percentile_us(total_ns, 95),
        throughput_rps=measured / wall_s if wall_s > 0 else 0.0,
        rounds_measured=measured,
    )


def _benchmark_bare_router(
    router_cls: type,
    name: str,
    dim: int,
    rounds: int,
    warmup: int,
    contexts: np.ndarray,
    rewards: np.ndarray,
) -> LatencyResult:
    """Benchmark any bare router class with the standard timing loop.

    Args:
        router_cls: One of :class:`BareSMRouter`, :class:`NaiveCachedInvRouter`,
            or :class:`NaivePerRouteInvRouter`.
        name: Human-readable label stored in the result (e.g. ``"Bare SM (d=26)"``).
        dim: Context-vector dimension (including bias).
        rounds: Total route+update cycles.
        warmup: Cycles to exclude from timing.
        contexts: Pre-generated context matrix ``(rounds, dim)``.
        rewards: Pre-generated reward vector ``(rounds,)``.

    Returns:
        :class:`LatencyResult` with timing percentiles and throughput.
    """
    registry = _synthetic_registry()
    router = router_cls(list(registry.keys()), dim)

    route_ns: List[int] = []
    update_ns: List[int] = []

    for i in range(rounds):
        x = contexts[i]
        t0 = time.perf_counter_ns()
        model_id = router.route(x)
        t1 = time.perf_counter_ns()
        router.update(model_id, x, reward=float(rewards[i]))
        t2 = time.perf_counter_ns()
        if i >= warmup:
            route_ns.append(t1 - t0)
            update_ns.append(t2 - t1)

    total_ns = [r + u for r, u in zip(route_ns, update_ns)]
    measured = rounds - warmup
    wall_s = sum(total_ns) / 1e9
    return LatencyResult(
        name=name,
        dimension=dim,
        route_p50_us=_percentile_us(route_ns, 50),
        route_p95_us=_percentile_us(route_ns, 95),
        route_p99_us=_percentile_us(route_ns, 99),
        update_p50_us=_percentile_us(update_ns, 50),
        update_p95_us=_percentile_us(update_ns, 95),
        update_p99_us=_percentile_us(update_ns, 99),
        total_p50_us=_percentile_us(total_ns, 50),
        total_p95_us=_percentile_us(total_ns, 95),
        throughput_rps=measured / wall_s if wall_s > 0 else 0.0,
        rounds_measured=measured,
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency microbenchmark for ParetoBandit routing operations.",
    )
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DIM_PCA = 26    # 25 PCA components + bias
    DIM_RAW = 385   # 384-D raw embedding + bias

    logger.info("=" * 70)
    logger.info("Latency Microbenchmark: ParetoBandit Routing Operations")
    logger.info("=" * 70)
    logger.info(f"  Rounds: {args.rounds}  |  Warmup: {args.warmup}  |  Seed: {args.seed}")
    logger.info(f"  Dimensions: PCA={DIM_PCA}, Raw={DIM_RAW}")

    # Pre-generate contexts at both dimensions
    ctx_pca = _make_contexts(args.rounds, DIM_PCA, seed=args.seed)
    ctx_raw = _make_contexts(args.rounds, DIM_RAW, seed=args.seed)
    rewards = _make_rewards(args.rounds, seed=args.seed)

    results: List[LatencyResult] = []

    # (label, benchmark_fn_or_class, dim, contexts)
    # Production configs use benchmark_paretobandit; bare configs use a class.
    bare_configs: List[tuple] = [
        ("Bare SM (d=26)", BareSMRouter, DIM_PCA, ctx_pca),
        ("Bare SM (d=385)", BareSMRouter, DIM_RAW, ctx_raw),
        ("Cached Inv. (d=26)", NaiveCachedInvRouter, DIM_PCA, ctx_pca),
        ("Cached Inv. (d=385)", NaiveCachedInvRouter, DIM_RAW, ctx_raw),
        ("Per-Route Inv. (d=26)", NaivePerRouteInvRouter, DIM_PCA, ctx_pca),
        ("Per-Route Inv. (d=385)", NaivePerRouteInvRouter, DIM_RAW, ctx_raw),
    ]

    # --- Production configs ---
    for label, dim, ctx in [
        ("ParetoBandit (d=26)", DIM_PCA, ctx_pca),
        ("ParetoBandit (d=385)", DIM_RAW, ctx_raw),
    ]:
        logger.info(f"\n  Benchmarking: {label} ...")
        t_start = time.perf_counter()
        result = benchmark_paretobandit(
            dim=dim, rounds=args.rounds, warmup=args.warmup,
            contexts=ctx, rewards=rewards,
        )
        elapsed = time.perf_counter() - t_start
        results.append(result)
        logger.info(
            f"    route p50={result.route_p50_us:.1f}us  "
            f"p95={result.route_p95_us:.1f}us  |  "
            f"update p50={result.update_p50_us:.1f}us  "
            f"p95={result.update_p95_us:.1f}us  |  "
            f"throughput={result.throughput_rps:,.0f} req/s  "
            f"({elapsed:.1f}s wall)"
        )

    # --- Bare algorithm configs ---
    for label, router_cls, dim, ctx in bare_configs:
        logger.info(f"\n  Benchmarking: {label} ...")
        t_start = time.perf_counter()
        result = _benchmark_bare_router(
            router_cls=router_cls, name=label,
            dim=dim, rounds=args.rounds, warmup=args.warmup,
            contexts=ctx, rewards=rewards,
        )
        elapsed = time.perf_counter() - t_start
        results.append(result)
        logger.info(
            f"    route p50={result.route_p50_us:.1f}us  "
            f"p95={result.route_p95_us:.1f}us  |  "
            f"update p50={result.update_p50_us:.1f}us  "
            f"p95={result.update_p95_us:.1f}us  |  "
            f"throughput={result.throughput_rps:,.0f} req/s  "
            f"({elapsed:.1f}s wall)"
        )

    prod_pca = results[0]       # ParetoBandit d=26
    prod_raw = results[1]       # ParetoBandit d=385
    bare_sm_pca = results[2]    # Bare SM d=26
    bare_sm_raw = results[3]    # Bare SM d=385
    cached_pca = results[4]     # Cached Inv d=26
    cached_raw = results[5]     # Cached Inv d=385
    perroute_pca = results[6]   # Per-Route Inv d=26
    perroute_raw = results[7]   # Per-Route Inv d=385

    speedups = {
        # --- Algorithmic isolation: Bare SM vs. Cached Inv. (same route code) ---
        "algo_sm_vs_cached_d26_route": cached_pca.route_p50_us / max(bare_sm_pca.route_p50_us, 1e-3),
        "algo_sm_vs_cached_d26_update": cached_pca.update_p50_us / max(bare_sm_pca.update_p50_us, 1e-3),
        "algo_sm_vs_cached_d26_throughput": bare_sm_pca.throughput_rps / max(cached_pca.throughput_rps, 1),
        "algo_sm_vs_cached_d385_route": cached_raw.route_p50_us / max(bare_sm_raw.route_p50_us, 1e-3),
        "algo_sm_vs_cached_d385_update": cached_raw.update_p50_us / max(bare_sm_raw.update_p50_us, 1e-3),
        "algo_sm_vs_cached_d385_throughput": bare_sm_raw.throughput_rps / max(cached_raw.throughput_rps, 1),
        # --- Production overhead: ParetoBandit vs. Bare SM ---
        "prod_overhead_d26_route": prod_pca.route_p50_us / max(bare_sm_pca.route_p50_us, 1e-3),
        "prod_overhead_d26_update": prod_pca.update_p50_us / max(bare_sm_pca.update_p50_us, 1e-3),
        "prod_overhead_d385_route": prod_raw.route_p50_us / max(bare_sm_raw.route_p50_us, 1e-3),
        "prod_overhead_d385_update": prod_raw.update_p50_us / max(bare_sm_raw.update_p50_us, 1e-3),
        # --- PCA dimensionality reduction (within production configs) ---
        "pca_vs_raw_prod_throughput": prod_pca.throughput_rps / max(prod_raw.throughput_rps, 1),
    }

    logger.info("\n" + "=" * 70)
    logger.info("Speedup Ratios")
    logger.info("=" * 70)
    for k, v in speedups.items():
        logger.info(f"  {k}: {v:.1f}x")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "rounds": args.rounds,
        "warmup": args.warmup,
        "seed": args.seed,
        "results": [asdict(r) for r in results],
        "speedups": speedups,
    }
    out_path = RESULTS_DIR / "latency_benchmark_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
