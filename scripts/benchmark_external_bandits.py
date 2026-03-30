#!/usr/bin/env python3
"""Benchmark ParetoBandit against external contextual-bandit libraries.

Compares online routing performance (cumulative reward and latency) of
ParetoBandit's DisjointLinUCBPolicy against two third-party LinUCB
implementations from the ``bench`` extras:

1. **MABWiser** (``mabwiser.mab.MAB`` with ``LearningPolicy.LinUCB``)
2. **contextualbandits** (``contextualbandits.online.LinUCB``)

A synthetic contextual-bandit environment generates stochastic rewards
from per-arm linear weight vectors so cumulative reward reflects each
algorithm's ability to learn arm-context associations online.

Usage (matches the CI invocation in ``.github/workflows/performance.yml``):

    python scripts/benchmark_external_bandits.py \\
        --rounds 2000 --warmup-rounds 200 --strict \\
        --output-json benchmark-results/external_bandits.json \\
        --output-md  benchmark-results/external_bandits.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pareto_bandit import BanditRouter
from pareto_bandit.feature_service import FeatureService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for _noisy in (
    "pareto_bandit.router",
    "pareto_bandit.feature_service",
    "pareto_bandit.policy",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ======================================================================
# Synthetic environment
# ======================================================================

ARM_NAMES = ["budget-model", "mid-model", "frontier-model"]
CONTEXT_DIM = 16  # 15 features + bias

_REGISTRY: dict[str, dict[str, Any]] = {
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


def _make_contexts(n: int, dim: int, seed: int) -> np.ndarray:
    """Unit-norm context vectors with a trailing bias term."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim - 1))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    raw /= np.maximum(norms, 1e-12)
    bias = np.ones((n, 1), dtype=np.float64)
    return np.hstack([raw, bias])


@dataclass
class SyntheticEnv:
    """Contextual-bandit environment with linear mean rewards per arm.

    Attributes:
        theta_star: ``(K, dim)`` array of true weight vectors per arm.
        noise_std: Standard deviation of Gaussian reward noise.
    """

    theta_star: np.ndarray
    noise_std: float
    _rng: np.random.Generator

    @classmethod
    def create(cls, n_arms: int, dim: int, noise_std: float = 0.1, seed: int = 0) -> SyntheticEnv:
        rng = np.random.default_rng(seed)
        theta = rng.standard_normal((n_arms, dim)) * 0.3
        return cls(theta_star=theta, noise_std=noise_std, _rng=rng)

    def reward(self, arm_idx: int, context: np.ndarray) -> float:
        """Stochastic reward: clip(θ*_a · x + noise, 0, 1)."""
        mean = float(self.theta_star[arm_idx] @ context)
        noisy = mean + self._rng.normal(0, self.noise_std)
        return float(np.clip(noisy, 0.0, 1.0))

    def oracle_arm(self, context: np.ndarray) -> int:
        """Return the arm index with the highest expected reward."""
        means = self.theta_star @ context
        return int(np.argmax(means))


# ======================================================================
# Result container
# ======================================================================

@dataclass
class BenchmarkResult:
    """Aggregate metrics for a single algorithm run."""

    name: str
    cumulative_reward: float
    oracle_cumulative_reward: float
    regret: float
    regret_fraction: float
    select_p50_us: float
    select_p95_us: float
    update_p50_us: float
    update_p95_us: float
    total_p50_us: float
    total_p95_us: float
    throughput_rps: float
    rounds_measured: int


def _pct_us(values_ns: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), q) / 1e3)


# ======================================================================
# Algorithm wrappers (unified select / update interface)
# ======================================================================

class _ParetoBanditRunner:
    """Wrapper that drives ParetoBandit through the synthetic environment."""

    def __init__(self, dim: int, alpha: float = 0.1) -> None:
        fs = FeatureService.for_precomputed(dim)
        self.router = BanditRouter.create(
            model_registry=_REGISTRY,
            feature_service=fs,
            priors="none",
            alpha=alpha,
        )

    def select(self, context: np.ndarray) -> str:
        model_id, _ = self.router.route(context)
        return model_id

    def update(self, arm_name: str, context: np.ndarray, reward: float) -> None:
        self.router.update(arm_name, context, reward=reward)


class _MABWiserRunner:
    """Wrapper around MABWiser's LinUCB."""

    def __init__(self, arms: list[str], dim: int, alpha: float = 1.0) -> None:
        from mabwiser.mab import MAB, LearningPolicy

        self.mab = MAB(arms, LearningPolicy.LinUCB(alpha=alpha))
        self.arms = arms
        self._fitted = False
        self._dim = dim

    def select(self, context: np.ndarray) -> str:
        if not self._fitted:
            return self.arms[0]
        result = self.mab.predict(context.reshape(1, -1))
        return result if isinstance(result, str) else result[0]

    def update(self, arm_name: str, context: np.ndarray, reward: float) -> None:
        ctx = context.reshape(1, -1)
        if not self._fitted:
            self.mab.fit(
                decisions=[arm_name],
                rewards=[reward],
                contexts=ctx,
            )
            self._fitted = True
        else:
            self.mab.partial_fit(
                decisions=[arm_name],
                rewards=[reward],
                contexts=ctx,
            )


class _ContextualBanditsRunner:
    """Wrapper around contextualbandits.online.LinUCB."""

    def __init__(self, arms: list[str], dim: int, alpha: float = 1.0) -> None:
        from contextualbandits.online import LinUCB

        self.model = LinUCB(nchoices=len(arms), alpha=alpha)
        self.arms = arms
        self._arm_to_idx = {name: i for i, name in enumerate(arms)}
        self._fitted = False
        self._dim = dim

    def select(self, context: np.ndarray) -> str:
        if not self._fitted:
            return self.arms[0]
        idx = int(self.model.predict(context.reshape(1, -1))[0])
        return self.arms[idx]

    def update(self, arm_name: str, context: np.ndarray, reward: float) -> None:
        ctx = context.reshape(1, -1)
        a = np.array([self._arm_to_idx[arm_name]])
        r = np.array([reward])
        if not self._fitted:
            self.model.fit(ctx, a, r)
            self._fitted = True
        else:
            self.model.partial_fit(ctx, a, r)


# ======================================================================
# Benchmark loop
# ======================================================================

def run_benchmark(
    runner: _ParetoBanditRunner | _MABWiserRunner | _ContextualBanditsRunner,
    name: str,
    env: SyntheticEnv,
    contexts: np.ndarray,
    rounds: int,
    warmup: int,
) -> BenchmarkResult:
    """Run a single algorithm through the synthetic environment and collect metrics.

    Args:
        runner: Algorithm wrapper with ``select`` and ``update`` methods.
        name: Human-readable label.
        env: Synthetic contextual-bandit environment.
        contexts: Pre-generated ``(rounds, dim)`` context matrix.
        rounds: Total route+update cycles.
        warmup: Cycles excluded from latency statistics.

    Returns:
        Aggregated result with reward, regret, and latency metrics.
    """
    arm_name_to_idx = {a: i for i, a in enumerate(ARM_NAMES)}
    cumulative_reward = 0.0
    oracle_cumulative_reward = 0.0
    select_ns: list[int] = []
    update_ns: list[int] = []

    for i in range(rounds):
        ctx = contexts[i]

        t0 = time.perf_counter_ns()
        chosen_name = runner.select(ctx)
        t1 = time.perf_counter_ns()

        chosen_idx = arm_name_to_idx[chosen_name]
        r = env.reward(chosen_idx, ctx)
        cumulative_reward += r

        oracle_idx = env.oracle_arm(ctx)
        oracle_cumulative_reward += env.reward(oracle_idx, ctx)

        runner.update(chosen_name, ctx, r)
        t2 = time.perf_counter_ns()

        if i >= warmup:
            select_ns.append(t1 - t0)
            update_ns.append(t2 - t1)

    total_ns = [s + u for s, u in zip(select_ns, update_ns)]
    measured = rounds - warmup
    wall_s = sum(total_ns) / 1e9
    regret = oracle_cumulative_reward - cumulative_reward

    return BenchmarkResult(
        name=name,
        cumulative_reward=round(cumulative_reward, 4),
        oracle_cumulative_reward=round(oracle_cumulative_reward, 4),
        regret=round(regret, 4),
        regret_fraction=round(regret / max(oracle_cumulative_reward, 1e-9), 4),
        select_p50_us=_pct_us(select_ns, 50),
        select_p95_us=_pct_us(select_ns, 95),
        update_p50_us=_pct_us(update_ns, 50),
        update_p95_us=_pct_us(update_ns, 95),
        total_p50_us=_pct_us(total_ns, 50),
        total_p95_us=_pct_us(total_ns, 95),
        throughput_rps=round(measured / wall_s, 1) if wall_s > 0 else 0.0,
        rounds_measured=measured,
    )


# ======================================================================
# Output formatters
# ======================================================================

def _write_json(results: list[BenchmarkResult], meta: dict[str, Any], path: Path) -> None:
    payload = {
        **meta,
        "results": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("JSON report written to %s", path)


def _write_markdown(results: list[BenchmarkResult], meta: dict[str, Any], path: Path) -> None:
    lines = [
        "# External Bandit Benchmark",
        "",
        f"**Rounds:** {meta['rounds']}  |  "
        f"**Warmup:** {meta['warmup_rounds']}  |  "
        f"**Seed:** {meta['seed']}",
        "",
        "## Cumulative Reward & Regret",
        "",
        "| Algorithm | Cum. Reward | Oracle Reward | Regret | Regret % |",
        "|-----------|------------|---------------|--------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.cumulative_reward:.2f} | "
            f"{r.oracle_cumulative_reward:.2f} | {r.regret:.2f} | "
            f"{r.regret_fraction * 100:.1f}% |"
        )

    lines += [
        "",
        "## Latency (microseconds)",
        "",
        "| Algorithm | Select p50 | Select p95 | Update p50 | Update p95 | Throughput |",
        "|-----------|-----------|-----------|-----------|-----------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.select_p50_us:.1f} | {r.select_p95_us:.1f} | "
            f"{r.update_p50_us:.1f} | {r.update_p95_us:.1f} | "
            f"{r.throughput_rps:,.0f} req/s |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report written to %s", path)


# ======================================================================
# Strict-mode assertions
# ======================================================================

def _strict_checks(results: list[BenchmarkResult]) -> None:
    """Sanity checks that fail the CI step when ``--strict`` is set.

    Checks:
        1. ParetoBandit cumulative reward is positive.
        2. ParetoBandit regret fraction is below 60 % (very lenient; the
           real value is usually <30 % but CI runners are noisy).
        3. All algorithms complete without producing NaN metrics.
    """
    for r in results:
        for field in ("cumulative_reward", "regret", "select_p50_us", "update_p50_us"):
            val = getattr(r, field)
            if np.isnan(val) or np.isinf(val):
                raise AssertionError(f"[{r.name}] {field} is {val}")

    pb = next(r for r in results if r.name.startswith("ParetoBandit"))
    if pb.cumulative_reward <= 0:
        raise AssertionError(
            f"ParetoBandit cumulative reward is non-positive: {pb.cumulative_reward}"
        )
    if pb.regret_fraction > 0.60:
        raise AssertionError(
            f"ParetoBandit regret fraction {pb.regret_fraction:.2%} exceeds 60% budget"
        )

    logger.info("Strict checks passed.")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ParetoBandit against external contextual-bandit libraries.",
    )
    parser.add_argument("--rounds", type=int, default=2000,
                        help="Total route+update cycles per algorithm (default: 2000).")
    parser.add_argument("--warmup-rounds", type=int, default=200,
                        help="Warmup rounds excluded from latency stats (default: 200).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42).")
    parser.add_argument("--strict", action="store_true",
                        help="Enable strict sanity assertions (fail on anomalies).")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path for JSON results (e.g. benchmark-results/external_bandits.json).")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Path for Markdown results (e.g. benchmark-results/external_bandits.md).")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("External Bandit Benchmark")
    logger.info("=" * 70)
    logger.info(
        "  Rounds: %d  |  Warmup: %d  |  Seed: %d  |  Strict: %s",
        args.rounds, args.warmup_rounds, args.seed, args.strict,
    )

    contexts = _make_contexts(args.rounds, CONTEXT_DIM, seed=args.seed)
    env = SyntheticEnv.create(
        n_arms=len(ARM_NAMES), dim=CONTEXT_DIM, noise_std=0.1, seed=args.seed + 1000,
    )

    algorithms: list[tuple[str, Any]] = [
        ("ParetoBandit LinUCB", _ParetoBanditRunner(dim=CONTEXT_DIM, alpha=0.1)),
        ("MABWiser LinUCB", _MABWiserRunner(ARM_NAMES, dim=CONTEXT_DIM, alpha=1.0)),
        ("contextualbandits LinUCB", _ContextualBanditsRunner(ARM_NAMES, dim=CONTEXT_DIM, alpha=1.0)),
    ]

    results: list[BenchmarkResult] = []
    for label, runner in algorithms:
        env_copy = SyntheticEnv.create(
            n_arms=len(ARM_NAMES), dim=CONTEXT_DIM, noise_std=0.1, seed=args.seed + 1000,
        )
        logger.info("  Benchmarking: %s ...", label)
        t_start = time.perf_counter()
        result = run_benchmark(
            runner=runner,
            name=label,
            env=env_copy,
            contexts=contexts,
            rounds=args.rounds,
            warmup=args.warmup_rounds,
        )
        elapsed = time.perf_counter() - t_start
        results.append(result)
        logger.info(
            "    reward=%.2f  regret=%.2f (%.1f%%)  "
            "select_p50=%.1fus  update_p50=%.1fus  throughput=%s req/s  (%.1fs wall)",
            result.cumulative_reward,
            result.regret,
            result.regret_fraction * 100,
            result.select_p50_us,
            result.update_p50_us,
            f"{result.throughput_rps:,.0f}",
            elapsed,
        )

    meta = {
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "seed": args.seed,
        "context_dim": CONTEXT_DIM,
        "arms": ARM_NAMES,
    }

    if args.output_json:
        _write_json(results, meta, Path(args.output_json))
    if args.output_md:
        _write_markdown(results, meta, Path(args.output_md))

    if args.strict:
        _strict_checks(results)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
