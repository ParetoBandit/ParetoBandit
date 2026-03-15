from __future__ import annotations

import os
import time

import numpy as np
import pytest

from bandit_gpt import BanditRouter
from tests.stress_utils import (
    DEFAULT_DIMENSION,
    make_context,
    precomputed_feature_service,
    synthetic_registry,
)


def _percentile_ms(values_ns: list[int], q: float) -> float:
    return float(np.percentile(np.asarray(values_ns, dtype=np.float64), q) / 1e6)


@pytest.mark.performance
@pytest.mark.slow
def test_route_update_latency_budget():
    if os.getenv("RUN_PERFORMANCE_TESTS", "0") != "1":
        pytest.skip("Set RUN_PERFORMANCE_TESTS=1 to run performance budget checks.")

    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
    )

    rounds = int(os.getenv("BANDITGPT_PERF_ROUNDS", "4000"))
    warmup = int(os.getenv("BANDITGPT_PERF_WARMUP", "400"))
    route_ns: list[int] = []
    update_ns: list[int] = []

    for i in range(rounds):
        x = make_context(seed=i)
        t0 = time.perf_counter_ns()
        model_id, _ = router.route(x)
        t1 = time.perf_counter_ns()
        router.update(model_id, x, reward=0.8 if i % 2 else 0.3)
        t2 = time.perf_counter_ns()
        if i >= warmup:
            route_ns.append(t1 - t0)
            update_ns.append(t2 - t1)

    route_p95 = _percentile_ms(route_ns, 95)
    update_p95 = _percentile_ms(update_ns, 95)

    # Conservative defaults; can be tightened per machine in CI vars.
    route_budget = float(os.getenv("BANDITGPT_ROUTE_P95_MS_BUDGET", "50"))
    update_budget = float(os.getenv("BANDITGPT_UPDATE_P95_MS_BUDGET", "20"))
    assert route_p95 <= route_budget, f"route p95 {route_p95:.3f}ms exceeds {route_budget}ms"
    assert update_p95 <= update_budget, f"update p95 {update_p95:.3f}ms exceeds {update_budget}ms"
