from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from bandit_gpt import BanditRouter, RouterConfig
from bandit_gpt.storage import EphemeralContextStore
from tests.stress_utils import (
    DEFAULT_DIMENSION,
    make_context,
    precomputed_feature_service,
    synthetic_registry,
)


@pytest.mark.stress
def test_feedback_loop_handles_delayed_and_out_of_order_updates():
    rng = random.Random(123)
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
        config=RouterConfig(max_log_size=32),
        context_store=EphemeralContextStore(max_size=5000),
    )

    request_ids: list[str] = []
    selected_models: list[str] = []
    for i in range(600):
        context = make_context(seed=i)
        kwargs = {}
        if i % 3 == 0:
            kwargs["max_cost"] = 0.002
            kwargs["input_tokens"] = 160
            kwargs["output_tokens"] = 120
        elif i % 3 == 1:
            kwargs["max_latency"] = 0.25
        else:
            kwargs["quality_floor"] = {"hle": 0.6}
        model_id, log = router.route(context, **kwargs)
        request_ids.append(log.request_id)
        selected_models.append(model_id)

    # Verify constraints were honored at route-time.
    for i, model_id in enumerate(selected_models):
        meta = router.registry[model_id]
        if i % 3 == 0:
            assert meta["input_cost_per_m"] <= 0.12
        elif i % 3 == 1:
            assert meta["time_to_first_token_seconds"] <= 0.25
        else:
            assert meta.get("hle", 0.0) >= 0.6

    shuffled = request_ids[:]
    rng.shuffle(shuffled)
    for idx, req_id in enumerate(shuffled):
        reward = 1.0 if idx % 5 else -0.25  # includes out-of-range values for clamp path
        router.process_feedback(req_id, reward)

    # Sanity: updates happened and internal matrices changed.
    assert router.bandit.t > 0
    norms = [float(np.linalg.norm(router.bandit.b[m])) for m in router.bandit.models]
    assert any(n > 0.0 for n in norms)


@pytest.mark.stress
def test_feedback_loop_thread_contention_does_not_corrupt_state():
    router = BanditRouter.create(
        model_registry=synthetic_registry(),
        feature_service=precomputed_feature_service(DEFAULT_DIMENSION),
        priors="none",
        context_store=EphemeralContextStore(max_size=10000),
    )
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(worker_id: int) -> int:
        local_updates = 0
        try:
            for step in range(150):
                context = make_context(seed=worker_id * 10_000 + step)
                model, log = router.route(context)
                reward = 0.9 if step % 4 else 0.2
                router.process_feedback(log.request_id, reward)
                # Also exercise direct update path.
                router.update(model, context, reward, weight=1.0)
                local_updates += 2
            return local_updates
        except Exception as exc:  # pragma: no cover - captured for assertion below
            with lock:
                errors.append(exc)
            return 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(worker, range(8)))

    assert not errors, f"Concurrent route/feedback loop raised: {errors}"
    assert sum(counts) > 0
    assert router.bandit.t >= 8 * 150
