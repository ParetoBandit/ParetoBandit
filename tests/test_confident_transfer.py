"""
Test: Model Admission — No Confidence Inheritance

Validates that dynamically registered models get A = λI (fresh uncertainty)
regardless of how mature existing models are.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from pareto_bandit.router import BanditRouter


def test_new_model_gets_fresh_precision():
    """New model gets A = λI even when existing models have large eigenvalues."""

    registry = {
        "gpt-4-base": {
            "display_name": "GPT-4 Base Model for Complex Reasoning",
            "model_id": "gpt-4-base",
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "initial_quality": 0.85
        }
    }

    router = BanditRouter(
        model_registry=registry,
        alpha=0.1,
        init_lambda=1.0,
    )

    model_id = "gpt-4-base"

    for i in range(100):
        context = router._get_context_vector(f"test prompt {i}")
        reward = 0.8 + 0.1 * np.random.rand()
        router.bandit.A[model_id] += np.outer(context, context)
        router.bandit.b[model_id] += reward * context

    router.bandit.refresh_inverse_cache()

    eigenvalues_before = np.linalg.eigvalsh(router.bandit.A[model_id])
    max_eigenvalue_before = eigenvalues_before.max()
    assert max_eigenvalue_before > 10 * router.bandit.init_lambda, \
        "Precondition: mature model should have large eigenvalues"

    router.register_model(
        "gpt-4-turbo",
        speed="slow",
        capabilities=["reasoning", "coding"],
        cost_usd=8.0,
        latency_s=1.5,
    )

    A_new = router.bandit.A["gpt-4-turbo"]
    eigenvalues_new = np.linalg.eigvalsh(A_new)
    max_eigenvalue_new = eigenvalues_new.max()

    is_A_fresh = abs(max_eigenvalue_new - router.bandit.init_lambda) < 0.1
    assert is_A_fresh, (
        f"New model A should have max eigenvalue ≈ init_lambda "
        f"({router.bandit.init_lambda}), got {max_eigenvalue_new:.2f}"
    )

    b_new = router.bandit.b["gpt-4-turbo"]
    has_prior = np.linalg.norm(b_new) > 1e-6
    assert has_prior, (
        "New model should have non-zero b from T-shirt prior (speed='slow' "
        "sets slow_bias=0.05)"
    )


if __name__ == "__main__":
    success = test_new_model_gets_fresh_precision()
    sys.exit(0)
