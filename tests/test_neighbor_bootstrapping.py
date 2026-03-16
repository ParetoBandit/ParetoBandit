#!/usr/bin/env python3
"""
Test: Model Admission — Uniform Prior Initialization

New models are initialized with A = λI and b = λ·θ (T-shirt prior),
relying on family-level β_F sharing for knowledge transfer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from pareto_bandit import BanditRouter


def test_register_model_uses_uniform_prior():
    """
    Verify that register_model() initialises new arms with A = λI
    and does not copy θ from any existing model.
    """
    registry = {
        "python_specialist": {
            "model_id": "provider/python-coder",
            "display_name": "Python coding specialist expert",
            "hle": 0.75,
            "input_cost_per_m": 2.0,
            "output_cost_per_m": 6.0
        }
    }

    router = BanditRouter.create(model_registry=registry, priors="none")

    # Manually inject non-trivial state into the existing model so we can
    # verify the new model doesn't inherit it.
    dim = router.bandit.dim
    rng = np.random.RandomState(42)
    for _ in range(30):
        x = rng.randn(dim)
        x /= np.linalg.norm(x)
        router.bandit.update("python_specialist", x, reward=0.9)

    theta_trained = (
        router.bandit.A_inv["python_specialist"] @ router.bandit.b["python_specialist"]
    )
    assert np.linalg.norm(theta_trained) > 0.01, (
        "Precondition: trained model should have non-trivial θ"
    )

    router.register_model(
        "ruby_specialist",
        speed="fast",
        capabilities=["coding"],
    )

    A_ruby = router.bandit.A["ruby_specialist"]
    lam = router.bandit.init_lambda

    # A should be λI
    off_diag = A_ruby - np.diag(np.diag(A_ruby))
    assert np.allclose(off_diag, 0, atol=1e-10), "A should be diagonal (λI)"
    assert np.allclose(np.diag(A_ruby), lam, rtol=1e-6), \
        f"Diagonal should equal init_lambda={lam}"

    # θ should NOT mirror the trained model
    b_ruby = router.bandit.b["ruby_specialist"]
    theta_ruby = router.bandit.A_inv["ruby_specialist"] @ b_ruby

    if np.linalg.norm(theta_ruby) > 1e-6 and np.linalg.norm(theta_trained) > 1e-6:
        cosine = np.dot(theta_trained, theta_ruby) / (
            np.linalg.norm(theta_trained) * np.linalg.norm(theta_ruby)
        )
        assert cosine < 0.95, (
            f"New model θ should NOT mirror trained neighbor (cosine={cosine:.3f})"
        )


if __name__ == "__main__":
    test_register_model_uses_uniform_prior()
    print("\nAll model admission tests passed!")
    sys.exit(0)
