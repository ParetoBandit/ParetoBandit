"""Warmup utilities for BanditRouter initialisation.

Provides heuristic prior construction (``get_heuristic_prior``) so that
newly registered models start with a reasonable (A, b) pair rather than an
uninformative identity covariance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def safe_inv(A: np.ndarray) -> np.ndarray:
    """Safe matrix inversion with pseudo-inverse fallback for stability."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def get_heuristic_prior(
    model_data: dict[str, Any],
    dim: int,
    init_lambda: float = 1.0,
    n_effective: float = 5.0,
    default_quality: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute heuristic prior (A, b) for a new model not in the warmup joblib.

    Strategy:
    Constructs a synthetic prior that mimics having seen 'n_effective' samples
    with an average reward equal to the model's quality score.

    **Numerical Stability Note:**
    By initializing A = n_effective * I, we ensure the matrix scales with confidence.
    BanditRouter will add init_lambda * I to A afterwards. We set b such that the
    mean prediction exactly equals the target quality, preventing Bayesian shrinkage.

    Args:
        model_data: Dictionary containing model metadata (quality_score, etc.)
        dim: Feature vector dimension (including bias)
        init_lambda: Regularization strength (default: 1.0)
        n_effective: Effective number of samples to represent in the prior (default: 5.0)
        default_quality: Fallback quality score if none found in metadata (default: 0.5)

    Returns:
        Tuple of (A_prior, b_prior)
    """
    # 1. Initialize A (Covariance) to represent n_effective samples
    # (Router will later add init_lambda * I to this)
    A = n_effective * np.eye(dim)

    # 2. Initialize b (Reward Vector)
    b = np.zeros(dim)

    # 3. Apply the "Prior Belief"
    quality = model_data.get("initial_quality")

    if quality is None:
        logger.warning(
            f"Model missing 'initial_quality' field, using default={default_quality}. "
            f"This may cause inconsistent initialization."
        )
        quality = default_quality

    # CRITICAL: b[-1] assumes the BIAS term is the LAST feature in the vector.
    # We want final mean theta[-1] = quality.
    # Since final A will be (n_effective + init_lambda) * I,
    # we set b[-1] = quality * (n_effective + init_lambda).
    b[-1] = float(quality) * (float(n_effective) + float(init_lambda))

    return A, b
