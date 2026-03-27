"""
Shared numerical utilities for the BanditRouter.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def sigmoid(x: float) -> float:
    """Standard logistic function mapping (-inf, inf) to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def argmax_random_tiebreak(
    scores: Dict[str, float],
    rng: np.random.Generator | None = None,
) -> str:
    """Return the key with the maximum value, breaking ties uniformly at random.

    Standard ``max(scores, key=scores.get)`` is deterministic when values are
    tied (returns the first key in insertion order).  For bandit algorithms this
    introduces a silent bias: e.g. a freshly-initialized policy always picks the
    first model in the list before any learning has occurred.

    This helper collects all keys sharing the maximum value and returns one
    uniformly at random, eliminating initialization-order bias.

    Args:
        scores: Mapping of candidate names to their scores.
        rng: Explicit NumPy generator for reproducibility.  Falls back to
            the global ``np.random`` state if *None* (legacy callers).
    """
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        keys = list(scores.keys())
        idx = rng.integers(len(keys)) if rng is not None else np.random.randint(len(keys))
        return keys[idx]
    max_val = max(finite.values())
    tied = [k for k, v in finite.items() if abs(v - max_val) < 1e-12]
    if len(tied) == 1:
        return tied[0]
    idx = rng.integers(len(tied)) if rng is not None else np.random.randint(len(tied))
    return tied[idx]
