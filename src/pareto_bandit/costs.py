"""Shared cost normalization utilities.

Provides a single-source-of-truth logarithmic cost normalization used by both
:class:`~pareto_bandit.router.BanditRouter` (production routing) and offline
evaluation scripts (experiment baselines).  Any change to the normalization
formula or market anchors propagates automatically to all consumers.
"""
from __future__ import annotations

import math


def log_normalize_cost(
    cost: float,
    floor: float,
    ceiling: float,
) -> float:
    """Logarithmic normalization of a cost value to [0, 1].

    Maps ``cost`` into the range [0, 1] using fixed log-scale market
    anchors.  Values at or below ``floor`` map to 0; values at or above
    ``ceiling`` map to 1.

    This is the canonical normalization shared by ``BanditRouter`` and all
    offline evaluation baselines.

    Args:
        cost: Raw cost value (must be > 0; same unit as ``floor``/``ceiling``).
        floor: Market floor anchor (cheapest expected cost).
        ceiling: Market ceiling anchor (most expensive expected cost).

    Returns:
        Normalized cost in [0.0, 1.0].
    """
    safe = max(cost, floor)
    log_range = math.log(ceiling) - math.log(floor)
    if log_range <= 0:
        return 0.5
    penalty = (math.log(safe) - math.log(floor)) / log_range
    return max(0.0, min(1.0, penalty))
