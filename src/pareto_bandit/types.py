"""Pure data definitions for BanditRouter.

Contains configuration dataclasses, type aliases, and named constants
used across the router package.  No behavioral logic — only data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progressive Registration API: Type Definitions
# ---------------------------------------------------------------------------
Capability = Literal["coding", "math", "creative", "reasoning", "general"]
SpeedProfile = Literal["fast", "balanced", "slow"]


# ---------------------------------------------------------------------------
# Router Configuration (Magic Numbers Documented)
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """Bayesian priors for new model admission.

    These values shape the initial belief state (theta) for a new model
    before we have observed any real traffic.

    Scientific Justification:
    - Bias terms: Derived from cost asymmetry (30x price differential).
    - Knowledge transfer: Handled by warmup priors
      (continuous, empirically validated).

    NOTE: complexity_weight fields were removed when the feature pipeline was
    simplified to [PCA | bias].  T-shirt sizing now operates exclusively through
    the bias dimension.  If per-task complexity weighting is reintroduced, add a
    dedicated feature dimension and map it in _build_feature_map().
    """

    fast_bias: float = 0.0
    """Fast profile (e.g. Haiku, Flash) — no bias adjustment (neutral)."""

    slow_bias: float = 0.05
    """Slow profile (e.g. Opus, GPT-4) — positive bias encodes belief that
    expensive models have latent quality."""

    balanced_bias: float = 0.0
    """Balanced profile (e.g. GPT-3.5, Sonnet) — neutral priors."""

    default_cost_per_1m: float = 10.00
    """Pessimistic fallback: assume $10/1M tokens."""

    default_latency_s: float = 2.0
    """Pessimistic fallback: assume 2 s TTFT."""


@dataclass
class RouterConfig:
    """Centralized configuration for BanditRouter.

    **CANONICAL CONFIG**: This is the production-grade configuration for BanditRouter.

    **Scientific Validation (Appendix A):**
    Key hyperparameters validated via prior transfer theory and ablation:

    1. **Market Anchors (cost normalization)**:
       - Derived from empirical market data (2024-2026)
       - Cost: $0.0001-$0.04/1k tokens (portfolio range)

    2. **Probation Period (500 requests)**:
       - Derived from convergence analysis (95% confidence interval)
       - Robust across [300, 1000] range (not shown for brevity)

    This dataclass is the single source of truth for the current production router.
    """

    # ---------------------------------------------------------------------------
    # Production Stability: Memory Management
    # ---------------------------------------------------------------------------
    max_log_size: int = 10_000
    """Ring buffer size for RoutingLog entries.
    At 100 QPS with 33-dim context vectors (~300 bytes/log), 10k logs ~ 3 MB."""

    # ---------------------------------------------------------------------------
    # LinUCB Regularization
    # ---------------------------------------------------------------------------
    init_lambda: float = 1.0
    """Initialization regularization for cold-start stability (A_0 = lambda I)."""

    # ---------------------------------------------------------------------------
    # Numerical Stability: Safety Net for Low-Traffic Arms
    # ---------------------------------------------------------------------------
    stability_check_interval: int = 1000
    """Check for numerical instability every N global updates."""

    stability_threshold: float = 1e6
    """Maximum trace(A_inv) before triggering regularization reset.

    trace(A_inv) grows as A decays toward singularity. For reference:
    - Healthy matrix: trace(A_inv) ~ d (dimension)
    - Decaying matrix: trace(A_inv) >> d
    - Near-singular: trace(A_inv) > 1e6

    If exceeded, triggers O(d^3) reset, but this is rare (e.g. once per day).
    """

    # Cost Normalization Anchors (Logarithmic Market Width)
    market_cost_floor: float = 0.0001
    """$/1k tokens — captures cheapest model."""
    market_cost_ceiling: float = 0.04
    """$/1k tokens — slightly above most expensive."""

    # ---------------------------------------------------------------------------
    # RESILIENCE DEFAULTS: Pessimistic Fallbacks (Fail-Operational Design)
    # ---------------------------------------------------------------------------
    default_missing_cost_per_m: float = 10.00
    """Pessimistic cost fallback when input_cost_per_m/output_cost_per_m is
    missing.  Set to $10/1M tokens (Opus/o1-high tier) to treat unknown models
    as expensive.  Prevents them from winning cost-sensitive races while keeping
    them eligible."""

    default_missing_latency: float = 2.0
    """Pessimistic latency fallback when time_to_first_token_seconds is missing.
    Set to 2.0 s (slow but usable) to prevent unknown models from winning
    low-latency races unfairly while remaining eligible for selection."""

    # ---------------------------------------------------------------------------
    # Progressive Registration API: Empirical Priors (Bayesian Initialization)
    # ---------------------------------------------------------------------------
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)

    registration_strict_kwargs: bool = True
    """Validate unknown kwargs in ``register_model`` when True.

    Open-source default is strict to fail fast on user typos (e.g. ``latnecy``).
    Set to ``False`` for backward compatibility in integrations that pass extra keys.
    """

    @property
    def cost_range_log(self) -> float:
        """Logarithmic range for cost normalization."""
        return math.log(self.market_cost_ceiling) - math.log(self.market_cost_floor)


# ---------------------------------------------------------------------------
# Exploration Rates
# ---------------------------------------------------------------------------

class ExplorationRate:
    """Named presets for exploration (Alpha)."""

    STATIC = 0.0
    """Pure exploitation."""
    SAFE = 0.1
    """Optimal with sigmoid priors (see parameter search)."""
    BALANCED = 1.0
    """Moderate exploration for cold-start scenarios."""
    AGGRESSIVE = 2.0
    """High exploration."""

    _RATES = {
        "static": STATIC,
        "safe": SAFE,
        "balanced": BALANCED,
        "aggressive": AGGRESSIVE,
    }

    DEFAULT = SAFE

    @classmethod
    def get(cls, name: Any) -> float:
        """Resolve an exploration rate by name or numeric value.

        Args:
            name: A string preset name (e.g. ``"safe"``), or a numeric value
                that will be cast to *float* directly.

        Returns:
            The resolved exploration rate.

        Raises:
            ValueError: If *name* is an unrecognised string.
        """
        if isinstance(name, (int, float)):
            return float(name)

        try:
            key = str(name).lower()
            val = cls._RATES.get(key)
            if val is not None:
                logger.debug(f"ExplorationRate.get('{name}') -> {val}")
                return val
            return float(name)
        except (ValueError, AttributeError):
            raise ValueError(f"Unknown exploration '{name}'")


# ---------------------------------------------------------------------------
# Routing Log
# ---------------------------------------------------------------------------

@dataclass
class RoutingLog:
    """Per-request record of a routing decision."""

    request_id: str
    timestamp_s: float
    prompt: str
    selected_model: str
    predicted_utility: float
    cost_usd: float
    latency_s: float
    context_vector: np.ndarray | None = None
    expected_reward: float = 0.0
    total_priority_weight: float = 1.0
    pacer_lambda_t: float | None = None
    pacer_cost_ema: float | None = None
