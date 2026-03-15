"""
Production-grade contextual bandit router (Hot Path).

Core Features:
1. Warmup Priors: Initializes with learned preferences from 80k battles.
2. Default Registry: Automatically loads 80+ models with cost/latency data.
3. Constraints: Supports max_cost, max_latency, and quality floors.

New Model Registration:
- Progressive API: register_model() accepts varying levels of detail
  - Tier A (Archetypes): capabilities=["coding", "math"]
  - Tier B (T-Shirt Sizing): speed="fast" (cheap), "slow" (expensive)
  - Tier C (Agnostic): Just model_id (cold start with high exploration)
"""

from __future__ import annotations

import json
import math
import time
import uuid
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter, deque, defaultdict
from typing import Any, Dict, Generator, List, Tuple, Optional, Literal, TypedDict, Union
import re
import copy

import numpy as np

# Prevent tokenizers parallelism hangs (common with SentenceTransformers).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# SentenceTransformer is optional; only needed when the default embedding
# pipeline is used (i.e. no custom_encoder or pre-computed vectors).
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[misc,assignment]


try:
    from bandit_gpt.cluster_detector import ClusterDetector
except ImportError:
    ClusterDetector = None  # Optional feature


try:
    import joblib
except ImportError:
    joblib = None



# ---------------------------------------------------------------------------
# Progressive Registration API: Type Definitions
# ---------------------------------------------------------------------------
Capability = Literal["coding", "math", "creative", "reasoning", "general"]
SpeedProfile = Literal["fast", "balanced", "slow"]

# ---------------------------------------------------------------------------
# Imports: Storage, Features, Utils
# ---------------------------------------------------------------------------
try:
    from bandit_gpt.storage import ContextStore, EphemeralContextStore, SqliteContextStore
    from bandit_gpt.utils import sigmoid, procedural_warmup, safe_inv, get_heuristic_prior
except ImportError:
    # Fallback for direct file import (not installed as package)
    from .storage import ContextStore, EphemeralContextStore, SqliteContextStore
    from .utils import sigmoid, procedural_warmup, safe_inv, get_heuristic_prior

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Staleness Variance Inflation Cap
# ---------------------------------------------------------------------------
# When forgetting_factor (gamma) < 1.0 the UCB variance for an arm that has
# not been selected recently is inflated by dividing the base variance by
# gamma^dt.  This implements "optimism in the face of uncertainty" for stale
# arms: an arm the router has avoided for a long time is treated as more
# uncertain, encouraging re-exploration.
#
# Without a floor on gamma^dt the inflation is unbounded as dt → ∞.  Because
# cost penalties are additive constants, a sufficiently large dt would cause
# the exploration bonus (alpha * sqrt(var / gamma^dt)) to dominate any finite
# cost_penalty, guaranteeing selection of stale arms regardless of cost.
#
# This constant caps the variance multiplier at _MAX_VAR_INFLATION_FACTOR,
# i.e., the effective floor on gamma^dt is 1 / _MAX_VAR_INFLATION_FACTOR.
# At maximum inflation the exploration bonus grows by at most
# sqrt(200) ≈ 14× relative to the uninflated baseline — strong enough to
# force re-exploration but bounded enough to preserve the cost signal.
#
# **This path is only active when forgetting_factor < 1.0 (non-default).**
# The default configuration is stationary (forgetting_factor = 1.0) and
# never applies variance inflation; the cost penalty remains fully effective
# at all times in the default deployment.
#
# If stricter cost control is required alongside non-stationary forgetting,
# reduce alpha or increase cost_penalty rather than relaxing this cap.
_MAX_VAR_INFLATION_FACTOR: float = 200.0
"""Maximum factor by which staleness may inflate the UCB variance term.

Applies only when forgetting_factor < 1.0.  Adjust downward to tighten cost
enforcement in non-stationary deployments, or upward for more aggressive
staleness-driven exploration.  The default (200) keeps the worst-case
exploration bonus within ~14× of the steady-state value.
"""

# ---------------------------------------------------------------------------
# Numerical Stability Constants
# ---------------------------------------------------------------------------

_MAX_STALENESS_DT: int = 1000
"""Maximum dt (time steps since last update) used in gamma^dt decay calculations.

Caps the exponent so that gamma^dt does not underflow to exactly 0.0 for any
gamma in (0, 1).  At dt=1000 and gamma=0.99, gamma^dt ≈ 4.3e-5, which is
comfortably above float64 underflow.  Arms unseen for longer than this are
treated identically to arms unseen for exactly _MAX_STALENESS_DT steps.
"""

_REGULARIZATION_FLOOR_FRACTION: float = 0.1
"""Fraction of init_lambda used as the proactive-regularization trigger threshold.

During each update, if effective regularization (estimated from the diagonal
of A) has decayed to below init_lambda * _REGULARIZATION_FLOOR_FRACTION,
a top-up injection is triggered before the rank-1 Sherman-Morrison update.
This prevents the covariance matrix from approaching singularity in
high-traffic, non-stationary settings before the SM denominator check fires.
"""

_SM_DENOMINATOR_THRESHOLD: float = 1e-6
"""Minimum absolute denominator for the Sherman-Morrison rank-1 update.

The SM denominator is 1 + weight * x^T A^{-1} x.  For any positive-definite
A (guaranteed by initialisation and rank-1 PD updates) this quantity is
always >= 1.  Falling below this threshold therefore signals floating-point
corruption of the cached A_inv matrix rather than a legitimate near-zero
value, and triggers a full O(d^3) recomputation from scratch.
"""

_OUTPUT_COST_MULTIPLIER: float = 3.0
"""Heuristic ratio of output-token cost to input-token cost used to derive
blended_cost_per_m when only cost_usd (input-token price) is provided.

Typical LLM pricing charges 3–5× more for output than input tokens.  The
default of 3.0 is conservative; increase for providers with higher output
premiums.  The blended cost is (input_cost + output_cost) / 2.
"""


def _argmax_random_tiebreak(scores: Dict[str, float]) -> str:
    """
    Return the key with the maximum value, breaking ties uniformly at random.

    Standard ``max(scores, key=scores.get)`` is deterministic when values are
    tied (returns the first key in insertion order).  For bandit algorithms this
    introduces a silent bias: e.g. a freshly-initialized policy always picks the
    first model in the list before any learning has occurred.

    This helper collects all keys sharing the maximum value and returns one
    uniformly at random, eliminating initialization-order bias.
    """
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        keys = list(scores.keys())
        return keys[np.random.randint(len(keys))]
    max_val = max(finite.values())
    tied = [k for k, v in finite.items() if abs(v - max_val) < 1e-12]
    if len(tied) == 1:
        return tied[0]
    return tied[np.random.randint(len(tied))]


def _inflate_variance(
    var: float,
    gamma: float,
    dt: int,
    max_staleness_dt: int = _MAX_STALENESS_DT,
    max_var_inflation: float = _MAX_VAR_INFLATION_FACTOR,
) -> float:
    """Apply staleness-based variance inflation with a bounded cap.

    When ``gamma < 1.0``, divides ``var`` by ``gamma^dt`` to widen the
    confidence interval for stale arms, capped at *max_var_inflation*
    to prevent the exploration bonus from overwhelming additive cost penalties.

    For stationary settings (``gamma == 1.0``) or ``dt == 0``, returns ``var``
    unchanged.

    Args:
        var: Base variance (x^T A^{-1} x).
        gamma: Forgetting factor in (0, 1].
        dt: Steps since last update or selection (non-negative).
        max_staleness_dt: Cap on ``dt`` to avoid float64 underflow in
            ``gamma^dt``.
        max_var_inflation: Maximum multiplicative factor for the variance.

    Returns:
        Inflated (or unchanged) variance.
    """
    if gamma >= 1.0 or dt <= 0:
        return var
    decay_factor = gamma ** min(dt, max_staleness_dt)
    inflation_floor = 1.0 / max_var_inflation
    return var / max(decay_factor, inflation_floor, 1e-12)



def _effective_staleness(
    t: int,
    last_update: Dict[str, int],
    last_played: Dict[str, int],
    model: str,
) -> int:
    """Request-count intervals since the most recent event for *model*.

    ``t`` is a *request counter*: it is incremented at route time by
    ``mark_selected()``, not at feedback time.  Uses
    ``max(last_update, last_played)`` as the reference to prevent
    artificial uncertainty inflation for arms whose feedback is
    still in flight (delayed RLHF scenario).

    Args:
        t: Global logical clock (request counter).
        last_update: Per-model timestamp of last update.
        last_played: Per-model timestamp of last selection.
        model: Model identifier.

    Returns:
        Non-negative staleness in request-count units.
    """
    most_recent = max(
        last_update.get(model, 0),
        last_played.get(model, 0),
    )
    return t - most_recent


@dataclass
class _SMUpdateResult:
    """Return value of :func:`_sherman_morrison_update`."""
    A: np.ndarray
    A_inv: np.ndarray
    b: np.ndarray
    regularization_floor: float
    used_fallback: bool


def _sherman_morrison_update(
    A: np.ndarray,
    A_inv: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    reward: float,
    weight: float,
    init_lambda: float,
    regularization_floor: float,
    model_name: str,
    reg_floor_fraction: float = _REGULARIZATION_FLOOR_FRACTION,
) -> _SMUpdateResult:
    """Perform a rank-1 Sherman-Morrison update on (A, A_inv, b).

    Encapsulates the O(d^2) rank-1 update and the O(d^3) near-singularity
    fallback with gap-based regularization injection.  Callers handle locking
    and time advancement; this function is pure linear algebra.

    Args:
        A: Current precision matrix (d x d).
        A_inv: Cached inverse of A (d x d).
        b: Target vector (d,).
        x: Context vector (d,).
        reward: Observed reward for this observation.
        weight: Importance weight (must be > 0).
        init_lambda: Baseline regularization strength (lambda_0).
        regularization_floor: Current tracked floor for this arm.
        model_name: For logging only.
        reg_floor_fraction: Minimum fraction of *init_lambda* used as
            the fallback regularization injection floor.

    Returns:
        :class:`_SMUpdateResult` containing updated matrices and floor.
    """
    x_outer = weight * np.outer(x, x)
    reward_x = weight * reward * x

    u = x * np.sqrt(weight)
    A_inv_u = A_inv @ u
    v_A_inv = u @ A_inv
    denominator = 1.0 + (u @ A_inv_u)

    if abs(denominator) > _SM_DENOMINATOR_THRESHOLD:
        new_A_inv = A_inv - np.outer(A_inv_u, v_A_inv) / denominator
        new_A = A + x_outer
        new_b = b + reward_x
        return _SMUpdateResult(
            A=new_A, A_inv=new_A_inv, b=new_b,
            regularization_floor=regularization_floor,
            used_fallback=False,
        )

    # Fallback: A_inv has drifted — rebuild from scratch with gap-based
    # regularization injection.  See DisjointLinUCBPolicy.update() docstring
    # for the full mathematical rationale.
    logger.warning(
        f"[WARN] Sherman-Morrison near-singularity for {model_name}: "
        f"|denominator|={abs(denominator):.2e} < {_SM_DENOMINATOR_THRESHOLD}. "
        f"A_inv has numerically drifted; rebuilding with gap-based "
        f"regularisation injection."
    )
    needed = max(init_lambda - regularization_floor, init_lambda * reg_floor_fraction)
    dim = A.shape[0]
    new_A = A + x_outer + needed * np.eye(dim)
    new_A_inv = safe_inv(new_A)
    new_b = b + reward_x
    return _SMUpdateResult(
        A=new_A, A_inv=new_A_inv, b=new_b,
        regularization_floor=regularization_floor + needed,
        used_fallback=True,
    )


def _safe_multivariate_normal(
    mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw from a multivariate normal with numerical safety fallbacks.

    Attempts ``rng.multivariate_normal`` first, then adds jitter,
    then falls back to diagonal sampling if the covariance is too
    ill-conditioned.

    Args:
        mean: Mean vector of shape ``(dim,)``.
        cov: Covariance matrix of shape ``(dim, dim)``.
        n_samples: Number of draws.
        dim: Dimensionality (used for diagonal fallback).
        rng: Explicit NumPy random generator for reproducibility.
            Falls back to ``np.random.default_rng()`` if *None*.

    Returns:
        Array of shape ``(n_samples, dim)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    try:
        return rng.multivariate_normal(mean, cov, n_samples)
    except np.linalg.LinAlgError:
        pass

    jitter = max(np.trace(cov) / dim, 1e-12) * 1e-6
    cov_safe = cov + jitter * np.eye(dim)
    try:
        return rng.multivariate_normal(mean, cov_safe, n_samples)
    except np.linalg.LinAlgError:
        avg_var = max(np.trace(cov) / dim, 1e-12)
        return rng.normal(
            loc=mean, scale=np.sqrt(avg_var), size=(n_samples, dim)
        )


# ---------------------------------------------------------------------------
# Exception Classes
# ---------------------------------------------------------------------------

class MissingCostError(ValueError):
    """Raised when a model has no blended cost and it cannot be derived.

    The router requires every registered model to have a known
    ``blended_cost_per_m`` ($/M tokens).  This can be provided directly,
    via ``price_1m_blended``, or derived from both ``input_cost_per_m``
    and ``output_cost_per_m``.  If none of these are available (or only
    one of input/output is present), this error is raised at init time
    so the user can fix the registry entry.
    """


class NoEligibleModelsError(Exception):
    """Raised when no models pass the hard cost/latency/quality constraints.

    Attributes:
        reasons: Mapping of model_id -> list of human-readable exclusion strings.
    """

    def __init__(
        self,
        reasons: Dict[str, List[str]],
        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] | None = None,
    ):
        self.reasons = reasons
        lines = ["No models meet the specified constraints:"]
        if max_cost is not None:
            lines.append(f"  max_cost: ${max_cost:.6f}/1k tokens")
        if max_latency is not None:
            lines.append(f"  max_latency: {max_latency:.3f}s")
        if quality_floor:
            lines.append(f"  quality_floor: {quality_floor}")
        lines.append("")
        for model, model_reasons in reasons.items():
            lines.append(f"  {model}: {', '.join(model_reasons)}")
        super().__init__("\n".join(lines))


class NoModelScoredError(ValueError):
    """Raised when model scoring receives no eligible/scorable candidates.

    This error is used by lower-level selectors (e.g. cost-aware expert
    adapters) to provide a strict API contract for open-source consumers:
    selection methods that are typed to return ``str`` never return ``None``.
    """


# ---------------------------------------------------------------------------
# Router Configuration (Magic Numbers Documented)
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """
    Bayesian priors for new model admission.
    
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
    # Fast Profile (e.g., Haiku, Flash) -> No bias adjustment (neutral)
    fast_bias: float = 0.0
    
    # Slow Profile (e.g., Opus, GPT-4) -> Bias TOWARDS usage (believe expensive = high quality)
    # Positive bias encodes belief that expensive models have latent quality
    slow_bias: float = 0.05
    
    # Balanced Profile (e.g., GPT-3.5, Sonnet) -> Neutral priors
    balanced_bias: float = 0.0
    
    # Fallback Metadata (Pessimistic Defaults for Resilience)
    default_cost_per_1m: float = 10.00  # Assume expensive ($10/1M)
    default_latency_s: float = 2.0      # Assume slow (2s)

@dataclass
class RouterConfig:
    """
    Centralized configuration for BanditRouter.
    
    **CANONICAL CONFIG**: This is the production-grade configuration for BanditRouter.
    
    **Scientific Validation (Appendix A):**
    Key hyperparameters validated via prior transfer theory and ablation:
    
    1. **Market Anchors (cost/latency normalization)**:
       - Derived from empirical market data (2024-2026)
       - Cost: $0.0001-$0.04/1k tokens (portfolio range)
       - Latency: 0.05s-5.0s (instant to timeout threshold)
    
    2. **Probation Period (500 requests)**:
       - Derived from convergence analysis (95% confidence interval)
       - Robust across [300, 1000] range (not shown for brevity)
    
    This dataclass is the single source of truth for the current production router.
    """
    
    # ---------------------------------------------------------------------------
    # Production Stability: Memory Management
    # ---------------------------------------------------------------------------
    # Prevent OOM from unbounded log growth.
    # At 100 QPS with 33-dim context vectors (~300 bytes/log), 10k logs ≈ 3MB.
    # Adjust based on deployment memory constraints and feedback latency.
    max_log_size: int = 10_000         # Ring buffer size for RoutingLog entries
    
    # ---------------------------------------------------------------------------
    # Procedural Warmup: Covariance Shaping
    # ---------------------------------------------------------------------------
    # Number of synthetic samples for procedural warmup to shape covariance matrix.
    # Mathematical requirement: Need at least d samples to span the space.
    # Recommendation: 2d for robust covariance estimation.
    # 
    # With 5 archetypes, samples_per_archetype = procedural_warmup_samples // 5
    # Default 100 → 20 samples per archetype → sufficient to shape 33D covariance
    procedural_warmup_samples: int = 100  # Warmup samples (~3*d for d=33)
    
    # ---------------------------------------------------------------------------
    # LinUCB Regularization
    # ---------------------------------------------------------------------------
    # Sherman-Morrison only handles rank-1 updates (O(d²)).
    # A diagonal injection (+λI) is full-rank and forces O(d³) inversion.
    # init_lambda is applied once at initialization (A₀ = λI) to ensure
    # cold-start stability. Data terms (xx^T) keep A well-conditioned at
    # runtime. Performance: 2,710 updates/sec @ d=384 (O(d²) path).
    init_lambda: float = 1.0
    """Initialization regularization for cold-start stability (A₀ = λI)."""

    # ---------------------------------------------------------------------------
    # Numerical Stability: Safety Net for Low-Traffic Arms
    # ---------------------------------------------------------------------------
    # With initialization-only regularization, matrices can decay toward
    # singularity if an arm receives zero traffic for extended periods.
    # This safety check triggers a regularization reset when numerical
    # instability is detected.
    #
    # **Cost**: O(d) trace computation every N updates (cheap)
    # **Benefit**: Prevents singular matrices in edge cases
    # **Frequency**: Default every 1000 updates ≈ once per 10 seconds @ 100 QPS
    stability_check_interval: int = 1000
    """Check for numerical instability every N global updates."""
    
    stability_threshold: float = 1e6
    """
    Maximum trace(A_inv) before triggering regularization reset.
    
    trace(A_inv) grows as A decays toward singularity. For reference:
    - Healthy matrix: trace(A_inv) ≈ d (dimension)
    - Decaying matrix: trace(A_inv) >> d
    - Near-singular: trace(A_inv) > 1e6
    
    If exceeded, triggers O(d³) reset, but this is rare (e.g., once per day).
    """
    
    # Cost Normalization Anchors (Logarithmic Market Width)
    # Tightened to actual portfolio range ($0.0001-$0.0375/1k) for better
    # penalty differentiation (1.39x improvement over a wider default range).
    market_cost_floor: float = 0.0001  # $/1k tokens (captures cheapest model)
    market_cost_ceiling: float = 0.04  # $/1k tokens (slightly above most expensive)
    
    # Latency Normalization Anchors
    # Floor: 50ms (instant/cached responses)
    # Ceiling: 5.0s (reasonable timeout threshold)
    market_latency_floor: float = 0.05  # seconds
    market_latency_ceiling: float = 5.0  # seconds
    
    # ---------------------------------------------------------------------------
    # RESILIENCE DEFAULTS: Pessimistic Fallbacks (Fail-Operational Design)
    # ---------------------------------------------------------------------------
    # Used when registry metadata is missing or malformed.
    # 
    # **Philosophy: "Pessimistic" vs "Fail-Secure" vs "Optimistic"**
    # - Fail-Secure (float('inf')): Model is banned → All models missing data = OUTAGE
    # - Optimistic ($0.00): Router floods unknown models → Potential budget blowout
    # - Pessimistic (expensive/slow): Model treated as luxury → Service UP, conservative
    # 
    # By assuming unknown models are expensive (Opus tier) and slow, we:
    # 1. Keep traffic flowing during metadata corruption/config failures
    # 2. Prevent budget blowouts (unknown models only picked if strictly necessary)
    # 3. Quality becomes the primary differentiator among "expensive" models
    # ---------------------------------------------------------------------------
    
    default_missing_cost_per_m: float = 10.00
    """
    Pessimistic cost fallback when input_cost_per_m/output_cost_per_m is missing.
    
    Set to $10/1M tokens (Opus/o1-high tier) to treat unknown models as expensive.
    This prevents them from winning cost-sensitive races while keeping them eligible.
    """
    
    default_missing_latency: float = 2.0
    """
    Pessimistic latency fallback when time_to_first_token_seconds is missing.
    
    Set to 2.0 seconds (slow but usable) to prevent unknown models from winning
    low-latency races unfairly while remaining eligible for selection.
    """
    
    # ---------------------------------------------------------------------------
    # Progressive Registration API: Empirical Priors (Bayesian Initialization)
    # ---------------------------------------------------------------------------
    # These values encode domain knowledge from LLM ecosystem cost/performance analysis.
    # They initialize the bandit with reasonable defaults to accelerate convergence.
    # All parameters are tunable via RouterConfig for custom deployments.
    # 
    # Scientific Justification:
    #   - Speed-based biases reflect cost asymmetry (30x difference between tiers)
    #   - Complexity weights encode known conditional failure probabilities
    #   - Anchor boosts quantify task-specific performance differentials
    # 
    # Optimization: Run tune_registration_priors.py on your data to find optimal values.
    # ---------------------------------------------------------------------------
    
    
    # [RESTORED] Registration Priors for Progressive Model Admission
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    registration_strict_kwargs: bool = True
    """Validate unknown kwargs in ``register_model`` when True.

    Open-source default is strict to fail fast on user typos (e.g. ``latnecy``).
    Set to ``False`` for backward compatibility in legacy integrations that pass
    extra keys.
    """
    
    @property
    def cost_range_log(self) -> float:
        """Logarithmic range for cost normalization."""
        return np.log(self.market_cost_ceiling) - np.log(self.market_cost_floor)
    
    @property
    def latency_range_log(self) -> float:
        """Logarithmic range for latency normalization."""
        return np.log(self.market_latency_ceiling) - np.log(self.market_latency_floor)


from .config import DEFAULT_SENTENCE_TRANSFORMER

DEFAULT_CONTEXT_MODEL = DEFAULT_SENTENCE_TRANSFORMER

# ---------------------------------------------------------------------------
# Exploration Rates
# ---------------------------------------------------------------------------
class ExplorationRate:
    """Named presets for exploration (Alpha)."""
    STATIC = 0.0       # Pure exploitation
    SAFE = 0.1         # Optimal with sigmoid priors (see parameter search)
    BALANCED = 1.0     # Legacy setting for cold-start scenarios
    AGGRESSIVE = 2.0   # High exploration

    _RATES = {
        "static": STATIC, "safe": SAFE, "balanced": BALANCED, "aggressive": AGGRESSIVE
    }
    
    DEFAULT = SAFE  # Optimal: α=0.1 with sigmoid-transformed priors

    @classmethod
    def get(cls, name: Any) -> float:
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
# Helper Functions
# ---------------------------------------------------------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    return x / n if n > eps else x

def estimate_tokens_rough(text: str) -> int:
    """Estimate token count from whitespace word count (word_count * 1.3).

    This is a coarse heuristic used only for cost logging, **not** for
    billing or hard budget enforcement.  Typical error bounds:

    - Natural-language prose: ~5-15% overestimate vs. GPT tokenizers.
    - Code / punctuation-heavy text: 30-50% underestimate (subword
      tokenizers split identifiers and operators into multiple tokens).
    - Non-Latin scripts: highly variable; CJK text has far more tokens
      per whitespace word.

    For accurate token counts, use the model's actual tokenizer.
    """
    if not text:
        return 0
    return int(max(0, round(len(str(text).split()) * 1.3)))

# ---------------------------------------------------------------------------
# Model Family Inference (for analytics and correlation-based grouping)
# ---------------------------------------------------------------------------

def infer_model_family(model_id: str) -> str:
    """
    Infer model family from a model_id by stripping variant suffixes.

    Models within the same family are expected to have similar reward
    functions.  Used by ``compute_correlation_families`` and family-aware
    analytics.

    Strips size qualifiers (-mini, -large), instruction tuning (-instruct),
    quality tiers (-turbo, -pro), date stamps (-2024-04-09), parameter
    counts (-70b), and trailing minor versions (.1, .2).

    Override the inference by setting an explicit ``family`` field in the
    model registry entry.

    Examples:
        "openai/gpt-4-turbo"                -> "openai/gpt-4"
        "openai/gpt-4o-mini"                -> "openai/gpt-4o"
        "openai/gpt-5.1"                    -> "openai/gpt-5"
        "openai/o1-mini"                    -> "openai/o1"
        "anthropic/claude-3.5-sonnet"       -> "anthropic/claude-3"
        "anthropic/claude-3-haiku"          -> "anthropic/claude-3"
        "mistralai/mixtral-8x7b-instruct"   -> "mistralai/mixtral-8x7b"
        "meta-llama/llama-3.1-70b-instruct" -> "meta-llama/llama-3"
        "google/gemini-2.0-flash"           -> "google/gemini-2"
    """
    if "/" not in model_id:
        return model_id

    provider, model = model_id.split("/", 1)

    _SUFFIXES = (
        "-turbo", "-mini", "-small", "-medium", "-large", "-xl", "-xxl",
        "-instruct", "-chat", "-preview", "-latest", "-pro", "-flash",
        "-lite", "-haiku", "-sonnet", "-opus", "-nano", "-micro",
        "-thinking", "-online", "-free", "-nightly", "-exp",
    )

    # Iteratively strip date stamps, parameter counts, and known suffixes
    # until no further changes occur.  Interleaving is necessary because a
    # date stamp or param count may be followed by a suffix (or vice-versa).
    changed = True
    while changed:
        changed = False

        # Date stamps: -2024-04-09, -20240409
        stripped = re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", model)
        if stripped != model:
            model = stripped
            changed = True

        # Known qualifiers
        for suffix in _SUFFIXES:
            if model.endswith(suffix):
                model = model[: -len(suffix)]
                changed = True

        # Simple parameter counts: -8b, -70b, -405b.  The regex does NOT
        # match mixture-of-experts specs like -8x7b because the 'x'
        # character breaks the \d+ run before '-'.
        stripped = re.sub(r"-\d+b$", "", model)
        if stripped != model:
            model = stripped
            changed = True

    # Strip trailing minor version: gpt-5.1 -> gpt-5, claude-3.5 -> claude-3
    model = re.sub(r"(\d+)\.\d+$", r"\1", model)

    return f"{provider}/{model}"


# ---------------------------------------------------------------------------
# Tetrachoric Correlation & Data-Driven Family Assignment
# ---------------------------------------------------------------------------
# For binary (0/1) rewards, Pearson r equals the phi coefficient, which has a
# ceiling effect: when base rates are extreme or differ between two models,
# phi_max << 1 even for perfectly correlated failure patterns.  The
# tetrachoric correlation estimates the latent continuous correlation
# underlying the binary observations, correcting for this attenuation.
#
# Reference: Drasgow, F. (1986). "Polychoric and polyserial correlations."
# ---------------------------------------------------------------------------

def tetrachoric_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Tetrachoric correlation for two binary (0/1) vectors.

    Solves for the bivariate normal correlation *r* such that
    P(Z₁ > c₁, Z₂ > c₂ ; r) equals the observed joint success rate,
    where c₁, c₂ are the normal thresholds implied by each variable's
    marginal success rate.

    Applies Yates' continuity correction (+0.5 to each cell) when any
    cell of the 2×2 table is zero, preventing degenerate solutions.

    Returns NaN if the solver fails to converge (e.g. all-same vectors).
    """
    from scipy.stats import norm, multivariate_normal as mvn_dist
    from scipy.optimize import brentq

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = float(len(x))

    n11 = float(np.sum((x == 1) & (y == 1)))
    n10 = float(np.sum((x == 1) & (y == 0)))
    n01 = float(np.sum((x == 0) & (y == 1)))
    n00 = float(np.sum((x == 0) & (y == 0)))

    if n00 == 0 or n11 == 0 or n10 == 0 or n01 == 0:
        n11 += 0.5; n10 += 0.5; n01 += 0.5; n00 += 0.5
        n += 2.0

    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    p_obs = n11 / n

    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return np.nan

    c1 = norm.ppf(1.0 - p1)
    c2 = norm.ppf(1.0 - p2)

    def _objective(r: float) -> float:
        r = np.clip(r, -0.999, 0.999)
        dist = mvn_dist(mean=[0, 0], cov=[[1, r], [r, 1]])
        return dist.cdf([-c1, -c2]) - p_obs

    try:
        return float(brentq(_objective, -0.999, 0.999, xtol=1e-8))
    except ValueError:
        return np.nan


def compute_correlation_families(
    reward_vectors: dict[str, np.ndarray],
    threshold: float = 0.6,
    method: str = "tetrachoric",
) -> dict[str, str]:
    """Build a family map from within-provider reward correlations.

    Parameters
    ----------
    reward_vectors : dict[str, np.ndarray]
        Mapping from model ID (e.g. ``"openai/gpt-5"``) to a reward vector
        of shape ``(n_prompts,)``.  All vectors must have the same length
        and be aligned to the same prompt ordering.  For ``method="tetrachoric"``
        the vectors are treated as binary; for ``method="pearson"`` they are
        used as continuous values.
    threshold : float
        Minimum correlation for two models to be placed in the same family.
        Typical defaults: 0.6 for tetrachoric, 0.3 for Pearson.
    method : str
        Correlation measure: ``"tetrachoric"`` (default) computes the
        tetrachoric correlation on binarised rewards; ``"pearson"`` computes
        Pearson correlation on continuous rewards.

    Returns
    -------
    family_map : dict[str, str]
        Mapping from model ID to family label.  Models within the same
        provider whose pairwise correlation meets the threshold are grouped
        via connected-components clustering.  Cross-provider grouping is
        intentionally excluded.

    Raises
    ------
    ValueError
        If *method* is not one of ``"tetrachoric"`` or ``"pearson"``.

    Notes
    -----
    Falls back to :func:`infer_model_family` for providers with only one
    model in *reward_vectors*, preserving the syntactic heuristic as a
    default for models without reward history.
    """
    if method not in ("tetrachoric", "pearson"):
        raise ValueError(f"Unknown method {method!r}; expected 'tetrachoric' or 'pearson'")

    from itertools import combinations

    providers: dict[str, list[str]] = defaultdict(list)
    for m in sorted(reward_vectors):
        prov = m.split("/")[0] if "/" in m else "__none__"
        providers[prov].append(m)

    # Union-Find for connected-components clustering
    parent: dict[str, str] = {m: m for m in reward_vectors}

    def _find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for prov, models in providers.items():
        if len(models) < 2:
            continue
        for m1, m2 in combinations(models, 2):
            if method == "tetrachoric":
                corr = tetrachoric_corr(reward_vectors[m1], reward_vectors[m2])
            else:
                corr = float(np.corrcoef(reward_vectors[m1], reward_vectors[m2])[0, 1])
            if not np.isnan(corr) and corr >= threshold:
                _union(m1, m2)

    # Build family labels from connected components
    family_map: dict[str, str] = {}
    for m in sorted(reward_vectors):
        root = _find(m)
        family_map[m] = root

    return family_map


# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------
# **COMPLEXITY ANALYSIS**
#
# update() complexity depends on forgetting_factor (gamma):
#
# Non-stationary (gamma < 1.0) → O(d²) per update
#   - Exponential decay: A *= gamma^dt, b *= gamma^dt (scalar multiply)
#   - A_inv updated via scalar division: A_inv /= gamma^dt
#   - Sherman-Morrison rank-1 correction for new observation
#   - Rare O(d³) maintenance cycle when regularization floor drops below
#     10% of init_lambda (prevents singularity under prolonged silence)
#   - Performance: ~2,710 updates/sec @ d=384
#
# Stationary (gamma = 1.0, default) → O(d²) always
#   - No decay; standard Sherman-Morrison for each observation
#   - Performance: ~3,051 updates/sec @ d=384
#
# Empirical validation: See benchmarks/diagnose_performance.py
# ---------------------------------------------------------------------------

# Type definition for bandit state snapshot
class BanditState(TypedDict):
    """Snapshot of bandit state during update operations."""
    A: np.ndarray
    b: np.ndarray
    A_inv: np.ndarray
    timestamp: int
    needs_full_inversion: bool


class DisjointLinUCBPolicy:
    """Disjoint LinUCB: one ridge regression per arm."""
    def __init__(
        self,
        model_names: List[str],
        dim: int = 384,
        alpha: float = 0.1,
        init_lambda: float = 1.0,
        forgetting_factor: float = 1.0,
        seed: int | None = None,
        max_staleness_dt: int = _MAX_STALENESS_DT,
        reg_floor_fraction: float = _REGULARIZATION_FLOOR_FRACTION,
        max_var_inflation: float = _MAX_VAR_INFLATION_FACTOR,
    ):
        """Initialize Disjoint LinUCB policy.

        REGULARIZATION NOTE (isotropic prior after PCA):
        We initialize A₀ = λI, an isotropic regularizer in the PCA-transformed
        feature space.  After PCA, principal components have decreasing empirical
        variance, so equal regularization across all directions does not match
        the per-component scale—effectively over-shrinking low-variance components
        relative to their scale.  This is a deliberate simplicity choice: isotropic
        ridge in a PCA basis is a standard, stable baseline.  Designing anisotropic
        or variance-matched regularization (e.g., diagonal A₀ scaled by component
        variance, or full whitening before the bandit) is a natural extension left
        for future work.

        Args:
            model_names: List of model identifiers (arms).
            dim: Context vector dimension.
            alpha: Exploration coefficient (UCB bonus multiplier).
            init_lambda: Initialization regularization (A₀ = λI). Default 1.0 for
                cold-start stability.
            forgetting_factor: Exponential decay factor (1.0 = stationary,
                <1.0 = adaptive).
            seed: Seed for the internal ``np.random.Generator`` used by
                Thompson Sampling (``get_probabilities``).  *None* creates
                an unseeded generator.
            max_staleness_dt: Cap on ``dt`` in ``gamma^dt`` to prevent float64
                underflow.  Default 1000.
            reg_floor_fraction: Fraction of *init_lambda* below which proactive
                regularization injects a top-up into A.  Default 0.1.
            max_var_inflation: Maximum multiplicative factor for staleness-based
                variance inflation.  Default 200.0.
        """
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(forgetting_factor)
        self.init_lambda = float(init_lambda)
        self._rng = np.random.default_rng(seed)
        self.max_staleness_dt = int(max_staleness_dt)
        self.reg_floor_fraction = float(reg_floor_fraction)
        self.max_var_inflation = float(max_var_inflation)

        self.model_locks: Dict[str, threading.Lock] = {
            m: threading.Lock() for m in self.models
        }
        self._lock = threading.Lock()
        
        # Initialize A=I*init_lambda, b=0
        self.A = {m: np.eye(self.dim) * self.init_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        
        # Precompute A_inv for hot-path speed
        self.A_inv = {m: safe_inv(self.A[m]) for m in self.models}
        
        self.last_update = {m: 0 for m in self.models}  # Track last reward-update step
        self.last_played = {m: 0 for m in self.models}  # Track last selection step
        self.t = 0  # Global time step
        
        # Track per-model regularization floor to keep A well-conditioned under
        # forgetting/decay in low-traffic regimes.
        self.regularization_floor = {m: self.init_lambda for m in self.models}

    def reset_to_tabula_rasa(self) -> None:
        """Reset all learned state to cold-start initialization.

        Discards A, b, and A_inv matrices for every arm, restoring them
        to ``A = init_lambda * I``, ``b = 0``.  Time counters are zeroed
        so the policy re-explores as if no observations had been seen.

        Thread-safe: acquires all per-model locks and the global lock.
        """
        with self._lock:
            for m in self.models:
                with self.model_locks[m]:
                    self.A[m] = np.eye(self.dim) * self.init_lambda
                    self.b[m] = np.zeros(self.dim, dtype=np.float64)
                    self.A_inv[m] = safe_inv(self.A[m])
                    self.last_update[m] = 0
                    self.last_played[m] = 0
                    self.regularization_floor[m] = self.init_lambda
            self.t = 0

    def __deepcopy__(self, memo):
        """
        Custom deepcopy to handle thread locks.
        
        Locks cannot be pickled or deepcopied directly. We create new locks
        for the clone while deepcopying all numerical state (A, b, A_inv, etc.).
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy basic attributes
        result.models = copy.deepcopy(self.models, memo)
        result.dim = self.dim
        result.alpha = self.alpha
        result.gamma = self.gamma
        result.init_lambda = self.init_lambda
        result.max_staleness_dt = self.max_staleness_dt
        result.reg_floor_fraction = self.reg_floor_fraction
        result.max_var_inflation = self.max_var_inflation
        result.t = self.t
        result.last_update = copy.deepcopy(self.last_update, memo)
        result.last_played = copy.deepcopy(self.last_played, memo)

        result.A = copy.deepcopy(self.A, memo)
        result.b = copy.deepcopy(self.b, memo)
        result.A_inv = copy.deepcopy(self.A_inv, memo)

        result.model_locks = {m: threading.Lock() for m in result.models}
        result._lock = threading.Lock()

        result.regularization_floor = copy.deepcopy(self.regularization_floor, memo)
        result._rng = copy.deepcopy(self._rng, memo)

        return result

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically.

        Prepares all state outside the lock, then publishes atomically.
        Uses double-checked locking: the outer ``if`` is a fast-path
        optimization; the inner re-check under ``self._lock`` prevents
        duplicate registration when two threads call ``add_arm`` for the
        same model concurrently.
        """
        if model_name in self.models:
            return

        new_A = np.eye(self.dim) * self.init_lambda
        new_b = np.zeros(self.dim, dtype=np.float64)
        new_A_inv = safe_inv(new_A)

        with self._lock:
            if model_name in self.models:
                return
            self.A[model_name] = new_A
            self.b[model_name] = new_b
            self.A_inv[model_name] = new_A_inv
            self.last_update[model_name] = self.t
            self.last_played[model_name] = self.t
            self.regularization_floor[model_name] = self.init_lambda
            self.model_locks[model_name] = threading.Lock()
            self.models.append(model_name)

    def delete_arm(self, model_name: str) -> None:
        """Remove an arm from the bandit.

        Acquires the model's per-arm lock before ``self._lock`` to
        prevent a race with a concurrent ``update()`` that has already
        acquired ``model_locks[model_name]`` and is reading arm state.
        """
        model_lock = self.model_locks.get(model_name)
        if model_lock is None:
            with self._lock:
                self.models = [m for m in self.models if m != model_name]
            return
        with model_lock:
            with self._lock:
                self.models = [m for m in self.models if m != model_name]
                for attr in (self.A, self.b, self.A_inv, self.last_update,
                             self.last_played, self.regularization_floor):
                    attr.pop(model_name, None)
                self.model_locks.pop(model_name, None)

    def refresh_inverse_cache(self) -> None:
        """Recompute ``A_inv`` for all models after a bulk load.

        Builds the new inverse dict outside the lock, then atomically
        swaps the reference so that concurrent ``update()`` calls (which
        read ``self.A_inv[model]`` under ``model_locks`` only) never see
        a partially populated dictionary.
        """
        with self._lock:
            snapshot = {m: self.A[m] for m in self.models if m in self.A}
        new_A_inv = {m: safe_inv(A_m) for m, A_m in snapshot.items()}
        with self._lock:
            self.A_inv = new_A_inv


    def select_arm(
        self, 
        x: np.ndarray, 
        candidates: List[str] | None = None,
        cost_penalties: Dict[str, float] | None = None,
    ) -> Tuple[str, float]:
        """
        Select the best arm (model) using Upper Confidence Bound (UCB).
        
        Implements paper Eq. 4:
          a_t = argmax (x^T θ_hat + α √(x^T A^{-1} x) - cost_penalty)
        
        **Commensurability Note:**
        The UCB term (mean + α·std) is in reward units. When processing feedback,
        `BanditRouter.process_feedback()` strictly clamps rewards to [0, 1].
        Furthermore, `calibrate_priors()` guarantees that initialized priors
        satisfy |x^T θ_hat| ≤ 0.9 on the calibration suite, and the exploration
        bonus `α·std` is structurally bounded by the feature embedding space
        (PCA-whitened to unit variance) and the Tikhonov regularization `λ`.
        
        Because the quality score is bounded to a known, stable scale (~[0, 1]
        range), the additive cost penalty `λ_c * norm_cost` (where `norm_cost`
        is also in [0, 1]) is provably commensurate. The multiplier `λ_c`
        (e.g., `cost_penalty=0.1`) represents a direct exchange rate:
        "Sacrifice 10% expected reward to choose the cheapest over the most
        expensive model." Early in training (cold start), α·std may exceed 1.0,
        intentionally dominating the penalty to ensure exploration until the
        variance shrinks.
        
        Args:
            x: Context vector
            candidates: List of candidate model IDs (None = all models)
            cost_penalties: Optional per-model cost penalty {model_id: λ * norm_cost}.
                          Subtracted from UCB score at selection time.
            
        Returns:
            Tuple of (best_model_id, best_score)
        """
        ucb_scores: Dict[str, float] = {}

        with self._lock:
            candidates = self.models if candidates is None else candidates
            candidates = [m for m in candidates if m in self.A]
            if not candidates:
                raise ValueError("No candidates available")
            for m in candidates:
                theta = self.A_inv[m] @ self.b[m]
                mean = float(theta.dot(x))

                dt = self._effective_staleness(m)
                var = float(x.dot(self.A_inv[m]).dot(x))
                var_inflated = _inflate_variance(
                    var, self.gamma, dt,
                    max_staleness_dt=self.max_staleness_dt,
                    max_var_inflation=self.max_var_inflation,
                )

                std = float(np.sqrt(max(var_inflated, 1e-12)))
                ucb = mean + self.alpha * std

                if cost_penalties and m in cost_penalties:
                    ucb -= cost_penalties[m]

                ucb_scores[m] = ucb

        best_model = _argmax_random_tiebreak(ucb_scores)
        return best_model, float(ucb_scores[best_model])

    def _effective_staleness(self, model: str) -> int:
        """Delegate to module-level :func:`_effective_staleness`."""
        return _effective_staleness(self.t, self.last_update, self.last_played, model)

    def mark_selected(self, model: str) -> None:
        """Record that *model* was deployed this round (for staleness tracking).

        Called at selection time so that ``_effective_staleness`` can
        distinguish "no update yet because feedback is delayed" from "arm
        genuinely unused."

        **Time Advancement (Request Counter):**
        This method also increments the global logical clock `self.t` by 1.
        By advancing time at selection (route time) rather than at feedback
        time, `self.t` acts as a *request counter* rather than a *feedback counter*.
        This ensures that when a burst of delayed feedback arrives (e.g., 100
        RLHF ratings at once), the updates do not artificially inflate the
        decay factor `gamma ** dt` for each other. Time correctly represents
        the number of environmental interactions, not the processing speed.
        """
        with self._lock:
            self.t += 1
            self.last_played[model] = self.t

    def get_expected_reward(self, model: str, x: np.ndarray) -> float:
        """Expected reward for *model* given context *x*.

        Computes ``x^T theta_hat`` where ``theta_hat = A_inv @ b``.

        Args:
            model: Model identifier.
            x: Context feature vector.

        Returns:
            Scalar expected reward (may exceed [0, 1] before clamping).
        """
        theta = self.A_inv[model] @ self.b[model]
        return float(theta.dot(x))

    def get_ucb_variance(self, model: str, x: np.ndarray) -> float:
        """UCB variance term for *model* given context *x*.

        Computes ``x^T A_inv x`` with staleness-based inflation when the
        forgetting factor is active (gamma < 1).

        Args:
            model: Model identifier.
            x: Context feature vector.

        Returns:
            Variance term (non-negative).  Take ``sqrt`` for the UCB bonus.
        """
        var = float(x.dot(self.A_inv[model]).dot(x))
        dt = self._effective_staleness(model)
        return _inflate_variance(
            var, self.gamma, dt,
            max_staleness_dt=self.max_staleness_dt,
            max_var_inflation=self.max_var_inflation,
        )

    def get_probabilities(self, x: np.ndarray, models: List[str], n_samples: int = 1000,
                          noise_variance: float = 0.25) -> Dict[str, float]:
        """
        Probability each model has the highest *quality* (expected reward).

        Samples from the Bayesian posterior for ridge regression::

            θ | D ~ N(A⁻¹b,  σ² · A⁻¹_eff)

        where ``A⁻¹_eff`` incorporates staleness inflation (see
        ``_effective_staleness``).  The model whose posterior draw yields
        the largest ``θᵀx`` wins a sample; probabilities are the empirical
        win fractions across *n_samples* draws.

        **Important:** These probabilities reflect the *quality-only* reward
        model.  Cost and latency penalties applied by ``select_arm()`` are
        **not** incorporated.  Use this for posterior calibration, monitoring,
        and explainability — not as a substitute for the full utility-based
        selection rule.

        Args:
            x: Context vector.
            models: List of model IDs to compare.
            n_samples: Number of Monte Carlo samples (default: 1000).
            noise_variance: σ² for the posterior covariance.  Default 0.25 is
                the variance of a Bernoulli(0.5) reward, appropriate for
                binary win/loss feedback.  Override for non-binary rewards.

        Returns:
            Dictionary mapping model_id to probability of being the
            highest-quality arm (sums to 1.0).
        """
        model_samples = {}
        valid_models = [m for m in models if m in self.A]
        
        snapshots = {}
        with self._lock:
            for m in valid_models:
                A_inv_m = self.A_inv[m].copy()
                theta_hat = A_inv_m @ self.b[m]
                dt = self._effective_staleness(m)
                snapshots[m] = (A_inv_m, theta_hat, dt)
        
        # Return uniform distribution instead of all-zeros when no
        # models have initialized state.  All-zeros violates the probability
        # contract (sum should be 1.0) and can cause division-by-zero in callers.
        if not snapshots:
            n = len(models) or 1
            return {m: 1.0 / n for m in models}
        
        for m, (A_inv_m, theta_hat, dt) in snapshots.items():
            scalar_var = float(np.trace(A_inv_m))
            inflated = _inflate_variance(
                scalar_var, self.gamma, dt,
                max_staleness_dt=self.max_staleness_dt,
                max_var_inflation=self.max_var_inflation,
            )
            if scalar_var > 0 and inflated != scalar_var:
                cov = noise_variance * A_inv_m * (inflated / scalar_var)
            else:
                cov = noise_variance * A_inv_m
            samples = _safe_multivariate_normal(
                theta_hat, cov, n_samples, self.dim, rng=self._rng,
            )
            model_samples[m] = samples @ x
            
        # Determine how many times each model was the winner across samples
        stacked_samples = np.stack([model_samples[m] for m in valid_models])
        winners = np.argmax(stacked_samples, axis=0)
        
        counts = Counter(winners)
        probs = {m: 0.0 for m in models}
        for i, m in enumerate(valid_models):
            probs[m] = counts[i] / n_samples
        return probs

    
    # Snapshot-swap helper methods removed - replaced with simple per-model locking
    # This eliminates the lost update race condition identified in conference review
    
    def update(self, model: str, x: np.ndarray, reward: float, weight: float = 1.0, advance_time: bool = True) -> None:
        """
        Update the model's A and b matrices with new observation.
        
        **Per-Model Locking**
        Replaced snapshot-swap pattern with fine-grained locking to eliminate
        lost update race condition. Each model has its own lock, so updates to
        Model A don't block updates to Model B.
        
        **Proactive Regularization Floor**
        Tracks effective lambda decay and proactively maintains eigenvalue floor.
        Prevents singularity in low-traffic regimes with forgetting factor < 1.0.
        Amortized O(d²) with rare O(d³) maintenance cycles.
        
        **Performance:**
        Sherman-Morrison update is O(d²) ≈ 0.5ms for d=33, negligible compared
        to network latency. Holding lock during update is acceptable.
        
        **Time Convention (`advance_time`):**
        `self.t` is a *request counter*, not a *feedback counter*.  It is
        advanced at route/selection time by ``mark_selected()``, so that a
        burst of delayed feedback arriving at once (e.g., a daily RLHF batch)
        does not artificially inflate ``gamma**dt`` decay for each feedback
        event processed in the batch.

        When called from ``process_feedback()`` (the standard online path),
        pass ``advance_time=False`` because time was already advanced by
        ``mark_selected()`` at route time.

        When called directly for offline/batch replay (i.e., without a
        preceding ``route()`` call), leave ``advance_time=True`` (the default)
        so each replayed observation still increments time.
        
        Args:
            model: Model identifier
            x: Context vector
            reward: Observed reward
            weight: Importance weight for this update (default 1.0).
                    Use weight = (1 - cluster_mu) for difficulty-based weighting.
                    Hard tasks (μ=0.5) get weight=0.5, easy tasks (μ=0.95) get weight=0.05.
            advance_time: Whether to increment ``self.t``. Defaults to True.
                    Set to False when time was already advanced at route time via
                    ``mark_selected()`` to prevent double-counting.
        """
        if model not in self.A:
            return
        
        # Guard against negative weight, which would produce NaN
        # via np.sqrt(negative) in the Sherman-Morrison u = x * sqrt(w) path,
        # permanently corrupting the model's A_inv with NaN values.
        if weight < 0:
            logger.warning(f"Negative weight={weight:.4f} for {model}; skipping update (negative weight would corrupt A_inv via sqrt(w))")
            return
        
        # weight=0 contributes zero information (x_outer=0, reward_x=0)
        # but advancing self.t inflates dt for ALL other models' staleness
        # computations, artificially increasing their exploration bonuses.
        if weight == 0:
            return
        
        # Hold model-specific lock for entire update (eliminates lost-update race).
        # Per-model locks allow concurrent updates to different arms; the global
        # self._lock is acquired in short nested sections below for self.t reads
        # and matrix pointer swaps.
        with self.model_locks[model]:
            with self._lock:
                if model not in self.A:
                    return
                current_t = self.t

            # 1. Calculate Time Decay
            dt = 0
            decay_factor = 1.0
            if self.gamma < 1.0:
                dt = current_t - self.last_update[model]
                # Clamp dt to prevent numerical underflow when gamma is small
                decay_factor = self.gamma ** min(dt, self.max_staleness_dt)

            current_lambda = self.regularization_floor.get(model, self.init_lambda)
            new_lambda = current_lambda * decay_factor
            lambda_threshold = self.init_lambda * self.reg_floor_fraction

            if new_lambda < lambda_threshold:
                # MAINTENANCE MODE: Inject fresh regularization (Rare O(d³))
                #
                # WHEN this triggers:
                #   Only when gamma < 1.0 AND the arm has gone so long without
                #   an update that gamma^dt < 0.1 — i.e., >90% of the original
                #   prior has been forgotten.  For gamma=0.99 this requires
                #   dt > 230 steps; for gamma=0.95 it requires dt > 44 steps.
                #   In any normally-trafficked deployment this is a rare edge
                #   case.  With the default gamma=1.0 this branch is never
                #   reached.
                #
                # WHAT it does:
                #   Decays A and b equally, then reinjects missing_lambda * I
                #   into A only.  Because theta_hat = A^{-1} b, adding lambda*I
                #   to A while leaving b unchanged shrinks theta_hat toward zero
                #   — the standard Tikhonov/ridge shrinkage effect at rate
                #   O(lambda / (lambda + n_eff)).
                #
                # WHY this shrinkage is intentional and safe:
                #   1. Well-observed arms: data eigenvalues in A dominate lambda
                #      (n_eff >> 1), so |Δtheta| ≈ lambda/n_eff is negligible.
                #   2. Lightly-observed arms: after >90% forgetting the old data
                #      is nearly irrelevant; regressing to the zero-mean prior
                #      is the statistically correct posterior.
                #   3. After a maintenance reset the arm's UCB exploration bonus
                #      increases (lower data mass → higher A_inv → higher UCB),
                #      driving immediate re-exploration that rebuilds theta_hat.
                #
                # The 10% threshold is deliberately conservative: maintenance
                # mode fires only after extreme staleness, keeping the fast
                # O(d²) path dominant during normal operation.
                logger.info(
                    f"[FIX] Maintenance: Restoring regularization floor for {model} "
                    f"(λ_eff={new_lambda:.2e} < {lambda_threshold:.2e})"
                )
                
                missing_lambda = self.init_lambda - new_lambda
                
                new_A = (self.A[model] * decay_factor) + (missing_lambda * np.eye(self.dim))
                new_b = self.b[model] * decay_factor
                new_A_inv = safe_inv(new_A)
                
                self.regularization_floor[model] = self.init_lambda
                
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
                    self.last_update[model] = current_t  # use snapped clock
                
            else:
                # STANDARD MODE: Fast Decay (Common O(d²))
                if self.gamma < 1.0:
                    self.regularization_floor[model] = new_lambda  # Update tracker
                    new_A = self.A[model] * decay_factor
                    new_b = self.b[model] * decay_factor
                    # Update A_inv to match decayed A.
                    # Since A_new = A_old * gamma^dt, the correct inverse is:
                    #   A_inv_new = A_inv_old / gamma^dt
                    # Without this, Sherman-Morrison below applies its rank-1
                    # correction to a stale pre-decay inverse, causing the cached
                    # A_inv to drift arbitrarily far from the true inv(A) over
                    # successive updates.  O(d²) scalar division — no perf hit.
                    new_A_inv = self.A_inv[model] / decay_factor

                    with self._lock:
                        self.A[model] = new_A
                        self.b[model] = new_b
                        self.A_inv[model] = new_A_inv
                        self.last_update[model] = current_t  # use snapped clock
            
            # 3. Rank-1 Sherman-Morrison Update (Data Integration)
            result = _sherman_morrison_update(
                A=self.A[model],
                A_inv=self.A_inv[model],
                b=self.b[model],
                x=x,
                reward=reward,
                weight=weight,
                init_lambda=self.init_lambda,
                regularization_floor=self.regularization_floor.get(model, self.init_lambda),
                model_name=model,
                reg_floor_fraction=self.reg_floor_fraction,
            )
            self.regularization_floor[model] = result.regularization_floor

            with self._lock:
                self.A[model] = result.A
                self.b[model] = result.b
                self.A_inv[model] = result.A_inv
                if advance_time:
                    self.t += 1


    def _check_numerical_stability(self, model: str, config: 'RouterConfig' = None) -> None:
        """
        Safety check for numerical stability using trace of inverse.
        
        Eigenvalue decomposition is O(d³) ≈ 20ms, causing
        1-second P99 latency spikes with 50 models. Use trace instead.
        
        **Mathematical Insight**: If A decays toward singularity (λ → 0),
        then A^{-1} eigenvalues → ∞, so trace(A^{-1}) → ∞.
        
        **Cost**: O(d) - just summing diagonal elements
        **Trigger**: Only when trace(A_inv) > threshold (rare)
        **Frequency**: Every N updates (e.g., 1000)
        
        Args:
            model: Model identifier to check
            config: RouterConfig with stability thresholds (optional)
        """
        if config is None or model not in self.A_inv:
            return
        
        # O(d) operation: compute trace(A_inv)
        trace = np.trace(self.A_inv[model])
        
        # Check if inverse is exploding (matrix approaching singularity)
        # Default threshold: 1000 * d (well-conditioned trace ≈ d)
        threshold = getattr(config, 'stability_threshold', 1000 * self.dim)
        
        if trace > threshold:
            logger.warning(
                f"[GUARD] Numerical instability detected for {model}: "
                f"trace(A_inv)={trace:.2e} > {threshold:.2e}. "
                f"Triggering regularization reset."
            )
            
            # Acquire per-model lock, then global lock, before
            # mutating A and A_inv.  Without locking, a concurrent select_arm()
            # could read A_inv between the += and the safe_inv(), observing an
            # inconsistent (A, A_inv) pair.  NumPy releases the GIL during
            # matrix operations, making this a real race even in CPython.
            # Use self.init_lambda (the bandit's own regularization
            # parameter) instead of config.init_lambda.  config may carry a different
            # value (e.g., from a stale or mis-matched RouterConfig), while
            # self.init_lambda is the authoritative regularization strength that was
            # used to initialize this bandit's A matrices.
            reg_lambda = self.init_lambda
            with self.model_locks[model]:
                self.A[model] += reg_lambda * np.eye(self.dim)
                new_A_inv = safe_inv(self.A[model])
                with self._lock:
                    self.A_inv[model] = new_A_inv
                    # Update regularization_floor tracker to reflect the
                    # injected regularization.  Without this, the forgetting-factor
                    # code under-estimates how much lambda has been added.
                    self.regularization_floor[model] = self.regularization_floor.get(
                        model, self.init_lambda
                    ) + reg_lambda
                    # Snapshot the trace while still holding the global lock so
                    # the sanity-check log reflects the value we just wrote, not
                    # a potentially stale read that a concurrent update could have
                    # modified between here and the next line.
                    new_trace = np.trace(new_A_inv)

            logger.info(
                f"[OK] Regularization reset complete for {model}. "
                f"New trace(A_inv)={new_trace:.2f}"
            )



    def save_state(self, path: Path | str) -> None:
        """Save A, b, and temporal metadata to a compressed NPZ file.

        Persists sufficient statistics (A, b) and temporal metadata
        (t, last_update, last_played, regularization_floor) so that
        forgetting-factor and staleness logic resume correctly after
        a checkpoint restore.
        """
        data: Dict[str, Any] = {
            "_metadata_dim": self.dim,
            "_metadata_models": list(self.models),
            "_metadata_policy": "disjoint",
            "__temporal__t": self.t,
        }
        last_update_arr = np.array(
            [self.last_update.get(m, 0) for m in self.models], dtype=np.int64
        )
        last_played_arr = np.array(
            [self.last_played.get(m, 0) for m in self.models], dtype=np.int64
        )
        reg_floor_arr = np.array(
            [self.regularization_floor.get(m, self.init_lambda)
             for m in self.models],
            dtype=np.float64,
        )
        data["__temporal__last_update"] = last_update_arr
        data["__temporal__last_played"] = last_played_arr
        data["__temporal__reg_floor"] = reg_floor_arr

        for m in self.models:
            data[f"{m}_A"] = self.A[m]
            data[f"{m}_b"] = self.b[m]
        np.savez_compressed(path, **data)

    def load_state(self, path: Path | str) -> None:
        """
        Load A and b matrices from a compressed NPZ file with dimension validation.

        All file I/O, validation, and ``safe_inv`` computation happen
        outside any lock.  State references are then swapped atomically
        under ``self._lock`` so that concurrent readers (``select_arm``,
        ``get_probabilities``) never observe torn state (e.g. a new ``A``
        paired with an old ``A_inv``).

        Raises:
            ValueError: If saved dimension doesn't match current bandit dimension.
                       Suggests clearing state or updating feature configuration.
        """
        data = np.load(path)

        if '_metadata_dim' in data:
            saved_dim = int(data['_metadata_dim'])
            if saved_dim != self.dim:
                raise ValueError(
                    f"Dimension mismatch: saved state has dim={saved_dim}, "
                    f"but current bandit expects dim={self.dim}. "
                    f"This can happen when:\n"
                    f"  1. PCA configuration changes (raw embeddings vs PCA-compressed)\n"
                    f"  2. Virtual anchor set is modified\n"
                    f"  3. Feature engineering pipeline changes\n"
                    f"To fix:\n"
                    f"  - Delete the saved state file to start fresh, OR\n"
                    f"  - Ensure PCA and feature config match the saved state"
                )
        else:
            logger.warning(
                f"Loading state from {path} without dimension metadata. "
                f"This may cause issues if dimensions have changed. "
                f"Current dim={self.dim}"
            )

        # Stage 1: load, validate, and compute inverses outside any lock.
        staged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for m in self.models:
            a_key = f"{m}_A"
            b_key = f"{m}_b"
            if a_key in data and b_key in data:
                A_loaded = data[a_key]
                b_loaded = data[b_key]

                if A_loaded.shape != (self.dim, self.dim):
                    raise ValueError(
                        f"Matrix A for model '{m}' has wrong shape: "
                        f"expected ({self.dim}, {self.dim}), got {A_loaded.shape}"
                    )
                if b_loaded.shape != (self.dim,):
                    raise ValueError(
                        f"Vector b for model '{m}' has wrong shape: "
                        f"expected ({self.dim},), got {b_loaded.shape}"
                    )
                staged[m] = (A_loaded, b_loaded, safe_inv(A_loaded))

        # Restore temporal metadata (outside lock).
        saved_t: int | None = None
        saved_last_update: Dict[str, int] = {}
        saved_last_played: Dict[str, int] = {}
        saved_reg_floor: Dict[str, float] = {}
        if "__temporal__t" in data:
            saved_t = int(data["__temporal__t"])
        if "__temporal__last_update" in data:
            arr = data["__temporal__last_update"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_last_update[m] = int(arr[i])
        if "__temporal__last_played" in data:
            arr = data["__temporal__last_played"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_last_played[m] = int(arr[i])
        if "__temporal__reg_floor" in data:
            arr = data["__temporal__reg_floor"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_reg_floor[m] = float(arr[i])

        # Stage 2: swap all references atomically under the global lock.
        with self._lock:
            for m, (A_new, b_new, A_inv_new) in staged.items():
                self.A[m] = A_new
                self.b[m] = b_new
                self.A_inv[m] = A_inv_new
            if saved_t is not None:
                self.t = saved_t
            for m in self.models:
                if m in saved_last_update:
                    self.last_update[m] = saved_last_update[m]
                if m in saved_last_played:
                    self.last_played[m] = saved_last_played[m]
                if m in saved_reg_floor:
                    self.regularization_floor[m] = saved_reg_floor[m]


# ---------------------------------------------------------------------------
# Main Router Class
# ---------------------------------------------------------------------------

@dataclass
class RoutingLog:
    request_id: str
    timestamp_s: float
    prompt: str
    selected_model: str
    predicted_utility: float
    cost_usd: float
    latency_s: float
    context_vector: np.ndarray | None = None # Cached embedding for updates
    expected_reward: float = 0.0             # θᵀx at route time (for drift detection residuals)
    total_priority_weight: float = 1.0       # Sum of w_q, w_c, w_l for normalization
    pacer_lambda_t: float | None = None      # BudgetPacer dual variable at route time
    pacer_cost_ema: float | None = None      # BudgetPacer cost EMA at route time

class BanditRouter:
    """
    The primary entry point for routing.
    """
    # --- VIRTUAL ANCHORS (Zero-Shot) ---
    # Declarative semantic landmarks using natural language descriptions.
    # Replaces the data-dependent "Anchor Cluster ID" system.
    DEFAULT_VIRTUAL_ANCHORS = {
        "coding": "Python code programming software engineering script development computer science",
        "math": "mathematics arithmetic calculus equations reasoning proof algebra geometry",
        "creative": "creative writing poetry fiction storytelling narrative prose",
        "jokes": "humor jokes comedy funny wit sarcasm riddles",
        "reasoning": "step-by-step reasoning logic puzzle analysis critical thinking deduction"
    }
    
    # Heuristic seeds for generating a Complexity Vector if missing
    HARD_REASONING_SEEDS = [
        "complex mathematical proof", "advanced algorithmic optimization",
        "system architecture design", "quantum physics derivation",
        "intricate logic puzzle", "technical debugging",
        "multi-step analytical reasoning", "scientific research analysis"
    ]

    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        # Feature extraction (The Eyes) - now injectable
        feature_service: 'FeatureService | None' = None,
        # Legacy params for backward compatibility
        context_model: str = DEFAULT_CONTEXT_MODEL,
        context_encoder=None,
        pca_path: Path | str | None = None,
        # Bandit parameters (The Brain)
        alpha: float = 0.1,
        embedding_dim: int = 384,
        init_lambda: float = 1.0,
        forgetting_factor: float = 1.0,
        context_store: ContextStore | None = None,
        config: RouterConfig | None = None,
        verbose_routing: bool = False,
        cost_penalty: float = 0.3,  # λ_c for UCB cost penalty (paper Eq. 4)
        latency_penalty: float = 0.0,  # λ_l for UCB latency penalty
        drift_threshold: float = 0.0,
        drift_method: str = "centroid",
        drift_burn_in_steps: int = 50,
        drift_ema_alpha: float = 0.05,
        drift_confirmation_window: int = 20,
        budget_pacer: "BudgetPacer | None" = None,
    ):
        """
        Initialize BanditRouter with separated feature extraction.
        
        **Architectural Separation (Eyes, Brain, Memory):**
        - FeatureService (The Eyes): Feature extraction (or legacy fallback)
        - RouterCore (The Brain): LinUCB selection
        - FeedbackLoop (The Memory): Matrix updates
        
        Args:
            model_registry: Dictionary of model configurations
            feature_service: Optional FeatureService instance for custom feature extraction.
                           If None, falls back to legacy default service using context_model/pca_path.
            context_model: Encoder model name (used if feature_service=None)
            context_encoder: Pre-initialized encoder (legacy, overrides context_model)
            pca_path: Path to PCA model (used if feature_service=None)
            alpha: Exploration coefficient for UCB
            embedding_dim: Dimension override (auto-detected if feature_service provided)
            init_lambda: Initialization regularization (A₀ = λI)
            forgetting_factor: Temporal decay (1.0 = stationary)
            context_store: Persistent storage for delayed feedback
            config: Router configuration object
            verbose_routing: Enable detailed breakdown logs for each routing decision
            cost_penalty: λ_c for UCB cost penalty (paper Eq. 4). At selection
                       time, each arm's score includes -λ_c·normalized_cost(model).
                       Does NOT affect learned quality estimates — only biases
                       selection toward cheaper models.
                       - 0.0 = quality-only
                       - 0.3 = moderate cost awareness (default)
                       - 0.5+ = aggressive cost preference
            latency_penalty: λ_l for UCB latency penalty. Mirrors cost_penalty
                       but for latency. Each arm's score includes
                       -λ_l·normalized_latency(model). Normalized to [0, 1]
                       using the same log-scale market anchor approach as cost.
                       - 0.0 = no latency preference (default, backward-compatible)
                       - 0.1 = mild preference for faster models
                       - 0.3 = moderate latency awareness
            drift_threshold: Sigma-based threshold for automatic covariate
                       shift detection on prompt embeddings.  When the
                       drift score exceeds
                       ``baseline + threshold * baseline_std``
                       for a sustained confirmation window, the router
                       performs a **cold-start reset**: all bandit matrices
                       (A, b) are cleared to initial values and the
                       drift detector re-enters burn-in on the new traffic.
                       This detect → reset → re-learn cycle repeats if a
                       subsequent shift is detected.
                       - 0 = disabled (default, backward-compatible)
                       - 1.5 = sensitive
                       - 2.0 = conservative
            drift_method: Detection algorithm.  ``"centroid"`` (default) uses
                       running-centroid cosine distance — sensitive to
                       topic/domain rotations in embedding space.
                       ``"chi2"`` uses the legacy diagonal chi-squared test
                       on per-component z-scores.
            drift_burn_in_steps: Observations used to establish the embedding
                       distribution baseline (default: 50).
            drift_ema_alpha: Smoothing factor for the drift detector's EMA
                       (default: 0.05, half-life ≈ 14 observations).
            drift_confirmation_window: Consecutive above-threshold steps
                       required before drift is confirmed (default: 20).
            budget_pacer: Optional :class:`BudgetPacer` instance for online
                       budget pacing (Primal-Dual CBwK).  When provided,
                       injects adaptive cost constraints (hard ceiling
                       and/or soft penalty) into each routing decision and
                       updates pacing state in ``process_feedback()``.
                       ``None`` (default) disables pacing entirely.
        """
        self.config = config or RouterConfig()
        self.verbose_routing = verbose_routing
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.drift_threshold = drift_threshold
        self.drift_method = drift_method
        self.drift_burn_in_steps = drift_burn_in_steps
        self.drift_ema_alpha = drift_ema_alpha
        self.drift_confirmation_window = drift_confirmation_window
        self._drift_adapted = False
        self.drift_detector = None
        self.budget_pacer = budget_pacer

        if model_registry is None:
            # Load default models.json from config/
            base_dir = Path(__file__).parent
            models_path = base_dir / "config" / "models.json"
            if not models_path.exists():
                logger.warning(f"Default models.json not found at {models_path}. Initializing with empty registry.")
                model_registry = {}
            else:
                import json
                with open(models_path) as f:
                    data = json.load(f)
                model_registry = {m["model_id"]: m for m in data["models"]}

        # Copy the registry defensively. We intentionally avoid retaining
        # references to caller-owned nested dicts because we normalise and add
        # derived fields (e.g., blended_cost_per_m). A shallow copy would mutate
        # the caller's objects and can create test-order dependence.
        self.registry = {k: dict(v) for k, v in model_registry.items()}

        self._resolve_registry_costs()
        
        # -----------------------------------------------------------------------
        # FEATURE SERVICE (The Eyes) - Dependency Injection
        # -----------------------------------------------------------------------
        if feature_service is not None:
            # Use provided service (custom feature engineering)
            self.features = feature_service
            logger.info("Using injected FeatureService")
        else:
            # Create default service from legacy parameters
            # Feature extraction is delegated to FeatureService
            # Dimension is auto-detected from PCA file (PCA components + 1 bias)
            from .feature_service import FeatureService as FS
            self.features = FS(
                encoder_model=context_model,
                pca_path=pca_path,
                allow_jit_training=True
            )
            logger.info(f"Created default FeatureService with encoder={context_model}")
        
        # Backward-compatible aliases.  We avoid eagerly touching
        # features.encoder / features.pca here — that would force a
        # multi-GB download even for users who injected a lightweight
        # FeatureService or use pre-computed vectors.
        self._encoder_resolved = False
        self._pca_resolved = False
        
        # Calculate dimension dynamically from feature service
        # Default is 33 (32 PCA + 1 bias) with pca_32.joblib
        embedding_dim = self.features.dimension
        
        logger.debug(f"Feature dimensions: total={embedding_dim} (including bias)")
        
        # Initialize bandit with calculated dimension.
        # NOTE: Features are [PCA_0, ..., PCA_{d-2}, bias].  We apply PCA for
        # dimensionality reduction and initialize the covariance/prior as
        # A₀ = λI, which corresponds to an isotropic regularizer in the PCA
        # space; this choice does not distinguish between high- and low-variance
        # components and may over-regularize the latter.  See the
        # DisjointLinUCBPolicy docstring for further discussion.
        model_ids = list(self.registry.keys())
        self.bandit = DisjointLinUCBPolicy(
            model_ids,
            dim=embedding_dim,
            alpha=alpha,
            init_lambda=init_lambda,
            forgetting_factor=forgetting_factor,
        )
        
        self._toxicity_scanner = None


        # ---------------------------------------------------------------------------
        # Tiered Context Storage
        # ---------------------------------------------------------------------------
        # Default: SqliteContextStore (production, zero dependencies, 7-day TTL)
        # Alternative: EphemeralContextStore (testing, RAM-only, 100s horizon)
        self.context_store = context_store or SqliteContextStore()
        logger.info(f"Context store: {type(self.context_store).__name__}")

        # ---------------------------------------------------------------------------
        # Production Stability: Bounded Log Buffer
        # ---------------------------------------------------------------------------
        # Using deque with maxlen prevents unbounded memory growth.
        # At 100 QPS with ~500 bytes/log, 10k entries ≈ 5MB max footprint.
        # Oldest logs are automatically evicted when buffer is full.
        # IMPORTANT: process_feedback() must be called before log is evicted!
        # Use instance config, not class default, so custom
        # RouterConfig(max_log_size=...) is respected.
        self.logs: deque[RoutingLog] = deque(maxlen=self.config.max_log_size)
        # Parallel index for O(1) feedback lookups
        self.log_index: Dict[str, RoutingLog] = {}
        # Lock to protect the logs/log_index pair from concurrent writes
        self._log_lock = threading.Lock()
        self.model_priors: Dict[str, float] = {} 
        
        # Feature name to index mapping for Progressive Registration
        self._feature_map = self._build_feature_map()
        
        # Precompute market anchors to avoid redundant log calls in hot loop
        self._market_cost_floor = self.config.market_cost_floor
        self._market_cost_floor_log = np.log(self.config.market_cost_floor)
        self._market_cost_range = self.config.cost_range_log
        
        self._market_lat_floor = self.config.market_latency_floor
        self._market_lat_floor_log = np.log(self.config.market_latency_floor)
        self._market_lat_range = self.config.latency_range_log

    def _resolve_registry_costs(self) -> None:
        """Ensure every model in ``self.registry`` has a ``blended_cost_per_m``.

        Resolves from available fields (legacy keys, input/output split, or
        pessimistic defaults).  Raises :class:`MissingCostError` for partial
        schemas (only input or only output) that are likely registry bugs.

        Mutates ``self.registry`` in place.
        """
        for m_id, m_data in self.registry.items():
            if "time_to_first_token_seconds" not in m_data and "median_latency_s" in m_data:
                try:
                    m_data["time_to_first_token_seconds"] = float(m_data["median_latency_s"])
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid median_latency_s for model '%s': %r. "
                        "Leaving time_to_first_token_seconds unset.",
                        m_id,
                        m_data.get("median_latency_s"),
                    )

            if "blended_cost_per_m" in m_data:
                continue
            if "price_1m_blended" in m_data:
                m_data["blended_cost_per_m"] = float(m_data["price_1m_blended"])
                continue
            if "cost_per_1m_tokens" in m_data:
                m_data["blended_cost_per_m"] = float(m_data["cost_per_1m_tokens"])
                m_data.setdefault("input_cost_per_m", float(m_data["blended_cost_per_m"]))
                m_data.setdefault("output_cost_per_m", float(m_data["blended_cost_per_m"]))
                continue

            inp = m_data.get("input_cost_per_m")
            out = m_data.get("output_cost_per_m")
            inp_valid = isinstance(inp, (int, float))
            out_valid = isinstance(out, (int, float))

            if inp_valid and out_valid:
                m_data["blended_cost_per_m"] = (float(inp) + float(out)) / 2.0
            elif inp_valid and not out_valid:
                raise MissingCostError(
                    f"Model '{m_id}' has input_cost_per_m=${inp}/M but is missing "
                    f"output_cost_per_m. Provide both or set blended_cost_per_m directly."
                )
            elif out_valid and not inp_valid:
                raise MissingCostError(
                    f"Model '{m_id}' has output_cost_per_m=${out}/M but is missing "
                    f"input_cost_per_m. Provide both or set blended_cost_per_m directly."
                )
            else:
                fallback = float(getattr(self.config, "default_missing_cost_per_m", 10.0))
                m_data["blended_cost_per_m"] = fallback
                m_data.setdefault("input_cost_per_m", fallback)
                m_data.setdefault("output_cost_per_m", fallback)

    def __deepcopy__(self, memo):
        """Custom deepcopy for BanditRouter to handle unpicklable components.

        Strategy:
        1. SHARE stateless / lock-containing objects (encoder, features, context_store)
        2. DEEPCOPY all mutable state (bandit, logs, counters, config)
        3. COPY scalar / immutable values directly
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # --- Configuration (deepcopy to isolate) ---
        result.config = copy.deepcopy(self.config, memo)
        result.registry = copy.deepcopy(self.registry, memo)
        
        # --- Feature Service (SHARE: stateless, contains locks & GPU state) ---
        result.features = self.features
        result._encoder_resolved = self._encoder_resolved
        result._pca_resolved = self._pca_resolved
        
        # --- Bandit Policy (deepcopy: has its own __deepcopy__ for locks) ---
        result.bandit = copy.deepcopy(self.bandit, memo)
        
        result.cost_penalty = self.cost_penalty
        result.latency_penalty = self.latency_penalty
        
        # --- Logs and Counters (deepcopy: mutable collections) ---
        result.logs = copy.deepcopy(self.logs, memo)
        result.log_index = copy.deepcopy(self.log_index, memo)
        result._log_lock = threading.Lock()  # Fresh lock for clone
        result.model_priors = copy.deepcopy(self.model_priors, memo)
        
        # --- Scalar / Immutable Settings (direct copy) ---
        result.verbose_routing = self.verbose_routing
        result._feature_map = copy.deepcopy(self._feature_map, memo)
        result._toxicity_scanner = None  # Lazy-init; don't share
        
        # --- Precomputed Market Anchors (scalars) ---
        result._market_cost_floor = self._market_cost_floor
        result._market_cost_floor_log = self._market_cost_floor_log
        result._market_cost_range = self._market_cost_range
        result._market_lat_floor = self._market_lat_floor
        result._market_lat_floor_log = self._market_lat_floor_log
        result._market_lat_range = self._market_lat_range
        
        # --- Context Store (SHARE: DB connection, thread-safe) ---
        result.context_store = self.context_store
        
        return result


    def _build_feature_map(self) -> Dict[str, int]:
        """
        Build a mapping from feature names to vector indices.
        
        This enables the Progressive Registration API to translate human-friendly
        feature names (e.g., 'anchor_coding', 'complexity_score') into the exact
        indices within the theta vector.
        
        Returns:
            Dictionary mapping feature name to index in the context vector
        """
        feature_map = {}
        
        embedding_dim = self.features.dimension - 1  # exclude bias
        
        # PCA components  
        for i in range(embedding_dim):
            feature_map[f"pca_{i}"] = i
        
        # Bias term (always last)
        feature_map["bias"] = embedding_dim
        
        return feature_map

    def register_model(
        self,
        model_id: str,
        speed: SpeedProfile = "balanced",
        cost_usd: float = None,
        latency_s: float = None,
        blended_cost_per_m: float = None,
        initial_weights: Optional[Dict[str, float]] = None,
        strict_kwargs: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Universal entry point for adding models with Progressive Registration.

        Combines basic user knowledge with bandit math. This method translates
        human-friendly inputs (speed profiles like "fast") into the mathematical
        priors (theta vectors) needed by LinUCB.

        **Two Tiers of Knowledge:**

        **Tier A: T-Shirt Sizing** - "I know cost/speed but not priors"
            speed="fast" sets positive bias (cheap -> use by default)
            speed="slow" sets positive bias (expensive -> reserve for hard tasks)

        **Tier B: Agnostic** - "I have no information"
            Just model_id initializes with neutral priors and high variance

        **Power User Override:**
            initial_weights={"complexity_score": 3.0} for explicit control

        Args:
            model_id: Unique model identifier
            speed: T-shirt speed profile ("fast", "balanced", "slow")
            cost_usd: Input cost in $/M tokens (used with output estimate to
                     derive blended cost if ``blended_cost_per_m`` is not set)
            latency_s: Time-to-first-token in seconds
            blended_cost_per_m: Weighted average cost in $/M tokens for hard
                              constraint filtering.  If not provided, derived
                              from ``cost_usd`` (treated as input, output
                              estimated as 3x input).
            initial_weights: Explicit feature weight overrides for power users
            strict_kwargs: Override for unknown-kwarg validation. If ``None``,
                          uses ``RouterConfig.registration_strict_kwargs``.
            **kwargs: Accepted for backward compatibility (e.g. ``capabilities``).
                     Unknown keys raise ``TypeError`` in strict mode.

        Raises:
            MissingCostError: If ``blended_cost_per_m`` is None and ``cost_usd``
                            is also None (cannot derive a blended cost).

        Examples:
            # Local Llama: Fast and general purpose
            router.register_model("llama-3-8b", speed="fast",
                                  blended_cost_per_m=0.2)

            # Specialist: Slow but great at coding
            router.register_model("deepseek-coder", speed="slow",
                                  blended_cost_per_m=2.0)

            # Mystery model: No information
            router.register_model("model-x", speed="balanced", blended_cost_per_m=5.0)
        """
        strict_mode = (
            self.config.registration_strict_kwargs
            if strict_kwargs is None else strict_kwargs
        )
        known_kwargs = {"capabilities"}
        unknown_kwargs = set(kwargs.keys()) - known_kwargs
        if unknown_kwargs:
            unknown_list = ", ".join(sorted(unknown_kwargs))
            if strict_mode:
                raise TypeError(
                    f"register_model() got unknown keyword argument(s): {unknown_list}. "
                    f"Allowed extra kwargs: {sorted(known_kwargs)}"
                )
            logger.warning(
                "Ignoring unknown register_model kwargs for '%s': %s",
                model_id,
                unknown_list,
            )

        capabilities = kwargs.get("capabilities", [])
            
        if model_id in self.bandit.models:
            logger.warning(f"[WARN] Model {model_id} already registered. Skipping.")
            return
        
        # 1. Initialize zero state (the canvas)
        weights = {}
        bias = 0.0
        
        # 2. Apply T-Shirt Sizing (The Bias Term)
        # Use Speed/Cost as prior for "Default Mode" when no warmup priors exist.
        # NOTE: complexity_weight fields were removed when the feature pipeline
        # was simplified to [PCA | bias].  T-shirt sizing now operates solely
        # through the bias dimension.
        reg_config = self.config.registration
        
        if speed == "fast":
            bias = reg_config.fast_bias
        elif speed == "slow":
            bias = reg_config.slow_bias
        else:  # balanced
            bias = reg_config.balanced_bias
        
        # 4. Apply Power User Overrides (Explicit Weights)
        # If the user DOES know specifics, let them overwrite our guesses
        if initial_weights:
            for k, v in initial_weights.items():
                weights[k] = v
        
        # 5. Compile into Theta Vector (The Math)
        dim = self.bandit.dim
        theta_vector = np.zeros(dim, dtype=np.float64)
        
        # Fill the bias term (explicit indexing)
        theta_vector[self.features.bias_index] = bias
        
        # Map dictionary keys to vector indices
        for feature_name, val in weights.items():
            if feature_name in self._feature_map:
                idx = self._feature_map[feature_name]
                theta_vector[idx] = val
            else:
                logger.warning(f"Unknown feature '{feature_name}' in initial_weights. Skipping.")
        
        # 6. Initialize bandit arm with T-shirt prior
        #
        # All new models start with A = λI (identity-scaled precision) and
        # b = λ·θ (prior encoding from T-shirt sizing / capabilities).
        self.bandit.add_arm(model_id)

        # Inject T-shirt prior into arm-specific b vector
        new_b = self.bandit.init_lambda * theta_vector
        with self.bandit._lock:
            self.bandit.b[model_id] = new_b
            
        # 8. Prepare registry entry (but don't publish yet)
        # Use defaults from config if not provided
        if cost_usd is None:
            cost_usd = reg_config.default_cost_per_1m
        if latency_s is None:
            latency_s = reg_config.default_latency_s

        if blended_cost_per_m is None:
            # cost_usd is guaranteed non-None here: the block above falls back to
            # reg_config.default_cost_per_1m when not explicitly provided.
            output_est = cost_usd * _OUTPUT_COST_MULTIPLIER
            blended_cost_per_m = (cost_usd + output_est) / 2.0
        else:
            # Back-derive output cost from the caller-provided blended cost
            # so that _get_normalized_cost (which reads input/output from the
            # registry) stays consistent with the blended figure.
            output_est = 2.0 * blended_cost_per_m - cost_usd

        registry_entry = {
            "cost_per_1m_tokens": cost_usd,
            "input_cost_per_m": cost_usd,
            "output_cost_per_m": output_est,
            "blended_cost_per_m": float(blended_cost_per_m),
            "time_to_first_token_seconds": latency_s,
            "median_latency_s": latency_s,
            "capabilities": capabilities,
            "speed_profile": speed,
        }
        
        # 9. Publish to registry — model is now fully initialized
        self.registry[model_id] = registry_entry
        
        boost_summary = ", ".join(f"{k}={v:.1f}" for k, v in list(weights.items())[:5])
        if len(weights) > 5:
            boost_summary += "..."
        
        logger.info(
            f"[OK] Registered {model_id} | "
            f"Bias: {bias:.1f} | "
            f"Boosts: {boost_summary} | "
            f"Cost: ${cost_usd:.2f}/1M | "
            f"Latency: {latency_s:.2f}s"
        )


    # ---------------------------------------------------------------------------
    # Tier 1 Safety: Fast Toxicity Heuristic
    # ---------------------------------------------------------------------------
    
    # Feature and Context Extraction (Delegated to FeatureService)
    # ---------------------------------------------------------------------------
    
    def _get_context_vector(self, prompt: str | np.ndarray) -> np.ndarray:
        """
        Extract features via the FeatureService.
        
        Args:
            prompt: Input text or pre-encoded vector
            
        Returns:
            Normalized feature vector [PCA, bias]
        """
        return self.features.extract_features(prompt)

    @property
    def reference_model(self) -> Dict[str, Any]:
        """
        Dynamically identifies the 'Flagship' model to use as a baseline reference.
        
        **Selection Criteria:**
        The model with the **highest initial_quality score** in the current registry.
        This ensures the reference point adapts automatically when you upgrade your
        model portfolio (e.g., adding GPT-5).
        
        Returns:
            Dictionary containing flagship model metadata with keys:
                - id: Model identifier (string)
                - initial_quality: Quality score (float, typically 0.0-1.0 range)
                - input_cost_per_m: Cost in $/million tokens (float)
                - output_cost_per_m: Cost in $/million tokens (float)
                - ... (other registry metadata)
        
        Example:
            >>> router = BanditRouter.create()
            >>> ref = router.reference_model
            >>> print(f"Current flagship: {ref['id']} (Quality: {ref.get('initial_quality', 0):.3f})")
            Current flagship: google/gemini-exp-1206 (Quality: 0.348)
        """
        if not self.registry:
            # Fallback if registry is empty (should never happen in production)
            logger.warning("Registry is empty, using fallback reference model")
            return {
                "id": "fallback-flagship",
                "initial_quality": 1.0,
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 10.0
            }
            
        # Find the model with the maximum quality score
        champion_id = max(
            self.registry,
            key=lambda m: self.registry[m].get("initial_quality", 0.0)
        )
        
        # Return a copy of the registry entry with the ID included
        data = dict(self.registry[champion_id])
        data["id"] = champion_id
        return data


    @classmethod
    def create(
        cls,
        model_registry: Dict[str, Any] | None = None,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        priors: str = "warmup",
        prior_n_effective: float = 5000.0,
        **kwargs
    ) -> "BanditRouter":
        """Factory method to create a fully initialized router.

        Args:
            model_registry: Dictionary of model configurations.
            context_model: Model to use for embedding generation.
            priors: Prior initialization strategy. ``"warmup"`` (default) loads
                the shipped K=3 warmup priors for an informed cold-start.
                ``"none"`` starts with standard LinUCB cold-start (identity
                covariance + quality-based bias).  Pass a path to a ``.joblib``
                file to load custom priors generated via
                :func:`generate_warmup_priors`.
            prior_n_effective: Effective sample count attributed to loaded
                priors.  Controls how strongly the offline priors are trusted:
                ``scale = prior_n_effective / n_warmup`` where ``n_warmup`` is
                the number of samples the priors were trained on.  Default
                5000.0 with 80k warmup data gives scale = 6.25%, meaning the
                priors contribute as if they were 5000 real observations.
                Higher values trust priors more (slower adaptation); lower
                values trust them less (faster override by online evidence).
            **kwargs: Additional arguments passed to __init__ or prior loading
        
        Returns:
            Fully initialized BanditRouter instance
        """
        # 1. Extract factory-specific arguments (not passed to __init__)
        state_path = kwargs.pop("state_path", None)
        warmup_path = kwargs.pop("warmup_path", None)
        
        # Legacy support: map old 'exploration' parameter to 'alpha'
        exploration = kwargs.pop("exploration", None)
        alpha = kwargs.pop("alpha", None)
        
        if alpha is None and exploration is not None:
            alpha = ExplorationRate.get(exploration)
        elif alpha is None:
            alpha = 0.1

        # 2a. Guard: custom encoder with explicit warmup priors path
        #     must have matching priors (encoder embedding space must match)
        _using_custom_encoder = context_model != DEFAULT_CONTEXT_MODEL
        _wants_file_priors = (
            isinstance(priors, str)
            and priors not in ("none",)
            and (priors.endswith(".joblib") or "/" in priors)
        )
        if _using_custom_encoder and _wants_file_priors and warmup_path is None:
            logger.info(
                f"Loading priors for custom encoder '{context_model}'. "
                f"Ensure the priors were generated with the same encoder."
            )

        # 2. Filter kwargs to only include those accepted by __init__
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = sig.parameters.keys()
        init_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # 3. Initialize the Router (Standard)
        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=alpha,
            **init_kwargs
        )
        
        # 4. Load Priors from explicit path or shipped default
        if priors == "warmup" and warmup_path is None:
            from .config import DEFAULT_WARMUP_PRIORS_PATH
            if DEFAULT_WARMUP_PRIORS_PATH.exists():
                warmup_path = str(DEFAULT_WARMUP_PRIORS_PATH)
            else:
                logger.warning(
                    "Shipped warmup priors not found at %s; "
                    "falling back to cold-start.",
                    DEFAULT_WARMUP_PRIORS_PATH,
                )
        _priors_path_str = warmup_path or (priors if priors != "none" else None)
        if isinstance(_priors_path_str, str) and (
            _priors_path_str.endswith(".joblib") or "/" in _priors_path_str
        ):
            priors_path = Path(_priors_path_str)
            if priors_path and priors_path.exists():
                import joblib
                warmup_data = joblib.load(priors_path)
                # -----------------------------------------------------------------
                # Feature-space compatibility: PCA whitening
                # -----------------------------------------------------------------
                # FeatureService may whiten PCA coordinates at runtime.  Warmup
                # priors generated before whitening (or with whitening disabled)
                # are in a different coordinate system.  For diagonal whitening
                # x_new = D x_old, the sufficient statistics transform as:
                #   A_new = D A_old D,   b_new = D b_old.
                # We apply this conversion once at load time so users can keep
                # older prior artifacts without silent scale mismatch.
                try:
                    scales = None
                    if hasattr(router.features, "get_pca_whitening_scales"):
                        scales = np.asarray(router.features.get_pca_whitening_scales(), dtype=np.float64)
                    # Determine whether the router's *actual feature vectors*
                    # are whitened (could be via PCA.whiten=True or external scaling).
                    router_whitens = False
                    if scales is not None and scales.shape[0] >= 2:
                        router_whitens = not np.allclose(scales[:-1], 1.0)

                    priors_whitened = bool(warmup_data.get("pca_whitened", False))
                    if scales is not None and priors_whitened != router_whitens:
                        if priors_whitened and not router_whitens:
                            # Convert whitened priors -> unwhitened router space.
                            scales = 1.0 / np.maximum(scales, 1e-12)
                        # Else: unwhitened priors -> whitened router space uses scales as-is.
                        warmup_data = dict(warmup_data)  # shallow copy to avoid mutating shared object
                        A_map = warmup_data.get("A", {})
                        b_map = warmup_data.get("b", {})
                        if isinstance(A_map, dict) and isinstance(b_map, dict):
                            A_new = {}
                            b_new = {}
                            for m in A_map:
                                if m not in b_map:
                                    continue
                                A_m = np.asarray(A_map[m], dtype=np.float64)
                                b_m = np.asarray(b_map[m], dtype=np.float64)
                                if A_m.shape[0] == scales.shape[0] and A_m.shape[1] == scales.shape[0]:
                                    A_new[m] = A_m * scales.reshape(-1, 1) * scales.reshape(1, -1)
                                else:
                                    A_new[m] = A_m
                                if b_m.shape[0] == scales.shape[0]:
                                    b_new[m] = b_m * scales
                                else:
                                    b_new[m] = b_m
                            warmup_data["A"] = A_new
                            warmup_data["b"] = b_new
                            warmup_data["pca_whitened"] = router_whitens
                            logger.info(
                                "🔄 Converted warmup priors PCA whitening: "
                                f"priors_whitened={priors_whitened} -> router_whitens={router_whitens}"
                            )
                except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
                    logger.warning(
                        f"Warmup priors whitening compatibility conversion failed: {exc}. "
                        "Proceeding without conversion (may degrade performance)."
                    )
                # Guard against n=0 in warmup file (ZeroDivisionError).
                # .get("n", 20000) protects against missing key, but not zero value.
                n_warmup = max(warmup_data.get("n", 20000), 1)
                scale = prior_n_effective / float(n_warmup)
                
                missing_models = []
                for model_id in router.bandit.models:
                    # Layer 1: Try Robust Offline Priors
                    if (model_id in warmup_data.get("A", {})) and (model_id in warmup_data.get("b", {})):
                        router.bandit.A[model_id] = warmup_data["A"][model_id] * scale
                        router.bandit.b[model_id] = warmup_data["b"][model_id] * scale
                    # Layer 2: Gap-Filling (Cascading Fallback)
                    else:
                        missing_models.append(model_id)
                        model_data = router.registry.get(model_id, {})
                        
                        A_heuristic, b_heuristic = get_heuristic_prior(
                            model_data=model_data,
                            dim=router.bandit.dim,
                            init_lambda=router.bandit.init_lambda,
                            n_effective=prior_n_effective
                        )
                        router.bandit.A[model_id] = A_heuristic
                        router.bandit.b[model_id] = b_heuristic
                
                if missing_models:
                    logger.warning(
                        f"[WARN] Warmup Partial Miss: {len(missing_models)} models not in joblib. "
                        f"Applied heuristic initialization for: {missing_models}"
                    )
                else:
                    logger.info("[OK] Warmup Complete: All models initialized from offline priors.")
                
                # =====================================================================
                # POST-WARMUP REGULARIZATION: Bayesian Shrinkage Toward Zero
                # =====================================================================
                # We deliberately add λI to A without adjusting b.  Since
                # θ = A⁻¹b, increasing A without increasing b shrinks θ toward
                # the origin at rate O(λ / (λ + n_eff)), where n_eff is the
                # effective sample count encoded in the warmup priors.
                #
                # This is intentional — NOT a bug:
                #
                #   1. Safety valve against mismatched priors.  The 80k offline
                #      battle priors may come from a different traffic distribution
                #      than the deployment environment.  Without shrinkage, a strong
                #      but wrong prior could lock the router into suboptimal
                #      selections for many rounds.
                #
                #   2. Online evidence always wins.  Each online observation adds
                #      xx^T to A and reward·x to b, so the warmup's contribution
                #      to θ naturally dilutes.  The post-warmup λI accelerates
                #      this dilution, ensuring the system remains responsive.
                #
                #   3. Defense in depth.  The forgetting factor (γ) provides
                #      exponential discounting of stale observations.
                #
                # Net effect: numerical stability + controlled prior decay.
                # =====================================================================
                for model_id in router.bandit.models:
                    router.bandit.A[model_id] += np.eye(router.bandit.dim) * router.bandit.init_lambda
                
                # Single refresh_inverse_cache() after all A matrices are
                # finalized.  Previously there were two calls — one before and one
                # after the regularization loop — wasting O(K·d³) at startup.
                router.bandit.refresh_inverse_cache()
                logger.info(f"[OK] Applied post-warmup regularization (λ={router.bandit.init_lambda}) from {priors_path}")

                # Activate embedding-based covariate drift detection now
                # that priors are loaded.  The detector monitors the prompt
                # embedding distribution to catch distribution shift —
                # meaningful only with priors (no point detecting shift
                # when learning from scratch).
                if router.drift_threshold > 0:
                    from bandit_gpt.drift import CentroidDriftDetector, DriftDetector
                    detector_kwargs = dict(
                        threshold=router.drift_threshold,
                        burn_in_steps=router.drift_burn_in_steps,
                        ema_alpha=router.drift_ema_alpha,
                        confirmation_window=router.drift_confirmation_window,
                    )
                    if router.drift_method == "centroid":
                        router.drift_detector = CentroidDriftDetector(**detector_kwargs)
                    else:
                        router.drift_detector = DriftDetector(**detector_kwargs)
                    logger.info(
                        "[OK] Embedding drift detection enabled "
                        "(method=%s, threshold=%.1fσ, burn_in=%d, "
                        "ema_alpha=%.3f, confirm=%d).",
                        router.drift_method,
                        router.drift_threshold,
                        router.drift_burn_in_steps,
                        router.drift_ema_alpha,
                        router.drift_confirmation_window,
                    )
            else:
                logger.warning(f"[WARN] Priors file not found at {priors_path}. Using cold start.")
        
        if router.drift_detector is None and router.drift_threshold > 0:
            logger.info(
                "Drift detection requires warmup priors; skipping "
                "(drift_threshold=%.1fσ has no effect without priors).",
                router.drift_threshold,
            )

        # =====================================================================
        # LAYER 3: T-SHIRT SIZING INJECTION (Business Logic)
        # =====================================================================
        # Warm-Start Architecture:
        # - Layer 1: User-supplied priors (if provided) → Already loaded above
        # - Layer 2 (HERE): T-shirt sizing → Business logic on top of data
        #
        # This layer applies human-provided speed profile priors (fast/slow)
        # *on top* of data-driven warmup priors, with proper confidence scaling.
        #
        # Why confidence scaling?
        # After warmup, b[bias] might be ~1000 (from 80k battles).
        # Naive: b[bias] += 0.5 → 1000.5 (0.05% change, negligible)
        # Scaled: b[bias] += confidence × 0.5 → 1500 (50% change, meaningful)
        #
        # Mathematical justification:
        # θ[bias] = b[bias] / A[bias, bias]
        # To shift θ by Δ: b_new = b_old + A[bias, bias] × Δ
        #
        # Example:
        # - Fast model (Mixtral): bias_shift = +0.5 → Encourage selection
        # - Slow model (GPT-4): bias_shift = -0.5 → Reserve for hard tasks
        reg_config = router.config.registration
        bias_idx = router.features.bias_index
        
        logger.info("💉 Layer 3: Injecting T-Shirt Sizing biases into warmed-up state...")
        
        for model_id in router.bandit.models:
            # Check model speed profile from registry
            speed = router.registry.get(model_id, {}).get("speed_profile", "balanced")
            
            # Determine Shift Amount
            bias_shift = 0.0
            if speed == "fast":
                bias_shift = reg_config.fast_bias      # e.g., +0.5
            elif speed == "slow":
                bias_shift = reg_config.slow_bias      # e.g., -0.5 or -1.0
            
            if abs(bias_shift) > 0.0:
                # Correct matrix algebra for theta shift.
                # To shift theta by delta * e_i (unit vector in dimension i):
                #   b_new = b_old + delta * A[:, i]   (entire i-th column of A)
                #
                # Previously, only the diagonal element was used:
                #   b[i] += A[i,i] * delta
                # This missed off-diagonal contributions A[j,i] for j != i,
                # leaking the bias shift into PCA dimensions proportional to
                # A_inv[j, bias] — contaminating learned feature preferences.
                injection_col = bias_shift * router.bandit.A[model_id][:, bias_idx]
                router.bandit.b[model_id] += injection_col
                
                logger.debug(
                    f"   - {model_id} ({speed}): Bias {bias_shift} "
                    f"-> Added ||{np.linalg.norm(injection_col):.2f}|| to b-vector."
                )

        # 6. Refresh inverse cache to be safe (though b-update doesn't strictly require it)
        router.bandit.refresh_inverse_cache()
        
        # 6b. Calibrate priors on the canonical bandit state (catches scale
        #     explosion from warmup or T-shirt sizing before the adapter sees it).
        calibrate_priors(router.bandit, target_max_pred=0.9)

        # 7. Load state if provided (overwrites any priors applied above)
        if state_path:
            router.load_state(state_path)
                
        return router





    # ---------------------------------------------------------------------------
    # Self-Healing PCA (JIT Calibration)
    # ---------------------------------------------------------------------------
    
    def _generate_synthetic_data(self, n: int = 1000) -> List[str]:
        """
        Generate synthetic prompts for PCA calibration.
        
        Uses the same archetypes as procedural warmup to ensure consistency
        between PCA manifold and warmup covariance structure.
        
        Args:
            n: Number of synthetic prompts to generate (default: 1000)
               For robust PCA, need ~10x the target dimensionality (32 dims → ~320 samples)
               
        Returns:
            List of synthetic prompt strings
        """
        import random
        
        # Template patterns matching procedural warmup archetypes
        templates = {
            "math": [
                "Solve the integral of {expr} with respect to {var}",
                "Prove that {theorem} using mathematical induction",
                "Find the derivative of {function} and explain each step",
                "Calculate the eigenvalues of the matrix {matrix}",
                "Determine if the series {series} converges or diverges"
            ],
            "coding": [
                "Write a Python function to {task} using {library}",
                "Implement {algorithm} in {language} with time complexity analysis",
                "Debug this {language} code that {problem}",
                "Create a {language} class for {task} with unit tests",
                "Optimize this {algorithm} implementation for {constraint}"
            ],
            "reasoning": [
                "Analyze the logical structure of {argument} and identify fallacies",
                "Develop a step-by-step solution for {problem}",
                "Compare and contrast {concept_a} with {concept_b}",
                "Explain the causal relationship between {cause} and {effect}",
                "Evaluate the validity of {claim} given {evidence}"
            ],
            "creative": [
                "Write a {genre} story about {topic} in {style}",
                "Compose a poem about {subject} using {form}",
                "Create a dialogue between {character_a} and {character_b} about {topic}",
                "Describe {scene} from the perspective of {viewpoint}",
                "Develop a plot outline for a {genre} involving {element}"
            ],
            "chat": [
                "What is {simple_concept} and why is it important?",
                "Can you explain {topic} in simple terms?",
                "Tell me about {subject}",
                "Why does {phenomenon} happen?",
                "What's the difference between {concept_a} and {concept_b}?"
            ]
        }
        
        # Fill placeholders with variations
        fill_values = {
            "expr": ["x^2 + 3x + 2", "sin(x)cos(x)", "e^(2x)", "ln(x^2)"],
            "var": ["x", "y", "t", "theta"],
            "theorem": ["Fermat's Last Theorem", "the Pythagorean identity", "Euler's formula"],
            "function": ["f(x) = x^3 + 2x", "g(x) = sqrt(x+1)", "h(x) = e^x / x"],
            "matrix": ["[[1,2],[3,4]]", "a 3x3 identity matrix", "[[2,-1],[4,3]]"],
            "series": ["sum(1/n^2)", "sum((-1)^n/n)", "sum(1/n!)"],
            "task": ["parse JSON", "sort a list", "find duplicates", "merge dictionaries"],
            "library": ["pandas", "numpy", "requests", "pathlib"],
            "algorithm": ["binary search", "quicksort", "dijkstra's", "BFS"],
            "language": ["Python", "JavaScript", "Java", "C++"],
            "problem": ["throws TypeError", "has memory leak", "returns wrong output"],
            "constraint": ["memory", "speed", "readability"],
            "argument": ["this logical claim", "the premise that AI is conscious"],
            "concept_a": ["AI", "machine learning", "neural networks"],
            "concept_b": ["automation", "deep learning", "decision trees"],
            "cause": ["climate change", "urbanization", "technology adoption"],
            "effect": ["sea level rise", "habitat loss", "social transformation"],
            "claim": ["this hypothesis", "the assertion", "the theory"],
            "evidence": ["the data", "experimental results", "historical records"],
            "genre": ["science fiction", "mystery", "romance", "thriller"],
            "topic": ["time travel", "AI", "space exploration", "ancient civilizations"],
            "style": ["Hemingway's style", "a humorous tone", "dark and moody"],
            "subject": ["autumn", "technology", "love", "nature"],
            "form": ["haiku", "sonnet", "free verse"],
            "character_a": ["a scientist", "an AI", "a detective"],
            "character_b": ["a philosopher", "a child", "a criminal"],
            "scene": ["a futuristic city", "a quiet forest", "a busy marketplace"],
            "viewpoint": ["a bird", "an alien observer", "a time traveler"],
            "element": ["time loops", "parallel universes", "mind reading"],
            "simple_concept": ["photosynthesis", "gravity", "democracy"],
            "phenomenon": ["rain", "lightning", "the aurora borealis"]
        }
        
        prompts = []
        rng = random.Random(42)

        archetype_keys = list(templates.keys())
        for _ in range(n):
            archetype = rng.choice(archetype_keys)
            template = rng.choice(templates[archetype])

            prompt = template
            for placeholder, values in fill_values.items():
                if f"{{{placeholder}}}" in prompt:
                    prompt = prompt.replace(f"{{{placeholder}}}", rng.choice(values))

            prompts.append(prompt)
        
        return prompts
    

    def _procedural_warmup(self, n_samples: int = 50):

        """
        Shape the covariance matrix A using synthetic archetypal prompts.
        
        Delegates to utils.procedural_warmup for actual warmup logic.
        See utils/warmup.py for full implementation details.
        
        Args:
            n_samples: Number of synthetic samples (default: 50)
        """
        procedural_warmup(self, n_samples=n_samples)



    # prune_arms removed - trusting UCB confidence bounds to naturally downweight bad models
    # Bad models get minimal traffic (~0.001%) without explicit pruning


    # _detect_difficulty_score removed - feature engineering should be done externally
    # The router is now a pure "Decision Engine"






    # -------------------------------------------------------------------------
    # Helper Methods for route() - Atomicity Refactoring
    # -------------------------------------------------------------------------
    
    def _build_routing_features(self, prompt: str | np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Build context vector with embeddings, features, and anchors.
        
        Args:
            prompt: Input prompt (string or pre-embedded vector)
            
        Returns:
            Tuple of (context_vector, prompt_text)
        """
        prompt_text = prompt if isinstance(prompt, str) else "[Pre-embedded Prompt]"
        # Delegate to FeatureService (The Eyes)
        x = self.features.extract_features(prompt)
        return x, prompt_text
    
    
    def _filter_by_constraints(
        self,
        candidates: List[str],
        max_cost: float | None,
        max_latency: float | None,
        quality_floor: Dict[str, float | None] | None,
    ) -> List[str]:
        """
        Apply hard constraints (cost, latency, quality floor).

        Cost filtering interprets ``max_cost`` as a unit price ceiling in
        ``$/1k tokens`` (as documented in the README). Each model's
        ``blended_cost_per_m`` (stored in ``$/M``) is converted to ``$/1k`` by
        dividing by 1000.

        Latency filtering uses ``time_to_first_token_seconds`` from the registry.
        All constraints are enforced on actual registry metadata, not predictions.

        Raises:
            NoEligibleModelsError: If no candidate passes all constraints.
                The exception message lists every candidate with the specific
                reason(s) it was excluded.

        Args:
            candidates: List of candidate model IDs
            max_cost: Maximum blended price in ``$/1k tokens`` (optional)
            max_latency: Maximum time-to-first-token in seconds (optional)
            quality_floor: Minimum quality scores per metric (optional)

        Returns:
            List of models passing all constraints
        """
        filtered = []
        reasons: Dict[str, List[str]] = {}

        for m in candidates:
            m_data = self.registry.get(m, {})
            m_reasons: List[str] = []

            if max_cost is not None:
                blended_m = m_data.get("blended_cost_per_m")
                if blended_m is not None:
                    blended_per_1k = float(blended_m) / 1000.0
                    if blended_per_1k > max_cost:
                        m_reasons.append(
                            f"blended_cost=${blended_per_1k:.6f}/1k > max_cost=${max_cost:.6f}/1k"
                        )

            if max_latency is not None:
                lat = m_data.get("time_to_first_token_seconds")
                if lat is not None and isinstance(lat, (int, float)) and float(lat) > max_latency:
                    m_reasons.append(
                        f"latency={float(lat):.3f}s > max_latency={max_latency:.3f}s"
                    )

            if quality_floor:
                scores = m_data.get("scores", {})
                for k, v in quality_floor.items():
                    if v is None:
                        continue
                    actual = float(scores.get(k, 0))
                    if actual < v:
                        m_reasons.append(f"{k}={actual:.3f} < floor={v:.3f}")

            if m_reasons:
                reasons[m] = m_reasons
            else:
                filtered.append(m)

        if not filtered:
            raise NoEligibleModelsError(reasons, max_cost, max_latency, quality_floor)

        return filtered
    
    
    def _create_routing_log(
        self,
        prompt_text: str,
        model: str,
        utility: float,
        x: np.ndarray,
        input_tokens: int,
        output_tokens: int,
        total_weight: float = 1.0
    ) -> RoutingLog:
        """
        Create and persist routing log.
        
        Args:
            prompt_text: Input prompt text
            model: Selected model ID
            utility: Predicted utility score
            x: Context vector (cached for feedback loop)
            input_tokens: Input token count
            output_tokens: Output token count
            
        Returns:
            RoutingLog object
        """
        log = RoutingLog(
            # Use uuid4 instead of time.time_ns() to avoid
            # request_id collisions at high QPS (clock resolution varies by OS).
            request_id=str(uuid.uuid4()),
            timestamp_s=time.time(),
            prompt=prompt_text,
            selected_model=model,
            predicted_utility=float(utility),
            cost_usd=self._estimate_cost(model, input_tokens, output_tokens),
            latency_s=self._estimate_latency(model, output_tokens),
            context_vector=x,  # Cache for feedback loop
            total_priority_weight=total_weight
        )
        # Protect deque eviction + append + index write with a lock.
        # Without this, two concurrent route() calls could both check len(self.logs),
        # evict the same entry, or leave log_index pointing at evicted logs.
        # Use `is not None` instead of truthiness for maxlen.
        # deque(maxlen=0) has maxlen=0 which is falsy, causing `0 or inf` = inf,
        # skipping eviction while deque silently drops items → unbounded log_index.
        with self._log_lock:
            if self.logs.maxlen is not None and len(self.logs) >= self.logs.maxlen:
                if len(self.logs) > 0:
                    old_log = self.logs[0]
                    self.log_index.pop(old_log.request_id, None)
            self.logs.append(log)
            # Only index if deque actually kept it (maxlen > 0)
            if self.logs.maxlen is None or self.logs.maxlen > 0:
                self.log_index[log.request_id] = log
        
        return log

    def route(
        self,
        prompt: str | np.ndarray,
        *,
        profile: str | Dict[str, float] = "auto",  # Deprecated, ignored
        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] = None,
        input_tokens: int | None = None,
        output_tokens: int = 600,
        total_steps: int = 1,  # Deprecated, ignored
    ) -> Tuple[str, RoutingLog]:
        """
        Route a prompt to the best model using LinUCB with cost/latency penalties.
        
        Raises:
            NoEligibleModelsError: If no models pass the hard constraints.
        
        Args:
            prompt: Input text or pre-embedded vector
            max_cost: Hard budget ceiling in ``$/1k tokens``. Compared against
                each model's registry price (derived from ``blended_cost_per_m``).
            max_latency: Hard latency ceiling (seconds), compared against each
                        model's ``time_to_first_token_seconds``
            quality_floor: Minimum quality scores per metric (e.g.
                          ``{"hle": 0.7}``)
            input_tokens: Input token count (auto-estimated if None)
            output_tokens: Expected output tokens (default 600)
            total_steps: Deprecated, ignored. Retained for backward
                compatibility.
        
        Returns:
            Tuple of (selected_model_id, routing_log)
        """
        # Build features and apply constraints
        x, prompt_text = self._build_routing_features(prompt)

        # Covariate drift detection: feed the context vector BEFORE arm
        # selection so the router adapts proactively (no reward needed).
        if self.drift_detector is not None:
            self.drift_detector.update(x)
            if self.drift_detector.is_drifting:
                self._n_resets = getattr(self, "_n_resets", 0) + 1
                pre_reset_score = self.drift_detector.drift_score
                self.bandit.reset_to_tabula_rasa()
                self.drift_detector.reset()
                self._drift_adapted = True
                logger.info(
                    "[DRIFT] Covariate shift detected — resetting to "
                    "tabula rasa (reset #%d, drift_score=%.4f, "
                    "threshold=%.1fσ). Bandit matrices cleared; "
                    "drift detector re-entering burn-in.",
                    self._n_resets,
                    pre_reset_score,
                    self.drift_threshold,
                )

        # Budget pacing: compute effective cost ceiling and soft penalties.
        effective_max_cost = max_cost
        extra_cost_penalties: Dict[str, float] | None = None
        _pacer_ceiling_relaxed = False

        if self.budget_pacer is not None and self.budget_pacer.uses_hard:
            max_model_cost_per_1k = max(
                float(self.registry[m].get("blended_cost_per_m", 0)) / 1000.0
                for m in self.registry
            )
            ceiling = self.budget_pacer.get_cost_ceiling_per_1k(
                max_model_cost_per_1k
            )
            if ceiling is not None:
                effective_max_cost = (
                    min(effective_max_cost, ceiling)
                    if effective_max_cost is not None
                    else ceiling
                )

        candidates = list(self.registry.keys())
        try:
            filtered = self._filter_by_constraints(
                candidates,
                effective_max_cost,
                max_latency,
                quality_floor,
            )
        except NoEligibleModelsError:
            if effective_max_cost is not max_cost:
                _pacer_ceiling_relaxed = True
                logger.warning(
                    "[PACER] Hard ceiling $%.6f/1k excluded all models; "
                    "relaxing to user max_cost=%s.",
                    effective_max_cost,
                    max_cost,
                )
                filtered = self._filter_by_constraints(
                    candidates,
                    max_cost,
                    max_latency,
                    quality_floor,
                )
            else:
                raise

        # Estimate tokens for logging and cost_usd estimates.
        in_tok = input_tokens or estimate_tokens_rough(prompt_text)

        if self.budget_pacer is not None and self.budget_pacer.uses_soft:
            model_costs = {
                m: self._get_normalized_cost(m) for m in filtered
            }
            extra_cost_penalties = self.budget_pacer.get_extra_cost_penalties(
                model_costs
            )

        # LinUCB selection with optional cost+latency penalty (paper Eq. 4)
        cp = None
        if self.cost_penalty > 0 or self.latency_penalty > 0:
            cp = {}
            for m in filtered:
                p = 0.0
                if self.cost_penalty > 0:
                    p += self.cost_penalty * self._get_normalized_cost(m)
                if self.latency_penalty > 0:
                    p += self.latency_penalty * self._get_normalized_latency(m)
                cp[m] = p
        if extra_cost_penalties is not None:
            cp = cp or {}
            for m, pen in extra_cost_penalties.items():
                cp[m] = cp.get(m, 0.0) + pen
        best_model, best_utility = self.bandit.select_arm(
            x, candidates=filtered, cost_penalties=cp
        )
        
        total_weight = 1.0

        self.bandit.mark_selected(best_model)
        
        # Create routing log
        log = self._create_routing_log(
            prompt_text, best_model, best_utility, x, in_tok, output_tokens, total_weight
        )
        if self.budget_pacer is not None:
            log.pacer_lambda_t = self.budget_pacer.lambda_t
            log.pacer_cost_ema = self.budget_pacer.cost_ema

        self.context_store.save_context(log.request_id, x, best_model)
        
        return best_model, log

    def route_and_call(
        self,
        prompt: str | np.ndarray,
        client: "LLMClient",
        *,
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **route_kwargs,
    ) -> Tuple[str, str, "RoutingLog"]:
        """Route a prompt and call the selected model in one step.

        Parameters:
            prompt: The prompt text (also used for routing features).
            client: Any object satisfying the ``LLMClient`` protocol
                    (see ``bandit_gpt.providers``).
            messages: Chat messages to send.  Defaults to a single user
                      message containing *prompt* (when *prompt* is a string).
            max_tokens: Passed to ``client.complete()``.
            temperature: Passed to ``client.complete()``.
            **route_kwargs: Forwarded to ``route()`` (e.g. *max_cost*,
                            *max_latency*).

        Returns:
            ``(model_id, response_text, routing_log)``
        """
        model_id, log = self.route(prompt, **route_kwargs)
        if messages is None:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError(
                    "When prompt is a pre-computed feature vector, "
                    "you must pass explicit messages."
                )
        response = client.complete(
            model_id, messages, max_tokens=max_tokens, temperature=temperature
        )
        return model_id, response, log

    def process_feedback(
        self,
        request_id: str,
        reward: float,
    ) -> None:
        """
        Process feedback for a previous routing decision.

        Looks up the stored context vector for *request_id*, clamps the reward
        to [0, 1], and performs a LinUCB update on the bandit.

        **Reward clamping**: Values outside [0, 1] are clipped silently.

        **Delayed feedback (RLHF)**: If the in-memory log has been evicted,
        the method falls back to the ``SqliteContextStore``.  Feedback can
        arrive hours or days after routing as long as the context has not
        expired (default TTL: 7 days).

        Args:
            request_id: The ``RoutingLog.request_id`` returned by ``route()``.
            reward: Observed quality signal in [0, 1].  Values outside this
                range are clamped.  Typical sources: LLM-as-judge score,
                user thumbs-up/down (0 or 1), or normalised task metric.

        Raises:
            No exceptions are raised.  If *request_id* is unknown (evicted
            from both in-memory log and persistent store), a warning is
            logged and the call is a no-op.
        """
        # O(1) lookup via parallel index instead of O(N) linear scan
        log = self.log_index.get(request_id)
        
        # Fallback to context_store for delayed feedback (RLHF)
        if log is None:
            context, model_id, _stored_token = self.context_store.get_context(
                request_id
            )
            if context is None:
                logger.warning(f"Context not found for request_id={request_id}")
                return
            log = RoutingLog(
                request_id=request_id, timestamp_s=time.time(),
                prompt="[Delayed Feedback]", selected_model=model_id,
                predicted_utility=0.0, cost_usd=0.0, latency_s=0.0,
                context_vector=context,
            )
        
        # Reject non-finite rewards (NaN/inf) that would corrupt IPW estimates.
        if not np.isfinite(reward):
            logger.warning(
                "process_feedback: non-finite reward=%s for request_id=%s; "
                "skipping update.",
                reward, request_id,
            )
            return

        reward = float(np.clip(reward, 0.0, 1.0))

        # Use cached context vector to avoid re-encoding
        x = log.context_vector if log.context_vector is not None else self._get_context_vector(log.prompt)
        
        self.bandit.update(log.selected_model, x, reward, advance_time=False)
        
        if self.budget_pacer is not None:
            self.budget_pacer.observe(log.cost_usd)

        # Periodic stability check (cheap O(d) operation).
        if (self.config.stability_check_interval > 0 and 
            self.bandit.t % self.config.stability_check_interval == 0 and
            self.bandit.t > 0):
            for model in self.bandit.models:
                self.bandit._check_numerical_stability(model, self.config)

    def get_probabilities(self, context: str | np.ndarray, model_ids: List[str] | None = None) -> Dict[str, float]:
        """
        Estimate the probability each model has the highest *quality* for *context*.

        Uses Thompson Sampling (posterior draws from the LinUCB ridge
        regression posterior) to produce a probability distribution over
        models.

        **Quality-only:** These probabilities reflect the learned reward
        model and do **not** incorporate cost or latency penalties.  The
        actual routing decision (``route()``) optimises a composite utility
        ``UCB - λ_c·cost - λ_ℓ·latency``; this method answers the narrower
        question "which model is most likely to produce the highest-quality
        response?" — useful for dashboards, explainability, and posterior
        calibration.

        Args:
            context: Prompt string or pre-computed feature vector.
            model_ids: Subset of models to evaluate.  ``None`` means all
                registered models.

        Returns:
            Dictionary mapping model IDs to quality-best probabilities
            that sum to 1.0.  A uniform distribution is returned when no
            valid models remain after filtering.
        """
        x = self.features.extract_features(context)
        models = model_ids if model_ids else self.bandit.models
        
        return self.bandit.get_probabilities(x, models)

    def update(self, model_id: str, context: str | np.ndarray, reward: float, weight: float = 1.0, advance_time: bool = True) -> None:
        """
        Perform a direct bandit update (bypass ``process_feedback`` flow).

        Use this for batch/offline learning where you already have
        ``(model, context, reward)`` triples.  For the standard online
        workflow, prefer ``route()`` → ``process_feedback()``.

        Rewards are clamped to [0, 1].

        Args:
            model_id: Model that was selected.
            context: Prompt string or pre-computed feature vector.
            reward: Observed quality signal in [0, 1] (clamped).
            weight: Importance weight for this observation (default 1.0).
            advance_time: Whether to increment the global time step `t`.
                Default `True` is correct for offline/batch learning.

        Raises:
            ValueError: If *context* is an ``np.ndarray`` with wrong dimension.
            KeyError: If *model_id* is not registered.
        """
        # Clamp reward to [0, 1] (same as process_feedback)
        reward = float(np.clip(reward, 0.0, 1.0))
        x = self.features.extract_features(context)
        
        self.bandit.update(model_id, x, reward, weight, advance_time=advance_time)
        
        # Periodic stability check.
        if (self.config.stability_check_interval > 0 and
            self.bandit.t % self.config.stability_check_interval == 0 and
            self.bandit.t > 0):
            for model in self.bandit.models:
                self.bandit._check_numerical_stability(model, self.config)

    # -------------------------------------------------------------------------
    # Observability: Feature Contribution Analysis
    # -------------------------------------------------------------------------

    @staticmethod
    def _explain_coefficients(
        policy: "DisjointLinUCBPolicy",
        model_id: str,
    ) -> np.ndarray:
        """Return the combined coefficient vector for interpretability.

        For :class:`DisjointLinUCBPolicy` (or adapters wrapping one), the
        coefficient vector is simply ``A_inv @ b``.

        **Thread safety:** This method does NOT acquire any locks.  The
        caller must hold ``bandit._lock`` (or ensure no concurrent
        mutations) before calling.

        Returns:
            1-D coefficient array of shape ``(dim,)``.

        Raises:
            ValueError: If *model_id* is not in the policy.
        """
        if model_id not in policy.A_inv:
            raise ValueError(
                f"Model {model_id} not found in bandit registry"
            )
        return policy.A_inv[model_id] @ policy.b[model_id]

    def explain_decision(
        self, 
        model_id: str, 
        context_vector: np.ndarray,
        threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        Feature Contribution Analysis: Why did LinUCB pick this model?
        
        This method provides mathematical transparency into the router's decision-making
        by decomposing the model's score into individual feature contributions.
        
        **Mathematical Foundation:**
        LinUCB computes a score as: score = θ^T · x
        This method shows which features in x contributed most to the final score.
        
        **Use Case:**
        Instead of guessing "Did it pick Claude Opus because of code?", you can inspect:
        ```
        explanation = router.explain_decision("claude-opus", context_vector)
        # Returns: {"PCA_0": +0.8, "PCA_5": +0.3, "bias": +0.2}
        ```
        
        This tells you that PCA_0 (which might capture "mathematical reasoning") 
        contributed +0.8 to the score, making Opus the winner.
        
        Args:
            model_id: The model to explain (e.g., "claude-opus")
            context_vector: The context vector for the prompt
            threshold: Minimum absolute contribution to include (default: 0.01)
                      Filters out noise from features with negligible impact
        
        Returns:
            Dictionary mapping feature names to their contribution scores
            Sorted by absolute contribution (highest to lowest)
            
        Example:
            >>> prompt = "Solve the integral of x^2"
            >>> x = router._get_context_vector(prompt)
            >>> selected_model, log = router.route(prompt)
            >>> explanation = router.explain_decision(selected_model, x)
            >>> print(explanation)
            {'PCA_0': 0.85, 'PCA_12': 0.42, 'bias': 0.15}
        """
        with self.bandit._lock:
            combined = self._explain_coefficients(self.bandit, model_id)

        contributions = combined * context_vector
        
        # 3. Map back to feature names
        explanation = {}
        
        # Structure: [PCA (d-1) | Bias (1)], e.g. 33-D with 32 PCA + 1 bias
        pca_dims = len(context_vector) - 1  # All except last dimension
        
        for idx in range(pca_dims):
            score = float(contributions[idx])
            if abs(score) > threshold:
                explanation[f"PCA_{idx}"] = score
        
        # Bias term (last dimension)
        bias_score = float(contributions[-1])
        if abs(bias_score) > threshold:
            explanation["bias"] = bias_score
        
        # Sort by absolute contribution (highest impact first)
        explanation = dict(
            sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        return explanation
    
    def explain_selection(
        self, 
        prompt: str, 
        top_k: int = 3,
        threshold: float = 0.01
    ) -> Dict[str, Dict[str, float]]:
        """
        Explain why the router selected a model over alternatives.
        
        This is a convenience wrapper that:
        1. Extracts the context vector from the prompt
        2. Shows feature contributions for the top-k models
        
        **Use Case:**
        Instead of manually extracting context vectors, you can directly:
        ```
        explanations = router.explain_selection(
            "Prove Fermat's Last Theorem", 
            top_k=3
        )
        # Returns feature contributions for top 3 models
        ```
        
        Args:
            prompt: Input prompt text
            top_k: Number of top models to explain (default: 3)
            threshold: Minimum absolute contribution to include (default: 0.01)
        
        Returns:
            Dictionary mapping model_id -> feature contributions
            
        Example:
            >>> explanations = router.explain_selection("Debug this Python code", top_k=2)
            >>> for model, features in explanations.items():
            ...     print(f"{model}: {features}")
            claude-opus: {'PCA_7': 0.92, 'PCA_3': 0.41, 'bias': 0.18}
            gpt-4: {'PCA_7': 0.78, 'PCA_12': 0.35, 'bias': 0.15}
        """
        # Extract context vector
        x = self._get_context_vector(prompt)
        
        model_scores = []
        coeff_cache: Dict[str, np.ndarray] = {}
        with self.bandit._lock:
            for model_id in self.bandit.models:
                if model_id not in self.bandit.A_inv:
                    continue
                coeffs = self._explain_coefficients(self.bandit, model_id)
                coeff_cache[model_id] = coeffs
                score = float(np.dot(coeffs, x))
                model_scores.append((model_id, score))
        
        # Sort by score (highest first) and take top-k
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [m[0] for m in model_scores[:top_k]]
        
        # Generate explanations for top-k models using cached coefficients
        # (no second lock acquisition needed — uses the same snapshot)
        pca_dims = len(x) - 1  # All except last dimension (bias)
        explanations = {}
        for model_id in top_models:
            contributions = coeff_cache[model_id] * x
            explanation = {}
            for idx in range(pca_dims):
                score = float(contributions[idx])
                if abs(score) > threshold:
                    explanation[f"PCA_{idx}"] = score
            bias_score = float(contributions[-1])
            if abs(bias_score) > threshold:
                explanation["bias"] = bias_score
            explanations[model_id] = explanation
        
        return explanations





    def save_state(self, path: Path | str) -> None:
        """Save the bandit's learned state (A, b matrices) to disk."""
        self.bandit.save_state(path)

    def load_state(self, path: Path | str) -> None:
        """Load the bandit's learned state from disk."""
        self.bandit.load_state(path)

    @contextmanager
    def exploit(self) -> Generator[None, None, None]:
        """Context manager for greedy exploitation (frozen policy evaluation).

        Temporarily sets the bandit's ``alpha`` to 0 so that ``route()``
        selects ``argmax(theta^T x)`` with no UCB exploration bonus.

        State is restored on exit (including after exceptions), analogous to
        ``torch.no_grad()`` in PyTorch.

        Usage::

            with router.exploit():
                model, log = router.route(x)
        """
        saved_alpha = self.bandit.alpha
        self.bandit.alpha = 0.0

        try:
            yield
        finally:
            self.bandit.alpha = saved_alpha

    def _calculate_absolute_penalty(self, cost_per_1k: float) -> float:
        """Stable 0.0-1.0 cost penalty via logarithmic market anchors.

        Delegates to :func:`bandit_gpt.costs.log_normalize_cost` — the
        canonical implementation shared with offline evaluation baselines.
        Anchors are read from ``self.config`` (single source of truth).

        Args:
            cost_per_1k: Cost in dollars per 1000 tokens.

        Returns:
            Penalty in [0.0, 1.0].
        """
        from bandit_gpt.costs import log_normalize_cost

        return log_normalize_cost(
            cost_per_1k,
            floor=self.config.market_cost_floor,
            ceiling=self.config.market_cost_ceiling,
        )

    def _get_normalized_cost(self, model_id: str) -> float:
        """Compute normalized [0, 1] cost for a model from registry metadata."""
        m_data = self.registry.get(model_id, {})
        default_cost = self.config.default_missing_cost_per_m
        input_cost = m_data.get("input_cost_per_m")
        output_cost = m_data.get("output_cost_per_m")
        if not isinstance(input_cost, (int, float)):
            input_cost = default_cost
        if not isinstance(output_cost, (int, float)):
            output_cost = default_cost * _OUTPUT_COST_MULTIPLIER
        avg_cost_per_1k = ((input_cost + output_cost) / 2.0) / 1000.0
        return self._calculate_absolute_penalty(avg_cost_per_1k)

    def _calculate_absolute_latency_penalty(self, latency_s: float) -> float:
        """
        Calculate stable 0.0-1.0 latency penalty based on Fixed Market Anchors.

        Uses the same logarithmic normalization as cost to ensure penalties
        are absolute (not relative to currently loaded models).

        Market Anchors:
        - Floor: 0.05s  (streaming-first models, e.g. Flash, Haiku)
        - Ceiling: 5.0s (slow batch inference or timeout threshold)
        - Log range: ln(5.0) - ln(0.05) ≈ 4.61

        Args:
            latency_s: Estimated time-to-first-token in seconds.

        Returns:
            Penalty in range [0.0, 1.0].
            - 0.0 = at or below market floor (fastest)
            - 1.0 = at or above market ceiling (slowest)
        """
        safe_lat = max(latency_s, self._market_lat_floor)
        log_lat = math.log(safe_lat)
        penalty = (log_lat - self._market_lat_floor_log) / self._market_lat_range
        return max(0.0, min(1.0, penalty))

    def _get_normalized_latency(self, model_id: str) -> float:
        """Compute normalized [0, 1] latency for a model from registry metadata."""
        m_data = self.registry.get(model_id, {})
        val = m_data.get("time_to_first_token_seconds")
        if val is None or not isinstance(val, (int, float)) or val <= 0.0:
            val = self.config.default_missing_latency
        return self._calculate_absolute_latency_penalty(float(val))

    def _estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        """
        Estimate cost with Pessimistic Defaults for resilience.
        
        Prevents 'All-Infinity' outage if registry schema breaks or config
        update fails. Unknown models are treated as Opus-tier expensive,
        keeping the service operational in conservative mode.
        
        Args:
            model: Model identifier
            in_tok: Input token count
            out_tok: Output token count
            
        Returns:
            Estimated cost in USD
        """
        m = self.registry.get(model, {})
        
        # Extract costs with type validation
        input_cost = m.get("input_cost_per_m")
        output_cost = m.get("output_cost_per_m")
        
        # Validate: Must be numbers (guard against schema corruption)
        if input_cost is None or not isinstance(input_cost, (int, float)):
            input_cost = self.config.default_missing_cost_per_m
            
        if output_cost is None or not isinstance(output_cost, (int, float)):
            output_cost = self.config.default_missing_cost_per_m * _OUTPUT_COST_MULTIPLIER
        
        # Calculation: now guaranteed to return valid float, never inf
        return (input_cost * in_tok + output_cost * out_tok) / 1e6

    def _estimate_latency(self, model: str, out_tok: int) -> float:
        """
        Estimate latency with Pessimistic Defaults for resilience.
        
        Prevents routing failures when time_to_first_token_seconds is missing.
        Unknown models are treated as slow (2.0s) but usable.
        
        Args:
            model: Model identifier
            out_tok: Output token count (unused, for API consistency)
            
        Returns:
            Estimated time to first token in seconds
        """
        m = self.registry.get(model, {})
        val = m.get("time_to_first_token_seconds")
        
        # Validate: Must be positive number
        if val is None or not isinstance(val, (int, float)) or val <= 0.0:
            # Fallback to "slow but usable" instead of infinity
            return self.config.default_missing_latency
            
        return float(val)



# ---------------------------------------------------------------------------
# Standalone prior calibration (operates on a DisjointLinUCBPolicy)
# ---------------------------------------------------------------------------


def calibrate_priors(
    bandit: 'DisjointLinUCBPolicy',
    target_max_pred: float = 0.9,
    calibration_contexts: List[np.ndarray] | None = None,
) -> None:
    """Auto-calibrate loaded priors on *bandit* so predictions stay in a safe range.

    Two-pass calibration:

    **Pass 1 — Bias probe** (fast, catches the most common failure):
    Probes each model with ``[0,...,0,1]`` (bias-only context).  If the bias
    prediction exceeds 1.5, the bias component of theta is clamped via
    theta-reconstruction (``b = A @ theta_new``).

    **Pass 2 — Suite probe** (comprehensive, catches PCA-dimension explosions):
    Probes each model with a built-in suite of basis-independent feature
    vectors (axis-aligned, random unit-norm, uniform).  If the caller supplies
    *calibration_contexts*, those are appended to the suite.  If any prediction
    exceeds 1.5, theta is globally rescaled so the worst-case prediction equals
    *target_max_pred*.

    .. warning::

        **Not thread-safe.**  This function writes ``bandit.b[m]`` without
        acquiring any lock.  It must be called during single-threaded
        initialization (e.g., inside ``BanditRouter.create()``) *before*
        the router is exposed to concurrent ``route()`` / ``update()``
        traffic.  Calling it on a live router will race with readers and
        writers.

    Args:
        bandit: A ``DisjointLinUCBPolicy`` whose A/b matrices will be modified
                in place.
        target_max_pred: Target maximum absolute prediction over the probe suite
                       (default: 0.9).
        calibration_contexts: Optional list of domain-specific context vectors
                       (numpy arrays of shape ``(dim,)``) appended to the
                       built-in geometry probes.
    """
    d = bandit.dim
    probes: List[tuple] = []

    bias_probe = np.zeros(d)
    bias_probe[-1] = 1.0
    probes.append(("bias", bias_probe))

    pca_dims = list(range(d - 1))
    if len(pca_dims) > 8:
        step = max(1, len(pca_dims) // 8)
        pca_dims = pca_dims[::step][:8]
    for i in pca_dims:
        e_i = np.zeros(d)
        e_i[i] = 1.0
        probes.append((f"axis_{i}", e_i))

    rng = np.random.default_rng(42)
    for k in range(4):
        v = rng.standard_normal(d)
        v /= (np.linalg.norm(v) + 1e-12)
        probes.append((f"random_{k}", v))

    uniform = np.ones(d) / np.sqrt(d)
    probes.append(("uniform", uniform))

    if calibration_contexts is not None:
        for i, ctx in enumerate(calibration_contexts):
            ctx = np.asarray(ctx, dtype=float).flatten()
            if ctx.shape[0] != d:
                logger.warning(
                    f"Skipping calibration_context[{i}]: shape {ctx.shape} "
                    f"!= dim {d}"
                )
                continue
            probes.append((f"user_{i}", ctx))

    for m in bandit.models:
        try:
            theta = bandit.A_inv[m] @ bandit.b[m]

            bias_pred = float(theta @ bias_probe)
            if abs(bias_pred) > 1.5:
                theta_new = theta.copy()
                theta_new[-1] = target_max_pred * (1.0 if bias_pred > 0 else -1.0)
                logger.warning(
                    f"[FIX] Calibration pass 1 ({m}): bias prediction "
                    f"{bias_pred:.2f} -> {theta_new[-1]:.2f} (theta-reconstruction)"
                )
                bandit.b[m] = bandit.A[m] @ theta_new
                theta = bandit.A_inv[m] @ bandit.b[m]

            max_abs_pred = 0.0
            worst_probe = "none"
            for name, x in probes:
                pred = abs(float(theta @ x))
                if pred > max_abs_pred:
                    max_abs_pred = pred
                    worst_probe = name

            if max_abs_pred > 1.5:
                scale = target_max_pred / max_abs_pred
                theta_new = theta * scale
                logger.warning(
                    f"[FIX] Calibration pass 2 ({m}): worst-case prediction "
                    f"{max_abs_pred:.2f} on probe '{worst_probe}' "
                    f"-> global theta scale {scale:.4f}"
                )
                bandit.b[m] = bandit.A[m] @ theta_new

        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Failed to calibrate prior for {m}: {e}")
            continue


