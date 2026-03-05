"""
Production-grade contextual bandit router (Hot Path).

Core Features:
1. Warmup Priors: Initializes with learned preferences from 80k battles.
2. Default Registry: Automatically loads 80+ models with cost/latency data.
3. Corralling: Meta-learning over warmup and tabula rasa experts.
4. Constraints: Supports max_cost, max_latency, and quality floors.

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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter, deque, defaultdict
from typing import Any, Dict, List, Tuple, Optional, Literal, TypedDict
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
    introduces a silent bias: e.g. the tabula-rasa expert always picks the
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
) -> float:
    """Apply staleness-based variance inflation with a bounded cap.

    When ``gamma < 1.0``, divides ``var`` by ``gamma^dt`` to widen the
    confidence interval for stale arms, capped at ``_MAX_VAR_INFLATION_FACTOR``
    to prevent the exploration bonus from overwhelming additive cost penalties.

    For stationary settings (``gamma == 1.0``) or ``dt == 0``, returns ``var``
    unchanged.

    Args:
        var: Base variance (x^T A^{-1} x).
        gamma: Forgetting factor in (0, 1].
        dt: Steps since last update or selection (non-negative).

    Returns:
        Inflated (or unchanged) variance.
    """
    if gamma >= 1.0 or dt <= 0:
        return var
    decay_factor = gamma ** min(dt, _MAX_STALENESS_DT)
    inflation_floor = 1.0 / _MAX_VAR_INFLATION_FACTOR
    return var / max(decay_factor, inflation_floor, 1e-12)


def _linear_alpha_decay(
    t: int,
    total_steps: int,
    alpha_start: float,
    alpha_end: float,
) -> float:
    """Linearly interpolate alpha from ``alpha_start`` to ``alpha_end``.

    Returns ``alpha_end`` when ``total_steps == 0`` (evaluation mode).

    Args:
        t: Current step.
        total_steps: Total training horizon.
        alpha_start: Initial exploration coefficient.
        alpha_end: Terminal exploration coefficient.

    Returns:
        Interpolated alpha value.
    """
    if total_steps == 0:
        return alpha_end
    fraction = min(t / total_steps, 1.0)
    return alpha_start + fraction * (alpha_end - alpha_start)


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
        f"⚠️ Sherman-Morrison near-singularity for {model_name}: "
        f"|denominator|={abs(denominator):.2e} < {_SM_DENOMINATOR_THRESHOLD}. "
        f"A_inv has numerically drifted; rebuilding with gap-based "
        f"regularisation injection."
    )
    needed = max(init_lambda - regularization_floor, init_lambda * _REGULARIZATION_FLOOR_FRACTION)
    dim = A.shape[0]
    new_A = A + x_outer + needed * np.eye(dim)
    new_A_inv = safe_inv(new_A)
    new_b = b + reward_x
    return _SMUpdateResult(
        A=new_A, A_inv=new_A_inv, b=new_b,
        regularization_floor=regularization_floor + needed,
        used_fallback=True,
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
    - Knowledge transfer: Handled by the Corralling meta-learner and
      warmup priors (continuous, empirically validated).
    
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
    
    ✅ **CANONICAL CONFIG**: This is the production-grade configuration for BanditRouter.
    
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
    if not text: return 0
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
    def __init__(self, model_names: List[str], dim: int = 384, alpha: float = 0.1,
                 init_lambda: float = 1.0,
                 forgetting_factor: float = 1.0):
        """
        Initialize Disjoint LinUCB policy.
        
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
            model_names: List of model identifiers (arms)
            dim: Context vector dimension
            alpha: Exploration coefficient (UCB bonus multiplier)
            init_lambda: Initialization regularization (A₀ = λI). Default 1.0 for
                       cold-start stability.  This is an isotropic prior that
                       regularizes all principal directions equally; it does not
                       distinguish between high- and low-variance PCA components.
            forgetting_factor: Exponential decay factor (1.0 = stationary, <1.0 = adaptive). Default 1.0.
        """
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(forgetting_factor)
        self.init_lambda = float(init_lambda)

        # Thread safety: Per-model locks to eliminate lost-update race conditions.
        # Updates to Model A don't block updates to Model B.
        from collections import defaultdict
        self.model_locks = defaultdict(threading.Lock)
        
        # Global lock for read operations (select_arm, refresh_inverse_cache)
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

    def __deepcopy__(self, memo):
        """
        Custom deepcopy to handle thread locks.
        
        Locks cannot be pickled or deepcopied directly. We create new locks
        for the clone while deepcopying all numerical state (A, b, A_inv, etc.).
        """
        from collections import defaultdict
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy basic attributes
        result.models = copy.deepcopy(self.models, memo)
        result.dim = self.dim
        result.alpha = self.alpha
        result.gamma = self.gamma
        result.init_lambda = self.init_lambda
        result.t = self.t
        result.last_update = copy.deepcopy(self.last_update, memo)
        result.last_played = copy.deepcopy(self.last_played, memo)
        
        # Copy major state (numpy arrays copy well)
        result.A = copy.deepcopy(self.A, memo)
        result.b = copy.deepcopy(self.b, memo)
        result.A_inv = copy.deepcopy(self.A_inv, memo)
        
        # Create FRESH locks for the clone (per-model locks)
        result.model_locks = defaultdict(threading.Lock)
        
        # Create fresh global lock for the clone
        result._lock = threading.Lock()
        
        # Copy regularization_floor so the clone's decay path
        # (which accesses self.regularization_floor[model]) doesn't crash
        # with AttributeError on the first update when gamma < 1.0.
        result.regularization_floor = copy.deepcopy(self.regularization_floor, memo)
        
        return result

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically.
        
Prepare all state outside the lock, then publish atomically.
        Previously, a concurrent select_arm() could see the model in self.A
        before self.A_inv was assigned, causing a KeyError.
        """
        if model_name in self.models: return
        
        # Prepare outside lock
        new_A = np.eye(self.dim) * self.init_lambda
        new_b = np.zeros(self.dim, dtype=np.float64)
        new_A_inv = safe_inv(new_A)
        
        # Publish atomically under global lock
        with self._lock:
            self.A[model_name] = new_A
            self.b[model_name] = new_b
            self.A_inv[model_name] = new_A_inv
            self.last_update[model_name] = self.t
            self.last_played[model_name] = self.t
            self.regularization_floor[model_name] = self.init_lambda
            self.models.append(model_name)  # Last: select_arm sees it only after state is ready

    def delete_arm(self, model_name: str) -> None:
        """Remove an arm from the bandit.
        
Wrap in lock and clean up regularization_floor and
        model_locks to prevent unbounded memory growth under model churn.
        """
        with self._lock:
            if model_name in self.models:
                self.models.remove(model_name)
            if model_name in self.A: del self.A[model_name]
            if model_name in self.b: del self.b[model_name]
            if model_name in self.A_inv: del self.A_inv[model_name]
            if model_name in self.last_update: del self.last_update[model_name]
            if model_name in self.last_played: del self.last_played[model_name]
            if model_name in self.regularization_floor: del self.regularization_floor[model_name]
            if model_name in self.model_locks: del self.model_locks[model_name]

    def refresh_inverse_cache(self) -> None:
        """
        Recomputes A_inv for all models after a bulk load.
        
        This is needed when loading pre-trained warmup state, where A matrices
        are updated directly but the inverse cache becomes stale.
        
        Thread-safe: Uses lock to prevent concurrent reads during refresh.
        """
        with self._lock:
            self.A_inv = {}
            for m in self.models:
                if m in self.A:
                    # Recompute inverse using safe_inv (handles near-singular matrices)
                    self.A_inv[m] = safe_inv(self.A[m])


    def select_arm(
        self, 
        x: np.ndarray, 
        candidates: List[str | None] = None,
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
        expensive model." Early in training (tabula rasa), α·std may exceed 1.0,
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
        # Snapshot references under lock (O(1) dict lookups).  Because update()
        # assigns new NumPy arrays (pointer swaps) rather than mutating in place,
        # we can safely compute O(d²) matrix math lock-free using the snapshots.
        with self._lock:
            candidates = candidates or self.models
            candidates = [m for m in candidates if m in self.A_inv]
            if not candidates:
                raise ValueError("No candidates available")
            snapshots = {
                m: (self.A_inv[m], self.b[m], self._effective_staleness(m))
                for m in candidates
            }

        # Execute O(d²) matrix math entirely lock-free.
        ucb_scores: Dict[str, float] = {}
        for m, (A_inv, b, dt) in snapshots.items():
            theta = A_inv @ b
            mean = float(theta.dot(x))
            var = float(x.dot(A_inv).dot(x))
            var_inflated = _inflate_variance(var, self.gamma, dt)
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
            # Inflate the posterior covariance for staleness, matching
            # the scalar inflation applied in select_arm() for consistency.
            scalar_var = float(np.trace(A_inv_m))
            inflated = _inflate_variance(scalar_var, self.gamma, dt)
            if scalar_var > 0 and inflated != scalar_var:
                cov = noise_variance * A_inv_m * (inflated / scalar_var)
            else:
                cov = noise_variance * A_inv_m
            try:
                samples = np.random.multivariate_normal(theta_hat, cov, n_samples)
            except np.linalg.LinAlgError:
                # Jitter the diagonal to restore positive-definiteness while
                # preserving the off-diagonal covariance structure (confidence
                # ellipsoid geometry).  The jitter magnitude is proportional to
                # the average variance so it doesn't dominate a well-conditioned
                # matrix or vanish on a poorly-scaled one.
                jitter = max(np.trace(cov) / self.dim, 1e-12) * 1e-6
                cov_safe = cov + jitter * np.eye(self.dim)
                try:
                    samples = np.random.multivariate_normal(
                        theta_hat, cov_safe, n_samples,
                    )
                except np.linalg.LinAlgError:
                    # Truly degenerate — fall back to isotropic sampling
                    avg_var = max(np.trace(cov) / self.dim, 1e-12)
                    samples = np.random.normal(
                        loc=theta_hat, scale=np.sqrt(avg_var),
                        size=(n_samples, self.dim),
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
            # Snapshot the logical clock atomically before any computation.
            # mark_selected() increments self.t under self._lock; reading self.t
            # here outside that lock and again later (for last_update assignment)
            # would create a formal race where dt is computed against one value of
            # self.t but last_update is written with a different, later value.
            # Snapping current_t once under self._lock ensures both uses are
            # consistent, whether or not mark_selected() fires between them.
            with self._lock:
                current_t = self.t

            # 1. Calculate Time Decay
            dt = 0
            decay_factor = 1.0
            if self.gamma < 1.0:
                dt = current_t - self.last_update[model]
                # Clamp dt to prevent numerical underflow when gamma is small
                decay_factor = self.gamma ** min(dt, _MAX_STALENESS_DT)

            # 2. Proactive regularization maintenance
            # Ensure A remains well-conditioned under decay (A >= lambda_min I).
            current_lambda = self.regularization_floor.get(model, self.init_lambda)
            new_lambda = current_lambda * decay_factor
            
            # Threshold: Reinject if prior strength drops below _REGULARIZATION_FLOOR_FRACTION of init
            lambda_threshold = self.init_lambda * _REGULARIZATION_FLOOR_FRACTION

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
                    f"🔧 Maintenance: Restoring regularization floor for {model} "
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
                f"🛡️ Numerical instability detected for {model}: "
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
                f"✅ Regularization reset complete for {model}. "
                f"New trace(A_inv)={new_trace:.2f}"
            )



    def save_state(self, path: Path | str) -> None:
        """
        Save A and b matrices to a compressed NPZ file with metadata.
        
        Stores dimension metadata to enable validation on load, preventing
        crashes from dimension mismatches due to PCA fallback or feature changes.
        """
        data = {}
        # Save metadata for validation
        data['_metadata_dim'] = self.dim
        data['_metadata_models'] = list(self.models)
        
        for m in self.models:
            data[f"{m}_A"] = self.A[m]
            data[f"{m}_b"] = self.b[m]
        np.savez_compressed(path, **data)

    def load_state(self, path: Path | str) -> None:
        """
        Load A and b matrices from a compressed NPZ file with dimension validation.
        
        Validates that saved dimension matches current bandit dimension to prevent
        silent matrix misalignment crashes. Raises clear error if dimensions don't match.
        
        Raises:
            ValueError: If saved dimension doesn't match current bandit dimension.
                       Suggests clearing state or updating feature configuration.
        """
        data = np.load(path)
        
        # Validate dimension compatibility
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
            # Legacy state file without metadata - warn but proceed
            logger.warning(
                f"Loading state from {path} without dimension metadata. "
                f"This may cause issues if dimensions have changed. "
                f"Current dim={self.dim}"
            )
        
        # Load matrices with dimension validation
        for m in self.models:
            a_key = f"{m}_A"
            b_key = f"{m}_b"
            if a_key in data and b_key in data:
                A_loaded = data[a_key]
                b_loaded = data[b_key]
                
                # Validate shapes
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
                
                self.A[m] = A_loaded
                self.b[m] = b_loaded
                self.A_inv[m] = safe_inv(self.A[m])


# ---------------------------------------------------------------------------
# Main Router Class
# ---------------------------------------------------------------------------
# (HybridLinUCBPolicy was removed — the router exclusively uses
#  DisjointLinUCBPolicy for arm selection.  Family-level transfer is
#  handled externally by Corralling experts if needed.)
# ---------------------------------------------------------------------------

# Backward-compatible public alias. Older code and stress tests import
# HybridLinUCBPolicy from the top-level package; it now maps to the
# disjoint-arm LinUCB implementation used by BanditRouter.
HybridLinUCBPolicy = DisjointLinUCBPolicy

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
    total_priority_weight: float = 1.0       # Sum of w_q, w_c, w_l for normalization
    corralling_token: Dict | None = None     # Selection token for Corralling meta-weight attribution

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
        use_corralling: bool = True,  # Enable corralling by default
        corralling_learning_rate: float = 0.1,
        corralling_gamma: float = 0.05,
        cost_penalty: float = 0.3,  # λ_c for UCB cost penalty (paper Eq. 4)
        latency_penalty: float = 0.0,  # λ_l for UCB latency penalty
        tabula_rasa_alpha: float | None = None,
        tabula_rasa_forgetting_factor: float | None = None,
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
            use_corralling: Enable Corralling meta-learner (default: True)
            corralling_learning_rate: Meta-learning rate for expert weight updates (default: 0.1)
            corralling_gamma: Mixing parameter (default: 0.05, empirically validated optimal)
            cost_penalty: λ_c for UCB cost penalty (paper Eq. 4). At selection
                       time, each arm's score includes -λ_c·normalized_cost(model).
                       Applied consistently in both Corralling experts and the
                       singleton fallback path. Does NOT affect learned quality
                       estimates — only biases selection toward cheaper models.
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
            tabula_rasa_alpha: Per-expert exploration coefficient for the
                       tabula-rasa expert inside Corralling.  When ``None``
                       (default), the tabula-rasa expert uses ``2 * alpha``
                       (legacy behaviour).
            tabula_rasa_forgetting_factor: Per-expert forgetting factor for
                       the tabula-rasa expert inside Corralling.  When ``None``
                       (default), inherits the canonical bandit's
                       ``forgetting_factor`` (legacy behaviour).
        """
        self.config = config or RouterConfig()
        self.verbose_routing = verbose_routing
        self.use_corralling = use_corralling
        self.corralling_learning_rate = corralling_learning_rate
        self.corralling_gamma = corralling_gamma
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.tabula_rasa_alpha = tabula_rasa_alpha
        self.tabula_rasa_forgetting_factor = tabula_rasa_forgetting_factor
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
        
        # Initialize Security Scanner (Lazy)
        self._toxicity_scanner = None
        
        # Initialize Corralling Router (if enabled)
        # Properly initialized in create() after warmup priors are loaded.
        self.corralling_router = None


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

    def _init_corralling(self, alpha: float) -> None:
        """Set up the Corralling meta-learner (Log-Barrier OMD) with heterogeneous experts.

        Creates two experts — an informed explorer (adapter over the canonical
        bandit with constant alpha) and a tabula-rasa converger (decaying alpha,
        no priors) — and wires them into a :class:`CorrallingRouter` that
        aggregates expert advice via Log-Barrier Online Mirror Descent
        (Agarwal et al., 2017).

        Per-expert hyperparameters (``tabula_rasa_alpha``,
        ``tabula_rasa_forgetting_factor``) are read from the instance
        attributes set at construction time.  When these are ``None`` the
        legacy schedule is used (tabula-rasa alpha = 2× warmup alpha,
        forgetting factor inherited from the canonical bandit).

        Called once from :meth:`create` after warmup priors and calibration are
        finalised.

        Args:
            alpha: Base exploration coefficient for the warmup expert.
        """
        logger.info("🎯 Initializing Corralling Router with Heterogeneous Experts Strategy...")

        target_alpha = alpha if alpha is not None else 0.5

        model_costs: Dict[str, Dict[str, float]] = {}
        for model_id in self.bandit.models:
            m_data = self.registry.get(model_id, {})
            input_cost = m_data.get("input_cost_per_m", self.config.default_missing_cost_per_m)
            output_cost = m_data.get(
                "output_cost_per_m",
                self.config.default_missing_cost_per_m * _OUTPUT_COST_MULTIPLIER,
            )
            avg_cost_per_1k = ((input_cost + output_cost) / 2.0) / 1000.0
            norm_cost = self._calculate_absolute_penalty(avg_cost_per_1k)
            norm_latency = self._get_normalized_latency(model_id)
            model_costs[model_id] = {
                "normalized_cost": norm_cost,
                "normalized_latency": norm_latency,
            }

        # Expert 1: Informed Explorer — constant alpha hedges against prior mismatch
        expert_warmup = CostAwareLinUCBAdapter(
            bandit=self.bandit,
            model_costs=model_costs,
            alpha_start=target_alpha,
            alpha_end=target_alpha,
            cost_penalty=self.cost_penalty,
            latency_penalty=self.latency_penalty,
        )

        # Expert 2: Learning Converger
        # Per-expert parameters from Appendix H ablation when available;
        # otherwise fall back to the legacy 2× alpha → 0.01 decay schedule.
        tr_alpha = self.tabula_rasa_alpha
        tr_gamma = (
            self.tabula_rasa_forgetting_factor
            if self.tabula_rasa_forgetting_factor is not None
            else self.bandit.gamma
        )
        if tr_alpha is not None:
            tr_alpha_start = tr_alpha
            tr_alpha_end = 0.01
        else:
            tr_alpha_start = target_alpha * 2.0
            tr_alpha_end = 0.01

        expert_tabula_rasa = CostAwareTabulaRasaRouter(
            models=self.bandit.models,
            context_dim=self.bandit.dim,
            model_costs=model_costs,
            alpha_start=tr_alpha_start,
            alpha_end=tr_alpha_end,
            cost_penalty=self.cost_penalty,
            latency_penalty=self.latency_penalty,
            ridge_lambda=1.0,
            forgetting_factor=tr_gamma,
        )

        self.corralling_router = CorrallingRouter(
            experts=[expert_warmup, expert_tabula_rasa],
            models=self.bandit.models,
            learning_rate=self.corralling_learning_rate,
            gamma=self.corralling_gamma,
            model_costs=model_costs,
        )

        logger.info("✅ Heterogeneous Experts Strategy Initialized:")
        logger.info(f"   📊 Expert 1 (Informed):     Constant Alpha {target_alpha:.2f} (Sustained Discovery)")
        logger.info(f"   🔍 Expert 2 (Uninformed):   Decaying Alpha {tr_alpha_start:.2f}→{tr_alpha_end} (Explore-then-Exploit)")
        logger.info(f"   ⏳ Forgetting (warmup):      γ={self.bandit.gamma:.4f} ({'stationary' if self.bandit.gamma >= 1.0 else 'adaptive'})")
        logger.info(f"   ⏳ Forgetting (tabula rasa): γ={tr_gamma:.4f} ({'stationary' if tr_gamma >= 1.0 else 'adaptive'})")
        logger.info("   🎯 Meta-Learner:            Corralling (Log-Barrier OMD) selects expert based on prompt context")

    def __deepcopy__(self, memo):
        """
        Custom deepcopy for BanditRouter to handle unpicklable components.
        
Previous version referenced non-existent attributes
        (anchor_vectors, complexity_vector, cluster_detector) and omitted
        many attributes that exist in __init__, producing a broken clone.
        
        Strategy:
        1. SHARE stateless / lock-containing objects (encoder, features, context_store)
        2. DEEPCOPY all mutable state (bandit, corralling, logs, counters, config)
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
        
        # --- Corralling (deepcopy: independent mutable state) ---
        result.use_corralling = self.use_corralling
        result.corralling_learning_rate = self.corralling_learning_rate
        result.corralling_gamma = self.corralling_gamma
        result.cost_penalty = self.cost_penalty
        result.latency_penalty = self.latency_penalty
        result.tabula_rasa_alpha = self.tabula_rasa_alpha
        result.tabula_rasa_forgetting_factor = self.tabula_rasa_forgetting_factor
        result.corralling_router = copy.deepcopy(self.corralling_router, memo) if self.corralling_router else None
        
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
            logger.warning(f"⚠️ Model {model_id} already registered. Skipping.")
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
        new_A = np.eye(self.bandit.dim) * self.bandit.init_lambda
        new_b = self.bandit.init_lambda * theta_vector
        new_A_inv = safe_inv(new_A)

        with self.bandit._lock:
            self.bandit.A[model_id] = new_A
            self.bandit.b[model_id] = new_b
            self.bandit.A_inv[model_id] = new_A_inv
            self.bandit.last_update[model_id] = self.bandit.t
            self.bandit.last_played[model_id] = self.bandit.t
            self.bandit.regularization_floor[model_id] = self.bandit.init_lambda
            self.bandit.models.append(model_id)  # LAST: visible only after state is ready
            
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
        
        registry_entry = {
            "cost_per_1m_tokens": cost_usd,
            "blended_cost_per_m": float(blended_cost_per_m),
            "time_to_first_token_seconds": latency_s,
            "median_latency_s": latency_s,
            "capabilities": capabilities,
            "speed_profile": speed,
        }
        
        # 9. Propagate to Corralling Experts BEFORE registry publication
        # The bandit arm was already added at step 6 above.  Expert 1 (adapter)
        # shares the bandit reference, so it sees the new arm automatically.
        # We only need to register cost metadata with the adapter and propagate
        # the arm to Expert 2 (tabula rasa) and the corralling manager.
        if self.use_corralling and self.corralling_router:
            logger.info(f"🔄 Propagating {model_id} to Corralling experts...")
            
            output_cost = cost_usd * _OUTPUT_COST_MULTIPLIER
            avg_cost_per_1k = ((cost_usd + output_cost) / 2.0) / 1000.0
            norm_cost = self._calculate_absolute_penalty(avg_cost_per_1k)
            norm_latency = self._calculate_absolute_latency_penalty(latency_s)
            
            self.corralling_router.add_model(model_id)
            
            expert_warmup = self.corralling_router.experts[0]
            if hasattr(expert_warmup, 'add_model'):
                expert_warmup.add_model(
                    model_id, norm_cost,
                    normalized_latency=norm_latency,
                )
            
            expert_tr = self.corralling_router.experts[1]
            if hasattr(expert_tr, 'add_model'):
                expert_tr.add_model(
                    model_id, norm_cost,
                    normalized_latency=norm_latency,
                )
                
            logger.info(f"✅ {model_id} added to Corralling system")
        
        # 10. Publish to registry LAST — model is now fully initialized everywhere
        self.registry[model_id] = registry_entry
        
        boost_summary = ", ".join(f"{k}={v:.1f}" for k, v in list(weights.items())[:5])
        if len(weights) > 5:
            boost_summary += "..."
        
        logger.info(
            f"✅ Registered {model_id} | "
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
        priors: str = "none",
        **kwargs
    ) -> "BanditRouter":
        """
        Factory method to create a fully initialized router.
        
        Args:
            model_registry: Dictionary of model configurations
            context_model: Model to use for embedding generation
            priors: Prior initialization strategy. ``"none"`` (default) starts
                with standard LinUCB cold-start (identity covariance + quality-based
                bias).  Pass a path to a ``.joblib`` file to load custom priors
                generated via :func:`generate_warmup_priors`.
            **kwargs: Additional arguments passed to __init__ or prior loading
        
        Returns:
            Fully initialized BanditRouter instance
        """
        # 1. Extract factory-specific arguments (not passed to __init__)
        state_path = kwargs.pop("state_path", None)
        prior_n_effective = kwargs.pop("prior_n_effective", 5000.0)
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
        
        # 4. Load Priors from explicit path
        # Users generate their own priors via generate_warmup_priors() and
        # pass the .joblib path here.  No default artifact is shipped.
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
                        f"⚠️ Warmup Partial Miss: {len(missing_models)} models not in joblib. "
                        f"Applied heuristic initialization for: {missing_models}"
                    )
                else:
                    logger.info("✅ Warmup Complete: All models initialized from offline priors.")
                
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
                #      exponential discounting of stale observations, and the
                #      Corralling meta-learner can further compensate by shifting
                #      traffic to the tabula-rasa expert when warmup priors prove
                #      harmful.
                #
                # Net effect: numerical stability + controlled prior decay.
                # =====================================================================
                for model_id in router.bandit.models:
                    router.bandit.A[model_id] += np.eye(router.bandit.dim) * router.bandit.init_lambda
                
                # Single refresh_inverse_cache() after all A matrices are
                # finalized.  Previously there were two calls — one before and one
                # after the regularization loop — wasting O(K·d³) at startup.
                router.bandit.refresh_inverse_cache()
                logger.info(f"✅ Applied post-warmup regularization (λ={router.bandit.init_lambda}) from {priors_path}")
            else:
                logger.warning(f"⚠️ Priors file not found at {priors_path}. Using cold start.")
        
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
        
        # 7. Initialize Corralling Router (if enabled)
        if router.use_corralling:
            router._init_corralling(alpha)
        
        # 8. Load state if provided (overwrites any priors applied above)
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
        random.seed(42)  # Deterministic for reproducibility
        
        # Generate n prompts by sampling templates and filling placeholders
        archetype_keys = list(templates.keys())
        for _ in range(n):
            archetype = random.choice(archetype_keys)
            template = random.choice(templates[archetype])
            
            # Fill placeholders
            prompt = template
            for placeholder, values in fill_values.items():
                if f"{{{placeholder}}}" in prompt:
                    prompt = prompt.replace(f"{{{placeholder}}}", random.choice(values))
            
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
        profile: str | Dict[str, float] = "auto",
        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] = None,
        input_tokens: int | None = None,
        output_tokens: int = 600,
        total_steps: int = 1,
    ) -> Tuple[str, RoutingLog]:
        """
        Route a prompt to the best model using Corralling (meta-learning over experts).
        
        **Corralling Architecture:**
        Maintains multiple expert strategies and learns which one works best for your data:
        - Expert 1: Warmup strategy (uses pre-trained priors from 80k battles)
        - Expert 2: Tabula rasa (learns from scratch on your data)
        
        The router automatically adapts weights to the expert that performs better,
        providing robustness against domain mismatch while leveraging priors when helpful.
        
        **Usage:**
        ```python
        # Simple: Just use defaults (corralling with warmup + tabula rasa)
        model_id, log = router.route("Write a Python function")
        ```
        
        Raises:
            NoEligibleModelsError: If no models pass the hard constraints.
        
        Args:
            prompt: Input text or pre-embedded vector
            profile: (Ignored, kept for API compatibility)
            max_cost: Hard budget ceiling in ``$/1k tokens``. Compared against
                each model's registry price (derived from ``blended_cost_per_m``).
            max_latency: Hard latency ceiling (seconds), compared against each
                        model's ``time_to_first_token_seconds``
            quality_floor: Minimum quality scores per metric (e.g.
                          ``{"hle": 0.7}``)
            input_tokens: Input token count (auto-estimated if None)
            output_tokens: Expected output tokens (default 600)
            total_steps: Total training steps for alpha decay (default 1 for production use)
        
        Returns:
            Tuple of (selected_model_id, routing_log)
        """
        # Build features and apply constraints
        x, prompt_text = self._build_routing_features(prompt)
        candidates = list(self.registry.keys())
        filtered = self._filter_by_constraints(
            candidates,
            max_cost,
            max_latency,
            quality_floor,
        )
        
        # Estimate tokens for logging and cost_usd estimates.
        in_tok = input_tokens or estimate_tokens_rough(prompt_text)

        # Use Corralling if enabled, otherwise fall back to simple LinUCB
        corralling_token = None
        if self.use_corralling and self.corralling_router is not None:
            # Pass total_steps to enable proper alpha decay in experts
            # For experiments: Pass actual timestep for decay schedule
            # For production: Default total_steps=1 uses alpha_end (stable exploitation)
            # Pass filtered candidates so cost/latency/quality constraints
            # are enforced.  Previously, corralling selected from ALL models,
            # silently ignoring max_cost, max_latency, and quality_floor.
            best_model, corralling_token = self.corralling_router.select_model(
                x, total_steps=total_steps, candidates=filtered
            )
            best_utility = 0.0  # Placeholder, corralling doesn't expose utility
        else:
            # Fallback: LinUCB selection with optional cost+latency penalty (paper Eq. 4)
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
            best_model, best_utility = self.bandit.select_arm(
                x, candidates=filtered, cost_penalties=cp
            )
        
        total_weight = 1.0

        # Record that this arm was selected so staleness inflation
        # distinguishes "feedback in flight" from "genuinely unused".
        # Only endorsing experts advance their clock — non-endorsing experts
        # did not play this arm and should not have their staleness reset.
        if self.use_corralling and self.corralling_router is not None:
            self.corralling_router.mark_selected(
                best_model,
                endorsing_experts=corralling_token.get("endorsing_experts"),
            )
        else:
            self.bandit.mark_selected(best_model)
        
        # Create routing log
        log = self._create_routing_log(
            prompt_text, best_model, best_utility, x, in_tok, output_tokens, total_weight
        )
        log.corralling_token = corralling_token

        # Persist context + corralling token for delayed feedback (RLHF).
        # Moved here from _create_routing_log so the corralling_token is
        # available at save time.  Without persisting the token, delayed
        # feedback that arrives after in-memory log eviction silently skips
        # the Corralling meta-weight update (only base experts learn).
        self.context_store.save_context(
            log.request_id, x, best_model,
            corralling_token=corralling_token,
        )
        
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
                            *max_latency*, *profile*).

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
        to [0, 1], and performs an importance-weighted update through the
        Corralling meta-learner (or a direct LinUCB update when Corralling is
        disabled).

        **Reward clamping**: Values outside [0, 1] are clipped silently.
        The importance-weighted loss estimator ``ℓ = (1 - r) / p``
        requires bounded rewards for valid regret guarantees.

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
            context, model_id, stored_token = self.context_store.get_context(
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
            log.corralling_token = stored_token
        
        # Reject non-finite rewards (NaN/inf) that would corrupt IPW estimates.
        if not np.isfinite(reward):
            logger.warning(
                "process_feedback: non-finite reward=%s for request_id=%s; "
                "skipping update.",
                reward, request_id,
            )
            return

        # Clamp reward to [0, 1] at the feedback entry point.
        # The Corralling importance-weighted loss estimator ℓ = (1 - r) / p
        # requires bounded rewards for valid regret guarantees (Auer et al., 2002).
        # reward > 1 would produce negative loss (artificially boosting an expert);
        # reward < 0 would spike the importance-weighted loss, destabilizing meta-weights.
        reward = float(np.clip(reward, 0.0, 1.0))
        
        # Use cached context vector to avoid re-encoding
        x = log.context_vector if log.context_vector is not None else self._get_context_vector(log.prompt)
        
        # Update corralling router if enabled
        if self.use_corralling and self.corralling_router is not None:
            # Pass the selection token so the importance-weighted
            # meta-weight update uses the correct expert_idx and probability.
            self.corralling_router.update(
                x, log.selected_model, reward,
                selection_token=getattr(log, 'corralling_token', None),
                advance_time=False
            )
        else:
            # Fallback: Update bandit directly
            self.bandit.update(log.selected_model, x, reward, advance_time=False)
        
        # Periodic stability check (cheap O(d) operation).
        # The canonical bandit is always live (under Corralling, the adapter
        # delegates update() to self.bandit, so self.bandit.t is incremented).
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
        models.  When Corralling is enabled, delegates to the warmup
        expert (expert 0) which receives all online observations.

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
        
        # Under corralling the warmup expert (experts[0]) delegates A/b/A_inv
        # to the canonical self.bandit — so calling self.bandit.get_probabilities
        # produces identical posteriors without duplicating 60+ lines of sampling
        # and jitter-fallback logic.
        return self.bandit.get_probabilities(x, models)

    def update(self, model_id: str, context: str | np.ndarray, reward: float, weight: float = 1.0, advance_time: bool = True) -> None:
        """
        Perform a direct bandit update (bypass ``process_feedback`` flow).

        Use this for batch/offline learning where you already have
        ``(model, context, reward)`` triples.  For the standard online
        workflow, prefer ``route()`` → ``process_feedback()``.

        Rewards are clamped to [0, 1].  When Corralling is enabled, the
        update is forwarded to the expert bandits but *not* the meta-weights
        (no selection token is available without a preceding ``route()``).

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
        
        if self.use_corralling and self.corralling_router:
            # Propagate weight so difficulty-based weighting
            # isn't silently dropped under Corralling.
            self.corralling_router.update(x, model_id, reward, weight=weight, advance_time=advance_time)
        else:
            self.bandit.update(model_id, x, reward, weight, advance_time=advance_time)
        
        # Periodic stability check — the canonical bandit receives all updates
        # (even under corralling, the adapter delegates to self.bandit).
        if (self.config.stability_check_interval > 0 and
            self.bandit.t % self.config.stability_check_interval == 0 and
            self.bandit.t > 0):
            for model in self.bandit.models:
                self.bandit._check_numerical_stability(model, self.config)

    # -------------------------------------------------------------------------
    # Observability: Feature Contribution Analysis
    # -------------------------------------------------------------------------
    
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
        # Under Corralling, delegate to the warmup expert (adapter).
        # The adapter's A_inv/b delegate to the canonical bandit.
        if self.use_corralling and self.corralling_router:
            expert = self.corralling_router.experts[0]
            with expert._lock:
                if model_id not in expert.A_inv:
                    raise ValueError(f"Model {model_id} not found in active expert")
                theta = expert.A_inv[model_id] @ expert.b[model_id]
        else:
            with self.bandit._lock:
                if model_id not in self.bandit.A_inv:
                    raise ValueError(f"Model {model_id} not found in bandit registry")
                theta = self.bandit.A_inv[model_id] @ self.bandit.b[model_id]
        
        # 2. Element-wise multiplication shows contribution of each feature
        contributions = theta * context_vector
        
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
        
        # Under Corralling, use the warmup expert's current state
        # (mirrors get_probabilities() and explain_decision() delegation).
        # Snapshot state under lock to prevent reading mid-update.
        if self.use_corralling and self.corralling_router:
            expert = self.corralling_router.experts[0]
            source_lock = expert._lock
            source_A_inv = expert.A_inv
            source_b = expert.b
            source_models = self.bandit.models
        else:
            source_lock = self.bandit._lock
            source_A_inv = self.bandit.A_inv
            source_b = self.bandit.b
            source_models = self.bandit.models
        
        # Snapshot ALL theta vectors under a single lock acquisition
        # so scoring and explanations are from the same consistent state.
        # Previously used separate lock acquisitions for scoring vs. explanation,
        # creating a TOCTOU gap where a concurrent update could cause the
        # per-model explanations to disagree with the top-k ranking.
        model_scores = []
        theta_cache = {}  # {model_id: theta} for explanation reuse
        with source_lock:
            for model_id in source_models:
                if model_id not in source_A_inv:
                    continue
                theta = source_A_inv[model_id] @ source_b[model_id]
                theta_cache[model_id] = theta
                score = float(np.dot(theta, x))
                model_scores.append((model_id, score))
        
        # Sort by score (highest first) and take top-k
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [m[0] for m in model_scores[:top_k]]
        
        # Generate explanations for top-k models using cached theta
        # (no second lock acquisition needed — uses the same snapshot)
        pca_dims = len(x) - 1  # All except last dimension (bias)
        explanations = {}
        for model_id in top_models:
            contributions = theta_cache[model_id] * x
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
        """
        Save the bandit's learned state to disk.
        
        KNOWN LIMITATION: Only the base DisjointLinUCBPolicy (A, b matrices)
        is persisted.  When Corralling is enabled, the expert bandits' learned
        state (both warmup and tabula rasa A/b matrices), meta-weights, and
        cumulative losses are NOT saved.  After a restart/reload, the Corralling
        layer resets to its initial 50/50 expert allocation and warmup-era
        expert state.  Extending persistence to the full Corralling stack is a
        natural enhancement for long-running production deployments.
        """
        self.bandit.save_state(path)

    def load_state(self, path: Path | str) -> None:
        """
        Load the bandit's learned state from disk.
        
        See save_state() for known limitations regarding Corralling persistence.
        """
        self.bandit.load_state(path)

    def _calculate_absolute_penalty(self, cost_per_1k: float) -> float:
        """
        Calculate stable 0.0-1.0 cost penalty based on Fixed Market Anchors.
        
        Uses Logarithmic Market Width to ensure penalties are absolute, 
        not relative to currently loaded models.
        
        Market Anchors (Mathematically Derived):
        - Floor: $0.0005/1k (DeepSeek V3, Flash, Haiku tier) → ln(0.0005) ≈ -7.60
        - Ceiling: $10.00/1k (Future o1-high/Opus tiers) → ln(10.00) ≈ +2.30
        - Range: 2.30 - (-7.60) = 9.90 → Use 10.0 for clean scaling
        
        Args:
            cost_per_1k: Cost in dollars per 1000 tokens
            
        Returns:
            Penalty in range [0.0, 1.0]
            - 0.0 = At or below market floor
            - 1.0 = At or above market ceiling
        """
        # Use precomputed market anchors for performance
        safe_cost = max(cost_per_1k, self._market_cost_floor)
        log_cost = math.log(safe_cost)
        
        # Normalize: (Current - Floor) / Range
        penalty = (log_cost - self._market_cost_floor_log) / self._market_cost_range
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, penalty))

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
# Corralling Router: Robust Warmup with Safety Guarantees
# ---------------------------------------------------------------------------

class CorrallingRouter:
    """
    Corralling meta-learner via Log-Barrier Online Mirror Descent (Agarwal et al., 2017).

    Maintains a distribution over K base experts and updates it using Online
    Mirror Descent (OMD) with a log-barrier regularizer.  Each expert receives
    a per-expert adaptive learning rate ``eta_i = eta_0 / sqrt(S_i + eps)``
    where ``S_i`` is the cumulative squared importance-weighted loss for
    expert *i*.  This adapts automatically: noisy experts with high-variance
    loss estimates get slower, more conservative learning rates.

    The log-barrier regularizer yields a closed-form weight update:
        ``w_{t+1,i} = w_{t,i} / (1 + eta_i * loss_hat_i * w_{t,i})``
    which naturally prevents expert death (weights cannot reach zero through
    the barrier), complementing the explicit gamma exploration floor.

    **High-Level Idea (Non-Technical):**
    Instead of betting everything on warmup priors, we hedge our bets by running
    both "warmup" and "tabula rasa" in parallel. Over time, we give more weight
    to whichever strategy is performing better.

    **Why This Matters:**
    If warmup priors are harmful (domain mismatch), the algorithm automatically
    shifts weight to tabula rasa. If warmup priors are helpful, they dominate.
    This provides safety guarantees against negative transfer.

    **Expert Death Prevention:**
    Two complementary mechanisms prevent expert death:
    1. The log-barrier regularizer makes it mathematically impossible for weights
       to reach zero through the OMD update alone.
    2. The mixing parameter (gamma) provides an additional explicit exploration
       floor, ensuring every expert maintains minimum probability gamma/K.

    **Non-Stationarity Scope:**
    The meta-learner incorporates loss decay (applied to cumulative squared
    losses) to adapt expert weights under non-stationary conditions (new models,
    traffic distribution shifts).  Each expert bandit is itself trained under a
    stationary-reward assumption (monotone A accumulation without forgetting).
    Meta-level non-stationarity only; extending expert-level updates with
    explicit forgetting is a direction for future work.

    **Theoretical Guarantee (vanilla Corralling):**
    Agarwal et al. (2017) prove that Corralling with Log-Barrier OMD achieves
    a master regret of O(sqrt(K * T * ln K)), which is near-optimal for
    combining K bandit experts over T rounds.  This bound holds under the
    standard assumptions: unbiased IPS loss estimates, deterministic experts,
    and a monotone (non-decaying) squared-loss accumulator.

    **Practical Extensions (void formal bound — empirically validated):**
    The following production-motivated extensions break one or more assumptions
    of the formal bound.  Each is disabled by default or set to a near-neutral
    value, so the vanilla algorithm is recoverable:

    - ``loss_decay`` (default 0.999): Decays cumulative squared losses for
      non-stationarity.  Set to 1.0 to recover the monotone accumulator.
    - ``meta_lr_halflife`` (default 60s): Staleness-aware meta-weight updates
      for delayed feedback.  Set to ``inf`` to disable.
    - ``ipw_clip`` (default 20.0): Caps importance weights fed to base experts,
      introducing bounded bias in exchange for variance reduction.
      Set to ``inf`` to recover unbiased IPS.
    - ``gamma`` floor (default 0.05): Explicit exploration floor complementing
      the log-barrier.  Part of the original algorithm; set to 0.0 for
      pure log-barrier exploration.

    **Deterministic expert assumption:**
    The loss estimator attributes loss to every expert that endorsed the played
    action: ``ℓ̂_j = I(expert_j → a) · ℓ / π(a)``.  This is unbiased when
    experts are deterministic (LinUCB argmax).  If experts become stochastic
    (e.g., Thompson Sampling), ``I(expert_j → a)`` must be replaced with
    ``P(expert_j → a)`` to maintain unbiasedness.

    **Empirical Validation (gamma=0.05):**
    - Validated across 4 dimensions using 18,750 trials (5 values x 5 seeds x 750 prompts)
    - Performance: 43.8 +/- 5.4 regret (near-optimal, <1% cost vs. gamma=0.0)
    - Safety: 80% variance reduction vs. gamma=0.0 (prevents stochastic expert death)
    - See: experiments/appendix/E_prior_degradation/results/gamma_ablation/

    **Computational Overhead:**
    - Memory: 2x (store two sets of A/b matrices)
    - Inference: O(Kd) extra (query all K experts; K=2 -> ~0.05ms)
    - Update: 2x (update both strategies, but they're independent)

    In practice, the overhead is negligible (~0.1ms) compared to LLM inference (~100ms).

    **Implementation Note:**
    At each round we query ALL K experts, compute the marginal action
    probability pi(a) = sum_j p_j * I(expert_j chose a), and attribute the
    importance-weighted loss l_obs / pi(a) to every expert that endorsed the
    played action.  This has dramatically lower variance than penalising only
    the sampled expert when experts agree on the same action, which is the
    common case.  With K=2 deterministic experts the extra query cost is O(d).

    Args:
        experts: List of bandit instances (typically [warmup_router, tabula_rasa_router])
        models: List of model IDs (must match across all experts)
        learning_rate: Base learning rate eta_0 for Log-Barrier OMD (default: 0.1).
               Per-expert effective rates are eta_0 / sqrt(S_i + eps).
        gamma: Mixing parameter gamma. Minimum prob for any expert is gamma/N.
               Prevents 'Expert Death' when the meta-learner's environment
               shifts. (default: 0.05, empirically validated as optimal across
               performance, safety, decisiveness, and predictability)

    Example:
        >>> # Create two experts
        >>> warmup = SimpleLinUCBRouter(models, warmup_priors, alpha=1.0)
        >>> tabula_rasa = TabulaRasaRouter(models, context_dim=33, alpha=1.0)
        >>>
        >>> # Wrap them in Corralling (Log-Barrier OMD)
        >>> corral = CorrallingRouter(experts=[warmup, tabula_rasa], models=models, gamma=0.05)
        >>>
        >>> # Use like any other router
        >>> selected = corral.select_model(context)
        >>> corral.update(context, selected, reward)
    """
    
    def __init__(
        self,
        experts: List,
        models: List[str],
        learning_rate: float = 0.1,
        gamma: float = 0.05,  # [VALIDATED] Empirically optimal (see experiments/appendix/E_prior_degradation/results/gamma_ablation/)
        loss_decay: float = 0.999,  # Meta-level adaptation decay
        meta_lr_halflife: float = 60.0,  # Staleness half-life in seconds for delayed feedback
        initial_weights: Optional[np.ndarray] = None,  # Prior-trust bias
        model_costs: Optional[Dict] = None,  # {model_id: {"normalized_cost": float}}
        epsilon: float = 1e-8,  # Regularizer for adaptive learning rate denominator
        ipw_clip: float = 20.0,  # Cap on importance weights fed to base experts
    ):
        """
        Initialize Corralling meta-learner (Log-Barrier OMD, Agarwal et al. 2017).

        Uses Online Mirror Descent with a log-barrier regularizer and per-expert
        adaptive learning rates.  Each expert's learning rate decays as
        ``eta_0 / sqrt(sum_squared_losses_i + epsilon)``, so noisy experts
        are down-weighted automatically.  The log-barrier naturally prevents
        expert death (weights cannot reach zero), complementing the explicit
        gamma exploration floor.

        The ``loss_decay`` parameter is a practical extension (not part of
        the original Corralling theory) that enables meta-level adaptation
        under non-stationarity by decaying the cumulative squared-loss
        history that drives the adaptive learning rates.

        NOTE: loss_decay operates at the META level only.  Each expert bandit
        is a stationary learner (monotone A accumulation, no forgetting).

        Args:
            learning_rate: Base learning rate eta_0 for the Log-Barrier OMD.
                       Per-expert effective rates are
                       ``eta_0 / sqrt(sum_squared_losses_i + epsilon)``.
            gamma: Mixing parameter gamma. Minimum prob for any expert is gamma/N.
                   Provides an explicit exploration floor beyond the log-barrier's
                   implicit prevention of zero weights.
            loss_decay: Decay factor applied to cumulative squared losses
                       (default: 0.999).  Controls how quickly the adaptive
                       learning rates forget old variance estimates.
                       - 1.0 = stationary (no decay, standard Corralling)
                       - 0.999 = mild adaptation (half-life ~693 steps)
                       - 0.99 = moderate adaptation (half-life ~69 steps)
            meta_lr_halflife: tau (seconds) for staleness-aware meta-weight learning.
                       When delayed feedback arrives, the meta-weight update's
                       effective learning rate is scaled by 1 / (1 + delay/tau).
                       Expert internal updates are always at full strength.
            initial_weights: Optional array of initial expert weights.
                       Must sum to 1 and have length == len(experts).
                       Default: uniform (1/K each).
            model_costs: Optional dict mapping model_id to cost metadata.
                       Stored for reference; cost/latency trade-offs are
                       handled by each expert's ``cost_penalty`` parameter.
            epsilon: Small constant added to the squared-loss denominator
                       for numerical stability (default: 1e-8).
            ipw_clip: Maximum importance weight ``weight / action_prob``
                       applied to base-expert updates (default: 20.0).
                       Pure IPW can produce extreme weights when an action's
                       marginal probability is small (e.g., with K=10 and
                       gamma=0.05, max theoretical IPW = K/gamma = 200).
                       Feeding uncapped weights into LinUCB's precision matrix
                       (A += w·xxᵀ) lets a single observation dominate A,
                       causing erratic exploration.  Clipping trades negligible
                       bias for large variance reduction — standard practice in
                       production bandit systems (e.g., Vowpal Wabbit's
                       ``--cb_type ips`` uses capped IPS by default).
                       Set to ``float('inf')`` to disable clipping.
        """
        self.experts = experts
        self.models = models
        self.eta_0 = learning_rate
        self.gamma = gamma
        self.loss_decay = loss_decay
        self.meta_lr_halflife = meta_lr_halflife
        self.model_costs = model_costs or {}
        self.n_experts = len(experts)
        self.epsilon = epsilon
        self.ipw_clip = ipw_clip
        # Thread safety — CorrallingRouter mutates shared state
        # (weights, sum_squared_losses) in update() and reads it in select_model().
        self._lock = threading.Lock()

        # Expert weights — uniform by default, or biased via initial_weights
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            if len(w) != self.n_experts:
                raise ValueError(
                    f"initial_weights length {len(w)} != n_experts {self.n_experts}"
                )
            if not np.isclose(w.sum(), 1.0):
                raise ValueError(
                    f"initial_weights must sum to 1, got {w.sum():.6f}"
                )
            self.weights = w.copy()
        else:
            self.weights = np.ones(self.n_experts) / self.n_experts

        # Per-expert cumulative squared importance-weighted losses.
        # Drives the adaptive learning rate: eta_i = eta_0 / sqrt(S_i + eps).
        # Initialized to 1.0 (not 0.0) to avoid degenerate initial learning
        # rates: with S_i=0, eta = eta_0/sqrt(eps) ≈ eta_0 * 10^4, causing
        # violent weight oscillations.  Initializing to 1.0 gives
        # eta ≈ eta_0 on the first step (the intended behavior), matching
        # standard AdaGrad practice (cf. TensorFlow's initial_accumulator_value).
        self.sum_squared_losses = np.ones(self.n_experts)

        # Exploit mode — when True, select_model picks argmax(weights)
        # deterministically instead of sampling.  Standard practice for
        # offline policy evaluation and frozen deployment.
        self.exploit_mode: bool = False

        # Diagnostics
        self.expert_selections = [0] * self.n_experts
        self.selections = {m: 0 for m in models}
    
    def _get_mixed_distribution(self) -> np.ndarray:
        """
        Compute P_t = (1-γ) * w_t + γ/K
        This mixes the learned policy (w_t) with uniform exploration (1/K).
        
        Returns:
            Mixed probability distribution over experts
        """
        uniform_dist = np.ones(self.n_experts) / self.n_experts
        return (1 - self.gamma) * self.weights + self.gamma * uniform_dist
    
    def select_model(self, context: np.ndarray, total_steps: int = 0,
                     candidates: List[str] | None = None) -> Tuple[str, Dict]:
        """
        Select model via Corralling: query ALL experts, sample from the marginal
        action distribution π(a) = Σ_j p_j · I(expert_j chose a).

        Querying all K experts is O(Kd).  With K=2, this is negligible
        compared to the LLM inference that follows (~0.1 ms vs ~100 ms).

        Args:
            context: Context vector for selection.
            total_steps: Total training steps (passed to experts for alpha decay).
            candidates: Optional list of eligible model IDs after constraint
                       filtering.  If provided, experts only score these models.

        Returns:
            Tuple of (selected_model_id, selection_token).
            The selection_token must be passed back to ``update()`` for correct
            importance-weighted meta-weight attribution.  Without it, the
            meta-weight update is skipped (only base experts learn).
        """
        with self._lock:
            probs = self._get_mixed_distribution()
            weights_snapshot = self.weights.copy()
            use_exploit = self.exploit_mode

        # Query ALL experts for their deterministic recommendations.
        recommendations = [
            expert.select_model(context, total_steps=total_steps,
                                candidates=candidates)
            for expert in self.experts
        ]

        if use_exploit:
            # Deterministic greedy: pick the highest-weight expert.
            # Ties broken by lowest index (the warmup expert by convention).
            expert_idx = int(np.argmax(weights_snapshot))
        else:
            # Stochastic Corralling: sample an expert from the mixed distribution.
            # Mathematically equivalent to sampling from π(·) when experts
            # are deterministic.
            expert_idx = np.random.choice(self.n_experts, p=probs)

        model = recommendations[expert_idx]

        # Marginal action probability: π(model) = Σ_j p_j · I(rec_j == model)
        action_prob = sum(
            float(probs[j]) for j in range(self.n_experts)
            if recommendations[j] == model
        )
        endorsing = [
            j for j in range(self.n_experts)
            if recommendations[j] == model
        ]

        with self._lock:
            self.expert_selections[expert_idx] += 1
            if model not in self.selections:
                self.selections[model] = 0
            self.selections[model] += 1

        selection_token = {
            "action_prob": float(action_prob),
            "endorsing_experts": endorsing,
            "timestamp": time.time(),
            "was_exploit": use_exploit,
        }

        return model, selection_token

    def _log_barrier_project(
        self, A: np.ndarray, eta: np.ndarray,
        tol: float = 1e-12, max_iter: int = 64,
    ) -> np.ndarray:
        """Project onto the probability simplex under the log-barrier regularizer.

        After the OMD mirror step, we have unprojected inverse weights
        ``A_i = 1/w_{t,i} + η_i · ℓ̂_{t,i}``.  The true projection finds
        the Lagrange multiplier ``μ`` such that::

            Σ_i  1 / (A_i + η_i · μ)  =  1

        and returns ``w_i = 1 / (A_i + η_i · μ)``.

        This is *not* equivalent to L1 normalization (``w̃ / Σ w̃``), which
        corresponds to the KL/negative-entropy projection.  The log-barrier
        projection preserves the Itakura-Saito geometry required for the
        O(√(T ln K)) master regret bound (Agarwal et al., 2017).

        Parameters
        ----------
        A : np.ndarray
            Unprojected inverse weights, shape ``(K,)``.
        eta : np.ndarray
            Per-expert adaptive learning rates, shape ``(K,)``.
        tol : float
            Convergence tolerance for the bisection root finder.
        max_iter : int
            Maximum bisection iterations (64 gives ~1e-19 precision).

        Returns
        -------
        np.ndarray
            Projected weights on the probability simplex, shape ``(K,)``.
        """
        K = len(A)

        # Unprojected weights (before projection)
        w_tilde = 1.0 / np.maximum(A, 1e-30)
        s = w_tilde.sum()

        # If already on the simplex (within tolerance), no projection needed
        if abs(s - 1.0) < tol:
            return w_tilde

        if K == 2:
            # Exact analytic solution via quadratic formula.
            # We solve: 1/(A_1 + η_1·μ) + 1/(A_2 + η_2·μ) = 1
            # Cross-multiply: (A_2 + η_2·μ) + (A_1 + η_1·μ) = (A_1 + η_1·μ)(A_2 + η_2·μ)
            # Rearranging: η_1·η_2·μ² + [η_1(A_2-1) + η_2(A_1-1)]·μ + (A_1·A_2 - A_1 - A_2) = 0
            e1, e2 = eta[0], eta[1]
            a1, a2 = A[0], A[1]

            qa = e1 * e2
            qb = e1 * (a2 - 1.0) + e2 * (a1 - 1.0)
            qc = a1 * a2 - a1 - a2

            disc = qb * qb - 4.0 * qa * qc
            if disc < 0:
                # Degenerate case — fall back to L1 normalization
                return w_tilde / s

            sqrt_disc = np.sqrt(disc)

            # Both roots: we need the one that keeps all weights positive.
            # w_i > 0 requires A_i + η_i·μ > 0  ⟹  μ > -A_i/η_i.
            # The larger root satisfies this when losses are non-negative.
            mu = (-qb + sqrt_disc) / (2.0 * qa)

            w = 1.0 / (A + eta * mu)

            # Validate: if numerical noise produces negative weights,
            # try the smaller root.
            if np.any(w < 0):
                mu = (-qb - sqrt_disc) / (2.0 * qa)
                w = 1.0 / (A + eta * mu)

            if np.any(w < 0):
                return w_tilde / s

            return w

        # General K > 2: bisection on μ.
        # f(μ) = Σ 1/(A_i + η_i·μ) is strictly decreasing in μ.
        # f(0) = Σ 1/A_i = Σ w̃_i.  We need f(μ) = 1.
        # Lower bound: μ must satisfy A_i + η_i·μ > 0 for all i,
        #   so μ > -min(A_i/η_i).
        mu_lo = -np.min(A / np.maximum(eta, 1e-30)) + tol
        mu_hi = 0.0 if s <= 1.0 else mu_lo

        # Expand upper bound until f(mu_hi) < 1
        if s > 1.0:
            mu_hi = 1.0
            for _ in range(max_iter):
                f_hi = np.sum(1.0 / (A + eta * mu_hi))
                if f_hi < 1.0:
                    break
                mu_hi *= 2.0
        else:
            # s <= 1: μ is negative — expand lower bound
            mu_lo_candidate = -1.0
            for _ in range(max_iter):
                if mu_lo_candidate <= mu_lo:
                    mu_lo_candidate = mu_lo
                    break
                f_lo = np.sum(1.0 / (A + eta * mu_lo_candidate))
                if f_lo > 1.0:
                    break
                mu_lo_candidate *= 2.0
            mu_lo, mu_hi = mu_lo_candidate, 0.0

        for _ in range(max_iter):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            f_mid = np.sum(1.0 / (A + eta * mu_mid))
            if abs(f_mid - 1.0) < tol:
                break
            if f_mid > 1.0:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid

        mu = 0.5 * (mu_lo + mu_hi)
        w = 1.0 / (A + eta * mu)

        if np.any(w < 0) or np.any(np.isnan(w)):
            return w_tilde / s

        return w

    def update(self, context: np.ndarray, model: str, reward: float,
               selection_token: Dict | None = None, weight: float = 1.0,
               advance_time: bool = True):
        """
        Two-level update: meta-weights (which expert to trust) + base-level
        (each expert's internal LinUCB learning).

        .. note::

            If the selection token was generated during exploit mode
            (``was_exploit=True``), the meta-weight update is silently
            skipped because deterministic selection invalidates the
            stochastic action probabilities in the token.  Base-expert
            updates still proceed normally.

        **Level 1 — Meta-Weight Update (Log-Barrier OMD, Agarwal et al. 2017):**
        Only performed when a valid ``selection_token`` is provided (returned
        by ``select_model()``).

        For the played action *a* with observed loss ℓ, the estimated loss
        for expert *j* is::

            ℓ̂_j = I(expert_j recommended a) · ℓ / π(a)

        where π(a) = Σ_j p_j · I(expert_j chose a) is the marginal
        probability of *a* under the mixed policy.  All experts that
        endorsed the chosen action share the same importance-weighted
        loss; experts that recommended a different action receive zero.

        The squared losses feed the per-expert adaptive learning rate
        ``eta_i = eta_0 / sqrt(S_i + eps)``.  The mirror step computes
        unprojected inverse weights ``A_i = 1/w_i + η_i · ℓ̂_i``, then
        projects onto the simplex via the true log-barrier (Itakura-Saito)
        projection: find μ s.t. ``Σ 1/(A_i + η_i·μ) = 1`` and set
        ``w_i = 1/(A_i + η_i·μ)``.  For K=2 this is an exact O(1)
        quadratic solve; for K>2 a fast bisection is used.

        When ``selection_token`` is None (e.g. external ``BanditRouter.update()``
        calls without a preceding ``select_model()``), the meta-weight update
        is skipped entirely to prevent using stale probabilities.

        **Staleness-Aware Meta-Learning Rate:**
        For delayed feedback (RLHF, human ratings), the meta-weight learning
        rate is scaled by 1 / (1 + delay/τ) where τ = meta_lr_halflife.

        Rationale:  The importance weight 1/π(a) captured at selection time
        becomes less reliable as meta-weights drift.  Rather than discarding
        delayed feedback entirely (losing signal) or trusting it fully (high
        variance), we smoothly discount the meta-weight update while keeping
        expert internal updates at full strength.

        - Fresh feedback (< τ):   meta-lr ≈ full  → unbiased, low variance
        - Stale feedback (≈ τ):   meta-lr ≈ 50%   → conservative update
        - Very stale (>> τ):      meta-lr → 0      → experts learn, meta stable

        **Level 2 — Base Algorithm Update (Importance-Weighted):**
        Base algorithms observe (context, model_played, reward), but updates
        are properly corrected for off-policy evaluation.

        Because the model was chosen by the mixed policy π, not necessarily by
        a specific expert, feeding uncorrected observations to all experts
        violates the independence assumptions underlying LinUCB regret analysis
        (correlated feedback bias).

        To maintain valid regret guarantees under Corralling theory, we
        apply Inverse Probability Weighting (IPW). An expert receives the update
        scaled by 1/π(a) if it endorsed the chosen action, and no update
        otherwise.

        **Overhead:** Up to K expert updates per observation (negligible
        compared to LLM inference latency).

        Args:
            context: Context vector used for selection.
            model: Model that was selected.
            reward: Observed reward (0-1 typically).
            selection_token: Token returned by ``select_model()`` containing the
                marginal action probability and endorsing expert list.  Required
                for correct importance-weighted meta-weight attribution.
                If None, only base experts are updated (no meta-weight change).
            weight: Observation importance weight (passed to expert updates).

        """
        # Exploit-mode guard: the determinism constraint belongs to the
        # *action* (token), not the router's current state.  Delayed feedback
        # from a stochastic selection is valid even if the router has since
        # switched to exploit mode.
        was_exploit = (
            selection_token.get("was_exploit", False) if selection_token else False
        )

        # ===================================================================
        # LEVEL 1: Meta-Weight Update (which expert to trust)
        # ===================================================================
        # Skip the meta-weight update if the selection was deterministic
        # (exploit mode).  The stochastic action_prob in the token is invalid
        # for IPW when selection was deterministic, but base-expert updates
        # (Level 2) remain valid.
        if selection_token is not None and not was_exploit:
            action_prob = max(selection_token["action_prob"], 1e-6)
            endorsing_experts = selection_token["endorsing_experts"]

            # ---------------------------------------------------------
            # Staleness-Aware Meta-Learning Rate
            # ---------------------------------------------------------
            token_time = selection_token.get("timestamp")
            if token_time is not None and self.meta_lr_halflife < float('inf'):
                delay_seconds = max(time.time() - token_time, 0.0)
                staleness_factor = 1.0 / (1.0 + delay_seconds / self.meta_lr_halflife)
            else:
                staleness_factor = 1.0

            observed_loss = 1.0 - reward
            losses = np.zeros(self.n_experts)

            # Importance-weighted loss estimator (Agarwal et al., 2017):
            #   ℓ̂_j = I(expert_j endorsed a) · ℓ_obs / π(a)
            # Since π(a) >= γ/K (bounded by the mixing parameter),
            # the estimator is bounded (max loss <= K/γ).
            iw_loss = observed_loss / action_prob
            for j in endorsing_experts:
                losses[j] = iw_loss

            with self._lock:
                # Adaptive per-expert learning rate (Corralling, Agarwal et al. 2017).
                # Decay old squared-loss history for non-stationarity, then
                # accumulate the new squared losses.
                #
                # The decay is applied as loss_decay**2 because S_i tracks
                # *squared* losses: if each raw loss decays by λ (i.e.,
                # ℓ_old → λ·ℓ_old), the squared term decays by λ²
                # (ℓ_old² → λ²·ℓ_old²).  Equivalently, this makes the
                # adaptive learning rate η_i = η_0/√S_i forget old variance
                # at the same effective rate as the loss decay.
                scaled_losses = staleness_factor * losses
                self.sum_squared_losses *= self.loss_decay ** 2
                self.sum_squared_losses += scaled_losses ** 2
                eta = self.eta_0 / np.sqrt(self.sum_squared_losses + self.epsilon)

                # Log-Barrier OMD mirror step (unprojected):
                #   A_i = 1/w_{t,i} + η_i · ℓ̂_{t,i}
                # The projected weight is p_i = 1/(A_i + μ) where μ is the
                # Lagrange multiplier enforcing Σ p_i = 1.
                inv_w = 1.0 / np.maximum(self.weights, 1e-30)
                A = inv_w + eta * scaled_losses

                self.weights = self._log_barrier_project(A, eta)

                # NaN guard: if overflow/underflow corrupts weights, reset
                # both weights AND sum_squared_losses to prevent cascading
                # corruption (NaN in S_i → NaN eta → NaN weights forever).
                if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                    logger.warning(
                        "CorrallingRouter: NaN/inf in weights after OMD update; "
                        "resetting to uniform."
                    )
                    self.weights = np.ones(self.n_experts) / self.n_experts
                    self.sum_squared_losses = np.ones(self.n_experts)
                else:
                    self.weights = np.maximum(self.weights, 1e-12)
        
        # ===================================================================
        # LEVEL 2: Base Algorithm Update (Importance-Weighted)
        # ===================================================================
        # To maintain the independence assumptions underlying the LinUCB regret
        # analysis, we must correct for off-policy evaluation. The model was
        # chosen by the mixed policy π, not necessarily by expert j.
        # We apply Inverse Probability Weighting (IPW): p_j_model / π_total_model.
        # Since our experts are deterministic, p_j_model is 1 if the expert
        # endorsed the model, and 0 otherwise. Thus, only endorsing experts
        # receive the update, scaled by 1 / π(a).
        #
        # ---------------------------------------------------------------
        # DESIGN NOTE: Why `weight` is passed to experts but NOT to the
        # meta-weight update (Level 1) above.
        # ---------------------------------------------------------------
        # The Corralling master uses importance-weighted loss
        #   ℓ_j = (1 - r) / π(a)   for each endorsing expert j
        # to correct for action-selection bias.  The regret guarantees
        # (Agarwal et al., 2017) assume an unweighted loss stream and
        # rely on the 1/π(a) factor being the *only* source of scaling.
        #
        # If we additionally multiplied by an application-level weight w_t,
        # the effective step size would become η·w_t/π(a), which can vary
        # across orders of magnitude, causing:
        #   (a) High-variance spikes that collapse expert probabilities
        #   (b) Difficulty tuning η when w_t is heavy-tailed
        #   (c) Conflation of two concerns — debiasing bandit feedback vs.
        #       encoding business importance — that are best kept separate
        #
        # Industry practice (ad-tech, recommendations, marketing bandits)
        # handles observation importance via one of:
        #   1. Encoding it into the reward itself (composite reward)
        #   2. Stratification (separate bandits for VIP vs. normal traffic)
        #   3. Offline resampling in supervised components
        #
        # **When this separation is safe:**  With homogeneous expert
        # architectures (same objective, differing only in initialization
        # or hyperparameters — e.g., warmup vs. tabula rasa LinUCB), the
        # expert that handles high-importance traffic well will naturally
        # produce higher rewards on those prompts, and the unweighted
        # meta-learner will favour it through the standard loss signal.
        #
        # **When weighted meta-regret matters:**  If future expert pools
        # include heterogeneous architectures (e.g., a cheap heuristic
        # alongside a neural model), an expert could exploit a shortcut
        # that performs well on high-volume low-value traffic while
        # failing on rare high-value prompts.  In that setting, prefer:
        #   (a) Stratified meta-learners (separate pools per traffic tier)
        #   (b) Composite rewards (reward = quality × importance)
        #   (c) Rolling-window-normalised weighted loss (last resort)
        # over naïve loss clipping, which introduces a hard-to-tune bound.
        # ---------------------------------------------------------------
        if selection_token is not None:
            # action_prob represents π(a) in the importance-weighted estimator.
            # Bounded below by γ/K, but we add a safety floor for numerical stability.
            action_prob = max(selection_token["action_prob"], 1e-6)
            endorsing_experts = selection_token["endorsing_experts"]
            for j, expert in enumerate(self.experts):
                if j in endorsing_experts:
                    ipw_weight = min(weight / action_prob, self.ipw_clip)
                    expert.update(context, model, reward, ipw_weight, advance_time=advance_time)
        else:
            # Fallback for direct updates without a preceding selection
            for expert in self.experts:
                expert.update(context, model, reward, weight, advance_time=advance_time)
    
    def mark_selected(
        self,
        model: str,
        endorsing_experts: Optional[List[int]] = None,
    ) -> None:
        """Advance the staleness clock only for experts that endorsed the action.

        Non-endorsing experts did not play *model* and will not receive an
        IPW update for it.  Marking them as having played the arm would
        under-inflate their uncertainty, suppressing exploration of off-policy
        arms under non-stationary forgetting (gamma < 1).

        Args:
            model: The model that was selected this round.
            endorsing_experts: Indices of experts that endorsed *model*.
                If ``None``, falls back to marking all experts (backward
                compatibility for callers that lack endorsement info).
        """
        for j, expert in enumerate(self.experts):
            if endorsing_experts is not None and j not in endorsing_experts:
                continue
            if hasattr(expert, 'mark_selected'):
                expert.mark_selected(model)

    def get_expert_weights(self) -> Dict[str, float]:
        """Get current expert weights for diagnostics."""
        return {
            f"expert_{i} ({type(self.experts[i]).__name__})": float(w) 
            for i, w in enumerate(self.weights)
        }
    
    def add_model(self, model_id: str) -> None:
        """
        Add model to the internal list (for dynamic model registration).
        
        Note: Experts must be updated separately via their own add_model() methods.
        This only updates the Corralling manager's model list and selection counters.
        
        Args:
            model_id: New model identifier
        """
        # Lock the check-then-append to prevent TOCTOU race where
        # concurrent calls both see the model as absent and double-append.
        with self._lock:
            if model_id not in self.models:
                self.models.append(model_id)
                # Initialize selection counter for new model
                self.selections[model_id] = 0
        logger.debug(f"✅ Added {model_id} to Corralling model list")

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle unpicklable threading.Lock."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_lock':
                setattr(result, k, threading.Lock())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


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

    rng = np.random.RandomState(42)
    for k in range(4):
        v = rng.randn(d)
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
                    f"🔧 Calibration pass 1 ({m}): bias prediction "
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
                    f"🔧 Calibration pass 2 ({m}): worst-case prediction "
                    f"{max_abs_pred:.2f} on probe '{worst_probe}' "
                    f"-> global theta scale {scale:.4f}"
                )
                bandit.b[m] = bandit.A[m] @ theta_new

        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Failed to calibrate prior for {m}: {e}")
            continue


# ---------------------------------------------------------------------------
# Prediction Monitor: Runtime score-range tracking for deployed routers
# ---------------------------------------------------------------------------

class PredictionMonitor:
    """
    Lightweight runtime monitor that tracks per-model prediction statistics
    to detect score-range anomalies (scale explosion, drift, collapsed arms)
    in deployed routers.
    
    **Why this exists:**
    Static `_calibrate_priors()` catches scale issues at initialization, but
    cannot detect problems that emerge after deployment:
    - Gradual drift as A/b accumulate biased updates
    - New models registered with bad priors after startup
    - Feature distribution shift (new prompt patterns activating high-PCA dims)
    
    **Design principles (production best practice):**
    1. O(1) per observation — no storage of individual predictions
    2. Per-model rolling statistics (min, max, mean, variance, count)
    3. Configurable alert threshold and cooldown to avoid log spam
    4. Thread-safe (shares the caller's lock context)
    5. Queryable health report for CI/canary checks
    
    **What it tracks per model:**
    - `expected_reward`: θ^T · x (the mean prediction, before UCB bonus)
    - `ucb_score`: full score including exploration bonus and cost penalty
    
    Usage:
        >>> monitor = PredictionMonitor(alert_threshold=2.0)
        >>> monitor.record("gpt-4", expected_reward=0.72, ucb_score=0.85)
        >>> health = monitor.get_health_report()
        >>> assert health["gpt-4"]["alerts"] == 0
    """
    
    def __init__(self, alert_threshold: float = 2.0, alert_cooldown: int = 100):
        """
        Args:
            alert_threshold: Absolute prediction value above which to log a warning.
                           Binary rewards should produce predictions in [0, 1]; values
                           above this threshold indicate possible scale explosion.
            alert_cooldown: Minimum observations between repeated alerts for the
                          same model (prevents log spam under sustained drift).
        """
        self.alert_threshold = alert_threshold
        self.alert_cooldown = alert_cooldown
        
        # Per-model statistics: {model_id: {metric: {min, max, sum, sum_sq, count}}}
        self._stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Per-model alert suppression counters
        self._alert_counter: Dict[str, int] = {}
    
    def _ensure_model(self, model_id: str):
        """Lazily initialize stats for a model on first observation."""
        if model_id not in self._stats:
            self._stats[model_id] = {
                "expected_reward": {"min": float("inf"), "max": float("-inf"),
                                    "sum": 0.0, "sum_sq": 0.0, "count": 0},
                "ucb_score": {"min": float("inf"), "max": float("-inf"),
                              "sum": 0.0, "sum_sq": 0.0, "count": 0},
            }
            # Start counter at cooldown so the very first
            # violation fires immediately instead of being silently swallowed.
            self._alert_counter[model_id] = self.alert_cooldown
    
    def record(self, model_id: str, expected_reward: float, ucb_score: float):
        """
        Record one prediction observation. O(1), no allocation.
        
        Args:
            model_id: Which model produced this prediction
            expected_reward: θ^T · x (mean prediction before UCB bonus)
            ucb_score: Full utility score (reward + exploration - cost)
        """
        self._ensure_model(model_id)
        
        for metric_name, value in [("expected_reward", expected_reward),
                                    ("ucb_score", ucb_score)]:
            s = self._stats[model_id][metric_name]
            s["min"] = min(s["min"], value)
            s["max"] = max(s["max"], value)
            s["sum"] += value
            s["sum_sq"] += value * value
            s["count"] += 1
        
        # Alert check (on expected_reward, the most interpretable signal)
        # Check for violation FIRST, then manage cooldown.
        # Previously, incrementing before checking meant the very first violation
        # was silenced (counter=1 < cooldown=100).  Now the first violation
        # always fires, and cooldown prevents repeated alerts for the SAME drift.
        # _alert_counter tracks observations-since-last-alert (not total violations).
        if abs(expected_reward) > self.alert_threshold:
            if self._alert_counter[model_id] >= self.alert_cooldown:
                count = self._stats[model_id]["expected_reward"]["count"]
                logger.warning(
                    f"⚠️ PredictionMonitor: {model_id} expected_reward="
                    f"{expected_reward:.4f} exceeds threshold "
                    f"±{self.alert_threshold} (observation #{count})"
                )
                self._alert_counter[model_id] = 0
            else:
                self._alert_counter[model_id] += 1
        else:
            # Non-violating observations still advance the cooldown counter
            self._alert_counter[model_id] += 1
    
    def get_health_report(self) -> Dict[str, Dict]:
        """
        Return a per-model health report suitable for CI checks or dashboards.
        
        Returns:
            Dict mapping model_id -> {
                "expected_reward": {min, max, mean, std, count},
                "ucb_score": {min, max, mean, std, count},
                "alerts": int  (number of threshold violations)
            }
        """
        report = {}
        for model_id, metrics in self._stats.items():
            model_report = {}
            alert_count = 0
            for metric_name, s in metrics.items():
                n = s["count"]
                if n == 0:
                    model_report[metric_name] = {
                        "min": None, "max": None, "mean": None,
                        "std": None, "count": 0
                    }
                    continue
                mean = s["sum"] / n
                variance = max(0.0, s["sum_sq"] / n - mean * mean)
                model_report[metric_name] = {
                    "min": s["min"],
                    "max": s["max"],
                    "mean": mean,
                    "std": variance ** 0.5,
                    "count": n,
                }
                # Count alerts: how many times max exceeded threshold
                if metric_name == "expected_reward":
                    if abs(s["max"]) > self.alert_threshold or abs(s["min"]) > self.alert_threshold:
                        alert_count += 1
            model_report["alerts"] = alert_count
            report[model_id] = model_report
        return report
    
    def reset(self, model_id: str | None = None):
        """
        Reset monitoring stats (e.g., after a planned recalibration).
        
        Args:
            model_id: Reset a specific model, or None to reset all.
        """
        if model_id is not None:
            self._stats.pop(model_id, None)
            self._alert_counter.pop(model_id, None)
        else:
            self._stats.clear()
            self._alert_counter.clear()


# ---------------------------------------------------------------------------
# Cost-Aware LinUCB Adapter: Thin wrapper for Corralling integration
# ---------------------------------------------------------------------------

class CostAwareLinUCBAdapter:
    """Thin adapter wrapping a shared :class:`DisjointLinUCBPolicy` for use as
    Expert 1 in :class:`CorrallingRouter`.

    Unlike the previous ``CostAwareLinUCBRouter`` which maintained a **copy** of
    the bandit's A/b matrices, this adapter holds a **shared reference** to the
    canonical ``DisjointLinUCBPolicy``.  All matrix state (A, b, A_inv, forgetting
    factor, regularization, Sherman-Morrison counters) lives in the bandit; the
    adapter adds only:

    - Alpha scheduling (constant or decaying exploration coefficient)
    - Cost and latency penalty integration in the UCB score
    - Runtime prediction monitoring

    **Why a shared reference?**

    Following the Google SmartChoices single-policy-adapter pattern:

    - Eliminates redundant O(K d^2) state and O(d^2) Sherman-Morrison per update
    - The main bandit stays *live* when corralling is active (previously it was a
      dead snapshot because ``process_feedback`` skipped ``self.bandit.update()``)
    - ``get_probabilities`` and ``explain_decision`` read from the same canonical
      state that routing uses
    - When corralling is disabled, the same bandit object handles routing directly

    **Thread safety:**

    ``select_model`` acquires ``self.bandit._lock`` (the global read lock) to
    take a consistent snapshot of scores across all arms.  ``update`` delegates
    to ``self.bandit.update()`` which uses per-model locks internally.
    """

    def __init__(
        self,
        bandit: 'DisjointLinUCBPolicy',
        model_costs: Dict[str, Dict[str, float]],
        alpha_start: float = 1.0,
        alpha_end: float = 0.1,
        cost_penalty: float = 0.0,
        latency_penalty: float = 0.0,
    ):
        """
        Args:
            bandit: Shared DisjointLinUCBPolicy instance (NOT copied).
            model_costs: Per-model cost metadata, each entry mapping
                        ``model_id -> {"normalized_cost": float,
                        "normalized_latency": float}``.
            alpha_start: Initial exploration coefficient.
            alpha_end: Final exploration coefficient after burn-in.
            cost_penalty: Weight for cost penalty (lambda_c).
            latency_penalty: Weight for latency penalty (lambda_l).
        """
        self.bandit = bandit
        self.model_costs = model_costs
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.t = 0
        self.prediction_monitor = PredictionMonitor(
            alert_threshold=2.0, alert_cooldown=100
        )

    # --- Properties delegating to the shared bandit ---

    @property
    def _lock(self) -> threading.Lock:
        """Delegate locking to the shared bandit's global read lock."""
        return self.bandit._lock

    @property
    def models(self) -> List[str]:
        return self.bandit.models

    @property
    def A(self) -> Dict[str, np.ndarray]:
        return self.bandit.A

    @property
    def b(self) -> Dict[str, np.ndarray]:
        return self.bandit.b

    @property
    def A_inv(self) -> Dict[str, np.ndarray]:
        return self.bandit.A_inv

    @property
    def last_played(self) -> Dict[str, int]:
        return self.bandit.last_played

    @property
    def context_dim(self) -> int:
        return self.bandit.dim

    def mark_selected(self, model: str) -> None:
        """Delegate to the shared bandit's selection tracker."""
        self.bandit.mark_selected(model)

    def get_current_alpha(self, total_steps: int) -> float:
        """Delegate to module-level :func:`_linear_alpha_decay`."""
        return _linear_alpha_decay(self.t, total_steps, self.alpha_start, self.alpha_end)

    def select_model(
        self,
        context: np.ndarray,
        total_steps: int = 0,
        candidates: List[str] | None = None,
    ) -> str:
        """Select the best model using cost-and-latency-aware LinUCB.

        Snapshots bandit state under lock, then computes O(d²) matrix math
        lock-free to avoid read-path contention under concurrent inference.

        Args:
            context: Context feature vector.
            total_steps: Total training steps (for alpha decay schedule).
            candidates: Optional constraint-filtered candidate list.

        Returns:
            Selected model identifier.
        """
        alpha = self.get_current_alpha(total_steps)

        with self.bandit._lock:
            eligible = candidates if candidates is not None else self.bandit.models
            snapshots = {}
            for model in eligible:
                if model not in self.bandit.A_inv:
                    continue
                meta = self.model_costs.get(model, {})
                snapshots[model] = (
                    self.bandit.A_inv[model],
                    self.bandit.b[model],
                    self.bandit._effective_staleness(model),
                    meta.get("normalized_cost", 1.0),
                    meta.get("normalized_latency", 1.0),
                )

        ucb_scores: Dict[str, float] = {}
        expected_rewards: Dict[str, float] = {}
        for model, (A_inv, b, dt, norm_cost, norm_latency) in snapshots.items():
            theta = A_inv @ b
            expected_reward = float(theta @ context)
            var = float(context @ A_inv @ context)
            var = _inflate_variance(var, self.bandit.gamma, dt)
            uncertainty = np.sqrt(max(var, 1e-12))
            score = (
                (expected_reward + alpha * uncertainty)
                - (self.cost_penalty * norm_cost)
                - (self.latency_penalty * norm_latency)
            )
            ucb_scores[model] = score
            expected_rewards[model] = expected_reward

        for model, score in ucb_scores.items():
            self.prediction_monitor.record(
                model,
                expected_reward=expected_rewards[model],
                ucb_score=float(score),
            )

        if not ucb_scores:
            eligible = candidates if candidates is not None else self.bandit.models
            raise NoModelScoredError(
                "CostAwareLinUCBAdapter.select_model() could not score any model. "
                f"candidates={eligible}"
            )

        return _argmax_random_tiebreak(ucb_scores)

    def update(
        self,
        context: np.ndarray,
        model: str,
        reward: float,
        weight: float = 1.0,
        advance_time: bool = True,
    ) -> None:
        """Delegate the update to the shared bandit.

        The bandit handles forgetting factor decay, Sherman-Morrison, proactive
        regularization, and periodic A_inv refresh internally.
        """
        self.bandit.update(model, context, reward, weight, advance_time=advance_time)
        if advance_time:
            self.t += 1

    def add_model(
        self,
        model_id: str,
        normalized_cost: float,
        normalized_latency: float = 1.0,
        **_kwargs,
    ) -> None:
        """Register cost metadata for a dynamically added model.

        The bandit arm itself is added by ``BanditRouter.register_model()``
        before this method is called; we only need to track the cost data
        used by ``select_model`` for penalty computation.
        """
        self.model_costs[model_id] = {
            "normalized_cost": normalized_cost,
            "normalized_latency": normalized_latency,
        }
        logger.debug(
            f"✅ Added {model_id} to LinUCB Adapter "
            f"(cost={normalized_cost:.2f}, latency={normalized_latency:.2f})"
        )

    def __deepcopy__(self, memo):
        """Deepcopy that resolves the shared bandit reference through *memo*.

        ``BanditRouter.__deepcopy__`` copies ``self.bandit`` first, placing it
        in *memo*.  When the corralling router (containing this adapter) is
        deepcopied afterwards, ``copy.deepcopy(self.bandit, memo)`` returns the
        already-cloned bandit, preserving the shared-reference invariant.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.bandit = copy.deepcopy(self.bandit, memo)
        result.model_costs = copy.deepcopy(self.model_costs, memo)
        result.alpha_start = self.alpha_start
        result.alpha_end = self.alpha_end
        result.cost_penalty = self.cost_penalty
        result.latency_penalty = self.latency_penalty
        result.t = self.t
        result.prediction_monitor = copy.deepcopy(self.prediction_monitor, memo)
        return result


class CostAwareTabulaRasaRouter:
    """
    Cost-and-latency-aware tabula rasa router (learns from scratch with cost/latency penalty).
    
    Uses Tikhonov regularization (Ridge regression) to prevent infinite initial uncertainty.
    Initializes A = λI where λ is automatically calculated based on empirical variance
    of reward signals or manually specified.
    Implements α-scheduling: Linear decay from α_start to α_end during burn-in.
    Supports forgetting factor (gamma) for non-stationary adaptation, with proactive
    regularization maintenance and staleness-inflated UCB.
    
    This is the "blank slate" expert in Corralling that learns purely from online data
    without warmup priors. Paired with CostAwareLinUCBAdapter (warmup expert) to provide
    robustness against domain mismatch.
    """
    def __init__(self, models: List[str], context_dim: int, model_costs: Dict,
                 alpha_start: float = 1.0, alpha_end: float = 0.1, cost_penalty: float = 0.0,
                 latency_penalty: float = 0.0,
                 ridge_lambda: Optional[float] = None, reward_std: Optional[float] = None,
                 forgetting_factor: float = 1.0):
        """
        Initialize tabula rasa router with automatic or manual ridge regularization.
        
        Args:
            models: List of model identifiers
            context_dim: Dimension of context vectors
            model_costs: Dict mapping model_id -> {"normalized_cost": float,
                       "normalized_latency": float}
            alpha_start: Initial exploration coefficient (default: 1.0)
            alpha_end: Final exploration coefficient (default: 0.1)
            cost_penalty: Weight for cost penalty (default: 0.0)
            latency_penalty: Weight for latency penalty (default: 0.0)
            ridge_lambda: Ridge regularization parameter (default: None, auto-calculated)
            reward_std: Standard deviation of rewards for auto-calculation (optional)
            forgetting_factor: Exponential decay for past observations
                             (1.0 = stationary, <1.0 = adaptive). Default 1.0.
        """
        self.models = models
        self.alpha_start = alpha_start  # Initial exploration (e.g., 1.0)
        self.alpha_end = alpha_end      # Final exploitation (e.g., 0.1)
        self.cost_penalty = cost_penalty
        self.latency_penalty = latency_penalty
        self.model_costs = model_costs
        self.t = 0  # Step counter for linear decay
        
        # Automatic Ridge Lambda Calculation
        # Based on empirical reward variance from 80k offline battles
        # Higher variance → stronger regularization needed
        if ridge_lambda is None:
            if reward_std is not None:
                ridge_lambda = max(1.0, 10.0 * reward_std)
                logger.info(f"Auto-calculated ridge_lambda={ridge_lambda:.2f} from reward_std={reward_std:.3f}")
            else:
                ridge_lambda = 1.0  # Safe default
        
        self.ridge_lambda = ridge_lambda
        # Store context_dim so add_model() can use it instead of
        # hardcoding 33 when no existing matrices are available.
        self.context_dim = context_dim
        # Thread safety — matches CostAwareLinUCBAdapter pattern.
        self._lock = threading.Lock()
        
        self.gamma = float(forgetting_factor)
        self.last_update: Dict[str, int] = {m: 0 for m in models}
        self.last_played: Dict[str, int] = {m: 0 for m in models}
        self.regularization_floor: Dict[str, float] = {
            m: self.ridge_lambda for m in models
        }
        
        # Bayesian Prior Regularization: A = λI
        # λ > 1: More regularization (smoother, evidence-based updates)
        # λ = 1: Standard identity (high initial uncertainty)
        # This prevents the "spiky jagged weights" from being purely random
        self.A = {m: self.ridge_lambda * np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        
        # Cache A_inv to avoid O(d³) recomputation on every select_model()
        self.A_inv = {m: safe_inv(self.A[m]) for m in models}
        
        # Runtime prediction monitor (matches CostAwareLinUCBAdapter)
        self.prediction_monitor = PredictionMonitor(
            alert_threshold=2.0, alert_cooldown=100
        )
        
        # Per-model Sherman-Morrison update counter for periodic refresh.
        self._sm_update_count: Dict[str, int] = {m: 0 for m in models}
    
    def get_current_alpha(self, total_steps: int) -> float:
        """Delegate to module-level :func:`_linear_alpha_decay`."""
        return _linear_alpha_decay(self.t, total_steps, self.alpha_start, self.alpha_end)

    def _effective_staleness(self, model: str) -> int:
        """Delegate to module-level :func:`_effective_staleness`."""
        return _effective_staleness(self.t, self.last_update, self.last_played, model)

    def mark_selected(self, model: str) -> None:
        """Record that *model* was deployed this round.

        Advances the global logical clock `self.t` by 1 so that `t` tracks
        the number of routing *requests*, not feedback arrivals.  This
        matches the convention in :class:`DisjointLinUCBPolicy`: staleness
        (`dt = t - last_played`) measures intervals between request events,
        making the forgetting factor `gamma**dt` independent of delayed
        feedback batch sizes.
        """
        with self._lock:
            self.t += 1
            self.last_played[model] = self.t

    def select_model(self, context: np.ndarray, total_steps: int = 0,
                     candidates: List[str] | None = None) -> str:
        """
        Select model using cost-and-latency-aware UCB with dynamic α (tabula rasa, no priors).

        Score = (Predicted Reward + α_t × Uncertainty) - λ_c × NormCost - λ_l × NormLatency
        """
        alpha = self.get_current_alpha(total_steps)

        # Snapshot references under lock; compute O(d²) math lock-free.
        with self._lock:
            eligible = candidates if candidates is not None else self.models
            snapshots = {}
            for model in eligible:
                if model not in self.A_inv:
                    continue
                meta = self.model_costs.get(model, {})
                snapshots[model] = (
                    self.A_inv[model],
                    self.b[model],
                    self._effective_staleness(model),
                    meta.get("normalized_cost", 1.0),
                    meta.get("normalized_latency", 1.0),
                )

        ucb_scores: Dict[str, float] = {}
        expected_rewards: Dict[str, float] = {}
        for model, (A_inv, b, dt, norm_cost, norm_latency) in snapshots.items():
            theta = A_inv @ b
            expected_reward = float(theta @ context)
            var = float(context @ A_inv @ context)
            var = _inflate_variance(var, self.gamma, dt)
            uncertainty = np.sqrt(max(var, 1e-12))
            score = (
                (expected_reward + alpha * uncertainty)
                - (self.cost_penalty * norm_cost)
                - (self.latency_penalty * norm_latency)
            )
            ucb_scores[model] = score
            expected_rewards[model] = expected_reward

        for model, score in ucb_scores.items():
            self.prediction_monitor.record(
                model, expected_reward=expected_rewards[model], ucb_score=float(score)
            )

        if not ucb_scores:
            eligible = candidates if candidates is not None else self.models
            raise NoModelScoredError(
                "CostAwareTabulaRasaRouter.select_model() could not score any model. "
                f"candidates={eligible}"
            )

        return _argmax_random_tiebreak(ucb_scores)

    def update(self, context: np.ndarray, model: str, reward: float, weight: float = 1.0,
               advance_time: bool = True):
        """
        Update arm-specific matrices via Sherman-Morrison.

        When forgetting_factor < 1.0, applies exponential decay to A and b before
        the rank-1 update, with proactive regularization maintenance to prevent
        the effective regularization from collapsing toward zero.

        Args:
            context: Context vector used for selection
            model: Model that was selected
            reward: Observed reward (0-1 typically)
            weight: Importance/difficulty weight (default 1.0)
            advance_time: Whether to increment `self.t`. Set to False when
                time was already advanced at route time (via `mark_selected`)
                to prevent double-counting feedback arrivals as requests.
        """
        if model not in self.A:
            return
        if weight < 0:
            logger.warning(
                f"Negative weight={weight:.4f} for {model}; "
                f"skipping update (negative weight would corrupt A_inv via sqrt(w))"
            )
            return
        if weight == 0:
            return

        x = context.flatten()

        with self._lock:
            # --- Forgetting factor: decay A, b, A_inv before rank-1 update ---
            # CostAwareTabulaRasaRouter uses a single unified lock (self._lock)
            # for all state, so there is no clock-snapshot race (unlike
            # DisjointLinUCBPolicy which has per-model + global locks).
            if self.gamma < 1.0:
                dt = self.t - self.last_update.get(model, 0)
                decay_factor = self.gamma ** min(dt, _MAX_STALENESS_DT)

                current_floor = self.regularization_floor.get(model, self.ridge_lambda)
                new_floor = current_floor * decay_factor

                if new_floor < self.ridge_lambda * _REGULARIZATION_FLOOR_FRACTION:
                    self.A[model] = (
                        self.A[model] * decay_factor
                        + self.ridge_lambda * np.eye(self.context_dim)
                    )
                    self.b[model] = self.b[model] * decay_factor
                    self.A_inv[model] = safe_inv(self.A[model])
                    self.regularization_floor[model] = self.ridge_lambda
                else:
                    self.A[model] *= decay_factor
                    self.b[model] *= decay_factor
                    self.A_inv[model] /= decay_factor
                    self.regularization_floor[model] = new_floor

                self.last_update[model] = self.t

            # --- Rank-1 Sherman-Morrison update (shared implementation) ---
            result = _sherman_morrison_update(
                A=self.A[model],
                A_inv=self.A_inv[model],
                b=self.b[model],
                x=x,
                reward=reward,
                weight=weight,
                init_lambda=self.ridge_lambda,
                regularization_floor=self.regularization_floor.get(model, self.ridge_lambda),
                model_name=model,
            )
            self.A[model] = result.A
            self.b[model] = result.b
            self.A_inv[model] = result.A_inv
            self.regularization_floor[model] = result.regularization_floor

            if advance_time:
                self.t += 1

            # Periodic full recomputation to correct accumulated float drift.
            self._sm_update_count[model] = self._sm_update_count.get(model, 0) + 1
            if self._sm_update_count[model] % 1000 == 0:
                self.A_inv[model] = safe_inv(self.A[model])
    
    def add_model(self, model_id: str, normalized_cost: float,
                  normalized_latency: float = 1.0) -> None:
        """
        Dynamically register a new model with cold-start state (for Corralling integration).
        
        This enables the Tabula Rasa expert to route to newly added models.
        Initializes with ridge regularization (Identity matrix) for maximum plasticity.
        
        Args:
            model_id: New model identifier
            normalized_cost: Cost penalty in [0, 1]
            normalized_latency: Latency penalty in [0, 1] (default 1.0, pessimistic)
        """
        with self._lock:
            if model_id not in self.models:
                self.models.append(model_id)
            
            dim = self.context_dim
            
            self.A[model_id] = self.ridge_lambda * np.eye(dim)
            self.b[model_id] = np.zeros(dim)
            self.A_inv[model_id] = safe_inv(self.A[model_id])
            self.model_costs[model_id] = {
                "normalized_cost": normalized_cost,
                "normalized_latency": normalized_latency,
            }
            self.last_update[model_id] = self.t
            self.last_played[model_id] = self.t
            self.regularization_floor[model_id] = self.ridge_lambda

        logger.debug(f"✅ Added {model_id} to Tabula Rasa Expert with cold start (ridge_λ={self.ridge_lambda:.2f})")

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle unpicklable threading.Lock."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('_lock',):
                setattr(result, k, threading.Lock())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class NonStationaryBudgetTracker:
    """Primal-dual budget controller with exponential moving average smoothing.

    Adjusts the cost penalty (dual variable) via projected stochastic gradient
    ascent on the Lagrangian relaxation of the per-query budget constraint.

    An **exponential moving average** (EMA) of observed costs replaces the
    previous uniform sliding window.  EMA provides equivalent variance
    reduction without the hard lag cliff that a finite window introduces: old
    observations decay exponentially (half-life ≈ ``window_size * ln2 / 2``)
    rather than vanishing abruptly after ``window_size`` steps.  This makes
    the dual variable more responsive to distribution shifts while retaining
    smoothness against bursty cost outliers.

    The smoothing factor is derived from ``window_size`` via the standard
    correspondence ``beta = 2 / (window_size + 1)``, so existing
    ``window_size`` tuning carries over with minimal re-calibration.
    """

    def __init__(
        self,
        target_cost_per_query: float,
        eta: float = 0.05,
        window_size: int = 100,
    ):
        """
        Args:
            target_cost_per_query: Target average normalized cost per query.
            eta: Learning rate for the dual variable update.
            window_size: Controls the EMA smoothing span.  Equivalent to the
                number of observations in a simple moving average with the
                same centre-of-mass lag (``beta = 2 / (window_size + 1)``).
        """
        self.target_cost_per_query = target_cost_per_query
        self.eta = eta
        self.window_size = window_size
        self.beta: float = 2.0 / (window_size + 1)

        # Initialise EMA to the target so the dual starts neutral.
        self._ema_cost: float = target_cost_per_query

        self.dual_weight: float = 0.5

    def update(self, actual_cost: float) -> None:
        """Incorporate a new cost observation and adjust the dual variable.

        Args:
            actual_cost: Normalised cost of the model selected this round.
        """
        self._ema_cost += self.beta * (actual_cost - self._ema_cost)

        gradient = self._ema_cost - self.target_cost_per_query
        self.dual_weight = max(0.0, self.dual_weight + self.eta * gradient)

    def get_cost_weight(self) -> float:
        """Return the current dynamically-tuned cost penalty weight."""
        return self.dual_weight


class ChebyshevCostAwareRouter(CostAwareTabulaRasaRouter):
    """
    Cost-aware router using Augmented Chebyshev Scalarization and Non-Stationary BwK.
    
    Instead of linear scalarization: Score = UCB - lambda * cost
    It uses Minimax distance: Score = - max(w_q * (Ideal_Q - Q), w_c * (Cost - Ideal_C))
    This guarantees finding points in non-convex regions of the Pareto frontier.
    
    The cost weight (w_c) is dynamically controlled by a NonStationaryBudgetTracker.
    """
    def __init__(self, models: List[str], context_dim: int, model_costs: Dict,
                 budget_tracker: NonStationaryBudgetTracker,
                 ideal_quality: float = 1.0,
                 quality_weight: float = 1.0,
                 rho: float = 0.05,
                 **kwargs):
        """
        Args:
            models: List of model identifiers
            context_dim: Dimension of context vectors
            model_costs: Dict mapping model_id -> normalized costs
            budget_tracker: NS-BwK tracker that provides the dynamic w_c penalty
            ideal_quality: Floor for the quality utopia point (default 1.0 for
                normalized rewards).  At each round the effective utopia is
                ``max(ideal_quality, max_ucb_across_arms)`` so that the UCB
                exploration bonus is never suppressed by clamping.
            quality_weight: w_q scaling factor for the quality distance (default 1.0)
            rho: Augmentation factor to prevent weakly Pareto optimal solutions (default 0.05)
        """
        # We don't use static cost_penalty from kwargs
        if 'cost_penalty' in kwargs:
            del kwargs['cost_penalty']
        super().__init__(models, context_dim, model_costs, cost_penalty=0.0, **kwargs)

        self.budget_tracker = budget_tracker
        self.ideal_quality = ideal_quality
        self.quality_weight = quality_weight
        self.rho = rho

        # Compute the ideal (utopia) cost which is the cheapest market floor among candidates
        costs = [meta.get("normalized_cost", 1.0) for meta in model_costs.values()]
        self.ideal_cost = min(costs) if costs else 0.0

    def select_model(self, context: np.ndarray, total_steps: int = 0,
                     candidates: List[str] | None = None) -> str:
        """Select model using Chebyshev Minimax distance with dynamic NS-BwK weights.

        Uses a **per-round adaptive utopia point** for quality: the ideal
        quality each round is ``max(ucb_quality)`` across eligible arms rather
        than the static ``self.ideal_quality``.  This prevents the UCB
        exploration bonus from being suppressed when optimistic bounds exceed
        the static utopia (which collapses quality distances to zero and
        degenerates selection into a pure cost-minimiser).

        Args:
            context: Context feature vector.
            total_steps: Total training horizon (for alpha decay).
            candidates: Optional constraint-filtered candidate list.

        Returns:
            Selected model identifier.

        Raises:
            NoModelScoredError: If no eligible model could be scored.
        """
        alpha = self.get_current_alpha(total_steps)
        w_c = self.budget_tracker.get_cost_weight()
        w_q = self.quality_weight

        # Snapshot references under lock; compute O(d²) math lock-free.
        with self._lock:
            eligible = candidates if candidates is not None else self.models
            snapshots = {}
            for model in eligible:
                if model not in self.A_inv:
                    continue
                meta = self.model_costs.get(model, {})
                snapshots[model] = (
                    self.A_inv[model],
                    self.b[model],
                    self._effective_staleness(model),
                    meta.get("normalized_cost", 1.0),
                )

        if not snapshots:
            eligible = candidates if candidates is not None else self.models
            raise NoModelScoredError(
                "ChebyshevCostAwareRouter.select_model() could not score "
                f"any model. candidates={eligible}"
            )

        # --- Pass 1: UCB quality and cost (lock-free) ---
        arm_stats: Dict[str, Tuple[float, float, float]] = {}
        for model, (A_inv, b, dt, norm_cost) in snapshots.items():
            theta = A_inv @ b
            expected_reward = float(theta @ context)
            var = float(context @ A_inv @ context)
            var = _inflate_variance(var, self.gamma, dt)
            uncertainty = np.sqrt(max(var, 1e-12))
            ucb_quality = expected_reward + alpha * uncertainty
            arm_stats[model] = (expected_reward, ucb_quality, norm_cost)

        round_ideal_q = max(
            self.ideal_quality,
            max(ucb for _, ucb, _ in arm_stats.values()),
        )

        # --- Pass 2: Chebyshev scalarization (lock-free) ---
        chebyshev_scores: Dict[str, float] = {}
        expected_rewards: Dict[str, float] = {}
        for model, (exp_r, ucb_q, norm_cost) in arm_stats.items():
            dist_q = round_ideal_q - ucb_q
            dist_c = max(0.0, norm_cost - self.ideal_cost)
            weighted_dist_q = w_q * dist_q
            weighted_dist_c = w_c * dist_c
            cheby_max = max(weighted_dist_q, weighted_dist_c)
            cheby_augmented = cheby_max + self.rho * (weighted_dist_q + weighted_dist_c)
            chebyshev_scores[model] = -cheby_augmented
            expected_rewards[model] = exp_r

        for model, score in chebyshev_scores.items():
            self.prediction_monitor.record(
                model,
                expected_reward=expected_rewards[model],
                ucb_score=float(score),
            )

        return _argmax_random_tiebreak(chebyshev_scores)

    def update(self, context: np.ndarray, model: str, reward: float, weight: float = 1.0,
               advance_time: bool = True):
        """
        Update the model parameters and feed the actual cost into the NS-BwK budget tracker.
        """
        model_meta = self.model_costs.get(model, {})
        actual_cost = model_meta.get("normalized_cost", 1.0)
        
        # Update the dynamic cost penalty based on the chosen model's cost
        self.budget_tracker.update(actual_cost)

        # Proceed with the standard LinUCB parameter update
        super().update(context, model, reward, weight, advance_time)
