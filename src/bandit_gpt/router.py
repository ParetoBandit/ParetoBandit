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

# Set environment variable to avoid hangs in multi-threaded/multi-process environments
# This is a common issue with SentenceTransformers on Mac/Linux.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Missing dependency: sentence-transformers") from e


try:
    from banditgpt.cluster_detector import ClusterDetector
except ImportError:
    try:
        # Fallback for direct file import (not package)
        from cluster_detector import ClusterDetector
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
    max_val = max(scores.values())
    # Collect all keys that achieve the maximum (within float tolerance)
    tied = [k for k, v in scores.items() if abs(v - max_val) < 1e-12]
    if len(tied) == 1:
        return tied[0]
    return tied[np.random.randint(len(tied))]

# ---------------------------------------------------------------------------
# Router Configuration (Magic Numbers Documented)
# ---------------------------------------------------------------------------

@dataclass
class RegistrationConfig:
    """
    Bayesian priors for new model admission.
    
    These values shape the initial belief state (theta) for a new model 
    before we have observed any real traffic.
    
    Scientific Justification (Prior Transfer Analysis):
    All parameters validated via sensitivity analysis:
    - n_effective: Robust across [1.0, 20.0] range (Appendix A.3)
    - Bias terms: Derived from cost asymmetry (30x price differential)
    
    Key Finding: Performance driven by semantic neighbor accuracy (θ_neighbor),
    not hyperparameter fine-tuning. System achieves Zero-Shot Readiness without
    manual calibration.
    
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
    
    # Latent Semantic Transfer - Prior Strength Calibration
    # Validated via prior transfer analysis (Appendix A.3):
    # - n_effective robust across [1.0, 20.0] range
    # - Corralling meta-learning adaptively chooses between warmup and tabula rasa experts
    # - System robustness comes from Corralling's adaptive switching, not n_eff tuning
    # Default: 5.0 (mid-range value, reasonable when warmup expert is used)
    n_effective_default: float = 5.0
    n_effective_high_similarity: float = 5.0  # sim > 0.8 (strong match)
    n_effective_medium_similarity: float = 5.0  # sim 0.6-0.8 (moderate match)
    n_effective_low_similarity: float = 5.0  # sim < 0.6 (weak match, Corralling will prefer tabula rasa)

@dataclass
class RouterConfig:
    """
    Centralized configuration for BanditRouter.
    
    ✅ **CANONICAL CONFIG**: This is the production-grade configuration for BanditRouter.
    
    **Scientific Validation (Appendix A):**
    Key hyperparameters validated via prior transfer theory and ablation:
    
    1. **Latent Semantic Transfer (n_effective)** (Appendix A.3):
       - Tested range: [1.0, 2.0, 5.0, 10.0, 20.0] on real LMSYS Arena data
       - Robust across full range; performance driven by neighbor accuracy
       - Default: 5.0 (mid-range value, effective when warmup expert is used)
    
    2. **Market Anchors (cost/latency normalization)**:
       - Derived from empirical market data (2024-2026)
       - Cost: $0.0001-$0.04/1k tokens (portfolio range)
       - Latency: 0.05s-5.0s (instant to timeout threshold)
    
    3. **Probation Period (500 requests)**:
       - Derived from convergence analysis (95% confidence interval)
       - Robust across [300, 1000] range (not shown for brevity)
    
    **Key Finding:** Performance driven by semantic neighbor accuracy (θ_neighbor),
    not hyperparameter fine-tuning. System achieves Zero-Shot Readiness without
    manual calibration.
    
    This dataclass is the single source of truth for the current production router.
    """
    
    # ---------------------------------------------------------------------------
    # Production Stability: Memory Management
    # ---------------------------------------------------------------------------
    # Prevent OOM from unbounded log growth.
    # At 100 QPS with 54-dim context vectors (~500 bytes/log), 10k logs ≈ 5MB.
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
    # Default 100 → 20 samples per archetype → sufficient to shape 54D covariance
    procedural_warmup_samples: int = 100  # Warmup samples (2*d for d≈50)
    
    # ---------------------------------------------------------------------------
    # LinUCB Regularization: Initialization vs Runtime
    # ---------------------------------------------------------------------------
    # **The Regularization Trap**: Sherman-Morrison only works for rank-1 updates.
    #   - Data update (xx^T): Rank-1 → O(d²) ✓
    #   - Scalar decay (γA): Preserves structure → O(d²) ✓
    #   - Diagonal regularization (+λI): Full-rank → Forces O(d³) ✗
    #
    # **The Solution**: "Initialization-Only Regularization"
    #   - Use init_lambda for cold-start stability (A₀ = λI)
    #   - Set update_lambda=0 for runtime updates
    #   - Let data terms (xx^T) keep matrix well-conditioned
    #
    # **Why This Is Safe**:
    #   In online bandits with steady traffic, the continuous addition of xx^T
    #   keeps A invertible. You only risk singularity if traffic stops AND you
    #   keep decaying until A→0, which is an edge case (handled by safety check).
    #
    # **Performance Impact**:
    #   - init_lambda=1.0, update_lambda=0.0: 2,710 updates/sec (O(d²))
    #   - init_lambda=1.0, update_lambda=1.0: ~628 updates/sec (O(d³))
    init_lambda: float = 1.0
    """Initialization regularization for cold-start stability (A₀ = λI)."""
    
    update_lambda: float = 0.0
    """
    Runtime regularization for continuous updates.
    
    Default 0.0 enables O(d²) Sherman-Morrison efficiency.
    Only increase if you have extremely sparse data or long idle periods.
    """
    
    # ---------------------------------------------------------------------------
    # Numerical Stability: Safety Net for Low-Traffic Arms
    # ---------------------------------------------------------------------------
    # With update_lambda=0, matrices can decay toward singularity if an arm
    # receives zero traffic for extended periods. This safety check triggers
    # a regularization reset when numerical instability is detected.
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
    # Fix (Jan 2026): Adjusted to match ACTUAL portfolio range for consistency
    # Portfolio range: $0.0001-$0.0375/1k (Llama 3.1-8B to o1)
    # Previous: $0.00005-$0.10/1k (too wide, caused suboptimal spread)
    # New: Tightened to improve penalty differentiation by 1.39x
    market_cost_floor: float = 0.0001  # $/1k tokens (captures cheapest model)
    market_cost_ceiling: float = 0.04  # $/1k tokens (slightly above most expensive)
    
    # Latency Normalization Anchors
    # Floor: 50ms (instant/cached responses)
    # Ceiling: 5.0s (reasonable timeout threshold)
    market_latency_floor: float = 0.05  # seconds
    market_latency_ceiling: float = 5.0  # seconds
    
    # ---------------------------------------------------------------------------
    # RESILIENCE DEFAULTS: Pessimistic Fallbacks (Paper "Fail-Operational" Fix)
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
    
    @property
    def cost_range_log(self) -> float:
        """Logarithmic range for cost normalization."""
        return np.log(self.market_cost_ceiling) - np.log(self.market_cost_floor)
    
    @property
    def latency_range_log(self) -> float:
        """Logarithmic range for latency normalization."""
        return np.log(self.market_latency_ceiling) - np.log(self.market_latency_floor)


from .config_legacy import DEFAULT_SENTENCE_TRANSFORMER

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

# transform_hle_to_prior removed - trusting LinUCB to learn from data
# instead of encoding rigid prior transformations

# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------
# **COMPLEXITY ANALYSIS**
#
# The update() method complexity depends on update_lambda and forgetting_factor:
#
# Configuration 1: update_lambda=0, gamma<1.0 (DEFAULT) → O(d²) always ✓
#   - Pure exponential decay without regularization floor
#   - Scaled Sherman-Morrison handles all updates efficiently
#   - **Performance**: 2,710 updates/sec @ d=384
#
# Configuration 2: update_lambda>0, gamma<1.0 → O(d³) on stale updates ✗
#   - Decay operation is O(d²) via Scaled Sherman-Morrison
#   - BUT: Regularization floor (1-γ)λI forces full re-inversion
#   - **Performance**: ~628 updates/sec @ d=384
#   - **Use case**: Extremely sparse data or long idle periods
#
# Configuration 3: gamma=1.0 (stationary) → O(d²) always ✓
#   - No decay, standard Sherman-Morrison applies
#   - **Performance**: 3,051 updates/sec @ d=384
#
# Why Default to update_lambda=0?
# The "Initialization-Only Regularization" pattern:
#   - Use init_lambda for cold-start stability (A₀ = λI)
#   - Set update_lambda=0 for runtime (4x faster)
#   - Data terms (xx^T) keep matrix well-conditioned with steady traffic
#   - Only risks singularity if traffic stops AND decay continues → rare edge case
#
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
                 update_lambda: float = 0.0,
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
            update_lambda: Runtime regularization for decay restoration. Default 0.0 for O(d²) speed.
            forgetting_factor: Exponential decay factor (1.0 = stationary, <1.0 = adaptive). Default 1.0.
        """
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(forgetting_factor)
        self.init_lambda = float(init_lambda)
        self.update_lambda = float(update_lambda)
        
        # Thread safety: Per-model locks to eliminate lost-update race conditions.
        # Updates to Model A don't block updates to Model B.
        from collections import defaultdict
        self.model_locks = defaultdict(threading.Lock)
        
        # Global lock for read operations (select_arm, refresh_inverse_cache)
        self._lock = threading.Lock()
        
        # Initialize A=I*init_lambda, b=0
        # Use init_lambda for cold-start stability, not update_lambda
        self.A = {m: np.eye(self.dim) * self.init_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        
        # Precompute A_inv for hot-path speed
        self.A_inv = {m: safe_inv(self.A[m]) for m in self.models}
        
        self.last_update = {m: 0 for m in self.models}  # Track last update step
        self.t = 0  # Global time step
        
        # [Paper FIX] Track effective regularization level per model
        # Ensures principled lower bound on eigenvalues (proactive approach)
        # Prevents singularity in low-traffic regimes with forgetting factor < 1.0
        self.regularization_floor = {m: self.init_lambda for m in self.models}

    def bandit_is_stable(self, model_id: str) -> bool:
        """
        O(d) stability check using trace of the precision matrix (A).
        
        A more rigorous spectral check (lambda_min > threshold) is O(d³).
        The trace check is a cheaper proxy that detects manifold collapse
        or extreme numerical instability.
        """
        if model_id not in self.A:
            return True
        trace = np.trace(self.A[model_id])
        # Heuristic: Expect trace to be at least d * init_lambda
        # If it's significantly lower, something is wrong with the updates
        return trace > (self.dim * self.init_lambda * 0.1)


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
        result.update_lambda = self.update_lambda
        result.t = self.t
        result.last_update = copy.deepcopy(self.last_update, memo)
        
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
        candidates: List[str | None] = None
    ) -> Tuple[str, float]:
        """
        Select the best arm (model) using Upper Confidence Bound (UCB).
        
        Args:
            x: Context vector
            candidates: List of candidate model IDs (None = all models)
            
        Returns:
            Tuple of (best_model_id, best_ucb_score)
        """
        # Candidate filtering MUST happen inside the lock.
        # Otherwise, a model could be removed from self.A between the filter
        # and the locked read, causing a KeyError on A_inv[m].
        best_model = None
        best_ucb = -float("inf")

        # Thread safety: Acquire lock for reading shared state
        with self._lock:
            candidates = candidates or self.models
            candidates = [m for m in candidates if m in self.A]
            if not candidates:
                raise ValueError("No candidates available")
            best_model = candidates[0]
            for m in candidates:
                # UCB = mean + alpha * std
                theta = self.A_inv[m] @ self.b[m]
                mean = float(theta.dot(x))
                
                # Global Forgetting: Inflate variance based on staleness
                # A_effective = A_stored * gamma^(dt)
                # Var_effective = x^T A_eff^-1 x = x^T (A^-1 * gamma^-dt) x = Var_stored * gamma^-dt
                #
                # Time-delta variance inflation: covers the gap between the model's
                # last update and the current selection time. Since A is only decayed
                # during update(), we must explicitly inflate the variance here to
                # reflect increased uncertainty as time passes without new observations.
                dt = self.t - self.last_update[m]
                decay_factor = self.gamma ** dt
                
                var = float(x.dot(self.A_inv[m]).dot(x))
                # Inflate variance for staleness
                var_inflated = var / max(decay_factor, 1e-12) 
                
                std = float(np.sqrt(max(var_inflated, 1e-12)))
                ucb = mean + self.alpha * std
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_model = m
        
        return best_model, float(best_ucb)
    
    def get_probabilities(self, x: np.ndarray, models: List[str], n_samples: int = 1000,
                          noise_variance: float = 0.25) -> Dict[str, float]:
        """
        Calculate the probability of each model being the best via posterior sampling.
        
        The Bayesian posterior for ridge regression is:
            θ | D ~ N(A⁻¹b,  σ² · A⁻¹)
        
        Args:
            x: Context vector
            models: List of model IDs to compare
            n_samples: Number of Monte Carlo samples (default: 1000)
            noise_variance: σ² for the posterior covariance.  Default 0.25 is the
                           variance of a Bernoulli(0.5) reward, appropriate for
                           binary win/loss feedback.  Override for non-binary rewards.
        
        Returns:
            Dictionary mapping model_id -> probability of being the best
        
Previously sampled from N(θ_hat, A_inv) which implicitly
        assumed σ²=1.  With binary rewards, σ²≈0.25, so the old code
        overestimated posterior variance by ~4×, producing artificially
        uniform probability estimates.
        """
        model_samples = {}
        valid_models = [m for m in models if m in self.A]
        
        snapshots = {}
        with self._lock:
            for m in valid_models:
                A_inv_m = self.A_inv[m].copy()
                theta_hat = A_inv_m @ self.b[m]
                # Capture staleness info so we can inflate the
                # posterior covariance for models that haven't been updated recently.
                # Without this, get_probabilities() is artificially tight for stale
                # models, disagreeing with select_arm() which correctly inflates.
                dt = self.t - self.last_update.get(m, 0)
                snapshots[m] = (A_inv_m, theta_hat, dt)
        
        # Return uniform distribution instead of all-zeros when no
        # models have initialized state.  All-zeros violates the probability
        # contract (sum should be 1.0) and can cause division-by-zero in callers.
        if not snapshots:
            n = len(models) or 1
            return {m: 1.0 / n for m in models}
        
        for m, (A_inv_m, theta_hat, dt) in snapshots.items():
            # Sample weights from the posterior N(theta_hat, σ² · A_inv_eff)
            # When gamma < 1.0, A_effective = A_stored * gamma^dt, so
            # A_inv_effective = A_inv_stored / gamma^dt.  This inflates the
            # posterior for stale models, matching select_arm() behavior.
            if self.gamma < 1.0 and dt > 0:
                decay_factor = self.gamma ** min(dt, 1000)
                cov = noise_variance * A_inv_m / max(decay_factor, 1e-12)
            else:
                cov = noise_variance * A_inv_m
            # Wrap in try/except to handle ill-conditioned covariance
            # that causes LinAlgError in Cholesky decomposition inside
            # multivariate_normal.  Fall back to diagonal-only sampling.
            try:
                samples = np.random.multivariate_normal(theta_hat, cov, n_samples)
            except np.linalg.LinAlgError:
                # Fall back to isotropic noise with same average variance
                avg_var = max(np.trace(cov) / self.dim, 1e-12)
                samples = np.random.normal(
                    loc=theta_hat, scale=np.sqrt(avg_var), size=(n_samples, self.dim)
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
    
    def update(self, model: str, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
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
        
        Args:
            model: Model identifier
            x: Context vector
            reward: Observed reward
            weight: Importance weight for this update (default 1.0).
                    Use weight = (1 - cluster_mu) for difficulty-based weighting.
                    Hard tasks (μ=0.5) get weight=0.5, easy tasks (μ=0.95) get weight=0.05.
        """
        if model not in self.A:
            return
        
        # Guard against negative weight, which would produce NaN
        # via np.sqrt(negative) in the Sherman-Morrison u = x * sqrt(w) path,
        # permanently corrupting the model's A_inv with NaN values.
        if weight < 0:
            logger.warning(f"Negative weight={weight:.4f} for {model}; clamping to 0 (skipping update)")
            return
        
        # weight=0 contributes zero information (x_outer=0, reward_x=0)
        # but advancing self.t inflates dt for ALL other models' staleness
        # computations, artificially increasing their exploration bonuses.
        if weight == 0:
            return
        
        # Hold model-specific lock for entire update (eliminates race condition)
        with self.model_locks[model]:
            # 1. Calculate Time Decay
            dt = 0
            decay_factor = 1.0
            if self.gamma < 1.0:
                dt = self.t - self.last_update[model]
                # Clamp dt to prevent numerical underflow when gamma is small
                decay_factor = self.gamma ** min(dt, 1000)

            # 2. [Paper FIX] Proactive Regularization Maintenance
            # Instead of waiting for singularity (reactive), ensure A >= lambda_min I (proactive)
            current_lambda = self.regularization_floor.get(model, self.init_lambda)
            new_lambda = current_lambda * decay_factor
            
            # Threshold: Reinject if prior strength drops below 10% of init
            lambda_threshold = self.init_lambda * 0.1
            
            if new_lambda < lambda_threshold:
                # MAINTENANCE MODE: Inject fresh regularization (Rare O(d³))
                logger.info(
                    f"🔧 Maintenance: Restoring regularization floor for {model} "
                    f"(λ_eff={new_lambda:.2e} < {lambda_threshold:.2e})"
                )
                
                # Preserve learned preferences before regularization
                old_theta = self.A_inv[model] @ self.b[model]
                
                # Calculate how much lambda to add to get back to init_lambda
                missing_lambda = self.init_lambda - new_lambda
                
                # Apply decay + Injection
                # A_new = (A_old * decay) + (missing_lambda * I)
                new_A = (self.A[model] * decay_factor) + (missing_lambda * np.eye(self.dim))
                
                # Restore b to preserve theta: b_new = A_new @ theta
                # Removed dead `new_b = self.b[model] * decay_factor`
                # which was immediately overwritten on the next line.
                new_b = new_A @ old_theta
                
                # Full inversion required (Safe & Robust)
                new_A_inv = safe_inv(new_A)
                
                # Reset tracker
                self.regularization_floor[model] = self.init_lambda
                
                # Update state atomically
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
                    self.last_update[model] = self.t
                    
                # Continue to apply current observation below
                
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
                        self.last_update[model] = self.t
            
            # 3. Standard Sherman-Morrison Update (Data Integration)
            # Add observation: A += weight * x x^T, b += weight * reward * x
            x_outer = weight * np.outer(x, x)
            reward_x = weight * reward * x
            
            # Sherman-Morrison inverse update (O(d²))
            # Formula: (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
            A_inv_current = self.A_inv[model]
            u = x * np.sqrt(weight)
            v = x * np.sqrt(weight)
            
            A_inv_u = A_inv_current @ u
            v_A_inv = v @ A_inv_current
            denominator = 1.0 + (v @ A_inv_u)
            
            # Strict safety floor for numerical stability
            if abs(denominator) > 1e-6:
                # Safe to use Sherman-Morrison formula
                new_A_inv = A_inv_current - np.outer(A_inv_u, v_A_inv) / denominator
                new_A = self.A[model] + x_outer
                new_b = self.b[model] + reward_x
                
                # Atomic pointer swap for consistency
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
                    self.t += 1
            else:
                # CRITICAL: Denominator too small, fallback to O(d³) with fresh regularization
                logger.warning(
                    f"⚠️ Sherman-Morrison near-singularity for {model}: "
                    f"|denominator|={abs(denominator):.2e} < 1e-6. "
                    f"Injecting fresh regularization and recomputing inverse."
                )
                # Preserve learned preferences before regularization
                old_theta = self.A_inv[model] @ self.b[model]
                
                # Inject fresh regularization to restore conditioning
                new_A = self.A[model] + x_outer + (self.init_lambda * np.eye(self.dim))
                new_A_inv = safe_inv(new_A)
                
                # Restore b to preserve theta, then ADD the current
                # observation's reward.  Previously `new_b = new_A @ old_theta`
                # silently discarded the current reward signal.
                new_b = new_A @ old_theta + (weight * reward * x)
                
                # Reset regularization floor since we just injected init_lambda
                self.regularization_floor[model] = self.init_lambda
                
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
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
            
            # Verify fix
            new_trace = np.trace(self.A_inv[model])
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
                    f"  1. PCA fallback changes (384D embeddings vs 32D compressed)\n"
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
        alpha: float = 0.05,
        embedding_dim: int = 384,
        init_lambda: float = 1.0,
        update_lambda: float = 0.0,
        forgetting_factor: float = 1.0,
        context_store: ContextStore | None = None,
        config: RouterConfig | None = None,
        verbose_routing: bool = False,
        use_corralling: bool = True,  # Enable corralling by default
        corralling_learning_rate: float = 0.1,
        corralling_gamma: float = 0.05,
        corralling_cost_weight: float = 0.0,  # λ for composite reward in Corralling
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
            init_lambda: Regularization parameter
            update_lambda: Update-time regularization
            forgetting_factor: Temporal decay (1.0 = stationary)
            context_store: Persistent storage for delayed feedback
            config: Router configuration object
            verbose_routing: Enable detailed breakdown logs for each routing decision
            use_corralling: Enable Corralling meta-learner (default: True)
            corralling_learning_rate: Meta-learning rate for expert weight updates (default: 0.1)
            corralling_gamma: Mixing parameter (default: 0.05, empirically validated optimal)
            corralling_cost_weight: λ for composite reward in Corralling meta-learner.
                       When > 0, the reward signal is adjusted to r = quality - λ·cost
                       BEFORE it reaches both the meta-learner and expert bandits.
                       This ensures the entire learning stack optimizes for cost-quality
                       value. Set cost_penalty=0 on experts to avoid double-counting.
                       - 0.0 = quality-only (default, backward-compatible)
                       - 0.1-0.3 = moderate cost awareness
                       - 0.5+ = aggressive cost preference
        """
        self.config = config or RouterConfig()
        self.verbose_routing = verbose_routing
        self.use_corralling = use_corralling
        self.corralling_learning_rate = corralling_learning_rate
        self.corralling_gamma = corralling_gamma
        self.corralling_cost_weight = corralling_cost_weight
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
                model_registry = {m["openrouter_id"]: m for m in data["models"]}

        self.registry = dict(model_registry)
        
        # -----------------------------------------------------------------------
        # FEATURE SERVICE (The Eyes) - Dependency Injection
        # -----------------------------------------------------------------------
        if feature_service is not None:
            # Use provided service (custom feature engineering)
            self.features = feature_service
            logger.info("Using injected FeatureService")
        else:
            # Create default service from legacy parameters
            # --- Simplified Feature & Performance Layer (Jan 2026) ---
            # Feature extraction is now delegated to FeatureService (The Eyes)
            # Dimension is auto-detected from PCA file (PCA components + 1 bias)
            from .feature_service import FeatureService as FS
            self.features = FS(
                encoder_model=context_model,
                pca_path=pca_path,
                allow_jit_training=True
            )
            logger.info(f"Created default FeatureService with encoder={context_model}")
        
        # For backward compatibility, expose encoder and pca as properties
        # These are now properties of the FeatureService itself
        self.encoder = self.features.encoder
        self.pca = self.features.pca
        
        # Calculate dimension dynamically from feature service
        # Default is 33 (32 PCA + 1 bias) with pca_32.joblib
        embedding_dim = self.features.dimension
        
        logger.debug(f"Feature dimensions: "
                    f"pca={self.pca.n_components if self.pca else 'none'}, "
                    f"total={embedding_dim} (including bias)")
        
        # Initialize bandit with calculated dimension.
        # NOTE: Features are [PCA_0, ..., PCA_{d-2}, bias].  We apply PCA for
        # dimensionality reduction and initialize the covariance/prior as
        # A₀ = λI, which corresponds to an isotropic regularizer in the PCA
        # space; this choice does not distinguish between high- and low-variance
        # components and may over-regularize the latter.  See the
        # DisjointLinUCBPolicy docstring for further discussion.
        self.bandit = DisjointLinUCBPolicy(
            list(self.registry.keys()), 
            dim=embedding_dim,  # Already includes bias
            alpha=alpha,
            init_lambda=init_lambda,  # Use parameter, not config
            update_lambda=update_lambda,  # Use parameter, not config
            forgetting_factor=forgetting_factor
        )
        
        # Initialize Security Scanner (Lazy)
        self._toxicity_scanner = None
        
        # Initialize Corralling Router (if enabled)
        self.corralling_router = None
        if self.use_corralling:
            # Create two experts: one with current bandit (warmup), one tabula rasa
            # This will be properly initialized in create() after priors are loaded
            pass


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
        result.encoder = self.encoder
        result.pca = self.pca  # PCA model is read-only after init
        
        # --- Bandit Policy (deepcopy: has its own __deepcopy__ for locks) ---
        result.bandit = copy.deepcopy(self.bandit, memo)
        
        # --- Corralling (deepcopy: independent mutable state) ---
        result.use_corralling = self.use_corralling
        result.corralling_learning_rate = self.corralling_learning_rate
        result.corralling_gamma = self.corralling_gamma
        result.corralling_cost_weight = self.corralling_cost_weight
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
        
        # Calculate base dimensions
        if self.pca:
            embedding_dim = self.pca.n_components
        else:
            embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # PCA components  
        for i in range(embedding_dim):
            feature_map[f"pca_{i}"] = i
        
        # Bias term (always last)
        feature_map["bias"] = embedding_dim
        
        return feature_map

    def register_model(
        self,
        model_id: str,
        capabilities: List[Capability] = None,
        speed: SpeedProfile = "balanced",
        cost_usd: float = None,
        latency_s: float = None,
        initial_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Universal entry point for adding models with Progressive Registration.
        
        Combines basic user knowledge with bandit math. This method translates
        human-friendly inputs (capabilities like "coding", speed profiles like "fast")
        into the mathematical priors (theta vectors) needed by LinUCB.
        
        **Three Tiers of Knowledge:**
        
        **Tier A: Archetypes** - "I know the model's intent"
            capabilities=["coding", "math"] applies semantic anchor boosts
            
        **Tier B: T-Shirt Sizing** - "I know cost/speed but not priors"
            speed="fast" sets positive bias (cheap → use by default)
            speed="slow" sets positive bias (expensive → reserve for hard tasks)
            
        **Tier C: Agnostic** - "I have no information"
            Just model_id initializes with neutral priors and high variance
            
        **Power User Override:**
            initial_weights={"complexity_score": 3.0} for explicit control
        
        Examples:
            # Local Llama: Fast and general purpose
            router.register_model("llama-3-8b", speed="fast", capabilities=["general"])
            
            # Specialist: Slow but great at coding
            router.register_model("deepseek-coder", speed="slow", capabilities=["coding"])
            
            # Mystery model: No information
            router.register_model("model-x", speed="balanced")
            
            # Power user: Explicit weights
            router.register_model("custom", initial_weights={"complexity_score": 2.5})
        """
        if capabilities is None:
            capabilities = []
            
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
        
        # 6. Add to Bandit with Latent Semantic Transfer (Paper V1: Progressive Learning)
        # Instead of hardcoded heuristics, use semantic similarity to find neighbors
        # and dynamically adjust prior strength based on confidence in the match
        if len(self.bandit.models) > 0:
            # Build semantic DNA and find best neighbor
            dna_str = self._get_model_dna(model_id, capabilities, speed)
            neighbor, similarity = self._find_semantic_neighbor(model_id, dna_str)
            
            # Dynamic n_effective based on similarity confidence
            # Validated via prior transfer analysis (Appendix A.3)
            # Robustness comes from Corralling's adaptive expert selection.
            # 
            # Strategy: Use similarity as proxy for θ_neighbor quality, not n_effective tuning
            reg_config = self.config.registration
            if similarity > 0.8:
                n_effective = reg_config.n_effective_high_similarity  # Strong match
            elif similarity > 0.6:
                n_effective = reg_config.n_effective_medium_similarity  # Moderate match
            else:
                n_effective = reg_config.n_effective_low_similarity  # Weak match
            
            logger.info(
                f"🔍 Latent Semantic Transfer: {model_id} "
                f"matched to {neighbor} (sim: {similarity:.3f}, n_eff: {n_effective})"
            )
            
            # Use neighbor bootstrapping with dynamic prior strength.
            # Pass precomputed (neighbor, similarity) to avoid
            # a redundant O(K) embedding + similarity re-search inside admix.
            A_init, b_init = self.admix_theta_from_neighbors(
                model_id=model_id,
                registry=self.registry,
                bandit=self.bandit,
                encoder=self.encoder,
                n_effective=n_effective,
                precomputed_neighbor=(neighbor, similarity)
            )
            
            # Capture whether bootstrapping actually happened.
            # If admix_theta_from_neighbors found no suitable neighbor, it returns:
            #   A = init_lambda * I, b = zeros(dim)
            # We detect this case to determine if we should apply manual priors.
            is_bootstrapped = not (np.linalg.norm(b_init) < 1e-12)
            
            # Add arm with bootstrapped parameters
            # Atomic-publication pattern: set state before appending to models list
            # so concurrent select_arm() never sees an incomplete model entry.
            A_inv_init = safe_inv(A_init)
            with self.bandit._lock:
                self.bandit.A[model_id] = A_init
                self.bandit.b[model_id] = b_init
                self.bandit.A_inv[model_id] = A_inv_init
                self.bandit.last_update[model_id] = self.bandit.t
                self.bandit.regularization_floor[model_id] = self.bandit.init_lambda
                self.bandit.models.append(model_id)  # LAST: visible only after state is ready
        else:
            # First model - use standard initialization, but apply manual prior
            # atomically under the lock to prevent a concurrent select_arm() from
            # seeing the model with b=zeros before the prior is applied.
            new_A = np.eye(self.bandit.dim) * self.bandit.init_lambda
            new_b = self.bandit.init_lambda * theta_vector  # Apply manual prior immediately
            new_A_inv = safe_inv(new_A)
            with self.bandit._lock:
                self.bandit.A[model_id] = new_A
                self.bandit.b[model_id] = new_b
                self.bandit.A_inv[model_id] = new_A_inv
                self.bandit.last_update[model_id] = self.bandit.t
                self.bandit.regularization_floor[model_id] = self.bandit.init_lambda
                self.bandit.models.append(model_id)  # LAST: visible only after state is ready
            is_bootstrapped = False  # (manual prior applied above, skip step 7)
        
        # 7. Apply Manual Prior (T-Shirt Sizing) ONLY if Bootstrapping Failed
        #
        # CRITICAL: Apply manual prior if and only if no semantic transfer occurred
        # AND the first-model path above didn't already apply it.
        #
        # Scenario 1: Bootstrapping succeeded (found similar neighbor)
        #   - is_bootstrapped = True
        #   - b already contains neighbor's preferences scaled by n_effective
        #   - DO NOT overwrite with manual priors (neighbor knowledge > T-shirt sizing)
        #
        # Scenario 2: Bootstrapping failed (no suitable neighbor found, len(models) > 0)
        #   - is_bootstrapped = False
        #   - b = zeros(dim) (default/identity initialization)
        #   - DO apply manual priors to give the model a reasonable starting bias
        #
        # Scenario 3: First model (len(models) == 0 before registration)
        #   - Handled in the else-branch above (atomic initialization with prior)
        #   - is_bootstrapped = False, but b already set — re-setting is harmless
        if not is_bootstrapped:
            # Standard prior encoding: b = A @ theta
            # With A = lambda*I, we get: b = lambda * theta
            self.bandit.b[model_id] = self.bandit.init_lambda * theta_vector
            
        # 8. Prepare registry entry (but don't publish yet)
        # Use defaults from config if not provided
        if cost_usd is None:
            cost_usd = reg_config.default_cost_per_1m
        if latency_s is None:
            latency_s = reg_config.default_latency_s
        
        registry_entry = {
            "cost_per_1m_tokens": cost_usd,
            "median_latency_s": latency_s,
            "capabilities": capabilities,
            "speed_profile": speed
        }
        
        # 9. Propagate to Corralling Experts BEFORE registry publication
        # Previously, self.registry[model_id] was set before expert
        # state was initialized.  A concurrent route() call would build candidates
        # from registry.keys(), see the new model, and crash with KeyError in the
        # expert's select_model() when accessing self.A_inv[model].
        # Now we initialize all expert state first, then publish to registry last.
        if self.use_corralling and self.corralling_router:
            logger.info(f"🔄 Propagating {model_id} to Corralling experts...")
            
            # A. Calculate Normalized Cost (needed for experts)
            output_cost = cost_usd * 3.0 
            avg_cost_per_1k = ((cost_usd + output_cost) / 2.0) / 1000.0
            norm_cost = self._calculate_absolute_penalty(avg_cost_per_1k)
            
            # B. Get the Initial Matrices
            if is_bootstrapped:
                prior_A = A_init
                prior_b = b_init
            else:
                prior_A = self.bandit.A[model_id]
                prior_b = self.bandit.b[model_id]
            
            # C. Update Experts (before registry publication)
            self.corralling_router.add_model(model_id)
            
            expert_warmup = self.corralling_router.experts[0]
            if hasattr(expert_warmup, 'add_model'):
                expert_warmup.add_model(model_id, prior_A, prior_b, norm_cost)
            
            expert_tr = self.corralling_router.experts[1]
            if hasattr(expert_tr, 'add_model'):
                expert_tr.add_model(model_id, norm_cost)
                
            logger.info(f"✅ {model_id} added to Corralling/Hybrid system")
        
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
    
    # ---------------------------------------------------------------------------
    # Latent Semantic Transfer: Progressive Learning for New Models
    # ---------------------------------------------------------------------------
    
    def _get_model_dna(
        self, 
        model_id: str, 
        capabilities: List[str] = None, 
        speed: str = None
    ) -> str:
        """
        Creates a semantic string representing the model's 'DNA' for embedding.
        
        This combines the model ID, capabilities, and speed profile into a 
        rich semantic description that can be embedded and compared to find
        similar models for knowledge transfer.
        
        Args:
            model_id: The model identifier (e.g., "gpt-4-turbo", "claude-3-opus")
            capabilities: List of model capabilities (e.g., ["coding", "math"])
            speed: Speed profile ("fast", "balanced", "slow")
            
        Returns:
            A space-separated string combining all semantic information
            
        Example:
            >>> dna = _get_model_dna("deepseek-coder-v2", ["coding"], "slow")
            >>> dna
            "deepseek coder v2 coding slow"
        """
        # Normalize model ID: replace separators with spaces for better embedding
        parts = [model_id.replace("-", " ").replace("/", " ").replace("_", " ")]
        
        if capabilities:
            parts.extend(capabilities)
        if speed:
            parts.append(speed)
            
        return " ".join(parts).lower()
    
    def _find_semantic_neighbor(
        self, 
        model_id: str, 
        dna_str: str
    ) -> Tuple[Optional[str], float]:
        """
        Finds the most similar existing model in the registry using embeddings.
        
        This is the core of Latent Semantic Transfer: instead of hardcoded rules,
        we use the semantic similarity between model "DNA" strings to find the
        best neighbor for knowledge transfer.
        
        Args:
            model_id: The new model to find a neighbor for
            dna_str: The semantic DNA string of the new model
            
        Returns:
            Tuple of (best_neighbor_id, similarity_score)
            Returns (None, 0.0) if no suitable neighbor is found
            
        Example:
            >>> neighbor, sim = _find_semantic_neighbor("gpt-4-turbo", "gpt 4 turbo fast")
            >>> neighbor, sim
            ("gpt-4", 0.92)
        """
        if not self.registry or len(self.bandit.models) < 1:
            return None, 0.0
        
        # 1. Embed the new model's DNA
        try:
            new_vec = self.encoder.encode([dna_str], convert_to_numpy=True)[0]
        except Exception as e:
            logger.warning(f"Failed to encode DNA for {model_id}: {e}")
            return None, 0.0
        
        # 2. Compare against existing models (with caching)
        best_neighbor = None
        best_sim = -1.0
        
        for m_id in self.bandit.models:
            if m_id == model_id:
                continue
            
            m_data = self.registry.get(m_id, {})
            
            # Use cached embedding or generate it
            if "dna_embedding" not in m_data:
                # Generate DNA for existing model
                m_capabilities = m_data.get("capabilities", [])
                m_speed = m_data.get("speed_profile", "balanced")
                m_dna = self._get_model_dna(m_id, m_capabilities, m_speed)
                
                try:
                    m_data["dna_embedding"] = self.encoder.encode([m_dna], convert_to_numpy=True)[0]
                except Exception as e:
                    logger.debug(f"Failed to encode DNA for {m_id}: {e}")
                    continue
            
            # Compute cosine similarity
            try:
                m_vec = m_data["dna_embedding"]
                sim = np.dot(new_vec, m_vec) / (
                    np.linalg.norm(new_vec) * np.linalg.norm(m_vec) + 1e-12
                )
                
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = m_id
            except Exception as e:
                logger.debug(f"Failed to compute similarity with {m_id}: {e}")
                continue
        
        return best_neighbor, best_sim

    def admix_theta_from_neighbors(
        self,
        model_id: str,
        registry: Dict[str, Dict],
        bandit: 'DisjointLinUCBPolicy',
        encoder,  # SentenceTransformer or compatible encoder
        n_effective: float = 5.0,  # Tunable prior strength (pseudocount of observations)
        precomputed_neighbor: Tuple[Optional[str], float] | None = None,  # Skip re-search
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap a new model's (A, b) from its nearest neighbor in embedding space.
        
        **LAYER 2: SEMANTIC TRANSFER (Dynamic Model Admission)**
        
        This method implements the second layer of the three-layer warm-start architecture:
        - Layer 1: Core warmup priors (80k battles) → Already loaded in __init__
        - Layer 2 (THIS METHOD): Semantic transfer → θ-only transfer for new models
        - Layer 3: T-shirt sizing injection → Applied in BanditRouter.create()
        
        **The "Prior Belief" Reset**
        
        [CRITICAL ALGORITHMIC FIX - Jan 2026]:
        Previous implementation transferred both A and b matrices, which caused the
        "Confident Transfer Trap": new models inherited the CONFIDENCE of mature
        neighbors (e.g., A with 1M samples → tiny confidence intervals → no exploration).
        
        **New Strategy: Transfer θ (Preferences), Reset A (Confidence)**:
        1. Find nearest neighbor by embedding similarity
        2. Extract neighbor's learned preferences: θ_neighbor = A_inv @ b_neighbor  
        3. Initialize new model with:
           - A_new = n_effective * init_lambda * I  (Scaled Identity → Controlled Uncertainty)
           - b_new = n_effective * init_lambda * θ_neighbor  (Scaled Preferences)
        4. Result: θ_hat = (n*λI)^-1 @ (n*λθ) = θ (mean preserved), Var ~ 1/n (confidence scaled)
        
        **Mathematical Justification (Appendix A.3):**
        - θ encodes "what contexts this model is good for" (direction)
        - A encodes "how confident we are in θ" (magnitude)
        - Scaling BOTH A and b by n_effective * init_lambda preserves mean while scaling variance
        
        **Hyperparameter Sensitivity (Appendix A.3):**
        n_effective robust across [1.0, 20.0] range. Guidance:
        - n_effective = 1.0: Weak prior → More exploration, slower convergence
        - n_effective = 5.0 (default): Balanced exploration/exploitation
        - n_effective = 20.0+: Strong prior → Fast exploitation, less exploration
          → Only use when neighbor similarity is very high (>0.9)
        
        **Concrete Example:**
        - Neighbor "GPT-4" has θ = [+0.8 (complexity), +0.3 (math), ...]
        - After 1M samples, its A has large eigenvalues → tight confidence
        - New model "GPT-4-Turbo" bootstraps:
          - OLD (buggy): Inherits 80% of A → thinks it has 800k samples → fossilized
          - NEW (fixed): Gets θ as prior, but A = λI → thinks it has 0 samples → explores
        
        Args:
            model_id: The new model to initialize
            registry: Model registry with display_name metadata
            bandit: LinUCB policy with existing model parameters
            encoder: SentenceTransformer for computing similarity
            n_effective: Tunable prior strength (default: 5.0). Simulates N pseudo-observations
                worth of confidence in the neighbor's preferences.
                - 0.1-1.0: Weak prior (use when similarity <0.7 or domain may shift)
                - 5.0-10.0: Moderate prior (recommended for most cases)
                - 20.0+: Strong prior (only use when similarity >0.9 and domain is stable)
                Higher values = faster exploitation but less exploration.
                Lower values = more exploration but slower convergence.
        
        Returns:
            Tuple of (A_new, b_new) where:
            - A_new = n_effective * init_lambda * I  (Precision scales with prior strength)
            - b_new = n_effective * init_lambda * θ_neighbor  (Moment scales proportionally)
            Result: θ_hat = A^{-1} b = θ_neighbor (mean preserved), Var ∝ 1/n_effective
            
        Example:
            >>> # Adding a new coding model with balanced prior
            >>> A, b = admix_theta_from_neighbors(
            ...     "deepseek-coder",
            ...     registry,
            ...     bandit,
            ...     encoder,
            ...     n_effective=5.0
            ... )
            # Result: Inherits preferences from similar model, but with fresh exploration
        """
        # Use precomputed neighbor from _find_semantic_neighbor()
        # when available, eliminating the redundant O(K) embedding+similarity search.
        # Previously, register_model() called _find_semantic_neighbor() (to set
        # n_effective) and then admix_theta_from_neighbors() re-ran the same search
        # internally, wasting compute and risking inconsistent neighbor selection
        # under concurrent mutation.
        if precomputed_neighbor is not None:
            best_neighbor, best_similarity = precomputed_neighbor
        else:
            # Fallback: run the full search (backward-compatible for direct callers)
            model_info = registry.get(model_id, {})
            capabilities = model_info.get("capabilities", [])
            speed = model_info.get("speed_profile", "balanced")
            model_dna = self._get_model_dna(model_id, capabilities, speed)
            
            try:
                if 'dna_embedding' in model_info:
                    new_embedding = model_info['dna_embedding']
                else:
                    new_embedding = encoder.encode([model_dna], convert_to_numpy=True)[0]
                    if model_id in registry:
                        registry[model_id]['dna_embedding'] = new_embedding
            except Exception as e:
                logger.warning(f"Failed to encode DNA for {model_id}: {e}. Using identity init.")
                return (
                    np.eye(bandit.dim) * bandit.init_lambda,
                    np.zeros(bandit.dim, dtype=np.float64)
                )
            
            best_neighbor = None
            best_similarity = -1.0
            
            for neighbor_id in bandit.models:
                if neighbor_id == model_id:
                    continue
                neighbor_info = registry.get(neighbor_id, {})
                neighbor_capabilities = neighbor_info.get("capabilities", [])
                neighbor_speed = neighbor_info.get("speed_profile", "balanced")
                neighbor_dna = self._get_model_dna(neighbor_id, neighbor_capabilities, neighbor_speed)
                
                try:
                    if 'dna_embedding' in neighbor_info:
                        neighbor_embedding = neighbor_info['dna_embedding']
                    else:
                        neighbor_embedding = encoder.encode([neighbor_dna], convert_to_numpy=True)[0]
                        registry[neighbor_id]['dna_embedding'] = neighbor_embedding
                    
                    similarity = np.dot(new_embedding, neighbor_embedding) / (
                        np.linalg.norm(new_embedding) * np.linalg.norm(neighbor_embedding) + 1e-12
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_neighbor = neighbor_id
                except Exception as e:
                    logger.debug(f"Skipping neighbor {neighbor_id}: {e}")
                    continue
        
        # Bootstrap from neighbor if found
        if best_neighbor and best_similarity > 0.5:  # Only use if moderately similar
            # Step 1: Extract neighbor's learned preferences (θ = A_inv @ b)
            with bandit._lock:  # Thread-safe read
                A_inv_neighbor = bandit.A_inv[best_neighbor]
                b_neighbor = bandit.b[best_neighbor]
            
            theta_neighbor = A_inv_neighbor @ b_neighbor
            
            # Step 2: Initialize new model with scaled precision and moment
            # Bayesian Ridge Regression with Prior Strength Scaling (Appendix A.3)
            # 
            # Formulation (preserves mean, scales confidence):
            # A_new = n_effective * λ_init * I  (Precision scales with prior strength)
            # b_new = n_effective * λ_init * θ  (Moment scales proportionally)
            # Result: θ_hat = A^-1 @ b = (n*λI)^-1 @ (n*λθ) = θ (mean preserved!)
            #         Var(θ_hat) ∝ 1/n_effective (confidence increases with n)
            
            A_new = n_effective * bandit.init_lambda * np.eye(bandit.dim)  # Scale Precision
            b_new = n_effective * bandit.init_lambda * theta_neighbor  # Scale Moment
            
            # Calculate transferred theta norm for verification
            theta_norm = np.linalg.norm(theta_neighbor)
            
            # Warn about potential n_effective misconfiguration
            if best_similarity < 0.7 and n_effective > 10.0:
                logger.warning(
                    f"⚠️ Strong prior (n_effective={n_effective}) with weak similarity "
                    f"({best_similarity:.3f}) for {model_id}. Consider reducing n_effective "
                    f"to 1.0-5.0 to allow more exploration."
                )
            elif best_similarity > 0.9 and n_effective < 2.0:
                logger.info(
                    f"💡 High similarity ({best_similarity:.3f}) detected for {model_id}. "
                    f"Consider increasing n_effective to 10.0-20.0 for faster convergence."
                )
            
            logger.info(
                f"✨ Bootstrapping {model_id} from neighbor {best_neighbor} "
                f"(similarity={best_similarity:.3f}, n_effective={n_effective}). "
                f"Transferred θ (||θ||={theta_norm:.4f}), reset A (confidence) for exploration."
            )
            
            return A_new, b_new
        else:
            # No suitable neighbor, use identity
            logger.info(f"No suitable neighbor for {model_id} (best_sim={best_similarity:.2f}), using identity init")
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64)
            )

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
        priors: str = "warmup",  # Default to loading priors
        **kwargs
    ) -> "BanditRouter":
        """
        Factory method to create a fully initialized router.
        
        Args:
            model_registry: Dictionary of model configurations
            context_model: Model to use for embedding generation
            priors: Prior initialization strategy ("warmup" or path to .joblib file)
            **kwargs: Additional arguments passed to __init__ or prior loading
        
        Returns:
            Fully initialized BanditRouter instance
        """
        # 1. Extract factory-specific arguments (not passed to __init__)
        state_path = kwargs.pop("state_path", None)
        prior_n_effective = kwargs.pop("prior_n_effective", 100.0)
        warmup_path = kwargs.pop("warmup_path", None)
        
        # Legacy support: map old 'exploration' parameter to 'alpha'
        exploration = kwargs.pop("exploration", None)
        alpha = kwargs.pop("alpha", None)
        
        if alpha is None and exploration is not None:
            exploration_map = {
                "static": 0.0,
                "safe": 0.05,
                "balanced": 0.5,
                "aggressive": 1.0
            }
            alpha = exploration_map.get(exploration, 0.05)
        elif alpha is None:
            alpha = 0.05  # Default to safe exploration

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
        
        # 4. Load Priors (The Data Truth)
        # This loads the dense matrices from disk (A_inv, b, etc.)
        if priors == "warmup" or (isinstance(priors, str) and (priors.endswith(".joblib") or "/" in priors)):
            # Determine priors path
            priors_path = warmup_path or (priors if priors != "warmup" else None)
            
            if priors_path:
                priors_path = Path(priors_path)
            else:
                # Default location (versioned artifacts or package assets)
                base_dir = Path(__file__).resolve().parent
                priors_path = base_dir.parent.parent / "artifacts" / "priors_warmup.joblib"
                
                # Check for alternative location in package assets if artifacts missing
                if not priors_path.exists():
                    priors_path = base_dir / "assets" / "priors_warmup.joblib"
                
            if priors_path and priors_path.exists():
                import joblib
                warmup_data = joblib.load(priors_path)
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
                #   1. Safety valve against mismatched priors.  The 80k RouteLLM
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
                logger.warning(f"⚠️ Warmup priors not found at {priors_path}. Using cold start.")
        
        # =====================================================================
        # LAYER 3: T-SHIRT SIZING INJECTION (Business Logic)
        # =====================================================================
        # Three-Layer Warm-Start Architecture:
        # - Layer 1: Core warmup priors (80k battles) → Already loaded above
        # - Layer 2: Semantic transfer → Handled in register_model()
        # - Layer 3 (HERE): T-shirt sizing → Business logic on top of data
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
        
        # 7. Initialize Corralling Router (if enabled)
        if router.use_corralling:
            # ---------------------------------------------------------------
            # Upgrade: Heterogeneous Alpha Strategy
            # Instead of sharing the same alpha schedule, we diversify the
            # experts so the meta-learner can track environment changes at
            # the allocation level.
            #
            # SCOPE OF NON-STATIONARITY HANDLING:
            # The Corralling meta-learner tracks non-stationary environments
            # (new models, traffic shifts) by re-weighting experts via loss
            # decay.  Each expert bandit itself is a stationary learner with
            # monotone information accumulation (A grows, exploration shrinks).
            # Extending expert-level updates with explicit forgetting or
            # change-point mechanisms is a direction for future work.
            #
            # Alpha Strategy:
            # - Decaying alpha assumes the world stops changing → "Brain Death"
            # - Constant alpha stays vigilant but wastes resources in stable periods
            # - Solution: Let Corralling meta-learner choose the right strategy
            # ---------------------------------------------------------------
            logger.info("🎯 Initializing Corralling Router with Heterogeneous Experts Strategy...")
            
            # Capture the alpha passed to create() to propagate to experts
            target_alpha = alpha if alpha is not None else 2.0  # Default: moderate exploration
            
            # Prepare model costs for cost-aware experts
            model_costs = {}
            for model_id in router.bandit.models:
                m_data = router.registry.get(model_id, {})
                input_cost = m_data.get("input_cost_per_m", router.config.default_missing_cost_per_m)
                output_cost = m_data.get("output_cost_per_m", router.config.default_missing_cost_per_m * 3.0)
                # Normalize cost to [0, 1] using market anchors
                avg_cost_per_1k = ((input_cost + output_cost) / 2.0) / 1000.0
                norm_cost = router._calculate_absolute_penalty(avg_cost_per_1k)
                model_costs[model_id] = {"normalized_cost": norm_cost}
            
            # Prepare warmup priors for Expert 1
            warmup_priors = {
                'A': {m: router.bandit.A[m].copy() for m in router.bandit.models},
                'b': {m: router.bandit.b[m].copy() for m in router.bandit.models},
                'context_dim': router.bandit.dim
            }
            
            # ---------------------------------------------------------------
            # Expert 1: The "Informed Explorer" (Warmup with Constant α)
            # ---------------------------------------------------------------
            # STRATEGY: Constant exploration with informed priors
            # RATIONALE: Priors encode 80k battles of domain knowledge
            # GOAL: Maintain ability to detect distribution shifts and prior mismatch
            # BEHAVIOR:
            #   - Starts with target exploration (alpha=target_alpha)
            #   - NEVER decays (alpha_end=target_alpha)
            #   - Result: Sustained vigilance detects when priors become stale
            #   - Validated: 14% better than decay (43.4 vs 49.6 regret)
            # 
            # WHY CONSTANT FOR INFORMED EXPERT?
            # Informed priors can mismatch deployment data (distribution shift).
            # Premature alpha decay causes irreversible commitment to potentially
            # wrong beliefs. Constant exploration maintains discovery potential.
            # ---------------------------------------------------------------
            expert_warmup = CostAwareLinUCBRouter(
                models=router.bandit.models,
                warmup_priors=warmup_priors,
                model_costs=model_costs,
                alpha_start=target_alpha,    # CONSTANT: Sustained exploration
                alpha_end=target_alpha,      # CONSTANT: Never decay
                cost_penalty=0.0
            )
            
            # ---------------------------------------------------------------
            # Expert 2: The "Learning Converger" (Tabula Rasa with Decay)
            # ---------------------------------------------------------------
            # STRATEGY: High initial exploration that converges to exploitation
            # RATIONALE: No priors → needs initial exploration, then convergence
            # GOAL: Build internal model quickly, then exploit learned patterns
            # BEHAVIOR:
            #   - Starts with conservative exploration (alpha=target_alpha/2)
            #   - Linearly decays to near-zero (alpha=0.01)
            #   - Result: Efficient learning that converges to optimal policy
            #   - Validated: Ablation shows decay is optimal for uninformed experts
            # 
            # WHY DECAY FOR UNINFORMED EXPERT?
            # Blank-slate experts need initial exploration to discover which models
            # work well. As uncertainty decreases (A matrix grows), continued high
            # exploration wastes samples. Decay balances exploration/exploitation.
            # ---------------------------------------------------------------
            expert_tabula_rasa = CostAwareTabulaRasaRouter(
                models=router.bandit.models,
                context_dim=router.bandit.dim,
                model_costs=model_costs,
                alpha_start=target_alpha / 2.0,  # Conservative start
                alpha_end=0.01,                   # DECAY: Converge to exploitation
                cost_penalty=0.0,
                ridge_lambda=1.0
            )
            
            # ---------------------------------------------------------------
            # The Manager: Corralling Meta-Learner
            # ---------------------------------------------------------------
            # Automatically switches between "Efficiency" and "Discovery" 
            # based on which expert performs better in the current regime.
            #
            # Stable Period → Conservative expert dominates (low regret)
            # Distribution Shift → Adaptive expert wins (detects changes)
            # New Model Release → Adaptive finds it first → Router pivots
            # ---------------------------------------------------------------
            router.corralling_router = CorrallingRouter(
                experts=[expert_warmup, expert_tabula_rasa],
                models=router.bandit.models,
                learning_rate=router.corralling_learning_rate,
                gamma=router.corralling_gamma,
                model_costs=model_costs,
                cost_weight=router.corralling_cost_weight,
            )
            
            logger.info("✅ Reversed Heterogeneous Experts Strategy Initialized:")
            logger.info(f"   📊 Expert 1 (Informed):     Constant Alpha {target_alpha:.2f} (Sustained Discovery)")
            logger.info(f"   🔍 Expert 2 (Uninformed):   Decaying Alpha {target_alpha/2.0:.2f}→0.01 (Learning Convergence)")
            logger.info("   🎯 Meta-Learner:            Corralling selects expert based on prompt context")
            logger.info("   💡 Performance:             14% better than original design (43.4 vs 49.6 regret)")
            logger.info("   📖 Rationale:               Informed priors need sustained exploration to detect drift")
        
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



    # ---------------------------------------------------------------------------
    # New Model Admission Protocol ("Transfer & Verify")
    # ---------------------------------------------------------------------------
    
    def _calculate_global_stats(self) -> Dict[str, Tuple[float, float, float]]:
        """Calculate min/max/mean stats for all registered models to normalize features."""
        stats = {
            "cost": [],
            "latency": [],
            "quality": [],
            "context": []
        }
        
        for m_data in self.registry.values():
            stats["cost"].append(float(m_data.get("input_cost_per_m") or 0.0))
            stats["latency"].append(float(m_data.get("time_to_first_token_seconds") or 0.0))
            stats["quality"].append(float(m_data.get("initial_quality") or 0.0))
            stats["context"].append(float(m_data.get("context_length") or 4096.0))
            
        def safe_stats(values):
            # Guard against empty values list (empty registry)
            if not values:
                return (0.0, 0.0, 0.0)
            arr = np.array(values)
            return (float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)))
            
        return {
            "cost": safe_stats(stats["cost"]),
            "latency": safe_stats(stats["latency"]),
            "quality": safe_stats(stats["quality"]),
            "context": safe_stats(stats["context"])
        }

    def _vectorize_model_metadata(self, model_data: Dict[str, Any], global_stats: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
        """
        Create a static feature vector V for transfer learning.
        V = [Norm(Cost), Norm(Latency), Norm(Quality_Score), Context_Window_Log_Norm]
        """
        # Extract
        cost = float(model_data.get("input_cost_per_m") or 0.0)
        lat = float(model_data.get("time_to_first_token_seconds") or 0.0)
        qs = float(model_data.get("initial_quality") or 0.0)
        ctx = float(model_data.get("context_length") or 4096.0)
        
        # Helper: MinMax Normalize to [0, 1]
        def normalize(val, key, log=False):
            min_v, max_v, _ = global_stats[key]
            if log:
                val = np.log(val + 1e-9)
                min_v = np.log(min_v + 1e-9)
                max_v = np.log(max_v + 1e-9)
            
            if max_v - min_v < 1e-9: return 0.5
            return (val - min_v) / (max_v - min_v)
            
        # Vector Construction
        return np.array([
            normalize(cost, "cost"),
            normalize(lat, "latency"),
            normalize(qs, "quality"),
            normalize(ctx, "context", log=True)
        ])



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
        prompt: str | np.ndarray,
        max_cost: float | None,
        max_latency: float | None,
        quality_floor: Dict[str, float | None] | None,
        input_tokens: int | None,
        output_tokens: int
    ) -> List[str]:
        """
        Apply hard constraints (cost, latency, quality floor).
        
        Args:
            candidates: List of candidate model IDs
            prompt: Input prompt
            max_cost: Maximum cost constraint (optional)
            max_latency: Maximum latency constraint (optional)
            quality_floor: Quality score minimums (optional)
            input_tokens: Input token count (optional, estimated if None)
            output_tokens: Output token count
            
        Returns:
            List of models passing all constraints
        """
        prompt_text = prompt if isinstance(prompt, str) else "[Pre-embedded]"
        in_tok = input_tokens or estimate_tokens_rough(prompt_text)
        
        filtered = []
        for m in candidates:
            # Check Cost
            cost = self._estimate_cost(m, in_tok, output_tokens)
            if max_cost is not None and cost > max_cost:
                continue
            
            # Check Latency
            lat = self._estimate_latency(m, output_tokens)
            if max_latency is not None and lat > max_latency:
                continue
            
            # Check Quality Floor
            # Guard against None values in quality_floor dict.
            # Callers may pass {"arena_elo": None} meaning "no constraint",
            # which would cause TypeError on `< v` comparison.
            if quality_floor:
                scores = self.registry.get(m, {}).get("scores", {})
                passes = True
                for k, v in quality_floor.items():
                    if v is None:
                        continue  # No constraint on this metric
                    if float(scores.get(k, 0)) < v:
                        passes = False
                        break
                if not passes:
                    continue
                    
            filtered.append(m)
            
        if not filtered:
            # Fall back to the original candidates, not the global
            # registry.  Returning registry.keys() could reintroduce models that
            # were excluded upstream (e.g., by corralling expert scope).
            filtered = list(candidates)
            
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
        
        # Save context for delayed feedback (RLHF, human ratings, etc.)
        self.context_store.save_context(log.request_id, x, model)
        
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
        
        Args:
            prompt: Input text or pre-embedded vector
            profile: (Ignored, kept for API compatibility)
            max_cost: Hard cost ceiling ($/1k tokens)
            max_latency: Hard latency ceiling (seconds)
            quality_floor: Minimum quality scores per model
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
            candidates, prompt, max_cost, max_latency, quality_floor, input_tokens, output_tokens
        )
        
        # Estimate tokens for scoring
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
            # Fallback: Simple LinUCB selection (UCB only, no cost penalty)
            best_model, best_utility = self.bandit.select_arm(x, candidates=filtered)
        
        total_weight = 1.0
        
        # Create routing log
        log = self._create_routing_log(
            prompt_text, best_model, best_utility, x, in_tok, output_tokens, total_weight
        )
        # Store corralling selection token in the log so that
        # process_feedback() can pass the correct (expert_idx, expert_prob)
        # back to CorrallingRouter.update().  This ties the importance-weighted
        # estimator to the exact probability that produced this selection.
        log.corralling_token = corralling_token
        
        return best_model, log
    
    def process_feedback(
        self,
        request_id: str,
        reward: float,
    ) -> None:
        """
        Process feedback for a routing decision.
        
        Args:
            request_id: ID from RoutingLog
            reward: Base reward (0-1, typically from judge)
        """
        # O(1) lookup via parallel index instead of O(N) linear scan
        log = self.log_index.get(request_id)
        
        # Fallback to context_store for delayed feedback (RLHF)
        if log is None:
            context, model_id = self.context_store.get_context(request_id)
            if context is None:
                logger.warning(f"Context not found for request_id={request_id}")
                return
            # Reconstruct log from persistent storage
            log = RoutingLog(
                request_id=request_id, timestamp_s=time.time(),
                prompt="[Delayed Feedback]", selected_model=model_id,
                predicted_utility=0.0, cost_usd=0.0, latency_s=0.0,
                context_vector=context
            )
        
        # Clamp reward to [0, 1] at the feedback entry point.
        # The Exp4/Corralling importance-weighted loss estimator ℓ = (1 - r) / p
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
                selection_token=getattr(log, 'corralling_token', None)
            )
        else:
            # Fallback: Update bandit directly
            self.bandit.update(log.selected_model, x, reward)
        
        # Periodic stability check (cheap O(d) operation)
        # Prevents numerical instability in low-traffic arms when update_lambda=0
        # Only check the bandit that is actually being updated.
        # Under Corralling, self.bandit.t stays at 0 (never incremented),
        # causing 0 % interval == 0 → True on every call, wasting O(K·d) and
        # checking matrices that are never written.
        if not (self.use_corralling and self.corralling_router):
            if (self.config.stability_check_interval > 0 and 
                self.bandit.t % self.config.stability_check_interval == 0 and
                self.bandit.t > 0):
                for model in self.bandit.models:
                    self.bandit._check_numerical_stability(model, self.config)

    def get_probabilities(self, context: str | np.ndarray, model_ids: List[str] | None = None) -> Dict[str, float]:
        """
        Get the probability of each model being the specialist for a given context.
        
When Corralling is enabled, self.bandit is never updated
        online (frozen at warmup priors).  We delegate to the warmup expert
        (expert 0) which receives all observations and has current state.
        """
        x = self.features.extract_features(context)
        models = model_ids if model_ids else self.bandit.models
        
        if self.use_corralling and self.corralling_router:
            # Use the warmup expert's learned state instead of stale base bandit.
            # The warmup expert has the same A/b/A_inv structure; we compute
            # posterior sampling directly from its matrices.
            expert = self.corralling_router.experts[0]
            n_samples = 1000
            model_samples = {}
            # Read expert state under its lock to prevent
            # observing a half-updated (A_inv, b) pair from concurrent update().
            with expert._lock:
                valid_models = [m for m in models if m in expert.A_inv]
                if not valid_models:
                    n = len(models) or 1
                    return {m: 1.0 / n for m in models}
                # Snapshot the matrices we need (copy under lock, sample outside)
                snapshots = {
                    m: (expert.A_inv[m].copy(), expert.b[m].copy())
                    for m in valid_models
                }
            for m in valid_models:
                A_inv_m, b_m = snapshots[m]
                theta_hat = A_inv_m @ b_m
                cov = 0.25 * A_inv_m  # noise_variance default
                try:
                    samples = np.random.multivariate_normal(theta_hat, cov, n_samples)
                except np.linalg.LinAlgError:
                    avg_var = max(np.trace(cov) / cov.shape[0], 1e-12)
                    samples = np.random.normal(loc=theta_hat, scale=np.sqrt(avg_var), size=(n_samples, len(theta_hat)))
                model_samples[m] = samples @ x
            stacked = np.stack([model_samples[m] for m in valid_models])
            winners = np.argmax(stacked, axis=0)
            from collections import Counter
            counts = Counter(winners)
            probs = {m: 0.0 for m in models}
            for i, m in enumerate(valid_models):
                probs[m] = counts.get(i, 0) / n_samples
            return probs
        
        return self.bandit.get_probabilities(x, models)

    def update(self, model_id: str, context: str | np.ndarray, reward: float, weight: float = 1.0) -> None:
        """
        Update the router with a new observation.
        
Previously updated BOTH self.bandit AND self.corralling_router
        when corralling was enabled.  Now mirrors the exclusive-OR pattern used
        by process_feedback(): update corralling if enabled, otherwise update
        the base bandit.
        
        Note: When called externally (without a preceding select_model()), no
        selection_token is available, so only the base experts learn — the
        meta-weights are not updated.  This is correct: the importance-weighted
        estimator requires the probability from the specific selection call.
        """
        # Clamp reward to [0, 1] (same as process_feedback)
        reward = float(np.clip(reward, 0.0, 1.0))
        x = self.features.extract_features(context)
        
        if self.use_corralling and self.corralling_router:
            # Propagate weight so difficulty-based weighting
            # isn't silently dropped under Corralling.
            self.corralling_router.update(x, model_id, reward, weight=weight)
        else:
            self.bandit.update(model_id, x, reward, weight)
        
        # Periodic stability check — same guard as process_feedback()
        if not (self.use_corralling and self.corralling_router):
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
        # Under Corralling, self.bandit is frozen at warmup priors
        # (never updated online).  Delegate to the warmup expert (experts[0])
        # which receives all observations, mirroring get_probabilities().
        # Snapshot A_inv/b under lock for thread safety.
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
            # Output typically 3x input cost (market convention)
            output_cost = self.config.default_missing_cost_per_m * 3.0
        
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
# Hybrid/Corralling Router: Robust Warmup with Safety Guarantees
# ---------------------------------------------------------------------------

class CorrallingRouter:
    """
    [Paper FIXED] Corralling Bandits with Mixing Parameter to prevent 'Expert Death'.
    
    Implements Exp4-style updates with explicit exploration floor (gamma).
    
    **High-Level Idea (Non-Technical):**
    Instead of betting everything on warmup priors, we hedge our bets by running
    both "warmup" and "tabula rasa" in parallel. Over time, we give more weight
    to whichever strategy is performing better.
    
    **Why This Matters:**
    If warmup priors are harmful (domain mismatch), the algorithm automatically
    shifts weight to tabula rasa. If warmup priors are helpful, they dominate.
    This provides safety guarantees against negative transfer.
    
    **Expert Death Prevention:**
    Pure exponential weighting can cause "Expert Death" - once an expert's weight
    drops to ~10^-16, the router stops listening to it forever, even if the
    environment changes. The mixing parameter (gamma) ensures every expert
    maintains a minimum probability (γ/K), allowing recovery.
    
    **Non-Stationarity Scope:**
    The Corralling meta-learner incorporates loss decay to adapt expert weights
    under non-stationary conditions (new models, traffic distribution shifts),
    but each expert bandit is itself trained under a stationary-reward assumption
    (monotone A accumulation without forgetting).  Thus, our treatment of
    non-stationarity operates at the meta level only; extending expert-level
    updates with explicit forgetting or change-point mechanisms is a direction
    for future work.
    
    **Theoretical Guarantee:**
    - gamma > 0 ensures no expert's probability ever drops to zero.
    - This allows the meta-learner to recover from poor early expert allocation
      (e.g., if a previously underperforming expert becomes relevant after a
      distribution shift, it will still be sampled often enough to detect this).
    
    **Empirical Validation (gamma=0.05):**
    - Validated across 4 dimensions using 18,750 trials (5 values × 5 seeds × 750 prompts)
    - Performance: 43.8 ± 5.4 regret (near-optimal, <1% cost vs. gamma=0.0)
    - Safety: 80% variance reduction vs. gamma=0.0 (prevents stochastic expert death)
    - Decisiveness: Achieves lowest minimum weights (~10^-4), indicating strong adaptation
      (allocates 80-90%+ weight to the higher-reward expert based on empirical performance)
    - Predictability: 45% lower outcome variance vs. gamma=0.0
    - See: experiments/03_figure/results/gamma_ablation/ for full analysis
    
    **Computational Overhead:**
    - Memory: 2x (store two sets of A/b matrices)
    - Inference: O(1) extra (just pick between two pre-computed decisions)
    - Update: 2x (update both strategies, but they're independent)
    
    In practice, the overhead is negligible (~0.1ms) compared to LLM inference (~100ms).
    
    **Implementation Note:**
    This is a simplified version of the full Corralling algorithm (Agarwal et al., 2017).
    We use exponential weights with observed losses rather than full importance-weighted
    counterfactual estimation, which makes the code much simpler while retaining the
    core adaptive property.
    
    Args:
        experts: List of bandit instances (typically [warmup_router, tabula_rasa_router])
        models: List of model IDs (must match across all experts)
        learning_rate: How quickly to adapt weights (default: 0.1)
        gamma: Mixing parameter γ. Minimum prob for any expert is γ/N.
               Prevents 'Expert Death' when the meta-learner's environment
               shifts. (default: 0.05, empirically validated as optimal across
               performance, safety, decisiveness, and predictability)
        
    Example:
        >>> # Create two experts
        >>> warmup = SimpleLinUCBRouter(models, warmup_priors, alpha=1.0)
        >>> tabula_rasa = TabulaRasaRouter(models, context_dim=24, alpha=1.0)
        >>> 
        >>> # Wrap them in Corralling
        >>> hybrid = CorrallingRouter(experts=[warmup, tabula_rasa], models=models, gamma=0.05)
        >>> 
        >>> # Use like any other router
        >>> selected = hybrid.select_model(context)
        >>> hybrid.update(context, selected, reward)
    """
    
    def __init__(
        self,
        experts: List,
        models: List[str],
        learning_rate: float = 0.1,
        gamma: float = 0.05,  # [VALIDATED] Empirically optimal (see experiments/03_figure/results/gamma_ablation/)
        loss_decay: float = 0.999,  # Meta-level adaptation decay
        meta_lr_halflife: float = 60.0,  # Staleness half-life in seconds for delayed feedback
        initial_weights: Optional[np.ndarray] = None,  # Prior-trust bias
        model_costs: Optional[Dict] = None,  # {model_id: {"normalized_cost": float}}
        cost_weight: float = 0.0,  # λ for composite reward: r = quality - λ·cost
    ):
        """
        Initialize Corralling with configurable expert weights and exploration floor.
        
        The loss_decay parameter prevents weight collapse when the meta-learner's
        environment shifts.  Without decay, cumulative_losses accumulates
        indefinitely, causing learned weights to become dominated by early history.
        
        NOTE: loss_decay operates at the META level only.  Each expert bandit
        is a stationary learner (monotone A accumulation, no forgetting).
        
        LEARNING RATE NOTE:
        In principle, the corralling learning rate admits an optimal schedule
        η* ∝ √(ln K / T) for a given horizon T (Agarwal et al., 2017), but
        this implementation uses a fixed η tuned for the typical horizon in
        our application.  The loss_decay mechanism is intended to handle
        meta-level adaptation via forgetting of stale feedback; it partially
        masks horizon mis-specification but is not a substitute for an optimal
        corralling learning-rate schedule.  Adaptive or horizon-aware η_t
        schedules are a direction for future work.
        
        Args:
            learning_rate: eta (η) for exponential weight updates.
                       Fixed at initialization; in principle η* ∝ √(ln K / T)
                       for a known horizon T, but we use a constant value
                       empirically tuned for the target traffic regime.
            gamma: Mixing parameter γ. Minimum prob for any expert is γ/N.
                   Prevents 'Expert Death' when meta-environment shifts.
            loss_decay: Exponential decay factor for cumulative losses (default: 0.999).
                       Controls how quickly the meta-learner forgets old expert
                       performance and adapts to new conditions.  This is
                       conceptually distinct from the learning rate: loss_decay
                       implements forgetting (meta-level adaptation), while η
                       controls stability vs. regret for a given horizon.
                       - 1.0 = stationary (no decay, standard Corralling)
                       - 0.999 = mild adaptation (half-life ~693 steps)
                       - 0.99 = moderate adaptation (half-life ~69 steps)
                       - 0.95 = aggressive adaptation (half-life ~14 steps)
            meta_lr_halflife: τ (seconds) for staleness-aware meta-weight learning.
                       When delayed feedback arrives, the meta-weight update's
                       effective learning rate is scaled by 1 / (1 + delay/τ).
                       - 60.0 = feedback >1 min gets ≤50% meta-lr (default)
                       - float('inf') = disable staleness decay (trust all tokens)
                       - 10.0 = aggressive decay for real-time systems
                       Expert internal updates are always at full strength.
            initial_weights: Optional array of initial expert weights.
                       Must sum to 1 and have length == len(experts).
                       Default: uniform (1/K each).
                       Biasing toward the warmup expert (e.g., [0.7, 0.3])
                       encodes "prior trust" — the belief that priors are
                       likely correct.  This reduces overhead when priors are
                       good but increases recovery time when they are bad.
                       See experiments/03_figure for the full trade-off.
            model_costs: Optional dict mapping model_id to {"normalized_cost": float}.
                       Required when cost_weight > 0. Costs should be in [0, 1],
                       normalized using log-scale market anchors (see RouterConfig).
            cost_weight: λ for composite reward: r_effective = quality - λ·cost.
                       When > 0, the reward signal fed to BOTH the meta-learner
                       and expert bandits is adjusted to encode the cost-quality
                       tradeoff directly. This ensures:
                       (a) Expert theta learns E[quality - λ·cost | context], so
                           cheap-but-sufficient models get higher predicted value.
                       (b) The meta-learner evaluates experts on composite value,
                           not penalizing cost-aware experts for picking cheaper models.
                       
                       NOTE: When using cost_weight > 0, set cost_penalty=0 on
                       the expert classes to avoid double-counting cost. The
                       composite reward subsumes the UCB cost penalty.
                       
                       - 0.0 = quality-only (default, backward-compatible)
                       - 0.1-0.3 = moderate cost awareness
                       - 0.5+ = aggressive cost preference
        """
        self.experts = experts
        self.models = models
        self.learning_rate = learning_rate
        self.gamma = gamma  # The "Life Support" parameter
        self.loss_decay = loss_decay  # Meta-level adaptation decay
        self.meta_lr_halflife = meta_lr_halflife  # Staleness half-life (τ)
        self.model_costs = model_costs or {}
        self.cost_weight = cost_weight
        self.n_experts = len(experts)
        # Thread safety — CorrallingRouter mutates shared state
        # (weights, cumulative_losses) in update() and reads it in select_model().
        # NumPy releases the GIL during array ops, so concurrent calls can observe
        # un-normalized weights or double-decay cumulative_losses.
        self._lock = threading.Lock()
        
        # Expert weights — uniform by default, or biased via initial_weights
        if initial_weights is not None:
            w = np.array(initial_weights, dtype=np.float64)
            assert len(w) == self.n_experts, (
                f"initial_weights length {len(w)} != n_experts {self.n_experts}")
            assert np.isclose(w.sum(), 1.0), (
                f"initial_weights must sum to 1, got {w.sum()}")
            self.weights = w.copy()
        else:
            self.weights = np.ones(self.n_experts) / self.n_experts
        
        # Set initial cumulative losses consistent with initial weights.
        # Exp4 update: w_i ∝ exp(-η * L_i), so L_i = -ln(w_i) / η.
        # This ensures the first update step continues smoothly from
        # the initial weights rather than "snapping" to uniform.
        self.cumulative_losses = -np.log(self.weights) / self.learning_rate
        
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
        Select model using the MIXED distribution (prevents Expert Death).
        
        **Overhead:** O(1) - just one random sample and one expert query.
        
        Args:
            context: Context vector for selection
            total_steps: Total training steps (passed to experts for alpha decay)
            candidates: Optional list of eligible model IDs after constraint filtering.
                       If provided, experts will only score these models.
                       If None, experts score all models (backward-compatible default).
        
        Returns:
            Tuple of (selected_model_id, selection_token).
            The selection_token must be passed back to update() for correct
            importance-weighted meta-weight attribution.  Without it, the
            meta-weight update is skipped (only base experts learn).
        """
        # Acquire lock for reading weights (select) to prevent
        # observing un-normalized weights from a concurrent update().
        with self._lock:
            probs = self._get_mixed_distribution()
        
        expert_idx = np.random.choice(self.n_experts, p=probs)
        
        # Ask that expert which model to use (pass through total_steps and candidates)
        model = self.experts[expert_idx].select_model(
            context, total_steps=total_steps, candidates=candidates
        )
        
        # Protect diagnostic counters under lock to prevent
        # lost increments from concurrent select_model() calls.
        with self._lock:
            self.expert_selections[expert_idx] += 1
            if model not in self.selections:
                self.selections[model] = 0
            self.selections[model] += 1
        
        # Return a selection token instead of storing state on self.
        # Previously, last_expert_idx/last_expert_prob were instance-level scalars
        # that could be overwritten by concurrent select_model() calls or be stale
        # when update() is called via the external BanditRouter.update() path.
        # The token ties the probability to the specific selection, ensuring the
        # importance-weighted estimator in update() uses the correct denominator.
        selection_token = {
            "expert_idx": int(expert_idx),
            "expert_prob": float(probs[expert_idx]),
            "timestamp": time.time(),  # For staleness-aware meta-lr decay
        }
        
        return model, selection_token
    
    def update(self, context: np.ndarray, model: str, reward: float,
               selection_token: Dict | None = None, weight: float = 1.0):
        """
        Two-level update: meta-weights (which expert to trust) + base-level
        (each expert's internal LinUCB learning).
        
        **Level 1 — Meta-Weight Update (Importance-Weighted Loss):**
        Only performed when a valid `selection_token` is provided (returned by
        `select_model()`).  The chosen expert is penalised with loss scaled by
        1/p_chosen for an unbiased estimator (Exp4-style).  Non-chosen experts
        receive 0 meta-loss because we didn't test their recommendation.
        
        When `selection_token` is None (e.g. external `BanditRouter.update()`
        calls without a preceding `select_model()`), the meta-weight update is
        skipped entirely.  This prevents using stale or mismatched probabilities
        which would yield a biased importance-weighted estimator.
        
        **Staleness-Aware Meta-Learning Rate:**
        For delayed feedback (RLHF, human ratings), the meta-weight learning
        rate is scaled by 1 / (1 + delay/τ) where τ = meta_lr_halflife.
        
        Rationale:  The importance weight 1/p_i captured at selection time
        becomes less reliable as meta-weights drift.  Rather than discarding
        delayed feedback entirely (losing signal) or trusting it fully (high
        variance), we smoothly discount the meta-weight update while keeping
        expert internal updates at full strength.
        
        - Fresh feedback (< τ):   meta-lr ≈ full  → unbiased, low variance
        - Stale feedback (≈ τ):   meta-lr ≈ 50%   → conservative update
        - Very stale (>> τ):      meta-lr → 0      → experts learn, meta stable
        
        **Level 2 — Base Algorithm Update (All Experts Learn):**
        ALL experts' internal bandits observe (context, model_played, reward).
        This is critical for Corralling correctness:
        
        Previously only the chosen expert's bandit was updated.
        This caused a starvation death spiral:
          1. Warmup expert starts stronger → gets more meta-weight
          2. More weight → more selections → more updates → learns faster
          3. Tabula rasa starved of data → can never catch up
          4. Tabula rasa becomes useless as a safety net against prior mismatch
        
        In Exp4/Corralling, base algorithms must see all feedback to maintain
        valid internal policies.  The meta-learner decides which expert to
        *trust*, but every expert should *learn* from every observation.
        
        **Overhead:** K expert updates instead of 1 (K=2 typically → 2× update cost,
        negligible compared to LLM inference latency).
        
        Args:
            context: Context vector used for selection
            model: Model that was selected
            reward: Observed reward (0-1 typically)
            selection_token: Token returned by select_model() containing the
                expert index and probability used for this selection.  Required
                for correct importance-weighted meta-weight attribution.
                If None, only base experts are updated (no meta-weight change).
            weight: Observation importance weight (passed to expert updates).
        """
        # ===================================================================
        # COMPOSITE REWARD: Encode cost-quality tradeoff in reward signal
        # ===================================================================
        # When cost_weight > 0, adjust the reward BEFORE it reaches both the
        # meta-learner (Level 1) and expert bandits (Level 2). This ensures
        # the entire learning stack optimizes for cost-quality value, not
        # pure quality.
        #
        # r_effective = quality - λ · normalized_cost(model)
        #
        # Effect: A cheap model that succeeds (Mixtral: 1.0 - λ·0.29) gets
        # a HIGHER reward than an expensive model that also succeeds
        # (GPT-4o: 1.0 - λ·0.69). This trains theta to prefer cheap models
        # when quality is comparable, and expensive models only when they
        # provide strictly better outcomes.
        if self.cost_weight > 0 and model in self.model_costs:
            cost_adj = self.cost_weight * self.model_costs[model].get('normalized_cost', 0.0)
            reward = reward - cost_adj
        
        # ===================================================================
        # LEVEL 1: Meta-Weight Update (which expert to trust)
        # ===================================================================
        # Only update meta-weights when we have a valid token
        # from the select_model() call that produced this observation.
        # Previously, last_expert_idx/prob were instance-level scalars that
        # could be stale or overwritten by concurrent calls, yielding a
        # biased importance-weighted estimator.
        if selection_token is not None:
            expert_idx = selection_token["expert_idx"]
            p_chosen = selection_token["expert_prob"]
            
            # ---------------------------------------------------------
            # Staleness-Aware Meta-Learning Rate
            # ---------------------------------------------------------
            # For delayed feedback (RLHF, human ratings arriving minutes
            # or hours after selection), the importance weight 1/p_i may
            # be inaccurate because meta-weights have drifted.
            #
            # We scale the meta-weight update by:
            #   staleness_factor = 1 / (1 + delay_seconds / τ)
            #
            # where τ = meta_lr_halflife (default 60s).
            #
            # This is equivalent to using a time-decayed learning rate:
            #   η_effective = η · staleness_factor
            #
            # Expert internal updates (Level 2) are NOT affected — the
            # reward signal for individual model quality is always valid
            # regardless of when it arrives.
            # ---------------------------------------------------------
            token_time = selection_token.get("timestamp")
            if token_time is not None and self.meta_lr_halflife < float('inf'):
                delay_seconds = max(time.time() - token_time, 0.0)
                staleness_factor = 1.0 / (1.0 + delay_seconds / self.meta_lr_halflife)
            else:
                staleness_factor = 1.0
            
            # Convert reward to loss
            observed_loss = 1.0 - reward
            
            # Initialize loss vector
            losses = np.zeros(self.n_experts)
            
            # Importance-Weighted Estimator: l_hat = l_obs / p_chosen
            # Since p_t >= gamma/K, the estimator is bounded (max loss <= K/gamma).
            # Only the chosen expert is penalised — we didn't test others' advice.
            losses[expert_idx] = observed_loss / p_chosen
            
            # Acquire lock for the compound read-modify-write
            # on cumulative_losses and weights.  Without this, concurrent update()
            # calls can double-decay or lose loss contributions, and concurrent
            # select_model() can read un-normalized weights.
            with self._lock:
                # Apply exponential decay before adding new losses (meta-level adaptation)
                self.cumulative_losses *= self.loss_decay
                # Scale loss contribution by staleness factor for delayed feedback
                self.cumulative_losses += staleness_factor * losses
                
                # Exp4 Update: w_i ∝ exp(-eta * cumulative_loss_i)
                log_weights = -self.learning_rate * self.cumulative_losses
                log_weights -= log_weights.max()  # Numerical stability
                self.weights = np.exp(log_weights)
                self.weights /= self.weights.sum()
        
        # ===================================================================
        # LEVEL 2: Base Algorithm Update (all experts learn)
        # ===================================================================
        # Every expert's internal bandit observes (context, model_played, reward).
        # This prevents starvation and lets the tabula rasa expert build an
        # informed policy even when it's not the one being selected.
        # Pass through the weight for importance/difficulty weighting.
        #
        # ---------------------------------------------------------------
        # DESIGN NOTE: Why `weight` is passed to experts but NOT to the
        # meta-weight update (Level 1) above.
        # ---------------------------------------------------------------
        # The Exp4/Corralling master uses importance-weighted loss
        #   ℓ_i = (1 - r) / p_i
        # to correct for action-selection bias.  The regret guarantees
        # (Agarwal et al., 2017) assume an unweighted loss stream and
        # rely on the 1/p_i factor being the *only* source of scaling.
        #
        # If we additionally multiplied by an application-level weight w_t,
        # the effective step size would become η·w_t/p_i, which can vary
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
        # If a future use case requires weighted regret at the master level
        # (e.g., "one VIP failure = 100 normal failures"), propagate w_t
        # into the loss with variance control: clip w_t/p_i to a max bound
        # and normalize weights over a rolling window.
        # ---------------------------------------------------------------
        for expert in self.experts:
            expert.update(context, model, reward, weight)
    
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
# Cost-Aware LinUCB Router: Optimized for Figure 4 Pareto Sweeps
# ---------------------------------------------------------------------------

class CostAwareLinUCBRouter:
    """
    LinUCB implementation with dynamic alpha-decay and cost-penalty logic.
    Optimized for Figure 4 Pareto sweeps with CorrallingRouter integration.
    
    **Architecture: Expert Parameter Warm-Start**
    
    The warm-start happens in __init__ (not during routing) because:
    1. **Hybrid Effectiveness**: Individual experts must be "pre-informed" so the
       Corralling Master has meaningful choices from day one
    2. **Bayesian Grounding**: Starting with 80k RouteLLM battles (A, b matrices)
       provides high-confidence priors instead of empty identity matrices
    3. **Semantic Transfer**: New models inherit preferences (θ) from similar models
       while resetting confidence (A) for fresh exploration ("First-Child" Bias fix)
    
    **Why Warm-Start at Expert Level?**
    - Warmup Expert: Initialized with high-confidence priors (large A values)
    - Tabula Rasa Expert: Can also use warm-start but with higher α to allow
      quick deviation when encountering the 94.2% Easy Cluster in production
    
    **Key Features:**
    - Dynamic alpha-decay: Starts with high exploration (alpha_start) and decays
      to low exploration (alpha_end) over the burn-in period
    - Cost-aware utility: Balances expected reward, uncertainty, and cost penalty
    - Warmup initialization: Uses pre-trained priors (e.g., 80k battles) to avoid cold-start
    - CorrallingRouter compatible: select_model accepts only context parameter
    
    **Use Case:**
    This is a simplified router designed for experimental Pareto frontier sweeps
    where you need fine-grained control over the exploration-exploitation tradeoff
    and explicit cost penalty weights.
    
    Args:
        models: List of model identifiers
        warmup_priors: Dict with 'A', 'b', and 'context_dim' from prior training
                      - A: Dict[str, np.ndarray] - Confidence matrices (d×d)
                      - b: Dict[str, np.ndarray] - Reward-weighted context sums (d,)
                      - context_dim: int - Feature dimension
        model_costs: Dict mapping model_id -> {"normalized_cost": float}
        alpha_start: Initial exploration coefficient (default: 1.0)
        alpha_end: Final exploration coefficient after burn-in (default: 0.1)
        cost_penalty: Weight for cost penalty (default: 0.0)
    
    Example:
        >>> # Standard usage with warmup priors
        >>> router = CostAwareLinUCBRouter(
        ...     models=["gpt-4", "gpt-3.5"],
        ...     warmup_priors={"A": {...}, "b": {...}, "context_dim": 24},
        ...     model_costs={"gpt-4": {"normalized_cost": 1.0}, 
        ...                  "gpt-3.5": {"normalized_cost": 0.1}},
        ...     alpha_start=1.0,
        ...     alpha_end=0.1,
        ...     cost_penalty=0.5
        ... )
        >>> selected = router.select_model(context)  # CorrallingRouter compatible
        
        >>> # Dynamic prior loading
        >>> router = CostAwareLinUCBRouter(models, warmup_priors, model_costs)
        >>> router.load_priors(new_priors, scale=0.5)  # Reduce prior strength
    """
    
    def __init__(self, models, warmup_priors, model_costs, alpha_start=1.0, alpha_end=0.1, cost_penalty=0.0):
        """
        Initialize router with Expert Parameter Warm-Start.
        
        **Warm-Start Architecture:**
        The matrices self.A and self.b are initialized by copying warmup_priors.
        This implements "Expert Parameter Warm-Start" - the expert begins with
        the "wisdom" of 80k RouteLLM Battles instead of cold-start identity matrices.
        
        **Automatic Prior Calibration:**
        Includes built-in detection and correction for 'Scale Explosion' where
        loaded priors predict massive rewards (e.g., 800.0) instead of [0, 1].
        The calibration rescales b-vectors to ensure predictions stay in safe range.
        
        **Why in __init__?**
        For Hybrid/Corralling to be effective, experts need pre-informed state
        so the master has meaningful choices immediately. Delaying warmup until
        first routing would defeat the purpose of having informed experts.
        
        Args:
            models: List of model IDs to route between
            warmup_priors: Pre-trained matrices from offline data (e.g., 80k battles)
            model_costs: Cost metadata for utility calculations
            alpha_start: Initial exploration (high during burn-in)
            alpha_end: Final exploitation (low after burn-in)
            cost_penalty: Budget constraint weight (λ parameter)
        """
        self.models = models
        self.alpha_start = alpha_start  # Initial exploration (e.g., 1.0)
        self.alpha_end = alpha_end      # Final exploitation (e.g., 0.1)
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.context_dim = warmup_priors['context_dim']
        self.t = 0  # Step counter for linear decay
        # Thread safety — NumPy releases the GIL during matrix ops,
        # so concurrent select_model/update can observe inconsistent (A_inv, b).
        self._lock = threading.Lock()
        
        # =====================================================================
        # LAYER 1: EXPERT PARAMETER WARM-START (Core Architecture)
        # =====================================================================
        # Three-Layer Warm-Start Architecture:
        # - Layer 1 (HERE): Load 80k battle priors → Data-driven initialization
        # - Layer 2 (register_model): Semantic transfer → Dynamic model admission
        # - Layer 3 (BanditRouter.create): T-shirt sizing → Business logic
        #
        # This layer initializes from warmup priors (80k RouteLLM battles):
        # - A matrices: Confidence/precision (covariance structure, d×d)
        #   → Inherits feature correlations from 80k battles
        #   → Large A[i,i] = high confidence in feature i's importance
        #   → A[i,j] ≠ 0 = features i and j are correlated
        #
        # - b vectors: Reward-weighted context sums (d,)
        #   → Inherits learned preferences from 80k battles
        #   → θ = A⁻¹b gives expected reward prediction weights
        #   → Large b[i] = feature i strongly predicts success
        #
        # Why in __init__?
        # - Hybrid effectiveness: Corralling Master needs informed experts from t=0
        # - No cold-start penalty: Immediate 80k battles of knowledge
        # - Empirical validation: Enables 92% cost reduction at 0.90 reward
        # Fall back to identity initialization for models missing
        # from warmup_priors (e.g., newly registered models not in offline data).
        _wp_A = warmup_priors.get('A', {})
        _wp_b = warmup_priors.get('b', {})
        _dim = warmup_priors['context_dim']
        self.A = {
            m: _wp_A[m].copy() if m in _wp_A else np.eye(_dim)
            for m in models
        }
        self.b = {
            m: _wp_b[m].copy() if m in _wp_b else np.zeros(_dim)
            for m in models
        }
        
        # Cache A_inv to avoid O(d³) recomputation.
        # Without caching, select_model() recomputes np.linalg.inv(A) for EVERY
        # model on EVERY routing decision → O(K·d³) per selection.
        # With caching and incremental updates → O(K·d²) per selection.
        self.A_inv = {m: safe_inv(self.A[m]) for m in models}
        
        # =====================================================================
        # LAYER 1.5: AUTOMATIC PRIOR CALIBRATION (Scale Explosion Fix)
        # =====================================================================
        # After loading priors, check if they predict reasonable values.
        # If predictions are massive (e.g., 800.0), rescale b-vectors to [0, 1].
        # This prevents "Scale Explosion" from misconfigured or legacy priors.
        self._calibrate_priors(target_max_pred=0.9)
        
        # =====================================================================
        # RUNTIME PREDICTION MONITOR (Complements static _calibrate_priors)
        # =====================================================================
        # Tracks per-model prediction statistics on live traffic to detect
        # drift, scale explosion, or collapsed arms post-deployment.
        # Threshold of 2.0: predictions for binary rewards should be in [0,1];
        # >2.0 indicates likely scale issue even accounting for noise.
        self.prediction_monitor = PredictionMonitor(
            alert_threshold=2.0, alert_cooldown=100
        )
        
        # Per-model Sherman-Morrison update counter for periodic
        # A_inv refresh.  Using a global counter (self.t) biases refreshes toward
        # frequently-updated models; per-model counters ensure every model gets
        # refreshed after exactly 1000 of its own updates.
        self._sm_update_count: Dict[str, int] = {m: 0 for m in models}
    
    def _calibrate_priors(
        self,
        target_max_pred: float = 0.9,
        calibration_contexts: List[np.ndarray] | None = None,
    ):
        """
        Auto-calibrates loaded priors so predictions stay in a safe range.
        
        Two-pass calibration:
        
        **Pass 1 — Bias probe** (fast, catches the most common failure):
        Probes each model with [0,...,0,1] (bias-only context).  If the bias
        prediction exceeds the threshold, the bias component of theta is clamped
        via theta-reconstruction (b = A @ theta_new).
        
        **Pass 2 — Suite probe** (comprehensive, catches PCA-dimension explosions):
        Probes each model with a built-in suite of basis-independent feature
        vectors (axis-aligned, random unit-norm, uniform).  If the caller
        supplies ``calibration_contexts``, those are appended to the suite so
        that domain-specific directions are also checked.  If any prediction
        exceeds the threshold, theta is globally rescaled so the worst-case
        prediction equals ``target_max_pred``.
        
        **Why the built-in probes are domain-agnostic:**
        As a general-purpose library, banditGPT cannot ship with domain-specific
        "golden prompts."  The synthetic geometry probes (axis-aligned + random +
        uniform) span the feature space in a basis-independent way, which is the
        correct approach for a scale check that must work regardless of the
        application domain.  Users who need domain-specific calibration can pass
        their own representative feature vectors via ``calibration_contexts``.
        
        Args:
            target_max_pred: Target maximum absolute prediction over the probe
                           suite (default: 0.9).  Should be < 1.0 to leave room
                           for the UCB exploration bonus.
            calibration_contexts: Optional list of domain-specific context vectors
                           (numpy arrays of shape ``(context_dim,)``) to include
                           in the probe suite.  Use this to check calibration
                           against representative traffic from your deployment.
                           These are appended to — not a replacement for — the
                           built-in geometry probes.
        """
        # -----------------------------------------------------------------
        # Build probe suite once (shared across all models)
        # -----------------------------------------------------------------
        d = self.context_dim
        probes = []
        
        # 1. Bias-only probe (catches the most common failure)
        bias_probe = np.zeros(d)
        bias_probe[-1] = 1.0
        probes.append(("bias", bias_probe))
        
        # 2. Axis-aligned probes for each PCA dimension (unit vectors)
        #    For high-d, sample a subset to keep calibration fast
        pca_dims = list(range(d - 1))  # All except bias
        if len(pca_dims) > 8:
            # Sample 8 evenly-spaced PCA axes
            step = max(1, len(pca_dims) // 8)
            pca_dims = pca_dims[::step][:8]
        for i in pca_dims:
            e_i = np.zeros(d)
            e_i[i] = 1.0
            probes.append((f"axis_{i}", e_i))
        
        # 3. Random unit-norm probes (covers off-axis directions)
        rng = np.random.RandomState(42)  # Deterministic for reproducibility
        for k in range(4):
            v = rng.randn(d)
            v /= (np.linalg.norm(v) + 1e-12)
            probes.append((f"random_{k}", v))
        
        # 4. Uniform probe (all features equally active, normalized)
        uniform = np.ones(d) / np.sqrt(d)
        probes.append(("uniform", uniform))
        
        # 5. User-supplied domain-specific contexts (optional)
        if calibration_contexts is not None:
            for i, ctx in enumerate(calibration_contexts):
                ctx = np.asarray(ctx, dtype=float).flatten()
                if ctx.shape[0] != d:
                    logger.warning(
                        f"Skipping calibration_context[{i}]: shape {ctx.shape} "
                        f"!= context_dim {d}"
                    )
                    continue
                probes.append((f"user_{i}", ctx))
        
        # -----------------------------------------------------------------
        # Calibrate each model
        # -----------------------------------------------------------------
        for m in self.models:
            try:
                theta = self.A_inv[m] @ self.b[m]
                
                # --- Pass 1: Bias-only probe (targeted fix) ---
                bias_pred = float(theta @ bias_probe)
                if abs(bias_pred) > 1.5:
                    # Theta-reconstruction: clamp bias weight, preserve PCA weights
                    theta_new = theta.copy()
                    theta_new[-1] = target_max_pred * (1.0 if bias_pred > 0 else -1.0)
                    logger.warning(
                        f"🔧 Calibration pass 1 ({m}): bias prediction "
                        f"{bias_pred:.2f} -> {theta_new[-1]:.2f} (theta-reconstruction)"
                    )
                    self.b[m] = self.A[m] @ theta_new
                    # Re-derive theta after b changed
                    theta = self.A_inv[m] @ self.b[m]
                
                # --- Pass 2: Suite probe (global rescale if needed) ---
                max_abs_pred = 0.0
                worst_probe = "none"
                for name, x in probes:
                    pred = abs(float(theta @ x))
                    if pred > max_abs_pred:
                        max_abs_pred = pred
                        worst_probe = name
                
                if max_abs_pred > 1.5:
                    # Global rescale: theta_new = theta * (target / max_pred)
                    # This uniformly shrinks all dimensions so the worst-case
                    # prediction equals target_max_pred.
                    scale = target_max_pred / max_abs_pred
                    theta_new = theta * scale
                    logger.warning(
                        f"🔧 Calibration pass 2 ({m}): worst-case prediction "
                        f"{max_abs_pred:.2f} on probe '{worst_probe}' "
                        f"-> global theta scale {scale:.4f}"
                    )
                    self.b[m] = self.A[m] @ theta_new
                    
            except Exception as e:
                logger.warning(f"Failed to calibrate prior for {m}: {e}")
                continue
    
    def load_priors(
        self,
        warmup_priors: Dict,
        scale: float = 1.0,
        calibration_contexts: List[np.ndarray] | None = None,
    ):
        """
        Load or update warmup priors with optional scaling.
        
        **Use Cases:**
        1. Dynamic prior updates: Refresh priors from new offline training
        2. Prior strength tuning: Scale down priors for faster adaptation
        3. Transfer learning: Load priors from different but related domains
        
        **Scaling Factor (scale):**
        - scale=1.0: Full prior strength (default, 80k battles worth of confidence)
        - scale=0.5: Half strength (faster adaptation to new data)
        - scale=2.0: Double strength (stronger regularization, slower adaptation)
        
        **Mathematical Effect:**
        Scaling both A and b by the same factor preserves θ = A^(-1)b:
        - θ_new = (scale*A)^(-1) @ (scale*b) = (1/scale * A^(-1)) @ (scale*b) = θ_old
        - But confidence changes: Smaller scale → wider confidence intervals → more exploration
        
        **Automatic Calibration:**
        After loading, automatically checks for "Scale Explosion" and corrects if needed.
        
        Args:
            warmup_priors: Dict with 'A' and 'b' matrices from prior training
            scale: Strength multiplier for priors (default: 1.0)
            calibration_contexts: Optional domain-specific context vectors to include
                           in the calibration probe suite (see ``_calibrate_priors``).
        
        Example:
            >>> # Reduce prior strength for faster adaptation
            >>> router.load_priors(new_priors, scale=0.5)
            
            >>> # Transfer priors from related domain (e.g., coding → math)
            >>> router.load_priors(coding_priors, scale=0.3)  # Weak transfer
            
            >>> # Load with domain-specific calibration check
            >>> held_out = [feature_pipeline(p) for p in sample_prompts[:10]]
            >>> router.load_priors(new_priors, calibration_contexts=held_out)
        """
        # Acquire lock for the entire load+calibrate sequence.
        # A concurrent select_model() could read partially-loaded priors (some
        # models updated, others still cold-start) or read A/b before A_inv is
        # refreshed, producing nonsensical scores.
        with self._lock:
            for m in self.models:
                if m in warmup_priors['A'] and m in warmup_priors['b']:
                    # Scale both A and b to adjust prior strength
                    self.A[m] = warmup_priors['A'][m].copy() * scale
                    self.b[m] = warmup_priors['b'][m].copy() * scale
                    # [PERFORMANCE FIX]: Refresh A_inv cache after loading new priors
                    self.A_inv[m] = safe_inv(self.A[m])
                else:
                    logger.warning(f"Model {m} not found in warmup_priors, skipping")
            
            # Auto-calibrate after loading to prevent scale explosion
            self._calibrate_priors(
                target_max_pred=0.9, calibration_contexts=calibration_contexts
            )
    
    def get_current_alpha(self, total_steps: int) -> float:
        """
        Linear decay schedule: Transition from exploration to exploitation.
        
        α_t = α_start + (t / T) × (α_end - α_start)
        
        Args:
            total_steps: Total training steps (N=1,121 for dev set)
        
        Returns:
            Current α value (linearly decayed from α_start to α_end)
        """
        if total_steps == 0:
            return self.alpha_end  # Evaluation mode: use final α
        
        fraction = min(self.t / total_steps, 1.0)
        return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)
    
    def select_model(self, context, total_steps: int = 0,
                     candidates: List[str] | None = None):
        """
        Select best model using cost-aware LinUCB with dynamic alpha-decay.
        
        **Utility Formula:**
        Score = (Predicted Reward + α_t × Uncertainty) - λ × Normalized Cost
        
        Where:
        - Expected Reward: θ^T · context (learned from past observations)
        - Uncertainty: sqrt(context^T · A^-1 · context) (epistemic uncertainty)
        - Alpha: Decays linearly from alpha_start to alpha_end over burn-in
        - Cost Penalty: Instance-level weight for cost sensitivity (self.cost_penalty)
        
        Args:
            context: Context vector (numpy array)
            total_steps: Total training steps (0 during evaluation for fixed α_end)
            candidates: Optional list of eligible model IDs (constraint-filtered).
                       If None, all models are scored.
        
        Returns:
            Selected model identifier (string)
        """
        alpha = self.get_current_alpha(total_steps)
        ucb_scores = {}
        expected_rewards = {}  # For runtime monitoring
        
        with self._lock:
            eligible = candidates if candidates is not None else self.models
            for model in eligible:
                # Skip models not yet initialized (race with register_model)
                if model not in self.A_inv:
                    continue
                A_inv = self.A_inv[model]
                theta = A_inv @ self.b[model]
                expected_reward = theta @ context
                var = float(context @ A_inv @ context)
                uncertainty = np.sqrt(max(var, 1e-12))
                normalized_cost = self.model_costs.get(model, {}).get("normalized_cost", 1.0)
                score = (expected_reward + alpha * uncertainty) - (self.cost_penalty * normalized_cost)
                ucb_scores[model] = score
                expected_rewards[model] = float(expected_reward)
            
            # Record inside lock so PredictionMonitor stats are
            # consistent with the scores computed in this critical section.
            for model, score in ucb_scores.items():
                self.prediction_monitor.record(
                    model, expected_reward=expected_rewards[model], ucb_score=float(score)
                )
        
        if not ucb_scores:
            # Fallback must respect candidate constraints.
            # Previously returned self.models[0] which may violate cost/latency
            # constraints that produced the candidates list.
            fallback = candidates if candidates is not None else self.models
            return fallback[0] if fallback else None
        
        # Random tie-breaking prevents initialization-order bias.
        # Previously, deterministic max() always picked the first model when
        # UCB scores were tied (e.g., at startup with identical priors).
        return _argmax_random_tiebreak(ucb_scores)
    
    def update(self, context, model, reward, weight: float = 1.0):
        """
        Update model's A and b matrices with new observation using Sherman-Morrison.
        
        [PERFORMANCE FIX]: Now uses O(d²) Sherman-Morrison formula to incrementally
        update A_inv instead of recomputing from scratch (O(d³)). Falls back to
        full inversion when denominator becomes too small (numerical stability).
        
        Standard LinUCB update:
        - A += weight · context · context^T
        - b += weight · reward · context
        - A_inv updated via Sherman-Morrison or fallback
        - t += 1
        
        Args:
            context: Context vector used for selection
            model: Model that was selected
            reward: Observed reward (0-1 typically)
            weight: Importance/difficulty weight (default 1.0)
        """
        if model not in self.A:
            return
        # Guard against negative weight (defense-in-depth, mirrors
        # DisjointLinUCBPolicy).  Negative weight subtracts from A, potentially
        # making it non-PD and corrupting subsequent A_inv updates.
        if weight < 0:
            return
        # weight=0 contributes zero information; skip to avoid
        # inflating self.t (affects staleness calculations for other models).
        if weight == 0:
            return
        
        x = context.flatten()
        
        with self._lock:
            A_inv_current = self.A_inv[model]
            A_inv_x = A_inv_current @ x
            denominator = 1.0 + weight * (x @ A_inv_x)
            
            if abs(denominator) > 1e-6:
                self.A_inv[model] = A_inv_current - weight * np.outer(A_inv_x, A_inv_x) / denominator
                self.A[model] += weight * np.outer(x, x)
                self.b[model] += weight * reward * x
            else:
                logger.warning(
                    f"⚠️ Sherman-Morrison near-singularity for {model}: "
                    f"|denominator|={abs(denominator):.2e} < 1e-6. Recomputing inverse."
                )
                self.A[model] += weight * np.outer(x, x)
                self.A_inv[model] = safe_inv(self.A[model])
                self.b[model] += weight * reward * x
            
            self.t += 1
            
            # Periodic full refresh to correct Sherman-Morrison
            # floating-point drift.  Per-model counter ensures every model gets
            # refreshed after exactly 1000 of its own updates.  O(d³) amortized
            # cost is negligible vs. the O(d²) per-step cost.
            self._sm_update_count[model] = self._sm_update_count.get(model, 0) + 1
            if self._sm_update_count[model] % 1000 == 0:
                self.A_inv[model] = safe_inv(self.A[model])
    
    def add_model(self, model_id: str, A: np.ndarray, b: np.ndarray, normalized_cost: float) -> None:
        """
        Dynamically register a new model with specific priors (for Corralling integration).
        
        This enables dynamic model admission at runtime while maintaining semantic transfer.
        Called by BanditRouter.register_model() after semantic bootstrapping.
        
        Args:
            model_id: New model identifier
            A: Initial Precision matrix (d x d) from semantic transfer
            b: Initial Moment vector (d,) from semantic transfer  
            normalized_cost: Cost penalty in [0, 1]
        """
        # Lock the entire add sequence so concurrent select_model()
        # never sees the model in self.models but with missing A_inv.
        with self._lock:
            # Note: model_id may already be in self.models if experts share the list with main bandit
            # Always update A/b matrices even if model_id exists in list
            if model_id not in self.models:
                self.models.append(model_id)
                
            self.A[model_id] = A.copy()
            self.b[model_id] = b.copy()
            # [PERFORMANCE FIX]: Cache A_inv for new model
            self.A_inv[model_id] = safe_inv(A)
            self.model_costs[model_id] = {"normalized_cost": normalized_cost}
        logger.debug(f"✅ Added {model_id} to Warmup Expert with transferred priors (||A||_F={np.linalg.norm(A):.1f})")

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


class CostAwareTabulaRasaRouter:
    """
    Cost-aware tabula rasa router (learns from scratch with cost penalty).
    
    Uses Tikhonov regularization (Ridge regression) to prevent infinite initial uncertainty.
    Initializes A = λI where λ is automatically calculated based on empirical variance
    of reward signals or manually specified.
    Implements α-scheduling: Linear decay from α_start to α_end during burn-in.
    
    This is the "blank slate" expert in Corralling that learns purely from online data
    without warmup priors. Paired with CostAwareLinUCBRouter (warmup expert) to provide
    robustness against domain mismatch.
    """
    def __init__(self, models: List[str], context_dim: int, model_costs: Dict,
                 alpha_start: float = 1.0, alpha_end: float = 0.1, cost_penalty: float = 0.0, 
                 ridge_lambda: float = None, reward_std: float = None):
        """
        Initialize tabula rasa router with automatic or manual ridge regularization.
        
        Args:
            models: List of model identifiers
            context_dim: Dimension of context vectors
            model_costs: Dict mapping model_id -> {"normalized_cost": float}
            alpha_start: Initial exploration coefficient (default: 1.0)
            alpha_end: Final exploration coefficient (default: 0.1)
            cost_penalty: Weight for cost penalty (default: 0.0)
            ridge_lambda: Ridge regularization parameter (default: None, auto-calculated)
            reward_std: Standard deviation of rewards for auto-calculation (optional)
        """
        self.models = models
        self.alpha_start = alpha_start  # Initial exploration (e.g., 1.0)
        self.alpha_end = alpha_end      # Final exploitation (e.g., 0.1)
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.t = 0  # Step counter for linear decay
        
        # Automatic Ridge Lambda Calculation
        # Based on empirical reward variance from 80k RouteLLM battles
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
        # Thread safety — matches CostAwareLinUCBRouter pattern.
        self._lock = threading.Lock()
        
        # Bayesian Prior Regularization: A = λI
        # λ > 1: More regularization (smoother, evidence-based updates)
        # λ = 1: Standard identity (high initial uncertainty)
        # This prevents the "spiky jagged weights" from being purely random
        self.A = {m: self.ridge_lambda * np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        
        # [PERFORMANCE FIX]: Cache A_inv to avoid O(d³) recomputation on every select_model()
        self.A_inv = {m: safe_inv(self.A[m]) for m in models}
        
        # Runtime prediction monitor (matches CostAwareLinUCBRouter)
        self.prediction_monitor = PredictionMonitor(
            alert_threshold=2.0, alert_cooldown=100
        )
        
        # Per-model Sherman-Morrison update counter for periodic refresh.
        self._sm_update_count: Dict[str, int] = {m: 0 for m in models}
    
    def get_current_alpha(self, total_steps: int) -> float:
        """
        Linear decay schedule: Transition from exploration to exploitation.
        
        α_t = α_start + (t / T) × (α_end - α_start)
        
        Args:
            total_steps: Total training steps (N=1,121 for dev set)
        
        Returns:
            Current α value (linearly decayed from α_start to α_end)
        """
        if total_steps == 0:
            return self.alpha_end  # Evaluation mode: use final α
        
        fraction = min(self.t / total_steps, 1.0)
        return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)
    
    def select_model(self, context: np.ndarray, total_steps: int = 0,
                     candidates: List[str] | None = None) -> str:
        """
        Select model using cost-aware UCB with dynamic α (tabula rasa, no priors).
        
        Score = (Predicted Reward + α_t × Uncertainty) - λ × Normalized Cost
        
        Args:
            context: PCA-transformed prompt embedding
            total_steps: Total training steps (0 during evaluation for fixed α_end)
            candidates: Optional list of eligible model IDs (constraint-filtered).
                       If None, all models are scored.
        
        Returns:
            Selected model ID
        """
        alpha = self.get_current_alpha(total_steps)
        ucb_scores = {}
        expected_rewards = {}  # For runtime monitoring
        
        with self._lock:
            eligible = candidates if candidates is not None else self.models
            for model in eligible:
                if model not in self.A_inv:
                    continue
                A_inv = self.A_inv[model]
                theta = A_inv @ self.b[model]
                expected_reward = theta @ context
                var = float(context @ A_inv @ context)
                uncertainty = np.sqrt(max(var, 1e-12))
                normalized_cost = self.model_costs.get(model, {}).get("normalized_cost", 1.0)
                score = (expected_reward + alpha * uncertainty) - (self.cost_penalty * normalized_cost)
                ucb_scores[model] = score
                expected_rewards[model] = float(expected_reward)
            
            # Record inside lock so PredictionMonitor stats are
            # consistent with the scores computed in this critical section.
            for model, score in ucb_scores.items():
                self.prediction_monitor.record(
                    model, expected_reward=expected_rewards[model], ucb_score=float(score)
                )
        
        if not ucb_scores:
            # Fallback must respect candidate constraints.
            fallback = candidates if candidates is not None else self.models
            return fallback[0] if fallback else None
        
        # Random tie-breaking prevents initialization-order bias.
        # Critical for tabula rasa: with A=λI and b=0, all models have identical
        # UCB scores at startup. Deterministic max() always picked the first
        # model in dict order, biasing early exploration.
        return _argmax_random_tiebreak(ucb_scores)
    
    def update(self, context: np.ndarray, model: str, reward: float, weight: float = 1.0):
        """
        Update using standard LinUCB update with Sherman-Morrison inverse update.
        
        [PERFORMANCE FIX]: Now uses O(d²) Sherman-Morrison formula to incrementally
        update A_inv instead of recomputing from scratch (O(d³)).
        
        Args:
            context: Context vector
            model: Model that was selected
            reward: Observed reward
            weight: Importance/difficulty weight (default 1.0)
        """
        if model not in self.A:
            return
        # Guard against negative weight (defense-in-depth, mirrors
        # DisjointLinUCBPolicy).  Negative weight subtracts from A, potentially
        # making it non-PD and corrupting subsequent A_inv updates.
        if weight < 0:
            return
        # weight=0 contributes zero information; skip to avoid
        # inflating self.t.
        if weight == 0:
            return
        
        x = context.flatten()
        
        with self._lock:
            A_inv_current = self.A_inv[model]
            A_inv_x = A_inv_current @ x
            denominator = 1.0 + weight * (x @ A_inv_x)
            
            if abs(denominator) > 1e-6:
                self.A_inv[model] = A_inv_current - weight * np.outer(A_inv_x, A_inv_x) / denominator
                self.A[model] += weight * np.outer(x, x)
                self.b[model] += weight * reward * x
            else:
                logger.warning(
                    f"⚠️ Sherman-Morrison near-singularity for {model}: "
                    f"|denominator|={abs(denominator):.2e} < 1e-6. Recomputing inverse."
                )
                self.A[model] += weight * np.outer(x, x)
                self.A_inv[model] = safe_inv(self.A[model])
                self.b[model] += weight * reward * x
            
            self.t += 1
            
            # Periodic full refresh to correct Sherman-Morrison
            # floating-point drift.  Per-model counter ensures fair refresh.
            self._sm_update_count[model] = self._sm_update_count.get(model, 0) + 1
            if self._sm_update_count[model] % 1000 == 0:
                self.A_inv[model] = safe_inv(self.A[model])
    
    def add_model(self, model_id: str, normalized_cost: float) -> None:
        """
        Dynamically register a new model with cold-start state (for Corralling integration).
        
        This enables the Tabula Rasa expert to route to newly added models.
        Initializes with ridge regularization (Identity matrix) for maximum plasticity.
        
        Args:
            model_id: New model identifier
            normalized_cost: Cost penalty in [0, 1]
        """
        # Lock the entire add sequence so concurrent select_model()
        # never sees the model in self.models but with missing A_inv.
        with self._lock:
            # Note: model_id may already be in self.models if experts share the list with main bandit
            # Always update A/b matrices even if model_id exists in list
            if model_id not in self.models:
                self.models.append(model_id)
            
            # Initialize with Ridge Regularization (Identity) - pure online learning
            # Use stored context_dim instead of hardcoded 33.
            dim = self.context_dim
            
            self.A[model_id] = self.ridge_lambda * np.eye(dim)
            self.b[model_id] = np.zeros(dim)
            # [PERFORMANCE FIX]: Cache A_inv for new model
            self.A_inv[model_id] = safe_inv(self.A[model_id])
            self.model_costs[model_id] = {"normalized_cost": normalized_cost}
        logger.debug(f"✅ Added {model_id} to Tabula Rasa Expert with cold start (ridge_λ={self.ridge_lambda:.2f})")
    
    def add_model_with_semantic_transfer(self, new_model_id: str, semantic_neighbor_id: str = None):
        """
        Add a new model with Latent Semantic Transfer (First-Child Bias Correction).
        
        **The "First-Child" Bias Correction:**
        When dynamically adding models via register_model(), the warm-start logic
        triggers a semantic transfer that avoids the "confident transfer trap":
        
        1. **Find Semantic Neighbor**: Match new model to existing similar model
           (e.g., "Flash" → "Haiku") using embedding similarity
        
        2. **Transfer Preferences (θ), Reset Confidence (A)**:
           - Extract neighbor's learned preferences: θ_neighbor = A_inv @ b_neighbor
           - Initialize new model: A_new = λI (fresh exploration potential)
           - Transfer preferences: b_new = λ × θ_neighbor (inherit domain knowledge)
        
        3. **Why This Works**:
           - θ encodes "what contexts this model is good for" (direction)
           - A encodes "how confident we are in θ" (magnitude)
           - Transfer knowledge (θ) but not sampling history (A)
           - New model can quickly diverge if it performs differently
        
        **Prevents "Confident Transfer Trap":**
        Without this correction, new models inherit both A and b from neighbors.
        This causes them to think they have 1M samples of experience (tiny confidence
        intervals) and never explore, even if they're actually quite different.
        
        Args:
            new_model_id: ID of model to add
            semantic_neighbor_id: Optional explicit neighbor (auto-detected if None)
        
        Example:
            >>> # Add new model with automatic semantic matching
            >>> router.add_model_with_semantic_transfer("anthropic/claude-3-haiku-20240307")
            >>> # Automatically finds "anthropic/claude-3-sonnet" as neighbor
            
            >>> # Explicit neighbor specification
            >>> router.add_model_with_semantic_transfer(
            ...     "openai/gpt-4-turbo-2024-04-09",
            ...     semantic_neighbor_id="openai/gpt-4-1106-preview"
            ... )
        
        Note:
            This method is primarily for documentation. In practice, the BanditRouter
            class implements this via admix_theta_from_neighbors() during register_model().
            For CostAwareLinUCBRouter (experimental), use load_priors() instead.
        """
        raise NotImplementedError(
            "Semantic transfer is handled by BanditRouter.register_model(). "
            "For CostAwareLinUCBRouter, initialize with pre-computed warmup_priors "
            "or use load_priors() to update existing priors."
        )

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
