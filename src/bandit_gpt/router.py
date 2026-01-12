"""
Production-grade contextual bandit router (Hot Path).

Core Features:
1. HLE Prior as Default: Initializes with "expert intuition" from 26k prompts.
2. Default Registry: Automatically loads 80+ models with cost/latency data.
3. Multi-Objective: Balances Quality, Cost, and Latency.
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
    from bandit_gpt.utils import sigmoid, calibrate_complexity, procedural_warmup, safe_inv, get_heuristic_prior
except ImportError:
    # Fallback for direct file import (not installed as package)
    from .storage import ContextStore, EphemeralContextStore, SqliteContextStore
    from .utils import sigmoid, calibrate_complexity, procedural_warmup, safe_inv, get_heuristic_prior

logger = logging.getLogger(__name__)

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
    - Bias: The intercept term. A positive bias (e.g., +0.5) gives a model 
      a ~62% starting probability of being picked, encouraging exploration.
      A negative bias (e.g., -0.5) makes it a "backup" (~38% prob).
    - Complexity Weight: How the model responds to hard prompts. 
      "Fast" models usually struggle (-0.5), "Slow" models usually excel (+0.5).
    """
    # Fast Profile (e.g., Haiku, Flash) -> No bias adjustment (neutral)
    fast_bias: float = 0.0
    fast_complexity_weight: float = -0.5
    
    # Slow Profile (e.g., Opus, GPT-4) -> Bias TOWARDS usage (believe expensive = high quality)
    # KDD FIX: Positive bias encodes belief that expensive models have latent quality
    slow_bias: float = 0.05
    slow_complexity_weight: float = 0.5
    
    # Balanced Profile (e.g., GPT-3.5, Sonnet) -> Neutral priors
    balanced_bias: float = 0.0
    balanced_complexity_weight: float = 0.0
    
    # Fallback Metadata (Pessimistic Defaults for Resilience)
    default_cost_per_1m: float = 10.00  # Assume expensive ($10/1M)
    default_latency_s: float = 2.0      # Assume slow (2s)

@dataclass
class RouterConfig:
    """
    Centralized configuration for BanditRouter magic numbers.
    
    ✅ **CANONICAL CONFIG**: This is the production-grade configuration for BanditRouter.
    
    All values are derived from empirical analysis or market data.
    
    **NOTE**: A legacy `LegacyRouterConfig` (Pydantic) exists in config.py for the deprecated
    virtual anchors architecture. That config is for `core.py` (BanditGPT), not this router.
    This dataclass is the single source of truth for the current production router.
    """
    
    # ---------------------------------------------------------------------------
    # HLE → Utility Transformation Parameters (Two-Tiered Calibration)
    # ---------------------------------------------------------------------------
    # Empirical Basis (validated on N=1000 LMSYS prompts):
    #   - HLE range across 35 models: [0.05, 0.35]
    #   - Complexity μ=-0.0037, σ=0.095 (see validate_complexity_bounds.py)
    #   - Ablation sensitivity: easy_floor ±2% regret, hard_exponent ±5% regret
    # 
    # Two-Tiered Approach:
    #   EASY PROMPTS: utility = easy_floor + easy_slope * hle_score
    #   HARD PROMPTS: utility = (hle_score / hard_max_benchmark) ^ hard_exponent
    # ---------------------------------------------------------------------------
    easy_floor: float = 0.95           # Base success rate for easy prompts
    easy_slope: float = 0.05           # HLE contribution slope for easy prompts  
    hard_max_benchmark: float = 0.35   # Best-in-class HLE score (GPT-4/Claude-3)
    hard_exponent: float = 2.0         # Power-law exponent (2.0 optimal from grid search)
    calibration_validated: bool = True # ✓ Validated on N=1000 LMSYS train prompts
    
    # ---------------------------------------------------------------------------
    # Production Stability: Memory Management
    # ---------------------------------------------------------------------------
    # KDD Reviewer Fix: Prevent OOM from unbounded log growth.
    # At 100 QPS with 54-dim context vectors (~500 bytes/log), 10k logs ≈ 5MB.
    # Adjust based on deployment memory constraints and feedback latency.
    max_log_size: int = 10_000         # Ring buffer size for RoutingLog entries
    
    # ---------------------------------------------------------------------------
    # New Model Admission: Probation Period
    # ---------------------------------------------------------------------------
    # Number of requests a new model must serve during probation before full admission.
    # Used to validate empirical performance vs. optimistic initialization.
    probation_requests: int = 500      # Probation period length (requests)
    pruning_min_samples: int = 30      # Min samples for probation subsidy decay
    probation_bonus: float = 0.10      # Quality boost for probationary models
    max_probation_models: int = 10     # [KDD FIX]: Max models allowed in probation simultaneously
    
    # Pruning constants removed - relying on UCB natural exploration/exploitation balance
    # No explicit model removal or probation periods required.
    
    # ---------------------------------------------------------------------------
    # Procedural Warmup: Covariance Shaping (KDD Reviewer Fix)
    # ---------------------------------------------------------------------------
    # Number of synthetic samples for procedural warmup to shape covariance matrix.
    # 
    # KDD Critique: Previously 15 samples for d≈54 dimensions was insufficient.
    # With only 15 rank-1 updates, cannot meaningfully override isotropic λI prior.
    # 
    # Mathematical requirement: Need at least d samples to span the space.
    # Recommendation: 2d for robust covariance estimation.
    # 
    # With 5 archetypes, samples_per_archetype = procedural_warmup_samples // 5
    # Default 100 → 20 samples per archetype → sufficient to shape 54D covariance
    procedural_warmup_samples: int = 100  # Warmup samples (2*d for d≈50)
    
    # ---------------------------------------------------------------------------
    # LinUCB Regularization: Initialization vs Runtime (KDD Performance Fix)
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
    # KDD FIX (Jan 2026): Adjusted to match ACTUAL portfolio range for consistency
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
    # RESILIENCE DEFAULTS: Pessimistic Fallbacks (KDD "Fail-Operational" Fix)
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


DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Optimization Profiles
# ---------------------------------------------------------------------------
class OptimizationProfile:
    """
    Named presets for utility function weights.
    
    Weights are Exchange Rates:
    - w_c = 1.0  (Base Currency: $1.00 USD)
    - w_q = X    (Value of 1% Quality Gain in USD)
    
    alpha_scale controls Risk Tolerance:
    - 0.01: Risk Averse (Strict Exploitation)
    - 1.00: Risk Neutral (Standard Exploration)
    - 2.00: Risk Seeking (High Exploration)
    """
    
    # 1. MAX QUALITY ("Rational Luxury")
    # Target: GPT-4.1 @ $5.00
    # BALANCE FIX: Set w_c=0.5 to balance cost sensitivity with quality differentiation
    # 
    # Analysis from warmup priors:
    # - Quality predictions cluster in [0.70, 0.85] range (IRT simulation)
    # - Models differ by only 0.5-1.5% in quality predictions
    # - Need to balance: avoid waste on noise, but allow quality to justify cost
    # 
    # Ratio 60:1 (w_q=30, w_c=0.5):
    # - Simple prompt (0.67% quality edge): 30×0.0067=0.20 vs 0.5×0.69=0.345 → Cheap wins ✓
    # - Hard prompt (1.5% quality edge): 30×0.015=0.45 vs 0.5×0.69=0.345 → Quality wins ✓
    # 
    # Interpretation: "Quality matters, but don't waste money on statistical noise"
    MAX_QUALITY = {"w_q": 100.0, "w_c": 0.5, "w_l": 0.0, "alpha_scale": 0.3}
    
    # 2. ARBITRAGE ("Smart Shopper")
    # Bias towards Cost (w_c=1.0) to break Prior Lock and find value (Grok/Flash).
    # High Exploration (alpha=1.0) ensures we sample cheap models to verify quality.
    ARBITRAGE = {"w_q": 0.8, "w_c": 1.0, "w_l": 0.0, "alpha_scale": 1.0}
    
    # 3. COST SAVER ("The Penny Pincher")
    # Target: Gemma-3-12b @ $0.24
    # "The Penny Pincher" - w_q=0.1 forces it to grab the cheapest model that 
    # isn't broken. Aggressively optimizes for the $0.16-$0.24 range.
    COST_SAVER = {"w_q": 0.1, "w_c": 1.0, "w_l": 0.0, "alpha_scale": 0.5}

    # 4. LOW LATENCY ("Real Time")
    # Goal: For chat/voice interfaces.
    LOW_LATENCY = {"w_q": 5.0, "w_c": 0.5, "w_l": 50.0, "alpha_scale": 0.5}

    _PROFILES = {
        "max_quality": MAX_QUALITY,
        "arbitrage": ARBITRAGE,
        "cost_saver": COST_SAVER,
        "low_latency": LOW_LATENCY,
    }

    @classmethod
    def get(cls, name: Union[str, Dict[str, float]]) -> Dict[str, float]:
        """Get profile weights by name or return dict if already a profile."""
        if isinstance(name, dict):
            # Pass-through for custom weight dicts with normalization
            weights = dict(name) # Shallow copy
            
            # Ensure at least one key exists
            if not any(k in weights for k in ["w_q", "w_c", "w_l"]):
                raise ValueError("Custom profile must contain at least one of ['w_q', 'w_c', 'w_l']")
            
            # Fill missing keys with 0.0
            for k in ["w_q", "w_c", "w_l"]:
                if k not in weights:
                    weights[k] = 0.0
            
            # NOTE: We NO LONGER normalize weights to sum to 1.0.
            # This allows "Unbounded Weights" where users can set high priorities
            # for multiple metrics simultaneously (e.g., w_q=1.0, w_c=1.0).
            # The bandit's exploration (alpha) scales naturally with w_q.
            
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("Weights must sum to a positive value")
            
            return weights
            
        if not isinstance(name, str):
            raise TypeError(f"Profile must be a string or dict, got {type(name)}")
            
        key = name.lower().replace("-", "_")
        if key not in cls._PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Valid: {list(cls._PROFILES.keys())}")
        return cls._PROFILES[key]
    
    @classmethod
    def from_reference(
        cls,
        quality_tolerance: float = 0.05,
        cost_savings: float = 0.50,
        latency_savings: float = 0.0
    ) -> Dict[str, float]:
        """
        Create optimization weights from intuitive trade-off percentages.
        
        This implements **Reference Point Normalization**, a concept from decision 
        theory that translates relative preferences into utility weights.
        
        **Mathematical Foundation:**
        
        If you're willing to trade a 5% quality drop (ΔQ=0.05) for a 50% cost 
        reduction (ΔC=0.50), you're defining the slope of your indifference curve:
        
            w_q × Loss_in_Quality = w_c × Gain_in_Cost
            w_q × 0.05 = w_c × 0.50
            
        This gives the exchange rate:
        
            w_q / w_c = 0.50 / 0.05 = 10
            
        Interpretation: Quality is 10x more valuable than Cost.
        
        **Connection to Pareto Frontier:**
        
        - **Steep Slope** (w_q ≫ w_c): Top-left of curve (Max Quality)
          → Small quality tolerance, large cost savings needed
          
        - **Shallow Slope** (w_q ≈ w_c): Moving down curve (Cost Saver)
          → Larger quality tolerance, moderate cost savings
          
        **Usage Examples:**
        
        1. **The Arbitrageur** - "I want GPT-4 quality (99%) at half price (50%)"
           ```python
           profile = OptimizationProfile.from_reference(
               quality_tolerance=0.01,  # 1% quality drop
               cost_savings=0.50        # 50% cost reduction
           )
           # Result: {"w_q": 50.0, "w_c": 1.0, "w_l": 0.0}
           # Interpretation: Extremely sensitive to quality drops
           ```
        
        2. **The Budget User** - "I can lose 20% quality for 90% cost savings"
           ```python
           profile = OptimizationProfile.from_reference(
               quality_tolerance=0.20,  # 20% quality drop
               cost_savings=0.90        # 90% cost reduction
           )
           # Result: {"w_q": 4.5, "w_c": 1.0, "w_l": 0.0}
           # Interpretation: Quality matters (4.5x), but router has room to optimize cost
           ```
        
        3. **Speed Matters** - "10% quality loss for 50% cost + 30% latency savings"
           ```python
           profile = OptimizationProfile.from_reference(
               quality_tolerance=0.10,
               cost_savings=0.50,
               latency_savings=0.30
           )
           # Result: {"w_q": 5.0, "w_c": 1.0, "w_l": 3.0}
           ```
        
        Args:
            quality_tolerance: Percentage quality drop you can accept (e.g., 0.05 = 5%)
            cost_savings: Percentage cost reduction desired (e.g., 0.50 = 50%)
            latency_savings: Percentage latency reduction desired (optional, default 0.0)
            
        Returns:
            Weight dictionary compatible with router's profile parameter
            
        Raises:
            ValueError: If parameters are not positive
        """
        # Validation
        if quality_tolerance <= 0:
            raise ValueError(f"quality_tolerance must be positive, got {quality_tolerance}")
        if cost_savings <= 0:
            raise ValueError(f"cost_savings must be positive, got {cost_savings}")
        if latency_savings < 0:
            raise ValueError(f"latency_savings must be non-negative, got {latency_savings}")
        
        # Avoid division by zero with minimum threshold
        q_drop = max(quality_tolerance, 0.001)
        c_save = max(cost_savings, 0.001)
        
        # Calculate the Exchange Rate
        # If you need a huge cost saving (50%) to justify a tiny quality drop (1%),
        # then Quality is VERY important.
        # Ratio = Cost_Savings / Quality_Drop
        exchange_rate = c_save / q_drop
        
        # Base weights
        # w_c is the "numeraire" (set to 1.0)
        # w_q is the exchange rate
        w_c = 1.0
        w_q = exchange_rate
        
        # Latency handling (optional)
        w_l = 0.0
        if latency_savings > 0:
            # How much latency reduction would you trade for the same quality drop?
            # Use the same exchange rate logic
            l_save = max(latency_savings, 0.001)
            w_l = l_save / q_drop
        
        return {"w_q": w_q, "w_c": w_c, "w_l": w_l}

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
# **COMPLEXITY ANALYSIS (KDD Reviewer Concern - RESOLVED)**
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
        
        Args:
            model_names: List of model identifiers (arms)
            dim: Context vector dimension
            alpha: Exploration coefficient (UCB bonus multiplier)
            init_lambda: Initialization regularization (A₀ = λI). Default 1.0 for cold-start stability.
            update_lambda: Runtime regularization for decay restoration. Default 0.0 for O(d²) speed.
            forgetting_factor: Exponential decay factor (1.0 = stationary, <1.0 = adaptive). Default 1.0.
        """
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(forgetting_factor)
        self.init_lambda = float(init_lambda)
        self.update_lambda = float(update_lambda)
        
        # Thread safety: Per-model locks (KDD Review Fix: eliminates lost update race condition)
        # Updates to Model A don't block updates to Model B
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

    def _check_numerical_stability(self, model_id: str, config: Any) -> None:
        """Periodic stability check and repair."""
        if not self.bandit_is_stable(model_id):
            logger.warning(f"⚠️ Stability alert for {model_id}! Trace={np.trace(self.A[model_id]):.4f}. Resetting.")
            self.A[model_id] = np.eye(self.dim) * self.init_lambda
            self.b[model_id] = np.zeros(self.dim)
            self.A_inv[model_id] = safe_inv(self.A[model_id])

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
        
        return result

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically."""
        if model_name in self.models: return
        
        self.models.append(model_name)
        self.A[model_name] = np.eye(self.dim) * self.init_lambda
        self.b[model_name] = np.zeros(self.dim, dtype=np.float64)
        self.A_inv[model_name] = safe_inv(self.A[model_name])
        self.last_update[model_name] = self.t

    def delete_arm(self, model_name: str) -> None:
        """Remove an arm from the bandit."""
        if model_name in self.models:
            self.models.remove(model_name)
        if model_name in self.A: del self.A[model_name]
        if model_name in self.b: del self.b[model_name]
        if model_name in self.A_inv: del self.A_inv[model_name]
        if model_name in self.last_update: del self.last_update[model_name]

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
        candidates = candidates or self.models
        candidates = [m for m in candidates if m in self.A]
        if not candidates: raise ValueError("No candidates available")

        best_model = candidates[0]
        best_ucb = -float("inf")

        # Thread safety: Acquire lock for reading shared state
        with self._lock:
            for m in candidates:
                # UCB = mean + alpha * std
                theta = self.A_inv[m] @ self.b[m]
                mean = float(theta.dot(x))
                
                # Global Forgetting: Inflate variance based on staleness
                # A_effective = A_stored * gamma^(dt)
                # Var_effective = x^T A_eff^-1 x = x^T (A^-1 * gamma^-dt) x = Var_stored * gamma^-dt
                #
                # [KDD REVIEW FIX C: Time-Delta Logic]
                # This inflation covers the "gap" between the model's last update and 
                # the current selection time. Since A is only decayed during update(),
                # we must explicitly inflate the variance here to reflect increased
                # uncertainty as time passes without new observations for this model.
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
    
    def get_probabilities(self, x: np.ndarray, models: List[str], n_samples: int = 1000) -> Dict[str, float]:
        """Calculate the probability of each model being the best via posterior sampling."""
        model_samples = {}
        valid_models = [m for m in models if m in self.A]
        
        snapshots = {}
        with self._lock:
            for m in valid_models:
                A_inv_m = self.A_inv[m]
                theta_hat = A_inv_m @ self.b[m]
                snapshots[m] = (A_inv_m, theta_hat)
        
        if not snapshots: return {m: 0.0 for m in models}
        
        for m, (A_inv_m, theta_hat) in snapshots.items():
            # Sample weights from the posterior N(theta_hat, A_inv)
            # Computation is outside global lock to maintain latency
            samples = np.random.multivariate_normal(theta_hat, A_inv_m, n_samples)
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
    # This eliminates the lost update race condition identified in KDD review
    
    def update(self, model: str, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """
        Update the model's A and b matrices with new observation.
        
        **KDD REVIEW FIX: Per-Model Locking**
        Replaced snapshot-swap pattern with fine-grained locking to eliminate
        lost update race condition. Each model has its own lock, so updates to
        Model A don't block updates to Model B.
        
        **Performance:**
        Sherman-Morrison update is O(d²) ≈ 0.5ms for d=24, negligible compared
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
        
        # Hold model-specific lock for entire update (eliminates race condition)
        with self.model_locks[model]:
            # Apply time-proportional decay based on elapsed steps
            # KDD Review Fix: Use time-based decay (gamma^dt) to match variance inflation in select_arm()
            #
            # [KDD REVIEW FIX C: Time-Delta Logic]
            # This decay catches A up from its last update to the current global time.
            # Because select_arm() inflates variance based on the same dt, 
            # this logic is theoretically sound and avoids "double-dipping" 
            # (at dt=0, inflation factor is 1.0).
            if self.gamma < 1.0:
                dt = self.t - self.last_update[model]
                # Clamp dt to prevent numerical underflow when gamma is small
                decay_factor = self.gamma ** min(dt, 1000)
                
                # [KDD REVIEW FIX]: Atomic Pointer Swap
                # Read old/calculate new locally, then swap under global lock
                # to ensure select_arm sees consistent A/b pair.
                new_A = self.A[model] * decay_factor
                new_b = self.b[model] * decay_factor
                
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.last_update[model] = self.t
            
            # KDD REVIEW FIX (Critique B): JIT Regularization Injection
            # Check for numerical instability BEFORE Sherman-Morrison update
            # In low-traffic regimes with gamma < 1.0, A can decay toward singularity
            # Use trace(A_inv) as O(d) proxy: if A → 0, then A_inv → ∞
            trace = np.trace(self.A_inv[model])
            threshold = 100 * self.dim  # Conservative: trigger at 100x expected trace
            
            if trace > threshold:
                logger.warning(
                    f"🛡️ JIT regularization for {model}: "
                    f"trace(A_inv)={trace:.2e} > {threshold:.2e}. "
                    f"Injecting λI to restore conditioning."
                )
                # KDD OPTIMIZATION: Preserve Theta During Stability Reset
                # Capture learned preferences before regularization
                old_theta = self.A_inv[model] @ self.b[model]
                
                # Inject identity regularization to restore numerical stability
                # [KDD REVIEW FIX A2]: COW re-assignment instead of +=
                self.A[model] = self.A[model] + (self.init_lambda * np.eye(self.dim))
                
                # Must recompute inverse after manual regularization injection
                # [KDD REVIEW FIX]: Atomic Pointer Swap
                new_A_inv = safe_inv(new_A)
                new_b = new_A @ old_theta
                
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
            
            # Add observation: A += weight * x x^T, b += weight * reward * x
            # [KDD REVIEW FIX A5]: COW re-assignment instead of +=
            self.A[model] = self.A[model] + (weight * np.outer(x, x))
            self.b[model] = self.b[model] + (weight * reward * x)
            
            # Sherman-Morrison inverse update (O(d²))
            # Formula: (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
            A_inv_current = self.A_inv[model] # Capture reference
            u = x * np.sqrt(weight)
            v = x * np.sqrt(weight)
            
            A_inv_u = A_inv_current @ u
            v_A_inv = v @ A_inv_current
            denominator = 1.0 + (v @ A_inv_u)
            
            # KDD REVIEW FIX: Stricter safety floor (1e-6 instead of 1e-10)
            if abs(denominator) > 1e-6:
                # Safe to use Sherman-Morrison formula
                new_A_inv = A_inv_current - np.outer(A_inv_u, v_A_inv) / denominator
                new_A = self.A[model] + (weight * np.outer(x, x))
                new_b = self.b[model] + (weight * reward * x)
                
                # [KDD REVIEW FIX]: Atomic Pointer Swap for Consistency
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
                # Capture learned preferences before regularization
                old_theta = self.A_inv[model] @ self.b[model]
                
                # Inject fresh regularization to restore conditioning
                new_A = self.A[model] + (weight * np.outer(x, x)) + (self.init_lambda * np.eye(self.dim))
                new_A_inv = safe_inv(new_A)
                new_b = new_A @ old_theta
                
                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
                    self.t += 1


    def _check_numerical_stability(self, model: str, config: 'RouterConfig' = None) -> None:
        """
        Safety check for numerical stability using trace of inverse.
        
        **KDD REVIEW FIX v2**: Eigenvalue decomposition is O(d³) ≈ 20ms, causing
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
            
            # Reset matrix with fresh regularization
            self.A[model] += config.init_lambda * np.eye(self.dim)
            self.A_inv[model] = safe_inv(self.A[model])
            
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
    cluster_id: int | None = None  # Detected semantic cluster
    cluster_similarity: float | None = None  # Similarity to cluster centroid
    context_vector: np.ndarray | None = None # Cached embedding for updates
    total_priority_weight: float = 1.0       # Sum of w_q, w_c, w_l for normalization

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
        cluster_boost_weight:float = 0.0,
        complexity_path: Path | str | None = None,
        anchors: Dict[str, str | None] = None,
        context_store: ContextStore | None = None,
        config: RouterConfig | None = None,
        verbose_routing: bool = False,
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
            cluster_boost_weight: Diversity boost weight
            complexity_path: (Deprecated) Path to complexity vectors
            anchors: (Deprecated) Custom virtual anchor definitions
            context_store: Persistent storage for delayed feedback
            config: Router configuration object
            verbose_routing: Enable detailed breakdown logs for each routing decision
        """
        self.config = config or RouterConfig()
        self.verbose_routing = verbose_routing
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
            # Standard Dimension: 23 PCA + 1 Bias = 24D
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
        # Default is 24 (23 PCA + 1 bias)
        embedding_dim = self.features.dimension
        
        logger.debug(f"Feature dimensions: "
                    f"pca={self.pca.n_components if self.pca else 'none'}, "
                    f"total={embedding_dim} (including bias)")
        
        # Initialize bandit with calculated dimension
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

        # [NEW] Pareto Configuration
        # Lambda values for Pareto utility: utility = quality - (λ * cost_penalty)
        # [KDD FIX]: Rescaled for normalized [0,1] quality and cost_penalty inputs
        # With both in [0,1] range, λ controls the quality-cost tradeoff:
        #   λ=1.0: Equal weight (50/50 quality-cost tradeoff)
        #   λ=0.5: Quality-biased (67% quality, 33% cost)
        #   λ=0.05: Quality-focused (95% quality, 5% cost)
        # Previous cost_saver λ=10.0 was calibrated for unnormalized cost,
        # causing utility range [-10, 1] where cost dominated 10x over quality.
        self.PARETO_PROFILES = {
            "cost_saver": 1.0,        # Balanced: Equal weight to quality and cost (50/50)
            "smart_shopper": 0.02,    # Push toward grok: stronger cost penalty
            "rational_luxury": 0.0     # Pure Quality: Cost-invariant (κ=0 is rational)
        }
        # Controls the "Optimism" of the Pareto Filter (UCB)
        # 1.0 = Standard UCB. Higher = Keep uncertain models alive longer.
        # [KDD REVIEW FIX]: Increased to 2.0 to ensure gatekeeper is more generous than judge.
        self.PARETO_EXPLORATION_CONSTANT = 2.0


        # ---------------------------------------------------------------------------
        # Tiered Context Storage (KDD Review Fix: "Feedback Horizon Fallacy")
        # ---------------------------------------------------------------------------
        # Default: SqliteContextStore (production, zero dependencies, 7-day TTL)
        # Alternative: EphemeralContextStore (testing, RAM-only, 100s horizon)
        self.context_store = context_store or SqliteContextStore()
        logger.info(f"Context store: {type(self.context_store).__name__}")

        # ---------------------------------------------------------------------------
        # Production Stability: Bounded Log Buffer (KDD Fix)
        # ---------------------------------------------------------------------------
        # Using deque with maxlen prevents unbounded memory growth.
        # At 100 QPS with ~500 bytes/log, 10k entries ≈ 5MB max footprint.
        # Oldest logs are automatically evicted when buffer is full.
        # IMPORTANT: process_feedback() must be called before log is evicted!
        self.logs: deque[RoutingLog] = deque(maxlen=RouterConfig.max_log_size)
        # [KDD REVIEW FIX]: Parallel index for O(1) feedback lookups
        self.log_index: Dict[str, RoutingLog] = {}
        self.model_priors: Dict[str, float] = {} 
        self.cluster_boost_weight = cluster_boost_weight
        
        # [KDD REVIEW FIX]: Persistent Tracking (Monotonic Probation)
        # Prevents "Rolling Window Fallacy" where models receive a probation bonus 
        # after their early logs are evicted from self.logs.
        self.model_counts: Dict[str, int] = defaultdict(int)
        
        # New Model Admission: Probation List
        self.probation_models: Dict[str, Dict[str, Any]] = {} 
        # Feature name to index mapping for Progressive Registration
        self._feature_map = self._build_feature_map()
        
        # [KDD REVIEW FIX]: Precompute Market Anchors for Performance
        # CPU profiling showed redundant log calls and Config creation in hot loop
        self._market_cost_floor = self.config.market_cost_floor
        self._market_cost_floor_log = np.log(self.config.market_cost_floor)
        self._market_cost_range = self.config.cost_range_log
        
        self._market_lat_floor = self.config.market_latency_floor
        self._market_lat_floor_log = np.log(self.config.market_latency_floor)
        self._market_lat_range = self.config.latency_range_log


    def __deepcopy__(self, memo):
        """
        Custom deepcopy for BanditRouter to handle unpicklable components.
        
        1. Shared Encoder: The SentenceTransformer is stateless and contains 
           locks. We share it across clones rather than copying.
        2. Bandit Policy: Uses its own custom __deepcopy__ for its internal lock.
        3. Context Store: Re-initialized or shared depending on type.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy configuration and registry
        result.config = copy.deepcopy(self.config, memo)
        result.registry = copy.deepcopy(self.registry, memo)
        
        # SHARE the encoder (stateless, contains locks)
        result.encoder = self.encoder
        
        # Deepcopy the bandit policy (calls its custom __deepcopy__)
        result.bandit = copy.deepcopy(self.bandit, memo)
        
        # Re-copy other stateful/cached components
        result.pca = copy.deepcopy(self.pca, memo)
        result.anchor_vectors = copy.deepcopy(self.anchor_vectors, memo)
        result.complexity_vector = copy.deepcopy(self.complexity_vector, memo)
        result.cluster_detector = copy.deepcopy(self.cluster_detector, memo)
        result.logs = copy.deepcopy(self.logs, memo)
        result.model_priors = copy.deepcopy(self.model_priors, memo)
        result.probation_models = copy.deepcopy(self.probation_models, memo)
        result._feature_map = copy.deepcopy(self._feature_map, memo)
        
        # Handle Context Store: Share the connection
        
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
            
        **Tier B: T-Shirt Sizing** - "I know cost/speed but not HLE"
            speed="fast" sets positive bias (cheap → use by default)
            speed="slow" sets negative bias (expensive → use selectively)
            
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
        # If HLE is unknown, use Speed/Cost as prior for "Default Mode"
        # Values from RouterConfig.registration (cientifically justified)
        reg_config = self.config.registration
        
        if speed == "fast":
            bias = reg_config.fast_bias
            # Fast models usually struggle with high complexity
            weights["complexity_score"] = reg_config.fast_complexity_weight
        elif speed == "slow":
            bias = reg_config.slow_bias
            # Slow models are usually meant for high complexity
            weights["complexity_score"] = reg_config.slow_complexity_weight
        else:  # balanced
            bias = reg_config.balanced_bias
            weights["complexity_score"] = reg_config.balanced_complexity_weight
        
        
        # OLD: Archetype mapping to virtual anchors - REMOVED
        # Anchors removed in KDD simplification
        
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
        
        # 6. Add to Bandit with Neighbor Bootstrapping (KDD Review Fix - Critique C)
        # Instead of cold-starting with A=I, b=0, bootstrap from similar models
        if len(self.bandit.models) > 0:
            # Use neighbor bootstrapping if there are existing models
            A_init, b_init = self.admix_theta_from_neighbors(
                model_id=model_id,
                registry=self.registry,
                bandit=self.bandit,
                encoder=self.encoder,
                alpha=0.8,  # DEPRECATED: kept for API compatibility
                n_effective=5.0  # Balanced prior strength for neighbor bootstrapping
            )
            
            # [KDD REVIEW FIX - Bug A: "First-Child" Bias Correction]
            # Capture whether bootstrapping actually happened.
            # If admix_theta_from_neighbors found no suitable neighbor, it returns:
            #   A = init_lambda * I, b = zeros(dim)
            # We detect this case to determine if we should apply manual priors.
            is_bootstrapped = not (np.linalg.norm(b_init) < 1e-12)
            
            # Add arm with bootstrapped parameters
            self.bandit.models.append(model_id)
            self.bandit.A[model_id] = A_init
            self.bandit.b[model_id] = b_init
            self.bandit.A_inv[model_id] = safe_inv(A_init)
            self.bandit.last_update[model_id] = self.bandit.t
        else:
            # First model - use standard initialization
            self.bandit.add_arm(model_id)
            is_bootstrapped = False
        
        # 7. Apply Manual Prior (T-Shirt Sizing) ONLY if Bootstrapping Failed
        # [KDD REVIEW FIX - Bug A]: The "First-Child" Bias Correction
        #
        # CRITICAL: Apply manual prior if and only if no semantic transfer occurred.
        #
        # Scenario 1: Bootstrapping succeeded (found similar neighbor)
        #   - is_bootstrapped = True
        #   - b already contains neighbor's preferences scaled by n_effective
        #   - DO NOT overwrite with manual priors (neighbor knowledge > T-shirt sizing)
        #
        # Scenario 2: Bootstrapping failed (no suitable neighbor found)
        #   - is_bootstrapped = False
        #   - b = zeros(dim) (default/identity initialization)
        #   - DO apply manual priors to give the model a reasonable starting bias
        #
        # This fixes the original bug where manual priors were only applied to the
        # very first model (len(models)==1), causing subsequent models without neighbors
        # to start with b=0 and lose the "fast"/"slow" signal from speed parameter.
        if not is_bootstrapped:
            # Standard prior encoding: b = A @ theta
            # With A = lambda*I, we get: b = lambda * theta
            self.bandit.b[model_id] = self.bandit.init_lambda * theta_vector
            
        # 8. Add to Model Registry (for cost/latency lookup during routing)
        # Use defaults from config if not provided
        if cost_usd is None:
            cost_usd = reg_config.default_cost_per_1m
        if latency_s is None:
            latency_s = reg_config.default_latency_s
            
        self.registry[model_id] = {
            "cost_per_1m_tokens": cost_usd,
            "median_latency_s": latency_s,
            "hle_score": None,  # Unknown HLE (will be learned)
            "capabilities": capabilities,
            "speed_profile": speed
        }
        
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
    





    @classmethod
    def admix_theta_from_neighbors(
        cls,
        model_id: str,
        registry: Dict[str, Dict],
        bandit: 'DisjointLinUCBPolicy',
        encoder,  # SentenceTransformer or compatible encoder
        alpha: float = 0.8,  # DEPRECATED: kept for API compatibility
        n_effective: float = 5.0,  # Tunable prior strength (pseudocount of observations)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap a new model's (A, b) from its nearest neighbor in embedding space.
        
        **KDD REVIEW FIX (Concern B)**: The "Prior Belief" Reset
        
        [CRITICAL ALGORITHMIC FIX - Jan 2026]:
        Previous implementation transferred both A and b matrices, which caused the
        "Confident Transfer Trap": new models inherited the CONFIDENCE of mature
        neighbors (e.g., A with 1M samples → tiny confidence intervals → no exploration).
        
        **New Strategy: Transfer θ (Preferences), Reset A (Confidence)**:
        1. Find nearest neighbor by embedding similarity
        2. Extract neighbor's learned preferences: θ_neighbor = A_inv @ b_neighbor  
        3. Initialize new model with:
           - A_new = init_lambda * I  (Identity → Maximum Uncertainty)
           - b_new = init_lambda * θ_neighbor * n_effective  (Tunable prior strength)
        4. Result: Same preferences, but high exploration potential
        
        **Mathematical Justification:**
        - θ encodes "what contexts this model is good for" (direction)
        - A encodes "how confident we are in θ" (magnitude)
        - We want to transfer domain knowledge (θ) but not sampling history (A)
        - By resetting A to λI, the new model starts with wide confidence intervals,
          allowing it to quickly diverge from the neighbor if it performs differently.
        - n_effective controls prior strength: low (1.0) = weak prior/high exploration,
          high (10.0) = strong prior/quick exploitation, default (5.0) = balanced warmup
        
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
            alpha: DEPRECATED (kept for backward compatibility, not used)
            n_effective: Tunable prior strength (default: 5.0). Simulates N samples
                worth of confidence in the neighbor's preferences. Higher values mean
                stronger priors (faster exploitation), lower means weaker priors
                (more exploration).
        
        Returns:
            Tuple of (A_new, b_new) where:
            - A_new = init_lambda * I (fresh identity, maximum uncertainty)
            - b_new = init_lambda * θ_neighbor * n_effective (scaled prior strength)
            
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
        # Get model description for embedding
        model_info = registry.get(model_id, {})
        model_desc = model_info.get("display_name", model_id)
        
        # Compute embedding for new model (with caching)
        # [KDD OPTIMIZATION]: Cache embeddings to avoid recomputation
        try:
            # Check if embedding is already cached
            if '_embedding' in model_info:
                new_embedding = model_info['_embedding']
            else:
                new_embedding = encoder.encode([model_desc], convert_to_numpy=True)[0]
                # Cache for future use (Pareto checks, etc.)
                if model_id in registry:
                    registry[model_id]['_embedding'] = new_embedding
        except Exception as e:
            logger.warning(f"Failed to encode {model_id}: {e}. Using identity init.")
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64)
            )
        
        # Find nearest neighbor among existing models
        best_neighbor = None
        best_similarity = -1.0
        
        for neighbor_id in bandit.models:
            if neighbor_id == model_id:
                continue
            neighbor_info = registry.get(neighbor_id, {})
            neighbor_desc = neighbor_info.get("display_name", neighbor_id)
            
            try:
                # [KDD OPTIMIZATION]: Use cached embedding if available
                if '_embedding' in neighbor_info:
                    neighbor_embedding = neighbor_info['_embedding']
                else:
                    neighbor_embedding = encoder.encode([neighbor_desc], convert_to_numpy=True)[0]
                    # Cache for future use
                    registry[neighbor_id]['_embedding'] = neighbor_embedding
                
                # Cosine similarity
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
            # [KDD REVIEW FIX]: Extract θ from neighbor, reset A to identity
            
            # Step 1: Extract neighbor's learned preferences (θ = A_inv @ b)
            with bandit._lock:  # Thread-safe read
                A_inv_neighbor = bandit.A_inv[best_neighbor]
                b_neighbor = bandit.b[best_neighbor]
            
            theta_neighbor = A_inv_neighbor @ b_neighbor
            
            # Step 2: Initialize new model with fresh uncertainty but neighbor's preferences
            # [KDD ENHANCEMENT]: Tunable prior strength via n_effective parameter
            # Low n_effective (e.g., 1.0) = Weak prior, high exploration
            # High n_effective (e.g., 10.0) = Strong prior, quick exploitation
            # Default 5.0 = Balanced warmup
            
            A_new = np.eye(bandit.dim) * bandit.init_lambda  # Maximum uncertainty
            b_new = (bandit.init_lambda * theta_neighbor) * n_effective  # Scaled prior strength
            
            logger.info(
                f"✨ Bootstrapping {model_id} from neighbor {best_neighbor} "
                f"(similarity={best_similarity:.2f}, n_effective={n_effective}). "
                f"Transferred θ (preferences), reset A (confidence) for exploration."
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
        Proxy method to extract features via the FeatureService.
        
        This method is maintained for backward compatibility with 
        experiment scripts and internal feedback loops.
        
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
        
        This property supports the **Reference Point Normalization** logic in
        `OptimizationProfile.from_reference()`. When users specify preferences like
        "I want 95% of the quality for 50% of the cost," they implicitly define
        these percentages relative to the best available model.
        
        **Selection Criteria:**
        The model with the **highest HLE (Human-Like Excellence) score** in the
        current registry. This ensures the reference point adapts automatically
        when you upgrade your model portfolio (e.g., adding GPT-5).
        
        **Why HLE?**
        HLE score is the canonical quality metric in BanditGPT, representing
        empirical win-rate against Claude 3.5 Sonnet on the LMSYS Arena dataset.
        It directly captures "how good" a model is at satisfying users.
        
        **Market-Relative Interpretation:**
        Note that `from_reference()` actually uses **market-wide normalization**
        (RouterConfig.market_cost_ceiling = $10.00) rather than this specific
        model's cost, which provides stability across portfolio changes.
        This property serves as a **transparent reference** for users to understand
        what "100% quality" means at any given time.
        
        Returns:
            Dictionary containing flagship model metadata with keys:
                - id: Model identifier (string)
                - hle: HLE score (float, typically 0.0-0.4 range)
                - input_cost_per_m: Cost in $/million tokens (float)
                - output_cost_per_m: Cost in $/million tokens (float)
                - ... (other registry metadata)
        
        Example:
            >>> router = BanditRouter.create()
            >>> ref = router.reference_model
            >>> print(f"Current flagship: {ref['id']} (HLE: {ref['hle']:.3f})")
            Current flagship: google/gemini-exp-1206 (HLE: 0.348)
        """
        if not self.registry:
            # Fallback if registry is empty (should never happen in production)
            logger.warning("Registry is empty, using fallback reference model")
            return {
                "id": "fallback-flagship",
                "hle": 1.0,
                "input_cost_per_m": 10.0,
                "output_cost_per_m": 10.0
            }
            
        # Find the model with the maximum quality score (composite metric)
        champion_id = max(
            self.registry,
            key=lambda m: self.registry[m].get("initial_quality") or self.registry[m].get("hle", 0.0) or 0.0
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
        context_encoder=None,
        alpha: float | None = None,
        exploration: str = "safe",
        priors: str = "hle",
        **kwargs
    ) -> "BanditRouter":
        """
        Backward compatibility factory method to create a BanditRouter.
        """
        # 1. Resolve Alpha from exploration string
        if alpha == None:
            exploration_map = {
                "static": 0.0,
                "safe": 0.05,
                "balanced": 0.5,
                "aggressive": 1.0
            }
            alpha = exploration_map.get(exploration, 0.5)
            
        # 2. Extract arguments for the factory, not the constructor
        state_path = kwargs.pop("state_path", None)
        prior_n_effective = kwargs.pop("prior_n_effective", 20.0) # Reduced for faster adaptation
        warmup_path = kwargs.pop("warmup_path", None)

        # 3. Initialize Router
        # Filter kwargs to only include those accepted by __init__
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = sig.parameters.keys()
        init_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            context_encoder=context_encoder,
            alpha=alpha,
            **init_kwargs
        )
        
        # 4. Apply Priors
        if priors == "hle":
            # Diagonal injection of benchmark scores
            for model_id in router.bandit.models:
                # [KDD FIX]: Use ONLY initial_quality (composite: 40% HLE, 25% GPQA, 20% Livecode, 15% IFbench)
                # Previous code cascaded through empirical_hle→raw_hle→hle, mixing semantically different metrics.
                # This caused initialization inconsistency - all models should use the same composite metric.
                m_data = router.registry.get(model_id, {})
                quality_score = m_data.get("initial_quality")
                
                if quality_score is None:
                    raise ValueError(
                        f"Model '{model_id}' missing 'initial_quality' field in registry. "
                        f"All models must have a composite quality score for consistent HLE priors."
                    )
                
                # KDD Simplification: Only set prior on bias term (last dimension)
                router.bandit.b[model_id][-1] += (quality_score * prior_n_effective)
                
        elif priors == "warmup" or (isinstance(priors, str) and (priors.endswith(".joblib") or "/" in priors)):
            # Load pre-computed matrices from disk
            priors_path = warmup_path or (priors if priors != "warmup" else None)
            
            if priors_path:
                priors_path = Path(priors_path)
            else:
                # Default location (KDD versioned artifacts or package assets)
                # Primary: artifacts/ (versioned, committed to git)
                # Fallback: assets/ (bundled with pip install)
                base_dir = Path(__file__).resolve().parent
                priors_path = base_dir.parent.parent / "artifacts" / "priors_warmup.joblib"
                
                # Check for alternative location in package assets if artifacts missing
                if not priors_path.exists():
                    priors_path = base_dir / "assets" / "priors_warmup.joblib"
                
            if priors_path and priors_path.exists():
                import joblib
                warmup_data = joblib.load(priors_path)
                n_warmup = warmup_data.get("n", 20000)
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
                
                router.bandit.refresh_inverse_cache()
                
                # CRITICAL FIX: Add regularization after scaling to prevent numerical instability
                for model_id in router.bandit.models:
                    router.bandit.A[model_id] += np.eye(router.bandit.dim) * router.bandit.init_lambda
                
                router.bandit.refresh_inverse_cache()
                logger.info(f"✅ Applied post-warmup regularization (λ={router.bandit.init_lambda}) from {priors_path}")
            else:
                logger.warning(f"Warmup priors not found at {priors_path}. Using cold start.")
        
        # 5. Post-Warmup Bias Injection (KDD FIX: Apply T-Shirt Sizing After Loading Priors)
        # We do this AFTER loading priors to ensure business logic adjusts data.
        # The bias is scaled by confidence to avoid "Dilution" problem.
        reg_config = RegistrationConfig()
        
        for model_id in router.bandit.models:
            speed = router.registry.get(model_id, {}).get("speed_profile", "balanced")
            
            # 1. Determine the Bias Shift (Delta Theta)
            bias_shift = 0.0
            if speed == "fast":
                bias_shift = reg_config.fast_bias
            elif speed == "slow":
                bias_shift = reg_config.slow_bias
            
            if abs(bias_shift) > 0.01:  # Skip negligible biases
                # 2. Scale by Confidence (The "Dilution" Fix)
                # We want to shift prediction θ by 'bias_shift'.
                # Since θ = A_inv @ b, we must shift b by (A @ delta_theta).
                # For the bias feature (last index), this simplifies to:
                # b[-1] += A[-1, -1] * bias_shift
                
                bias_idx = router.features.bias_index
                
                # Get current confidence level for the bias term
                confidence = router.bandit.A[model_id][bias_idx, bias_idx]
                
                # Apply the scaled shift
                router.bandit.b[model_id][bias_idx] += (confidence * bias_shift)
                
                logger.info(
                    f"⚖️ Applied {speed} bias ({bias_shift:+.2f}) to {model_id}. "
                    f"Scaled impact: {confidence:.1f} * {bias_shift} = {confidence*bias_shift:+.2f} added to b."
                )
        
        # Refresh inverse cache after bias injection
        router.bandit.refresh_inverse_cache()
        
        # 6. Load state if provided (overwrites any priors applied above)
        if state_path:
            router.load_state(state_path)
                
        return router




    def _load_zero_shot_priors(self, prior_n_effective: float):
        """
        Load pretrained weights from default_weights.json.
        
        Philosophy:
        - A = λI (Identity → High Plasticity, adapts to user's data)
        - θ = pretrained from JSON (Initial Intuition for reasoning/turbo archetypes)
        - b = prior_n_effective * init_lambda * θ
        """
        logger.info(f"Loading pretrained weights (N_eff={prior_n_effective:.1f})")
        
        weights_path = Path(__file__).parent / "priors" / "default_weights.json"
        
        if not weights_path.exists():
            # Use HLE scores directly as priors
            for model_id in self.bandit.models:
                raw_hle = float(self.registry.get(model_id, {}).get("hle", 0.15))
                neutral_ctx = np.zeros(self.bandit.dim)
                neutral_ctx[-1] = 1.0
                self.bandit.b[model_id] = prior_n_effective * raw_hle * neutral_ctx
            return
        
        with open(weights_path) as f:
            data = json.load(f)
        
        # Map model IDs to archetypes (reasoning vs turbo)
        for model_id in self.bandit.models:
            # Determine archetype from model metadata or heuristics
            hle = float(self.registry.get(model_id, {}).get("hle", 0.15))
            is_reasoning = hle > 0.15  # High HLE = reasoning model
            
            archetype = "reasoning_model" if is_reasoning else "turbo_model"
            
            if archetype in data.get("models", {}):
                weights_config = data["models"][archetype]["weights"]
                
                # KDD Simplification: Pure Decision Engine (24D)
                # We no longer use handcrafted or anchor features.
                # Initialize bias term with HLE-based specialist prior.
                theta = np.zeros(self.bandit.dim)
                raw_hle = float(self.registry.get(model_id, {}).get("hle", 0.15))
                theta[-1] = raw_hle  # Bias term captures global specialist probability
                
                # Compute b = N_eff * λ * θ (pseudocounts for θ)
                self.bandit.b[model_id] = prior_n_effective * self.bandit.init_lambda * theta
            else:
                # Fallback to HLE
                raw_hle = float(self.registry.get(model_id, {}).get("hle", 0.15))
                neutral_ctx = np.zeros(self.bandit.dim)
                neutral_ctx[-1] = 1.0
                self.bandit.b[model_id] = prior_n_effective * raw_hle * neutral_ctx

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

    def _is_pareto_dominated(self, new_model_data: Dict[str, Any]) -> bool:
        """
        Phase 1: Optimizer Gatekeeper (Corrected).
        
        Checks if the new model (with Optimistic Reward=0.95) is dominated by existing models
        using ABSOLUTE MARKET ANCHORS.
        
        The baseline check must use absolute market width to prevent local entrapment.
        Previous logic used relative stats from the current registry, which 
        caused false rejections for models that looked "expensive" locally but were
        actually "cheap" globally.
        
        [KDD REVIEW FIX - Improvement B]: Minimum Novelty Check
        Also rejects "feature spam" - near-duplicate models that differ only in price.
        """
        # [KDD REVIEW FIX - Improvement B]: Minimum Novelty Check
        # Prevent "Feature Spam" where providers release 100 variations of a model,
        # each $0.0001 cheaper. While they aren't strictly dominated by the 0.05 margin,
        # they shouldn't all enter the registry and dilute attention.
        #
        # Strategy: Compute embedding similarity. If new model is too similar to ANY
        # existing model (cosine similarity > 0.9, i.e., distance < 0.1), reject it
        # ONLY IF probation is at capacity. If we have room, allow the variation through
        # so it can compete in probation (max_probation_models limit provides the throttle).
        
        new_model_id = new_model_data.get("openrouter_id", "unknown")
        new_model_desc = new_model_data.get("display_name", new_model_id)
        
        try:
            # [KDD OPTIMIZATION]: Use cached embedding if available
            if '_embedding' in new_model_data:
                new_embedding = new_model_data['_embedding']
            else:
                new_embedding = self.encoder.encode([new_model_desc], convert_to_numpy=True)[0]
                # Cache for admix_theta_from_neighbors reuse
                new_model_data['_embedding'] = new_embedding
                
            new_embedding_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-12)
            
            # Check similarity to all existing models
            for m_id, m_data in self.registry.items():
                m_desc = m_data.get("display_name", m_id)
                try:
                    # [KDD OPTIMIZATION]: Use cached embedding if available
                    if '_embedding' in m_data:
                        m_embedding = m_data['_embedding']
                    else:
                        m_embedding = self.encoder.encode([m_desc], convert_to_numpy=True)[0]
                        # Cache for future checks
                        self.registry[m_id]['_embedding'] = m_embedding
                    m_embedding_norm = m_embedding / (np.linalg.norm(m_embedding) + 1e-12)
                    
                    # Cosine similarity
                    similarity = float(np.dot(new_embedding_norm, m_embedding_norm))
                    
                    # If too similar (similarity > 0.9), this is likely a near-duplicate
                    # [KDD REVIEW FIX - Improvement B]: Check probation capacity first
                    if similarity > 0.9:
                        # Check if we have room in probation to evaluate this variation
                        try:
                            probation_count = sum(1 for m in self.probation_models.values() 
                                                if self.bandit.t < m.get('immune_until', 0))
                            
                            if probation_count < self.config.max_probation_models:
                                # We have capacity - allow this variation to compete in probation
                                logger.info(
                                    f"⚠️ Near-duplicate detected: {new_model_id} similar to {m_id} "
                                    f"(similarity={similarity:.3f}), but probation has room "
                                    f"({probation_count}/{self.config.max_probation_models}). Allowing through."
                                )
                                # Don't reject - let it through to Pareto check
                            else:
                                # Probation is full - activate spam protection
                                logger.info(
                                    f"🚫 Novelty Rejection: {new_model_id} is near-duplicate of {m_id} "
                                    f"(similarity={similarity:.3f}). Probation full ({probation_count}/"
                                    f"{self.config.max_probation_models}). Feature spam protection active."
                                )
                                return True  # Reject as dominated (spam)
                        except Exception as e:
                            # If probation check fails, fall through to Pareto check (don't reject on errors)
                            logger.debug(f"Probation check failed for novelty protection: {e}")
                        
                except Exception as e:
                    logger.debug(f"Failed to encode {m_id} for novelty check: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to encode new model {new_model_id} for novelty check: {e}")
            # If encoding fails, fall through to Pareto check (don't reject on errors)
        
        # Proceed with standard Pareto dominance check
        def get_absolute_utility(
            quality_score: float, 
            cost_per_m: float, 
            lat_s: float, 
            profile: Dict[str, float]
        ) -> float:
            """Calculate utility using absolute market normalization."""
            # Cost Utility (Higher is Cheaper)
            c_norm = (np.log(10.00) - np.log(max(cost_per_m/1000.0, 0.0005))) / (np.log(10.00) - np.log(0.0005))
            c_norm = np.clip(c_norm, 0, 1)
            
            # Latency Utility (Higher is Faster)
            l_norm = (np.log(5.0) - np.log(max(lat_s, 0.05))) / (np.log(5.0) - np.log(0.05))
            l_norm = np.clip(l_norm, 0, 1)
            
            # Linear weighted sum
            return profile['w_q'] * quality_score + profile['w_c'] * c_norm + profile['w_l'] * l_norm

        for m_id, m_data in self.registry.items():
            domination_count = 0
            profiles = ["max_quality", "arbitrage", "cost_saver", "low_latency"]
            
            for p_name in profiles:
                p = OptimizationProfile.get(p_name)
                u_existing = get_absolute_utility(
                    m_data.get('initial_quality', 0.5), # KDD FIX
                    m_data.get('cost_per_1m_tokens', 10.0),
                    m_data.get('median_latency_s', 2.0),
                    p
                )
                # [KDD REVIEW FIX]: Relax optimism from 1.0 to 0.95 (Refined Admissions)
                # Prevents "Spam Models" that are slightly cheaper from flooding the registry
                # even if they likely have terrible quality.
                u_new = get_absolute_utility(
                    0.95, 
                    new_model_data.get('cost_per_1m_tokens', 10.0),
                    new_model_data.get('median_latency_s', 2.0),
                    p
                )
                if u_existing > u_new + 0.05:
                    domination_count += 1
            
            if domination_count == len(profiles):
                logger.info(f"🚫 Pareto Rejection: {new_model_data.get('openrouter_id')} dominated by {m_id}")
                return True
                
        return False

    def _get_contextual_stats(self, model_id: str, x: np.ndarray, in_tok: int, out_tok: int) -> Dict[str, Any]:
        """
        Get context-aware statistics (Mean, Uncertainty, Cost) for a single model.
        Used to build the dynamic Pareto frontier for a specific prompt.
        """
        # 1. Get LinUCB Predictions (Quality)
        with self.bandit._lock: # Thread-safe read
            # Mean (Predicted Quality)
            theta = self.bandit.A_inv[model_id] @ self.bandit.b[model_id]
            mean_quality = float(theta.dot(x))
            
            # Uncertainty (Std Dev) for Optimism
            dt = self.bandit.t - self.bandit.last_update[model_id]
            decay_factor = self.bandit.gamma ** dt
            var = float(x.dot(self.bandit.A_inv[model_id]).dot(x))
            var_inflated = var / max(decay_factor, 1e-12)
            uncertainty = float(np.sqrt(max(var_inflated, 1e-12)))

        # 2. Get Cost (Estimated)
        cost_usd = self._estimate_cost(model_id, in_tok, out_tok)
        cost_per_1k = cost_usd * 1000.0
        
        return {
            "id": model_id,
            "mean_quality": mean_quality,
            "uncertainty": uncertainty,
            "cost": cost_per_1k
        }

    def _filter_pareto_frontier(self, candidates: List[str], x: np.ndarray, in_tok: int, out_tok: int) -> List[str]:
        """
        Step A: The Pareto Filter.
        Prunes models that are strictly dominated by others based on (Cost vs. Mean Quality).
        
        [KDD ARCHITECTURAL FIX]: Use ONLY mean quality for Pareto filtering, NOT UCB.
        - Pareto filtering = hard exclusion → miscalibration causes permanent damage
        - UCB selection = soft exploration → miscalibration self-corrects with data
        
        By using only mean quality (no exploration bonus), we prevent inflated UCB
        from allowing dominated models to survive the filter.
        """
        stats = {
            m: self._get_contextual_stats(m, x, in_tok, out_tok) 
            for m in candidates
        }
        
        survivors = []
        for cand_id in candidates:
            cand = stats[cand_id]
            # Use ONLY mean quality, not UCB (no exploration bonus for hard filtering)
            cand_quality = cand['mean_quality']
            
            is_dominated = False
            for opp_id in candidates:
                if cand_id == opp_id: continue
                opp = stats[opp_id]
                # Use ONLY mean quality for opponent too
                opp_quality = opp['mean_quality']
                
                # Dominated if opponent is cheaper AND has higher mean quality
                if (opp['cost'] <= cand['cost']) and (opp_quality > cand_quality):
                    if (opp['cost'] < cand['cost']) or (opp_quality > cand_quality + 1e-6):
                        is_dominated = True
                        break
            
            if not is_dominated:
                survivors.append(cand_id)
        
        return survivors if survivors else candidates


    def admit_new_model(self, model_data: Dict[str, Any], dampening: float = 0.1) -> bool:
        """
        Validate and Initialize a new model using semantic neighbor transfer.
        
        [KDD REVIEW FIX - Concern B]: Consolidated with register_model pipeline.
        Both startup and runtime paths now use the same admix_theta_from_neighbors
        logic (fixed to transfer θ only, reset A for exploration).
        
        Args:
            model_data: Model metadata dictionary with openrouter_id, cost, etc.
            dampening: DEPRECATED (kept for API compatibility, not used)
        
        Returns:
            True if admitted, False if rejected
        """
        model_id = model_data["openrouter_id"]
        
        # Phase 1: Admission Gatekeeping
        # Check if model is Pareto dominated (with optimistic quality=0.95)
        if self._is_pareto_dominated(model_data):
            logger.info(f"Refusing admission to {model_id}: Pareto Dominated (even with 0.95 optimistic reward).")
            return False
            
        # [KDD REVIEW FIX]: Probation Spam Guard
        # Check how many models are currently in probation
        probation_count = sum(1 for m in self.probation_models.values() 
                            if self.bandit.t < m.get('immune_until', 0))
        
        if probation_count >= self.config.max_probation_models:
            logger.warning(
                f"🚫 Admission Denied for {model_id}: Too many models in probation ({probation_count}). "
                f"Wait for existing models to graduate or increase max_probation_models."
            )
            return False
            
        # Phase 2: Initialization via Semantic Transfer
        # [KDD REVIEW FIX - Concern B]: Use the same logic as register_model
        # This ensures both paths use the fixed θ-only transfer (no confident transfer trap)
        
        # Add to registry first (needed by admix_theta_from_neighbors)
        self.registry[model_id] = model_data
        
        # Use the consolidated bootstrapping logic
        # This will:
        # 1. Find semantically similar neighbor via embedding
        # 2. Extract neighbor's learned preferences (θ = A_inv @ b)
        # 3. Initialize with A = λI (fresh), b = λ × θ (preferences)
        A_init, b_init = self.admix_theta_from_neighbors(
            model_id=model_id,
            registry=self.registry,
            bandit=self.bandit,
            encoder=self.encoder,
            n_effective=5.0  # Balanced prior strength for neighbor bootstrapping
        )
        
        # Add arm to bandit with bootstrapped parameters
        self.bandit.models.append(model_id)
        self.bandit.A[model_id] = A_init
        self.bandit.b[model_id] = b_init
        self.bandit.A_inv[model_id] = safe_inv(A_init)
        self.bandit.last_update[model_id] = self.bandit.t
        
        logger.info(
            f"✅ Admitted {model_id} via semantic transfer. "
            f"Initialized with θ from neighbor, fresh A for exploration."
        )

        # Phase 3: Probation Tracking
        # New models get 500-request probation period
        self.probation_models[model_id] = {
            "start_t": self.bandit.t,
            "status": "PROBATION",
            "immune_until": self.bandit.t + self.config.probation_requests
        }
        
        return True


    def _get_sample_counts(self, arms: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Count selectors in logs using O(N) Counter optimization.
        
        Args:
            arms: List of arm IDs to count (None = all arms in bandit)
            
        Returns:
            Dictionary mapping arm ID to sample count
        """
        # [KDD REVIEW FIX]: Use persistent counts to avoid Rolling Window fallacy
        # Ephemeral log counting via Counter(self.logs) is only used for debugging or 
        # when persistent counts are not yet initialized (bulk load logic).
        arms_to_count = arms if arms is not None else self.bandit.models
        return {arm: self.model_counts.get(arm, 0) for arm in arms_to_count}



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
    
    def _resolve_utility_weights(
        self,
        profile: str | Dict[str, float],
        max_cost: float | None,
        max_latency: float | None
    ) -> Tuple[float, float, float, float]:
        """
        Resolve optimization profile weights.

        **KDD COMPLIANCE UPDATE (Jan 2026): Exchange Rate Logic**
        Weights are NO LONGER normalized to sum to 1.0.
        
        They act as economic exchange rates:
        - w_c = 1.0  (Base currency: $1.00)
        - w_q = 100.0 (Value of 1% Quality gain relative to $1.00)
        
        This allows the router to correctly trade off inputs with vastly 
        different scales (e.g., Cost in [0, 15] vs Quality in [0, 1]).

        Args:
            profile: Optimization profile name
            max_cost: Hard cost constraint (optional)
            max_latency: Hard latency constraint (optional)

        Returns:
            Tuple of (w_q, w_c, w_l, alpha_scale) raw weights
        """
        # Get raw weights (copy to avoid mutating class profile)
        weights = OptimizationProfile.get(profile).copy()
        
        # Extract individual components (default to 0.0 if missing)
        w_q = weights.get("w_q", 0.0)
        w_c = weights.get("w_c", 0.0)
        w_l = weights.get("w_l", 0.0)
        
        # EXTRACT ALPHA SCALE (Risk Aversion Factor)
        # Default to 1.0 (Standard Exploration) if missing
        alpha_scale = weights.get("alpha_scale", 1.0)

        # Orthogonal Optimization:
        # If a hard constraint exists, disable the soft optimization for that 
        # dimension and re-allocate its "importance" to Quality.
        # Since we aren't summing to 1, we just add the raw value.
        if max_cost is not None:
            w_q += w_c
            w_c = 0.0
        
        if max_latency is not None:
            w_q += w_l
            w_l = 0.0

        # KDD FIX: Removed normalization block.
        # We return the raw exchange rates directly.
        
        return w_q, w_c, w_l, alpha_scale
    

    
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
            if quality_floor:
                scores = self.registry.get(m, {}).get("scores", {})
                passes = True
                for k, v in quality_floor.items():
                    if float(scores.get(k, 0)) < v:
                        passes = False
                        break
                if not passes:
                    continue
                    
            filtered.append(m)
            
        if not filtered:
            filtered = list(self.registry.keys())  # Ultimate fallback
            
        return filtered
    
    def _calculate_penalties(
        self,
        filtered: List[str],
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate absolute cost and latency penalties for each model.
        
        Uses market anchors (not relative to current pool) for stable penalties:
        - Cost: Floor=$0.0005/1k, Ceiling=$10.00/1k
        - Latency: Floor=0.05s, Ceiling=5.0s
        
        Args:
            filtered: List of candidate model IDs
            input_tokens: Input token count
            output_tokens: Output token count
            
        Returns:
            Tuple of (cost_penalties, latency_penalties) dicts
        """
        # Calculate costs and latencies
        costs = {m: self._estimate_cost(m, input_tokens, output_tokens) for m in filtered}
        lats = {m: self._estimate_latency(m, output_tokens) for m in filtered}
        
        # Cost penalties (absolute, using market anchors)
        cost_penalties = {}
        total_tokens = max(1, input_tokens + output_tokens)
        
        for m in filtered:
            # KDD FIX: Calculate Rate ($/1k) = (Total Cost / Total Tokens) * 1000
            cost_per_1k = (costs[m] / total_tokens) * 1000
            cost_penalties[m] = self._calculate_absolute_penalty(cost_per_1k)
        
        # [KDD REVIEW FIX]: Use precomputed anchors
        latency_penalties = {}
        for m in filtered:
            safe_lat = max(lats[m], self._market_lat_floor)
            log_lat = np.log(safe_lat)
            norm_lat = (log_lat - self._market_lat_floor_log) / self._market_lat_range
            latency_penalties[m] = max(0.0, min(1.0, norm_lat))
        
        return cost_penalties, latency_penalties
    
    def _score_candidates(
        self,
        filtered: List[str],
        x: np.ndarray,
        w_q: float,
        w_c: float,
        w_l: float,
        alpha_scale: float,
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[str, float, float]:
        """
        Calculate utility scores and select best model.
        
        **KDD FINAL VERSION (Jan 2026)**:
        Uses an ADDITIVE utility formula that separates deterministic trade-offs 
        from exploration uncertainty.
        
        Formula:
          Utility = Base_Utility + Exploration_Bonus
          Base_Utility = (w_q * mean_quality) + (w_c * cost_savings) + (w_l * lat_savings)
          Exploration_Bonus = alpha * scaling_factor * w_q * std_quality
          
        This ensures:
        1. Exploration only scales with quality importance (w_q).
        2. Cost signals are never drowned out in cost-sensitive profiles.
        3. The trade-off between mean quality and cost is perfectly linear.
        4. Risk-averse profiles (Max Quality) can opt-out of exploration.
        """
        best_model = filtered[0]
        best_utility = -float("inf")
        
        sample_counts = self._get_sample_counts(filtered)
        cost_penalties, latency_penalties = self._calculate_penalties(filtered, input_tokens, output_tokens)
        
        for m in filtered:
            # 1. Calculate deterministic quality prediction (mean)
            with self.bandit._lock:
                theta = self.bandit.A_inv[m] @ self.bandit.b[m]
                mean_quality = float(theta.dot(x))
                
                # Global Forgetting Adjustment
                dt = self.bandit.t - self.bandit.last_update[m]
                decay_factor = self.bandit.gamma ** dt
                
                # 2. Calculate exploration uncertainty (std)
                var = float(x.dot(self.bandit.A_inv[m]).dot(x))
                var_inflated = var / max(decay_factor, 1e-12) 
                std = float(np.sqrt(max(var_inflated, 1e-12)))
            
            # 3. Calculate separate utility components
            norm_cost = cost_penalties[m]
            norm_lat = latency_penalties[m]
            
            # Use raw quality prediction with floor at 0
            # The warmup priors were trained on [0,1] rewards, so predictions 
            # should naturally be on the correct scale. We just clip negative values.
            # Sigmoid was compressing differences too much (e.g., 0.9→0.71, 1.1→0.75)
            norm_quality = max(0.0, mean_quality)
            
            # Base Trade-off Utility (Deterministic)
            base_utility = (
                w_q * norm_quality + 
                w_c * (1.0 - norm_cost) + 
                w_l * (1.0 - norm_lat)
            )
            
            # Exploration Bonus (Scaled by Quality Weight)
            # CRITICAL FIX: Exploration must be scaled by w_q to match base utility scale.
            # Without w_q: MAX_QUALITY exploration is 2500x weaker than exploitation!
            # With w_q: exploration becomes proportional to value at risk.
            exploration_bonus = self.bandit.alpha * alpha_scale * w_q * std
            
            # Probation Bonus (Legacy support)
            # [KDD REVIEW FIX - Improvement A]: Link to probation_models list
            # Only apply bonus if model is ACTUALLY in probation, not just low sample count
            probation_bonus = 0.0
            if self.config.probation_bonus > 0 and m in self.probation_models:
                count = sample_counts.get(m, 0)
                if count < self.config.pruning_min_samples:
                    decay = 1.0 - (count / self.config.pruning_min_samples)
                    probation_bonus = self.config.probation_bonus * w_q * decay
            
            # FINAL ADDITIVE UTILITY
            total_utility = base_utility + exploration_bonus + probation_bonus
            
            if self.verbose_routing:
                logger.info(f"Scoring {m:15s} | Utility: {total_utility:8.4f}")
                logger.info(f"  > [Base]   Q_util: {w_q*norm_quality:6.4f} (raw={mean_quality:.3f}, norm={norm_quality:.3f}) | C_util: {w_c*(1-norm_cost):6.4f} | L_util: {w_l*(1-norm_lat):6.4f}")
                logger.info(f"  > [Bonus]  Explore: {exploration_bonus:6.4f} (std={std:.3f}) | Probation: {probation_bonus:6.4f}")
            
            if total_utility > best_utility:
                best_utility = total_utility
                best_model = m
                
        if self.verbose_routing:
            logger.info(f"WINNER: {best_model} (Utility={best_utility:.4f})")
                
        return best_model, best_utility, (w_q + w_c + w_l)
    
    
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
            request_id=str(time.time_ns()),
            timestamp_s=time.time(),
            prompt=prompt_text,
            selected_model=model,
            predicted_utility=float(utility),
            cost_usd=self._estimate_cost(model, input_tokens, output_tokens),
            latency_s=self._estimate_latency(model, output_tokens),
            cluster_id=None,  # Legacy: replaced by Virtual Anchors
            cluster_similarity=None,
            context_vector=x,  # Cache for feedback loop
            total_priority_weight=total_weight
        )
        # [KDD REVIEW FIX]: Manage parallel index eviction before deque append
        if len(self.logs) >= (self.logs.maxlen or float('inf')):
            old_log = self.logs[0]
            self.log_index.pop(old_log.request_id, None)
            
        self.logs.append(log)
        self.log_index[log.request_id] = log
        
        # Save context for delayed feedback (RLHF, human ratings, etc.)
        self.context_store.save_context(log.request_id, x, model)
        
        return log

    def route(
        self,
        prompt: str | np.ndarray,
        *,
        profile: str | Dict[str, float] = "smart_shopper", # Changed default
        # Reference Point Normalization (Convenience Parameters)
        quality_tolerance: float | None = None,
        cost_savings: float | None = None,
        latency_savings: float | None = None,
        # Other routing parameters

        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] = None,
        input_tokens: int | None = None,
        output_tokens: int = 600,
    ) -> Tuple[str, RoutingLog]:
        """
        Route a prompt to the best model.
        
        **Reference Point Normalization (New):**
        You can now specify preferences using intuitive trade-off percentages:
        
        >>> router.route(
        ...     prompt="Explain quantum computing",
        ...     quality_tolerance=0.05,  # Accept 5% quality drop
        ...     cost_savings=0.50         # To save 50% cost
        ... )
        
        This is equivalent to calling `OptimizationProfile.from_reference()` manually.
        If both `quality_tolerance` and `cost_savings` are provided, they override
        the `profile` parameter.
        
        Args:
            prompt: Input prompt text or pre-embedded vector
            profile: Named profile or custom weight dict (e.g., "max_quality", "arbitrage")
            quality_tolerance: **[Reference Point]** % quality drop acceptable (e.g., 0.05 = 5%)
            cost_savings: **[Reference Point]** % cost reduction desired (e.g., 0.50 = 50%)
            latency_savings: **[Reference Point]** % latency reduction desired (optional)
            max_cost: Hard cost constraint ($/1k tokens)
            max_latency: Hard latency constraint (seconds)
            quality_floor: Minimum quality scores per model
            input_tokens: Input token count (auto-estimated if not provided)
            output_tokens: Expected output token count (default 600)
        
        Returns:
            Tuple of (model_id, routing_log)
        """
        # --- NEW LOGIC: RESOLVE PROFILE ---
        # If user provides tolerances, generate the profile on the fly
        if quality_tolerance is not None and cost_savings is not None:
            profile_weights = OptimizationProfile.from_reference(
                quality_tolerance=quality_tolerance,
                cost_savings=cost_savings,
                latency_savings=latency_savings or 0.0
            )
        else:
            # Fallback to standard named profile or custom dict
            profile_weights = profile
        # ----------------------------------
        
        # Orchestrate the routing process using focused helper methods
        x, prompt_text = self._build_routing_features(prompt)
        candidates = list(self.registry.keys())
        filtered = self._filter_by_constraints(
            candidates, prompt, max_cost, max_latency, quality_floor, input_tokens, output_tokens
        )
        
        # Estimate tokens for scoring
        in_tok = input_tokens or estimate_tokens_rough(prompt_text)

        # [NEW] Logic Branch: Pareto vs Legacy
        # Check if the requested profile is one of our new Pareto/Lambda profiles
        is_pareto_mode = isinstance(profile_weights, str) and profile_weights in self.PARETO_PROFILES
        
        best_model = None
        best_utility = -float('inf')
        total_weight = 1.0

        if is_pareto_mode:
            # --- PATH A: NEW PARETO LOGIC ---
            
            # Step 1: Pareto Filter (BYPASSED - portfolio is pre-curated to Pareto-optimal models)
            # All models in the portfolio are Pareto-optimal by construction.
            efficient_models = filtered
            
            # Step 2: Linear Utility Selection
            # Score = Quality - (Lambda * Cost)
            lambda_val = self.PARETO_PROFILES[profile_weights]
            
            # [KDD FIX]: Use consistent exploration (UCB) in both filter and selection
            # Previously used optimistic UCB in filter but pessimistic mean in selection,
            # causing uncertain models to pass filter but then lose in selection.
            # This creates "Explore-then-Exploit Disconnect" where new models can't win.
            exploration_alpha = self.PARETO_EXPLORATION_CONSTANT
            
            for m in efficient_models:
                stats = self._get_contextual_stats(m, x, in_tok, output_tokens)
                
                # Use UCB (Optimistic) for final selection (consistent with filter)
                # Exploration bonus allows uncertain models to win and gather data
                ucb_quality = stats['mean_quality'] + (exploration_alpha * stats['uncertainty'])
                utility = ucb_quality - (lambda_val * stats['cost'])
                
                if utility > best_utility:
                    best_utility = utility
                    best_model = m
            
            # For logging compatibility
            total_weight = lambda_val 

        else:
            # --- PATH B: LEGACY LOGIC (Keep for backward compatibility) ---
            # Use _resolve_utility_weights and _score_candidates as before
            w_q, w_c, w_l, alpha_scale = self._resolve_utility_weights(profile_weights, max_cost, max_latency)
            best_model, best_utility, total_weight = self._score_candidates(
                filtered, x, w_q, w_c, w_l, alpha_scale, in_tok, output_tokens
            )

        # Pruning removed for V1 (fixed portfolio)
        log = self._create_routing_log(
            prompt_text, best_model, best_utility, x, in_tok, output_tokens, total_weight
        )
        
        return best_model, log
    
    def process_feedback(
        self,
        request_id: str,
        reward: float,
        *,
        cluster_boost: bool = True
    ) -> None:
        """
        Process feedback for a routing decision with optional cluster-aware boost.
        
        Args:
            request_id: ID from RoutingLog
            reward: Base reward (0-1, typically from judge)
            cluster_boost: Whether to apply cluster-aware reward boosting
        """
        # [KDD REVIEW FIX]: O(1) lookup via parallel index instead of O(N) linear scan
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
                cluster_id=None, cluster_similarity=None, context_vector=context
            )
        
        # Apply cluster boost if enabled and cluster was detected
        boosted_reward = reward
        boost_amount = 0.0
        
        if cluster_boost and log.cluster_id is not None:
            # Look up model's z-score for this cluster
            model_data = self.registry.get(log.selected_model, {})
            z_scores = model_data.get('cluster_z_scores')
            
            if z_scores and len(z_scores) > log.cluster_id:
                z_score = z_scores[log.cluster_id]
                
                # Boost formula: reward *= (1 + z_score * boost_weight)
                # Positive z-score → model excels at this cluster → get bonus
                # Negative z-score → model weak at this cluster → get penalty
                boost_factor = 1.0 + (z_score * self.cluster_boost_weight)
                boosted_reward = reward * boost_factor
                boost_amount = boosted_reward - reward
                
                # Log significant boosts
                if abs(boost_amount) > 0.01:
                    logger.info(
                        f"Cluster boost: model={log.selected_model}, "
                        f"cluster={log.cluster_id}, z={z_score:.2f}, "
                        f"reward: {reward:.3f} → {boosted_reward:.3f} ({boost_amount:+.3f})"
                    )
        
        # [KDD REVIEW FIX]: Persistent monotonicity (Probation Fix)
        self.model_counts[log.selected_model] += 1
        
        # Update bandit with boosted reward
        # Use cached context vector to avoid re-encoding
        x = log.context_vector if log.context_vector is not None else self._get_context_vector(log.prompt)
        self.bandit.update(log.selected_model, x, boosted_reward)
        
        # Periodic stability check (cheap O(d) operation)
        # Prevents numerical instability in low-traffic arms when update_lambda=0
        if (self.config.stability_check_interval > 0 and 
            self.bandit.t % self.config.stability_check_interval == 0):
            # Check all arms for numerical stability
            for model in self.bandit.models:
                self.bandit._check_numerical_stability(model, self.config)

    def get_probabilities(self, context: str | np.ndarray, model_ids: List[str] | None = None) -> Dict[str, float]:
        """Get the probability of each model being the specialist for a given context."""
        x = self.features.extract_features(context)
        models = model_ids if model_ids else self.bandit.models
        return self.bandit.get_probabilities(x, models)

    def update(self, model_id: str, context: str | np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update the bandit's internal state with a new observation."""
        x = self.features.extract_features(context)
        self.bandit.update(model_id, x, reward, weight)
        
        # Periodic stability check (cheap O(d) operation)
        # Prevents numerical instability in low-traffic arms when update_lambda=0
        if (self.config.stability_check_interval > 0 and 
            self.bandit.t % self.config.stability_check_interval == 0):
            # Check all arms for numerical stability  
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
        if model_id not in self.bandit.A_inv:
            raise ValueError(f"Model {model_id} not found in bandit registry")
        
        # 1. Get the learned weights (theta) for this model
        theta = self.bandit.A_inv[model_id] @ self.bandit.b[model_id]
        
        # 2. Element-wise multiplication shows contribution of each feature
        contributions = theta * context_vector
        
        # 3. Map back to feature names
        explanation = {}
        
        # Based on the 24-D structure: [PCA (23) | Bias (1)]
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
        
        # Get scores for all models
        model_scores = []
        for model_id in self.bandit.models:
            if model_id not in self.bandit.A_inv:
                continue
            theta = self.bandit.A_inv[model_id] @ self.bandit.b[model_id]
            score = float(np.dot(theta, x))
            model_scores.append((model_id, score))
        
        # Sort by score (highest first) and take top-k
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [m[0] for m in model_scores[:top_k]]
        
        # Generate explanations for top-k models
        explanations = {}
        for model_id in top_models:
            explanations[model_id] = self.explain_decision(model_id, x, threshold)
        
        return explanations



    def add_model(self, model_id: str, definition: Dict[str, Any]) -> None:
        """
        Add a new model to the router dynamically.
        
        Args:
            model_id: Unique identifier for the model (e.g. 'provider/model-name').
            definition: Dict containing metadata. MUST include:
                        - 'input_cost_per_m': Cost per million input tokens (float).
                        Optional:
                        - 'benchmark_score': Score for the active benchmark (float).
                        - 'description': Text description for clustering.
                        - 'tags': List of tags for clustering.
        """
        # 1. Validation
        if "input_cost_per_m" not in definition:
            raise ValueError(f"Model definition for '{model_id}' must include 'input_cost_per_m'")
            
        # 2. Update Registry
        self.registry[model_id] = definition
        
        # 3. Add to Bandit
        self.bandit.add_arm(model_id)
        
        # 4. Initialize Prior (HLE)
        # We reuse the logic from load_from_benchmark but for a single model
        # This ensures the new model gets the same "Smart Prior" treatment
        
        # Get score (default 0.05 if missing)
        raw_score = float(definition.get("hle") or 0.05)
        score = transform_hle_to_prior(raw_score)
        
        if score > 0:
            # Initialize with prior belief
            # Set the bias term (last element) of b to prior_strength * score
            # This effectively gives it a "mean reward" of 'score' for the bias feature.
            config = RouterConfig()
            pseudocounts = config.registration.prior_pseudocounts
            self.bandit.b[model_id][-1] = pseudocounts * score
            # Also increase confidence in the bias term
            self.bandit.A[model_id][-1, -1] += pseudocounts
            self.bandit.A_inv[model_id] = safe_inv(self.bandit.A[model_id])


    def save_state(self, path: Path | str) -> None:
        """Save the bandit's learned state to disk."""
        self.bandit.save_state(path)

    def load_state(self, path: Path | str) -> None:
        """Load the bandit's learned state from disk."""
        self.bandit.load_state(path)

    def calibrate(self, prompts: List[str], *, apply: bool = True, verbose: bool = False) -> Dict[str, float]:
        """
        Auto-calibrate complexity normalization parameters from user's dataset.
        
        Delegates to utils.calibrate_complexity for actual calibration.
        See utils/calibration.py for full implementation details.
        
        Args:
            prompts: List of representative prompts from your production traffic.
                    Recommended: 500-1000 samples for stable estimates.
            apply: If True, update the router's COMPLEXITY_MU and COMPLEXITY_SIGMA.
                   If False, just return statistics without modifying the router.
            verbose: If True, print detailed statistics and recommendations.
        
        Returns:
            Dict with calibration statistics (mean, std, min, max, p1, p99, n_samples)
        
        Raises:
            ValueError: If prompts list is empty or too small (<10 samples)
        """
        return calibrate_complexity(self, prompts, apply=apply, verbose=verbose)

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
        # [KDD REVIEW FIX]: Use precomputed market anchors (Performance)
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
