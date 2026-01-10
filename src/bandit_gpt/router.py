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
from collections import Counter, deque
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
    from bandit_gpt.utils import sigmoid, calibrate_complexity, procedural_warmup, safe_inv
except ImportError:
    # Fallback for direct file import (not installed as package)
    from .storage import ContextStore, EphemeralContextStore, SqliteContextStore
    from .utils import sigmoid, calibrate_complexity, procedural_warmup, safe_inv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router Configuration (Magic Numbers Documented)
# ---------------------------------------------------------------------------

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
    # Based on 2024-2025 Market Analysis:
    # Floor: $0.0005/1k (DeepSeek V3, Gemini Flash tier)
    # Ceiling: $10.00/1k (o1-high, Claude Opus reasoning tier)
    # If market changes (e.g., GPT-5 costs $50/1k), update ceiling.
    market_cost_floor: float = 0.0005  # $/1k tokens
    market_cost_ceiling: float = 10.00  # $/1k tokens
    
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
    
    # RegistrationConfig removed - trusting LinUCB to learn from data
    # instead of encoding rigid priors
    
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
    Named presets for utility function weights (Quality vs Cost vs Latency).
    
    **KDD FIX (Jan 2026)**: All weights are pre-normalized to sum to 1.0.
    This ensures:
    - Interpretable trade-off ratios (w_q/w_c represents economic exchange rate)
    - Consistent exploration scaling (α_eff = α * w_q)
    - Predictable utility scores in [0, 1] range
    
    **Economic Interpretation**:
    The ratio w_q/w_c represents "How much would I pay (in % cost) for 1% quality gain?"
    Examples:
    - w_q=0.99, w_c=0.01 → willing to pay 99¢ for 1¢ quality → extremely quality-sensitive
    - w_q=0.50, w_c=0.50 → willing to pay 50¢ for 1¢ quality → balanced
    """
    
    # Premium User: "Quality is paramount, cost is almost irrelevant"
    # w_q/w_c = 0.99/0.01 = 99 → willing to pay 99x more for 1% better quality
    MAX_QUALITY = {"w_q": 0.99, "w_c": 0.01, "w_l": 0.00}
    
    # Smart Shopper: "Flagship quality at reasonable cost"
    # w_q/w_c = 0.85/0.15 = 5.67 → willing to pay 5.67x more for 1% better quality
    ARBITRAGE = {"w_q": 0.85, "w_c": 0.15, "w_l": 0.00}
    
    # Balanced Default: "Solid trade-off between quality and cost"
    # w_q/w_c = 0.70/0.30 = 2.33 → willing to pay 2.33x more for 1% better quality
    BEST_VALUE = {"w_q": 0.70, "w_c": 0.30, "w_l": 0.00}
    
    # Budget User: "Cost matters more, acceptable quality drop for savings"
    # w_q/w_c = 0.40/0.60 = 0.67 → only willing to pay 0.67x more for 1% quality
    COST_SAVER = {"w_q": 0.40, "w_c": 0.60, "w_l": 0.00}
    
    # Real-time Applications: Speed is critical
    # w_l dominates, willing to sacrifice both quality and cost for speed
    LOW_LATENCY = {"w_q": 0.20, "w_c": 0.10, "w_l": 0.70}

    _PROFILES = {
        "max_quality": MAX_QUALITY,
        "arbitrage": ARBITRAGE,
        "best_value": BEST_VALUE,
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
        
        for m in valid_models:
            theta_hat = self.A_inv[m] @ self.b[m]
            # Sample weights from the posterior N(theta_hat, A_inv)
            samples = np.random.multivariate_normal(theta_hat, self.A_inv[m], n_samples)
            model_samples[m] = samples @ x
            
        if not model_samples: return {m: 0.0 for m in models}
        
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
            if self.gamma < 1.0:
                dt = self.t - self.last_update[model]
                # Clamp dt to prevent numerical underflow when gamma is small
                decay_factor = self.gamma ** min(dt, 1000)
                
                self.A[model] *= decay_factor
                self.b[model] *= decay_factor
                
                # Update timestamp after applying decay
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
                self.A[model] += self.init_lambda * np.eye(self.dim)
                # Must recompute inverse after manual regularization injection
                self.A_inv[model] = safe_inv(self.A[model])
                
                # Update b to preserve theta direction: b_new = A_new @ theta_old
                # This prevents "amnesia effect" where model forgets learned preferences
                self.b[model] = self.A[model] @ old_theta
            
            # Add observation: A += weight * x x^T, b += weight * reward * x
            self.A[model] += weight * np.outer(x, x)
            self.b[model] += weight * reward * x
            
            # Sherman-Morrison inverse update (O(d²))
            # Formula: (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
            A_inv = self.A_inv[model]
            u = x * np.sqrt(weight)
            v = x * np.sqrt(weight)
            
            A_inv_u = A_inv @ u
            v_A_inv = v @ A_inv
            denominator = 1.0 + (v @ A_inv_u)
            
            # KDD REVIEW FIX: Stricter safety floor (1e-6 instead of 1e-10)
            # Near-zero denominator indicates numerical instability in Sherman-Morrison
            if abs(denominator) > 1e-6:
                # Safe to use Sherman-Morrison formula
                self.A_inv[model] = A_inv - np.outer(A_inv_u, v_A_inv) / denominator
            else:
                # CRITICAL: Denominator too small, fallback to O(d³) with fresh regularization
                logger.warning(
                    f"⚠️ Sherman-Morrison near-singularity for {model}: "
                    f"|denominator|={abs(denominator):.2e} < 1e-6. "
                    f"Injecting fresh regularization and recomputing inverse."
                )
                # KDD OPTIMIZATION: Preserve Theta During Stability Reset
                # Capture learned preferences before regularization
                old_theta = self.A_inv[model] @ self.b[model]
                
                # Inject fresh regularization to restore conditioning
                self.A[model] += self.init_lambda * np.eye(self.dim)
                # Full O(d³) recomputation with regularized matrix
                self.A_inv[model] = safe_inv(self.A[model])
                
                # Update b to preserve theta direction: b_new = A_new @ theta_old
                # This prevents "amnesia effect" where model forgets learned preferences
                self.b[model] = self.A[model] @ old_theta
            
            # Global counter only (timestamp already updated above in decay block)
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
        alpha: float = 0.1,
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
        - FeatureService (The Eyes): Feature extraction
        - RouterCore (The Brain): LinUCB selection
        - FeedbackLoop (The Memory): Matrix updates
        
        Args:
            model_registry: Dictionary of model configurations
            feature_service: Optional FeatureService for custom feature extraction
                           If None, creates default service using context_model/pca_path
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
        self.model_priors: Dict[str, float] = {} 
        self.cluster_boost_weight = cluster_boost_weight
        
        # New Model Admission: Probation List
        # Stores model_id -> request_count_at_admission (or just boolean check in pruner)
        self.probation_models: Dict[str, Dict[str, Any]] = {} 
        # Feature name to index mapping for Progressive Registration
        self._feature_map = self._build_feature_map()


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
        
        Args:
            model_id: Unique model identifier
            capabilities: List of capability tags (coding, math, creative, reasoning, general)
            speed: Speed profile proxy for cost/HLE (fast, balanced, slow)
            cost_usd: Optional cost per 1M tokens (for registry metadata)
            latency_s: Optional median latency (for registry metadata)
            initial_weights: Optional explicit feature weights for power users
        
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
        
        # Fill the bias term (always last index)
        theta_vector[-1] = bias
        
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
                alpha=0.8  # 80% neighbor knowledge, 20% regularization
            )
            
            # Add arm with bootstrapped parameters
            self.bandit.models.append(model_id)
            self.bandit.A[model_id] = A_init
            self.bandit.b[model_id] = b_init
            self.bandit.A_inv[model_id] = safe_inv(A_init)
            self.bandit.last_update[model_id] = self.bandit.t
        else:
            # First model - use standard initialization
            self.bandit.add_arm(model_id)
        
        # 7. Override b vector to encode the prior (only if using Tier A/B/C knowledge)
        # If we bootstrapped from a neighbor, we DON'T want to overwrite with hand-coded priors
        # The neighbor's learned knowledge is more valuable than our guesses
        if len(self.bandit.models) == 1:  # Only for the very first model
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
        alpha: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap a new model's (A, b) from its nearest neighbor in embedding space.
        
        **KDD REVIEW FIX (Critique C)**: Sample Efficiency via Shared Priors
        
        With d=24 features, LinUCB needs ~240 samples (10×d) to learn stable parameters.
        For a 20-model registry, that's 5,000 requests just for warmup.
        
        This method addresses the cold-start problem by:
        1. Computing embedding similarity between model descriptions
        2. Finding the nearest neighbor among existing models
        3. Inheriting theta parameters from that neighbor (weighted by alpha)
        
        **Mathematical Justification:**
        If models A and B are semantically similar (e.g., both are coding specialists),
        then their ideal theta vectors should also be similar. By bootstrapping from
        a neighbor's learned parameters, we can reduce warmup time from 240 to ~50 samples.
        
        Args:
            model_id: The new model to initialize
            registry: Model registry with display_name metadata
            bandit: LinUCB policy with existing model parameters
            encoder: SentenceTransformer for computing similarity
            alpha: Mixing coefficient (0.0 = pure identity, 1.0 = pure neighbor)
                   Default 0.8 = 80% neighbor knowledge, 20% regularization
        
        Returns:
            Tuple of (A_bootstrapped, b_bootstrapped)
            
        Example:
            >>> # Adding a new coding model
            >>> A, b = admix_theta_from_neighbors(
            ...     "deepseek-coder",
            ...     registry,
            ...     bandit,
            ...     encoder,
            ...     alpha=0.8
            ... )
            # Result: Inherits 80% of theta from similar model (e.g., "codellama")
        """
        # Get model description for embedding
        model_info = registry.get(model_id, {})
        model_desc = model_info.get("display_name", model_id)
        
        # Compute embedding for new model
        try:
            new_embedding = encoder.encode([model_desc], convert_to_numpy=True)[0]
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
                neighbor_embedding = encoder.encode([neighbor_desc], convert_to_numpy=True)[0]
                
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
            # Mix neighbor's parameters with identity init
            A_neighbor = bandit.A[best_neighbor]
            b_neighbor = bandit.b[best_neighbor]
            
            A_identity = np.eye(bandit.dim) * bandit.init_lambda
            b_identity = np.zeros(bandit.dim, dtype=np.float64)
            
            # Weighted combination
            A_bootstrapped = alpha * A_neighbor + (1 - alpha) * A_identity
            b_bootstrapped = alpha * b_neighbor + (1 - alpha) * b_identity
            
            logger.info(
                f"✨ Bootstrapping {model_id} from neighbor {best_neighbor} "
                f"(similarity={best_similarity:.2f}, alpha={alpha})"
            )
            
            return A_bootstrapped, b_bootstrapped
        else:
            # No suitable neighbor, use identity
            logger.info(f"No suitable neighbor for {model_id} (best_sim={best_similarity:.2f}), using identity init")
            return (
                np.eye(bandit.dim) * bandit.init_lambda,
                np.zeros(bandit.dim, dtype=np.float64)
            )

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
            
        # Find the model with the maximum HLE score
        champion_id = max(
            self.registry,
            key=lambda m: self.registry[m].get("hle", 0.0) or 0.0
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
                "safe": 0.2,
                "balanced": 0.5,
                "aggressive": 1.0
            }
            alpha = exploration_map.get(exploration, 0.5)
            
        # 2. Pop arguments that shouldn't be passed to __init__
        state_path = kwargs.pop("state_path", None)
        prior_n_effective = kwargs.pop("prior_n_effective", 10.0)

        # 3. Initialize Router
        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            context_encoder=context_encoder,
            alpha=alpha,
            **kwargs
        )
        
        # 4. Apply Priors
        if priors == "hle":
            # Diagonal injection of benchmark scores
            for model_id in router.bandit.models:
                hle = router.registry.get(model_id, {}).get("hle", 0.15)
                # KDD Simplification: Only set prior on bias term (last dimension)
                router.bandit.b[model_id][-1] += (hle * prior_n_effective)
                
        elif priors == "warmup":
            # Load pre-computed matrices from disk
            priors_path = Path(__file__).parent.parent.parent / "data" / "priors_warmup.joblib"
            if priors_path.exists():
                import joblib
                warmup_data = joblib.load(priors_path)
                n_warmup = warmup_data.get("n", 20000)
                scale = prior_n_effective / float(n_warmup)
                
                for model_id in router.bandit.models:
                    if model_id in warmup_data["A"] and model_id in warmup_data["b"]:
                        router.bandit.A[model_id] = warmup_data["A"][model_id] * scale
                        router.bandit.b[model_id] = warmup_data["b"][model_id] * scale
                
                router.bandit.refresh_inverse_cache()
                
                # CRITICAL FIX: Add regularization after scaling to prevent numerical instability
                # When prior_n_effective is very small (e.g., 0.1), the scale factor (0.1/20000 = 5e-6)  
                # makes matrices extremely small, causing A_inv to explode.
                # Solution: Add init_lambda regularization to ensure matrices stay well-conditioned.
                for model_id in router.bandit.models:
                    router.bandit.A[model_id] += np.eye(router.bandit.dim) * router.bandit.init_lambda
                
                router.bandit.refresh_inverse_cache()
                logger.info(f"✅ Applied post-warmup regularization (λ={router.bandit.init_lambda}) for stability")
            else:
                logger.warning(f"Warmup priors not found at {priors_path}. Using cold start.")
        
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
            "hle": [],
            "context": []
        }
        
        for m_data in self.registry.values():
            stats["cost"].append(float(m_data.get("input_cost_per_m") or 0.0))
            stats["latency"].append(float(m_data.get("time_to_first_token_seconds") or 0.0))
            stats["hle"].append(float(m_data.get("hle") or 0.0))
            stats["context"].append(float(m_data.get("context_length") or 4096.0))
            
        def safe_stats(values):
            arr = np.array(values)
            return (float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)))
            
        return {
            "cost": safe_stats(stats["cost"]),
            "latency": safe_stats(stats["latency"]),
            "hle": safe_stats(stats["hle"]),
            "context": safe_stats(stats["context"])
        }

    def _vectorize_model_metadata(self, model_data: Dict[str, Any], global_stats: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
        """
        Create a static feature vector V for transfer learning.
        V = [Norm(Cost), Norm(Latency), Norm(HLE_Score), Context_Window_Log_Norm]
        """
        # Extract
        cost = float(model_data.get("input_cost_per_m") or 0.0)
        lat = float(model_data.get("time_to_first_token_seconds") or 0.0)
        hle = float(model_data.get("hle") or 0.0)
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
            normalize(hle, "hle"),
            normalize(ctx, "context", log=True)
        ])

    def _is_pareto_dominated(self, new_model_data: Dict[str, Any]) -> bool:
        """
        Phase 1: Optimizer Gatekeeper.
        Checks if the new model (with Optimistic Reward=1.0) is dominated by existing models
        across all major utility profiles (Quality, Cost, Latency).
        """
        # Hypothetical perfect score components
        # We calculate what the score WOULD be if Reward=1.0
        
        # 1. Normalize New Model inputs
        # We need the runtime normalization statistics (from active pool) used in select_arm
        # But here we can approximate using the global registry stats for "Admission"
        
        costs = [float(m.get("input_cost_per_m") or 0.0) for m in self.registry.values()]
        lats = [float(m.get("time_to_first_token_seconds") or 0.0) for m in self.registry.values()]
        
        min_c, max_c = min(costs), max(costs)
        min_l, max_l = min(lats), max(lats)
        
        def get_score(reward, cost, lat, profile):
            # Normalize Cost/Lat (Log-MinMax as per router logic)
            # Clip for safety
            cost = max(cost, 1e-9)
            lat = max(lat, 1e-9)
            
            # Log transform
            c_log = math.log(cost)
            l_log = math.log(lat)
            
            min_c_log, max_c_log = math.log(max(min_c, 1e-9)), math.log(max_c)
            min_l_log, max_l_log = math.log(max(min_l, 1e-9)), math.log(max_l)
            
            c_norm = (c_log - min_c_log) / (max_c_log - min_c_log) if max_c_log > min_c_log else 0.0
            l_norm = (l_log - min_l_log) / (max_l_log - min_l_log) if max_l_log > min_l_log else 0.0
            
            # Weight-based utility Score = (w_q * Quality) + (w_c * (1 - C)) + (w_l * (1 - L))
            w_q = profile.get("w_q", 1.0 - profile.get("w_c", 0.0) - profile.get("w_l", 0.0))
            w_c = profile.get("w_c", 0.0)
            w_l = profile.get("w_l", 0.0)
            
            return (w_q * reward) + (w_c * (1.0 - c_norm)) + (w_l * (1.0 - l_norm))

        new_cost = float(new_model_data.get("input_cost_per_m") or 0.0)
        new_lat = float(new_model_data.get("time_to_first_token_seconds") or 0.0)
        
        profiles = [
            OptimizationProfile.MAX_QUALITY,
            OptimizationProfile.ARBITRAGE,
            OptimizationProfile.BEST_VALUE,
            OptimizationProfile.COST_SAVER,
            OptimizationProfile.LOW_LATENCY
        ]
        
        # Check dominance
        is_useful_in_any_profile = False
        
        for profile in profiles:
            # Optimistic Score for New Model (Reward = 1.0)
            opt_score = get_score(1.0, new_cost, new_lat, profile)
            
            # Best Score among existing models (using their HLE directly)
            # No transformation - trust the bandit to learn from data
            best_existing = -float("inf")
            for m_id, m_data in self.registry.items():
                if m_id == new_model_data["openrouter_id"]: continue
                
                # Use raw HLE with slight positive bias (0.7) as simple prior
                m_hle = float(m_data.get("hle") or 0.15) * 0.7
                m_cost = float(m_data.get("input_cost_per_m") or 0.0)
                m_lat = float(m_data.get("time_to_first_token_seconds") or 0.0)
                
                score = get_score(m_hle, m_cost, m_lat, profile)
                if score > best_existing:
                    best_existing = score
            
            # If the new model (even with perfect score) can beat the best existing model
            # in THIS profile, then it is NOT dominated.
            if opt_score >= best_existing:
                is_useful_in_any_profile = True
                break
        
        return not is_useful_in_any_profile

    def admit_new_model(self, model_data: Dict[str, Any], dampening: float = 0.1) -> bool:
        """
        Validate and Initialize a new model using Ridge Regression Transfer.
        Returns True if admitted, False if rejected.
        """
        model_id = model_data["openrouter_id"]
        
        # 1. Update Registry temporarily to include new stats (or just use local var)
        # We need it in registry for stats calculation, but if we reject, we remove it.
        # Ideally check BEFORE adding.
        
        # Phase 1: Admission Check
        if self._is_pareto_dominated(model_data):
            logger.info(f"Refusing admission to {model_id}: Pareto Dominated (even with optimistic reward).")
            return False
            
        # Add to registry
        self.registry[model_id] = model_data
        
        # Phase 2: Initialization (Transfer)
        stats = self._calculate_global_stats()
        new_vec = self._vectorize_model_metadata(model_data, stats)
        
        # Find Neighbors
        candidates = []
        for m_id in self.bandit.models:
            if m_id == model_id: continue
            if m_id not in self.registry: continue
            
            m_vec = self._vectorize_model_metadata(self.registry[m_id], stats)
            dist = np.linalg.norm(new_vec - m_vec)
            candidates.append((dist, m_id))
            
        # Sort by distance
        candidates.sort(key=lambda x: x[0])
        neighbors = candidates[:3] # Top 3
        
        if not neighbors:
             # Fallback: Just add initialized (Identity)
             self.bandit.add_arm(model_id)
        else:
            # Average Matrices
            A_sum = np.zeros_like(self.bandit.A[neighbors[0][1]])
            b_sum = np.zeros_like(self.bandit.b[neighbors[0][1]])
            
            for _, n_id in neighbors:
                A_sum += self.bandit.A[n_id]
                b_sum += self.bandit.b[n_id]
                
            A_avg = A_sum / len(neighbors)
            b_avg = b_sum / len(neighbors)
            
            # Dampen (Inflate Variance)
            # A_new = eps * A_avg + (1-eps) * I * lambda
            lambda_reg = self.bandit.init_lambda
            identity = np.eye(A_avg.shape[0]) * lambda_reg
            
            A_new = (dampening * A_avg) + ((1.0 - dampening) * identity)
            b_new = dampening * b_avg
            
            # Register in Bandit
            self.bandit.add_arm(model_id)
            self.bandit.A[model_id] = A_new
            self.bandit.b[model_id] = b_new
            self.bandit.A_inv[model_id] = safe_inv(A_new)
            
            logger.info(f"Initialized {model_id} via transfer from: {[n[1] for n in neighbors]}")

        # Phase 3: Probation
        # Probation period = 500 requests
        # Rationale: ~5x the convergence window (estimated 100 requests for LinUCB)
        # This gives new models time to receive feedback before pruning evaluation.
        # Configurable via: self.probation_period (set in __init__ if needed)

        self.probation_models[model_id] = {
            "start_t": self.bandit.t,
            "status": "PROBATION",
            "immune_until": self.bandit.t + RouterConfig.probation_requests
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
        from collections import Counter
        counts = Counter(log.selected_model for log in self.logs)
        
        arms_to_count = arms if arms is not None else self.bandit.models
        return {arm: counts.get(arm, 0) for arm in arms_to_count}



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
    ) -> Tuple[float, float, float]:
        """
        Resolve optimization profile weights and apply orthogonal optimization.
        
        **Orthogonal Optimization:**
        If a hard constraint is active, disable the soft penalty for that dimension
        and re-allocate weight to Quality to avoid "Double Penalty".
        
        **KDD FIX (Jan 2026 - Phase 1):**
        Weights are normalized to sum to 1.0 to ensure:
        - Scale-invariant trade-off ratios (w_q=100, w_c=1 ≡ w_q=1, w_c=0.01)
        - Interpretable utility scores in [0, 1] range
        - Exploration bonus proportional to quality importance
        
        Args:
            profile: Optimization profile name
            max_cost: Hard cost constraint (optional)
            max_latency: Hard latency constraint (optional)
            
        Returns:
            Tuple of (w_q, w_c, w_l) weights (normalized to sum to 1.0)
        """
        weights = OptimizationProfile.get(profile).copy()
        w_q = weights.get("w_q", 1.0 - weights.get("w_c", 0.0) - weights.get("w_l", 0.0))
        w_c = weights.get("w_c", 0.0)
        w_l = weights.get("w_l", 0.0)
        
        # Orthogonal Optimization
        if max_cost is not None:
            w_q += w_c
            w_c = 0.0
        if max_latency is not None:
            w_q += w_l
            w_l = 0.0
        
        # KDD FIX: Normalize weights to sum to 1.0
        total = w_q + w_c + w_l
        if total > 1e-9:  # Avoid division by zero
            w_q = w_q / total
            w_c = w_c / total
            w_l = w_l / total
        else:
            # Degenerate case: default to quality-only
            logger.warning(f"Weight normalization failed (total={total}). Defaulting to pure quality.")
            w_q, w_c, w_l = 1.0, 0.0, 0.0
            
        return w_q, w_c, w_l
    

    
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
        for m in filtered:
            cost_per_1k = costs[m] * 1000  # Convert to $/1k tokens
            cost_penalties[m] = self._calculate_absolute_penalty(cost_per_1k)
        
        # Latency penalties (absolute)
        config = RouterConfig()
        LATENCY_FLOOR = config.market_latency_floor
        LATENCY_RANGE = config.latency_range_log
        
        latency_penalties = {}
        for m in filtered:
            safe_lat = max(lats[m], LATENCY_FLOOR)
            log_lat = np.log(safe_lat)
            norm_lat = (log_lat - np.log(LATENCY_FLOOR)) / LATENCY_RANGE
            latency_penalties[m] = max(0.0, min(1.0, norm_lat))
        
        return cost_penalties, latency_penalties
    
    def _score_candidates(
        self,
        filtered: List[str],
        x: np.ndarray,
        w_q: float,
        w_c: float,
        w_l: float,
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
          Exploration_Bonus = alpha * w_q * std_quality
          
        This ensures:
        1. Exploration only scales with quality importance (w_q).
        2. Cost signals are never drowned out in cost-sensitive profiles.
        3. The trade-off between mean quality and cost is perfectly linear.
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
            
            # Base Trade-off Utility (Deterministic)
            base_utility = (
                w_q * mean_quality + 
                w_c * (1.0 - norm_cost) + 
                w_l * (1.0 - norm_lat)
            )
            
            # Exploration Bonus (Proportional to w_q)
            exploration_bonus = self.bandit.alpha * w_q * std
            
            # Probation Bonus (Legacy support)
            probation_bonus = 0.0
            if self.config.probation_bonus > 0:
                count = sample_counts.get(m, 0)
                if count < self.config.pruning_min_samples:
                    decay = 1.0 - (count / self.config.pruning_min_samples)
                    probation_bonus = self.config.probation_bonus * w_q * decay
            
            # FINAL ADDITIVE UTILITY
            total_utility = base_utility + exploration_bonus + probation_bonus
            
            if self.verbose_routing:
                logger.info(f"Scoring {m:15s} | Utility: {total_utility:8.4f}")
                logger.info(f"  > [Base]   Q_util: {w_q*mean_quality:6.4f} (m={mean_quality:.3f}) | C_util: {w_c*(1-norm_cost):6.4f} | L_util: {w_l*(1-norm_lat):6.4f}")
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
        self.logs.append(log)
        
        # Save context for delayed feedback (RLHF, human ratings, etc.)
        self.context_store.save_context(log.request_id, x, model)
        
        return log

    def route(
        self,
        prompt: str | np.ndarray,
        *,
        profile: str | Dict[str, float] = "arbitrage",
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
        # Use profile_weights instead of profile here
        w_q, w_c, w_l = self._resolve_utility_weights(profile_weights, max_cost, max_latency)
        candidates = list(self.registry.keys())
        filtered = self._filter_by_constraints(
            candidates, prompt, max_cost, max_latency, quality_floor, input_tokens, output_tokens
        )
        
        # Estimate tokens for scoring
        in_tok = input_tokens or estimate_tokens_rough(prompt_text)
        
        best_model, best_utility, total_weight = self._score_candidates(
            filtered, x, w_q, w_c, w_l, in_tok, output_tokens
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
        # Find the routing log
        log = None
        for l in self.logs:
            if l.request_id == request_id:
                log = l
                break
        
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
        # Constants (Derived from Logarithmic Market Width)
        # Use config for consistency
        config = RouterConfig()
        MARKET_FLOOR_LOG = np.log(config.market_cost_floor)
        MARKET_RANGE = config.cost_range_log
        
        # Log Transform (clip to floor to avoid log(0))
        safe_cost = max(cost_per_1k, config.market_cost_floor)
        log_cost = math.log(safe_cost)
        
        # Normalize: (Current - Floor) / Range
        penalty = (log_cost - MARKET_FLOOR_LOG) / MARKET_RANGE
        
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
