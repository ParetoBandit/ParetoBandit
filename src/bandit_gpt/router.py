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

import copy
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

# Prevent tokenizers parallelism hangs (common with SentenceTransformers).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
# Canonical modules — re-exported here for backward compatibility so that
# ``from bandit_gpt.router import X`` continues to work for all consumers.
# ---------------------------------------------------------------------------
from bandit_gpt.policy import (  # noqa: F401 — re-exported
    DisjointLinUCBPolicy,
    BanditState,
    calibrate_priors,
    _SMUpdateResult,
    _argmax_random_tiebreak,
    _inflate_variance,
    _effective_staleness,
    _sherman_morrison_update,
    _safe_multivariate_normal,
    _MAX_VAR_INFLATION_FACTOR,
    _MAX_STALENESS_DT,
    _REGULARIZATION_FLOOR_FRACTION,
    _SM_DENOMINATOR_THRESHOLD,
    _OUTPUT_COST_MULTIPLIER,
)
from bandit_gpt.types import (  # noqa: F401 — re-exported
    RouterConfig,
    RegistrationConfig,
    ExplorationRate,
    RoutingLog,
    Capability,
    SpeedProfile,
)

from bandit_gpt.storage import ContextStore, EphemeralContextStore, SqliteContextStore
from bandit_gpt.utils import sigmoid, safe_inv, get_heuristic_prior

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception Classes (canonical definitions in bandit_gpt.exceptions)
# ---------------------------------------------------------------------------
from bandit_gpt.exceptions import (  # noqa: F401 — re-exported for backward compat
    MissingCostError,
    NoEligibleModelsError,
    NoModelScoredError,
)


from .config import DEFAULT_SENTENCE_TRANSFORMER

DEFAULT_CONTEXT_MODEL = DEFAULT_SENTENCE_TRANSFORMER

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
# Model Family Inference (canonical definitions in bandit_gpt.family)
# ---------------------------------------------------------------------------
from bandit_gpt.family import (  # noqa: F401 — re-exported for backward compat
    infer_model_family,
    tetrachoric_corr,
    compute_correlation_families,
)



# ---------------------------------------------------------------------------
# Main Router Class
# ---------------------------------------------------------------------------

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
        feature_service: 'FeatureService | None' = None,
        context_model: str = DEFAULT_CONTEXT_MODEL,
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
        budget_pacer: "BudgetPacer | None" = None,
    ):
        """Initialize BanditRouter with separated feature extraction.

        **Architectural Separation (Eyes, Brain, Memory):**
        - FeatureService (The Eyes): Feature extraction
        - RouterCore (The Brain): LinUCB selection
        - FeedbackLoop (The Memory): Matrix updates

        Args:
            model_registry: Dictionary of model configurations.
            feature_service: Optional FeatureService instance for custom feature
                extraction.  If *None*, a default service is created from
                *context_model* and *pca_path*.
            context_model: Encoder model name (used if feature_service=None).
            pca_path: Path to PCA model (used if feature_service=None).
            alpha: Exploration coefficient for UCB
            embedding_dim: Dimension override (auto-detected if feature_service provided)
            init_lambda: Initialization regularization (A₀ = λI)
            forgetting_factor: Temporal decay (1.0 = stationary)
            context_store: Storage backend for context vectors. Defaults to
                EphemeralContextStore (RAM-only, no disk). Pass
                SqliteContextStore() for persistence across restarts
                (required for delayed feedback / RLHF workflows).
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
            # Create default service from context_model / pca_path
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
        # Context Storage (opt-in persistence)
        # ---------------------------------------------------------------------------
        # Default: EphemeralContextStore (RAM-only, no disk I/O)
        # For delayed feedback / RLHF: pass SqliteContextStore() explicitly
        self.context_store = context_store or EphemeralContextStore()
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

        Resolves from available fields (alternate key names, input/output split,
        or pessimistic defaults).  Raises :class:`MissingCostError` for partial
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

        # --- Budget Pacer (deepcopy: has mutable EMA state) ---
        result.budget_pacer = copy.deepcopy(self.budget_pacer, memo)

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
        alpha = kwargs.pop("alpha", 0.1)

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
            else:
                logger.warning(f"[WARN] Priors file not found at {priors_path}. Using cold start.")

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
        """Generate synthetic prompts for PCA calibration.

        Delegates to :func:`bandit_gpt.utils.synthetic.generate_synthetic_prompts`.
        """
        from bandit_gpt.utils.synthetic import generate_synthetic_prompts

        return generate_synthetic_prompts(n)
    

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
        max_cost: float | None = None,
        max_latency: float | None = None,
        quality_floor: Dict[str, float | None] = None,
        input_tokens: int | None = None,
        output_tokens: int = 600,
    ) -> Tuple[str, RoutingLog]:
        """Route a prompt to the best model using LinUCB with cost/latency penalties.

        Raises:
            NoEligibleModelsError: If no models pass the hard constraints.

        Args:
            prompt: Input text or pre-embedded vector.
            max_cost: Hard budget ceiling in ``$/1k tokens``. Compared against
                each model's registry price (derived from ``blended_cost_per_m``).
            max_latency: Hard latency ceiling (seconds), compared against each
                model's ``time_to_first_token_seconds``.
            quality_floor: Minimum quality scores per metric (e.g.
                ``{"hle": 0.7}``).
            input_tokens: Input token count (auto-estimated if None).
            output_tokens: Expected output tokens (default 600).

        Returns:
            Tuple of (selected_model_id, routing_log).
        """
        # Build features and apply constraints
        x, prompt_text = self._build_routing_features(prompt)

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

        # Pre-compute normalized cost/latency once per candidate to avoid
        # redundant registry lookups + log() calls across pacer and penalty paths.
        need_cost = self.cost_penalty > 0 or (
            self.budget_pacer is not None and self.budget_pacer.uses_soft
        )
        need_latency = self.latency_penalty > 0
        norm_costs = (
            {m: self._get_normalized_cost(m) for m in filtered}
            if need_cost else {}
        )
        norm_latencies = (
            {m: self._get_normalized_latency(m) for m in filtered}
            if need_latency else {}
        )

        if self.budget_pacer is not None and self.budget_pacer.uses_soft:
            extra_cost_penalties = self.budget_pacer.get_extra_cost_penalties(
                norm_costs
            )

        # LinUCB selection with optional cost+latency penalty (paper Eq. 4)
        cp = None
        if self.cost_penalty > 0 or self.latency_penalty > 0:
            cp = {}
            for m in filtered:
                p = 0.0
                if self.cost_penalty > 0:
                    p += self.cost_penalty * norm_costs[m]
                if self.latency_penalty > 0:
                    p += self.latency_penalty * norm_latencies[m]
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
        the method falls back to the configured ``context_store``.  To support
        feedback arriving hours or days after routing, pass a
        ``SqliteContextStore`` when constructing the router.

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
            context, model_id = self.context_store.get_context(request_id)
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
        if model_id not in policy.theta:
            raise ValueError(
                f"Model {model_id} not found in bandit registry"
            )
        return policy.theta[model_id]

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

