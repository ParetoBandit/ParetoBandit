"""
Production-grade contextual bandit router (Hot Path).

Core Features:
1. Warmup Priors: Initializes with learned preferences from 80k battles.
2. Default Registry: Automatically loads 80+ models with cost data.
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
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical modules — re-exported here for backward compatibility so that
# ``from pareto_bandit.router import X`` continues to work for all consumers.
# ---------------------------------------------------------------------------
from pareto_bandit.policy import (  # noqa: F401 — re-exported
    DisjointLinUCBPolicy,
    BanditState,
    calibrate_priors,
    _OUTPUT_COST_MULTIPLIER,
)
from pareto_bandit.types import (  # noqa: F401 — re-exported
    RouterConfig,
    RegistrationConfig,
    ExplorationRate,
    RoutingLog,
    Capability,
    SpeedProfile,
)

from pareto_bandit.storage import ContextStore, EphemeralContextStore, SqliteContextStore
from pareto_bandit.utils import sigmoid, safe_inv, get_heuristic_prior

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception Classes (canonical definitions in pareto_bandit.exceptions)
# ---------------------------------------------------------------------------
from pareto_bandit.exceptions import (  # noqa: F401 — re-exported for backward compat
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
# Model Family Inference (canonical definitions in pareto_bandit.family)
# ---------------------------------------------------------------------------
from pareto_bandit.family import (  # noqa: F401 — re-exported for backward compat
    infer_model_family,
    tetrachoric_corr,
    compute_correlation_families,
)
from pareto_bandit.costs import log_normalize_cost



# ---------------------------------------------------------------------------
# Main Router Class
# ---------------------------------------------------------------------------

class BanditRouter:
    """Contextual bandit router that learns to select the best LLM per request.

    **Quick start with your own models** (no warmup priors needed)::

        registry = {
            "gpt-4o": {
                "input_cost_per_m": 2.50,
                "output_cost_per_m": 10.00,
                "time_to_first_token_seconds": 0.8,
            },
            "llama-3-70b": {
                "input_cost_per_m": 0.50,
                "output_cost_per_m": 0.50,
                "time_to_first_token_seconds": 0.3,
            },
        }
        router = BanditRouter.create(model_registry=registry, priors="none")
        model, log = router.route("Explain quantum computing")
        # ... get response, compute reward ...
        router.process_feedback(log.request_id, reward=0.85)

    The router supports three initialization modes via ``create()``:

    - ``priors="warmup"`` (default): Loads shipped offline priors for the
      K=3 paper portfolio.  Models not in the prior file receive heuristic
      initialization automatically.
    - ``priors="none"``: Clean cold-start with identity covariance.
      Recommended when using entirely custom model portfolios.
    - ``priors="path/to/custom.joblib"``: Load your own offline priors
      generated via :func:`pareto_bandit.generate_warmup_priors`.

    See :meth:`create` and :meth:`register_model` for full details.
    """
    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        feature_service: 'FeatureService | None' = None,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        pca_path: Path | str | None = None,
        # Bandit parameters (The Brain)
        alpha: float = 0.01,
        embedding_dim: int = 384,
        init_lambda: float = 1.0,
        forgetting_factor: float = 0.997,
        context_store: ContextStore | None = None,
        config: RouterConfig | None = None,
        verbose_routing: bool = False,
        cost_penalty: float = 0.3,  # λ_c for UCB cost penalty (paper Eq. 4)
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
        self.budget_pacer = budget_pacer

        if model_registry is None:
            # Load default models.json from config/
            base_dir = Path(__file__).parent
            models_path = base_dir / "config" / "models.json"
            if not models_path.exists():
                logger.warning("Default models.json not found at %s. Initializing with empty registry.", models_path)
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
            logger.info("Created default FeatureService with encoder=%s", context_model)
        
        # Calculate dimension dynamically from feature service
        # Default is 33 (32 PCA + 1 bias) with pca_32.joblib
        embedding_dim = self.features.dimension
        
        logger.debug("Feature dimensions: total=%d (including bias)", embedding_dim)
        
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
        
        # ---------------------------------------------------------------------------
        # Context Storage (opt-in persistence)
        # ---------------------------------------------------------------------------
        # Default: EphemeralContextStore (RAM-only, no disk I/O)
        # For delayed feedback / RLHF: pass SqliteContextStore() explicitly
        self.context_store = context_store or EphemeralContextStore()
        logger.info("Context store: %s", type(self.context_store).__name__)

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
        # Feature name to index mapping for Progressive Registration
        self._feature_map = self._build_feature_map()
        
        # Thread-local storage for per-thread overrides (e.g. exploit mode).
        # This avoids mutating shared bandit.alpha, which races with
        # concurrent route() calls on other threads.
        self._thread_local = threading.local()
        
        # Precompute market anchors to avoid redundant log calls in hot loop
        self._market_cost_floor = self.config.market_cost_floor
        self._market_cost_floor_log = np.log(self.config.market_cost_floor)
        self._market_cost_range = self.config.cost_range_log

    def update_model_pricing(self, model_id: str, **pricing_fields: float) -> None:
        """Update pricing for a model and recompute derived cost fields.

        Use this when simulating or reacting to mid-stream price changes
        (e.g., a provider price drop).  The method updates the specified
        fields, clears the stale ``blended_cost_per_m``, and re-resolves
        costs for the entire registry.

        Parameters
        ----------
        model_id : str
            Model identifier that must already exist in ``self.registry``.
        **pricing_fields : float
            Pricing fields to set (e.g., ``input_cost_per_m=0.10``,
            ``output_cost_per_m=0.10``).

        Raises
        ------
        KeyError
            If *model_id* is not in the registry.
        """
        if model_id not in self.registry:
            raise KeyError(f"Model '{model_id}' not in registry")
        entry = self.registry[model_id]
        for key, value in pricing_fields.items():
            entry[key] = value
        entry.pop("blended_cost_per_m", None)
        self._resolve_registry_costs()

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
        
        # --- Bandit Policy (deepcopy: has its own __deepcopy__ for locks) ---
        result.bandit = copy.deepcopy(self.bandit, memo)
        
        result.cost_penalty = self.cost_penalty
        
        # --- Logs and Counters (deepcopy: mutable collections) ---
        result.logs = copy.deepcopy(self.logs, memo)
        result.log_index = copy.deepcopy(self.log_index, memo)
        result._log_lock = threading.Lock()  # Fresh lock for clone
        result._thread_local = threading.local()  # Fresh thread-local for clone
        # --- Scalar / Immutable Settings (direct copy) ---
        result.verbose_routing = self.verbose_routing
        result._feature_map = copy.deepcopy(self._feature_map, memo)

        # --- Budget Pacer (deepcopy: has mutable EMA state) ---
        result.budget_pacer = copy.deepcopy(self.budget_pacer, memo)

        # --- Precomputed Market Anchors (scalars) ---
        result._market_cost_floor = self._market_cost_floor
        result._market_cost_floor_log = self._market_cost_floor_log
        result._market_cost_range = self._market_cost_range
        
        # --- Context Store ---
        # SqliteContextStore: SHARE (thread-safe DB connection, expensive to dup).
        # EphemeralContextStore: DEEPCOPY (unprotected dict/deque, racy if shared).
        if isinstance(self.context_store, EphemeralContextStore):
            result.context_store = copy.deepcopy(self.context_store, memo)
        else:
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
        cost_usd: float | None = None,
        latency_s: float | None = None,
        blended_cost_per_m: float | None = None,
        input_cost_per_m: float | None = None,
        output_cost_per_m: float | None = None,
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

        **Cost specification** (in order of precedence):

        1. ``input_cost_per_m`` + ``output_cost_per_m`` — exact per-token
           costs; blended cost is derived as their average.
        2. ``blended_cost_per_m`` — single blended rate; input/output are
           back-derived assuming the 3:1 output/input heuristic.
        3. ``cost_usd`` (legacy) — treated as input cost; output is estimated
           as ``cost_usd * 3``.
        4. None — falls back to ``RegistrationConfig.default_cost_per_1m``.

        Args:
            model_id: Unique model identifier
            speed: T-shirt speed profile ("fast", "balanced", "slow")
            cost_usd: *Deprecated — prefer* ``input_cost_per_m`` *+*
                ``output_cost_per_m``.  Input cost in $/M tokens.
            latency_s: Time-to-first-token in seconds
            blended_cost_per_m: Weighted average cost in $/M tokens for hard
                constraint filtering.
            input_cost_per_m: Input token cost in $/M tokens.
            output_cost_per_m: Output token cost in $/M tokens.
            initial_weights: Explicit feature weight overrides for power users
            strict_kwargs: Override for unknown-kwarg validation. If ``None``,
                uses ``RouterConfig.registration_strict_kwargs``.
            **kwargs: Accepted for backward compatibility (e.g. ``capabilities``).
                Unknown keys raise ``TypeError`` in strict mode.

        Raises:
            MissingCostError: If no cost information can be resolved.
            ValueError: If only one of ``input_cost_per_m`` /
                ``output_cost_per_m`` is provided without the other.

        Examples:
            # Exact pricing (preferred)
            router.register_model("gpt-4o", speed="balanced",
                                  input_cost_per_m=2.50,
                                  output_cost_per_m=10.00)

            # Single blended rate
            router.register_model("llama-3-8b", speed="fast",
                                  blended_cost_per_m=0.2)

            # Mystery model: No information (pessimistic defaults)
            router.register_model("model-x")
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

        # Validate: both or neither of input/output must be provided
        _have_input = input_cost_per_m is not None
        _have_output = output_cost_per_m is not None
        if _have_input != _have_output:
            raise ValueError(
                f"register_model('{model_id}'): provide both input_cost_per_m "
                f"and output_cost_per_m, or neither. "
                f"Got input_cost_per_m={input_cost_per_m}, "
                f"output_cost_per_m={output_cost_per_m}."
            )

        capabilities = kwargs.get("capabilities", [])
            
        if model_id in self.bandit.models:
            logger.warning("Model '%s' already registered. Skipping.", model_id)
            return
        
        # 1. Build initial theta vector from T-shirt sizing + overrides
        weights = {}
        reg_config = self.config.registration

        if speed == "fast":
            bias = reg_config.fast_bias
        elif speed == "slow":
            bias = reg_config.slow_bias
        else:
            bias = reg_config.balanced_bias

        if initial_weights:
            for k, v in initial_weights.items():
                weights[k] = v

        # 2. Compile theta vector
        dim = self.bandit.dim
        theta_vector = np.zeros(dim, dtype=np.float64)
        theta_vector[self.features.bias_index] = bias
        for feature_name, val in weights.items():
            if feature_name in self._feature_map:
                theta_vector[self._feature_map[feature_name]] = val
            else:
                logger.warning("Unknown feature '%s' in initial_weights. Skipping.", feature_name)

        # 3. Initialize bandit arm: A = λI, b = λ·θ
        self.bandit.add_arm(model_id)
        new_b = self.bandit.init_lambda * theta_vector
        with self.bandit._lock:
            self.bandit.b[model_id] = new_b
            self.bandit._refresh_theta(model_id)

        # 4. Resolve costs (precedence: explicit pair > blended > cost_usd > default)
        if latency_s is None:
            latency_s = reg_config.default_latency_s

        if input_cost_per_m is not None and output_cost_per_m is not None:
            # Tier 1: Exact per-token costs provided
            final_input = float(input_cost_per_m)
            final_output = float(output_cost_per_m)
            final_blended = (final_input + final_output) / 2.0
        elif blended_cost_per_m is not None:
            # Tier 2: Single blended rate; back-derive input/output
            final_blended = float(blended_cost_per_m)
            if cost_usd is not None:
                final_input = float(cost_usd)
                final_output = 2.0 * final_blended - final_input
            else:
                final_input = final_blended
                final_output = final_blended
        elif cost_usd is not None:
            # Tier 3: Legacy cost_usd (input price); estimate output via multiplier
            final_input = float(cost_usd)
            final_output = final_input * _OUTPUT_COST_MULTIPLIER
            final_blended = (final_input + final_output) / 2.0
        else:
            # Tier 4: No cost info — pessimistic defaults
            final_input = reg_config.default_cost_per_1m
            final_output = reg_config.default_cost_per_1m * _OUTPUT_COST_MULTIPLIER
            final_blended = (final_input + final_output) / 2.0

        registry_entry = {
            "cost_per_1m_tokens": final_input,
            "input_cost_per_m": final_input,
            "output_cost_per_m": final_output,
            "blended_cost_per_m": final_blended,
            "time_to_first_token_seconds": latency_s,
            "median_latency_s": latency_s,
            "capabilities": capabilities,
            "speed_profile": speed,
        }
        
        self.registry[model_id] = registry_entry
        
        boost_summary = ", ".join(f"{k}={v:.1f}" for k, v in list(weights.items())[:5])
        if len(weights) > 5:
            boost_summary += "..."
        
        logger.info(
            "Registered %s | Bias: %.1f | Boosts: %s | "
            "Cost: in=$%.2f out=$%.2f blend=$%.2f /1M | Latency: %.2fs",
            model_id, bias, boost_summary,
            final_input, final_output, final_blended, latency_s,
        )


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
        prior_n_effective: float = 1163.9,
        **kwargs
    ) -> "BanditRouter":
        """Factory method to create a fully initialized router.

        Args:
            model_registry: Dictionary of model configurations.  Each key is
                a model ID; each value is a dict with cost/latency metadata.
                Required keys: ``input_cost_per_m``, ``output_cost_per_m``.
                Optional: ``time_to_first_token_seconds``, ``speed_profile``,
                ``initial_quality``, ``capabilities``.
                When ``None``, loads the shipped K=3 paper portfolio from
                ``config/models.json``.
            context_model: Model to use for embedding generation.
            priors: Prior initialization strategy:

                - ``"warmup"`` (default): Loads shipped K=3 warmup priors.
                  Models not in the prior file receive heuristic
                  initialization based on ``initial_quality``.
                - ``"none"``: Clean cold-start (identity covariance +
                  quality-based bias).  Recommended for custom portfolios.
                - ``"path/to/priors.joblib"``: Load custom offline priors
                  generated via :func:`generate_warmup_priors`.

            prior_n_effective: Effective sample count attributed to loaded
                priors.  Controls how strongly the offline priors are trusted:
                ``scale = prior_n_effective / A[-1,-1]`` where ``A[-1,-1]``
                is the total precision mass in the bias direction of the
                warmup precision matrix (``lambda + sum(weights)``), not the
                raw number of training samples.  Default 1163.9, derived from
                the T_adapt-constrained Pareto knee-point selection
                (Experiment 05) with ``gamma=0.997`` and ``T_adapt=500``
                via ``n_eff = (gamma^{-T_adapt} - 1) / (1 - gamma)``.
                Higher values trust priors more (slower adaptation); lower
                values trust them less (faster override by online evidence).
            **kwargs: Additional arguments passed to __init__ or prior loading.
                Notable: ``config`` (:class:`RouterConfig`) to customise
                reward range, cost anchors, etc.
        
        Returns:
            Fully initialized BanditRouter instance

        Examples:
            Bring your own models (BYOM) — cold-start with custom fleet::

                registry = {
                    "gpt-4o": {
                        "input_cost_per_m": 2.50,
                        "output_cost_per_m": 10.00,
                        "time_to_first_token_seconds": 0.8,
                    },
                    "claude-sonnet": {
                        "input_cost_per_m": 3.00,
                        "output_cost_per_m": 15.00,
                    },
                }
                router = BanditRouter.create(
                    model_registry=registry,
                    priors="none",
                )

            Custom reward scale (e.g. preference pairs in [-1, 1])::

                from pareto_bandit.types import RouterConfig
                router = BanditRouter.create(
                    model_registry=registry,
                    priors="none",
                    config=RouterConfig(reward_min=-1.0, reward_max=1.0),
                )

            Add a model at runtime::

                router.register_model(
                    "deepseek-v3",
                    input_cost_per_m=0.27,
                    output_cost_per_m=1.10,
                    speed="fast",
                )
        """
        # 1. Extract factory-specific arguments (not passed to __init__)
        state_path = kwargs.pop("state_path", None)
        warmup_path = kwargs.pop("warmup_path", None)
        alpha = kwargs.pop("alpha", 0.01)

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

        # 2. Filter kwargs to only those accepted by __init__
        _INIT_PARAMS = {
            "feature_service", "context_model", "pca_path", "alpha",
            "embedding_dim", "init_lambda", "forgetting_factor",
            "context_store", "config", "verbose_routing", "cost_penalty",
            "budget_pacer",
        }
        unknown = set(kwargs) - _INIT_PARAMS
        if unknown:
            logger.warning(
                "create() ignoring unknown kwargs not accepted by __init__: %s",
                sorted(unknown),
            )
        init_kwargs = {k: v for k, v in kwargs.items() if k in _INIT_PARAMS}

        # 3. Initialize the Router (Standard)
        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=alpha,
            **init_kwargs
        )
        
        # 4. Resolve priors path (shipped default or explicit)
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

        priors_path_str = warmup_path or (priors if priors != "none" else None)
        if isinstance(priors_path_str, str) and (
            priors_path_str.endswith(".joblib") or "/" in priors_path_str
        ):
            priors_path = Path(priors_path_str)
            if priors_path.exists():
                cls._load_warmup_priors(router, priors_path, prior_n_effective)
            else:
                logger.warning(
                    "Priors file not found at %s. Using cold start.", priors_path
                )

        # 5. T-shirt sizing injection on top of warmup priors
        cls._inject_tshirt_biases(router)

        # 6. Calibrate priors (catches scale explosion before the adapter sees it)
        router.bandit.refresh_inverse_cache()
        calibrate_priors(router.bandit, target_max_pred=0.9)

        # 7. Load state if provided (overwrites any priors applied above)
        if state_path:
            router.load_state(state_path)

        return router

    # ------------------------------------------------------------------
    # Factory helpers (extracted from create() for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def _align_whitening(
        warmup_data: Dict[str, Any],
        features: Any,
    ) -> Dict[str, Any]:
        """Align warmup priors to the router's PCA whitening convention.

        For diagonal whitening ``x_new = D x_old`` the sufficient statistics
        transform as ``A_new = D A_old D``, ``b_new = D b_old``.  This
        conversion is applied once at load time so older prior artifacts
        remain usable without silent scale mismatch.

        Args:
            warmup_data: Loaded warmup dictionary (mutated in-place).
            features: FeatureService instance (checked for
                ``get_pca_whitening_scales``).

        Returns:
            The (possibly modified) warmup_data dict.
        """
        scales = None
        if hasattr(features, "get_pca_whitening_scales"):
            scales = np.asarray(features.get_pca_whitening_scales(), dtype=np.float64)

        router_whitens = False
        if scales is not None and scales.shape[0] >= 2:
            router_whitens = not np.allclose(scales[:-1], 1.0)

        priors_whitened = bool(warmup_data.get("pca_whitened", False))
        if scales is None or priors_whitened == router_whitens:
            return warmup_data

        if priors_whitened and not router_whitens:
            scales = 1.0 / np.maximum(scales, 1e-12)

        warmup_data = dict(warmup_data)
        A_map = warmup_data.get("A", {})
        b_map = warmup_data.get("b", {})
        if not (isinstance(A_map, dict) and isinstance(b_map, dict)):
            return warmup_data

        A_new: Dict[str, np.ndarray] = {}
        b_new: Dict[str, np.ndarray] = {}
        for m in A_map:
            if m not in b_map:
                continue
            A_m = np.asarray(A_map[m], dtype=np.float64)
            b_m = np.asarray(b_map[m], dtype=np.float64)
            d = scales.shape[0]
            A_new[m] = (
                A_m * scales.reshape(-1, 1) * scales.reshape(1, -1)
                if A_m.shape == (d, d) else A_m
            )
            b_new[m] = b_m * scales if b_m.shape[0] == d else b_m

        warmup_data["A"] = A_new
        warmup_data["b"] = b_new
        warmup_data["pca_whitened"] = router_whitens
        logger.info(
            "Converted warmup priors PCA whitening: "
            "priors_whitened=%s -> router_whitens=%s",
            priors_whitened, router_whitens,
        )
        return warmup_data

    @staticmethod
    def _load_warmup_priors(
        router: "BanditRouter",
        priors_path: Path,
        prior_n_effective: float,
    ) -> None:
        """Load offline warmup priors, align whitening, and regularize.

        Mutates ``router.bandit`` in place.  Models present in the warmup
        file receive scaled offline priors; missing models fall back to
        heuristic initialization based on registry metadata.

        Post-warmup regularization (``A += lambda * I`` without adjusting
        ``b``) implements Bayesian shrinkage toward zero — a safety valve
        against mismatched priors from a different traffic distribution.

        Args:
            router: Partially initialized router (bandit + registry ready).
            priors_path: Path to a ``.joblib`` warmup priors artifact.
            prior_n_effective: Effective sample count attributed to priors.
        """
        import joblib as _joblib

        warmup_data = _joblib.load(priors_path)

        try:
            warmup_data = BanditRouter._align_whitening(warmup_data, router.features)
        except (KeyError, TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
            logger.warning(
                "Warmup priors whitening conversion failed: %s. "
                "Proceeding without conversion (may degrade performance).",
                exc,
            )

        missing_models: List[str] = []
        for model_id in router.bandit.models:
            if (model_id in warmup_data.get("A", {})) and (model_id in warmup_data.get("b", {})):
                # A[-1,-1] gives the total precision weight in the bias
                # direction: lambda + sum(w_i) (regularization + data).
                # This is a proxy for prior strength, not a raw observation
                # count — it automatically accounts for importance weights
                # and regularization used during warmup generation.
                warmup_eff_strength = warmup_data["A"][model_id][-1, -1]
                scale = prior_n_effective / max(float(warmup_eff_strength), 1.0)
                
                # Original matrices
                A_orig = warmup_data["A"][model_id]
                b_orig = warmup_data["b"][model_id]
                
                # Scale data covariance and target
                A_scaled = A_orig * scale
                b_scaled = b_orig * scale
                
                # True prior mean (from offline data)
                # We use safe_inv because A might be singular if n_warmup is small or data is degenerate
                try:
                    theta_true = np.linalg.solve(A_orig, b_orig)
                except np.linalg.LinAlgError:
                    theta_true = np.linalg.pinv(A_orig) @ b_orig
                    
                router.bandit.A[model_id] = A_scaled
                
                # We will add init_lambda * I to A below, so we must add init_lambda * theta_true to b 
                # to prevent Bayesian shrinkage from pulling the mean to zero.
                # The user expects the confidence (n_eff) to change the variance, not the mean!
                router.bandit.b[model_id] = b_scaled + router.bandit.init_lambda * theta_true
                
            else:
                missing_models.append(model_id)
                A_h, b_h = get_heuristic_prior(
                    model_data=router.registry.get(model_id, {}),
                    dim=router.bandit.dim,
                    init_lambda=router.bandit.init_lambda,
                    n_effective=prior_n_effective,
                )
                router.bandit.A[model_id] = A_h
                router.bandit.b[model_id] = b_h

        if missing_models:
            logger.warning(
                "Warmup partial miss: %d models not in joblib. "
                "Applied heuristic initialization for: %s",
                len(missing_models), missing_models,
            )
        else:
            logger.info("Warmup complete: all models initialized from offline priors.")

        # Post-warmup regularization: A += lambda*I
        # We explicitly preserve the prior mean (via b_scaled + lambda * theta_true)
        # to ensure that quality differences remain properly scaled against the cost penalty.
        # Adding lambda*I guarantees numerical stability and smooths out extreme
        # singular values from degenerate offline data.
        reg_eye = np.eye(router.bandit.dim) * router.bandit.init_lambda
        for model_id in router.bandit.models:
            router.bandit.A[model_id] += reg_eye

        router.bandit.refresh_inverse_cache()
        logger.info(
            "Applied post-warmup regularization (lambda=%.2f) from %s",
            router.bandit.init_lambda, priors_path,
        )

    @staticmethod
    def _inject_tshirt_biases(router: "BanditRouter") -> None:
        """Inject T-shirt sizing speed-profile biases into warmed-up state.

        Applies human-provided speed profile priors (fast/slow) *on top* of
        data-driven warmup priors with proper confidence scaling.  To shift
        theta by ``delta`` in the bias dimension ``i``, the correct update is
        ``b += delta * A[:, i]`` (the full i-th column of A), which accounts
        for off-diagonal covariance.

        Args:
            router: Router instance with bandit and registry already initialized.
        """
        reg_config = router.config.registration
        bias_idx = router.features.bias_index

        for model_id in router.bandit.models:
            speed = router.registry.get(model_id, {}).get("speed_profile", "balanced")

            if speed == "fast":
                bias_shift = reg_config.fast_bias
            elif speed == "slow":
                bias_shift = reg_config.slow_bias
            else:
                bias_shift = 0.0

            if abs(bias_shift) > 0.0:
                injection_col = bias_shift * router.bandit.A[model_id][:, bias_idx]
                router.bandit.b[model_id] += injection_col
                logger.debug(
                    "  %s (%s): bias_shift=%.2f, ||injection||=%.2f",
                    model_id, speed, bias_shift, np.linalg.norm(injection_col),
                )


    # -------------------------------------------------------------------------
    # Constraint Filtering and Logging Helpers
    # -------------------------------------------------------------------------

    def _filter_by_constraints(
        self,
        candidates: List[str],
        max_cost: float | None,
        max_latency: float | None,
        quality_floor: Dict[str, float | None] | None,
    ) -> List[str]:
        """
        Apply hard constraints (cost, latency, quality floor).

        Cost filtering interprets ``max_cost`` as a **unit price ceiling** in
        ``$/1k tokens`` (as documented in the README). Each model's
        ``blended_cost_per_m`` (stored in ``$/M``) is converted to ``$/1k`` by
        dividing by 1000.

        Design note: the filter uses a static blended rate rather than a
        token-weighted estimate because ``max_cost`` is a per-model *tier*
        gate ("exclude models above price X"), not a per-request budget cap.
        Log-scale normalization preserves model cost ranking across any
        reasonable input:output ratio.  For per-request budget enforcement,
        the ``BudgetPacer`` uses exact observed costs from the billing loop.

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
        quality_floor: Dict[str, float | None] | None = None,
        input_tokens: int | None = None,
        output_tokens: int = 600,
    ) -> Tuple[str, RoutingLog]:
        """Route a prompt to the best model using LinUCB with cost penalties.

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
        # Extract features and derive prompt text for logging
        prompt_text = prompt if isinstance(prompt, str) else "[Pre-embedded Prompt]"
        x = self._get_context_vector(prompt)

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

        # Pre-compute normalized cost once per candidate to avoid
        # redundant registry lookups + log() calls across pacer and penalty paths.
        need_cost = self.cost_penalty > 0 or (
            self.budget_pacer is not None and self.budget_pacer.uses_soft
        )
        norm_costs = (
            {m: self._get_normalized_cost(m) for m in filtered}
            if need_cost else {}
        )

        if self.budget_pacer is not None and self.budget_pacer.uses_soft:
            extra_cost_penalties = self.budget_pacer.get_extra_cost_penalties(
                norm_costs
            )

        # LinUCB selection with optional cost penalty (paper Eq. 4)
        cp = None
        if self.cost_penalty > 0:
            cp = {m: self.cost_penalty * norm_costs[m] for m in filtered}
        if extra_cost_penalties is not None:
            cp = cp or {}
            for m, pen in extra_cost_penalties.items():
                cp[m] = cp.get(m, 0.0) + pen
        alpha_override = getattr(self._thread_local, "alpha_override", None)
        best_model, best_utility = self.bandit.select_arm(
            x, candidates=filtered, cost_penalties=cp,
            alpha_override=alpha_override,
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
                    (see ``pareto_bandit.providers``).
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
        # O(1) lookup via parallel index instead of O(N) linear scan.
        with self._log_lock:
            log = self.log_index.get(request_id)
        
        # Fallback to context_store for delayed feedback (RLHF)
        if log is None:
            context, model_id = self.context_store.get_context(request_id)
            if context is None:
                logger.warning("Context not found for request_id=%s", request_id)
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

        r_min, r_max = self.config.reward_min, self.config.reward_max
        if reward < r_min or reward > r_max:
            logger.warning(
                "process_feedback: reward=%.4f outside [%.4f, %.4f] for "
                "request_id=%s; clamping. Adjust RouterConfig.reward_min / "
                "reward_max if your reward scale differs.",
                reward, r_min, r_max, request_id,
            )
        reward = float(np.clip(reward, r_min, r_max))

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
        model and do **not** incorporate cost penalties.  The actual routing
        decision (``route()``) optimises a composite utility
        ``UCB - λ_c·cost``; this method answers the narrower question
        "which model is most likely to produce the highest-quality
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
        r_min, r_max = self.config.reward_min, self.config.reward_max
        if reward < r_min or reward > r_max:
            logger.warning(
                "update: reward=%.4f outside [%.4f, %.4f] for model '%s'; "
                "clamping. Adjust RouterConfig.reward_min / reward_max if "
                "your reward scale differs.",
                reward, r_min, r_max, model_id,
            )
        reward = float(np.clip(reward, r_min, r_max))
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
        """Decompose the **mean reward prediction** ``θ^T x`` into per-feature contributions.

        **Scope — mean prediction only.**  This method decomposes ``θ^T x``
        (the learned quality estimate).  It does **not** include the UCB
        exploration bonus (``α √(x^T A⁻¹ x)``) or cost penalties
        (``−λ_c · norm_cost``), both of which affect the actual routing
        decision via ``select_arm()``.  During cold start or high-exploration
        phases, the exploration bonus may dominate the mean; during
        budget-constrained routing, the cost penalty may be decisive.  Use
        this method to understand the *learned quality signal* for a model,
        not to fully explain why a particular model was selected.

        Args:
            model_id: The model to explain (e.g., "claude-opus").
            context_vector: The context vector for the prompt.
            threshold: Minimum absolute contribution to include (default 0.01).
                Filters out noise from features with negligible impact.

        Returns:
            Dictionary mapping feature names to their contribution scores,
            sorted by absolute contribution (highest to lowest).

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
        """Decompose the mean reward prediction for the top-*k* models by ``θ^T x``.

        Convenience wrapper that extracts the context vector and returns
        per-feature contributions for the models with the highest learned
        quality estimate.

        **Scope — mean prediction only.**  Rankings are by ``θ^T x`` (the
        learned quality estimate) and do not reflect the exploration bonus
        or cost penalties used by the actual routing decision.  See
        :meth:`explain_decision` for details on what is and is not included.

        Args:
            prompt: Input prompt text.
            top_k: Number of top models to explain (default 3).
            threshold: Minimum absolute contribution to include (default 0.01).

        Returns:
            Dictionary mapping model_id to per-feature contribution dict.

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
        model_scores.sort(key=lambda item: item[1], reverse=True)
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

        Within this block, ``route()`` selects ``argmax(theta^T x)`` with no
        UCB exploration bonus (alpha=0).

        **Thread safety:** Uses ``threading.local()`` so concurrent threads
        retain their own alpha.  Unlike the previous implementation that
        mutated ``bandit.alpha`` (shared across threads), this version is
        safe for multi-threaded deployments (e.g. FastAPI, gunicorn).

        Usage::

            with router.exploit():
                model, log = router.route(x)
        """
        self._thread_local.alpha_override = 0.0
        try:
            yield
        finally:
            self._thread_local.alpha_override = None

    def _calculate_absolute_penalty(self, cost_per_1k: float) -> float:
        """Stable 0.0-1.0 cost penalty via logarithmic market anchors.

        Delegates to :func:`pareto_bandit.costs.log_normalize_cost` — the
        canonical implementation shared with offline evaluation baselines.
        Anchors are read from ``self.config`` (single source of truth).

        Args:
            cost_per_1k: Cost in dollars per 1000 tokens.

        Returns:
            Penalty in [0.0, 1.0].
        """
        return log_normalize_cost(
            cost_per_1k,
            floor=self.config.market_cost_floor,
            ceiling=self.config.market_cost_ceiling,
        )

    def _get_normalized_cost(self, model_id: str) -> float:
        """Compute normalized [0, 1] cost for a model from registry metadata.

        Uses a static 1:1 blend of input/output rates rather than weighting
        by the current request's token counts.  This is intentional: the UCB
        penalty is a *model-level* bias term (paper Eq. 4), not a per-request
        cost estimate.  Log-scale normalization ensures the cost ranking is
        preserved for any plausible token ratio; the bandit's reward model
        corrects any residual mis-pricing through online learning.  Exact
        per-request costs are handled by ``_estimate_cost`` and the
        ``BudgetPacer`` feedback loop.

        Note: the 1:1 blend here differs from the 1:3 ratio used in
        ``_estimate_cost()`` (via ``_OUTPUT_COST_MULTIPLIER``) and in
        ``register_model()``.  The distinction is intentional — this method
        produces a model-level *ranking signal* for the UCB penalty, while
        ``_estimate_cost()`` produces a per-request *dollar amount* for
        budget tracking.  See ``_OUTPUT_COST_MULTIPLIER`` in
        ``pareto_bandit.policy`` for the output-cost heuristic.
        """
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

