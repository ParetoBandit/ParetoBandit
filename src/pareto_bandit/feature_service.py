"""
Feature Service: The Eyes of the BanditRouter.

Handles all feature extraction logic independently from the LinUCB math.
This separation allows iterating on feature engineering (regex, PCA, encoders)
without risking breaking the router core.

Three embedding paths are supported (in order of priority):

1. **Pre-computed vectors** — ``FeatureService.for_precomputed(dim)``
   No encoder or PCA loaded; pass ``np.ndarray`` directly.

2. **Custom encoder callable** — ``FeatureService(custom_encoder=fn)``
   Any ``Callable[[str], np.ndarray]`` that maps text → 1-D float array.
   Paired with an optional PCA artifact for dimensionality reduction.
   Does *not* require ``sentence-transformers``.

3. **SentenceTransformer encoder** (default) — ``FeatureService()``
   Requires the ``sentence-transformers`` package
   (``pip install paretobandit[embeddings]``).

**Optional text features** — ``FeatureService(use_text_features=True)``
appends four z-score normalized, regex-based features (logical operator
count, constraint keyword count, average word length, instruction×vague
density) between the embedding/PCA components and the bias term.  These complement the
semantic PCA dimensions with structural signals that are empirically
predictive of model-arm reward gaps (see ``experiments/eda_pareto_features.py``).
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from .config import DEFAULT_SENTENCE_TRANSFORMER  # noqa: E402

# Default context model
DEFAULT_CONTEXT_MODEL = DEFAULT_SENTENCE_TRANSFORMER

# Sentinels for lazy initialization
_ENCODER_NOT_LOADED = object()
_PCA_NOT_LOADED = object()


# Maximum prompt length to prevent OOM on very long inputs
MAX_PROMPT_LENGTH = 50000  # ~12k tokens


# ---------------------------------------------------------------------------
# Text feature extraction (lightweight regex-based features)
# ---------------------------------------------------------------------------

_LOGICAL_OPS_RE = re.compile(
    r"\b(if|then|and|or|not|but|unless|given that|only if|"
    r"provided that|neither|nor|therefore|hence|however|"
    r"although|whereas|assuming|suppose)\b",
    re.IGNORECASE,
)

_CONSTRAINT_RE = re.compile(
    r"\b(must|ensure|exactly|at least|at most|no more than|strictly|"
    r"required|do not|don't|never|always|make sure|constraint|"
    r"limit|restrict)\b",
    re.IGNORECASE,
)

# Vague/subjective adjectives that signal ambiguity (prompts that are both
# complex and ambiguous may benefit from stronger models).
_VAGUE_ADJ_RE = re.compile(
    r"\b(various|several|many|some|certain|numerous|few|appropriate|"
    r"relevant|suitable|reasonable|significant|substantial|different|"
    r"similar|general|specific|typical|average|normal|possible|likely|"
    r"good|bad|nice|proper|adequate|effective|efficient|clear|simple|"
    r"complex|basic|advanced|best|worst|better|worse)\b",
    re.IGNORECASE,
)

TEXT_FEATURE_NAMES: list[str] = [
    "n_logical_ops",
    "n_constraints",
    "avg_word_len",
    "instruction_x_vague_density",
]
N_TEXT_FEATURES: int = len(TEXT_FEATURE_NAMES)

# Z-score normalization constants derived from the pareto dataset
# (N=11,983 prompts across 13 public benchmarks).  These produce
# features with approximately zero mean and unit variance, matching
# the whitened PCA feature scale (~0.76 std per component).
# The 4th feature (instruction_x_vague_density) uses a conservative
# mean/std since the interaction term is sparse; tune with calibration.
_TEXT_FEATURE_MEANS = np.array([1.8188, 0.0562, 4.7897, 0.02], dtype=np.float64)
_TEXT_FEATURE_STDS = np.array([2.2717, 0.2932, 0.6948, 0.08], dtype=np.float64)

# Clipping bound for z-scored text features.  Prevents extreme
# outliers (e.g. a prompt with 20 logical operators) from producing
# values >10 that would destabilize LinUCB's matrix inverse.
_TEXT_FEATURE_CLIP = 3.0


def extract_text_features(prompt: str) -> np.ndarray:
    """Extract lightweight text-derived features from a prompt.

    Returns a 1-D array of shape ``(N_TEXT_FEATURES,)`` with values
    z-score normalized and clipped to ``[-3, 3]`` for compatibility
    with the whitened PCA features used by LinUCB.

    Features
    --------
    0. ``n_logical_ops``  — count of logical connectives (if, then, and, or,
       not, but, unless, ...), z-scored.
    1. ``n_constraints``  — count of constraint/instruction keywords (must,
       ensure, exactly, at least, ...), z-scored.
    2. ``avg_word_len``   — mean word length in characters, z-scored.
    3. ``instruction_x_vague_density`` — Instruction_Count * Vague_Adjective
       Density. Identifies prompts that are both complex and ambiguous.

    Args:
        prompt: Raw prompt string.

    Returns:
        1-D ``np.ndarray`` of shape ``(4,)``.
    """
    n_logical = len(_LOGICAL_OPS_RE.findall(prompt))
    n_constraints = len(_CONSTRAINT_RE.findall(prompt))
    words = prompt.split()
    n_words = max(len(words), 1)
    avg_wl = float(np.mean([len(w) for w in words])) if words else 0.0
    n_vague = len(_VAGUE_ADJ_RE.findall(prompt))
    vague_density = n_vague / n_words
    instruction_x_vague = float(n_constraints) * vague_density
    raw = np.array(
        [n_logical, n_constraints, avg_wl, instruction_x_vague], dtype=np.float64
    )
    z = (raw - _TEXT_FEATURE_MEANS) / _TEXT_FEATURE_STDS
    return np.clip(z, -_TEXT_FEATURE_CLIP, _TEXT_FEATURE_CLIP)


def extract_text_features_batch(prompts: list[str]) -> np.ndarray:
    """Vectorized text feature extraction for a list of prompts.

    Args:
        prompts: List of raw prompt strings.

    Returns:
        2-D ``np.ndarray`` of shape ``(len(prompts), N_TEXT_FEATURES)``.
    """
    return np.vstack([extract_text_features(p) for p in prompts])


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization with numerical stability."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm < eps:
        logger.warning(f"Near-zero norm ({norm:.2e}) in l2_normalize, returning original")
        return x
    return np.asarray(x / norm)


def validate_feature_vector(x: np.ndarray, context: str = "") -> np.ndarray:
    """
    Validate feature vector for numerical issues.

    Checks for NaN, Inf, and extreme values that could destabilize LinUCB.

    Args:
        x: Feature vector to validate
        context: Description for error messages (e.g., "prompt: 'hello world'")

    Returns:
        Validated (and potentially clipped) feature vector

    Raises:
        ValueError: If vector contains NaN values
    """
    if np.any(np.isnan(x)):
        raise ValueError(f"Feature vector contains NaN values. {context}")

    if np.any(np.isinf(x)):
        logger.warning(f"Feature vector contains Inf values, clipping. {context}")
        x = np.clip(x, -1e6, 1e6)

    # Check for extreme values in PCA components (not bias)
    pca_components = x[:-1]
    if np.any(np.abs(pca_components) > 10):
        logger.warning(
            f"Feature vector has extreme values (max={np.max(np.abs(pca_components)):.2f}). "
            f"This may indicate PCA calibration issues. {context}"
        )

    return x


class FeatureService:
    """
    Feature extraction service for BanditRouter.

    **Responsibility**: Convert prompts to feature vectors
    **Output**: [PCA_0...PCA_24, bias] = 26-dimensional vector (with default pca_25.joblib)

    **PCA provenance guarantee:**
    The PCA projection matrix shipped with the package (``pca_25.joblib``) is
    fitted *once*, *offline*, on ~46K LMSYS Arena prompts.  Train-split
    prompts (``train.jsonl``) are excluded so PCA directions are not
    optimally aligned with warmup-prior data.  Val/test prompts are *not*
    excluded because PCA is unsupervised (no reward labels), so their
    presence cannot leak evaluation signal.  At runtime this class only
    calls ``pca.transform()`` — the projection is never re-fitted on
    evaluation data.  The JIT fallback path (when the artifact is missing)
    fits on synthetic prompts, not on the incoming stream.

    **Design Philosophy:**
    - Isolated from router logic (no LinUCB dependencies)
    - Easily swappable for custom feature engineering
    - Self-healing PCA loading with JIT calibration

    Example:
        >>> features = FeatureService()
        >>> vector = features.extract_features("Solve x^2 + 2x + 1 = 0")
        >>> vector.shape  # depends on PCA artifact; 26 with default pca_25.joblib
        (26,)
    """

    def __init__(
        self,
        encoder_model: str = DEFAULT_CONTEXT_MODEL,
        pca_path: Path | str | None = None,
        pca_components: int | None = None,
        target_variance: float = 0.60,
        whiten_pca: bool = True,
        allow_jit_training: bool = True,
        calibration_file: Path | str | None = None,
        custom_encoder: Callable[[str], np.ndarray] | None = None,
        embedding_dim: int | None = None,
        use_text_features: bool = False,
    ):
        """
        Initialize FeatureService with sentence encoder and optional PCA.

        Three paths are available (see module docstring for details):

        1. **Default SentenceTransformer** — leave *custom_encoder* as ``None``.
           Requires ``pip install paretobandit[embeddings]``.
        2. **Custom encoder callable** — pass any function that maps
           ``str → np.ndarray`` (1-D float vector).  The library handles PCA
           and bias-term appending.  ``embedding_dim`` is required so the
           service can validate PCA compatibility without calling the encoder.
        3. **Pre-computed vectors** — use ``FeatureService.for_precomputed(dim)``
           instead of this constructor.

        Args:
            encoder_model: SentenceTransformer model name (ignored when
                *custom_encoder* is provided).
            pca_path: Path to a pre-trained PCA model (``.joblib``).
                When ``None`` and using the default encoder, the shipped
                ``pca_25.joblib`` is loaded.  When ``None`` and a
                *custom_encoder* is given, **no PCA** is applied and raw
                embeddings (+ bias) are used directly.
            pca_components: Number of PCA components (auto-detected from PCA
                file if ``None``).
            target_variance: Minimum explained variance for PCA (default 0.60).
            whiten_pca: If ``True`` (default), scale PCA coordinates by
                ``1/sqrt(explained_variance)`` so each component has roughly
                unit variance under the PCA training distribution. This makes
                the downstream LinUCB isotropic prior (A₀=λI) better matched to
                the feature scale without requiring a new PCA artifact.
            allow_jit_training: Allow JIT PCA training when the PCA artifact
                is missing (default ``True``).  Set to ``False`` in strict
                production to crash-fast instead of hanging.
            calibration_file: Path to a line-delimited text file of real
                prompts for PCA calibration.
            custom_encoder: A callable ``(str) -> np.ndarray`` that produces
                a 1-D embedding vector for a given prompt.  When provided,
                ``sentence-transformers`` is **not** required.
            embedding_dim: Dimensionality of vectors produced by
                *custom_encoder*.  **Required** when *custom_encoder* is
                provided; ignored otherwise.
            use_text_features: If ``True``, append four lightweight
                regex-based text features (logical operator count, constraint
                keyword count, average word length, instruction×vague density)
                between the PCA components and the bias term.  Increases the
                feature vector by 4 dimensions.  Requires string prompts —
                incompatible with ``FeatureService.for_precomputed()``.
                Default ``False`` for backward compatibility.
        """
        self._custom_encoder = custom_encoder
        self.encoder_model = encoder_model
        self.whiten_pca = bool(whiten_pca)
        self._pca_whitening_scale: np.ndarray | None = None
        self.use_text_features = bool(use_text_features)

        if custom_encoder is not None:
            if embedding_dim is None:
                raise ValueError(
                    "embedding_dim is required when using a custom_encoder so "
                    "the service can validate PCA compatibility and set the "
                    "feature-vector dimension without calling the encoder."
                )
            self._custom_embedding_dim: int | None = embedding_dim
            self.encoder_model = "custom"
            # Custom encoders skip JIT training (synthetic prompts are tuned
            # for the default SentenceTransformer and would be misleading).
            allow_jit_training = False
        else:
            self._custom_embedding_dim = None

        _using_nondefault_encoder = (
            custom_encoder is None and encoder_model != DEFAULT_CONTEXT_MODEL
        )
        if _using_nondefault_encoder and pca_path is None:
            raise ValueError(
                f"Encoder '{encoder_model}' differs from the default "
                f"('{DEFAULT_CONTEXT_MODEL}'), so the shipped PCA artifact "
                f"is incompatible.  Options:\n\n"
                f"  1. Generate a matching PCA artifact:\n"
                f"       from pareto_bandit import train_pca\n"
                f"       train_pca(prompts, encoder_model='{encoder_model}', "
                f"output_path='my_pca.joblib')\n"
                f"     Then pass pca_path='my_pca.joblib'.\n\n"
                f"  2. Use a custom_encoder callable instead (skips PCA):\n"
                f"       fs = FeatureService(\n"
                f"           custom_encoder=my_encode_fn,\n"
                f"           embedding_dim=768,\n"
                f"       )\n"
                f"       router = BanditRouter(registry, feature_service=fs)\n\n"
                f"  3. Pass pca_path to an artifact you have already generated."
            )

        if pca_path is None and custom_encoder is None:
            from .config import DEFAULT_PCA_PATH
            self.pca_path: Path | None = DEFAULT_PCA_PATH
        elif pca_path is not None:
            self.pca_path = Path(pca_path)
        else:
            # custom_encoder with no PCA → raw embeddings
            self.pca_path = None

        self.pca_components = pca_components
        self.target_variance = target_variance

        if _using_nondefault_encoder or custom_encoder is not None:
            self.allow_jit_training = False
        else:
            self.allow_jit_training = allow_jit_training
        self.calibration_file = Path(calibration_file) if calibration_file else None

        # Lazy initialization
        self._encoder = _ENCODER_NOT_LOADED
        self._pca = _PCA_NOT_LOADED
        self._dimension: int | None = None

        # When custom_encoder is given without a PCA, eagerly set dimension
        # so callers can query .dimension before the first encode call.
        if custom_encoder is not None and self.pca_path is None:
            if pca_components is not None:
                raise ValueError(
                    "pca_components was set but no pca_path was provided for "
                    "the custom encoder.  Either supply a PCA artifact or "
                    "omit pca_components to use raw embeddings."
                )
            assert embedding_dim is not None
            self.pca_components = embedding_dim
            n_text = N_TEXT_FEATURES if self.use_text_features else 0
            self._dimension = embedding_dim + n_text + 1
            self._pca = None  # intentionally no PCA
            self._pca_whitening_scale = None

    def _apply_pca_whitening(self, pca_features: np.ndarray) -> np.ndarray:
        """Optionally whiten PCA coordinates to unit variance.

        Args:
            pca_features: 1-D PCA feature vector of shape ``(pca_components,)``.

        Returns:
            The (possibly) whitened PCA feature vector of the same shape.
        """
        if not self.whiten_pca:
            return pca_features
        if self._pca_whitening_scale is None:
            return pca_features
        if pca_features.shape != self._pca_whitening_scale.shape:
            raise ValueError(
                "PCA whitening scale shape mismatch: "
                f"features={pca_features.shape}, scale={self._pca_whitening_scale.shape}"
            )
        return np.asarray(pca_features * self._pca_whitening_scale)

    def _apply_pca_whitening_batch(self, pca_features: np.ndarray) -> np.ndarray:
        """Vectorized whitening for batched PCA features.

        Args:
            pca_features: 2-D array of shape ``(n, pca_components)``.

        Returns:
            Whitened array of the same shape when whitening is enabled, else the
            input array.
        """
        if not self.whiten_pca or self._pca_whitening_scale is None:
            return pca_features
        if pca_features.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {pca_features.shape}")
        if pca_features.shape[1] != self._pca_whitening_scale.shape[0]:
            raise ValueError(
                "PCA whitening scale shape mismatch: "
                f"features={pca_features.shape}, scale={self._pca_whitening_scale.shape}"
            )
        return np.asarray(pca_features * self._pca_whitening_scale.reshape(1, -1))

    @staticmethod
    def _compute_whitening_scale_from_pca(pca: Any) -> np.ndarray | None:
        """Return per-component whitening scales from a fitted sklearn PCA.

        Args:
            pca: Fitted ``sklearn.decomposition.PCA`` instance.

        Returns:
            1-D array of ``1/sqrt(explained_variance_)`` or ``None`` if the PCA
            object lacks ``explained_variance_``.
        """
        ev = getattr(pca, "explained_variance_", None)
        if ev is None:
            return None
        ev = np.asarray(ev, dtype=np.float64)
        return np.asarray(1.0 / np.sqrt(np.maximum(ev, 1e-12)))

    def _pca_has_builtin_whitening(self) -> bool:
        """Whether the loaded PCA artifact already whitens outputs."""
        if self._pca is _PCA_NOT_LOADED:
            _ = self.pca
        if self._pca is None:
            return False
        return bool(getattr(self._pca, "whiten", False))

    def get_pca_whitening_scales(self) -> np.ndarray:
        """Return per-dimension feature scaling implied by PCA whitening.

        This is useful for transforming warmup priors across feature conventions.

        Returns:
            1-D array of shape ``(dimension,)``.  The last element (bias term)
            is always 1.0.  Text feature slots (when enabled) are also 1.0
            since they are already z-score normalized.  When PCA is active and
            whitening is enabled, the PCA slots are ``1/sqrt(explained_variance)``.
        """
        dim = int(self.dimension)
        scales = np.ones(dim, dtype=np.float64)
        if self.using_pca:
            wants_whitened_outputs = self.whiten_pca or self._pca_has_builtin_whitening()
            if wants_whitened_outputs:
                pca_obj = self.pca
                if pca_obj is not None:
                    scale = self._compute_whitening_scale_from_pca(pca_obj)
                    if scale is not None:
                        n_pca = len(scale)
                        scales[:n_pca] = scale
        return scales

    @classmethod
    def for_precomputed(cls, dimension: int) -> FeatureService:
        """Create a lightweight service for pre-computed embedding vectors.

        No sentence-transformer model or PCA artifact is loaded.  The
        resulting instance only validates vector dimension when
        ``extract_features`` receives an ``np.ndarray``.  Passing a
        string prompt will raise because there is no encoder.

        Text features are disabled (requires string prompts).

        Args:
            dimension: Total feature-vector length (PCA components + bias).
        """
        instance = cls.__new__(cls)
        instance.pca_components = dimension - 1
        instance._encoder = _ENCODER_NOT_LOADED
        instance._custom_encoder = None
        instance._custom_embedding_dim = None
        instance._pca = None  # intentionally no PCA
        instance._dimension = dimension
        instance.encoder_model = "precomputed"
        instance.pca_path = None
        instance.target_variance = 0.0
        instance.allow_jit_training = False
        instance.calibration_file = None
        instance.use_text_features = False
        instance.whiten_pca = False
        instance._pca_whitening_scale = None
        return instance

    @property
    def encoder(self) -> Any:
        """Lazy-load the SentenceTransformer encoder on first use.

        When a *custom_encoder* callable was supplied at init time, accessing
        this property raises ``RuntimeError`` — use ``encode_prompt`` instead.
        """
        if self._custom_encoder is not None:
            raise RuntimeError(
                "A custom_encoder callable is configured; the SentenceTransformer "
                "encoder is not available.  Use encode_prompt() for text encoding."
            )
        if self._encoder is _ENCODER_NOT_LOADED:
            import os
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the default embedding "
                    "pipeline but is not installed.  Install it with:\n\n"
                    "    pip install paretobandit[embeddings]\n\n"
                    "Alternatively, pass a custom_encoder callable or use "
                    "FeatureService.for_precomputed() to avoid this dependency."
                ) from exc

            if sys.stdout.isatty():
                print(f"Loading embedding model '{self.encoder_model}'...", file=sys.stderr)

            self._encoder = SentenceTransformer(self.encoder_model)
            logger.info(f"Loaded encoder: {self.encoder_model}")
        return self._encoder

    @property
    def has_encoder(self) -> bool:
        """Whether this service can encode string prompts (custom or ST)."""
        if self._custom_encoder is not None:
            return True
        if self.encoder_model == "precomputed":
            return False
        return True

    def encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode a single prompt to a 1-D embedding vector.

        Dispatches to the custom encoder callable when available, otherwise
        falls through to the SentenceTransformer encoder.

        Args:
            prompt: Input text.

        Returns:
            1-D ``np.ndarray`` of floats (L2-normalized).
        """
        if self._custom_encoder is not None:
            vec = np.asarray(self._custom_encoder(prompt), dtype=np.float64)
            if vec.ndim != 1:
                raise ValueError(
                    f"custom_encoder must return a 1-D array, got shape {vec.shape}"
                )
            return l2_normalize(vec)
        return l2_normalize(
            self.encoder.encode(prompt, normalize_embeddings=True, show_progress_bar=False)
        )

    def encode_prompts_batch(self, prompts: list[str]) -> np.ndarray:
        """Encode multiple prompts to a 2-D embedding matrix.

        Args:
            prompts: List of input texts.

        Returns:
            2-D ``np.ndarray`` of shape ``(len(prompts), embedding_dim)``.
        """
        if self._custom_encoder is not None:
            vecs = np.array(
                [np.asarray(self._custom_encoder(p), dtype=np.float64) for p in prompts]
            )
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            return np.asarray(vecs / norms)
        return np.asarray(self.encoder.encode(
            prompts, normalize_embeddings=True,
            show_progress_bar=len(prompts) > 100,
        ))

    def get_sentence_embedding_dimension(self) -> int:
        """Return the raw embedding dimension (before PCA).

        Works for custom encoders (via *embedding_dim*) and for
        SentenceTransformer encoders (queried from the model).
        """
        if self._custom_embedding_dim is not None:
            return self._custom_embedding_dim
        return int(self.encoder.get_sentence_embedding_dimension())

    @property
    def pca(self) -> Any:
        """Lazy load PCA on first use with self-healing."""
        if self._pca is _PCA_NOT_LOADED:
            self._ensure_pca_ready()
        return self._pca

    @property
    def dimension(self) -> int:
        """Total feature dimension (PCA [+ text features] + bias)."""
        if self.pca_components is None:
            _ = self.pca  # trigger lazy load which sets pca_components
        assert self.pca_components is not None
        n_text = N_TEXT_FEATURES if self.use_text_features else 0
        return self.pca_components + n_text + 1

    @property
    def bias_index(self) -> int:
        """Bias term is always the last element."""
        return -1

    @property
    def using_pca(self) -> bool:
        """Check if PCA compression is active (vs raw embeddings)."""
        if self._pca is _PCA_NOT_LOADED:
            _ = self.pca  # trigger lazy load
        return self._pca is not None

    def get_dimension(self) -> int:
        """
        Get feature vector dimensionality.

        Returns:
            Dimension of output vectors (pca_components [+ text features] + bias)
        """
        if self._dimension is None:
            n_text = N_TEXT_FEATURES if self.use_text_features else 0
            assert self.pca_components is not None
            self._dimension = self.pca_components + n_text + 1
        return self._dimension

    def get_feature_names(self) -> list[str]:
        """
        Get human-readable feature names for interpretability.

        Returns:
            List of feature names matching vector indices

        Example:
            >>> fs = FeatureService()
            >>> names = fs.get_feature_names()
            >>> names[:3]
            ['PCA_0', 'PCA_1', 'PCA_2']
            >>> names[-1]
            'bias'
        """
        n_text = N_TEXT_FEATURES if self.use_text_features else 0
        n_emb = self.pca_components if self.pca_components is not None else (self.dimension - n_text - 1)

        # Check if using raw embeddings (fallback mode)
        is_raw = self._pca is None and self._dimension and self._dimension > n_emb + n_text + 1
        if is_raw:
            names = [f"emb_{i}" for i in range(n_emb)]
        else:
            names = [f"PCA_{i}" for i in range(n_emb)]

        if self.use_text_features:
            names.extend(TEXT_FEATURE_NAMES)

        names.append("bias")
        return names

    def extract_features(self, prompt: str | np.ndarray) -> np.ndarray:
        """
        Convert prompt to feature vector.

        **Feature Structure (with default pca_25.joblib):**
        [PCA_0, PCA_1, ..., PCA_24, bias] = 26 dimensions

        The actual dimension is determined by the PCA artifact loaded at init.
        Default production artifact: pca_25.joblib (25 PCA + 1 bias = 26D).

        Args:
            prompt: Input text or pre-computed vector

        Returns:
            Feature vector of dimension (pca_components + 1 bias)

        Raises:
            ValueError: If prompt is empty or feature extraction fails
            TypeError: If prompt is wrong type

        Example:
            >>> features = FeatureService()
            >>> vector = features.extract_features("Explain quantum computing")
            >>> vector.shape  # 26 with default pca_25.joblib
            (26,)
            >>> vector[-1]  # Bias term
            1.0
        """
        # Handle pre-computed vectors
        if isinstance(prompt, np.ndarray):
            # Validate dimension
            if len(prompt) != self.dimension:
                raise ValueError(
                    f"Pre-computed vector has dimension {len(prompt)}, "
                    f"expected {self.dimension}"
                )
            return prompt

        # Type validation
        if not isinstance(prompt, str):
            raise TypeError(f"Expected str or np.ndarray, got {type(prompt)}")

        # Guard: precomputed mode cannot encode strings
        if self.encoder_model == "precomputed":
            raise TypeError(
                "This FeatureService was created with for_precomputed() and "
                "cannot encode string prompts.  Pass a pre-computed "
                "np.ndarray of shape (dimension,) instead, or create a "
                "FeatureService with an encoder (e.g. FeatureService() or "
                "FeatureService(custom_encoder=fn, embedding_dim=N))."
            )

        # Empty/whitespace validation
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")

        # Length validation (prevent OOM)
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(
                f"Prompt length ({len(prompt)}) exceeds maximum ({MAX_PROMPT_LENGTH}). "
                f"Truncating to prevent OOM."
            )
            prompt = prompt[:MAX_PROMPT_LENGTH]

        # 1. Semantic Embedding (delegates to custom_encoder or SentenceTransformer)
        emb_full = self.encode_prompt(prompt)

        # 2. PCA Compression
        if self.pca is not None:
            emb_reduced = self.pca.transform(emb_full.reshape(1, -1)).flatten()
            emb_reduced = self._apply_pca_whitening(emb_reduced)
        else:
            # Fallback: use raw embeddings (no PCA)
            emb_reduced = emb_full

        # 3. Optionally append text features
        if self.use_text_features:
            text_feats = extract_text_features(prompt)
            emb_reduced = np.concatenate([emb_reduced, text_feats])

        # 4. Append bias term
        result = np.append(emb_reduced, 1.0)

        # 5. Validate output
        result = validate_feature_vector(result, context=f"prompt: '{prompt[:50]}...'")

        return result

    def extract_features_batch(self, prompts: list[str]) -> np.ndarray:
        """
        Extract features for multiple prompts efficiently.

        Uses batch encoding which is faster than sequential calls.

        Args:
            prompts: List of prompt strings

        Returns:
            Array of shape (n_prompts, dimension)

        Example:
            >>> fs = FeatureService()
            >>> vectors = fs.extract_features_batch(["Hello", "World"])
            >>> vectors.shape
            (2, 26)
        """
        if not prompts:
            return np.empty((0, self.dimension))

        # Validate all prompts
        valid_prompts = []
        for i, p in enumerate(prompts):
            if not isinstance(p, str):
                raise TypeError(f"Prompt {i} is not a string: {type(p)}")
            if not p.strip():
                raise ValueError(f"Prompt {i} is empty or whitespace-only")
            if len(p) > MAX_PROMPT_LENGTH:
                logger.warning(f"Prompt {i} truncated from {len(p)} to {MAX_PROMPT_LENGTH}")
                p = p[:MAX_PROMPT_LENGTH]
            valid_prompts.append(p)

        # Batch encode (dispatches to custom_encoder or SentenceTransformer)
        embeddings = self.encode_prompts_batch(valid_prompts)

        # PCA transform
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            embeddings = self._apply_pca_whitening_batch(embeddings)

            # Validate and handle numerical issues
            if np.any(np.isnan(embeddings)):
                logger.warning(f"PCA transform produced NaN values for {np.sum(np.any(np.isnan(embeddings), axis=1))} prompts. Replacing with zeros.")
                embeddings = np.nan_to_num(embeddings, nan=0.0)

            if np.any(np.isinf(embeddings)):
                logger.warning("PCA transform produced Inf values. Clipping to ±1e6.")
                embeddings = np.clip(embeddings, -1e6, 1e6)

        # Optionally append text features
        if self.use_text_features:
            text_feats = extract_text_features_batch(valid_prompts)
            embeddings = np.hstack([embeddings, text_feats])

        # Append bias column
        bias_column = np.ones((len(embeddings), 1))
        result = np.hstack([embeddings, bias_column])

        return result

    def _ensure_pca_ready(self) -> None:
        """
        Self-Healing PCA: Load existing PCA, validate it, or train new one via JIT calibration.

        This prevents production outages from:
        - Missing PCA artifacts
        - Dimension mismatches (encoder upgrades)
        - Manifold collapse (low variance capture)
        """
        # No PCA path → intentionally skip PCA (custom encoder w/o PCA, etc.)
        if self.pca_path is None:
            self._pca = None
            self._pca_whitening_scale = None
            return

        pca_loaded = False

        # Check if joblib is available
        try:
            import joblib as jl
        except ImportError:
            logger.warning("joblib not available - cannot use PCA compression")
            self._pca = None
            return

        # Phase 1: Try loading existing PCA
        if self.pca_path:
            logger.info(f"Attempting to load PCA from: {self.pca_path.absolute()}")
            if self.pca_path.exists():
                try:
                    candidate_pca = jl.load(self.pca_path)

                    # Validation: Dimension check
                    expected_dim = self.get_sentence_embedding_dimension()
                    actual_dim = candidate_pca.n_features_in_

                    if actual_dim == expected_dim:
                        self._pca = candidate_pca
                        # Only apply external whitening when the PCA artifact
                        # does NOT already whiten its transform().
                        if self.whiten_pca and not bool(getattr(candidate_pca, "whiten", False)):
                            self._pca_whitening_scale = self._compute_whitening_scale_from_pca(candidate_pca)
                        else:
                            self._pca_whitening_scale = None
                        # Auto-detect components from loaded PCA if not specified
                        if self.pca_components is None:
                            self.pca_components = candidate_pca.n_components_
                        explained_var = np.sum(candidate_pca.explained_variance_ratio_)
                        logger.info(
                            f"✓ PCA loaded successfully "
                            f"({actual_dim}→{candidate_pca.n_components_}, "
                            f"variance={explained_var:.1%})"
                        )
                        pca_loaded = True
                    else:
                        logger.warning(
                            f"⚠️ PCA dimension mismatch! "
                            f"Encoder: {expected_dim}D, PCA: {actual_dim}D. "
                            f"Re-training with JIT calibration."
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load PCA artifact at {self.pca_path}: {e}. Re-training.")
            else:
                logger.warning(f"⚠️ PCA artifact not found at {self.pca_path.absolute()}")

        # Phase 2: JIT Calibration (if needed)
        if not pca_loaded:
            # Gate JIT training for strict production mode
            if not self.allow_jit_training:
                raise RuntimeError(
                    "PCA artifact not found and JIT training is disabled (allow_jit_training=False). "
                    "Deploy correct PCA artifact or enable JIT training for development."
                )

            # Log CRITICAL warning for configuration drift
            logger.critical(
                "🚨 JIT PCA TRAINING TRIGGERED! 🚨\n"
                "This indicates configuration drift:\n"
                "  - PCA artifact missing from expected path\n"
                "  - Dimension mismatch (encoder version changed?)\n"
                "Generating PCA from SYNTHETIC data.\n"
                "WARNING: This will hang the first request for 2-5 seconds!\n"
                "Synthetic distribution may not match production traffic!\n"
                "ACTION: Verify PCA artifact is deployed correctly."
            )
            logger.info("⚡ JIT PCA Calibration: Training new PCA on synthetic data...")

            # Generate synthetic prompts matching procedural warmup
            synthetic_prompts = self._generate_synthetic_data(n_samples=1000)
            logger.info(f"  Generated {len(synthetic_prompts)} synthetic prompts")

            # Encode to get embeddings
            logger.info("  Encoding prompts...")
            embeddings = self.encode_prompts_batch(synthetic_prompts)
            logger.info(f"  Embeddings shape: {embeddings.shape}")

            # Fit PCA
            from sklearn.decomposition import PCA
            # If pca_components not specified, default to 25 to match paper (d=26)
            n_components = self.pca_components if self.pca_components is not None else 25
            new_pca = PCA(n_components=n_components, whiten=bool(self.whiten_pca))
            new_pca.fit(embeddings)

            # Update pca_components from fitted PCA
            if self.pca_components is None:
                self.pca_components = new_pca.n_components_

            # Strict PCA variance validation:
            # Low variance capture indicates manifold collapse or insufficient components
            explained_var = np.sum(new_pca.explained_variance_ratio_)
            logger.info(f"  JIT PCA Explained Variance: {explained_var:.1%}")

            if explained_var < self.target_variance:
                # Safe fallback to raw embeddings
                #
                # CRITICAL: Proceeding with low-variance PCA means >40% of semantic
                # signal is lost, effectively routing on noise rather than meaning.
                #
                # Better to fallback to raw (uncompressed) embeddings:
                # - Slower: O(raw_dim²) updates vs O(pca_dim²)
                # - Correct: Full semantic routing vs noise-based routing
                #
                # This prevents silent performance degradation. Users will see critical
                # log and know to retrain PCA with more data or higher n_components.
                raw_dim = self.get_sentence_embedding_dimension()
                logger.critical(
                    f"🛑 PCA VARIANCE TOO LOW: {explained_var:.2%} < {self.target_variance:.2%}\n"
                    f"   ⚠️  FALLBACK TO RAW EMBEDDINGS ({raw_dim}D) FOR SAFETY\n"
                    f"   📊 Impact: Slower updates (O({raw_dim}²) vs O({self.pca_components}²)) but CORRECT semantic routing\n"
                    f"   🔧 Fix: Retrain PCA with more data or increase n_components in config\n"
                    f"   📍 PCA path: {self.pca_path}"
                )
                # Disable PCA - use raw embeddings
                self._pca = None
                self._pca_whitening_scale = None
                n_text = N_TEXT_FEATURES if self.use_text_features else 0
                self._dimension = raw_dim + n_text + 1
                logger.info(f"   ✅ Using raw {raw_dim}D embeddings (+ {n_text} text + 1 bias) = {self._dimension}D features")
                return  # Skip setting self._pca, will use raw in extract_features()

            self._pca = new_pca
            if self.whiten_pca and not bool(getattr(new_pca, "whiten", False)):
                self._pca_whitening_scale = self._compute_whitening_scale_from_pca(new_pca)
            else:
                self._pca_whitening_scale = None
            logger.info(f"  ✓ JIT PCA ready ({embeddings.shape[1]}→{self.pca_components})")

            # Phase 3: Persist for next startup (cache-aside pattern)
            if self.pca_path:
                try:
                    self.pca_path.parent.mkdir(parents=True, exist_ok=True)
                    jl.dump(new_pca, self.pca_path)
                    logger.info(f"  💾 Saved JIT PCA to {self.pca_path} for future use")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not persist PCA (non-fatal): {e}")

    def _generate_synthetic_data(self, n_samples: int = 1000) -> list[str]:
        """
        Generate synthetic prompts for PCA training.

        **Conference REVIEW WARNING: Domain Bias Risk**

        Synthetic data is biased toward English math/coding tasks. If production
        traffic is in a different domain (e.g., Japanese legal contracts), the PCA
        projection may filter out critical semantic variance.

        **Solution**: Use calibration_file parameter in __init__() to load real
        prompts from your domain before falling back to synthetic data.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            List of synthetic prompt strings
        """
        # If calibration file provided, load real prompts
        if self.calibration_file and self.calibration_file.exists():
            logger.info(f"Loading calibration prompts from {self.calibration_file}")
            try:
                with open(self.calibration_file, encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
                if len(prompts) >= n_samples:
                    logger.info(f"  ✓ Loaded {len(prompts)} real prompts (domain-specific)")
                    return prompts[:n_samples]
                else:
                    logger.warning(
                        f"  ⚠️  Only {len(prompts)} prompts in calibration file, "
                        f"need {n_samples}. Supplementing with synthetic data."
                    )
                    # Use what we have + synthetic to fill gap
                    synthetic = self._generate_synthetic_fallback(n_samples - len(prompts))
                    return prompts + synthetic
            except Exception as e:
                logger.error(f"Failed to load calibration file: {e}. Using synthetic data.")

        # Fallback to synthetic data
        return self._generate_synthetic_fallback(n_samples)

    def _generate_synthetic_fallback(self, n_samples: int) -> list[str]:
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
        for _ in range(n_samples):
            archetype = rng.choice(archetype_keys)
            template = rng.choice(templates[archetype])

            prompt = template
            for placeholder, values in fill_values.items():
                if f"{{{placeholder}}}" in prompt:
                    prompt = prompt.replace(f"{{{placeholder}}}", rng.choice(values))

            prompts.append(prompt)

        return prompts
