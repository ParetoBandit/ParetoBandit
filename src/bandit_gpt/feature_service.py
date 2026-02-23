"""
Feature Service: The Eyes of the BanditRouter.

Handles all feature extraction logic independently from the LinUCB math.
This separation allows iterating on feature engineering (regex, PCA, encoders)
without risking breaking the router core.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
import numpy as np

logger = logging.getLogger(__name__)

# Import from centralized config
from .config_legacy import DEFAULT_SENTENCE_TRANSFORMER

# Default context model
DEFAULT_CONTEXT_MODEL = DEFAULT_SENTENCE_TRANSFORMER


# Maximum prompt length to prevent OOM on very long inputs
MAX_PROMPT_LENGTH = 50000  # ~12k tokens


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization with numerical stability."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm < eps:
        logger.warning(f"Near-zero norm ({norm:.2e}) in l2_normalize, returning original")
        return x
    return x / norm


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
    **Output**: [PCA_0...PCA_31, bias] = 33-dimensional vector (with default pca_32.joblib)
    
    **Design Philosophy:**
    - Isolated from router logic (no LinUCB dependencies)
    - Easily swappable for custom feature engineering
    - Self-healing PCA loading with JIT calibration
    
    Example:
        >>> features = FeatureService()
        >>> vector = features.extract_features("Solve x^2 + 2x + 1 = 0")
        >>> vector.shape  # depends on PCA artifact; 33 with default pca_32.joblib
        (33,)
    """
    
    def __init__(
        self,
        encoder_model: str = DEFAULT_CONTEXT_MODEL,
        pca_path: Optional[Path | str] = None,
        pca_components: int = None,  # Auto-detect from PCA file if not specified
        target_variance: float = 0.60,
        allow_jit_training: bool = True,
        calibration_file: Optional[Path | str] = None,
    ):
        """
        Initialize FeatureService with sentence encoder and optional PCA.
        
        Args:
            encoder_model: SentenceTransformer model name
            pca_components: Number of PCA components (auto-detected from PCA file if None)
            pca_path: Path to pre-trained PCA model (optional, defaults to DEFAULT_PCA_PATH)
            target_variance: Minimum explained variance for PCA (default 0.60)
            allow_jit_training: Allow JIT PCA training if artifact missing (default: True)
                              Set to False in strict production to crash-fast instead of hanging
            calibration_file: Path to real prompts for PCA calibration (optional)
                             Line-delimited text file. Used instead of synthetic data
                             to train domain-specific PCA projections.
        """
        self.encoder_model = encoder_model
        
        # If no PCA path provided, use the default from config_legacy
        if pca_path is None:
            from .config_legacy import DEFAULT_PCA_PATH
            self.pca_path = DEFAULT_PCA_PATH
        else:
            self.pca_path = Path(pca_path)
            
        self.pca_components = pca_components  # Will be set from loaded PCA if None
        self.target_variance = target_variance
        self.allow_jit_training = allow_jit_training
        self.calibration_file = Path(calibration_file) if calibration_file else None
        
        # Lazy initialization
        self._encoder = None
        self._pca = None
        self._dimension = None
    
    @classmethod
    def for_precomputed(cls, dimension: int) -> "FeatureService":
        """Create a lightweight service for pre-computed embedding vectors.

        No sentence-transformer model or PCA artifact is loaded.  The
        resulting instance only validates vector dimension when
        ``extract_features`` receives an ``np.ndarray``.  Passing a
        string prompt will raise because there is no encoder.

        Args:
            dimension: Total feature-vector length (PCA components + bias).
        """
        instance = cls.__new__(cls)
        instance.pca_components = dimension - 1
        instance._encoder = None
        instance._pca = None
        instance._dimension = dimension
        instance.encoder_model = "precomputed"
        instance.pca_path = None
        instance.target_variance = 0.0
        instance.allow_jit_training = False
        instance.calibration_file = None
        return instance

    @property
    def encoder(self):
        """Lazy load encoder on first use."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.encoder_model)
                logger.info(f"Loaded encoder: {self.encoder_model}")
            except ImportError as e:
                raise ImportError(
                    "Missing dependency: sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._encoder
    
    @property
    def pca(self):
        """Lazy load PCA on first use with self-healing."""
        if self._pca is None:
            self._ensure_pca_ready()
        return self._pca

    @property
    def dimension(self) -> int:
        """Total feature dimension (PCA + bias)."""
        return self.pca_components + 1
    
    @property
    def bias_index(self) -> int:
        """Bias term is always the last element."""
        return -1
    
    @property
    def using_pca(self) -> bool:
        """Check if PCA compression is active (vs raw embeddings)."""
        # Trigger PCA loading if not done yet
        if self._pca is None:
            _ = self.pca
        # Check if we fell back to raw embeddings
        return self._pca is not None
    
    def get_dimension(self) -> int:
        """
        Get feature vector dimensionality.
        
        Returns:
            Dimension of output vectors (pca_components + 1 bias term)
        """
        if self._dimension is None:
            self._dimension = self.pca_components + 1  # PCA + bias
        return self._dimension
    
    def get_feature_names(self) -> List[str]:
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
        dim = self.dimension
        # Check if using raw embeddings (fallback mode)
        if self._pca is None and self._dimension and self._dimension > self.pca_components + 1:
            # Raw embedding mode
            names = [f"emb_{i}" for i in range(dim - 1)]
        else:
            names = [f"PCA_{i}" for i in range(dim - 1)]
        names.append("bias")
        return names
    
    def extract_features(self, prompt: Union[str, np.ndarray]) -> np.ndarray:
        """
        Convert prompt to feature vector.
        
        **Feature Structure (with default pca_32.joblib):**
        [PCA_0, PCA_1, ..., PCA_31, bias] = 33 dimensions
        
        The actual dimension is determined by the PCA artifact loaded at init.
        Default production artifact: pca_32.joblib (32 PCA + 1 bias = 33D).
        
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
            >>> vector.shape  # 33 with default pca_32.joblib
            (33,)
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
        
        # 1. Semantic Embedding
        emb_full = self.encoder.encode(prompt, normalize_embeddings=True, show_progress_bar=False)
        emb_full = l2_normalize(emb_full)
        
        # 2. PCA Compression
        if self.pca:
            emb_reduced = self.pca.transform(emb_full.reshape(1, -1)).flatten()
        else:
            # Fallback: use raw embeddings (no PCA)
            emb_reduced = emb_full
        
        # 3. Append bias term
        result = np.append(emb_reduced, 1.0)
        
        # 4. Validate output
        result = validate_feature_vector(result, context=f"prompt: '{prompt[:50]}...'")
        
        return result
    
    def extract_features_batch(self, prompts: List[str]) -> np.ndarray:
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
            (2, 24)
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
        
        # Batch encode
        embeddings = self.encoder.encode(
            valid_prompts,
            normalize_embeddings=True,  # Already normalized by encoder
            show_progress_bar=len(valid_prompts) > 100
        )
        
        # Note: Embeddings already normalized by encoder, no need for double normalization
        # embeddings = np.array([l2_normalize(e) for e in embeddings])  # Redundant
        
        # PCA transform
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            
            # Validate and handle numerical issues
            if np.any(np.isnan(embeddings)):
                logger.warning(f"PCA transform produced NaN values for {np.sum(np.any(np.isnan(embeddings), axis=1))} prompts. Replacing with zeros.")
                embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            if np.any(np.isinf(embeddings)):
                logger.warning(f"PCA transform produced Inf values. Clipping to ±1e6.")
                embeddings = np.clip(embeddings, -1e6, 1e6)
        
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
        pca_loaded = False
        
        # Check if joblib is available
        try:
            import joblib as jl
        except ImportError:
            logger.warning("joblib not available - cannot use PCA compression")
            return
        
        # Phase 1: Try loading existing PCA
        if self.pca_path:
            logger.info(f"Attempting to load PCA from: {self.pca_path.absolute()}")
            if self.pca_path.exists():
                try:
                    candidate_pca = jl.load(self.pca_path)
                    
                    # Validation: Dimension check
                    expected_dim = self.encoder.get_sentence_embedding_dimension()
                    actual_dim = candidate_pca.n_features_in_
                    
                    if actual_dim == expected_dim:
                        self._pca = candidate_pca
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
            embeddings = self.encoder.encode(
                synthetic_prompts,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            logger.info(f"  Embeddings shape: {embeddings.shape}")
            
            # Fit PCA
            from sklearn.decomposition import PCA
            # If pca_components not specified, default to 32 for JIT training
            n_components = self.pca_components if self.pca_components is not None else 32
            new_pca = PCA(n_components=n_components)
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
                # Better to fallback to raw 384D embeddings:
                # - Slower: O(384²) updates vs O(24²)  
                # - Correct: Full semantic routing vs noise-based routing
                # 
                # This prevents silent performance degradation. Users will see critical
                # log and know to retrain PCA with more data or higher n_components.
                logger.critical(
                    f"🛑 PCA VARIANCE TOO LOW: {explained_var:.2%} < {self.target_variance:.2%}\n"
                    f"   ⚠️  FALLBACK TO RAW EMBEDDINGS ({self.encoder.get_sentence_embedding_dimension()}D) FOR SAFETY\n"
                    f"   📊 Impact: Slower updates (O({self.encoder.get_sentence_embedding_dimension()}²) vs O({self.pca_components}²)) but CORRECT semantic routing\n"
                    f"   🔧 Fix: Retrain PCA with more data or increase n_components in config\n"
                    f"   📍 PCA path: {self.pca_path}"
                )
                # Disable PCA - use raw embeddings
                self._pca = None
                # Update dimension to raw embedding size + bias term
                raw_dim = self.encoder.get_sentence_embedding_dimension()
                self._dimension = raw_dim + 1  # 384 + 1 = 385
                logger.info(f"   ✅ Using raw {raw_dim}D embeddings (+ 1 bias) = {self._dimension}D features")
                return  # Skip setting self._pca, will use raw in extract_features()
            
            self._pca = new_pca
            logger.info(f"  ✓ JIT PCA ready ({embeddings.shape[1]}→{self.pca_components})")
            
            # Phase 3: Persist for next startup (cache-aside pattern)
            if self.pca_path:
                try:
                    self.pca_path.parent.mkdir(parents=True, exist_ok=True)
                    jl.dump(new_pca, self.pca_path)
                    logger.info(f"  💾 Saved JIT PCA to {self.pca_path} for future use")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not persist PCA (non-fatal): {e}")
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> List[str]:
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
                with open(self.calibration_file, 'r', encoding='utf-8') as f:
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
    
    def _generate_synthetic_fallback(self, n_samples: int) -> List[str]:
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
        for _ in range(n_samples):
            archetype = random.choice(archetype_keys)
            template = random.choice(templates[archetype])
            
            # Fill placeholders
            prompt = template
            for placeholder, values in fill_values.items():
                if f"{{{placeholder}}}" in prompt:
                    prompt = prompt.replace(f"{{{placeholder}}}", random.choice(values))
            
            prompts.append(prompt)
        
        return prompts
