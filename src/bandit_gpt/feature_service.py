"""
Feature Service: The Eyes of the BanditRouter.

Handles all feature extraction logic independently from the LinUCB math.
This separation allows iterating on feature engineering (regex, PCA, encoders)
without risking breaking the router core.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Default context model
DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization with numerical stability."""
    norm = np.linalg.norm(x)
    return x / (norm + eps)


class FeatureService:
    """
    Feature extraction service for BanditRouter.
    
    **Responsibility**: Convert prompts to feature vectors
    **Output**: [PCA_0...PCA_22, bias] = 24-dimensional vector
    
    **Design Philosophy:**
    - Isolated from router logic (no LinUCB dependencies)
    - Easily swappable for custom feature engineering
    - Self-healing PCA loading with JIT calibration
    
    Example:
        >>> features = FeatureService()
        >>> vector = features.extract_features("Solve x^2 + 2x + 1 = 0")
        >>> vector.shape
        (24,)
    """
    
    def __init__(
        self,
        encoder_model: str = DEFAULT_CONTEXT_MODEL,
        pca_path: Optional[Path | str] = None,
        pca_components: int = 23,
        target_variance: float = 0.60,
        allow_jit_training: bool = True
    ):
        """
        Initialize feature extraction service.
        
        Args:
            encoder_model: Sentence transformer model for embeddings
            pca_path: Path to PCA model (optional, will JIT calibrate if missing)
            pca_components: Number of PCA components (default: 23)
            target_variance: Minimum explained variance threshold for PCA
            allow_jit_training: Allow JIT PCA training if artifact missing (default: True)
                              Set to False in strict production to crash-fast instead of hanging
        """
        self.encoder_model = encoder_model
        self.pca_path = Path(pca_path) if pca_path else None
        self.pca_components = pca_components
        self.target_variance = target_variance
        self.allow_jit_training = allow_jit_training
        
        # Lazy initialization
        self._encoder = None
        self._pca = None
        self._dimension = None
    
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
    
    def get_dimension(self) -> int:
        """
        Get feature vector dimensionality.
        
        Returns:
            Dimension of output vectors (pca_components + 1 bias term)
        """
        if self._dimension is None:
            self._dimension = self.pca_components + 1  # PCA + bias
        return self._dimension
    
    def extract_features(self, prompt: str | np.ndarray) -> np.ndarray:
        """
        Convert prompt to feature vector.
        
        **Feature Structure:**
        [PCA_0, PCA_1, ..., PCA_22, bias] = 24 dimensions
        
        Args:
            prompt: Input text or pre-computed vector
        
        Returns:
            24-dimensional feature vector (23 PCA + 1 bias)
            
        Example:
            >>> features = FeatureService()
            >>> vector = features.extract_features("Explain quantum computing")
            >>> vector.shape
            (24,)
            >>> vector[-1]  # Bias term
            1.0
        """
        if isinstance(prompt, np.ndarray):
            # Already a feature vector
            return prompt
        
        # 1. Semantic Embedding
        emb_full = self.encoder.encode(prompt, normalize_embeddings=True)
        emb_full = l2_normalize(emb_full)
        
        # 2. PCA Compression
        if self.pca:
            emb_reduced = self.pca.transform(emb_full.reshape(1, -1)).flatten()
        else:
            emb_reduced = emb_full[:self.pca_components]
        
        # 3. Append bias term
        return np.append(emb_reduced, 1.0)
    
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
        if self.pca_path and self.pca_path.exists():
            try:
                candidate_pca = jl.load(self.pca_path)
                
                # Validation: Dimension check
                expected_dim = self.encoder.get_sentence_embedding_dimension()
                actual_dim = candidate_pca.n_features_in_
                
                if actual_dim == expected_dim:
                    self._pca = candidate_pca
                    explained_var = np.sum(candidate_pca.explained_variance_ratio_)
                    logger.info(
                        f"✓ PCA loaded from {self.pca_path} "
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
                logger.warning(f"⚠️ Failed to load PCA artifact: {e}. Re-training.")
        
        # Phase 2: JIT Calibration (if needed)
        if not pca_loaded:
            # KDD REVIEW FIX v2: Gate JIT training for strict production mode
            if not self.allow_jit_training:
                raise RuntimeError(
                    "PCA artifact not found and JIT training is disabled (allow_jit_training=False). "
                    "Deploy correct PCA artifact or enable JIT training for development."
                )
            
            # KDD REVIEW FIX: Log CRITICAL warning for configuration drift
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
            synthetic_prompts = self._generate_synthetic_data(n=1000)
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
            new_pca = PCA(n_components=self.pca_components)
            new_pca.fit(embeddings)
            
            # KDD REVIEW FIX (Critique C): Strict PCA Variance Validation
            # Low variance capture indicates manifold collapse or insufficient components
            explained_var = np.sum(new_pca.explained_variance_ratio_)
            logger.info(f"  JIT PCA Explained Variance: {explained_var:.1%}")
            
            if explained_var < self.target_variance:
                error_msg = (
                    f"PCA variance too low: {explained_var:.2%} < target {self.target_variance:.2%}. "
                    f"This indicates poor embedding quality. Recommendations:\n"
                    f"  1. Increase n_components (currently {self.pca_components})\n"
                    f"  2. Check encoder quality ({self.encoder_model})\n"
                    f"  3. Review synthetic data distribution\n"
                    f"With d={self.pca_components + 1}, LinUCB needs ~{10 * (self.pca_components + 1)} samples to warm up. "
                    f"Low-quality features will hurt sample efficiency."
                )
                logger.error(f"🛑 {error_msg}")
                raise ValueError(error_msg)
            
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
    
    def _generate_synthetic_data(self, n: int = 1000) -> list[str]:
        """
        Generate synthetic prompts for PCA calibration.
        
        Uses the same archetypes as procedural warmup to ensure consistency
        between PCA manifold and warmup covariance structure.
        
        Args:
            n: Number of synthetic prompts to generate (default: 1000)
               For robust PCA, need ~10x the target dimensionality (23 dims → ~230 samples)
               
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
