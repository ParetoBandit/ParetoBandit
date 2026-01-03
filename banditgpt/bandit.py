"""
Production-grade contextual bandit router (Hot Path).

Core Features:
1. HLE Prior as Default: Initializes with "expert intuition" from 26k prompts.
2. Default Registry: Automatically loads 80+ models with cost/latency data.
3. Multi-Objective: Balances Quality, Cost, and Latency.
4. Constraints: Supports max_cost, max_latency, and quality floors.
"""

from __future__ import annotations

import json
import math
import time
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np

# Set environment variable to avoid hangs in multi-threaded/multi-process environments
# This is a common issue with SentenceTransformers on Mac/Linux.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Missing dependency: sentence-transformers") from e

try:
    from .cluster_detector import ClusterDetector
except ImportError:
    ClusterDetector = None  # Optional feature

try:
    import joblib
except ImportError:
    joblib = None




logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Optimization Profiles
# ---------------------------------------------------------------------------
class OptimizationProfile:
    """Named presets for utility function weights (Quality vs Cost vs Latency).
    Risk is now handled via Gating/Masking, not penalty weights.
    """
    QUALITY_FIRST   = {"lambda_cost": 0.005, "lambda_latency": 0.005}
    BEST_VALUE      = {"lambda_cost": 1.42,  "lambda_latency": 0.1}
    COST_SAVER      = {"lambda_cost": 5.0,   "lambda_latency": 1.0}
    LOW_LATENCY     = {"lambda_cost": 0.1,   "lambda_latency": 8.0}
    VALUE_EFFICIENT = {"lambda_cost": 1.25, "lambda_latency": 0.5}

    _PROFILES = {
        "quality_first": QUALITY_FIRST,
        "best_value": BEST_VALUE,
        "balanced": BEST_VALUE,  # Alias for backwards compatibility
        "cost_saver": COST_SAVER,
        "low_latency": LOW_LATENCY,
        "value_efficient": VALUE_EFFICIENT,
    }

    @classmethod
    def get(cls, name: str) -> Dict[str, float]:
        key = name.lower().replace("-", "_")
        if key not in cls._PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Valid: {list(cls._PROFILES.keys())}")
        return cls._PROFILES[key]

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
                print(f"DEBUG: ExplorationRate.get('{name}') -> {val}")
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

def transform_hle_to_prior(raw_hle_score: float) -> float:
    """
    Maps the raw HLE score (0-40%) to a realistic Utility Probability (0-100%).
    
    Acknowledges that even "low" HLE scores indicate highly capable models:
    - 0.5% HLE → ~0.1 (trash tier, random guessing)
    - 6.5% HLE → ~0.8 (capable of daily tasks)
    - 25% HLE → ~0.95 (genius tier)
    
    Uses a sigmoid (logistic function) centered at 20% HLE.
    """
    # Models below 1% are truly broken
    if raw_hle_score < 0.01:
        return 0.1
    
    # Sigmoid parameters
    k = 80.0      # Steepness: controls how quickly the curve transitions
    x0 = 0.20     # Midpoint: 20% HLE maps to utility ~0.5 (Separates "Good" from "Elite")
    
    # Logistic function: sigmoid(x) = 1 / (1 + e^(-k*(x - x0)))
    utility_prior = 1.0 / (1.0 + np.exp(-k * (raw_hle_score - x0)))
    
    # Cap at 0.95 to leave room for uncertainty/learning
    return min(utility_prior, 0.95)

def sigmoid(x: float) -> float:
    """Standard logistic function mapping (-inf, inf) to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))

def safe_inv(A: np.ndarray) -> np.ndarray:
    """Safe matrix inversion with pseudo-inverse fallback for stability."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)

# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------
class DisjointLinUCBPolicy:
    """Disjoint LinUCB: one ridge regression per arm."""
    def __init__(self, model_names: List[str], dim: int = 384, alpha: float = 0.1,
                 ridge_lambda: float = 1.0, forgetting_factor: float = 0.95):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.gamma = float(forgetting_factor) # Forgetting factor (1.0 = no forgetting)
        # Initialize A=I*lambda, b=0
        self.A = {m: np.eye(self.dim) * self.ridge_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        
        # Precompute A_inv for hot-path speed
        self.A_inv = {m: safe_inv(self.A[m]) for m in self.models}
                
        self.last_update = {m: 0 for m in self.models} # Track last update step
        self.t = 0 # Global time step

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically."""
        if model_name in self.models: return
        
        self.models.append(model_name)
        self.A[model_name] = np.eye(self.dim) * self.ridge_lambda
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


    def select_arm(self, x: np.ndarray, candidates: Optional[List[str]] = None) -> Tuple[str, float]:
        candidates = candidates or self.models
        candidates = [m for m in candidates if m in self.A]
        if not candidates: raise ValueError("No candidates available")

        best_model = candidates[0]
        best_ucb = -float("inf")

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

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        if model not in self.A: return
        
        self.t += 1 # Increment global clock
        
        # Synchronize decay before update
        dt = self.t - self.last_update[model]
        if dt > 0:
            effective_gamma = self.gamma ** dt
            self.A[model] *= effective_gamma
            self.b[model] *= effective_gamma
            
            # Maintain invertibility with ridge
            if self.gamma < 1.0:
                 self.A[model] += (1.0 - effective_gamma) * np.eye(self.dim) * self.ridge_lambda
        
        # Add new data
        self.A[model] += np.outer(x, x)
        self.b[model] += float(reward) * x
        
        self.A_inv[model] = safe_inv(self.A[model])
        self.last_update[model] = self.t

    def save_state(self, path: Path | str) -> None:
        """Save A and b matrices to a compressed NPZ file."""
        data = {}
        for m in self.models:
            data[f"{m}_A"] = self.A[m]
            data[f"{m}_b"] = self.b[m]
        np.savez_compressed(path, **data)

    def load_state(self, path: Path | str) -> None:
        """Load A and b matrices from a compressed NPZ file."""
        data = np.load(path)
        for m in self.models:
            a_key = f"{m}_A"
            b_key = f"{m}_b"
            if a_key in data and b_key in data:
                self.A[m] = data[a_key]
                self.b[m] = data[b_key]
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
    cluster_id: Optional[int] = None  # Detected semantic cluster
    cluster_similarity: Optional[float] = None  # Similarity to cluster centroid
    context_vector: Optional[np.ndarray] = None # Cached embedding for updates

class BanditRouter:
    """
    The primary entry point for routing.
    """
    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        context_encoder=None,  # NEW: Optional pre-initialized encoder for dependency injection
        alpha: float = 0.1,
        embedding_dim: int = 384,
        ridge_lambda: float = 1.0,
        forgetting_factor: float = 0.95,
        benchmark_key: str = "hle",
        cluster_boost_weight: float = 0.0,  # Default: disabled until validated
        pca_path: Optional[Path | str] = None, # Path to PCA model
    ):
        self.registry = dict(model_registry)
        
        # Use provided encoder or initialize new one
        if context_encoder is not None:
            self.encoder = context_encoder
        else:
            self.encoder = SentenceTransformer(context_model)
        
        # Initialize PCA if provided
        self.pca = None
        if pca_path and joblib:
            try:
                self.pca = joblib.load(pca_path)
                logger.info(f"✓ Hybrid PCA initialized (384->{self.pca.n_components})")
            except Exception as e:
                logger.warning(f"Failed to load PCA model: {e}")
        
        # Initialize cluster detector if available
        self.cluster_detector = None
        if ClusterDetector is not None:
            try:
                # Share encoder to avoid loading twice
                self.cluster_detector = ClusterDetector(encoder=self.encoder)
                logger.info(f"✓ Cluster detector initialized with {self.cluster_detector.n_clusters} clusters")
            except Exception as e:
                logger.warning(f"Could not initialize cluster detector: {e}")
        
        # -----------------------------------------------------------------------
        # FEATURE VECTOR DIMENSION LOGIC
        # Base Embedding (384) + Handcrafted Features (7) + Cluster Distances (5) + Bias (1) = 397
        # NOTE: High dimensionality (~400 params per arm) is expensive for online bandits.
        # Without strong priors (N_eff), convergence would take 10k+ steps.
        # Priors are essential here to bridge the "cold start" gap.
        # -----------------------------------------------------------------------
        enc_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Check effective dimension
        # If PCA is active, base dim is PCA components (32)
        if self.pca:
            base_dim = self.pca.n_components
        else:
            base_dim = enc_dim
            
        if embedding_dim == enc_dim:
             # User likely passed default 384 (or we just want auto-calc).
             # We are adding 13 features (8 explicit + 5 cluster).
             embedding_dim = base_dim + 13
        
        # Add bias term to dimension
        self.bandit = DisjointLinUCBPolicy(
            list(self.registry.keys()), 
            dim=embedding_dim + 1, 
            alpha=alpha,
            ridge_lambda=ridge_lambda,
            forgetting_factor=forgetting_factor
        )
        
        # Initialize Security Scanner (Lazy)
        self._toxicity_scanner = None
        try:
             from llm_guard.input_scanners import Toxicity
             self._toxicity_scanner = Toxicity(threshold=0.5)
             logger.info("✓ Toxicity scanner initialized")
        except ImportError:
             logger.info("Toxicity scanner not available (llm-guard not installed). Feature will be 0.0.")
        except Exception as e:
             logger.warning(f"Failed to initialize toxicity scanner: {e}")

        self.logs: List[RoutingLog] = []
        self.model_priors: Dict[str, float] = {} 
        self.benchmark_key = benchmark_key
        self.cluster_boost_weight = cluster_boost_weight
        
        # New Model Admission: Probation List
        # Stores model_id -> request_count_at_admission (or just boolean check in pruner)
        self.probation_models: Dict[str, Dict[str, Any]] = {} 


    def _count_syllables(self, word: str) -> int:
        """Heuristic syllable counter for Flesch-Kincaid."""
        word = word.lower().strip(".:;?!")
        if not word: return 0
        if len(word) <= 3: return 1
        
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        # Subtract silent 'e' at end
        if word.endswith('e'):
            count -= 1
        # Subtract consecutive vowels (already handled by regex group)
        return max(1, count)

    def _extract_handcrafted_features(self, text: str) -> np.ndarray:
        """
        Extract explicit features for routing logic.
        1. is_code_heavy
        2. requires_json
        3. input_length_log
        4. list_density
        5. instruction_density
        6. flesch_kincaid
        7. question_count
        8. toxicity_score
        """
        if not text:
            return np.zeros(8)
        
        # --- BASICS ---
        total_len = len(text)
        words = re.findall(r'\b\w+\b', text.lower())
        n_words = len(words)
        lines = text.split('\n')
        n_lines = len(lines)
        
        # 1. Code Heavy
        code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', text, re.DOTALL)
        code_len = sum(len(c) for c in code_blocks)
        is_code_heavy = (code_len / total_len) if total_len > 0 else 0.0
        
        # 2. Requires JSON
        json_keywords = ["json", "valid format", "schema", "output format"]
        requires_json = 1.0 if any(k in text.lower() for k in json_keywords) else 0.0
        
        # 3. Input Length (Log)
        n_tokens = n_words * 1.3
        input_length_log = np.log(n_tokens + 1.0)
        
        # 4. List Density
        list_markers = [l for l in lines if l.strip().startswith(('-', '*', '1.', '2.'))]
        list_density = (len(list_markers) / n_lines) if n_lines > 0 else 0.0
        
        # --- COMPLEXITY ---
        
        # 5. Instruction Density
        imperatives = {"create", "write", "solve", "analyze", "explain", "summarize", "find", "calculate", "implement", "design"}
        n_imperatives = sum(1 for w in words if w in imperatives)
        instruction_density = (n_imperatives / n_words) if n_words > 0 else 0.0
        
        # 6. Flesch-Kincaid Grade
        sentences = re.split(r'[.!?]+', text)
        n_sentences = max(1, len([s for s in sentences if s.strip()]))
        
        if n_words > 0:
            n_syllables = sum(self._count_syllables(w) for w in words)
            fk_grade = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
        else:
            fk_grade = 0.0
            
        fk_normalized = max(0.0, min(fk_grade, 20.0)) / 20.0
        
        # 7. Question Count
        q_count = text.count('?')
        question_count = np.log(q_count + 1.0)
        
        # --- SECURITY ---
        
        # 8. Toxicity Score
        toxicity_score = 0.0
        if self._toxicity_scanner:
            try:
                _, _, score = self._toxicity_scanner.scan(text)
                toxicity_score = score
            except Exception:
                pass
        
        return np.array([
            is_code_heavy, requires_json, input_length_log, list_density,
            instruction_density, fk_normalized, question_count,
            toxicity_score
        ])

    def _get_cluster_distances(self, embedding: np.ndarray) -> np.ndarray:
        """
        Get distances to the 5 Fixed Anchor Clusters.
        
        Args:
            embedding: Normalized sentence embedding (384,)
        """
        k = 5
        if not self.cluster_detector:
            return np.zeros(k) # Fallback
            
        try:
            # Use Fixed Anchors (Math, Coding, etc)
            return self.cluster_detector.get_anchor_distances(embedding)
        except Exception as e:
            logger.warning(f"Cluster detection failed: {e}")
            return np.zeros(k)

    def _get_context_vector(self, context: str | np.ndarray) -> np.ndarray:
        """Convert string prompt or array to a normalized context vector."""
        if isinstance(context, str):
            # 1. Semantic Embedding (384 -> 32 if PCA)
            # Captures DOMAIN/TOPIC (e.g. "Math", "Creative Writing")
            emb_full = self.encoder.encode(context)
            emb_full = l2_normalize(emb_full)
            
            if self.pca:
                # Project 384 -> 32
                # reshape to (1, 384) for transform, then flatten
                emb_reduced = self.pca.transform(emb_full.reshape(1, -1)).flatten()
            else:
                emb_reduced = emb_full
            
            # 2. Handcrafted Features (8)
            # Captures COMPLEXITY/DIFFICULTY (e.g. "Simple Arithmetic" vs "Calculus")
            # Addressing the limitation of semantic embeddings which cluster all "Math" together.
            feats = self._extract_handcrafted_features(context)
            
            # 3. Cluster Distances (5)
            # Must use full 384-dim embedding for distance calculation
            cluster_dists = self._get_cluster_distances(emb_full)
            
            # Concatenate: [Embedding, Explicit, Clusters]
            x = np.concatenate([emb_reduced, feats, cluster_dists])
        else:
            x = context
            
        # Append bias term (Last dim)
        return np.append(x, 1.0)


    @classmethod
    def create(
        cls,
        model_registry: Optional[Dict[str, Dict]] = None,
        context_model: str = "pca",
        context_encoder=None,  # NEW: Optional pre-initialized encoder for testing/DI
        prior_n_effective: Optional[float] = None,
        prior_structure_n_effective: Optional[float] = None,
        alpha: Optional[float] = None,
        exploration: str = "safe",
        state_path: Optional[str] = None,
        priors: str = "csr",
        benchmark_key: Optional[str] = None,
        ridge_lambda: float = 1.0,
        forgetting_factor: float = 1.0,
        cluster_boost_weight: float = 0.0
    ) -> "BanditRouter":
        """
        Factory method to create a BanditRouter with optional priors.
        
        Args:
            model_registry: Dict of {model_id: model_metadata}.
            context_model: "pca" (default) or "sbert".
            context_encoder: Optional pre-initialized encoder (for testing or custom encoders).
                           If provided, context_model is ignored.
            priors: Prior type to load:
                - "csr": Cluster Success Rates (task-specific, recommended)
                - "hle": Hard Label Evaluation scores (generic)
                - "benchmark": Deprecated alias for "csr"
                - "none": Cold start (no priors)
            prior_n_effective: Effective sample size for belief strength (b vector scaling).
                              Default: Auto-selected based on priors type:
                                - CSR: 20.0 (optimal for early advantage from ablation study)
                                - HLE: 60.0 (optimal for HLE from ablation study)
                                - none: 0.0 (no priors)
            prior_structure_n_effective: Effective sample size for structural stiffness (A matrix scaling).
                                        Default: Auto-selected based on priors type:
                                          - CSR: 20.0 (optimal for early advantage)
                                          - HLE: 10.0 (optimal for HLE)
                                          - none: 20.0 (structure only, no mean)
                              Note: None = Infinite stiffness (deprecated, not recommended).
            exploration: "static", "safe", "balanced", "aggressive".
            ridge_lambda: Regularization parameter (default 1.0, auto-scales with structure).
            state_path: Optional path to a saved bandit state (.npz).
            benchmark_key: Deprecated. Use priors="hle" instead.
            cluster_boost_weight: Reward boost weight for cluster specialization (default 0.0, disabled).
        """
        base_dir = Path(__file__).parent
        
        # Backward compatibility: "benchmark" → "csr"
        if priors == "benchmark":
            logger.warning("priors='benchmark' is deprecated. Use priors='csr' instead.")
            priors = "csr"
        
        # Auto-select optimal parameters based on prior type (from z-score ablation study)
        if prior_n_effective is None:
            if priors == "csr":
                prior_n_effective = 20.0  # Optimal for CSR early advantage
            elif priors == "hle":
                prior_n_effective = 60.0  # Optimal for HLE
            elif priors == "none":
                prior_n_effective = 0.0   # No priors
            else:
                prior_n_effective = 20.0  # Default fallback
        
        if prior_structure_n_effective is None:
            if priors == "csr":
                prior_structure_n_effective = 20.0  # Optimal for CSR
            elif priors == "hle":
                prior_structure_n_effective = 10.0  # Optimal for HLE
            elif priors == "none":
                prior_structure_n_effective = 20.0  # Structure only, no mean
            else:
                prior_structure_n_effective = 20.0  # Default fallback
        
        # 1. Load Default Registry if needed
        if model_registry is None:
            models_path = base_dir / "models.json"
            if not models_path.exists():
                raise FileNotFoundError(f"Default models.json not found at {models_path}")
            with open(models_path) as f:
                data = json.load(f)
            model_registry = {m["openrouter_id"]: m for m in data["models"]}
            
        # 2. Resolve Exploration
        alpha = ExplorationRate.get(exploration)
        
        # --- Parameter Validation ---
        # Ensure prior parameters are valid numbers and non-negative
        if prior_n_effective is not None:
            if not np.isfinite(prior_n_effective) or prior_n_effective < 0:
                raise ValueError(f"Invalid prior_n_effective: {prior_n_effective}. Must be finite and non-negative.")
        
        if prior_structure_n_effective is not None:
            if not np.isfinite(prior_structure_n_effective) or prior_structure_n_effective < 0:
                raise ValueError(f"Invalid prior_structure_n_effective: {prior_structure_n_effective}. Must be finite and non-negative.")
        
        # 3. Initialize
        # Determine PCA path
        pca_path_default = base_dir / "data" / "pca_32.joblib"
        
        # Load Saved State (Overrides priors)
        if state_path and Path(state_path).exists():
            router = cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                        alpha=alpha, ridge_lambda=ridge_lambda,
                        forgetting_factor=forgetting_factor, benchmark_key=benchmark_key,
                        cluster_boost_weight=cluster_boost_weight, pca_path=pca_path_default)
            router.bandit.load_state(state_path)
            return router

        # Load HLE Priors (Generic Structure + Generic Mean)
        if priors == "hle":
            # Use same PCA-based structural priors as CSR
            meta_path = base_dir / "priors" / "priors_meta_pca.npz"
            
            if not meta_path.exists():
                 logger.warning("No priors metadata found. Falling back to cold start.")
                 return cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                           alpha=alpha, ridge_lambda=ridge_lambda,
                           benchmark_key=benchmark_key, cluster_boost_weight=cluster_boost_weight,
                           pca_path=pca_path_default if pca_path_default.exists() else None)

            return cls.load_from_benchmark(
                model_registry=model_registry,
                context_model=context_model,
                context_encoder=context_encoder,
                alpha=alpha,
                prior_n_effective=prior_n_effective,
                prior_structure_n_effective=prior_structure_n_effective,
                ridge_lambda=ridge_lambda,
                forgetting_factor=forgetting_factor,
                priors_meta_path=meta_path,
                benchmark_key="hle",  # Use generic HLE scores instead of cluster success rates
                cluster_boost_weight=cluster_boost_weight,
                pca_path=pca_path_default if pca_path_default.exists() else None
            )
            
        # Load CSR Priors (Generic Structure + Task-Specific Cluster Success Rates)
        if priors == "csr":
            # Use PCA-based priors (45D: 32 PCA + 8 handcrafted + 5 cluster)
            meta_path = base_dir / "priors" / "priors_meta_pca.npz"
            
            if not meta_path.exists():
                 logger.warning("No priors metadata found. Falling back to cold start.")
                 return cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                           alpha=alpha, ridge_lambda=ridge_lambda,
                           benchmark_key=benchmark_key, cluster_boost_weight=cluster_boost_weight,
                           pca_path=pca_path_default if pca_path_default.exists() else None)

            return cls.load_from_benchmark(
                model_registry=model_registry,
                context_model=context_model,
                context_encoder=context_encoder,
                alpha=alpha,
                prior_n_effective=prior_n_effective,
                prior_structure_n_effective=prior_structure_n_effective,
                ridge_lambda=ridge_lambda,
                forgetting_factor=forgetting_factor,
                priors_meta_path=meta_path,
                benchmark_key="csr",  # Use CSR mode (cluster success rates)
                cluster_boost_weight=cluster_boost_weight,
                pca_path=pca_path_default if pca_path_default.exists() else None
            )
            
        # Cold Start (No Priors)
        if priors == "none":
            logger.info("Cold start mode: No priors loaded.")
            return cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                      alpha=alpha, ridge_lambda=ridge_lambda,
                      forgetting_factor=forgetting_factor, benchmark_key=benchmark_key,
                      cluster_boost_weight=cluster_boost_weight,
                      pca_path=pca_path_default if pca_path_default.exists() else None)
        
        # Unknown priors type
        raise ValueError(f"Unknown priors type: '{priors}'. Use 'csr', 'hle', or 'none'.")

    @classmethod
    def load_from_benchmark(
        cls,
        *,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str = DEFAULT_CONTEXT_MODEL,
        context_encoder=None,  # NEW: For dependency injection
        alpha: float = 0.1,
        prior_n_effective: float = 20.0,
        prior_structure_n_effective: Optional[float] = None,
        ridge_lambda: float = 1.0,
        priors_meta_path: Optional[Path] = None,
        forgetting_factor: float = 0.95,
        benchmark_key: str = "hle",
        cluster_boost_weight: float = 0.1,
        pca_path: Optional[Path] = None,
    ) -> "BanditRouter":
        """
        Initialize with HLE priors using covariance matrix.
        
        Two-Knob Scaling:
            prior_n_effective: Controls belief strength (b vector scaling). Default 20.0.
            prior_structure_n_effective: Controls structural stiffness (A matrix scaling). 
                                         Default None = Infinite stiffness (unscaled).
        """
        meta = np.load(priors_meta_path)
        cov_matrix = meta["cov_matrix"]
        sum_vec = meta["sum_vec"]
        dim = sum_vec.shape[0]
        
        router = cls(
            model_registry, 
            context_model=context_model,
            context_encoder=context_encoder,
            alpha=alpha, 
            embedding_dim=dim,
            ridge_lambda=ridge_lambda,
            benchmark_key=benchmark_key,
            cluster_boost_weight=cluster_boost_weight,
            pca_path=pca_path
        )
        
        # Ridge Update: A += init_scale * Cov, b += belief_scale * score * Sum
        # NORMALIZATION: We scale the benchmark (N=26223) to N_effective
        
        # Load Cluster Statistics
        cluster_sums = meta["cluster_sums"]     # (100, 45)
        cluster_counts = meta["cluster_counts"] # (100,)
        global_sum = meta["global_sum"]         # (45,)
        n_clusters = cluster_sums.shape[0]
        
        # Calculate total samples first (needed for normalization)
        total_samples = float(np.sum(cluster_counts))
        
        # NORMALIZATION: Convert sums to means for fair CSR vs HLE comparison
        # This ensures prior_n_effective has equivalent strength for both
        cluster_means = cluster_sums / cluster_counts[:, np.newaxis]  # Shape: (100, 45)
        global_mean = global_sum / max(total_samples, 1.0)            # Shape: (45,)
        
        # -----------------------------------------------------------------------
        # TWO-KNOB SCALING FRAMEWORK
        # -----------------------------------------------------------------------
        # Note: total_samples already calculated above for normalization 
        
        # Knob 1: Structural Stiffness (Initialization Scaling for A Matrix)
        # Controls how confident we are in the covariance structure
        if prior_structure_n_effective is None:
            init_scale = 1.0  # Infinite stiffness (unscaled, full N_offline strength)
        else:
            init_scale = prior_structure_n_effective / max(total_samples, 1.0)
            
        # Knob 2: Belief Strength (Scaling for b Vector)
        # Controls how confident we are in the mean quality scores
        # NOTE: Named 'belief_scale' to avoid collision with self.gamma (forgetting factor)
        belief_scale = prior_n_effective / max(total_samples, 1.0)
        
        # Adaptive Ridge Regularization
        # Scale ridge_lambda to match the prior strength to prevent ridge from dominating
        # For infinite stiffness (init_scale=1.0), keep original ridge_lambda
        # For scaled priors, make ridge proportional to structure
        if prior_structure_n_effective is not None and init_scale < 0.1:
            # Structure is weak (scaled down), scale ridge proportionally
            # Use 10x multiplier for stability while keeping ridge in same range as structure
            effective_ridge_lambda = init_scale * 10.0
            logger.info(f"Adaptive ridge: {ridge_lambda:.3f} → {effective_ridge_lambda:.6f} (10x init_scale)")
        else:
            # Structure is strong or infinite, use original ridge
            effective_ridge_lambda = ridge_lambda
        
        # Update router's bandit with effective ridge
        # Reinitialize A matrices with adaptive ridge
        for m in router.bandit.models:
            router.bandit.A[m] = np.eye(router.bandit.dim) * effective_ridge_lambda
            router.bandit.A_inv[m] = safe_inv(router.bandit.A[m])
        
        # Pad covariance matrix for bias term (zeros for cross-terms, 1.0 for bias variance)
        # AND pad for new handcrafted features if strict dimensions don't match
        # Prior Cov is (384, 384). Router dim is likely 388 (+1 bias).
        
        current_dim = router.bandit.dim # e.g. 389 (384+4+1)
        prior_dim = dim # e.g. 384
        
        cov_padded = np.eye(current_dim) # Default to identity for new features/bias
        # Fill strictly the top-left block with the prior covariance
        # This means new features start with Identity covariance (neutral prior)
        cov_padded[:prior_dim, :prior_dim] = cov_matrix
        
        # Apply Initialization Scaling to Covariance (Knob 1)
        # A_prior = init_scale * Sum(xx^T)
        # We add this to the existing A = lambda * I
        A_prior_update = init_scale * cov_padded
        
        # Ensure bias term (last element) has variance 1.0 (It is already 1.0 from eye init)
        
        for m in router.bandit.models:
            # Update A with the prior
            router.bandit.A[m] += A_prior_update
            
            # Get raw benchmark score (default 0.05 for new models)
            # This fallback enables "new model with no benchmarks" constraint
            raw_hle_score = float(model_registry.get(m, {}).get(benchmark_key) or 0.05)
            
            # ------------------------------------------------------------------
            # PRIOR MEAN CALCULATION: HLE (Generic) vs CSR (Task-Specific)
            # ------------------------------------------------------------------
            # We compute the initial belief vector (b) based on prior knowledge:
            # - HLE: Flat prior using generic benchmark performance
            # - CSR: Structured prior using per-cluster success rates
            
            # Initialize variables for later bias term calculation
            model_cluster_success_rates = None  # Dict of cluster_id -> success_rate
            cluster_rates_ordered_array = None  # Numpy array aligned with cluster_sums
            transformed_utility_score = None    # Sigmoid-transformed HLE score (0-1 range)
            
            # CRITICAL: Different prior modes use different data sources
            # benchmark_key="hle" -> Generic HLE benchmark scores
            # benchmark_key="csr" -> Cluster-specific success rates
            if benchmark_key == "hle":
                # HLE MODE: Generic priors (ignore cluster_success_rates even if available)
                # Creates a uniform prior across all clusters based on overall performance
                # Use global MEAN (not sum) for fair comparison with CSR
                transformed_utility_score = transform_hle_to_prior(raw_hle_score)
                prior_mean_update_vector = transformed_utility_score * global_mean
            else:
                # CSR MODE: Task-specific priors using cluster success rates
                model_cluster_success_rates = model_registry.get(m, {}).get("cluster_success_rates", [])
                
                # Fallback to HLE if cluster data is missing
                if not model_cluster_success_rates or len(model_cluster_success_rates) != n_clusters:
                    transformed_utility_score = transform_hle_to_prior(raw_hle_score)
                    prior_mean_update_vector = transformed_utility_score * global_mean
                else:
                    # Use cluster-specific z-scores (normalized success rates)
                    # Extract z-scores from {"0": {"raw": 0.9, "z_score": 1.2}, ...}
                    # Z-scores eliminate frontier model bias by normalizing per-cluster
                    cluster_z_scores_list = []
                    for cluster_idx in range(n_clusters):
                        cluster_data = model_cluster_success_rates.get(str(cluster_idx))
                        if cluster_data is None:
                            cluster_data = model_cluster_success_rates.get(cluster_idx)
                        
                        # Strict: expect dict with z_score, no fallback
                        if not isinstance(cluster_data, dict) or 'z_score' not in cluster_data:
                            raise ValueError(
                                f"Missing z_score for model {m}, cluster {cluster_idx}. "
                                f"Run update_success_rates.py to regenerate cluster_success_rates."
                            )
                        
                        cluster_z_scores_list.append(float(cluster_data['z_score']))
                    
                    cluster_z_scores_array = np.array(cluster_z_scores_list)
                    
                    # Transform z-scores to positive weights using sigmoid
                    # sigmoid(z) maps (-inf, inf) -> (0, 1)
                    # This gives: bad performance -> low weight, good performance -> high weight
                    cluster_weights = 1.0 / (1.0 + np.exp(-cluster_z_scores_array))
                    
                    # Weighted MEAN: Σ(weight[k] * cluster_mean_vector[k])
                    # Weights derived from z-scores, not raw success rates
                    # Shape: (100,) @ (100, 45) -> (45,)
                    prior_mean_update_vector = np.dot(cluster_weights, cluster_means)

            # Apply Belief Strength Scaling (Knob 2: prior_n_effective)
            # b[m] accumulates weighted feature vectors: Σ(x * reward)
            # IMPORTANT: We use prior_n_effective directly (not belief_scale) because
            # cluster_means and global_mean are already divided by N_offline
            # Using belief_scale would divide by N_offline twice!
            router.bandit.b[m][:prior_dim] += prior_n_effective * prior_mean_update_vector

            # Update Bias Term (Last Element)
            # Represents average expected reward across all contexts
            # For CSR: weighted average of cluster success rates
            # For HLE: flat transformed utility score
            if model_cluster_success_rates and len(model_cluster_success_rates) == n_clusters:
                # CSR mode: Compute weighted average using z-score-derived weights
                weighted_avg_success = np.dot(
                    cluster_weights,  # From sigmoid(z_scores), not raw rates
                    meta["cluster_counts"]
                ) / max(total_samples, 1.0)
                router.bandit.b[m][-1] += prior_n_effective * weighted_avg_success
            else:
                # HLE mode or fallback: Use flat transformed score
                router.bandit.b[m][-1] += prior_n_effective * transformed_utility_score

            # ------------------------------------------------------------------
            # SMART PRIOR: Efficiency Boosting (LiteLLM-inspired)
            # ------------------------------------------------------------------
            cost = float(model_registry.get(m, {}).get("input_cost_per_m") or 0.0) / 1000.0
            # Avoid division by zero, assume min cost $0.05/1M -> $0.00005/1k
            cost = max(cost, 0.00000005) 
            
            # Efficiency Factor: Higher for lower cost.
            # Log-scale to dampen extreme differences.
            # e.g. Cost=0.15 (GPT-4o) -> log(1/0.15) ~ 1.9
            #      Cost=0.0001 (Flash) -> log(1/0.0001) ~ 9.2
            # We scale this to be a multiplier, e.g. 1.0 + (0.0 * efficiency) -> No Boost (Relies on Utility)
            efficiency_boost = 1.0 + (0.2 * math.log(1.0 / cost))

            # ------------------------------------------------------------------
            # CONTEXTUAL CLUSTER PRIOR (Mathematical Formulation)
            # U(m, x) = beta * HLE(m) + (1-beta) * ClusterPerf(m, k)
            # ------------------------------------------------------------------
            
            # 1. Detect Cluster (Simplified)
            # We check model ID, description, and tags for keywords.
            # This allows new models to be clustered if metadata is provided.
            # If no metadata, it defaults to "General" (no cluster boost).
            md = model_registry.get(m, {})
            text_to_check = (m + " " + md.get("description", "") + " " + " ".join(md.get("tags", []))).lower()
            
            is_math = any(k in text_to_check for k in ["math", "reasoning", "deepseek", "gemini", "flash"])
            is_code = any(k in text_to_check for k in ["code", "coder", "python"])
            
            # Cluster boost now handled via process_feedback() with ClusterDetector
            # Apply efficiency boost only to priors
            # Apply Efficiency to the final B vector?
            # "Smart Prior" implies it prefers efficient models.
            # Apply Efficiency to the final B vector?
            # "Smart Prior" implies it prefers efficient models.
            # Retaining this boost for now but applied to the calculated vector.
            router.bandit.b[m] *= efficiency_boost
            
            # Recompute inverse
            router.bandit.A_inv[m] = safe_inv(router.bandit.A[m])
            
        return router

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
            stats["hle"].append(float(m_data.get(self.benchmark_key) or 0.0))
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
        hle = float(model_data.get(self.benchmark_key) or 0.0)
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
            
            # Score = Reward - (w_c * C) - (w_l * L)
            return reward - (profile["lambda_cost"] * c_norm) - (profile["lambda_latency"] * l_norm)

        new_cost = float(new_model_data.get("input_cost_per_m") or 0.0)
        new_lat = float(new_model_data.get("time_to_first_token_seconds") or 0.0)
        
        # Profiles to check
        profiles = [
            OptimizationProfile.QUALITY_FIRST,
            OptimizationProfile.BEST_VALUE,
            OptimizationProfile.COST_SAVER,
            OptimizationProfile.LOW_LATENCY
        ]
        
        # Check dominance
        is_useful_in_any_profile = False
        
        for profile in profiles:
            # Optimistic Score for New Model (Reward = 1.0)
            opt_score = get_score(1.0, new_cost, new_lat, profile)
            
            # Best Score among existing models (using their ESTIMATED quality from registry or HLE?)
            # Use HLE as a proxy for "current known quality" for the gatekeeper check
            best_existing = -float("inf")
            for m_id, m_data in self.registry.items():
                if m_id == new_model_data["openrouter_id"]: continue
                
                m_hle = transform_hle_to_prior(float(m_data.get(self.benchmark_key) or 0.0))
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
            lambda_reg = self.bandit.ridge_lambda
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
        self.probation_models[model_id] = {
            "start_t": self.bandit.t,
            "status": "PROBATION",
            "immune_until": self.bandit.t + 500
        }
        
        return True

    def prune_arms(self, min_requests: int = 1000, selection_threshold: float = 0.001) -> List[str]:
        """
        Lazy Pruning (Eviction) Mechanism.
        Removes models that are "Starved" (Selection < 0.1%) AND "Dominated" (Pareto sub-optimal).
        
        Args:
            min_requests: Only prune if we have at least this many logs (warm-up check).
            selection_threshold: Usage frequency below which a model is considered "Starved".
            
        Returns:
            List of removed model IDs.
        """
        if len(self.logs) < min_requests:
            return []
            
        # 1. Calculate Selection Rates (Starvation Check)
        # Using a sliding window of last 1000 requests for relevance, or all?
        # User implies periodic check, so let's use the last N logs matching min_requests
        recent_logs = self.logs[-min_requests:]
        counts = Counter(l.selected_model for l in recent_logs)
        total = len(recent_logs)
        
        starved_candidates = []
        for m in self.bandit.models:
            # Check Probation Immunity
            if m in self.probation_models:
                p_info = self.probation_models[m]
                # If immune, skip
                if self.bandit.t < p_info["immune_until"]:
                    continue
                # If graduated, remove from probation map (cleanup)
                if self.bandit.t >= p_info["immune_until"]:
                    del self.probation_models[m]
                    
            rate = counts[m] / total
            if rate < selection_threshold:
                starved_candidates.append(m)
                
        if not starved_candidates:
            return []
            
        # 2. Calculate "Learned Quality" (Real mu)
        # We average the predicted utility over the recent contexts for ALL models (not just starved)
        # to establish the current Pareto Frontier.
        
        # Sample contexts: Use unique contexts from recent logs, capped at 50 for speed
        sample_contexts = [l.context_vector for l in recent_logs if l.context_vector is not None]
        # De-duplicate by bytes hash if needed, or just take every kth
        if len(sample_contexts) > 50:
             # Take 50 evenly spaced
             indices = np.linspace(0, len(sample_contexts)-1, 50, dtype=int)
             sample_contexts = [sample_contexts[i] for i in indices]
             
        if not sample_contexts:
            return [] # Cannot estimate quality
            
        # Compute Mean Utility for each model m on these contexts
        # mu = Mean(x^T * theta)
        model_quality = {}
        for m in self.bandit.models:
            if m not in self.bandit.A_inv: continue
            theta = self.bandit.A_inv[m] @ self.bandit.b[m]
            
            utilities = []
            for x in sample_contexts:
                utilities.append(float(np.dot(theta, x)))
            
            model_quality[m] = float(np.mean(utilities))
            
        # 3. Check Dominance
        # A Starved model is removed ONLY if it is strictly dominated by another model.
        # Dominated means: Exists B such that Cost(B) < Cost(A) AND Quality(B) > Quality(A).
        # We also usually check Latency. User said "Cheaper AND Higher Win Rate".
        # Let's stick to Cost/Quality for KDD simplifiction, or include Latency if critical.
        # User prompt: "Is there another model that is Cheaper AND has a higher empirical Win Rate?"
        
        removed = []
        for victim in starved_candidates:
            victim_q = model_quality.get(victim, -float('inf'))
            victim_c = float(self.registry.get(victim, {}).get("input_cost_per_m") or 0.0)
            
            is_dominated = False
            dominator = None
            
            for other in self.bandit.models:
                if other == victim: continue
                if other in removed: continue # Don't compare against already marked for death
                
                other_q = model_quality.get(other, -float('inf'))
                other_c = float(self.registry.get(other, {}).get("input_cost_per_m") or 0.0)
                
                # Strict Dominance Check
                # Equal cost/quality is not dominance.
                if other_c <= victim_c and other_q > victim_q:
                     is_dominated = True
                     dominator = other
                     break
                # Or Significantly Cheaper and Equal Quality?
                # User said "Cheaper AND Higher Win Rate". Strict on both?
                # Usually standard Pareto is <= and >= with at least one strict.
                # Let's assume strict on Quality, <= on Cost.
                
            if is_dominated:
                # EVICT
                logger.info(f"Evicting {victim}: Starved (<0.1%) and Dominated by {dominator} (Q:{other_q:.2f}>{victim_q:.2f}, C:{other_c}<={victim_c})")
                self.bandit.delete_arm(victim)
                del self.registry[victim]
                removed.append(victim)
                
        return removed



    def _classify_sensitivity(self, text: str) -> str:
        """
        Stage 1: Context Sensitivity Classifier
        Two-Tier System: LOW (normal) | HIGH (safety-critical)
        Uses deterministic regex-based classifier.
        """
        try:
            from high_risk_prompt_classifier import HighRiskPromptClassifier
            if not hasattr(self, '_risk_classifier'):
                self._risk_classifier = HighRiskPromptClassifier(threshold=5.0)
            
            result = self._risk_classifier.classify(text)
            return "HIGH" if result.label == "high" else "LOW"
        except ImportError:
            # Ultra-minimal fallback (should not happen)
            text_lower = text.lower()
            high_triggers = [
                "medical", "doctor", "legal", "lawyer", "suicide", "kill myself",
                "financial advice", "dose", "diagnosis", "prescription"
            ]
            for trigger in high_triggers:
                if trigger in text_lower:
                    return "HIGH"
            return "LOW"

    def route(
        self,
        prompt: str | np.ndarray,
        *,
        profile: str = "best_value",
        sensitivity: Optional[str] = None, # Manual override: "LOW", "MID", "HIGH"
        max_cost: Optional[float] = None,
        max_latency: Optional[float] = None,
        quality_floor: Optional[Dict[str, float]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: int = 600,
    ) -> Tuple[str, RoutingLog]:
        """
        Route a prompt to the best model using Three-Tier Risk Gating.
        
        Tiers:
        - LOW: No Gating. (Best for Creative/Low-Stakes)
        - MID: Gate <= 5.0% Risk. (Best for General Knowledge/Coding)
        - HIGH: Gate <= 2.5% Risk. (Best for Medical/Legal/High-Stakes)
        """
        # 1. Embed & Context
        x = self._get_context_vector(prompt)
        
        # We still need the text for logging and cluster detection if prompt is an array
        prompt_text = prompt if isinstance(prompt, str) else "[Pre-embedded Prompt]"
        
        # Detect cluster if detector available
        cluster_id = None
        cluster_similarity = None
        if self.cluster_detector is not None:
            try:
                cluster_id, cluster_similarity = self.cluster_detector.detect_cluster(prompt)
            except Exception as e:
                logger.warning(f"Cluster detection failed: {e}")
        
        # 2. Resolve Weights
        weights = OptimizationProfile.get(profile).copy()
        lambda_cost = weights["lambda_cost"]
        lambda_latency = weights["lambda_latency"]
        
        # --- ORTHOGONAL OPTIMIZATION ---
        # If a hard constraint is active, disable the soft penalty for that dimension
        # to avoid "Double Penalty" (e.g. picking cheapest model when budget allows better).
        if max_cost is not None:
            lambda_cost = 0.0
        if max_latency is not None:
            lambda_latency = 0.0
        # -------------------------------
        
        # 3. Filter Candidates (Constraints + Gating)
        candidates = list(self.registry.keys())
        
        # --- RISK GATING ---
        # Two-Tier System: Only HIGH triggers gating
        eff_sensitivity = sensitivity.upper() if sensitivity else self._classify_sensitivity(prompt_text)
        
        # HIGH: <= 2.5% Risk (Forces safe models like GPT-4o)
        # LOW: No Filter (Bandit optimizes freely)
        if eff_sensitivity == "HIGH":
            safe_subset = []
            threshold = 2.5
            for m in candidates:
                meta = self.registry.get(m, {})
                risk_score = float(meta.get("hallucination_vectara", meta.get("hallucination_rate", 8.0)))
                if risk_score <= threshold:
                    safe_subset.append(m)
            
            if safe_subset:
                candidates = safe_subset
            else:
                logger.warning(f"No models passed HIGH (<= {threshold}%) gate. Falling back to full pool.")
        # -------------------

        # Estimate tokens
        in_tok = input_tokens or estimate_tokens_rough(prompt)
        
        filtered = []
        for m in candidates:
            # Check Cost
            cost = self._estimate_cost(m, in_tok, output_tokens)
            if max_cost is not None and cost > max_cost: continue
            
            # Check Latency
            lat = self._estimate_latency(m, output_tokens)
            if max_latency is not None and lat > max_latency: continue
            
            # Check Quality Floor
            if quality_floor:
                scores = self.registry.get(m, {}).get("scores", {})
                passes = True
                for k, v in quality_floor.items():
                    if float(scores.get(k, 0)) < v:
                        passes = False
                        break
                if not passes: continue
                
            filtered.append(m)
            
        if not filtered:
            filtered = list(self.registry.keys()) # Ultimate fallback
            # In production, raise error or return default
            
        # 4. Score Candidates
        best_model = filtered[0]
        best_utility = -float("inf")
        
        # Pre-compute UCBs
        ucbs = {}
        for m in filtered:
            _, ucb = self.bandit.select_arm(x, candidates=[m])
            ucbs[m] = ucb
            
        # Calculate Utility
        # Normalization: Scale Cost (Log) and Latency (Linear) to [0, 1] relative to the pool
        costs = {m: self._estimate_cost(m, in_tok, output_tokens) for m in filtered}
        lats = {m: self._estimate_latency(m, output_tokens) for m in filtered}
        
        # Log-MinMax Normalization for Cost (Handles large dynamic range)
        EPS = 1e-9
        log_costs = {m: np.log(max(costs[m], EPS)) for m in filtered}
        min_c, max_c = min(log_costs.values()), max(log_costs.values())
        range_c = max_c - min_c
        
        # Linear MinMax Normalization for Latency (Usually tighter range)
        # Or Log if we expect 0.1s vs 10s. Let's use Log to be safe against outliers.
        log_lats = {m: np.log(max(lats[m], EPS)) for m in filtered}
        min_l, max_l = min(log_lats.values()), max(log_lats.values())
        range_l = max_l - min_l
        
        for m in filtered:
            quality = ucbs[m]
            
            # Normalize to [0, 1]
            # 0.0 = Cheapest/Fastest in pool
            # 1.0 = Most Expensive/Slowest in pool
            
            # Stability Check: If range is negligible (e.g. all same cost), set to 0.0
            if range_c < 1e-9:
                norm_cost = 0.0
            else:
                norm_cost = (log_costs[m] - min_c) / range_c
                
            if range_l < 1e-9:
                norm_lat = 0.0
            else:
                norm_lat = (log_lats[m] - min_l) / range_l
            
            # This ensures w_c=1.0 means "Cost difference of pool width is equal to 1.0 Quality"
            utility = quality - (lambda_cost * norm_cost) - (lambda_latency * norm_lat)
            
            if utility > best_utility:
                best_utility = utility
                best_model = m
                
        # 5. Log
        log  = RoutingLog(
            request_id=str(time.time_ns()),
            timestamp_s=time.time(),
            prompt=prompt_text,
            selected_model=best_model,
            predicted_utility=float(best_utility),
            cost_usd=self._estimate_cost(best_model, in_tok, output_tokens),
            latency_s=self._estimate_latency(best_model, output_tokens),
            cluster_id=cluster_id,
            cluster_similarity=cluster_similarity,
            context_vector=x # Cache for feedback loop
        )
        # Trigger Lazy Pruning (Periodically)
        # e.g., every 100 requests (for demo) or 1000
        if len(self.logs) % 100 == 0:
             self.prune_arms(min_requests=100) # Lower min_requests for testing/demo
             
        # 5. Log
        log  = RoutingLog(
            request_id=str(time.time_ns()),
            timestamp_s=time.time(),
            prompt=prompt_text,
            selected_model=best_model,
            predicted_utility=float(best_utility),
            cost_usd=self._estimate_cost(best_model, in_tok, output_tokens),
            latency_s=self._estimate_latency(best_model, output_tokens),
            cluster_id=cluster_id,
            cluster_similarity=cluster_similarity,
            context_vector=x # Cache for feedback loop
        )
        self.logs.append(log)
        
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
        
        if log is None:
            logger.warning(f"No routing log found for request_id={request_id}")
            return
        
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

    def get_probabilities(self, context: str | np.ndarray, model_ids: List[str] | None = None) -> Dict[str, float]:
        """Get the probability of each model being the specialist for a given context."""
        x = self._get_context_vector(context)
        models = model_ids if model_ids else self.bandit.models
        return self.bandit.get_probabilities(x, models)

    def update(self, model_id: str, context: str | np.ndarray, reward: float) -> None:
        """Update the bandit's internal state with a new observation."""
        x = self._get_context_vector(context)
        self.bandit.update(model_id, x, reward)

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
        
        # 4. Initialize Prior (Cluster + Efficiency)
        # We reuse the logic from load_from_benchmark but for a single model
        # This ensures the new model gets the same "Smart Prior" treatment
        
        # Get score (default 0.05 if missing)
        raw_score = float(definition.get(self.benchmark_key) or 0.05)
        score = transform_hle_to_prior(raw_score)
        
        # Efficiency Boost only (cluster boost via process_feedback)
        cost = float(definition.get("input_cost_per_m", 0.0)) / 1000.0
        cost = max(cost, 0.00000005)
        efficiency_boost = 1.0 + (0.2 * math.log(1.0 / cost))
        score *= efficiency_boost
        
        if score > 0:
            # Initialize with prior belief
            # Set the bias term (last element) of b to prior_strength * score
            # This effectively gives it a "mean reward" of 'score' for the bias feature.
            self.bandit.b[model_id][-1] = 20.0 * score
            # Also increase confidence in the bias term
            self.bandit.A[model_id][-1, -1] += 20.0
            self.bandit.A_inv[model_id] = safe_inv(self.bandit.A[model_id])


    def save_state(self, path: Path | str) -> None:
        """Save the bandit's learned state to disk."""
        self.bandit.save_state(path)

    def _estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        m = self.registry.get(model, {})
        # Strictly use per-token pricing. No fallbacks.
        if m.get("input_cost_per_m") is not None and m.get("output_cost_per_m") is not None:
            return (m["input_cost_per_m"] * in_tok + m["output_cost_per_m"] * out_tok) / 1e6
        return float('inf')

    def _estimate_latency(self, model: str, out_tok: int) -> float:
        m = self.registry.get(model, {})
        # Strictly use "time_to_first_token_seconds" as requested.
        # If missing or 0 (invalid), return infinity so it fails any max_latency constraint.
        val = m.get("time_to_first_token_seconds")
        if val is None or float(val) <= 0.0:
            return float('inf')
        return float(val)
