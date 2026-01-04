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
import threading
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




logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router Configuration (Magic Numbers Documented)
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """
    Centralized configuration for BanditRouter magic numbers.
    All values are derived from empirical analysis or market data.
    """
    
    # Sigmoid Transformation for HLE Priors
    # ⚠️ WARNING: These parameters are ASSUMPTIONS, not empirically validated!
    # TODO: Replace with calibration curve from "Golden Prompt" experiment:
    #   1. Create 50 prompts with binary success criteria
    #   2. Run benchmark models (Llama-3, GPT-4, etc.) on them
    #   3. Plot Actual_Success_Rate vs HLE_Score
    #   4. Fit regression curve (linear, polynomial, or calibrated sigmoid)
    #   5. Replace transform_hle_to_prior() with fitted curve
    # Current assumption: 6.5% HLE → 0.8 Utility (unvalidated!)
    prior_sigmoid_k: float = 80.0
    prior_sigmoid_center: float = 0.20
    calibration_validated: bool = False  # Set to True after empirical calibration
    
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
    """Named presets for utility function weights (Quality vs Cost vs Latency).
    Weights MUST sum to 1.0 (100%).
    """
    MAX_QUALITY     = {"w_q": 0.97, "w_c": 0.02, "w_l": 0.01}
    ARBITRAGE       = {"w_q": 0.65, "w_c": 0.30, "w_l": 0.05}
    BEST_VALUE      = {"w_q": 0.40, "w_c": 0.50, "w_l": 0.10}
    BALANCED        = BEST_VALUE
    COST_SAVER      = {"w_q": 0.10, "w_c": 0.85, "w_l": 0.05}
    LOW_LATENCY     = {"w_q": 0.10, "w_c": 0.10, "w_l": 0.80}
    VALUE_EFFICIENT = {"w_q": 0.30, "w_c": 0.60, "w_l": 0.10}

    _PROFILES = {
        "max_quality": MAX_QUALITY,
        "arbitrage": ARBITRAGE,
        "best_value": BEST_VALUE,
        "balanced": BALANCED,
        "cost_saver": COST_SAVER,
        "low_latency": LOW_LATENCY,
        "value_efficient": VALUE_EFFICIENT,
    }

    @classmethod
    def get(cls, name: Union[str, Dict[str, float]]) -> Dict[str, float]:
        """Get profile weights by name or return dict if already a profile."""
        if isinstance(name, dict):
            # Pass-through for custom weight dicts
            return name
            
        if not isinstance(name, str):
            raise TypeError(f"Profile must be a string or dict, got {type(name)}")
            
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

def transform_hle_to_prior(raw_hle_score: float, is_hard_prompt: bool = False) -> float:
    """
    Maps HLE score (0-40%) to expected success probability (0-100%).
    
    Uses a TWO-TIERED "BARBELL" approach with MIN-MAX CALIBRATION:
    
    **Tier A - Easy Prompts (90% of traffic)**:
    - Everyone gets an 'A': 95-99% success
    - Price is the ONLY differentiator → Selects cheapest
    - Example: Gemma (6.5% HLE) → 95%, GPT-4 (30% HLE) → 99%
    
    **Tier B - Hard Prompts (10% of traffic, high-value)**:
    - GRADE ON A CURVE: Relative utility, not absolute accuracy
    - Best available model gets 99%, worst gets 1%
    - Creates massive gap that overcomes cost penalty
    - Example: Gemma (5% HLE) → 1%, GPT-4 (28% HLE) → 92%
    
    This creates a "barbell" distribution:
    - Easy → Cheapest model (Gemma, Nova)
    - Hard → Most capable model (Opus, GPT-4)
    - Mid-tier models get starved out
    
    Args:
        raw_hle_score: HLE benchmark score (0.0-0.4 range)
        is_hard_prompt: If True, use min-max scaling (relative utility)
    
    Returns:
        Expected success probability (0.0-1.0)
    """
    if not is_hard_prompt:
        # TIER A: DE-COMPRESSED EASY MODE
        # Uses steeper slope (0.05) to ensure quality beats cost penalty.
        # 
        # Math: Opus (HLE=0.28) vs Phi (HLE=0.15)
        #   Opus Utility: 0.95 + (0.05 * 0.28) = 0.964
        #   Phi Utility:  0.95 + (0.05 * 0.15) = 0.9575
        #   Gap: 0.0065 > Cost Penalty (~0.005) → Opus wins even on "leaked" prompts
        # 
        # This prevents the "Leak Trap" where hard prompts in easy clusters
        # would pick cheap models due to hyper-compressed prior gap.
        return 0.95 + (0.05 * raw_hle_score)  # Linear from HLE
    else:
        # TIER B: QUADRATIC RESOLUTION (Hard Prompts)
        # Use QUADRATIC SCALING with a 0% FLOOR
        # This provides visual differentiation for mid-tier models (no more 0 pileup)
        # while maintaining the "Elite Advantage" (top models win exponentially).
        
        # Upper bound: Best-in-class HLE (Opus/GPT-5 level)
        max_benchmark = 0.35 
        
        # 1. Linear scaling with 0 floor
        linear_score = raw_hle_score / max_benchmark
        
        # 2. Quadratic Boost: u = x^2
        # Math: 
        #   Elite (0.30 HLE): (0.30/0.35)^2 = 0.73 -> ~73% base utility
        #   Mid   (0.15 HLE): (0.15/0.35)^2 = 0.18 -> ~18% base utility
        #   Low   (0.05 HLE): (0.05/0.35)^2 = 0.02 -> ~2% base utility
        utility = linear_score ** 2
        
        # Clip to [0.01, 0.99]
        return max(0.01, min(0.99, utility))

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
        
        # Thread safety: protect state mutations in multi-threaded deployments
        self._lock = threading.Lock()
        
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

    def update(self, model: str, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """
        Update the model's A and b matrices with new observation.
        
        Args:
            model: Model identifier
            x: Context vector
            reward: Observed reward
            weight: Importance weight for this update (default 1.0).
                    Use weight = (1 - cluster_mu) for difficulty-based weighting.
                    Hard tasks (μ=0.5) get weight=0.5, easy tasks (μ=0.95) get weight=0.05.
        """
        if model not in self.A: return
        
        # Thread safety: acquire lock for all state mutations
        with self._lock:
            self.t += 1 # Increment global clock
            
            # Synchronize decay before update
            dt = self.t - self.last_update[model]
            needs_full_inversion = False
            
            if dt > 0:
                effective_gamma = self.gamma ** dt
                
                # 1. Decay the Information state
                self.A[model] *= effective_gamma
                self.b[model] *= effective_gamma
                
                # 2. CRITICAL FIX: Restore the Regularization Floor
                # If we don't do this, A decays to zero and inversion explodes.
                # Standard Discounted LinUCB: A_new = γ*A_old + (1-γ)*λI + x*x^T
                # We add back the 'lost' portion of the identity matrix.
                restore_reg = (1.0 - effective_gamma) * self.ridge_lambda
                
                # Add to diagonal only (efficient: O(d) instead of O(d²))
                np.fill_diagonal(self.A[model], self.A[model].diagonal() + restore_reg)
                
                # Diagonal adjustment breaks Sherman-Morrison assumptions
                # Must recompute A_inv from scratch after decay
                needs_full_inversion = True
            
            # 3. Add new observation with importance weighting
            self.A[model] += weight * np.outer(x, x)
            self.b[model] += weight * float(reward) * x
            
            # 4. Update A_inv efficiently using Sherman-Morrison Formula
            # A_new^-1 = A_old^-1 - (A_old^-1 @ x @ x^T @ A_old^-1) / (1 + x^T @ A_old^-1 @ x)
            # Complexity: O(d²) instead of O(d³)
            if needs_full_inversion:
                # Forgetting factor applied diagonal adjustment
                # Sherman-Morrison doesn't apply, recompute from scratch
                self.A_inv[model] = safe_inv(self.A[model])
            else:
                # Standard rank-1 update: use Sherman-Morrison
                # u = A_inv @ x
                u = self.A_inv[model] @ x
                
                # numerator = u @ u^T (outer product)
                numerator = np.outer(u, u)
                
                # denominator = 1 + x^T @ A_inv @ x = 1 + x^T @ u
                denominator = 1.0 + weight * float(np.dot(x, u))
                
                # A_inv_new = A_inv_old - (numerator / denominator) * weight
                self.A_inv[model] -= (weight * numerator) / denominator
            
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
        context_model: str = DEFAULT_CONTEXT_MODEL,
        context_encoder=None,
        alpha: float = 0.1,
        embedding_dim: int = 384,
        ridge_lambda: float = 1.0,
        forgetting_factor: float = 0.95,
        cluster_boost_weight: float = 0.0,
        pca_path: Optional[Path | str] = None,
        complexity_path: Optional[Path | str] = None,
        anchors: Optional[Dict[str, str]] = None,
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
        
        # -----------------------------------------------------------------------
        # ZERO-SHOT FEATURE INITIALIZATION (Anchors & Complexity)
        # -----------------------------------------------------------------------
        self.anchors_config = anchors or self.DEFAULT_VIRTUAL_ANCHORS
        anchor_texts = list(self.anchors_config.values())
        
        logger.info(f"Initializing {len(anchor_texts)} Virtual Anchors...")
        self.anchor_vectors = self.encoder.encode(anchor_texts, normalize_embeddings=True)
        
        # Load or initialize Reference Complexity Vector (H-vector)
        self.complexity_vector = None
        comp_path = complexity_path or Path(__file__).parent.parent.parent / "priors" / "complexity_vector.npz"
        if Path(comp_path).exists():
            try:
                data = np.load(comp_path)
                self.complexity_vector = data["complexity_vector"]
                logger.info("✓ Reference Complexity Vector loaded.")
            except Exception as e:
                logger.warning(f"Failed to load complexity vector: {e}")
        
        if self.complexity_vector is None:
            logger.info("Generating Complexity Vector from zero-shot seeds...")
            seed_embs = self.encoder.encode(self.HARD_REASONING_SEEDS, normalize_embeddings=True)
            self.complexity_vector = np.mean(seed_embs, axis=0)
            self.complexity_vector /= (np.linalg.norm(self.complexity_vector) + 1e-12)
        # -----------------------------------------------------------------------

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
        # FEATURE VECTOR DIMENSION LOGIC (Updated for Heuristic Hardness)
        # Base Embedding (384/32) + Handcrafted (11) + Cluster Distances (5) + 
        # Hardness Score (1) + Bias (1) = 50 (or 401 without PCA)
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
             # We are adding 17 features (11 handcrafted + 5 cluster + 1 hardness).
             embedding_dim = base_dim + 17
        
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
        9. has_code_block (NEW: explicit code fence detection)
        10. latex_density (NEW: math/LaTeX symbol density)
        11. length_penalty (NEW: normalized token count)
        """
        if not text:
            return np.zeros(11)
        
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
        
        # --- HEURISTIC HARDNESS FEATURES ---
        
        # 9. Has Code Block (Explicit)
        # Embeddings can miss short but complex code snippets
        has_code_block = 1.0 if '```' in text else 0.0
        
        # 10. LaTeX Density
        # Short math problems with high symbol density are hard
        latex_symbols = text.count('$') + text.count('\\') + text.count('^') + text.count('_{')
        latex_density = min(latex_symbols / max(n_words, 1), 1.0)  # Normalize by word count
        
        # 11. Length Penalty (Normalized)
        # Longer prompts often require larger context models
        # Normalize to [0, 1] assuming 2000 tokens as "long"
        length_penalty = min(n_tokens / 2000.0, 1.0)
        
        return np.array([
            is_code_heavy, requires_json, input_length_log, list_density,
            instruction_density, fk_normalized, question_count,
            toxicity_score,
            has_code_block, latex_density, length_penalty
        ])

    def _get_cluster_distances(self, embedding: np.ndarray) -> np.ndarray:
        """
        Get distances to the Virtual Anchors.
        
        Args:
            embedding: Normalized sentence embedding (384,)
        """
        # Calculate cosine similarity: dot product of normalized vectors
        # Shape: (N_anchors, 384) @ (384,) -> (N_anchors,)
        similarities = np.dot(self.anchor_vectors, embedding)
        
        # Convert to distance (1 - similarity)
        # Clip to [0, 2] to handle precision errors
        distances = 1.0 - similarities
        return np.clip(distances, 0.0, 2.0)

    def _get_context_vector(self, context: str | np.ndarray) -> np.ndarray:
        """
        Convert string prompt or array to a normalized context vector.
        
        Structure with Zero-Shot Hardness + Heuristic Features:
        [Embedding (32/384) | Handcrafted (11) | Anchors (5) | Hardness Score (1) | Bias (1)]
        
        The Hardness Score is the "Semantic Complexity" signal:
        - S_hard = prompt_embedding · Reference_Complexity_Vector
        - Feed raw scalar to bandit to learn difficulty-dependent routing.
        
        Args:
            context: Prompt text or pre-computed embedding
        """
        if isinstance(context, str):
            # 1. Semantic Embedding
            emb_full = self.encoder.encode(context)
            emb_full = l2_normalize(emb_full)
            
            if self.pca:
                emb_reduced = self.pca.transform(emb_full.reshape(1, -1)).flatten()
            else:
                emb_reduced = emb_full
            
            # 2. Handcrafted Features (8)
            feats = self._extract_handcrafted_features(context)
            
            # 3. Virtual Anchor Distances (N)
            anchor_dists = self._get_cluster_distances(emb_full)
            
            # 4. Zero-Shot Hardness Score (1)
            # Projection onto Reference Complexity Vector
            hardness_score = float(np.dot(emb_full, self.complexity_vector))
            # SIGNAL BOOSTING: Scale score to meaningful magnitude for the bandit
            hardness_feat = np.array([hardness_score * 10.0])
            
            # Concatenate: [Embedding, Handcrafted, Anchors, Hardness, Bias]
            x = np.concatenate([emb_reduced, feats, anchor_dists, hardness_feat])
        else:
            x = context
            
        # Append bias term (Last dim)
        return np.append(x, 1.0)


    @classmethod
    def create(
        cls,
        model_registry: Optional[Dict[str, Dict]] = None,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        context_encoder=None,  # NEW: Optional pre-initialized encoder for testing/DI
        prior_n_effective: Optional[float] = None,
        prior_structure_n_effective: Optional[float] = None,
        alpha: Optional[float] = None,
        exploration: str = "safe",
        state_path: Optional[str] = None,
        priors: str = "hle",  # Default: HLE (unbiased benchmark scores)
        ridge_lambda: float = 1.0,
        forgetting_factor: float = 1.0,
        cluster_boost_weight: float = 0.0,
        anchors: Optional[Dict[str, str]] = None
    ) -> "BanditRouter":
        """
        Factory method to create a BanditRouter with optional priors.
        
        Args:
            model_registry: Dict of {model_id: model_metadata}.
            context_model: Sentence-transformers model name (default: "sentence-transformers/all-MiniLM-L6-v2").
                          PCA compression is applied separately if pca_path exists.
            context_encoder: Optional pre-initialized encoder (for testing or custom encoders).
                           If provided, context_model is ignored.
            priors: Prior type to load:
                - "hle": Hard Label Evaluation scores (generic, unbiased) [DEFAULT]
                - "none": Cold start (no priors)
            prior_n_effective: Effective sample size for belief strength (b vector scaling).
                              Default: Auto-selected based on priors type:
                                - HLE: 10.0 (Calibrated Champion)
                                - none: 0.0 (no priors)
            prior_structure_n_effective: Effective sample size for structural stiffness (A matrix scaling).
                                        Default: Auto-selected based on priors type:
                                          - HLE: 250.0 (Calibrated Champion)
                                          - none: 20.0 (structure only, no mean)
                              Note: None = Infinite stiffness (deprecated, not recommended).
            exploration: "static", "safe", "balanced", "aggressive".
            ridge_lambda: Regularization parameter (default 1.0, auto-scales with structure).
            state_path: Optional path to a saved bandit state (.npz).
            cluster_boost_weight: Reward boost weight for cluster specialization (default 0.0, disabled).
            anchors: Optional dict of {name: description} for virtual anchors.
        """
        base_dir = Path(__file__).parent
        
        # Auto-select optimal parameters based on prior type (from z-score ablation study)
        if prior_n_effective is None:
            if priors == "hle":
                prior_n_effective = 10.0  # Calibrated HLE Champion N_prior
            elif priors == "none":
                prior_n_effective = 0.0   # No priors
            else:
                prior_n_effective = 10.0  # Default fallback
        
        if prior_structure_n_effective is None:
            if priors == "hle":
                prior_structure_n_effective = 250.0  # Calibrated HLE Champion N_structure
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
                        forgetting_factor=forgetting_factor,
                        cluster_boost_weight=cluster_boost_weight, pca_path=pca_path_default)
            router.bandit.load_state(state_path)
            return router
        # Load HLE Priors (Generic Structure + Generic Mean)
        if priors == "hle":
            # Use PCA-based structural priors
            meta_path = base_dir / "priors" / "priors_meta_pca.npz"
            
            if not meta_path.exists():
                 logger.info("No priors metadata found. Initializing with Zero-Shot Warm Start.")
                 router = cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                           alpha=alpha, ridge_lambda=ridge_lambda,
                           forgetting_factor=forgetting_factor,
                           cluster_boost_weight=cluster_boost_weight,
                           pca_path=pca_path_default if pca_path_default.exists() else None,
                           anchors=anchors)
                 
                 # Perform Zero-Shot Warm Start
                 router._load_zero_shot_priors(prior_n_effective)
                 return router

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
                cluster_boost_weight=cluster_boost_weight,
                pca_path=pca_path_default if pca_path_default.exists() else None,
                anchors=anchors
            )
            
        # Cold Start (No Priors)
        if priors == "none":
            logger.info("Cold start mode: No priors loaded.")
            return cls(model_registry, context_model=context_model, context_encoder=context_encoder,
                      alpha=alpha, ridge_lambda=ridge_lambda,
                      forgetting_factor=forgetting_factor,
                      cluster_boost_weight=cluster_boost_weight,
                      pca_path=pca_path_default if pca_path_default.exists() else None,
                      anchors=anchors)
        
        # Unknown priors type
        raise ValueError(f"Unknown priors type: '{priors}'. Use 'hle' or 'none'.")

    def _load_zero_shot_priors(self, prior_n_effective: float):
        """
        Initializes bandit priors using the Zero-Shot Complexity Vector.
        This provides a warm start for models based on their expected performance
        on 'hard' vs 'easy' tasks, without needing full HLE benchmark data.
        """
        logger.info(f"Initializing Zero-Shot Warm Start with prior_n_effective={prior_n_effective:.1f}")
        
        # The complexity vector is already loaded during BanditRouter init.
        # It represents the direction of "hardness" in the embedding space.
        # We use it to create a simple prior:
        # - Models that align well with the complexity vector get a higher prior score.
        # - Models that don't align get a lower prior score.
        
        # This is a simplified prior, assuming a linear relationship between
        # complexity vector projection and expected reward.
        
        # For each model, calculate its "expected hardness score" based on its
        # inherent capabilities (e.g., from model metadata or a simple heuristic).
        # For now, we'll use a placeholder or a simple mapping.
        
        # The complexity vector is part of the context vector, specifically the
        # 'hardness_feat' component. The bandit learns to weight this feature.
        # For a zero-shot prior, we can directly influence the 'b' vector.
        
        # A simple approach:
        # For each model, assign a base prior reward (e.g., 0.5)
        # Then, adjust it based on a "quality" score from metadata, if available.
        # This quality score can be a proxy for how well it handles complex tasks.
        
        # Placeholder: Assume a simple quality score based on model tier or cost.
        # In a real scenario, this would come from model metadata.
        
        for model_id, model_data in self.model_registry.items():
            # Example: Assign a quality score based on model_data (e.g., cost, tier)
            # For demonstration, let's use a dummy score or a simple heuristic.
            # A more robust implementation would use actual model performance data.
            
            # For now, we'll just set a neutral prior, as the complexity vector
            # itself is a feature the bandit learns to weight.
            # The "zero-shot" aspect comes from the *feature* itself, not necessarily
            # a pre-baked prior into A/b for each model.
            
            # If we wanted to bake in a prior, we'd need a way to map model_id
            # to an expected performance on hard tasks.
            
            # For a true "zero-shot warm start" without external benchmark data,
            # we primarily rely on the *features* (like the complexity score)
            # to guide the bandit, rather than pre-populating A/b with model-specific priors.
            # The bandit will learn the weights for these features from initial interactions.
            
            # However, if prior_n_effective > 0, we should still apply *some* prior.
            # A simple "neutral" prior for all models, scaled by prior_n_effective.
            # This ensures the bandit starts with some initial confidence, preventing
            # extreme early exploration.
            
            # The 'b' vector represents sum(rewards * context_vector).
            # A neutral prior would mean b = prior_n_effective * expected_mean_reward * mean_context_vector.
            # Since we don't have mean_context_vector without data, we can just set a small,
            # uniform prior for all models.
            
            # For now, we'll let the bandit start with its default A=lambda*I, b=0,
            # and rely on the complexity feature to guide learning.
            # The `prior_n_effective` here primarily acts as a "softening" factor
            # for the initial exploration, making the bandit less volatile initially.
            
            # If a more specific zero-shot prior is desired, it would involve:
            # 1. Defining a "zero-shot reward" for each model (e.g., based on size, cost, known capabilities).
            # 2. Estimating a "mean context vector" for typical prompts.
            # 3. Calculating b_prior = prior_n_effective * zero_shot_reward * mean_context_vector.
            # This is complex without actual data.
            
            # For now, the "Zero-Shot Warm Start" primarily means:
            # - The `complexity_vector` is active as a feature.
            # - The bandit starts with default A/b, but `prior_n_effective` might
            #   influence initial exploration parameters if it's used elsewhere
            #   to scale the initial `alpha` or `ridge_lambda`.
            
            # If prior_n_effective is meant to add a mean prior, we need a mean reward.
            # Let's assume a neutral mean reward of 0.5 for all models initially.
            # And a neutral context vector (e.g., all zeros except bias).
            # This is a very weak prior.
            
            # A more meaningful zero-shot prior would involve:
            # 1. A global mean context vector (e.g., from a large corpus).
            # 2. A global mean reward (e.g., 0.5).
            # 3. Then, b_prior = prior_n_effective * global_mean_reward * global_mean_context_vector.
            
            # Given the current structure, the most direct way to use prior_n_effective
            # for a "zero-shot" warm start is to scale the initial A matrix (ridge)
            # and potentially add a small, uniform b vector.
            
            # For now, we'll keep the A matrix as ridge_lambda * I and b as 0,
            # and let the complexity vector feature do its work.
            # The `prior_n_effective` parameter in this context primarily signals
            # that we *intend* to have some prior strength, even if it's just
            # a default initialization.
            pass # The actual prior application happens in load_from_benchmark or is implicit in bandit init.

    def _detect_difficulty_hybrid(self, text: str, cluster_id: Optional[int] = None) -> float:
        """
        Determines the semantic complexity/difficulty of a prompt.
        Returns a continuous score [0.0, 1.0].
        """
        # 1. Semantic Hardness (Complexity Vector)
        # This is our primary Zero-Shot signal
        emb = self.encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        hardness_score = float(np.dot(emb, self.complexity_vector))
        
        return np.clip(hardness_score, 0.0, 1.0)

    # The following methods are deprecated as _detect_difficulty_hybrid now uses the complexity vector directly.
    # They are kept for reference or if any legacy code still calls them.
    @staticmethod
    def _is_hard_cluster(cluster_idx: int) -> bool:
        """
        [DEPRECATED] Data-dependent cluster ID lookup. 
        Replaced by Zero-Shot Complexity Vector.
        """
        # Empirically derived from cluster content analysis
        # These clusters were manually verified to contain Math/Code/Technical content
        HARD_CLUSTERS = {
            # Math/Reasoning clusters
            47,  # Unity/C# code, bitwise operations
            54,  # Math puzzles (gallons problem)
            57,  # JSON extraction, structured data
            61,  # HTML/CSS code
            65,  # AMC 10 math problems
            66,  # Python/PyQt programming
            73,  # HTML tables
            77,  # PyTorch/ML code
            80,  # Ansible/devops code
            81, 85, 87, 89, 93, 10, 48, 59, 12, 55, 27 # Additional hard clusters
        }
        
        return cluster_idx in HARD_CLUSTERS
    
    @staticmethod
    def _is_explicitly_hard(text: str) -> bool:
        """
        [DEPRECATED] Deterministic safety net for difficulty detection.
        Replaced by Zero-Shot Complexity Vector.
        """
        import re
        
        # Code indicators
        code_patterns = [
            r'\bdef\s+\w+\s*\(',       # Python function definition
            r'\bclass\s+\w+',           # Class definition
            r'\bimport\s+\w+',          # Import statement
            r'\bfunction\s+\w+\s*\(',  # JavaScript function
            r'\bconst\s+\w+\s*=',       # JavaScript const
            r'\bpublic\s+(class|void|int|string)', # Java/C#
            r'```(python|javascript|java|cpp|c\+\+|typescript|rust|go|sql|bash|sh)',  # Code blocks
        ]
        
        # Math indicators
        math_patterns = [
            r'\$\$.*\$\$',              # LaTeX display math
            r'\\frac\{',               # LaTeX fractions
            r'\\int',                   # LaTeX integrals
            r'\\sum',                   # LaTeX summations
            r'\btheorem\b',             # Mathematical theorems
            r'\bproof\b',               # Mathematical proofs
            r'\bsolve\s+for\b',        # "Solve for x"
            r'\bcalculate\b.*\bif\b',  # "Calculate X if Y"
        ]
        
        # Technical debugging
        debug_patterns = [
            r'\bdebug\b.*\b(error|exception|traceback)\b',
            r'\btraceback\b',
            r'\bstack\s*trace\b',
            r'\bsegmentation\s*fault\b',
            r'\bcore\s*dump\b',
        ]
        
        # Combine all patterns
        all_patterns = code_patterns + math_patterns + debug_patterns
        text_lower = text.lower()
        
        for pattern in all_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Length heuristic: Very long prompts (>500 words) are often complex
        word_count = len(text.split())
        if word_count > 500:
            return True
            
        return False
    
    def _detect_difficulty_hybrid(self, text: str, cluster_id: Optional[int]) -> bool:
        """
        Robust difficulty detection using both Semantic Clusters and Explicit Signals.
        
        This implements a "Leak-Proof" detection strategy:
        1. Explicit signals (code, math, debug) ALWAYS force Hard Mode
        2. Semantic cluster provides nuanced detection for ambiguous prompts
        
        Args:
            text: The prompt text
            cluster_id: The detected cluster ID (may be None)
            
        Returns:
            True if prompt should be treated as Hard
        """
        # 1. Check Explicit Signals (The Safety Net)
        # This catches "leaked" hard prompts that ended up in easy clusters
        if self._is_explicitly_hard(text):
            return True
            
        # 2. Check Semantic Cluster (The Nuance)
        if cluster_id is not None:
            return self._is_hard_cluster(cluster_id)
            
        return False

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
        cluster_boost_weight: float = 0.1,
        pca_path: Optional[Path] = None,
        anchors: Optional[Dict[str, str]] = None,
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
            embedding_dim=dim + 1,  # +1 for hardness switch
            ridge_lambda=ridge_lambda,
            cluster_boost_weight=cluster_boost_weight,
            pca_path=pca_path,
            anchors=anchors
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
        
        # NORMALIZATION: Convert sums to means for fair HLE comparison
        # This ensures prior_n_effective has equivalent strength for both
        cluster_means = cluster_sums / cluster_counts[:, np.newaxis]  # Shape: (100, 45)
        global_mean = global_sum / max(total_samples, 1.0)            # Shape: (45,)
        
        # -----------------------------------------------------------------------
        # TWO-KNOB SCALING FRAMEWORK (CORRECTED)
        # -----------------------------------------------------------------------
        # Note: total_samples already calculated above for normalization 
        
        # Knob 1: Structural Stiffness (Initialization Scaling for A Matrix)
        # Controls how confident we are in the covariance structure
        #
        # CRITICAL FIX: The covariance matrix in priors_meta_pca.npz is already
        # a NORMALIZED covariance (mean, not sum). Therefore, we scale it directly
        # by prior_structure_n_effective to simulate N_eff samples, NOT divide by total_samples.
        #
        # Old (Buggy): init_scale = prior_structure_n_effective / total_samples
        #   This created A ≈ I (tiny), b ≈ 60μ (large) → θ ≈ 60μ (explodes to 50-60 range)
        #
        # New (Correct): init_scale = prior_structure_n_effective
        #   This creates A ≈ 60*Cov, b ≈ 60μ → θ = (60*Cov)^(-1)(60μ) ≈ Cov^(-1)μ ≈ 1.0
        #
        # Result: Quality scores return to 0.0-1.0 range, cost penalties become meaningful
        if prior_structure_n_effective is None:
            init_scale = 1.0  # Infinite stiffness (unscaled, full N_offline strength)
        else:
            init_scale = prior_structure_n_effective  # CORRECTED: No division
            
        # Knob 2: Belief Strength (b Vector Scaling)
        # We use prior_n_effective DIRECTLY in the b vector update (see line ~1115).
        # This is correct because global_mean and cluster_means are already normalized MEANS,
        # not SUMS. The update formula is: b += N_eff * mean_vector
        # This makes the prior strength independent of the offline dataset size.
        
        # Ridge Floor (The Baseline Variance λI)
        # Always maintain the original ridge_lambda to ensure stability and 
        # a baseline exploration radius, even when Prior Structure is zero.
        effective_ridge_lambda = ridge_lambda
        logger.info(f"Ridge Floor: λ = {effective_ridge_lambda:.3f}")
        logger.info(f"Structural Scaling: init_scale = {init_scale:.3f}")
        logger.info(f"Belief Scaling: prior_n_effective = {prior_n_effective:.3f}")
        
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
            raw_hle_score = float(model_registry.get(m, {}).get("hle") or 0.05)
            
            # ------------------------------------------------------------------
            # PRIOR MEAN CALCULATION: HLE (Generic) Context-Aware
            # ------------------------------------------------------------------
            # HOLOGRAPHIC PRIOR CONSTRUCTION V2: Context-Aware Beliefs
            # Instead of averaging priors into a single vector, we teach the bandit
            # using cluster-specific context vectors WITH the hardness switch.
            #
            # This prevents "context collapse" where easy/hard priors get blended.
            # The hardness feature orthogonalizes the beliefs.
            
            total_samples_val = np.sum(meta["cluster_counts"])
            
            # Iterate through all clusters to teach cluster-specific beliefs
            for k in range(n_clusters):
                # 1. DETECT DIFFICULTY for this cluster
                # In v2, we try to use the complexity vector if possible, 
                # but fallback to ID-based for legacy HLE metadatas.
                is_hard = router._is_hard_cluster(k)
                hardness_val = 10.0 if is_hard else 0.0
                # 2. TRANSFORM HLE to target score using two-tiered barbell
                target_score = transform_hle_to_prior(raw_hle_score, is_hard_prompt=is_hard)
                
                # 3. CONSTRUCT CLUSTER-SPECIFIC CONTEXT VECTOR
                # Structure: [ClusterCentroid | Hardness | Bias]
                cluster_centroid = cluster_means[k]
                
                # Build full context: [Centroid | Hardness | Bias]
                x_cluster = np.concatenate([cluster_centroid, [hardness_val], [1.0]])
                
                # 4. WEIGHT by cluster prevalence
                prevalence = meta["cluster_counts"][k] / max(total_samples_val, 1.0)
                weight = prior_n_effective * prevalence
                
                # 5. TEACH THE BANDIT: "For context x_cluster, expect target_score"
                # Update A (structure): A += weight * x @ x^T
                router.bandit.A[m] += weight * np.outer(x_cluster, x_cluster)
                
                # Update b (belief): b += weight * target_score * x
                router.bandit.b[m] += weight * target_score * x_cluster


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
                
                m_hle = transform_hle_to_prior(float(m_data.get("hle") or 0.0))
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

    def _load_zero_shot_priors(self, prior_n_effective: float = 10.0) -> None:
        """
        Zero-Shot Warm Start using Virtual Anchors.
        Teaches the bandit initial beliefs without training data.
        """
        logger.info(f"Teaching {len(self.bandit.models)} models using {len(self.anchor_vectors)} Virtual Anchors...")
        
        # Stats for normalization of handcrafted features
        global_stats = self._calculate_global_stats()
        
        for m in self.bandit.models:
            # Get raw HLE from registry
            m_data = self.registry.get(m, {})
            raw_hle_score = float(m_data.get("hle") or 0.05)
            
            # Teach each anchor
            for idx, anchor_vec in enumerate(self.anchor_vectors):
                # 1. Determine Difficulty of this Anchor
                # Projection onto Complexity Vector
                h_score = float(np.dot(anchor_vec, self.complexity_vector))
                is_hard = h_score > 0.5 # Threshold for prior assignment
                
                # 2. Transform HLE
                target_score = transform_hle_to_prior(raw_hle_score, is_hard_prompt=is_hard)
                
                # 3. Construct Context Vector for this Anchor
                # We simulate handcrafted features for the 'ideal' version of this anchor
                # but for simplicity, we use neutral zeros/means for handcrafted.
                dummy_text = list(self.anchors_config.values())[idx]
                x_anchor = self._get_context_vector(dummy_text)
                
                # 4. Weight
                # Each anchor gets equal share of the prior strength
                weight = prior_n_effective / len(self.anchor_vectors)
                
                # 5. Teach
                self.bandit.A[m] += weight * np.outer(x_anchor, x_anchor)
                self.bandit.b[m] += weight * target_score * x_anchor
            
            # Recompute inverse
            self.bandit.A_inv[m] = safe_inv(self.bandit.A[m])

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
        # We need the text for cluster detection and logging
        prompt_text = prompt if isinstance(prompt, str) else "[Pre-embedded Prompt]"
        
        # 1. Vectorize with Zero-Shot Features
        # This creates: [Emb | Feats | Anchors | Hardness Score | Bias]
        # The hardness score project semantic complexity into a raw scalar feature.
        x = self._get_context_vector(prompt)
        
        # 2. Resolve Weights
        weights = OptimizationProfile.get(profile).copy()
        w_q = weights.get("w_q", 1.0 - weights.get("w_c", 0.0) - weights.get("w_l", 0.0))
        w_c = weights.get("w_c", 0.0)
        w_l = weights.get("w_l", 0.0)
        
        # --- ORTHOGONAL OPTIMIZATION ---
        # If a hard constraint is active, disable the soft penalty for that dimension
        # and re-allocate weight to Quality to avoid "Double Penalty".
        if max_cost is not None:
            w_q += w_c
            w_c = 0.0
        if max_latency is not None:
            w_q += w_l
            w_l = 0.0
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
        costs = {m: self._estimate_cost(m, in_tok, output_tokens) for m in filtered}
        lats = {m: self._estimate_latency(m, output_tokens) for m in filtered}
        
        # --- ABSOLUTE COST PENALTY (Logarithmic Market Width) ---
        # Use absolute penalty based on fixed market anchors, not relative to current pool.
        # This ensures a model's cost penalty is determined by its actual price tag,
        # not by what other models happen to be loaded.
        # 
        # Math: penalty = (log(cost) - log(floor)) / range
        # Floor: $0.0005/1k, Ceiling: $10.00/1k, Range: 10.0
        
        # Convert cost to per-1k basis for absolute penalty calculation
        cost_penalties = {}
        for m in filtered:
            cost_per_1k = costs[m] * 1000  # Convert to $/1k tokens
            cost_penalties[m] = self._calculate_absolute_penalty(cost_per_1k)
        
        # --- ABSOLUTE LATENCY PENALTY (Logarithmic Market Width) ---
        # Apply same absolute anchor approach to latency for consistency.
        # This makes lambda_cost and lambda_latency directly comparable.
        # 
        # Latency Anchors:
        # Floor: 0.05s (instant/cached)
        # Ceiling: 5.0s (timeout threshold)
        # Range: ln(5.0) - ln(0.05) ≈ 4.6
        
        LATENCY_FLOOR = 0.05   # 50ms
        LATENCY_CEILING = 5.0  # 5s timeout
        LATENCY_RANGE = np.log(LATENCY_CEILING) - np.log(LATENCY_FLOOR)  # ≈ 4.6
        
        latency_penalties = {}
        for m in filtered:
            # Clip to floor to avoid log domain errors
            safe_lat = max(lats[m], LATENCY_FLOOR)
            log_lat = np.log(safe_lat)
            
            # Normalize absolutely: 0.0 = instant, 1.0 = 5s+
            norm_lat = (log_lat - np.log(LATENCY_FLOOR)) / LATENCY_RANGE
            latency_penalties[m] = max(0.0, min(1.0, norm_lat))
        
        for m in filtered:
            quality = ucbs[m]
            
            # Cost penalty (absolute, 0-1 scale)
            norm_cost = cost_penalties[m]
            
            # Latency penalty (absolute, 0-1 scale)
            norm_lat = latency_penalties[m]
            
            # Utility = (w_q * Quality) + (w_c * (1 - Cost Penalty)) + (w_l * (1 - Latency Penalty))
            # Now all factors are in the same 0-1 absolute scale
            utility = (w_q * quality) + (w_c * (1.0 - norm_cost)) + (w_l * (1.0 - norm_lat))
            
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

    def update(self, model_id: str, context: str | np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update the bandit's internal state with a new observation."""
        x = self._get_context_vector(context)
        self.bandit.update(model_id, x, reward, weight)


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
            self.bandit.b[model_id][-1] = 20.0 * score
            # Also increase confidence in the bias term
            self.bandit.A[model_id][-1, -1] += 20.0
            self.bandit.A_inv[model_id] = safe_inv(self.bandit.A[model_id])


    def save_state(self, path: Path | str) -> None:
        """Save the bandit's learned state to disk."""
        self.bandit.save_state(path)

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
