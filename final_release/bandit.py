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
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Missing dependency: sentence-transformers") from e


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
    def get(cls, name: str) -> float:
        key = name.lower()
        val = cls._RATES.get(key)
        if val is not None: 
            print(f"DEBUG: ExplorationRate.get('{name}') -> {val}")
            return val
        try: return float(name)
        except ValueError: raise ValueError(f"Unknown exploration '{name}'")

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

# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------
class DisjointLinUCBPolicy:
    """Disjoint LinUCB: one ridge regression per arm."""
    def __init__(self, model_names: List[str], dim: int = 384, alpha: float = 0.1,
                 prior_strength: float = 40.0, ridge_lambda: float = 1.0, forgetting_factor: float = 0.95):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.gamma = float(forgetting_factor) # Forgetting factor (1.0 = no forgetting)
        # Initialize A=I*lambda, b=0
        self.A = {m: np.eye(self.dim) * self.ridge_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        self.A_inv = {m: np.linalg.inv(self.A[m]) for m in self.models}
        self.last_update = {m: 0 for m in self.models} # Track last update step
        self.t = 0 # Global time step

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically."""
        if model_name in self.models: return
        
        self.models.append(model_name)
        self.A[model_name] = np.eye(self.dim) * self.ridge_lambda
        self.b[model_name] = np.zeros(self.dim, dtype=np.float64)
        self.A_inv[model_name] = np.linalg.inv(self.A[model_name])
        self.last_update[model_name] = self.t

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
        
        self.A_inv[model] = np.linalg.inv(self.A[model])
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
                self.A_inv[m] = np.linalg.inv(self.A[m])

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

class BanditRouter:
    """
    The primary entry point for routing.
    """
    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        alpha: float = 0.1,
        embedding_dim: int = 384,
        forgetting_factor: float = 0.95,
        benchmark_key: str = "hle",
    ):
        self.registry = dict(model_registry)
        self.encoder = SentenceTransformer(context_model)
        # Add bias term to dimension
        self.bandit = DisjointLinUCBPolicy(
            list(self.registry.keys()), 
            dim=embedding_dim + 1, 
            alpha=alpha,
            forgetting_factor=forgetting_factor
        )
        self.logs: List[RoutingLog] = []
        self.model_priors: Dict[str, float] = {} # Optional scalar priors
        self.benchmark_key = benchmark_key


    @classmethod
    def create(
        cls,
        model_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        priors: str = "benchmark", # Default to HLE with sigmoid transformation
        prior_strength: float = 40.0,
        exploration: str = "safe",  # Optimal: α=0.1
        forgetting_factor: float = 0.95,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        state_path: Optional[Path | str] = None,
        benchmark_key: str = "hle",
    ) -> "BanditRouter":
        """
        Create a configured router.
        
        Args:
            model_registry: Dict of models. If None, loads default `models.json`.
            priors: "benchmark" (HLE) or "none".
            prior_strength: Strength of the prior (default 40.0).
            exploration: "static", "safe", "balanced", "aggressive".
            state_path: Optional path to a saved bandit state (.npz).
            benchmark_key: Key in models.json to use for priors (default "hle").
        """
        base_dir = Path(__file__).parent
        
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
        
        # 3. Initialize
        # Load Saved State (Overrides priors)
        if state_path and Path(state_path).exists():
            router = cls(model_registry, context_model=context_model, alpha=alpha, forgetting_factor=forgetting_factor, benchmark_key=benchmark_key)
            router.bandit.load_state(state_path)
            return router

        # Load HLE Priors (Default)
        if priors == "benchmark":
            meta_path = base_dir / "data" / "priors_meta_large.npz"
            if not meta_path.exists():
                # Fallback to small if large doesn't exist (e.g. in test env)
                meta_path = base_dir / "data" / "priors_meta.npz"
            
            if not meta_path.exists():
                 logger.warning("No priors metadata found. Falling back to cold start.")
                 return cls(model_registry, context_model=context_model, alpha=alpha, benchmark_key=benchmark_key)

            return cls.load_from_benchmark(
                model_registry=model_registry,
                context_model=context_model,
                alpha=alpha,
                prior_strength=prior_strength,
                forgetting_factor=forgetting_factor,
                priors_meta_path=meta_path,
                benchmark_key=benchmark_key
            )
            
        # Cold Start
        return cls(model_registry, context_model=context_model, alpha=alpha, forgetting_factor=forgetting_factor, benchmark_key=benchmark_key)

    @classmethod
    def load_from_benchmark(
        cls,
        *,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str = DEFAULT_CONTEXT_MODEL,
        alpha: float = 0.1,
        prior_strength: float = 40.0,
        priors_meta_path: Optional[Path] = None,
        forgetting_factor: float = 0.95,
        benchmark_key: str = "hle",
    ) -> "BanditRouter":
        """Initialize with HLE priors using covariance matrix."""
        meta = np.load(priors_meta_path)
        cov_matrix = meta["cov_matrix"]
        sum_vec = meta["sum_vec"]
        dim = sum_vec.shape[0]
        
        router = cls(
            model_registry, 
            context_model=context_model, 
            alpha=alpha, 
            embedding_dim=dim,
            forgetting_factor=forgetting_factor,
            benchmark_key=benchmark_key
        )
        
        # Ridge Update: A += strength * Cov, b += strength * score * Sum
        # NORMALIZATION: We scale the benchmark (N=26223) to match prior_strength
        N = 26223.0
        
        # Pad covariance matrix for bias term (zeros for cross-terms, 1.0 for bias variance)
        cov_padded = np.zeros((dim + 1, dim + 1))
        cov_padded[:dim, :dim] = cov_matrix
        cov_padded[dim, dim] = 1.0 # Bias variance
        
        for m in router.bandit.models:
            # Scale initial identity to match prior_strength. 
            router.bandit.A[m] *= prior_strength
            
            # Get score (default 0.05 if missing to allow new models)
            # This satisfies "new model with no benchmarks" constraint
            raw_score = float(model_registry.get(m, {}).get(benchmark_key) or 0.05)
            
            # Transform HLE score to realistic utility prior
            # Raw HLE scores are 0-40%, but even 6% indicates a highly capable model
            score = transform_hle_to_prior(raw_score)
            
            # ------------------------------------------------------------------
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
            efficiency_boost = 1.0 + (0.0 * math.log(1.0 / cost))

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
            
            # 2. Apply Cluster Performance Boost
            cluster_boost = 1.0
            if is_math:
                cluster_boost = 1.5 # Boost math specialists
            
            # Combine: Prior = Score * Efficiency * ClusterBoost
            score *= efficiency_boost * cluster_boost
            
            if score > 0:
                # Update A with padded covariance
                router.bandit.A[m] += prior_strength * (cov_padded / N)
                
                # Update b: Set the bias term to the score
                # The embedding part of b is 0 (assuming average prompt is neutral)
                # b = [0, ..., 0, prior_strength * score]
                bias_update = np.zeros(dim + 1)
                bias_update[dim] = prior_strength * score
                router.bandit.b[m] += bias_update
                
            # Recompute inverse
            router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])
            
        return router

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
        prompt: str,
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
        eff_sensitivity = sensitivity.upper() if sensitivity else self._classify_sensitivity(prompt)
        
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
        # Normalization: Scale Cost (Log) and Latency (Linear) to [0, 1]
        costs = {m: self._estimate_cost(m, in_tok, output_tokens) for m in filtered}
        lats = {m: self._estimate_latency(m, output_tokens) for m in filtered}
        
        # Log-MinMax Normalization
        EPS = 1e-9
        
        log_costs = {m: np.log(max(costs[m], EPS)) for m in filtered}
        log_lats = {m: np.log(max(lats[m], EPS)) for m in filtered}
        
        min_c, max_c = min(log_costs.values()), max(log_costs.values())
        range_c = max_c - min_c if max_c > min_c else 1.0
        
        min_l, max_l = min(log_lats.values()), max(log_lats.values())
        range_l = max_l - min_l if max_l > min_l else 1.0
        
        for m in filtered:
            quality = ucbs[m]
            
            # Normalize to [0, 1] in Log Space
            norm_cost = (log_costs[m] - min_c) / range_c
            norm_lat = (log_lats[m] - min_l) / range_l
            
            # Utility = Quality - (w_c * NormCost) - (w_l * NormLatency)
            # Risk is handled by GATING, so no penalty here.
            utility = quality - (lambda_cost * norm_cost) - (lambda_latency * norm_lat)
            
            if utility > best_utility:
                best_utility = utility
                best_model = m
                
        # 5. Log
        log = RoutingLog(
            request_id=str(time.time_ns()),
            timestamp_s=time.time(),
            prompt=prompt,
            selected_model=best_model,
            predicted_utility=float(best_utility),
            cost_usd=self._estimate_cost(best_model, in_tok, output_tokens),
            latency_s=self._estimate_latency(best_model, output_tokens)
        )
        self.logs.append(log)
        
        return best_model, log

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
        
        # Efficiency Boost
        cost = float(definition.get("input_cost_per_m", 0.0)) / 1000.0
        cost = max(cost, 0.00000005)
        efficiency_boost = 1.0 + (0.2 * math.log(1.0 / cost))
        
        # Cluster Boost
        # Check ID, description, and tags
        text_to_check = (model_id + " " + definition.get("description", "") + " " + " ".join(definition.get("tags", []))).lower()
        is_math = any(k in text_to_check for k in ["math", "reasoning", "deepseek", "gemini", "flash"])
        
        cluster_boost = 1.0
        if is_math:
            cluster_boost = 1.5
            
        # Apply Boosts
        score *= efficiency_boost * cluster_boost
        
        if score > 0:
            # Initialize with prior belief
            # Set the bias term (last element) of b to prior_strength * score
            # This effectively gives it a "mean reward" of 'score' for the bias feature.
            self.bandit.b[model_id][-1] = 20.0 * score
            # Also increase confidence in the bias term
            self.bandit.A[model_id][-1, -1] += 20.0
            self.bandit.A_inv[model_id] = np.linalg.inv(self.bandit.A[model_id])

    def _get_context_vector(self, context: str | np.ndarray) -> np.ndarray:
        """Convert string prompt or array to a normalized context vector."""
        if isinstance(context, str):
            x = self.encoder.encode(context)
            x = l2_normalize(x)
        else:
            x = context
            
        # Append bias term
        return np.append(x, 1.0)

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
