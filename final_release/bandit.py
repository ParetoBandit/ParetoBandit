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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Missing dependency: sentence-transformers") from e

from .quality_predictor import (
    QualityCostPredictor,
    LogitReward,
    RunningZScoreNormalizer,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Optimization Profiles
# ---------------------------------------------------------------------------
class OptimizationProfile:
    """Named presets for utility function weights (Quality vs Cost vs Latency)."""
    QUALITY_FIRST = {"lambda_cost": 0.1, "lambda_latency": 0.05}
    BALANCED = {"lambda_cost": 10.0, "lambda_latency": 0.10}
    COST_SAVER = {"lambda_cost": 50.0, "lambda_latency": 0.20}
    LOW_LATENCY = {"lambda_cost": 1.0, "lambda_latency": 0.50}

    _PROFILES = {
        "quality_first": QUALITY_FIRST,
        "balanced": BALANCED,
        "cost_saver": COST_SAVER,
        "low_latency": LOW_LATENCY,
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
    SAFE = 0.1         # Minimal exploration (Default)
    BALANCED = 0.5     # Standard bandit
    AGGRESSIVE = 2.0   # High exploration

    _RATES = {
        "static": STATIC, "safe": SAFE, "balanced": BALANCED, "aggressive": AGGRESSIVE
    }

    @classmethod
    def get(cls, name: str) -> float:
        key = name.lower()
        if key in cls._RATES: return cls._RATES[key]
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

# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------
class DisjointLinUCBPolicy:
    """Disjoint LinUCB: one ridge regression per arm."""
    def __init__(self, model_names: List[str], dim: int = 384, alpha: float = 0.1, ridge_lambda: float = 1.0):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        # Initialize A=I*lambda, b=0
        self.A = {m: np.eye(self.dim) * self.ridge_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        self.A_inv = {m: np.linalg.inv(self.A[m]) for m in self.models}

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
            var = float(x.dot(self.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + self.alpha * std
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = m
        
        return best_model, float(best_ucb)

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        if model not in self.A: return
        self.A[model] += np.outer(x, x)
        self.b[model] += float(reward) * x
        self.A_inv[model] = np.linalg.inv(self.A[model])

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
    ):
        self.registry = dict(model_registry)
        self.encoder = SentenceTransformer(context_model)
        self.bandit = DisjointLinUCBPolicy(list(self.registry.keys()), dim=embedding_dim, alpha=alpha)
        self.logs: List[RoutingLog] = []
        self.model_priors: Dict[str, float] = {} # Optional scalar priors

    @classmethod
    def create(
        cls,
        model_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        priors: str = "benchmark", # Default to HLE
        prior_strength: float = 20.0,
        exploration: str = "safe",
        context_model: str = DEFAULT_CONTEXT_MODEL,
        state_path: Optional[Path | str] = None,
    ) -> "BanditRouter":
        """
        Create a configured router.
        
        Args:
            model_registry: Dict of models. If None, loads default `models.json`.
            priors: "benchmark" (HLE) or "none".
            prior_strength: Strength of the prior (default 20.0).
            exploration: "static", "safe", "balanced", "aggressive".
            state_path: Optional path to a saved bandit state (.npz).
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
            router = cls(model_registry, context_model=context_model, alpha=alpha)
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
                 return cls(model_registry, context_model=context_model, alpha=alpha)

            return cls.load_from_benchmark(
                model_registry=model_registry,
                context_model=context_model,
                alpha=alpha,
                prior_strength=prior_strength,
                priors_meta_path=meta_path
            )
            
        # Cold Start
        return cls(model_registry, context_model=context_model, alpha=alpha)

    @classmethod
    def load_from_benchmark(
        cls,
        *,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        alpha: float,
        prior_strength: float,
        priors_meta_path: Path,
        benchmark_key: str = "hle",
    ) -> "BanditRouter":
        """Initialize with HLE priors using covariance matrix."""
        meta = np.load(priors_meta_path)
        cov_matrix = meta["cov_matrix"]
        sum_vec = meta["sum_vec"]
        dim = sum_vec.shape[0]
        
        router = cls(model_registry, context_model=context_model, alpha=alpha, embedding_dim=dim)
        
        # Ridge Update: A += strength * Cov, b += strength * score * Sum
        A_update = prior_strength * cov_matrix
        
        for m in router.bandit.models:
            # Scale initial identity
            router.bandit.A[m] *= prior_strength
            
            # Get score (default 0 if missing)
            score = float(model_registry.get(m, {}).get(benchmark_key) or 0.0)
            
            if score > 0:
                router.bandit.A[m] += A_update
                router.bandit.b[m] += prior_strength * score * sum_vec
                
            # Recompute inverse
            router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])
            
        return router

    def route(
        self,
        prompt: str,
        *,
        profile: str = "balanced",
        max_cost: Optional[float] = None,
        max_latency: Optional[float] = None,
        quality_floor: Optional[Dict[str, float]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: int = 600,
    ) -> Tuple[str, RoutingLog]:
        """
        Route a prompt to the best model based on Quality, Cost, and Latency.
        
        Args:
            prompt: User query.
            profile: "quality_first", "balanced", "cost_saver", "low_latency".
            max_cost: Hard limit on $/request.
            max_latency: Hard limit on seconds/request.
            quality_floor: Min benchmark scores (e.g. {"math": 80}).
        """
        # 1. Embed
        x = self.encoder.encode(prompt)
        x = l2_normalize(x)
        
        # 2. Resolve Weights
        weights = OptimizationProfile.get(profile)
        lambda_cost = weights["lambda_cost"]
        lambda_latency = weights["lambda_latency"]
        
        # 3. Filter Candidates (Constraints)
        candidates = list(self.registry.keys())
        
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
            raise ValueError("No models satisfy the constraints.")
            
        # 4. Score Candidates
        best_model = filtered[0]
        best_utility = -float("inf")
        
        # Pre-compute UCBs
        ucbs = {}
        for m in filtered:
            _, ucb = self.bandit.select_arm(x, candidates=[m])
            ucbs[m] = ucb
            
        # Calculate Utility
        for m in filtered:
            quality = ucbs[m]
            cost = self._estimate_cost(m, in_tok, output_tokens)
            lat = self._estimate_latency(m, output_tokens)
            
            # Utility = Quality - (w_c * Cost) - (w_l * Latency)
            utility = quality - (lambda_cost * cost) - (lambda_latency * lat)
            
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

    def save_state(self, path: Path | str) -> None:
        """Save the bandit's learned state to disk."""
        self.bandit.save_state(path)

    def _estimate_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        m = self.registry.get(model, {})
        # Try per-token pricing first
        if m.get("input_cost_per_m") and m.get("output_cost_per_m"):
            return (m["input_cost_per_m"] * in_tok + m["output_cost_per_m"] * out_tok) / 1e6
        # Fallback to fixed cost
        return float(m.get("cost", 0.0))

    def _estimate_latency(self, model: str, out_tok: int) -> float:
        m = self.registry.get(model, {})
        ttft = m.get("time_to_first_token_seconds", 0.0) or 0.0
        otps = m.get("output_tokens_per_second", 0.0) or 0.0
        gen_time = (out_tok / otps) if otps > 0 else 0.0
        return float(ttft + gen_time) or float(m.get("latency_s", 1.0))
