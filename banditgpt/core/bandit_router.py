"""
Production-grade contextual bandit router (Hot Path) + async grader loop (Cold Path).

Design goals (per our architecture discussion):
  - Router (hot path): millisecond-scale model selection using LinUCB.
  - Grader (cold path): asynchronous competence grading using QualityCostPredictor.
  - Learning signal: reward_z (clipped z-score) derived from reward_raw (clipped P_correct).
  - Persistence: store bandit parameters AND reward normalizer state so the system
    is "primed" immediately on startup (no day-1 cold start).

Important terminology:
  - This is Type B "competence risk" routing (avoid low-competence answers).
  - Policy safety (Type A) should be handled by provider / separate safety filter.

How Routing Works: Prompt → Prediction
---------------------------------------

1. THE MAPPER: Embedding Model
   When a user sends a prompt like "Write a Python script to parse JSON",
   the Sentence Transformer (all-MiniLM-L6-v2) converts it into a 384-dimensional
   context vector x:

       Prompt: "Write Python..."  →  x = [0.05, -0.92, 0.44, ...]

   This vector describes the prompt's position in "Meaning Space." The numbers
   essentially say: "This is close to Coding, far from Poetry, somewhat close to Logic."

2. THE MEMORY: Weight Vector (θ)
   Each model (e.g., Llama-3) has its own weight vector θ that acts as its "Profile."
   This vector was learned during prior generation (the archetype grid or synthetic warmup).

   Because Llama-3 did well on the "Coding" cluster, its weights are high in the
   dimensions that correspond to coding:

       θ = A⁻¹ @ b   (solved via linear regression on observed rewards)

3. THE LOOKUP: Dot Product
   The bandit calculates the predicted quality using a dot product:

       predicted_score = θ · x

   - If the vectors align (prompt asks for Python, Llama-3's weights "point" towards
     Python), the math produces a high score (e.g., 0.95).
   - If they oppose (prompt asks for French History, Llama-3's weights point away
     from History), the math produces a low score (e.g., 0.20).

4. THE DECISION: UCB with Utility
   We add exploration bonus and apply cost/latency penalties:

       UCB = (θ·x + prior) + α·√(x'A⁻¹x)
       Utility = UCB - λ_cost·Cost - λ_latency·Latency

   The model with highest Utility wins.
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
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: sentence-transformers") from e

from banditgpt.core.quality_cost_predictor import (
    QualityCostPredictor,
    LogitReward,
    RunningZScoreNormalizer,
)

logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast 384-dim


# ---------------------------------------------------------------------------
# Optimization Profiles (User-Friendly Presets)
# ---------------------------------------------------------------------------
#
# The utility function is: U = Q - (w_cost * C) - (w_latency * L)
#
# Weights act as "exchange rates" converting money/time into quality points:
#   - w_cost: Quality per dollar (ΔQ/$) - "How much quality to sacrifice to save $1?"
#   - w_latency: Quality per second (ΔQ/s) - "How much quality to sacrifice to save 1s?"
#
# Reference Table:
# ┌─────────────────┬──────────┬────────────┬─────────────────────────────────────────┐
# │ Profile         │ w_cost   │ w_latency  │ Behavior                                │
# ├─────────────────┼──────────┼────────────┼─────────────────────────────────────────┤
# │ quality_first   │ 0.1      │ 0.05       │ Ignore price, maximize quality          │
# │ balanced        │ 10.0     │ 0.10       │ Trade 1% quality to save $0.001         │
# │ cost_saver      │ 50.0     │ 0.20       │ Aggressive cost optimization            │
# │ low_latency     │ 1.0      │ 0.50       │ Prioritize speed over cost              │
# └─────────────────┴──────────┴────────────┴─────────────────────────────────────────┘
#
# Think of weights as PENALTIES:
#   "For every $0.01 spent, penalize the model's score by w_cost * 0.01"
#   "For every second waited, penalize the model's score by w_latency * 1.0"


class OptimizationProfile:
    """
    Named presets for the utility function weights.

    Instead of tuning raw floats, users can pick a profile:
        router.route("Write code", profile="balanced")

    Profiles:
        quality_first: Maximize quality, ignore cost/latency
        balanced: Reasonable trade-off (default for most apps)
        cost_saver: Aggressive cost optimization
        low_latency: Prioritize speed for real-time apps
    """

    QUALITY_FIRST = {"lambda_cost": 0.1, "lambda_latency": 0.05}
    BALANCED = {"lambda_cost": 10.0, "lambda_latency": 0.10}
    COST_SAVER = {"lambda_cost": 50.0, "lambda_latency": 0.20}
    LOW_LATENCY = {"lambda_cost": 1.0, "lambda_latency": 0.50}

    _PROFILES = {
        "quality_first": QUALITY_FIRST,
        "quality-first": QUALITY_FIRST,
        "balanced": BALANCED,
        "cost_saver": COST_SAVER,
        "cost-saver": COST_SAVER,
        "penny_pincher": COST_SAVER,
        "low_latency": LOW_LATENCY,
        "low-latency": LOW_LATENCY,
        "realtime": LOW_LATENCY,
    }

    @classmethod
    def get(cls, name: str) -> Dict[str, float]:
        """Get profile by name (case-insensitive, supports hyphens/underscores)."""
        key = name.lower().replace("-", "_")
        if key not in cls._PROFILES:
            valid = ", ".join(sorted(set(cls._PROFILES.keys())))
            raise ValueError(f"Unknown profile '{name}'. Valid profiles: {valid}")
        return cls._PROFILES[key]

    @classmethod
    def list_profiles(cls) -> List[str]:
        """List all available profile names."""
        return ["quality_first", "balanced", "cost_saver", "low_latency"]


# ---------------------------------------------------------------------------
# Exploration Rate (User-Friendly Alpha Setting)
# ---------------------------------------------------------------------------
#
# The exploration bonus is: Q_final = μ(x) + α·σ(x)
#
# Alpha (α) controls the RISK APPETITE of the router:
#   - Higher α → More exploration (try unproven models to find better options)
#   - Lower α → More exploitation (stick with known winners, minimize risk)
#
# Different product stages need different risk profiles:
#   - Day 1 User: "Test everything! I want to learn which models work best."
#   - Production Bank: "Never route to an unproven model. Zero risk."
#
# Reference Table:
# ┌─────────────────┬───────┬────────────────────────────────────────────────────┐
# │ Setting         │ Alpha │ Behavior                                           │
# ├─────────────────┼───────┼────────────────────────────────────────────────────┤
# │ static          │ 0.0   │ Pure exploitation. Trust mean only. (Bank Mode)    │
# │ safe            │ 0.1   │ Minimal risk. Only explore if upside is huge.      │
# │ balanced        │ 0.5   │ Standard bandit behavior. (Default)                │
# │ aggressive      │ 2.0   │ Try everything! Fast learning. (Day 1 / Shadow)    │
# └─────────────────┴───────┴────────────────────────────────────────────────────┘
#
# Think of it as: "How often should the router try unproven models to find
# cheaper/better options?"


class ExplorationRate:
    """
    Named presets for the exploration parameter.

    Controls how often the router tries unproven models to find better options.
    Higher values = more exploration (riskier but learns faster).

    Usage:
        # At router creation
        router = BanditRouter.create(registry, exploration="safe")

        # During routing (override default)
        model, log = router.route("...", exploration="aggressive")

        # Get the raw alpha value
        alpha = ExplorationRate.get("safe")  # Returns 0.1

    Presets:
        static: Pure exploitation, zero risk (α=0.0) - for production/fintech
        safe: Minimal exploration (α=0.1) - RECOMMENDED DEFAULT
        balanced: Standard bandit (α=0.5) - reasonable exploration
        aggressive: Maximum exploration (α=2.0) - for day-1/shadow mode
    """

    STATIC = 0.0       # Trust mean only. No gambling. (Bank Mode)
    SAFE = 0.1         # Only gamble if upside is huge. (Production Default)
    BALANCED = 0.5     # Standard Bandit behavior.
    AGGRESSIVE = 2.0   # Try everything! (Calibration/Day 1 Mode)

    _RATES = {
        # Primary names
        "static": STATIC,
        "safe": SAFE,
        "balanced": BALANCED,
        "aggressive": AGGRESSIVE,
        # Aliases
        "none": STATIC,
        "zero": STATIC,
        "off": STATIC,
        "production": SAFE,
        "default": SAFE,
        "low": SAFE,
        "medium": BALANCED,
        "normal": BALANCED,
        "high": AGGRESSIVE,
        "calibration": AGGRESSIVE,
        "shadow": AGGRESSIVE,
        "day1": AGGRESSIVE,
        "day_1": AGGRESSIVE,
        "learning": AGGRESSIVE,
    }

    @classmethod
    def get(cls, name: str) -> float:
        """
        Get alpha value by name (case-insensitive).

        Args:
            name: Preset name ("safe", "aggressive") or a float string ("0.75")

        Returns:
            Alpha value as float

        Examples:
            ExplorationRate.get("safe")        # 0.1
            ExplorationRate.get("aggressive")  # 2.0
            ExplorationRate.get("0.75")        # 0.75
        """
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key in cls._RATES:
            return cls._RATES[key]
        # Allow direct float values like "0.75"
        try:
            return float(name)
        except ValueError:
            valid = ["static", "safe", "balanced", "aggressive"]
            raise ValueError(f"Unknown exploration '{name}'. Valid: {valid} (or a float like '0.75')")

    @classmethod
    def list_rates(cls) -> List[str]:
        """List all available rate names."""
        return ["static", "safe", "balanced", "aggressive"]


def build_cost_proportional_priors(
    registry: Dict[str, Dict[str, Any]],
    *,
    gamma: float = 0.25,
    min_cost_usd: float = 1e-6,
    clamp: float = 3.0,
) -> Dict[str, float]:
    """
    Cost-proportional cold-start priors (no benchmarks).

    Philosophy: if you have no data, assume price ≈ quality.

    Prior formula:
      prior(model) = clamp(gamma * log(max(cost_usd, min_cost_usd)), [-clamp, +clamp])

    Notes:
      - cost_usd here is the router's estimated $/request (from the model cache).
      - This is intentionally simple and *meant to be overwritten quickly* by real
        feedback / warmup priors.
    """
    g = float(gamma)
    lo = float(min_cost_usd)
    c = float(max(0.0, clamp))
    priors: Dict[str, float] = {}
    for mid, meta in (registry or {}).items():
        try:
            cost = float((meta or {}).get("cost", 0.0) or 0.0)
        except Exception:
            cost = 0.0
        x = max(cost, lo)
        val = g * float(math.log(x))
        if c > 0:
            val = max(min(val, c), -c)
        priors[str(mid)] = float(val)
    return priors


def estimate_tokens_rough(text: str) -> int:
    """
    Cheap token estimator (no tokenizer dependency).

    Rule of thumb: ~1 token ~= 0.75 words in English prose.
    We use 1.3 * word_count as a conservative approximation.
    """
    if text is None:
        return 0
    wc = len(str(text).split())
    return int(max(0, round(wc * 1.3)))

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to unit L2 norm.

    LinUCB is sensitive to feature scale. Normalizing x keeps the uncertainty term
    and learned weights in a stable, comparable range across prompts.
    """
    x = np.asarray(x, dtype=np.float64)
    n = float(np.linalg.norm(x))
    if n <= eps:
        return x
    return x / n


def build_registry_from_models_cache(
    cache_path: str | Path,
    *,
    default_output_tokens: int = 600,
    default_input_tokens: Optional[int] = None,
    latency_mode: str = "ttft+gen",
) -> Dict[str, Dict[str, Any]]:
    """
    Build a BanditRouter-compatible registry from `data/models_cache.json`.

    The bandit router expects:
      registry[openrouter_id] = {"cost": <float>, "latency_s": <float>, ...}

    We derive:
      - cost: estimated $ per request (based on $/1M input/output tokens)
      - latency_s: estimated seconds per request (TTFT + generation time if available)
      - scores: benchmark scores for quality masking (math, code, reasoning, mmlu, avg)

    Args:
      cache_path: path to models_cache.json
      default_output_tokens: assumed completion size for cost/latency estimation
      default_input_tokens: assumed prompt size; if None, caller can use prompt-based estimation externally
      latency_mode:
        - "ttft": use time_to_first_token_seconds only
        - "ttft+gen": TTFT + (output_tokens / output_tokens_per_second) when available

    Returns:
      Dict mapping model_id to metadata including:
        - cost: estimated $ per request
        - latency_s: estimated seconds per request
        - scores: {"math": 0-100, "code": 0-100, "reasoning": 0-100, "mmlu": 0-100, "avg": 0-100}
        - display_name, input_cost_per_m, output_cost_per_m, etc.

    Example:
        registry = build_registry_from_models_cache("models_cache.json")
        # Filter to models with math score > 80:
        math_capable = [m for m in registry if registry[m]["scores"]["math"] > 80]
    """
    cache_path = Path(cache_path)
    d = json.loads(cache_path.read_text())
    models = d.get("models", [])
    if not isinstance(models, list):
        raise ValueError("models_cache.json expected {'models': [...]} format")

    mode = str(latency_mode).lower().strip()
    if mode not in {"ttft", "ttft+gen"}:
        raise ValueError("latency_mode must be 'ttft' or 'ttft+gen'")

    registry: Dict[str, Dict[str, Any]] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        or_id = m.get("openrouter_id")
        if not isinstance(or_id, str) or not or_id.strip():
            continue
        or_id = or_id.strip()

        # Costs are in $ per 1M tokens.
        in_cost_raw = m.get("input_cost_per_m", m.get("price_1m_input"))
        out_cost_raw = m.get("output_cost_per_m", m.get("price_1m_output"))
        blended_raw = m.get("price_1m_blended", m.get("price_1m_blended_3_to_1", m.get("price_1m_blended_3_to_1")))

        def _pos_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                fv = float(v)
            except Exception:
                return None
            return fv if fv > 0 else None

        in_cost_per_m = _pos_float(in_cost_raw)
        out_cost_per_m = _pos_float(out_cost_raw)
        blended_cost_per_m = _pos_float(blended_raw)

        in_tok = int(default_input_tokens) if default_input_tokens is not None else 0
        out_tok = int(default_output_tokens)
        if in_cost_per_m is not None and out_cost_per_m is not None:
            cost_est = (in_cost_per_m * (in_tok / 1_000_000.0)) + (out_cost_per_m * (out_tok / 1_000_000.0))
        elif blended_cost_per_m is not None:
            cost_est = blended_cost_per_m * ((in_tok + out_tok) / 1_000_000.0)
        else:
            # Unknown pricing: do NOT treat as free; set a conservative non-zero default.
            cost_est = 1e-3

        ttft = _pos_float(m.get("time_to_first_token_seconds", m.get("measured_ttft_seconds")))
        otps = _pos_float(m.get("output_tokens_per_second"))
        median_latency_ms = _pos_float(m.get("median_latency_ms"))
        gen_t = (out_tok / float(otps)) if (mode == "ttft+gen" and (otps or 0.0) > 0.0) else 0.0
        latency_est = float((ttft or 0.0) + gen_t)
        if latency_est <= 0.0 and (median_latency_ms or 0.0) > 0:
            latency_est = float(float(median_latency_ms) / 1000.0)
        if latency_est <= 0.0:
            # Unknown latency: conservative default to avoid "0 latency wins"
            latency_est = 1.0

        # Extract benchmark scores for quality masking
        # Normalize all scores to 0-100 scale
        def _norm_score(v: Any, scale_if_small: bool = True) -> float:
            """Normalize score to 0-100 scale."""
            if v is None:
                return 0.0
            try:
                fv = float(v)
            except Exception:
                return 0.0
            # If score is <= 1.0, assume it's a fraction and scale to 100
            if scale_if_small and 0 < fv <= 1.0:
                fv *= 100.0
            return max(0.0, fv)

        benchmark_scores = {
            "math": _norm_score(m.get("math_500")),
            "code": _norm_score(m.get("humaneval_score"), scale_if_small=False),  # Already 0-100
            "reasoning": _norm_score(m.get("reasoning_score"), scale_if_small=False),
            "mmlu": _norm_score(m.get("mmlu_pro")),
        }
        # Compute average score for general quality floor
        # USE ONLY 3 BENCHMARKS: math, reasoning, mmlu (all have 100% coverage)
        # HumanEval (code) is excluded from avg due to sparse coverage (67/81 models)
        # but preserved in scores["code"] for domain-specific analysis.
        # This creates a balanced "General Capability" score:
        #   - Math + Reasoning = Fluid Intelligence (Logic, Problem Solving)
        #   - MMLU = Crystallized Intelligence (Facts, World Knowledge)
        core_scores = [benchmark_scores["math"], benchmark_scores["reasoning"], benchmark_scores["mmlu"]]
        benchmark_scores["avg"] = sum(core_scores) / 3.0

        registry[or_id] = {
            "display_name": m.get("display_name", m.get("name")),
            "cost": float(cost_est),
            "latency_s": float(latency_est),
            # NEW: Benchmark scores for quality masking
            "scores": benchmark_scores,
            # Keep raw fields around for debugging / alternative cost models
            "input_cost_per_m": in_cost_per_m,
            "output_cost_per_m": out_cost_per_m,
            "price_1m_blended": blended_cost_per_m,
            "time_to_first_token_seconds": float(ttft or 0.0),
            "output_tokens_per_second": float(otps or 0.0),
            "median_latency_ms": float(median_latency_ms or 0.0),
        }

    return registry


@dataclass
class RoutingLog:
    """
    The golden tuple recorded per request.

    Store this (and the response later) to enable the asynchronous learning loop.
    """

    request_id: str
    timestamp_s: float
    prompt: str
    context_vector: List[float]
    selected_model: str
    # Bandit prediction (quality signal) for the chosen model.
    # Note: selection uses predicted_utility, not just predicted_quality.
    predicted_quality: float
    # Utility score used for routing:
    #   U = quality_hat - lambda_cost * cost - lambda_latency * latency
    predicted_utility: float
    propensity: float  # P(selected_model | policy, x) needed for off-policy eval later
    # Filled in by the cold path:
    response_text: Optional[str] = None
    reward_raw: Optional[float] = None
    reward_logit: Optional[float] = None
    reward_z: Optional[float] = None
    grader_meta: Optional[Dict[str, Any]] = None


class DisjointLinUCBPolicy:
    """
    Disjoint LinUCB: one ridge regression per arm.

    We learn directly in normalized reward space (reward_z), which gives stronger
    signal and faster convergence under score compression.
    """

    def __init__(
        self,
        model_names: List[str],
        dim: int = 384,
        alpha: float = 0.5,
        ridge_lambda: float = 1.0,
        recompute_inv_every: int = 50,
    ):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.recompute_inv_every = int(recompute_inv_every)

        self.A: Dict[str, np.ndarray] = {m: np.eye(self.dim) * self.ridge_lambda for m in self.models}
        self.b: Dict[str, np.ndarray] = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        self.A_inv: Dict[str, np.ndarray] = {m: np.linalg.inv(self.A[m]) for m in self.models}
        self._updates: Dict[str, int] = {m: 0 for m in self.models}

    def select_arm(
        self,
        x: np.ndarray,
        *,
        candidate_models: Optional[List[str]] = None,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, float, float]:
        """
        Returns (model_name, predicted_ucb, propensity).

        - epsilon-greedy exploration is supported (propensity is computed accordingly).
        """
        rng = rng or np.random.default_rng()
        candidates = candidate_models if candidate_models else self.models
        candidates = [m for m in candidates if m in self.A]
        if not candidates:
            raise ValueError("No candidate models available for selection.")

        eps = float(max(0.0, min(1.0, epsilon)))
        if eps > 0 and rng.random() < eps:
            chosen = candidates[int(rng.integers(0, len(candidates)))]
            return chosen, float("nan"), (eps / len(candidates))

        best_model = candidates[0]
        best_ucb = -float("inf")

        for m in candidates:
            theta = self.A_inv[m] @ self.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(self.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            ucb = mean + self.alpha * std
            if ucb > best_ucb:
                best_ucb = ucb
                best_model = m

        # propensity under epsilon-greedy:
        #   P(best) = (1-eps) + eps/|candidates|
        #   P(other) = eps/|candidates|
        prop = (1.0 - eps) + (eps / len(candidates))
        return best_model, float(best_ucb), float(prop)

    def update(self, model: str, x: np.ndarray, reward_z: float) -> None:
        if model not in self.A:
            return
        self.A[model] += np.outer(x, x)
        self.b[model] += float(reward_z) * x
        self._updates[model] += 1
        if self._updates[model] % self.recompute_inv_every == 0:
            self.A_inv[model] = np.linalg.inv(self.A[model])

    def add_model(
        self,
        new_model: str,
        *,
        clone_from: Optional[str] = None,
        clone_decay: float = 0.9,
    ) -> bool:
        """
        Dynamically add a new model to the bandit ("brain surgery").

        This allows installing new models at runtime without restarting the router.

        Args:
            new_model: Model ID to add (e.g., "deepseek/deepseek-v3")
            clone_from: Optional model ID to clone weights from
            clone_decay: Multiply cloned weights by this factor (default 0.9)
                         to increase uncertainty and encourage exploration

        Returns:
            True if model was added, False if it already exists
        """
        if new_model in self.A:
            return False  # Already exists

        self.models.append(new_model)

        if clone_from and clone_from in self.A:
            # CLONING: Copy from existing model with decay
            self.A[new_model] = self.A[clone_from].copy() * float(clone_decay)
            self.b[new_model] = self.b[clone_from].copy() * float(clone_decay)
        else:
            # COLD START: Identity matrix for A, zeros for b
            self.A[new_model] = np.eye(self.dim) * self.ridge_lambda
            self.b[new_model] = np.zeros(self.dim, dtype=np.float64)

        # Pre-compute inverse for immediate inference
        self.A_inv[new_model] = np.linalg.inv(self.A[new_model])
        self._updates[new_model] = 0

        return True

    def remove_model(self, model: str) -> bool:
        """
        Remove a model from the bandit.

        Args:
            model: Model ID to remove

        Returns:
            True if model was removed, False if it didn't exist
        """
        if model not in self.A:
            return False

        self.models.remove(model)
        del self.A[model]
        del self.b[model]
        del self.A_inv[model]
        del self._updates[model]

        return True

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "alpha": self.alpha,
            "ridge_lambda": self.ridge_lambda,
            "recompute_inv_every": self.recompute_inv_every,
            "models": self.models,
            "A": {m: self.A[m].tolist() for m in self.models},
            "b": {m: self.b[m].tolist() for m in self.models},
            "updates": dict(self._updates),
        }

    def to_meta_dict(self) -> Dict[str, Any]:
        """
        Metadata-only state (no big matrices). Intended to be paired with `.npz`.
        """
        return {
            "dim": self.dim,
            "alpha": self.alpha,
            "ridge_lambda": self.ridge_lambda,
            "recompute_inv_every": self.recompute_inv_every,
            "models": self.models,
            "updates": dict(self._updates),
        }

    def save_npz(self, path: Path, *, dtype: Any = np.float32) -> None:
        """
        Save bandit matrices compactly as a compressed NPZ.

        This avoids massive JSON state files (A is ~#models * dim^2 floats).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        models = list(self.models)
        A = np.stack([np.asarray(self.A[m], dtype=dtype) for m in models], axis=0)
        b = np.stack([np.asarray(self.b[m], dtype=dtype) for m in models], axis=0)
        np.savez_compressed(p, models=np.asarray(models, dtype=object), A=A, b=b)

    @classmethod
    def from_meta_and_npz(cls, meta: Dict[str, Any], npz_path: Path) -> "DisjointLinUCBPolicy":
        p = Path(npz_path)
        z = np.load(p, allow_pickle=True)
        models = [str(x) for x in list(z["models"])]
        obj = cls(
            model_names=models,
            dim=int(meta["dim"]),
            alpha=float(meta["alpha"]),
            ridge_lambda=float(meta.get("ridge_lambda", 1.0)),
            recompute_inv_every=int(meta.get("recompute_inv_every", 50)),
        )
        A = np.asarray(z["A"], dtype=np.float64)  # shape: (n_models, dim, dim)
        b = np.asarray(z["b"], dtype=np.float64)  # shape: (n_models, dim)
        for i, m in enumerate(models):
            obj.A[m] = A[i]
            obj.b[m] = b[i]
            obj.A_inv[m] = np.linalg.inv(obj.A[m])
        obj._updates = {k: int(v) for k, v in (meta.get("updates", {}) or {}).items()}
        return obj

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "DisjointLinUCBPolicy":
        obj = cls(
            model_names=list(d["models"]),
            dim=int(d["dim"]),
            alpha=float(d["alpha"]),
            ridge_lambda=float(d.get("ridge_lambda", 1.0)),
            recompute_inv_every=int(d.get("recompute_inv_every", 50)),
        )
        for m in obj.models:
            obj.A[m] = np.asarray(d["A"][m], dtype=np.float64)
            obj.b[m] = np.asarray(d["b"][m], dtype=np.float64)
            obj.A_inv[m] = np.linalg.inv(obj.A[m])
        obj._updates = {k: int(v) for k, v in d.get("updates", {}).items()}
        return obj


class SharedCovarianceLinUCBPolicy:
    """
    LinUCB variant with a *shared* covariance matrix A across all models.

    Motivation:
      - When you pre-warm on a shared proxy dataset, the embedding covariance
        structure is (approximately) shared. Models differ primarily in b (reward).
      - This allows very compact priors: one A (or A_inv) + N b vectors.

    Tradeoff:
      - Exploration variance term becomes shared across arms.
      - Works well for warm-start priors and is dramatically smaller to serialize.
    """

    def __init__(
        self,
        model_names: List[str],
        dim: int = 384,
        alpha: float = 0.5,
        ridge_lambda: float = 1.0,
        recompute_inv_every: int = 50,
    ):
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.recompute_inv_every = int(recompute_inv_every)

        self.A = np.eye(self.dim, dtype=np.float64) * self.ridge_lambda
        self.A_inv = np.linalg.inv(self.A)
        self.b: Dict[str, np.ndarray] = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}
        self._updates: Dict[str, int] = {m: 0 for m in self.models}
        self._a_updates = 0

    def update(self, model: str, x: np.ndarray, reward: float) -> None:
        if model not in self.b:
            return
        x = np.asarray(x, dtype=np.float64)
        self.A += np.outer(x, x)
        self.b[model] += float(reward) * x
        self._updates[model] += 1
        self._a_updates += 1
        if self._a_updates % self.recompute_inv_every == 0:
            self.A_inv = np.linalg.inv(self.A)

    def predict(self, x: np.ndarray, model: str) -> float:
        if model not in self.b:
            return 0.0
        x = np.asarray(x, dtype=np.float64)
        theta = self.A_inv @ self.b[model]
        mean = float(theta.dot(x))
        var = float(x.dot(self.A_inv).dot(x))
        std = float(np.sqrt(max(var, 1e-12)))
        return float(mean + self.alpha * std)

    def to_shippable_priors_npz(
        self,
        path: Path,
        *,
        dtype: Any = np.float16,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save compact priors bundle: A_shared + per-model b vectors.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        models = np.asarray(self.models, dtype=object)
        A = np.asarray(self.A, dtype=dtype)
        b = np.stack([np.asarray(self.b[m], dtype=dtype) for m in self.models], axis=0)
        meta = {
            "dim": int(self.dim),
            "alpha": float(self.alpha),
            "ridge_lambda": float(self.ridge_lambda),
            "recompute_inv_every": int(self.recompute_inv_every),
            "updates": {k: int(v) for k, v in self._updates.items()},
        }
        if extra_meta:
            meta.update(extra_meta)
        np.savez_compressed(p, models=models, A_shared=A, b=b, meta=np.asarray([json.dumps(meta)], dtype=object))

    @classmethod
    def from_shippable_priors_npz(cls, path: Path) -> "SharedCovarianceLinUCBPolicy":
        p = Path(path)
        z = np.load(p, allow_pickle=True)
        models = [str(x) for x in list(z["models"])]
        meta_s = str(list(z["meta"])[0])
        meta = json.loads(meta_s)
        obj = cls(
            model_names=models,
            dim=int(meta.get("dim", 384)),
            alpha=float(meta.get("alpha", 0.5)),
            ridge_lambda=float(meta.get("ridge_lambda", 1.0)),
            recompute_inv_every=int(meta.get("recompute_inv_every", 50)),
        )
        obj.A = np.asarray(z["A_shared"], dtype=np.float64)
        obj.A_inv = np.linalg.inv(obj.A)
        b = np.asarray(z["b"], dtype=np.float64)
        for i, m in enumerate(models):
            obj.b[m] = b[i]
        obj._updates = {k: int(v) for k, v in (meta.get("updates", {}) or {}).items()}
        return obj


class BanditRouter:
    """
    Public entrypoint for production routing.

    Hot path:
      - embed prompt -> x
      - choose model via LinUCB (in reward_z space)
      - return choice + request_id (and a log record)

    Cold path:
      - given (request_id -> response_text), grade and update bandit
    """

    def __init__(
        self,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        alpha: float = 0.1,  # Default to "safe" exploration
        exploration: Optional[str] = None,  # User-friendly exploration rate
        state_path: Optional[Path] = None,
        normalizer_init: Optional[RunningZScoreNormalizer] = None,
        reward_mode: str = "z",
        model_priors: Optional[Dict[str, float]] = None,
        embedding_dim: int = 384,
    ):
        # Resolve exploration rate: user-friendly name takes precedence
        if exploration is not None:
            alpha = ExplorationRate.get(exploration)
        self._default_exploration = float(alpha)

        self.registry = dict(model_registry)
        self.encoder = SentenceTransformer(context_model)
        self.bandit = DisjointLinUCBPolicy(list(self.registry.keys()), dim=embedding_dim, alpha=alpha)
        # Optional cold-start priors (no benchmarks): learned from offline warmup grading of
        # real model outputs. This prevents "all models equal => pick cheapest".
        self.model_priors: Dict[str, float] = {str(k): float(v) for k, v in (model_priors or {}).items()}

        # Reward normalizer for competence reward_z. Persist this state in production.
        self.normalizer = normalizer_init or RunningZScoreNormalizer(
            mean_init=0.65,
            std_init=0.05,
            alpha=0.01,
            clamp=3.0,
            auto_init_from_first_sample=True,
        )

        # Optional: logit reward transform (stationary stretch). This is safe due to strict clipping.
        self.reward_mode = str(reward_mode).lower().strip()
        if self.reward_mode not in {"z", "logit"}:
            raise ValueError("reward_mode must be 'z' or 'logit'")
        self.logit_reward = LogitReward(epsilon=1e-4)

        self.state_path = Path(state_path) if state_path is not None else None
        self.logs: List[RoutingLog] = []

    def add_model(
        self,
        new_model_id: str,
        *,
        clone_from: Optional[str] = None,
        clone_decay: float = 0.9,
        registry_entry: Optional[Dict[str, Any]] = None,
        prior: Optional[float] = None,
    ) -> bool:
        """
        Dynamically add a new model to the router ("brain surgery").

        This is how you "install" a new model (e.g., DeepSeek-V3, GPT-5) into
        a live router without restarting the server.

        Strategies:
            1. CLONING (recommended for upgrades): Clone weights from a similar model
               (e.g., clone DeepSeek-V3 from DeepSeek-V2). The new model inherits
               learned capabilities with slight uncertainty boost.
            2. COLD START: Initialize with identity/zeros. The bandit will explore
               this model to learn its capabilities.

        Args:
            new_model_id: Model ID to add (e.g., "deepseek/deepseek-v3")
            clone_from: Optional model ID to clone weights from
            clone_decay: Multiply cloned weights by this factor (default 0.9)
            registry_entry: Optional registry metadata (cost, latency, display_name).
                            If None, will use minimal defaults.
            prior: Optional prior value for this model. If cloning, inherits from source.

        Returns:
            True if model was added, False if it already exists

        Example:
            # Add GPT-5 by cloning from GPT-4o
            router.add_model(
                "openai/gpt-5",
                clone_from="openai/gpt-4o",
                registry_entry={"display_name": "GPT-5", "cost_per_1k_input": 0.005}
            )

            # Cold start for a completely new model
            router.add_model("brand-new/model-v1")
        """
        if new_model_id in self.registry:
            return False  # Already exists

        # Add to registry
        if registry_entry:
            self.registry[new_model_id] = dict(registry_entry)
        else:
            # Minimal registry entry
            self.registry[new_model_id] = {"display_name": new_model_id}

        # Add to bandit
        added = self.bandit.add_model(new_model_id, clone_from=clone_from, clone_decay=clone_decay)
        if not added:
            # Bandit already had it - sync registry
            return False

        # Set prior
        if prior is not None:
            self.model_priors[new_model_id] = float(prior)
        elif clone_from and clone_from in self.model_priors:
            # Inherit prior from source model
            self.model_priors[new_model_id] = self.model_priors[clone_from]

        return True

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the router.

        Args:
            model_id: Model ID to remove

        Returns:
            True if model was removed, False if it didn't exist
        """
        if model_id not in self.registry:
            return False

        del self.registry[model_id]
        self.bandit.remove_model(model_id)
        self.model_priors.pop(model_id, None)

        return True

    def list_models(self) -> List[str]:
        """Return list of all model IDs in the router."""
        return list(self.registry.keys())

    def _compute_candidate_scores(
        self,
        candidates: List[str],
        x: np.ndarray,
        alpha: float,
        lambda_cost: float,
        lambda_latency: float,
        in_tok: Optional[int],
        out_tok: int,
        use_ucb_for_quality: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized scoring for candidate models.

        Returns:
            quality_hat: predicted quality per model
            utility: utility per model (quality minus cost/latency penalties)
            cost: estimated cost per model
            latency: estimated latency per model
        """
        theta = np.stack([self.bandit.A_inv[m] @ self.bandit.b[m] for m in candidates])
        mean = theta @ x

        A_inv_x = np.stack([self.bandit.A_inv[m] @ x for m in candidates])
        var = (A_inv_x * x).sum(axis=1)
        std = np.sqrt(np.maximum(var, 1e-12))

        prior = np.array([self.model_priors.get(m, 0.0) for m in candidates], dtype=float)
        base_quality = mean + prior
        quality_hat = base_quality + (alpha * std if use_ucb_for_quality else 0.0)

        cost = np.array(
            [self._estimate_cost_usd(m, input_tokens=in_tok, output_tokens=out_tok) for m in candidates],
            dtype=float,
        )
        latency = np.array(
            [self._estimate_latency_s(m, output_tokens=out_tok) for m in candidates],
            dtype=float,
        )
        utility = quality_hat - float(lambda_cost) * cost - float(lambda_latency) * latency
        return quality_hat, utility, cost, latency

    def route(
        self,
        prompt: str,
        *,
        profile: Optional[str] = None,
        exploration: Optional[str] = None,
        lambda_cost: Optional[float] = None,
        lambda_latency: Optional[float] = None,
        max_latency_s: Optional[float] = None,
        max_cost: Optional[float] = None,
        quality_floor: Optional[Dict[str, float]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimate_input_tokens: bool = True,
        default_output_tokens: int = 600,
        epsilon: float = 0.05,
        candidate_models: Optional[List[str]] = None,
        use_ucb_for_quality: bool = True,
    ) -> Tuple[str, RoutingLog]:
        """
        Hot path: route by maximizing real-time utility.

        For each candidate model:
            U_model = quality_hat - lambda_cost * cost - lambda_latency * latency

        - quality_hat is predicted by the bandit (mean or UCB).
        - cost/latency are deterministic lookups from the registry.
        - Learning still uses the grader's reward (quality-only).

        Args:
            profile: Named optimization preset. One of:
                - "quality_first": Maximize quality, ignore cost/latency
                - "balanced": Reasonable trade-off (default for most apps)
                - "cost_saver": Aggressive cost optimization
                - "low_latency": Prioritize speed for real-time apps
                If provided, overrides lambda_cost/lambda_latency unless explicitly set.
            exploration: Controls how often the router tries unproven models.
                - "static": Zero exploration, pure exploitation (fintech/production)
                - "safe": Minimal exploration (DEFAULT, recommended)
                - "balanced": Standard bandit behavior
                - "aggressive": Maximum exploration (day-1/shadow mode)
                Can also be a float like "0.75".
            lambda_cost: penalty per unit cost (user/business knob)
            lambda_latency: penalty per second (user/business knob)
            max_latency_s: optional hard constraint (filter models exceeding this)
            max_cost: optional hard budget constraint in $ per request.
                Models exceeding this are masked from consideration.
            quality_floor: benchmark-based filtering. Dict mapping score type to minimum:
                - {"avg": 70} - only models with avg benchmark >= 70%
                - {"math": 80} - only models with math_500 >= 80%
                - {"code": 60} - only models with humaneval >= 60%
                - {"reasoning": 50, "math": 70} - multiple constraints (AND)
                This acts as a "safety rail" preventing the bandit from picking
                cheap models that are statistically incapable of the task.
            input_tokens: override estimated input tokens (for cost/latency math)
            output_tokens: override expected output tokens (for cost/latency math)
            estimate_input_tokens: if True and input_tokens is None, estimate from prompt text
            default_output_tokens: used when output_tokens is None
            epsilon: epsilon-greedy exploration rate
            candidate_models: optional allowlist of models (e.g. compliance mask)
            use_ucb_for_quality: if True use UCB as quality_hat, else use mean.

        Example:
            # Using named presets (recommended)
            model, log = router.route("Write code", profile="balanced", exploration="safe")

            # Day-1 calibration mode (aggressive learning)
            model, log = router.route("Write code", exploration="aggressive")

            # Production fintech (zero exploration)
            model, log = router.route("Analyze risk", exploration="static")
        """
        # Resolve optimization profile
        if profile is not None:
            profile_weights = OptimizationProfile.get(profile)
            if lambda_cost is None:
                lambda_cost = profile_weights["lambda_cost"]
            if lambda_latency is None:
                lambda_latency = profile_weights["lambda_latency"]

        # Default to 0.0 if still None (legacy behavior)
        if lambda_cost is None:
            lambda_cost = 0.0
        if lambda_latency is None:
            lambda_latency = 0.0

        # Resolve exploration rate (alpha for UCB)
        if exploration is not None:
            alpha = ExplorationRate.get(exploration)
        else:
            alpha = self._default_exploration

        x = self.encoder.encode(prompt)
        x = l2_normalize(np.asarray(x, dtype=np.float64))

        in_tok = int(input_tokens) if input_tokens is not None else None
        if in_tok is None and estimate_input_tokens:
            in_tok = estimate_tokens_rough(prompt)
        out_tok = int(output_tokens) if output_tokens is not None else int(default_output_tokens)

        candidates = candidate_models if candidate_models else list(self.registry.keys())
        candidates = [m for m in candidates if m in self.bandit.A]

        # Optional hard constraint: latency ceiling
        if max_latency_s is not None:
            cap = float(max_latency_s)
            filtered: List[str] = []
            for m in candidates:
                lat = float(self._estimate_latency_s(m, output_tokens=out_tok))
                if lat <= cap:
                    filtered.append(m)
            candidates = filtered

        # Optional hard constraint: cost ceiling
        if max_cost is not None:
            cost_cap = float(max_cost)
            filtered = []
            for m in candidates:
                model_cost = float(self._estimate_cost(m, in_tok or 100, out_tok))
                if model_cost <= cost_cap:
                    filtered.append(m)
            candidates = filtered

        # Optional hard constraint: quality floor (benchmark-based masking)
        # This prevents the bandit from picking cheap but weak models
        if quality_floor is not None and isinstance(quality_floor, dict):
            filtered = []
            for m in candidates:
                scores = self.registry.get(m, {}).get("scores", {})
                passes = True
                for score_type, min_val in quality_floor.items():
                    model_score = float(scores.get(score_type, 0))
                    if model_score < float(min_val):
                        passes = False
                        break
                if passes:
                    filtered.append(m)
            candidates = filtered

        if not candidates:
            raise ValueError("No candidate models available after applying constraints.")

        eps = float(max(0.0, min(1.0, epsilon)))
        rng = np.random.default_rng()
        explore = eps > 0 and rng.random() < eps

        quality_hat, utility, cost_arr, latency_arr = self._compute_candidate_scores(
            candidates=candidates,
            x=x,
            alpha=float(alpha),
            lambda_cost=float(lambda_cost),
            lambda_latency=float(lambda_latency),
            in_tok=in_tok,
            out_tok=out_tok,
            use_ucb_for_quality=use_ucb_for_quality,
        )

        best_idx = int(np.argmax(utility))
        best_model = candidates[best_idx]
        best_quality = float(quality_hat[best_idx])
        best_utility = float(utility[best_idx])

        if explore:
            best_idx = int(rng.integers(0, len(candidates)))
            best_model = candidates[best_idx]
            best_quality = float(quality_hat[best_idx])
            best_utility = float(utility[best_idx])

        # propensity under epsilon-greedy with deterministic argmax utility
        prop = (eps / len(candidates)) if explore else ((1.0 - eps) + (eps / len(candidates)))

        model = best_model
        pred_quality = best_quality
        pred_utility = best_utility
        try:
            logger.debug(
                "route_decision",
                extra={
                    "prompt_len": len(prompt or ""),
                    "selected_model": model,
                    "pred_quality": float(pred_quality),
                    "pred_utility": float(pred_utility),
                    "candidates": len(candidates),
                    "explore": bool(explore),
                    "epsilon": eps,
                    "alpha": float(alpha),
                    "lambda_cost": float(lambda_cost),
                    "lambda_latency": float(lambda_latency),
                },
            )
        except Exception:
            pass
        req_id = str(time.time_ns())
        log = RoutingLog(
            request_id=req_id,
            timestamp_s=time.time(),
            prompt=prompt,
            context_vector=x.tolist(),  # NOTE: stored normalized
            selected_model=model,
            predicted_quality=float(pred_quality),
            predicted_utility=float(pred_utility),
            propensity=float(prop),
        )
        self.logs.append(log)
        return model, log

    def rank_prompt(
        self,
        prompt: str,
        *,
        top_k: int = 10,
        profile: Optional[str] = None,
        exploration: Optional[str] = None,
        lambda_cost: Optional[float] = None,
        lambda_latency: Optional[float] = None,
        max_latency_s: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimate_input_tokens: bool = True,
        default_output_tokens: int = 600,
        candidate_models: Optional[List[str]] = None,
        use_ucb_for_quality: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Debug/testing helper: return the top-k model recommendations with full breakdown.

        Args:
            profile: Named optimization preset ("quality_first", "balanced", "cost_saver", "low_latency")
            exploration: Exploration rate ("static", "safe", "balanced", "aggressive")

        Returns rows of:
          {model_id, utility, quality_hat, cost_usd, latency_s, display_name}
        """
        # Resolve optimization profile
        if profile is not None:
            profile_weights = OptimizationProfile.get(profile)
            if lambda_cost is None:
                lambda_cost = profile_weights["lambda_cost"]
            if lambda_latency is None:
                lambda_latency = profile_weights["lambda_latency"]

        if lambda_cost is None:
            lambda_cost = 0.0
        if lambda_latency is None:
            lambda_latency = 0.0

        # Resolve exploration rate
        if exploration is not None:
            alpha = ExplorationRate.get(exploration)
        else:
            alpha = self._default_exploration

        x = self.encoder.encode(prompt)
        x = l2_normalize(np.asarray(x, dtype=np.float64))

        in_tok = int(input_tokens) if input_tokens is not None else None
        if in_tok is None and estimate_input_tokens:
            in_tok = estimate_tokens_rough(prompt)
        out_tok = int(output_tokens) if output_tokens is not None else int(default_output_tokens)

        candidates = candidate_models if candidate_models else list(self.registry.keys())
        candidates = [m for m in candidates if m in self.bandit.A]

        if max_latency_s is not None:
            cap = float(max_latency_s)
            candidates = [m for m in candidates if float(self._estimate_latency_s(m, output_tokens=out_tok)) <= cap]

        quality_hat, utility, cost_arr, latency_arr = self._compute_candidate_scores(
            candidates=candidates,
            x=x,
            alpha=float(alpha),
            lambda_cost=float(lambda_cost),
            lambda_latency=float(lambda_latency),
            in_tok=in_tok,
            out_tok=out_tok,
            use_ucb_for_quality=use_ucb_for_quality,
        )

        k = int(max(1, top_k))
        idx = np.arange(len(candidates))
        if len(candidates) > k:
            top_idx = np.argpartition(utility, -k)[-k:]
        else:
            top_idx = idx
        # Stable ordering: primary by utility desc, secondary by original index to avoid
        # tie-induced drift across runs.
        order = np.lexsort((idx[top_idx], -utility[top_idx]))
        top_idx = top_idx[order]

        rows: List[Dict[str, Any]] = []
        for idx in top_idx:
            m = candidates[idx]
            reg = self.registry.get(m, {})
            rows.append(
                {
                    "model_id": m,
                    "display_name": reg.get("display_name"),
                    "utility": float(utility[idx]),
                    "quality_hat": float(quality_hat[idx]),
                    "prior": float(self.model_priors.get(m, 0.0)),
                    "cost_usd": float(cost_arr[idx]),
                    "latency_s": float(latency_arr[idx]),
                }
            )

        return rows

    def _estimate_cost_usd(self, model_id: str, *, input_tokens: Optional[int], output_tokens: int) -> float:
        """
        Estimate $/request from cached $/1M token prices when available.
        Falls back to registry['cost'] if token pricing isn't present.
        """
        reg = self.registry.get(model_id, {})
        in_cost_per_m = reg.get("input_cost_per_m")
        out_cost_per_m = reg.get("output_cost_per_m")
        if (
            in_cost_per_m is None
            or out_cost_per_m is None
            or float(in_cost_per_m) <= 0
            or float(out_cost_per_m) <= 0
        ):
            return float(reg.get("cost", 0.0) or 0.0)
        in_tok = int(input_tokens) if input_tokens is not None else 0
        out_tok = int(output_tokens)
        return float(float(in_cost_per_m) * (in_tok / 1_000_000.0) + float(out_cost_per_m) * (out_tok / 1_000_000.0))

    def _estimate_latency_s(self, model_id: str, *, output_tokens: int) -> float:
        """
        Estimate latency using cached TTFT + tokens/sec when available.
        Falls back to registry['latency_s'] if not present.
        """
        reg = self.registry.get(model_id, {})
        ttft = reg.get("time_to_first_token_seconds")
        otps = reg.get("output_tokens_per_second")
        if (ttft is None and otps is None) or (float(ttft or 0.0) <= 0.0 and float(otps or 0.0) <= 0.0):
            return float(reg.get("latency_s", 0.0) or 0.0)
        ttft_s = float(ttft or 0.0)
        otps_f = float(otps or 0.0)
        gen_t = (float(output_tokens) / otps_f) if otps_f > 0 else 0.0
        return float(ttft_s + gen_t)

    def process_feedback(
        self,
        grader: Any,
        *,
        responses_by_request_id: Dict[str, str],
    ) -> List[RoutingLog]:
        """
        Cold path: attach responses, grade, compute reward_z, update bandit.

        Returns the updated logs for storage.
        """
        updated: List[RoutingLog] = []
        for log in self.logs:
            resp = responses_by_request_id.get(log.request_id)
            if resp is None:
                continue

            # `grader` can be:
            # - QualityCostPredictor (local competence/vibe grader)
            # - TieredGrader (soft grader + optional teacher/verifier for hard prompts)
            prod = grader.predict_production(
                log.prompt, 
                resp, 
                model_id=log.selected_model,
                reward_normalizer=self.normalizer
            )
            if self.reward_mode == "logit":
                reward_for_update = float(prod["reward_logit"])
            else:
                reward_for_update = prod.get("reward_z")
                if reward_for_update is None:
                    # If no normalizer used, fall back to raw reward (not recommended).
                    reward_for_update = float(prod["reward_raw"])

            # Stored vectors are normalized, but normalize again for safety/back-compat.
            x = l2_normalize(np.asarray(log.context_vector, dtype=np.float64))
            self.bandit.update(log.selected_model, x, float(reward_for_update))

            log.response_text = resp
            log.reward_raw = float(prod["reward_raw"])
            log.reward_logit = float(prod.get("reward_logit")) if prod.get("reward_logit") is not None else None
            log.reward_z = float(prod.get("reward_z")) if prod.get("reward_z") is not None else None
            log.grader_meta = {
                k: v
                for k, v in prod.items()
                if k not in {"reward_raw", "reward_logit", "reward_z"}
            }
            updated.append(log)

        # Clear buffer (production would stream these logs to storage instead).
        self.logs = []
        return updated

    def report_feedback(
        self,
        request_id: str,
        reward: float,
        *,
        response_text: Optional[str] = None,
    ) -> bool:
        """
        Report direct feedback (Human Truth or Hard Truth) for a routed request.

        This is the simple API for:
          - **Human feedback**: User clicks thumbs-up/down, regenerate button
          - **Code execution**: Did the SQL execute? Did the Python parse?

        The router learns immediately via a rank-one update (microseconds).

        Args:
            request_id: The request_id returned by route()
            reward: The reward signal:
                    - 1.0 = success (user accepted, code ran)
                    - 0.0 = neutral (user edited, inconclusive)
                    - -1.0 = failure (user rejected, syntax error)
            response_text: Optional response text (for logging/debugging)

        Returns:
            True if the request was found and updated, False otherwise.

        Example (Hard Truth - Code Execution):
            ```python
            model, log = router.route("Write SQL to get users")
            sql = client.generate(model, ...)

            try:
                db.execute(sql)
                router.report_feedback(log.request_id, reward=1.0)
            except:
                router.report_feedback(log.request_id, reward=0.0)
            ```

        Example (Human Truth - User Feedback):
            ```python
            model, log = router.route(prompt)
            response = client.generate(model, prompt)

            # User clicks "thumbs up"
            router.report_feedback(log.request_id, reward=1.0)

            # User clicks "regenerate"
            router.report_feedback(log.request_id, reward=-0.5)
            ```

        Notes:
            - This bypasses the grader entirely (no LLM-as-a-Judge call).
            - The reward is used directly for the bandit update.
            - For automatic grading, use process_feedback() instead.
        """
        # Find the log
        target_log: Optional[RoutingLog] = None
        for log in self.logs:
            if log.request_id == request_id:
                target_log = log
                break

        if target_log is None:
            return False

        # Normalize context vector and update bandit
        x = l2_normalize(np.asarray(target_log.context_vector, dtype=np.float64))
        self.bandit.update(target_log.selected_model, x, float(reward))

        # Update log metadata
        target_log.response_text = response_text
        target_log.reward_raw = float(reward)

        # Remove from pending logs
        self.logs = [log for log in self.logs if log.request_id != request_id]
        return True

    def save_state(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path is not None else self.state_path
        if p is None:
            raise ValueError("No state_path provided.")
        p.parent.mkdir(parents=True, exist_ok=True)
        # Version 2: store big matrices in a compressed NPZ sidecar.
        npz_path = p.with_suffix(".bandit.npz")
        self.bandit.save_npz(npz_path)
        state = {
            "version": 2,
            "reward_mode": self.reward_mode,
            "bandit_meta": self.bandit.to_meta_dict(),
            "bandit_npz": npz_path.name,
            "normalizer": self.normalizer.state_dict(),
            "registry": self.registry,
            "model_priors": dict(self.model_priors),
        }
        p.write_text(json.dumps(state, indent=2))

    def save_shippable_priors(self, path: Path, *, dtype: Any = np.float16) -> None:
        """
        Export a compact, check-in-friendly priors bundle (<1MB typical).

        Strategy:
          - Use a shared covariance A from an *aggregate* of per-model A matrices.
          - Store per-model b vectors.
          - Persist normalizer state to prevent "calibration shift".

        This is intended for shipping with the library repo (or as a small artifact).
        """
        # Derive a shared A by averaging per-model A matrices.
        # This preserves the overall covariance structure without storing 81 copies.
        models = list(self.bandit.models)
        A_stack = np.stack([np.asarray(self.bandit.A[m], dtype=np.float64) for m in models], axis=0)
        A_shared = np.mean(A_stack, axis=0)

        # Build shared policy and save.
        sp = SharedCovarianceLinUCBPolicy(models, dim=int(self.bandit.dim), alpha=float(self.bandit.alpha), ridge_lambda=float(self.bandit.ridge_lambda), recompute_inv_every=int(self.bandit.recompute_inv_every))
        sp.A = A_shared
        sp.A_inv = np.linalg.inv(sp.A)
        for m in models:
            sp.b[m] = np.asarray(self.bandit.b[m], dtype=np.float64)
            sp._updates[m] = int(self.bandit._updates.get(m, 0))
            
        # Include normalizer state
        extra = {"normalizer": self.normalizer.state_dict()}
        sp.to_shippable_priors_npz(Path(path), dtype=dtype, extra_meta=extra)

    @classmethod
    def load_from_shippable_priors(
        cls,
        *,
        priors_npz: Path,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str = DEFAULT_CONTEXT_MODEL,
        reward_mode: str = "logit",
        alpha: float = 0.5,
        prior_strength: float = 50.0,
    ) -> "BanditRouter":
        """
        Create a router from a compact priors bundle.

        Inflates into a normal disjoint policy (each model gets its own copy of A)
        so the models can diverge online.

        Args:
            priors_npz: Path to priors file (supports both shared and expert formats)
            model_registry: Dict of model_id -> metadata
            context_model: Sentence transformer model for embeddings
            reward_mode: Reward normalization mode ("logit" or "z")
            alpha: UCB exploration parameter
            prior_strength: Confidence multiplier for priors (default 50.0).
                           This calibrates agent confidence to match the reliability
                           of the distillation source. Higher = more exploitation.
                           - 1.0: Use priors as-is (weak, for uniform exploration priors)
                           - 50.0: Strong confidence (recommended for expert-distilled priors)
        """
        priors_data = np.load(priors_npz, allow_pickle=True)
        
        # Extract normalizer state if present
        normalizer_init = None
        if "meta" in priors_data:
            meta_s = str(list(priors_data["meta"])[0])
            meta = json.loads(meta_s)
            if "normalizer" in meta:
                normalizer_init = RunningZScoreNormalizer.from_state_dict(meta["normalizer"])

        # Detect prior format: expert (A_stack) vs shared (A)
        if "A_stack" in priors_data:
            # Expert-distilled priors (already disjoint format)
            return cls._load_expert_priors(
                priors_data=priors_data,
                model_registry=model_registry,
                context_model=context_model,
                reward_mode=reward_mode,
                alpha=alpha,
                prior_strength=prior_strength,
                normalizer_init=normalizer_init,
            )
        else:
            # Shared covariance priors (need inflation)
            return cls._load_shared_priors(
                priors_npz=priors_npz,
                model_registry=model_registry,
                context_model=context_model,
                reward_mode=reward_mode,
                alpha=alpha,
                prior_strength=prior_strength,
                normalizer_init=normalizer_init,
            )

    @classmethod
    def _load_shared_priors(
        cls,
        *,
        priors_npz: Path,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        reward_mode: str,
        alpha: float,
        prior_strength: float,
        normalizer_init: Optional[RunningZScoreNormalizer] = None,
    ) -> "BanditRouter":
        """Load shared covariance priors and inflate to disjoint."""
        shared = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(priors_npz)
        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=float(alpha),
            reward_mode=str(reward_mode),
            embedding_dim=int(shared.dim),
            normalizer_init=normalizer_init,
        )
        # Inflate: copy shared A into every arm, set b, apply strength multiplier
        for m in router.bandit.models:
            # FIX: Only apply prior_strength if it's != 1.0. 
            # Note: A already contains ridge_lambda * I.
            router.bandit.A[m] = np.asarray(shared.A, dtype=np.float64).copy() * prior_strength
            router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])
            router.bandit.b[m] = np.asarray(shared.b.get(m, np.zeros(shared.dim)), dtype=np.float64).copy() * prior_strength
        return router

    @classmethod
    def _load_expert_priors(
        cls,
        *,
        priors_data: Any,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        reward_mode: str,
        alpha: float,
        prior_strength: float,
        normalizer_init: Optional[RunningZScoreNormalizer] = None,
    ) -> "BanditRouter":
        """
        Load expert-distilled priors (already in disjoint format).

        Expert priors are generated via teacher demonstration (80% optimal picks)
        rather than uniform exploration. They encode "expert intuition" and
        benefit significantly from the prior_strength boost.
        """
        model_names = [str(m) for m in priors_data["model_names"]]
        dim = int(priors_data["dim"])
        A_stack = np.asarray(priors_data["A_stack"], dtype=np.float64)
        b_stack = np.asarray(priors_data["b_stack"], dtype=np.float64)

        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=float(alpha),
            reward_mode=str(reward_mode),
            embedding_dim=dim,
            normalizer_init=normalizer_init,
        )

        # Load with strength multiplier applied
        for i, m in enumerate(model_names):
            if m in router.bandit.A:
                router.bandit.A[m] = A_stack[i] * prior_strength
                router.bandit.b[m] = b_stack[i] * prior_strength
                router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])

        return router

    @classmethod
    def load_state(
        cls,
        path: Path,
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
    ) -> "BanditRouter":
        p = Path(path)
        d = json.loads(p.read_text())
        ver = int(d.get("version", 1))
        if ver <= 1:
            # Back-compat: huge JSON state
            router = cls(
                model_registry=d["registry"],
                context_model=context_model,
                alpha=float(d["bandit"]["alpha"]),
                state_path=p,
                normalizer_init=RunningZScoreNormalizer.from_state_dict(d["normalizer"]),
                reward_mode=str(d.get("reward_mode", "z")),
                model_priors=d.get("model_priors", None),
                embedding_dim=int(d["bandit"]["dim"]),
            )
            router.bandit = DisjointLinUCBPolicy.from_state_dict(d["bandit"])
            return router

        bandit_meta = dict(d.get("bandit_meta", {}))
        npz_name = str(d.get("bandit_npz", ""))
        npz_path = (p.parent / npz_name) if npz_name else p.with_suffix(".bandit.npz")
        router = cls(
            model_registry=d["registry"],
            context_model=context_model,
            alpha=float(bandit_meta.get("alpha", 0.5)),
            state_path=p,
            normalizer_init=RunningZScoreNormalizer.from_state_dict(d["normalizer"]),
            reward_mode=str(d.get("reward_mode", "z")),
            model_priors=d.get("model_priors", None),
            embedding_dim=int(bandit_meta.get("dim", 384)),
        )
        router.bandit = DisjointLinUCBPolicy.from_meta_and_npz(bandit_meta, npz_path)
        return router

    @classmethod
    def create(
        cls,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        exploration: str = "safe",
        alpha: Optional[float] = None,
        reward_mode: str = "logit",
        priors: str = "auto",
        prior_strength: float = 50.0,
        user_priors_path: Optional[Path] = None,
        bundled_priors_path: Optional[Path] = None,
    ) -> "BanditRouter":
        """
        Create a router with configurable prior loading.

        Args:
            model_registry: Dict of model_id -> metadata
            context_model: Sentence transformer model for embeddings
            exploration: How often to try unproven models (controls risk appetite):
                - "static": Zero exploration, pure exploitation (fintech/production)
                - "safe": Minimal exploration (DEFAULT, recommended for production)
                - "balanced": Standard bandit behavior
                - "aggressive": Maximum exploration (day-1/shadow mode)
                Can also be a float like "0.75".
            alpha: Raw UCB parameter (overrides exploration if set)
            reward_mode: "logit" or "z"
            priors: Loading strategy:
                - "auto": Try user, then bundled, then cold start
                - "merged": Bundled as base + user additions layered on top (recommended)
                - "user": Only user priors, else cold start
                - "bundled": Only bundled priors, else cold start
                - "none": Cold start (no priors)
            prior_strength: Confidence multiplier for priors (default 50.0).
                           Calibrates agent confidence to match distillation source reliability.
                           - 1.0: Use priors as-is (for uniform exploration priors)
                           - 50.0: Strong confidence (recommended for expert-distilled priors)
                           Higher values = more exploitation, less exploration.
            user_priors_path: Override user priors location
            bundled_priors_path: Override bundled priors location

        Returns:
            Configured BanditRouter

        Prior Locations:
            - BUNDLED: <package>/data/priors/expert_priors.npz (expert-distilled defaults)
            - USER:    ~/.banditgpt/priors/user_priors.npz (user additions)

        Example:
            # Production mode (safe exploration, merged priors, 62% regret reduction)
            router = BanditRouter.create(registry, exploration="safe", priors="merged")

            # Day-1 calibration mode (aggressive learning)
            router = BanditRouter.create(registry, exploration="aggressive")

            # Fintech mode (zero exploration, only exploit known winners)
            router = BanditRouter.create(registry, exploration="static")

            # Cold start (for testing)
            router = BanditRouter.create(registry, priors="none")
        """
        # Resolve exploration -> alpha
        if alpha is not None:
            resolved_alpha = float(alpha)
        else:
            resolved_alpha = ExplorationRate.get(exploration)
        # Resolve paths - prefer expert_priors.npz, fall back to shippable_priors.npz
        from banditgpt._resources import (
            get_user_priors_path,
            get_expert_priors_path,
            get_bundled_priors_path,
            get_priors_path,
        )
        from banditgpt.core.prior_manifest import verify_bundled_prior, load_priors_manifest

        import logging

        logger = logging.getLogger(__name__)

        user_path = user_priors_path or get_user_priors_path()
        default_bundled = get_expert_priors_path()
        fallback_bundled = get_bundled_priors_path()
        priors_dir = get_priors_path().parent
        manifest = load_priors_manifest()

        if bundled_priors_path:
            bundled_path = bundled_priors_path
        elif default_bundled.exists():
            bundled_path = default_bundled
        else:
            bundled_path = fallback_bundled

        # Validate bundled priors when they come from the package (skip custom overrides)
        try:
            if bundled_path.resolve().is_relative_to(priors_dir.resolve()):
                verify_bundled_prior(bundled_path, manifest=manifest)
        except AttributeError:
            # Python <3.9 fallback: best-effort directory check
            if str(priors_dir.resolve()) in str(bundled_path.resolve()):
                verify_bundled_prior(bundled_path, manifest=manifest)
        logger.debug(
            "BanditRouter init resolved priors",
            extra={
                "priors_mode": priors,
                "user_priors_path": str(user_path),
                "bundled_priors_path": str(bundled_path),
                "alpha": resolved_alpha,
                "exploration": exploration,
                "prior_strength": prior_strength,
            },
        )

        # Handle "merged" mode specially
        if priors == "merged":
            return cls._create_merged(
                model_registry=model_registry,
                context_model=context_model,
                alpha=resolved_alpha,
                reward_mode=reward_mode,
                prior_strength=prior_strength,
                user_path=user_path,
                bundled_path=bundled_path,
            )

        priors_to_load: Optional[Path] = None
        priors_source = "none"

        if priors == "auto":
            if user_path.exists():
                priors_to_load = user_path
                priors_source = "user"
            elif bundled_path.exists():
                priors_to_load = bundled_path
                priors_source = "bundled"
        elif priors == "user":
            if user_path.exists():
                priors_to_load = user_path
                priors_source = "user"
        elif priors == "bundled":
            if bundled_path.exists():
                priors_to_load = bundled_path
                priors_source = "bundled"
        # priors == "none" -> priors_to_load stays None

        if priors_to_load is not None:
            router = cls.load_from_shippable_priors(
                priors_npz=priors_to_load,
                model_registry=model_registry,
                context_model=context_model,
                alpha=resolved_alpha,
                reward_mode=reward_mode,
                prior_strength=prior_strength,
            )
            router._priors_source = priors_source
            router._priors_path = priors_to_load
            return router

        # Cold start
        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=resolved_alpha,
            reward_mode=reward_mode,
        )
        router._priors_source = "none"
        router._priors_path = None
        return router

    @classmethod
    def _create_merged(
        cls,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        alpha: float,
        reward_mode: str,
        prior_strength: float,
        user_path: Path,
        bundled_path: Path,
    ) -> "BanditRouter":
        """
        Create router with merged priors (bundled + user additions).

        Merge strategy:
            - Load bundled priors as base (all library models)
            - Layer user priors on top (user's additions/updates)
            - b_vectors: union, user takes precedence on conflicts
            - A_shared: use user's if exists, else bundled
        """
        from banditgpt.core.judge import PriorManager, PriorConfig

        bundled_priors = None
        user_priors = None

        if bundled_path.exists():
            bundled_priors = PriorManager(PriorConfig(source="file", path=bundled_path))._load_npz(bundled_path)
        if user_path.exists():
            user_priors = PriorManager(PriorConfig(source="file", path=user_path))._load_npz(user_path)

        # Determine what we have
        if bundled_priors is None and user_priors is None:
            # Cold start
            router = cls(
                model_registry=model_registry,
                context_model=context_model,
                alpha=alpha,
                reward_mode=reward_mode,
            )
            router._priors_source = "none"
            router._priors_path = None
            return router

        if bundled_priors is None:
            # Only user priors
            merged = user_priors
            priors_source = "user"
        elif user_priors is None:
            # Only bundled priors
            merged = bundled_priors
            priors_source = "bundled"
        else:
            # Merge both: bundled as base, user as overlay
            merged = PriorManager.merge_priors(bundled_priors, user_priors)
            priors_source = "merged"

        # Build router from merged priors
        router = cls._load_from_priors_dict(
            priors=merged,
            model_registry=model_registry,
            context_model=context_model,
            alpha=alpha,
            reward_mode=reward_mode,
            prior_strength=prior_strength,
        )
        router._priors_source = priors_source
        router._priors_path = user_path if user_priors else bundled_path
        return router

    @classmethod
    def _load_from_priors_dict(
        cls,
        priors: Dict[str, Any],
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        alpha: float,
        reward_mode: str,
        prior_strength: float = 50.0,
    ) -> "BanditRouter":
        """Load router from a priors dict (instead of NPZ file)."""
        dim = int(priors.get("dim", 384))
        A_shared = priors.get("A_shared")
        b_vectors = priors.get("b_vectors", {})

        router = cls(
            model_registry=model_registry,
            context_model=context_model,
            alpha=alpha,
            reward_mode=reward_mode,
            embedding_dim=dim,
        )

        # Apply priors to bandit with strength multiplier
        if A_shared is not None:
            A_shared = np.asarray(A_shared, dtype=np.float64) * prior_strength
            for m in router.bandit.models:
                router.bandit.A[m] = A_shared.copy()
                router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])

        for m in router.bandit.models:
            if m in b_vectors:
                router.bandit.b[m] = np.asarray(b_vectors[m], dtype=np.float64).copy() * prior_strength

        return router

    @property
    def priors_source(self) -> str:
        """
        Which priors are currently loaded.

        Returns:
            "user", "bundled", or "none"
        """
        return getattr(self, "_priors_source", "unknown")

    @property
    def priors_path(self) -> Optional[Path]:
        """Path to the loaded priors file, or None if cold start."""
        return getattr(self, "_priors_path", None)


# ---------------------------------------------------------------------------
# Hybrid Router: Bandit-Guided Cascade (Dynamic Chain Architecture)
# ---------------------------------------------------------------------------


@dataclass
class HybridRoutingLog(RoutingLog):
    """Extended routing log for hybrid routing decisions."""
    
    routing_mode: str = "single_shot"  # "single_shot" or "cascade"
    cascade_models: Optional[List[str]] = None
    cascade_attempts: int = 0


class HybridRouter:
    """
    Constraint-Aware Router: Filter, Select, Verify Architecture.
    
    THREE-PHASE ROUTING ARCHITECTURE
    =================================
    
    PHASE 1: HARD FILTERING (SLA Compliance)
    ----------------------------------------
    Mask out any model that violates business constraints:
        - max_cost: Budget constraint ($/1k queries)
        - max_latency: Speed constraint (seconds)
        - min_quality: Benchmark floor (e.g., HumanEval > 70%)
    
    PHASE 2: BANDIT SELECTION (Expertise)
    -------------------------------------
    Pick the best remaining specialist using the learned prior.
    The bandit only sees "legal" candidates that passed Phase 1.
    
    PHASE 3: HYBRID VERIFICATION (Lambda Tuning)
    --------------------------------------------
    Apply cascade_rate (λ) to decide if we need to verify the selection.
    This trades cost for reliability on the selected model.
    
    CASCADE RATE CONTROL (λ):
    -------------------------
    λ directly controls the cascade probability:
        λ = 0.0  → 0% cascade   (pure single-shot, Standard mode)
        λ = 0.3  → ~30% cascade (chatbots)
        λ = 0.5  → ~50% cascade (balanced)
        λ = 0.8  → ~80% cascade (code generation)
        λ = 1.0  → 100% cascade (always verify, max accuracy)
    
    THE BUSINESS KNOBS:
    -------------------
    1. min_quality (Safety Floor): Prevents "cheap but dumb" routing
    2. max_cost / max_latency (FinOps Guardrails): Hard SLA limits
    3. cascade_rate (Quality/Cost Slider): Trades money for reliability
    
    UNIFIED ARCHITECTURE: Standard Mode = λ=0
    =========================================
    
    Rather than maintaining separate codepaths, BanditGPT implements a unified
    routing logic where **Standard Mode is simply the special case of λ=0**.
    
    This ensures that critical safety features—such as hard budget constraints
    (max_cost) and benchmark quality floors (min_quality)—are universally applied
    to all queries, regardless of the verification strategy selected.
    
    Standard Mode (λ=0, Default):
        - Hard Filters: ✓ Applied (max_cost, min_quality)
        - Bandit Selection: ✓ Applied
        - Cascade Verification: ✗ Skipped
        - Result: Pure O(1) speed with constraint enforcement
    
    Hybrid Mode (λ>0):
        - Hard Filters: ✓ Applied (max_cost, min_quality)
        - Bandit Selection: ✓ Applied
        - Cascade Verification: ✓ Applied (λ% of predictions)
        - Result: Higher accuracy with controlled verification cost
    
    THE KEY INSIGHT: FrugalGPT Cannot Scale to 80+ Models
    =====================================================
    
    Standard FrugalGPT (Fixed Chain):
        - Mechanism: Try Model A → Verifier → Fail → Try Model B → ...
        - Latency Scaling: O(N) - linear with chain length
        - Model Pool: Limited to 2-3 models (otherwise latency explodes)
        - Specialist Access: POOR - must be hardcoded in chain
    
    Our Hybrid (Dynamic Chain):
        - Step 1 (Bandit): O(1) vector search over 80+ models
        - Step 2 (Cascade): Execute selected model with optional fallback
        - Latency Scaling: O(1) - constant regardless of pool size
        - Specialist Access: EXCELLENT - dynamically fetched
    
    The "Confident Failure" Hypothesis:
        - FrugalGPT relies on ex-post verification (checking after generation)
        - This fails for complex constraints where "checking" is as hard as "doing"
        - Our Hybrid uses ex-ante prediction to identify high-risk prompts BEFORE generation
        - Result: +2-4% accuracy on Instruction tasks
    
    Usage:
        # SLA-Aware: Use verification_threshold to tune cost/quality
        router = HybridRouter.create(
            model_registry=registry,
            verification_threshold=0.5,  # Balanced for chatbots
        )
        
        # Or use named presets
        router = HybridRouter.create(
            model_registry=registry,
            mode="cost_optimal",  # λ=0.0
        )
        
        result = router.route_with_cascade(
            prompt="Explain CRISPR-Cas9",
            generate_fn=lambda model, prompt: call_llm(model, prompt),
            verify_fn=lambda response: check_quality(response),
        )
    """
    
    # Named presets for cascade_rate (λ)
    CASCADE_PRESETS = {
        "cost_optimal": 0.0,       # Never cascade - pure single-shot (Standard mode)
        "speed": 0.0,              # Alias
        "chatbot": 0.3,            # Low-risk apps, verify ~30%
        "balanced": 0.5,           # Reasonable trade-off, verify ~50%
        "code": 0.8,               # Higher stakes, verify ~80%
        "high_assurance": 0.9,     # Quality-critical, verify ~90%
        "max_accuracy": 1.0,       # Always cascade (like FrugalGPT)
    }
    
    # Backward compatibility alias
    VERIFICATION_PRESETS = CASCADE_PRESETS
    
    def __init__(
        self,
        bandit_router: BanditRouter,
        *,
        fallback_model: str = "openai/gpt-4o",
        cascade_rate: float = 0.0,
        verification_threshold: Optional[float] = None,  # Deprecated alias for cascade_rate
        confidence_threshold: Optional[float] = None,  # Deprecated
        max_cascade_attempts: int = 2,
    ):
        """
        Initialize HybridRouter with a configured BanditRouter.
        
        Args:
            bandit_router: Pre-configured BanditRouter instance
            fallback_model: Model to use when confidence is low or verification fails
            cascade_rate: The λ parameter (0.0-1.0) controlling verification frequency:
                - 0.0: Never cascade (pure single-shot, Standard mode)
                - 0.3: ~30% cascade (chatbots)
                - 0.5: ~50% cascade (balanced)
                - 0.8: ~80% cascade (code generation)
                - 1.0: Always cascade (max accuracy, like FrugalGPT)
            verification_threshold: DEPRECATED. Use cascade_rate instead.
            confidence_threshold: DEPRECATED. Use cascade_rate instead.
            max_cascade_attempts: Maximum models to try in cascade before giving up
        """
        self.router = bandit_router
        self.fallback_model = fallback_model
        self.max_cascade_attempts = int(max_cascade_attempts)
        
        # Handle deprecated confidence_threshold
        if confidence_threshold is not None:
            logger.warning("confidence_threshold is deprecated. Use cascade_rate instead.")
            # Convert: old confidence_threshold was "cascade if below", 
            # cascade_rate is "how aggressive to verify"
            # Higher confidence_threshold meant more single-shot, so invert
            self._cascade_rate = 1.0 - float(confidence_threshold)
        else:
            self._cascade_rate = float(cascade_rate)
        
        # Validate fallback model exists
        if fallback_model not in bandit_router.registry:
            logger.warning(f"Fallback model '{fallback_model}' not in registry. "
                          f"Cascade may fail if primary routing fails.")
    
    @property
    def cascade_rate(self) -> float:
        """
        The λ parameter controlling cascade/verification frequency.
        
        λ = 0.0: Never cascade (Standard mode, pure single-shot)
        λ = 0.5: ~50% cascade (balanced)
        λ = 1.0: Always cascade (max accuracy)
        """
        return self._cascade_rate
    
    @cascade_rate.setter
    def cascade_rate(self, value: float) -> None:
        """Set cascade rate (can be adjusted at runtime)."""
        self._cascade_rate = float(max(0.0, min(1.0, value)))
    
    # Backward compatibility alias
    @property
    def verification_threshold(self) -> float:
        """DEPRECATED: Use cascade_rate instead."""
        return self._cascade_rate
    
    @verification_threshold.setter
    def verification_threshold(self, value: float) -> None:
        """DEPRECATED: Use cascade_rate instead."""
        self._cascade_rate = float(max(0.0, min(1.0, value)))
    
    @property
    def confidence_threshold(self) -> float:
        """
        Internal confidence threshold derived from verification_threshold.
        
        Maps λ to the confidence level below which we cascade:
        - λ=0.0 → confidence_threshold=+∞ (never cascade)
        - λ=0.5 → confidence_threshold=0.5 (cascade when <50% confident)
        - λ=1.0 → confidence_threshold=+∞ (always cascade, handled specially)
        """
        if self._verification_threshold >= 1.0:
            return float('inf')  # Always cascade
        if self._verification_threshold <= 0.0:
            return float('-inf')  # Never cascade
        # Linear mapping: higher λ = higher threshold = more cascading
        return self._verification_threshold
    
    def set_mode(self, mode: str) -> None:
        """
        Set verification threshold using a named preset.
        
        Available modes:
            - "cost_optimal" / "speed": λ=0.0 (pure single-shot)
            - "chatbot": λ=0.5 (balanced for low-risk)
            - "balanced": λ=0.7 (reasonable trade-off)
            - "code": λ=0.85 (higher stakes)
            - "high_assurance": λ=0.9 (quality-critical)
            - "max_accuracy": λ=1.0 (always cascade)
        """
        key = mode.lower().replace("-", "_").replace(" ", "_")
        if key not in self.VERIFICATION_PRESETS:
            valid = list(self.VERIFICATION_PRESETS.keys())
            raise ValueError(f"Unknown mode '{mode}'. Valid modes: {valid}")
        self._verification_threshold = self.VERIFICATION_PRESETS[key]
    
    @classmethod
    def create(
        cls,
        model_registry: Dict[str, Dict[str, Any]],
        *,
        fallback_model: str = "openai/gpt-4o",
        cascade_rate: Optional[float] = None,
        mode: Optional[str] = None,
        verification_threshold: Optional[float] = None,  # Deprecated alias
        confidence_threshold: Optional[float] = None,  # Deprecated
        max_cascade_attempts: int = 2,
        exploration: str = "safe",
        priors: str = "auto",
        prior_strength: float = 50.0,  # Reduced from 1000.0 to prevent exploration collapse
        **router_kwargs,
    ) -> "HybridRouter":
        """
        Create a HybridRouter with a fresh BanditRouter.
        
        The default prior_strength is higher (1000.0) because hybrid mode
        relies on confident predictions to decide between single-shot and cascade.
        
        Args:
            model_registry: Dict of model_id -> metadata
            fallback_model: Model for cascade fallback (should be high-quality)
            cascade_rate: The λ parameter (0.0-1.0) for controlling verification:
                - 0.0: Never cascade (Standard mode, cost-optimal)
                - 0.3: ~30% cascade (chatbots)
                - 0.5: ~50% cascade (balanced)
                - 0.8: ~80% cascade (code generation)
                - 1.0: Always cascade (max accuracy, like FrugalGPT)
            mode: Named preset for cascade_rate. One of:
                - "cost_optimal" / "speed": λ=0.0
                - "chatbot": λ=0.3
                - "balanced": λ=0.5
                - "code": λ=0.8
                - "high_assurance": λ=0.9
                - "max_accuracy": λ=1.0
            verification_threshold: DEPRECATED. Use cascade_rate instead.
            max_cascade_attempts: Max cascade depth
            exploration: Exploration rate for bandit ("safe", "static", etc.)
            priors: Prior loading strategy ("auto", "bundled", etc.)
            prior_strength: Confidence multiplier (default 1000 for hybrid)
            **router_kwargs: Additional args passed to BanditRouter.create()
        
        Returns:
            Configured HybridRouter
        
        Example:
            # Using cascade_rate directly (the λ parameter)
            hybrid = HybridRouter.create(
                model_registry=registry,
                cascade_rate=0.5,  # Balanced - verify ~50%
            )
            
            # Using named preset
            hybrid = HybridRouter.create(
                model_registry=registry,
                mode="code",  # λ=0.8 for code generation
            )
            
            # SLA-aware: adjust at runtime
            hybrid.cascade_rate = 0.9  # Increase verification for high-stakes
        """
        # Resolve cascade_rate from mode preset if provided
        resolved_rate = 0.0  # Default: cost-optimal (Standard mode)
        if mode is not None:
            key = mode.lower().replace("-", "_").replace(" ", "_")
            if key not in cls.CASCADE_PRESETS:
                valid = list(cls.CASCADE_PRESETS.keys())
                raise ValueError(f"Unknown mode '{mode}'. Valid modes: {valid}")
            resolved_rate = cls.CASCADE_PRESETS[key]
        
        # cascade_rate takes precedence over mode
        if cascade_rate is not None:
            resolved_rate = float(cascade_rate)
        
        # Handle deprecated verification_threshold (alias)
        if verification_threshold is not None:
            resolved_rate = float(verification_threshold)
        
        # Handle deprecated confidence_threshold
        if confidence_threshold is not None:
            logger.warning("confidence_threshold is deprecated. Use cascade_rate or mode instead.")
            resolved_rate = 1.0 - float(confidence_threshold)
        
        bandit = BanditRouter.create(
            model_registry=model_registry,
            exploration=exploration,
            priors=priors,
            prior_strength=prior_strength,
            **router_kwargs,
        )
        
        return cls(
            bandit_router=bandit,
            fallback_model=fallback_model,
            cascade_rate=resolved_rate,
            max_cascade_attempts=max_cascade_attempts,
        )
    
    # Latency threshold below which cascade mode is auto-disabled
    # (Cascades typically take 2+ seconds due to multiple LLM calls)
    CASCADE_LATENCY_THRESHOLD = 2.0
    
    def _get_model_cost(self, model_id: str, output_tokens: int = 600) -> float:
        """Get estimated cost for a model in $/1k queries."""
        reg = self.router.registry.get(model_id, {})
        cost = self.router._estimate_cost_usd(model_id, input_tokens=100, output_tokens=output_tokens)
        return cost * 1000  # Convert per-query to per-1k
    
    def _get_model_latency(self, model_id: str, output_tokens: int = 600) -> float:
        """Get estimated latency for a model in seconds."""
        return self.router._estimate_latency_s(model_id, output_tokens=output_tokens)
    
    def _filter_candidates(
        self,
        candidates: List[str],
        *,
        max_cost: Optional[float] = None,
        max_latency: Optional[float] = None,
        quality_floor: Optional[Dict[str, float]] = None,
        output_tokens: int = 600,
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Apply hard constraints to filter the candidate model pool.
        
        Args:
            candidates: List of model IDs to filter
            max_cost: Maximum cost in $/1k queries
            max_latency: Maximum latency in seconds
            quality_floor: Benchmark-based filtering, e.g. {"avg": 70, "math": 80}
            output_tokens: Assumed output tokens for cost/latency estimation
        
        Returns:
            Tuple of (filtered_candidates, exclusion_reasons)
        """
        excluded: Dict[str, str] = {}
        filtered = []
        
        for m in candidates:
            cost = self._get_model_cost(m, output_tokens)
            latency = self._get_model_latency(m, output_tokens)
            
            if max_cost is not None and cost > max_cost:
                excluded[m] = f"cost ${cost:.2f}/1k > ${max_cost:.2f}/1k"
                continue
            if max_latency is not None and latency > max_latency:
                excluded[m] = f"latency {latency:.1f}s > {max_latency:.1f}s"
                continue
            
            # Quality floor: benchmark-based masking
            if quality_floor is not None and isinstance(quality_floor, dict):
                scores = self.router.registry.get(m, {}).get("scores", {})
                failed_constraint = None
                for score_type, min_val in quality_floor.items():
                    model_score = float(scores.get(score_type, 0))
                    if model_score < float(min_val):
                        failed_constraint = f"{score_type}={model_score:.1f}% < {min_val}%"
                        break
                if failed_constraint:
                    excluded[m] = failed_constraint
                    continue
            
            filtered.append(m)
        
        return filtered, excluded
    
    def _find_closest_match(
        self,
        candidates: List[str],
        *,
        max_cost: Optional[float] = None,
        max_latency: Optional[float] = None,
        output_tokens: int = 600,
    ) -> str:
        """
        Find the "least bad" model when all models violate constraints.
        
        Strategy: Score by how close each model is to the constraints.
        """
        best_model = candidates[0] if candidates else self.fallback_model
        best_score = float('inf')
        
        for m in candidates:
            cost = self._get_model_cost(m, output_tokens)
            latency = self._get_model_latency(m, output_tokens)
            
            # Score: sum of constraint violations (lower is better)
            score = 0.0
            if max_cost is not None:
                score += max(0, cost - max_cost) / max_cost  # Normalized overage
            if max_latency is not None:
                score += max(0, latency - max_latency) / max_latency
            
            if score < best_score:
                best_score = score
                best_model = m
        
        return best_model
    
    def route(
        self,
        prompt: str,
        *,
        cascade_rate: Optional[float] = None,
        min_quality: Optional[float] = None,
        max_cost: Optional[float] = None,
        max_latency: Optional[float] = None,
        quality_floor: Optional[Dict[str, float]] = None,
        verification_threshold: Optional[float] = None,  # Deprecated alias
        **route_kwargs,
    ) -> Tuple[str, HybridRoutingLog, str]:
        """
        Constraint-Aware Routing: Filter, Select, Verify.
        
        THREE-PHASE ARCHITECTURE:
        =========================
        
        PHASE 1: HARD FILTERING (SLA Compliance)
            Filter out models violating max_cost, max_latency, min_quality.
        
        PHASE 2: BANDIT SELECTION (Expertise)
            Pick the best remaining specialist using learned prior.
        
        PHASE 3: HYBRID VERIFICATION (Lambda Tuning)
            Apply cascade_rate (λ) to decide if we verify the selection.
        
        Args:
            prompt: The user's prompt
            
            # === THE BUSINESS KNOBS ===
            
            cascade_rate: The λ parameter (0.0-1.0) for verification frequency:
                - 0.0: Never cascade (Standard mode, single-shot)
                - 0.3: ~30% cascade (chatbots)
                - 0.5: ~50% cascade (balanced)
                - 0.8: ~80% cascade (code generation)
                - 1.0: Always cascade (max accuracy)
                If None, uses the router's default cascade_rate.
            
            min_quality: Safety floor - minimum average benchmark score.
                Shorthand for quality_floor={"avg": min_quality}.
                Prevents "cheap but dumb" routing.
            
            max_cost: Budget constraint in $/1k queries.
                Models exceeding this are masked. FinOps guardrail.
            
            max_latency: Latency constraint in seconds.
                Models exceeding this are masked.
                Note: If max_latency < 2.0s, cascade is auto-disabled.
            
            quality_floor: Advanced quality filtering (dict).
                - {"avg": 70} - only models with avg benchmark >= 70%
                - {"math": 80} - only models with math_500 >= 80%
                - {"code": 60, "reasoning": 50} - multiple constraints (AND)
            
            verification_threshold: DEPRECATED. Use cascade_rate instead.
            **route_kwargs: Additional args passed to BanditRouter.route()
        
        Returns:
            Tuple of:
                - model_id: The recommended model (within constraints)
                - log: Extended routing log with constraint info
                - mode: "single_shot" or "cascade"
        
        Example:
            # Budget-constrained routing (FinOps)
            model, log, mode = hybrid.route(
                "Summarize this article",
                max_cost=0.50,  # Hard budget limit
            )
            
            # Quality-assured code generation
            model, log, mode = hybrid.route(
                "Write a Python function",
                min_quality=70,      # Benchmark floor
                cascade_rate=0.8,    # Verify 80% of predictions
            )
            
            # Full SLA control
            model, log, mode = hybrid.route(
                "Generate SQL query",
                max_cost=1.00,
                min_quality=60,
                cascade_rate=0.5,
            )
        """
        # Handle deprecated verification_threshold
        if verification_threshold is not None:
            cascade_rate = verification_threshold
        
        # Handle min_quality shorthand
        if min_quality is not None:
            if quality_floor is None:
                quality_floor = {}
            quality_floor["avg"] = float(min_quality)
        # Get all candidate models
        all_candidates = list(self.router.registry.keys())
        
        # Apply hard constraints (The Mask)
        # This includes cost, latency, AND quality floor constraints
        filtered_candidates, exclusions = self._filter_candidates(
            all_candidates,
            max_cost=max_cost,
            max_latency=max_latency,
            quality_floor=quality_floor,
        )
        
        # Edge case: All models filtered out
        if not filtered_candidates:
            logger.warning(
                f"All {len(all_candidates)} models filtered by constraints "
                f"(max_cost=${max_cost}, max_latency={max_latency}s). "
                f"Falling back to closest match."
            )
            fallback = self._find_closest_match(
                all_candidates, max_cost=max_cost, max_latency=max_latency
            )
            filtered_candidates = [fallback]
        
        # Pass filtered candidates to the underlying bandit
        route_kwargs['candidate_models'] = filtered_candidates
        model_id, base_log = self.router.route(prompt, **route_kwargs)
        confidence = base_log.predicted_quality
        
        # =====================================================================
        # PHASE 3: HYBRID VERIFICATION (Lambda Tuning)
        # =====================================================================
        # 
        # UNIFIED ARCHITECTURE: Standard Mode is just λ=0
        # ------------------------------------------------
        # This is the key design insight: we don't maintain separate codepaths.
        # Standard and Hybrid modes share Phases 1 & 2. Only Phase 3 differs:
        #   - λ=0.0: Skip verification entirely (Standard Mode)
        #   - λ>0.0: Apply cascade verification (Hybrid Mode)
        #
        # This means Standard Mode STILL gets hard constraints (max_cost, min_quality)
        # applied in Phases 1 & 2 above. It's a "Constrained Bandit" not vanilla.
        #
        # Use per-request cascade_rate if provided, otherwise use router default
        lambda_rate = cascade_rate if cascade_rate is not None else self._cascade_rate
        
        # Direct cascade rate control (avoids "Calibration Skew" problem):
        # Model confidences cluster in a narrow range (0.85-0.99), so raw
        # thresholds are ineffective. Instead, λ directly controls cascade %:
        #   λ = 0.0 → 0% cascade   (Standard Mode - default)
        #   λ = 0.3 → ~30% cascade (chatbots)
        #   λ = 0.5 → ~50% cascade (balanced)
        #   λ = 1.0 → 100% cascade (always verify)
        
        if lambda_rate <= 0.0:
            # STANDARD MODE: The confidence threshold is effectively -1.0
            # The cascade check ALWAYS FAILS, so we always take the single-shot path.
            # This is the pure O(1) path with constraint enforcement from Phases 1 & 2.
            mode = "single_shot"
        else:
            # HYBRID MODE: Apply cascade verification based on λ
            # 
            # Direct cascade rate control with uncertainty weighting.
            # Formula: cascade_prob = λ * (1.0 + 0.3 * uncertainty)
            # 
            # This ensures:
            #   λ=0.5 → ~50% cascade (bottom 50% of predictions get verified)
            #   λ=1.0 → ~100% cascade (always verify)
            # 
            # The uncertainty weighting means we cascade MORE on uncertain queries,
            # while still respecting the user's target verification rate.
            import random
            uncertainty = 1.0 - confidence
            cascade_prob = lambda_rate * (1.0 + 0.3 * uncertainty)
            cascade_prob = min(cascade_prob, 1.0)
            mode = "cascade" if random.random() < cascade_prob else "single_shot"
        
        # Auto-disable cascade if latency constraint is tight
        # (Cascades typically take 2+ seconds due to multiple LLM calls)
        if max_latency is not None and max_latency < self.CASCADE_LATENCY_THRESHOLD:
            if mode == "cascade":
                logger.info(
                    f"Latency constraint ({max_latency}s < {self.CASCADE_LATENCY_THRESHOLD}s) "
                    f"forces single_shot mode (cascade disabled)."
                )
                mode = "single_shot"
        
        # Create extended log with constraint info
        log = HybridRoutingLog(
            request_id=base_log.request_id,
            timestamp_s=base_log.timestamp_s,
            prompt=base_log.prompt,
            context_vector=base_log.context_vector,
            selected_model=base_log.selected_model,
            predicted_quality=base_log.predicted_quality,
            predicted_utility=base_log.predicted_utility,
            propensity=base_log.propensity,
            routing_mode=mode,
            cascade_models=[model_id] if mode == "cascade" else None,
        )
        
        # Store constraint info in grader_meta for debugging
        log.grader_meta = {
            "constraints": {
                "max_cost": max_cost,
                "max_latency": max_latency,
                "candidates_before_filter": len(all_candidates),
                "candidates_after_filter": len(filtered_candidates),
                "models_excluded": len(exclusions),
            }
        }
        
        return model_id, log, mode
    
    def route_with_cascade(
        self,
        prompt: str,
        *,
        generate_fn: Any,  # Callable[[str, str], str]
        verify_fn: Optional[Any] = None,  # Callable[[str], bool]
        verification_threshold: Optional[float] = None,
        **route_kwargs,
    ) -> Dict[str, Any]:
        """
        Route and execute with automatic cascade fallback.
        
        This is the "complete" flow - prediction + execution + optional cascade.
        
        Flow (controlled by verification_threshold λ):
            1. Bandit predicts best model + confidence P(Success)
            2. If P(Success) > λ: single-shot (return immediately)
            3. If P(Success) ≤ λ OR verification fails: cascade to fallback
        
        Args:
            prompt: The user's prompt
            generate_fn: Function(model_id, prompt) -> response_text
            verification_threshold: Override λ for this request (0.0-1.0)
            verify_fn: Optional Function(response) -> bool (True = accept)
            **route_kwargs: Additional args passed to BanditRouter.route()
        
        Returns:
            Dict with:
                - response: The final response text
                - model_used: The model that generated the accepted response
                - mode: "single_shot" or "cascade"
                - confidence: Bandit's confidence score
                - attempts: Number of generation attempts
                - log: Full routing log
        
        Example:
            result = hybrid.route_with_cascade(
                prompt="Write SQL to get all users",
                generate_fn=lambda m, p: openrouter.generate(m, p),
                verify_fn=lambda r: validate_sql(r),
            )
            
            print(f"Response from {result['model_used']} ({result['mode']})")
            print(result['response'])
        """
        model_id, log, mode = self.route(
            prompt, 
            verification_threshold=verification_threshold,
            **route_kwargs
        )
        confidence = log.predicted_quality
        
        # Track attempts for cascade
        attempts = []
        
        # === SINGLE-SHOT MODE ===
        if mode == "single_shot":
            response = generate_fn(model_id, prompt)
            
            # If verify_fn provided, check the response
            if verify_fn is not None:
                is_valid = verify_fn(response)
                if is_valid:
                    return {
                        "response": response,
                        "model_used": model_id,
                        "mode": "single_shot",
                        "confidence": confidence,
                        "attempts": 1,
                        "log": log,
                    }
                else:
                    # Verification failed - fall through to cascade
                    mode = "cascade"
                    attempts.append((model_id, response, False))
            else:
                # No verification - trust the bandit
                return {
                    "response": response,
                    "model_used": model_id,
                    "mode": "single_shot",
                    "confidence": confidence,
                    "attempts": 1,
                    "log": log,
                }
        
        # === CASCADE MODE ===
        # Try the bandit's pick first (if not already tried)
        if not attempts:
            response = generate_fn(model_id, prompt)
            is_valid = verify_fn(response) if verify_fn else True
            attempts.append((model_id, response, is_valid))
            
            if is_valid:
                log.routing_mode = "cascade"
                log.cascade_attempts = len(attempts)
                return {
                    "response": response,
                    "model_used": model_id,
                    "mode": "cascade",
                    "confidence": confidence,
                    "attempts": len(attempts),
                    "log": log,
                }
        
        # Try fallback model
        if len(attempts) < self.max_cascade_attempts and self.fallback_model != model_id:
            response = generate_fn(self.fallback_model, prompt)
            is_valid = verify_fn(response) if verify_fn else True
            attempts.append((self.fallback_model, response, is_valid))
            
            if is_valid:
                log.routing_mode = "cascade"
                log.cascade_models = [m for m, _, _ in attempts]
                log.cascade_attempts = len(attempts)
                return {
                    "response": response,
                    "model_used": self.fallback_model,
                    "mode": "cascade",
                    "confidence": confidence,
                    "attempts": len(attempts),
                    "log": log,
                }
        
        # All attempts exhausted - return last response
        last_model, last_response, _ = attempts[-1]
        log.routing_mode = "cascade"
        log.cascade_models = [m for m, _, _ in attempts]
        log.cascade_attempts = len(attempts)
        
        return {
            "response": last_response,
            "model_used": last_model,
            "mode": "cascade_exhausted",
            "confidence": confidence,
            "attempts": len(attempts),
            "log": log,
        }
    
    def report_feedback(
        self,
        request_id: str,
        reward: float,
        *,
        response_text: Optional[str] = None,
    ) -> bool:
        """
        Report feedback for a hybrid routing decision.
        
        Delegates to the underlying BanditRouter for learning.
        """
        return self.router.report_feedback(request_id, reward, response_text=response_text)
    
    @property
    def registry(self) -> Dict[str, Dict[str, Any]]:
        """Access the underlying model registry."""
        return self.router.registry
    
    @property
    def bandit(self) -> DisjointLinUCBPolicy:
        """Access the underlying bandit policy."""
        return self.router.bandit

