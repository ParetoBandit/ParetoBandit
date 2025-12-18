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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    raise ImportError("Missing dependency: sentence-transformers") from e

try:
    from llm_jury.async_bandit.complexity import LocalComplexityClassifier, NvidiaComplexityClassifier
except Exception:  # pragma: no cover
    LocalComplexityClassifier = None  # type: ignore[assignment]
    NvidiaComplexityClassifier = None  # type: ignore[assignment]

from llm_jury.async_bandit.quality_cost_predictor import (
    QualityCostPredictor,
    LogitReward,
    RunningZScoreNormalizer,
)


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


# For complexity-gated routing (no model-quality priors):
# - We use a prompt complexity model to decide when to *gate* to a strong allowlist.
# - This avoids the cold-start failure mode where cheap models are chosen for
#   technically tricky prompts (e.g., weak-acid/autoionization edge cases).
DEFAULT_STRONG_MODEL_ALLOWLIST = [
    # Reasoning-strong / frontier-ish baselines (OpenRouter ids)
    "deepseek/deepseek-r1",
    "openai/o3-mini-high",
    "openai/o1",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
]

# For domain-knowledge-heavy prompts, avoid "mini" class models by default.
DEFAULT_STRONG_DOMAIN_ALLOWLIST = [
    "deepseek/deepseek-r1",
    "openai/o3-mini-high",
    "openai/o1",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-pro",
]

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

    Args:
      default_output_tokens: assumed completion size for cost/latency estimation
      default_input_tokens: assumed prompt size; if None, caller can use prompt-based estimation externally
      latency_mode:
        - "ttft": use time_to_first_token_seconds only
        - "ttft+gen": TTFT + (output_tokens / output_tokens_per_second) when available
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

        registry[or_id] = {
            "display_name": m.get("display_name", m.get("name")),
            "cost": float(cost_est),
            "latency_s": float(latency_est),
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
        alpha: float = 0.5,
        state_path: Optional[Path] = None,
        normalizer_init: Optional[RunningZScoreNormalizer] = None,
        reward_mode: str = "z",
        model_priors: Optional[Dict[str, float]] = None,
        complexity_classifier: Optional[Any] = None,
        strong_model_allowlist: Optional[List[str]] = None,
        strong_domain_allowlist: Optional[List[str]] = None,
        embedding_dim: int = 384,
    ):
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

        # Optional complexity classifier for gating decisions.
        # This is NOT a model-quality prior; it is a prompt-level policy signal.
        self.complexity_classifier = complexity_classifier
        if self.complexity_classifier is None and LocalComplexityClassifier is not None:
            # Prefer local trained model (no network download).
            cc = LocalComplexityClassifier(device="cpu")
            if cc.is_available():
                self.complexity_classifier = cc
        if self.complexity_classifier is None and NvidiaComplexityClassifier is not None:
            # Fallback: HF model (may download on first use).
            self.complexity_classifier = NvidiaComplexityClassifier(device="cpu")
        self.strong_model_allowlist = list(strong_model_allowlist or DEFAULT_STRONG_MODEL_ALLOWLIST)
        self.strong_domain_allowlist = list(strong_domain_allowlist or DEFAULT_STRONG_DOMAIN_ALLOWLIST)

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

        # Also remove from allowlists if present
        if model_id in self.strong_model_allowlist:
            self.strong_model_allowlist.remove(model_id)
        if model_id in self.strong_domain_allowlist:
            self.strong_domain_allowlist.remove(model_id)

        return True

    def list_models(self) -> List[str]:
        """Return list of all model IDs in the router."""
        return list(self.registry.keys())

    def _complexity_gate_candidates(
        self,
        prompt: str,
        candidates: List[str],
        *,
        enabled: bool,
        min_complexity: float = 0.25,
        easy_confidence_min: float = 0.45,
        min_domain_knowledge: float = 0.60,
        gate_math: bool = True,
    ) -> Tuple[List[str], Optional[Any], bool]:
        """
        Optionally restrict candidates based on prompt complexity.

        Policy:
          - If complexity score >= min_complexity OR (task_type == Math and gate_math),
            restrict to a configured strong allowlist (intersection with candidates).
          - If the intersection is empty, fall back to original candidates.
        """
        if not enabled or self.complexity_classifier is None:
            return candidates, None, False

        try:
            res = self.complexity_classifier.classify(prompt)
        except Exception:
            return candidates, None, False

        # Support both:
        # - LocalComplexityResult (label/confidence/prompt_complexity_score)
        # - NvidiaComplexityResult (prompt_complexity_score/domain_knowledge/task_type_1)
        try:
            score = float(getattr(res, "prompt_complexity_score", 0.0))
        except Exception:
            score = 0.0

        task = str(getattr(res, "task_type_1", "") or "")
        try:
            domain_knowledge = float(getattr(res, "domain_knowledge", 0.0))
        except Exception:
            domain_knowledge = 0.0

        label = getattr(res, "label", None)
        confidence = getattr(res, "confidence", None)
        is_local = label is not None and confidence is not None
        if is_local:
            try:
                label_i = int(label)
            except Exception:
                label_i = -1
            try:
                conf_f = float(confidence)
            except Exception:
                conf_f = 0.0
            # If the local classifier is not confidently "easy", gate upward.
            local_easy = (label_i == 0) and (conf_f >= float(easy_confidence_min)) and (score < float(min_complexity))
            should_gate = not local_easy
        else:
            # NVIDIA model (or others with similar fields): gate if overall complexity high,
            # domain knowledge high, or task type is explicitly Math.
            should_gate = (
                (score >= float(min_complexity))
                or (domain_knowledge >= float(min_domain_knowledge))
                or (gate_math and task.lower() == "math")
            )
        if not should_gate:
            return candidates, res, False

        # Use a stricter allowlist for high domain_knowledge prompts.
        # For the local classifier, "not confidently easy" should route to the *strong domain* set,
        # because these are exactly the brittle edge-cases (math/chem/probability) where small
        # models are often confidently wrong.
        if is_local:
            base_allowlist = self.strong_domain_allowlist
        else:
            base_allowlist = self.strong_domain_allowlist if domain_knowledge >= float(min_domain_knowledge) else self.strong_model_allowlist
        allow = [m for m in base_allowlist if m in candidates]
        return (allow if allow else candidates), res, True

    def route(
        self,
        prompt: str,
        *,
        profile: Optional[str] = None,
        lambda_cost: Optional[float] = None,
        lambda_latency: Optional[float] = None,
        max_latency_s: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimate_input_tokens: bool = True,
        default_output_tokens: int = 600,
        use_complexity_gating: bool = False,
        complexity_min_score: float = 0.25,
        complexity_easy_confidence_min: float = 0.45,
        complexity_min_domain_knowledge: float = 0.60,
        complexity_gate_math: bool = True,
        auto_knobs_from_complexity: bool = True,
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
            lambda_cost: penalty per unit cost (user/business knob)
            lambda_latency: penalty per second (user/business knob)
            max_latency_s: optional hard constraint (filter models exceeding this)
            input_tokens: override estimated input tokens (for cost/latency math)
            output_tokens: override expected output tokens (for cost/latency math)
            estimate_input_tokens: if True and input_tokens is None, estimate from prompt text
            default_output_tokens: used when output_tokens is None
            use_complexity_gating: if True, use prompt complexity to gate to a strong allowlist
            complexity_min_score: gate when complexity score >= this
            complexity_gate_math: gate when task_type_1 == 'Math'
            epsilon: epsilon-greedy exploration rate
            candidate_models: optional allowlist of models (e.g. compliance mask)
            use_ucb_for_quality: if True use UCB as quality_hat, else use mean.

        Example:
            # Using a named profile (recommended for most users)
            model, log = router.route("Write code", profile="balanced")

            # Using explicit weights (for power users)
            model, log = router.route("Write code", lambda_cost=25.0, lambda_latency=0.15)
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

        x = self.encoder.encode(prompt)
        x = l2_normalize(np.asarray(x, dtype=np.float64))

        in_tok = int(input_tokens) if input_tokens is not None else None
        if in_tok is None and estimate_input_tokens:
            in_tok = estimate_tokens_rough(prompt)
        out_tok = int(output_tokens) if output_tokens is not None else int(default_output_tokens)

        candidates = candidate_models if candidate_models else list(self.registry.keys())
        candidates = [m for m in candidates if m in self.bandit.A]
        candidates, _complexity, did_gate = self._complexity_gate_candidates(
            prompt,
            candidates,
            enabled=use_complexity_gating,
            min_complexity=complexity_min_score,
            easy_confidence_min=complexity_easy_confidence_min,
            min_domain_knowledge=complexity_min_domain_knowledge,
            gate_math=complexity_gate_math,
        )

        # If we gated to a strong set, make sure default knobs don't force the
        # router back into "cheapest within strong set" for hard prompts.
        # We only auto-adjust when the caller hasn't provided explicit knobs.
        if auto_knobs_from_complexity and did_gate and lambda_cost == 0.0 and lambda_latency == 0.0 and max_latency_s is None:
            # More quality-biased defaults for complex/domain prompts.
            lambda_cost = 50.0
            lambda_latency = 0.05

        # Optional hard constraint: latency ceiling
        if max_latency_s is not None:
            cap = float(max_latency_s)
            filtered: List[str] = []
            for m in candidates:
                lat = float(self._estimate_latency_s(m, output_tokens=out_tok))
                if lat <= cap:
                    filtered.append(m)
            candidates = filtered

        if not candidates:
            raise ValueError("No candidate models available after applying constraints.")

        eps = float(max(0.0, min(1.0, epsilon)))
        rng = np.random.default_rng()
        explore = eps > 0 and rng.random() < eps

        best_model = candidates[0]
        best_utility = -float("inf")
        best_quality = float("nan")

        # Compute utilities for all candidates (fast: O(#arms * d^2) dominated by dot products).
        for m in candidates:
            theta = self.bandit.A_inv[m] @ self.bandit.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(self.bandit.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            prior = float(self.model_priors.get(m, 0.0))
            ucb = (mean + prior) + self.bandit.alpha * std
            quality_hat = float(ucb if use_ucb_for_quality else (mean + prior))

            # Deterministic penalties (not learned)
            cost = float(self._estimate_cost_usd(m, input_tokens=in_tok, output_tokens=out_tok))
            latency = float(self._estimate_latency_s(m, output_tokens=out_tok))
            utility = quality_hat - float(lambda_cost) * cost - float(lambda_latency) * latency

            if utility > best_utility:
                best_utility = float(utility)
                best_model = m
                best_quality = float(quality_hat)

        if explore:
            best_model = candidates[int(rng.integers(0, len(candidates)))]
            # Recompute predicted quality/utility for the randomly chosen model (for logging).
            theta = self.bandit.A_inv[best_model] @ self.bandit.b[best_model]
            mean = float(theta.dot(x))
            var = float(x.dot(self.bandit.A_inv[best_model]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            prior = float(self.model_priors.get(best_model, 0.0))
            ucb = (mean + prior) + self.bandit.alpha * std
            best_quality = float(ucb if use_ucb_for_quality else (mean + prior))
            cost = float(self._estimate_cost_usd(best_model, input_tokens=in_tok, output_tokens=out_tok))
            latency = float(self._estimate_latency_s(best_model, output_tokens=out_tok))
            best_utility = float(best_quality - float(lambda_cost) * cost - float(lambda_latency) * latency)

        # propensity under epsilon-greedy with deterministic argmax utility
        prop = (eps / len(candidates)) if explore else ((1.0 - eps) + (eps / len(candidates)))

        model = best_model
        pred_quality = best_quality
        pred_utility = best_utility
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
        lambda_cost: Optional[float] = None,
        lambda_latency: Optional[float] = None,
        max_latency_s: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimate_input_tokens: bool = True,
        default_output_tokens: int = 600,
        use_complexity_gating: bool = False,
        complexity_min_score: float = 0.25,
        complexity_easy_confidence_min: float = 0.45,
        complexity_min_domain_knowledge: float = 0.60,
        complexity_gate_math: bool = True,
        auto_knobs_from_complexity: bool = True,
        candidate_models: Optional[List[str]] = None,
        use_ucb_for_quality: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Debug/testing helper: return the top-k model recommendations with full breakdown.

        Args:
            profile: Named optimization preset ("quality_first", "balanced", "cost_saver", "low_latency")

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

        x = self.encoder.encode(prompt)
        x = l2_normalize(np.asarray(x, dtype=np.float64))

        in_tok = int(input_tokens) if input_tokens is not None else None
        if in_tok is None and estimate_input_tokens:
            in_tok = estimate_tokens_rough(prompt)
        out_tok = int(output_tokens) if output_tokens is not None else int(default_output_tokens)

        candidates = candidate_models if candidate_models else list(self.registry.keys())
        candidates = [m for m in candidates if m in self.bandit.A]
        candidates, complexity, did_gate = self._complexity_gate_candidates(
            prompt,
            candidates,
            enabled=use_complexity_gating,
            min_complexity=complexity_min_score,
            easy_confidence_min=complexity_easy_confidence_min,
            min_domain_knowledge=complexity_min_domain_knowledge,
            gate_math=complexity_gate_math,
        )

        if auto_knobs_from_complexity and did_gate and lambda_cost == 0.0 and lambda_latency == 0.0 and max_latency_s is None:
            lambda_cost = 50.0
            lambda_latency = 0.05

        if max_latency_s is not None:
            cap = float(max_latency_s)
            candidates = [m for m in candidates if float(self._estimate_latency_s(m, output_tokens=out_tok)) <= cap]

        rows: List[Dict[str, Any]] = []
        for m in candidates:
            theta = self.bandit.A_inv[m] @ self.bandit.b[m]
            mean = float(theta.dot(x))
            var = float(x.dot(self.bandit.A_inv[m]).dot(x))
            std = float(np.sqrt(max(var, 1e-12)))
            prior = float(self.model_priors.get(m, 0.0))
            ucb = (mean + prior) + self.bandit.alpha * std
            quality_hat = float(ucb if use_ucb_for_quality else (mean + prior))

            cost = float(self._estimate_cost_usd(m, input_tokens=in_tok, output_tokens=out_tok))
            latency = float(self._estimate_latency_s(m, output_tokens=out_tok))
            utility = float(quality_hat - float(lambda_cost) * cost - float(lambda_latency) * latency)

            reg = self.registry.get(m, {})
            rows.append(
                {
                    "model_id": m,
                    "display_name": reg.get("display_name"),
                    "utility": utility,
                    "quality_hat": quality_hat,
                    "prior": float(prior),
                    "cost_usd": cost,
                    "latency_s": latency,
                }
            )

        rows.sort(key=lambda r: float(r["utility"]), reverse=True)
        k = int(max(1, top_k))
        out = rows[:k]
        # Attach complexity metadata to the first row for convenience.
        if out and complexity is not None:
            out[0]["_complexity"] = {
                # Common (both local + NVIDIA)
                "prompt_complexity_score": getattr(complexity, "prompt_complexity_score", None),
                # Local model fields (if present)
                "label": getattr(complexity, "label", None),
                "confidence": getattr(complexity, "confidence", None),
                # NVIDIA fields (if present)
                "task_type_1": getattr(complexity, "task_type_1", None),
                "task_type_2": getattr(complexity, "task_type_2", None),
                "task_type_prob": getattr(complexity, "task_type_prob", None),
                "reasoning": getattr(complexity, "reasoning", None),
                "domain_knowledge": getattr(complexity, "domain_knowledge", None),
            }
        return out

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
            prod = grader.predict_production(log.prompt, resp, reward_normalizer=self.normalizer)
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
        sp.to_shippable_priors_npz(Path(path), dtype=dtype)

    @classmethod
    def load_from_shippable_priors(
        cls,
        *,
        priors_npz: Path,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str = DEFAULT_CONTEXT_MODEL,
        reward_mode: str = "logit",
        alpha: float = 0.5,
    ) -> "BanditRouter":
        """
        Create a router from a compact priors bundle.

        Inflates into a normal disjoint policy (each model gets its own copy of A)
        so the models can diverge online.
        """
        shared = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(priors_npz)
        router = cls(model_registry=model_registry, context_model=context_model, alpha=float(alpha), reward_mode=str(reward_mode), embedding_dim=int(shared.dim))
        # Inflate: copy shared A into every arm, set b.
        for m in router.bandit.models:
            router.bandit.A[m] = np.asarray(shared.A, dtype=np.float64).copy()
            router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])
            router.bandit.b[m] = np.asarray(shared.b.get(m, np.zeros(shared.dim)), dtype=np.float64).copy()
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
        alpha: float = 0.5,
        reward_mode: str = "logit",
        priors: str = "auto",
        user_priors_path: Optional[Path] = None,
        bundled_priors_path: Optional[Path] = None,
    ) -> "BanditRouter":
        """
        Create a router with configurable prior loading.

        Args:
            model_registry: Dict of model_id -> metadata
            context_model: Sentence transformer model for embeddings
            alpha: UCB exploration parameter
            reward_mode: "logit" or "z"
            priors: Loading strategy:
                - "auto": Try user, then bundled, then cold start
                - "merged": Bundled as base + user additions layered on top (recommended)
                - "user": Only user priors, else cold start
                - "bundled": Only bundled priors, else cold start
                - "none": Cold start (no priors)
            user_priors_path: Override user priors location
            bundled_priors_path: Override bundled priors location

        Returns:
            Configured BanditRouter

        Prior Locations:
            - BUNDLED: <package>/data/priors/shippable_priors.npz (library defaults)
            - USER:    ~/.llm_jury/priors/user_priors.npz (user additions)

        Example:
            # Merged mode: bundled + user additions (recommended for most users)
            router = BanditRouter.create(registry, priors="merged")

            # Auto mode: user takes precedence if exists, else bundled
            router = BanditRouter.create(registry, priors="auto")

            # Force bundled priors only (ignore user customizations)
            router = BanditRouter.create(registry, priors="bundled")

            # Cold start (for testing)
            router = BanditRouter.create(registry, priors="none")
        """
        # Resolve paths
        user_path = user_priors_path or (Path.home() / ".llm_jury" / "priors" / "user_priors.npz")
        bundled_path = bundled_priors_path or (Path(__file__).parent.parent.parent / "data" / "priors" / "shippable_priors.npz")

        # Handle "merged" mode specially
        if priors == "merged":
            return cls._create_merged(
                model_registry=model_registry,
                context_model=context_model,
                alpha=alpha,
                reward_mode=reward_mode,
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
                alpha=alpha,
                reward_mode=reward_mode,
            )
            router._priors_source = priors_source
            router._priors_path = priors_to_load
            return router

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

    @classmethod
    def _create_merged(
        cls,
        model_registry: Dict[str, Dict[str, Any]],
        context_model: str,
        alpha: float,
        reward_mode: str,
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
        from llm_jury.async_bandit.judge import PriorManager, PriorConfig

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

        # Apply priors to bandit
        if A_shared is not None:
            A_shared = np.asarray(A_shared, dtype=np.float64)
            for m in router.bandit.models:
                router.bandit.A[m] = A_shared.copy()
                router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])

        for m in router.bandit.models:
            if m in b_vectors:
                router.bandit.b[m] = np.asarray(b_vectors[m], dtype=np.float64).copy()

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

