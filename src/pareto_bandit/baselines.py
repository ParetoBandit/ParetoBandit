"""
Extended baseline policies for comprehensive comparison.

Implements three additional algorithmic families that complement the
existing ablation suite (Random, EMA, ε-greedy, LinUCB, Corralling):

1. **CostAwareLinTSRouter** — Linear Thompson Sampling (posterior sampling)
   Algorithmic family: Thompson Sampling / Bayesian bandits
   Key difference from LinUCB: Samples θ ~ N(A⁻¹b, v²·A⁻¹) instead of UCB bonus.
   Expected advantage: Better empirical performance in K≥3 regimes where UCB's
   deterministic bonus creates correlated arm selection.

2. **CostAwareLearnedProjRouter** — Learned linear projection + LinUCB
   Algorithmic family: Representation learning bandits (tests PILOT's hypothesis)
   Key difference: Adds a trainable d_proj × d_raw projection matrix that adapts
   to the bandit reward signal via online gradient updates. Tests whether
   learned features improve routing quality within the paretobandit framework.

3. **CostThresholdRouter** — Difficulty-threshold heuristic
   Algorithmic family: Heuristic / rule-based routing
   Key difference: No bandit learning. Routes based on a single difficulty score
   derived from the embedding norm. If difficulty > threshold → expensive model.
   Provides a "how much does learning actually help?" reference point.

All routers follow the same API contract:
    - select_model(context, total_steps=0, candidates=None) → str
    - update(context, model, reward, weight=1.0) → None
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

import numpy as np

try:
    from pareto_bandit.utils import safe_inv
except ImportError:
    from .utils import safe_inv

logger = logging.getLogger(__name__)


def _argmax_random_tiebreak(
    scores: Dict[str, float],
    rng: np.random.Generator | None = None,
) -> str:
    """Return key with max value, breaking ties uniformly at random.

    Args:
        scores: Mapping of candidate names to their scores.
        rng: Explicit NumPy generator for reproducibility. Falls back to
            the global ``np.random`` state if *None*.
    """
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        keys = list(scores.keys())
        idx = rng.integers(len(keys)) if rng is not None else np.random.randint(len(keys))
        return keys[idx]
    max_val = max(finite.values())
    tied = [k for k, v in finite.items() if abs(v - max_val) < 1e-12]
    if len(tied) == 1:
        return tied[0]
    idx = rng.integers(len(tied)) if rng is not None else np.random.randint(len(tied))
    return tied[idx]


# =============================================================================
# 1. Linear Thompson Sampling (LinTS)
# =============================================================================

class CostAwareLinTSRouter:
    """
    Cost-aware Linear Thompson Sampling router.

    At each step, samples θ̃ ~ N(θ̂, v² · A⁻¹) for each arm and selects the
    arm maximizing  θ̃ᵀx - λ·cost.  This replaces the deterministic UCB bonus
    (α√(xᵀA⁻¹x)) with stochastic posterior sampling, which:

    - Naturally balances exploration and exploitation via posterior uncertainty
    - Avoids the need for α-scheduling (no alpha_start/alpha_end)
    - Provides better empirical performance in many multi-arm settings
    - Is the dominant production bandit algorithm (Google, Netflix, Spotify)

    The noise_variance parameter v² controls exploration intensity:
    - v²=1.0: Standard posterior (may over-explore with binary rewards)
    - v²=0.25: Calibrated for binary rewards (Var[Bernoulli(0.5)] = 0.25)
    - v²=0.1: Conservative exploration (closer to greedy)

    Supports both warmup priors and tabula rasa initialization.
    """

    _AINV_REFRESH_INTERVAL = 200
    _COV_JITTER = 1e-8
    _PRED_CLIP = (-1.0, 2.0)

    def __init__(
        self,
        models: List[str],
        context_dim: int,
        model_costs: Dict,
        cost_penalty: float = 0.0,
        noise_variance: float = 0.25,
        warmup_priors: Optional[Dict] = None,
        ridge_lambda: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize LinTS router.

        Args:
            models: List of model identifiers
            context_dim: Dimension of context vectors
            model_costs: Dict mapping model_id → {"normalized_cost": float}
            cost_penalty: λ weight for cost penalty (default: 0.0)
            noise_variance: v² for posterior sampling (default: 0.25 for binary rewards)
            warmup_priors: Optional dict with 'A', 'b', 'context_dim' for warm-start
            ridge_lambda: Ridge regularization for tabula rasa init (default: 1.0)
            seed: RNG seed for reproducibility. *None* creates an unseeded generator.
        """
        self.models = models
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.noise_variance = noise_variance
        self.context_dim = context_dim
        self.t = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

        if warmup_priors is not None:
            _wp_A = warmup_priors.get("A", {})
            _wp_b = warmup_priors.get("b", {})
            _dim = warmup_priors["context_dim"]
            self.A = {
                m: _wp_A[m].copy() if m in _wp_A else np.eye(_dim)
                for m in models
            }
            self.b = {
                m: _wp_b[m].copy() if m in _wp_b else np.zeros(_dim)
                for m in models
            }
        else:
            self.A = {m: ridge_lambda * np.eye(context_dim) for m in models}
            self.b = {m: np.zeros(context_dim) for m in models}

        self.A_inv = {m: safe_inv(self.A[m]) for m in models}

    def select_model(
        self,
        context: np.ndarray,
        total_steps: int = 0,
        candidates: Optional[List[str]] = None,
    ) -> str:
        """
        Select model via Thompson Sampling: sample θ̃ ~ posterior, pick argmax.

        Score = θ̃ᵀx - λ·normalized_cost

        Unlike UCB, this requires no α parameter — the posterior width
        naturally controls exploration.
        """
        scores = {}

        with self._lock:
            eligible = candidates if candidates is not None else self.models
            for model in eligible:
                if model not in self.A_inv:
                    continue

                A_inv_m = self.A_inv[model]
                theta_hat = A_inv_m @ self.b[model]

                # Sample θ̃ from posterior N(θ̂, v² · A⁻¹)
                # Add jitter for positive-definiteness after Sherman-Morrison drift
                cov = self.noise_variance * A_inv_m
                cov += self._COV_JITTER * np.eye(self.context_dim)
                try:
                    theta_sample = self._rng.multivariate_normal(theta_hat, cov)
                except np.linalg.LinAlgError:
                    diag_var = np.maximum(np.diag(cov), 1e-12)
                    theta_sample = self._rng.normal(
                        loc=theta_hat, scale=np.sqrt(diag_var)
                    )

                expected_reward = float(theta_sample @ context)
                expected_reward = np.clip(
                    expected_reward, self._PRED_CLIP[0], self._PRED_CLIP[1]
                )
                normalized_cost = self.model_costs.get(model, {}).get(
                    "normalized_cost", 1.0
                )
                score = expected_reward - self.cost_penalty * normalized_cost
                scores[model] = score

        if not scores:
            fallback = candidates if candidates is not None else self.models
            return fallback[0] if fallback else None

        return _argmax_random_tiebreak(scores, rng=self._rng)

    def update(
        self,
        context: np.ndarray,
        model: str,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        """Standard ridge regression update (same as LinUCB)."""
        if model not in self.A or weight <= 0:
            return

        x = context.flatten()

        with self._lock:
            A_inv_current = self.A_inv[model]
            A_inv_x = A_inv_current @ x
            denominator = 1.0 + weight * (x @ A_inv_x)

            if abs(denominator) > 1e-6:
                self.A_inv[model] = (
                    A_inv_current
                    - weight * np.outer(A_inv_x, A_inv_x) / denominator
                )
                self.A[model] += weight * np.outer(x, x)
                self.b[model] += weight * reward * x
            else:
                self.A[model] += weight * np.outer(x, x)
                self.A_inv[model] = safe_inv(self.A[model])
                self.b[model] += weight * reward * x

            self.t += 1

            # Periodic A_inv recomputation to correct Sherman-Morrison drift
            if self.t % self._AINV_REFRESH_INTERVAL == 0:
                for m in self.models:
                    self.A_inv[m] = safe_inv(self.A[m])


# =============================================================================
# 2. Learned Projection + LinUCB (tests PILOT's representation hypothesis)
# =============================================================================

class CostAwareLearnedProjRouter:
    """
    LinUCB with a trainable linear projection layer on the feature space.

    Architecture:
        raw_features (d_raw) → W (d_proj × d_raw) → projected (d_proj) → LinUCB

    The projection W is updated via online gradient descent on the squared
    prediction error:  L = (reward - θ̂ᵀ W x)²

    This tests the PILOT hypothesis: "Does jointly learning the feature
    representation alongside the bandit policy improve routing quality?"

    Key design choices:
    - Linear projection only (not a full MLP) to stay comparable to LinUCB
    - Gradient updates are rank-1 (cheap, O(d_proj · d_raw) per update)
    - Projection is shared across all models (common feature space)
    - After projection update, A_inv must be fully recomputed (expensive but rare)

    Args:
        models: List of model identifiers
        raw_dim: Dimension of raw feature vectors (e.g., 33 with PCA+bias)
        proj_dim: Dimension of projected feature space (default: 16)
        model_costs: Cost metadata
        cost_penalty: λ weight
        proj_lr: Learning rate for projection updates (default: 0.01)
        alpha_start: Initial UCB exploration coefficient
        alpha_end: Final UCB exploration coefficient
    """

    _AINV_REFRESH_INTERVAL = 200
    _PRED_CLIP = (-1.0, 2.0)

    def __init__(
        self,
        models: List[str],
        raw_dim: int,
        model_costs: Dict,
        proj_dim: int = 16,
        cost_penalty: float = 0.0,
        proj_lr: float = 0.01,
        alpha_start: float = 2.0,
        alpha_end: float = 0.1,
        ridge_lambda: float = 1.0,
        seed: int | None = None,
    ):
        self.models = models
        self.raw_dim = raw_dim
        self.proj_dim = proj_dim
        self.cost_penalty = cost_penalty
        self.model_costs = model_costs
        self.proj_lr = proj_lr
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.t = 0
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / (raw_dim + proj_dim))
        self.W = self._rng.standard_normal((proj_dim, raw_dim)) * scale

        # LinUCB state in projected space
        self.A = {m: ridge_lambda * np.eye(proj_dim) for m in models}
        self.b = {m: np.zeros(proj_dim) for m in models}
        self.A_inv = {m: safe_inv(self.A[m]) for m in models}

    def _project(self, x_raw: np.ndarray) -> np.ndarray:
        """Apply learned projection: z = Wx."""
        return self.W @ x_raw

    def get_current_alpha(self, total_steps: int) -> float:
        """Linear decay schedule matching other routers."""
        if total_steps == 0:
            return self.alpha_end
        fraction = min(self.t / total_steps, 1.0)
        return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)

    def select_model(
        self,
        context: np.ndarray,
        total_steps: int = 0,
        candidates: Optional[List[str]] = None,
    ) -> str:
        """
        Select model using UCB in the learned projected space.

        Score = θ̂ᵀz + α√(zᵀA⁻¹z) - λ·cost,  where z = Wx
        """
        alpha = self.get_current_alpha(total_steps)
        z = self._project(context)
        scores = {}

        with self._lock:
            eligible = candidates if candidates is not None else self.models
            for model in eligible:
                if model not in self.A_inv:
                    continue
                A_inv = self.A_inv[model]
                theta = A_inv @ self.b[model]
                expected_reward = float(theta @ z)
                expected_reward = np.clip(
                    expected_reward, self._PRED_CLIP[0], self._PRED_CLIP[1]
                )
                var = float(z @ A_inv @ z)
                uncertainty = np.sqrt(max(var, 1e-12))
                normalized_cost = self.model_costs.get(model, {}).get(
                    "normalized_cost", 1.0
                )
                score = (
                    expected_reward + alpha * uncertainty
                ) - self.cost_penalty * normalized_cost
                scores[model] = score

        if not scores:
            fallback = candidates if candidates is not None else self.models
            return fallback[0] if fallback else None

        return _argmax_random_tiebreak(scores, rng=self._rng)

    def update(
        self,
        context: np.ndarray,
        model: str,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        """
        Two-phase update:
        1. Update LinUCB (A, b) in projected space (fast, O(d_proj²))
        2. Update projection W via gradient descent on prediction error (O(d_proj·d_raw))
        """
        if model not in self.A or weight <= 0:
            return

        x_raw = context.flatten()
        z = self._project(x_raw)

        with self._lock:
            # Phase 1: Standard LinUCB update in projected space
            A_inv_current = self.A_inv[model]
            A_inv_z = A_inv_current @ z
            denominator = 1.0 + weight * (z @ A_inv_z)

            if abs(denominator) > 1e-6:
                self.A_inv[model] = (
                    A_inv_current
                    - weight * np.outer(A_inv_z, A_inv_z) / denominator
                )
                self.A[model] += weight * np.outer(z, z)
                self.b[model] += weight * reward * z
            else:
                self.A[model] += weight * np.outer(z, z)
                self.A_inv[model] = safe_inv(self.A[model])
                self.b[model] += weight * reward * z

            # Phase 2: Update projection W
            # Gradient of L = (reward - θ̂ᵀ Wx)² w.r.t. W:
            #   dL/dW = -2(reward - θ̂ᵀz) · θ̂ · xᵀ  (outer product, rank-1)
            theta_hat = self.A_inv[model] @ self.b[model]
            prediction = float(theta_hat @ z)
            error = reward - prediction

            # Gradient descent: W ← W + lr · error · θ̂ · xᵀ
            # (dropping the factor of 2 into the learning rate)
            grad = weight * error * np.outer(theta_hat, x_raw)
            self.W += self.proj_lr * grad

            # Clip W to prevent explosion
            w_norm = np.linalg.norm(self.W)
            max_norm = np.sqrt(self.proj_dim * self.raw_dim) * 2.0
            if w_norm > max_norm:
                self.W *= max_norm / w_norm

            self.t += 1

            # Periodic A_inv recomputation to correct Sherman-Morrison drift
            if self.t % self._AINV_REFRESH_INTERVAL == 0:
                for m in self.models:
                    self.A_inv[m] = safe_inv(self.A[m])


# =============================================================================
# 3. Cost Threshold Heuristic (no learning, rule-based)
# =============================================================================

class CostThresholdRouter:
    """
    Simple difficulty-threshold routing heuristic.

    Routes based on a single scalar "difficulty score" derived from the
    embedding: if score > threshold → expensive model, else → cheap model.

    The difficulty score is the L2 norm of the context vector (after PCA).
    Intuition: prompts that are "far from the centroid" in embedding space
    are more unusual/difficult and may benefit from the stronger model.

    This requires no bandit learning and provides a reference for "how much
    does online learning actually contribute?"

    The threshold is swept to generate a Pareto curve:
    - Low threshold → mostly expensive model (high quality, high cost)
    - High threshold → mostly cheap model (lower quality, lower cost)
    """

    def __init__(
        self,
        models: List[str],
        model_costs: Dict,
        threshold: float = 1.0,
        difficulty_feature: str = "norm",
    ):
        """
        Args:
            models: List of model identifiers (exactly 2: cheap and expensive)
            model_costs: Cost metadata
            threshold: Difficulty threshold for routing to expensive model
            difficulty_feature: Method for computing difficulty score
                - "norm": L2 norm of context vector
                - "pc1": First principal component value (absolute)
        """
        self.models = models
        self.model_costs = model_costs
        self.threshold = threshold
        self.difficulty_feature = difficulty_feature

        # Identify cheap and expensive models
        costs = {
            m: model_costs.get(m, {}).get("cost", model_costs.get(m, {}).get("normalized_cost", 0.5))
            for m in models
        }
        sorted_models = sorted(costs.items(), key=lambda x: x[1])
        self.cheap_model = sorted_models[0][0]
        self.expensive_model = sorted_models[-1][0]

    def _difficulty_score(self, context: np.ndarray) -> float:
        """Compute scalar difficulty score from context vector."""
        if self.difficulty_feature == "pc1":
            return abs(float(context[0]))  # First PCA component
        else:  # "norm"
            return float(np.linalg.norm(context[:-1]))  # Exclude bias term

    def select_model(
        self,
        context: np.ndarray,
        total_steps: int = 0,
        candidates: Optional[List[str]] = None,
    ) -> str:
        """Route based on difficulty threshold."""
        score = self._difficulty_score(context)
        if score > self.threshold:
            return self.expensive_model
        else:
            return self.cheap_model

    def update(
        self,
        context: np.ndarray,
        model: str,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        """No-op: threshold router does not learn."""
        pass
