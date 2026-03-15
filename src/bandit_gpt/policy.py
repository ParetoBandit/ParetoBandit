"""Disjoint LinUCB contextual bandit policy.

Self-contained bandit "brain" with no dependency on :class:`BanditRouter`.
Implements the per-arm ridge regression model, Sherman-Morrison rank-1
updates, Thompson Sampling win-probability estimation, and prior
calibration.

**Complexity analysis**

``update()`` complexity depends on forgetting_factor (gamma):

Non-stationary (gamma < 1.0) -- O(d^2) per update:
  - Exponential decay: ``A *= gamma^dt``, ``b *= gamma^dt`` (scalar multiply)
  - ``A_inv`` updated via scalar division: ``A_inv /= gamma^dt``
  - Sherman-Morrison rank-1 correction for new observation
  - Rare O(d^3) maintenance cycle when regularization floor drops below
    10% of ``init_lambda`` (prevents singularity under prolonged silence)
  - Performance: ~2,710 updates/sec @ d=384

Stationary (gamma = 1.0, default) -- O(d^2) always:
  - No decay; standard Sherman-Morrison for each observation
  - Performance: ~3,051 updates/sec @ d=384

Empirical validation: See ``benchmarks/diagnose_performance.py``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict

import numpy as np

from bandit_gpt.utils import sigmoid, safe_inv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Staleness Variance Inflation Cap
# ---------------------------------------------------------------------------
# When forgetting_factor (gamma) < 1.0 the UCB variance for an arm that has
# not been selected recently is inflated by dividing the base variance by
# gamma^dt.  This implements "optimism in the face of uncertainty" for stale
# arms: an arm the router has avoided for a long time is treated as more
# uncertain, encouraging re-exploration.
#
# Without a floor on gamma^dt the inflation is unbounded as dt -> inf.
# Because cost penalties are additive constants, a sufficiently large dt
# would cause the exploration bonus (alpha * sqrt(var / gamma^dt)) to
# dominate any finite cost_penalty, guaranteeing selection of stale arms
# regardless of cost.
#
# This constant caps the variance multiplier at _MAX_VAR_INFLATION_FACTOR,
# i.e., the effective floor on gamma^dt is 1 / _MAX_VAR_INFLATION_FACTOR.
# At maximum inflation the exploration bonus grows by at most
# sqrt(200) ~ 14x relative to the uninflated baseline -- strong enough to
# force re-exploration but bounded enough to preserve the cost signal.
#
# **This path is only active when forgetting_factor < 1.0 (non-default).**
# The default configuration is stationary (forgetting_factor = 1.0) and
# never applies variance inflation; the cost penalty remains fully effective
# at all times in the default deployment.
#
# If stricter cost control is required alongside non-stationary forgetting,
# reduce alpha or increase cost_penalty rather than relaxing this cap.
_MAX_VAR_INFLATION_FACTOR: float = 200.0
"""Maximum factor by which staleness may inflate the UCB variance term.

Applies only when forgetting_factor < 1.0.  Adjust downward to tighten cost
enforcement in non-stationary deployments, or upward for more aggressive
staleness-driven exploration.  The default (200) keeps the worst-case
exploration bonus within ~14x of the steady-state value.
"""

# ---------------------------------------------------------------------------
# Numerical Stability Constants
# ---------------------------------------------------------------------------

_MAX_STALENESS_DT: int = 1000
"""Maximum dt (time steps since last update) used in gamma^dt decay calculations.

Caps the exponent so that gamma^dt does not underflow to exactly 0.0 for any
gamma in (0, 1).  At dt=1000 and gamma=0.99, gamma^dt ~ 4.3e-5, which is
comfortably above float64 underflow.  Arms unseen for longer than this are
treated identically to arms unseen for exactly _MAX_STALENESS_DT steps.
"""

_REGULARIZATION_FLOOR_FRACTION: float = 0.1
"""Fraction of init_lambda used as the proactive-regularization trigger threshold.

During each update, if effective regularization (estimated from the diagonal
of A) has decayed to below init_lambda * _REGULARIZATION_FLOOR_FRACTION,
a top-up injection is triggered before the rank-1 Sherman-Morrison update.
This prevents the covariance matrix from approaching singularity in
high-traffic, non-stationary settings before the SM denominator check fires.
"""

_SM_DENOMINATOR_THRESHOLD: float = 1e-6
"""Minimum absolute denominator for the Sherman-Morrison rank-1 update.

The SM denominator is 1 + weight * x^T A^{-1} x.  For any positive-definite
A (guaranteed by initialisation and rank-1 PD updates) this quantity is
always >= 1.  Falling below this threshold therefore signals floating-point
corruption of the cached A_inv matrix rather than a legitimate near-zero
value, and triggers a full O(d^3) recomputation from scratch.
"""

_OUTPUT_COST_MULTIPLIER: float = 3.0
"""Heuristic ratio of output-token cost to input-token cost used to derive
blended_cost_per_m when only cost_usd (input-token price) is provided.

Typical LLM pricing charges 3-5x more for output than input tokens.  The
default of 3.0 is conservative; increase for providers with higher output
premiums.  The blended cost is (input_cost + output_cost) / 2.
"""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _argmax_random_tiebreak(scores: Dict[str, float]) -> str:
    """Return the key with the maximum value, breaking ties uniformly at random.

    Standard ``max(scores, key=scores.get)`` is deterministic when values are
    tied (returns the first key in insertion order).  For bandit algorithms this
    introduces a silent bias: e.g. a freshly-initialized policy always picks the
    first model in the list before any learning has occurred.

    This helper collects all keys sharing the maximum value and returns one
    uniformly at random, eliminating initialization-order bias.
    """
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        keys = list(scores.keys())
        return keys[np.random.randint(len(keys))]
    max_val = max(finite.values())
    tied = [k for k, v in finite.items() if abs(v - max_val) < 1e-12]
    if len(tied) == 1:
        return tied[0]
    return tied[np.random.randint(len(tied))]


def _inflate_variance(
    var: float,
    gamma: float,
    dt: int,
    max_staleness_dt: int = _MAX_STALENESS_DT,
    max_var_inflation: float = _MAX_VAR_INFLATION_FACTOR,
) -> float:
    """Apply staleness-based variance inflation with a bounded cap.

    When ``gamma < 1.0``, divides ``var`` by ``gamma^dt`` to widen the
    confidence interval for stale arms, capped at *max_var_inflation*
    to prevent the exploration bonus from overwhelming additive cost penalties.

    For stationary settings (``gamma == 1.0``) or ``dt == 0``, returns ``var``
    unchanged.

    Args:
        var: Base variance (x^T A^{-1} x).
        gamma: Forgetting factor in (0, 1].
        dt: Steps since last update or selection (non-negative).
        max_staleness_dt: Cap on ``dt`` to avoid float64 underflow in
            ``gamma^dt``.
        max_var_inflation: Maximum multiplicative factor for the variance.

    Returns:
        Inflated (or unchanged) variance.
    """
    if gamma >= 1.0 or dt <= 0:
        return var
    decay_factor = gamma ** min(dt, max_staleness_dt)
    inflation_floor = 1.0 / max_var_inflation
    return var / max(decay_factor, inflation_floor, 1e-12)


def _effective_staleness(
    t: int,
    last_update: Dict[str, int],
    last_played: Dict[str, int],
    model: str,
) -> int:
    """Request-count intervals since the most recent event for *model*.

    ``t`` is a *request counter*: it is incremented at route time by
    ``mark_selected()``, not at feedback time.  Uses
    ``max(last_update, last_played)`` as the reference to prevent
    artificial uncertainty inflation for arms whose feedback is
    still in flight (delayed RLHF scenario).

    Args:
        t: Global logical clock (request counter).
        last_update: Per-model timestamp of last update.
        last_played: Per-model timestamp of last selection.
        model: Model identifier.

    Returns:
        Non-negative staleness in request-count units.
    """
    most_recent = max(
        last_update.get(model, 0),
        last_played.get(model, 0),
    )
    return t - most_recent


@dataclass
class _SMUpdateResult:
    """Return value of :func:`_sherman_morrison_update`."""

    A: np.ndarray
    A_inv: np.ndarray
    b: np.ndarray
    regularization_floor: float
    used_fallback: bool


def _sherman_morrison_update(
    A: np.ndarray,
    A_inv: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    reward: float,
    weight: float,
    init_lambda: float,
    regularization_floor: float,
    model_name: str,
    reg_floor_fraction: float = _REGULARIZATION_FLOOR_FRACTION,
) -> _SMUpdateResult:
    """Perform a rank-1 Sherman-Morrison update on (A, A_inv, b).

    Encapsulates the O(d^2) rank-1 update and the O(d^3) near-singularity
    fallback with gap-based regularization injection.  Callers handle locking
    and time advancement; this function is pure linear algebra.

    Args:
        A: Current precision matrix (d x d).
        A_inv: Cached inverse of A (d x d).
        b: Target vector (d,).
        x: Context vector (d,).
        reward: Observed reward for this observation.
        weight: Importance weight (must be > 0).
        init_lambda: Baseline regularization strength (lambda_0).
        regularization_floor: Current tracked floor for this arm.
        model_name: For logging only.
        reg_floor_fraction: Minimum fraction of *init_lambda* used as
            the fallback regularization injection floor.

    Returns:
        :class:`_SMUpdateResult` containing updated matrices and floor.
    """
    x_outer = weight * np.outer(x, x)
    reward_x = weight * reward * x

    u = x * np.sqrt(weight)
    A_inv_u = A_inv @ u
    v_A_inv = u @ A_inv
    denominator = 1.0 + (u @ A_inv_u)

    if abs(denominator) > _SM_DENOMINATOR_THRESHOLD:
        new_A_inv = A_inv - np.outer(A_inv_u, v_A_inv) / denominator
        new_A = A + x_outer
        new_b = b + reward_x
        return _SMUpdateResult(
            A=new_A, A_inv=new_A_inv, b=new_b,
            regularization_floor=regularization_floor,
            used_fallback=False,
        )

    logger.warning(
        f"[WARN] Sherman-Morrison near-singularity for {model_name}: "
        f"|denominator|={abs(denominator):.2e} < {_SM_DENOMINATOR_THRESHOLD}. "
        f"A_inv has numerically drifted; rebuilding with gap-based "
        f"regularisation injection."
    )
    needed = max(init_lambda - regularization_floor, init_lambda * reg_floor_fraction)
    dim = A.shape[0]
    new_A = A + x_outer + needed * np.eye(dim)
    new_A_inv = safe_inv(new_A)
    new_b = b + reward_x
    return _SMUpdateResult(
        A=new_A, A_inv=new_A_inv, b=new_b,
        regularization_floor=regularization_floor + needed,
        used_fallback=True,
    )


def _safe_multivariate_normal(
    mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    dim: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw from a multivariate normal with numerical safety fallbacks.

    Attempts ``rng.multivariate_normal`` first, then adds jitter,
    then falls back to diagonal sampling if the covariance is too
    ill-conditioned.

    Args:
        mean: Mean vector of shape ``(dim,)``.
        cov: Covariance matrix of shape ``(dim, dim)``.
        n_samples: Number of draws.
        dim: Dimensionality (used for diagonal fallback).
        rng: Explicit NumPy random generator for reproducibility.
            Falls back to ``np.random.default_rng()`` if *None*.

    Returns:
        Array of shape ``(n_samples, dim)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    try:
        return rng.multivariate_normal(mean, cov, n_samples)
    except np.linalg.LinAlgError:
        pass

    jitter = max(np.trace(cov) / dim, 1e-12) * 1e-6
    cov_safe = cov + jitter * np.eye(dim)
    try:
        return rng.multivariate_normal(mean, cov_safe, n_samples)
    except np.linalg.LinAlgError:
        avg_var = max(np.trace(cov) / dim, 1e-12)
        return rng.normal(
            loc=mean, scale=np.sqrt(avg_var), size=(n_samples, dim)
        )


# ---------------------------------------------------------------------------
# Bandit State TypedDict
# ---------------------------------------------------------------------------

class BanditState(TypedDict):
    """Snapshot of bandit state during update operations."""

    A: np.ndarray
    b: np.ndarray
    A_inv: np.ndarray
    timestamp: int
    needs_full_inversion: bool


# ---------------------------------------------------------------------------
# Core Bandit Policy (Disjoint LinUCB)
# ---------------------------------------------------------------------------

class DisjointLinUCBPolicy:
    """Disjoint LinUCB: one ridge regression per arm."""

    def __init__(
        self,
        model_names: List[str],
        dim: int = 384,
        alpha: float = 0.1,
        init_lambda: float = 1.0,
        forgetting_factor: float = 1.0,
        seed: int | None = None,
        max_staleness_dt: int = _MAX_STALENESS_DT,
        reg_floor_fraction: float = _REGULARIZATION_FLOOR_FRACTION,
        max_var_inflation: float = _MAX_VAR_INFLATION_FACTOR,
    ):
        """Initialize Disjoint LinUCB policy.

        REGULARIZATION NOTE (isotropic prior after PCA):
        We initialize A_0 = lambda I, an isotropic regularizer in the
        PCA-transformed feature space.  After PCA, principal components have
        decreasing empirical variance, so equal regularization across all
        directions does not match the per-component scale — effectively
        over-shrinking low-variance components relative to their scale.
        This is a deliberate simplicity choice: isotropic ridge in a PCA
        basis is a standard, stable baseline.  Designing anisotropic or
        variance-matched regularization (e.g., diagonal A_0 scaled by
        component variance, or full whitening before the bandit) is a
        natural extension left for future work.

        Args:
            model_names: List of model identifiers (arms).
            dim: Context vector dimension.
            alpha: Exploration coefficient (UCB bonus multiplier).
            init_lambda: Initialization regularization (A_0 = lambda I).
                Default 1.0 for cold-start stability.
            forgetting_factor: Exponential decay factor (1.0 = stationary,
                <1.0 = adaptive).
            seed: Seed for the internal ``np.random.Generator`` used by
                Thompson Sampling (``get_probabilities``).  *None* creates
                an unseeded generator.
            max_staleness_dt: Cap on ``dt`` in ``gamma^dt`` to prevent float64
                underflow.  Default 1000.
            reg_floor_fraction: Fraction of *init_lambda* below which proactive
                regularization injects a top-up into A.  Default 0.1.
            max_var_inflation: Maximum multiplicative factor for staleness-based
                variance inflation.  Default 200.0.
        """
        self.models = list(model_names)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(forgetting_factor)
        self.init_lambda = float(init_lambda)
        self._rng = np.random.default_rng(seed)
        self.max_staleness_dt = int(max_staleness_dt)
        self.reg_floor_fraction = float(reg_floor_fraction)
        self.max_var_inflation = float(max_var_inflation)

        self.model_locks: Dict[str, threading.Lock] = {
            m: threading.Lock() for m in self.models
        }
        self._lock = threading.Lock()

        self.A = {m: np.eye(self.dim) * self.init_lambda for m in self.models}
        self.b = {m: np.zeros(self.dim, dtype=np.float64) for m in self.models}

        self.A_inv = {m: safe_inv(self.A[m]) for m in self.models}

        # Cached theta = A_inv @ b.  Avoids an O(d^2) matrix-vector product
        # per candidate on every select_arm() call (hot path becomes O(d) dot).
        self.theta: Dict[str, np.ndarray] = {
            m: self.A_inv[m] @ self.b[m] for m in self.models
        }

        self.last_update = {m: 0 for m in self.models}
        self.last_played = {m: 0 for m in self.models}
        self.t = 0

        self.regularization_floor = {m: self.init_lambda for m in self.models}

    # ------------------------------------------------------------------
    # Reset / copy
    # ------------------------------------------------------------------

    def reset_to_tabula_rasa(self) -> None:
        """Reset all learned state to cold-start initialization.

        Discards A, b, and A_inv matrices for every arm, restoring them
        to ``A = init_lambda * I``, ``b = 0``.  Time counters are zeroed
        so the policy re-explores as if no observations had been seen.

        Thread-safe: acquires all per-model locks and the global lock.
        """
        with self._lock:
            for m in self.models:
                with self.model_locks[m]:
                    self.A[m] = np.eye(self.dim) * self.init_lambda
                    self.b[m] = np.zeros(self.dim, dtype=np.float64)
                    self.A_inv[m] = safe_inv(self.A[m])
                    self._refresh_theta(m)
                    self.last_update[m] = 0
                    self.last_played[m] = 0
                    self.regularization_floor[m] = self.init_lambda
            self.t = 0

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle thread locks.

        Locks cannot be pickled or deepcopied directly. We create new locks
        for the clone while deepcopying all numerical state (A, b, A_inv, etc.).
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.models = copy.deepcopy(self.models, memo)
        result.dim = self.dim
        result.alpha = self.alpha
        result.gamma = self.gamma
        result.init_lambda = self.init_lambda
        result.max_staleness_dt = self.max_staleness_dt
        result.reg_floor_fraction = self.reg_floor_fraction
        result.max_var_inflation = self.max_var_inflation
        result.t = self.t
        result.last_update = copy.deepcopy(self.last_update, memo)
        result.last_played = copy.deepcopy(self.last_played, memo)

        result.A = copy.deepcopy(self.A, memo)
        result.b = copy.deepcopy(self.b, memo)
        result.A_inv = copy.deepcopy(self.A_inv, memo)
        result.theta = copy.deepcopy(self.theta, memo)

        result.model_locks = {m: threading.Lock() for m in result.models}
        result._lock = threading.Lock()

        result.regularization_floor = copy.deepcopy(self.regularization_floor, memo)
        result._rng = copy.deepcopy(self._rng, memo)

        return result

    # ------------------------------------------------------------------
    # Arm management
    # ------------------------------------------------------------------

    def add_arm(self, model_name: str) -> None:
        """Add a new arm (model) to the bandit dynamically.

        Prepares all state outside the lock, then publishes atomically.
        Uses double-checked locking: the outer ``if`` is a fast-path
        optimization; the inner re-check under ``self._lock`` prevents
        duplicate registration when two threads call ``add_arm`` for the
        same model concurrently.
        """
        if model_name in self.models:
            return

        new_A = np.eye(self.dim) * self.init_lambda
        new_b = np.zeros(self.dim, dtype=np.float64)
        new_A_inv = safe_inv(new_A)
        new_theta = new_A_inv @ new_b

        with self._lock:
            if model_name in self.models:
                return
            self.A[model_name] = new_A
            self.b[model_name] = new_b
            self.A_inv[model_name] = new_A_inv
            self.theta[model_name] = new_theta
            self.last_update[model_name] = self.t
            self.last_played[model_name] = self.t
            self.regularization_floor[model_name] = self.init_lambda
            self.model_locks[model_name] = threading.Lock()
            self.models.append(model_name)

    def delete_arm(self, model_name: str) -> None:
        """Remove an arm from the bandit.

        Acquires the model's per-arm lock before ``self._lock`` to
        prevent a race with a concurrent ``update()`` that has already
        acquired ``model_locks[model_name]`` and is reading arm state.
        """
        model_lock = self.model_locks.get(model_name)
        if model_lock is None:
            with self._lock:
                self.models = [m for m in self.models if m != model_name]
            return
        with model_lock:
            with self._lock:
                self.models = [m for m in self.models if m != model_name]
                for attr in (self.A, self.b, self.A_inv, self.theta,
                             self.last_update, self.last_played,
                             self.regularization_floor):
                    attr.pop(model_name, None)
                self.model_locks.pop(model_name, None)

    # ------------------------------------------------------------------
    # Cache maintenance
    # ------------------------------------------------------------------

    def refresh_inverse_cache(self) -> None:
        """Recompute ``A_inv`` for all models after a bulk load.

        Builds the new inverse dict outside the lock, then atomically
        swaps the reference so that concurrent ``update()`` calls (which
        read ``self.A_inv[model]`` under ``model_locks`` only) never see
        a partially populated dictionary.
        """
        with self._lock:
            snapshot = {m: (self.A[m], self.b[m]) for m in self.models if m in self.A}
        new_A_inv = {m: safe_inv(A_m) for m, (A_m, _) in snapshot.items()}
        new_theta = {m: new_A_inv[m] @ b_m for m, (_, b_m) in snapshot.items()}
        with self._lock:
            self.A_inv = new_A_inv
            self.theta = new_theta

    def _refresh_theta(self, model: str) -> None:
        """Recompute cached ``theta`` for a single arm after A_inv or b changes.

        Must be called under ``self._lock`` (or at init when no contention exists).
        """
        self.theta[model] = self.A_inv[model] @ self.b[model]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_arm(
        self,
        x: np.ndarray,
        candidates: List[str] | None = None,
        cost_penalties: Dict[str, float] | None = None,
    ) -> Tuple[str, float]:
        """Select the best arm (model) using Upper Confidence Bound (UCB).

        Implements paper Eq. 4::

            a_t = argmax (x^T theta_hat + alpha * sqrt(x^T A^{-1} x) - cost_penalty)

        **Commensurability Note:**
        The UCB term (mean + alpha * std) is in reward units. When processing
        feedback, ``BanditRouter.process_feedback()`` strictly clamps rewards to
        [0, 1].  Furthermore, ``calibrate_priors()`` guarantees that initialized
        priors satisfy ``|x^T theta_hat| <= 0.9`` on the calibration suite, and
        the exploration bonus ``alpha * std`` is structurally bounded by the
        feature embedding space (PCA-whitened to unit variance) and the Tikhonov
        regularization ``lambda``.

        Because the quality score is bounded to a known, stable scale (~[0, 1]
        range), the additive cost penalty ``lambda_c * norm_cost`` (where
        ``norm_cost`` is also in [0, 1]) is provably commensurate. The
        multiplier ``lambda_c`` (e.g. ``cost_penalty=0.1``) represents a direct
        exchange rate: "Sacrifice 10% expected reward to choose the cheapest
        over the most expensive model."  Early in training (cold start),
        ``alpha * std`` may exceed 1.0, intentionally dominating the penalty to
        ensure exploration until the variance shrinks.

        Args:
            x: Context vector.
            candidates: List of candidate model IDs (None = all models).
            cost_penalties: Optional per-model cost penalty
                ``{model_id: lambda * norm_cost}``.  Subtracted from UCB score
                at selection time.

        Returns:
            Tuple of (best_model_id, best_score).
        """
        with self._lock:
            candidates = self.models if candidates is None else candidates
            candidates = [m for m in candidates if m in self.A]
            if not candidates:
                raise ValueError("No candidates available")
            snapshots = {
                m: (self.theta[m], self.A_inv[m], self._effective_staleness(m))
                for m in candidates
            }
            alpha = self.alpha
            gamma = self.gamma

        ucb_scores: Dict[str, float] = {}
        for m, (theta_m, A_inv_m, dt) in snapshots.items():
            mean = float(theta_m.dot(x))

            var = float(x.dot(A_inv_m).dot(x))
            var_inflated = _inflate_variance(
                var, gamma, dt,
                max_staleness_dt=self.max_staleness_dt,
                max_var_inflation=self.max_var_inflation,
            )

            std = float(np.sqrt(max(var_inflated, 1e-12)))
            ucb = mean + alpha * std

            if cost_penalties and m in cost_penalties:
                ucb -= cost_penalties[m]

            ucb_scores[m] = ucb

        best_model = _argmax_random_tiebreak(ucb_scores)
        return best_model, float(ucb_scores[best_model])

    def _effective_staleness(self, model: str) -> int:
        """Delegate to module-level :func:`_effective_staleness`."""
        return _effective_staleness(self.t, self.last_update, self.last_played, model)

    def mark_selected(self, model: str) -> None:
        """Record that *model* was deployed this round (for staleness tracking).

        Called at selection time so that ``_effective_staleness`` can
        distinguish "no update yet because feedback is delayed" from "arm
        genuinely unused."

        **Time Advancement (Request Counter):**
        This method also increments the global logical clock ``self.t`` by 1.
        By advancing time at selection (route time) rather than at feedback
        time, ``self.t`` acts as a *request counter* rather than a *feedback
        counter*.  This ensures that when a burst of delayed feedback arrives
        (e.g. 100 RLHF ratings at once), the updates do not artificially
        inflate the decay factor ``gamma ** dt`` for each other.  Time
        correctly represents the number of environmental interactions, not
        the processing speed.
        """
        with self._lock:
            self.t += 1
            self.last_played[model] = self.t

    # ------------------------------------------------------------------
    # Inference queries
    # ------------------------------------------------------------------

    def get_expected_reward(self, model: str, x: np.ndarray) -> float:
        """Expected reward for *model* given context *x*.

        Uses the cached ``theta`` vector (``A_inv @ b``).

        Args:
            model: Model identifier.
            x: Context feature vector.

        Returns:
            Scalar expected reward (may exceed [0, 1] before clamping).
        """
        return float(self.theta[model].dot(x))

    def get_ucb_variance(self, model: str, x: np.ndarray) -> float:
        """UCB variance term for *model* given context *x*.

        Computes ``x^T A_inv x`` with staleness-based inflation when the
        forgetting factor is active (gamma < 1).

        Args:
            model: Model identifier.
            x: Context feature vector.

        Returns:
            Variance term (non-negative).  Take ``sqrt`` for the UCB bonus.
        """
        var = float(x.dot(self.A_inv[model]).dot(x))
        dt = self._effective_staleness(model)
        return _inflate_variance(
            var, self.gamma, dt,
            max_staleness_dt=self.max_staleness_dt,
            max_var_inflation=self.max_var_inflation,
        )

    def get_probabilities(
        self,
        x: np.ndarray,
        models: List[str],
        n_samples: int = 1000,
        noise_variance: float = 0.25,
    ) -> Dict[str, float]:
        """Probability each model has the highest *quality* (expected reward).

        Samples from the Bayesian posterior for ridge regression::

            theta | D ~ N(A^{-1} b,  sigma^2 * A^{-1}_eff)

        where ``A^{-1}_eff`` incorporates staleness inflation (see
        ``_effective_staleness``).  The model whose posterior draw yields
        the largest ``theta^T x`` wins a sample; probabilities are the
        empirical win fractions across *n_samples* draws.

        **Important:** These probabilities reflect the *quality-only* reward
        model.  Cost and latency penalties applied by ``select_arm()`` are
        **not** incorporated.  Use this for posterior calibration, monitoring,
        and explainability — not as a substitute for the full utility-based
        selection rule.

        Args:
            x: Context vector.
            models: List of model IDs to compare.
            n_samples: Number of Monte Carlo samples (default: 1000).
            noise_variance: sigma^2 for the posterior covariance.  Default 0.25
                is the variance of a Bernoulli(0.5) reward, appropriate for
                binary win/loss feedback.  Override for non-binary rewards.

        Returns:
            Dictionary mapping model_id to probability of being the
            highest-quality arm (sums to 1.0).
        """
        model_samples = {}
        valid_models = [m for m in models if m in self.A]

        snapshots = {}
        with self._lock:
            for m in valid_models:
                A_inv_m = self.A_inv[m].copy()
                theta_hat = self.theta[m].copy()
                dt = self._effective_staleness(m)
                snapshots[m] = (A_inv_m, theta_hat, dt)

        if not snapshots:
            n = len(models) or 1
            return {m: 1.0 / n for m in models}

        for m, (A_inv_m, theta_hat, dt) in snapshots.items():
            # Known approximation: staleness inflation uses trace(A_inv) as a
            # scalar proxy and scales the full covariance uniformly, rather than
            # inflating x^T A_inv x per-direction.  This preserves the O(d^2)
            # cost of _safe_multivariate_normal but loses anisotropic structure:
            # high-variance PCA directions are over-inflated relative to
            # low-variance ones.  For the intended use cases (dashboards,
            # explainability, posterior calibration) this is acceptable;
            # the actual routing decision uses per-direction inflation in
            # select_arm().
            scalar_var = float(np.trace(A_inv_m))
            inflated = _inflate_variance(
                scalar_var, self.gamma, dt,
                max_staleness_dt=self.max_staleness_dt,
                max_var_inflation=self.max_var_inflation,
            )
            if scalar_var > 0 and inflated != scalar_var:
                cov = noise_variance * A_inv_m * (inflated / scalar_var)
            else:
                cov = noise_variance * A_inv_m
            samples = _safe_multivariate_normal(
                theta_hat, cov, n_samples, self.dim, rng=self._rng,
            )
            model_samples[m] = samples @ x

        stacked_samples = np.stack([model_samples[m] for m in valid_models])
        winners = np.argmax(stacked_samples, axis=0)

        counts = Counter(winners)
        probs = {m: 0.0 for m in models}
        for i, m in enumerate(valid_models):
            probs[m] = counts[i] / n_samples
        return probs

    # ------------------------------------------------------------------
    # Learning (update)
    # ------------------------------------------------------------------

    def update(
        self,
        model: str,
        x: np.ndarray,
        reward: float,
        weight: float = 1.0,
        advance_time: bool = True,
    ) -> None:
        """Update the model's A and b matrices with a new observation.

        **Per-Model Locking:**
        Fine-grained locking eliminates the lost-update race condition. Each
        model has its own lock, so updates to Model A don't block updates to
        Model B.

        **Proactive Regularization Floor:**
        Tracks effective lambda decay and proactively maintains eigenvalue
        floor.  Prevents singularity in low-traffic regimes with forgetting
        factor < 1.0.  Amortized O(d^2) with rare O(d^3) maintenance cycles.

        **Time Convention (advance_time):**
        ``self.t`` is a *request counter*, not a *feedback counter*.  It is
        advanced at route/selection time by ``mark_selected()``, so that a
        burst of delayed feedback arriving at once (e.g. a daily RLHF batch)
        does not artificially inflate ``gamma**dt`` decay for each feedback
        event processed in the batch.

        When called from ``process_feedback()`` (the standard online path),
        pass ``advance_time=False`` because time was already advanced by
        ``mark_selected()`` at route time.

        When called directly for offline/batch replay (i.e. without a
        preceding ``route()`` call), leave ``advance_time=True`` (the default)
        so each replayed observation still increments time.

        Args:
            model: Model identifier.
            x: Context vector.
            reward: Observed reward.
            weight: Importance weight for this update (default 1.0).
            advance_time: Whether to increment ``self.t``.  Defaults to True.
                Set to False when time was already advanced at route time via
                ``mark_selected()`` to prevent double-counting.
        """
        if model not in self.A:
            return

        if weight < 0:
            logger.warning(
                f"Negative weight={weight:.4f} for {model}; skipping update "
                f"(negative weight would corrupt A_inv via sqrt(w))"
            )
            return

        if weight == 0:
            return

        with self.model_locks[model]:
            with self._lock:
                if model not in self.A:
                    return
                current_t = self.t

            dt = 0
            decay_factor = 1.0
            if self.gamma < 1.0:
                dt = current_t - self.last_update[model]
                decay_factor = self.gamma ** min(dt, self.max_staleness_dt)

            current_lambda = self.regularization_floor.get(model, self.init_lambda)
            new_lambda = current_lambda * decay_factor
            lambda_threshold = self.init_lambda * self.reg_floor_fraction

            if new_lambda < lambda_threshold:
                # MAINTENANCE MODE: Inject fresh regularization (Rare O(d^3))
                #
                # Only when gamma < 1.0 AND the arm has gone so long without
                # an update that gamma^dt < 0.1 — i.e., >90% of the original
                # prior has been forgotten.
                logger.info(
                    f"[FIX] Maintenance: Restoring regularization floor for {model} "
                    f"(lambda_eff={new_lambda:.2e} < {lambda_threshold:.2e})"
                )

                missing_lambda = self.init_lambda - new_lambda

                new_A = (self.A[model] * decay_factor) + (missing_lambda * np.eye(self.dim))
                new_b = self.b[model] * decay_factor
                new_A_inv = safe_inv(new_A)

                self.regularization_floor[model] = self.init_lambda

                with self._lock:
                    self.A[model] = new_A
                    self.b[model] = new_b
                    self.A_inv[model] = new_A_inv
                    self._refresh_theta(model)
                    self.last_update[model] = current_t

            else:
                if self.gamma < 1.0:
                    self.regularization_floor[model] = new_lambda
                    new_A = self.A[model] * decay_factor
                    new_b = self.b[model] * decay_factor
                    new_A_inv = self.A_inv[model] / decay_factor

                    with self._lock:
                        self.A[model] = new_A
                        self.b[model] = new_b
                        self.A_inv[model] = new_A_inv
                        self._refresh_theta(model)
                        self.last_update[model] = current_t

            result = _sherman_morrison_update(
                A=self.A[model],
                A_inv=self.A_inv[model],
                b=self.b[model],
                x=x,
                reward=reward,
                weight=weight,
                init_lambda=self.init_lambda,
                regularization_floor=self.regularization_floor.get(model, self.init_lambda),
                model_name=model,
                reg_floor_fraction=self.reg_floor_fraction,
            )
            self.regularization_floor[model] = result.regularization_floor

            with self._lock:
                self.A[model] = result.A
                self.b[model] = result.b
                self.A_inv[model] = result.A_inv
                self._refresh_theta(model)
                if advance_time:
                    self.t += 1

    # ------------------------------------------------------------------
    # Numerical stability
    # ------------------------------------------------------------------

    def _check_numerical_stability(
        self,
        model: str,
        config: "RouterConfig | None" = None,
    ) -> None:
        """Safety check for numerical stability using trace of inverse.

        Uses O(d) trace computation instead of O(d^3) eigenvalue decomposition.
        If trace(A_inv) exceeds the threshold, triggers a regularization reset.

        Args:
            model: Model identifier to check.
            config: RouterConfig with stability thresholds (optional).
        """
        # Import here to avoid circular import at module level.
        # policy.py -> types.py is fine, but we avoid a hard top-level
        # dependency so the module can be imported independently.
        if config is None or model not in self.A_inv:
            return

        trace = np.trace(self.A_inv[model])

        threshold = getattr(config, "stability_threshold", 1000 * self.dim)

        if trace > threshold:
            logger.warning(
                f"[GUARD] Numerical instability detected for {model}: "
                f"trace(A_inv)={trace:.2e} > {threshold:.2e}. "
                f"Triggering regularization reset."
            )

            reg_lambda = self.init_lambda
            with self.model_locks[model]:
                self.A[model] += reg_lambda * np.eye(self.dim)
                new_A_inv = safe_inv(self.A[model])
                with self._lock:
                    self.A_inv[model] = new_A_inv
                    self._refresh_theta(model)
                    self.regularization_floor[model] = self.regularization_floor.get(
                        model, self.init_lambda
                    ) + reg_lambda
                    new_trace = np.trace(new_A_inv)

            logger.info(
                f"[OK] Regularization reset complete for {model}. "
                f"New trace(A_inv)={new_trace:.2f}"
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path | str) -> None:
        """Save A, b, and temporal metadata to a compressed NPZ file.

        Persists sufficient statistics (A, b) and temporal metadata
        (t, last_update, last_played, regularization_floor) so that
        forgetting-factor and staleness logic resume correctly after
        a checkpoint restore.
        """
        data: Dict[str, Any] = {
            "_metadata_dim": self.dim,
            "_metadata_models": list(self.models),
            "_metadata_policy": "disjoint",
            "__temporal__t": self.t,
        }
        last_update_arr = np.array(
            [self.last_update.get(m, 0) for m in self.models], dtype=np.int64
        )
        last_played_arr = np.array(
            [self.last_played.get(m, 0) for m in self.models], dtype=np.int64
        )
        reg_floor_arr = np.array(
            [self.regularization_floor.get(m, self.init_lambda)
             for m in self.models],
            dtype=np.float64,
        )
        data["__temporal__last_update"] = last_update_arr
        data["__temporal__last_played"] = last_played_arr
        data["__temporal__reg_floor"] = reg_floor_arr

        for m in self.models:
            data[f"{m}_A"] = self.A[m]
            data[f"{m}_b"] = self.b[m]
        np.savez_compressed(path, **data)

    def load_state(self, path: Path | str) -> None:
        """Load A and b matrices from a compressed NPZ file with dimension validation.

        All file I/O, validation, and ``safe_inv`` computation happen
        outside any lock.  State references are then swapped atomically
        under ``self._lock`` so that concurrent readers (``select_arm``,
        ``get_probabilities``) never observe torn state.

        Raises:
            ValueError: If saved dimension doesn't match current bandit dimension.
        """
        data = np.load(path)

        if "_metadata_dim" in data:
            saved_dim = int(data["_metadata_dim"])
            if saved_dim != self.dim:
                raise ValueError(
                    f"Dimension mismatch: saved state has dim={saved_dim}, "
                    f"but current bandit expects dim={self.dim}. "
                    f"This can happen when:\n"
                    f"  1. PCA configuration changes (raw embeddings vs PCA-compressed)\n"
                    f"  2. Virtual anchor set is modified\n"
                    f"  3. Feature engineering pipeline changes\n"
                    f"To fix:\n"
                    f"  - Delete the saved state file to start fresh, OR\n"
                    f"  - Ensure PCA and feature config match the saved state"
                )
        else:
            logger.warning(
                f"Loading state from {path} without dimension metadata. "
                f"This may cause issues if dimensions have changed. "
                f"Current dim={self.dim}"
            )

        staged: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for m in self.models:
            a_key = f"{m}_A"
            b_key = f"{m}_b"
            if a_key in data and b_key in data:
                A_loaded = data[a_key]
                b_loaded = data[b_key]

                if A_loaded.shape != (self.dim, self.dim):
                    raise ValueError(
                        f"Matrix A for model '{m}' has wrong shape: "
                        f"expected ({self.dim}, {self.dim}), got {A_loaded.shape}"
                    )
                if b_loaded.shape != (self.dim,):
                    raise ValueError(
                        f"Vector b for model '{m}' has wrong shape: "
                        f"expected ({self.dim},), got {b_loaded.shape}"
                    )
                staged[m] = (A_loaded, b_loaded, safe_inv(A_loaded))

        saved_t: int | None = None
        saved_last_update: Dict[str, int] = {}
        saved_last_played: Dict[str, int] = {}
        saved_reg_floor: Dict[str, float] = {}
        if "__temporal__t" in data:
            saved_t = int(data["__temporal__t"])
        if "__temporal__last_update" in data:
            arr = data["__temporal__last_update"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_last_update[m] = int(arr[i])
        if "__temporal__last_played" in data:
            arr = data["__temporal__last_played"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_last_played[m] = int(arr[i])
        if "__temporal__reg_floor" in data:
            arr = data["__temporal__reg_floor"]
            models_list = (
                list(data["_metadata_models"])
                if "_metadata_models" in data else list(self.models)
            )
            for i, m in enumerate(models_list):
                if i < len(arr):
                    saved_reg_floor[m] = float(arr[i])

        with self._lock:
            for m, (A_new, b_new, A_inv_new) in staged.items():
                self.A[m] = A_new
                self.b[m] = b_new
                self.A_inv[m] = A_inv_new
                self._refresh_theta(m)
            if saved_t is not None:
                self.t = saved_t
            for m in self.models:
                if m in saved_last_update:
                    self.last_update[m] = saved_last_update[m]
                if m in saved_last_played:
                    self.last_played[m] = saved_last_played[m]
                if m in saved_reg_floor:
                    self.regularization_floor[m] = saved_reg_floor[m]


# ---------------------------------------------------------------------------
# Standalone prior calibration (operates on a DisjointLinUCBPolicy)
# ---------------------------------------------------------------------------

def calibrate_priors(
    bandit: DisjointLinUCBPolicy,
    target_max_pred: float = 0.9,
    calibration_contexts: List[np.ndarray] | None = None,
) -> None:
    """Auto-calibrate loaded priors on *bandit* so predictions stay in a safe range.

    Two-pass calibration:

    **Pass 1 -- Bias probe** (fast, catches the most common failure):
    Probes each model with ``[0,...,0,1]`` (bias-only context).  If the bias
    prediction exceeds 1.5, the bias component of theta is clamped via
    theta-reconstruction (``b = A @ theta_new``).

    **Pass 2 -- Suite probe** (comprehensive, catches PCA-dimension explosions):
    Probes each model with a built-in suite of basis-independent feature
    vectors (axis-aligned, random unit-norm, uniform).  If the caller supplies
    *calibration_contexts*, those are appended to the suite.  If any prediction
    exceeds 1.5, theta is globally rescaled so the worst-case prediction equals
    *target_max_pred*.

    .. warning::

        **Not thread-safe.**  This function writes ``bandit.b[m]`` without
        acquiring any lock.  It must be called during single-threaded
        initialization (e.g. inside ``BanditRouter.create()``) *before*
        the router is exposed to concurrent ``route()`` / ``update()``
        traffic.

    Args:
        bandit: A ``DisjointLinUCBPolicy`` whose A/b matrices will be modified
                in place.
        target_max_pred: Target maximum absolute prediction over the probe suite
                       (default: 0.9).
        calibration_contexts: Optional list of domain-specific context vectors
                       (numpy arrays of shape ``(dim,)``) appended to the
                       built-in geometry probes.
    """
    d = bandit.dim
    probes: List[tuple] = []

    bias_probe = np.zeros(d)
    bias_probe[-1] = 1.0
    probes.append(("bias", bias_probe))

    pca_dims = list(range(d - 1))
    if len(pca_dims) > 8:
        step = max(1, len(pca_dims) // 8)
        pca_dims = pca_dims[::step][:8]
    for i in pca_dims:
        e_i = np.zeros(d)
        e_i[i] = 1.0
        probes.append((f"axis_{i}", e_i))

    rng = np.random.default_rng(42)
    for k in range(4):
        v = rng.standard_normal(d)
        v /= (np.linalg.norm(v) + 1e-12)
        probes.append((f"random_{k}", v))

    uniform = np.ones(d) / np.sqrt(d)
    probes.append(("uniform", uniform))

    if calibration_contexts is not None:
        for i, ctx in enumerate(calibration_contexts):
            ctx = np.asarray(ctx, dtype=float).flatten()
            if ctx.shape[0] != d:
                logger.warning(
                    f"Skipping calibration_context[{i}]: shape {ctx.shape} "
                    f"!= dim {d}"
                )
                continue
            probes.append((f"user_{i}", ctx))

    for m in bandit.models:
        try:
            theta = bandit.A_inv[m] @ bandit.b[m]

            bias_pred = float(theta @ bias_probe)
            if abs(bias_pred) > 1.5:
                theta_new = theta.copy()
                theta_new[-1] = target_max_pred * (1.0 if bias_pred > 0 else -1.0)
                logger.warning(
                    f"[FIX] Calibration pass 1 ({m}): bias prediction "
                    f"{bias_pred:.2f} -> {theta_new[-1]:.2f} (theta-reconstruction)"
                )
                bandit.b[m] = bandit.A[m] @ theta_new
                bandit.theta[m] = theta_new.copy()
                theta = theta_new

            max_abs_pred = 0.0
            worst_probe = "none"
            for name, x in probes:
                pred = abs(float(theta @ x))
                if pred > max_abs_pred:
                    max_abs_pred = pred
                    worst_probe = name

            if max_abs_pred > 1.5:
                scale = target_max_pred / max_abs_pred
                theta_new = theta * scale
                logger.warning(
                    f"[FIX] Calibration pass 2 ({m}): worst-case prediction "
                    f"{max_abs_pred:.2f} on probe '{worst_probe}' "
                    f"-> global theta scale {scale:.4f}"
                )
                bandit.b[m] = bandit.A[m] @ theta_new
                bandit.theta[m] = theta_new.copy()

        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Failed to calibrate prior for {m}: {e}")
            continue
