"""Primal-Dual budget pacing for cost-constrained LLM routing.

The :class:`BudgetPacer` implements an online Lagrangian relaxation
(Primal-Dual CBwK; Agrawal & Devanur 2014) that enforces a per-request
average cost target without requiring a known time horizon.

Architecture
------------
Three user-selectable modes share the same dual-variable tracking but
differ in enforcement:

- **HARD** -- Adaptive ``max_cost`` ceiling fed through the router's
  existing constraint filter.  This is a practical safety mechanism
  (not part of the CBwK formulation) that excludes models whose blended
  cost/1k exceeds a ceiling derived from the dual variable.
- **SOFT** -- Dynamic per-request cost penalties injected into the
  UCB scoring function.  This is the theoretically grounded path:
  expensive models are penalized proportional to
  ``lambda_t * normalized_cost``, keeping them eligible but
  less attractive.
- **ADAPTIVE** -- Both mechanisms active.  Hard ceiling as safety net,
  soft penalty as optimizer.

Dual-variable dynamics
----------------------
After each request, the pacer observes the actual cost and performs
two updates:

1. **EMA update** (smoothing)::

       cost_ema = (1 - alpha) * cost_ema + alpha * actual_cost

2. **Dual update** using the **smoothed EMA**, not the raw per-request
   cost::

       c_norm = cost_ema / target    # 1.0 at budget, >1 overspend
       lambda_t = min(lambda_max,
                      max(0, lambda_t + lr * (c_norm - 1.0)))

Using the EMA rather than the raw cost is critical when model costs
span orders of magnitude (e.g., $0.00003 for Llama vs. $0.015 for
Gemini).  A single expensive request would otherwise spike lambda by
``lr * (c_t / target - 1)``, which can exceed 1.0, causing sawtooth
oscillations that never converge.  The EMA smooths this variance,
producing a stable gradient signal whose magnitude reflects the
*average* cost regime.

Normalizing by target makes ``lr`` portfolio-independent: the same
``lr`` value works regardless of whether the budget is $0.001/req or
$10/req.  ``lambda_max`` prevents the dual variable from growing
unboundedly, which in HARD mode would shrink the ceiling to zero
and exclude all models.

Design choices
--------------
- ``cost_ema`` warm-starts at ``target`` (not 0) to avoid cold-start
  overshoot.
- ``lambda_t`` starts at 0 (no penalty initially; ramps only if
  overspending).
- Thread-safe via a dedicated lock.
"""

from __future__ import annotations

import enum
import math
import threading
from typing import Dict, Optional


class PacingMode(enum.Enum):
    """How the budget constraint is enforced."""

    HARD = "hard"
    SOFT = "soft"
    ADAPTIVE = "adaptive"


class BudgetPacer:
    """Online budget pacer using Primal-Dual Lagrangian relaxation.

    Parameters
    ----------
    target_avg_spend_usd : float
        Desired average cost per request in USD.  Must be positive.
    mode : PacingMode
        Enforcement mechanism (see module docstring).
    lr : float
        Learning rate for the dual variable (lambda) gradient step.
        Operates on target-normalized costs, so the same ``lr`` value
        works regardless of absolute cost scale.  ``0.05`` means
        "each request at 2x budget increases lambda by 0.05."
    ema_alpha : float
        Smoothing factor for the cost EMA that drives the dual update.
        Must be in (0, 1].  ``0.05`` has a half-life of ~14 observations.
    hard_ceiling_multiplier : float
        Controls how aggressively the hard ceiling tightens as the
        dual variable grows.  The ceiling is computed relative to the
        portfolio's most expensive model.
    lambda_max : float
        Upper bound on the dual variable.  Prevents the hard ceiling
        from shrinking to zero and excluding all models.  Must be
        positive.

    Attributes
    ----------
    lambda_t : float
        Current Lagrange multiplier (dual variable).  Zero means no
        cost pressure; positive means overspending has been detected.
        Dimensionless (operates on target-normalized costs).
    cost_ema : float
        Exponential moving average of observed per-request costs (USD).
        Drives the dual-variable update as a smoothed gradient signal.
    n_observations : int
        Total number of cost observations processed.

    Raises
    ------
    ValueError
        If ``target_avg_spend_usd <= 0`` or ``ema_alpha`` is out of range.
    """

    def __init__(
        self,
        target_avg_spend_usd: float,
        mode: PacingMode = PacingMode.ADAPTIVE,
        lr: float = 0.05,
        ema_alpha: float = 0.05,
        hard_ceiling_multiplier: float = 1.0,
        lambda_max: float = 5.0,
    ) -> None:
        if target_avg_spend_usd <= 0:
            raise ValueError(
                f"target_avg_spend_usd must be positive, "
                f"got {target_avg_spend_usd}"
            )
        if not (0 < ema_alpha <= 1):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if lambda_max <= 0:
            raise ValueError(f"lambda_max must be positive, got {lambda_max}")

        self.target_avg_spend_usd: float = float(target_avg_spend_usd)
        self.mode: PacingMode = mode
        self.lr: float = float(lr)
        self.ema_alpha: float = float(ema_alpha)
        self.hard_ceiling_multiplier: float = float(hard_ceiling_multiplier)
        self.lambda_max: float = float(lambda_max)

        self.lambda_t: float = 0.0
        self.cost_ema: float = float(target_avg_spend_usd)
        self.n_observations: int = 0

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mode properties
    # ------------------------------------------------------------------

    @property
    def uses_hard(self) -> bool:
        """Whether the hard ceiling mechanism is active."""
        return self.mode in (PacingMode.HARD, PacingMode.ADAPTIVE)

    @property
    def uses_soft(self) -> bool:
        """Whether the soft penalty mechanism is active."""
        return self.mode in (PacingMode.SOFT, PacingMode.ADAPTIVE)

    # ------------------------------------------------------------------
    # Hard ceiling
    # ------------------------------------------------------------------

    def get_cost_ceiling_per_1k(
        self, max_model_cost_per_1k: float
    ) -> Optional[float]:
        """Return a blended cost-per-1k-token ceiling, or ``None``.

        The ceiling is derived from the portfolio's most expensive model
        rate, scaled down by the dual variable::

            ceiling = max_model_cost_per_1k / (1 + multiplier * lambda_t)

        This ensures the ceiling is always in the correct units ($/1k
        tokens) and self-calibrates to the portfolio.

        When ``lambda_t == 0`` (no overspend detected), returns ``None``
        (no constraint needed).

        .. note::

            The hard ceiling is a practical safety mechanism, not part
            of the Primal-Dual CBwK formulation (which operates through
            soft penalties in the UCB scoring function).

        Parameters
        ----------
        max_model_cost_per_1k : float
            Blended cost per 1k tokens of the most expensive model in
            the portfolio.  Computed by the caller from registry metadata.

        Returns
        -------
        float or None
            Cost ceiling in USD/1k tokens, or ``None`` if no constraint
            is active.
        """
        with self._lock:
            if self.lambda_t <= 0.0:
                return None
            ceiling = max_model_cost_per_1k / (
                1.0 + self.hard_ceiling_multiplier * self.lambda_t
            )
            return ceiling

    # ------------------------------------------------------------------
    # Soft penalties
    # ------------------------------------------------------------------

    def get_extra_cost_penalties(
        self, model_costs: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute per-model cost penalties scaled by the dual variable.

        Each model receives a penalty of ``lambda_t * normalized_cost``
        where ``normalized_cost`` is the value passed in ``model_costs``.
        The caller is responsible for normalizing costs to [0, 1] so the
        penalty is commensurate with the UCB reward scale.

        Parameters
        ----------
        model_costs : dict[str, float]
            ``{model_id: normalized_cost}`` where ``normalized_cost`` is
            typically the output of
            ``BanditRouter._get_normalized_cost(model_id)``.

        Returns
        -------
        dict[str, float]
            ``{model_id: penalty}`` to subtract from UCB scores.
        """
        with self._lock:
            lam = self.lambda_t
        return {m: lam * c for m, c in model_costs.items()}

    # ------------------------------------------------------------------
    # Observation (dual update)
    # ------------------------------------------------------------------

    def observe(self, actual_cost_usd: float) -> None:
        """Record one realized cost and update pacing state.

        Performs two updates atomically:

        1. **EMA update**: smooths the per-request cost into a running
           average::

               cost_ema = (1 - alpha) * cost_ema + alpha * actual_cost

        2. **Dual update**: projected gradient ascent on the Lagrange
           multiplier using the **smoothed EMA** (not the raw cost)::

               c_norm = cost_ema / target
               lambda_t = min(lambda_max,
                              max(0, lambda_t + lr * (c_norm - 1)))

           Using the EMA eliminates high-variance spikes from single
           expensive requests, producing stable lambda convergence even
           when model costs span 500x.

        Parameters
        ----------
        actual_cost_usd : float
            Realized cost in USD for the most recent request.
        """
        with self._lock:
            self.n_observations += 1

            self.cost_ema = (
                (1.0 - self.ema_alpha) * self.cost_ema
                + self.ema_alpha * actual_cost_usd
            )

            c_normalized = self.cost_ema / self.target_avg_spend_usd
            self.lambda_t = min(
                self.lambda_max,
                max(0.0, self.lambda_t + self.lr * (c_normalized - 1.0)),
            )

    # ------------------------------------------------------------------
    # Reset / lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-initialize pacing state, preserving configuration.

        Called after a drift-triggered bandit reset to clear stale
        dual-variable state that reflected the pre-drift cost regime.
        """
        with self._lock:
            self.lambda_t = 0.0
            self.cost_ema = self.target_avg_spend_usd
            self.n_observations = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a serializable snapshot for logging / checkpointing.

        Returns
        -------
        dict
            Contains ``lambda_t``, ``cost_ema``, ``n_observations``,
            ``target``, ``mode``, and derived quantities.
        """
        with self._lock:
            return {
                "lambda_t": self.lambda_t,
                "cost_ema": self.cost_ema,
                "n_observations": self.n_observations,
                "target_avg_spend_usd": self.target_avg_spend_usd,
                "mode": self.mode.value,
                "lr": self.lr,
                "ema_alpha": self.ema_alpha,
                "lambda_max": self.lambda_max,
                "uses_hard": self.uses_hard,
                "uses_soft": self.uses_soft,
            }

    def __repr__(self) -> str:
        return (
            f"BudgetPacer(target={self.target_avg_spend_usd:.6f}, "
            f"mode={self.mode.value}, lambda={self.lambda_t:.4f}, "
            f"ema={self.cost_ema:.6f}, n={self.n_observations})"
        )
