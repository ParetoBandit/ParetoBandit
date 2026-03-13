"""Lightweight covariate shift detection via prompt-embedding monitoring.

The :class:`DriftDetector` tracks the distribution of incoming context vectors
(PCA-reduced prompt embeddings) and detects when the live traffic distribution
has shifted away from the distribution observed during burn-in.

Detection metric
----------------
For each prompt, the detector computes a **mean squared z-score** across PCA
components::

    chi2_score(x) = mean_j [ (x_j - mu_j)^2 / sigma_j^2 ]

where ``mu`` and ``sigma`` are the per-component mean and standard deviation
from the burn-in period.  Under the null hypothesis (no shift), ``chi2_score``
has expected value ~1.0 (since each ``z_j^2 ~ chi2(1)``).

Since PCA produces orthogonal components, the diagonal covariance assumption
is principled — no full covariance matrix is needed, making the estimator
stable even with small burn-in samples (50 prompts for 26 dimensions).

Threshold semantics (standard deviations)
------------------------------------------
The threshold is expressed in **baseline standard deviations** of the
chi-squared score observed during burn-in.

- ``threshold = 1.5``: trigger when EMA exceeds baseline by 1.5 sigma
  — sensitive (FPR ~3%, TPR 100% on Pareto→K4).
- ``threshold = 2.0``: trigger at 2.0 sigma — conservative
  (FPR ~1%, TPR 99% on Pareto→K4).
- ``threshold = 0``: immediate trigger on any increase (for testing).

Robustness guarantees
---------------------
1. **O(d) per observation** — one z-score computation per prompt (d = PCA dim).
2. **Stateless after burn-in** — only per-component mean/std (2·d floats),
   plus a handful of scalars for EMA and confirmation state.
3. **EMA smoothing** absorbs isolated outlier prompts.
4. **Confirmation window** requires sustained drift before triggering,
   preventing false positives from transient spikes.
5. **Variance-aware baseline** uses ``mean + 2·std`` of burn-in chi-squared
   scores, capturing normal fluctuation range.
6. **Proactive**: fires at route time (when the embedding arrives), not at
   feedback time (when the reward is observed).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


class DriftDetector:
    """Detect covariate shift via per-component z-scores on context embeddings.

    During the **burn-in phase** (first ``burn_in_steps`` observations), the
    detector accumulates context vectors and computes per-component reference
    statistics (mean and std).  After burn-in, it maintains an EMA of the
    mean-squared-z-score (chi-squared statistic) and compares it to the
    baseline.  The drift condition must hold for ``confirmation_window``
    consecutive steps before :attr:`is_drifting` returns ``True``.

    Parameters
    ----------
    threshold : float
        Number of baseline standard deviations the EMA must exceed the
        baseline chi-squared score for drift to be signalled.
        ``threshold = 2.0`` requires a 2-sigma sustained increase.
    burn_in_steps : int
        Number of initial observations used to establish reference statistics.
        Must be >= 4 (need at least 2 in the second half for std estimation).
    ema_alpha : float
        Smoothing factor for the exponential moving average of chi-squared
        scores.  Smaller values give more smoothing.
        ``0.05`` has a half-life of ~14 observations.
    confirmation_window : int
        Number of consecutive above-threshold EMA readings required before
        drift is confirmed.

    Attributes
    ----------
    baseline : float
        Variance-aware baseline: ``mean + 2·std`` of chi-squared scores from
        the second half of burn-in.  Zero before burn-in completes.
    baseline_std : float
        Standard deviation of chi-squared scores from the second half of
        burn-in, used to scale the threshold.  Zero before burn-in completes.
    total_steps : int
        Total observations processed (burn-in + monitoring).
    """

    def __init__(
        self,
        threshold: float = 2.0,
        burn_in_steps: int = 50,
        ema_alpha: float = 0.05,
        confirmation_window: int = 20,
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if burn_in_steps < 4:
            raise ValueError(f"burn_in_steps must be >= 4, got {burn_in_steps}")
        if not (0 < ema_alpha <= 1):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if confirmation_window < 1:
            raise ValueError(
                f"confirmation_window must be >= 1, got {confirmation_window}"
            )

        self.threshold: float = float(threshold)
        self.burn_in_steps: int = int(burn_in_steps)
        self.ema_alpha: float = float(ema_alpha)
        self.confirmation_window: int = int(confirmation_window)

        self._burn_in_vectors: List[np.ndarray] = []
        self._burned_in: bool = False

        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std: Optional[np.ndarray] = None

        self.baseline: float = 0.0
        self.baseline_std: float = 0.0
        self._ema_chi2: float = 0.0
        self.total_steps: int = 0
        self._consecutive_above: int = 0
        self._confirmed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, context_vector: np.ndarray) -> bool:
        """Record one context vector and return drift status.

        Parameters
        ----------
        context_vector : np.ndarray
            The PCA-reduced prompt embedding (typically d=26 for 25 PCA
            components + 1 bias term).

        Returns
        -------
        bool
            ``True`` if drift is currently confirmed, ``False`` otherwise.
        """
        self.total_steps += 1

        if not self._burned_in:
            self._burn_in_vectors.append(np.asarray(context_vector, dtype=np.float64))
            if len(self._burn_in_vectors) >= self.burn_in_steps:
                self._finalize_burn_in()
            return False

        chi2 = self._chi2_score(context_vector)
        self._ema_chi2 = (
            self.ema_alpha * chi2 + (1.0 - self.ema_alpha) * self._ema_chi2
        )

        if self._ema_above_threshold():
            self._consecutive_above += 1
            if self._consecutive_above >= self.confirmation_window:
                self._confirmed = True
        else:
            self._consecutive_above = 0

        return self.is_drifting

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chi2_score(self, x: np.ndarray) -> float:
        """Mean squared z-score across components (diagonal chi-squared).

        Under the null (in-distribution), each component's z-score is
        approximately standard normal, so the mean of z_j^2 has expected
        value ~1.0.
        """
        z = (np.asarray(x, dtype=np.float64) - self._ref_mean) / self._ref_std
        return float(np.mean(z ** 2))

    def _finalize_burn_in(self) -> None:
        """Compute reference statistics and baseline from burn-in vectors."""
        vectors = np.array(self._burn_in_vectors, dtype=np.float64)
        half = self.burn_in_steps // 2
        second_half = vectors[half:]

        self._ref_mean = second_half.mean(axis=0)
        raw_std = second_half.std(axis=0)
        self._ref_std = np.where(raw_std > 1e-10, raw_std, 1e-10)

        chi2_scores = [self._chi2_score(v) for v in second_half]
        n = len(chi2_scores)
        mean_chi2 = sum(chi2_scores) / n
        var_chi2 = sum((s - mean_chi2) ** 2 for s in chi2_scores) / n
        std_chi2 = math.sqrt(var_chi2)

        self.baseline = mean_chi2 + 2.0 * std_chi2
        self.baseline_std = std_chi2
        self._ema_chi2 = mean_chi2
        self._burned_in = True
        self._burn_in_vectors = []

    def _ema_above_threshold(self) -> bool:
        """Whether the current EMA exceeds baseline + threshold * baseline_std."""
        trigger_level = self.baseline + self.threshold * self.baseline_std
        return self._ema_chi2 > trigger_level

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_drifting(self) -> bool:
        """Whether drift has been confirmed (sustained EMA above threshold)."""
        if not self._burned_in:
            return False
        return self._confirmed

    @property
    def drift_score(self) -> float:
        """Current chi-squared excess above baseline.

        Returns 0.0 before burn-in or when the EMA is at or below baseline.
        """
        if not self._burned_in:
            return 0.0
        return max(0.0, self._ema_chi2 - self.baseline)

    @property
    def drift_ratio(self) -> float:
        """Fraction of the trigger level consumed.

        ``drift_score / (threshold * baseline_std)``.

        - ``< 1.0``: below threshold.
        - ``>= 1.0``: drift confirmed (assuming confirmation window met).
        - Returns ``0.0`` before burn-in.
        - Returns ``inf`` if ``threshold * baseline_std == 0``.
        """
        if not self._burned_in:
            return 0.0
        denominator = self.threshold * self.baseline_std
        if denominator == 0:
            return math.inf if self.drift_score > 0 else 0.0
        return self.drift_score / denominator

    @property
    def ema_chi2(self) -> float:
        """Current EMA of the mean-squared-z-score (chi-squared statistic)."""
        return self._ema_chi2

    @property
    def is_burned_in(self) -> bool:
        """Whether the burn-in phase has completed."""
        return self._burned_in

    @property
    def consecutive_above(self) -> int:
        """Current streak of consecutive above-threshold EMA readings."""
        return self._consecutive_above

    # ------------------------------------------------------------------
    # Reset / lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and re-enter burn-in.

        After a drift-triggered adaptation (e.g., tabula rasa reset),
        the detector must re-establish a baseline on the new traffic.
        Calling ``reset()`` discards the old reference statistics and
        begins accumulating fresh burn-in vectors so that subsequent
        drift detection reflects the *post-adaptation* distribution.

        The configuration parameters (threshold, burn_in_steps,
        ema_alpha, confirmation_window) are preserved.
        """
        self._burn_in_vectors = []
        self._burned_in = False
        self._ref_mean = None
        self._ref_std = None
        self.baseline = 0.0
        self.baseline_std = 0.0
        self._ema_chi2 = 0.0
        self.total_steps = 0
        self._consecutive_above = 0
        self._confirmed = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return a serialisable snapshot for logging / checkpointing."""
        return {
            "total_steps": self.total_steps,
            "burned_in": self._burned_in,
            "baseline": self.baseline,
            "baseline_std": self.baseline_std,
            "ema_chi2": self._ema_chi2,
            "drift_score": self.drift_score,
            "drift_ratio": self.drift_ratio,
            "is_drifting": self.is_drifting,
            "consecutive_above": self._consecutive_above,
            "confirmed": self._confirmed,
            "threshold": self.threshold,
            "burn_in_steps": self.burn_in_steps,
            "ema_alpha": self.ema_alpha,
            "confirmation_window": self.confirmation_window,
        }
