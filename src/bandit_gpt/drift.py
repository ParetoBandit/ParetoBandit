"""Lightweight covariate shift detection via prompt-embedding monitoring.

Two detector implementations are provided:

:class:`DriftDetector`
    Diagonal chi-squared test on per-component z-scores.  Effective when
    the shift manifests as per-feature mean/variance changes (e.g.,
    different embedding scale or missing components).

:class:`CentroidDriftDetector`
    Cosine-distance test between a running EMA centroid and the burn-in
    reference centroid.  Effective when the shift is a **topic rotation**
    (e.g., general NLP → math reasoning) that moves the centroid in
    high-dimensional space without necessarily changing per-component
    variance.  This is the recommended default for LLM routing.

Both classes share the same public API (``update``, ``is_drifting``,
``reset``, ``get_state``, etc.) so they are drop-in replaceable in the
router.

Detection metric — DriftDetector (chi-squared)
-----------------------------------------------
For each prompt, the detector computes a **mean squared z-score** across PCA
components::

    chi2_score(x) = mean_j [ (x_j - mu_j)^2 / sigma_j^2 ]

where ``mu`` and ``sigma`` are the per-component mean and standard deviation
from the burn-in period.  Under the null hypothesis (no shift), ``chi2_score``
has expected value ~1.0 (since each ``z_j^2 ~ chi2(1)``).

Detection metric — CentroidDriftDetector (cosine distance)
-----------------------------------------------------------
For each prompt, the detector computes the cosine distance between the
incoming vector and the burn-in centroid::

    d(x) = 1 - cos(x, centroid_ref)

An EMA of ``d(x)`` is compared to the burn-in baseline distance (mean + 2*std
of burn-in distances).  Cosine distance is rotation-sensitive and
L2-normalization-invariant, making it ideal for SentenceTransformer
embeddings that lie on the unit sphere.

Threshold semantics (standard deviations)
------------------------------------------
The threshold is expressed in **baseline standard deviations** of the
detection score observed during burn-in.

- ``threshold = 1.5``: sensitive (early detection).
- ``threshold = 2.0``: conservative (fewer false positives).
- ``threshold = 0``: immediate trigger on any increase (for testing).

Robustness guarantees
---------------------
1. **O(d) per observation** — one distance computation per prompt.
2. **Stateless after burn-in** — reference centroid (d floats),
   plus a handful of scalars for EMA and confirmation state.
3. **EMA smoothing** absorbs isolated outlier prompts.
4. **Confirmation window** requires sustained drift before triggering,
   preventing false positives from transient spikes.
5. **Variance-aware baseline** uses ``mean + 2·std`` of burn-in scores,
   capturing normal fluctuation range.
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


class CentroidDriftDetector:
    """Detect covariate shift via running-centroid cosine distance.

    Maintains an **EMA of the embedding vectors** themselves (not
    per-sample distances), producing a smoothed running centroid that
    tracks the current traffic distribution.  Drift is detected when
    the cosine distance between this running centroid and the burn-in
    reference centroid exceeds a threshold.

    This is far more powerful than per-sample distance tests because
    averaging embeddings cancels within-class noise, amplifying the
    between-class signal.  For L2-normalized SentenceTransformer
    embeddings, topic/domain shifts (e.g., general NLP → math reasoning)
    manifest as centroid rotations that this detector reliably captures.

    Detection score at each step::

        centroid_ema_t = (1 - alpha) * centroid_ema_{t-1} + alpha * x_t
        score_t = 1 - cos(centroid_ema_t, centroid_ref)

    The score is compared to a variance-aware baseline derived from
    the burn-in period.

    Parameters
    ----------
    threshold : float
        Number of baseline standard deviations the score must exceed
        the baseline for drift to be signalled.
    burn_in_steps : int
        Number of initial observations for establishing the reference
        centroid and baseline score distribution.
    ema_alpha : float
        Smoothing factor for the running embedding centroid.
        ``0.05`` gives a half-life of ~14 observations, producing a
        centroid that averages over ~28 recent prompts.
    confirmation_window : int
        Consecutive above-threshold readings required to confirm drift.

    Attributes
    ----------
    baseline : float
        ``mean + 2·std`` of cosine-distance scores from burn-in.
    baseline_std : float
        Standard deviation of burn-in scores.
    total_steps : int
        Total observations processed.
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

        self._ref_centroid: Optional[np.ndarray] = None
        self._ema_centroid: Optional[np.ndarray] = None

        self.baseline: float = 0.0
        self.baseline_std: float = 0.0
        self._current_score: float = 0.0
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
            Prompt embedding (any dimensionality).

        Returns
        -------
        bool
            ``True`` if drift is currently confirmed.
        """
        self.total_steps += 1
        x = np.asarray(context_vector, dtype=np.float64)

        if not self._burned_in:
            self._burn_in_vectors.append(x)
            if len(self._burn_in_vectors) >= self.burn_in_steps:
                self._finalize_burn_in()
            return False

        self._ema_centroid = (
            (1.0 - self.ema_alpha) * self._ema_centroid
            + self.ema_alpha * x
        )
        self._current_score = self._centroid_distance()

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

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm

    def _centroid_distance(self) -> float:
        """Cosine distance between running centroid and reference."""
        ema_norm = self._l2_normalize(self._ema_centroid)
        sim = float(np.dot(ema_norm, self._ref_centroid))
        return 1.0 - max(-1.0, min(1.0, sim))

    def _finalize_burn_in(self) -> None:
        """Compute reference centroid and baseline from burn-in vectors."""
        vectors = np.array(self._burn_in_vectors, dtype=np.float64)
        half = self.burn_in_steps // 2
        second_half = vectors[half:]

        raw_centroid = second_half.mean(axis=0)
        self._ref_centroid = self._l2_normalize(raw_centroid)
        self._ema_centroid = raw_centroid.copy()

        scores: List[float] = []
        running = vectors[0].copy()
        for i in range(1, len(vectors)):
            running = (1.0 - self.ema_alpha) * running + self.ema_alpha * vectors[i]
            if i >= half:
                running_norm = self._l2_normalize(running)
                sim = float(np.dot(running_norm, self._ref_centroid))
                scores.append(1.0 - max(-1.0, min(1.0, sim)))

        n = len(scores)
        mean_s = sum(scores) / n
        var_s = sum((s - mean_s) ** 2 for s in scores) / n
        std_s = math.sqrt(var_s)

        self.baseline = mean_s + 2.0 * std_s
        self.baseline_std = std_s
        self._current_score = scores[-1] if scores else mean_s
        self._burned_in = True
        self._burn_in_vectors = []

    def _ema_above_threshold(self) -> bool:
        trigger_level = self.baseline + self.threshold * self.baseline_std
        return self._current_score > trigger_level

    # ------------------------------------------------------------------
    # Properties (same API as DriftDetector)
    # ------------------------------------------------------------------

    @property
    def is_drifting(self) -> bool:
        """Whether drift has been confirmed."""
        if not self._burned_in:
            return False
        return self._confirmed

    @property
    def drift_score(self) -> float:
        """Current centroid cosine-distance excess above baseline."""
        if not self._burned_in:
            return 0.0
        return max(0.0, self._current_score - self.baseline)

    @property
    def drift_ratio(self) -> float:
        """Fraction of the trigger level consumed (>=1.0 means triggered)."""
        if not self._burned_in:
            return 0.0
        denominator = self.threshold * self.baseline_std
        if denominator == 0:
            return math.inf if self.drift_score > 0 else 0.0
        return self.drift_score / denominator

    @property
    def ema_chi2(self) -> float:
        """Current score (centroid distance, named for API compat)."""
        return self._current_score

    @property
    def is_burned_in(self) -> bool:
        """Whether the burn-in phase has completed."""
        return self._burned_in

    @property
    def consecutive_above(self) -> int:
        """Current streak of consecutive above-threshold readings."""
        return self._consecutive_above

    # ------------------------------------------------------------------
    # Reset / lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and re-enter burn-in."""
        self._burn_in_vectors = []
        self._burned_in = False
        self._ref_centroid = None
        self._ema_centroid = None
        self.baseline = 0.0
        self.baseline_std = 0.0
        self._current_score = 0.0
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
            "ema_chi2": self._current_score,
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
