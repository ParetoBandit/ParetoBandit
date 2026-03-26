"""Bootstrap confidence interval utilities for experiment analysis.

Provides vectorised percentile-bootstrap CIs over per-seed scalar data
and time-series data.  Used both in figure scripts (CI bands) and in
``generate_latex.py`` scripts (CI bounds for reported statistics).

The standard workflow is:

    1. The run script stores a ``per_seed_<metric>`` list at each
       checkpoint (length = *n_seeds*).
    2. The figure script reshapes the lists into a (n_checkpoints,
       n_seeds) matrix and calls :func:`bootstrap_ci_series`.
    3. The returned ``(lo, hi)`` arrays are passed directly to
       ``ax.fill_between``.

Typical overhead for 10 000 resamples × 40 seeds × 40 checkpoints is
< 0.3 s on a single core.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile-bootstrap confidence interval for an arbitrary statistic.

    Parameters
    ----------
    values : array-like, shape (n_seeds,)
        One scalar observation per seed.
    statistic :
        Callable ``f(array) -> scalar``.  Defaults to ``np.mean``.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g. 0.95 for 95 % CI).
    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    lo, hi : float
        Lower and upper CI bounds.
    """
    if statistic is None:
        statistic = np.mean
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([statistic(values[idx]) for idx in indices])
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_ci_mean(
    values: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile-bootstrap CI for the mean (fast vectorised path)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[indices].mean(axis=1)
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_ci_median(
    values: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile-bootstrap CI for the median (fast vectorised path)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_medians = np.median(values[indices], axis=1)
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_medians, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_medians, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_ci_series(
    per_seed_matrix: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile-bootstrap CIs along a checkpoint time-series.

    Parameters
    ----------
    per_seed_matrix : ndarray, shape (n_checkpoints, n_seeds)
        Row *t* contains the metric value from each seed at checkpoint *t*.
    n_bootstrap : int
        Number of bootstrap resamples (shared across checkpoints).
    ci_level : float
        Confidence level (e.g. 0.95 for 95 % CI).
    seed : int
        PRNG seed for reproducibility.

    Returns
    -------
    lo, hi : ndarray, each shape (n_checkpoints,)
        Lower and upper CI bounds per checkpoint.
    """
    rng = np.random.default_rng(seed)
    per_seed_matrix = np.asarray(per_seed_matrix, dtype=np.float64)
    n_checkpoints, n_seeds = per_seed_matrix.shape
    alpha = 1.0 - ci_level

    # Pre-draw one shared index matrix — same resamples at every checkpoint
    # so that the CI band is smooth across time steps.
    indices = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))

    lo = np.empty(n_checkpoints)
    hi = np.empty(n_checkpoints)
    for t in range(n_checkpoints):
        boot_means = per_seed_matrix[t, indices].mean(axis=1)
        lo[t] = np.percentile(boot_means, 100.0 * alpha / 2.0)
        hi[t] = np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))

    return lo, hi
