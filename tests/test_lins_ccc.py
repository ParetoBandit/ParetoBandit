"""Unit tests for Lin's Concordance Correlation Coefficient.

Validates the hand-rolled implementation against known-answer cases from
Lin (1989) and algebraic identities.  CCC = 0.649 and 0.718 are key
numbers in the judge-robustness appendix, so correctness is critical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "experiments" / "appendix" / "judge_robustness"),
)

from judge_robustness_utils import lins_ccc


class TestLinsCCC:
    """Known-answer and algebraic-identity tests for lins_ccc."""

    def test_perfect_agreement(self) -> None:
        """Identical arrays must yield CCC = 1."""
        x = np.array([0.1, 0.4, 0.7, 0.9, 1.0])
        assert lins_ccc(x, x) == pytest.approx(1.0, abs=1e-12)

    def test_perfect_negative_agreement(self) -> None:
        """CCC of x and -x (after mean-centering) should be -1."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert lins_ccc(x, -x) == pytest.approx(-1.0, abs=1e-12)

    def test_constant_shift_penalised(self) -> None:
        """A location shift (y = x + c) should yield CCC < 1 despite r = 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x + 2.0
        ccc = lins_ccc(x, y)
        assert ccc < 1.0, "Location shift should penalise CCC"
        assert ccc > 0.0, "CCC should remain positive for monotone shift"

    def test_scale_shift_penalised(self) -> None:
        """A scale shift (y = 2x) should yield CCC < 1 despite r = 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x
        ccc = lins_ccc(x, y)
        assert ccc < 1.0, "Scale shift should penalise CCC"
        assert ccc > 0.0, "CCC should remain positive for monotone scale"

    def test_symmetry(self) -> None:
        """CCC(x, y) must equal CCC(y, x)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0.8, 0.1, size=200)
        y = x + rng.normal(0.0, 0.05, size=200)
        assert lins_ccc(x, y) == pytest.approx(lins_ccc(y, x), abs=1e-12)

    def test_known_numerical_example(self) -> None:
        """Verify against independently computed CCC value.

        Data: x = [1..8], y = x with small perturbations.
        Independently verified via the formula:
            CCC = 2*cov / (var_x + var_y + (mean_x - mean_y)^2)
        using numpy with ddof=1 for sample variance/covariance.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([0.8, 2.5, 2.8, 4.1, 5.2, 5.9, 7.2, 7.8])
        mx, my = np.mean(x), np.mean(y)
        sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
        sxy = np.cov(x, y, ddof=1)[0, 1]
        expected = 2.0 * sxy / (sx2 + sy2 + (mx - my) ** 2)
        ccc = lins_ccc(x, y)
        assert ccc == pytest.approx(expected, abs=1e-12)

    def test_ccc_bounded(self) -> None:
        """CCC should lie in [-1, 1] for random data."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            x = rng.uniform(0, 1, size=100)
            y = rng.uniform(0, 1, size=100)
            ccc = lins_ccc(x, y)
            assert -1.0 <= ccc <= 1.0, f"CCC out of bounds: {ccc}"

    def test_ccc_leq_pearson(self) -> None:
        """CCC must be <= |Pearson r| for any dataset (McBride 2005)."""
        from scipy.stats import pearsonr

        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.normal(0.85, 0.1, size=200)
            y = x + rng.normal(0.03, 0.08, size=200)
            ccc = lins_ccc(x, y)
            r, _ = pearsonr(x, y)
            assert ccc <= abs(r) + 1e-10, (
                f"CCC ({ccc:.6f}) should not exceed |r| ({abs(r):.6f})"
            )
