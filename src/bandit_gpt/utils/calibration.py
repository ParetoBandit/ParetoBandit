"""
Sigmoid Normalization Logic

Calibration utilities for the BanditRouter.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: float) -> float:
    """Standard logistic function mapping (-inf, inf) to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))
