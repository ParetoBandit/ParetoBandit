"""
Reward utilities for BanditGPT.

This module provides tools for normalizing and transforming reward signals
received from external feedback (user, programmatic, or oracle).
"""

import numpy as np
from typing import Dict, Any

def clip01(x: float, eps: float = 0.01) -> float:
    """Clip a probability-like value into [eps, 1-eps]."""
    x = float(x)
    eps = float(eps)
    return float(min(max(x, eps), 1.0 - eps))

class LogitReward:
    """
    Safe logit transform with strict clipping.

    Stretches probabilities p in (0, 1) to (-inf, +inf):
        logit(p) = log(p / (1 - p))
    """

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = float(epsilon)
        e = self.epsilon
        self.min_val = float(np.log(e / (1.0 - e)))
        self.max_val = float(np.log((1.0 - e) / e))

    def transform(self, p_correct: float) -> float:
        p = clip01(float(p_correct), eps=self.epsilon)
        return float(np.log(p / (1.0 - p)))

class RunningZScoreNormalizer:
    """
    Online reward normalization using exponential moving averages.

    Stretches the signal by converting raw rewards into a clamped z-score:
        z = (r - mean) / (std + eps)
    """

    def __init__(
        self,
        mean_init: float = 0.65,
        std_init: float = 0.05,
        alpha: float = 0.01,
        clamp: float = 3.0,
        eps: float = 1e-9,
        auto_init_from_first_sample: bool = False,
    ):
        self.mean = float(mean_init)
        self.var = float(max(std_init, eps) ** 2)
        self.alpha = float(alpha)
        self.clamp = float(clamp)
        self.eps = float(eps)
        self.auto_init_from_first_sample = bool(auto_init_from_first_sample)
        self.n_seen = 0

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, self.eps)))

    def update(self, x: float) -> None:
        x = float(x)
        new_mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        err = x - new_mean
        new_var = (1.0 - self.alpha) * self.var + self.alpha * (err * err)
        self.mean = float(new_mean)
        self.var = float(max(new_var, self.eps))

    def normalize(self, x: float, *, update: bool = True) -> float:
        x = float(x)
        if self.auto_init_from_first_sample and self.n_seen == 0:
            self.mean = float(x)
            self.n_seen = 1
            return 0.0

        z = (x - self.mean) / (self.std + self.eps)
        if self.clamp > 0:
            z = max(min(z, self.clamp), -self.clamp)
        if update:
            self.update(x)
        self.n_seen += 1
        return float(z)

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "alpha": float(self.alpha),
            "clamp": float(self.clamp),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RunningZScoreNormalizer":
        obj = cls(
            mean_init=float(d.get("mean", 0.65)),
            std_init=float(np.sqrt(max(float(d.get("var", 0.05**2)), 1e-12))),
            alpha=float(d.get("alpha", 0.01)),
            clamp=float(d.get("clamp", 3.0)),
            eps=float(d.get("eps", 1e-9)),
        )
        obj.mean = float(d.get("mean", obj.mean))
        obj.var = float(d.get("var", obj.var))
        return obj
