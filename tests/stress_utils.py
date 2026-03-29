from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pareto_bandit import FeatureService

DEFAULT_DIMENSION = 16  # 15 PCA dimensions + bias


@dataclass(frozen=True)
class SyntheticModelTier:
    model_id: str
    input_cost_per_m: float
    output_cost_per_m: float
    latency_s: float
    hle: float
    speed_profile: str


DEFAULT_TIERS = [
    SyntheticModelTier(
        model_id="cheap-fast/model-a",
        input_cost_per_m=0.08,
        output_cost_per_m=0.18,
        latency_s=0.15,
        hle=0.42,
        speed_profile="fast",
    ),
    SyntheticModelTier(
        model_id="cheap-fast/model-b",
        input_cost_per_m=0.12,
        output_cost_per_m=0.25,
        latency_s=0.20,
        hle=0.51,
        speed_profile="fast",
    ),
    SyntheticModelTier(
        model_id="mid/model-c",
        input_cost_per_m=1.50,
        output_cost_per_m=3.00,
        latency_s=0.45,
        hle=0.67,
        speed_profile="balanced",
    ),
    SyntheticModelTier(
        model_id="premium/model-d",
        input_cost_per_m=7.00,
        output_cost_per_m=14.00,
        latency_s=0.90,
        hle=0.80,
        speed_profile="slow",
    ),
]


def synthetic_registry() -> dict[str, dict]:
    """Registry with realistic cost/latency/quality spread."""
    registry: dict[str, dict] = {}
    for tier in DEFAULT_TIERS:
        registry[tier.model_id] = {
            "model_id": tier.model_id,
            "display_name": tier.model_id.split("/")[-1],
            "scores": {"hle": tier.hle},
            "hle": tier.hle,
            "input_cost_per_m": tier.input_cost_per_m,
            "output_cost_per_m": tier.output_cost_per_m,
            "time_to_first_token_seconds": tier.latency_s,
            "speed_profile": tier.speed_profile,
        }
    return registry


def precomputed_feature_service(dimension: int = DEFAULT_DIMENSION) -> FeatureService:
    """Feature service that accepts precomputed vectors only."""
    return FeatureService.for_precomputed(dimension)


def make_context(seed: int, dimension: int = DEFAULT_DIMENSION) -> np.ndarray:
    """Deterministic unit vector with bias term fixed to 1.0."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dimension - 1)
    base = base / (np.linalg.norm(base) + 1e-12)
    return np.append(base, 1.0).astype(np.float64)
