"""Warmup utilities for BanditRouter initialisation.

Provides heuristic prior construction (``get_heuristic_prior``) so that
newly registered models start with a reasonable (A, b) pair rather than an
uninformative identity covariance.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


def safe_inv(A: np.ndarray) -> np.ndarray:
    """Safe matrix inversion with pseudo-inverse fallback for stability."""
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def get_heuristic_prior(
    model_data: Dict[str, Any],
    dim: int,
    init_lambda: float = 1.0,
    n_effective: float = 5.0,
    default_quality: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heuristic prior (A, b) for a new model not in the warmup joblib.
    
    Strategy: 
    Constructs a synthetic prior that mimics having seen 'n_effective' samples
    with an average reward equal to the model's quality score.
    
    **Numerical Stability Note:**
    By initializing A = init_lambda * I, we ensure the matrix is invertible
    at t=0, matching the standard LinUCB regularization.
    
    Args:
        model_data: Dictionary containing model metadata (quality_score, etc.)
        dim: Feature vector dimension (including bias)
        init_lambda: Regularization strength (default: 1.0)
        n_effective: Effective number of samples to represent in the prior (default: 5.0)
        default_quality: Fallback quality score if none found in metadata (default: 0.5)
        
    Returns:
        Tuple of (A_prior, b_prior)
    """
    # 1. Initialize A (Covariance) with regularization
    A = init_lambda * np.eye(dim)
    
    # 2. Initialize b (Reward Vector)
    b = np.zeros(dim)
    
    # 3. Apply the "Prior Belief"
    # Use only initial_quality (composite metric) for consistency with the router
    # initialization path.
    quality = model_data.get("initial_quality")
    
    if quality is None:
        logger.warning(
            f"Model missing 'initial_quality' field, using default={default_quality}. "
            f"This may cause inconsistent initialization."
        )
        quality = default_quality
    
    # CRITICAL: b[-1] assumes the BIAS term is the LAST feature in the vector.
    # Verification Reference: src.pareto_bandit.feature_service.FeatureService.extract_features
    # Logic: np.append(emb_reduced, 1.0) -> bias is absolutely the last element.
    prior_reward_sum = float(quality) * float(n_effective)
    b[-1] = prior_reward_sum
    
    return A, b
