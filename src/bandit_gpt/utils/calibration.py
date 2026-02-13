"""
Sigmoid Normalization Logic

Calibration utilities for complexity normalization in the BanditRouter.

**Conference Review Response: "Feature: router.calibrate(dataset)"**

This addresses the critique: "Hardcoded clipping loses data (Normalization Cliff)."
Instead of using hardcoded μ and σ from LMSYS (generic traffic), users can tune
the sigmoid normalization to their specific data distribution.
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING
import logging

import numpy as np

if TYPE_CHECKING:
    from banditgpt.router import BanditRouter

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Standard logistic function mapping (-inf, inf) to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_complexity(
    router: BanditRouter,
    prompts: List[str],
    *,
    apply: bool = True,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Auto-calibrate complexity normalization parameters from user's dataset.
    
    **Use Cases:**
    - Medical/legal applications (harder than internet average)
    - Creative writing apps (easier than internet average)
    - Domain-specific chatbots with unusual complexity distributions
    
    **Algorithm:**
    1. Encode all prompts using the router's embedding model
    2. Project onto the complexity vector to get raw scores
    3. Compute empirical mean (μ) and std (σ)
    4. Optionally update the router's normalization parameters
    
    **Example:**
    ```python
    # Calibrate on your production traffic
    router = BanditRouter.create(...)
    stats = calibrate_complexity(router, my_prompts, apply=True)
    print(f"Your traffic: μ={stats['mean']:.4f}, σ={stats['std']:.4f}")
    ```
    
    Args:
        router: BanditRouter instance to calibrate
        prompts: List of representative prompts from your production traffic.
                Recommended: 500-1000 samples for stable estimates.
        apply: If True, update the router's COMPLEXITY_MU and COMPLEXITY_SIGMA.
               If False, just return statistics without modifying the router.
        verbose: If True, print detailed statistics and recommendations.
    
    Returns:
        Dict with calibration statistics:
        - 'mean': Empirical mean of complexity projections
        - 'std': Empirical std dev of complexity projections
        - 'min': Minimum projection
        - 'max': Maximum projection
        - 'p1': 1st percentile
        - 'p99': 99th percentile
        - 'n_samples': Number of prompts analyzed
    
    Raises:
        ValueError: If prompts list is empty or too small (<10 samples)
    """
    if not prompts or len(prompts) < 10:
        raise ValueError(f"Need at least 10 prompts for calibration, got {len(prompts)}")
    
    if verbose:
        logger.info(f"Calibrating complexity normalization on {len(prompts)} prompts...")
    
    # Project all prompts onto complexity vector
    projections = []
    for i, prompt in enumerate(prompts):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(prompts)}")
        
        # Encode and project
        emb = router.encoder.encode(prompt, normalize_embeddings=True)
        projection = float(np.dot(emb, router.complexity_vector))
        projections.append(projection)
    
    projections = np.array(projections)
    
    # Compute statistics
    mean = float(np.mean(projections))
    std = float(np.std(projections))
    min_val = float(np.min(projections))
    max_val = float(np.max(projections))
    p1 = float(np.percentile(projections, 1))
    p99 = float(np.percentile(projections, 99))
    
    stats = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'p1': p1,
        'p99': p99,
        'n_samples': len(prompts)
    }
    
    if verbose:
        logger.info(f"\nCalibration Results:")
        logger.info(f"  Mean (μ):     {mean:7.4f}")
        logger.info(f"  Std Dev (σ):  {std:7.4f}")
        logger.info(f"  Range:        [{min_val:7.4f}, {max_val:7.4f}]")
        logger.info(f"  P1-P99:       [{p1:7.4f}, {p99:7.4f}]")
        logger.info(f"  Samples:      {len(prompts)}")
        
        # Compare with default LMSYS calibration
        default_mu = -0.0037
        default_sigma = 0.0950
        logger.info(f"\nComparison with LMSYS defaults:")
        logger.info(f"  Δμ:  {mean - default_mu:+.4f} ({'harder' if mean > default_mu else 'easier'} traffic)")
        logger.info(f"  Δσ:  {std - default_sigma:+.4f} ({'more' if std > default_sigma else 'less'} varied)")
    
    # Apply calibration if requested
    if apply:
        # Store calibrated values for use in _get_context_vector()
        # We need to update the constants used in the normalization
        # Since COMPLEXITY_MU and COMPLEXITY_SIGMA are local variables in _get_context_vector(),
        # we'll add them as instance attributes
        router.calibrated_complexity_mu = mean
        router.calibrated_complexity_sigma = std
        
        if verbose:
            logger.info(f"\n✓ Applied calibration: μ={mean:.4f}, σ={std:.4f}")
            logger.info("  Future calls to route() will use these parameters.")
    
    return stats
