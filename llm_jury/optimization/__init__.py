"""
Optimization module for multi-objective model scoring.

This module contains algorithms for scoring and ranking models based on
multiple objectives: quality, cost, and latency.

Key concepts:

1. **Total Cost of Inference (TCI)**: Handles the $0 open-source pricing problem
   by assigning shadow compute costs to self-hosted models. This ensures fair
   comparison between API-managed and self-hosted options.

2. **Metric Orientation**: All metrics are normalized to share direction (higher = better)
   for consistent Pareto analysis:
   - Quality: Maximize → use as-is
   - Cost: Minimize → invert using 1/(1+ratio)
   - Latency: Minimize → invert using 1/(1+ratio)
"""

# from llm_jury.optimization.chebyshev_scorer import ChebyshevScorer
# from llm_jury.optimization.total_cost_inference import (
#     # TCI approach (data engineering fix)
#     calculate_tci,
#     calculate_shadow_price,
#     get_effective_cost,
#     is_self_hosted,
#     MIN_COST_FLOOR,
#     COMPUTE_PROFILES,
#     set_tci_minimum_cost,
#     get_tci_minimum_cost,
#     # Utopian distance approach (mathematical fix)
#     minmax_normalize,
#     calculate_utopian_distance,
#     find_utopian_knee,
#     # Kneedle algorithm approach (2D projection)
#     LOG_EPSILON,
#     calculate_performance_score,
#     log_transform_cost,
#     find_pareto_frontier_2d,
#     find_kneedle_point,
#     find_kneedle_knee,
#     get_kneedle_visualization_data,
# )
# from llm_jury.optimization.augmented_chebyshev import (
#     AugmentedChebyshevScorer,
#     AugmentedScore,
#     MetricConfig,
#     MetricDirection,
#     WEIGHT_PRESETS as AUGMENTED_WEIGHT_PRESETS,
#     utopian_normalize,
#     calculate_efficiency_index,
#     calculate_trust_score,
# )
from llm_jury.optimization.pareto_chebyshev import (
    ParetoChebyshevOptimizer,
    OptimizationConfig,
    BusinessTargets,
    StrictnessWeights,
    ChebyshevResult,
    ModelMetrics,
    OptimizationResult,
    PRESET_CONFIGS,
    calculate_effective_cost,
    pareto_filter,
    calculate_regret,
    set_minimum_cost_from_population,
    get_minimum_cost,
)

__all__ = [
    # 'ChebyshevScorer',
    # # TCI exports (data engineering fix)
    # 'calculate_tci',
    # 'calculate_shadow_price', 
    # 'get_effective_cost',
    # 'is_self_hosted',
    # 'MIN_COST_FLOOR',
    # 'COMPUTE_PROFILES',
    # 'set_tci_minimum_cost',
    # 'get_tci_minimum_cost',
    # # Utopian distance exports (mathematical fix)
    # 'minmax_normalize',
    # 'calculate_utopian_distance',
    # 'find_utopian_knee',
    # # Kneedle algorithm exports (2D projection)
    # 'LOG_EPSILON',
    # 'calculate_performance_score',
    # 'log_transform_cost',
    # 'find_pareto_frontier_2d',
    # 'find_kneedle_point',
    # 'find_kneedle_knee',
    # 'get_kneedle_visualization_data',
    # # Augmented Chebyshev exports (5D with Utopian normalization)
    # 'AugmentedChebyshevScorer',
    # 'AugmentedScore',
    # 'MetricConfig',
    # 'MetricDirection',
    # 'AUGMENTED_WEIGHT_PRESETS',
    # 'utopian_normalize',
    # 'calculate_efficiency_index',
    # 'calculate_trust_score',
    # Pareto-Chebyshev exports (3-Phase optimization)
    'ParetoChebyshevOptimizer',
    'OptimizationConfig',
    'BusinessTargets',
    'StrictnessWeights',
    'ChebyshevResult',
    'ModelMetrics',
    'OptimizationResult',
    'PRESET_CONFIGS',
    'calculate_effective_cost',
    'pareto_filter',
    'calculate_regret',
    'set_minimum_cost_from_population',
    'get_minimum_cost',
]
