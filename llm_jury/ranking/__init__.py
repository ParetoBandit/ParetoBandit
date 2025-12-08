"""Ranking module for model quality scoring and optimization."""

from llm_jury.ranking.quality_scorer import QualityScorer
from llm_jury.ranking.optimizer import (
    Optimizer,
    OptimizationStrategy,
    MissingDataStrategy,
    Objective,
    ObjectiveRegistry,
    NormalizationMethod,
    # Metric orientation utilities for Pareto analysis
    orient_to_maximize,
    orient_cost,
    orient_latency,
)
from llm_jury.optimization.pareto_chebyshev import (
    ParetoChebyshevOptimizer,
    OptimizationConfig,
    BusinessTargets,
    StrictnessWeights,
    PRESET_CONFIGS,
)
from llm_jury.ranking.constraints import (
    ConstraintConfig,
    CapabilityRequirement,
    UseCaseConstraints,
    apply_constraints,
    get_constrained_recommendations,
    create_context_objective,
    create_capability_objective,
    check_model_capability,
    get_model_context_k,
)
from llm_jury.ranking.use_case_weights import (
    UseCaseWeights,
    USE_CASE_WEIGHTS,
    get_weights_for_use_case,
    get_recommended_models,
    get_primary_metric,
    list_all_use_cases,
)

__all__ = [
    # Optimizer
    "QualityScorer",
    "Optimizer",
    "OptimizationStrategy",
    "MissingDataStrategy",
    "Objective",
    "ObjectiveRegistry",
    "NormalizationMethod",
    # Metric orientation utilities
    "orient_to_maximize",
    "orient_cost",
    "orient_latency",
    # Constraints
    "ConstraintConfig",
    "CapabilityRequirement",
    "UseCaseConstraints",
    "apply_constraints",
    "get_constrained_recommendations",
    "create_context_objective",
    "create_capability_objective",
    "check_model_capability",
    "get_model_context_k",
    # Use Case Weights
    "UseCaseWeights",
    "USE_CASE_WEIGHTS",
    "get_weights_for_use_case",
    "get_recommended_models",
    "get_primary_metric",
    "list_all_use_cases",
    # Pareto-Chebyshev (3-Phase Algorithm)
    "ParetoChebyshevOptimizer",
    "OptimizationConfig",
    "BusinessTargets",
    "StrictnessWeights",
    "PRESET_CONFIGS",
]
