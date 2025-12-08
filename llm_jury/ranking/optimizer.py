"""
Multi-objective optimizer for LLM model selection.

Supports two optimization approaches:
1. Chebyshev Scalarization - minimizes distance from utopia point
2. Knee Point Detection - finds best "bang for buck" on Pareto frontier

Uses a pluggable Objective system for extensibility. Default objectives:
- Quality (from AA benchmarks: intelligence, coding, math indices + raw benchmarks)
- Cost (Total Cost of Inference - handles $0 open-source pricing)
- Latency (Time To First Token - TTFT)
- Hallucination Rate (from Vectara leaderboard)
- Refusal Rate (how often model refuses to answer)

Adding new objectives (e.g., ethics, safety) is as simple as defining an
Objective and registering it with the optimizer.

Phase 1: Data Engineering - Total Cost of Inference (TCI)
=========================================================
Open-source models are not free - they require compute resources. TCI assigns
a "shadow price" based on GPU costs:
- Managed Models (GPT-4, Claude): TCI = API Price per 1M tokens
- Self-Hosted (Llama, Mistral): TCI = (Hourly GPU Cost / Tokens per Hour)

This converts $0 to a small positive number (e.g., $0.02-$0.50), enabling
fair comparison and preventing divide-by-zero issues.

Phase 2: Metric Orientation
===========================
For Pareto analysis, all metrics are normalized to share direction (higher = better):
- Quality: Maximize (higher is better) → use as-is
- Latency: Minimize (lower is better) → invert: 1 - normalized_value
- Cost: Minimize (lower is better) → invert: 1 / (1 + cost_ratio)

This ensures consistent Pareto frontier construction where the utopia point
is always at (1, 1, 1, ...) in normalized space.
"""

import numpy as np
from dataclasses import dataclass, field as dataclass_field
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
from enum import Enum

from llm_jury.core.models import (
    ModelMetadata, RoutingDecision, RecommendationResult, 
    RankedModel, PromptCategory, ProductArchetype
)
from llm_jury.ranking.quality_scorer import QualityScorer
# from llm_jury.optimization.total_cost_inference import (
#     calculate_tci, 
#     MIN_COST_FLOOR,
#     minmax_normalize,
#     calculate_utopian_distance,
#     find_kneedle_point,
#     log_transform_cost,
#     calculate_performance_score,
# )
# from llm_jury.optimization.augmented_chebyshev import (
#     AugmentedChebyshevScorer,
#     WEIGHT_PRESETS as AUGMENTED_WEIGHT_PRESETS,
# )
from llm_jury.optimization.pareto_chebyshev import (
    ParetoChebyshevOptimizer,
    OptimizationConfig,
    BusinessTargets,
    StrictnessWeights,
    PRESET_CONFIGS,
    set_minimum_cost_from_population,
)


# =============================================================================
# Objective System - Extensible metric definitions
# =============================================================================

class NormalizationMethod(Enum):
    """How to normalize a metric to 0-1 scale."""
    PERCENTAGE = "percentage"           # Value is 0-100, divide by 100
    INVERSE_PERCENTAGE = "inverse_pct"  # Value is 0-100, lower is better: 1 - (v/100)
    RATIO_TO_BASELINE = "ratio"         # Normalize relative to baseline model
    MIN_MAX = "minmax"                  # Use population min/max
    INVERSE_MIN_MAX = "inverse_minmax"  # Min/max but inverted (lower is better)
    CUSTOM = "custom"                   # Use custom normalization function


class MissingDataStrategy(Enum):
    """How to handle models with missing attribute data."""
    STRICT = "strict"       # Only use models with complete data (default, highest confidence)
    IMPUTE = "impute"       # Use default/custom values for missing data (lower confidence)


@dataclass
class Objective:
    """
    Defines a single optimization objective.
    
    This is the core abstraction for extensibility. To add a new metric:
    1. Create an Objective with appropriate settings
    2. Register it with the optimizer
    
    Attributes:
        name: Unique identifier for this objective (e.g., "quality", "ethics")
        display_name: Human-readable name for summaries
        direction: "maximize" or "minimize" 
        default_weight: Default weight in balanced optimization (should sum to 1.0 across all)
        default_value: Value to use when metric is missing from model (used in IMPUTE mode)
        extractor: Function to extract raw value from (model, decision, context) -> float
        normalization: How to normalize to 0-1 scale
        custom_normalizer: Custom function for CUSTOM normalization
        summary_format: Format string for summary output (use {value}, {diff}, {direction})
        required_fields: List of model attribute names required for this objective.
                        Used in STRICT mode to filter models with incomplete data.
    
    Example:
        # Define an ethics objective
        ethics_objective = Objective(
            name="ethics",
            display_name="Ethics Score",
            direction="maximize",
            default_weight=0.15,
            default_value=50.0,
            extractor=lambda m, d, ctx: getattr(m, 'ethics_score', 50.0),
            normalization=NormalizationMethod.PERCENTAGE,
            required_fields=["ethics_score"],  # For STRICT mode filtering
            summary_format="Ethics: {value:.1f} ({diff:+.1f})"
        )
    """
    name: str
    display_name: str
    direction: str  # "maximize" or "minimize"
    default_weight: float
    default_value: float
    extractor: Callable[[ModelMetadata, RoutingDecision, Dict[str, Any]], float]
    normalization: NormalizationMethod = NormalizationMethod.MIN_MAX
    custom_normalizer: Optional[Callable[[float, float, Dict], float]] = None
    summary_format: Optional[str] = None
    required_fields: List[str] = dataclass_field(default_factory=list)
    
    def extract(
        self, 
        model: ModelMetadata, 
        decision: RoutingDecision, 
        context: Dict[str, Any],
        imputation_value: Optional[float] = None
    ) -> float:
        """
        Extract raw metric value from model.
        
        Args:
            model: Model to extract from
            decision: Routing decision context
            context: Additional context (scorer, etc.)
            imputation_value: Custom value to use if extraction fails (overrides default_value)
        """
        fallback = imputation_value if imputation_value is not None else self.default_value
        try:
            value = self.extractor(model, decision, context)
            return value if value is not None else fallback
        except Exception:
            return fallback
    
    def has_complete_data(self, model: ModelMetadata) -> bool:
        """Check if model has all required fields for this objective."""
        if not self.required_fields:
            return True  # No specific requirements
        
        for field_name in self.required_fields:
            value = getattr(model, field_name, None)
            if value is None:
                return False
        return True
    
    def get_missing_fields(self, model: ModelMetadata) -> List[str]:
        """Get list of required fields that are missing from the model."""
        missing = []
        for field_name in self.required_fields:
            value = getattr(model, field_name, None)
            if value is None:
                missing.append(field_name)
        return missing
    
    def normalize(
        self, 
        value: float, 
        baseline_value: float,
        population_stats: Dict[str, float]
    ) -> float:
        """
        Normalize value to 0-1 scale where 1 is best (closest to utopia).
        
        Args:
            value: Raw metric value
            baseline_value: Baseline model's value for this metric
            population_stats: Dict with "min" and "max" for this metric
            
        Returns:
            Normalized value in [0, 1], higher is better
        """
        if self.normalization == NormalizationMethod.PERCENTAGE:
            # 0-100 scale, higher is better
            return min(1.0, max(0.0, value / 100.0))
        
        elif self.normalization == NormalizationMethod.INVERSE_PERCENTAGE:
            # 0-100 scale, lower is better (e.g., hallucination rate)
            return min(1.0, max(0.0, 1.0 - value / 100.0))
        
        elif self.normalization == NormalizationMethod.RATIO_TO_BASELINE:
            # Normalize relative to baseline using sigmoid-like function
            if baseline_value > 0:
                ratio = value / baseline_value
                if self.direction == "maximize":
                    # Higher ratio = better, use ratio directly capped at 1
                    return min(1.0, ratio)
                else:
                    # Lower ratio = better, invert: 1/(1+ratio)
                    return 1.0 / (1.0 + ratio)
            return 0.5
        
        elif self.normalization == NormalizationMethod.MIN_MAX:
            # Use population min/max
            min_val = population_stats.get("min", 0)
            max_val = population_stats.get("max", 1)
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                return min(1.0, max(0.0, normalized))
            return 1.0
        
        elif self.normalization == NormalizationMethod.INVERSE_MIN_MAX:
            # Min/max but inverted (lower raw value = higher normalized score)
            min_val = population_stats.get("min", 0)
            max_val = population_stats.get("max", 1)
            if max_val > min_val:
                normalized = 1.0 - (value - min_val) / (max_val - min_val)
                return min(1.0, max(0.0, normalized))
            return 1.0
        
        elif self.normalization == NormalizationMethod.CUSTOM:
            if self.custom_normalizer:
                return self.custom_normalizer(value, baseline_value, population_stats)
            return 0.5
        
        return 0.5


class ObjectiveRegistry:
    """
    Registry of optimization objectives.
    
    Provides default objectives and allows adding custom ones.
    
    Example:
        # Get default registry
        registry = ObjectiveRegistry.default()
        
        # Add custom objective
        registry.register(ethics_objective)
        
        # Create optimizer with custom objectives
        optimizer = Optimizer(baseline, data, objectives=registry)
    """
    
    def __init__(self):
        self._objectives: Dict[str, Objective] = {}
    
    def register(self, objective: Objective) -> "ObjectiveRegistry":
        """Register an objective. Returns self for chaining."""
        self._objectives[objective.name] = objective
        return self
    
    def unregister(self, name: str) -> "ObjectiveRegistry":
        """Remove an objective by name. Returns self for chaining."""
        self._objectives.pop(name, None)
        return self
    
    def get(self, name: str) -> Optional[Objective]:
        """Get objective by name."""
        return self._objectives.get(name)
    
    def all(self) -> List[Objective]:
        """Get all registered objectives."""
        return list(self._objectives.values())
    
    def names(self) -> List[str]:
        """Get names of all registered objectives."""
        return list(self._objectives.keys())
    
    def __iter__(self):
        return iter(self._objectives.values())
    
    def __len__(self):
        return len(self._objectives)
    
    @classmethod
    def default(cls) -> "ObjectiveRegistry":
        """Create registry with default objectives."""
        registry = cls()
        
        # Quality objective - uses QualityScorer
        # No strict required_fields since QualityScorer handles missing benchmarks gracefully
        # It will use whatever benchmarks are available
        registry.register(Objective(
            name="quality",
            display_name="Quality",
            direction="maximize",
            default_weight=0.35,
            default_value=50.0,
            extractor=lambda m, d, ctx: ctx["scorer"].calculate_quality_score(
                ctx["model_to_dict"](m), d.category
            ),
            normalization=NormalizationMethod.PERCENTAGE,
            summary_format="Quality: {value:.1f} ({diff:+.1f})",
            required_fields=[],  # QualityScorer handles missing data internally
        ))
        
        # Cost objective - Total Cost of Inference (TCI)
        # TCI handles $0 open-source pricing by applying shadow compute costs
        # This ensures fair comparison between API-managed and self-hosted models
        registry.register(Objective(
            name="cost",
            display_name="Cost (TCI)",
            direction="minimize",
            default_weight=0.20,
            default_value=1.0,
            extractor=_extract_tci,
            normalization=NormalizationMethod.RATIO_TO_BASELINE,
            summary_format="Cost: {pct:.0f}% {direction}",
            required_fields=[],  # TCI can estimate from model name if pricing missing
        ))
        
        # Latency objective - Time To First Token
        # Any of the latency fields is acceptable
        registry.register(Objective(
            name="latency",
            display_name="Latency",
            direction="minimize",
            default_weight=0.15,
            default_value=1.0,
            extractor=_extract_ttft,
            normalization=NormalizationMethod.INVERSE_MIN_MAX,
            summary_format="TTFT: {value_ms:.0f}ms ({diff_pct:+.0f}%)",
            required_fields=[],  # Has fallback chain, no strict requirement
        ))
        
        # Hallucination objective
        # Returns None when missing to allow imputation logic to work
        registry.register(Objective(
            name="hallucination",
            display_name="Hallucination",
            direction="minimize",
            default_weight=0.20,
            default_value=15.0,
            extractor=lambda m, d, ctx: (
                float(m.hallucination_rate) 
                if hasattr(m, 'hallucination_rate') and m.hallucination_rate is not None 
                else None
            ),
            normalization=NormalizationMethod.INVERSE_PERCENTAGE,
            summary_format="Halluc: {value:.1f}% ({diff:+.1f})",
            required_fields=["hallucination_rate"],
        ))
        
        # Refusal objective
        # Returns None when missing to allow imputation logic to work
        registry.register(Objective(
            name="refusal",
            display_name="Refusal",
            direction="minimize",
            default_weight=0.10,
            default_value=5.0,
            extractor=lambda m, d, ctx: (
                float(m.refusal_rate)
                if hasattr(m, 'refusal_rate') and m.refusal_rate is not None
                else None
            ),
            normalization=NormalizationMethod.INVERSE_PERCENTAGE,
            summary_format="Refusal: {value:.1f}%",
            required_fields=["refusal_rate"],
        ))
        
        return registry


# =============================================================================
# Metric Orientation Utilities
# =============================================================================

def orient_to_maximize(
    value: float,
    direction: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """
    Orient a metric value so higher is always better.
    
    For Pareto analysis, all metrics must share a direction. This function
    converts any metric to "maximize" orientation where higher = better.
    
    Args:
        value: Raw metric value
        direction: "maximize" or "minimize"
        min_val: Minimum possible value (for normalization)
        max_val: Maximum possible value (for normalization)
        
    Returns:
        Oriented value where higher is better
        
    Example:
        >>> orient_to_maximize(0.8, "maximize")  # Quality: 80% is good
        0.8
        >>> orient_to_maximize(0.2, "minimize")  # Latency: 20% of max is good
        0.8  # Inverted: 1 - 0.2 = 0.8
    """
    if direction == "minimize":
        # Invert: lower raw value → higher oriented value
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
            return 1.0 - min(1.0, max(0.0, normalized))
        return 1.0 - value
    return value


def orient_cost(cost: float, baseline_cost: float = 1.0) -> float:
    """
    Orient cost metric for Pareto analysis.
    
    Uses inverse ratio: lower cost → higher value.
    Formula: 1 / (1 + cost_ratio) where cost_ratio = cost / baseline
    
    This has nice properties:
    - cost = 0 → value = 1.0 (free is best)
    - cost = baseline → value = 0.5
    - cost → ∞ → value → 0
    
    Args:
        cost: Cost per 1M tokens (TCI)
        baseline_cost: Reference cost for normalization
        
    Returns:
        Oriented value in (0, 1] where higher is better (cheaper)
    """
    if baseline_cost <= 0:
        baseline_cost = 1.0
    cost_ratio = cost / baseline_cost
    return 1.0 / (1.0 + cost_ratio)


def orient_latency(latency: float, baseline_latency: float = 1.0) -> float:
    """
    Orient latency metric for Pareto analysis.
    
    Uses inverse ratio: lower latency → higher value.
    Formula: 1 / (1 + latency_ratio) where latency_ratio = latency / baseline
    
    Args:
        latency: Latency in seconds (TTFT)
        baseline_latency: Reference latency for normalization
        
    Returns:
        Oriented value in (0, 1] where higher is better (faster)
    """
    if baseline_latency <= 0:
        baseline_latency = 1.0
    latency_ratio = latency / baseline_latency
    return 1.0 / (1.0 + latency_ratio)


def _extract_ttft(m: ModelMetadata, d: RoutingDecision, ctx: Dict) -> float:
    """Extract Time To First Token from model."""
    if hasattr(m, 'measured_ttft_seconds') and m.measured_ttft_seconds:
        return float(m.measured_ttft_seconds)
    if hasattr(m, 'time_to_first_token_seconds') and m.time_to_first_token_seconds:
        return float(m.time_to_first_token_seconds)
    if m.median_latency_ms and m.median_latency_ms > 0:
        return m.median_latency_ms / 1000.0
    return 1.0


def _extract_tci(m: ModelMetadata, d: RoutingDecision, ctx: Dict) -> float:
    """
    Extract Total Cost of Inference (TCI) from model.
    
    TCI addresses the $0 problem in Pareto optimization:
    - Managed Models (GPT-4, Claude): Uses API pricing
    - Self-Hosted (Llama, Mistral): Uses shadow compute cost
    
    This ensures open-source models with $0 API pricing don't
    artificially dominate cost-sensitive optimization.
    
    Returns:
        Cost per 1M blended tokens (75% input, 25% output)
    """
    return calculate_tci(
        input_cost_per_m=m.input_cost_per_m,
        output_cost_per_m=m.output_cost_per_m,
        param_count_b=m.param_count_b,
        model_name=m.name,
    )


# =============================================================================
# Strategy definitions
# =============================================================================

class OptimizationStrategy(Enum):
    """Strategy for multi-objective optimization."""
    HYBRID = "hybrid"                       # Pareto-Chebyshev Fusion (default) - best of both worlds
    QUALITY_FOCUSED = "quality"             # Prioritize quality (high weight on quality)
    COST_FOCUSED = "cost"                   # Prioritize cost savings
    BALANCED = "balanced"                   # Balanced optimization (Chebyshev only)
    SPEED_FOCUSED = "speed"                 # Prioritize low latency
    RELIABILITY_FOCUSED = "reliability"     # Prioritize low hallucination/refusal
    VALUE_OPTIMIZED = "value"               # Sweet spot: good quality at low cost (constrained)
    KNEE = "knee"                           # Find the "knee" point - best bang for buck (Pareto only)
    UTOPIAN = "utopian"                     # Distance to Utopia - handles $0 costs mathematically
    KNEEDLE = "kneedle"                     # Kneedle algorithm on 2D projected curve (max curvature)
    UTOPIAN_CHEBYSHEV = "utopian_chebyshev" # 5D Augmented Chebyshev with Utopian normalization
    PARETO_CHEBYSHEV = "pareto_chebyshev"   # 3-Phase: Shadow Price → Pareto Filter → Chebyshev (RECOMMENDED)


# Strategy weight presets - these auto-adjust to include all registered objectives
STRATEGY_WEIGHT_PRESETS: Dict[OptimizationStrategy, Dict[str, float]] = {
    OptimizationStrategy.HYBRID: {
        # Balanced 4D weights for Chebyshev component of hybrid
        "quality": 0.35,
        "cost": 0.25,
        "latency": 0.20,
        "hallucination": 0.15,
        "refusal": 0.05,
    },
    OptimizationStrategy.QUALITY_FOCUSED: {
        "quality": 0.50,
        "cost": 0.10,
        "latency": 0.10,
        "hallucination": 0.20,
        "refusal": 0.10,
    },
    OptimizationStrategy.COST_FOCUSED: {
        "quality": 0.20,
        "cost": 0.40,
        "latency": 0.15,
        "hallucination": 0.15,
        "refusal": 0.10,
    },
    OptimizationStrategy.SPEED_FOCUSED: {
        "quality": 0.20,
        "cost": 0.15,
        "latency": 0.40,
        "hallucination": 0.15,
        "refusal": 0.10,
    },
    OptimizationStrategy.RELIABILITY_FOCUSED: {
        "quality": 0.25,
        "cost": 0.10,
        "latency": 0.10,
        "hallucination": 0.35,
        "refusal": 0.20,
    },
    OptimizationStrategy.VALUE_OPTIMIZED: {
        "quality": 0.35,
        "cost": 0.25,
        "latency": 0.15,
        "hallucination": 0.15,
        "refusal": 0.10,
    },
    # UTOPIAN uses Euclidean distance, weights control axis importance
    OptimizationStrategy.UTOPIAN: {
        "quality": 0.40,
        "cost": 0.30,
        "latency": 0.30,
        "hallucination": 0.0,  # Not used in 3D utopian distance
        "refusal": 0.0,
    },
    # KNEEDLE projects to 2D (Performance vs Cost) and finds max curvature
    OptimizationStrategy.KNEEDLE: {
        "quality": 0.60,  # Used in Performance = α*Quality + (1-α)*Latency
        "cost": 0.0,      # Cost is on separate axis
        "latency": 0.40,  # Combined with quality into Performance
        "hallucination": 0.0,
        "refusal": 0.0,
    },
    # UTOPIAN_CHEBYSHEV: 5D Augmented Chebyshev with Utopian normalization
    # These weights map to: efficiency, trust, quality, latency, cost
    OptimizationStrategy.UTOPIAN_CHEBYSHEV: {
        "quality": 0.30,       # Task-specific benchmark performance
        "cost": 0.20,          # Affordability
        "latency": 0.15,       # Response speed
        "hallucination": 0.20, # Trust (combined with refusal)
        "refusal": 0.0,        # Included in trust
        # Note: efficiency (0.15) is derived from quality/params, not a separate weight here
    },
    # PARETO_CHEBYSHEV: 3-Phase optimization with business targets
    # Strictness weights (high = severe punishment for missing target)
    OptimizationStrategy.PARETO_CHEBYSHEV: {
        "quality": 0.35,       # Task performance
        "cost": 0.25,          # Budget constraint
        "latency": 0.20,       # Speed requirement
        "hallucination": 0.20, # Trust requirement
        "refusal": 0.0,        # Included in trust
    },
}

# Hybrid strategy parameters - Pareto dominance as 5th Chebyshev dimension
HYBRID_DEFAULTS = {
    # Use quality-cost for Pareto dominance, but weight by quality
    # High-quality dominators score more, low-quality dominators score less
    "pareto_dimensions": ["quality", "cost"],
}


# =============================================================================
# Main Optimizer class
# =============================================================================

class Optimizer:
    """
    Multi-objective optimizer for LLM model selection.
    
    Supports three optimization approaches:
    
    1. **HYBRID (5D Chebyshev with Net Pareto Dominance)** - DEFAULT:
       Extends Chebyshev to 5 dimensions by adding Pareto dominance as an objective:
       - Quality: Task-specific benchmark score
       - Cost: Pricing per 1M tokens
       - Latency: Time To First Token
       - Trust: Hallucination/refusal rates
       - Net Pareto Dominance: (quality-weighted dominations) - (quality-weighted dominated-by)
       
       High-quality models that dominate many but are dominated by few score highest.
       Low-quality models get heavily penalized by being dominated by many others.
       No magic thresholds - just uses quality as a natural weight.
    
    2. **Chebyshev Scalarization** (BALANCED and other weight-based strategies):
       Minimizes distance to utopia point across 4 objectives.
       Good for when you have clear preferences via weights.
    
    3. **Knee Point Detection** (KNEE strategy):
       Finds the point of maximum curvature on the Pareto frontier.
       Best "bang for buck" - further improvements require disproportionate sacrifices.
    
    Extensibility:
        Add new objectives by creating an Objective and passing a custom
        ObjectiveRegistry to the constructor.
        
        Example - adding an ethics metric:
        
            registry = ObjectiveRegistry.default()
            registry.register(Objective(
                name="ethics",
                display_name="Ethics",
                direction="maximize",
                default_weight=0.15,
                default_value=50.0,
                extractor=lambda m, d, ctx: getattr(m, 'ethics_score', 50.0),
                normalization=NormalizationMethod.PERCENTAGE,
            ))
            
            optimizer = Optimizer(
                baseline, models_data,
                objectives=registry,
                custom_weights={"quality": 0.30, "ethics": 0.15, ...}
            )
    """
    
    def __init__(
        self, 
        baseline_model: ModelMetadata, 
        all_models_data: List[Dict],
        strategy: OptimizationStrategy = OptimizationStrategy.HYBRID,
        objectives: Optional[ObjectiveRegistry] = None,
        quality_range: Optional[tuple] = None,
        cost_range: Optional[tuple] = None,
        speed_range: Optional[tuple] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        knee_position_weight: float = 0.3,
        knee_objective_weights: Optional[Dict[str, float]] = None,
        missing_data: MissingDataStrategy = MissingDataStrategy.STRICT,
        imputation_values: Optional[Dict[str, float]] = None,
        # Hybrid strategy parameters
        hybrid_pareto_dimensions: Optional[List[str]] = None,
    ):
        """
        Initialize optimizer with baseline model and population for quality scoring.
        
        Args:
            baseline_model: Reference model for comparison (e.g., Gemini 3 Pro)
            all_models_data: Full model dataset (for QualityScorer initialization)
            strategy: Optimization strategy to use. Default is HYBRID (Pareto-Chebyshev Fusion).
            objectives: Custom ObjectiveRegistry (default: ObjectiveRegistry.default())
            quality_range: (min, max) quality ratios for VALUE_OPTIMIZED (default: (0.80, 0.95))
            cost_range: (min, max) cost ratios for VALUE_OPTIMIZED (default: (0.10, 0.30))
            speed_range: (min, max) speed ratios for VALUE_OPTIMIZED (default: None = no constraint)
            custom_weights: Override default weights for objectives
            knee_position_weight: Weight for position bonus in KNEE strategy (0-1, default: 0.3)
            knee_objective_weights: Custom weights for objectives in KNEE benefit/cost calculation.
            missing_data: How to handle models with missing data:
                - STRICT (default): Only optimize models with complete data for all objectives.
                  Highest confidence results, but may reduce the candidate pool.
                - IMPUTE: Use default values for missing data. Includes more models but
                  with lower confidence since some values are estimated.
            imputation_values: Custom values to use when imputing missing data (IMPUTE mode only).
                Maps objective names to values, e.g., {"hallucination": 20.0, "refusal": 10.0}.
                If not specified, uses the objective's default_value.
            hybrid_pareto_dimensions: Dimensions for Pareto dominance (default: all 4D for full balance)
        
        Example:
            # Standard usage with HYBRID strategy (default)
            optimizer = Optimizer(baseline, models_data)
            
            # Use pure Knee Point strategy
            optimizer = Optimizer(
                baseline, models_data, 
                strategy=OptimizationStrategy.KNEE,
            )
            
            # Custom hybrid weights
            optimizer = Optimizer(
                baseline, models_data,
                hybrid_chebyshev_weight=0.7,  # More emphasis on balance
                hybrid_knee_weight=0.3,       # Less on value
            )
        """
        self.baseline = baseline_model
        self.strategy = strategy
        self.scorer = QualityScorer(all_models_data=all_models_data)
        
        # Missing data handling
        self.missing_data = missing_data
        self.imputation_values = imputation_values or {}
        
        # Use provided objectives or default
        self.objectives = objectives or ObjectiveRegistry.default()
        
        # Configurable constraint ranges for VALUE_OPTIMIZED
        # None means no constraint for that dimension
        self.quality_range = quality_range  # None = no relative quality constraint
        self.cost_range = cost_range  # None = no relative cost constraint
        self.speed_range = speed_range
        
        # Custom weights override
        self.custom_weights = custom_weights
        
        # Hybrid strategy parameters - quality-cost dominance, weighted by quality
        self.hybrid_pareto_dimensions = hybrid_pareto_dimensions or ["quality", "cost"]
        
        # KNEE strategy customization
        self.knee_position_weight = knee_position_weight
        self.knee_objective_weights = knee_objective_weights or {
            obj.name: 1.0 for obj in self.objectives
        }
        
        # Context passed to objective extractors
        self._context = {
            "scorer": self.scorer,
            "model_to_dict": self._model_to_dict,
        }

    @property
    def DEFAULT_WEIGHTS(self) -> Dict[str, float]:
        """Get default weights from registered objectives."""
        return {obj.name: obj.default_weight for obj in self.objectives}

    def _get_weights(self) -> Dict[str, float]:
        """
        Get optimization weights based on strategy.
        
        For strategies with presets, uses those weights.
        For unknown objectives, uses their default_weight.
        
        Returns:
            Dict mapping objective names to weights
        """
        if self.custom_weights:
            return self._fill_missing_weights(self.custom_weights)
        
        preset = STRATEGY_WEIGHT_PRESETS.get(self.strategy)
        if preset:
            return self._fill_missing_weights(preset)
        
        # BALANCED or unknown - use default weights
        return self.DEFAULT_WEIGHTS
    
    def _fill_missing_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Fill in missing objective weights with defaults."""
        result = dict(weights)
        for obj in self.objectives:
            if obj.name not in result:
                result[obj.name] = obj.default_weight
        return result

    def _get_model_metrics(self, m: ModelMetadata, decision: RoutingDecision) -> Dict[str, float]:
        """
        Extract all metrics from a model for optimization.
        
        Uses imputation_values if provided and missing_data is IMPUTE.
        
        Args:
            m: Model metadata
            decision: Routing decision for task-specific quality
            
        Returns:
            Dict mapping objective names to raw metric values
        """
        return {
            obj.name: obj.extract(
                m, decision, self._context,
                imputation_value=self.imputation_values.get(obj.name)
            )
            for obj in self.objectives
        }
    
    def has_complete_data(self, model: ModelMetadata) -> bool:
        """
        Check if a model has complete data for all objectives.
        
        Args:
            model: Model to check
            
        Returns:
            True if model has all required fields for all objectives
        """
        for obj in self.objectives:
            if not obj.has_complete_data(model):
                return False
        return True
    
    def get_missing_data_report(self, model: ModelMetadata) -> Dict[str, List[str]]:
        """
        Get a report of missing data for a model.
        
        Args:
            model: Model to check
            
        Returns:
            Dict mapping objective names to lists of missing fields
        """
        report = {}
        for obj in self.objectives:
            missing = obj.get_missing_fields(model)
            if missing:
                report[obj.name] = missing
        return report
    
    def filter_complete_models(
        self, 
        models: List[ModelMetadata],
        verbose: bool = False
    ) -> List[ModelMetadata]:
        """
        Filter to only models with complete data for all objectives.
        
        Args:
            models: List of models to filter
            verbose: Whether to print filtering summary
            
        Returns:
            List of models with complete data
        """
        complete = []
        incomplete_count = 0
        
        for m in models:
            if self.has_complete_data(m):
                complete.append(m)
            else:
                incomplete_count += 1
        
        if verbose and incomplete_count > 0:
            print(f"ℹ️  STRICT mode: Filtered out {incomplete_count} models with incomplete data")
            print(f"   Remaining: {len(complete)} models with complete data")
        
        return complete

    def get_completeness_stats(
        self, 
        models: List[ModelMetadata]
    ) -> Dict[str, Any]:
        """
        Get statistics about data completeness for a list of models.
        
        Args:
            models: List of models to analyze
            
        Returns:
            Dictionary with completeness statistics:
            - total_models: Total number of models
            - complete_models: Number of models with complete data
            - incomplete_models: Number of models with missing data
            - completeness_rate: Percentage of models with complete data
            - missing_by_objective: Dict of objective name -> count of models missing that objective
        """
        total = len(models)
        complete = 0
        incomplete = 0
        missing_by_objective: Dict[str, int] = {obj.name: 0 for obj in self.objectives}
        
        for m in models:
            if self.has_complete_data(m):
                complete += 1
            else:
                incomplete += 1
                # Count which objectives are missing
                for obj in self.objectives:
                    if not obj.has_complete_data(m):
                        missing_by_objective[obj.name] += 1
        
        return {
            "total_models": total,
            "complete_models": complete,
            "incomplete_models": incomplete,
            "completeness_rate": (complete / total * 100) if total > 0 else 0.0,
            "missing_by_objective": missing_by_objective,
        }

    def _normalize_metrics(
        self, 
        metrics: Dict[str, float], 
        baseline_metrics: Dict[str, float],
        population_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Normalize metrics to 0-1 scale where 1 is best (closest to utopia).
        
        Args:
            metrics: Raw metrics for a model
            baseline_metrics: Raw metrics for baseline model
            population_stats: Min/max stats for normalization
            
        Returns:
            Dict with normalized metrics (0-1, higher is better)
        """
        normalized = {}
        for obj in self.objectives:
            normalized[obj.name] = obj.normalize(
                metrics[obj.name],
                baseline_metrics[obj.name],
                population_stats.get(obj.name, {"min": 0, "max": 100})
            )
        return normalized

    def _calculate_population_stats(
        self, 
        models: List[ModelMetadata], 
        decision: RoutingDecision
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate min/max statistics for normalization.
        
        Args:
            models: List of models
            decision: Routing decision
            
        Returns:
            Dict with min/max for each objective
        """
        stats = {
            obj.name: {"min": float('inf'), "max": float('-inf')}
            for obj in self.objectives
        }
        
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            for name, value in metrics.items():
                stats[name]["min"] = min(stats[name]["min"], value)
                stats[name]["max"] = max(stats[name]["max"], value)
        
        return stats

    def _calculate_knee_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
        baseline_metrics: Dict[str, float],
        population_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate knee point scores for all models.
        
        The "knee" is where you get the best "bang for your buck" - 
        the point of MAXIMUM UTILITY on the Pareto frontier.
        
        Algorithm:
        1. Extract quality and cost for each model
        2. Normalize both to [0, 1] range
        3. Find the Pareto frontier (non-dominated models)
        4. Use ADDITIVE UTILITY to avoid division-by-zero:
           U = α·norm_quality + (1-α)·(1 - norm_cost)
           where α controls quality-vs-cost preference (default 0.6)
        5. Score all models by utility (higher = better)
        
        This is mathematically sound because:
        - No division by cost (handles free models naturally)
        - Free models compete fairly on quality
        - Expensive models need higher quality to compensate
        
        Args:
            models: List of models
            decision: Routing decision
            baseline_metrics: Baseline model metrics
            population_stats: Population statistics for normalization
            
        Returns:
            Dict mapping model names to knee scores (lower = better)
        """
        # Quality weight (higher = prioritize quality over cost savings)
        alpha = 0.6
        
        # Step 1: Extract quality and cost for each model
        model_data = []
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            quality = metrics.get('quality', 0)
            cost = metrics.get('cost', 0)
            
            # Skip models with zero or negative quality
            if quality <= 0:
                continue
                
            model_data.append({
                "model": m,
                "name": m.name,
                "quality": quality,
                "cost": cost,
            })
        
        if len(model_data) < 2:
            return {d["name"]: 0.0 for d in model_data}
        
        # Step 2: Normalize quality and cost to [0, 1]
        min_quality = min(d["quality"] for d in model_data)
        max_quality = max(d["quality"] for d in model_data)
        min_cost = min(d["cost"] for d in model_data)
        max_cost = max(d["cost"] for d in model_data)
        
        quality_range = max_quality - min_quality if max_quality > min_quality else 1.0
        cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
        
        for d in model_data:
            d["norm_quality"] = (d["quality"] - min_quality) / quality_range
            d["norm_cost"] = (d["cost"] - min_cost) / cost_range
        
        # Step 3: Find Pareto frontier (non-dominated on quality-cost)
        # Sort by cost ascending
        sorted_by_cost = sorted(model_data, key=lambda x: x["cost"])
        
        pareto_frontier = []
        max_quality_seen = -float('inf')
        
        for d in sorted_by_cost:
            # A model is on the frontier if it has higher quality than all cheaper models
            if d["quality"] > max_quality_seen:
                pareto_frontier.append(d)
                max_quality_seen = d["quality"]
        
        frontier_names = {d["name"] for d in pareto_frontier}
        
        # Step 4: Calculate ADDITIVE UTILITY for all models
        # U = α·norm_quality + (1-α)·(1 - norm_cost)
        # Higher utility = better (quality contributes positively, cost negatively)
        
        for d in model_data:
            # Base utility: weighted sum of normalized quality and cost savings
            utility = alpha * d["norm_quality"] + (1 - alpha) * (1 - d["norm_cost"])
            
            # Bonus for being on Pareto frontier (non-dominated)
            if d["name"] in frontier_names:
                utility += 0.1  # 10% bonus for Pareto-optimal models
            
            d["utility"] = utility
        
        # Step 5: Convert utility to scores (lower = better for ranking)
        max_utility = max(d["utility"] for d in model_data)
        min_utility = min(d["utility"] for d in model_data)
        utility_range = max_utility - min_utility if max_utility > min_utility else 1.0
        
        knee_scores = {}
        for d in model_data:
            # Invert: high utility -> low score (better)
            knee_scores[d["name"]] = 1.0 - (d["utility"] - min_utility) / utility_range
        
        return knee_scores

    def _calculate_pareto_dominance_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
        dimensions: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate net Pareto dominance with quality weighting.
        
        Net dominance = (quality-weighted dominations) - (quality-weighted dominated_by)
        
        This has two effects:
        1. Models dominated by many others get penalized
        2. High-quality dominators score more than low-quality ones
        
        Example:
        - Q:90 model dominates 5 others, dominated by 1 high-quality model
          Score = 0.90*5 - 0.95*1 = 3.55
        - Q:40 model dominates 10 others, dominated by 15 mid-quality models
          Score = 0.40*10 - 0.60*15 = -5.0 (bad)
        
        Args:
            models: List of models to score
            decision: Routing decision for task-specific quality
            dimensions: Which objectives to consider
            
        Returns:
            Dict mapping model names to dominance scores (0-1, higher = better)
        """
        if not models or len(models) < 2:
            return {m.name: 1.0 for m in models}
        
        dims = dimensions or self.hybrid_pareto_dimensions
        
        # Get metrics for all models
        model_metrics = []
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            model_metrics.append({
                "model": m,
                "name": m.name,
                "metrics": metrics,
            })
        
        # Calculate domination matrix
        # dominates[i][j] = True if model i dominates model j
        n = len(model_metrics)
        dominates = [[False] * n for _ in range(n)]
        
        for i, candidate in enumerate(model_metrics):
            for j, other in enumerate(model_metrics):
                if i == j:
                    continue
                
                # Check if 'candidate' dominates 'other'
                dominates_all = True
                strictly_better_in_one = False
                
                for dim in dims:
                    c_val = candidate["metrics"].get(dim, 0)
                    o_val = other["metrics"].get(dim, 0)
                    
                    obj = self.objectives.get(dim)
                    if obj and obj.direction == "minimize":
                        if c_val > o_val:
                            dominates_all = False
                        elif c_val < o_val:
                            strictly_better_in_one = True
                    else:
                        if c_val < o_val:
                            dominates_all = False
                        elif c_val > o_val:
                            strictly_better_in_one = True
                
                dominates[i][j] = dominates_all and strictly_better_in_one
        
        # Calculate net dominance with quality weighting
        raw_scores = {}
        
        for i, candidate in enumerate(model_metrics):
            candidate_quality = candidate["metrics"].get("quality", 50) / 100.0
            
            # Sum of (my quality) for each model I dominate
            dominating_score = sum(
                candidate_quality
                for j in range(n) if dominates[i][j]
            )
            
            # Sum of (their quality) for each model that dominates me
            dominated_score = sum(
                model_metrics[j]["metrics"].get("quality", 50) / 100.0
                for j in range(n) if dominates[j][i]
            )
            
            # Net dominance: positive = good, negative = bad
            raw_scores[candidate["name"]] = dominating_score - dominated_score
        
        # Shift and normalize to 0-1
        min_score = min(raw_scores.values())
        max_score = max(raw_scores.values())
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        return {
            name: (score - min_score) / score_range
            for name, score in raw_scores.items()
        }

    def _calculate_hybrid_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
        baseline_metrics: Dict[str, float],
        population_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate hybrid Pareto-Chebyshev scores.
        
        The hybrid approach adds Pareto dominance as a 5th dimension to Chebyshev:
        - Quality, Cost, Latency, Trust (original 4D)
        - Pareto Dominance: How many models does this one dominate on quality-cost?
        
        This naturally penalizes models that are dominated by many others
        (low quality OR high cost relative to alternatives).
        
        Formula:
            hybrid_score = max(
                w_quality * quality_regret,
                w_cost * cost_regret,
                w_latency * latency_regret,
                w_trust * trust_regret,
                w_dominance * dominance_regret  # NEW: Pareto efficiency
            )
        
        Args:
            models: List of models to score
            decision: Routing decision
            baseline_metrics: Baseline model metrics
            population_stats: Population statistics for normalization
            
        Returns:
            Dict mapping model names to hybrid scores (lower = better)
        """
        weights = self._get_weights()
        
        # Calculate Pareto dominance scores (0-1, higher = dominates more models)
        dominance_scores = self._calculate_pareto_dominance_scores(
            models, decision, self.hybrid_pareto_dimensions
        )
        
        # Get weight for dominance dimension (use average of quality and cost weights)
        dominance_weight = (weights.get("quality", 0.35) + weights.get("cost", 0.25)) / 2
        
        hybrid_scores = {}
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            norm = self._normalize_metrics(metrics, baseline_metrics, population_stats)
            regrets = {k: max(0, 1.0 - v) for k, v in norm.items()}
            
            # Add dominance regret (1 - dominance_score, so low dominance = high regret)
            dominance = dominance_scores.get(m.name, 0.5)
            dominance_regret = 1.0 - dominance
            
            # Chebyshev distance: max weighted regret across ALL dimensions including dominance
            weighted_regrets = [
                weights.get(obj.name, 0.1) * regrets[obj.name]
                for obj in self.objectives
            ]
            # Add dominance as 5th dimension
            weighted_regrets.append(dominance_weight * dominance_regret)
            
            hybrid_scores[m.name] = max(weighted_regrets)
        
        return hybrid_scores

    def _calculate_utopian_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
        population_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate Utopian Distance scores for all models.
        
        The Utopian Distance method:
        1. Uses Min-Max normalization where 1 = optimal for all metrics
        2. Calculates weighted Euclidean distance from Utopia (1, 1, 1)
        3. Handles $0 costs naturally (they become 1.0 = optimal)
        
        Formula:
            D = sqrt(w_q*(1-Q_norm)² + w_l*(1-L_norm)² + w_c*(1-C_norm)²)
        
        The model with MINIMUM distance is optimal.
        
        Args:
            models: List of models to score
            decision: Routing decision
            population_stats: Population statistics for normalization
            
        Returns:
            Dict mapping model names to utopian distance scores (lower = better)
        """
        weights = self._get_weights()
        
        # Get population min/max for quality, cost, latency
        q_stats = population_stats.get("quality", {"min": 0, "max": 100})
        c_stats = population_stats.get("cost", {"min": 0, "max": 10})
        l_stats = population_stats.get("latency", {"min": 0, "max": 5})
        
        utopian_scores = {}
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            
            # Get raw values
            quality = metrics.get("quality", 50)
            cost = metrics.get("cost", 1.0)
            latency = metrics.get("latency", 1.0)
            
            # Min-Max normalize with proper direction
            # Quality: maximize → (x - min) / (max - min)
            q_norm = minmax_normalize(
                quality, q_stats["min"], q_stats["max"], direction="maximize"
            )
            # Cost: minimize → (max - x) / (max - min) 
            # $0 cost → normalized to 1.0 (optimal)
            c_norm = minmax_normalize(
                cost, c_stats["min"], c_stats["max"], direction="minimize"
            )
            # Latency: minimize → (max - x) / (max - min)
            l_norm = minmax_normalize(
                latency, l_stats["min"], l_stats["max"], direction="minimize"
            )
            
            # Calculate weighted Euclidean distance to utopia (1, 1, 1)
            distance = calculate_utopian_distance(
                quality_norm=q_norm,
                latency_norm=l_norm,
                cost_norm=c_norm,
                weight_quality=weights.get("quality", 0.4),
                weight_latency=weights.get("latency", 0.3),
                weight_cost=weights.get("cost", 0.3),
            )
            
            utopian_scores[m.name] = distance
        
        return utopian_scores

    def _calculate_utopian_chebyshev_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
    ) -> Dict[str, float]:
        """
        Calculate 5D Augmented Chebyshev scores with Utopian normalization.
        
        This implements the improved algorithm:
        1. Directional Unification: All metrics oriented to 1.0 = utopia
        2. 5 Dimensions: Efficiency, Trust, Quality, Latency, Cost
        3. Augmented Chebyshev: max(w_i · dev_i) + ρ · Σdev_i
        
        The model with MINIMUM score is optimal.
        
        Args:
            models: List of models to score
            decision: Routing decision for task-specific quality
            
        Returns:
            Dict mapping model names to augmented Chebyshev scores (lower = better)
        """
        # Build model data for the scorer
        model_data_list = []
        quality_scores = {}
        
        for m in models:
            model_dict = self._model_to_dict(m)
            # Add additional fields needed by AugmentedChebyshevScorer
            model_dict['param_count_b'] = m.param_count_b
            model_dict['hallucination_rate'] = getattr(m, 'hallucination_rate', None)
            model_dict['refusal_rate'] = getattr(m, 'refusal_rate', None)
            model_dict['factual_consistency_rate'] = getattr(m, 'factual_consistency_rate', None)
            model_dict['measured_ttft_seconds'] = getattr(m, 'measured_ttft_seconds', None)
            model_dict['time_to_first_token_seconds'] = getattr(m, 'time_to_first_token_seconds', None)
            model_dict['median_latency_ms'] = m.median_latency_ms
            model_dict['input_cost_per_m'] = m.input_cost_per_m
            model_dict['output_cost_per_m'] = m.output_cost_per_m
            model_dict['name'] = m.name
            
            model_data_list.append(model_dict)
            
            # Calculate quality score for this category
            quality = self.scorer.calculate_quality_score(model_dict, decision.category)
            quality_scores[m.name] = quality
        
        # Get weights from strategy presets
        weights = self._get_weights()
        
        # Map to AugmentedChebyshev weights
        # Note: efficiency is derived from quality/params, trust from hallucination/refusal
        augmented_weights = {
            "efficiency": 0.15,  # Fixed weight for efficiency
            "trust": weights.get("hallucination", 0.20),
            "quality": weights.get("quality", 0.30),
            "latency": weights.get("latency", 0.15),
            "cost": weights.get("cost", 0.20),
        }
        
        # Create scorer with these weights
        scorer = AugmentedChebyshevScorer(weights=augmented_weights)
        
        # Score all models
        scores = scorer.score_models(model_data_list, quality_scores)
        
        # Return as dict
        return {s.model_name: s.augmented_score for s in scores}

    def _calculate_pareto_chebyshev_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Calculate 3-Phase Pareto-Chebyshev scores.
        
        Phase 1: Shadow Price Fix (no $0 costs)
        Phase 2: Pareto Filter (remove dominated models)
        Phase 3: Augmented Chebyshev (rank by regret from business targets)
        
        Args:
            models: List of models to score
            decision: Routing decision for task-specific quality
            
        Returns:
            Tuple of (scores_dict, removed_model_names)
            - scores_dict: model name → regret score (lower = better)
            - removed_model_names: list of dominated models that were filtered out
        """
        # Build model data for the optimizer
        model_data_list = []
        quality_scores = {}
        
        for m in models:
            model_dict = self._model_to_dict(m)
            # Add additional fields
            model_dict['param_count_b'] = m.param_count_b
            model_dict['hallucination_rate'] = getattr(m, 'hallucination_rate', None)
            model_dict['refusal_rate'] = getattr(m, 'refusal_rate', None)
            model_dict['measured_ttft_seconds'] = getattr(m, 'measured_ttft_seconds', None)
            model_dict['time_to_first_token_seconds'] = getattr(m, 'time_to_first_token_seconds', None)
            model_dict['median_latency_ms'] = m.median_latency_ms
            model_dict['input_cost_per_m'] = m.input_cost_per_m
            model_dict['output_cost_per_m'] = m.output_cost_per_m
            model_dict['name'] = m.name
            
            model_data_list.append(model_dict)
            
            # Calculate quality score for this category
            quality = self.scorer.calculate_quality_score(model_dict, decision.category)
            quality_scores[m.name] = quality
        
        # Get weights from strategy presets
        weights = self._get_weights()
        
        # Set minimum cost from population (for shadow pricing free models)
        set_minimum_cost_from_population(model_data_list)
        
        # Create business targets based on population
        all_costs = [calculate_tci(m.get('input_cost_per_m'), m.get('output_cost_per_m'), 
                                   model_name=m.get('name')) for m in model_data_list]
        all_qualities = list(quality_scores.values())
        
        import numpy as np
        # Cost target: 25th percentile (achievable for good models)
        target_cost = np.percentile(all_costs, 25)
        
        # Quality target: Use the LEAD MODEL's quality (the best available)
        # This asks: "How close can you get to the best model while meeting other constraints?"
        lead_quality = max(all_qualities)
        
        # Configure the optimizer
        config = OptimizationConfig(
            targets=BusinessTargets(
                cost=max(0.10, target_cost),  # Min $0.10 target
                latency=0.300,  # 300ms default target
                quality=lead_quality,  # Target = lead model's quality
                trust=85.0,  # 85% trust target
            ),
            weights=StrictnessWeights(
                cost=weights.get("cost", 0.25),
                latency=weights.get("latency", 0.20),
                quality=weights.get("quality", 0.35),
                trust=weights.get("hallucination", 0.20),
            ),
            apply_pareto_filter=True,
        )
        
        # Run three-phase optimization
        optimizer = ParetoChebyshevOptimizer(config)
        result = optimizer.optimize(model_data_list, quality_scores, verbose=False)
        
        # Convert results to scores dict
        scores = {r.name: r.regret for r in result.ranked}
        
        # Models that were filtered out get a high score (they're dominated)
        for name in result.dominated_removed:
            scores[name] = float('inf')  # Dominated models rank last
        
        return scores, result.dominated_removed

    def _calculate_kneedle_scores(
        self,
        models: List[ModelMetadata],
        decision: RoutingDecision,
        population_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate Kneedle scores for all models.
        
        The Kneedle algorithm:
        1. Projects 3D (Quality, Latency, Cost) to 2D (Performance, Cost)
        2. Uses log(Cost + ε) to handle $0 costs
        3. Finds the Pareto frontier
        4. Draws a line from cheapest to most expensive
        5. The "knee" is the point with maximum perpendicular distance
        
        Models are scored by their distance to the efficiency line,
        with the knee point getting the best score (lowest).
        
        Args:
            models: List of models to score
            decision: Routing decision
            population_stats: Population statistics for normalization
            
        Returns:
            Dict mapping model names to kneedle scores (lower = better)
        """
        weights = self._get_weights()
        alpha = weights.get("quality", 0.6)
        
        # Build model data for kneedle algorithm
        model_data = []
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            model_data.append({
                'name': m.name,
                'quality': metrics.get("quality", 50),
                'latency': metrics.get("latency", 1.0),
                'cost': metrics.get("cost", 1.0),
            })
        
        # Run kneedle algorithm
        knee_point, processed = find_kneedle_point(
            model_data,
            quality_key='quality',
            latency_key='latency',
            cost_key='cost',
            alpha=alpha,
            use_pareto_only=True,
        )
        
        # Convert kneedle distances to scores (higher distance = lower score = worse)
        # Invert so knee point has lowest score
        kneedle_scores = {}
        max_dist = max((m.get('kneedle_distance', 0) for m in processed), default=1)
        
        for m in processed:
            dist = m.get('kneedle_distance', 0)
            is_knee = m.get('is_knee', False)
            
            if is_knee:
                # Knee point gets best score
                kneedle_scores[m['name']] = 0.0
            elif max_dist > 0:
                # Others scored by inverse of their distance
                # Higher kneedle_distance = better (closer to knee behavior)
                # So we invert: low kneedle_distance = high score (bad)
                kneedle_scores[m['name']] = 1.0 - (dist / max_dist)
            else:
                kneedle_scores[m['name']] = 0.5
        
        return kneedle_scores

    def rank(
        self, 
        models: List[ModelMetadata], 
        decision: RoutingDecision, 
        top_k: int = 3, 
        return_detailed: bool = False,
        verbose: bool = True,
        # Constraint-aware ranking parameters
        min_quality_pct: Optional[float] = None,
        max_cost_pct: Optional[float] = None,
        max_latency_pct: Optional[float] = None,
    ) -> List[RecommendationResult]:
        """
        Rank models using multi-objective optimization.
        
        Uses Chebyshev scalarization for most strategies, or Knee point
        detection for KNEE strategy.
        
        Missing data handling depends on the missing_data strategy:
        - STRICT: Only models with complete data are included (default)
        - IMPUTE: Models with missing data use imputation values
        
        Constraint-aware ranking:
        When constraint parameters are provided (min_quality_pct, max_cost_pct, 
        max_latency_pct), models are sorted FIRST by how many constraints they 
        meet, THEN by their optimization score. This ensures models meeting your
        requirements appear at the top, even if other models have slightly better
        optimization scores but fail critical constraints.
        
        Args:
            models: List of candidate models
            decision: Routing decision (determines task-specific quality weights)
            top_k: Number of top models to return
            return_detailed: If True, return RankedModel objects instead
            verbose: Whether to print status messages
            min_quality_pct: Minimum quality as % of baseline (e.g., 80 for 80%)
            max_cost_pct: Maximum cost as % of baseline (e.g., 50 for 50%)
            max_latency_pct: Maximum latency as % of baseline (e.g., 100 for same as baseline)
            
        Returns:
            List of top-k RecommendationResult or RankedModel objects
        """
        # Step 1: Filter for pricing (always required)
        # Include free models (cost=0) - they are valid open source options
        valid_models = [
            m for m in models 
            if m.input_cost_per_m is not None 
            and m.output_cost_per_m is not None
            and m.input_cost_per_m >= 0
        ]
        
        if not valid_models:
            if verbose:
                print("⚠️ Warning: No models with complete pricing data found.")
            valid_models = models
        
        # Step 2: Apply missing data strategy
        if self.missing_data == MissingDataStrategy.STRICT:
            valid_models = self.filter_complete_models(valid_models, verbose=verbose)
            if not valid_models:
                if verbose:
                    print("⚠️ STRICT mode: No models have complete data for all objectives.")
                    print("   Consider using missing_data=MissingDataStrategy.IMPUTE")
                return []
        elif self.missing_data == MissingDataStrategy.IMPUTE and verbose:
            # Count models with imputed data
            imputed_count = sum(1 for m in valid_models if not self.has_complete_data(m))
            if imputed_count > 0:
                print(f"ℹ️  IMPUTE mode: {imputed_count} models using imputed values (lower confidence)")
        
        baseline_metrics = self._get_model_metrics(self.baseline, decision)
        population_stats = self._calculate_population_stats(valid_models, decision)
        
        if self.strategy == OptimizationStrategy.VALUE_OPTIMIZED:
            constrained_models = self._apply_constraints(
                valid_models, decision, baseline_metrics
            )
            if constrained_models:
                valid_models = constrained_models
        
        weights = self._get_weights()
        results = []
        
        # Pre-calculate scores based on strategy
        knee_scores = {}
        hybrid_scores = {}
        utopian_scores = {}
        kneedle_scores = {}
        utopian_chebyshev_scores = {}
        pareto_chebyshev_scores = {}
        pareto_chebyshev_removed = []
        pareto_models = valid_models  # Default to all models
        
        if self.strategy == OptimizationStrategy.HYBRID:
            # Calculate hybrid scores: Chebyshev with Pareto dominance as 5th dimension
            hybrid_scores = self._calculate_hybrid_scores(
                valid_models, decision, baseline_metrics, population_stats
            )
            if verbose:
                print(f"✓ HYBRID: 5D Chebyshev (Quality, Cost, Latency, Trust, Pareto Dominance) "
                      f"for {len(valid_models)} models")
            
        elif self.strategy == OptimizationStrategy.KNEE:
            knee_scores = self._calculate_knee_scores(
                valid_models, decision, baseline_metrics, population_stats
            )
            if verbose:
                print(f"✓ KNEE: Calculated knee point scores for {len(valid_models)} models")
        
        elif self.strategy == OptimizationStrategy.UTOPIAN:
            # Calculate utopian distance scores: Min-Max normalization + Euclidean distance
            # This handles $0 costs mathematically (they become 1.0 = optimal)
            utopian_scores = self._calculate_utopian_scores(
                valid_models, decision, population_stats
            )
            if verbose:
                print(f"✓ UTOPIAN: Distance to (1,1,1) with Min-Max normalization "
                      f"for {len(valid_models)} models")
        
        elif self.strategy == OptimizationStrategy.KNEEDLE:
            # Calculate kneedle scores: 2D projection + max curvature detection
            # Projects (Quality, Latency, Cost) to (Performance, log(Cost+ε))
            kneedle_scores = self._calculate_kneedle_scores(
                valid_models, decision, population_stats
            )
            if verbose:
                weights = self._get_weights()
                alpha = weights.get("quality", 0.6)
                print(f"✓ KNEEDLE: 2D projection (α={alpha:.1f}×Q + {1-alpha:.1f}×L vs log(Cost)) "
                      f"for {len(valid_models)} models")
        
        elif self.strategy == OptimizationStrategy.UTOPIAN_CHEBYSHEV:
            # Calculate 5D Augmented Chebyshev with Utopian normalization
            # Dimensions: Efficiency, Trust, Quality, Latency, Cost
            utopian_chebyshev_scores = self._calculate_utopian_chebyshev_scores(
                valid_models, decision
            )
            if verbose:
                print(f"✓ UTOPIAN_CHEBYSHEV: 5D Augmented Chebyshev (Efficiency, Trust, Quality, "
                      f"Latency, Cost) for {len(valid_models)} models")
        
        elif self.strategy == OptimizationStrategy.PARETO_CHEBYSHEV:
            # 3-Phase: Shadow Price → Pareto Filter → Chebyshev with Business Targets
            pareto_chebyshev_scores, pareto_chebyshev_removed = self._calculate_pareto_chebyshev_scores(
                valid_models, decision
            )
            if verbose:
                print(f"✓ PARETO_CHEBYSHEV: 3-Phase optimization")
                print(f"  Phase 1: Shadow pricing applied")
                print(f"  Phase 2: {len(pareto_chebyshev_removed)} dominated models removed")
                print(f"  Phase 3: {len(valid_models) - len(pareto_chebyshev_removed)} survivors ranked by regret")

        for m in valid_models:
            metrics = self._get_model_metrics(m, decision)
            norm = self._normalize_metrics(metrics, baseline_metrics, population_stats)
            regrets = {k: max(0, 1.0 - v) for k, v in norm.items()}
            
            if self.strategy == OptimizationStrategy.HYBRID:
                score = hybrid_scores.get(m.name, 0.5)
            elif self.strategy == OptimizationStrategy.KNEE:
                score = knee_scores.get(m.name, 0.5)
            elif self.strategy == OptimizationStrategy.UTOPIAN:
                score = utopian_scores.get(m.name, 0.5)
            elif self.strategy == OptimizationStrategy.KNEEDLE:
                score = kneedle_scores.get(m.name, 0.5)
            elif self.strategy == OptimizationStrategy.UTOPIAN_CHEBYSHEV:
                score = utopian_chebyshev_scores.get(m.name, 0.5)
            elif self.strategy == OptimizationStrategy.PARETO_CHEBYSHEV:
                score = pareto_chebyshev_scores.get(m.name, float('inf'))
            else:
                # Chebyshev distance: max weighted regret
                score = max(
                    weights.get(obj.name, 0.1) * regrets[obj.name]
                    for obj in self.objectives
                )

            summary = self._generate_summary(metrics, baseline_metrics)

            # Calculate constraint satisfaction info
            constraint_info = {}
            if any([min_quality_pct, max_cost_pct, max_latency_pct]):
                q_ratio = (metrics["quality"] / baseline_metrics["quality"] * 100) if baseline_metrics["quality"] > 0 else 0
                c_ratio = (metrics["cost"] / baseline_metrics["cost"] * 100) if baseline_metrics["cost"] > 0 else 100
                l_ratio = (metrics["latency"] / baseline_metrics["latency"] * 100) if baseline_metrics["latency"] > 0 else 100
                
                constraint_info = {
                    "quality_pct": round(q_ratio, 1),
                    "cost_pct": round(c_ratio, 1),
                    "latency_pct": round(l_ratio, 1),
                    "meets_quality": q_ratio >= min_quality_pct if min_quality_pct else True,
                    "meets_cost": c_ratio <= max_cost_pct if max_cost_pct else True,
                    "meets_latency": l_ratio <= max_latency_pct if max_latency_pct else True,
                }
                constraint_info["meets_all"] = all([
                    constraint_info["meets_quality"],
                    constraint_info["meets_cost"],
                    constraint_info["meets_latency"]
                ])

            if return_detailed:
                results.append(RankedModel(
                    name=m.name,
                    quality_score=metrics.get("quality", 0),
                    chebyshev_score=score,
                    tradeoff_summary=summary,
                    metadata=m
                ))
            else:
                results.append(RecommendationResult(
                    rank=0,
                    model_name=m.name,
                    score=score,
                    reasoning=summary,
                    cot_template=decision.cot_template or "",
                    optimization_metrics=constraint_info if constraint_info else None
                ))

        # Constraint-aware sorting: prioritize models meeting more constraints
        has_constraints = any([min_quality_pct, max_cost_pct, max_latency_pct])
        
        if has_constraints:
            # Calculate constraint satisfaction for sorting
            constraint_scores = {}
            for m in valid_models:
                metrics = self._get_model_metrics(m, decision)
                constraints_met = 0
                total_constraints = 0
                
                if min_quality_pct is not None:
                    total_constraints += 1
                    q_ratio = (metrics["quality"] / baseline_metrics["quality"] * 100) if baseline_metrics["quality"] > 0 else 0
                    if q_ratio >= min_quality_pct:
                        constraints_met += 1
                
                if max_cost_pct is not None:
                    total_constraints += 1
                    c_ratio = (metrics["cost"] / baseline_metrics["cost"] * 100) if baseline_metrics["cost"] > 0 else 100
                    if c_ratio <= max_cost_pct:
                        constraints_met += 1
                
                if max_latency_pct is not None:
                    total_constraints += 1
                    l_ratio = (metrics["latency"] / baseline_metrics["latency"] * 100) if baseline_metrics["latency"] > 0 else 100
                    if l_ratio <= max_latency_pct:
                        constraints_met += 1
                
                constraint_scores[m.name] = constraints_met
            
            # Sort by: (1) constraints met DESC, (2) optimization score ASC
            results.sort(key=lambda x: (
                -constraint_scores.get(x.name if return_detailed else x.model_name, 0),
                x.chebyshev_score if return_detailed else x.score
            ))
            
            if verbose:
                meets_all = sum(1 for v in constraint_scores.values() if v == total_constraints)
                print(f"✓ Constraint-aware ranking: {meets_all}/{len(valid_models)} models meet all constraints")
        else:
            # Standard sorting by optimization score only
            results.sort(key=lambda x: x.chebyshev_score if return_detailed else x.score)
        
        for i, res in enumerate(results):
            if not return_detailed:
                res.rank = i + 1
                
        return results[:top_k]

    def _apply_constraints(
        self, 
        models: List[ModelMetadata], 
        decision: RoutingDecision,
        baseline_metrics: Dict[str, float]
    ) -> List[ModelMetadata]:
        """Apply constraint filtering for VALUE_OPTIMIZED strategy."""
        # If no constraints specified, return all models
        if self.quality_range is None and self.cost_range is None and self.speed_range is None:
            return models
        
        constrained = []
        
        for m in models:
            metrics = self._get_model_metrics(m, decision)
            
            q_ratio = metrics["quality"] / baseline_metrics["quality"] if baseline_metrics["quality"] > 0 else 0
            c_ratio = metrics["cost"] / baseline_metrics["cost"] if baseline_metrics["cost"] > 0 else 1.0
            
            speed_ratio = baseline_metrics["latency"] / metrics["latency"] if metrics["latency"] > 0 else 1.0
            
            # Quality constraint is optional - None means no constraint
            passes_quality = True
            if self.quality_range is not None:
                passes_quality = self.quality_range[0] <= q_ratio <= self.quality_range[1]
            
            # Cost constraint is optional - None means no constraint
            passes_cost = True
            if self.cost_range is not None:
                passes_cost = self.cost_range[0] <= c_ratio <= self.cost_range[1]
            
            passes_speed = True
            if self.speed_range is not None:
                passes_speed = self.speed_range[0] <= speed_ratio <= self.speed_range[1]
            
            if passes_quality and passes_cost and passes_speed:
                constrained.append(m)
        
        if constrained:
            constraint_parts = []
            if self.quality_range is not None:
                q_min, q_max = self.quality_range
                constraint_parts.append(f"{q_min:.0%}-{q_max:.0%} quality")
            if self.cost_range is not None:
                c_min, c_max = self.cost_range
                constraint_parts.append(f"{c_min:.0%}-{c_max:.0%} cost")
            if self.speed_range is not None:
                s_min, _ = self.speed_range
                constraint_parts.append(f"≥{s_min:.0%} speed")
            if constraint_parts:
                print(f"✓ VALUE_OPTIMIZED: Filtered to {len(constrained)} models ({', '.join(constraint_parts)})")
        else:
            constraint_parts = []
            if self.quality_range is not None:
                constraint_parts.append(f"quality {self.quality_range}")
            if self.cost_range is not None:
                constraint_parts.append(f"cost {self.cost_range}")
            print(f"⚠️ No models found matching constraints: {', '.join(constraint_parts)}. Using all models.")
        
        return constrained

    def _generate_summary(
        self, 
        metrics: Dict[str, float], 
        baseline_metrics: Dict[str, float]
    ) -> str:
        """Generate human-readable summary of model vs baseline."""
        parts = []
        
        # Quality
        if "quality" in metrics:
            q_diff = metrics["quality"] - baseline_metrics["quality"]
            parts.append(f"Quality: {metrics['quality']:.1f} ({'+' if q_diff >= 0 else ''}{q_diff:.1f})")
        
        # Cost
        if "cost" in metrics:
            c_ratio = metrics["cost"] / baseline_metrics["cost"] if baseline_metrics["cost"] > 0 else 1.0
            c_diff = (1 - c_ratio) * 100
            parts.append(f"Cost: {abs(c_diff):.0f}% {'cheaper' if c_diff >= 0 else 'more expensive'}")
        
        # Latency
        if "latency" in metrics:
            lat_diff = baseline_metrics["latency"] - metrics["latency"]
            lat_pct = (lat_diff / baseline_metrics["latency"] * 100) if baseline_metrics["latency"] > 0 else 0
            parts.append(f"TTFT: {metrics['latency']*1000:.0f}ms ({'+' if lat_pct >= 0 else ''}{lat_pct:.0f}%)")
        
        # Hallucination
        if "hallucination" in metrics:
            hall_diff = baseline_metrics["hallucination"] - metrics["hallucination"]
            parts.append(f"Halluc: {metrics['hallucination']:.1f}% ({'+' if hall_diff >= 0 else ''}{hall_diff:.1f})")
        
        # Any additional objectives
        known = {"quality", "cost", "latency", "hallucination", "refusal"}
        for obj in self.objectives:
            if obj.name not in known and obj.name in metrics:
                diff = metrics[obj.name] - baseline_metrics[obj.name]
                if obj.direction == "minimize":
                    diff = -diff
                parts.append(f"{obj.display_name}: {metrics[obj.name]:.1f} ({'+' if diff >= 0 else ''}{diff:.1f})")
        
        return " | ".join(parts)

    def _model_to_dict(self, m: ModelMetadata) -> Dict:
        """Convert ModelMetadata to dict format expected by QualityScorer."""
        return {
            'name': m.name,
            'intelligence_index': getattr(m, 'intelligence_index', None),
            'coding_index': getattr(m, 'coding_index', None),
            'math_index': getattr(m, 'math_index', None),
            'mmlu_pro': getattr(m, 'mmlu_pro', None),
            'gpqa': getattr(m, 'gpqa', None),
            'hle': getattr(m, 'hle', None),
            'livecodebench': getattr(m, 'livecodebench', None),
            'scicode': getattr(m, 'scicode', None),
            'math_500': getattr(m, 'math_500', None),
            'aime': getattr(m, 'aime', None),
        }
