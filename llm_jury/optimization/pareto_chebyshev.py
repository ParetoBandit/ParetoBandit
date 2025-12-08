"""
Three-Phase Pareto-Chebyshev Optimization Algorithm.

This module implements a rigorous three-phase approach to multi-objective
model selection that combines Pareto filtering with Chebyshev ranking.

Phase 1: Shadow Price Fix (Data Engineering)
============================================
There is no such thing as $0. All costs are either CapEx or OpEx.

For SaaS models: Cost = API Price per 1M tokens
For Open Source: Cost = (Hourly GPU Cost / Tokens per Hour) + ε

The ε constant (1e-6) ensures we never hit literal zero.

Phase 2: Pareto Filter (Efficiency Gate)
========================================
Before ranking, we REMOVE dominated models. A model is dominated if
another model exists that is better in EVERY dimension.

This gives confidence: "We deleted options that were objectively worse."

Result: Non-Dominated Set (Pareto Frontier survivors)

Phase 3: Augmented Chebyshev (The Ranker)
=========================================
Rank survivors using BUSINESS TARGETS (not utopia).

Regret Formula:
    R(x) = max[w_c·(C(x)-C*), w_l·(L(x)-L*), w_q·(Q*-Q(x)), w_t·(T*-T(x))] + α·Σdev

Where:
- C*, L*, Q*, T* = Business Targets (e.g., Cost=$0.50, Latency=200ms)
- w = Strictness weights (high weight = severe punishment for missing target)
- α = Tie-breaker constant (default: 0.001)

The model with MINIMUM regret wins.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum


# =============================================================================
# PHASE 1: Shadow Price Fix
# =============================================================================

# Tiny epsilon to prevent literal zeros
EPSILON = 1e-6

# Default minimum cost for free models (will be updated from population)
# This is the fallback if no paid models exist in the population
DEFAULT_MIN_COST = 0.05  # $0.05/M tokens


# Global cache for minimum cost from population
_min_cost_cache: Optional[float] = None


def set_minimum_cost_from_population(models_data: List[Dict[str, Any]], discount: float = 0.9) -> float:
    """
    Calculate and cache the minimum cost from paid models in the population.
    
    Free models will use a slightly discounted version of this price as their 
    shadow price (default 90% of cheapest paid model).
    
    Args:
        models_data: List of model data dictionaries
        discount: Multiplier for shadow price (default 0.9 = 10% cheaper than cheapest paid)
        
    Returns:
        The shadow price for free models (slightly below cheapest paid model)
    """
    global _min_cost_cache
    
    paid_costs = []
    for m in models_data:
        input_cost = m.get('input_cost_per_m') or m.get('price_1m_input') or 0
        output_cost = m.get('output_cost_per_m') or m.get('price_1m_output') or 0
        blended = input_cost * 0.75 + output_cost * 0.25
        if blended > EPSILON:
            paid_costs.append(blended)
    
    if paid_costs:
        # Shadow price = slightly cheaper than the cheapest paid model
        _min_cost_cache = min(paid_costs) * discount
    else:
        _min_cost_cache = DEFAULT_MIN_COST
    
    return _min_cost_cache


def get_minimum_cost() -> float:
    """Get the cached minimum cost, or default if not set."""
    global _min_cost_cache
    return _min_cost_cache if _min_cost_cache is not None else DEFAULT_MIN_COST


def calculate_effective_cost(
    input_cost_per_m: Optional[float],
    output_cost_per_m: Optional[float],
    param_count_b: Optional[float] = None,
    model_name: Optional[str] = None,
    input_ratio: float = 0.75,
    output_ratio: float = 0.25,
) -> float:
    """
    Phase 1: Calculate Effective Cost with Shadow Pricing.
    
    There is no such thing as $0. Free models get the price of the 
    CHEAPEST PAID MODEL as their shadow price.
    
    Formula:
        Cost(m) = API_Price                    if SaaS (cost > 0)
                = min(paid_model_costs) + ε   if Open Source (cost = 0)
    
    Args:
        input_cost_per_m: API input cost per 1M tokens
        output_cost_per_m: API output cost per 1M tokens
        param_count_b: Parameter count in billions (unused, kept for compatibility)
        model_name: Model name (unused, kept for compatibility)
        input_ratio: Proportion of input tokens (default: 75%)
        output_ratio: Proportion of output tokens (default: 25%)
        
    Returns:
        Effective cost per 1M blended tokens (always > 0)
    """
    input_cost = input_cost_per_m if input_cost_per_m is not None else 0.0
    output_cost = output_cost_per_m if output_cost_per_m is not None else 0.0
    
    # Calculate blended API cost
    api_cost = input_cost * input_ratio + output_cost * output_ratio
    
    # If API cost is effectively zero, use the cheapest paid model's price
    if api_cost < EPSILON:
        return get_minimum_cost() + EPSILON
    
    return api_cost + EPSILON  # Always add epsilon


# =============================================================================
# PHASE 2: Pareto Filter (Efficiency Gate)
# =============================================================================

@dataclass
class ModelMetrics:
    """Metrics for a single model used in Pareto comparison."""
    name: str
    cost: float      # Effective cost (Phase 1 output)
    latency: float   # Seconds (lower is better)
    quality: float   # Score (higher is better)
    trust: float     # Score (higher is better)
    
    # Original data reference
    raw_data: Dict[str, Any] = field(default_factory=dict)


def is_dominated(model_a: ModelMetrics, model_b: ModelMetrics) -> bool:
    """
    Check if model_a is dominated by model_b.
    
    A model is dominated if another model is:
    - Better OR equal in ALL dimensions
    - Strictly better in AT LEAST ONE dimension
    
    Args:
        model_a: Model to check for domination
        model_b: Potential dominator
        
    Returns:
        True if model_a is dominated by model_b
    """
    # For "minimize" metrics (cost, latency): b <= a is better or equal
    # For "maximize" metrics (quality, trust): b >= a is better or equal
    
    cost_ok = model_b.cost <= model_a.cost
    latency_ok = model_b.latency <= model_a.latency
    quality_ok = model_b.quality >= model_a.quality
    trust_ok = model_b.trust >= model_a.trust
    
    # Must be at least as good in ALL dimensions
    all_at_least_as_good = cost_ok and latency_ok and quality_ok and trust_ok
    
    if not all_at_least_as_good:
        return False
    
    # Must be strictly better in AT LEAST ONE dimension
    cost_better = model_b.cost < model_a.cost
    latency_better = model_b.latency < model_a.latency
    quality_better = model_b.quality > model_a.quality
    trust_better = model_b.trust > model_a.trust
    
    strictly_better_in_one = cost_better or latency_better or quality_better or trust_better
    
    return strictly_better_in_one


def pareto_filter(models: List[ModelMetrics]) -> Tuple[List[ModelMetrics], List[str]]:
    """
    Phase 2: Remove dominated models (Efficiency Gate).
    
    A model is removed if another model exists that is better in EVERY way.
    
    This gives confidence: "We deleted options that were objectively worse."
    
    Args:
        models: List of ModelMetrics to filter
        
    Returns:
        Tuple of (survivors, removed_names)
    """
    if not models:
        return [], []
    
    n = len(models)
    dominated = [False] * n
    
    # Compare every model against every other model
    for i in range(n):
        if dominated[i]:
            continue  # Already marked as dominated
        
        for j in range(n):
            if i == j or dominated[j]:
                continue
            
            # Check if model i is dominated by model j
            if is_dominated(models[i], models[j]):
                dominated[i] = True
                break
    
    survivors = [m for i, m in enumerate(models) if not dominated[i]]
    removed = [m.name for i, m in enumerate(models) if dominated[i]]
    
    return survivors, removed


# =============================================================================
# PHASE 3: Augmented Chebyshev Ranker
# =============================================================================

@dataclass
class BusinessTargets:
    """
    Business targets for Chebyshev optimization.
    
    These are YOUR targets, not the theoretical utopia.
    
    For quality: Use the LEAD MODEL's score as the target.
    This asks: "How close can you get to the best while meeting cost/latency constraints?"
    
    Example: Target Cost = $0.50/M tokens, Target Latency = 200ms, Quality = lead model
    """
    cost: float = 0.50        # Target cost per 1M tokens
    latency: float = 0.200    # Target latency in seconds (200ms)
    quality: float = 70.0     # Target quality score (0-100) - use lead model's score
    trust: float = 85.0       # Target trust score (0-100, 85 = 15% hallucination)
    
    @classmethod
    def from_population(
        cls,
        quality_scores: Dict[str, float],
        costs: List[float],
        cost_percentile: float = 25,
        latency: float = 0.300,
        trust: float = 85.0,
    ) -> 'BusinessTargets':
        """
        Create targets from population statistics.
        
        - Quality: Uses the LEAD model's score (max quality)
        - Cost: Uses the specified percentile (default 25th = achievable)
        - Latency/Trust: User-specified defaults
        
        Args:
            quality_scores: Dict of model name → quality score
            costs: List of effective costs (TCI)
            cost_percentile: Percentile for cost target (default 25 = cheap)
            latency: Target latency in seconds
            trust: Target trust score
            
        Returns:
            BusinessTargets with lead model quality as target
        """
        import numpy as np
        
        lead_quality = max(quality_scores.values()) if quality_scores else 70.0
        target_cost = np.percentile(costs, cost_percentile) if costs else 0.50
        
        return cls(
            cost=max(0.10, target_cost),  # Minimum $0.10
            latency=latency,
            quality=lead_quality,
            trust=trust,
        )


@dataclass
class StrictnessWeights:
    """
    Strictness weights for Chebyshev optimization.
    
    High weight = "I will punish you severely if you miss this target"
    """
    cost: float = 0.25
    latency: float = 0.20
    quality: float = 0.35
    trust: float = 0.20
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.cost + self.latency + self.quality + self.trust
        if total > 0:
            self.cost /= total
            self.latency /= total
            self.quality /= total
            self.trust /= total


# Tie-breaker constant
ALPHA = 0.001


@dataclass
class ChebyshevResult:
    """Result of Chebyshev ranking for a single model."""
    name: str
    
    # Raw metrics
    cost: float
    latency: float
    quality: float
    trust: float
    
    # Deviations from targets
    cost_deviation: float      # Positive = over target (bad)
    latency_deviation: float   # Positive = over target (bad)
    quality_deviation: float   # Positive = under target (bad)
    trust_deviation: float     # Positive = under target (bad)
    
    # Weighted deviations
    cost_weighted: float
    latency_weighted: float
    quality_weighted: float
    trust_weighted: float
    
    # Final scores
    max_weighted_deviation: float  # The Chebyshev component
    sum_deviations: float          # For tie-breaking
    regret: float                  # Final score (lower = better)
    
    # Which metric has the worst deviation
    bottleneck: str


def calculate_regret(
    model: ModelMetrics,
    targets: BusinessTargets,
    weights: StrictnessWeights,
    population_stats: Dict[str, Tuple[float, float]],
) -> ChebyshevResult:
    """
    Phase 3: Calculate Augmented Chebyshev Regret.
    
    Regret Formula:
        R(x) = max[w_c·dev_c, w_l·dev_l, w_q·dev_q, w_t·dev_t] + α·Σdev
    
    Deviations are normalized and measured from BUSINESS TARGETS:
    - Cost: (actual - target) / range  [positive if over budget]
    - Latency: (actual - target) / range  [positive if too slow]
    - Quality: (target - actual) / range  [positive if under target]
    - Trust: (target - actual) / range  [positive if under target]
    
    Args:
        model: Model metrics
        targets: Business targets
        weights: Strictness weights
        population_stats: Min/max for each metric in the population
        
    Returns:
        ChebyshevResult with all components
    """
    # Get population ranges for normalization
    cost_min, cost_max = population_stats['cost']
    lat_min, lat_max = population_stats['latency']
    qual_min, qual_max = population_stats['quality']
    trust_min, trust_max = population_stats['trust']
    
    # Safe range calculation (avoid division by zero)
    cost_range = max(cost_max - cost_min, EPSILON)
    lat_range = max(lat_max - lat_min, EPSILON)
    qual_range = max(qual_max - qual_min, EPSILON)
    trust_range = max(trust_max - trust_min, EPSILON)
    
    # Calculate normalized deviations from targets
    # Positive deviation = bad (missing the target)
    
    # Cost: (actual - target) / range  [over budget is bad]
    cost_dev = (model.cost - targets.cost) / cost_range
    
    # Latency: (actual - target) / range  [too slow is bad]
    latency_dev = (model.latency - targets.latency) / lat_range
    
    # Quality: (target - actual) / range  [under target is bad]
    quality_dev = (targets.quality - model.quality) / qual_range
    
    # Trust: (target - actual) / range  [under target is bad]
    trust_dev = (targets.trust - model.trust) / trust_range
    
    # Apply strictness weights
    cost_weighted = weights.cost * cost_dev
    latency_weighted = weights.latency * latency_dev
    quality_weighted = weights.quality * quality_dev
    trust_weighted = weights.trust * trust_dev
    
    # Find max weighted deviation (Chebyshev component)
    weighted_devs = {
        'cost': cost_weighted,
        'latency': latency_weighted,
        'quality': quality_weighted,
        'trust': trust_weighted,
    }
    max_weighted = max(weighted_devs.values())
    bottleneck = max(weighted_devs.keys(), key=lambda k: weighted_devs[k])
    
    # Sum of deviations for tie-breaking
    sum_devs = cost_dev + latency_dev + quality_dev + trust_dev
    
    # Augmented Chebyshev regret
    regret = max_weighted + ALPHA * sum_devs
    
    return ChebyshevResult(
        name=model.name,
        cost=model.cost,
        latency=model.latency,
        quality=model.quality,
        trust=model.trust,
        cost_deviation=cost_dev,
        latency_deviation=latency_dev,
        quality_deviation=quality_dev,
        trust_deviation=trust_dev,
        cost_weighted=cost_weighted,
        latency_weighted=latency_weighted,
        quality_weighted=quality_weighted,
        trust_weighted=trust_weighted,
        max_weighted_deviation=max_weighted,
        sum_deviations=sum_devs,
        regret=regret,
        bottleneck=bottleneck,
    )


# =============================================================================
# INTEGRATED THREE-PHASE OPTIMIZER
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the three-phase optimizer."""
    targets: BusinessTargets = field(default_factory=BusinessTargets)
    weights: StrictnessWeights = field(default_factory=StrictnessWeights)
    apply_pareto_filter: bool = True  # Set False to skip Phase 2


class ParetoChebyshevOptimizer:
    """
    Three-Phase Pareto-Chebyshev Optimizer.
    
    Phase 1: Shadow Price Fix (no $0 costs)
    Phase 2: Pareto Filter (remove dominated models)
    Phase 3: Augmented Chebyshev (rank by regret from business targets)
    
    Example:
        >>> config = OptimizationConfig(
        ...     targets=BusinessTargets(cost=0.50, latency=0.200, quality=70, trust=85),
        ...     weights=StrictnessWeights(cost=0.25, latency=0.20, quality=0.35, trust=0.20),
        ... )
        >>> optimizer = ParetoChebyshevOptimizer(config)
        >>> results = optimizer.optimize(models, quality_scores)
        >>> print(f"Winner: {results.winner.name}")
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.config.weights.normalize()
    
    def _extract_metrics(
        self,
        model_data: Dict[str, Any],
        quality_score: float,
    ) -> ModelMetrics:
        """Extract metrics from model data."""
        
        # Phase 1: Effective Cost (Shadow Pricing)
        cost = calculate_effective_cost(
            input_cost_per_m=model_data.get('input_cost_per_m') or model_data.get('price_1m_input'),
            output_cost_per_m=model_data.get('output_cost_per_m') or model_data.get('price_1m_output'),
            param_count_b=model_data.get('param_count_b'),
            model_name=model_data.get('name'),
        )
        
        # Latency (in seconds)
        latency = (
            model_data.get('measured_ttft_seconds') or
            model_data.get('time_to_first_token_seconds') or
            (model_data.get('median_latency_ms', 1000) / 1000.0)
        )
        
        # Trust score (from hallucination/refusal rates)
        halluc = model_data.get('hallucination_rate', 15.0)
        refusal = model_data.get('refusal_rate', 5.0)
        trust = 100.0 - (halluc * 0.7 + refusal * 0.3)  # Higher = more trustworthy
        
        return ModelMetrics(
            name=model_data.get('name', 'Unknown'),
            cost=cost,
            latency=latency,
            quality=quality_score,
            trust=trust,
            raw_data=model_data,
        )
    
    def _calculate_population_stats(
        self,
        models: List[ModelMetrics]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate min/max for each metric."""
        return {
            'cost': (min(m.cost for m in models), max(m.cost for m in models)),
            'latency': (min(m.latency for m in models), max(m.latency for m in models)),
            'quality': (min(m.quality for m in models), max(m.quality for m in models)),
            'trust': (min(m.trust for m in models), max(m.trust for m in models)),
        }
    
    def optimize(
        self,
        models_data: List[Dict[str, Any]],
        quality_scores: Dict[str, float],
        top_k: Optional[int] = None,
        verbose: bool = False,
    ) -> 'OptimizationResult':
        """
        Run three-phase optimization.
        
        Args:
            models_data: List of model data dictionaries
            quality_scores: Dict mapping model names to quality scores
            top_k: Return only top K models (optional)
            verbose: Print progress messages
            
        Returns:
            OptimizationResult with ranked models
        """
        if verbose:
            print("=" * 70)
            print("THREE-PHASE PARETO-CHEBYSHEV OPTIMIZATION")
            print("=" * 70)
        
        # Set minimum cost from population (for shadow pricing free models)
        min_cost = set_minimum_cost_from_population(models_data)
        
        # Phase 1: Extract metrics with shadow pricing
        if verbose:
            print(f"\n[Phase 1] Shadow Price Fix: Free models get ${min_cost:.2f}/M (cheapest paid model)...")
        
        all_metrics = []
        for data in models_data:
            name = data.get('name', 'Unknown')
            quality = quality_scores.get(name, 50.0)
            metrics = self._extract_metrics(data, quality)
            all_metrics.append(metrics)
        
        if verbose:
            free_count = sum(1 for m in models_data 
                           if (m.get('input_cost_per_m') or 0) == 0 and 
                              (m.get('output_cost_per_m') or 0) == 0)
            print(f"  → {len(all_metrics)} models processed, {free_count} with shadow pricing applied")
        
        # Phase 2: Pareto Filter
        if self.config.apply_pareto_filter:
            if verbose:
                print("\n[Phase 2] Pareto Filter: Removing dominated models...")
            
            survivors, removed = pareto_filter(all_metrics)
            
            if verbose:
                print(f"  → {len(removed)} dominated models removed")
                print(f"  → {len(survivors)} Pareto-efficient models remain")
        else:
            survivors = all_metrics
            removed = []
            if verbose:
                print("\n[Phase 2] Pareto Filter: SKIPPED")
        
        # Phase 3: Augmented Chebyshev Ranking
        if verbose:
            print("\n[Phase 3] Augmented Chebyshev: Ranking by regret from business targets...")
            print(f"  Targets: Cost=${self.config.targets.cost:.2f}, "
                  f"Latency={self.config.targets.latency*1000:.0f}ms, "
                  f"Quality={self.config.targets.quality:.0f}, "
                  f"Trust={self.config.targets.trust:.0f}")
        
        # Calculate population stats for normalization
        pop_stats = self._calculate_population_stats(survivors)
        
        # Calculate regret for each survivor
        results = []
        for m in survivors:
            result = calculate_regret(m, self.config.targets, self.config.weights, pop_stats)
            results.append(result)
        
        # Sort by regret (lowest = best)
        results.sort(key=lambda x: x.regret)
        
        if verbose:
            print(f"  → Ranked {len(results)} models by regret")
            if results:
                print(f"\n  Winner: {results[0].name} (regret={results[0].regret:.4f})")
        
        if top_k:
            results = results[:top_k]
        
        return OptimizationResult(
            ranked=results,
            pareto_survivors=survivors,
            dominated_removed=removed,
            config=self.config,
        )


@dataclass
class OptimizationResult:
    """Result of three-phase optimization."""
    ranked: List[ChebyshevResult]          # Ranked by regret (best first)
    pareto_survivors: List[ModelMetrics]   # Models that survived Pareto filter
    dominated_removed: List[str]           # Names of removed (dominated) models
    config: OptimizationConfig
    
    @property
    def winner(self) -> Optional[ChebyshevResult]:
        """Get the winning model (lowest regret)."""
        return self.ranked[0] if self.ranked else None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "OPTIMIZATION SUMMARY",
            "=" * 60,
            f"Total candidates: {len(self.pareto_survivors) + len(self.dominated_removed)}",
            f"Pareto-efficient: {len(self.pareto_survivors)}",
            f"Dominated (removed): {len(self.dominated_removed)}",
            "",
            "Business Targets:",
            f"  Cost: ${self.config.targets.cost:.2f}/M tokens",
            f"  Latency: {self.config.targets.latency*1000:.0f}ms",
            f"  Quality: {self.config.targets.quality:.0f}",
            f"  Trust: {self.config.targets.trust:.0f}",
            "",
        ]
        
        if self.ranked:
            lines.append("Top 5 Recommendations:")
            for i, r in enumerate(self.ranked[:5], 1):
                lines.append(f"  {i}. {r.name}")
                lines.append(f"     Regret: {r.regret:.4f} (bottleneck: {r.bottleneck})")
                lines.append(f"     Cost: ${r.cost:.2f}, Latency: {r.latency*1000:.0f}ms, "
                           f"Quality: {r.quality:.1f}, Trust: {r.trust:.1f}")
        
        return "\n".join(lines)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_preset_configs() -> Dict[str, OptimizationConfig]:
    """Create preset configurations for common use cases."""
    return {
        # Balanced optimization
        "balanced": OptimizationConfig(
            targets=BusinessTargets(cost=0.50, latency=0.300, quality=70.0, trust=85.0),
            weights=StrictnessWeights(cost=0.25, latency=0.20, quality=0.35, trust=0.20),
        ),
        # Enterprise: quality and trust are critical
        "enterprise": OptimizationConfig(
            targets=BusinessTargets(cost=2.00, latency=0.500, quality=80.0, trust=90.0),
            weights=StrictnessWeights(cost=0.15, latency=0.15, quality=0.40, trust=0.30),
        ),
        # Startup: cost-conscious, speed matters
        "startup": OptimizationConfig(
            targets=BusinessTargets(cost=0.20, latency=0.200, quality=65.0, trust=80.0),
            weights=StrictnessWeights(cost=0.35, latency=0.25, quality=0.25, trust=0.15),
        ),
        # Real-time: latency is critical
        "realtime": OptimizationConfig(
            targets=BusinessTargets(cost=1.00, latency=0.100, quality=60.0, trust=80.0),
            weights=StrictnessWeights(cost=0.20, latency=0.45, quality=0.20, trust=0.15),
        ),
        # Medical/Legal: trust is non-negotiable
        "high_stakes": OptimizationConfig(
            targets=BusinessTargets(cost=5.00, latency=1.000, quality=75.0, trust=95.0),
            weights=StrictnessWeights(cost=0.10, latency=0.10, quality=0.30, trust=0.50),
        ),
    }


PRESET_CONFIGS = create_preset_configs()

