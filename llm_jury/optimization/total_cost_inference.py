"""
Total Cost of Inference (TCI) Calculator and Utopian Distance Method.

This module provides two complementary approaches for handling the $0 pricing
problem in Pareto optimization:

APPROACH 1: Total Cost of Inference (TCI) - Data Engineering Fix
================================================================
Assigns realistic compute costs to open-source/self-hosted models:
- Managed Models (API): TCI = API Price per 1M tokens
- Self-Hosted (Open Source): TCI = (Hourly GPU Cost / Tokens per Hour)

Use when: You want realistic cost comparison including infrastructure costs.

APPROACH 2: Utopian Distance Method - Mathematical Fix
======================================================
Uses Min-Max normalization where $0 cost naturally becomes optimal (1.0).
No shadow pricing needed - the math handles zeros gracefully.

Formula:
    D = sqrt(w_q*(1-Q_norm)² + w_l*(1-L_norm)² + w_c*(1-C_norm)²)

Where:
    - For "minimize" metrics (cost, latency): x_norm = (x_max - x) / (x_max - x_min)
    - For "maximize" metrics (quality): x_norm = (x - x_min) / (x_max - x_min)
    - Utopia point = (1.0, 1.0, 1.0) representing perfect scores

The model with MINIMUM distance D is the optimal "knee" point.

Use when: You want mathematical robustness without imputing costs.

Reference GPU Costs (as of 2024):
    - A100 80GB: ~$2.00/hour (AWS/GCP spot)
    - H100 80GB: ~$4.00/hour (AWS/GCP spot)
    - A10G 24GB: ~$1.00/hour (good for 7B-13B models)
    - T4 16GB: ~$0.35/hour (good for <7B models)

Reference Throughput (tokens/second):
    - 70B model on A100: ~30-50 tokens/sec output
    - 7B model on A10G: ~100-200 tokens/sec output
    - 405B model on 8xH100: ~20-30 tokens/sec output
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ComputeProfile:
    """Compute resource profile for a model size class."""
    gpu_type: str
    hourly_cost_usd: float
    tokens_per_hour_output: float
    tokens_per_hour_input: float  # Input is typically 5-10x faster than output
    description: str


# Compute profiles by model size (parameter count in billions)
# Based on optimized inference (vLLM, TensorRT-LLM) with spot/reserved pricing
COMPUTE_PROFILES: Dict[str, ComputeProfile] = {
    # Small models (<10B) - A10G or L4
    # Optimized: vLLM on spot instances achieves ~200 tok/sec output
    "small": ComputeProfile(
        gpu_type="A10G/L4",
        hourly_cost_usd=0.50,             # Spot pricing ~$0.30-0.70/hr
        tokens_per_hour_output=720_000,   # ~200 tok/sec (vLLM optimized)
        tokens_per_hour_input=3_600_000,  # ~1000 tok/sec
        description="Small models (<10B params)"
    ),
    # Medium models (10B-40B) - A100 40GB
    # Optimized: 70-100 tok/sec output typical
    "medium": ComputeProfile(
        gpu_type="A100-40GB",
        hourly_cost_usd=1.20,             # Spot pricing ~$1.00-1.50/hr
        tokens_per_hour_output=300_000,   # ~83 tok/sec
        tokens_per_hour_input=1_200_000,  # ~333 tok/sec
        description="Medium models (10B-40B params)"
    ),
    # Large models (40B-100B) - A100 80GB or 2xA100
    # Optimized: 40-60 tok/sec output
    "large": ComputeProfile(
        gpu_type="A100-80GB",
        hourly_cost_usd=1.80,             # Spot pricing ~$1.50-2.00/hr
        tokens_per_hour_output=180_000,   # ~50 tok/sec
        tokens_per_hour_input=720_000,    # ~200 tok/sec
        description="Large models (40B-100B params)"
    ),
    # Very large models (100B+) - Multi-GPU H100/A100
    # Typically run on inference providers with batching
    "xlarge": ComputeProfile(
        gpu_type="4xA100/2xH100",
        hourly_cost_usd=8.00,             # Multi-GPU spot ~$6-10/hr
        tokens_per_hour_output=120_000,   # ~33 tok/sec (with batching)
        tokens_per_hour_input=480_000,    # ~133 tok/sec
        description="Very large models (100B+ params)"
    ),
}

# Minimum cost floor to enable logarithmic scaling and prevent divide-by-zero
MIN_COST_FLOOR = 0.01  # $0.01 per 1M tokens minimum

# Global cache for minimum cost from population (for shadow pricing)
_tci_min_cost_cache: Optional[float] = None


def set_tci_minimum_cost(models_data: List[Dict], discount: float = 0.9) -> float:
    """
    Set the minimum cost from paid models for TCI shadow pricing.
    
    Free models will use a slightly discounted version of the cheapest paid 
    model's price as their shadow price (default 90% = 10% cheaper).
    
    Args:
        models_data: List of model data dictionaries
        discount: Multiplier for shadow price (default 0.9 = 10% cheaper than cheapest paid)
        
    Returns:
        The shadow price for free models (slightly below cheapest paid model)
    """
    global _tci_min_cost_cache
    
    paid_costs = []
    for m in models_data:
        input_cost = m.get('input_cost_per_m') or m.get('price_1m_input') or 0
        output_cost = m.get('output_cost_per_m') or m.get('price_1m_output') or 0
        blended = input_cost * 0.75 + output_cost * 0.25
        if blended > MIN_COST_FLOOR:
            paid_costs.append(blended)
    
    if paid_costs:
        # Shadow price = slightly cheaper than the cheapest paid model
        _tci_min_cost_cache = min(paid_costs) * discount
    else:
        _tci_min_cost_cache = MIN_COST_FLOOR
    
    return _tci_min_cost_cache


def get_tci_minimum_cost() -> Optional[float]:
    """Get the cached minimum cost, or None if not set."""
    global _tci_min_cost_cache
    return _tci_min_cost_cache


def estimate_param_count(model_name: str) -> Optional[float]:
    """
    Estimate parameter count from model name.
    
    Args:
        model_name: Model name (e.g., "Llama-3.1-70B", "Qwen-7B-Instruct")
        
    Returns:
        Estimated parameter count in billions, or None if unknown
    """
    name_lower = model_name.lower()
    
    # Look for explicit parameter counts (e.g., "70b", "7b", "405b")
    match = re.search(r'(\d+\.?\d*)b', name_lower)
    if match:
        return float(match.group(1))
    
    # Look for parameter counts in different formats
    match = re.search(r'(\d+)x(\d+)b', name_lower)  # MoE like "8x7b"
    if match:
        experts = int(match.group(1))
        per_expert = float(match.group(2))
        return experts * per_expert  # Rough approximation
    
    # Known model size heuristics
    if 'gpt-4' in name_lower and 'mini' not in name_lower:
        return 1800.0  # Estimated
    if 'claude-3' in name_lower and 'opus' in name_lower:
        return 175.0  # Estimated
    if 'gemini' in name_lower and 'pro' in name_lower:
        return 175.0  # Estimated
    if 'mini' in name_lower or 'small' in name_lower or 'tiny' in name_lower:
        return 7.0
    
    return None


def get_compute_profile(param_count_b: Optional[float]) -> ComputeProfile:
    """
    Get the appropriate compute profile for a model size.
    
    Args:
        param_count_b: Parameter count in billions
        
    Returns:
        ComputeProfile for the model size class
    """
    if param_count_b is None:
        return COMPUTE_PROFILES["medium"]  # Default assumption
    
    if param_count_b < 10:
        return COMPUTE_PROFILES["small"]
    elif param_count_b < 40:
        return COMPUTE_PROFILES["medium"]
    elif param_count_b < 100:
        return COMPUTE_PROFILES["large"]
    else:
        return COMPUTE_PROFILES["xlarge"]


def calculate_shadow_price(
    param_count_b: Optional[float] = None,
    model_name: Optional[str] = None,
    custom_hourly_cost: Optional[float] = None,
    custom_throughput_output: Optional[float] = None,
    custom_throughput_input: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate shadow price for a self-hosted model.
    
    Shadow price represents the compute cost of running the model,
    converted to per-1M-token pricing for comparison with API models.
    
    Formula:
        Cost per 1M tokens = (Hourly GPU Cost / Tokens per Hour) * 1,000,000
    
    Args:
        param_count_b: Parameter count in billions (optional if model_name provided)
        model_name: Model name for parameter estimation (optional)
        custom_hourly_cost: Override default hourly cost
        custom_throughput_output: Override default output throughput (tokens/hour)
        custom_throughput_input: Override default input throughput (tokens/hour)
        
    Returns:
        Tuple of (input_cost_per_m, output_cost_per_m) in USD
        
    Example:
        >>> calculate_shadow_price(param_count_b=70)
        (0.42, 1.67)  # Large model: ~$0.42/M input, ~$1.67/M output
        
        >>> calculate_shadow_price(model_name="Llama-3.1-8B")
        (0.032, 0.16)  # Small model: ~$0.03/M input, ~$0.16/M output
    """
    # Determine parameter count
    if param_count_b is None and model_name:
        param_count_b = estimate_param_count(model_name)
    
    # Get compute profile
    profile = get_compute_profile(param_count_b)
    
    # Apply custom overrides
    hourly_cost = custom_hourly_cost or profile.hourly_cost_usd
    throughput_output = custom_throughput_output or profile.tokens_per_hour_output
    throughput_input = custom_throughput_input or profile.tokens_per_hour_input
    
    # Calculate cost per 1M tokens
    # Formula: (hourly_cost / tokens_per_hour) * 1,000,000
    input_cost_per_m = (hourly_cost / throughput_input) * 1_000_000
    output_cost_per_m = (hourly_cost / throughput_output) * 1_000_000
    
    return (
        max(MIN_COST_FLOOR, round(input_cost_per_m, 4)),
        max(MIN_COST_FLOOR, round(output_cost_per_m, 4))
    )


def calculate_tci(
    input_cost_per_m: Optional[float],
    output_cost_per_m: Optional[float],
    param_count_b: Optional[float] = None,
    model_name: Optional[str] = None,
    input_ratio: float = 0.75,
    output_ratio: float = 0.25,
) -> float:
    """
    Calculate Total Cost of Inference (TCI) for a model.
    
    TCI is a unified cost metric that:
    1. Uses API pricing for managed/commercial models
    2. Applies shadow pricing for self-hosted models with $0 cost
    3. Ensures a minimum cost floor for mathematical stability
    
    Args:
        input_cost_per_m: API input cost per 1M tokens (can be 0 or None)
        output_cost_per_m: API output cost per 1M tokens (can be 0 or None)
        param_count_b: Parameter count for shadow pricing (optional)
        model_name: Model name for parameter estimation (optional)
        input_ratio: Proportion of tokens that are input (default: 75%)
        output_ratio: Proportion of tokens that are output (default: 25%)
        
    Returns:
        Total Cost of Inference per 1M blended tokens
        
    Example:
        # Commercial model with API pricing
        >>> calculate_tci(3.0, 15.0)
        6.0  # Blended: 3.0 * 0.75 + 15.0 * 0.25 = 6.0
        
        # Open source model with $0 API cost
        >>> calculate_tci(0.0, 0.0, model_name="Llama-3.1-70B")
        0.54  # Shadow price: ~$0.54/M blended tokens
    """
    # Handle None values
    input_cost = input_cost_per_m if input_cost_per_m is not None else 0.0
    output_cost = output_cost_per_m if output_cost_per_m is not None else 0.0
    
    # Check if we need shadow pricing
    api_cost = input_cost * input_ratio + output_cost * output_ratio
    
    if api_cost <= MIN_COST_FLOOR:
        # Apply shadow pricing for self-hosted/free models
        # Use minimum cost from population if set, otherwise fall back to compute-based
        min_cost = get_tci_minimum_cost()
        if min_cost is not None:
            return min_cost
        
        # Fallback: compute-based shadow pricing
        shadow_input, shadow_output = calculate_shadow_price(
            param_count_b=param_count_b,
            model_name=model_name
        )
        return shadow_input * input_ratio + shadow_output * output_ratio
    
    return max(MIN_COST_FLOOR, api_cost)


def is_self_hosted(
    input_cost_per_m: Optional[float],
    output_cost_per_m: Optional[float],
    model_name: Optional[str] = None,
) -> bool:
    """
    Determine if a model is likely self-hosted vs API-managed.
    
    Heuristics:
    - $0 pricing → self-hosted
    - Very low pricing (<$0.01) → likely self-hosted  
    - Model name contains open-source indicators → self-hosted
    
    Args:
        input_cost_per_m: API input cost
        output_cost_per_m: API output cost
        model_name: Model name for additional heuristics
        
    Returns:
        True if model is likely self-hosted
    """
    # Check pricing
    input_cost = input_cost_per_m or 0.0
    output_cost = output_cost_per_m or 0.0
    
    if input_cost <= MIN_COST_FLOOR and output_cost <= MIN_COST_FLOOR:
        return True
    
    # Check model name for open-source indicators
    if model_name:
        name_lower = model_name.lower()
        open_source_indicators = [
            'llama', 'mistral', 'mixtral', 'qwen', 'deepseek',
            'falcon', 'vicuna', 'phi-', 'gemma', 'yi-',
            'starcoder', 'codellama', 'openchat'
        ]
        if any(ind in name_lower for ind in open_source_indicators):
            # Open source model - check if pricing seems like API pricing
            # API pricing for open source is typically $0.10-$2.00/M
            if input_cost < 0.10 and output_cost < 0.10:
                return True
    
    return False


# Convenience functions for backward compatibility
def get_effective_cost(
    model_data: Dict,
    input_ratio: float = 0.75,
    output_ratio: float = 0.25,
) -> float:
    """
    Get effective cost (TCI) from model data dictionary.
    
    Convenience wrapper for calculate_tci that extracts data from
    the standard model dictionary format.
    
    Args:
        model_data: Dictionary with model information
        input_ratio: Proportion of input tokens (default: 75%)
        output_ratio: Proportion of output tokens (default: 25%)
        
    Returns:
        Total Cost of Inference per 1M blended tokens
    """
    return calculate_tci(
        input_cost_per_m=model_data.get('input_cost_per_m') or model_data.get('price_1m_input'),
        output_cost_per_m=model_data.get('output_cost_per_m') or model_data.get('price_1m_output'),
        param_count_b=model_data.get('param_count_b'),
        model_name=model_data.get('name') or model_data.get('display_name'),
        input_ratio=input_ratio,
        output_ratio=output_ratio,
    )


# =============================================================================
# APPROACH 2: Utopian Distance Method (Mathematical Fix)
# =============================================================================

def minmax_normalize(
    value: float,
    min_val: float,
    max_val: float,
    direction: str = "maximize"
) -> float:
    """
    Min-Max normalize a value to [0, 1] scale where 1 is always optimal.
    
    This normalization handles $0 costs gracefully:
    - For "minimize" metrics: x_norm = (x_max - x) / (x_max - x_min)
      → $0 cost becomes 1.0 (best)
    - For "maximize" metrics: x_norm = (x - x_min) / (x_max - x_min)
      → highest quality becomes 1.0 (best)
    
    Args:
        value: Raw metric value
        min_val: Minimum value in the population
        max_val: Maximum value in the population
        direction: "maximize" (higher is better) or "minimize" (lower is better)
        
    Returns:
        Normalized value in [0, 1] where 1 = optimal (utopia)
        
    Example:
        >>> minmax_normalize(0, min_val=0, max_val=10, direction="minimize")
        1.0  # $0 cost → optimal
        
        >>> minmax_normalize(10, min_val=0, max_val=10, direction="minimize")
        0.0  # Max cost → worst
        
        >>> minmax_normalize(100, min_val=50, max_val=100, direction="maximize")
        1.0  # Max quality → optimal
    """
    # Handle edge case where all values are the same
    if max_val <= min_val:
        return 1.0  # All models equal → all optimal
    
    if direction == "minimize":
        # Lower is better: x_norm = (x_max - x) / (x_max - x_min)
        normalized = (max_val - value) / (max_val - min_val)
    else:
        # Higher is better: x_norm = (x - x_min) / (x_max - x_min)
        normalized = (value - min_val) / (max_val - min_val)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, normalized))


def calculate_utopian_distance(
    quality_norm: float,
    latency_norm: float,
    cost_norm: float,
    weight_quality: float = 1.0,
    weight_latency: float = 1.0,
    weight_cost: float = 1.0,
) -> float:
    """
    Calculate weighted Euclidean distance from the Utopian point (1, 1, 1).
    
    The Utopian point represents a theoretical "Perfect Model":
    - Quality: 1.0 (maximum)
    - Latency: 1.0 (minimum, after normalization)
    - Cost: 1.0 (minimum/$0, after normalization)
    
    Formula:
        D = sqrt(w_q*(1-Q_norm)² + w_l*(1-L_norm)² + w_c*(1-C_norm)²)
    
    The model with MINIMUM distance D is the optimal choice (the "knee").
    
    Args:
        quality_norm: Normalized quality score [0, 1]
        latency_norm: Normalized latency score [0, 1] (1 = fastest)
        cost_norm: Normalized cost score [0, 1] (1 = cheapest/$0)
        weight_quality: Weight for quality dimension (default: 1.0)
        weight_latency: Weight for latency dimension (default: 1.0)
        weight_cost: Weight for cost dimension (default: 1.0)
        
    Returns:
        Distance from utopia (lower = better, closer to ideal)
        
    Example:
        >>> calculate_utopian_distance(0.9, 0.8, 1.0)  # Free, high-quality model
        0.224  # Close to utopia
        
        >>> calculate_utopian_distance(0.5, 0.5, 0.5)  # Average model
        0.866  # Farther from utopia
    """
    import math
    
    # Calculate regret (distance from utopia on each axis)
    quality_regret = 1.0 - quality_norm
    latency_regret = 1.0 - latency_norm
    cost_regret = 1.0 - cost_norm
    
    # Weighted Euclidean distance
    distance = math.sqrt(
        weight_quality * (quality_regret ** 2) +
        weight_latency * (latency_regret ** 2) +
        weight_cost * (cost_regret ** 2)
    )
    
    return distance


def find_utopian_knee(
    models: List[Dict],
    weight_quality: float = 1.0,
    weight_latency: float = 1.0,
    weight_cost: float = 1.0,
) -> List[Dict]:
    """
    Find the optimal "knee" point using the Utopian Distance method.
    
    This method:
    1. Extracts Quality, Latency, Cost from each model
    2. Applies Min-Max normalization (handles $0 costs naturally)
    3. Calculates weighted Euclidean distance from Utopia (1, 1, 1)
    4. Returns models sorted by distance (closest to utopia first)
    
    Args:
        models: List of model dictionaries with keys:
            - 'quality': Quality score (higher is better)
            - 'latency': Latency in seconds (lower is better)
            - 'cost': Cost per 1M tokens (lower is better, can be $0)
            - 'name': Model name (optional)
        weight_quality: Weight for quality (default: 1.0)
        weight_latency: Weight for latency (default: 1.0)
        weight_cost: Weight for cost (default: 1.0)
        
    Returns:
        List of models sorted by utopian distance, each with added fields:
            - 'quality_norm': Normalized quality [0, 1]
            - 'latency_norm': Normalized latency [0, 1]
            - 'cost_norm': Normalized cost [0, 1]
            - 'utopian_distance': Distance from (1, 1, 1)
        
    Example:
        >>> models = [
        ...     {'name': 'GPT-4o', 'quality': 88, 'latency': 0.5, 'cost': 7.5},
        ...     {'name': 'Llama-70B', 'quality': 82, 'latency': 0.3, 'cost': 0},  # $0!
        ...     {'name': 'GPT-4o-mini', 'quality': 75, 'latency': 0.2, 'cost': 0.6},
        ... ]
        >>> ranked = find_utopian_knee(models)
        >>> ranked[0]['name']  # Model closest to utopia
        'Llama-70B'  # High quality + $0 cost = near-optimal
    """
    if not models:
        return []
    
    # Extract metrics
    qualities = [m.get('quality', 0) for m in models]
    latencies = [m.get('latency', 1) for m in models]
    costs = [m.get('cost', 0) for m in models]
    
    # Calculate population min/max for normalization
    q_min, q_max = min(qualities), max(qualities)
    l_min, l_max = min(latencies), max(latencies)
    c_min, c_max = min(costs), max(costs)
    
    # Score each model
    scored_models = []
    for m in models:
        q = m.get('quality', 0)
        l = m.get('latency', 1)
        c = m.get('cost', 0)
        
        # Normalize (direction-aware)
        q_norm = minmax_normalize(q, q_min, q_max, direction="maximize")
        l_norm = minmax_normalize(l, l_min, l_max, direction="minimize")
        c_norm = minmax_normalize(c, c_min, c_max, direction="minimize")
        
        # Calculate distance to utopia
        distance = calculate_utopian_distance(
            q_norm, l_norm, c_norm,
            weight_quality, weight_latency, weight_cost
        )
        
        # Create result with normalized scores and distance
        result = dict(m)
        result['quality_norm'] = round(q_norm, 4)
        result['latency_norm'] = round(l_norm, 4)
        result['cost_norm'] = round(c_norm, 4)
        result['utopian_distance'] = round(distance, 4)
        scored_models.append(result)
    
    # Sort by distance (ascending - closest to utopia first)
    scored_models.sort(key=lambda x: x['utopian_distance'])
    
    return scored_models


# =============================================================================
# APPROACH 3: Kneedle Algorithm on Projected 2D Curve
# =============================================================================

# Small epsilon for log transformation of $0 costs
LOG_EPSILON = 0.0001


def calculate_performance_score(
    quality_norm: float,
    latency_norm: float,
    alpha: float = 0.6,
) -> float:
    """
    Combine Quality and Latency into a single "Performance" metric.
    
    This projects 3D space (Quality, Latency, Cost) down to 2D (Performance, Cost)
    to enable standard 2D knee detection.
    
    Formula:
        Performance = α * Quality_norm + (1-α) * Latency_norm
    
    Args:
        quality_norm: Normalized quality [0, 1] where 1 = best
        latency_norm: Normalized latency [0, 1] where 1 = fastest
        alpha: Weight for quality (default: 0.6, prioritizes quality over speed)
        
    Returns:
        Performance score [0, 1] where 1 = best
        
    Example:
        >>> calculate_performance_score(0.9, 0.8, alpha=0.6)
        0.86  # 0.6*0.9 + 0.4*0.8 = 0.86
    """
    return alpha * quality_norm + (1 - alpha) * latency_norm


def log_transform_cost(cost: float, epsilon: float = LOG_EPSILON) -> float:
    """
    Apply log transformation to cost for visualization.
    
    Handles $0 cost by adding a small epsilon before taking log.
    This allows free models to appear at the far left of the graph
    without breaking the axis.
    
    Formula:
        X_plot = log(Cost + ε)
    
    Args:
        cost: Raw cost (can be $0)
        epsilon: Small value to add before log (default: 0.0001)
        
    Returns:
        Log-transformed cost
        
    Example:
        >>> log_transform_cost(0)      # Free model
        -9.21  # log(0.0001) ≈ -9.21
        >>> log_transform_cost(1.0)    # $1/M model
        0.0    # log(1) = 0
        >>> log_transform_cost(10.0)   # $10/M model
        2.30   # log(10) ≈ 2.30
    """
    return math.log(cost + epsilon)


def find_pareto_frontier_2d(
    models: List[Dict],
    x_key: str = "cost_log",
    y_key: str = "performance",
) -> List[Dict]:
    """
    Find 2D Pareto frontier (non-dominated points).
    
    A point is on the frontier if no other point is both:
    - Lower on x-axis (cheaper)
    - Higher on y-axis (better performance)
    
    Handles ties correctly: when multiple models have the same x-value
    (e.g., multiple $0 models), only the one with highest y is kept.
    
    Args:
        models: List of model dicts with x_key and y_key values
        x_key: Key for x-axis metric (to minimize, e.g., cost)
        y_key: Key for y-axis metric (to maximize, e.g., performance)
        
    Returns:
        List of models on the Pareto frontier, sorted by x_key
    """
    if not models:
        return []
    
    # Step 1: For models with the same x-value, keep only the one with highest y
    # This handles multiple $0 models correctly
    best_at_x: Dict[float, Dict] = {}
    for m in models:
        x = m[x_key]
        y = m[y_key]
        # Round x to handle floating point comparison issues
        x_rounded = round(x, 6)
        if x_rounded not in best_at_x or y > best_at_x[x_rounded][y_key]:
            best_at_x[x_rounded] = m
    
    # Step 2: Sort unique models by x (cost) ascending
    sorted_models = sorted(best_at_x.values(), key=lambda m: m[x_key])
    
    # Step 3: Find frontier - model is on frontier if it has higher y than all cheaper models
    frontier = []
    max_y = float('-inf')
    
    for m in sorted_models:
        if m[y_key] > max_y:
            frontier.append(m)
            max_y = m[y_key]
    
    return frontier


def perpendicular_distance(
    x: float, y: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> float:
    """
    Calculate perpendicular distance from point (x, y) to line through (x1, y1) and (x2, y2).
    
    Formula:
        d = |((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)| / sqrt((y2-y1)² + (x2-x1)²)
    
    Args:
        x, y: Point coordinates
        x1, y1: Line start point
        x2, y2: Line end point
        
    Returns:
        Perpendicular distance (always positive)
    """
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def find_kneedle_point(
    models: List[Dict],
    quality_key: str = "quality",
    latency_key: str = "latency",
    cost_key: str = "cost",
    alpha: float = 0.6,
    use_pareto_only: bool = True,
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Find the "knee" point using the Kneedle algorithm on a 2D projected curve.
    
    The Kneedle algorithm:
    1. Project 3D data to 2D: Performance (combined Quality+Latency) vs Cost
    2. Transform Cost to log scale: log(Cost + ε) to handle $0
    3. Find the Pareto frontier (efficient points only)
    4. Draw a line from cheapest to most expensive point
    5. Find the point with maximum perpendicular distance to this line
    
    This point represents the "elbow" - where you get the most performance
    improvement per dollar spent before diminishing returns set in.
    
    Args:
        models: List of model dicts with quality, latency, cost keys
        quality_key: Key for quality metric (higher is better)
        latency_key: Key for latency metric (lower is better, in seconds)
        cost_key: Key for cost metric (lower is better, can be $0)
        alpha: Weight for quality in performance score (default: 0.6)
        use_pareto_only: If True, only consider Pareto-efficient points (default: True)
        
    Returns:
        Tuple of (knee_point_dict, all_processed_models)
        - knee_point_dict: The model at the knee, or None if not found
        - all_processed_models: All models with added projection fields
        
    Example:
        >>> models = [
        ...     {'name': 'GPT-4o', 'quality': 88, 'latency': 0.5, 'cost': 7.5},
        ...     {'name': 'Llama-70B', 'quality': 82, 'latency': 0.3, 'cost': 0},
        ...     {'name': 'GPT-4o-mini', 'quality': 75, 'latency': 0.2, 'cost': 0.6},
        ... ]
        >>> knee, processed = find_kneedle_point(models)
        >>> knee['name']
        'GPT-4o-mini'  # Best balance of performance vs cost
    """
    if not models or len(models) < 2:
        return (models[0] if models else None, models)
    
    # Step 1: Extract and normalize metrics
    qualities = [m.get(quality_key, 0) for m in models]
    latencies = [m.get(latency_key, 1) for m in models]
    costs = [m.get(cost_key, 0) for m in models]
    
    q_min, q_max = min(qualities), max(qualities)
    l_min, l_max = min(latencies), max(latencies)
    
    # Step 2: Calculate Performance and log(Cost) for each model
    processed = []
    for m in models:
        q = m.get(quality_key, 0)
        l = m.get(latency_key, 1)
        c = m.get(cost_key, 0)
        
        # Normalize quality (higher is better)
        q_norm = minmax_normalize(q, q_min, q_max, direction="maximize")
        
        # Normalize latency (lower is better → invert)
        l_norm = minmax_normalize(l, l_min, l_max, direction="minimize")
        
        # Calculate combined Performance score
        perf = calculate_performance_score(q_norm, l_norm, alpha)
        
        # Log-transform cost for visualization
        cost_log = log_transform_cost(c)
        
        result = dict(m)
        result['quality_norm'] = round(q_norm, 4)
        result['latency_norm'] = round(l_norm, 4)
        result['performance'] = round(perf, 4)
        result['cost_log'] = round(cost_log, 4)
        result['cost_raw'] = c
        processed.append(result)
    
    # Step 3: Find Pareto frontier (or use all points)
    if use_pareto_only:
        frontier = find_pareto_frontier_2d(processed, x_key="cost_log", y_key="performance")
    else:
        frontier = sorted(processed, key=lambda m: m['cost_log'])
    
    if len(frontier) < 2:
        return (frontier[0] if frontier else None, processed)
    
    # Step 4: Draw line from cheapest to most expensive point on frontier
    start = frontier[0]   # Cheapest (leftmost on log scale)
    end = frontier[-1]    # Most expensive (rightmost)
    
    x1, y1 = start['cost_log'], start['performance']
    x2, y2 = end['cost_log'], end['performance']
    
    # Step 5: Find point with maximum perpendicular distance
    max_distance = -1
    knee_point = None
    
    for m in frontier:
        x, y = m['cost_log'], m['performance']
        dist = perpendicular_distance(x, y, x1, y1, x2, y2)
        m['kneedle_distance'] = round(dist, 4)
        
        if dist > max_distance:
            max_distance = dist
            knee_point = m
    
    # Mark the knee point
    if knee_point:
        knee_point['is_knee'] = True
    
    return (knee_point, processed)


def find_kneedle_knee(
    models: List[Dict],
    weight_quality: float = 0.6,
    weight_latency: float = 0.4,
) -> List[Dict]:
    """
    Convenience function to find and rank models using the Kneedle algorithm.
    
    Returns all models sorted by their desirability, with the knee point first.
    
    Ranking criteria (in order):
    1. Knee point (maximum curvature)
    2. Other Pareto-efficient points (by distance from line)
    3. Non-Pareto points (by performance/cost ratio)
    
    Args:
        models: List of model dicts with 'quality', 'latency', 'cost' keys
        weight_quality: Weight for quality in performance calculation (default: 0.6)
        weight_latency: Weight for latency (default: 0.4, must sum to 1 with quality)
        
    Returns:
        List of models sorted by Kneedle ranking, with added fields:
        - 'performance': Combined quality+latency score [0,1]
        - 'cost_log': Log-transformed cost for visualization
        - 'kneedle_distance': Distance from efficiency line (higher = more "knee-like")
        - 'is_knee': True for the knee point
        - 'kneedle_rank': Ranking (1 = best)
        
    Example:
        >>> models = [
        ...     {'name': 'GPT-4o', 'quality': 88, 'latency': 0.5, 'cost': 7.5},
        ...     {'name': 'Llama-70B', 'quality': 82, 'latency': 0.3, 'cost': 0},
        ...     {'name': 'Claude', 'quality': 88, 'latency': 0.4, 'cost': 6.0},
        ... ]
        >>> ranked = find_kneedle_knee(models)
        >>> ranked[0]['name']  # The knee point
        'Llama-70B'
    """
    if not models:
        return []
    
    alpha = weight_quality / (weight_quality + weight_latency)
    
    knee_point, processed = find_kneedle_point(models, alpha=alpha)
    
    # Sort: knee first, then by kneedle_distance (descending), then by performance/cost
    def sort_key(m):
        is_knee = m.get('is_knee', False)
        knee_dist = m.get('kneedle_distance', 0)
        perf = m.get('performance', 0)
        cost_log = m.get('cost_log', 0)
        
        # Primary: is_knee (True sorts first as -1)
        # Secondary: kneedle_distance (higher = better, so negate)
        # Tertiary: performance minus cost penalty
        return (
            0 if is_knee else 1,
            -knee_dist,
            -(perf - cost_log * 0.1)  # Slight cost penalty for tiebreaking
        )
    
    processed.sort(key=sort_key)
    
    # Add ranking
    for i, m in enumerate(processed, 1):
        m['kneedle_rank'] = i
    
    return processed


def get_kneedle_visualization_data(
    models: List[Dict],
    alpha: float = 0.6,
) -> Dict:
    """
    Get data formatted for plotting the Kneedle curve.
    
    Returns data suitable for matplotlib or other plotting libraries.
    
    Args:
        models: List of model dicts with 'quality', 'latency', 'cost' keys
        alpha: Weight for quality in performance score
        
    Returns:
        Dictionary with:
        - 'all_points': List of (x, y, name) for all models
        - 'frontier_points': List of (x, y, name) for Pareto frontier
        - 'knee_point': (x, y, name) for the knee
        - 'efficiency_line': ((x1, y1), (x2, y2)) start and end points
        - 'x_label': Suggested x-axis label
        - 'y_label': Suggested y-axis label
        
    Example:
        >>> data = get_kneedle_visualization_data(models)
        >>> plt.scatter(*zip(*[(p[0], p[1]) for p in data['all_points']]))
        >>> plt.plot(*zip(*data['efficiency_line']), 'r--')
        >>> knee = data['knee_point']
        >>> plt.scatter(knee[0], knee[1], color='red', s=200, marker='*')
    """
    knee_point, processed = find_kneedle_point(models, alpha=alpha)
    frontier = find_pareto_frontier_2d(processed, x_key="cost_log", y_key="performance")
    
    all_points = [(m['cost_log'], m['performance'], m.get('name', '?')) for m in processed]
    frontier_points = [(m['cost_log'], m['performance'], m.get('name', '?')) for m in frontier]
    
    knee_data = None
    if knee_point:
        knee_data = (knee_point['cost_log'], knee_point['performance'], knee_point.get('name', '?'))
    
    efficiency_line = None
    if len(frontier) >= 2:
        start, end = frontier[0], frontier[-1]
        efficiency_line = (
            (start['cost_log'], start['performance']),
            (end['cost_log'], end['performance'])
        )
    
    return {
        'all_points': all_points,
        'frontier_points': frontier_points,
        'knee_point': knee_data,
        'efficiency_line': efficiency_line,
        'x_label': 'log(Cost + ε)  [Lower = Cheaper]',
        'y_label': f'Performance  [α={alpha:.1f}×Quality + {1-alpha:.1f}×Speed]',
        'processed_models': processed,
    }

