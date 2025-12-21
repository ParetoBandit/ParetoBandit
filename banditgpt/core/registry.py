"""
Model Registry Utilities

Provides default model registry with 81 models including:
- Benchmark scores (Math-500, MMLU-Pro, Reasoning)
- Cost information (input/output pricing)
- Latency estimates

The default registry enables "Metadata-Guided Cold Start" by initializing
the bandit with public benchmark scores instead of starting from scratch.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_default_registry() -> Dict[str, Dict[str, Any]]:
    """
    Load the default model registry with 81 models.
    
    Returns:
        Dictionary mapping model_id -> model_info, where model_info contains:
        - display_name: Human-readable name
        - cost_per_1k_input: Cost per 1K input tokens
        - cost_per_1k_output: Cost per 1K output tokens
        - benchmarks: Dict with math_500, mmlu_pro, reasoning, average scores
        - metadata: Additional info (latency, throughput, etc.)
    
    Example:
        >>> registry = load_default_registry()
        >>> len(registry)
        81
        >>> registry['openai/gpt-4o-mini']['benchmarks']['average']
        0.566
    """
    registry_path = Path(__file__).parent.parent / "data" / "model_registry.json"
    
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "This file should be included in the package."
        )
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    return registry


def get_benchmark_average(model_id: str, registry: Optional[Dict] = None) -> float:
    """
    Get the 3-benchmark average score for a model.
    
    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4o-mini')
        registry: Optional registry dict. If None, loads default.
    
    Returns:
        Average of Math-500, MMLU-Pro, and Reasoning scores (0.0-1.0)
        Returns 0.5 if model not found (neutral prior).
    
    Example:
        >>> get_benchmark_average('openai/gpt-4o-mini')
        0.566
    """
    if registry is None:
        registry = load_default_registry()
    
    if model_id not in registry:
        return 0.5  # Neutral prior for unknown models
    
    return registry[model_id]['benchmarks']['average']


def create_minimal_registry(model_ids: list) -> Dict[str, Dict[str, Any]]:
    """
    Create a minimal registry with only specified models.
    
    Useful for testing or focused experiments.
    
    Args:
        model_ids: List of model identifiers
    
    Returns:
        Registry containing only the specified models
    
    Example:
        >>> registry = create_minimal_registry([
        ...     'openai/gpt-4o-mini',
        ...     'anthropic/claude-3.5-haiku'
        ... ])
        >>> len(registry)
        2
    """
    full_registry = load_default_registry()
    
    minimal_registry = {}
    for model_id in model_ids:
        if model_id in full_registry:
            minimal_registry[model_id] = full_registry[model_id]
        else:
            # Create a placeholder entry for unknown models
            minimal_registry[model_id] = {
                "display_name": model_id.split('/')[-1],
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.001,
                "benchmarks": {
                    "math_500": 0.5,
                    "mmlu_pro": 0.5,
                    "reasoning": 0.5,
                    "average": 0.5
                },
                "metadata": {}
            }
    
    return minimal_registry


def get_models_by_benchmark_tier(
    tier: str = "high",
    registry: Optional[Dict] = None
) -> list:
    """
    Get models filtered by benchmark performance tier.
    
    Args:
        tier: 'high' (>0.7), 'medium' (0.5-0.7), 'low' (<0.5)
        registry: Optional registry dict. If None, loads default.
    
    Returns:
        List of model IDs in the specified tier, sorted by benchmark average
    
    Example:
        >>> high_performers = get_models_by_benchmark_tier('high')
        >>> len(high_performers)
        30
    """
    if registry is None:
        registry = load_default_registry()
    
    thresholds = {
        'high': (0.7, 1.0),
        'medium': (0.5, 0.7),
        'low': (0.0, 0.5)
    }
    
    if tier not in thresholds:
        raise ValueError(f"tier must be one of {list(thresholds.keys())}")
    
    min_score, max_score = thresholds[tier]
    
    models = [
        (model_id, info['benchmarks']['average'])
        for model_id, info in registry.items()
        if min_score <= info['benchmarks']['average'] < max_score
    ]
    
    # Sort by score descending
    models.sort(key=lambda x: x[1], reverse=True)
    
    return [model_id for model_id, _ in models]

