"""
LLM Jury - Intelligent LLM Model Routing and Recommendation Library

A library for routing prompts to optimal LLM models based on task complexity,
domain category, and cost-quality tradeoffs using multi-objective optimization.
"""

# IMPORTANT: Import torch first to avoid segfaults on Mac.
# The `datasets` library imports torch, and if torch is first imported during
# nested module imports (like when llm_jury.data imports datasets), it can cause
# segmentation faults during later PyTorch model operations. Importing torch
# at the top level ensures it's initialized in a clean context.
# See: https://github.com/pytorch/pytorch/issues/78490
try:
    import torch
except ImportError:
    pass  # torch is optional for some functionality

__version__ = "0.1.0"

# Default baseline model for optimization comparisons
DEFAULT_BASELINE_MODEL = "Gemini 2.5 Pro"

# Core models and enums
from llm_jury.core.models import (
    ProductArchetype,
    PromptCategory,
    ModelMetadata,
    ModelSpecs,
    RoutingDecision,
    RecommendationResult,
    RankedModel,
    ModelTier,
)

# Routing (basic)
from llm_jury.routing import PromptRouter, ArchetypeRouter

# Ranking
from llm_jury.ranking import (
    Optimizer,
    OptimizationStrategy,
    MissingDataStrategy,
    QualityScorer,
    Objective,
    ObjectiveRegistry,
    NormalizationMethod,
)

# Data sources
from llm_jury.data import ModelRegistry, HuggingFaceDataSource

# Main orchestrator
from llm_jury.orchestration import (
    get_recommendations,
    get_recommendations_for_use_case,
    get_best_models_for_budget,
    get_value_recommendations,
    analyze_prompt,
    list_use_cases,
    get_use_case_config,
    is_open_source,
    UseCase,
    UseCaseConfig,
    USE_CASE_CONFIGS,
    ValueRecommendation,
    OPEN_SOURCE_PATTERNS,
)

# Prompt classification
from llm_jury.routing import (
    PromptClassifier,
    ClassificationResult,
    UseCaseCategory,
    classify_prompt,
)

# Configuration
from llm_jury.config import get_config, Config


def list_available_models(cache_path=None, verbose=False):
    """
    List all available models in the cache.
    
    Args:
        cache_path: Optional path to custom model cache file.
        verbose: Whether to print loading messages.
        
    Returns:
        List of model names available for use as baseline or candidates.
        
    Example:
        >>> from llm_jury import list_available_models
        >>> models = list_available_models()
        >>> print(models[:5])
        ['Claude 3.5 Sonnet (new)', 'GPT-4o', 'Gemini 2.5 Pro', ...]
    """
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=verbose)
    return sorted([m.name for m in registry])


def get_model_by_name(name, cache_path=None, verbose=False):
    """
    Get a specific model by name from the cache.
    
    Args:
        name: Model name to look up.
        cache_path: Optional path to custom model cache file.
        verbose: Whether to print loading messages.
        
    Returns:
        ModelMetadata if found, None otherwise.
        
    Example:
        >>> from llm_jury import get_model_by_name
        >>> model = get_model_by_name("GPT-4o")
        >>> print(model.input_cost_per_m)
        2.5
    """
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=verbose)
    return next((m for m in registry if m.name == name), None)


__all__ = [
    "__version__",
    # Constants
    "DEFAULT_BASELINE_MODEL",
    # Core models
    "ProductArchetype",
    "PromptCategory",
    "ModelMetadata",
    "ModelSpecs",
    "RoutingDecision",
    "RecommendationResult",
    "RankedModel",
    "ModelTier",
    # Routing
    "PromptRouter",
    "ArchetypeRouter",
    # Prompt Classification (NEW)
    "PromptClassifier",
    "ClassificationResult",
    "UseCaseCategory",
    "classify_prompt",
    # Ranking
    "Optimizer",
    "OptimizationStrategy",
    "MissingDataStrategy",
    "QualityScorer",
    "Objective",
    "ObjectiveRegistry",
    "NormalizationMethod",
    # Data
    "ModelRegistry",
    "HuggingFaceDataSource",
    # Orchestration
    "get_recommendations",
    "get_recommendations_for_use_case",
    "get_best_models_for_budget",
    "get_value_recommendations",  # NEW: Main entry point
    "analyze_prompt",             # NEW: Prompt analysis
    "ValueRecommendation",        # NEW: Value result type
    "list_use_cases",
    "get_use_case_config",
    "is_open_source",             # Check if model is open source
    "UseCase",
    "UseCaseConfig",
    "USE_CASE_CONFIGS",
    "OPEN_SOURCE_PATTERNS",       # Patterns for open source model detection
    # Helpers
    "list_available_models",
    "get_model_by_name",
    # Configuration
    "get_config",
    "Config",
]
