"""
Main orchestrator for LLM model recommendations.

Consolidates get_recommendations() from:
- llm_router.py (with trust scoring and detailed output)
- llm_recommendation_orchestrator.py (simpler version)
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from llm_jury.core.models import ModelMetadata, RecommendationResult, RoutingDecision, PromptCategory, ProductArchetype
from llm_jury.data.registry import ModelRegistry
from llm_jury.routing.archetype_router import ArchetypeRouter
from llm_jury.routing.prompt_classifier import PromptClassifier, ClassificationResult
from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy

# Default baseline model for optimization
DEFAULT_BASELINE_MODEL = "Gemini 2.5 Pro"

# Open source model patterns for filtering
OPEN_SOURCE_PATTERNS = [
    'DeepSeek',
    'Qwen', 
    'GLM',
    'Llama',
    'Mistral',
    'Mixtral',  # Mistral's mixture-of-experts model
    'Gemma',
    'Phi-',
    'gpt-oss',  # Open source GPT variants
]


def is_open_source(model_name: str) -> bool:
    """
    Check if a model is open source based on its name.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if the model is open source, False otherwise
        
    Example:
        >>> is_open_source("DeepSeek V3.1 Terminus")
        True
        >>> is_open_source("GPT-5.1 (high)")
        False
    """
    return any(pattern in model_name for pattern in OPEN_SOURCE_PATTERNS)


# =============================================================================
# Use Case Definitions
# =============================================================================

class UseCase(Enum):
    """
    Predefined use cases for LLM model selection.
    
    Each use case maps to optimized settings for ranking models based on
    the specific requirements of that task type.
    
    Example:
        >>> from llm_jury import get_recommendations_for_use_case, UseCase
        >>> results = get_recommendations_for_use_case(UseCase.CODE_GENERATION, top_k=3)
    """
    
    # Development & Engineering
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "code_refactoring"
    TECHNICAL_DOCS = "technical_docs"
    
    # Data & Analytics
    DATA_ANALYSIS = "data_analysis"
    SQL_GENERATION = "sql_generation"
    MATH_REASONING = "math_reasoning"
    
    # Content & Communication
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    
    # Specialized Domains
    LEGAL_REVIEW = "legal_review"
    FINANCIAL_ANALYSIS = "financial_analysis"
    RESEARCH_ASSISTANT = "research_assistant"
    
    # Conversational & Support
    CUSTOMER_SUPPORT = "customer_support"
    TUTORING = "tutoring"
    GENERAL_QA = "general_qa"
    
    # Technical Capabilities
    RAG_PIPELINE = "rag_pipeline"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_EXTRACTION = "structured_extraction"
    LONG_CONTEXT = "long_context"
    
    # Cost/Performance Focused
    COST_OPTIMIZED = "cost_optimized"
    LOW_LATENCY = "low_latency"
    MAXIMUM_QUALITY = "maximum_quality"


@dataclass
class UseCaseConfig:
    """Configuration for a use case's optimization settings."""
    strategy: OptimizationStrategy
    weights: Dict[str, float]
    description: str
    prompt_category: PromptCategory
    quality_range: Optional[tuple] = None
    cost_range: Optional[tuple] = None
    speed_range: Optional[tuple] = None


# Use case configurations with optimized weights
USE_CASE_CONFIGS: Dict[UseCase, UseCaseConfig] = {
    # ==========================================================================
    # Development & Engineering
    # ==========================================================================
    UseCase.CODE_GENERATION: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.45,      # High weight on coding benchmarks
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.20,  # Important for correct code
            "refusal": 0.10,
        },
        description="Writing new code, implementing features, algorithms",
        prompt_category=PromptCategory.CODING,
    ),
    
    UseCase.CODE_REVIEW: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.35,
            "cost": 0.10,
            "latency": 0.10,
            "hallucination": 0.35,  # Must not miss bugs or invent issues
            "refusal": 0.10,
        },
        description="Reviewing code for bugs, security issues, best practices",
        prompt_category=PromptCategory.CODING,
    ),
    
    UseCase.CODE_REFACTORING: UseCaseConfig(
        strategy=OptimizationStrategy.BALANCED,
        weights={
            "quality": 0.40,
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.25,
            "refusal": 0.10,
        },
        description="Improving, modernizing, or restructuring existing code",
        prompt_category=PromptCategory.CODING,
    ),
    
    UseCase.TECHNICAL_DOCS: UseCaseConfig(
        strategy=OptimizationStrategy.BALANCED,
        weights={
            "quality": 0.35,
            "cost": 0.20,
            "latency": 0.10,
            "hallucination": 0.25,  # Accuracy matters for docs
            "refusal": 0.10,
        },
        description="API documentation, READMEs, technical writing",
        prompt_category=PromptCategory.CODING,
    ),
    
    # ==========================================================================
    # Data & Analytics
    # ==========================================================================
    UseCase.DATA_ANALYSIS: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.40,
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.25,  # Must not fabricate insights
            "refusal": 0.10,
        },
        description="Analyzing datasets, generating insights, statistics",
        prompt_category=PromptCategory.DATA_SCIENCE,
    ),
    
    UseCase.SQL_GENERATION: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.35,
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.30,  # Wrong SQL can be dangerous
            "refusal": 0.10,
        },
        description="Database queries, schema design, query optimization",
        prompt_category=PromptCategory.DATA_SCIENCE,
    ),
    
    UseCase.MATH_REASONING: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.50,      # Math benchmarks are key
            "cost": 0.10,
            "latency": 0.10,
            "hallucination": 0.20,
            "refusal": 0.10,
        },
        description="Complex calculations, proofs, mathematical problems",
        prompt_category=PromptCategory.DATA_SCIENCE,
    ),
    
    # ==========================================================================
    # Content & Communication
    # ==========================================================================
    UseCase.CREATIVE_WRITING: UseCaseConfig(
        strategy=OptimizationStrategy.BALANCED,
        weights={
            "quality": 0.40,
            "cost": 0.20,
            "latency": 0.10,
            "hallucination": 0.10,  # Less critical for creative work
            "refusal": 0.20,        # Low refusal important for creativity
        },
        description="Stories, marketing copy, creative content",
        prompt_category=PromptCategory.CREATIVE,
    ),
    
    UseCase.SUMMARIZATION: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.30,
            "cost": 0.15,
            "latency": 0.15,
            "hallucination": 0.30,  # Must not add false info
            "refusal": 0.10,
        },
        description="Summarizing documents, articles, meeting notes",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.TRANSLATION: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.45,
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.20,
            "refusal": 0.10,
        },
        description="Language translation between languages",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    # ==========================================================================
    # Specialized Domains
    # ==========================================================================
    UseCase.LEGAL_REVIEW: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.30,
            "cost": 0.10,
            "latency": 0.05,
            "hallucination": 0.45,  # Critical: must not fabricate clauses
            "refusal": 0.10,
        },
        description="Contract review, regulatory compliance, legal analysis",
        prompt_category=PromptCategory.LEGAL,
    ),
    
    UseCase.FINANCIAL_ANALYSIS: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.30,
            "cost": 0.10,
            "latency": 0.05,
            "hallucination": 0.45,  # Critical: financial accuracy
            "refusal": 0.10,
        },
        description="Financial modeling, market analysis, risk assessment",
        prompt_category=PromptCategory.FINANCE,
    ),
    
    UseCase.RESEARCH_ASSISTANT: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.45,
            "cost": 0.10,
            "latency": 0.05,
            "hallucination": 0.30,  # Must cite accurately
            "refusal": 0.10,
        },
        description="Academic research, literature review, scientific analysis",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    # ==========================================================================
    # Conversational & Support
    # ==========================================================================
    UseCase.CUSTOMER_SUPPORT: UseCaseConfig(
        strategy=OptimizationStrategy.SPEED_FOCUSED,
        weights={
            "quality": 0.25,
            "cost": 0.20,
            "latency": 0.30,        # Fast responses critical
            "hallucination": 0.15,
            "refusal": 0.10,
        },
        description="Chatbots, help desk, customer service",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.TUTORING: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.40,
            "cost": 0.15,
            "latency": 0.15,
            "hallucination": 0.20,
            "refusal": 0.10,
        },
        description="Educational explanations, teaching, learning assistance",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.GENERAL_QA: UseCaseConfig(
        strategy=OptimizationStrategy.BALANCED,
        weights={
            "quality": 0.35,
            "cost": 0.20,
            "latency": 0.15,
            "hallucination": 0.20,
            "refusal": 0.10,
        },
        description="General question answering, factual queries",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    # ==========================================================================
    # Technical Capabilities
    # ==========================================================================
    UseCase.RAG_PIPELINE: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.30,
            "cost": 0.15,
            "latency": 0.15,
            "hallucination": 0.30,  # Must use retrieved context accurately
            "refusal": 0.10,
        },
        description="Retrieval-augmented generation with external documents",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.FUNCTION_CALLING: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.35,
            "cost": 0.15,
            "latency": 0.15,
            "hallucination": 0.25,  # Must call correct functions
            "refusal": 0.10,
        },
        description="Tool use, API integration, function execution",
        prompt_category=PromptCategory.CODING,
    ),
    
    UseCase.STRUCTURED_EXTRACTION: UseCaseConfig(
        strategy=OptimizationStrategy.RELIABILITY_FOCUSED,
        weights={
            "quality": 0.30,
            "cost": 0.15,
            "latency": 0.15,
            "hallucination": 0.30,  # Must extract accurately
            "refusal": 0.10,
        },
        description="JSON extraction, form filling, schema-based output",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.LONG_CONTEXT: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.40,
            "cost": 0.15,
            "latency": 0.10,
            "hallucination": 0.25,
            "refusal": 0.10,
        },
        description="Processing very long documents (100K+ tokens)",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    # ==========================================================================
    # Cost/Performance Focused
    # ==========================================================================
    UseCase.COST_OPTIMIZED: UseCaseConfig(
        strategy=OptimizationStrategy.COST_FOCUSED,
        weights={
            "quality": 0.25,
            "cost": 0.40,           # Minimize cost
            "latency": 0.10,
            "hallucination": 0.15,
            "refusal": 0.10,
        },
        description="Budget-conscious applications, high-volume processing",
        prompt_category=PromptCategory.GENERAL,
        quality_range=(0.70, 1.0),  # Accept 70%+ quality
        cost_range=(0.0, 0.30),     # Under 30% of baseline cost
    ),
    
    UseCase.LOW_LATENCY: UseCaseConfig(
        strategy=OptimizationStrategy.SPEED_FOCUSED,
        weights={
            "quality": 0.25,
            "cost": 0.15,
            "latency": 0.40,        # Minimize latency
            "hallucination": 0.10,
            "refusal": 0.10,
        },
        description="Real-time applications, interactive chat, streaming",
        prompt_category=PromptCategory.GENERAL,
    ),
    
    UseCase.MAXIMUM_QUALITY: UseCaseConfig(
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        weights={
            "quality": 0.55,        # Maximum quality
            "cost": 0.05,
            "latency": 0.05,
            "hallucination": 0.25,
            "refusal": 0.10,
        },
        description="Best possible results regardless of cost",
        prompt_category=PromptCategory.GENERAL,
    ),
}


def get_recommendations(
    prompt: str,
    has_search_tools: bool = True,
    baseline_model_name: Optional[str] = None,
    ranking_strategy: OptimizationStrategy = OptimizationStrategy.KNEE,
    quality_range: Optional[tuple] = None,
    cost_range: Optional[tuple] = None,
    speed_range: Optional[tuple] = None,
    cache_path: Optional[Union[str, Path]] = None,
    top_k: int = 3,
    verbose: bool = True,
    force_refresh: bool = False
) -> List[RecommendationResult]:
    """
    Get model recommendations for a given prompt.
    
    This is the main entry point for the library. It:
    1. Loads models from the cache (default or user-provided)
    2. Routes the prompt to an archetype and category
    3. Filters models by archetype and pricing source
    4. Ranks models using Knee Point optimization (best bang for buck)
    5. Returns top-k recommendations
    
    Args:
        prompt: User prompt to route and rank models for
        has_search_tools: Whether search/RAG tools are available
        baseline_model_name: Name of baseline/reference model for comparison.
            If None, defaults to DEFAULT_BASELINE_MODEL ("Gemini 2.5 Pro").
            Use list_available_models() to see available options.
        ranking_strategy: Strategy for ranking. Default is KNEE (best value tradeoff).
            Options: KNEE, BALANCED, VALUE_OPTIMIZED, QUALITY_FOCUSED, COST_FOCUSED
        quality_range: (min, max) quality ratios for VALUE_OPTIMIZED (e.g., (0.80, 0.95))
        cost_range: (min, max) cost ratios for VALUE_OPTIMIZED (e.g., (0.10, 0.30))
        speed_range: (min, max) speed ratios for VALUE_OPTIMIZED (optional)
        cache_path: Path to custom model cache file. If None, uses default cache.
        top_k: Number of top models to return
        verbose: Whether to print detailed output
        force_refresh: Deprecated - use cache regeneration instead
        
    Returns:
        List of top-k RecommendationResult objects
        
    Example:
        >>> from llm_jury import get_recommendations, OptimizationStrategy
        
        # Basic usage with default baseline (Gemini 2.5 Pro)
        >>> results = get_recommendations(
        ...     "Write a Python function to parse JSON",
        ...     has_search_tools=False
        ... )
        
        # Use a different baseline model
        >>> results = get_recommendations(
        ...     "Explain quantum computing",
        ...     baseline_model_name="Claude 3.5 Sonnet (new)"
        ... )
        
        # See available models for baseline selection
        >>> from llm_jury import list_available_models
        >>> models = list_available_models()
        >>> print(models[:5])
        
        # Custom baseline with value-optimized strategy
        >>> results = get_recommendations(
        ...     "Write a Python function to parse JSON",
        ...     baseline_model_name="GPT-4o",
        ...     ranking_strategy=OptimizationStrategy.VALUE_OPTIMIZED,
        ...     quality_range=(0.80, 0.95),
        ...     cost_range=(0.10, 0.30)
        ... )
    """
    # 1. Setup - Load from cache (default or custom)
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=verbose)
    
    if not registry:
        if verbose:
            print("⚠️  No models loaded from cache.")
            if cache_path:
                print(f"   Check that the file exists: {cache_path}")
            else:
                print("   Run 'python run_etl.py --complete-only' first.")
        return []
    
    # Use provided baseline or default
    baseline_name = baseline_model_name or DEFAULT_BASELINE_MODEL
    baseline = next((m for m in registry if m.name == baseline_name), None)
    
    if baseline is None:
        if verbose:
            print(f"⚠️  Baseline model '{baseline_name}' not found in registry.")
            print(f"   Available models: {[m.name for m in registry[:10]]}...")
            print(f"   Using first model as fallback: {registry[0].name}")
        baseline = registry[0]
    
    # Load raw model data for quality scorer initialization
    raw_models_data = ModelRegistry.load_raw_cache(cache_path=cache_path)
    
    if not raw_models_data:
        if verbose:
            print("⚠️  Could not load raw model data for quality scoring.")
        return []
    
    if not raw_models_data:
        if verbose:
            print("⚠️  Could not load raw model data for quality scoring.")
        return []
    
    # Initialize Router with Hybrid Classification
    # We default to local model (use_api=False) for privacy/offline support
    # but this can be exposed via config in the future
    router = ArchetypeRouter(use_api=False, fallback_threshold=0.75)
    
    optimizer = Optimizer(
        baseline, 
        all_models_data=raw_models_data, 
        strategy=ranking_strategy,
        quality_range=quality_range,
        cost_range=cost_range,
        speed_range=speed_range
    )

    # 2. Route
    decision = router.route(prompt, has_search_tools)
    
    # 3. Filter candidates by archetype
    # Accept models with pricing data from any source
    candidates = [
        m for m in registry 
        if m.archetype == decision.archetype 
        and m.input_cost_per_m is not None
        and m.output_cost_per_m is not None
    ]
    
    # Fallback if cluster is empty - try frontier models
    if not candidates:
        from llm_jury.core.models import ProductArchetype
        candidates = [
            m for m in registry 
            if m.archetype == ProductArchetype.FRONTIER
            and m.input_cost_per_m is not None
            and m.output_cost_per_m is not None
        ]
    
    # If still no candidates, use all models with pricing
    if not candidates:
        candidates = [
            m for m in registry
            if m.input_cost_per_m is not None
            and m.output_cost_per_m is not None
        ]
    
    if not candidates:
        if verbose:
            print("\n⚠️  No models found with pricing data")
            print("   Ensure your cache includes input_cost_per_m and output_cost_per_m fields")
        return []

    # 4. Rank
    ranked = optimizer.rank(candidates, decision, top_k=top_k)

    # 5. Output (if verbose)
    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 PROMPT: \"{prompt}\"")
        print(f"⚙️  TOOLS: {has_search_tools}")
        print(f"{'='*60}")
        print(f"🎯 INTENT: {decision.category.name} | ARCHETYPE: {decision.archetype.name}")
        
        if decision.recommend_cot:
            print(f"🧠 STRATEGY: {decision.reason} (Injecting CoT)")
            print(f"   Template: \"{decision.cot_template}\"")
        else:
            print(f"⏩ STRATEGY: Direct Generation ({decision.reason})")

        print(f"\n🏆 TOP RECOMMENDATIONS (Benchmarked vs {baseline.name})")
        print(f"{'-'*60}")
        
        for res in ranked:
            print(f"#{res.rank} {res.model_name}")
            print(f"   └─ {res.reasoning}")
            print(f"   └─ Chebyshev Score: {res.score:.4f}")
            print("")

    return ranked


def get_recommendations_for_use_case(
    use_case: UseCase,
    baseline_model_name: Optional[str] = None,
    cache_path: Optional[Union[str, Path]] = None,
    top_k: int = 3,
    verbose: bool = True,
) -> List[RecommendationResult]:
    """
    Get model recommendations optimized for a specific use case.
    
    This is a simplified API that selects optimal models based on predefined
    use case configurations. Each use case has tuned optimization weights
    for quality, cost, latency, hallucination, and refusal rates.
    
    Args:
        use_case: The UseCase enum value specifying the task type
        baseline_model_name: Name of baseline model for comparison.
            If None, uses DEFAULT_BASELINE_MODEL.
        cache_path: Path to custom model cache file. If None, uses default.
        top_k: Number of top models to return
        verbose: Whether to print detailed output
        
    Returns:
        List of top-k RecommendationResult objects optimized for the use case
        
    Example:
        >>> from llm_jury import get_recommendations_for_use_case, UseCase
        
        # Get best models for code generation
        >>> results = get_recommendations_for_use_case(UseCase.CODE_GENERATION)
        
        # Get cost-optimized models
        >>> results = get_recommendations_for_use_case(
        ...     UseCase.COST_OPTIMIZED,
        ...     top_k=5
        ... )
        
        # Get models for legal review (high reliability)
        >>> results = get_recommendations_for_use_case(
        ...     UseCase.LEGAL_REVIEW,
        ...     baseline_model_name="GPT-4o"
        ... )
        
        # List all available use cases
        >>> for uc in UseCase:
        ...     print(f"{uc.name}: {USE_CASE_CONFIGS[uc].description}")
    """
    # Get configuration for this use case
    config = USE_CASE_CONFIGS[use_case]
    
    # 1. Setup - Load from cache
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=verbose)
    
    if not registry:
        if verbose:
            print("⚠️  No models loaded from cache.")
        return []
    
    # Use provided baseline or default
    baseline_name = baseline_model_name or DEFAULT_BASELINE_MODEL
    baseline = next((m for m in registry if m.name == baseline_name), None)
    
    if baseline is None:
        if verbose:
            print(f"⚠️  Baseline model '{baseline_name}' not found. Using fallback.")
        baseline = registry[0]
    
    # Load raw model data for quality scorer
    raw_models_data = ModelRegistry.load_raw_cache(cache_path=cache_path)
    
    if not raw_models_data:
        if verbose:
            print("⚠️  Could not load raw model data for quality scoring.")
        return []
    
    # 2. Create optimizer with use case-specific settings
    optimizer = Optimizer(
        baseline,
        all_models_data=raw_models_data,
        strategy=config.strategy,
        custom_weights=config.weights,
        quality_range=config.quality_range,
        cost_range=config.cost_range,
        speed_range=config.speed_range,
    )
    
    # 3. Create a routing decision based on the use case's category
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,  # Use all high-quality models
        category=config.prompt_category,
        reason=f"Use case: {use_case.value}",
        recommend_cot=False,
        cot_template="",
    )
    
    # 4. Filter to models with pricing
    candidates = [
        m for m in registry
        if m.input_cost_per_m is not None
        and m.output_cost_per_m is not None
    ]
    
    if not candidates:
        if verbose:
            print("\n⚠️  No models found with pricing data")
        return []
    
    # 5. Rank
    ranked = optimizer.rank(candidates, decision, top_k=top_k)
    
    # 6. Output (if verbose)
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 USE CASE: {use_case.name}")
        print(f"📋 {config.description}")
        print(f"{'='*60}")
        print(f"⚙️  Strategy: {config.strategy.name}")
        print(f"📊 Weights: quality={config.weights['quality']:.0%}, "
              f"cost={config.weights['cost']:.0%}, "
              f"halluc={config.weights['hallucination']:.0%}")
        
        print(f"\n🏆 TOP {top_k} MODELS (vs {baseline.name})")
        print(f"{'-'*60}")
        
        for res in ranked:
            print(f"#{res.rank} {res.model_name}")
            print(f"   └─ {res.reasoning}")
            print(f"   └─ Score: {res.score:.4f}")
            print("")
    
    return ranked


def list_use_cases() -> Dict[str, str]:
    """
    List all available use cases with their descriptions.
    
    Returns:
        Dictionary mapping use case names to descriptions
        
    Example:
        >>> from llm_jury import list_use_cases
        >>> for name, desc in list_use_cases().items():
        ...     print(f"{name}: {desc}")
    """
    return {
        uc.name: USE_CASE_CONFIGS[uc].description
        for uc in UseCase
    }


def get_use_case_config(use_case: UseCase) -> UseCaseConfig:
    """
    Get the configuration details for a specific use case.
    
    Args:
        use_case: The UseCase enum value
        
    Returns:
        UseCaseConfig with strategy, weights, and constraints
        
    Example:
        >>> from llm_jury import get_use_case_config, UseCase
        >>> config = get_use_case_config(UseCase.CODE_GENERATION)
        >>> print(config.weights)
        {'quality': 0.45, 'cost': 0.15, ...}
    """
    return USE_CASE_CONFIGS[use_case]


def get_best_models_for_budget(
    max_budget: float,
    quality_weight: float = 0.6,
    latency_weight: float = 0.4,
    use_input_cost: bool = True,
    use_output_cost: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    verbose: bool = True,
    open_source_only: bool = False,
    baseline_model_name: Optional[str] = "Gemini 3 Pro Preview (high)",
) -> List[RecommendationResult]:
    """
    Find the best models within a budget constraint.
    
    This is a simple, intuitive way to find models: "I have $X to spend per 
    million tokens, give me the best quality and fastest models within that budget."
    
    Models are filtered by budget, then ranked by a weighted combination of
    quality (higher is better) and latency (lower is better).
    
    Args:
        max_budget: Maximum cost in $/1M tokens (blended input+output).
            Examples: 1.0 = $1/M tokens, 0.5 = $0.50/M tokens
        quality_weight: Weight for quality in ranking (0-1, default: 0.6)
        latency_weight: Weight for latency in ranking (0-1, default: 0.4)
        use_input_cost: Include input cost in budget calculation (default: True)
        use_output_cost: Include output cost in budget calculation (default: True)
        cache_path: Path to custom model cache file. If None, uses default.
        top_k: Number of top models to return (default: 5)
        verbose: Whether to print detailed output (default: True)
        open_source_only: If True, only consider open source models (default: False)
        baseline_model_name: Name of reference model for quality comparison.
            Default is "Gemini 3 Pro Preview (high)". Use list_available_models() 
            to see options. Common choices: "GPT-4.1", "GPT-4o (Nov '24)".
        
    Returns:
        List of top-k RecommendationResult objects within budget, ranked by
        quality and latency.
        
    Example:
        >>> from llm_jury import get_best_models_for_budget
        
        # Find best models under $1/M tokens
        >>> results = get_best_models_for_budget(max_budget=1.0)
        
        # Find best open source models under $0.50/M tokens
        >>> results = get_best_models_for_budget(max_budget=0.50, open_source_only=True)
        
        # Find best models with GPT-4.1 as baseline reference
        >>> results = get_best_models_for_budget(
        ...     max_budget=7.5,
        ...     baseline_model_name="GPT-4.1"
        ... )
        
        # Find best models under $5/M tokens
        >>> results = get_best_models_for_budget(max_budget=5.0, top_k=10)
    """
    # Normalize weights
    total_weight = quality_weight + latency_weight
    q_weight = quality_weight / total_weight
    l_weight = latency_weight / total_weight
    
    # Load models
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=verbose)
    
    if not registry:
        if verbose:
            print("⚠️  No models loaded from cache.")
        return []
    
    # Load raw data for quality scoring
    raw_models_data = ModelRegistry.load_raw_cache(cache_path=cache_path)
    
    if not raw_models_data:
        if verbose:
            print("⚠️  Could not load raw model data for quality scoring.")
        return []
    
    # Filter models by budget
    def get_blended_cost(m: ModelMetadata) -> Optional[float]:
        """Calculate blended cost ($/1M tokens)."""
        input_cost = m.input_cost_per_m if use_input_cost else 0
        output_cost = m.output_cost_per_m if use_output_cost else 0
        
        if input_cost is None or output_cost is None:
            return None
        
        # Standard blending: 75% input, 25% output (typical token ratio)
        return input_cost * 0.75 + output_cost * 0.25
    
    within_budget = []
    for m in registry:
        # Skip non-open-source if filter is enabled
        if open_source_only and not is_open_source(m.name):
            continue
        cost = get_blended_cost(m)
        if cost is not None and cost <= max_budget:
            within_budget.append((m, cost))
    
    if not within_budget:
        if verbose:
            # Find cheapest model for reference
            all_costs = [(m, get_blended_cost(m)) for m in registry]
            valid_costs = [(m, c) for m, c in all_costs if c is not None]
            if valid_costs:
                cheapest = min(valid_costs, key=lambda x: x[1])
                print(f"\n⚠️  No models found under ${max_budget:.2f}/M tokens")
                print(f"   Cheapest available: {cheapest[0].name} at ${cheapest[1]:.2f}/M")
            else:
                print(f"\n⚠️  No models found with pricing data")
        return []
    
    if verbose:
        print(f"\n💰 BUDGET CONSTRAINT: ${max_budget:.2f}/M tokens")
        if baseline_model_name:
            print(f"📊 BASELINE: {baseline_model_name}")
        if open_source_only:
            print(f"🔓 Open Source Only: {len(within_budget)} models within budget")
        else:
            print(f"   Found {len(within_budget)} models within budget")
    
    # Get quality scores using QualityScorer
    from llm_jury.ranking.quality_scorer import QualityScorer
    scorer = QualityScorer(all_models_data=raw_models_data)
    
    # Build raw data lookup
    raw_data_by_name = {d.get('name'): d for d in raw_models_data}
    
    # Get baseline model quality for comparison
    baseline_quality = None
    baseline_cost = None
    if baseline_model_name:
        baseline_data = raw_data_by_name.get(baseline_model_name)
        if baseline_data:
            baseline_quality = scorer.calculate_quality_score(baseline_data, PromptCategory.GENERAL)
            # Find baseline cost
            for m in registry:
                if m.name == baseline_model_name:
                    baseline_cost = get_blended_cost(m)
                    break
    
    # Score each model
    scored_models = []
    for m, cost in within_budget:
        # Get quality score
        model_data = raw_data_by_name.get(m.name, {'name': m.name})
        quality = scorer.calculate_quality_score(model_data, PromptCategory.GENERAL)
        
        # Get latency (TTFT in seconds, lower is better)
        latency = 1.0  # default
        if hasattr(m, 'measured_ttft_seconds') and m.measured_ttft_seconds:
            latency = m.measured_ttft_seconds
        elif hasattr(m, 'time_to_first_token_seconds') and m.time_to_first_token_seconds:
            latency = m.time_to_first_token_seconds
        elif m.median_latency_ms:
            latency = m.median_latency_ms / 1000.0
        
        scored_models.append({
            'model': m,
            'cost': cost,
            'quality': quality,
            'latency': latency,
        })
    
    # Normalize scores for ranking
    if scored_models:
        max_quality = max(s['quality'] for s in scored_models)
        min_quality = min(s['quality'] for s in scored_models)
        max_latency = max(s['latency'] for s in scored_models)
        min_latency = min(s['latency'] for s in scored_models)
        
        quality_range = max_quality - min_quality if max_quality > min_quality else 1
        latency_range = max_latency - min_latency if max_latency > min_latency else 1
        
        for s in scored_models:
            # Normalize quality: higher is better (0-1)
            norm_quality = (s['quality'] - min_quality) / quality_range
            
            # Normalize latency: lower is better, so invert (0-1)
            norm_latency = 1.0 - (s['latency'] - min_latency) / latency_range
            
            # Combined score (higher is better)
            s['combined_score'] = q_weight * norm_quality + l_weight * norm_latency
    
    # Sort by combined score (higher is better)
    scored_models.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Take top_k
    top_models = scored_models[:top_k]
    
    # Build results
    results = []
    for rank, s in enumerate(top_models, 1):
        m = s['model']
        
        # Build reasoning string
        reasoning = (
            f"Quality: {s['quality']:.1f} | "
            f"TTFT: {s['latency']*1000:.0f}ms | "
            f"Cost: ${s['cost']:.2f}/M tokens"
        )
        
        result = RecommendationResult(
            rank=rank,
            model_name=m.name,
            score=1.0 - s['combined_score'],  # Convert to "lower is better" for consistency
            reasoning=reasoning,
        )
        results.append(result)
    
    # Output
    if verbose:
        print(f"📊 Ranking by: {q_weight:.0%} quality + {l_weight:.0%} speed")
        print(f"\n{'='*60}")
        print(f"🏆 TOP {len(results)} MODELS WITHIN BUDGET")
        print(f"{'-'*60}")
        
        for res in results:
            s = top_models[res.rank - 1]
            print(f"#{res.rank} {res.model_name}")
            
            # Show quality relative to baseline if available
            quality_str = f"Quality: {s['quality']:.1f}"
            if baseline_quality:
                quality_pct = (s['quality'] / baseline_quality) * 100
                quality_str += f" ({quality_pct:.0f}% of {baseline_model_name})"
            
            # Show cost savings vs baseline if available
            cost_str = f"Cost: ${s['cost']:.2f}/M"
            if baseline_cost:
                savings = ((baseline_cost - s['cost']) / baseline_cost) * 100
                cost_str += f" ({savings:.0f}% cheaper than baseline)"
            
            print(f"   └─ {quality_str} | TTFT: {s['latency']*1000:.0f}ms")
            print(f"   └─ {cost_str}")
            print("")
    
    return results


@dataclass
class ValueRecommendation:
    """
    A value-optimized model recommendation.
    
    Extends RecommendationResult with additional context about
    the detected use case and value analysis.
    """
    rank: int
    model_name: str
    score: float
    reasoning: str
    
    # Value analysis
    quality_score: float  # 0-100
    cost_per_m: float     # Blended $/1M tokens
    value_ratio: float    # Quality per dollar (higher is better)
    
    # Constraint matching (vs baseline)
    quality_ratio: float  # Quality as ratio of baseline (e.g., 0.85 = 85%)
    cost_ratio: float     # Cost as ratio of baseline (e.g., 0.25 = 25%)
    speed_ratio: float    # Speed as ratio of baseline (e.g., 1.2 = 20% faster)
    meets_quality: bool   # True if quality >= min_quality_ratio
    meets_cost: bool      # True if cost <= max_cost_ratio
    meets_speed: bool     # True if speed meets latency constraint
    meets_all: bool       # True if all constraints are met
    
    # Use case context
    detected_use_case: str
    use_case_confidence: float
    use_case_description: str
    
    # Alternative use cases detected
    alternative_use_cases: Optional[List[str]] = None


def get_value_recommendations(
    prompt: str,
    baseline_model_name: Optional[str] = None,
    cache_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    verbose: bool = True,
    # Relative constraints (require baseline)
    min_quality_ratio: Optional[float] = None,
    max_cost_ratio: Optional[float] = None,
    max_latency_ratio: Optional[float] = None,
    # Absolute constraints (no baseline needed)
    min_quality_score: Optional[float] = None,
    max_absolute_cost: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    # Flags
    latency_flexible: bool = False,
    open_source_only: bool = False,
) -> Tuple[List[ValueRecommendation], ClassificationResult]:
    """
    Get value-optimized model recommendations based on prompt analysis.
    
    This is the main "smart" entry point that:
    1. Analyzes the prompt to detect the use case (coding, data analysis, etc.)
    2. Applies use-case-specific optimization weights
    3. Returns models ranked by value (quality vs cost)
    
    Two Modes of Operation:
    
    1. **With Baseline** (default): Compare models relative to a reference model
       - Use `min_quality_ratio`, `max_cost_ratio`, `max_latency_ratio`
       - Example: "Find models with 90% of GPT-4's quality at 50% the cost"
       
    2. **Without Baseline** (`baseline_model_name=None`): Use absolute targets
       - Use `min_quality_score`, `max_absolute_cost`, `max_latency_ms`
       - Example: "Find models with quality ≥80, cost ≤$1/M, latency ≤500ms"
    
    You can mix relative and absolute constraints. When both are specified
    for the same dimension (e.g., both `max_cost_ratio` and `max_absolute_cost`),
    BOTH constraints must be satisfied.
    
    Args:
        prompt: Natural language prompt describing your task
        baseline_model_name: Model to compare against. Set to None to use only
            absolute constraints without a baseline comparison.
        cache_path: Path to custom model cache file
        top_k: Number of models to return (default: 5)
        verbose: Whether to print detailed output
        
        # Relative constraints (percentage of baseline):
        min_quality_ratio: Minimum quality as ratio of baseline (0-1, e.g., 0.90 = 90%)
        max_cost_ratio: Maximum cost as ratio of baseline (0-1, e.g., 0.50 = 50%)
        max_latency_ratio: Maximum latency as ratio of baseline (e.g., 1.0 = same speed)
        
        # Absolute constraints (fixed targets):
        min_quality_score: Minimum absolute quality score (0-100 scale)
        max_absolute_cost: Maximum cost in $/M tokens (e.g., 1.0 = $1/M)
        max_latency_ms: Maximum latency in milliseconds (e.g., 500 = 500ms TTFT)
        
        # Flags:
        latency_flexible: If True, ignore all latency constraints (default: False)
        open_source_only: If True, only consider open source models (default: False)
        
    Returns:
        Tuple of:
        - List of ValueRecommendation objects
        - ClassificationResult with detected use case info
        
    Example:
        >>> from llm_jury import get_value_recommendations
        
        # MODE 1: With baseline - relative constraints
        >>> results, _ = get_value_recommendations(
        ...     "Write a complex algorithm",
        ...     baseline_model_name="Gemini 3 Pro Preview (high)",
        ...     min_quality_ratio=0.90,      # At least 90% of baseline quality
        ...     max_cost_ratio=0.30,         # At most 30% of baseline cost
        ... )
        
        # MODE 2: No baseline - absolute targets only
        >>> results, _ = get_value_recommendations(
        ...     "Analyze financial data",
        ...     baseline_model_name=None,    # No baseline comparison
        ...     min_quality_score=80,        # Quality ≥80 (0-100 scale)
        ...     max_absolute_cost=1.0,       # Cost ≤$1/M tokens
        ...     max_latency_ms=500,          # TTFT ≤500ms
        ... )
        
        # MODE 3: Mixed - both relative and absolute
        >>> results, _ = get_value_recommendations(
        ...     "Write production code",
        ...     baseline_model_name="GPT-4o",
        ...     min_quality_ratio=0.85,      # ≥85% of GPT-4o quality
        ...     max_absolute_cost=2.0,       # But hard cap at $2/M tokens
        ... )
    """
    # Determine if we're in baseline mode or absolute mode
    use_baseline = baseline_model_name is not None
    
    # Handle constraint defaults based on mode
    if use_baseline:
        # With baseline: default to relative constraints
        if min_quality_ratio is None and min_quality_score is None:
            min_quality_ratio = 0.70  # Default: 70% of baseline quality
        if max_cost_ratio is None and max_absolute_cost is None:
            max_cost_ratio = 0.50  # Default: 50% of baseline cost
        if max_latency_ratio is None and max_latency_ms is None and not latency_flexible:
            max_latency_ratio = 1.0  # Default: same speed as baseline
    else:
        # Without baseline: must have at least one constraint specified
        has_any_constraint = any([
            min_quality_score is not None,
            max_absolute_cost is not None,
            max_latency_ms is not None,
        ])
        if not has_any_constraint:
            # Set sensible defaults for absolute mode
            min_quality_score = 70  # Quality ≥70 (decent models)
            max_absolute_cost = 5.0  # Cost ≤$5/M tokens
    
    # 1. Classify the prompt
    classifier = PromptClassifier()
    classification = classifier.classify(prompt)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📝 PROMPT: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
        print(f"{'='*60}")
        print(f"🎯 DETECTED USE CASE: {classification.use_case.upper()}")
        print(f"   Confidence: {classification.confidence:.0%}")
        print(f"   Category: {classification.category.value}")
        print(f"   Description: {classifier.get_use_case_description(classification.use_case)}")
        
        if classification.alternative_use_cases:
            alts = ", ".join([f"{uc} ({conf:.0%})" for uc, conf in classification.alternative_use_cases])
            print(f"   Alternatives: {alts}")
        
        # Show mode
        if use_baseline:
            print(f"📊 MODE: Relative to baseline ({baseline_model_name})")
        else:
            print(f"📊 MODE: Absolute targets (no baseline)")
    
    # 2. Map to UseCase enum
    use_case_name = classification.use_case.upper()
    try:
        use_case = UseCase[use_case_name]
    except KeyError:
        # Fallback to GENERAL_QA if use case not in enum
        use_case = UseCase.GENERAL_QA
        if verbose:
            print(f"⚠️  UseCase '{use_case_name}' not found, using GENERAL_QA")
    
    # 3. Get use case config
    config = USE_CASE_CONFIGS[use_case]
    
    # 4. Load models
    registry = ModelRegistry.load_cache(cache_path=cache_path, verbose=False)
    
    if not registry:
        if verbose:
            print("⚠️  No models loaded from cache.")
        return [], classification
    
    # Get baseline (if using baseline mode)
    baseline = None
    if use_baseline:
        baseline = next((m for m in registry if m.name == baseline_model_name), None)
        
        if baseline is None:
            if verbose:
                print(f"⚠️  Baseline '{baseline_model_name}' not found, using fallback")
            baseline = registry[0]
    
    # Load raw data for quality scoring
    raw_models_data = ModelRegistry.load_raw_cache(cache_path=cache_path)
    
    if not raw_models_data:
        if verbose:
            print("⚠️  Could not load raw model data.")
        return [], classification
    
    # 5. Create value-focused optimizer
    # Adjust weights to emphasize value (quality and cost)
    value_weights = config.weights.copy()
    value_weights["quality"] = max(value_weights["quality"], 0.35)
    value_weights["cost"] = max(value_weights["cost"], 0.25)
    
    # Normalize weights
    total = sum(value_weights.values())
    value_weights = {k: v/total for k, v in value_weights.items()}
    
    # For optimizer, we need a baseline. If none provided, use first model as reference
    # (the actual filtering will use absolute constraints)
    optimizer_baseline = baseline if baseline else registry[0]
    
    # Determine speed constraint (relative)
    speed_range = None
    if not latency_flexible and max_latency_ratio is not None:
        min_speed_ratio = 1.0 / max_latency_ratio
        speed_range = (min_speed_ratio, float('inf'))
    
    # Build relative constraint ranges (only if using baseline)
    relative_quality_range = None
    relative_cost_range = None
    
    if use_baseline:
        if min_quality_ratio is not None:
            relative_quality_range = (min_quality_ratio, 1.0)
        if max_cost_ratio is not None:
            relative_cost_range = (0.0, max_cost_ratio)
    
    optimizer = Optimizer(
        optimizer_baseline,
        all_models_data=raw_models_data,
        strategy=OptimizationStrategy.VALUE_OPTIMIZED,
        custom_weights=value_weights,
        quality_range=relative_quality_range,
        cost_range=relative_cost_range,
        speed_range=speed_range,
    )
    
    # 6. Create routing decision based on use case
    decision = RoutingDecision(
        archetype=ProductArchetype.FRONTIER,  # Consider all models
        category=config.prompt_category,
        reason=f"Auto-detected: {classification.use_case}",
        recommend_cot=False,
        cot_template="",
    )
    
    # 7. Filter to models with pricing (and optionally open source only)
    candidates = [
        m for m in registry
        if m.input_cost_per_m is not None
        and m.output_cost_per_m is not None
        and (not open_source_only or is_open_source(m.name))
    ]
    
    if not candidates:
        if verbose:
            if open_source_only:
                print("\n⚠️  No open source models found with pricing data")
            else:
                print("\n⚠️  No models found with pricing data")
        return [], classification
    
    if verbose and open_source_only:
        print(f"🔓 Open Source Only: {len(candidates)} models")
    
    # Helper functions for filtering
    def get_blended_cost(m: ModelMetadata) -> float:
        """Calculate blended cost ($/1M tokens) - 75% input, 25% output."""
        return (m.input_cost_per_m or 0) * 0.75 + (m.output_cost_per_m or 0) * 0.25
    
    def get_latency_ms(m: ModelMetadata) -> float:
        """Get latency in milliseconds."""
        if hasattr(m, 'measured_ttft_seconds') and m.measured_ttft_seconds:
            return m.measured_ttft_seconds * 1000
        if hasattr(m, 'time_to_first_token_seconds') and m.time_to_first_token_seconds:
            return m.time_to_first_token_seconds * 1000
        if m.median_latency_ms:
            return m.median_latency_ms
        return 1000.0  # Default 1 second
    
    # 7b. Apply absolute cost filter if specified
    if max_absolute_cost is not None:
        candidates_before = len(candidates)
        candidates = [m for m in candidates if get_blended_cost(m) <= max_absolute_cost]
        
        if verbose:
            print(f"💰 Absolute Cost Filter: ≤${max_absolute_cost:.2f}/M tokens → {len(candidates)}/{candidates_before} models")
        
        if not candidates:
            if verbose:
                print(f"\n⚠️  No models found under ${max_absolute_cost:.2f}/M tokens")
                all_with_pricing = [m for m in registry if m.input_cost_per_m is not None]
                if all_with_pricing:
                    cheapest = min(all_with_pricing, key=get_blended_cost)
                    print(f"   Cheapest available: {cheapest.name} at ${get_blended_cost(cheapest):.2f}/M")
            return [], classification
    
    # 7c. Apply absolute latency filter if specified
    if max_latency_ms is not None and not latency_flexible:
        candidates_before = len(candidates)
        candidates = [m for m in candidates if get_latency_ms(m) <= max_latency_ms]
        
        if verbose:
            print(f"⏱️  Absolute Latency Filter: ≤{max_latency_ms:.0f}ms → {len(candidates)}/{candidates_before} models")
        
        if not candidates:
            if verbose:
                print(f"\n⚠️  No models found with latency ≤{max_latency_ms:.0f}ms")
            return [], classification
    
    # 7d. Apply absolute quality filter if specified (after getting quality scores)
    # Note: This is done after ranking since quality scores require computation
    
    # 8. Rank models
    ranked = optimizer.rank(candidates, decision, top_k=top_k * 2, verbose=False)  # Get more for filtering
    
    # 9. Build value recommendations
    from llm_jury.ranking.quality_scorer import QualityScorer
    scorer = QualityScorer(all_models_data=raw_models_data)
    raw_data_by_name = {d.get('name'): d for d in raw_models_data}
    
    # Calculate baseline metrics for constraint comparison (if using baseline)
    baseline_quality = 100.0  # Default for absolute mode
    baseline_cost = 10.0      # Default for absolute mode
    baseline_latency = 1.0    # Default for absolute mode
    
    if use_baseline and baseline:
        baseline_data = raw_data_by_name.get(baseline.name, {'name': baseline.name})
        baseline_quality = scorer.calculate_quality_score(baseline_data, config.prompt_category)
        baseline_cost = (baseline.input_cost_per_m or 0) * 0.75 + (baseline.output_cost_per_m or 0) * 0.25
        baseline_latency = baseline.measured_ttft_seconds or (baseline.median_latency_ms / 1000 if baseline.median_latency_ms else 1.0)
    
    # Determine speed constraint threshold
    min_speed_ratio = 1.0 / max_latency_ratio if (max_latency_ratio and not latency_flexible) else 0.0
    
    value_results = []
    for res in ranked[:top_k]:
        # Find model metadata
        model = next((m for m in candidates if m.name == res.model_name), None)
        if not model:
            continue
        
        # Get quality score
        model_data = raw_data_by_name.get(model.name, {'name': model.name})
        quality = scorer.calculate_quality_score(model_data, config.prompt_category)
        
        # Apply absolute quality filter if specified
        if min_quality_score is not None and quality < min_quality_score:
            continue  # Skip models below quality threshold
        
        # Calculate blended cost
        cost = (model.input_cost_per_m or 0) * 0.75 + (model.output_cost_per_m or 0) * 0.25
        
        # Calculate latency (in seconds for ratio, ms for display)
        model_latency = model.measured_ttft_seconds or (model.median_latency_ms / 1000 if model.median_latency_ms else 1.0)
        model_latency_ms = model_latency * 1000
        
        # Calculate ratios vs baseline (meaningful only in baseline mode)
        quality_ratio = quality / baseline_quality if baseline_quality > 0 else 1.0
        cost_ratio = cost / baseline_cost if baseline_cost > 0 else 1.0
        speed_ratio = baseline_latency / model_latency if model_latency > 0 else 1.0  # Higher = faster
        
        # Check constraints - handle both relative and absolute
        # Quality: relative OR absolute
        meets_quality = True
        if min_quality_ratio is not None and use_baseline:
            meets_quality = meets_quality and (quality_ratio >= min_quality_ratio)
        if min_quality_score is not None:
            meets_quality = meets_quality and (quality >= min_quality_score)
        
        # Cost: relative OR absolute
        meets_cost = True
        if max_cost_ratio is not None and use_baseline:
            meets_cost = meets_cost and (cost_ratio <= max_cost_ratio)
        if max_absolute_cost is not None:
            meets_cost = meets_cost and (cost <= max_absolute_cost)
        
        # Speed/Latency: relative OR absolute
        meets_speed = True
        if not latency_flexible:
            if max_latency_ratio is not None and use_baseline:
                meets_speed = meets_speed and (speed_ratio >= min_speed_ratio)
            if max_latency_ms is not None:
                meets_speed = meets_speed and (model_latency_ms <= max_latency_ms)
        
        meets_all = meets_quality and meets_cost and meets_speed
        
        # Value ratio: quality per dollar (higher is better)
        # Free models (cost=0) get maximum value - use a large multiplier
        value_ratio = quality / cost if cost > 0 else quality * 1000
        
        value_rec = ValueRecommendation(
            rank=len(value_results) + 1,
            model_name=model.name,
            score=res.score,
            reasoning=res.reasoning,
            quality_score=quality,
            cost_per_m=cost,
            value_ratio=value_ratio,
            quality_ratio=quality_ratio,
            cost_ratio=cost_ratio,
            speed_ratio=speed_ratio,
            meets_quality=meets_quality,
            meets_cost=meets_cost,
            meets_speed=meets_speed,
            meets_all=meets_all,
            detected_use_case=classification.use_case,
            use_case_confidence=classification.confidence,
            use_case_description=classifier.get_use_case_description(classification.use_case),
            alternative_use_cases=[uc for uc, _ in classification.alternative_use_cases] if classification.alternative_use_cases else None,
        )
        value_results.append(value_rec)
    
    # 10. Output
    if verbose:
        print(f"\n💰 VALUE OPTIMIZATION")
        
        # Build constraint description
        constraint_parts = []
        
        # Quality constraints
        if min_quality_ratio is not None and use_baseline:
            constraint_parts.append(f"Quality ≥{min_quality_ratio:.0%} baseline")
        if min_quality_score is not None:
            constraint_parts.append(f"Quality ≥{min_quality_score:.0f}")
        if not constraint_parts:
            constraint_parts.append("Quality: any")
        
        # Cost constraints
        cost_parts = []
        if max_cost_ratio is not None and use_baseline:
            cost_parts.append(f"≤{max_cost_ratio:.0%} baseline")
        if max_absolute_cost is not None:
            cost_parts.append(f"≤${max_absolute_cost:.2f}/M")
        if cost_parts:
            constraint_parts.append(f"Cost {' AND '.join(cost_parts)}")
        
        # Latency constraints
        if latency_flexible:
            constraint_parts.append("Latency: flexible")
        else:
            latency_parts = []
            if max_latency_ratio is not None and use_baseline:
                latency_parts.append(f"≤{max_latency_ratio:.0%} baseline")
            if max_latency_ms is not None:
                latency_parts.append(f"≤{max_latency_ms:.0f}ms")
            if latency_parts:
                constraint_parts.append(f"Latency {' AND '.join(latency_parts)}")
        
        print(f"   Constraints: {', '.join(constraint_parts)}")
        print(f"   Weights: quality={value_weights['quality']:.0%}, cost={value_weights['cost']:.0%}")
        
        if use_baseline and baseline:
            print(f"\n🏆 TOP {len(value_results)} VALUE MODELS (vs {baseline.name})")
        else:
            print(f"\n🏆 TOP {len(value_results)} VALUE MODELS (absolute targets)")
        print(f"{'-'*70}")
        
        for rec in value_results:
            # Constraint status indicators
            q_check = "✓" if rec.meets_quality else "✗"
            c_check = "✓" if rec.meets_cost else "✗"
            s_check = "✓" if rec.meets_speed else "✗"
            status = f"[{q_check}Q {c_check}C {s_check}S]"
            all_met = "✓ ALL MET" if rec.meets_all else ""
            
            print(f"#{rec.rank} {rec.model_name} {status} {all_met}")
            
            # Show quality - include baseline comparison only if using baseline
            if use_baseline:
                print(f"   └─ Quality: {rec.quality_score:.1f} ({rec.quality_ratio:.0%} of baseline)")
            else:
                print(f"   └─ Quality: {rec.quality_score:.1f}/100")
            
            # Show cost
            if use_baseline:
                print(f"   └─ Cost: ${rec.cost_per_m:.2f}/M ({rec.cost_ratio:.0%} of baseline)")
            else:
                print(f"   └─ Cost: ${rec.cost_per_m:.2f}/M tokens")
            
            # Show speed/latency
            latency_ms = 1000 / rec.speed_ratio if rec.speed_ratio > 0 else 0  # Approximate
            if use_baseline:
                print(f"   └─ Speed: {rec.speed_ratio:.1f}x baseline")
            else:
                # In absolute mode, show actual latency
                model = next((m for m in candidates if m.name == rec.model_name), None)
                if model:
                    actual_latency_ms = get_latency_ms(model)
                    print(f"   └─ Latency: {actual_latency_ms:.0f}ms TTFT")
            
            print(f"   └─ Value Ratio: {rec.value_ratio:.1f} (quality/$)")
            print("")
    
    return value_results, classification


def analyze_prompt(prompt: str, verbose: bool = True) -> ClassificationResult:
    """
    Analyze a prompt to understand its use case without making recommendations.
    
    Useful for understanding what category a prompt falls into before
    deciding on optimization strategy.
    
    Args:
        prompt: Natural language prompt to analyze
        verbose: Whether to print analysis details
        
    Returns:
        ClassificationResult with use case, confidence, and signals
        
    Example:
        >>> from llm_jury import analyze_prompt
        >>> result = analyze_prompt("Write a Python function to sort a list")
        >>> print(result.use_case)  # "code_generation"
        >>> print(result.confidence)  # 0.95
    """
    classifier = PromptClassifier()
    result = classifier.classify(prompt)
    
    if verbose:
        print(f"\n📝 PROMPT ANALYSIS")
        print(f"{'='*60}")
        print(f"Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"")
        print(f"\n🎯 Classification:")
        print(f"   Use Case: {result.use_case.upper()}")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   Category: {result.category.value}")
        print(f"   Description: {classifier.get_use_case_description(result.use_case)}")
        
        print(f"\n🔍 Signals detected:")
        for signal in result.signals:
            print(f"   • {signal}")
        
        if result.alternative_use_cases:
            print(f"\n🔄 Alternative classifications:")
            for uc, conf in result.alternative_use_cases:
                print(f"   • {uc}: {conf:.0%}")
    
    return result
