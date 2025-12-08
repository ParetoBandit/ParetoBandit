"""Routing module for prompt classification and model selection."""

from llm_jury.routing.prompt_router import PromptRouter
from llm_jury.routing.archetype_router import ArchetypeRouter
from llm_jury.routing.prompt_classifier import (
    PromptClassifier,
    ClassificationResult,
    UseCaseCategory,
    classify_prompt,
)
from llm_jury.routing.use_case_router import (
    UseCaseRouter,
    UseCaseRoutingResult,
    UseCaseConstraintMapping,
    USE_CASE_CONSTRAINTS,
    route_prompt,
    get_use_case_constraints,
    detect_use_case,
    estimate_context_requirement,
)
from llm_jury.routing.hybrid_classifier import (
    HybridClassifier,
    HybridClassificationResult,
    HuggingFaceClassifier,
    HuggingFaceAPIClassifier,
    classify_prompt_hybrid,
    get_hybrid_classifier,
    benchmark_classifier,
    HF_LABEL_TO_USE_CASE,
    DEFAULT_ZS_LABELS,
)
from llm_jury.routing.complexity_classifier import (
    ComplexityClassifier,
    ComplexityResult,
    ComplexityLevel,
    HybridComplexityClassifier,
    HybridComplexityResult,
    HuggingFaceComplexityClassifier,
    classify_complexity,
    classify_complexity_hybrid,
)

__all__ = [
    # Original exports
    "PromptRouter",
    "ArchetypeRouter",
    "PromptClassifier",
    "ClassificationResult",
    "UseCaseCategory",
    "classify_prompt",
    # Use case-aware routing
    "UseCaseRouter",
    "UseCaseRoutingResult",
    "UseCaseConstraintMapping",
    "USE_CASE_CONSTRAINTS",
    "route_prompt",
    "get_use_case_constraints",
    "detect_use_case",
    "estimate_context_requirement",
    # Hybrid use case classifier (regex + HuggingFace)
    "HybridClassifier",
    "HybridClassificationResult",
    "HuggingFaceClassifier",
    "HuggingFaceAPIClassifier",
    "classify_prompt_hybrid",
    "get_hybrid_classifier",
    "benchmark_classifier",
    "HF_LABEL_TO_USE_CASE",
    "DEFAULT_ZS_LABELS",
    # Complexity classifier (regex + HuggingFace)
    "ComplexityClassifier",
    "ComplexityResult",
    "ComplexityLevel",
    "HybridComplexityClassifier",
    "HybridComplexityResult",
    "HuggingFaceComplexityClassifier",
    "classify_complexity",
    "classify_complexity_hybrid",
]
