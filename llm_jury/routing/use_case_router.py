"""
Use Case-Aware Router for LLM Model Selection.

Connects use case detection to model constraints and optimization for
intelligent end-to-end routing. This is the unified routing pipeline that:

1. Detects use case from prompt (what task type?)
2. Maps use case to constraints (what capabilities needed?)
3. Filters models by constraints (which models qualify?)
4. Optimizes ranking (which is best value?)

Architecture:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                           User Prompt                                │
    │  "Using the provided documents, answer questions about our policy"  │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              Stage 1: USE CASE DETECTION                            │
    │  PromptClassifier analyzes prompt → "rag_pipeline" (0.92 conf)      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              Stage 2: CONSTRAINT MAPPING                            │
    │  USE_CASE_CONSTRAINTS["rag_pipeline"] →                             │
    │    - min_context_k: 64                                              │
    │    - capabilities: [FUNCTION_CALLING]                               │
    │    - preferred: [LONG_CONTEXT]                                      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              Stage 3: MODEL FILTERING                               │
    │  apply_constraints(all_models, constraints)                         │
    │    51 models → 35 models pass constraints                           │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │              Stage 4: MULTI-OBJECTIVE OPTIMIZATION                  │
    │  Optimizer with USE_CASE_CONFIGS["rag_pipeline"] weights:           │
    │    quality=0.35, hallucination=0.30, cost=0.15, latency=0.10...    │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        TOP RECOMMENDATIONS                          │
    │  1. DeepSeek V3.1 (131K ctx, low halluc, good value)               │
    │  2. GPT-4o (128K ctx, strong RAG, reliable)                        │
    │  3. Claude 3.5 Sonnet (200K ctx, excellent quality)                │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from llm_jury.routing import UseCaseRouter, route_prompt
    
    # Quick usage
    result = route_prompt("Using these docs, answer my question about...")
    print(result.detected_use_case)  # "rag_pipeline"
    print(result.recommendations)     # [DeepSeek V3.1, GPT-4o, ...]
    
    # Advanced usage with custom context requirements
    router = UseCaseRouter()
    result = router.route(
        prompt="Analyze this 100K token document...",
        expected_context_tokens=100_000,  # Override context requirement
        require_capabilities=["vision"],   # Add extra requirements
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

from llm_jury.core.models import (
    ModelMetadata, 
    RoutingDecision, 
    PromptCategory, 
    ProductArchetype,
    RecommendationResult,
)
from llm_jury.routing.prompt_classifier import PromptClassifier, ClassificationResult
from llm_jury.ranking.constraints import (
    ConstraintConfig,
    CapabilityRequirement,
    apply_constraints,
    create_context_objective,
    get_model_context_k,
    check_model_capability,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Use Case to Constraint Mapping
# =============================================================================

@dataclass
class UseCaseConstraintMapping:
    """
    Maps a use case to its model constraints.
    
    This defines what capabilities a use case requires or prefers,
    enabling automatic model filtering based on detected intent.
    """
    min_context_k: int = 8                                    # Minimum context window (K tokens)
    target_context_k: Optional[int] = None                    # Target for soft bonus
    capabilities: List[CapabilityRequirement] = field(default_factory=list)      # Required
    preferred_capabilities: List[CapabilityRequirement] = field(default_factory=list)  # Nice-to-have
    max_input_cost_per_m: Optional[float] = None              # Budget constraint
    prefer_excess_context: bool = False                       # Bonus for context > target
    context_weight: float = 0.0                               # Weight for context objective (0 = don't add)
    
    def to_constraint_config(self) -> ConstraintConfig:
        """Convert to ConstraintConfig for filtering."""
        return ConstraintConfig(
            min_context_k=self.min_context_k,
            target_context_k=self.target_context_k or self.min_context_k,
            capabilities=self.capabilities,
            preferred_capabilities=self.preferred_capabilities,
            max_input_cost_per_m=self.max_input_cost_per_m,
            prefer_excess_context=self.prefer_excess_context,
        )


# Default constraint mappings for each use case
# These define what capabilities each use case typically needs
USE_CASE_CONSTRAINTS: Dict[str, UseCaseConstraintMapping] = {
    # =========================================================================
    # Development & Engineering
    # =========================================================================
    "code_generation": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],  # Most models can code
        preferred_capabilities=[CapabilityRequirement.JSON_MODE],
    ),
    
    "code_review": UseCaseConstraintMapping(
        min_context_k=64,  # Need to see full codebase context
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "code_refactoring": UseCaseConstraintMapping(
        min_context_k=64,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "technical_docs": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
    ),
    
    # =========================================================================
    # Data & Analytics
    # =========================================================================
    "data_analysis": UseCaseConstraintMapping(
        min_context_k=64,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "sql_generation": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
        preferred_capabilities=[CapabilityRequirement.JSON_MODE],
    ),
    
    "math_reasoning": UseCaseConstraintMapping(
        min_context_k=32,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    # =========================================================================
    # Content & Communication
    # =========================================================================
    "creative_writing": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
    ),
    
    "summarization": UseCaseConstraintMapping(
        min_context_k=64,  # Need to fit the content to summarize
        target_context_k=128,
        prefer_excess_context=True,
        context_weight=0.10,
    ),
    
    "translation": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
    ),
    
    # =========================================================================
    # Specialized Domains
    # =========================================================================
    "legal_review": UseCaseConstraintMapping(
        min_context_k=128,  # Legal docs are often long
        target_context_k=200,
        prefer_excess_context=True,
        context_weight=0.15,
    ),
    
    "financial_analysis": UseCaseConstraintMapping(
        min_context_k=64,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "research_assistant": UseCaseConstraintMapping(
        min_context_k=128,  # Research papers, multiple sources
        target_context_k=200,
        prefer_excess_context=True,
        context_weight=0.12,
    ),
    
    # =========================================================================
    # Conversational & Support
    # =========================================================================
    "customer_support": UseCaseConstraintMapping(
        min_context_k=16,  # Short conversations
        capabilities=[],
    ),
    
    "tutoring": UseCaseConstraintMapping(
        min_context_k=32,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "general_qa": UseCaseConstraintMapping(
        min_context_k=16,
        capabilities=[],  # No special requirements
    ),
    
    # =========================================================================
    # Technical Capabilities
    # =========================================================================
    "rag_pipeline": UseCaseConstraintMapping(
        min_context_k=64,           # Retrieved docs + prompt + response
        target_context_k=128,       # Ideal headroom
        capabilities=[],            # Function calling nice but not required
        preferred_capabilities=[
            CapabilityRequirement.FUNCTION_CALLING,
            CapabilityRequirement.LONG_CONTEXT,
        ],
        prefer_excess_context=True,
        context_weight=0.15,        # Context is important for RAG
    ),
    
    "function_calling": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[CapabilityRequirement.FUNCTION_CALLING],  # Required!
        preferred_capabilities=[CapabilityRequirement.JSON_MODE],
    ),
    
    "structured_extraction": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
        preferred_capabilities=[CapabilityRequirement.JSON_MODE],
    ),
    
    "long_context": UseCaseConstraintMapping(
        min_context_k=128,
        target_context_k=256,
        capabilities=[CapabilityRequirement.LONG_CONTEXT],
        preferred_capabilities=[CapabilityRequirement.VERY_LONG_CONTEXT],
        prefer_excess_context=True,
        context_weight=0.20,
    ),
    
    # =========================================================================
    # Cost/Performance Focused
    # =========================================================================
    "cost_optimized": UseCaseConstraintMapping(
        min_context_k=8,
        max_input_cost_per_m=2.0,  # Budget constraint
    ),
    
    "low_latency": UseCaseConstraintMapping(
        min_context_k=8,
        capabilities=[CapabilityRequirement.STREAMING],
    ),
    
    "maximum_quality": UseCaseConstraintMapping(
        min_context_k=64,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    # =========================================================================
    # Multimodal & Vision
    # =========================================================================
    "image_understanding": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[CapabilityRequirement.VISION],  # Required!
    ),
    
    "vision_qa": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[CapabilityRequirement.VISION],  # Required!
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    # =========================================================================
    # Embeddings & Similarity
    # =========================================================================
    "embeddings": UseCaseConstraintMapping(
        min_context_k=8,
        capabilities=[CapabilityRequirement.EMBEDDINGS],  # Required!
    ),
    
    "semantic_similarity": UseCaseConstraintMapping(
        min_context_k=8,
        capabilities=[CapabilityRequirement.EMBEDDINGS],  # Required!
    ),
    
    # =========================================================================
    # Agentic & Tool Use
    # =========================================================================
    "agent_workflow": UseCaseConstraintMapping(
        min_context_k=64,
        capabilities=[CapabilityRequirement.FUNCTION_CALLING],  # Required!
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    "tool_use": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[CapabilityRequirement.FUNCTION_CALLING],  # Required!
    ),
    
    "planning": UseCaseConstraintMapping(
        min_context_k=32,
        preferred_capabilities=[CapabilityRequirement.REASONING],
    ),
    
    # =========================================================================
    # Classification & Analysis
    # =========================================================================
    "text_classification": UseCaseConstraintMapping(
        min_context_k=16,
        max_input_cost_per_m=5.0,  # Often high volume
    ),
    
    "sentiment_analysis": UseCaseConstraintMapping(
        min_context_k=8,
        max_input_cost_per_m=5.0,  # Often high volume
    ),
    
    "entity_extraction": UseCaseConstraintMapping(
        min_context_k=16,
        preferred_capabilities=[CapabilityRequirement.JSON_MODE],
    ),
    
    "content_moderation": UseCaseConstraintMapping(
        min_context_k=8,
        max_input_cost_per_m=3.0,  # Very high volume
    ),
    
    # =========================================================================
    # Text Transformation
    # =========================================================================
    "paraphrasing": UseCaseConstraintMapping(
        min_context_k=16,
        capabilities=[],
    ),
    
    "style_transfer": UseCaseConstraintMapping(
        min_context_k=16,
        capabilities=[],
    ),
    
    "grammar_correction": UseCaseConstraintMapping(
        min_context_k=8,
        max_input_cost_per_m=3.0,  # Often batch
    ),
    
    # =========================================================================
    # Creative & Ideation
    # =========================================================================
    "brainstorming": UseCaseConstraintMapping(
        min_context_k=32,
        capabilities=[],
    ),
    
    "roleplay": UseCaseConstraintMapping(
        min_context_k=64,  # Need conversation history
        capabilities=[],
    ),
}


# =============================================================================
# Context Estimation
# =============================================================================

def estimate_context_requirement(
    prompt: str,
    use_case: str,
    system_prompt: Optional[str] = None,
    expected_response_tokens: int = 2000,
) -> int:
    """
    Estimate context requirement based on prompt and use case.
    
    Considers:
    - Prompt length
    - System prompt (if any)
    - Use case typical patterns
    - Expected response length
    - Safety margin (1.5x)
    
    Args:
        prompt: User prompt
        use_case: Detected use case
        system_prompt: Optional system prompt
        expected_response_tokens: Expected response size
        
    Returns:
        Estimated context requirement in K tokens
    """
    # Rough token estimation (4 chars ≈ 1 token)
    prompt_tokens = len(prompt) // 4
    system_tokens = len(system_prompt) // 4 if system_prompt else 0
    
    # Base requirement from prompt + system + response
    base_tokens = prompt_tokens + system_tokens + expected_response_tokens
    
    # Use case multipliers for typical patterns
    USE_CASE_MULTIPLIERS = {
        "rag_pipeline": 3.0,        # Retrieved docs typically 2-3x prompt
        "long_context": 5.0,        # Long docs are the input
        "summarization": 2.5,       # Document to summarize
        "legal_review": 3.0,        # Legal docs are verbose
        "research_assistant": 2.5,  # Multiple papers
        "code_review": 2.0,         # Codebase context
        "data_analysis": 2.0,       # Data descriptions
    }
    
    multiplier = USE_CASE_MULTIPLIERS.get(use_case, 1.5)
    
    # Apply multiplier and safety margin
    estimated_tokens = int(base_tokens * multiplier * 1.5)
    
    # Convert to K and round up to standard sizes
    estimated_k = max(8, (estimated_tokens // 1000) + 1)
    
    # Round to standard context sizes
    standard_sizes = [8, 16, 32, 64, 100, 128, 200, 256, 512, 1000]
    for size in standard_sizes:
        if estimated_k <= size:
            return size
    
    return estimated_k


# =============================================================================
# Routing Result
# =============================================================================

@dataclass
class UseCaseRoutingResult:
    """
    Complete result from use case-aware routing.
    
    Contains all information about the routing decision and 
    can be used to understand why certain models were recommended.
    """
    # Detection results
    detected_use_case: str
    use_case_confidence: float
    alternative_use_cases: List[Tuple[str, float]]
    detection_signals: List[str]
    
    # Constraint application
    applied_constraints: ConstraintConfig
    estimated_context_k: int
    models_before_filter: int
    models_after_filter: int
    
    # Routing decision (for archetype/category)
    routing_decision: RoutingDecision
    
    # Final recommendations (if optimization was run)
    recommendations: List[RecommendationResult] = field(default_factory=list)
    
    # Metadata
    prompt_preview: str = ""  # First 100 chars of prompt
    
    def summary(self) -> str:
        """Get human-readable summary of routing decision."""
        lines = [
            f"Use Case: {self.detected_use_case} ({self.use_case_confidence:.0%} confidence)",
            f"Context Required: {self.estimated_context_k}K tokens",
            f"Models: {self.models_after_filter}/{self.models_before_filter} passed constraints",
        ]
        
        if self.applied_constraints.capabilities:
            caps = [c.name for c in self.applied_constraints.capabilities]
            lines.append(f"Required Capabilities: {', '.join(caps)}")
        
        if self.recommendations:
            lines.append(f"Top Recommendation: {self.recommendations[0].model_name}")
        
        return "\n".join(lines)


# =============================================================================
# Use Case Router
# =============================================================================

class UseCaseRouter:
    """
    Unified router that connects use case detection to model selection.
    
    This is the main entry point for intelligent, use case-aware routing.
    It handles:
    
    1. Use case detection from prompt
    2. Automatic constraint mapping
    3. Context requirement estimation
    4. Model filtering by constraints
    5. (Optional) Multi-objective optimization
    
    Example:
        >>> router = UseCaseRouter()
        >>> result = router.route("Using these documents, answer questions about...")
        >>> print(result.detected_use_case)
        "rag_pipeline"
        >>> print(result.applied_constraints.min_context_k)
        64
    """
    
    def __init__(
        self,
        constraint_mappings: Optional[Dict[str, UseCaseConstraintMapping]] = None,
    ):
        """
        Initialize router.
        
        Args:
            constraint_mappings: Custom use case → constraint mappings.
                If None, uses USE_CASE_CONSTRAINTS defaults.
        """
        self.classifier = PromptClassifier()
        self.constraint_mappings = constraint_mappings or USE_CASE_CONSTRAINTS
    
    def detect_use_case(self, prompt: str) -> ClassificationResult:
        """
        Detect use case from prompt.
        
        Args:
            prompt: User prompt to classify
            
        Returns:
            ClassificationResult with use_case, confidence, signals
        """
        return self.classifier.classify(prompt)
    
    def get_constraints_for_use_case(
        self,
        use_case: str,
        context_override_k: Optional[int] = None,
        extra_capabilities: Optional[List[CapabilityRequirement]] = None,
    ) -> ConstraintConfig:
        """
        Get constraint configuration for a use case.
        
        Args:
            use_case: Use case identifier (e.g., "rag_pipeline")
            context_override_k: Override minimum context requirement
            extra_capabilities: Additional required capabilities
            
        Returns:
            ConstraintConfig for model filtering
        """
        # Get base mapping
        mapping = self.constraint_mappings.get(
            use_case,
            UseCaseConstraintMapping()  # Default: minimal constraints
        )
        
        # Create config
        config = mapping.to_constraint_config()
        
        # Apply overrides
        if context_override_k is not None:
            config.min_context_k = context_override_k
            config.target_context_k = max(
                config.target_context_k or 0,
                context_override_k
            )
        
        if extra_capabilities:
            config.capabilities = list(set(config.capabilities + extra_capabilities))
        
        return config
    
    def route(
        self,
        prompt: str,
        models: Optional[List[Dict]] = None,
        expected_context_tokens: Optional[int] = None,
        require_capabilities: Optional[List[str]] = None,
        use_case_override: Optional[str] = None,
        verbose: bool = False,
    ) -> UseCaseRoutingResult:
        """
        Route a prompt to appropriate models based on detected use case.
        
        This is the main entry point. It:
        1. Detects the use case from the prompt
        2. Maps to appropriate constraints
        3. Filters models by constraints
        4. Returns routing result (optionally with recommendations)
        
        Args:
            prompt: User prompt to route
            models: List of model dicts (optional, for filtering)
            expected_context_tokens: Override context requirement (tokens)
            require_capabilities: Additional required capability names
            use_case_override: Force a specific use case instead of detection
            verbose: Print detailed routing info
            
        Returns:
            UseCaseRoutingResult with complete routing information
        """
        # Stage 1: Use case detection
        if use_case_override:
            classification = ClassificationResult(
                use_case=use_case_override,
                confidence=1.0,
                category=self.classifier.patterns.get(
                    use_case_override, {}
                ).get("category", None),
                signals=["override:user_specified"],
                alternative_use_cases=[],
            )
        else:
            classification = self.detect_use_case(prompt)
        
        if verbose:
            logger.info(f"Detected use case: {classification.use_case} "
                       f"({classification.confidence:.0%} confidence)")
        
        # Stage 2: Estimate context requirements
        estimated_context_k = estimate_context_requirement(
            prompt=prompt,
            use_case=classification.use_case,
        )
        
        # Override with user-specified context if provided
        if expected_context_tokens:
            estimated_context_k = max(
                estimated_context_k,
                (expected_context_tokens // 1000) + 1
            )
        
        # Stage 3: Build constraints
        extra_caps = []
        if require_capabilities:
            # Convert string names to enum
            cap_map = {c.name.lower(): c for c in CapabilityRequirement}
            for cap_name in require_capabilities:
                cap = cap_map.get(cap_name.lower())
                if cap:
                    extra_caps.append(cap)
        
        constraints = self.get_constraints_for_use_case(
            use_case=classification.use_case,
            context_override_k=estimated_context_k,
            extra_capabilities=extra_caps,
        )
        
        if verbose:
            logger.info(f"Applied constraints: min_ctx={constraints.min_context_k}K, "
                       f"caps={[c.name for c in constraints.capabilities]}")
        
        # Stage 4: Filter models (if provided)
        models_before = len(models) if models else 0
        models_after = models_before
        
        if models:
            filtered = apply_constraints(models, constraints, verbose=verbose)
            models_after = len(filtered)
        
        # Build routing decision (for archetype/category)
        routing_decision = RoutingDecision(
            archetype=self._map_use_case_to_archetype(classification.use_case),
            category=self._map_use_case_to_category(classification.use_case),
            reason=f"Use case: {classification.use_case}",
            recommend_cot=classification.use_case in [
                "math_reasoning", "code_review", "research_assistant"
            ],
        )
        
        return UseCaseRoutingResult(
            detected_use_case=classification.use_case,
            use_case_confidence=classification.confidence,
            alternative_use_cases=classification.alternative_use_cases,
            detection_signals=classification.signals,
            applied_constraints=constraints,
            estimated_context_k=estimated_context_k,
            models_before_filter=models_before,
            models_after_filter=models_after,
            routing_decision=routing_decision,
            prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt,
        )
    
    def _map_use_case_to_archetype(self, use_case: str) -> ProductArchetype:
        """Map use case to product archetype."""
        archetype_mapping = {
            # Bulk operations
            "structured_extraction": ProductArchetype.BULK_OPS,
            "summarization": ProductArchetype.BULK_OPS,
            "translation": ProductArchetype.BULK_OPS,
            
            # RAG specialist
            "rag_pipeline": ProductArchetype.RAG_SPECIALIST,
            "research_assistant": ProductArchetype.RAG_SPECIALIST,
            "customer_support": ProductArchetype.RAG_SPECIALIST,
            
            # Reasoning specialist
            "code_generation": ProductArchetype.REASONING_SPECIALIST,
            "code_review": ProductArchetype.REASONING_SPECIALIST,
            "code_refactoring": ProductArchetype.REASONING_SPECIALIST,
            "math_reasoning": ProductArchetype.REASONING_SPECIALIST,
            "data_analysis": ProductArchetype.REASONING_SPECIALIST,
            "sql_generation": ProductArchetype.REASONING_SPECIALIST,
            
            # Frontier (complex)
            "legal_review": ProductArchetype.FRONTIER,
            "financial_analysis": ProductArchetype.FRONTIER,
            "maximum_quality": ProductArchetype.FRONTIER,
            "creative_writing": ProductArchetype.FRONTIER,
        }
        return archetype_mapping.get(use_case, ProductArchetype.RAG_SPECIALIST)
    
    def _map_use_case_to_category(self, use_case: str) -> PromptCategory:
        """Map use case to prompt category."""
        category_mapping = {
            # Coding
            "code_generation": PromptCategory.CODING,
            "code_review": PromptCategory.CODING,
            "code_refactoring": PromptCategory.CODING,
            "technical_docs": PromptCategory.CODING,
            
            # Data science
            "data_analysis": PromptCategory.DATA_SCIENCE,
            "sql_generation": PromptCategory.DATA_SCIENCE,
            "math_reasoning": PromptCategory.DATA_SCIENCE,
            
            # Creative
            "creative_writing": PromptCategory.CREATIVE,
            "translation": PromptCategory.CREATIVE,
            
            # Specialized
            "legal_review": PromptCategory.LEGAL,
            "financial_analysis": PromptCategory.FINANCE,
        }
        return category_mapping.get(use_case, PromptCategory.GENERAL)


# =============================================================================
# Convenience Functions
# =============================================================================

def route_prompt(
    prompt: str,
    models: Optional[List[Dict]] = None,
    verbose: bool = False,
) -> UseCaseRoutingResult:
    """
    Quick routing helper.
    
    Args:
        prompt: User prompt to route
        models: Optional list of models to filter
        verbose: Print routing details
        
    Returns:
        UseCaseRoutingResult
    """
    router = UseCaseRouter()
    return router.route(prompt, models=models, verbose=verbose)


def get_use_case_constraints(use_case: str) -> ConstraintConfig:
    """
    Get constraint configuration for a use case.
    
    Args:
        use_case: Use case identifier
        
    Returns:
        ConstraintConfig for model filtering
    """
    router = UseCaseRouter()
    return router.get_constraints_for_use_case(use_case)


def detect_use_case(prompt: str) -> Tuple[str, float]:
    """
    Quick use case detection.
    
    Args:
        prompt: User prompt
        
    Returns:
        Tuple of (use_case, confidence)
    """
    router = UseCaseRouter()
    result = router.detect_use_case(prompt)
    return result.use_case, result.confidence

