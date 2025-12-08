"""
Use Case-Specific Ranking Weights.

Defines optimized weights for multi-objective ranking based on detected use case.
Each use case has different priorities for quality, cost, latency, and specialized
metrics like coding benchmarks or hallucination rates.

Architecture:
    
    Detected Use Case (e.g., "code_generation")
            │
            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │        USE_CASE_WEIGHTS["code_generation"]                    │
    │                                                               │
    │   quality_index:     0.25  ← General quality still matters    │
    │   coding_index:      0.35  ← Prioritize coding benchmarks!    │
    │   cost_efficiency:   0.20  ← Cost matters but not primary     │
    │   latency:           0.10  ← Speed secondary                  │
    │   context_score:     0.10  ← Some context helps               │
    │                                                               │
    └───────────────────────────────────────────────────────────────┘
            │
            ▼
    Optimizer uses these weights to rank models

Usage:
    from llm_jury.ranking.use_case_weights import (
        get_weights_for_use_case,
        UseCaseWeights,
        USE_CASE_WEIGHTS,
    )
    
    # Get weights for a detected use case
    weights = get_weights_for_use_case("code_generation")
    
    # Use with optimizer
    optimizer = Optimizer(
        quality_weight=weights.quality_index,
        cost_weight=weights.cost_efficiency,
        latency_weight=weights.latency,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum


@dataclass
class UseCaseWeights:
    """
    Optimized ranking weights for a specific use case.
    
    All weights should sum to approximately 1.0 for consistent scoring.
    """
    # Core metrics (always available)
    quality_index: float = 0.35       # General quality/intelligence score
    cost_efficiency: float = 0.25     # Cost per quality (value)
    latency: float = 0.15             # Speed (TTFT, tokens/sec)
    context_score: float = 0.10       # Context window adequacy
    
    # Specialized metrics (use when available, otherwise redistributed)
    coding_index: float = 0.0         # Coding benchmarks (HumanEval, etc.)
    math_index: float = 0.0           # Math benchmarks (GSM8K, MATH, etc.)
    hallucination_penalty: float = 0.0  # Low hallucination preference
    reasoning_score: float = 0.0      # Reasoning benchmarks (GPQA, etc.)
    
    # Capability bonuses (soft preferences)
    vision_bonus: float = 0.0         # Vision capability preference
    function_calling_bonus: float = 0.0  # Tool use preference
    long_context_bonus: float = 0.0   # Extra context preference
    
    # Metadata
    description: str = ""             # Human-readable description
    recommended_models: List[str] = field(default_factory=list)  # Model hints
    
    def to_optimizer_weights(self) -> Dict[str, float]:
        """Convert to optimizer-compatible weight dictionary."""
        return {
            "quality_index": self.quality_index,
            "cost_efficiency": self.cost_efficiency,
            "latency": self.latency,
            "context_score": self.context_score,
            "coding_index": self.coding_index,
            "math_index": self.math_index,
            "hallucination_penalty": self.hallucination_penalty,
            "reasoning_score": self.reasoning_score,
        }
    
    def get_primary_metric(self) -> str:
        """Get the most important metric for this use case."""
        weights = self.to_optimizer_weights()
        return max(weights.items(), key=lambda x: x[1])[0]


# =============================================================================
# Use Case Weight Definitions
# =============================================================================

USE_CASE_WEIGHTS: Dict[str, UseCaseWeights] = {
    # =========================================================================
    # Development & Engineering
    # =========================================================================
    "code_generation": UseCaseWeights(
        quality_index=0.20,
        coding_index=0.40,          # Prioritize coding benchmarks
        cost_efficiency=0.20,
        latency=0.10,
        context_score=0.10,
        description="Prioritizes coding benchmarks (HumanEval, MBPP) over general quality",
        recommended_models=["deepseek-v3", "claude-3.5-sonnet", "gpt-4o", "codestral"],
    ),
    
    "code_review": UseCaseWeights(
        quality_index=0.25,
        coding_index=0.30,
        reasoning_score=0.15,       # Need to reason about code
        cost_efficiency=0.15,
        latency=0.05,
        context_score=0.10,         # Need to see full context
        description="Balances coding ability with reasoning for thorough review",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "deepseek-v3"],
    ),
    
    "code_refactoring": UseCaseWeights(
        quality_index=0.20,
        coding_index=0.35,
        reasoning_score=0.15,
        cost_efficiency=0.15,
        latency=0.05,
        context_score=0.10,
        description="Strong coding + reasoning for safe refactoring",
        recommended_models=["claude-3.5-sonnet", "deepseek-v3", "gpt-4o"],
    ),
    
    "technical_docs": UseCaseWeights(
        quality_index=0.35,
        coding_index=0.20,          # Need to understand code
        cost_efficiency=0.25,
        latency=0.10,
        context_score=0.10,
        description="Clear writing with code understanding",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-pro"],
    ),
    
    # =========================================================================
    # Data & Analytics
    # =========================================================================
    "data_analysis": UseCaseWeights(
        quality_index=0.25,
        math_index=0.25,            # Math/analytical ability
        reasoning_score=0.20,
        cost_efficiency=0.15,
        latency=0.05,
        context_score=0.10,
        description="Strong analytical and mathematical reasoning",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"],
    ),
    
    "sql_generation": UseCaseWeights(
        quality_index=0.20,
        coding_index=0.35,          # SQL is code
        math_index=0.15,            # Data logic
        cost_efficiency=0.20,
        latency=0.10,
        description="Coding ability for SQL with data understanding",
        recommended_models=["deepseek-v3", "gpt-4o", "claude-3.5-sonnet"],
    ),
    
    "math_reasoning": UseCaseWeights(
        quality_index=0.15,
        math_index=0.45,            # Heavily prioritize math
        reasoning_score=0.20,
        cost_efficiency=0.10,
        latency=0.05,
        context_score=0.05,
        description="Maximum weight on math benchmarks (GSM8K, MATH)",
        recommended_models=["o1", "deepseek-r1", "claude-3.5-sonnet", "gpt-4o"],
    ),
    
    # =========================================================================
    # Content & Communication
    # =========================================================================
    "creative_writing": UseCaseWeights(
        quality_index=0.45,          # Quality is everything
        cost_efficiency=0.25,
        latency=0.15,
        context_score=0.15,
        description="Prioritizes output quality and creativity",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-pro"],
    ),
    
    "summarization": UseCaseWeights(
        quality_index=0.30,
        hallucination_penalty=0.25,  # Must be accurate!
        cost_efficiency=0.20,
        latency=0.10,
        context_score=0.15,          # Need to fit full document
        description="Accuracy critical - penalizes hallucination",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-1.5-pro"],
    ),
    
    "translation": UseCaseWeights(
        quality_index=0.40,
        cost_efficiency=0.30,        # Often high volume
        latency=0.15,
        context_score=0.15,
        description="Quality translation at reasonable cost",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gemini-pro"],
    ),
    
    "paraphrasing": UseCaseWeights(
        quality_index=0.40,
        cost_efficiency=0.35,        # Often batch processing
        latency=0.15,
        context_score=0.10,
        description="Quality output at good value",
    ),
    
    "style_transfer": UseCaseWeights(
        quality_index=0.45,
        cost_efficiency=0.30,
        latency=0.15,
        context_score=0.10,
        description="Creative transformation quality",
    ),
    
    "grammar_correction": UseCaseWeights(
        quality_index=0.35,
        cost_efficiency=0.40,        # High volume, simple task
        latency=0.15,
        context_score=0.10,
        description="Cost-effective for simple corrections",
    ),
    
    # =========================================================================
    # RAG & Document Processing
    # =========================================================================
    "rag_pipeline": UseCaseWeights(
        quality_index=0.25,
        hallucination_penalty=0.30,  # Critical for RAG!
        context_score=0.20,          # Need long context
        cost_efficiency=0.15,
        latency=0.10,
        long_context_bonus=0.05,
        function_calling_bonus=0.05,  # For retrieval calls
        description="Low hallucination + long context for RAG",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro", "deepseek-v3"],
    ),
    
    "long_context": UseCaseWeights(
        quality_index=0.25,
        context_score=0.35,          # Context is primary
        cost_efficiency=0.20,
        latency=0.10,
        hallucination_penalty=0.10,
        description="Maximum context window priority",
        recommended_models=["gemini-1.5-pro", "claude-3.5-sonnet", "gpt-4o"],
    ),
    
    # =========================================================================
    # Technical Capabilities
    # =========================================================================
    "function_calling": UseCaseWeights(
        quality_index=0.30,
        function_calling_bonus=0.25,  # Must have capability
        cost_efficiency=0.25,
        latency=0.15,
        context_score=0.05,
        description="Reliable function/tool calling",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "mistral-large"],
    ),
    
    "tool_use": UseCaseWeights(
        quality_index=0.30,
        function_calling_bonus=0.25,
        reasoning_score=0.15,         # Need to reason about tools
        cost_efficiency=0.20,
        latency=0.10,
        description="Tool integration with reasoning",
        recommended_models=["gpt-4o", "claude-3.5-sonnet"],
    ),
    
    "agent_workflow": UseCaseWeights(
        quality_index=0.25,
        reasoning_score=0.25,         # Multi-step reasoning
        function_calling_bonus=0.20,
        cost_efficiency=0.15,         # Many calls = cost matters
        latency=0.10,
        context_score=0.05,
        description="Reasoning + tool use for autonomous agents",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "o1"],
    ),
    
    "planning": UseCaseWeights(
        quality_index=0.30,
        reasoning_score=0.30,         # Planning needs reasoning
        cost_efficiency=0.20,
        latency=0.10,
        context_score=0.10,
        description="Strong reasoning for task decomposition",
        recommended_models=["o1", "claude-3.5-sonnet", "gpt-4o"],
    ),
    
    "structured_extraction": UseCaseWeights(
        quality_index=0.30,
        hallucination_penalty=0.20,   # Must extract accurately
        cost_efficiency=0.30,         # Often batch processing
        latency=0.15,
        context_score=0.05,
        description="Accurate extraction at scale",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "mistral-large"],
    ),
    
    # =========================================================================
    # Embeddings & Similarity
    # =========================================================================
    "embeddings": UseCaseWeights(
        quality_index=0.40,
        cost_efficiency=0.40,         # Very high volume
        latency=0.20,                 # Speed matters for search
        description="Quality embeddings at scale - use embedding models",
        recommended_models=["text-embedding-3-large", "voyage-3", "text-embedding-3-small"],
    ),
    
    "semantic_similarity": UseCaseWeights(
        quality_index=0.45,
        cost_efficiency=0.35,
        latency=0.20,
        description="Accurate similarity comparison",
        recommended_models=["text-embedding-3-large", "voyage-3"],
    ),
    
    # =========================================================================
    # Classification & Analysis
    # =========================================================================
    "text_classification": UseCaseWeights(
        quality_index=0.35,
        cost_efficiency=0.40,         # Often batch
        latency=0.20,
        context_score=0.05,
        description="Accurate classification at volume",
        recommended_models=["gpt-4o-mini", "claude-haiku", "gpt-4o"],
    ),
    
    "sentiment_analysis": UseCaseWeights(
        quality_index=0.35,
        cost_efficiency=0.40,
        latency=0.20,
        context_score=0.05,
        description="Sentiment detection at scale",
        recommended_models=["gpt-4o-mini", "claude-haiku", "gpt-4o"],
    ),
    
    "entity_extraction": UseCaseWeights(
        quality_index=0.35,
        hallucination_penalty=0.20,   # Must be accurate
        cost_efficiency=0.30,
        latency=0.15,
        description="Accurate entity extraction",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gpt-4o-mini"],
    ),
    
    "content_moderation": UseCaseWeights(
        quality_index=0.40,           # Accuracy critical
        cost_efficiency=0.35,         # High volume
        latency=0.25,                 # Real-time needed
        description="Fast, accurate content safety",
        recommended_models=["gpt-4o-mini", "claude-haiku", "gpt-4o"],
    ),
    
    # =========================================================================
    # Vision / Multimodal
    # =========================================================================
    "image_understanding": UseCaseWeights(
        quality_index=0.40,
        vision_bonus=0.30,            # Must have vision
        cost_efficiency=0.20,
        latency=0.10,
        description="Vision capability required",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"],
    ),
    
    "vision_qa": UseCaseWeights(
        quality_index=0.35,
        vision_bonus=0.30,
        reasoning_score=0.15,
        cost_efficiency=0.15,
        latency=0.05,
        description="Vision + reasoning for visual QA",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"],
    ),
    
    # =========================================================================
    # Specialized Domains
    # =========================================================================
    "legal_review": UseCaseWeights(
        quality_index=0.30,
        hallucination_penalty=0.30,   # Accuracy critical
        reasoning_score=0.15,
        context_score=0.15,           # Legal docs are long
        cost_efficiency=0.10,
        description="Accuracy and reasoning for legal analysis",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "o1"],
    ),
    
    "financial_analysis": UseCaseWeights(
        quality_index=0.25,
        math_index=0.25,              # Financial calculations
        hallucination_penalty=0.20,   # Must be accurate
        reasoning_score=0.15,
        cost_efficiency=0.10,
        context_score=0.05,
        description="Math + accuracy for financial work",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "o1"],
    ),
    
    "research_assistant": UseCaseWeights(
        quality_index=0.30,
        hallucination_penalty=0.25,   # Must cite accurately
        context_score=0.20,           # Multiple sources
        reasoning_score=0.15,
        cost_efficiency=0.10,
        description="Accurate research with source handling",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-1.5-pro"],
    ),
    
    # =========================================================================
    # Conversational & Support
    # =========================================================================
    "customer_support": UseCaseWeights(
        quality_index=0.30,
        cost_efficiency=0.35,         # High volume
        latency=0.25,                 # Real-time chat
        context_score=0.10,
        description="Fast, cost-effective support responses",
        recommended_models=["gpt-4o-mini", "claude-haiku", "gpt-4o"],
    ),
    
    "tutoring": UseCaseWeights(
        quality_index=0.35,
        reasoning_score=0.25,         # Explain clearly
        cost_efficiency=0.20,
        latency=0.15,
        context_score=0.05,
        description="Clear explanations with good reasoning",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-pro"],
    ),
    
    "general_qa": UseCaseWeights(
        quality_index=0.35,
        cost_efficiency=0.30,
        latency=0.20,
        context_score=0.15,
        description="Balanced general-purpose assistant",
        recommended_models=["gpt-4o", "claude-3.5-sonnet", "gemini-pro"],
    ),
    
    # =========================================================================
    # Creative & Ideation
    # =========================================================================
    "brainstorming": UseCaseWeights(
        quality_index=0.45,           # Creative quality
        cost_efficiency=0.30,
        latency=0.15,
        context_score=0.10,
        description="Creative ideation quality",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "gemini-pro"],
    ),
    
    "roleplay": UseCaseWeights(
        quality_index=0.50,           # Character quality
        context_score=0.20,           # Maintain character history
        cost_efficiency=0.20,
        latency=0.10,
        description="Engaging character interactions",
        recommended_models=["claude-3.5-sonnet", "gpt-4o", "llama-3.1-70b"],
    ),
    
    # =========================================================================
    # Optimization Focused
    # =========================================================================
    "cost_optimized": UseCaseWeights(
        quality_index=0.25,
        cost_efficiency=0.55,         # Cost is primary
        latency=0.15,
        context_score=0.05,
        description="Minimize cost while maintaining acceptable quality",
        recommended_models=["gpt-4o-mini", "claude-haiku", "deepseek-chat", "gemini-flash"],
    ),
    
    "low_latency": UseCaseWeights(
        quality_index=0.25,
        latency=0.50,                 # Speed is primary
        cost_efficiency=0.20,
        context_score=0.05,
        description="Fastest response time",
        recommended_models=["claude-haiku", "gpt-4o-mini", "gemini-flash"],
    ),
    
    "maximum_quality": UseCaseWeights(
        quality_index=0.50,           # Quality is everything
        reasoning_score=0.20,
        cost_efficiency=0.10,
        latency=0.10,
        context_score=0.10,
        description="Best quality regardless of cost",
        recommended_models=["o1", "claude-3.5-sonnet", "gpt-4o"],
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_weights_for_use_case(use_case: str) -> UseCaseWeights:
    """
    Get optimized weights for a specific use case.
    
    Falls back to general_qa weights if use case not found.
    
    Args:
        use_case: The detected use case string
        
    Returns:
        UseCaseWeights with optimized weight configuration
    """
    return USE_CASE_WEIGHTS.get(use_case, USE_CASE_WEIGHTS["general_qa"])


def get_recommended_models(use_case: str) -> List[str]:
    """
    Get recommended models for a use case.
    
    Args:
        use_case: The detected use case string
        
    Returns:
        List of recommended model slugs
    """
    weights = get_weights_for_use_case(use_case)
    return weights.recommended_models


def get_primary_metric(use_case: str) -> str:
    """
    Get the most important ranking metric for a use case.
    
    Args:
        use_case: The detected use case string
        
    Returns:
        Name of the primary metric (e.g., "coding_index", "quality_index")
    """
    weights = get_weights_for_use_case(use_case)
    return weights.get_primary_metric()


def list_all_use_cases() -> List[str]:
    """Get all available use case names."""
    return list(USE_CASE_WEIGHTS.keys())

