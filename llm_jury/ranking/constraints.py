"""
Constraint System for LLM Model Selection.

Provides hard constraints (pre-filtering) and soft constraints (objectives)
for context window requirements and model capabilities.

Architecture:
    
    User Request
         │
         ▼
    ┌─────────────────────┐
    │  Hard Constraints   │  ← Filter models that don't meet minimum requirements
    │  (Pre-filtering)    │     e.g., context_window >= 32K, supports_functions=True
    └─────────────────────┘
         │
         ▼
    ┌─────────────────────┐
    │  Soft Constraints   │  ← Objectives that reward/penalize during optimization
    │  (Scoring Bonus)    │     e.g., bonus for excess context, penalty for limited
    └─────────────────────┘
         │
         ▼
    ┌─────────────────────┐
    │   Optimization      │  ← Standard Chebyshev/Knee/Hybrid optimization
    │   (Ranking)         │
    └─────────────────────┘
         │
         ▼
    Recommendations

Usage:
    from llm_jury.ranking.constraints import (
        ConstraintConfig, 
        CapabilityRequirement,
        apply_constraints,
        create_context_objective,
    )
    
    # Define requirements
    constraints = ConstraintConfig(
        min_context_k=128,  # At least 128K context
        capabilities=[CapabilityRequirement.FUNCTION_CALLING],
        prefer_excess_context=True,  # Bonus for more context
    )
    
    # Filter models
    filtered = apply_constraints(models, constraints)
    
    # Add context as soft objective (optional)
    context_obj = create_context_objective(target_context_k=128)
    optimizer.objectives.register(context_obj)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Any, Set
import numpy as np

from llm_jury.core.models import ModelMetadata


# =============================================================================
# Capability Definitions
# =============================================================================

class CapabilityRequirement(Enum):
    """
    Model capabilities that can be required or preferred.
    
    These map to model metadata fields or can be derived from model properties.
    """
    # Output modalities
    FUNCTION_CALLING = auto()      # Tool use / function calling support
    JSON_MODE = auto()             # Structured JSON output
    STREAMING = auto()             # Streaming response support
    
    # Input modalities  
    VISION = auto()                # Image input support
    AUDIO = auto()                 # Audio input support
    FILE_UPLOAD = auto()           # Document/file processing
    
    # Special capabilities
    EMBEDDINGS = auto()            # Embedding generation
    FINE_TUNING = auto()           # Fine-tuning available
    REASONING = auto()             # Extended thinking / chain-of-thought
    
    # Context-related
    LONG_CONTEXT = auto()          # 100K+ context window
    VERY_LONG_CONTEXT = auto()     # 200K+ context window
    MILLION_CONTEXT = auto()       # 1M+ context window


# Providers/models known to support specific capabilities
# Used for heuristic detection when explicit capability flags aren't available
_FUNCTION_CALLING_PROVIDERS = {'anthropic', 'openai', 'google', 'cohere', 'mistral', 'meta'}
_FUNCTION_CALLING_MODELS = {'claude', 'gpt', 'gemini', 'command', 'llama', 'mixtral', 'mistral'}

_VISION_MODELS = {'gpt-4o', 'claude-3', 'gemini', 'llava', 'vision'}

_EMBEDDING_MODELS = {'embed', 'e5', 'bge', 'gte', 'instructor'}


def _has_function_calling(m: Dict) -> bool:
    """Detect function calling support using multiple signals."""
    # Explicit field
    if m.get('supports_functions'):
        return True
    if m.get('tool_use_ability') and float(m.get('tool_use_ability', 0)) > 0.3:
        return True
    
    # Heuristic: Major providers' chat models support function calling
    creator = (m.get('creator_slug') or m.get('provider') or '').lower()
    name_lower = m.get('name', '').lower()
    
    # Embedding models don't support function calling
    if any(emb in name_lower for emb in _EMBEDDING_MODELS):
        return False
    
    # Check by provider
    if creator in _FUNCTION_CALLING_PROVIDERS:
        return True
    
    # Check by model family
    if any(family in name_lower for family in _FUNCTION_CALLING_MODELS):
        return True
    
    # Default: assume chat models support basic tool use
    return True


def _has_vision(m: Dict) -> bool:
    """Detect vision/image input support."""
    if m.get('supports_vision'):
        return True
    
    name_lower = m.get('name', '').lower()
    openrouter_id = m.get('openrouter_id', '').lower()
    
    # Check for vision keywords
    if any(v in name_lower or v in openrouter_id for v in ['vision', 'llava', '4o', 'gemini-pro-vision']):
        return True
    
    # Check architecture data from OpenRouter
    arch = m.get('architecture', {})
    if arch.get('modality') == 'multimodal' or 'image' in str(arch.get('input_modalities', [])):
        return True
    
    return False


# Capability detection rules
# Maps CapabilityRequirement to check function
CAPABILITY_DETECTORS: Dict[CapabilityRequirement, Callable[[Dict], bool]] = {
    CapabilityRequirement.FUNCTION_CALLING: _has_function_calling,
    
    CapabilityRequirement.JSON_MODE: lambda m: (
        m.get('supports_json_mode', True)  # Most modern models support this
    ),
    
    CapabilityRequirement.STREAMING: lambda m: (
        m.get('supports_streaming', True)  # Most API models support streaming
    ),
    
    CapabilityRequirement.VISION: _has_vision,
    
    CapabilityRequirement.AUDIO: lambda m: (
        'audio' in m.get('name', '').lower() or
        m.get('supports_audio', False) or
        'audio' in str(m.get('architecture', {}).get('input_modalities', []))
    ),
    
    CapabilityRequirement.EMBEDDINGS: lambda m: (
        any(emb in m.get('name', '').lower() for emb in _EMBEDDING_MODELS) or
        m.get('supports_embeddings', False)
    ),
    
    CapabilityRequirement.REASONING: lambda m: (
        'reason' in m.get('name', '').lower() or
        'think' in m.get('name', '').lower() or
        ' r1' in m.get('name', '').lower() or
        m.get('name', '').lower().endswith(' r1') or
        m.get('is_reasoning_model', False)
    ),
    
    CapabilityRequirement.LONG_CONTEXT: lambda m: (
        (m.get('context_window_k') or 0) >= 100 or
        (m.get('context_length') or 0) >= 100_000
    ),
    
    CapabilityRequirement.VERY_LONG_CONTEXT: lambda m: (
        (m.get('context_window_k') or 0) >= 200 or
        (m.get('context_length') or 0) >= 200_000
    ),
    
    CapabilityRequirement.MILLION_CONTEXT: lambda m: (
        (m.get('context_window_k') or 0) >= 1000 or
        (m.get('context_length') or 0) >= 1_000_000
    ),
}


# =============================================================================
# Constraint Configuration
# =============================================================================

@dataclass
class ConstraintConfig:
    """
    Configuration for model constraints.
    
    Attributes:
        min_context_k: Minimum context window in thousands of tokens.
            Models with less context will be filtered out (hard constraint).
        
        target_context_k: Target context for soft scoring bonus.
            Models meeting/exceeding this get a bonus. If None, uses min_context_k.
        
        capabilities: Required capabilities (hard constraint).
            Models lacking ANY of these are filtered out.
        
        preferred_capabilities: Preferred capabilities (soft constraint).
            Models with these get a scoring bonus.
        
        prefer_excess_context: Whether to give bonus for context > target.
            If True, adds a soft objective that rewards excess context.
        
        context_utilization_estimate: Expected context utilization (0-1).
            Used to estimate if a model has enough headroom.
            e.g., 0.8 means you expect to use ~80% of context window.
        
        exclude_providers: Providers to exclude (e.g., ["x-ai"] to skip Grok).
        
        require_providers: Only include these providers (whitelist).
    """
    # Context constraints
    min_context_k: Optional[int] = None
    target_context_k: Optional[int] = None
    context_utilization_estimate: float = 0.8
    prefer_excess_context: bool = True
    
    # Capability constraints
    capabilities: List[CapabilityRequirement] = field(default_factory=list)
    preferred_capabilities: List[CapabilityRequirement] = field(default_factory=list)
    
    # Provider constraints
    exclude_providers: List[str] = field(default_factory=list)
    require_providers: List[str] = field(default_factory=list)
    
    # Cost constraints (optional hard limits)
    max_input_cost_per_m: Optional[float] = None
    max_output_cost_per_m: Optional[float] = None
    
    def __post_init__(self):
        """Set defaults."""
        if self.target_context_k is None and self.min_context_k is not None:
            self.target_context_k = self.min_context_k


# =============================================================================
# Constraint Application (Pre-filtering)
# =============================================================================

def check_model_capability(
    model: Dict,
    capability: CapabilityRequirement
) -> bool:
    """
    Check if a model has a specific capability.
    
    Args:
        model: Model data dictionary
        capability: Capability to check
        
    Returns:
        True if model has the capability
    """
    detector = CAPABILITY_DETECTORS.get(capability)
    if detector is None:
        # Unknown capability - assume model has it
        return True
    
    try:
        return detector(model)
    except Exception:
        return False


def get_model_context_k(model: Dict) -> int:
    """Get model's context window in thousands of tokens."""
    # Try context_window_k first (our standard field)
    ctx_k = model.get('context_window_k')
    if ctx_k is not None:
        return int(ctx_k)
    
    # Try context_length (raw tokens)
    ctx = model.get('context_length')
    if ctx is not None:
        return int(ctx) // 1000
    
    # Fallback: estimate from model name
    name = model.get('name', '').lower()
    if '1m' in name or '1000k' in name:
        return 1000
    elif '200k' in name:
        return 200
    elif '128k' in name:
        return 128
    elif '100k' in name:
        return 100
    elif '32k' in name:
        return 32
    elif '16k' in name:
        return 16
    
    # Default assumption for modern models
    return 8


def apply_constraints(
    models: List[Dict],
    config: ConstraintConfig,
    verbose: bool = False
) -> List[Dict]:
    """
    Apply hard constraints to filter models.
    
    This is the pre-filtering step before optimization.
    
    Args:
        models: List of model data dictionaries
        config: Constraint configuration
        verbose: Whether to print filtering stats
        
    Returns:
        Filtered list of models meeting all hard constraints
    """
    original_count = len(models)
    filtered = models
    
    # 1. Context window constraint
    if config.min_context_k is not None:
        filtered = [
            m for m in filtered
            if get_model_context_k(m) >= config.min_context_k
        ]
        if verbose:
            print(f"  Context >= {config.min_context_k}K: {len(filtered)}/{original_count} models")
    
    # 2. Required capabilities
    for capability in config.capabilities:
        before = len(filtered)
        filtered = [
            m for m in filtered
            if check_model_capability(m, capability)
        ]
        if verbose and len(filtered) < before:
            print(f"  {capability.name}: {len(filtered)}/{before} models")
    
    # 3. Provider constraints
    if config.exclude_providers:
        exclude_set = set(p.lower() for p in config.exclude_providers)
        filtered = [
            m for m in filtered
            if m.get('creator_slug', '').lower() not in exclude_set and
               m.get('provider', '').lower() not in exclude_set
        ]
        if verbose:
            print(f"  Excluded providers: {len(filtered)} models remaining")
    
    if config.require_providers:
        require_set = set(p.lower() for p in config.require_providers)
        filtered = [
            m for m in filtered
            if m.get('creator_slug', '').lower() in require_set or
               m.get('provider', '').lower() in require_set
        ]
        if verbose:
            print(f"  Required providers: {len(filtered)} models remaining")
    
    # 4. Cost constraints
    if config.max_input_cost_per_m is not None:
        filtered = [
            m for m in filtered
            if (m.get('input_cost_per_m') or m.get('price_1m_input') or 0) <= config.max_input_cost_per_m
        ]
        if verbose:
            print(f"  Max input cost ${config.max_input_cost_per_m}/M: {len(filtered)} models")
    
    if config.max_output_cost_per_m is not None:
        filtered = [
            m for m in filtered
            if (m.get('output_cost_per_m') or m.get('price_1m_output') or 0) <= config.max_output_cost_per_m
        ]
        if verbose:
            print(f"  Max output cost ${config.max_output_cost_per_m}/M: {len(filtered)} models")
    
    if verbose:
        print(f"  Final: {len(filtered)}/{original_count} models pass constraints")
    
    return filtered


# =============================================================================
# Soft Constraints (Objectives for Optimization)
# =============================================================================

def create_context_objective(
    target_context_k: int = 128,
    weight: float = 0.10,
    excess_bonus: bool = True,
) -> 'Objective':
    """
    Create an Objective for context window optimization.
    
    This objective rewards models with sufficient context and optionally
    provides bonus for excess context (headroom).
    
    Scoring Logic:
        - Below target: Penalized proportionally (context_k / target_k)
        - At target: Score = 1.0
        - Above target (if excess_bonus): Logarithmic bonus up to 1.2
    
    Args:
        target_context_k: Target context window in thousands
        weight: Weight for this objective (default 0.10 = 10%)
        excess_bonus: Whether to reward context > target
        
    Returns:
        Objective instance for the optimizer
    """
    from llm_jury.ranking.optimizer import Objective, NormalizationMethod
    
    def extract_context_score(model: ModelMetadata, decision, context) -> float:
        """Extract context score from model."""
        # Get context from metadata
        ctx_k = getattr(model, 'context_window_k', None)
        if ctx_k is None:
            # Try to get from raw dict via context
            model_dict_fn = context.get('model_to_dict')
            if model_dict_fn:
                model_dict = model_dict_fn(model)
                ctx_k = get_model_context_k(model_dict)
            else:
                ctx_k = 8  # Conservative default
        
        # Calculate score
        if ctx_k >= target_context_k:
            if excess_bonus:
                # Logarithmic bonus for excess (up to 20% bonus)
                excess_ratio = ctx_k / target_context_k
                bonus = min(np.log1p(excess_ratio - 1) * 0.2, 0.2)
                return 100 * (1.0 + bonus)
            return 100.0
        else:
            # Penalty for insufficient context
            return 100 * (ctx_k / target_context_k)
    
    return Objective(
        name="context",
        display_name="Context Window",
        direction="maximize",
        default_weight=weight,
        default_value=50.0,  # Assume 50% of target if unknown
        extractor=extract_context_score,
        normalization=NormalizationMethod.PERCENTAGE,
        required_fields=[],  # We handle missing data in extractor
        summary_format="Context: {value:.0f}K tokens"
    )


def create_capability_objective(
    preferred_capabilities: List[CapabilityRequirement],
    weight: float = 0.05,
) -> 'Objective':
    """
    Create an Objective that rewards models with preferred capabilities.
    
    This is for soft preferences - use ConstraintConfig.capabilities for hard requirements.
    
    Scoring: Each capability contributes equally. Score = (capabilities_met / total) * 100
    
    Args:
        preferred_capabilities: List of preferred capabilities
        weight: Weight for this objective
        
    Returns:
        Objective instance
    """
    from llm_jury.ranking.optimizer import Objective, NormalizationMethod
    
    def extract_capability_score(model: ModelMetadata, decision, context) -> float:
        """Count how many preferred capabilities the model has."""
        if not preferred_capabilities:
            return 100.0
        
        # Get model dict
        model_dict_fn = context.get('model_to_dict')
        if model_dict_fn:
            model_dict = model_dict_fn(model)
        else:
            model_dict = {'name': model.name}
        
        met_count = sum(
            1 for cap in preferred_capabilities
            if check_model_capability(model_dict, cap)
        )
        
        return 100 * (met_count / len(preferred_capabilities))
    
    cap_names = [c.name for c in preferred_capabilities]
    
    return Objective(
        name="capabilities",
        display_name="Capabilities",
        direction="maximize",
        default_weight=weight,
        default_value=50.0,
        extractor=extract_capability_score,
        normalization=NormalizationMethod.PERCENTAGE,
        summary_format=f"Caps ({', '.join(cap_names[:2])}...): {{value:.0f}}%"
    )


# =============================================================================
# Use Case Presets with Constraints
# =============================================================================

@dataclass
class UseCaseConstraints:
    """Predefined constraint configurations for common use cases."""
    
    @staticmethod
    def rag_pipeline(document_size_tokens: int = 50_000) -> ConstraintConfig:
        """
        Constraints for RAG/retrieval pipelines.
        
        Args:
            document_size_tokens: Expected size of retrieved context
        """
        # Need context for: prompt + retrieved docs + response headroom
        min_ctx = int(np.ceil((document_size_tokens + 10_000) / 1000))
        target_ctx = min_ctx * 2  # 2x headroom is ideal
        
        return ConstraintConfig(
            min_context_k=min_ctx,
            target_context_k=target_ctx,
            context_utilization_estimate=0.6,
            prefer_excess_context=True,
            capabilities=[],  # RAG doesn't require special capabilities
            preferred_capabilities=[CapabilityRequirement.JSON_MODE],
        )
    
    @staticmethod
    def function_calling() -> ConstraintConfig:
        """Constraints for agentic/function-calling use cases."""
        return ConstraintConfig(
            min_context_k=32,
            capabilities=[CapabilityRequirement.FUNCTION_CALLING],
            preferred_capabilities=[
                CapabilityRequirement.JSON_MODE,
                CapabilityRequirement.STREAMING,
            ],
        )
    
    @staticmethod
    def long_document_analysis(document_tokens: int = 100_000) -> ConstraintConfig:
        """Constraints for analyzing very long documents."""
        min_ctx = int(np.ceil(document_tokens / 1000)) + 10  # +10K headroom
        
        return ConstraintConfig(
            min_context_k=min_ctx,
            target_context_k=min_ctx * 1.5,
            capabilities=[CapabilityRequirement.LONG_CONTEXT],
            prefer_excess_context=True,
        )
    
    @staticmethod
    def vision_analysis() -> ConstraintConfig:
        """Constraints for image/vision tasks."""
        return ConstraintConfig(
            min_context_k=32,
            capabilities=[CapabilityRequirement.VISION],
            preferred_capabilities=[CapabilityRequirement.JSON_MODE],
        )
    
    @staticmethod
    def reasoning_heavy() -> ConstraintConfig:
        """Constraints for complex reasoning tasks."""
        return ConstraintConfig(
            min_context_k=64,
            capabilities=[],  # Don't require reasoning flag
            preferred_capabilities=[CapabilityRequirement.REASONING],
        )
    
    @staticmethod
    def embeddings() -> ConstraintConfig:
        """Constraints for embedding generation."""
        return ConstraintConfig(
            capabilities=[CapabilityRequirement.EMBEDDINGS],
        )
    
    @staticmethod
    def budget_conscious(max_cost_per_m: float = 1.0) -> ConstraintConfig:
        """Cost-constrained selection."""
        return ConstraintConfig(
            max_input_cost_per_m=max_cost_per_m,
            max_output_cost_per_m=max_cost_per_m * 3,  # Output typically 3x input
        )


# =============================================================================
# Integration Helper
# =============================================================================

def get_constrained_recommendations(
    models: List[Dict],
    constraints: ConstraintConfig,
    add_soft_objectives: bool = True,
) -> tuple:
    """
    Apply constraints and optionally create soft objectives.
    
    This is a helper that:
    1. Filters models by hard constraints
    2. Creates soft objective(s) if requested
    
    Args:
        models: All available models
        constraints: Constraint configuration
        add_soft_objectives: Whether to create soft objectives
        
    Returns:
        Tuple of (filtered_models, list_of_soft_objectives)
    """
    # Apply hard constraints
    filtered = apply_constraints(models, constraints, verbose=True)
    
    soft_objectives = []
    
    if add_soft_objectives:
        # Add context objective if we have context constraints
        if constraints.target_context_k is not None:
            soft_objectives.append(
                create_context_objective(
                    target_context_k=constraints.target_context_k,
                    excess_bonus=constraints.prefer_excess_context,
                )
            )
        
        # Add capability objective if we have preferred capabilities
        if constraints.preferred_capabilities:
            soft_objectives.append(
                create_capability_objective(
                    preferred_capabilities=constraints.preferred_capabilities,
                )
            )
    
    return filtered, soft_objectives

