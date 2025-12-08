"""
Archetype-based router for product-oriented model selection.

Consolidated from llm_router.py and llm_recommendation_orchestrator.py.
Routes prompts to product archetypes (Edge/Bulk/RAG/Reasoning/Frontier)
based on intent and complexity analysis.
"""

import re
from typing import Optional

from llm_jury.core.models import ProductArchetype, PromptCategory, RoutingDecision


class ArchetypeRouter:
    """
    Routes prompts to product archetypes using Hybrid Semantic-Symbolic classification.
    
    This router implements a Hierarchical Bayesian Classification approach:
    1. Estimates Latent State S = (Intent, Complexity)
    2. Uses Hybrid Cascade (Regex -> Zero-Shot) for high-precision state estimation
    3. Maps State S -> Optimal Archetype A* maximizing expected utility
    """
    
    def __init__(self, use_api: bool = False, fallback_threshold: float = 0.75):
        """
        Initialize the router with hybrid classifiers.
        
        Args:
            use_api: Whether to use HuggingFace Inference API (True) or local model (False)
            fallback_threshold: Confidence threshold for regex-only classification
        """
        from llm_jury.routing.hybrid_classifier import HybridClassifier
        from llm_jury.routing.complexity_classifier import HybridComplexityClassifier, ComplexityLevel
        
        # Initialize the hybrid classifiers
        self.intent_classifier = HybridClassifier(
            use_api=use_api, 
            fallback_threshold=fallback_threshold
        )
        self.complexity_classifier = HybridComplexityClassifier(
            use_api=use_api,
            fallback_threshold=fallback_threshold
        )

    def _get_cot_template(self, archetype: ProductArchetype, complexity_level: str) -> str:
        """Get Chain-of-Thought template based on archetype and complexity."""
        
        # Base templates
        templates = {
            ProductArchetype.FRONTIER: 
                "Let's first understand the problem and devise a plan. Then, carry out the plan step by step.",
            ProductArchetype.REASONING_SPECIALIST: 
                "Let's work this out in a step by step way to be sure we have the right answer.",
            ProductArchetype.RAG_SPECIALIST: 
                "Check your internal knowledge carefully. Answer ONLY based on verified facts from the context.",
        }
        
        template = templates.get(archetype, "")
        
        # Augment for very complex tasks
        if complexity_level == "complex_task" and archetype == ProductArchetype.FRONTIER:
            template += " Reflect on each step to ensure accuracy."
            
        return template

    def route(self, prompt: str, has_search_tools: bool = False) -> RoutingDecision:
        """
        Route prompt to appropriate archetype using intent and complexity analysis.
        
        Args:
            prompt: User prompt to route
            has_search_tools: Whether search/RAG tools are available
            
        Returns:
            RoutingDecision with archetype, category, and CoT recommendation
        """
        from llm_jury.routing.complexity_classifier import ComplexityLevel
        
        # 1. Estimate Latent State (Intent, Complexity)
        # Run classifiers (could be parallelized in future)
        intent_result = self.intent_classifier.classify(prompt)
        complexity_result = self.complexity_classifier.classify(prompt)
        
        category = intent_result.category
        use_case = intent_result.use_case
        complexity = complexity_result.level
        
        # 2. Utility Maximization (State -> Archetype Mapping)
        # Default to RAG Specialist (balanced/versatile)
        archetype = ProductArchetype.RAG_SPECIALIST
        reason = f"Default: {use_case} ({complexity.value})"
        rec_cot = False
        
        # --- LOGIC MATRIX ---
        
        # 0. Trivial / Direct Answer -> BULK_OPS (Highest priority for cost/speed)
        if complexity == ComplexityLevel.DIRECT_ANSWER:
            archetype = ProductArchetype.BULK_OPS
            reason = "Trivial Query - Cost Optimized"

        # A. High Complexity / Reasoning -> FRONTIER or REASONING_SPECIALIST
        elif complexity in [ComplexityLevel.COMPLEX_TASK, ComplexityLevel.MULTI_STEP_REASONING]:
            if category == PromptCategory.CODING or use_case in ["math_reasoning", "logic"]:
                archetype = ProductArchetype.REASONING_SPECIALIST
                reason = f"Complex Reasoning: {use_case}"
                rec_cot = True
            else:
                archetype = ProductArchetype.FRONTIER
                reason = f"Complex Strategy: {use_case}"
                rec_cot = True
                
        # B. Specific Intents
        elif use_case == "code_generation":
            archetype = ProductArchetype.REASONING_SPECIALIST
            reason = "Code Generation"
            
        elif use_case in ["creative_writing", "roleplay"]:
            archetype = ProductArchetype.FRONTIER  # Frontier models usually have best prose
            reason = "Creative Task"
            
        elif use_case in ["summarization", "translation", "text_classification"]:
            # Bulk ops for simple content tasks, unless complex
            if complexity == ComplexityLevel.SIMPLE_TASK:
                archetype = ProductArchetype.BULK_OPS
                reason = "Simple Content Task"
            else:
                archetype = ProductArchetype.RAG_SPECIALIST # Better quality for moderate tasks
                reason = "Moderate Content Task"

        elif use_case == "rag_pipeline" or (complexity == ComplexityLevel.SIMPLE_TASK and has_search_tools):
             archetype = ProductArchetype.RAG_SPECIALIST
             reason = "Context/Search Required"

        # C. Ambiguity Handling
        elif complexity == ComplexityLevel.AMBIGUOUS_QUERY:
            archetype = ProductArchetype.FRONTIER # Smartest model to handle ambiguity
            reason = "Ambiguous Query - Using most capable model"

        return RoutingDecision(
            archetype=archetype,
            category=category,
            reason=f"{reason} | Intent:{use_case} | Cpx:{complexity.value}",
            recommend_cot=rec_cot,
            cot_template=self._get_cot_template(archetype, complexity.value) if rec_cot else None
        )
