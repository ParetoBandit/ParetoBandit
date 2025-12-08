"""
Tier-based prompt router for simple model selection.

Moved from prompt_router.py with minimal changes.
Determines optimal model tier (Small/Composite/Large) based on prompt complexity.
"""

import re
import math
from typing import List
from dataclasses import dataclass

from llm_jury.core.models import ModelTier


@dataclass
class RoutingDecision:
    """Simple routing decision for tier-based routing."""
    tier: ModelTier
    confidence: float
    reason: str
    required_tools: List[str]


class PromptRouter:
    """
    Simple tier-based router for prompt classification.
    
    Routes prompts to:
    - SMALL: Simple, static queries
    - COMPOSITE: Queries needing search/live data
    - LARGE: Complex reasoning, coding, or long-context tasks
    """
    
    def __init__(self):
        # 1. TEMPORAL INDICATORS: Signal need for Search/Live Data
        self.search_triggers = [
            r"\b(current|latest|new|recent|today|now|live)\b",
            r"\b(price|stock|score|winner|weather|news)\b",
            r"\b(who won|when is|what happened)\b",
            r"\b(2024|2025)\b"  # Explicit recent years
        ]

        # 2. REASONING INDICATORS: Signal need for Large Models (Logic/Coding)
        self.reasoning_triggers = [
            r"\b(code|python|function|algorithm|debug|script)\b",
            r"\b(strategy|analyze|synthesis|evaluate|critique)\b",
            r"\b(math|calculus|physics|proof)\b",
            r"\b(step-by-step|chain of thought)\b",
            r"\b(nuance|style of|creative|poem|novel)\b"
        ]

        # 3. COMPLEXITY THRESHOLDS
        self.length_threshold_tokens = 500  # Approximated by chars
        
    def _calculate_perplexity_proxy(self, text: str) -> float:
        """
        A heuristic to estimate complexity. 
        Higher unique word density + longer words = higher 'cognitive load'.
        """
        words = text.split()
        if not words:
            return 0.0
        avg_word_len = sum(len(w) for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        return avg_word_len * unique_ratio * math.log(len(words) + 1)

    def route(self, prompt: str) -> RoutingDecision:
        """
        Determines the optimal model tier for a given prompt.
        
        Args:
            prompt: User prompt to route
            
        Returns:
            RoutingDecision with tier, confidence, reason, and required tools
        """
        prompt_lower = prompt.lower()
        
        # --- PHASE 1: SEARCH DETECTION (Composite Tier) ---
        search_score = 0
        for pattern in self.search_triggers:
            if re.search(pattern, prompt_lower):
                search_score += 1
        
        # If explicitly asking for recent info, it MUST go to Search (Composite)
        if search_score > 0:
            return RoutingDecision(
                tier=ModelTier.COMPOSITE,
                confidence=0.9,
                reason="Detected temporal keywords or information retrieval intent.",
                required_tools=["tavily_search"]
            )

        # --- PHASE 2: REASONING & COMPLEXITY (Large Tier) ---
        reasoning_score = 0
        for pattern in self.reasoning_triggers:
            if re.search(pattern, prompt_lower):
                reasoning_score += 1

        # Calculate a proxy for complexity (length & vocabulary density)
        complexity_metric = self._calculate_perplexity_proxy(prompt)
        is_long_context = len(prompt) > (self.length_threshold_tokens * 4)  # approx 4 chars/token

        if reasoning_score > 0 or is_long_context or complexity_metric > 15.0:
            reason_msg = []
            if reasoning_score > 0:
                reason_msg.append("Detected complex reasoning keywords.")
            if is_long_context:
                reason_msg.append("Prompt exceeds token threshold for small models.")
            if complexity_metric > 15.0:
                reason_msg.append("High linguistic complexity.")
            
            return RoutingDecision(
                tier=ModelTier.LARGE,
                confidence=0.85,
                reason=" ".join(reason_msg),
                required_tools=[]
            )

        # --- PHASE 3: DEFAULT FALLBACK (Small Tier) ---
        return RoutingDecision(
            tier=ModelTier.SMALL,
            confidence=0.7,
            reason="Prompt is concise, static, and low-complexity. Safe for optimization.",
            required_tools=[]
        )
