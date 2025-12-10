"""
Standard Intent Classifier with Explicit Uncertainty Handling.

This classifier uses the standard 8-category taxonomy (from standard_taxonomy.py)
and implements proper uncertainty handling to avoid the overfitting trap.

Key Features:
    1. Broad, benchmark-aligned categories (not custom fine-grained ones)
    2. Explicit "uncertain" bucket for low-confidence/ambiguous queries
    3. Conservative fallback routing for uncertain cases
    4. Hybrid approach: fast regex + HuggingFace zero-shot fallback

The "Uncertain" Philosophy:
    "Uncertain" is NOT a failure mode - it's a valid routing decision.
    For queries falling into the 'Uncertain/Other' cluster, the optimizer
    automatically reverts to a high-entropy routing strategy (conservative
    fallback), ensuring safety by recommending well-rounded generalist models.

Usage:
    from llm_jury.routing import StandardClassifier
    
    classifier = StandardClassifier()
    result = classifier.classify("Help me with this thing")
    
    if result.category == StandardCategory.UNCERTAIN:
        # Use conservative fallback routing
        models = get_generalist_models()
    else:
        # Use category-specific routing
        models = get_specialist_models(result.category)
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from llm_jury.routing.standard_taxonomy import (
    StandardCategory,
    CATEGORY_METADATA,
    FINE_TO_STANDARD_MAPPING,
    STANDARD_ZS_LABELS,
    ZS_LABEL_LIST,
    map_to_standard_category,
    map_zs_label_to_category,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Uncertainty detection thresholds
# Note: Use <= for threshold comparisons to catch borderline cases
UNCERTAINTY_THRESHOLD = 0.70  # At or below this confidence → uncertain
AMBIGUITY_GAP_THRESHOLD = 0.10  # If top-2 within this gap → uncertain
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Above this → skip ambiguity check


@dataclass
class StandardClassificationResult:
    """
    Result from standard classification with uncertainty awareness.
    
    Attributes:
        category: The standard category (one of 8, or UNCERTAIN)
        confidence: Classification confidence (0.0 to 1.0)
        is_uncertain: Whether this was explicitly routed to uncertain bucket
        uncertainty_reason: Why it was marked uncertain (if applicable)
        signals: Classification signals/triggers
        all_scores: Scores for all categories (for debugging)
        classification_method: "regex", "zero_shot", or "hybrid"
        latency_ms: Classification time in milliseconds
        
        # For backward compatibility with fine-grained routing
        fine_grained_hint: Best guess at fine-grained category (optional)
    """
    category: StandardCategory
    confidence: float
    is_uncertain: bool = False
    uncertainty_reason: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)
    classification_method: str = "regex"
    latency_ms: float = 0.0
    fine_grained_hint: Optional[str] = None
    
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence classification."""
        return self.confidence >= HIGH_CONFIDENCE_THRESHOLD and not self.is_uncertain
    
    def should_use_conservative_routing(self) -> bool:
        """Check if conservative (generalist) routing should be used."""
        return self.is_uncertain or self.confidence < UNCERTAINTY_THRESHOLD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
            "uncertainty_reason": self.uncertainty_reason,
            "signals": self.signals,
            "classification_method": self.classification_method,
            "latency_ms": self.latency_ms,
            "fine_grained_hint": self.fine_grained_hint,
        }


# =============================================================================
# Pattern Definitions for Standard Categories
# =============================================================================

# Patterns for each standard category
# These are broader than the fine-grained patterns and focus on category-level signals
STANDARD_PATTERNS: Dict[StandardCategory, List[Tuple[str, float]]] = {
    StandardCategory.REASONING: [
        # Math signals
        (r"\b(solve|calculate|compute|derive|prove)\b", 0.85),
        (r"\b(equation|integral|derivative|formula|theorem)\b", 0.90),
        (r"\b(∫|∑|∏|∂|∇|∞|√|\d+\s*[+\-*/]\s*\d+)\b", 0.92),
        # Logic signals
        (r"\b(if.+then|therefore|conclude|deduce|infer)\b", 0.80),
        (r"\b(logic|logical|reasoning|analysis|analyze)\b", 0.75),
        # Data/analytics signals
        (r"\b(statistics|statistical|probability|variance|correlation)\b", 0.82),
        (r"\b(trend|pattern|insight|forecast|predict)\b", 0.70),
    ],
    
    StandardCategory.CODING: [
        # Explicit code signals
        (r"\b(code|function|class|method|variable|bug|debug)\b", 0.88),
        (r"\b(python|javascript|typescript|java|rust|go|c\+\+|sql)\b", 0.92),
        (r"\b(implement|refactor|optimize|compile|runtime|syntax)\b", 0.85),
        # Programming concepts
        (r"\b(algorithm|data structure|api|endpoint|database|query)\b", 0.82),
        (r"\b(git|github|docker|kubernetes|ci/cd|deploy)\b", 0.80),
        # Code blocks in prompt
        (r"```[\s\S]*```", 0.95),  # Markdown code blocks
        (r"\bdef\s+\w+\s*\(", 0.95),  # Python function
        (r"\bfunction\s+\w+\s*\(", 0.95),  # JS function
        (r"\bclass\s+\w+", 0.90),  # Class definition
    ],
    
    StandardCategory.CREATIVE: [
        # Creative writing signals
        (r"\b(write|create|compose)\b.*\b(story|poem|essay|article|novel)\b", 0.92),
        (r"\b(creative|imaginative|artistic|fiction|narrative)\b", 0.85),
        (r"\b(character|plot|scene|dialogue|setting)\b", 0.80),
        # Marketing/copy
        (r"\b(marketing|ad|advertisement|slogan|tagline|copy)\b", 0.85),
        # Roleplay
        (r"\b(roleplay|pretend|act as|you are|persona)\b", 0.90),
        # Brainstorming
        (r"\b(brainstorm|ideas?|creative|innovative|imagine)\b", 0.75),
    ],
    
    StandardCategory.FACTUAL_QA: [
        # Question patterns
        (r"^(what|who|when|where|why|how)\b", 0.72),
        (r"\b(what is|who is|explain|define|describe)\b", 0.78),
        # Knowledge signals
        (r"\b(fact|factual|information|knowledge|learn|understand)\b", 0.75),
        (r"\b(research|study|paper|article|source)\b", 0.72),
        # Tutorial/explanation - higher weight for educational intent
        (r"\b(teach|tutor|explain|help me understand)\b", 0.82),
        (r"\b(can you|could you).*(explain|understand|teach|tell me about)\b", 0.80),
        (r"\b(step by step|breakdown|eli5|beginner)\b", 0.75),
    ],
    
    StandardCategory.SUMMARIZATION: [
        # Explicit summarization
        (r"\b(summarize|summarise|summary|tldr|tl;dr)\b", 0.95),
        (r"\b(condense|brief|concise|short version)\b", 0.88),
        (r"\b(key points|main ideas|highlights|takeaways)\b", 0.85),
        # Long context signals
        (r"\b(this (document|article|text|paper))\b", 0.70),
    ],
    
    StandardCategory.EXTRACTION: [
        # Explicit extraction
        (r"\b(extract|parse|identify|find)\b.*\b(entities?|names?|data|information)\b", 0.92),
        (r"\b(ner|named entity|entity recognition)\b", 0.95),
        # Structured output
        (r"\b(json|xml|yaml|csv|structured)\b.*\b(output|format|extract)\b", 0.90),
        (r"\b(schema|fields?|properties|attributes)\b", 0.80),
        # Classification
        (r"\b(classify|categorize|label|tag|sentiment)\b", 0.88),
        (r"\b(positive|negative|neutral)\b.*\b(sentiment|tone)\b", 0.85),
        # Function calling
        (r"\b(function calling|api call|tool use)\b", 0.92),
    ],
    
    StandardCategory.TRANSLATION: [
        # Explicit translation
        (r"\b(translate|translation)\b", 0.95),
        (r"\b(to|into|from)\b\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|russian|italian|hindi)\b", 0.90),
        # Paraphrasing (same-language translation)
        (r"\b(paraphrase|rephrase|reword|rewrite)\b", 0.85),
        (r"\b(different words|another way|same meaning)\b", 0.82),
        # Grammar
        (r"\b(grammar|spelling|proofread|correct)\b.*\b(text|writing|sentence)\b", 0.80),
    ],
    
    StandardCategory.CONVERSATION: [
        # Chat signals
        (r"\b(chat|talk|converse|discuss)\b", 0.75),
        # Help with specific domain/topic (meaningful context)
        (r"\b(help me|can you|could you).*(plan|organize|prepare|decide|choose)\b", 0.78),
        # Generic help requests get lower weight (might be uncertain)
        (r"\b(help me|can you|could you|please)\b", 0.55),  # Low weight - often vague
        # Support signals
        (r"\b(support|assist|service|customer)\b", 0.72),
        # Agent/tool signals
        (r"\b(agent|autonomous|workflow|multi-?step)\b", 0.80),
        (r"\b(use this tool|call the api|execute)\b", 0.82),
    ],
}

# Keywords that boost category scores (lower weight than patterns)
STANDARD_KEYWORDS: Dict[StandardCategory, List[str]] = {
    StandardCategory.REASONING: ["math", "calculate", "solve", "logic", "analysis", "prove", "equation"],
    StandardCategory.CODING: ["code", "function", "class", "bug", "python", "javascript", "api", "debug"],
    StandardCategory.CREATIVE: ["story", "creative", "write", "poem", "character", "marketing", "imagine"],
    StandardCategory.FACTUAL_QA: ["what", "who", "explain", "define", "how", "why", "learn", "understand"],
    StandardCategory.SUMMARIZATION: ["summarize", "summary", "tldr", "brief", "condense", "key points"],
    StandardCategory.EXTRACTION: ["extract", "parse", "json", "classify", "entity", "sentiment", "structure"],
    StandardCategory.TRANSLATION: ["translate", "translation", "spanish", "french", "paraphrase", "rewrite"],
    StandardCategory.CONVERSATION: ["chat", "help", "assist", "discuss", "talk", "support"],
}


# =============================================================================
# Standard Classifier
# =============================================================================

class StandardClassifier:
    """
    Standard intent classifier with explicit uncertainty handling.
    
    Uses broad, benchmark-aligned categories and properly handles
    ambiguous/low-confidence cases by routing to the "uncertain" bucket.
    
    Example:
        >>> classifier = StandardClassifier()
        >>> result = classifier.classify("Write a Python function to sort a list")
        >>> print(result.category)  # StandardCategory.CODING
        >>> print(result.confidence)  # 0.95
        
        >>> result = classifier.classify("Help me with this")
        >>> print(result.category)  # StandardCategory.UNCERTAIN
        >>> print(result.is_uncertain)  # True
        >>> print(result.uncertainty_reason)  # "low_confidence"
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
        ambiguity_gap_threshold: float = AMBIGUITY_GAP_THRESHOLD,
        high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the standard classifier.
        
        Args:
            uncertainty_threshold: Below this confidence → mark as uncertain
            ambiguity_gap_threshold: If top-2 within this gap → mark as uncertain
            high_confidence_threshold: Above this → skip ambiguity check
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.ambiguity_gap_threshold = ambiguity_gap_threshold
        self.high_confidence_threshold = high_confidence_threshold
        
        # Compile regex patterns for efficiency
        self._compiled_patterns: Dict[StandardCategory, List[Tuple[re.Pattern, float]]] = {}
        for category, patterns in STANDARD_PATTERNS.items():
            self._compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]
    
    def classify(self, prompt: str) -> StandardClassificationResult:
        """
        Classify a prompt into one of the 8 standard categories.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            StandardClassificationResult with category, confidence, and uncertainty info
        """
        start_time = time.perf_counter()
        prompt_lower = prompt.lower()
        
        # Score each category
        scores: Dict[StandardCategory, Tuple[float, List[str]]] = {}
        
        for category in StandardCategory:
            if category == StandardCategory.UNCERTAIN:
                continue  # Don't score uncertain directly
            
            score, signals = self._score_category(prompt, prompt_lower, category)
            if score > 0:
                scores[category] = (score, signals)
        
        # Handle no matches
        if not scores:
            return self._create_uncertain_result(
                reason="no_pattern_match",
                all_scores={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_category, (best_score, best_signals) = sorted_scores[0]
        
        # Build all_scores dict for debugging
        all_scores = {cat.value: score for cat, (score, _) in scores.items()}
        
        # Check for uncertainty conditions
        uncertainty_reason = self._check_uncertainty(sorted_scores)
        
        if uncertainty_reason:
            return self._create_uncertain_result(
                reason=uncertainty_reason,
                all_scores=all_scores,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                fine_grained_hint=best_category.value,  # What we think it might be
                original_confidence=best_score,
            )
        
        # Confident classification
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return StandardClassificationResult(
            category=best_category,
            confidence=min(best_score, 1.0),
            is_uncertain=False,
            uncertainty_reason=None,
            signals=best_signals[:5],
            all_scores=all_scores,
            classification_method="regex",
            latency_ms=latency_ms,
            fine_grained_hint=best_category.value,
        )
    
    def _score_category(
        self,
        prompt: str,
        prompt_lower: str,
        category: StandardCategory,
    ) -> Tuple[float, List[str]]:
        """Score how well a prompt matches a category."""
        score = 0.0
        signals = []
        
        # Pattern matching
        patterns = self._compiled_patterns.get(category, [])
        for pattern, weight in patterns:
            if pattern.search(prompt):
                if weight > score:
                    score = weight
                signals.append(f"pattern:{pattern.pattern[:30]}...")
        
        # Keyword boosting
        keywords = STANDARD_KEYWORDS.get(category, [])
        keyword_matches = sum(1 for kw in keywords if kw in prompt_lower)
        
        if keyword_matches > 0:
            # Small additive boost for keywords
            keyword_boost = min(0.15, 0.05 * keyword_matches)
            score = max(score, score + keyword_boost)
            signals.append(f"keywords:{keyword_matches}")
        
        return score, signals
    
    def _check_uncertainty(
        self,
        sorted_scores: List[Tuple[StandardCategory, Tuple[float, List[str]]]],
    ) -> Optional[str]:
        """
        Check if the classification should be marked as uncertain.
        
        Returns:
            Uncertainty reason string, or None if confident
        """
        best_category, (best_score, _) = sorted_scores[0]
        
        # Condition 1: Low confidence (use <= to catch borderline cases)
        if best_score <= self.uncertainty_threshold:
            return "low_confidence"
        
        # Condition 2: High confidence → skip ambiguity check
        if best_score >= self.high_confidence_threshold:
            return None
        
        # Condition 3: Ambiguity between top categories
        if len(sorted_scores) >= 2:
            second_category, (second_score, _) = sorted_scores[1]
            gap = best_score - second_score
            
            if gap < self.ambiguity_gap_threshold:
                return f"ambiguous:{best_category.value}≈{second_category.value}"
        
        return None
    
    def _create_uncertain_result(
        self,
        reason: str,
        all_scores: Dict[str, float],
        latency_ms: float,
        fine_grained_hint: Optional[str] = None,
        original_confidence: float = 0.0,
    ) -> StandardClassificationResult:
        """Create an uncertain classification result."""
        return StandardClassificationResult(
            category=StandardCategory.UNCERTAIN,
            confidence=original_confidence,
            is_uncertain=True,
            uncertainty_reason=reason,
            signals=[f"uncertainty:{reason}"],
            all_scores=all_scores,
            classification_method="regex",
            latency_ms=latency_ms,
            fine_grained_hint=fine_grained_hint,
        )
    
    def classify_batch(
        self,
        prompts: List[str],
    ) -> List[StandardClassificationResult]:
        """
        Classify multiple prompts.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            List of classification results
        """
        return [self.classify(prompt) for prompt in prompts]


# =============================================================================
# Hybrid Standard Classifier (with Zero-Shot Fallback)
# =============================================================================

class HybridStandardClassifier:
    """
    Hybrid classifier that uses regex + zero-shot fallback.
    
    Strategy:
    1. Run regex-based classification (fast, ~1ms)
    2. If uncertain or low-confidence, use zero-shot classification
    3. Return best result with proper uncertainty handling
    
    Usage:
        >>> classifier = HybridStandardClassifier()
        >>> result = classifier.classify("Write a haiku about coding")
        >>> print(result.category)  # StandardCategory.CREATIVE
    """
    
    def __init__(
        self,
        fallback_threshold: float = 0.75,
        use_api: bool = False,
        hf_api_token: Optional[str] = None,
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            fallback_threshold: Use zero-shot if regex confidence below this
            use_api: Use HuggingFace API (no local model download)
            hf_api_token: API token for HuggingFace (if use_api=True)
        """
        self.fallback_threshold = fallback_threshold
        self.use_api = use_api
        self.hf_api_token = hf_api_token
        
        self.regex_classifier = StandardClassifier()
        self._zs_classifier = None  # Lazy loading
    
    @property
    def zs_classifier(self):
        """Lazy load zero-shot classifier."""
        if self._zs_classifier is None:
            self._load_zs_classifier()
        return self._zs_classifier
    
    def _load_zs_classifier(self):
        """Load the zero-shot classifier."""
        # Import here to avoid circular imports and allow optional dependency
        from llm_jury.routing.hybrid_classifier import (
            HuggingFaceClassifier,
            HuggingFaceAPIClassifier,
        )
        
        if self.use_api:
            self._zs_classifier = HuggingFaceAPIClassifier(
                labels=ZS_LABEL_LIST,
                api_token=self.hf_api_token,
            )
        else:
            self._zs_classifier = HuggingFaceClassifier(
                labels=ZS_LABEL_LIST,
            )
    
    def classify(
        self,
        prompt: str,
        force_zs: bool = False,
    ) -> StandardClassificationResult:
        """
        Classify using hybrid approach.
        
        Args:
            prompt: Prompt to classify
            force_zs: Force zero-shot classification (skip regex)
            
        Returns:
            StandardClassificationResult
        """
        start_time = time.perf_counter()
        
        # Stage 1: Regex classification
        if not force_zs:
            regex_result = self.regex_classifier.classify(prompt)
            
            # If confident and not uncertain, return regex result
            if (regex_result.confidence >= self.fallback_threshold 
                and not regex_result.is_uncertain):
                return regex_result
        
        # Stage 2: Zero-shot fallback
        try:
            zs_label, zs_score, all_zs_scores = self.zs_classifier.classify(prompt)
            
            # Map zero-shot label to standard category
            category = map_zs_label_to_category(zs_label)
            
            # Check uncertainty for zero-shot result
            is_uncertain = zs_score < self.regex_classifier.uncertainty_threshold
            
            if category == StandardCategory.UNCERTAIN or is_uncertain:
                return StandardClassificationResult(
                    category=StandardCategory.UNCERTAIN,
                    confidence=zs_score,
                    is_uncertain=True,
                    uncertainty_reason="zs_low_confidence",
                    signals=[f"zs_label:{zs_label}"],
                    all_scores={k: v for k, v in all_zs_scores.items()},
                    classification_method="zero_shot",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )
            
            return StandardClassificationResult(
                category=category,
                confidence=zs_score,
                is_uncertain=False,
                signals=[f"zs_label:{zs_label}"],
                all_scores={k: v for k, v in all_zs_scores.items()},
                classification_method="zero_shot",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
            # Fall back to regex result or uncertain
            if not force_zs:
                return regex_result
            else:
                return StandardClassificationResult(
                    category=StandardCategory.UNCERTAIN,
                    confidence=0.0,
                    is_uncertain=True,
                    uncertainty_reason="zs_error",
                    signals=[f"error:{str(e)[:50]}"],
                    all_scores={},
                    classification_method="error",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                )


# =============================================================================
# Convenience Functions
# =============================================================================

# Global singleton
_default_classifier: Optional[StandardClassifier] = None


def get_standard_classifier() -> StandardClassifier:
    """Get or create the default standard classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = StandardClassifier()
    return _default_classifier


def classify_standard(prompt: str) -> StandardClassificationResult:
    """
    Quick classification using standard taxonomy.
    
    Args:
        prompt: Prompt to classify
        
    Returns:
        StandardClassificationResult
    """
    return get_standard_classifier().classify(prompt)


def is_uncertain(prompt: str) -> bool:
    """
    Check if a prompt should be routed to the uncertain bucket.
    
    Args:
        prompt: Prompt to check
        
    Returns:
        True if uncertain/ambiguous
    """
    result = classify_standard(prompt)
    return result.is_uncertain


def get_routing_strategy(prompt: str) -> str:
    """
    Get the recommended routing strategy for a prompt.
    
    Args:
        prompt: Prompt to analyze
        
    Returns:
        "conservative_fallback" for uncertain, or category-specific strategy
    """
    result = classify_standard(prompt)
    
    if result.should_use_conservative_routing():
        return "conservative_fallback"
    
    metadata = CATEGORY_METADATA.get(result.category)
    return metadata.routing_strategy if metadata else "generalist"

