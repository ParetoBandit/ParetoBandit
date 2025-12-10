"""
Intent Classifier for 5 Core Categories.

Simplified classifier that categorizes prompts into 5 broad intents:
    - REASONING: Math, logic, analytical problem-solving → Benchmark: CRS (reasoning_score)
    - CODING: Programming, code generation, debugging → Benchmark: LiveCodeBench
    - FACTUAL_QA: Knowledge retrieval, Q&A, explanations → Benchmark: MMLU_Pro
    - AGENTIC: Tool use, API calls, external actions → Benchmark: IFEval
    - GENERAL: Chitchat, creative writing, opinions → Benchmark: MixEval_Hard

This classifier is designed for:
    1. High accuracy on common prompt types
    2. Fast inference with regex-based matching
    3. Clear decision boundaries
    4. Easy to test and evaluate

Intent → Benchmark Mapping:
    REASONING → reasoning_score (CRS: Composite Reasoning Score from Bayesian latent factor model)
    CODING → livecodebench (Skill: Can it write running code?)
    FACTUAL_QA → mmlu_pro (Knowledge: Does it know the facts?)
    AGENTIC → ifeval_score (Obedience: Can it follow instructions?)
    GENERAL → mixeval_hard (Vibes: Can it handle real-world queries?)
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional


class IntentCategory(Enum):
    """
    Five core intent categories with benchmark mappings.
    
    Each intent maps to a primary benchmark for model selection:
        REASONING → reasoning_score (CRS: Composite Reasoning Score)
        CODING → livecodebench (code generation quality)
        FACTUAL_QA → mmlu_pro (knowledge/facts)
        AGENTIC → ifeval_score (instruction following)
        GENERAL → mixeval_hard (general capability)
    """
    REASONING = "reasoning"
    CODING = "coding"
    FACTUAL_QA = "factual_qa"
    AGENTIC = "agentic"  # Renamed from AGENTIC_EXECUTION
    GENERAL = "general"
    
    # Keep alias for backwards compatibility
    AGENTIC_EXECUTION = "agentic"
    
    @property
    def benchmark(self) -> str:
        """Return the primary benchmark field for this intent."""
        mapping = {
            "reasoning": "reasoning_score",  # CRS: Bayesian latent factor model
            "coding": "livecodebench", 
            "factual_qa": "mmlu_pro",
            "agentic": "ifeval_score",
            "general": "mixeval_hard",
        }
        return mapping.get(self.value, "intelligence_index")


@dataclass
class IntentClassificationResult:
    """Result of intent classification."""
    category: IntentCategory
    confidence: float
    signals: List[str] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "signals": self.signals,
            "all_scores": self.all_scores,
            "latency_ms": self.latency_ms,
        }


# =============================================================================
# Pattern Definitions
# =============================================================================

# Patterns: (regex, weight) where weight is confidence score [0.0, 1.0]
INTENT_PATTERNS: Dict[IntentCategory, List[Tuple[str, float]]] = {
    IntentCategory.REASONING: [
        # Mathematical reasoning
        (r"\b(solve|calculate|compute|derive|prove)\b", 0.90),
        (r"\b(equation|integral|derivative|formula|theorem)\b", 0.92),
        (r"\b(math|mathematics|algebra|calculus|geometry|trigonometry)\b", 0.90),
        (r"[∫∑∏∂∇√]|(\d+\s*[+\-×÷*/]\s*\d+)", 0.95),  # Math symbols/operations
        
        # Logical reasoning
        (r"\b(logic|logical|reasoning|deduce|infer|conclude)\b", 0.85),
        (r"\b(if\s+.+\s+then|therefore|thus|hence)\b", 0.82),
        (r"\b(proof|prove|demonstrate|show that)\b", 0.88),
        
        # Analytical reasoning
        (r"\b(analyze|analysis|analytical|evaluate|assess)\b", 0.80),
        (r"\b(compare|contrast|distinguish|differentiate)\b", 0.78),
        (r"\b(statistics|statistical|probability|variance|correlation)\b", 0.85),
        (r"\b(data analysis|statistical analysis|regression)\b", 0.87),
    ],
    
    IntentCategory.CODING: [
        # Programming languages
        (r"\b(python|javascript|typescript|java|c\+\+|c#|rust|go|ruby|php|swift|kotlin)\b", 0.95),
        (r"\b(sql|html|css|bash|shell|powershell)\b", 0.92),
        
        # Code-related actions
        (r"\b(code|function|method|class|variable|array|list|dict|object)\b", 0.90),
        (r"\b(implement|write\s+(a\s+)?(function|script|program))\b", 0.93),
        (r"\b(debug|fix|bug|error|exception|trace)\b", 0.91),
        (r"\b(refactor|optimize|improve\s+(code|performance))\b", 0.90),
        
        # Programming concepts
        (r"\b(algorithm|recursive|iteration|loop|for\s+loop|while\s+loop)\b", 0.88),
        (r"\b(api|endpoint|rest|graphql|http|request|response)\b", 0.86),
        (r"\b(database|query|table|schema|orm)\b", 0.85),
        (r"\b(git|github|version control|commit|branch|merge)\b", 0.83),
        (r"\b(docker|kubernetes|container|deployment|ci/cd)\b", 0.84),
        
        # Code blocks in prompt
        (r"```[\s\S]*?```", 0.98),  # Markdown code blocks
        (r"\bdef\s+\w+\s*\(", 0.96),  # Python function definition
        (r"\bfunction\s+\w+\s*\(", 0.96),  # JS function definition
        (r"\bclass\s+\w+[:\s]", 0.94),  # Class definition
        (r"\bimport\s+\w+", 0.92),  # Import statement
    ],
    
    IntentCategory.FACTUAL_QA: [
        # Question words
        (r"^(what|who|when|where|why|which|whose)\b", 0.80),
        (r"^(how\s+(does|do|did|can|to))\b", 0.82),
        (r"\b(what\s+is|who\s+is|what\s+are|who\s+are)\b", 0.85),
        
        # Knowledge retrieval
        (r"\b(explain|describe|define|tell\s+me\s+about)\b", 0.83),
        (r"\b(fact|factual|information|knowledge|details?)\b", 0.81),
        (r"\b(meaning|definition|concept|term)\b", 0.80),
        
        # Educational
        (r"\b(teach|tutor|learn|understand|study)\b", 0.82),
        (r"\b(help\s+me\s+(understand|learn|know))\b", 0.84),
        (r"\b(step\s+by\s+step|breakdown|eli5|simple\s+terms)\b", 0.78),
        
        # Research and reference
        (r"\b(research|paper|article|source|citation|reference)\b", 0.79),
        (r"\b(according\s+to|based\s+on|evidence)\b", 0.77),
        (r"\b(history|historical|origin|etymology)\b", 0.76),
    ],
    
    IntentCategory.AGENTIC_EXECUTION: [
        # Multi-step and planning
        (r"\b(plan|planning|strategy|roadmap|steps?)\b", 0.85),
        (r"\b(multi[- ]step|multiple\s+steps|several\s+steps)\b", 0.92),
        (r"\b(workflow|pipeline|process|procedure)\b", 0.88),
        (r"\b(first.+then.+finally|step\s+1.+step\s+2)\b", 0.90),
        
        # Tool and function usage
        (r"\b(use\s+(this\s+)?tool|call\s+(the\s+)?(api|function))\b", 0.93),
        (r"\b(tool\s+use|function\s+calling|api\s+call)\b", 0.95),
        (r"\b(execute|run|perform|carry\s+out)\b", 0.82),
        
        # Agent-like behavior
        (r"\b(agent|autonomous|automate|automation)\b", 0.91),
        (r"\b(orchestrate|coordinate|manage\s+task)\b", 0.89),
        (r"\b(search\s+(for|the\s+web)|look\s+up|find\s+information)\b", 0.86),
        (r"\b(book|schedule|reserve|order|purchase)\b", 0.84),
        
        # Complex task decomposition
        (r"\b(break\s+down|decompose|subtask|component)\b", 0.87),
        (r"\b(iterative|iterate|repeat\s+until|keep\s+trying)\b", 0.85),
    ],
    
    IntentCategory.GENERAL: [
        # Conversational
        (r"\b(chat|talk|discuss|conversation)\b", 0.75),
        (r"\b(hello|hi|hey|greetings)\b", 0.80),
        (r"\b(thank|thanks|appreciate)\b", 0.78),
        
        # Vague requests
        (r"^(help|help me|can you help)\b", 0.70),
        (r"\b(something|anything|stuff|things?)\b", 0.72),
        (r"\b(ideas?|suggestions?|thoughts?|opinions?)\b", 0.68),
        
        # General assistance
        (r"\b(assist|assistance|support|advice)\b", 0.70),
        (r"\b(recommend|recommendation|suggest)\b", 0.72),
        (r"\b(tell\s+me|show\s+me|give\s+me)\b", 0.65),
    ],
}


# Keywords for boosting scores (lower weight than patterns)
INTENT_KEYWORDS: Dict[IntentCategory, List[str]] = {
    IntentCategory.REASONING: [
        "solve", "calculate", "math", "logic", "proof", "equation", "analysis",
        "statistics", "probability", "theorem", "derive"
    ],
    IntentCategory.CODING: [
        "code", "function", "python", "javascript", "debug", "api", "class",
        "method", "variable", "algorithm", "programming"
    ],
    IntentCategory.FACTUAL_QA: [
        "what", "who", "when", "where", "why", "explain", "define", "fact",
        "information", "learn", "teach", "understand"
    ],
    IntentCategory.AGENTIC_EXECUTION: [
        "plan", "workflow", "tool", "agent", "execute", "multi-step", "automate",
        "function calling", "api call"
    ],
    IntentCategory.GENERAL: [
        "chat", "help", "discuss", "suggest", "recommend", "opinion", "idea"
    ],
}


# =============================================================================
# Intent Classifier
# =============================================================================

class IntentClassifier:
    """
    Fast, regex-based intent classifier for 5 core categories.
    
    Features:
        - Pattern matching with confidence scores
        - Keyword boosting for additional signals
        - Fast inference (~1-2ms per classification)
        - Transparent scoring for debugging
    
    Example:
        >>> classifier = IntentClassifier()
        >>> result = classifier.classify("Write a Python function to sort a list")
        >>> print(result.category)  # IntentCategory.CODING
        >>> print(result.confidence)  # 0.95
    """
    
    def __init__(
        self,
        min_confidence: float = 0.60,
        keyword_boost: float = 0.05,
    ):
        """
        Initialize the intent classifier.
        
        Args:
            min_confidence: Minimum confidence to avoid GENERAL fallback
            keyword_boost: Additional boost per keyword match
        """
        self.min_confidence = min_confidence
        self.keyword_boost = keyword_boost
        
        # Compile regex patterns for efficiency
        self._compiled_patterns: Dict[IntentCategory, List[Tuple[re.Pattern, float]]] = {}
        for category, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), weight)
                for pattern, weight in patterns
            ]
    
    def classify(self, prompt: str) -> IntentClassificationResult:
        """
        Classify a prompt into one of 5 intent categories.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            IntentClassificationResult with category, confidence, and signals
        """
        start_time = time.perf_counter()
        prompt_lower = prompt.lower().strip()
        
        # Score each category
        scores: Dict[IntentCategory, Tuple[float, List[str]]] = {}
        
        for category in IntentCategory:
            score, signals = self._score_category(prompt, prompt_lower, category)
            if score > 0:
                scores[category] = (score, signals)
        
        # Find best match
        if not scores:
            # No matches at all → GENERAL with low confidence
            latency_ms = (time.perf_counter() - start_time) * 1000
            return IntentClassificationResult(
                category=IntentCategory.GENERAL,
                confidence=0.30,
                signals=["default:no_match"],
                all_scores={},
                latency_ms=latency_ms,
            )
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_category, (best_score, best_signals) = sorted_scores[0]
        
        # Build all_scores dict
        all_scores = {cat.value: score for cat, (score, _) in scores.items()}
        
        # If best score is too low and it's not GENERAL, consider falling back to GENERAL
        if best_score < self.min_confidence and best_category != IntentCategory.GENERAL:
            # Check if GENERAL has a reasonable score
            general_score = scores.get(IntentCategory.GENERAL, (0.0, []))[0]
            if general_score > best_score * 0.8:  # GENERAL is competitive
                best_category = IntentCategory.GENERAL
                best_score = general_score
                best_signals = scores[IntentCategory.GENERAL][1]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return IntentClassificationResult(
            category=best_category,
            confidence=min(best_score, 1.0),
            signals=best_signals[:5],  # Top 5 signals
            all_scores=all_scores,
            latency_ms=latency_ms,
        )
    
    def _score_category(
        self,
        prompt: str,
        prompt_lower: str,
        category: IntentCategory,
    ) -> Tuple[float, List[str]]:
        """Score how well a prompt matches a category."""
        score = 0.0
        signals = []
        
        # Pattern matching (take highest scoring pattern)
        patterns = self._compiled_patterns.get(category, [])
        for pattern, weight in patterns:
            if pattern.search(prompt):
                if weight > score:
                    score = weight
                pattern_str = pattern.pattern[:40]
                signals.append(f"pattern:{pattern_str}")
        
        # Keyword boosting (additive)
        keywords = INTENT_KEYWORDS.get(category, [])
        keyword_matches = sum(1 for kw in keywords if kw in prompt_lower)
        
        if keyword_matches > 0:
            boost = min(self.keyword_boost * keyword_matches, 0.15)  # Cap at 0.15
            score = min(score + boost, 1.0)
            signals.append(f"keywords:{keyword_matches}")
        
        return score, signals
    
    def classify_batch(self, prompts: List[str]) -> List[IntentClassificationResult]:
        """
        Classify multiple prompts.
        
        Args:
            prompts: List of prompts to classify
            
        Returns:
            List of classification results
        """
        return [self.classify(prompt) for prompt in prompts]


# =============================================================================
# Convenience Functions
# =============================================================================

# Global singleton
_default_classifier: Optional[IntentClassifier] = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the default intent classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = IntentClassifier()
    return _default_classifier


def classify_intent(prompt: str) -> IntentClassificationResult:
    """
    Quick intent classification.
    
    Args:
        prompt: Prompt to classify
        
    Returns:
        IntentClassificationResult
    """
    return get_intent_classifier().classify(prompt)

