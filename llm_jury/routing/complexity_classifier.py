"""
Complexity Classifier for Adaptive Model Routing.

Classifies prompts by complexity level to enable intelligent routing decisions:
- Simple queries → faster/cheaper models
- Complex queries → more capable/expensive models
- Ambiguous queries → request clarification

Complexity Levels:
    
    DIRECT_ANSWER (Trivial)
        No retrieval, simple response. Fast models work well.
        Examples: "What is 2+2?", "Hello!", "What's the capital of France?"
    
    SIMPLE_TASK (Single-step)
        One operation or simple lookup. Most models handle well.
        Examples: "Write hello world in Python", "Translate to Spanish"
    
    MULTI_STEP_REASONING (Moderate)
        Multiple operations, comparisons, analysis. Needs capable models.
        Examples: "Compare X vs Y", "Analyze this and suggest improvements"
    
    COMPLEX_TASK (Deep reasoning)
        Multi-hop reasoning, long-form generation, expert knowledge.
        Examples: "Design a system architecture", "Write a business plan"
    
    AMBIGUOUS_QUERY (Needs clarification)
        Too vague to process effectively. Should ask for details.
        Examples: "Help me with this", "Tell me about it"

Usage:
    from llm_jury.routing import ComplexityClassifier, HybridComplexityClassifier
    
    # Regex-only (fast, no API needed)
    classifier = ComplexityClassifier()
    result = classifier.classify("What is 2+2?")
    print(result.complexity)  # "direct_answer"
    
    # Hybrid with HuggingFace fallback
    hybrid = HybridComplexityClassifier()
    result = hybrid.classify("Some ambiguous request")
    print(result.complexity)  # Uses HF if regex is uncertain

Adaptive RAG Integration:
    complexity = classifier.classify(query)
    
    if complexity.level == "direct_answer":
        # No retrieval needed, answer directly
        response = llm.generate(query)
    elif complexity.level == "simple_retrieval":
        # Single search
        docs = retriever.search(query, k=3)
        response = llm.generate(query, context=docs)
    elif complexity.level == "multi_step_reasoning":
        # Multiple searches, chain-of-thought
        response = agent.reason_and_retrieve(query)
    elif complexity.level == "ambiguous_query":
        # Ask for clarification
        response = "Could you please provide more details?"
"""

import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Complexity Levels
# =============================================================================

class ComplexityLevel(Enum):
    """Complexity levels for prompt classification."""
    DIRECT_ANSWER = "direct_answer"           # Trivial, no retrieval
    SIMPLE_TASK = "simple_task"               # Single-step operation
    MULTI_STEP_REASONING = "multi_step_reasoning"  # Multiple operations/analysis
    COMPLEX_TASK = "complex_task"             # Deep reasoning, expert knowledge
    AMBIGUOUS_QUERY = "ambiguous_query"       # Needs clarification


@dataclass
class ComplexityResult:
    """Result of complexity classification."""
    complexity: str                           # ComplexityLevel value
    level: ComplexityLevel                    # Enum for type safety
    confidence: float                         # 0.0 to 1.0
    signals: List[str]                        # What triggered this classification
    reasoning: str                            # Human-readable explanation
    suggested_action: str                     # What to do with this complexity
    
    # Routing hints
    needs_retrieval: bool = False             # Should we search for info?
    needs_clarification: bool = False         # Should we ask for more details?
    estimated_steps: int = 1                  # Estimated reasoning steps
    recommended_model_tier: str = "standard"  # "fast", "standard", "premium"


# =============================================================================
# Complexity Classifier (Regex-based)
# =============================================================================

class ComplexityClassifier:
    """
    Classifies prompt complexity using pattern matching.
    
    Uses a multi-signal approach:
    1. Structural patterns (question words, operators, length)
    2. Complexity indicators (comparisons, multi-step words)
    3. Ambiguity signals (vague references, missing context)
    """
    
    def __init__(self):
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize complexity detection patterns."""
        
        # =================================================================
        # DIRECT_ANSWER patterns - Trivial queries
        # =================================================================
        self.direct_answer_patterns = [
            # Greetings and small talk
            (r"^(hi|hello|hey|howdy|greetings|good (morning|afternoon|evening))[\s!?.]*$", 0.98),
            (r"^(how are you|what'?s up|how'?s it going)[\s?]*$", 0.95),
            (r"^(thanks|thank you|thx|ty)[\s!.]*$", 0.98),
            (r"^(yes|no|ok|okay|sure|alright)[\s!.]*$", 0.95),
            
            # Simple math
            (r"^what (is|are|\=) \d+\s*[\+\-\*\/\%]\s*\d+", 0.98),
            (r"^\d+\s*[\+\-\*\/]\s*\d+\s*[\=\?]?\s*$", 0.98),
            
            # Simple factual questions (well-known facts)
            (r"^what (is|are) the capital of", 0.92),
            (r"^who (is|was) the (president|ceo|founder|author) of", 0.88),
            (r"^when (was|did|is) .{3,30} (born|founded|created|released)", 0.85),
            (r"^how (many|much) .{3,20} (in|are there)", 0.85),
            
            # Definitions
            (r"^(what|who) (is|are|was|were) (a|an|the)?\s*\w+\s*\??$", 0.85),
            (r"^define\s+\w+", 0.90),
            
            # Yes/no questions about simple facts
            (r"^(is|are|was|were|do|does|did|can|could|will|would) .{5,40}\??$", 0.75),
        ]
        
        # =================================================================
        # SIMPLE_TASK patterns - Single operation
        # =================================================================
        self.simple_task_patterns = [
            # Single action requests
            (r"^(write|create|make|generate)\s+(a|an|the)?\s*(simple|basic|short)?\s*(hello world|function|script)", 0.92),
            (r"^(translate|convert)\s+.{3,50}\s+(to|into)\s+\w+", 0.90),
            (r"^(summarize|sum up)\s+(this|the|a)\s+\w+", 0.88),
            (r"^(list|show|give me)\s+(\d+|a few|some)\s+\w+", 0.85),
            
            # Simple lookups
            (r"^(find|search|look up|get)\s+(the|a)?\s*\w+\s*(of|for|about)?\s*\w*", 0.82),
            (r"^(what|when|where|who)\s+.{10,60}\??$", 0.75),
            
            # Format/style changes
            (r"^(fix|correct|improve)\s+(the|this)?\s*(grammar|spelling|punctuation)", 0.90),
            (r"^(rewrite|rephrase)\s+(this|the)\s+(sentence|paragraph|text)", 0.85),
            
            # Simple code tasks
            (r"^(write|create)\s+(a|an)?\s*(function|method|class)\s+(to|that|for)\s+\w+", 0.85),
            (r"^(explain|describe)\s+(this|the|what)\s+(code|function|error)", 0.82),
        ]
        
        # =================================================================
        # MULTI_STEP_REASONING patterns - Multiple operations
        # =================================================================
        self.multi_step_patterns = [
            # Comparison requests
            (r"\b(compare|contrast|versus|vs\.?|difference between)\b", 0.92),
            (r"\b(which (is|are) (better|worse|faster|cheaper))\b", 0.88),
            (r"\b(pros and cons|advantages and disadvantages)\b", 0.95),
            
            # Analysis requests
            (r"\b(analyze|analyse|evaluate|assess)\b.*\b(and|then)\b", 0.90),
            (r"\b(explain (why|how)|what (caused|led to))\b", 0.85),
            (r"\b(identify|find)\b.*\b(patterns?|trends?|issues?)\b", 0.85),
            
            # Multi-part questions
            (r"\b(first|then|after that|finally|next)\b.*\b(and|then)\b", 0.88),
            (r"\b(step by step|step-by-step)\b", 0.90),
            (r"\b(how (do|can|should) (i|we|you))\b.*\b(and|then|also)\b", 0.85),
            
            # Research/synthesis
            (r"\b(research|investigate|explore)\b.*\b(and|then)\b", 0.88),
            (r"\b(summarize|synthesize)\b.*\b(multiple|several|various)\b", 0.90),
            
            # Conditional reasoning
            (r"\b(if|when|assuming)\b.*\b(then|what|how)\b", 0.80),
            (r"\b(depends on|based on|considering)\b", 0.75),
        ]
        
        # =================================================================
        # COMPLEX_TASK patterns - Deep reasoning
        # =================================================================
        self.complex_task_patterns = [
            # System design
            (r"\b(design|architect|build)\b.*\b(system|architecture|platform|infrastructure)\b", 0.95),
            (r"\b(create|develop|build)\b.*\b(comprehensive|complete|full|end-to-end)\b", 0.92),
            
            # Long-form generation
            (r"\b(write|create|draft)\b.*\b(business plan|proposal|report|whitepaper)\b", 0.95),
            (r"\b(write|create)\b.*\b(detailed|comprehensive|thorough)\b.*\b(guide|documentation|analysis)\b", 0.92),
            
            # Expert-level tasks
            (r"\b(debug|troubleshoot|diagnose)\b.*\b(complex|production|system)\b", 0.90),
            (r"\b(optimize|refactor)\b.*\b(entire|whole|complete)\b", 0.88),
            
            # Multi-domain reasoning
            (r"\b(considering|taking into account)\b.*\b(multiple|various|different)\b.*\b(factors|aspects|perspectives)\b", 0.90),
            (r"\b(trade-?offs?|implications|consequences)\b", 0.82),
            
            # Strategic/planning tasks
            (r"\b(strategy|strategic|roadmap|plan)\b.*\b(for|to)\b.*\b(year|quarter|long-?term)\b", 0.92),
            (r"\b(recommend|suggest)\b.*\b(approach|solution|strategy)\b.*\b(for|to)\b", 0.85),
            
            # Length/depth indicators combined with complexity
            (r"\b(in-?depth|thorough|exhaustive|comprehensive)\b.*\b(analysis|review|examination)\b", 0.92),
        ]
        
        # =================================================================
        # AMBIGUOUS_QUERY patterns - Needs clarification
        # =================================================================
        self.ambiguous_patterns = [
            # Vague references
            (r"^(help|help me)(\s+with)?\s*(this|it|that|something)?\s*\.?$", 0.95),
            (r"^(tell me|explain|describe)\s+(about\s+)?(it|this|that)\s*\.?$", 0.95),
            (r"^(what|how)\s+about\s+(it|this|that)\s*\??$", 0.92),
            
            # Missing context
            (r"^(the|this|that)\s+\w+\s*\??$", 0.85),  # "the user?", "this thing?"
            (r"^(it|this|that)\s+(is|was|does|doesn't|won't)\s*", 0.82),
            
            # Too short/vague
            (r"^.{1,10}$", 0.70),  # Very short queries often lack context
            (r"^\w+\s*\??$", 0.80),  # Single word queries
            
            # Unclear intent
            (r"^(what do you think|your thoughts|your opinion)\s*\??$", 0.88),
            (r"^(i need|i want|can you)\s+(help|something|anything)\s*\.?$", 0.90),
            (r"^(do|can|could|would) (you|it)\s*\??$", 0.85),
            
            # Pronouns without antecedents
            (r"^(he|she|they|it|we)\s+\w+", 0.75),
            (r"\b(the user|the customer|the client|the system)\b(?!.*\b(named|called|at|from|in)\b)", 0.80),
        ]
        
        # =================================================================
        # Complexity multipliers based on structural features
        # =================================================================
        self.complexity_boosters = [
            # Length indicators (longer = likely more complex)
            (r".{200,}", 0.15),   # 200+ chars
            (r".{400,}", 0.25),   # 400+ chars
            
            # Multiple questions
            (r"\?.*\?", 0.20),    # Multiple question marks
            
            # Enumeration (multiple items to address)
            (r"\b(1\.|a\)|first,?)\b.*\b(2\.|b\)|second,?)\b", 0.25),
            
            # Technical depth
            (r"\b(algorithm|architecture|implementation|integration)\b", 0.15),
            
            # Temporal scope
            (r"\b(over time|historically|trend|evolution)\b", 0.15),
        ]
        
        self.complexity_reducers = [
            # Simplicity indicators
            (r"\b(simple|basic|quick|brief|short)\b", -0.15),
            (r"\b(just|only|simply)\b", -0.10),
            (r"\b(example|sample|demo)\b", -0.10),
        ]
    
    def classify(self, prompt: str) -> ComplexityResult:
        """
        Classify the complexity of a prompt.
        
        Args:
            prompt: The user's prompt/query
            
        Returns:
            ComplexityResult with complexity level and metadata
        """
        prompt_lower = prompt.lower().strip()
        signals = []
        
        # Score each complexity level
        scores = {
            ComplexityLevel.DIRECT_ANSWER: 0.0,
            ComplexityLevel.SIMPLE_TASK: 0.0,
            ComplexityLevel.MULTI_STEP_REASONING: 0.0,
            ComplexityLevel.COMPLEX_TASK: 0.0,
            ComplexityLevel.AMBIGUOUS_QUERY: 0.0,
        }
        
        # Check DIRECT_ANSWER patterns
        for pattern, weight in self.direct_answer_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ComplexityLevel.DIRECT_ANSWER] = max(scores[ComplexityLevel.DIRECT_ANSWER], weight)
                signals.append(f"direct:{pattern[:30]}")
        
        # Check SIMPLE_TASK patterns
        for pattern, weight in self.simple_task_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ComplexityLevel.SIMPLE_TASK] = max(scores[ComplexityLevel.SIMPLE_TASK], weight)
                signals.append(f"simple:{pattern[:30]}")
        
        # Check MULTI_STEP patterns
        for pattern, weight in self.multi_step_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ComplexityLevel.MULTI_STEP_REASONING] = max(scores[ComplexityLevel.MULTI_STEP_REASONING], weight)
                signals.append(f"multi:{pattern[:30]}")
        
        # Check COMPLEX_TASK patterns
        for pattern, weight in self.complex_task_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ComplexityLevel.COMPLEX_TASK] = max(scores[ComplexityLevel.COMPLEX_TASK], weight)
                signals.append(f"complex:{pattern[:30]}")
        
        # Check AMBIGUOUS patterns
        for pattern, weight in self.ambiguous_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                scores[ComplexityLevel.AMBIGUOUS_QUERY] = max(scores[ComplexityLevel.AMBIGUOUS_QUERY], weight)
                signals.append(f"ambiguous:{pattern[:30]}")
        
        # Apply boosters/reducers
        adjustment = 0.0
        for pattern, adj in self.complexity_boosters + self.complexity_reducers:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                adjustment += adj
                signals.append(f"adj:{adj:+.2f}")
        
        # Apply adjustment to multi-step and complex scores
        scores[ComplexityLevel.MULTI_STEP_REASONING] += adjustment * 0.5
        scores[ComplexityLevel.COMPLEX_TASK] += adjustment
        
        # Determine winner
        best_level = max(scores, key=scores.get)
        best_score = scores[best_level]
        
        # Default to SIMPLE_TASK if nothing matched strongly
        if best_score < 0.5:
            best_level = ComplexityLevel.SIMPLE_TASK
            best_score = 0.5
            signals.append("default:simple_task")
        
        # Generate result
        return self._create_result(best_level, best_score, signals, prompt)
    
    def _create_result(
        self, 
        level: ComplexityLevel, 
        confidence: float, 
        signals: List[str],
        prompt: str,
    ) -> ComplexityResult:
        """Create a ComplexityResult with appropriate metadata."""
        
        # Level-specific metadata
        metadata = {
            ComplexityLevel.DIRECT_ANSWER: {
                "reasoning": "Simple factual or conversational query that can be answered directly.",
                "action": "Answer directly without retrieval. Use fast/cheap model.",
                "needs_retrieval": False,
                "needs_clarification": False,
                "steps": 1,
                "tier": "fast",
            },
            ComplexityLevel.SIMPLE_TASK: {
                "reasoning": "Single-step task or simple lookup that requires one operation.",
                "action": "Execute single operation. May need basic retrieval.",
                "needs_retrieval": True,
                "needs_clarification": False,
                "steps": 1,
                "tier": "standard",
            },
            ComplexityLevel.MULTI_STEP_REASONING: {
                "reasoning": "Task requires multiple steps, comparisons, or analysis.",
                "action": "Break into steps, may need multiple retrievals. Use capable model.",
                "needs_retrieval": True,
                "needs_clarification": False,
                "steps": 3,
                "tier": "standard",
            },
            ComplexityLevel.COMPLEX_TASK: {
                "reasoning": "Complex task requiring deep reasoning or expert knowledge.",
                "action": "Use chain-of-thought, multiple retrievals. Premium model recommended.",
                "needs_retrieval": True,
                "needs_clarification": False,
                "steps": 5,
                "tier": "premium",
            },
            ComplexityLevel.AMBIGUOUS_QUERY: {
                "reasoning": "Query is too vague or lacks necessary context.",
                "action": "Ask for clarification before proceeding.",
                "needs_retrieval": False,
                "needs_clarification": True,
                "steps": 0,
                "tier": "fast",
            },
        }
        
        meta = metadata[level]
        
        return ComplexityResult(
            complexity=level.value,
            level=level,
            confidence=min(confidence, 1.0),
            signals=signals[:5],  # Keep top 5 signals
            reasoning=meta["reasoning"],
            suggested_action=meta["action"],
            needs_retrieval=meta["needs_retrieval"],
            needs_clarification=meta["needs_clarification"],
            estimated_steps=meta["steps"],
            recommended_model_tier=meta["tier"],
        )


# =============================================================================
# HuggingFace Complexity Classifier (Fallback)
# =============================================================================

# Labels for zero-shot classification
COMPLEXITY_LABELS = [
    "simple factual question",      # → direct_answer
    "basic single task",            # → simple_task
    "multi-step analysis",          # → multi_step_reasoning
    "complex reasoning task",       # → complex_task
    "vague or unclear request",     # → ambiguous_query
]

# Map HF labels to our complexity levels
HF_LABEL_TO_COMPLEXITY = {
    "simple factual question": ComplexityLevel.DIRECT_ANSWER,
    "basic single task": ComplexityLevel.SIMPLE_TASK,
    "multi-step analysis": ComplexityLevel.MULTI_STEP_REASONING,
    "complex reasoning task": ComplexityLevel.COMPLEX_TASK,
    "vague or unclear request": ComplexityLevel.AMBIGUOUS_QUERY,
}


class HuggingFaceComplexityClassifier:
    """
    Complexity classifier using HuggingFace zero-shot classification.
    
    Supports two modes:
    1. Local model (downloads on first use)
    2. Inference API (no download, needs API token)
    """
    
    API_URL = "https://router.huggingface.co/hf-inference/models/"
    DEFAULT_MODEL = "facebook/bart-large-mnli"
    
    def __init__(
        self,
        use_api: bool = False,
        model_name: str = DEFAULT_MODEL,
        api_token: Optional[str] = None,
    ):
        self.use_api = use_api
        self.model_name = model_name
        self.api_token = api_token or self._load_api_token()
        self._pipeline = None
    
    @staticmethod
    def _load_api_token() -> Optional[str]:
        """Load API token from .env or environment."""
        import os
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        for var in ["HUGGINGFACE_API_KEY", "HF_API_TOKEN", "HF_TOKEN"]:
            token = os.environ.get(var)
            if token:
                return token
        return None
    
    def _get_pipeline(self):
        """Lazy load the local pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
        return self._pipeline
    
    def classify(self, prompt: str) -> Tuple[ComplexityLevel, float, Dict[str, float]]:
        """
        Classify complexity using HuggingFace.
        
        Returns:
            Tuple of (complexity_level, confidence, all_scores)
        """
        if self.use_api:
            return self._classify_api(prompt)
        else:
            return self._classify_local(prompt)
    
    def _classify_local(self, prompt: str) -> Tuple[ComplexityLevel, float, Dict[str, float]]:
        """Classify using local model."""
        pipeline = self._get_pipeline()
        result = pipeline(prompt, COMPLEXITY_LABELS)
        
        scores = {label: score for label, score in zip(result["labels"], result["scores"])}
        best_label = result["labels"][0]
        best_score = result["scores"][0]
        
        level = HF_LABEL_TO_COMPLEXITY.get(best_label, ComplexityLevel.SIMPLE_TASK)
        return level, best_score, scores
    
    def _classify_api(self, prompt: str) -> Tuple[ComplexityLevel, float, Dict[str, float]]:
        """Classify using HuggingFace Inference API."""
        import requests
        
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {"candidate_labels": COMPLEXITY_LABELS}
        }
        
        url = f"{self.API_URL}{self.model_name}"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Handle response format
            if isinstance(result, list):
                scores = {item["label"]: item["score"] for item in result}
                best_label = result[0]["label"]
                best_score = result[0]["score"]
            else:
                scores = {label: score for label, score in zip(result["labels"], result["scores"])}
                best_label = result["labels"][0]
                best_score = result["scores"][0]
            
            level = HF_LABEL_TO_COMPLEXITY.get(best_label, ComplexityLevel.SIMPLE_TASK)
            return level, best_score, scores
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            return ComplexityLevel.SIMPLE_TASK, 0.5, {}


# =============================================================================
# Hybrid Complexity Classifier
# =============================================================================

class HybridComplexityClassifier:
    """
    Hybrid complexity classifier with regex-first and HuggingFace fallback.
    
    Strategy:
    1. Try regex classification first (fast, free, no API needed)
    2. If confidence < threshold, use HuggingFace for better accuracy
    
    Usage:
        # Default: regex only, HF fallback when needed
        classifier = HybridComplexityClassifier()
        
        # Force API mode for HF
        classifier = HybridComplexityClassifier(use_api=True)
        
        result = classifier.classify("Compare X vs Y")
        print(result.complexity)         # "multi_step_reasoning"
        print(result.classification_method)  # "regex" or "huggingface"
    """
    
    def __init__(
        self,
        fallback_threshold: float = 0.70,
        use_api: bool = False,
        hf_model: str = HuggingFaceComplexityClassifier.DEFAULT_MODEL,
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            fallback_threshold: Confidence below this triggers HF fallback
            use_api: Use HF Inference API instead of local model
            hf_model: HuggingFace model for zero-shot classification
        """
        self.fallback_threshold = fallback_threshold
        self.use_api = use_api
        
        self.regex_classifier = ComplexityClassifier()
        self._hf_classifier = None
        self._hf_model = hf_model
    
    @property
    def hf_classifier(self) -> HuggingFaceComplexityClassifier:
        """Lazy load HuggingFace classifier."""
        if self._hf_classifier is None:
            self._hf_classifier = HuggingFaceComplexityClassifier(
                use_api=self.use_api,
                model_name=self._hf_model,
            )
        return self._hf_classifier
    
    def classify(self, prompt: str) -> 'HybridComplexityResult':
        """
        Classify prompt complexity with regex-first, HF fallback.
        
        Args:
            prompt: The user's prompt/query
            
        Returns:
            HybridComplexityResult with complexity and method info
        """
        start_time = time.perf_counter()
        
        # Try regex first
        regex_result = self.regex_classifier.classify(prompt)
        
        # Check if we should use HuggingFace fallback
        use_hf = regex_result.confidence < self.fallback_threshold
        
        if use_hf:
            try:
                hf_level, hf_confidence, hf_scores = self.hf_classifier.classify(prompt)
                
                # Use HF result if it's more confident
                if hf_confidence > regex_result.confidence:
                    elapsed = (time.perf_counter() - start_time) * 1000
                    return HybridComplexityResult(
                        complexity=hf_level.value,
                        level=hf_level,
                        confidence=hf_confidence,
                        signals=regex_result.signals,
                        reasoning=self._get_reasoning(hf_level),
                        suggested_action=self._get_action(hf_level),
                        needs_retrieval=self._needs_retrieval(hf_level),
                        needs_clarification=hf_level == ComplexityLevel.AMBIGUOUS_QUERY,
                        estimated_steps=self._get_steps(hf_level),
                        recommended_model_tier=self._get_tier(hf_level),
                        classification_method="huggingface",
                        regex_confidence=regex_result.confidence,
                        hf_confidence=hf_confidence,
                        latency_ms=elapsed,
                    )
            except Exception as e:
                logger.warning(f"HuggingFace fallback failed: {e}, using regex result")
        
        # Use regex result
        elapsed = (time.perf_counter() - start_time) * 1000
        return HybridComplexityResult(
            complexity=regex_result.complexity,
            level=regex_result.level,
            confidence=regex_result.confidence,
            signals=regex_result.signals,
            reasoning=regex_result.reasoning,
            suggested_action=regex_result.suggested_action,
            needs_retrieval=regex_result.needs_retrieval,
            needs_clarification=regex_result.needs_clarification,
            estimated_steps=regex_result.estimated_steps,
            recommended_model_tier=regex_result.recommended_model_tier,
            classification_method="regex",
            regex_confidence=regex_result.confidence,
            hf_confidence=None,
            latency_ms=elapsed,
        )
    
    def _get_reasoning(self, level: ComplexityLevel) -> str:
        """Get reasoning text for a complexity level."""
        reasonings = {
            ComplexityLevel.DIRECT_ANSWER: "Simple factual or conversational query.",
            ComplexityLevel.SIMPLE_TASK: "Single-step task or lookup.",
            ComplexityLevel.MULTI_STEP_REASONING: "Requires multiple steps or analysis.",
            ComplexityLevel.COMPLEX_TASK: "Complex reasoning or expert knowledge needed.",
            ComplexityLevel.AMBIGUOUS_QUERY: "Query is vague, needs clarification.",
        }
        return reasonings.get(level, "Unknown complexity level.")
    
    def _get_action(self, level: ComplexityLevel) -> str:
        """Get suggested action for a complexity level."""
        actions = {
            ComplexityLevel.DIRECT_ANSWER: "Answer directly. Use fast model.",
            ComplexityLevel.SIMPLE_TASK: "Execute single operation.",
            ComplexityLevel.MULTI_STEP_REASONING: "Break into steps, use capable model.",
            ComplexityLevel.COMPLEX_TASK: "Use chain-of-thought, premium model.",
            ComplexityLevel.AMBIGUOUS_QUERY: "Ask for clarification.",
        }
        return actions.get(level, "Process normally.")
    
    def _needs_retrieval(self, level: ComplexityLevel) -> bool:
        """Check if complexity level needs retrieval."""
        return level in [
            ComplexityLevel.SIMPLE_TASK,
            ComplexityLevel.MULTI_STEP_REASONING,
            ComplexityLevel.COMPLEX_TASK,
        ]
    
    def _get_steps(self, level: ComplexityLevel) -> int:
        """Get estimated steps for complexity level."""
        steps = {
            ComplexityLevel.DIRECT_ANSWER: 1,
            ComplexityLevel.SIMPLE_TASK: 1,
            ComplexityLevel.MULTI_STEP_REASONING: 3,
            ComplexityLevel.COMPLEX_TASK: 5,
            ComplexityLevel.AMBIGUOUS_QUERY: 0,
        }
        return steps.get(level, 1)
    
    def _get_tier(self, level: ComplexityLevel) -> str:
        """Get recommended model tier for complexity level."""
        tiers = {
            ComplexityLevel.DIRECT_ANSWER: "fast",
            ComplexityLevel.SIMPLE_TASK: "standard",
            ComplexityLevel.MULTI_STEP_REASONING: "standard",
            ComplexityLevel.COMPLEX_TASK: "premium",
            ComplexityLevel.AMBIGUOUS_QUERY: "fast",
        }
        return tiers.get(level, "standard")


@dataclass
class HybridComplexityResult(ComplexityResult):
    """Extended result with hybrid classification metadata."""
    classification_method: str = "regex"      # "regex" or "huggingface"
    regex_confidence: float = 0.0
    hf_confidence: Optional[float] = None
    latency_ms: float = 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_complexity(prompt: str) -> ComplexityResult:
    """Quick complexity classification using regex."""
    classifier = ComplexityClassifier()
    return classifier.classify(prompt)


def classify_complexity_hybrid(
    prompt: str,
    fallback_threshold: float = 0.70,
    use_api: bool = False,
) -> HybridComplexityResult:
    """
    Classify complexity with HuggingFace fallback.
    
    Args:
        prompt: The user's prompt/query
        fallback_threshold: Confidence below this triggers HF fallback
        use_api: Use HF Inference API instead of local model
        
    Returns:
        HybridComplexityResult with complexity and method info
    """
    classifier = HybridComplexityClassifier(
        fallback_threshold=fallback_threshold,
        use_api=use_api,
    )
    return classifier.classify(prompt)

