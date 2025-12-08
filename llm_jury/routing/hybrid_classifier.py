"""
Hybrid Prompt Classifier with HuggingFace Zero-Shot Fallback.

Uses a two-stage approach for optimal speed and accuracy:

1. **Stage 1: Regex-based classification** (~1ms)
   - Fast pattern matching for high-confidence cases
   - If confidence >= threshold, return immediately
   
2. **Stage 2: HuggingFace zero-shot fallback**
   - Option A: Local model (~100-500ms, requires model download)
   - Option B: Inference API (~100-200ms, no download, needs API key)

Deployment Options:
    
    LOCAL MODEL (default, recommended):
        - Downloads ~500MB model on first use
        - Stored in ~/.cache/huggingface/
        - Faster after first load (~300ms)
        - Works offline
        - Requires: pip install transformers torch
        - No API key needed!
    
    INFERENCE API (alternative):
        - No model download needed
        - ~1-2 second latency (network call)
        - Free tier included with HuggingFace account
        - Requires: pip install huggingface_hub (optional) or requests

Getting an API Token:
    1. Go to: https://huggingface.co/settings/tokens
    2. Click "Create new token"
    3. Select "Read" access (this automatically includes Inference API access)
    4. Copy the token (starts with "hf_...")

Token Resolution (for use_api=True):
    The classifier automatically loads your token from:
    1. api_token parameter (if provided directly)
    2. .env file in project root (requires: pip install python-dotenv)
    3. Environment variables (checked in order):
       - HUGGINGFACE_API_KEY
       - HF_API_TOKEN
       - HF_TOKEN

Usage:
    # Option 1: Local model (downloads on first use, NO API KEY needed)
    from llm_jury.routing import HybridClassifier
    classifier = HybridClassifier(use_api=False)  # default
    
    # Option 2: API mode (needs token in .env or environment)
    # Add to your .env file: HUGGINGFACE_API_KEY=hf_...
    classifier = HybridClassifier(use_api=True)
    
    result = classifier.classify("Write a poem about rust")
    print(result.use_case)  # "creative_writing"
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from llm_jury.routing.prompt_classifier import (
    PromptClassifier,
    ClassificationResult,
    UseCaseCategory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Mapping from HuggingFace labels to our use case names
# The HF model uses natural language labels; we map them to internal use cases
HF_LABEL_TO_USE_CASE: Dict[str, str] = {
    # Core task types
    "coding task": "code_generation",
    "code generation": "code_generation",
    "programming": "code_generation",
    "code review": "code_review",
    "debugging": "code_review",
    "code refactoring": "code_refactoring",
    
    # Reasoning
    "logical reasoning": "math_reasoning",
    "mathematical problem": "math_reasoning",
    "data analysis": "data_analysis",
    "analytical task": "data_analysis",
    "sql query": "sql_generation",
    
    # Content
    "creative writing": "creative_writing",
    "storytelling": "creative_writing",
    "summarization": "summarization",
    "translation": "translation",
    "paraphrasing": "paraphrasing",
    "style transfer": "style_transfer",
    "grammar correction": "grammar_correction",
    "proofreading": "grammar_correction",
    
    # RAG / Document
    "document question answering": "rag_pipeline",
    "rag_query": "rag_pipeline",
    "retrieval augmented generation": "rag_pipeline",
    "knowledge base query": "rag_pipeline",
    
    # Conversation
    "chitchat": "general_qa",
    "general_chat": "general_qa",
    "casual conversation": "general_qa",
    "question answering": "general_qa",
    "factual question": "general_qa",
    
    # Technical
    "function calling": "function_calling",
    "api call": "function_calling",
    "tool use": "tool_use",
    "agent task": "agent_workflow",
    "multi-step task": "agent_workflow",
    "planning": "planning",
    "task planning": "planning",
    "structured extraction": "structured_extraction",
    "json extraction": "structured_extraction",
    
    # Embeddings & Similarity
    "embedding generation": "embeddings",
    "text embedding": "embeddings",
    "semantic search": "embeddings",
    "text similarity": "semantic_similarity",
    "semantic similarity": "semantic_similarity",
    
    # Classification & Analysis
    "text classification": "text_classification",
    "categorization": "text_classification",
    "sentiment analysis": "sentiment_analysis",
    "opinion mining": "sentiment_analysis",
    "entity extraction": "entity_extraction",
    "named entity recognition": "entity_extraction",
    "content moderation": "content_moderation",
    "toxicity detection": "content_moderation",
    
    # Vision / Multimodal
    "image understanding": "image_understanding",
    "image analysis": "image_understanding",
    "visual question answering": "vision_qa",
    "image caption": "image_understanding",
    
    # Specialized
    "legal analysis": "legal_review",
    "financial analysis": "financial_analysis",
    "research": "research_assistant",
    "academic": "research_assistant",
    
    # Support & Creative
    "customer support": "customer_support",
    "tutoring": "tutoring",
    "explanation": "tutoring",
    "brainstorming": "brainstorming",
    "idea generation": "brainstorming",
    "roleplay": "roleplay",
    "character acting": "roleplay",
}

# Default labels for zero-shot classification
# These are natural language descriptions that work well with MNLI models
# Keep this list focused (15-20 labels) for best classification accuracy
DEFAULT_ZS_LABELS = [
    # Development
    "coding task",
    "code review",
    
    # Reasoning & Analysis
    "logical reasoning",
    "data analysis",
    
    # Content
    "creative writing",
    "summarization",
    "translation",
    "paraphrasing",
    
    # RAG & Documents
    "document question answering",
    
    # Technical
    "function calling",
    "agent task",
    "embedding generation",
    
    # Classification
    "text classification",
    "sentiment analysis",
    "content moderation",
    
    # Vision
    "image understanding",
    
    # Specialized
    "legal analysis",
    "financial analysis",
    
    # Conversational
    "general_chat",
    "customer support",
    "brainstorming",
    "roleplay",
]


# =============================================================================
# Hybrid Classifier Result
# =============================================================================

@dataclass
class HybridClassificationResult:
    """
    Result from hybrid classification.
    
    Extends ClassificationResult with additional metadata about
    which classification method was used and timing information.
    """
    use_case: str
    confidence: float
    category: UseCaseCategory
    signals: List[str]
    alternative_use_cases: List[Tuple[str, float]]
    
    # Hybrid-specific fields
    classification_method: str  # "regex" or "huggingface"
    regex_confidence: float     # Original regex confidence
    hf_confidence: Optional[float] = None  # HF confidence if used
    latency_ms: float = 0.0     # Classification time in milliseconds
    
    def to_classification_result(self) -> ClassificationResult:
        """Convert to standard ClassificationResult."""
        return ClassificationResult(
            use_case=self.use_case,
            confidence=self.confidence,
            category=self.category,
            signals=self.signals,
            alternative_use_cases=self.alternative_use_cases,
        )


# =============================================================================
# HuggingFace Zero-Shot Classifier
# =============================================================================

class HuggingFaceClassifier:
    """
    Zero-shot classifier using HuggingFace transformers.
    
    Uses distilbart-mnli for fast, accurate zero-shot classification.
    The model is loaded lazily on first use and cached for subsequent calls.
    
    Performance:
        - Model loading: ~2-5s (first call only)
        - Inference (CPU): ~100-500ms per prompt
        - Inference (GPU): ~20-50ms per prompt
    """
    
    # Default model - distilled for speed, good accuracy
    DEFAULT_MODEL = "valhalla/distilbart-mnli-12-1"
    
    # Alternative models:
    # - "facebook/bart-large-mnli" - More accurate, slower
    # - "MoritzLawormer/bart-large-mnli" - Larger, most accurate
    # - "typeform/distilbert-base-uncased-mnli" - Fastest, less accurate
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace classifier.
        
        Args:
            model_name: HuggingFace model ID for zero-shot classification
            labels: Custom labels for classification. If None, uses DEFAULT_ZS_LABELS
            device: Device to run on ("cpu", "cuda", "mps"). None = auto-detect.
        """
        self.model_name = model_name
        self.labels = labels or DEFAULT_ZS_LABELS
        self.device = device
        self._pipeline = None  # Lazy loading
        self._load_time_ms = 0.0
    
    @property
    def pipeline(self):
        """Lazy load the classification pipeline."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self):
        """Load the HuggingFace pipeline."""
        try:
            from transformers import pipeline
            
            logger.info(f"Loading HuggingFace zero-shot model: {self.model_name}")
            start = time.perf_counter()
            
            # Build kwargs
            kwargs = {"model": self.model_name}
            if self.device:
                kwargs["device"] = self.device
            
            self._pipeline = pipeline("zero-shot-classification", **kwargs)
            
            self._load_time_ms = (time.perf_counter() - start) * 1000
            logger.info(f"Model loaded in {self._load_time_ms:.0f}ms")
            
        except ImportError:
            logger.error(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    def classify(
        self,
        prompt: str,
        labels: Optional[List[str]] = None,
        multi_label: bool = False,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a prompt using zero-shot classification.
        
        Args:
            prompt: Text to classify
            labels: Override labels for this classification
            multi_label: Whether to allow multiple labels
            
        Returns:
            Tuple of (best_label, confidence, all_scores)
        """
        use_labels = labels or self.labels
        
        result = self.pipeline(
            prompt,
            use_labels,
            multi_label=multi_label,
        )
        
        # Build scores dict
        scores = {
            label: score 
            for label, score in zip(result["labels"], result["scores"])
        }
        
        best_label = result["labels"][0]
        best_score = result["scores"][0]
        
        return best_label, best_score, scores
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipeline is not None
    
    def get_load_time_ms(self) -> float:
        """Get model load time in milliseconds."""
        return self._load_time_ms


# =============================================================================
# HuggingFace Inference API Classifier (No Local Model)
# =============================================================================

class HuggingFaceAPIClassifier:
    """
    Zero-shot classifier using HuggingFace Inference API.
    
    This option requires NO local model download - all inference happens
    on HuggingFace's servers. Great for:
    - Quick setup (no 500MB download)
    - Serverless deployments
    - Resource-constrained environments
    
    Getting a Token:
        1. Go to: https://huggingface.co/settings/tokens
        2. Create a "Read" access token (automatically includes Inference API)
        3. Add to .env: HUGGINGFACE_API_KEY=hf_...
           Or set env var: HF_API_TOKEN=hf_...
    
    Rate Limits (Free Tier):
        - Included with free HuggingFace account
        - For higher limits, use HF Pro or dedicated endpoints
    
    Performance:
        - Latency: ~1-2 seconds (network + model inference)
        - No cold start on your machine (model runs on HF servers)
    """
    
    # API endpoint for zero-shot classification (updated Nov 2024)
    API_URL = "https://router.huggingface.co/hf-inference/models/"
    DEFAULT_MODEL = "facebook/bart-large-mnli"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        labels: Optional[List[str]] = None,
        api_token: Optional[str] = None,
    ):
        """
        Initialize the API classifier.
        
        Args:
            model_name: HuggingFace model ID
            labels: Custom labels for classification
            api_token: HuggingFace API token. If None, reads from environment.
            
        Token Resolution Order:
            1. api_token parameter (if provided)
            2. .env file (if python-dotenv installed)
            3. Environment variables: HUGGINGFACE_API_KEY, HF_API_TOKEN, HF_TOKEN
        """
        self.model_name = model_name
        self.labels = labels or DEFAULT_ZS_LABELS
        self.api_token = api_token or self._load_api_token()
        
        if not self.api_token:
            logger.warning(
                "No HuggingFace API token found.\n"
                "  To get a token:\n"
                "    1. Go to: https://huggingface.co/settings/tokens\n"
                "    2. Create a 'Read' access token (includes Inference API)\n"
                "    3. Add to .env file: HUGGINGFACE_API_KEY=hf_...\n"
                "  Or set environment variable: HF_API_TOKEN=hf_..."
            )
    
    @staticmethod
    def _load_api_token() -> Optional[str]:
        """
        Load API token from .env file or environment variables.
        
        Resolution order:
            1. .env file (using python-dotenv if available)
            2. HUGGINGFACE_API_KEY environment variable
            3. HF_API_TOKEN environment variable
            4. HF_TOKEN environment variable
        """
        import os
        
        # Try loading from .env file first
        try:
            from dotenv import load_dotenv
            load_dotenv()  # Loads .env file into os.environ
            logger.debug("Loaded .env file")
        except ImportError:
            # python-dotenv not installed, continue with os.environ
            pass
        
        # Check multiple possible env var names
        env_var_names = [
            "HUGGINGFACE_API_KEY",  # User's preferred name
            "HF_API_TOKEN",          # Common alternative
            "HF_TOKEN",              # HuggingFace CLI default
            "HUGGINGFACE_TOKEN",     # Another common name
        ]
        
        for var_name in env_var_names:
            token = os.environ.get(var_name)
            if token:
                logger.debug(f"Found API token in {var_name}")
                return token
        
        return None
    
    def classify(
        self,
        prompt: str,
        labels: Optional[List[str]] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a prompt using the Inference API.
        
        Uses huggingface_hub.InferenceClient for reliable API access.
        Falls back to direct requests if huggingface_hub not installed.
        
        Args:
            prompt: Text to classify
            labels: Override labels for this classification
            
        Returns:
            Tuple of (best_label, confidence, all_scores)
        """
        use_labels = labels or self.labels
        
        # Try using official huggingface_hub library first (recommended)
        try:
            return self._classify_with_hub(prompt, use_labels)
        except ImportError:
            logger.debug("huggingface_hub not installed, using requests fallback")
        except Exception as e:
            logger.warning(f"huggingface_hub failed: {e}, trying requests fallback")
        
        # Fallback to direct requests
        return self._classify_with_requests(prompt, use_labels)
    
    def _classify_with_hub(
        self,
        prompt: str,
        labels: List[str],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify using huggingface_hub InferenceClient."""
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(
            model=self.model_name,
            token=self.api_token,
        )
        
        results = client.zero_shot_classification(
            text=prompt,
            candidate_labels=labels,
        )
        
        # Handle response - can be list of objects or dataclass objects
        scores = {}
        best_label = None
        best_score = 0.0
        
        for r in results:
            # Handle both dict-like and dataclass responses
            if hasattr(r, 'label'):
                label, score = r.label, r.score
            elif isinstance(r, dict):
                label, score = r['label'], r['score']
            else:
                continue
                
            scores[label] = score
            if score > best_score:
                best_score = score
                best_label = label
        
        if best_label is None:
            return "general_chat", 0.5, {}
            
        return best_label, best_score, scores
    
    def _classify_with_requests(
        self,
        prompt: str,
        labels: List[str],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify using direct HTTP requests (fallback)."""
        import requests
        
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "candidate_labels": labels,
            }
        }
        
        url = f"{self.API_URL}{self.model_name}"
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Handle API response format (new router format: list of {label, score})
            if isinstance(result, list) and len(result) > 0:
                # New format: [{"label": "creative writing", "score": 0.98}, ...]
                scores = {item["label"]: item["score"] for item in result}
                # Results are already sorted by score descending
                best_label = result[0]["label"]
                best_score = result[0]["score"]
                return best_label, best_score, scores
            # Legacy format: {"labels": [...], "scores": [...]}
            elif isinstance(result, dict) and "labels" in result:
                scores = {
                    label: score 
                    for label, score in zip(result["labels"], result["scores"])
                }
                best_label = result["labels"][0]
                best_score = result["scores"][0]
                return best_label, best_score, scores
            else:
                logger.error(f"Unexpected API response format: {result}")
                return "general_chat", 0.5, {}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API request failed: {e}")
            # Return fallback
            return "general_chat", 0.5, {}
    
    def is_loaded(self) -> bool:
        """API is always 'loaded' - no local model."""
        return True
    
    def get_load_time_ms(self) -> float:
        """No load time for API."""
        return 0.0


# =============================================================================
# Hybrid Classifier
# =============================================================================

class HybridClassifier:
    """
    Hybrid classifier combining regex and HuggingFace zero-shot.
    
    Strategy:
    1. Run regex-based classification (fast, ~1ms)
    2. If confidence >= fallback_threshold, return regex result
    3. If confidence < fallback_threshold, run HuggingFace zero-shot
    4. Return HuggingFace result (more accurate for ambiguous cases)
    
    Two deployment modes:
    
    LOCAL MODEL (use_api=False, default):
        - Downloads ~500MB model on first use
        - Faster after first load (~300ms per inference)
        - Works offline
        - Requires: pip install transformers torch
    
    INFERENCE API (use_api=True):
        - No model download needed!
        - ~100-200ms per inference (network call)
        - Requires: HF_API_TOKEN environment variable
        - Free tier: 1000 requests/day
    
    Example:
        # Local model (downloads on first use)
        >>> classifier = HybridClassifier(use_api=False)
        
        # API mode (no download)
        >>> import os
        >>> os.environ["HF_API_TOKEN"] = "hf_..."
        >>> classifier = HybridClassifier(use_api=True)
        
        >>> result = classifier.classify("Write a poem about rust")
        >>> print(result.use_case)
        "creative_writing"
    """
    
    def __init__(
        self,
        fallback_threshold: float = 0.75,
        use_api: bool = False,
        hf_model: str = HuggingFaceClassifier.DEFAULT_MODEL,
        hf_labels: Optional[List[str]] = None,
        hf_device: Optional[str] = None,
        hf_api_token: Optional[str] = None,
        lazy_load_hf: bool = True,
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            fallback_threshold: Minimum regex confidence to skip HuggingFace.
                Higher = more HF calls (slower but more accurate)
                Lower = fewer HF calls (faster but less accurate for edge cases)
            use_api: If True, use HuggingFace Inference API (no local model download).
                If False (default), download and run model locally.
            hf_model: HuggingFace model for zero-shot classification
            hf_labels: Custom labels for HuggingFace classification
            hf_device: Device for local HuggingFace model (ignored if use_api=True)
            hf_api_token: API token for HuggingFace (only used if use_api=True).
                If None, reads from HF_API_TOKEN environment variable.
            lazy_load_hf: If True, only load HF model when needed (only for local mode)
        """
        self.fallback_threshold = fallback_threshold
        self.use_api = use_api
        self.regex_classifier = PromptClassifier()
        
        # HuggingFace classifier (lazy or eager loading)
        self._hf_classifier = None
        self._hf_config = {
            "model_name": hf_model,
            "labels": hf_labels,
        }
        
        if use_api:
            # API mode - no model download
            self._hf_config["api_token"] = hf_api_token
        else:
            # Local mode - model download
            self._hf_config["device"] = hf_device
        
        if not lazy_load_hf and not use_api:
            self._load_hf_classifier()
        
        # Stats
        self._stats = {
            "total_calls": 0,
            "regex_only": 0,
            "hf_fallback": 0,
            "total_regex_ms": 0.0,
            "total_hf_ms": 0.0,
        }
    
    def _load_hf_classifier(self):
        """Load HuggingFace classifier (local or API)."""
        if self.use_api:
            self._hf_classifier = HuggingFaceAPIClassifier(
                model_name=self._hf_config["model_name"],
                labels=self._hf_config["labels"],
                api_token=self._hf_config.get("api_token"),
            )
        else:
            self._hf_classifier = HuggingFaceClassifier(
                model_name=self._hf_config["model_name"],
                labels=self._hf_config["labels"],
                device=self._hf_config.get("device"),
            )
    
    @property
    def hf_classifier(self):
        """Get HuggingFace classifier, loading if needed."""
        if self._hf_classifier is None:
            self._load_hf_classifier()
        return self._hf_classifier
    
    def classify(
        self,
        prompt: str,
        force_hf: bool = False,
        force_regex: bool = False,
    ) -> HybridClassificationResult:
        """
        Classify a prompt using hybrid approach.
        
        Args:
            prompt: Text to classify
            force_hf: Always use HuggingFace (skip regex)
            force_regex: Always use regex (skip HuggingFace fallback)
            
        Returns:
            HybridClassificationResult with use case and metadata
        """
        start_time = time.perf_counter()
        self._stats["total_calls"] += 1
        
        # Stage 1: Regex classification
        regex_start = time.perf_counter()
        regex_result = self.regex_classifier.classify(prompt)
        regex_ms = (time.perf_counter() - regex_start) * 1000
        self._stats["total_regex_ms"] += regex_ms
        
        # Check if we should use regex result
        use_regex = (
            force_regex or
            (not force_hf and regex_result.confidence >= self.fallback_threshold)
        )
        
        if use_regex:
            self._stats["regex_only"] += 1
            total_ms = (time.perf_counter() - start_time) * 1000
            
            return HybridClassificationResult(
                use_case=regex_result.use_case,
                confidence=regex_result.confidence,
                category=regex_result.category,
                signals=regex_result.signals + ["method:regex"],
                alternative_use_cases=regex_result.alternative_use_cases,
                classification_method="regex",
                regex_confidence=regex_result.confidence,
                hf_confidence=None,
                latency_ms=total_ms,
            )
        
        # Stage 2: HuggingFace fallback
        self._stats["hf_fallback"] += 1
        
        hf_start = time.perf_counter()
        hf_label, hf_score, all_scores = self.hf_classifier.classify(prompt)
        hf_ms = (time.perf_counter() - hf_start) * 1000
        self._stats["total_hf_ms"] += hf_ms
        
        # Map HF label to use case
        use_case = HF_LABEL_TO_USE_CASE.get(hf_label, "general_qa")
        
        # Get category from regex patterns
        category = self.regex_classifier.patterns.get(
            use_case, {}
        ).get("category", UseCaseCategory.CONVERSATIONAL)
        
        # Build alternatives from HF scores
        alternatives = []
        for label, score in sorted(all_scores.items(), key=lambda x: -x[1])[1:4]:
            mapped_uc = HF_LABEL_TO_USE_CASE.get(label, "general_qa")
            if mapped_uc != use_case:
                alternatives.append((mapped_uc, score))
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return HybridClassificationResult(
            use_case=use_case,
            confidence=hf_score,
            category=category,
            signals=[f"hf_label:{hf_label}", "method:huggingface"],
            alternative_use_cases=alternatives,
            classification_method="huggingface",
            regex_confidence=regex_result.confidence,
            hf_confidence=hf_score,
            latency_ms=total_ms,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics.
        
        Returns:
            Dict with call counts and timing statistics
        """
        total = self._stats["total_calls"]
        if total == 0:
            return self._stats
        
        return {
            **self._stats,
            "regex_only_pct": 100 * self._stats["regex_only"] / total,
            "hf_fallback_pct": 100 * self._stats["hf_fallback"] / total,
            "avg_regex_ms": self._stats["total_regex_ms"] / total,
            "avg_hf_ms": (
                self._stats["total_hf_ms"] / self._stats["hf_fallback"]
                if self._stats["hf_fallback"] > 0 else 0
            ),
            "avg_total_ms": (
                self._stats["total_regex_ms"] + self._stats["total_hf_ms"]
            ) / total,
        }
    
    def reset_stats(self):
        """Reset classification statistics."""
        self._stats = {
            "total_calls": 0,
            "regex_only": 0,
            "hf_fallback": 0,
            "total_regex_ms": 0.0,
            "total_hf_ms": 0.0,
        }
    
    def preload_hf_model(self):
        """
        Preload the HuggingFace model.
        
        Call this during application startup to avoid latency on first classify().
        """
        _ = self.hf_classifier  # Triggers lazy loading
    
    def is_hf_loaded(self) -> bool:
        """Check if HuggingFace model is loaded."""
        return self._hf_classifier is not None and self._hf_classifier.is_loaded()


# =============================================================================
# Convenience Functions
# =============================================================================

# Global singleton for convenience functions
_default_classifier: Optional[HybridClassifier] = None


def get_hybrid_classifier(
    fallback_threshold: float = 0.75,
    **kwargs
) -> HybridClassifier:
    """
    Get or create the default hybrid classifier.
    
    Args:
        fallback_threshold: Threshold for HuggingFace fallback
        **kwargs: Additional args for HybridClassifier
        
    Returns:
        HybridClassifier singleton
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = HybridClassifier(
            fallback_threshold=fallback_threshold,
            **kwargs
        )
    return _default_classifier


def classify_prompt_hybrid(
    prompt: str,
    fallback_threshold: float = 0.75,
) -> HybridClassificationResult:
    """
    Classify a prompt using hybrid regex + HuggingFace approach.
    
    Args:
        prompt: Text to classify
        fallback_threshold: Confidence threshold for HF fallback
        
    Returns:
        HybridClassificationResult
    """
    classifier = get_hybrid_classifier(fallback_threshold)
    return classifier.classify(prompt)


def benchmark_classifier(
    prompts: List[str],
    classifier: Optional[HybridClassifier] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark classifier performance on a set of prompts.
    
    Args:
        prompts: List of prompts to classify
        classifier: Classifier to benchmark (default: create new)
        warmup: Whether to do a warmup run first
        
    Returns:
        Dict with timing and accuracy statistics
    """
    if classifier is None:
        classifier = HybridClassifier()
    
    # Warmup (load model)
    if warmup:
        classifier.preload_hf_model()
        classifier.classify(prompts[0])
        classifier.reset_stats()
    
    # Benchmark
    results = []
    start = time.perf_counter()
    
    for prompt in prompts:
        result = classifier.classify(prompt)
        results.append({
            "prompt": prompt[:50],
            "use_case": result.use_case,
            "confidence": result.confidence,
            "method": result.classification_method,
            "latency_ms": result.latency_ms,
        })
    
    total_time = (time.perf_counter() - start) * 1000
    
    stats = classifier.get_stats()
    
    return {
        "total_prompts": len(prompts),
        "total_time_ms": total_time,
        "avg_latency_ms": total_time / len(prompts),
        "stats": stats,
        "results": results,
    }

