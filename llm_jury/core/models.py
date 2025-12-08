"""
Core data models and enumerations for LLM Jury.

This module consolidates all dataclasses and enums that were previously
duplicated across multiple files (llm_router.py, llm_recommendation_orchestrator.py,
rank_models.py, LLM_Capability.py, prompt_router.py).
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


# ==========================================
# ENUMERATIONS
# ==========================================

class ProductArchetype(Enum):
    """Product archetypes for model classification based on use case."""
    BULK_OPS = "Bulk Ops (High Throughput)"
    RAG_SPECIALIST = "RAG Specialist (Context Aware)"
    REASONING_SPECIALIST = "Reasoning Specialist"
    FRONTIER = "Frontier (Complex Reasoning)"


class PromptCategory(Enum):
    """Domain categories for prompt classification."""
    CODING = "Programming"
    DATA_SCIENCE = "Data/Math"
    CREATIVE = "Creative/Roleplay"
    GENERAL = "General Assistant"
    LEGAL = "Legal & Compliance"
    FINANCE = "Finance"
    HEALTH = "Health & Medical"
    QA = "Q&A"              # Question answering - trust/accuracy focused
    RAG = "RAG"             # Retrieval-augmented - context window + cost focused
    CHATBOT = "Chatbot"     # Conversational - latency + cost focused


class ModelTier(Enum):
    """Simple tier-based model classification for routing."""
    SMALL = "Small_LLM"                     # e.g., Llama 3 8B, Haiku
    COMPOSITE = "Small_LLM_w_Search_CoT"    # e.g., Llama + Tavily + ReAct
    LARGE = "Large_LLM"                     # e.g., GPT-4, Claude 3.5 Sonnet


# ==========================================
# DATA MODELS
# ==========================================

@dataclass
class ModelMetadata:
    """
    Comprehensive model metadata combining all fields from different implementations.
    
    This consolidates ModelMetadata from:
    - llm_router.py (most complete with HF signals)
    - llm_recommendation_orchestrator.py (simpler version)
    - rank_models.py (with popularity metrics)
    """
    name: str
    
    # --- Capability Scores (0-100) ---
    mmlu_score: Optional[float] = None          # Knowledge
    gpqa_score: Optional[float] = None          # Deep Reasoning
    math_score: Optional[float] = None          # Logic/Math
    ifeval_score: Optional[float] = None        # Instruction Following
    tool_use_ability: Optional[float] = None    # Agentic capability (0-1)
    context_window_k: Optional[int] = None      # Context window in thousands of tokens
    
    # --- Trust & Safety Metrics ---
    hallucination_rate: Optional[float] = None  # Estimated % of hallucination (Lower is better)
    ethics_score: Optional[float] = None        # SafetyBench/Alignment score (Higher is better)
    
    # --- Hugging Face Popularity Signals ---
    hf_downloads: Optional[int] = None          # "Production Proof"
    hf_likes: Optional[int] = None              # "Community Quality Sentiment"
    hf_created_at: Optional[str] = None         # YYYY-MM-DD (For age/velocity calc)
    
    # --- Production Metrics ---
    archetype: Optional[ProductArchetype] = None
    input_cost_per_m: Optional[float] = None    # Cost per million tokens (input)
    output_cost_per_m: Optional[float] = None   # Cost per million tokens (output)
    pricing_source: Optional[str] = None        # Source of pricing: 'artificial_analysis', 'openrouter', etc.
    median_latency_ms: Optional[float] = None   # Time to First Token (TTFT)
    param_count_b: Optional[float] = None       # Parameters in billions
    
    # --- Additional Popularity/Trust Metrics (from rank_models.py) ---
    is_top_10_used: Optional[bool] = None
    date_created: Optional[str] = None          # YYYY-MM-DD (alternative to hf_created_at)
    avg_uptime_90d: Optional[float] = None      # Percentage (e.g., 99.9)
    num_apps_using: Optional[int] = None
    num_notable_apps: Optional[int] = None
    daily_requests: Optional[int] = None
    
    # --- Artificial Analysis Benchmark Indices ---
    intelligence_index: Optional[float] = None  # AA composite intelligence score
    coding_index: Optional[float] = None        # AA composite coding score
    math_index: Optional[float] = None          # AA composite math score
    
    # --- Raw Benchmarks (from AA API) ---
    mmlu_pro: Optional[float] = None            # MMLU-Pro score
    gpqa: Optional[float] = None                # GPQA score
    hle: Optional[float] = None                 # HLE score
    livecodebench: Optional[float] = None       # LiveCodeBench score
    scicode: Optional[float] = None             # SciCode score
    math_500: Optional[float] = None            # MATH-500 score
    aime: Optional[float] = None                # AIME score
    
    # --- Performance Metrics ---
    output_tokens_per_second: Optional[float] = None  # Throughput (tokens/sec)
    measured_ttft_seconds: Optional[float] = None     # Measured Time To First Token (seconds)
    time_to_first_token_seconds: Optional[float] = None  # AA's TTFT measurement
    
    # --- Reliability Metrics (from Vectara) ---
    refusal_rate: Optional[float] = None        # % of prompts model refuses (lower is better)
    factual_consistency_rate: Optional[float] = None  # 100 - hallucination_rate

    def get_trust_score(self) -> Optional[float]:
        """
        Calculates a 0-100 Trust Score based on Hugging Face signals.
        
        Returns:
            Optional[float]: Trust score from 0-100 if data is available, None otherwise.
        """
        # Return None if we don't have the required data
        if self.hf_downloads is None or self.hf_likes is None:
            return None
            
        # Validation score based on downloads (log scale)
        log_downloads = np.log10(max(self.hf_downloads, 1))
        validation_score = min(log_downloads / 7.0, 1.0) * 100

        # Sentiment score based on likes velocity
        date_fmt = "%Y-%m-%d"
        created_date = self.hf_created_at or self.date_created
        if created_date:
            try:
                days_old = (datetime.now() - datetime.strptime(created_date, date_fmt)).days
                days_old = max(days_old, 1)
                velocity = self.hf_likes / days_old
                sentiment_score = min(velocity / 5.0, 1.0) * 100
            except ValueError:
                sentiment_score = 0.0
        else:
            sentiment_score = 0.0

        return (validation_score * 0.7) + (sentiment_score * 0.3)


@dataclass
class ModelSpecs:
    """
    Simplified model specifications for clustering analysis.
    Used by LLM_Capability.py for model classification.
    """
    name: str
    param_count_b: float
    mmlu_score: float           # Knowledge
    gpqa_score: float           # Reasoning
    math_score: float           # Hard Math/Logic
    ifeval_score: float         # Instruction Following
    context_window_k: int
    tool_use_ability: float     # Estimated 0-1


@dataclass
class RoutingDecision:
    """Decision output from routing logic."""
    archetype: ProductArchetype
    category: PromptCategory
    reason: str
    recommend_cot: bool = False
    cot_template: Optional[str] = None


@dataclass
class RecommendationResult:
    """Result from ranking/recommendation system."""
    rank: int
    model_name: str
    score: float
    reasoning: str
    cot_template: str = ""
    optimization_metrics: Optional[Dict[str, Any]] = None  # Constraint satisfaction info


@dataclass
class RankedModel:
    """
    Detailed ranking result with quality and Chebyshev scores.
    Used by rank_models.py for more detailed analysis.
    """
    name: str
    quality_score: float
    chebyshev_score: float      # Lower is better (Distance to Utopia)
    tradeoff_summary: str
    metadata: ModelMetadata
