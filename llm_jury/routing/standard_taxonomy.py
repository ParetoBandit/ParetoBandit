"""
Standard Intent Taxonomy for Generalizable LLM Routing.

This module defines a **benchmark-aligned taxonomy** of 8 broad categories
that avoid the overfitting trap of custom, fine-grained taxonomies.

Design Principles:
    1. Categories are grounded in established NLP benchmarks (GSM8K, HumanEval, MMLU, etc.)
    2. Broad categories that generalize to unseen query types
    3. Explicit "uncertain" bucket for ambiguous/low-confidence cases
    4. Mappings from fine-grained use cases to standard categories

The Trap We Avoid:
    Using a taxonomy too specific to training data (e.g., "Python Scripts" vs. "Java Scripts")
    makes the method look non-generalizable and fails on novel query types.

The Remedy:
    Use standard, broad taxonomy aligned with academic benchmarks.
    Explicitly handle the "Unknown" intent with conservative fallback routing.

References:
    - GSM8K: Grade school math reasoning
    - HumanEval/MBPP: Code generation benchmarks
    - MMLU: Massive Multitask Language Understanding
    - TruthfulQA: Factual accuracy benchmark
    - CNN/DailyMail, XSum: Summarization benchmarks
    - WMT, FLORES: Translation benchmarks
    - MT-Bench: Multi-turn conversation benchmark
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


class StandardCategory(Enum):
    """
    Standard taxonomy of 8 broad categories aligned with NLP benchmarks.
    
    Each category maps to well-known evaluation benchmarks, ensuring
    the taxonomy generalizes beyond any specific training distribution.
    """
    
    # Mathematical and logical reasoning
    # Benchmarks: GSM8K, MATH, ARC, HellaSwag, WinoGrande
    REASONING = "reasoning"
    
    # Programming and code-related tasks
    # Benchmarks: HumanEval, MBPP, SWE-bench, CodeContests
    CODING = "coding"
    
    # Creative and generative writing
    # Benchmarks: Creative writing evaluations, story generation
    CREATIVE = "creative"
    
    # Factual question answering and knowledge retrieval
    # Benchmarks: MMLU, TruthfulQA, NaturalQuestions, TriviaQA
    FACTUAL_QA = "factual_qa"
    
    # Condensing and summarizing content
    # Benchmarks: CNN/DailyMail, XSum, SAMSum
    SUMMARIZATION = "summarization"
    
    # Extracting structured information from text
    # Benchmarks: NER benchmarks (CoNLL), RE benchmarks, IE tasks
    EXTRACTION = "extraction"
    
    # Language translation between languages
    # Benchmarks: WMT, FLORES, NTREX
    TRANSLATION = "translation"
    
    # Multi-turn conversation and chat
    # Benchmarks: MT-Bench, Chatbot Arena, AlpacaEval
    CONVERSATION = "conversation"
    
    # Explicit uncertainty bucket - NOT a failure mode!
    # This is a valid routing decision for ambiguous queries
    UNCERTAIN = "uncertain"


@dataclass
class CategoryMetadata:
    """Metadata for a standard category."""
    category: StandardCategory
    description: str
    benchmark_grounding: List[str]
    example_queries: List[str]
    # Model selection hints
    prefer_capabilities: List[str]
    routing_strategy: str  # "specialist" or "generalist"


# Category metadata with benchmark grounding
CATEGORY_METADATA: Dict[StandardCategory, CategoryMetadata] = {
    StandardCategory.REASONING: CategoryMetadata(
        category=StandardCategory.REASONING,
        description="Mathematical reasoning, logical deduction, analytical problems",
        benchmark_grounding=["GSM8K", "MATH", "ARC", "HellaSwag", "WinoGrande"],
        example_queries=[
            "Solve this equation: 2x + 5 = 13",
            "If all A are B, and some B are C, what can we conclude?",
            "Calculate the compound interest on $1000 at 5% for 3 years",
        ],
        prefer_capabilities=["reasoning", "chain_of_thought"],
        routing_strategy="specialist",
    ),
    
    StandardCategory.CODING: CategoryMetadata(
        category=StandardCategory.CODING,
        description="Code generation, review, debugging, and programming tasks",
        benchmark_grounding=["HumanEval", "MBPP", "SWE-bench", "CodeContests", "DS-1000"],
        example_queries=[
            "Write a Python function to merge two sorted lists",
            "Debug this JavaScript code that's throwing an error",
            "Optimize this SQL query for better performance",
        ],
        prefer_capabilities=["coding", "json_mode"],
        routing_strategy="specialist",
    ),
    
    StandardCategory.CREATIVE: CategoryMetadata(
        category=StandardCategory.CREATIVE,
        description="Creative writing, storytelling, marketing copy, artistic content",
        benchmark_grounding=["Creative writing evaluations", "Story generation"],
        example_queries=[
            "Write a short story about a robot learning to paint",
            "Create a marketing tagline for an eco-friendly water bottle",
            "Compose a haiku about autumn leaves",
        ],
        prefer_capabilities=["creative", "long_form"],
        routing_strategy="specialist",
    ),
    
    StandardCategory.FACTUAL_QA: CategoryMetadata(
        category=StandardCategory.FACTUAL_QA,
        description="Factual questions, knowledge retrieval, explanations",
        benchmark_grounding=["MMLU", "TruthfulQA", "NaturalQuestions", "TriviaQA"],
        example_queries=[
            "What is the capital of France?",
            "Explain how photosynthesis works",
            "Who invented the telephone?",
        ],
        prefer_capabilities=["low_hallucination", "factual"],
        routing_strategy="generalist",
    ),
    
    StandardCategory.SUMMARIZATION: CategoryMetadata(
        category=StandardCategory.SUMMARIZATION,
        description="Condensing documents, articles, or conversations",
        benchmark_grounding=["CNN/DailyMail", "XSum", "SAMSum", "MultiNews"],
        example_queries=[
            "Summarize this article in 3 sentences",
            "Give me the key points from this meeting transcript",
            "TL;DR of this research paper",
        ],
        prefer_capabilities=["long_context", "compression"],
        routing_strategy="generalist",
    ),
    
    StandardCategory.EXTRACTION: CategoryMetadata(
        category=StandardCategory.EXTRACTION,
        description="Extracting structured information: entities, relations, JSON",
        benchmark_grounding=["CoNLL NER", "ACE RE", "SciERC", "DocRED"],
        example_queries=[
            "Extract all person names from this text",
            "Parse this receipt into JSON format",
            "Identify the sentiment of each sentence",
        ],
        prefer_capabilities=["json_mode", "structured_output"],
        routing_strategy="specialist",
    ),
    
    StandardCategory.TRANSLATION: CategoryMetadata(
        category=StandardCategory.TRANSLATION,
        description="Translation between human languages",
        benchmark_grounding=["WMT", "FLORES", "NTREX"],
        example_queries=[
            "Translate this to Spanish",
            "Convert this French paragraph to English",
            "How do you say 'hello' in Japanese?",
        ],
        prefer_capabilities=["multilingual"],
        routing_strategy="specialist",
    ),
    
    StandardCategory.CONVERSATION: CategoryMetadata(
        category=StandardCategory.CONVERSATION,
        description="Multi-turn chat, customer support, general dialogue",
        benchmark_grounding=["MT-Bench", "Chatbot Arena", "AlpacaEval"],
        example_queries=[
            "Can you help me plan a trip to Italy?",
            "I'm having trouble with my account",
            "Let's brainstorm ideas for my presentation",
        ],
        prefer_capabilities=["conversational", "helpful"],
        routing_strategy="generalist",
    ),
    
    StandardCategory.UNCERTAIN: CategoryMetadata(
        category=StandardCategory.UNCERTAIN,
        description="Ambiguous or low-confidence queries requiring conservative routing",
        benchmark_grounding=["N/A - Explicit uncertainty handling"],
        example_queries=[
            "Tell me about that thing we discussed",
            "Help me with this",
            "What do you think?",
        ],
        prefer_capabilities=["generalist", "well_rounded"],
        routing_strategy="conservative_fallback",
    ),
}


# =============================================================================
# Mapping from Fine-Grained Use Cases to Standard Categories
# =============================================================================

# Maps the existing fine-grained use cases (from prompt_classifier.py)
# to the 8 standard categories
FINE_TO_STANDARD_MAPPING: Dict[str, StandardCategory] = {
    # REASONING: Math, logic, analysis, planning
    "math_reasoning": StandardCategory.REASONING,
    "data_analysis": StandardCategory.REASONING,
    "planning": StandardCategory.REASONING,
    "financial_analysis": StandardCategory.REASONING,  # Analytical reasoning
    
    # CODING: All programming-related tasks
    "code_generation": StandardCategory.CODING,
    "code_review": StandardCategory.CODING,
    "code_refactoring": StandardCategory.CODING,
    "sql_generation": StandardCategory.CODING,
    "technical_docs": StandardCategory.CODING,  # Code documentation
    
    # CREATIVE: Writing, roleplay, brainstorming
    "creative_writing": StandardCategory.CREATIVE,
    "roleplay": StandardCategory.CREATIVE,
    "brainstorming": StandardCategory.CREATIVE,
    "style_transfer": StandardCategory.CREATIVE,
    
    # FACTUAL_QA: Questions, tutoring, research
    "general_qa": StandardCategory.FACTUAL_QA,
    "tutoring": StandardCategory.FACTUAL_QA,
    "research_assistant": StandardCategory.FACTUAL_QA,
    "legal_review": StandardCategory.FACTUAL_QA,  # Domain-specific QA
    "customer_support": StandardCategory.FACTUAL_QA,  # FAQ-style QA
    
    # SUMMARIZATION: Condensing content
    "summarization": StandardCategory.SUMMARIZATION,
    "long_context": StandardCategory.SUMMARIZATION,  # Often involves summarization
    
    # EXTRACTION: Structured output, NER, classification
    "structured_extraction": StandardCategory.EXTRACTION,
    "entity_extraction": StandardCategory.EXTRACTION,
    "text_classification": StandardCategory.EXTRACTION,
    "sentiment_analysis": StandardCategory.EXTRACTION,
    "content_moderation": StandardCategory.EXTRACTION,
    "function_calling": StandardCategory.EXTRACTION,  # Structured API calls
    
    # TRANSLATION: Language conversion
    "translation": StandardCategory.TRANSLATION,
    "paraphrasing": StandardCategory.TRANSLATION,  # Same-language "translation"
    "grammar_correction": StandardCategory.TRANSLATION,  # Text transformation
    
    # CONVERSATION: Chat, agents, tools
    "agent_workflow": StandardCategory.CONVERSATION,
    "tool_use": StandardCategory.CONVERSATION,
    
    # Multimodal (map to closest text equivalent)
    "image_understanding": StandardCategory.FACTUAL_QA,
    "vision_qa": StandardCategory.FACTUAL_QA,
    "embeddings": StandardCategory.EXTRACTION,
    "semantic_similarity": StandardCategory.EXTRACTION,
    
    # RAG maps to FACTUAL_QA (document-grounded QA)
    "rag_pipeline": StandardCategory.FACTUAL_QA,
    
    # Optimization-focused (map based on typical use)
    "cost_optimized": StandardCategory.CONVERSATION,  # General use
    "low_latency": StandardCategory.CONVERSATION,  # Chat/interactive
    "maximum_quality": StandardCategory.REASONING,  # Complex tasks
}


def map_to_standard_category(use_case: str) -> StandardCategory:
    """
    Map a use case (fine-grained or standard) to its standard category.
    
    Args:
        use_case: Use case identifier - can be either:
            - Fine-grained (e.g., "code_generation", "math_reasoning")
            - Standard category value (e.g., "coding", "reasoning")
        
    Returns:
        StandardCategory that this use case belongs to
    """
    # First, check if it's already a standard category value
    try:
        return StandardCategory(use_case)
    except ValueError:
        pass
    
    # Otherwise, look up in the fine-grained mapping
    return FINE_TO_STANDARD_MAPPING.get(
        use_case,
        StandardCategory.UNCERTAIN  # Default to uncertain if not mapped
    )


def get_category_description(category: StandardCategory) -> str:
    """Get human-readable description of a category."""
    metadata = CATEGORY_METADATA.get(category)
    return metadata.description if metadata else "Unknown category"


def get_benchmark_grounding(category: StandardCategory) -> List[str]:
    """Get the benchmarks that ground this category."""
    metadata = CATEGORY_METADATA.get(category)
    return metadata.benchmark_grounding if metadata else []


# =============================================================================
# Zero-Shot Classification Labels (for HuggingFace)
# =============================================================================

# Standard labels for zero-shot classification
# These are natural language descriptions that work well with MNLI models
# and map directly to our 8 standard categories
STANDARD_ZS_LABELS: Dict[str, StandardCategory] = {
    # REASONING
    "mathematical reasoning and problem solving": StandardCategory.REASONING,
    "logical analysis and deduction": StandardCategory.REASONING,
    
    # CODING
    "coding and programming": StandardCategory.CODING,
    "software development task": StandardCategory.CODING,
    
    # CREATIVE
    "creative writing and storytelling": StandardCategory.CREATIVE,
    "artistic and imaginative content": StandardCategory.CREATIVE,
    
    # FACTUAL_QA
    "factual question answering": StandardCategory.FACTUAL_QA,
    "knowledge and information retrieval": StandardCategory.FACTUAL_QA,
    
    # SUMMARIZATION
    "summarization and condensation": StandardCategory.SUMMARIZATION,
    
    # EXTRACTION
    "information extraction and structured output": StandardCategory.EXTRACTION,
    "classification and labeling": StandardCategory.EXTRACTION,
    
    # TRANSLATION
    "translation between languages": StandardCategory.TRANSLATION,
    
    # CONVERSATION
    "conversational dialogue and chat": StandardCategory.CONVERSATION,
    "interactive assistance": StandardCategory.CONVERSATION,
}

# Flat list of labels for zero-shot classifier
ZS_LABEL_LIST: List[str] = list(STANDARD_ZS_LABELS.keys())


def map_zs_label_to_category(label: str) -> StandardCategory:
    """Map a zero-shot label to its standard category."""
    return STANDARD_ZS_LABELS.get(label, StandardCategory.UNCERTAIN)

