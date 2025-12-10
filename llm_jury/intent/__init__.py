"""
Intent Classification Module for LLM Jury.

Classifies user prompts into intent categories to enable intent-aware routing.
Ground-truth labels come from domain-specific datasets (no teacher labeling needed).

Intent Classes:
- CODING: Programming tasks, debugging, code generation
- REASONING: Math problems, logical reasoning, proofs
- FACTUAL_QA: Factual questions, knowledge retrieval
- SUMMARIZATION: Document summarization, content condensation
- AGENTIC_EXECUTION: Tool use, multi-turn tasks, function calling
- GENERAL: Chitchat, creative writing, opinions

Usage:
    from llm_jury.intent import IntentClassifier
    
    classifier = IntentClassifier()
    result = classifier.predict("Write a function to sort a list")
    # {'intent': 'coding', 'confidence': 0.95}
"""

from .classifier import IntentClassifier

__all__ = ['IntentClassifier']
