"""
Neural routing module for IRT-based model selection.

This module provides a Neural Item Response Theory (IRT) router that learns:
- Prompt difficulty vectors (from text) using human-labeled complexity data
- Model skill vectors (learned embeddings) using real instance-level success labels
- Probability of success: P(correct | prompt, model)

Data Sources (all human-labeled, no synthetic benchmark scores):
1. Difficulty Training: complexity_training_data.jsonl
   - HelpSteer2: Human-annotated complexity scores (0-4)
   - GPQA Diamond: PhD-level expert-written questions
   - IFEval: Human-verified constraint annotations
   - GSM8K: Human-written math problems with solutions
   - BBH: Human-curated hard reasoning tasks

2. Full IRT Training: OpenCompass instance-level data
   - Real (prompt, model, success) tuples from actual LLM evaluations
   - Downloaded via: python KDD/data/core_scripts/build_instance_level_training_data.py
"""

from llm_jury.neural_routing.neural_IRTRouter import (
    NeuralIRTRouter,
    IRTRouterConfig,
    IRTRoutingResult,
    IRTDataset,
    ComplexityDataset,
    load_complexity_training_data,
    load_instance_level_data,
    get_device,
    train_difficulty_only,
    train_full_irt,
)

__all__ = [
    "NeuralIRTRouter",
    "IRTRouterConfig",
    "IRTRoutingResult", 
    "IRTDataset",
    "ComplexityDataset",
    "load_complexity_training_data",
    "load_instance_level_data",
    "get_device",
    "train_difficulty_only",
    "train_full_irt",
]
