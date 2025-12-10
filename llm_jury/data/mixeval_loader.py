"""
MixEval Data Loader.

MixEval is the "KDD Secret Weapon" - a benchmark that correlates r=0.96 
with Chatbot Arena (real user preferences) while being automated and fresh.

Citation: 
    Ni et al. "MixEval: Deriving Wisdom of the Crowd from LLM Benchmark 
    Mixtures." NeurIPS 2024.

To get MixEval scores:
    1. Clone: git clone https://github.com/JinjieNi/MixEval
    2. Run evaluation on your models
    3. Load results here

Usage:
    from llm_jury.data.mixeval_loader import load_mixeval_scores, merge_with_models
    
    # Option 1: Load from CSV you generated
    scores = load_mixeval_scores("path/to/mixeval_results.csv")
    
    # Option 2: Use manual scores
    scores = {
        "gpt-4o": 64.7,
        "claude-3-opus": 62.3,
        "llama-3.1-405b": 66.2,
    }
    
    # Merge with existing model data
    models_with_mixeval = merge_with_models(models, scores)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# Known MixEval scores from their leaderboard (as of late 2024)
# These can be updated when new scores are available
KNOWN_MIXEVAL_SCORES = {
    # Format: "model_name_pattern": score
    # Top performers
    "gpt-4o": 64.7,
    "gpt-4-turbo": 62.5,
    "claude-3-opus": 62.3,
    "claude-3-sonnet": 58.4,
    "llama-3.1-405b": 66.2,
    "llama-3.1-70b": 56.8,
    "gemini-1.5-pro": 61.3,
    "gemini-1.5-flash": 55.2,
    "mixtral-8x22b": 51.8,
    "qwen-2-72b": 55.1,
    # Add more as needed from https://mixeval.github.io/
}

# MixEval-Hard scores (more challenging subset)
KNOWN_MIXEVAL_HARD_SCORES = {
    "gpt-4o": 48.3,
    "claude-3-opus": 45.1,
    "llama-3.1-405b": 50.2,
    # Add more as available
}


def load_mixeval_scores(
    filepath: str,
    score_column: str = "mixeval_score",
    model_column: str = "model",
) -> Dict[str, float]:
    """
    Load MixEval scores from CSV file.
    
    Expected format:
        model,mixeval_score,mixeval_hard_score
        gpt-4o,64.7,48.3
        claude-3-opus,62.3,45.1
        ...
    
    Args:
        filepath: Path to CSV with MixEval results
        score_column: Column name for MixEval score
        model_column: Column name for model identifier
        
    Returns:
        Dict mapping model name to MixEval score
    """
    import csv
    
    scores = {}
    path = Path(filepath)
    
    if not path.exists():
        logger.warning(f"MixEval file not found: {filepath}")
        return scores
    
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get(model_column, "").strip()
            score = row.get(score_column)
            
            if model and score:
                try:
                    scores[model] = float(score)
                except ValueError:
                    continue
    
    logger.info(f"Loaded {len(scores)} MixEval scores from {filepath}")
    return scores


def get_known_scores(use_hard: bool = False) -> Dict[str, float]:
    """
    Get known MixEval scores from leaderboard.
    
    Args:
        use_hard: If True, return MixEval-Hard scores
        
    Returns:
        Dict of model pattern -> score
    """
    return KNOWN_MIXEVAL_HARD_SCORES if use_hard else KNOWN_MIXEVAL_SCORES


def match_model_to_mixeval(
    model_name: str, 
    scores: Dict[str, float],
) -> Optional[float]:
    """
    Try to match a model name to MixEval scores.
    
    Uses fuzzy matching on model name patterns.
    """
    model_lower = model_name.lower()
    
    for pattern, score in scores.items():
        pattern_lower = pattern.lower()
        
        # Exact match
        if pattern_lower in model_lower or model_lower in pattern_lower:
            return score
        
        # Try matching key parts
        parts = pattern_lower.replace("-", " ").split()
        if all(part in model_lower for part in parts):
            return score
    
    return None


def merge_with_models(
    models: List[Dict],
    mixeval_scores: Optional[Dict[str, float]] = None,
    use_known: bool = True,
) -> List[Dict]:
    """
    Merge MixEval scores into model data.
    
    Args:
        models: List of model dicts
        mixeval_scores: Custom MixEval scores (model -> score)
        use_known: Also use known scores from leaderboard
        
    Returns:
        Models with 'mixeval_score' field added
    """
    # Combine custom and known scores
    all_scores = {}
    if use_known:
        all_scores.update(KNOWN_MIXEVAL_SCORES)
    if mixeval_scores:
        all_scores.update(mixeval_scores)
    
    if not all_scores:
        logger.warning("No MixEval scores available")
        return models
    
    matched = 0
    for model in models:
        name = model.get("name", "")
        score = match_model_to_mixeval(name, all_scores)
        
        if score is not None:
            model["mixeval_score"] = score
            matched += 1
    
    logger.info(f"Matched {matched}/{len(models)} models with MixEval scores")
    return models


def create_mixeval_template(
    models: List[Dict],
    output_path: str = "mixeval_template.csv",
):
    """
    Create a CSV template for running MixEval on your models.
    
    Fill in the scores after running MixEval evaluation suite.
    """
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "model_id", "mixeval_score", "mixeval_hard_score"])
        
        for model in models:
            name = model.get("name", "")
            model_id = model.get("openrouter_id", "") or model.get("id", "")
            writer.writerow([name, model_id, "", ""])
    
    logger.info(f"Created MixEval template: {output_path}")
    logger.info("Run MixEval on these models and fill in the scores")


# Convenience function
def get_quality_score(
    model: Dict,
    prefer_mixeval: bool = True,
) -> Optional[float]:
    """
    Get quality score for a model, preferring MixEval if available.
    
    Falls back to Arena ELO or intelligence_index.
    """
    if prefer_mixeval and model.get("mixeval_score"):
        return model["mixeval_score"]
    
    if model.get("arena_elo"):
        # Normalize Arena ELO to similar scale as MixEval (0-100)
        # Arena ELO typically ranges 900-1400
        elo = float(model["arena_elo"])
        return (elo - 900) / 5  # Maps 900-1400 to 0-100
    
    if model.get("intelligence_index"):
        return float(model["intelligence_index"])
    
    return None

