"""
Load production XGBoost models from KDD/data/production_models/.

These models were trained on 113K instance-level examples and validated
with zero-shot transfer to proprietary models.
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models_dir() -> Path:
    """Get path to production models directory."""
    # From llm_jury/prediction/ → ../models/production/
    current_dir = Path(__file__).parent
    models_dir = current_dir.parent / 'models' / 'production'
    
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Production models directory not found: {models_dir}\n"
            "Expected structure: llm_jury/models/production/\n"
            "Run training scripts to generate models."
        )
    
    return models_dir


def load_model(intent: str) -> Tuple['xgb.XGBClassifier', dict]:
    """
    Load a trained XGBoost model and its metadata.
    
    Args:
        intent: One of 'reasoning', 'coding', 'summarization', 'rag'
    
    Returns:
        Tuple of (model, model_card)
    
    Raises:
        ValueError: If intent is invalid
        FileNotFoundError: If model files not found
        ImportError: If xgboost not installed
    
    Example:
        >>> model, card = load_model('rag')
        >>> print(f"Test AUC: {card['test_auc']:.3f}")
        Test AUC: 0.779
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost not installed. Install with: pip install xgboost"
        )
    
    valid_intents = ['reasoning', 'coding', 'summarization', 'rag']
    if intent not in valid_intents:
        raise ValueError(f"Invalid intent: {intent}. Must be one of {valid_intents}")
    
    models_dir = get_models_dir()
    
    # Load model
    model_path = models_dir / f'{intent}_xgboost_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run KDD/data/core_scripts/train_final_xgboost_models.py to train models."
        )
    
    model = joblib.load(model_path)
    
    # Load model card
    card_path = models_dir / f'{intent}_model_card.json'
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {card_path}")
    
    with open(card_path) as f:
        model_card = json.load(f)
    
    return model, model_card


def load_all_models() -> Dict[str, Tuple['xgb.XGBClassifier', dict]]:
    """
    Load all 4 production models.
    
    Returns:
        Dictionary mapping intent -> (model, model_card)
    
    Example:
        >>> models = load_all_models()
        >>> for intent, (model, card) in models.items():
        ...     print(f"{intent}: AUC={card['test_auc']:.3f}")
        reasoning: AUC=0.824
        coding: AUC=0.969
        summarization: AUC=0.896
        rag: AUC=0.779
    """
    intents = ['reasoning', 'coding', 'summarization', 'rag']
    
    models = {}
    for intent in intents:
        models[intent] = load_model(intent)
    
    return models


def get_model_info(intent: str) -> dict:
    """
    Get model card without loading the model.
    
    Useful for checking model metadata without loading the full XGBoost model.
    
    Args:
        intent: One of 'reasoning', 'coding', 'summarization', 'rag'
    
    Returns:
        Model card dictionary with metadata
    
    Example:
        >>> info = get_model_info('coding')
        >>> print(f"Training examples: {info['n_train_examples']:,}")
        Training examples: 5,576
    """
    valid_intents = ['reasoning', 'coding', 'summarization', 'rag']
    if intent not in valid_intents:
        raise ValueError(f"Invalid intent: {intent}. Must be one of {valid_intents}")
    
    models_dir = get_models_dir()
    card_path = models_dir / f'{intent}_model_card.json'
    
    if not card_path.exists():
        raise FileNotFoundError(f"Model card not found: {card_path}")
    
    with open(card_path) as f:
        return json.load(f)


def get_all_model_info() -> Dict[str, dict]:
    """
    Get model cards for all intents.
    
    Returns:
        Dictionary mapping intent -> model_card
    
    Example:
        >>> info = get_all_model_info()
        >>> total_examples = sum(card['n_total_examples'] for card in info.values())
        >>> print(f"Total training examples: {total_examples:,}")
        Total training examples: 113,383
    """
    intents = ['reasoning', 'coding', 'summarization', 'rag']
    return {intent: get_model_info(intent) for intent in intents}


def print_model_summary():
    """Print summary of all production models."""
    print("="*80)
    print("KDD/data Production Models Summary")
    print("="*80)
    
    try:
        info = get_all_model_info()
        
        print(f"\n{'Intent':<15} {'Training N':<12} {'Test AUC':<10} {'Test Acc':<10}")
        print("-"*80)
        
        for intent in ['reasoning', 'coding', 'summarization', 'rag']:
            card = info[intent]
            print(f"{intent:<15} {card['n_train_examples']:<12,} "
                  f"{card['test_auc']:<10.3f} {card['test_accuracy']:<10.1%}")
        
        total_examples = sum(card['n_total_examples'] for card in info.values())
        print("-"*80)
        print(f"Total: {total_examples:,} examples across 4 intents")
        print("\nAll models trained with 5-fold CV and 85/15 train/test split")
        print("Transfer validated on proprietary models (all p<0.0001)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo train production models, run:")
        print("  cd KDD/data/core_scripts")
        print("  python train_final_xgboost_models.py")
    
    print("="*80)


if __name__ == '__main__':
    # When run directly, print summary
    print_model_summary()
