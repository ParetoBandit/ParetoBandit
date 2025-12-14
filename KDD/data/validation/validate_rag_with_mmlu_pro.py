#!/usr/bin/env python3
"""
RAG Validation Using MMLU-Pro as Capability Proxy

This demonstrates TRUE zero-shot transfer:
- Use MMLU-Pro (world knowledge) to predict TriviaQA (factual QA)
- No circular dependency on TriviaQA aggregates
- More production-realistic
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr
import json
import sys

# Add parent directory to path to import mapping
sys.path.insert(0, str(Path(__file__).parent))
from opencompass_name_mappings import OPENCOMPASS_TO_CACHE

# Proprietary models
PROPRIETARY_MODELS = ['gpt-4o-mini-2024-07-18']


def load_mmlu_pro_scores():
    """Load MMLU-Pro scores from models_cache."""
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache_data = json.load(f)
        models = cache_data['models']
    
    # Create mapping: model_name -> mmlu_pro score
    mmlu_pro_map = {}
    for model in models:
        name = model['name']
        mmlu_pro = model.get('mmlu_pro', None)
        if mmlu_pro and mmlu_pro != 'N/A':
            mmlu_pro_map[name] = float(mmlu_pro) * 100  # Convert to percentage
    
    print(f"Loaded MMLU-Pro scores for {len(mmlu_pro_map)} models")
    return mmlu_pro_map


def map_model_names(model_name, score_map):
    """Map OpenCompass model names to cache names using the mapping file."""
    # First try the explicit mapping
    if model_name in OPENCOMPASS_TO_CACHE:
        cache_name = OPENCOMPASS_TO_CACHE[model_name]
        if cache_name in score_map:
            return score_map[cache_name]
    
    # Try direct match
    if model_name in score_map:
        return score_map[model_name]
    
    return None


def validate_rag_with_mmlu_pro():
    """Validate RAG using MMLU-Pro as capability proxy."""
    print("="*80)
    print("RAG VALIDATION: USING MMLU-PRO AS CAPABILITY PROXY")
    print("="*80)
    print("\nThis tests TRUE zero-shot transfer:")
    print("  Train: Learn patterns from open-source models")
    print("  Feature: MMLU-Pro (world knowledge benchmark)")
    print("  Predict: TriviaQA performance (factual QA)")
    print()
    
    # Load data
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    df = pd.read_csv(data_path)
    
    # Filter for RAG
    rag_df = df[df['intent'] == 'rag'].copy()
    print(f"Total RAG examples: {len(rag_df):,}")
    
    # Load MMLU-Pro scores
    mmlu_pro_map = load_mmlu_pro_scores()
    
    # Map model names and add MMLU-Pro scores
    print("\nMapping MMLU-Pro scores to models...")
    rag_df['mmlu_pro'] = rag_df['model'].apply(lambda m: map_model_names(m, mmlu_pro_map))
    
    # Check coverage
    missing = rag_df['mmlu_pro'].isna().sum()
    print(f"  Models with MMLU-Pro: {len(rag_df) - missing:,}/{len(rag_df):,}")
    print(f"  Missing: {missing:,}")
    
    if missing > 0:
        print("\n  Models missing MMLU-Pro:")
        for model in rag_df[rag_df['mmlu_pro'].isna()]['model'].unique():
            print(f"    - {model}")
    
    # Drop rows without MMLU-Pro
    rag_df = rag_df.dropna(subset=['mmlu_pro'])
    print(f"\nUsing {len(rag_df):,} examples with MMLU-Pro scores")
    
    # Split train/validation
    train_df = rag_df[~rag_df['model'].isin(PROPRIETARY_MODELS)].copy()
    val_df = rag_df[rag_df['model'].isin(PROPRIETARY_MODELS)].copy()
    
    print(f"\nTraining: {len(train_df):,} examples ({train_df['model'].nunique()} models)")
    print(f"Validation: {len(val_df):,} examples ({val_df['model'].nunique()} models)")
    
    if len(val_df) == 0:
        print("\n❌ No proprietary models with MMLU-Pro scores found!")
        return
    
    # Prepare features (using MMLU-Pro only)
    feature_cols = [
        'nvidia_creativity',
        'nvidia_reasoning',
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        'mmlu_pro'  # ← External benchmark (not TriviaQA aggregate)
    ]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['success'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['success'].values
    
    # Train XGBoost
    print("\nTraining XGBoost with MMLU-Pro as capability proxy...")
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train, verbose=False)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Training accuracy: {train_acc:.1%}")
    
    # Validate
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    correlation, p_value = pearsonr(y_pred_proba, y_val)
    calibration_error = abs(y_pred_proba.mean() - y_val.mean())
    
    print(f"\nOVERALL METRICS:")
    print(f"  N: {len(y_val):,}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Correlation: r = {correlation:.3f} (p = {p_value:.4f})")
    print(f"  Calibration Error: ±{calibration_error:.1%}")
    
    # Compare with original (self-calculated aggregate)
    print(f"\n" + "="*80)
    print("COMPARISON WITH ORIGINAL APPROACH")
    print("="*80)
    print(f"\nOriginal (self-calculated TriviaQA aggregate):")
    print(f"  Correlation: r = 0.431")
    print(f"  Feature importance: 43.7%")
    print(f"\nNew (MMLU-Pro external benchmark):")
    print(f"  Correlation: r = {correlation:.3f}")
    
    # Feature importance
    importances = model.feature_importances_
    mmlu_pro_importance = importances[-1]  # Last is mmlu_pro
    print(f"  MMLU-Pro importance: {mmlu_pro_importance:.1%}")
    
    if correlation > 0.431:
        print(f"\n✅ IMPROVEMENT! MMLU-Pro is a better proxy (+{(correlation - 0.431):.3f})")
    elif correlation > 0.40:
        print(f"\n✅ COMPARABLE! MMLU-Pro works well (Δ{(correlation - 0.431):.3f})")
    else:
        print(f"\n⚠️  WEAKER than self-calculated aggregate")
    
    print(f"\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    for name, importance in sorted(zip(feature_cols, importances), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {name:30s}: {importance:.1%}")


if __name__ == '__main__':
    validate_rag_with_mmlu_pro()
