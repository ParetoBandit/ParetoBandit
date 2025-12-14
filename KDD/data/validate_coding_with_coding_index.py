#!/usr/bin/env python3
"""
Coding Validation Using Coding_Index as Capability Proxy

Tests if external Coding_Index benchmark improves transfer over self-calculated aggregate.
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
PROPRIETARY_MODELS = [
    'gpt4o-20240806',
    'gpt4o-20241120', 
    'gpt-4o-mini-2024-07-18',
    'claude-3-5-sonnet-20241022',
    'claude-3-7-sonnet-20250219',
    'gemini-2.0-flash-exp',
    'gemini-1.5-pro-latest'
]


def load_coding_index_scores():
    """Load Coding_Index scores from models_cache."""
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache_data = json.load(f)
        models = cache_data['models']
    
    # Create mapping: model_name -> coding_index score
    coding_index_map = {}
    for model in models:
        name = model['name']
        coding_index = model.get('coding_index', None)
        if coding_index and coding_index != 'N/A':
            # coding_index is already 0-100 scale
            coding_index_map[name] = float(coding_index)
    
    print(f"Loaded Coding_Index scores for {len(coding_index_map)} models")
    return coding_index_map


def map_model_names(model_name, coding_index_map):
    """Map OpenCompass model names to cache names using the mapping file."""
    # First try the explicit mapping
    if model_name in OPENCOMPASS_TO_CACHE:
        cache_name = OPENCOMPASS_TO_CACHE[model_name]
        if cache_name in coding_index_map:
            return coding_index_map[cache_name]
    
    # Try direct match
    if model_name in coding_index_map:
        return coding_index_map[model_name]
    
    return None


def validate_coding_with_coding_index():
    """Validate coding using Coding_Index as capability proxy."""
    print("="*80)
    print("CODING VALIDATION: USING CODING_INDEX AS CAPABILITY PROXY")
    print("="*80)
    print("\nThis tests TRUE zero-shot transfer:")
    print("  Train: Learn patterns from open-source models")
    print("  Feature: Coding_Index (external coding benchmark)")
    print("  Predict: HumanEval performance")
    print()
    
    # Load data
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter for coding
    coding_df = df[df['intent'] == 'coding'].copy()
    print(f"Total Coding examples: {len(coding_df):,}")
    
    # Load Coding_Index scores
    coding_index_map = load_coding_index_scores()
    
    # Map model names and add Coding_Index scores
    print("\nMapping Coding_Index scores to models...")
    coding_df['coding_index'] = coding_df['model'].apply(lambda m: map_model_names(m, coding_index_map))
    
    # Check coverage
    missing = coding_df['coding_index'].isna().sum()
    print(f"  Models with Coding_Index: {len(coding_df) - missing:,}/{len(coding_df):,}")
    print(f"  Missing: {missing:,}")
    
    if missing > 0:
        print("\n  Models missing Coding_Index:")
        for model in coding_df[coding_df['coding_index'].isna()]['model'].unique()[:10]:
            print(f"    - {model}")
    
    # Drop rows without Coding_Index
    coding_df = coding_df.dropna(subset=['coding_index'])
    print(f"\nUsing {len(coding_df):,} examples with Coding_Index scores")
    
    # Split train/validation
    train_df = coding_df[~coding_df['model'].isin(PROPRIETARY_MODELS)].copy()
    val_df = coding_df[coding_df['model'].isin(PROPRIETARY_MODELS)].copy()
    
    print(f"\nTraining: {len(train_df):,} examples ({train_df['model'].nunique()} models)")
    print(f"Validation: {len(val_df):,} examples ({val_df['model'].nunique()} models)")
    
    if len(val_df) == 0:
        print("\n❌ No proprietary models with Coding_Index scores found!")
        return
    
    # Prepare features (using Coding_Index instead of self-calculated aggregate)
    feature_cols = [
        'nvidia_creativity',
        'nvidia_reasoning',
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        'coding_index'  # ← External benchmark (not HumanEval aggregate)
    ]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['success'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['success'].values
    
    # Train XGBoost
    print("\nTraining XGBoost with Coding_Index as capability proxy...")
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
    
    # Per-model breakdown
    print(f"\n" + "="*80)
    print("PER-MODEL RESULTS")
    print("="*80)
    
    for model_name in val_df['model'].unique():
        model_mask = val_df['model'] == model_name
        model_y = y_val[model_mask]
        model_pred_proba = y_pred_proba[model_mask]
        
        model_acc = accuracy_score(model_y, (model_pred_proba >= 0.5).astype(int))
        try:
            model_auc = roc_auc_score(model_y, model_pred_proba)
        except:
            model_auc = np.nan
        
        model_corr, model_p = pearsonr(model_pred_proba, model_y)
        actual_success = model_y.mean()
        pred_success = model_pred_proba.mean()
        
        print(f"\n{model_name}:")
        print(f"  N={len(model_y)}, r={model_corr:.3f}, Acc={model_acc:.1%}, AUC={model_auc:.3f}")
        print(f"  Actual: {actual_success:.1%}, Predicted: {pred_success:.1%}")
    
    # Compare with original (self-calculated aggregate)
    print(f"\n" + "="*80)
    print("COMPARISON WITH ORIGINAL APPROACH")
    print("="*80)
    print(f"\nOriginal (self-calculated HumanEval aggregate):")
    print(f"  Correlation: r = 0.480")
    print(f"  Feature importance: 55.7%")
    print(f"\nNew (Coding_Index external benchmark):")
    print(f"  Correlation: r = {correlation:.3f}")
    
    # Feature importance
    importances = model.feature_importances_
    coding_index_importance = importances[-1]
    print(f"  Coding_Index importance: {coding_index_importance:.1%}")
    
    delta = correlation - 0.480
    if delta > 0.02:
        print(f"\n✅ SIGNIFICANT IMPROVEMENT! Coding_Index is better (+{delta:.3f})")
    elif delta > 0:
        print(f"\n✅ SLIGHT IMPROVEMENT! Coding_Index works (+{delta:.3f})")
    elif delta > -0.02:
        print(f"\n⚠️  COMPARABLE! Coding_Index works similarly (Δ{delta:.3f})")
    else:
        print(f"\n❌ WEAKER than self-calculated aggregate ({delta:.3f})")
    
    print(f"\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    for name, importance in sorted(zip(feature_cols, importances), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {name:30s}: {importance:.1%}")
    
    return {
        'correlation': correlation,
        'accuracy': accuracy,
        'auc': auc,
        'improvement': delta
    }


if __name__ == '__main__':
    validate_coding_with_coding_index()
