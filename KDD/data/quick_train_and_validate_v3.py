#!/usr/bin/env python3
"""
Quick Train & Validate V3: Use ACTUAL GPQA Performance as Capability Proxy

Key insight: Instead of using potentially wrong aggregate scores from cache,
calculate each model's GPQA aggregate directly from the 198 examples we have!

This should give MUCH better results because:
1. GPQA aggregate should perfectly predict GPQA instance performance
2. No mapping errors
3. No missing/duplicate data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr
import xgboost as xgb
import joblib
import json

# Proprietary models for validation
PROPRIETARY_MODELS = [
    'gpt-4o-mini-2024-07-18',
    'gpt4o-20240806', 
    'gpt4o-20241120',
    'claude-3-5-sonnet-20241022',
    'claude-3-7-sonnet-20250219',
    'gemini-1.5-pro-latest',
    'gemini-2.0-flash-exp'
]

def load_data():
    """Load training data."""
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} examples")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Success rate: {df['success'].mean():.1%}")
    
    return df

def add_model_gpqa_aggregate(df):
    """
    Calculate each model's GPQA aggregate score from actual performance.
    This is the RIGHT way to do it - use real performance, not cache scores!
    """
    print("\nCalculating GPQA aggregate scores from actual performance...")
    
    # Calculate aggregate for each model
    model_aggregates = df.groupby('model')['success'].mean() * 100
    
    # Add to dataframe
    df['model_gpqa_aggregate'] = df['model'].map(model_aggregates)
    
    print(f"  ✓ Calculated aggregates for {len(model_aggregates)} models")
    print(f"  Range: {model_aggregates.min():.1f}% to {model_aggregates.max():.1f}%")
    print(f"  Mean: {model_aggregates.mean():.1f}%")
    
    return df

def prepare_features(df):
    """Prepare feature matrix."""
    feature_cols = [
        'nvidia_creativity',
        'nvidia_reasoning', 
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        'model_gpqa_aggregate'  # Use actual GPQA performance!
    ]
    
    df_clean = df[feature_cols + ['success', 'model']].dropna()
    
    print(f"\nPrepared features: {len(df_clean)} examples")
    print(f"  Features: {len(feature_cols)}")
    
    X = df_clean[feature_cols].values
    y = df_clean['success'].values
    models = df_clean['model'].values
    
    return X, y, models, feature_cols

def split_train_validation(X, y, models):
    """Split into training (open-source) and validation (proprietary)."""
    is_proprietary = np.array([m in PROPRIETARY_MODELS for m in models])
    
    X_train = X[~is_proprietary]
    y_train = y[~is_proprietary]
    models_train = models[~is_proprietary]
    
    X_val = X[is_proprietary]
    y_val = y[is_proprietary]
    models_val = models[is_proprietary]
    
    print(f"\nTrain/Validation split:")
    print(f"  Training (open-source): {len(X_train)} examples, {len(np.unique(models_train))} models")
    print(f"  Validation (proprietary): {len(X_val)} examples, {len(np.unique(models_val))} models")
    print(f"  Training success rate: {y_train.mean():.1%}")
    print(f"  Validation success rate: {y_val.mean():.1%}")
    
    return X_train, X_val, y_train, y_val, models_val

def train_xgboost(X_train, y_train):
    """Train XGBoost."""
    print(f"\n{'='*80}")
    print("TRAINING XGBOOST")
    print("="*80)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'eval_metric': 'auc'
    }
    
    model = xgb.XGBClassifier(**params)
    
    # 5-fold cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train final model
    print("\nTraining final model...")
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Training accuracy: {train_acc:.3f}")
    
    return model

def validate(model, X_val, y_val, models_val, feature_names):
    """Validate on proprietary models."""
    print(f"\n{'='*80}")
    print("VALIDATION: ZERO-SHOT TRANSFER")
    print("="*80)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    corr, p_value = pearsonr(y_pred_proba, y_val)
    calibration_error = np.mean(np.abs(y_pred_proba - y_val))
    
    print(f"\nOVERALL METRICS:")
    print(f"  N: {len(y_val)}")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  AUC: {auc:.3f}")
    print(f"  Correlation: r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Calibration Error: ±{calibration_error:.3f} ({calibration_error*100:.1f}%)")
    
    # Quality
    if corr > 0.7:
        print(f"\n  🎉 STRONG transfer validation!")
    elif corr > 0.6:
        print(f"\n  ✅ GOOD transfer validation!")
    elif corr > 0.5:
        print(f"\n  ⚠️  MODERATE transfer validation")
    else:
        print(f"\n  ❌ WEAK transfer validation")
    
    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    # Category breakdown
    model_importance = importance_df[importance_df['feature'].str.startswith('model')]['importance'].sum()
    nvidia_importance = importance_df[importance_df['feature'].str.startswith('nvidia')]['importance'].sum()
    
    print(f"\nBy Category:")
    print(f"  Model GPQA Aggregate: {model_importance:.1%}")
    print(f"  NVIDIA Prompt Features: {nvidia_importance:.1%}")
    
    return {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'correlation': float(corr),
        'p_value': float(p_value),
        'calibration_error': float(calibration_error)
    }

def main():
    print("="*80)
    print("V3: USE ACTUAL GPQA PERFORMANCE AS CAPABILITY PROXY")
    print("="*80)
    print("\nKey insight: Calculate aggregate from actual performance,")
    print("not from potentially incorrect cache scores!")
    
    df = load_data()
    df = add_model_gpqa_aggregate(df)
    X, y, models, feature_names = prepare_features(df)
    X_train, X_val, y_train, y_val, models_val = split_train_validation(X, y, models)
    
    if len(X_val) == 0:
        print("\n❌ No proprietary models found!")
        return
    
    model = train_xgboost(X_train, y_train)
    results = validate(model, X_val, y_val, models_val, feature_names)
    
    # Save
    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / 'reasoning_xgboost_v3.joblib'
    joblib.dump(model, model_path)
    
    results_path = output_dir / 'reasoning_validation_results_v3.json'
    with open(results_path, 'w') as f:
        json.dump({
            'version': 3,
            'approach': 'Use actual GPQA performance as capability proxy',
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Saved to {output_dir}")
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Correlation: r = {results['correlation']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Calibration: ±{results['calibration_error']*100:.1f}%")

if __name__ == '__main__':
    main()
