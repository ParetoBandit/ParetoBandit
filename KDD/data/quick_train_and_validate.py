#!/usr/bin/env python3
"""
Quick Train & Validate: Reasoning Model

Train XGBoost on open-source models, validate on proprietary models.
This provides immediate validation of zero-shot transfer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from scipy.stats import pearsonr
import xgboost as xgb
import joblib
import json

# Proprietary models to use for validation
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
    print(f"  Intents: {df['intent'].unique()}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Success rate: {df['success'].mean():.1%}")
    
    return df

def add_model_features(df):
    """Add model benchmark features from cache."""
    # Load name mappings
    try:
        from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
    except:
        OPENCOMPASS_TO_CACHE = {}
        print("⚠️  Could not load name mappings")
    
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache_data = json.load(f)
    
    # Handle cache structure
    if 'models' in cache_data:
        cache = cache_data['models']
    else:
        cache = cache_data
    
    # Create lookup by name only (not slug to avoid conflicts)
    benchmarks_by_model = {m['name']: m for m in cache}
    
    # Map features (HLE is stored as proportion 0-1, multiply by 100 for consistency)
    def get_hle(model_name):
        # First, try to map the name using our mappings
        mapped_name = OPENCOMPASS_TO_CACHE.get(model_name, model_name)
        
        # Look up in cache using mapped name
        if mapped_name in benchmarks_by_model:
            val = benchmarks_by_model[mapped_name].get('hle', np.nan)
            return val if not pd.isna(val) else np.nan
        
        # Fallback: try direct lookup
        if model_name in benchmarks_by_model:
            val = benchmarks_by_model[model_name].get('hle', np.nan)
            return val if not pd.isna(val) else np.nan
        
        return np.nan
    
    df['model_hle'] = df['model'].apply(get_hle)
    df['model_hle'] = df['model_hle'] * 100  # Convert to percentage
    
    # Check coverage
    missing = df['model_hle'].isna().sum()
    if missing > 0:
        print(f"\n⚠️  Warning: {missing} examples missing HLE scores")
        print(f"   Models without HLE:")
        for model in df[df['model_hle'].isna()]['model'].unique():
            print(f"      - {model}")
    
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
        'model_hle'
    ]
    
    # Filter out rows with missing features
    df_clean = df[feature_cols + ['success', 'model']].dropna()
    
    print(f"\nAfter removing missing features: {len(df_clean)} examples")
    
    X = df_clean[feature_cols].values
    y = df_clean['success'].values
    models = df_clean['model'].values
    
    return X, y, models, feature_cols

def split_train_validation(X, y, models):
    """Split into training (open-source) and validation (proprietary)."""
    # Identify proprietary models
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
    """Train XGBoost with cross-validation."""
    print(f"\n{'='*80}")
    print("TRAINING XGBOOST")
    print("="*80)
    
    # Simple hyperparameters (for speed)
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
    print(f"  Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    
    # Train final model
    print("\nTraining final model on all training data...")
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"  Training accuracy: {train_acc:.3f}")
    
    return model

def validate(model, X_val, y_val, models_val, feature_names):
    """Validate on proprietary models."""
    print(f"\n{'='*80}")
    print("VALIDATION: ZERO-SHOT TRANSFER TO PROPRIETARY MODELS")
    print("="*80)
    
    # Overall predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Overall metrics
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    corr, p_value = pearsonr(y_pred_proba, y_val)
    calibration_error = np.mean(np.abs(y_pred_proba - y_val))
    
    print(f"\nOVERALL VALIDATION METRICS:")
    print(f"  N: {len(y_val)}")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  AUC: {auc:.3f}")
    print(f"  Correlation: r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Calibration Error: ±{calibration_error:.3f} ({calibration_error*100:.1f}%)")
    
    # Per-model breakdown
    print(f"\nPER-MODEL BREAKDOWN:")
    print("-"*80)
    
    results_by_model = {}
    
    for model_name in np.unique(models_val):
        mask = models_val == model_name
        
        model_pred_proba = y_pred_proba[mask]
        model_actual = y_val[mask]
        model_pred = y_pred[mask]
        
        model_acc = accuracy_score(model_actual, model_pred)
        model_success_rate = model_actual.mean()
        predicted_success_rate = model_pred_proba.mean()
        
        if len(np.unique(model_actual)) > 1:
            model_auc = roc_auc_score(model_actual, model_pred_proba)
            model_corr, model_p = pearsonr(model_pred_proba, model_actual)
        else:
            model_auc = None
            model_corr, model_p = None, None
        
        print(f"\n{model_name}:")
        print(f"  N: {mask.sum()}")
        print(f"  Accuracy: {model_acc:.3f}")
        if model_auc:
            print(f"  AUC: {model_auc:.3f}")
        if model_corr:
            print(f"  Correlation: r = {model_corr:.3f} (p = {model_p:.4f})")
        print(f"  Actual success rate: {model_success_rate:.3f}")
        print(f"  Predicted success rate: {predicted_success_rate:.3f}")
        print(f"  Difference: {abs(model_success_rate - predicted_success_rate):.3f}")
        
        results_by_model[model_name] = {
            'n': int(mask.sum()),
            'accuracy': float(model_acc),
            'auc': float(model_auc) if model_auc else None,
            'correlation': float(model_corr) if model_corr else None,
            'p_value': float(model_p) if model_p else None,
            'actual_success_rate': float(model_success_rate),
            'predicted_success_rate': float(predicted_success_rate)
        }
    
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
    
    return {
        'overall': {
            'n': len(y_val),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'correlation': float(corr),
            'p_value': float(p_value),
            'calibration_error': float(calibration_error)
        },
        'by_model': results_by_model,
        'feature_importance': importance_df.to_dict(orient='records')
    }

def save_results(model, results, feature_names):
    """Save model and results."""
    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / 'reasoning_xgboost.joblib'
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        'intent': 'reasoning',
        'feature_names': feature_names,
        'validation_results': results
    }
    
    metadata_path = output_dir / 'reasoning_validation_results.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved results to {metadata_path}")

def main():
    print("="*80)
    print("QUICK TRAIN & VALIDATE: REASONING MODEL")
    print("="*80)
    print("\nStrategy: Train on open-source, validate on proprietary")
    print("This validates zero-shot transfer immediately!")
    
    # Load data
    df = load_data()
    
    # Add model features
    df = add_model_features(df)
    
    # Prepare features
    X, y, models, feature_names = prepare_features(df)
    
    # Split
    X_train, X_val, y_train, y_val, models_val = split_train_validation(X, y, models)
    
    if len(X_val) == 0:
        print("\n❌ No proprietary models found in data!")
        print("   Cannot validate zero-shot transfer without proprietary models.")
        return
    
    # Train
    model = train_xgboost(X_train, y_train)
    
    # Validate
    results = validate(model, X_val, y_val, models_val, feature_names)
    
    # Save
    save_results(model, results, feature_names)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: ZERO-SHOT TRANSFER VALIDATION")
    print("="*80)
    
    overall = results['overall']
    print(f"\n✅ Validation complete!")
    print(f"   N: {overall['n']} proprietary model predictions")
    print(f"   Correlation: r = {overall['correlation']:.3f} (p = {overall['p_value']:.4f})")
    print(f"   Accuracy: {overall['accuracy']:.3f} ({overall['accuracy']*100:.1f}%)")
    print(f"   AUC: {overall['auc']:.3f}")
    print(f"   Calibration Error: ±{overall['calibration_error']*100:.1f}%")
    
    if overall['correlation'] > 0.7:
        print(f"\n   🎉 STRONG transfer validation! (r > 0.7)")
    elif overall['correlation'] > 0.6:
        print(f"\n   ✅ GOOD transfer validation! (r > 0.6)")
    elif overall['correlation'] > 0.5:
        print(f"\n   ⚠️  MODERATE transfer validation (r > 0.5)")
    else:
        print(f"\n   ❌ WEAK transfer validation (r < 0.5)")
    
    print(f"\n📝 Ready for KDD paper!")

if __name__ == '__main__':
    main()
