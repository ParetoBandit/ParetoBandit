#!/usr/bin/env python3
"""
Quick Train & Validate V2: Improved Transfer Validation

Improvements over V1:
1. Uses intelligence_index instead of just HLE
2. Adds probability calibration
3. Creates interaction features between prompt complexity and model capability
4. Better feature engineering

Target: r > 0.65, calibration < 15%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
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
    """Add model benchmark features from cache - IMPROVED VERSION."""
    # Load name mappings
    try:
        from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
    except:
        OPENCOMPASS_TO_CACHE = {}
    
    cache_path = Path(__file__).parent.parent.parent / 'data' / 'models_cache.json'
    
    with open(cache_path) as f:
        cache_data = json.load(f)
    
    if 'models' in cache_data:
        cache = cache_data['models']
    else:
        cache = cache_data
    
    # Create lookup
    benchmarks_by_model = {}
    for model in cache:
        name = model['name']
        slug = model.get('slug', '')
        benchmarks_by_model[name] = model
        benchmarks_by_model[slug] = model
        benchmarks_by_model[name.lower()] = model
        benchmarks_by_model[slug.lower()] = model
    
    def get_feature(model_name, feature_name):
        # Try direct lookup
        if model_name in benchmarks_by_model:
            val = benchmarks_by_model[model_name].get(feature_name, np.nan)
            return val * 100 if not pd.isna(val) else np.nan
        
        # Try mapped name
        mapped_name = OPENCOMPASS_TO_CACHE.get(model_name, model_name)
        if mapped_name in benchmarks_by_model:
            val = benchmarks_by_model[mapped_name].get(feature_name, np.nan)
            return val * 100 if not pd.isna(val) else np.nan
        
        # Try lowercase
        if model_name.lower() in benchmarks_by_model:
            val = benchmarks_by_model[model_name.lower()].get(feature_name, np.nan)
            return val * 100 if not pd.isna(val) else np.nan
        
        return np.nan
    
    # Add multiple capability proxies
    df['model_hle'] = df['model'].apply(lambda m: get_feature(m, 'hle'))
    df['model_intelligence_index'] = df['model'].apply(lambda m: get_feature(m, 'intelligence_index'))
    df['model_gpqa'] = df['model'].apply(lambda m: get_feature(m, 'gpqa'))
    df['model_mmlu_pro'] = df['model'].apply(lambda m: get_feature(m, 'mmlu_pro'))
    
    # Report coverage
    print(f"\nModel feature coverage:")
    print(f"  HLE: {df['model_hle'].notna().sum()}/{len(df)} ({df['model_hle'].notna().mean():.1%})")
    print(f"  Intelligence Index: {df['model_intelligence_index'].notna().sum()}/{len(df)} ({df['model_intelligence_index'].notna().mean():.1%})")
    print(f"  GPQA: {df['model_gpqa'].notna().sum()}/{len(df)} ({df['model_gpqa'].notna().mean():.1%})")
    print(f"  MMLU-Pro: {df['model_mmlu_pro'].notna().sum()}/{len(df)} ({df['model_mmlu_pro'].notna().mean():.1%})")
    
    # Fill missing values with mean (for models that have some but not all benchmarks)
    for col in ['model_hle', 'model_intelligence_index', 'model_gpqa', 'model_mmlu_pro']:
        if df[col].notna().any():
            df[col].fillna(df[col].mean(), inplace=True)
    
    return df

def create_interaction_features(df):
    """Create interaction features between prompt complexity and model capability."""
    print("\nCreating interaction features...")
    
    # Use intelligence_index as primary capability proxy
    capability = df['model_intelligence_index']
    
    # Key interactions: How does prompt complexity interact with model capability?
    df['reasoning_x_capability'] = df['nvidia_reasoning'] * capability
    df['constraint_x_capability'] = df['nvidia_constraint'] * capability
    df['domain_x_capability'] = df['nvidia_domain_knowledge'] * capability
    
    # Ratio features: Is the model capable enough for this prompt?
    df['capability_per_reasoning'] = capability / (df['nvidia_reasoning'] + 0.01)
    df['capability_per_constraint'] = capability / (df['nvidia_constraint'] + 0.01)
    df['capability_per_domain'] = capability / (df['nvidia_domain_knowledge'] + 0.01)
    
    # Difficulty score (high reasoning + high constraints + high domain knowledge)
    df['prompt_difficulty'] = (
        df['nvidia_reasoning'] * 0.4 +
        df['nvidia_constraint'] * 0.3 + 
        df['nvidia_domain_knowledge'] * 0.3
    )
    
    # Capability gap: difference between required and available capability
    df['capability_gap'] = capability - (df['prompt_difficulty'] * 100)
    
    print(f"  Added 8 interaction features")
    
    return df

def prepare_features(df):
    """Prepare feature matrix with improved features."""
    feature_cols = [
        # Original NVIDIA features (6)
        'nvidia_creativity',
        'nvidia_reasoning', 
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        
        # Model capability features (4)
        'model_intelligence_index',
        'model_hle',
        'model_gpqa',
        'model_mmlu_pro',
        
        # Interaction features (8)
        'reasoning_x_capability',
        'constraint_x_capability',
        'domain_x_capability',
        'capability_per_reasoning',
        'capability_per_constraint',
        'capability_per_domain',
        'prompt_difficulty',
        'capability_gap'
    ]
    
    # Filter out rows with missing features
    df_clean = df[feature_cols + ['success', 'model']].dropna()
    
    print(f"\nAfter removing missing features: {len(df_clean)} examples")
    print(f"  Total features: {len(feature_cols)}")
    
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

def train_xgboost_calibrated(X_train, y_train):
    """Train XGBoost with calibration for better probability estimates."""
    print(f"\n{'='*80}")
    print("TRAINING XGBOOST WITH CALIBRATION")
    print("="*80)
    
    # Base XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,  # Slightly shallower to avoid overfitting
        'learning_rate': 0.05,  # Lower learning rate
        'n_estimators': 200,  # More trees
        'subsample': 0.8,
        'colsample_bytree': 0.7,  # Lower to force considering all features
        'min_child_weight': 5,  # Higher to avoid overfitting
        'gamma': 0.1,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'eval_metric': 'auc'
    }
    
    base_model = xgb.XGBClassifier(**params)
    
    # 5-fold cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"  Base model CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train base model
    print("\nTraining base model...")
    base_model.fit(X_train, y_train)
    
    # Calibrate probabilities using isotonic regression
    print("\nCalibrating probabilities (isotonic regression)...")
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',  # Better for tree models
        cv=3,
        n_jobs=-1
    )
    calibrated_model.fit(X_train, y_train)
    
    print("✓ Calibration complete")
    
    # Compare base vs calibrated on training data
    base_pred_proba = base_model.predict_proba(X_train)[:, 1]
    calib_pred_proba = calibrated_model.predict_proba(X_train)[:, 1]
    
    base_calib_error = np.mean(np.abs(base_pred_proba - y_train))
    calib_calib_error = np.mean(np.abs(calib_pred_proba - y_train))
    
    print(f"\nCalibration comparison (on training data):")
    print(f"  Base model calibration error: ±{base_calib_error:.3f} ({base_calib_error*100:.1f}%)")
    print(f"  Calibrated model calibration error: ±{calib_calib_error:.3f} ({calib_calib_error*100:.1f}%)")
    print(f"  Improvement: {(base_calib_error - calib_calib_error)*100:.1f}%")
    
    return calibrated_model, base_model

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
    
    # Quality assessment
    if corr > 0.7:
        quality = "🎉 STRONG"
    elif corr > 0.6:
        quality = "✅ GOOD"
    elif corr > 0.5:
        quality = "⚠️  MODERATE"
    else:
        quality = "❌ WEAK"
    
    if calibration_error < 0.15:
        calib_quality = "✅ EXCELLENT"
    elif calibration_error < 0.20:
        calib_quality = "✅ GOOD"
    elif calibration_error < 0.25:
        calib_quality = "⚠️  ACCEPTABLE"
    else:
        calib_quality = "❌ POOR"
    
    print(f"\n  Transfer Quality: {quality}")
    print(f"  Calibration Quality: {calib_quality}")
    
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
    
    # Feature importance (from base model if available)
    print(f"\n{'='*80}")
    print("TOP 15 FEATURE IMPORTANCES")
    print("="*80)
    
    # Get base model from calibrated wrapper
    if hasattr(model, 'base_estimator'):
        base_model = model.base_estimator
    else:
        base_model = model
    
    if hasattr(base_model, 'feature_importances_'):
        importance = base_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(15)
        
        print(importance_df.to_string(index=False))
        
        # Calculate importance by category
        model_importance = importance_df[importance_df['feature'].str.startswith('model')]['importance'].sum()
        nvidia_importance = importance_df[importance_df['feature'].str.startswith('nvidia')]['importance'].sum()
        interaction_importance = importance_df[importance_df['feature'].str.contains('_x_|_per_|capability_gap|difficulty')]['importance'].sum()
        
        print(f"\nFeature Importance by Category:")
        print(f"  Model features: {model_importance:.1%}")
        print(f"  NVIDIA features: {nvidia_importance:.1%}")
        print(f"  Interaction features: {interaction_importance:.1%}")
    
    return {
        'overall': {
            'n': len(y_val),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'correlation': float(corr),
            'p_value': float(p_value),
            'calibration_error': float(calibration_error),
            'quality': quality,
            'calibration_quality': calib_quality
        },
        'by_model': results_by_model
    }

def save_results(model, results, feature_names):
    """Save model and results."""
    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / 'reasoning_xgboost_v2.joblib'
    joblib.dump(model, model_path)
    print(f"\n✓ Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        'intent': 'reasoning',
        'version': 2,
        'improvements': [
            'Multiple capability proxies (intelligence_index, HLE, GPQA, MMLU-Pro)',
            'Interaction features',
            'Probability calibration',
            'Better hyperparameters'
        ],
        'feature_names': feature_names,
        'validation_results': results
    }
    
    metadata_path = output_dir / 'reasoning_validation_results_v2.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved results to {metadata_path}")

def main():
    print("="*80)
    print("QUICK TRAIN & VALIDATE V2: IMPROVED TRANSFER VALIDATION")
    print("="*80)
    print("\nImprovements:")
    print("  1. Multiple capability proxies (intelligence_index + HLE + GPQA + MMLU-Pro)")
    print("  2. Interaction features (8 new features)")
    print("  3. Probability calibration (isotonic regression)")
    print("  4. Better hyperparameters")
    print("\nTarget: r > 0.65, calibration < 15%")
    
    # Load data
    df = load_data()
    
    # Add model features
    df = add_model_features(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Prepare features
    X, y, models, feature_names = prepare_features(df)
    
    # Split
    X_train, X_val, y_train, y_val, models_val = split_train_validation(X, y, models)
    
    if len(X_val) == 0:
        print("\n❌ No proprietary models found in data!")
        return
    
    # Train with calibration
    model, base_model = train_xgboost_calibrated(X_train, y_train)
    
    # Validate
    results = validate(model, X_val, y_val, models_val, feature_names)
    
    # Save
    save_results(model, results, feature_names)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: IMPROVED ZERO-SHOT TRANSFER VALIDATION")
    print("="*80)
    
    overall = results['overall']
    print(f"\n✅ Validation complete!")
    print(f"   N: {overall['n']} proprietary model predictions")
    print(f"   Correlation: r = {overall['correlation']:.3f} (p = {overall['p_value']:.4f})")
    print(f"   Accuracy: {overall['accuracy']:.3f} ({overall['accuracy']*100:.1f}%)")
    print(f"   AUC: {overall['auc']:.3f}")
    print(f"   Calibration Error: ±{overall['calibration_error']*100:.1f}%")
    print(f"\n   Transfer Quality: {overall['quality']}")
    print(f"   Calibration Quality: {overall['calibration_quality']}")
    
    if overall['correlation'] >= 0.65 and overall['calibration_error'] < 0.15:
        print(f"\n   🎉 EXCELLENT! Ready for KDD paper!")
    elif overall['correlation'] >= 0.60 and overall['calibration_error'] < 0.20:
        print(f"\n   ✅ GOOD! Acceptable for KDD paper.")
    else:
        print(f"\n   ⚠️  Needs more improvement. See IMPROVE_TRANSFER_VALIDATION.md")

if __name__ == '__main__':
    main()
