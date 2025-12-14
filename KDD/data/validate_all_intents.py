#!/usr/bin/env python3
"""
Validate Zero-Shot Transfer Across ALL Intents

Tests transfer validation for:
1. Reasoning (GPQA)
2. Coding (HumanEval)
3. Summarization (IFEval)

Key: Use each intent's OWN aggregate score as capability proxy!
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

# Proprietary models
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
    """Load all training data."""
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} total examples")
    print(f"\nBy intent:")
    for intent in df['intent'].unique():
        count = (df['intent'] == intent).sum()
        models = df[df['intent'] == intent]['model'].nunique()
        print(f"  {intent}: {count:,} examples, {models} models")
    
    return df

def add_aggregate_scores(df):
    """Calculate aggregate score for each model-intent combination."""
    print("\nCalculating aggregate scores for each model-intent...")
    
    # For each intent, calculate each model's aggregate
    aggregates = df.groupby(['intent', 'model'])['success'].mean() * 100
    
    # Add as feature
    df['model_aggregate'] = df.apply(
        lambda row: aggregates.get((row['intent'], row['model']), np.nan),
        axis=1
    )
    
    print(f"  ✓ Added aggregate scores for all model-intent pairs")
    
    return df

def validate_intent(df_intent, intent_name):
    """Validate one intent."""
    print(f"\n{'='*80}")
    print(f"INTENT: {intent_name.upper()}")
    print("="*80)
    
    # Prepare features
    feature_cols = [
        'nvidia_creativity',
        'nvidia_reasoning', 
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        'model_aggregate'
    ]
    
    df_clean = df_intent[feature_cols + ['success', 'model']].dropna()
    
    if len(df_clean) == 0:
        print(f"❌ No data for {intent_name}")
        return None
    
    print(f"Examples: {len(df_clean)}")
    print(f"Models: {df_clean['model'].nunique()}")
    
    X = df_clean[feature_cols].values
    y = df_clean['success'].values
    models = df_clean['model'].values
    
    # Split train/validation
    is_proprietary = np.array([m in PROPRIETARY_MODELS for m in models])
    
    X_train = X[~is_proprietary]
    y_train = y[~is_proprietary]
    X_val = X[is_proprietary]
    y_val = y[is_proprietary]
    models_val = models[is_proprietary]
    
    n_train_models = len(np.unique(models[~is_proprietary]))
    n_val_models = len(np.unique(models_val))
    
    print(f"\nSplit:")
    print(f"  Train: {len(X_train)} examples, {n_train_models} models ({y_train.mean():.1%} success)")
    print(f"  Val: {len(X_val)} examples, {n_val_models} models ({y_val.mean():.1%} success)")
    
    if len(X_val) == 0:
        print("❌ No proprietary model data for validation")
        return None
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, n_train_models), shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"  Train Accuracy: {train_acc:.3f}")
    
    # Validate
    print("\nValidation Results:")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    corr, p_value = pearsonr(y_pred_proba, y_val)
    calibration_error = np.mean(np.abs(y_pred_proba - y_val))
    
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  AUC: {auc:.3f}")
    print(f"  Correlation: r = {corr:.3f} (p = {p_value:.4f})")
    print(f"  Calibration: ±{calibration_error:.3f} ({calibration_error*100:.1f}%)")
    
    # Quality
    if corr > 0.6:
        quality = "✅ GOOD"
    elif corr > 0.5:
        quality = "✓ MODERATE"
    else:
        quality = "⚠️  WEAK"
    print(f"  Transfer Quality: {quality}")
    
    # Feature importance
    importance = model.feature_importances_
    model_feat_idx = feature_cols.index('model_aggregate')
    model_importance = importance[model_feat_idx]
    
    print(f"\nFeature Importance:")
    print(f"  Model Aggregate: {model_importance:.1%}")
    print(f"  NVIDIA Features: {(1-model_importance):.1%}")
    
    # Per-model breakdown
    print(f"\nPer-Model Performance:")
    for model_name in np.unique(models_val):
        mask = models_val == model_name
        model_acc = accuracy_score(y_val[mask], y_pred[mask])
        actual_rate = y_val[mask].mean()
        pred_rate = y_pred_proba[mask].mean()
        print(f"  {model_name}: Acc={model_acc:.3f}, Actual={actual_rate:.1%}, Pred={pred_rate:.1%}")
    
    return {
        'intent': intent_name,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_train_models': n_train_models,
        'n_val_models': n_val_models,
        'cv_accuracy': float(cv_scores.mean()),
        'train_accuracy': float(train_acc),
        'val_accuracy': float(accuracy),
        'auc': float(auc),
        'correlation': float(corr),
        'p_value': float(p_value),
        'calibration_error': float(calibration_error),
        'model_importance': float(model_importance)
    }

def main():
    print("="*80)
    print("MULTI-INTENT TRANSFER VALIDATION")
    print("="*80)
    print("\nValidating zero-shot transfer for all intents using")
    print("intent-specific aggregate scores as capability proxies.")
    
    # Load data
    df = load_data()
    df = add_aggregate_scores(df)
    
    # Validate each intent
    results = []
    
    for intent in sorted(df['intent'].unique()):
        df_intent = df[df['intent'] == intent]
        result = validate_intent(df_intent, intent)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print("="*80)
    
    if results:
        summary_df = pd.DataFrame(results)
        
        print(f"\n{'Intent':<15} {'Train':<10} {'Val':<8} {'Accuracy':<10} {'AUC':<8} {'Corr.':<10} {'Quality'}")
        print("-"*80)
        
        for _, row in summary_df.iterrows():
            quality = "✅" if row['correlation'] > 0.6 else "✓" if row['correlation'] > 0.5 else "⚠️"
            print(f"{row['intent']:<15} {row['n_train']:<10} {row['n_val']:<8} "
                  f"{row['val_accuracy']:.3f} ({row['val_accuracy']*100:4.1f}%) "
                  f"{row['auc']:<8.3f} {row['correlation']:<10.3f} {quality}")
        
        # Overall statistics
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"Mean Correlation: {summary_df['correlation'].mean():.3f}")
        print(f"Mean Accuracy: {summary_df['val_accuracy'].mean():.3f}")
        print(f"Mean AUC: {summary_df['auc'].mean():.3f}")
        print(f"Mean Calibration Error: ±{summary_df['calibration_error'].mean():.3f}")
        
        # Count quality
        good = (summary_df['correlation'] > 0.6).sum()
        moderate = ((summary_df['correlation'] > 0.5) & (summary_df['correlation'] <= 0.6)).sum()
        weak = (summary_df['correlation'] <= 0.5).sum()
        
        print(f"\nTransfer Quality:")
        print(f"  ✅ GOOD (r > 0.6): {good}/{len(results)} intents")
        print(f"  ✓ MODERATE (0.5 < r ≤ 0.6): {moderate}/{len(results)} intents")
        print(f"  ⚠️  WEAK (r ≤ 0.5): {weak}/{len(results)} intents")
        
        if good + moderate == len(results):
            print(f"\n🎉 ALL intents show acceptable transfer (r > 0.5)!")
            print(f"   Ready for KDD paper!")
        elif good + moderate >= len(results) * 0.67:
            print(f"\n✅ Most intents show acceptable transfer.")
            print(f"   Ready for KDD paper with caveats.")
        else:
            print(f"\n⚠️  Some intents need improvement.")
        
        # Save results
        output_dir = Path(__file__).parent / 'validation_results'
        output_dir.mkdir(exist_ok=True)
        
        results_path = output_dir / 'multi_intent_validation.json'
        with open(results_path, 'w') as f:
            json.dump({
                'summary': results,
                'aggregate_stats': {
                    'mean_correlation': float(summary_df['correlation'].mean()),
                    'mean_accuracy': float(summary_df['val_accuracy'].mean()),
                    'mean_auc': float(summary_df['auc'].mean()),
                    'good_count': int(good),
                    'moderate_count': int(moderate),
                    'weak_count': int(weak)
                }
            }, f, indent=2)
        
        print(f"\n✓ Saved results to {results_path}")
    else:
        print("\n❌ No validation results obtained")

if __name__ == '__main__':
    main()
