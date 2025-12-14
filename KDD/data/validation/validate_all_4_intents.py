#!/usr/bin/env python3
"""
Zero-Shot Transfer Validation for All 4 Intents

Validates that XGBoost models trained on open-source data
can predict performance on proprietary models (GPT-4o, Claude, Gemini).

Intents: Reasoning, Coding, RAG, Summarization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.stats import pearsonr
import json
import warnings
warnings.filterwarnings('ignore')


# Proprietary models to use for validation
PROPRIETARY_MODELS = [
    'gpt-4o-mini-2024-07-18',
    'gpt4o-20240806',
    'gpt4o-20241120',
    'claude-3-5-sonnet-20241022',
    'claude-3-7-sonnet-20250219',
    'gemini-1.5-pro-latest',
    'gemini-2.0-flash-exp',
]


def load_and_prepare_data(intent: str):
    """Load training data and split train/validation."""
    data_path = Path(__file__).parent / 'instance_level_training_data' / 'instance_level_training_data.csv'
    
    print(f"\nLoading data for {intent}...")
    df = pd.read_csv(data_path)
    
    # Filter for this intent
    intent_df = df[df['intent'] == intent].copy()
    print(f"  Total {intent} examples: {len(intent_df):,}")
    
    # Calculate aggregate scores for each model
    print(f"  Calculating model aggregates...")
    model_aggregates = intent_df.groupby('model')['success'].mean() * 100
    intent_df['model_aggregate'] = intent_df['model'].map(model_aggregates)
    
    # Split train (open-source) vs validation (proprietary)
    train_df = intent_df[~intent_df['model'].isin(PROPRIETARY_MODELS)].copy()
    val_df = intent_df[intent_df['model'].isin(PROPRIETARY_MODELS)].copy()
    
    print(f"  Training: {len(train_df):,} examples ({train_df['model'].nunique()} models)")
    print(f"  Validation: {len(val_df):,} examples ({val_df['model'].nunique()} models)")
    
    if len(val_df) == 0:
        print(f"  ⚠️  No proprietary models found for {intent}!")
        return None, None
    
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
    
    X_train = train_df[feature_cols].values
    y_train = train_df['success'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['success'].values
    
    return (X_train, y_train, train_df), (X_val, y_val, val_df)


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    print(f"\n  Training XGBoost...")
    
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"  Training accuracy: {train_acc:.1%}")
    
    return model


def validate_transfer(model, X_val, y_val, val_df, intent):
    """Validate zero-shot transfer to proprietary models."""
    print(f"\n{'='*80}")
    print(f"VALIDATION: ZERO-SHOT TRANSFER ({intent.upper()})")
    print(f"{'='*80}")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Overall metrics
    accuracy = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
    except:
        auc = np.nan
    
    correlation, p_value = pearsonr(y_pred_proba, y_val)
    
    # Calibration error
    calibration_error = np.abs(y_pred_proba.mean() - y_val.mean())
    
    print(f"\nOVERALL METRICS:")
    print(f"  N: {len(y_val):,}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  AUC: {auc:.3f}")
    print(f"  Correlation: r = {correlation:.3f} (p = {p_value:.4f})")
    print(f"  Calibration Error: ±{calibration_error:.1%}")
    
    # Assess quality
    if correlation > 0.60:
        quality = "✅ EXCELLENT"
    elif correlation > 0.50:
        quality = "✅ GOOD"
    elif correlation > 0.40:
        quality = "⚠️  MODERATE"
    else:
        quality = "❌ POOR"
    
    print(f"\n  {quality} transfer validation!")
    
    # Per-model breakdown
    print(f"\nPER-MODEL RESULTS:")
    
    model_results = []
    for model_name in sorted(val_df['model'].unique()):
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
        
        print(f"  {model_name}:")
        print(f"    N={len(model_y)}, Acc={model_acc:.1%}, AUC={model_auc:.3f}, r={model_corr:.3f}***")
        print(f"    Actual: {actual_success:.1%}, Predicted: {pred_success:.1%}")
        
        model_results.append({
            'model': model_name,
            'n': len(model_y),
            'accuracy': model_acc,
            'auc': model_auc,
            'correlation': model_corr,
            'p_value': model_p,
            'actual_success': actual_success,
            'predicted_success': pred_success
        })
    
    # Feature importance
    print(f"\nFEATURE IMPORTANCE:")
    feature_names = [
        'nvidia_creativity',
        'nvidia_reasoning',
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots',
        'model_aggregate'
    ]
    
    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:30s}: {importance:.1%}")
    
    return {
        'intent': intent,
        'overall': {
            'n': len(y_val),
            'accuracy': accuracy,
            'auc': auc,
            'correlation': correlation,
            'p_value': p_value,
            'calibration_error': calibration_error,
            'quality': quality
        },
        'per_model': model_results,
        'feature_importance': dict(zip(feature_names, importances.tolist()))
    }


def save_results(results, output_dir):
    """Save validation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for result in results:
        intent = result['intent']
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_python(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(item) for item in obj]
            return obj
        
        result_clean = convert_to_python(result)
        
        # Save JSON
        json_path = output_dir / f'{intent}_validation_results.json'
        with open(json_path, 'w') as f:
            json.dump(result_clean, f, indent=2)
        print(f"\n✓ Saved {intent} results: {json_path}")


def main():
    """Run validation for all intents."""
    print("="*80)
    print("ZERO-SHOT TRANSFER VALIDATION: ALL 4 INTENTS")
    print("="*80)
    print("\nValidating that models trained on open-source data")
    print("can predict performance on proprietary models (GPT-4o, Claude, Gemini)")
    
    intents = ['reasoning', 'coding', 'rag', 'summarization']
    results = []
    
    for intent in intents:
        print(f"\n{'#'*80}")
        print(f"# {intent.upper()}")
        print(f"{'#'*80}")
        
        # Load data
        train_data = load_and_prepare_data(intent)
        if train_data[0] is None:
            print(f"  ⚠️  Skipping {intent} (no validation data)")
            continue
        
        (X_train, y_train, train_df), (X_val, y_val, val_df) = train_data
        
        # Train model
        model = train_xgboost(X_train, y_train)
        
        # Validate transfer
        result = validate_transfer(model, X_val, y_val, val_df, intent)
        results.append(result)
    
    # Save all results
    output_dir = Path(__file__).parent / 'validation_results'
    save_results(results, output_dir)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: ALL INTENTS")
    print(f"{'='*80}")
    
    for result in results:
        intent = result['intent']
        overall = result['overall']
        print(f"\n{intent.upper()}:")
        print(f"  Correlation: r = {overall['correlation']:.3f}")
        print(f"  Accuracy: {overall['accuracy']:.1%}")
        print(f"  AUC: {overall['auc']:.3f}")
        print(f"  Quality: {overall['quality']}")
    
    # Check if all passed
    all_good = all(r['overall']['correlation'] > 0.50 for r in results)
    
    if all_good:
        print(f"\n✅ ALL INTENTS SHOW GOOD TRANSFER (r > 0.50)!")
        print("   Ready for KDD submission!")
    else:
        print(f"\n⚠️  Some intents show weaker transfer")
        print("   Review results and consider improvements")
    
    print(f"\n✓ Validation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
