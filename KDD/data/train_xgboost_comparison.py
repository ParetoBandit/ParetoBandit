#!/usr/bin/env python3
"""
Compare Logistic Regression vs XGBoost for LLM Performance Prediction

Trains both models on the same data and compares performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
from typing import Tuple
import json

def load_data() -> pd.DataFrame:
    """Load instance-level training data"""
    data_path = Path(__file__).parent / "instance_level_training_data" / "instance_level_training_data.csv"
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} training examples")
    return df

def load_models_cache():
    """Load model benchmark scores"""
    imputed_path = Path(__file__).parent / "anchor_based_imputation" / "models_with_imputed_scores.csv"
    
    if imputed_path.exists():
        models_df = pd.read_csv(imputed_path)
        print(f"✓ Loaded {len(models_df)} models with imputed scores")
        return models_df
    else:
        cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
        with open(cache_path, 'r') as f:
            data = json.load(f)
        models_df = pd.DataFrame(data['models'])
        print(f"✓ Loaded {len(models_df)} models from cache")
        return models_df

def add_model_features(df: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    """Add HLE model feature"""
    try:
        from opencompass_name_mappings import OPENCOMPASS_TO_CACHE
    except:
        OPENCOMPASS_TO_CACHE = {}
    
    df = df.copy()
    df['model_normalized'] = df['model'].apply(lambda x: OPENCOMPASS_TO_CACHE.get(x, x))
    
    if 'hle' in models_df.columns:
        model_to_hle = models_df.set_index('name')['hle'].to_dict()
        df['model_hle'] = df['model_normalized'].map(model_to_hle)
        
        # Remove rows with missing HLE
        before = len(df)
        df = df.dropna(subset=['model_hle'])
        after = len(df)
        if before > after:
            print(f"  Removed {before - after} examples with missing HLE scores")
    
    df = df.drop(columns=['model_normalized'], errors='ignore')
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Prepare feature matrix and target"""
    nvidia_features = [
        'nvidia_creativity',
        'nvidia_reasoning',
        'nvidia_constraint',
        'nvidia_domain_knowledge',
        'nvidia_contextual_knowledge',
        'nvidia_few_shots'
    ]
    
    model_features = ['model_hle']
    
    all_features = nvidia_features + model_features
    available_features = [f for f in all_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['success'].astype(int).values
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"\nFeatures: {len(available_features)}")
    for feat in available_features:
        print(f"  - {feat}")
    
    return X, y, available_features

def train_logistic_regression(X_train, y_train, X_test, y_test, cv_folds=5):
    """Train Logistic Regression"""
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        penalty='l2',
        C=1.0
    )
    clf.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    
    # Predictions
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)
    y_proba_test = clf.predict_proba(X_test_scaled)[:, 1]
    
    results = {
        'model': 'Logistic Regression',
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'test_auc': roc_auc_score(y_test, y_proba_test),
        'y_pred': y_pred_test,
        'y_proba': y_proba_test,
        'feature_importance': dict(zip(X_train.columns, clf.coef_[0]))
    }
    
    print(f"\nTrain Accuracy: {results['train_acc']:.4f}")
    print(f"Test Accuracy:  {results['test_acc']:.4f}")
    print(f"CV Accuracy:    {results['cv_acc_mean']:.4f} ±{results['cv_acc_std']:.4f}")
    print(f"Test AUC:       {results['test_auc']:.4f}")
    print(f"CV AUC:         {results['cv_auc_mean']:.4f} ±{results['cv_auc_std']:.4f}")
    
    return results

def train_xgboost(X_train, y_train, X_test, y_test, cv_folds=5):
    """Train XGBoost"""
    print("\n" + "="*80)
    print("XGBOOST")
    print("="*80)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train XGBoost
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    clf.fit(X_train, y_train, verbose=False)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    
    results = {
        'model': 'XGBoost',
        'train_acc': accuracy_score(y_train, y_pred_train),
        'test_acc': accuracy_score(y_test, y_pred_test),
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'test_auc': roc_auc_score(y_test, y_proba_test),
        'y_pred': y_pred_test,
        'y_proba': y_proba_test,
        'feature_importance': dict(zip(X_train.columns, clf.feature_importances_))
    }
    
    print(f"\nTrain Accuracy: {results['train_acc']:.4f}")
    print(f"Test Accuracy:  {results['test_acc']:.4f}")
    print(f"CV Accuracy:    {results['cv_acc_mean']:.4f} ±{results['cv_acc_std']:.4f}")
    print(f"Test AUC:       {results['test_auc']:.4f}")
    print(f"CV AUC:         {results['cv_auc_mean']:.4f} ±{results['cv_auc_std']:.4f}")
    
    return results

def compare_models(lr_results, xgb_results, y_test, feature_names):
    """Compare both models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Metric': ['Train Acc', 'Test Acc', 'CV Acc', 'Test AUC', 'CV AUC'],
        'Logistic Regression': [
            f"{lr_results['train_acc']:.4f}",
            f"{lr_results['test_acc']:.4f}",
            f"{lr_results['cv_acc_mean']:.4f} ±{lr_results['cv_acc_std']:.4f}",
            f"{lr_results['test_auc']:.4f}",
            f"{lr_results['cv_auc_mean']:.4f} ±{lr_results['cv_auc_std']:.4f}",
        ],
        'XGBoost': [
            f"{xgb_results['train_acc']:.4f}",
            f"{xgb_results['test_acc']:.4f}",
            f"{xgb_results['cv_acc_mean']:.4f} ±{xgb_results['cv_acc_std']:.4f}",
            f"{xgb_results['test_auc']:.4f}",
            f"{xgb_results['cv_auc_mean']:.4f} ±{xgb_results['cv_auc_std']:.4f}",
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    # Determine winner
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    
    lr_test = lr_results['test_acc']
    xgb_test = xgb_results['test_acc']
    lr_auc = lr_results['test_auc']
    xgb_auc = xgb_results['test_auc']
    
    print(f"\nTest Accuracy:")
    if xgb_test > lr_test:
        improvement = (xgb_test - lr_test) * 100
        print(f"  🏆 XGBoost wins by {improvement:.2f} percentage points")
    elif lr_test > xgb_test:
        improvement = (lr_test - xgb_test) * 100
        print(f"  🏆 Logistic Regression wins by {improvement:.2f} percentage points")
    else:
        print(f"  🤝 Tie")
    
    print(f"\nTest AUC-ROC:")
    if xgb_auc > lr_auc:
        improvement = (xgb_auc - lr_auc) * 100
        print(f"  🏆 XGBoost wins by {improvement:.2f} percentage points")
    elif lr_auc > xgb_auc:
        improvement = (lr_auc - xgb_auc) * 100
        print(f"  🏆 Logistic Regression wins by {improvement:.2f} percentage points")
    else:
        print(f"  🤝 Tie")
    
    # Feature importance comparison
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*80)
    
    print("\nLogistic Regression (Coefficients):")
    lr_feat = sorted(lr_results['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in lr_feat[:6]:
        direction = "SUCCESS ↑" if coef > 0 else "FAILURE ↑"
        print(f"  {feat:35s} {coef:+.4f}  {direction}")
    
    print("\nXGBoost (Gain):")
    xgb_feat = sorted(xgb_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feat, importance in xgb_feat[:6]:
        print(f"  {feat:35s} {importance:.4f}")
    
    # Confusion matrices
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    for name, results in [('Logistic Regression', lr_results), ('XGBoost', xgb_results)]:
        cm = confusion_matrix(y_test, results['y_pred'])
        print(f"\n{name}:")
        print(f"                  Predicted")
        print(f"              Failure  Success")
        print(f"Actual Failure  {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"      Success  {cm[1,0]:6d}  {cm[1,1]:6d}")

def main():
    print("="*80)
    print("LOGISTIC REGRESSION vs XGBOOST COMPARISON")
    print("="*80)
    
    # Load data
    df = load_data()
    models_cache = load_models_cache()
    
    # Filter to reasoning only
    df = df[df['intent'] == 'reasoning'].copy()
    print(f"✓ Filtered to {len(df)} reasoning examples")
    
    # Add model features
    df = add_model_features(df, models_cache)
    print(f"✓ After adding model features: {len(df)} examples")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train/test split (same for both models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} examples")
    print(f"Test set:  {len(X_test)} examples")
    print(f"Class balance: {(y == 0).sum()}/{(y == 1).sum()} (Failure/Success)")
    
    # Train both models
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Compare
    compare_models(lr_results, xgb_results, y_test, feature_names)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
