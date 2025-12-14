#!/usr/bin/env python3
"""
Train XGBoost Model with Hyperparameter Tuning for LLM Performance Prediction

Performs grid search to find optimal hyperparameters and trains final model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, make_scorer
import xgboost as xgb
from typing import Tuple
import json
import joblib
from datetime import datetime

def load_data() -> pd.DataFrame:
    """Load instance-level training data"""
    data_path = Path(__file__).parent / "instance_level_training_data" / "instance_level_training_data.csv"
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} training examples")
    print(f"  - Unique prompts: {df['prompt'].nunique()}")
    print(f"  - Unique models: {df['model'].nunique()}")
    return df

def load_models_cache():
    """Load model benchmark scores with imputation"""
    imputed_path = Path(__file__).parent / "anchor_based_imputation" / "models_with_imputed_scores.csv"
    
    if imputed_path.exists():
        models_df = pd.read_csv(imputed_path)
        print(f"✓ Loaded {len(models_df)} models with anchor-based imputed scores")
        return models_df
    else:
        cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
        with open(cache_path, 'r') as f:
            data = json.load(f)
        models_df = pd.DataFrame(data['models'])
        print(f"✓ Loaded {len(models_df)} models from cache")
        return models_df

def add_model_features(df: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    """Add HLE model benchmark feature"""
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
            print(f"  ⚠️  Removed {before - after} examples with missing HLE scores")
    
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
    
    print(f"\nFeatures ({len(available_features)}):")
    for feat in available_features:
        print(f"  - {feat}")
    
    return X, y, available_features

def tune_hyperparameters(X_train, y_train, cv_folds=5):
    """Perform grid search to find optimal hyperparameters"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING (Grid Search)")
    print("="*80)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass balance: {(y_train == 0).sum()}/{(y_train == 1).sum()} (Failure/Success)")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist'  # Faster
    )
    
    # Grid search with stratified CV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print(f"\nSearching {np.prod([len(v) for v in param_grid.values()]):,} combinations...")
    print(f"This may take several minutes...\n")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',  # Optimize for AUC
        cv=cv,
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)
    for param, value in grid_search.best_params_.items():
        print(f"  {param:20s}: {value}")
    
    print(f"\nBest CV AUC Score: {grid_search.best_score_:.4f}")
    
    # Show top 5 configurations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS")
    print("="*80)
    for i, row in results_df.head(5).iterrows():
        print(f"\nRank {int(row['rank_test_score'])}:")
        print(f"  Test AUC: {row['mean_test_score']:.4f} ±{row['std_test_score']:.4f}")
        print(f"  Train AUC: {row['mean_train_score']:.4f}")
        params_str = str(row['params']).replace('{', '').replace('}', '').replace("'", "")
        print(f"  Params: {params_str}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_final_model(X_train, y_train, X_test, y_test, best_params, cv_folds=5):
    """Train final model with best parameters and evaluate"""
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80)
    
    # Train with best parameters
    clf = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
    
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
    
    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)[:, 1]
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"  Train Accuracy:  {train_acc:.4f}")
    print(f"  Test Accuracy:   {test_acc:.4f}")
    print(f"  CV Accuracy:     {cv_scores.mean():.4f} ±{cv_scores.std():.4f}")
    print(f"  Test AUC-ROC:    {test_auc:.4f}")
    print(f"  CV AUC-ROC:      {cv_auc.mean():.4f} ±{cv_auc.std():.4f}")
    
    # Cross-validation fold results
    print(f"\n5-Fold Cross-Validation Results:")
    for i, (acc, auc) in enumerate(zip(cv_scores, cv_auc), 1):
        print(f"  Fold {i}: Accuracy={acc:.4f}, AUC={auc:.4f}")
    
    # Feature importance
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE (Top Features)")
    print("="*80)
    feature_importance = sorted(
        zip(X_train.columns, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feat, importance in feature_importance:
        print(f"  {feat:35s} {importance:.4f}")
    
    # Classification report
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT (Test Set)")
    print("="*80)
    print(classification_report(y_test, y_pred_test, target_names=['Failure', 'Success']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX (Test Set)")
    print("="*80)
    print(f"\n                  Predicted")
    print(f"              Failure  Success")
    print(f"Actual Failure  {cm[0,0]:6d}  {cm[0,1]:6d}  (N={cm[0,0]+cm[0,1]})")
    print(f"      Success  {cm[1,0]:6d}  {cm[1,1]:6d}  (N={cm[1,0]+cm[1,1]})")
    print(f"             (N={cm[0,0]+cm[1,0]:,})  (N={cm[0,1]+cm[1,1]:,})")
    
    print(f"\nKey Metrics:")
    print(f"  - True Positives:  {cm[1,1]:6d}")
    print(f"  - True Negatives:  {cm[0,0]:6d}")
    print(f"  - False Positives: {cm[0,1]:6d}")
    print(f"  - False Negatives: {cm[1,0]:6d}")
    
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  - Precision:       {precision:.4f}")
    print(f"  - Recall:          {recall:.4f}")
    print(f"  - F1-Score:        {f1:.4f}")
    
    return clf, {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'test_auc': test_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'feature_importance': dict(zip(X_train.columns.tolist(), clf.feature_importances_.tolist()))
    }

def save_model(clf, metrics, best_params, feature_names):
    """Save trained model and metadata"""
    output_dir = Path(__file__).parent / "xgboost_models"
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_dir / "reasoning_xgboost.joblib"
    joblib.dump(clf, model_path)
    print(f"\n✓ Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'XGBClassifier',
        'intent': 'reasoning',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'best_hyperparameters': best_params,
        'metrics': metrics
    }
    
    metadata_path = output_dir / "reasoning_xgboost_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_path}")

def main():
    print("="*80)
    print("XGBOOST MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load data
    df = load_data()
    models_cache = load_models_cache()
    
    # Filter to reasoning
    df = df[df['intent'] == 'reasoning'].copy()
    print(f"✓ Filtered to {len(df)} reasoning examples")
    
    # Add model features
    df = add_model_features(df, models_cache)
    print(f"✓ After adding model features: {len(df)} examples")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n{'='*80}")
    print("DATA SPLIT")
    print("="*80)
    print(f"  Train set: {len(X_train):,} examples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set:  {len(X_test):,} examples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Train positive rate: {(y_train==1).sum()/len(y_train)*100:.2f}%")
    print(f"  Test positive rate:  {(y_test==1).sum()/len(y_test)*100:.2f}%")
    
    # Hyperparameter tuning
    best_model, best_params = tune_hyperparameters(X_train, y_train, cv_folds=5)
    
    # Train final model
    final_model, metrics = train_final_model(X_train, y_train, X_test, y_test, best_params, cv_folds=5)
    
    # Save model
    save_model(final_model, metrics, best_params, feature_names)
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Accuracy: {metrics['test_acc']:.2%}")
    print(f"Final Test AUC-ROC:  {metrics['test_auc']:.2%}")

if __name__ == '__main__':
    main()
