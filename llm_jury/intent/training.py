"""
Training utilities for intent classifier.

Keep training separate from inference for clean packaging.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)


def load_labeled_data(data_path: str) -> Tuple[List[str], List[str]]:
    """
    Load labeled intent data from JSON.
    
    Args:
        data_path: Path to labeled data JSON file
        
    Returns:
        (prompts, labels) tuple
    """
    with open(data_path) as f:
        data = json.load(f)
    
    prompts = [s['prompt'] for s in data['samples']]
    labels = [s['intent_label'] for s in data['samples']]
    
    return prompts, labels


def extract_embeddings(prompts: List[str], model_name: str = 'all-MiniLM-L6-v2'):
    """
    Extract sentence embeddings from prompts.
    
    Args:
        prompts: List of prompt strings
        model_name: SentenceTransformer model name
        
    Returns:
        Numpy array of embeddings
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        prompts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


def train_xgboost_cv(
    X: np.ndarray,
    y: List[str],
    n_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Train XGBoost with k-fold cross-validation.
    
    Args:
        X: Feature matrix (embeddings)
        y: Labels
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dictionary with model, predictions, and metrics
    """
    import xgboost as xgb
    
    # Encode labels
    unique_labels = sorted(set(y))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    all_y_true = []
    all_y_pred = []
    fold_scores = []
    
    print(f"\n🎯 Training XGBoost with {n_folds}-Fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Train
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(unique_labels),
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=random_state,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        print(f"   Fold {fold}/{n_folds}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        fold_scores.append({'fold': fold, 'accuracy': acc, 'f1': f1})
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    # Train final model on all data
    print("\n🔧 Training final model on full dataset...")
    final_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_labels),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=random_state
    )
    final_model.fit(X, y_encoded)
    
    # Convert predictions back to labels
    y_true_labels = [idx_to_label[idx] for idx in all_y_true]
    y_pred_labels = [idx_to_label[idx] for idx in all_y_pred]
    
    # Overall metrics
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    
    print(f"\n📈 Overall CV Results:")
    print(f"   Accuracy: {overall_acc:.4f}")
    print(f"   F1-score: {overall_f1:.4f}")
    
    return {
        'model': final_model,
        'y_true': y_true_labels,
        'y_pred': y_pred_labels,
        'labels': unique_labels,
        'fold_scores': fold_scores,
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label
    }


def save_model(model, output_path: str):
    """Save trained model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n💾 Saved model to: {output_path}")


def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]):
    """
    Compute confusion matrix (counts and normalized).
    
    Returns:
        (cm_counts, cm_normalized)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm, cm_normalized


def print_classification_report(y_true: List[str], y_pred: List[str], labels: List[str]):
    """Print detailed classification metrics."""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    print(report)
