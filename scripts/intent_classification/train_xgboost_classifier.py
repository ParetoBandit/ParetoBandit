#!/usr/bin/env python3
"""
Train XGBoost Intent Classifier with K-Fold Cross-Validation.

Purpose: Classify NEW prompts to determine their intent for model routing.

Features: Sentence embeddings (semantic representation of prompt text)
Model: XGBoost (handles missing values, robust, fast)
Validation: K-fold CV to ensure generalization to unseen prompts

Once we know a prompt's intent, we use composite scores (CCS, CRS, etc.)
to route it to the best-performing model for that intent.

Usage:
    python scripts/intent_classification/train_xgboost_classifier.py \
        --data data/real_intent_prompts_labeled.json \
        --n-folds 5
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    f1_score
)
from sentence_transformers import SentenceTransformer
import xgboost as xgb


def load_intent_data(data_path: str):
    """Load labeled intent data."""
    with open(data_path) as f:
        data = json.load(f)
    
    samples = data['samples']
    print(f"\n📊 Loaded {len(samples)} labeled prompts")
    print(f"   Sources: {', '.join(data['metadata']['sources'])}")
    print(f"   Label counts: {data['metadata']['label_counts']}")
    
    return samples


def extract_embeddings(prompts, model_name='all-MiniLM-L6-v2'):
    """Extract sentence embeddings from prompts."""
    print(f"\n🔢 Extracting embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(prompts, show_progress_bar=True, convert_to_numpy=True)
    print(f"   ✓ Shape: {embeddings.shape}")
    return embeddings


def train_xgboost_cv(X, y, labels, n_folds=5):
    """Train XGBoost with k-fold cross-validation."""
    print(f"\n🎯 Training XGBoost with {n_folds}-Fold Cross-Validation...")
    
    # Convert string labels to integers
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_encoded = np.array([label_to_idx[label] for label in y])
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Track predictions
    all_y_true = []
    all_y_pred = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n   Fold {fold}/{n_folds}:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(label_to_idx),
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
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
        
        print(f"      Accuracy: {acc:.4f}")
        print(f"      F1-score: {f1:.4f}")
        
        fold_scores.append({'fold': fold, 'accuracy': acc, 'f1': f1})
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
    
    # Overall metrics
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    
    print(f"\n📈 Overall 5-Fold CV Results:")
    print(f"   Accuracy: {overall_acc:.4f}")
    print(f"   F1-score: {overall_f1:.4f}")
    
    # Convert back to string labels for confusion matrix
    y_true_labels = [idx_to_label[idx] for idx in all_y_true]
    y_pred_labels = [idx_to_label[idx] for idx in all_y_pred]
    
    return y_true_labels, y_pred_labels, fold_scores, idx_to_label


def plot_confusion_matrix(y_true, y_pred, labels, output_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, pad=10)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Plot 2: Normalized
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, pad=10)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved confusion matrix to: {output_path}")
    
    plt.show()
    
    return cm, cm_normalized


def print_classification_report(y_true, y_pred, labels):
    """Print detailed classification report."""
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost intent classifier with 5-fold CV"
    )
    parser.add_argument(
        '--data',
        default='data/real_intent_prompts_labeled.json',
        help='Path to labeled intent data'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name'
    )
    parser.add_argument(
        '--output-dir',
        default='results/intent_classification',
        help='Directory to save results'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("XGBoost Intent Classification with 5-Fold CV")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    samples = load_intent_data(args.data)
    
    # Extract prompts and labels
    prompts = [s['prompt'] for s in samples]
    labels = [s['intent_label'] for s in samples]
    
    print(f"\n📝 Intent Distribution:")
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    for label, count in sorted(label_counts.items()):
        print(f"   {label:<20} {count:>5} samples ({count/len(labels)*100:.1f}%)")
    
    # Extract embeddings
    embeddings = extract_embeddings(prompts, args.embedding_model)
    
    # Train with cross-validation
    y_true, y_pred, fold_scores, idx_to_label = train_xgboost_cv(
        embeddings, 
        labels,
        labels,
        n_folds=args.n_folds
    )
    
    # Get unique labels in sorted order
    unique_labels = sorted(set(labels))
    
    # Classification report
    print_classification_report(y_true, y_pred, unique_labels)
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    cm, cm_norm = plot_confusion_matrix(y_true, y_pred, unique_labels, cm_path)
    
    # Save results
    results = {
        'method': 'xgboost',
        'embedding_model': args.embedding_model,
        'n_folds': args.n_folds,
        'n_samples': len(samples),
        'fold_scores': fold_scores,
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'overall_f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'labels': unique_labels,
    }
    
    results_path = output_dir / 'xgboost_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved results to: {results_path}")
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
