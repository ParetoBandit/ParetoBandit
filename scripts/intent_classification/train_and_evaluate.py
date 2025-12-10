#!/usr/bin/env python3
"""
Train and evaluate XGBoost intent classifier with 5-fold CV.

Shows confusion matrix and classification report to validate generalization.
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_jury.intent.training import (
    load_labeled_data,
    extract_embeddings,
    train_xgboost_cv,
    save_model,
    compute_confusion_matrix,
    print_classification_report
)


def plot_confusion_matrices(cm, cm_norm, labels, output_path=None):
    """Plot side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=10)
    
    # Normalized
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Proportion'}
    )
    axes[1].set_title('Confusion Matrix (Normalized by True Label)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to: {output_path}")
    
    plt.show()


def print_summary_table(cm, labels):
    """Print per-class accuracy and sample counts."""
    print("\n" + "="*80)
    print("PER-CLASS SUMMARY")
    print("="*80)
    print(f"{'Intent':<20} {'Samples':>10} {'Accuracy':>10} {'Most Confused With':<30}")
    print("-"*80)
    
    for i, label in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        
        # Find most confused class
        cm_row = cm[i].copy()
        cm_row[i] = 0  # Exclude diagonal
        if cm_row.max() > 0:
            confused_idx = cm_row.argmax()
            confused_label = labels[confused_idx]
            confused_count = cm_row[confused_idx]
            confused_str = f"{confused_label} ({confused_count})"
        else:
            confused_str = "None"
        
        print(f"{label:<20} {total:>10} {accuracy:>9.1%} {confused_str:<30}")
    
    print("-"*80)
    total_samples = cm.sum()
    total_correct = np.diag(cm).sum()
    overall_acc = total_correct / total_samples
    print(f"{'OVERALL':<20} {int(total_samples):>10} {overall_acc:>9.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost intent classifier with k-fold CV"
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
        help='Number of CV folds'
    )
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name'
    )
    parser.add_argument(
        '--output-dir',
        default='results/intent_classification',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model for deployment'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("XGBoost Intent Classifier - Training & Evaluation")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📚 Loading data from: {args.data}")
    prompts, labels = load_labeled_data(args.data)
    print(f"   Total samples: {len(prompts)}")
    
    # Show distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("\n📊 Intent Distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(labels) * 100
        print(f"   {label:<20} {count:>5} samples ({pct:>5.1f}%)")
    
    # Extract embeddings
    print(f"\n🔢 Extracting embeddings using {args.embedding_model}...")
    embeddings = extract_embeddings(prompts, args.embedding_model)
    print(f"   Embedding shape: {embeddings.shape}")
    
    # Train with CV
    results = train_xgboost_cv(
        embeddings,
        labels,
        n_folds=args.n_folds,
        random_state=42
    )
    
    # Compute confusion matrix
    cm, cm_norm = compute_confusion_matrix(
        results['y_true'],
        results['y_pred'],
        results['labels']
    )
    
    # Print classification report
    print_classification_report(
        results['y_true'],
        results['y_pred'],
        results['labels']
    )
    
    # Print summary table
    print_summary_table(cm, results['labels'])
    
    # Plot confusion matrix
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrices(cm, cm_norm, results['labels'], cm_path)
    
    # Save results
    results_to_save = {
        'method': 'xgboost',
        'embedding_model': args.embedding_model,
        'n_folds': args.n_folds,
        'n_samples': len(prompts),
        'label_distribution': dict(label_counts),
        'fold_scores': results['fold_scores'],
        'overall_accuracy': results['overall_accuracy'],
        'overall_f1': results['overall_f1'],
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist(),
        'labels': results['labels'],
    }
    
    results_path = output_dir / 'xgboost_cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n💾 Saved results to: {results_path}")
    
    # Save model if requested
    if args.save_model:
        model_path = output_dir / 'xgboost_intent_classifier.pkl'
        save_model(results['model'], model_path)
        print(f"✅ Model ready for deployment!")
    
    print("\n✅ Training and evaluation complete!")
    print(f"\n📁 All results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
