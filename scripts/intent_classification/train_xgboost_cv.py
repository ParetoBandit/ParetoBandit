"""
Train XGBoost Intent Classifier with K-Fold Cross-Validation.

Uses stratified k-fold to get robust performance estimates and
reduce variance from single train/test split.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.xgboost_intent_classifier import (
    XGBoostIntentClassifier,
    FeatureExtractor,
)


def load_dataset(dataset_path: str):
    """Load and combine all data (ignore existing splits for CV)."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Combine all samples regardless of split
    prompts = []
    labels = []
    
    for sample in data['samples']:
        prompts.append(sample['prompt'])
        labels.append(sample['label'])
    
    return prompts, labels


def train_and_evaluate_fold(X_train, y_train, X_test, y_test, feature_names, label_encoder, fold_num):
    """Train and evaluate on a single fold."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_num}")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Initialize classifier
    classifier = XGBoostIntentClassifier()
    
    # Update label mappings
    classifier.label_encoder = {label: i for i, label in enumerate(label_encoder.classes_)}
    classifier.label_decoder = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Train
    history = classifier.train(
        X_train, y_train,
        feature_names=feature_names,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_class=len(label_encoder.classes_),
        verbose=False,
    )
    
    # Predict
    y_pred, probs = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, 
        average=None,
        zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1: {macro_f1*100:.2f}%")
    
    # Per-class results
    results = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': {},
        'y_true': y_test,
        'y_pred': y_pred,
    }
    
    for i, label in enumerate(label_encoder.classes_):
        results['per_class'][label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i],
        }
    
    return results, classifier


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train XGBoost with k-fold cross-validation"
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to labeled dataset'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    parser.add_argument(
        '--output',
        default='results/xgboost_cv_results.json',
        help='Path to save CV results'
    )
    parser.add_argument(
        '--save-best-model',
        default='models/xgboost_intent_cv_best.json',
        help='Path to save best model'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("XGBoost Intent Classifier - K-Fold Cross-Validation")
    print("="*60)
    
    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    prompts, labels = load_dataset(args.dataset)
    print(f"Total samples: {len(prompts)}")
    
    # Count by label
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Extract features
    print(f"\nExtracting features...")
    extractor = FeatureExtractor()
    X, feature_names = extractor.extract_batch(prompts)
    print(f"Features: {X.shape[1]}")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Stratified K-Fold
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\n{'='*60}")
    print(f"Running {args.n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    best_f1 = 0
    best_model = None
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        results, model = train_and_evaluate_fold(
            X_train, y_train, X_test, y_test,
            feature_names, label_encoder, fold_num
        )
        
        fold_results.append(results)
        
        # Track best model
        if results['macro_f1'] > best_f1:
            best_f1 = results['macro_f1']
            best_model = model
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    # Overall metrics
    accuracies = [r['accuracy'] for r in fold_results]
    macro_f1s = [r['macro_f1'] for r in fold_results]
    macro_precisions = [r['macro_precision'] for r in fold_results]
    macro_recalls = [r['macro_recall'] for r in fold_results]
    
    print(f"\nOverall Performance ({args.n_folds} folds):")
    print(f"  Accuracy:       {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
    print(f"  Macro Precision: {np.mean(macro_precisions)*100:.2f}% ± {np.std(macro_precisions)*100:.2f}%")
    print(f"  Macro Recall:    {np.mean(macro_recalls)*100:.2f}% ± {np.std(macro_recalls)*100:.2f}%")
    print(f"  Macro F1:        {np.mean(macro_f1s)*100:.2f}% ± {np.std(macro_f1s)*100:.2f}%")
    
    # Per-class aggregated metrics
    print(f"\nPer-Class Performance (averaged across folds):")
    print(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    
    for label in label_encoder.classes_:
        precisions = [r['per_class'][label]['precision'] for r in fold_results]
        recalls = [r['per_class'][label]['recall'] for r in fold_results]
        f1s = [r['per_class'][label]['f1'] for r in fold_results]
        
        print(f"{label:<20} {np.mean(precisions)*100:>9.1f}% {np.mean(recalls)*100:>9.1f}% {np.mean(f1s)*100:>9.1f}%")
    
    # Individual fold results
    print(f"\nPer-Fold Results:")
    print(f"{'Fold':<10} {'Accuracy':>12} {'Macro F1':>12}")
    print("-" * 40)
    for i, results in enumerate(fold_results, 1):
        print(f"{i:<10} {results['accuracy']*100:>11.2f}% {results['macro_f1']*100:>11.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv_summary = {
        'n_folds': args.n_folds,
        'total_samples': len(prompts),
        'classes': list(label_encoder.classes_),
        'overall': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'macro_f1_mean': float(np.mean(macro_f1s)),
            'macro_f1_std': float(np.std(macro_f1s)),
            'macro_precision_mean': float(np.mean(macro_precisions)),
            'macro_precision_std': float(np.std(macro_precisions)),
            'macro_recall_mean': float(np.mean(macro_recalls)),
            'macro_recall_std': float(np.std(macro_recalls)),
        },
        'per_class': {},
        'per_fold': [
            {
                'fold': i,
                'accuracy': float(r['accuracy']),
                'macro_f1': float(r['macro_f1']),
            }
            for i, r in enumerate(fold_results, 1)
        ],
    }
    
    # Per-class aggregated
    for label in label_encoder.classes_:
        precisions = [r['per_class'][label]['precision'] for r in fold_results]
        recalls = [r['per_class'][label]['recall'] for r in fold_results]
        f1s = [r['per_class'][label]['f1'] for r in fold_results]
        
        cv_summary['per_class'][label] = {
            'precision_mean': float(np.mean(precisions)),
            'precision_std': float(np.std(precisions)),
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
        }
    
    with open(output_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Save best model
    if best_model:
        model_path = Path(args.save_best_model)
        best_model.save(str(model_path))
        print(f"💾 Best model saved to: {model_path}")
    
    print("\n" + "="*60)
    print("✅ Cross-validation complete!")
    print("="*60)
    
    return cv_summary


if __name__ == '__main__':
    # Check dependencies
    try:
        import sklearn
        import xgboost
    except ImportError as e:
        print(f"Error: Missing dependency: {e.name}")
        print("Install with: pip install scikit-learn xgboost")
        sys.exit(1)
    
    main()

