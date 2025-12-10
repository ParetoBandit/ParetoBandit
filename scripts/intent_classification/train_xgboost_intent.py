"""
Train XGBoost Intent Classifier.

Trains on 5 classes: REASONING, CODING, FACTUAL_QA, AGENTIC, GENERAL

Each intent maps to a primary benchmark for model selection:
    REASONING → math_500 (IQ: Can it solve the problem?)
    CODING → livecodebench (Skill: Can it write running code?)
    FACTUAL_QA → mmlu_pro (Knowledge: Does it know the facts?)
    AGENTIC → ifeval_score (Obedience: Can it follow instructions?)
    GENERAL → mixeval_hard (Vibes: Can it handle real-world queries?)

Supports both:
- 5-fold stratified cross-validation (default, more robust)
- Fixed train/val/test splits (if data has 'split' field)

Usage:
    # 5-fold CV on labeled data  
    python scripts/train_xgboost_intent.py --dataset data/intent_training_dataset.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.xgboost_intent_classifier import (
    XGBoostIntentClassifier,
    FeatureExtractor,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def load_dataset(dataset_path: str, use_splits: bool = False):
    """Load dataset for training.
    
    Args:
        dataset_path: Path to JSON file with labeled samples
        use_splits: If True, use existing train/val/test splits
                   If False, return all samples for CV
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', data if isinstance(data, list) else [])
    metadata = data.get('metadata', {})
    
    # Normalize labels (agentic_execution -> agentic for backwards compatibility)
    for s in samples:
        label = s.get('label', '').lower()
        if label == 'agentic_execution':
            s['label'] = 'agentic'
    
    if use_splits:
        # Split by train/val/test
        splits = defaultdict(lambda: {'prompts': [], 'labels': []})
        for sample in samples:
            split = sample.get('split', 'train')
            splits[split]['prompts'].append(sample['prompt'])
            splits[split]['labels'].append(sample['label'])
        return splits, metadata
    else:
        # Return all samples for CV
        prompts = [s['prompt'] for s in samples]
        labels = [s['label'] for s in samples]
        return prompts, labels, metadata


def prepare_features(prompts, labels, extractor):
    """Extract features and encode labels."""
    X, feature_names = extractor.extract_batch(prompts)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    return X, y, feature_names, le


def train_fold(X_train, y_train, X_test, y_test, feature_names, label_encoder, 
               n_estimators=200, max_depth=6, learning_rate=0.1):
    """Train and evaluate on a single fold."""
    
    classifier = XGBoostIntentClassifier()
    classifier.label_encoder = {label: i for i, label in enumerate(label_encoder.classes_)}
    classifier.label_decoder = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Train
    classifier.train(
        X_train, y_train,
        feature_names=feature_names,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_class=len(label_encoder.classes_),
        verbose=False,
    )
    
    # Predict
    y_pred, probs = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': np.mean(f1),
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'per_class': {
            label: {'precision': precision[i], 'recall': recall[i], 'f1': f1[i], 'support': support[i]}
            for i, label in enumerate(label_encoder.classes_)
        },
    }, classifier


def run_cross_validation(X, y, feature_names, label_encoder, n_folds=5, 
                         n_estimators=200, max_depth=6, learning_rate=0.1):
    """Run stratified k-fold cross-validation."""
    
    print(f"\n{'='*60}")
    print(f"Running {n_folds}-Fold Stratified Cross-Validation")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    best_f1 = 0
    best_model = None
    
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\nFold {fold_num}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        results, model = train_fold(
            X_train, y_train, X_test, y_test,
            feature_names, label_encoder,
            n_estimators, max_depth, learning_rate
        )
        
        print(f"  Accuracy: {results['accuracy']*100:.2f}%  |  Macro F1: {results['macro_f1']*100:.2f}%")
        
        fold_results.append(results)
        
        if results['macro_f1'] > best_f1:
            best_f1 = results['macro_f1']
            best_model = model
    
    return fold_results, best_model


def evaluate_model(classifier, X, y_true, split_name, label_names=None):
    """Evaluate model on a dataset."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    # Predict
    y_pred, probs = classifier.predict(X)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Per-class metrics (5 classes)
    if label_names is None:
        label_names = ['reasoning', 'coding', 'factual_qa', 'agentic', 'general']
    print(f"\n{classification_report(y_true, y_pred, target_names=label_names, digits=3, zero_division=0)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"{'':>15}", end='')
    for name in label_names:
        print(f"{name[:8]:>9}", end='')
    print()
    
    for i, name in enumerate(label_names):
        print(f"{name:>15}", end='')
        for j in range(len(label_names)):
            print(f"{cm[i][j]:>9}", end='')
        print()
    
    return accuracy, y_pred, probs


def print_cv_summary(fold_results, label_encoder, n_folds):
    """Print cross-validation summary."""
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    accuracies = [r['accuracy'] for r in fold_results]
    macro_f1s = [r['macro_f1'] for r in fold_results]
    macro_precisions = [r['macro_precision'] for r in fold_results]
    macro_recalls = [r['macro_recall'] for r in fold_results]
    
    print(f"\nOverall Performance ({n_folds} folds):")
    print(f"  Accuracy:        {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
    print(f"  Macro Precision: {np.mean(macro_precisions)*100:.2f}% ± {np.std(macro_precisions)*100:.2f}%")
    print(f"  Macro Recall:    {np.mean(macro_recalls)*100:.2f}% ± {np.std(macro_recalls)*100:.2f}%")
    print(f"  Macro F1:        {np.mean(macro_f1s)*100:.2f}% ± {np.std(macro_f1s)*100:.2f}%")
    
    # Per-class
    print(f"\nPer-Class Performance (averaged across folds):")
    print(f"{'Category':<25} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print("-" * 65)
    
    for label in label_encoder.classes_:
        precisions = [r['per_class'][label]['precision'] for r in fold_results]
        recalls = [r['per_class'][label]['recall'] for r in fold_results]
        f1s = [r['per_class'][label]['f1'] for r in fold_results]
        
        print(f"{label:<25} {np.mean(precisions)*100:>11.1f}% {np.mean(recalls)*100:>11.1f}% {np.mean(f1s)*100:>11.1f}%")
    
    # Per-fold
    print(f"\nPer-Fold Results:")
    print(f"{'Fold':<10} {'Accuracy':>12} {'Macro F1':>12}")
    print("-" * 40)
    for i, r in enumerate(fold_results, 1):
        print(f"{i:<10} {r['accuracy']*100:>11.2f}% {r['macro_f1']*100:>11.2f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train XGBoost intent classifier with 5-fold CV (default) or fixed splits"
    )
    parser.add_argument(
        '--dataset',
        default='data/real_intent_labeled_balanced.json',
        help='Path to labeled dataset'
    )
    parser.add_argument(
        '--model-path',
        default='models/xgboost_intent_classifier.json',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    parser.add_argument(
        '--use-splits',
        action='store_true',
        help='Use existing train/val/test splits instead of CV'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=200,
        help='Number of boosting rounds'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--output',
        default='results/xgboost_cv_results.json',
        help='Path to save CV results'
    )
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import xgboost as xgb
        print("✓ XGBoost available")
    except ImportError:
        print("Error: XGBoost not installed")
        print("Install with: pip install xgboost")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("XGBoost Intent Classifier Training")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    
    if args.use_splits:
        # Use existing splits
        splits, metadata = load_dataset(args.dataset, use_splits=True)
        print(f"Using existing splits: Train={len(splits['train']['prompts'])}, "
              f"Val={len(splits['val']['prompts'])}, Test={len(splits['test']['prompts'])}")
        
        # ... (keep the old split-based training logic if needed)
        print("⚠️  Fixed split training not yet implemented in unified script.")
        print("   Use --n-folds for cross-validation instead.")
        return
    
    # Cross-validation mode (default)
    prompts, labels, metadata = load_dataset(args.dataset, use_splits=False)
    print(f"Total samples: {len(prompts)}")
    
    # Show label distribution
    label_counts = Counter(labels)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(labels) * 100
        print(f"  {label:<20} {count:>5} ({pct:.1f}%)")
    
    # Extract features
    print(f"\nExtracting features...")
    extractor = FeatureExtractor()
    X, y, feature_names, label_encoder = prepare_features(prompts, labels, extractor)
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Run cross-validation
    fold_results, best_model = run_cross_validation(
        X, y, feature_names, label_encoder,
        n_folds=args.n_folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    
    # Print summary
    print_cv_summary(fold_results, label_encoder, args.n_folds)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv_summary = {
        'n_folds': args.n_folds,
        'total_samples': len(prompts),
        'classes': list(label_encoder.classes_),
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
        },
        'overall': {
            'accuracy_mean': float(np.mean([r['accuracy'] for r in fold_results])),
            'accuracy_std': float(np.std([r['accuracy'] for r in fold_results])),
            'macro_f1_mean': float(np.mean([r['macro_f1'] for r in fold_results])),
            'macro_f1_std': float(np.std([r['macro_f1'] for r in fold_results])),
        },
        'per_class': {
            label: {
                'f1_mean': float(np.mean([r['per_class'][label]['f1'] for r in fold_results])),
                'precision_mean': float(np.mean([r['per_class'][label]['precision'] for r in fold_results])),
                'recall_mean': float(np.mean([r['per_class'][label]['recall'] for r in fold_results])),
            }
            for label in label_encoder.classes_
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n💾 CV results saved to: {output_path}")
    
    # Save best model
    if best_model:
        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save(str(model_path))
        print(f"💾 Best model saved to: {model_path}")
    
    # Feature importance from best model
    if best_model:
        print(f"\n{'='*60}")
        print("TOP 15 MOST IMPORTANT FEATURES (from best fold)")
        print(f"{'='*60}")
        
        importances = best_model.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(15, len(feature_names))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:<40} {importances[idx]:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    
    return cv_summary


if __name__ == '__main__':
    main()

