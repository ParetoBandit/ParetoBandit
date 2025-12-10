"""
Evaluate Intent Classifier on Train/Val/Test Splits.

This script evaluates the intent classifier's accuracy across different data splits
and generates comprehensive reports including:
    - Accuracy, precision, recall, F1 scores per category and overall
    - Confusion matrices
    - Per-split performance metrics
    - Detailed misclassification analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

# Add parent directory to path to import llm_jury
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.intent_classifier import IntentClassifier, IntentCategory


def load_dataset(dataset_path: str) -> Dict:
    """Load the labeled dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def evaluate_split(
    classifier: IntentClassifier,
    samples: List[Dict],
    split_name: str,
) -> Dict:
    """
    Evaluate classifier on a specific split.
    
    Returns:
        Dictionary with metrics and predictions
    """
    true_labels = []
    pred_labels = []
    predictions = []
    misclassifications = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating on {split_name.upper()} set ({len(samples)} samples)")
    print(f"{'='*60}")
    
    for sample in samples:
        prompt = sample['prompt']
        true_label = sample['label']
        sample_id = sample['id']
        
        # Classify
        result = classifier.classify(prompt)
        pred_label = result.category.value
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        predictions.append({
            'id': sample_id,
            'prompt': prompt,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': result.confidence,
            'correct': true_label == pred_label,
            'signals': result.signals,
        })
        
        # Track misclassifications
        if true_label != pred_label:
            misclassifications.append({
                'id': sample_id,
                'prompt': prompt,
                'true': true_label,
                'predicted': pred_label,
                'confidence': result.confidence,
                'signals': result.signals,
            })
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, pred_labels)
    
    return {
        'split': split_name,
        'metrics': metrics,
        'predictions': predictions,
        'misclassifications': misclassifications,
    }


def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict:
    """Calculate classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Per-class metrics
    labels = sorted(set(true_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=labels,
        zero_division=0,
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    # Build per-class metrics
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
        }
    
    # Macro averages
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())
    
    return {
        'accuracy': float(accuracy),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
    }


def print_metrics(metrics: Dict, split_name: str):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} METRICS")
    print(f"{'='*60}")
    
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:  {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:     {metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1:         {metrics['macro_f1']*100:.2f}%")
    
    print(f"\n{'Per-Class Metrics':-^60}")
    print(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print('-' * 60)
    
    for label, scores in sorted(metrics['per_class'].items()):
        print(
            f"{label:<20} "
            f"{scores['precision']*100:>9.1f}% "
            f"{scores['recall']*100:>9.1f}% "
            f"{scores['f1']*100:>9.1f}% "
            f"{scores['support']:>8d}"
        )


def print_confusion_matrix(metrics: Dict):
    """Pretty print confusion matrix."""
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    
    print(f"\n{'Confusion Matrix':-^60}")
    
    # Header
    true_pred_label = "True \\ Pred"
    print(f"{true_pred_label:<20}", end='')
    for label in labels:
        print(f"{label[:8]:>9}", end='')
    print()
    print('-' * 60)
    
    # Rows
    for i, true_label in enumerate(labels):
        print(f"{true_label:<20}", end='')
        for j, pred_label in enumerate(labels):
            count = cm[i][j]
            # Highlight diagonal (correct predictions)
            if i == j:
                print(f"\033[92m{count:>9}\033[0m", end='')
            elif count > 0:
                print(f"\033[91m{count:>9}\033[0m", end='')
            else:
                print(f"{count:>9}", end='')
        print()


def print_misclassifications(misclassifications: List[Dict], split_name: str, max_show: int = 10):
    """Print misclassification examples."""
    if not misclassifications:
        print(f"\n{'='*60}")
        print(f"No misclassifications in {split_name.upper()} set! 🎉")
        print(f"{'='*60}")
        return
    
    print(f"\n{'='*60}")
    print(f"MISCLASSIFICATIONS in {split_name.upper()} ({len(misclassifications)} total)")
    print(f"{'='*60}")
    
    # Group by true -> predicted category
    error_types = defaultdict(list)
    for miss in misclassifications:
        key = f"{miss['true']} → {miss['predicted']}"
        error_types[key].append(miss)
    
    print(f"\nError Distribution:")
    for error_type, errors in sorted(error_types.items(), key=lambda x: -len(x[1])):
        print(f"  {error_type}: {len(errors)} errors")
    
    print(f"\nExample Misclassifications (showing up to {max_show}):")
    print('-' * 60)
    
    for i, miss in enumerate(misclassifications[:max_show], 1):
        print(f"\n{i}. [{miss['id']}]")
        print(f"   Prompt: {miss['prompt'][:80]}{'...' if len(miss['prompt']) > 80 else ''}")
        print(f"   True:      {miss['true']}")
        print(f"   Predicted: {miss['predicted']} (confidence: {miss['confidence']:.2f})")
        print(f"   Signals:   {', '.join(miss['signals'][:3])}")


def print_summary(results: Dict):
    """Print overall summary across all splits."""
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n{'Split':<15} {'Accuracy':>12} {'Macro F1':>12} {'Errors':>8}")
    print('-' * 60)
    
    for split in ['train', 'val', 'test']:
        if split in results:
            metrics = results[split]['metrics']
            errors = len(results[split]['misclassifications'])
            total = sum(m['support'] for m in metrics['per_class'].values())
            
            print(
                f"{split.upper():<15} "
                f"{metrics['accuracy']*100:>11.2f}% "
                f"{metrics['macro_f1']*100:>11.2f}% "
                f"{errors:>8}/{total}"
            )
    
    print()


def save_results(results: Dict, output_path: str):
    """Save detailed results to JSON."""
    # Convert to serializable format
    output = {}
    for split, data in results.items():
        output[split] = {
            'metrics': data['metrics'],
            'predictions': data['predictions'],
            'misclassifications': data['misclassifications'],
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate intent classifier")
    parser.add_argument(
        '--dataset',
        default='data/intent_classification_dataset.json',
        help='Path to labeled dataset'
    )
    parser.add_argument(
        '--output',
        default='results/intent_classifier_evaluation.json',
        help='Path to save results'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Which splits to evaluate'
    )
    parser.add_argument(
        '--show-misclass',
        type=int,
        default=5,
        help='Number of misclassifications to show per split'
    )
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    
    print(f"\nDataset: {dataset['metadata']['description']}")
    print(f"Categories: {', '.join(dataset['metadata']['categories'])}")
    print(f"Total samples: {dataset['metadata']['total_samples']}")
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = IntentClassifier()
    
    # Split samples by split type
    samples_by_split = defaultdict(list)
    for sample in dataset['samples']:
        samples_by_split[sample['split']].append(sample)
    
    # Evaluate each split
    results = {}
    for split_name in args.splits:
        if split_name not in samples_by_split:
            print(f"\nWarning: Split '{split_name}' not found in dataset, skipping...")
            continue
        
        samples = samples_by_split[split_name]
        results[split_name] = evaluate_split(classifier, samples, split_name)
        
        # Print metrics
        print_metrics(results[split_name]['metrics'], split_name)
        print_confusion_matrix(results[split_name]['metrics'])
        print_misclassifications(
            results[split_name]['misclassifications'],
            split_name,
            max_show=args.show_misclass
        )
    
    # Print overall summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_path))
    
    print("\n" + "="*60)
    print("Evaluation complete! ✅")
    print("="*60)


if __name__ == '__main__':
    # Check for sklearn
    try:
        import sklearn
    except ImportError:
        print("Error: scikit-learn is required for evaluation")
        print("Install with: pip install scikit-learn")
        sys.exit(1)
    
    main()

