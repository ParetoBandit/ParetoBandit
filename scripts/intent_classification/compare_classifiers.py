"""
Compare Regex vs XGBoost Intent Classifiers.

Evaluates both approaches on the same test set and generates comparison report.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.routing.intent_classifier import IntentClassifier
from llm_jury.routing.xgboost_intent_classifier import XGBoostIntentClassifier, FeatureExtractor


def load_dataset(dataset_path: str):
    """Load labeled dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Split by split field
    splits = defaultdict(lambda: {'prompts': [], 'labels': []})
    
    for sample in data['samples']:
        split = sample.get('split', 'test')
        splits[split]['prompts'].append(sample['prompt'])
        splits[split]['labels'].append(sample['label'])
    
    return splits, data.get('metadata', {})


def evaluate_classifier(name, classifier, prompts, labels):
    """Evaluate a classifier."""
    from sklearn.metrics import accuracy_score, classification_report
    
    print(f"\n{'='*60}")
    print(f"{name} Classifier")
    print(f"{'='*60}")
    
    predictions = []
    confidences = []
    latencies = []
    
    for prompt in prompts:
        result = classifier.classify(prompt)
        predictions.append(result.category.value)
        confidences.append(result.confidence)
        latencies.append(result.latency_ms)
    
    # Metrics
    accuracy = accuracy_score(labels, predictions)
    avg_confidence = sum(confidences) / len(confidences)
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Avg Confidence: {avg_confidence:.3f}")
    print(f"Avg Latency: {avg_latency:.2f}ms")
    
    # Per-class
    print(f"\nPer-Class Performance:")
    label_names = ['reasoning', 'coding', 'factual_qa', 'agentic_execution', 'general']
    print(classification_report(labels, predictions, target_names=label_names, digits=3))
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'avg_latency': avg_latency,
        'predictions': predictions,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare classifiers")
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to labeled dataset'
    )
    parser.add_argument(
        '--xgboost-model',
        help='Path to XGBoost model (if trained)'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("Classifier Comparison: Regex vs XGBoost")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    splits, metadata = load_dataset(args.dataset)
    
    test_prompts = splits['test']['prompts']
    test_labels = splits['test']['labels']
    
    print(f"Test set: {len(test_prompts)} samples")
    
    # Initialize classifiers
    print("\nInitializing classifiers...")
    
    # 1. Regex classifier
    regex_classifier = IntentClassifier()
    print("  ✓ Regex classifier ready")
    
    # 2. XGBoost classifier (if model provided)
    xgb_classifier = None
    if args.xgboost_model and Path(args.xgboost_model).exists():
        xgb_classifier = XGBoostIntentClassifier(model_path=args.xgboost_model)
        print(f"  ✓ XGBoost classifier loaded from {args.xgboost_model}")
    else:
        print("  ⚠️  XGBoost model not found, training required")
        print(f"     Run: python scripts/train_xgboost_intent.py --dataset {args.dataset}")
    
    # Evaluate Regex
    regex_results = evaluate_classifier("REGEX", regex_classifier, test_prompts, test_labels)
    
    # Evaluate XGBoost (if available)
    xgb_results = None
    if xgb_classifier:
        xgb_results = evaluate_classifier("XGBOOST", xgb_classifier, test_prompts, test_labels)
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Regex':>15} {'XGBoost':>15} {'Winner':>10}")
    print("-" * 62)
    
    if xgb_results:
        # Accuracy
        winner = "XGBoost" if xgb_results['accuracy'] > regex_results['accuracy'] else "Regex"
        print(f"{'Accuracy':<20} {regex_results['accuracy']*100:>14.2f}% {xgb_results['accuracy']*100:>14.2f}% {winner:>10}")
        
        # Confidence
        winner = "XGBoost" if xgb_results['avg_confidence'] > regex_results['avg_confidence'] else "Regex"
        print(f"{'Avg Confidence':<20} {regex_results['avg_confidence']:>14.3f}  {xgb_results['avg_confidence']:>14.3f}  {winner:>10}")
        
        # Latency (lower is better)
        winner = "Regex" if regex_results['avg_latency'] < xgb_results['avg_latency'] else "XGBoost"
        print(f"{'Avg Latency (ms)':<20} {regex_results['avg_latency']:>14.2f}  {xgb_results['avg_latency']:>14.2f}  {winner:>10}")
        
        # Improvement
        improvement = (xgb_results['accuracy'] - regex_results['accuracy']) * 100
        print(f"\n{'Accuracy Improvement':<20} {improvement:>+14.2f}%")
    else:
        print(f"{'Accuracy':<20} {regex_results['accuracy']*100:>14.2f}% {'N/A':>15} {'N/A':>10}")
    
    print("\n" + "="*60)
    print("✅ Comparison complete!")
    print("="*60)


if __name__ == '__main__':
    try:
        import sklearn
    except ImportError:
        print("Error: scikit-learn required")
        print("Install with: pip install scikit-learn")
        sys.exit(1)
    
    main()

