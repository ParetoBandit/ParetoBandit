"""
Visualize Intent Classifier Results.

Creates visualizations for intent classification evaluation:
    - Confusion matrices (heatmaps)
    - Per-category performance charts
    - Accuracy comparison across splits
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(results_path: str) -> dict:
    """Load evaluation results."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(metrics: dict, split_name: str, output_dir: Path):
    """Plot confusion matrix as heatmap."""
    cm = np.array(metrics['confusion_matrix'])
    labels = metrics['labels']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {split_name.upper()} Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / f'confusion_matrix_{split_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_per_class_metrics(results: dict, output_dir: Path):
    """Plot per-class precision, recall, F1 across splits."""
    # Collect data
    categories = set()
    for split_data in results.values():
        categories.update(split_data['metrics']['per_class'].keys())
    categories = sorted(categories)
    
    splits = ['train', 'val', 'test']
    metrics_names = ['precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        x = np.arange(len(categories))
        width = 0.25
        
        for i, split in enumerate(splits):
            if split not in results:
                continue
            
            values = []
            for cat in categories:
                cat_metrics = results[split]['metrics']['per_class'].get(cat, {})
                values.append(cat_metrics.get(metric_name, 0) * 100)
            
            ax.bar(x + i * width, values, width, label=split.upper())
        
        ax.set_xlabel('Category', fontsize=11)
        ax.set_ylabel(f'{metric_name.capitalize()} (%)', fontsize=11)
        ax.set_title(f'Per-Category {metric_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
    
    plt.tight_layout()
    output_path = output_dir / 'per_class_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall_comparison(results: dict, output_dir: Path):
    """Plot overall metrics comparison across splits."""
    splits = ['train', 'val', 'test']
    metrics_names = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(splits))
    width = 0.2
    
    for i, (metric_name, label) in enumerate(zip(metrics_names, metric_labels)):
        values = []
        for split in splits:
            if split in results:
                values.append(results[split]['metrics'][metric_name] * 100)
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=label)
    
    ax.set_xlabel('Split', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.upper() for s in splits])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'overall_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_distribution(results: dict, output_dir: Path):
    """Plot distribution of errors by category."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    splits = ['train', 'val', 'test']
    
    for idx, split in enumerate(splits):
        if split not in results:
            continue
        
        ax = axes[idx]
        
        # Count misclassifications by true label
        error_counts = {}
        total_counts = {}
        
        for pred in results[split]['predictions']:
            true_label = pred['true_label']
            total_counts[true_label] = total_counts.get(true_label, 0) + 1
            if not pred['correct']:
                error_counts[true_label] = error_counts.get(true_label, 0) + 1
        
        # Calculate error rates
        categories = sorted(total_counts.keys())
        error_rates = [
            (error_counts.get(cat, 0) / total_counts[cat] * 100)
            for cat in categories
        ]
        correct_rates = [100 - er for er in error_rates]
        
        x = np.arange(len(categories))
        
        ax.bar(x, correct_rates, label='Correct', color='green', alpha=0.7)
        ax.bar(x, error_rates, bottom=correct_rates, label='Errors', color='red', alpha=0.7)
        
        ax.set_xlabel('Category', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'{split.upper()} Set - Error Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'error_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize intent classifier results")
    parser.add_argument(
        '--results',
        default='results/intent_classifier_evaluation.json',
        help='Path to evaluation results JSON'
    )
    parser.add_argument(
        '--output-dir',
        default='results/intent_classifier_plots',
        help='Directory to save plots'
    )
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    print("\nGenerating visualizations...")
    
    # Plot confusion matrices for each split
    for split in ['train', 'val', 'test']:
        if split in results:
            plot_confusion_matrix(results[split]['metrics'], split, output_dir)
    
    # Plot per-class metrics
    plot_per_class_metrics(results, output_dir)
    
    # Plot overall comparison
    plot_overall_comparison(results, output_dir)
    
    # Plot error distribution
    plot_error_distribution(results, output_dir)
    
    print("\n" + "="*60)
    print("Visualizations complete! ✅")
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    # Check for dependencies
    try:
        import matplotlib
        import seaborn
    except ImportError as e:
        print(f"Error: Missing required package: {e.name}")
        print("Install with: pip install matplotlib seaborn")
        sys.exit(1)
    
    main()

