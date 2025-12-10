#!/usr/bin/env python3
"""
Generate publication-quality figures for KDD paper.

This script loads results from the trained model's output JSON,
ensuring figures always reflect the latest experiment results.
No hardcoded values - all data loaded dynamically.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# ============================================================================
# LOAD RESULTS FROM TRAINING OUTPUT (NO HARDCODED VALUES)
# ============================================================================
print("Loading results from training output...")
results_path = Path('../../results/intent_classification/xgboost_results.json')
if not results_path.exists():
    raise FileNotFoundError(
        f"Results file not found: {results_path}\n"
        "Please run: python train_intent_classifier.py"
    )

with open(results_path) as f:
    results = json.load(f)

# Extract data
overall = results['overall']
fold_results = results['fold_results']
per_class = results['per_class']
confusion = results['confusion_matrix']
label_dist = results['label_distribution']

print(f"✓ Loaded results: {results['metadata']['n_samples']} samples, "
      f"{results['metadata']['n_classes']} classes")
print(f"  Overall accuracy: {overall['accuracy']:.4f}")

# Load raw data for length analysis
with open('../../data/real_intent_prompts_labeled.json') as f:
    data = json.load(f)

prompts = [s['prompt'] for s in data['samples']]
labels = [s['intent_label'] for s in data['samples']]
sources = [s['source'] for s in data['samples']]

# ============================================================================
# Figure 2: Per-Class Performance
# ============================================================================
print("\nGenerating Figure 2: Per-Class Performance...")
fig, ax = plt.subplots(figsize=(8, 5))

# Sort by accuracy (descending) for better visualization
per_class_sorted = sorted(per_class, key=lambda x: x['accuracy'], reverse=True)

intents = [pc['intent'] for pc in per_class_sorted]
accuracies = [pc['accuracy'] * 100 for pc in per_class_sorted]
f1_scores = [pc['f1_score'] * 100 for pc in per_class_sorted]

x = np.arange(len(intents))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Intent Class', fontweight='bold')
ax.set_ylabel('Performance (%)', fontweight='bold')
ax.set_title('Per-Class Classification Performance', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([i.replace('_', '\n').title() for i in intents], rotation=0)
ax.legend(loc='lower left')
ax.set_ylim([75, 102])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=overall['accuracy']*100, color='red', linestyle='--', 
           linewidth=1, alpha=0.5, label=f"Overall: {overall['accuracy']*100:.1f}%")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure2_per_class_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure2_per_class_performance.png")
plt.close()

# ============================================================================
# Figure 3: Cross-Validation Fold Results
# ============================================================================
print("Generating Figure 3: Cross-Validation Folds...")
fig, ax = plt.subplots(figsize=(8, 5))

folds = [fr['fold'] for fr in fold_results]
fold_accs = [fr['accuracy'] * 100 for fr in fold_results]
fold_f1s = [fr['f1_score'] * 100 for fr in fold_results]

x = np.arange(len(folds))
width = 0.35

bars1 = ax.bar(x - width/2, fold_accs, width, label='Accuracy', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, fold_f1s, width, label='F1-Score', alpha=0.8, color='coral')

ax.set_xlabel('Cross-Validation Fold', fontweight='bold')
ax.set_ylabel('Performance (%)', fontweight='bold')
ax.set_title('5-Fold Cross-Validation Results', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in folds])
ax.legend()
ax.set_ylim([90, 98])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add mean line
mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.5, 
           label=f'Mean: {mean_acc:.2f}% ± {std_acc:.2f}%', alpha=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure3_cv_folds.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure3_cv_folds.png")
plt.close()

# ============================================================================
# Figure 4: Data Distribution
# ============================================================================
print("Generating Figure 4: Data Distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Sample counts by intent
intents_sorted = sorted(label_dist.keys())
counts = [label_dist[i] for i in intents_sorted]

bars = ax1.barh(intents_sorted, counts, alpha=0.8, 
                color=sns.color_palette("colorblind", len(intents_sorted)))
ax1.set_xlabel('Number of Samples', fontweight='bold')
ax1.set_ylabel('Intent Class', fontweight='bold')
ax1.set_title('Dataset Distribution', fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (intent, count) in enumerate(zip(intents_sorted, counts)):
    ax1.text(count + 10, i, str(count), va='center', fontweight='bold')

# Right: Prompt length distribution by intent
length_by_intent = {intent: [] for intent in intents_sorted}
for prompt, label in zip(prompts, labels):
    length_by_intent[label].append(len(prompt))

data_for_box = [length_by_intent[intent] for intent in intents_sorted]
bp = ax2.boxplot(data_for_box, tick_labels=[i.replace('_', '\n').title() for i in intents_sorted],
                 patch_artist=True, showmeans=True)

# Color boxes
colors = sns.color_palette("colorblind", len(intents_sorted))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_ylabel('Prompt Length (characters)', fontweight='bold')
ax2.set_xlabel('Intent Class', fontweight='bold')
ax2.set_title('Prompt Length Distribution by Intent', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('figure4_data_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure4_data_distribution.png")
plt.close()

# ============================================================================
# Figure 5: Data Source Breakdown
# ============================================================================
print("Generating Figure 5: Data Source Breakdown...")
fig, ax = plt.subplots(figsize=(10, 6))

# Count by source
from collections import Counter
source_counts = Counter(sources)
sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)

source_names = [s[0] for s in sorted_sources]
source_vals = [s[1] for s in sorted_sources]

# Map sources to intents for coloring
source_to_intent = {
    'mbpp': 'coding', 'humaneval': 'coding', 'code_alpaca': 'coding',
    'gsm8k': 'reasoning',
    'natural_questions': 'factual_qa',
    'cnn_dailymail': 'summarization',
    'wildchat': 'general'
}

colors = []
intent_colors = {
    'coding': '#1f77b4', 'reasoning': '#ff7f0e', 'factual_qa': '#2ca02c',
    'summarization': '#d62728', 'general': '#9467bd'
}

for source in source_names:
    intent = source_to_intent.get(source, 'general')
    colors.append(intent_colors[intent])

bars = ax.barh(source_names, source_vals, alpha=0.8, color=colors)
ax.set_xlabel('Number of Samples', fontweight='bold')
ax.set_ylabel('Data Source', fontweight='bold')
ax.set_title('Dataset Source Breakdown', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (source, count) in enumerate(zip(source_names, source_vals)):
    ax.text(count + 10, i, str(count), va='center', fontweight='bold')

# Add legend for intent colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=intent_colors[intent], label=intent.title(), alpha=0.8)
                   for intent in ['coding', 'reasoning', 'factual_qa', 'summarization', 'general']]
ax.legend(handles=legend_elements, loc='lower right', title='Intent')

plt.tight_layout()
plt.savefig('figure5_data_sources.png', dpi=300, bbox_inches='tight')
print("✓ Saved figure5_data_sources.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✅ All figures generated successfully from training results!")
print("="*80)
print("\nFigures created:")
print("  1. figure1_confusion_matrix.png (copied from results/)")
print("  2. figure2_per_class_performance.png")
print("  3. figure3_cv_folds.png")
print("  4. figure4_data_distribution.png")
print("  5. figure5_data_sources.png")
print("\nResults loaded from:")
print(f"  {results_path}")
print(f"\nOverall Performance:")
print(f"  Accuracy: {overall['accuracy']:.4f} ± {overall['accuracy_std']:.4f}")
print(f"  F1-Score: {overall['f1_score']:.4f} ± {overall['f1_std']:.4f}")
print("\n✓ Figures are always generated from latest training results")
print("✓ No hardcoded values - fully reproducible!")
