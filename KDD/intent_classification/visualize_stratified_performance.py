#!/usr/bin/env python3
"""
Visualize Stratified Performance Analysis

Creates publication-quality figures showing:
1. Training data length distribution by intent
2. Validation accuracy by length bucket
3. Per-intent performance across lengths
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Load results
with open('stratified_performance_analysis.json') as f:
    results = json.load(f)

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# FIGURE 1: TRAINING DATA LENGTH DISTRIBUTION
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

train_dist = results['training_distribution']
intents = sorted(train_dist.keys())
buckets = ['Short', 'Medium', 'Long']

# Prepare data for stacked bar chart
data = {bucket: [] for bucket in buckets}
for intent in intents:
    for bucket in buckets:
        data[bucket].append(train_dist[intent][bucket])

# Create stacked bar chart
x = np.arange(len(intents))
width = 0.6
bottom = np.zeros(len(intents))

colors = ['#3498db', '#f39c12', '#e74c3c']  # Blue, Orange, Red

for bucket, color in zip(buckets, colors):
    ax.bar(x, data[bucket], width, label=bucket, bottom=bottom, color=color)
    bottom += data[bucket]

ax.set_xlabel('Intent Class')
ax.set_ylabel('Number of Samples')
ax.set_title('Training Data: Length Distribution by Intent Class')
ax.set_xticks(x)
ax.set_xticklabels([intent.replace('_', ' ').title() for intent in intents], rotation=45, ha='right')
ax.legend(title='Length Bucket', loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Annotate SUMMARIZATION as 100% Long
summ_idx = intents.index('summarization')
ax.annotate('100% Long!', xy=(summ_idx, data['Long'][summ_idx]/2),
            xytext=(summ_idx + 0.5, data['Long'][summ_idx]/2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'training_length_distribution.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'training_length_distribution.png'}")
plt.close()

# ============================================================================
# FIGURE 2: VALIDATION ACCURACY BY LENGTH BUCKET
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 5))

overall_strat = results['overall_stratified']
bucket_names = ['Short', 'Medium', 'Long']
accuracies = [overall_strat[b]['accuracy'] * 100 for b in bucket_names]
n_samples = [overall_strat[b]['n'] for b in bucket_names]

colors = ['#3498db', '#f39c12', '#e74c3c']
bars = ax.bar(bucket_names, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

# Add sample counts on bars
for bar, n in zip(bars, n_samples):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'n={n}', ha='center', va='top', fontsize=9, color='white', weight='bold')

# Add percentage labels on top
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Length Bucket')
ax.set_title('Cross-Validation Accuracy by Prompt Length')
ax.set_ylim([85, 100])
ax.axhline(y=94.47, color='gray', linestyle='--', linewidth=1, label='Overall (94.5%)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Annotate the suspicious pattern
ax.annotate('Suspiciously high!\nSuggests length bias', 
            xy=(2, 98.15), xytext=(1.5, 90),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, color='red', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'stratified_accuracy.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'stratified_accuracy.png'}")
plt.close()

# ============================================================================
# FIGURE 3: PER-INTENT PERFORMANCE ACROSS LENGTHS
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

per_intent = results['per_intent_stratified']
intents = sorted(per_intent.keys())

# Prepare data
x = np.arange(len(buckets))
width = 0.15
colors_intent = plt.cm.Set3(np.linspace(0, 1, len(intents)))

for i, intent in enumerate(intents):
    intent_data = per_intent[intent]
    accuracies = []
    for bucket in buckets:
        if bucket in intent_data:
            accuracies.append(intent_data[bucket]['accuracy'] * 100)
        else:
            accuracies.append(np.nan)
    
    # Plot with offset
    offset = width * (i - len(intents)/2 + 0.5)
    ax.bar(x + offset, accuracies, width, label=intent.replace('_', ' ').title(),
           color=colors_intent[i], edgecolor='black', linewidth=0.5)

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Length Bucket')
ax.set_title('Per-Intent Performance Across Length Buckets')
ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.set_ylim([70, 102])
ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(loc='lower right', ncol=2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'per_intent_stratified.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'per_intent_stratified.png'}")
plt.close()

# ============================================================================
# FIGURE 4: HEATMAP OF TRAINING DISTRIBUTION
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 5))

# Prepare data for heatmap
intents = sorted(train_dist.keys())
matrix = []
for intent in intents:
    row = [train_dist[intent][bucket] for bucket in buckets]
    matrix.append(row)

# Create heatmap
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(buckets)))
ax.set_yticks(np.arange(len(intents)))
ax.set_xticklabels(buckets)
ax.set_yticklabels([intent.replace('_', ' ').title() for intent in intents])

# Add text annotations
for i in range(len(intents)):
    for j in range(len(buckets)):
        value = matrix[i][j]
        total = sum(matrix[i])
        pct = value / total * 100 if total > 0 else 0
        text = ax.text(j, i, f'{value}\n({pct:.0f}%)',
                      ha="center", va="center", color="black" if value < 250 else "white",
                      fontsize=9, weight='bold')

ax.set_title('Training Data Heatmap: Sample Count by Intent × Length')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Samples')

plt.tight_layout()
plt.savefig(output_dir / 'training_distribution_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'training_distribution_heatmap.png'}")
plt.close()

# ============================================================================
# FIGURE 5: STABILITY COMPARISON
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Accuracy range per intent
intent_ranges = {}
for intent in intents:
    intent_data = per_intent[intent]
    accs = [intent_data[b]['accuracy'] * 100 for b in buckets if b in intent_data]
    if len(accs) > 1:
        intent_ranges[intent] = max(accs) - min(accs)
    else:
        intent_ranges[intent] = 0

intent_names = [intent.replace('_', ' ').title() for intent in intent_ranges.keys()]
ranges = list(intent_ranges.values())

colors_bar = ['red' if r > 10 else 'orange' if r > 5 else 'green' for r in ranges]
bars = ax1.barh(intent_names, ranges, color=colors_bar, edgecolor='black', linewidth=1)

ax1.set_xlabel('Accuracy Range Across Lengths (%)')
ax1.set_title('Per-Intent Stability\n(Lower = Better)')
ax1.axvline(x=5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
ax1.axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim([0, max(ranges) * 1.1])

# Annotate worst performers
for i, (name, rng) in enumerate(zip(intent_names, ranges)):
    if rng > 10:
        ax1.text(rng + 0.5, i, f'{rng:.1f}%', va='center', fontsize=9, weight='bold', color='red')

# Right: Overall variance metric
metrics = ['Variance', 'Range']
values = [results['stability']['variance'] * 10000, results['stability']['range'] * 100]  # Scale for visibility
labels = [f"{results['stability']['variance']:.6f}", f"{results['stability']['range']:.4f}\n(±{results['stability']['range']/2:.2%})"]

bars = ax2.bar(metrics, values, color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=1.2, width=0.5)

for bar, label in zip(bars, labels):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            label, ha='center', va='center', fontsize=10, weight='bold', color='white')

ax2.set_ylabel('Scaled Values')
ax2.set_title('Overall Stability Metrics')
ax2.set_ylim([0, max(values) * 1.2])
ax2.text(0.5, max(values) * 1.05, '⚠️ Moderate Instability',
         ha='center', fontsize=11, weight='bold', color='orange',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'stability_metrics.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'stability_metrics.png'}")
plt.close()

print(f"\n✅ All figures saved to: {output_dir}/")
print(f"\nFigures created:")
print(f"  1. training_length_distribution.png - Shows SUMMARIZATION is 100% long")
print(f"  2. stratified_accuracy.png - Shows 'Long' bucket has highest accuracy (98.2%)")
print(f"  3. per_intent_stratified.png - Shows per-intent instability")
print(f"  4. training_distribution_heatmap.png - Heatmap of training imbalance")
print(f"  5. stability_metrics.png - Quantifies instability (6.1% range)")
