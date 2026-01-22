#!/usr/bin/env python3
"""
Create a simple quality vs cost visualization to explain the model selection.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Model data
models = {
    'GPT-5': {'cost': 15000, 'quality': 0.984, 'color': '#27ae60', 'marker': 'o', 'size': 200},
    'GPT-4o': {'cost': 10000, 'quality': 0.968, 'color': '#3498db', 'marker': 's', 'size': 150},
    'Mixtral': {'cost': 500, 'quality': 0.600, 'color': '#95a5a6', 'marker': '^', 'size': 120}
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Quality vs Cost
for name, data in models.items():
    ax1.scatter(data['cost'], data['quality'], 
               s=data['size'], c=data['color'], 
               marker=data['marker'], edgecolors='black', linewidth=2,
               label=name, alpha=0.8, zorder=3)
    
    # Annotate
    offset_x = 500 if name == 'GPT-5' else -500
    offset_y = 0.01 if name != 'Mixtral' else -0.02
    ax1.annotate(name, xy=(data['cost'], data['quality']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Pareto frontier
ax1.plot([500, 10000, 15000], [0.600, 0.968, 0.984], 
         'k--', alpha=0.3, linewidth=1, label='Pareto Frontier', zorder=1)

ax1.set_xlabel('Cost ($ per 1M tokens)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Quality (Win Rate)', fontsize=12, fontweight='bold')
ax1.set_title('Quality vs Cost Trade-off', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='lower right')
ax1.set_ylim(0.55, 1.0)

# Annotate "Frontier Model"
ax1.annotate('', xy=(15000, 0.984), xytext=(15000, 1.05),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
ax1.text(15000, 1.06, 'Frontier\nModel', ha='center', fontsize=9, 
        color='#27ae60', fontweight='bold')

# Plot 2: Selection Distribution
selection_data = {
    'Cold Start': {'GPT-4o': 500, 'GPT-5': 0, 'Mixtral': 0},
    'LST': {'GPT-4o': 0, 'GPT-5': 500, 'Mixtral': 0}
}

x = np.arange(2)
width = 0.25

for i, model in enumerate(['GPT-4o', 'Mixtral', 'GPT-5']):
    counts = [selection_data['Cold Start'][model], selection_data['LST'][model]]
    color = models[model]['color']
    ax2.bar(x + (i-1)*width, counts, width, label=model, color=color, 
           alpha=0.7, edgecolor='black')

ax2.set_xlabel('Initialization Strategy', fontsize=12, fontweight='bold')
ax2.set_ylabel('Selection Count (out of 500)', fontsize=12, fontweight='bold')
ax2.set_title('Model Selection Distribution', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Cold Start\n(No Transfer)', 'LST\n(Semantic Transfer)'])
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 550)

# Annotate insights
ax2.text(0, 520, '❌ Stuck on\nSuboptimal', ha='center', fontsize=9, 
        color='#e74c3c', fontweight='bold')
ax2.text(1, 520, '✅ Frontier\nExploitation', ha='center', fontsize=9, 
        color='#27ae60', fontweight='bold')

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "results" / "quality_cost_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

plt.show()

