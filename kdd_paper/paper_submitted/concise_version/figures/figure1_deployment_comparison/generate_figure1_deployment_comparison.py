#!/usr/bin/env python3
"""
Generate Figure 1: Deployment Workflow Comparison
Shows the "Heavy Router" vs. "Agile Router" workflows side-by-side.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Create figure
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 8))

# Colors
color_heavy = '#E74C3C'  # Red (traditional)
color_agile = '#27AE60'  # Green (BanditGPT)
color_neutral = '#95A5A6'  # Gray

def draw_workflow_box(ax, y_pos, text, color, time_label=None):
    """Draw a single workflow step box."""
    box = FancyBboxPatch(
        (0.15, y_pos), 0.7, 0.12,
        boxstyle="round,pad=0.01",
        edgecolor=color, facecolor=color, alpha=0.2,
        linewidth=2.5
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(0.5, y_pos + 0.06, text, 
            ha='center', va='center', fontsize=13, fontweight='bold')
    
    # Add time label if provided
    if time_label:
        ax.text(0.88, y_pos + 0.06, time_label,
                ha='left', va='center', fontsize=10, style='italic',
                color=color)

def draw_arrow(ax, y_start, y_end):
    """Draw a downward arrow between boxes."""
    arrow = FancyArrowPatch(
        (0.5, y_start), (0.5, y_end),
        arrowstyle='->', mutation_scale=30, linewidth=2.5,
        color='#34495E', alpha=0.7
    )
    ax.add_patch(arrow)

# ============================================================================
# LEFT PANEL: Traditional Routers (Heavy)
# ============================================================================
ax_left.set_xlim(0, 1)
ax_left.set_ylim(0, 1)
ax_left.axis('off')

# Title
ax_left.text(0.5, 0.96, 'Traditional Routers', 
             ha='center', va='top', fontsize=16, fontweight='bold',
             color=color_heavy)
ax_left.text(0.5, 0.91, '(FrugalGPT, RouteLLM)', 
             ha='center', va='top', fontsize=11, style='italic',
             color=color_neutral)

# Workflow steps
y_positions = [0.75, 0.58, 0.41, 0.24, 0.07]

draw_workflow_box(ax_left, y_positions[0], 'New Model\nReleased', color_neutral)
draw_arrow(ax_left, y_positions[0], y_positions[1] + 0.12)

draw_workflow_box(ax_left, y_positions[1], 'Collect\nDataset', color_heavy, '⏱️ Days')
draw_arrow(ax_left, y_positions[1], y_positions[2] + 0.12)

draw_workflow_box(ax_left, y_positions[2], 'Benchmark\nModels', color_heavy, '⏱️ Hours')
draw_arrow(ax_left, y_positions[2], y_positions[3] + 0.12)

draw_workflow_box(ax_left, y_positions[3], 'Train\nRouter', color_heavy, '⏱️ Hours')
draw_arrow(ax_left, y_positions[3], y_positions[4] + 0.12)

draw_workflow_box(ax_left, y_positions[4], 'Deploy\n(Static)', color_heavy)

# Summary stats box
summary_box = FancyBboxPatch(
    (0.05, -0.08), 0.9, 0.12,
    boxstyle="round,pad=0.01",
    edgecolor=color_heavy, facecolor='white', alpha=0.9,
    linewidth=2
)
ax_left.add_patch(summary_box)

ax_left.text(0.5, 0.01, '⚠️  Manual, Slow, Static', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=color_heavy)
ax_left.text(0.5, -0.04, 'Time: Days-Weeks | Data: 500-5k examples',
             ha='center', va='center', fontsize=10, color=color_neutral)

# ============================================================================
# RIGHT PANEL: BanditGPT (Agile)
# ============================================================================
ax_right.set_xlim(0, 1)
ax_right.set_ylim(0, 1)
ax_right.axis('off')

# Title
ax_right.text(0.5, 0.96, 'BanditGPT', 
              ha='center', va='top', fontsize=16, fontweight='bold',
              color=color_agile)
ax_right.text(0.5, 0.91, '(This Work)', 
              ha='center', va='top', fontsize=11, style='italic',
              color=color_neutral)

# Simplified workflow
y_positions_agile = [0.75, 0.58, 0.24]

draw_workflow_box(ax_right, y_positions_agile[0], 'New Model\nReleased', color_neutral)
draw_arrow(ax_right, y_positions_agile[0], y_positions_agile[1] + 0.12)

draw_workflow_box(ax_right, y_positions_agile[1], 'Update\nMetadata', color_agile, '⏱️ Minutes')
draw_arrow(ax_right, y_positions_agile[1], y_positions_agile[2] + 0.12)

# Merged "Deploy + Learn" box (taller)
merged_box = FancyBboxPatch(
    (0.15, 0.07), 0.7, 0.25,
    boxstyle="round,pad=0.01",
    edgecolor=color_agile, facecolor=color_agile, alpha=0.2,
    linewidth=2.5
)
ax_right.add_patch(merged_box)

ax_right.text(0.5, 0.235, 'Deploy', 
              ha='center', va='center', fontsize=13, fontweight='bold')
ax_right.text(0.5, 0.19, '↓', 
              ha='center', va='center', fontsize=16)
ax_right.text(0.5, 0.145, 'Learn Online', 
              ha='center', va='center', fontsize=13, fontweight='bold')
ax_right.text(0.5, 0.095, '(Continuous Adaptation)', 
              ha='center', va='center', fontsize=9, style='italic',
              color=color_neutral)

ax_right.text(0.88, 0.19, '⏱️ Immediate',
              ha='left', va='center', fontsize=10, style='italic',
              color=color_agile)

# Summary stats box
summary_box_agile = FancyBboxPatch(
    (0.05, -0.08), 0.9, 0.12,
    boxstyle="round,pad=0.01",
    edgecolor=color_agile, facecolor='white', alpha=0.9,
    linewidth=2
)
ax_right.add_patch(summary_box_agile)

ax_right.text(0.5, 0.01, '✅  Automatic, Fast, Adaptive', 
              ha='center', va='center', fontsize=12, fontweight='bold',
              color=color_agile)
ax_right.text(0.5, -0.04, 'Time: Minutes | Data: 0 examples',
              ha='center', va='center', fontsize=10, color=color_neutral)

# ============================================================================
# Final adjustments
# ============================================================================
plt.tight_layout(pad=2.0)

# Save
output_path = 'figure1_deployment_comparison.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✅ Saved: {output_path}")

output_png = 'figure1_deployment_comparison.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')
print(f"✅ Saved: {output_png}")

plt.close()

print("\n📊 Figure 1 Generation Complete")
print("=" * 60)
print("Purpose: Visual teaser showing deployment workflow comparison")
print("Message: BanditGPT eliminates the calibration bottleneck")
print("Placement: Page 1 of paper (after abstract)")
print("=" * 60)

