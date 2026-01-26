"""
Generate Figure 2: Corralled Architecture Diagram
Shows the coordinator-expert hierarchy with information flows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
color_coordinator = '#3498db'  # Blue
color_warmup = '#2ecc71'       # Green
color_tabula = '#e67e22'       # Orange
color_execution = '#95a5a6'    # Gray
color_feedback = '#34495e'     # Dark gray

# Helper function to create fancy boxes
def create_box(ax, x, y, width, height, color, alpha=0.15, linewidth=2):
    """Create a rounded rectangle box"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=linewidth
    )
    ax.add_patch(box)
    return box

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, style='solid', color='black', width=2, label='', label_pos=0.5):
    """Create an arrow with optional label"""
    if style == 'dashed':
        linestyle = '--'
    else:
        linestyle = '-'
    
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=color,
        linewidth=width,
        linestyle=linestyle,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        label_x = x1 + (x2 - x1) * label_pos
        label_y = y1 + (y2 - y1) * label_pos
        ax.text(label_x, label_y, label, fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    
    return arrow

# ============================================================================
# COORDINATOR LAYER (Top)
# ============================================================================
coord_x, coord_y = 2, 8
coord_width, coord_height = 10, 1.5

create_box(ax, coord_x, coord_y, coord_width, coord_height, color_coordinator, alpha=0.2, linewidth=3)

# Coordinator title
ax.text(7, 9.2, 'Coordinator Layer', fontsize=14, weight='bold', ha='center', color=color_coordinator)

# Coordinator state
ax.text(7, 8.7, r'Trust Distribution: $\pi = [0.72, 0.28]$', fontsize=11, ha='center', family='monospace')
ax.text(7, 8.4, r'Cumulative Losses: $L = [45.2, 89.7]$', fontsize=11, ha='center', family='monospace')
ax.text(7, 8.1, r'Learning Rate: $\eta = 0.1$', fontsize=11, ha='center', family='monospace')

# ============================================================================
# EXPERT LAYER (Middle)
# ============================================================================

# Warmup Expert (Left)
warmup_x, warmup_y = 1.5, 4.5
warmup_width, warmup_height = 4.5, 3

create_box(ax, warmup_x, warmup_y, warmup_width, warmup_height, color_warmup, alpha=0.15, linewidth=2.5)

ax.text(3.75, 7.2, 'Warmup Expert', fontsize=13, weight='bold', ha='center', color=color_warmup)
ax.text(3.75, 6.7, 'Initialization:', fontsize=10, ha='center', style='italic')
ax.text(3.75, 6.35, r'$A_0 = \lambda I + \sum_{i=1}^{N} \phi(x_i)\phi(x_i)^T$', fontsize=9, ha='center')
ax.text(3.75, 6.0, r'$b_0 = \sum_{i=1}^{N} r_i \phi(x_i)$', fontsize=9, ha='center')
ax.text(3.75, 5.6, r'Prior: 80K RouteLLM battles', fontsize=9, ha='center', color='#27ae60')

# Warmup state box
state_box = FancyBboxPatch(
    (2, 4.8), 3.5, 0.6,
    boxstyle="round,pad=0.05",
    edgecolor=color_warmup,
    facecolor='white',
    linewidth=1.5
)
ax.add_patch(state_box)
ax.text(3.75, 5.25, r'Samples: $n = 720$', fontsize=10, ha='center', weight='bold')
ax.text(3.75, 4.95, r'Recommendation: GPT-4', fontsize=9, ha='center')

# Tabula Rasa Expert (Right)
tabula_x, tabula_y = 8, 4.5
tabula_width, tabula_height = 4.5, 3

create_box(ax, tabula_x, tabula_y, tabula_width, tabula_height, color_tabula, alpha=0.15, linewidth=2.5)

ax.text(10.25, 7.2, 'Tabula Rasa Expert', fontsize=13, weight='bold', ha='center', color=color_tabula)
ax.text(10.25, 6.7, 'Initialization:', fontsize=10, ha='center', style='italic')
ax.text(10.25, 6.35, r'$A_0 = \lambda I$', fontsize=9, ha='center')
ax.text(10.25, 6.0, r'$b_0 = 0$', fontsize=9, ha='center')
ax.text(10.25, 5.6, r'No priors (pure online learning)', fontsize=9, ha='center', color='#d35400')

# Tabula Rasa state box
state_box2 = FancyBboxPatch(
    (8.5, 4.8), 3.5, 0.6,
    boxstyle="round,pad=0.05",
    edgecolor=color_tabula,
    facecolor='white',
    linewidth=1.5
)
ax.add_patch(state_box2)
ax.text(10.25, 5.25, r'Samples: $n = 280$', fontsize=10, ha='center', weight='bold')
ax.text(10.25, 4.95, r'Recommendation: Claude', fontsize=9, ha='center')

# ============================================================================
# EXECUTION LAYER (Middle-Bottom)
# ============================================================================
exec_x, exec_y = 4.5, 2.8
exec_width, exec_height = 5, 1

create_box(ax, exec_x, exec_y, exec_width, exec_height, color_execution, alpha=0.2, linewidth=2)

ax.text(7, 3.5, 'Selected Action', fontsize=12, weight='bold', ha='center')
ax.text(7, 3.15, r'Model: GPT-4  |  Reward: $r = 0.92$', fontsize=10, ha='center', family='monospace')

# ============================================================================
# FEEDBACK LAYER (Bottom)
# ============================================================================
feedback_x, feedback_y = 3.5, 0.8
feedback_width, feedback_height = 7, 1.5

create_box(ax, feedback_x, feedback_y, feedback_width, feedback_height, color_feedback, alpha=0.15, linewidth=2)

ax.text(7, 2.0, 'Feedback Phase', fontsize=12, weight='bold', ha='center', color=color_feedback)
ax.text(7, 1.6, r'Loss: $\ell = \frac{1 - r}{\pi_i} = \frac{1 - 0.92}{0.72} = 0.111$', fontsize=10, ha='center')
ax.text(7, 1.25, r'Update: $L[i] \leftarrow L[i] + \ell$,  $\pi \leftarrow \text{normalize}(\exp(-\eta L))$', fontsize=10, ha='center')
ax.text(7, 0.95, r'Expert Update: $A \leftarrow A + \phi(x,a)\phi(x,a)^T$,  $b \leftarrow b + r\phi(x,a)$', fontsize=9, ha='center')

# ============================================================================
# ARROWS - Information Flow
# ============================================================================

# Selection arrows (Coordinator -> Experts)
create_arrow(ax, 4.5, 8.0, 3.75, 7.5, style='dashed', color=color_coordinator, width=2.5, 
             label=r'$p = 0.72$', label_pos=0.4)
create_arrow(ax, 9.5, 8.0, 10.25, 7.5, style='dashed', color=color_coordinator, width=2.5,
             label=r'$p = 0.28$', label_pos=0.4)

# Recommendation arrows (Experts -> Execution)
create_arrow(ax, 3.75, 4.5, 5.5, 3.8, style='solid', color='#7f8c8d', width=2,
             label='UCB=0.85', label_pos=0.6)
create_arrow(ax, 10.25, 4.5, 8.5, 3.8, style='solid', color='#7f8c8d', width=2,
             label='UCB=0.78', label_pos=0.6)

# Feedback arrow (Execution -> Feedback)
create_arrow(ax, 7, 2.8, 7, 2.3, style='solid', color='black', width=2.5)

# Feedback to Coordinator (curved)
from matplotlib.patches import ConnectionPatch
feedback_coord = ConnectionPatch(
    (7, 2.3), (7, 8.0),
    "data", "data",
    arrowstyle='->,head_width=0.4,head_length=0.4',
    color=color_feedback,
    linewidth=3,
    connectionstyle="arc3,rad=0.3"
)
ax.add_artist(feedback_coord)
ax.text(9.5, 5.2, 'Update\nWeights', fontsize=9, ha='center', color=color_feedback, weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_feedback, linewidth=1.5))

# Feedback to Warmup Expert (curved)
feedback_warmup = ConnectionPatch(
    (5.5, 1.5), (3.75, 4.5),
    "data", "data",
    arrowstyle='->,head_width=0.4,head_length=0.4',
    color=color_feedback,
    linewidth=2.5,
    connectionstyle="arc3,rad=-0.2"
)
ax.add_artist(feedback_warmup)
ax.text(4.5, 2.8, 'Update\nExpert', fontsize=8, ha='center', color=color_feedback,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# ============================================================================
# LEGEND
# ============================================================================
legend_x, legend_y = 11.5, 8.5
legend_width, legend_height = 2, 1.2

# Legend box
legend_box = FancyBboxPatch(
    (legend_x, legend_y), legend_width, legend_height,
    boxstyle="round,pad=0.1",
    edgecolor='black',
    facecolor='white',
    linewidth=1.5
)
ax.add_patch(legend_box)

ax.text(12.5, 9.5, 'Information Flow', fontsize=9, weight='bold', ha='center')

# Legend items
ax.plot([11.7, 12.1], [9.2, 9.2], '--', color=color_coordinator, linewidth=2)
ax.text(12.3, 9.2, 'Selection', fontsize=8, va='center')

ax.plot([11.7, 12.1], [8.95, 8.95], '-', color='#7f8c8d', linewidth=2)
ax.text(12.3, 8.95, 'Recommend', fontsize=8, va='center')

ax.plot([11.7, 12.1], [8.7, 8.7], '-', color=color_feedback, linewidth=3)
ax.text(12.3, 8.7, 'Feedback', fontsize=8, va='center')

# ============================================================================
# ANNOTATIONS
# ============================================================================

# Phase labels on the left
ax.text(0.3, 8.7, 'Phase 1:\nSelection', fontsize=9, weight='bold', ha='left', va='center',
        color=color_coordinator)
ax.text(0.3, 6.0, 'Phase 2:\nRecommend', fontsize=9, weight='bold', ha='left', va='center',
        color='#7f8c8d')
ax.text(0.3, 3.3, 'Phase 3:\nExecution', fontsize=9, weight='bold', ha='left', va='center',
        color=color_execution)
ax.text(0.3, 1.5, 'Phase 4:\nFeedback', fontsize=9, weight='bold', ha='left', va='center',
        color=color_feedback)

# Title
fig.suptitle('Figure 2: Corralled Architecture - Coordinator-Expert Hierarchy', 
             fontsize=16, weight='bold', y=0.98)

# Subtitle
ax.text(7, 9.8, 
        'Meta-learning system that dynamically balances warmup priors and online adaptation',
        fontsize=11, ha='center', style='italic', color='#555555')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_path = '/Users/annette/repostitories/banditGPT/experiments_v1/02_figure/results/figure2_corralled_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved figure to: {output_path}")

# Also save as PDF for paper
output_path_pdf = '/Users/annette/repostitories/banditGPT/experiments_v1/02_figure/results/figure2_corralled_architecture.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ Saved PDF to: {output_path_pdf}")

# Also save to paper figures directory
paper_output_path = '/Users/annette/repostitories/banditGPT/paper/figures/figure2_corralled_architecture.png'
plt.savefig(paper_output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved to paper figures: {paper_output_path}")

paper_output_pdf = '/Users/annette/repostitories/banditGPT/paper/figures/figure2_corralled_architecture.pdf'
plt.savefig(paper_output_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ Saved PDF to paper figures: {paper_output_pdf}")

plt.show()

print("\n" + "="*60)
print("Figure 2 Generation Complete!")
print("="*60)
print("\nKey Features:")
print("  • Three-layer hierarchy (Coordinator, Experts, Execution)")
print("  • Color-coded components (Blue=Coordinator, Green=Warmup, Orange=Tabula Rasa)")
print("  • Information flow arrows with labels")
print("  • Mathematical formulas for initialization and updates")
print("  • Realistic example values from experiments")
print("  • Professional publication-ready quality (300 DPI)")
print("\nOutputs:")
print(f"  1. PNG (300 DPI): {output_path}")
print(f"  2. PDF (vector): {output_path_pdf}")
print(f"  3. Paper copy: {paper_output_path}")
print("="*60)

