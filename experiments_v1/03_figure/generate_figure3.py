"""
Figure 3: Corralled Architecture (KDD 2026)
Professional academic diagram - clean vertical flow with clear feedback path.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================================
# STYLE
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

# Academic color palette
C = {
    'blue': '#4A90A4',
    'green': '#5B8C5A', 
    'orange': '#C17F59',
    'gray': '#6E6E6E',
    'dark': '#2C3E50',
    'light': '#F5F5F5',
    'red': '#B85450',
}

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(-0.8, 10.2)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rounded_box(x, y, w, h, color, fill='white', lw=2):
    rect = FancyBboxPatch((x, y), w, h, 
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=fill, edgecolor=color, linewidth=lw, zorder=2)
    ax.add_patch(rect)

def simple_arrow(x1, y1, x2, y2, color=C['dark'], style='-', lw=1.5):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='-|>,head_width=0.1,head_length=0.1',
                          color=color, linewidth=lw, linestyle=style,
                          shrinkA=6, shrinkB=6, mutation_scale=10, zorder=3)
    ax.add_patch(arr)

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Coordinator (top center)
rounded_box(3.0, 5.0, 4, 0.85, C['blue'])
ax.text(5.0, 5.42, "Coordinator", ha='center', va='center', 
        fontsize=12, fontweight='bold', color=C['blue'])

# Expert 1 (left) - More compact box
rounded_box(0.8, 2.7, 3.0, 1.45, C['green'])
ax.text(2.3, 3.8, "Expert 1: Warmup", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['green'])
ax.text(2.3, 3.4, "LinUCB with Priors", ha='center', fontsize=10, 
        style='italic', color=C['gray'])
ax.text(2.3, 3.05, r"$\alpha = 2.0$", ha='center', fontsize=11, color=C['red'])
ax.text(2.3, 2.8, "(constant exploration)", ha='center', fontsize=8, color=C['gray'])

# Expert 2 (right) - More compact box
rounded_box(6.2, 2.7, 3.0, 1.45, C['orange'])
ax.text(7.7, 3.8, "Expert 2: Tabula Rasa", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['orange'])
ax.text(7.7, 3.4, "LinUCB (no priors)", ha='center', fontsize=10,
        style='italic', color=C['gray'])
ax.text(7.7, 3.05, r"$\alpha = 2.0$", ha='center', fontsize=11, color=C['red'])
ax.text(7.7, 2.8, "(constant exploration)", ha='center', fontsize=8, color=C['gray'])

# Model Selection (center)
rounded_box(3.0, 1.0, 4, 0.85, C['gray'])
ax.text(5.0, 1.42, "Model Selection", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['gray'])

# Feedback (bottom center)
rounded_box(3.0, -0.2, 4, 0.85, C['dark'])
ax.text(5.0, 0.22, "Feedback", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['dark'])

# ============================================================================
# ARROWS
# ============================================================================

# Coordinator -> Expert 1
simple_arrow(4.0, 5.0, 2.3, 4.15, C['blue'])
ax.text(2.9, 4.7, r"$P_t(1)$", fontsize=11, color=C['blue'])

# Coordinator -> Expert 2
simple_arrow(6.0, 5.0, 7.7, 4.15, C['blue'])
ax.text(6.9, 4.7, r"$P_t(2)$", fontsize=11, color=C['blue'])

# Expert 1 -> Model Selection
simple_arrow(2.3, 2.7, 4.0, 1.85, C['gray'], style='--')

# Expert 2 -> Model Selection  
simple_arrow(7.7, 2.7, 6.0, 1.85, C['gray'], style='--')

# Model Selection -> Feedback
simple_arrow(5.0, 1.0, 5.0, 0.65, C['dark'])

# Feedback -> Coordinator (move line further left to avoid text overlap)
feedback_x = -0.1
ax.plot([3.0, feedback_x, feedback_x, 3.0], [0.22, 0.22, 5.42, 5.42], 
        color=C['blue'], linestyle=':', linewidth=1.5, zorder=1)
ax.annotate('', xy=(3.0, 5.42), xytext=(0.3, 5.42),
            arrowprops=dict(arrowstyle='-|>,head_width=0.08,head_length=0.08',
                           color=C['blue'], lw=1.5))

# Label for feedback path (positioned to the right of the line)
ax.text(0.05, 1.8, "Update\nWeights", fontsize=10, color=C['blue'], 
        ha='left', va='center')

# ============================================================================
# ANNOTATIONS
# ============================================================================

# Formula under Coordinator
ax.text(5.0, 4.7, r"$P_t = (1-\gamma)\,w_t + \gamma/K$", 
        ha='center', fontsize=12, color=C['dark'])

# Formula under Feedback
ax.text(5.0, -0.55, r"$\hat{\ell}_t = \frac{1-r_t}{P_t}$", 
        ha='center', fontsize=12, color=C['dark'])

# Parameters box (top right, compact)
rounded_box(8.0, 5.0, 1.8, 1.15, C['gray'], fill=C['light'], lw=1)
ax.text(8.9, 5.85, "Parameters", fontsize=9, fontweight='bold', 
        ha='center', color=C['dark'])
ax.text(8.9, 5.58, r"$\alpha = 2.0$", fontsize=9, ha='center', color=C['red'])
ax.text(8.9, 5.35, r"$\eta = 1.0$", fontsize=9, ha='center', color=C['dark'])
ax.text(8.9, 5.12, r"$\gamma = 0.05$", fontsize=9, ha='center', color=C['dark'])

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout()
out = '/Users/annette/repostitories/banditGPT/experiments_v1/03_figure/results/figure3_corralled_architecture'
plt.savefig(f'{out}.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}.pdf', bbox_inches='tight', facecolor='white')
print(f"✓ Saved to {out}.png and .pdf")
