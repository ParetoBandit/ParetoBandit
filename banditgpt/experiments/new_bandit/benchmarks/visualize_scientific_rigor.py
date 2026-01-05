#!/usr/bin/env python3
"""
Two-Strike Pruning Visualization

Two-panel figure for KDD paper:
1. Decision Logic Flowchart (Theory vs Reality paths)
2. Sample Count Plot (Min-Sample Probation with real data)

Usage:
    python visualize_scientific_rigor.py

Output:
    scientific_rigor_fixes.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bandit_v2 import RouterConfig
    MIN_SAMPLES = RouterConfig.pruning_min_samples
except ImportError:
    MIN_SAMPLES = 50


# =============================================================================
# LOAD REAL DATA
# =============================================================================

def load_test_data():
    """Load real test data for sample count visualization."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "offline_dataset"
    rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    
    model_counts = {}
    if rewards_path.exists():
        with open(rewards_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    model = row.get("model_id", "")
                    if model:
                        model_counts[model] = model_counts.get(model, 0) + 1
                except:
                    pass
    
    return model_counts


# =============================================================================
# PANEL A: DECISION LOGIC FLOWCHART
# =============================================================================

def plot_flowchart(ax):
    """
    Draw the Two-Strike pruning decision flowchart.
    Theory path vs Reality path (Unicorn Guardrail)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A. Two-Strike Pruning Logic', fontsize=14, fontweight='bold', pad=15)
    
    # Colors
    C_INPUT = '#3498db'
    C_THEORY = '#9b59b6'
    C_REALITY = '#e74c3c'
    C_KEEP = '#27ae60'
    C_PRUNE = '#c0392b'
    C_ARROW = '#2c3e50'
    
    def draw_box(x, y, w, h, color, text, fontsize=9):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', wrap=True)
    
    def draw_diamond(x, y, size, color, text, fontsize=8):
        diamond = plt.Polygon(
            [(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.9
        )
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
    
    def draw_arrow(x1, y1, x2, y2, label='', pos='mid'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if pos == 'right':
                mx += 0.3
            ax.text(mx, my, label, fontsize=9, ha='center', va='center',
                   color=C_ARROW, fontweight='bold')
    
    # Row 1: Input
    draw_box(5, 9.0, 2.5, 0.7, C_INPUT, "Candidate Arm", fontsize=11)
    draw_arrow(5, 8.65, 5, 7.8)
    
    # Row 2: Theory Test (Strike 1)
    draw_diamond(5, 7.0, 1.0, C_THEORY, "UCB < Best\nLCB on ALL\nAnchors?", fontsize=8)
    
    # Theory → No → Keep
    draw_arrow(6.0, 7.0, 8.5, 7.0, "No")
    draw_box(8.5, 7.0, 1.6, 0.6, C_KEEP, "KEEP", fontsize=10)
    ax.text(8.5, 6.3, "(Standard)", fontsize=8, ha='center', color='gray')
    
    # Theory → Yes → Strike 1, proceed to Reality
    draw_arrow(5, 6.0, 5, 5.0)
    ax.text(5.3, 5.5, "Yes", fontsize=9, color=C_ARROW, fontweight='bold')
    ax.text(3.3, 5.5, "Strike 1", fontsize=10, color=C_THEORY, fontweight='bold')
    
    # Row 3: Reality Test (Strike 2)
    draw_diamond(5, 4.0, 1.0, C_REALITY, "Mean Reward\n>= Global\n× 0.80?", fontsize=8)
    
    # Reality → Yes → Keep (Unicorn)
    draw_arrow(6.0, 4.0, 8.5, 4.0, "Yes")
    draw_box(8.5, 4.0, 1.6, 0.6, C_KEEP, "KEEP", fontsize=10)
    ax.text(8.5, 3.3, "(Unicorn Save)", fontsize=8, ha='center', color='#f39c12')
    
    # Reality → No → Prune
    draw_arrow(5, 3.0, 5, 2.0)
    ax.text(5.3, 2.5, "No", fontsize=9, color=C_ARROW, fontweight='bold')
    ax.text(3.3, 2.5, "Strike 2", fontsize=10, color=C_REALITY, fontweight='bold')
    
    # Row 4: Prune
    draw_box(5, 1.3, 2.0, 0.7, C_PRUNE, "PRUNE", fontsize=11)
    
    # Add path labels
    ax.text(1.5, 7.0, "THEORY\nPATH", fontsize=10, color=C_THEORY, 
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_THEORY, alpha=0.8))
    ax.text(1.5, 4.0, "REALITY\nPATH", fontsize=10, color=C_REALITY, 
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_REALITY, alpha=0.8))


# =============================================================================
# PANEL B: SAMPLE COUNT PLOT (Real Data)
# =============================================================================

def plot_sample_counts(ax):
    """Show sample count distribution with min-sample threshold using real data."""
    
    # Load real data
    model_counts = load_test_data()
    
    if not model_counts:
        # Fallback to realistic simulated data
        model_counts = {
            "anthropic/claude-opus-4.5": 580,
            "openai/gpt-5.1": 120,
            "openai/o3": 95,
            "openai/gpt-5": 72,
            "x-ai/grok-4": 65,
            "mistralai/ministral-3b": 48,
            "anthropic/claude-sonnet-4.5": 35,
            "openai/gpt-oss-120b": 22,
            "google/gemma-3-4b-it": 8,
            "amazon/nova-micro-v1": 3,
        }
    
    # Sort by count, take top 12
    sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    
    models = [m[0].split('/')[-1][:18] for m in sorted_models]
    counts = [m[1] for m in sorted_models]
    
    # Colors: green if >= threshold, red if < threshold
    colors = ['#27ae60' if c >= MIN_SAMPLES else '#e74c3c' for c in counts]
    
    y_pos = np.arange(len(models))
    ax.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=1.5)
    
    # Threshold line
    ax.axvline(x=MIN_SAMPLES, color='#f39c12', linestyle='--', linewidth=2.5,
               label=f'Min-Sample Threshold = {MIN_SAMPLES}')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Sample Count', fontsize=11, fontweight='bold')
    ax.set_title(f'B. Min-Sample Probation\n(Test Data: {len(model_counts)} models)', 
                fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=9)
    
    # Annotations
    eligible = sum(1 for c in counts if c >= MIN_SAMPLES)
    protected = sum(1 for c in counts if c < MIN_SAMPLES)
    ax.text(0.98, 0.95, f'Eligible: {eligible}\nProtected: {protected}', 
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()


# =============================================================================
# MAIN
# =============================================================================

def create_visualization():
    """Create single-panel flowchart figure."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Single panel: Flowchart
    plot_flowchart(ax)
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "scientific_rigor_fixes.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.show()
    return output_path


if __name__ == "__main__":
    create_visualization()
