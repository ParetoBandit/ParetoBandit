#!/usr/bin/env python3
"""
THE TRAP DIAGRAM: Visualizing the "Confident Failure" Hypothesis

This diagram contrasts:
- FrugalGPT's failure mode (ex-post verification fails)
- BanditGPT's success mode (ex-ante prediction succeeds)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path


def create_trap_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    RED = '#D62728'
    GREEN = '#2CA02C'
    ORANGE = '#FF7F0E'
    BLUE = '#17BECF'
    GRAY = '#7F7F7F'
    
    # Title
    ax.text(7, 7.5, 'The "Confident Failure" Trap', fontsize=20, fontweight='bold',
            ha='center', va='center')
    ax.text(7, 7.0, 'Why Ex-Ante Prediction Beats Ex-Post Verification', fontsize=14,
            ha='center', va='center', style='italic', color=GRAY)
    
    # =========================================================================
    # TOP PATH: FrugalGPT (The Trap)
    # =========================================================================
    y_top = 5.0
    
    # Label
    ax.text(0.5, y_top + 0.8, 'FrugalGPT (Cascade)', fontsize=14, fontweight='bold',
            color=RED, va='center')
    ax.text(0.5, y_top + 0.4, '"The Trap"', fontsize=12, style='italic',
            color=RED, va='center')
    
    # Boxes
    boxes_top = [
        (1.5, y_top, 'Complex\nInstruction', GRAY),
        (4.0, y_top, 'DeepSeek\n(Cheap)', ORANGE),
        (6.5, y_top, 'Wrong Output\n(Looks Good!)', ORANGE),
        (9.0, y_top, 'Verifier\n(Fooled!)', RED),
        (11.5, y_top, 'FAILURE', RED),
    ]
    
    for x, y, text, color in boxes_top:
        box = FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, 
                fontweight='bold' if 'FAILURE' in text else 'normal',
                color=color if 'FAILURE' in text else 'black')
    
    # Arrows for top path
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    for i in range(len(boxes_top) - 1):
        x1 = boxes_top[i][0] + 0.9
        x2 = boxes_top[i+1][0] - 0.9
        arrow = FancyArrowPatch((x1, y_top), (x2, y_top),
                                arrowstyle=arrow_style, color=RED, 
                                mutation_scale=1, linewidth=1.5)
        ax.add_patch(arrow)
    
    # Add X mark on FAILURE
    ax.text(11.5, y_top - 0.8, '✗', fontsize=30, color=RED, ha='center', va='center')
    
    # =========================================================================
    # BOTTOM PATH: BanditGPT (The Dodge)
    # =========================================================================
    y_bot = 2.0
    
    # Label
    ax.text(0.5, y_bot + 0.8, 'BanditGPT (Hybrid)', fontsize=14, fontweight='bold',
            color=GREEN, va='center')
    ax.text(0.5, y_bot + 0.4, '"The Dodge"', fontsize=12, style='italic',
            color=GREEN, va='center')
    
    # Boxes
    boxes_bot = [
        (1.5, y_bot, 'Complex\nInstruction', GRAY),
        (4.0, y_bot, 'Bandit\n(Detects Trap!)', BLUE),
        (6.5, y_bot, 'SKIP\nDeepSeek', GREEN),
        (9.0, y_bot, 'GPT-4o\n(Teacher)', BLUE),
        (11.5, y_bot, 'SUCCESS', GREEN),
    ]
    
    for x, y, text, color in boxes_bot:
        box = FancyBboxPatch((x - 0.9, y - 0.5), 1.8, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold' if 'SUCCESS' in text else 'normal',
                color=color if 'SUCCESS' in text else 'black')
    
    # Arrows for bottom path
    for i in range(len(boxes_bot) - 1):
        x1 = boxes_bot[i][0] + 0.9
        x2 = boxes_bot[i+1][0] - 0.9
        arrow = FancyArrowPatch((x1, y_bot), (x2, y_bot),
                                arrowstyle=arrow_style, color=GREEN,
                                mutation_scale=1, linewidth=1.5)
        ax.add_patch(arrow)
    
    # Add checkmark on SUCCESS
    ax.text(11.5, y_bot - 0.8, '✓', fontsize=30, color=GREEN, ha='center', va='center')
    
    # =========================================================================
    # KEY INSIGHT BOX
    # =========================================================================
    insight_box = FancyBboxPatch((3.5, 0.3), 7, 1.2,
                                 boxstyle="round,pad=0.1,rounding_size=0.3",
                                 facecolor='#F0F8FF', edgecolor=BLUE, linewidth=2)
    ax.add_patch(insight_box)
    
    ax.text(7, 1.1, 'KEY INSIGHT', fontsize=12, fontweight='bold',
            ha='center', va='center', color=BLUE)
    ax.text(7, 0.6, 'Prediction (Bandit) is safer than Verification (Cascade)\n'
                   'when "checking the work" is as hard as "doing the work"',
            fontsize=11, ha='center', va='center', style='italic')
    
    # =========================================================================
    # ANNOTATIONS
    # =========================================================================
    # Annotation on "Looks Good"
    ax.annotate('DeepSeek output\nviolates 1 constraint\nbut looks plausible',
                xy=(6.5, y_top - 0.5), xytext=(6.5, y_top - 1.5),
                fontsize=9, ha='center', color=ORANGE,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1))
    
    # Annotation on "Detects Trap"
    ax.annotate('Bandit sees complex\nconstraints in prompt\nBEFORE generation',
                xy=(4.0, y_bot + 0.5), xytext=(4.0, y_bot + 1.5),
                fontsize=9, ha='center', color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
    
    # Save
    output_dir = Path("results/needle_in_haystack")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "confident_failure_trap.png", dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "confident_failure_trap.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"Saved: {output_dir / 'confident_failure_trap.png'}")
    
    # Also copy to paper figures
    paper_dir = Path("kdd_paper/paper_submitted/figures")
    paper_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(paper_dir / "figure_confident_failure.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.savefig(paper_dir / "figure_confident_failure.pdf",
                bbox_inches='tight', facecolor='white')
    
    print(f"Saved: {paper_dir / 'figure_confident_failure.png'}")
    
    plt.close()


if __name__ == "__main__":
    create_trap_diagram()
