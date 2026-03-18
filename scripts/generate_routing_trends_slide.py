"""Generate a professional presentation slide showing LLM routing research trends.

Visualizes the growth of overall LLM routing papers and the emerging 
sub-segment of Contextual Bandit routers from 2024 to 2026.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Professional Color Palette ---
TEXT_DARK = "#1f2937" 
TEXT_LIGHT = "#4b5563"
BG_WHITE = "#ffffff"

# Chart colors
BAR_BG = "#e5e7eb"        # Light gray for total papers
BAR_FILL = "#dbeafe"      # Soft blue
BAR_EDGE = "#3b82f6"      # Bright blue
LINE_COLOR = "#e50914"    # Netflix/attention red for Bandits
LINE_BG = "#fef2f2"       # Soft red background for callouts

def _draw_callout(ax, x, y, title, subtitle, bg_color, border_color):
    """Draw a clean text callout box."""
    box = FancyBboxPatch(
        (x, y), 0.28, 0.12,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc=bg_color, ec=border_color, lw=1.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    
    ax.text(
        x + 0.02, y + 0.08,
        title,
        ha="left", va="center",
        fontsize=16, fontweight="bold", color=LINE_COLOR,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    
    ax.text(
        x + 0.02, y + 0.04,
        subtitle,
        ha="left", va="center",
        fontsize=13, color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )

def create_slide(output_path: str = "blog/routing_trends_slide.png") -> Path:
    # Google Slides default is 16:9 ratio. 16 inches x 9 inches at 300 DPI is perfect.
    fig = plt.figure(figsize=(16, 9), facecolor=BG_WHITE)
    
    # ---------------------------------------------------------
    # LEFT COLUMN: Narrative & Typography
    # ---------------------------------------------------------
    ax_text = fig.add_axes([0.05, 0.0, 0.35, 1.0])
    ax_text.axis("off")
    
    ax_text.text(
        0.0, 0.85,
        "The Shift to\nDynamic Routing",
        ha="left", va="top",
        fontsize=42, fontweight="bold", color=TEXT_DARK,
        fontfamily="sans-serif", linespacing=1.2
    )
    
    ax_text.text(
        0.0, 0.65,
        "LLM routing research has\nmore than tripled since 2024.",
        ha="left", va="top",
        fontsize=22, color=TEXT_LIGHT,
        fontfamily="sans-serif", linespacing=1.4
    )

    # Key Takeaways
    y_start = 0.45
    
    # Takeaway 1
    ax_text.plot([-0.02, -0.02], [y_start, y_start - 0.1], color=BAR_EDGE, lw=4)
    ax_text.text(
        0.02, y_start - 0.02,
        "Explosive Overall Growth",
        ha="left", va="top",
        fontsize=20, fontweight="bold", color=TEXT_DARK,
        fontfamily="sans-serif",
    )
    ax_text.text(
        0.02, y_start - 0.06,
        "The field has moved past single models\ninto complex multi-model architectures.",
        ha="left", va="top",
        fontsize=16, color=TEXT_LIGHT,
        fontfamily="sans-serif", linespacing=1.4
    )
    
    # Takeaway 2
    y_start -= 0.18
    ax_text.plot([-0.02, -0.02], [y_start, y_start - 0.15], color=LINE_COLOR, lw=4)
    ax_text.text(
        0.02, y_start - 0.02,
        "The Bandit Era Arrives",
        ha="left", va="top",
        fontsize=20, fontweight="bold", color=TEXT_DARK,
        fontfamily="sans-serif",
    )
    ax_text.text(
        0.02, y_start - 0.06,
        "Early routers relied on static, supervised\nlearning. The frontier has now shifted\nto Contextual Bandits for online,\nreal-time budget adaptation.",
        ha="left", va="top",
        fontsize=16, color=TEXT_LIGHT,
        fontfamily="sans-serif", linespacing=1.4
    )

    # ---------------------------------------------------------
    # RIGHT COLUMN: Chart (Stacked Bar Chart)
    # ---------------------------------------------------------
    ax = fig.add_axes([0.45, 0.15, 0.50, 0.70])
    
    years = ["2024", "2025", "2026\n(Projected)"]
    total_papers = np.array([70, 150, 240])
    bandit_pct = np.array([0.0, 0.027, 0.033])  # 0%, 2.7%, 3.3%
    
    bandit_papers = np.round(total_papers * bandit_pct).astype(int)
    non_bandit_papers = total_papers - bandit_papers
    
    x = np.arange(len(years))
    
    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color(TEXT_DARK)
    ax.set_yticks([])
    ax.tick_params(axis='x', length=0, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=20, fontweight="bold", color=TEXT_DARK)
    
    bar_width = 0.55
    
    # Draw Non-Bandit Base
    base_bars = ax.bar(
        x, non_bandit_papers, 
        width=bar_width, 
        color=BAR_FILL, edgecolor=BAR_EDGE, linewidth=2, zorder=2,
        label="Static / Supervised Routing"
    )
    
    # Draw Bandit Stack on top
    bandit_bars = ax.bar(
        x, bandit_papers, 
        bottom=non_bandit_papers,
        width=bar_width, 
        color=LINE_COLOR, edgecolor="#b91c1c", linewidth=2, zorder=2,
        label="Contextual Bandit Routing"
    )
    
    # Label the Total Papers above the bars
    for i, (base, top, total) in enumerate(zip(base_bars, bandit_bars, total_papers)):
        # Annotate total
        ax.text(
            x[i], total + 5,
            f"{total}+",
            ha="center", va="bottom",
            fontsize=26, fontweight="bold", color=TEXT_DARK,
        )
        
        # Annotate Bandit percentage / count on the red stack if it exists
        if bandit_papers[i] > 0:
            # We want to draw a callout pointing to the red cap
            cap_y = non_bandit_papers[i] + (bandit_papers[i] / 2)
            pct_val = bandit_pct[i] * 100
            
            # Place the text box to the right of the bar
            ax.text(
                x[i] + 0.35, total,
                f"{pct_val:.1f}% Bandits",
                ha="left", va="center",
                fontsize=18, fontweight="bold", color=LINE_COLOR,
                bbox=dict(facecolor=LINE_BG, edgecolor='none', boxstyle='round,pad=0.3', alpha=0.9)
            )
            # Arrow pointing to the red slice
            ax.annotate("",
                        xy=(x[i] + bar_width/2, total),
                        xytext=(x[i] + 0.35, total),
                        arrowprops=dict(arrowstyle="-|>", color=LINE_COLOR, lw=2, mutation_scale=15))

    # Add a custom legend inside the chart area
    ax.legend(loc='upper left', frameon=False, fontsize=16, 
              labelcolor=TEXT_DARK, title="Routing Approach", title_fontsize=18)

    # Narrative Annotations in the chart area
    ax.text(
        0.0, 40,
        "Dominated by static\nsupervised models\n(e.g., RouteLLM)",
        ha="center", va="top",
        fontsize=14, color=TEXT_LIGHT, style="italic"
    )
    
    ax.text(
        1.0, 90,
        "First online routers\nappear (e.g., BaRP, PILOT)",
        ha="center", va="top",
        fontsize=14, color=TEXT_LIGHT, style="italic"
    )
    
    ax.text(
        2.0, 160,
        "Accelerating shift\nto autonomous routing",
        ha="center", va="top",
        fontsize=14, color=TEXT_LIGHT, style="italic"
    )

    # Set custom Y limits to ensure space for total numbers and callouts
    ax.set_ylim(0, 275)

    # Y-axis label
    ax.text(-0.05, 0.5, "Total ArXiv Routing Papers", rotation=90, va='center', ha='right', 
            fontsize=18, color=BAR_EDGE, transform=ax.transAxes, fontweight="bold")
    
    # Footnote
    fig.text(
        0.05, 0.05,
        "*Based on targeted analysis of ArXiv publications and major surveys (e.g., LLMRouterBench).",
        fontsize=13, color=TEXT_LIGHT, style="italic"
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Remove bbox_inches="tight" to enforce exactly the 16:9 aspect ratio requested in figsize.
    fig.savefig(out, dpi=300, facecolor=BG_WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_slide()
    print(f"Saved: {path}")
