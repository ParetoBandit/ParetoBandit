"""Generate a highly professional presentation slide comparing two anonymous LLM responses.

Shows the GSM8K jam-packing prompt with Model A (Mistral) and Model B (Gemini)
side by side in a clean, modern corporate presentation style. Text wrapping is
strictly enforced to fit within the designated container boxes.
"""

from __future__ import annotations

import os
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# --- Professional Color Palette ---
# Dark charcoal for primary text, pure white background
TEXT_DARK = "#1f2937" 
TEXT_LIGHT = "#4b5563"
BG_WHITE = "#ffffff"

# Subtle grays for the prompt box
PROMPT_BG = "#f3f4f6"
PROMPT_BORDER = "#d1d5db"
PROMPT_ACCENT = "#3b82f6"  # Blue accent line

# Model A (Winner) - Subtle green/teal
MOD_A_BG = "#f0fdf4"
MOD_A_BORDER = "#bbf7d0"
MOD_A_ACCENT = "#10b981"

# Model B (Loser) - Subtle red/rose
MOD_B_BG = "#fff1f2"
MOD_B_BORDER = "#fecdd3"
MOD_B_ACCENT = "#f43f5e"

PROMPT_TEXT = (
    "Cassy packs 12 jars of jam in 10 boxes while she packs "
    "10 jars of jam in 30 boxes. If she has 500 jars of jams, "
    "how many jars of jam will she have left when all the boxes are full?"
)

# Using bullet points rather than hard breaks to allow auto-wrapping
RESPONSE_A_LINES = [
    "Step 1: Jars packed in the first group",
    "• 12 jars/box × 10 boxes = 120 jars",
    "",
    "Step 2: Jars packed in the second group",
    "• 10 jars/box × 30 boxes = 300 jars",
    "",
    "Step 3: Total jars packed",
    "• 120 + 300 = 420 jars",
    "",
    "Step 4: Jars remaining",
    "• 500 − 420 = 80 jars"
]

RESPONSE_B_LINES = [
    "Step 1: Find total boxes",
    "• 10 + 30 = 40 boxes",
    "",
    "Step 2: Find total jars needed to fill all boxes",
    "• 12 + 10 = 22 jars",
    "",
    "Step 3: Calculate leftover jars",
    "• 500 − 22 = 478 jars"
]


def _draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    bg_color: str,
    border_color: str,
    accent_color: str,
    title: str,
    draw_separator: bool = True,
) -> None:
    # Main card body
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        fc=bg_color, ec=border_color, lw=1.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    
    # Top accent bar (gives a polished SaaS/corporate look)
    accent = FancyBboxPatch(
        (x, y + h - 0.02), w, 0.02,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        fc=accent_color, ec="none",
        transform=ax.transAxes, clip_on=False,
    )
    # Square off the bottom corners of the accent bar so it blends seamlessly
    rect = Rectangle(
        (x, y + h - 0.02), w, 0.02,
        fc=accent_color, ec="none",
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(accent)
    ax.add_patch(rect)
    
    # Card Title
    ax.text(
        x + 0.03, y + h - 0.055,
        title,
        ha="left", va="center",
        fontsize=18, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    
    # Subtle separator line under title
    if draw_separator:
        ax.plot([x + 0.03, x + w - 0.03], [y + h - 0.09, y + h - 0.09], 
                color=border_color, lw=1.5, transform=ax.transAxes, clip_on=False)


def _draw_text_lines(ax, x, y_start, lines, width_chars, line_height):
    y = y_start
    for line in lines:
        if not line:
            y -= line_height * 0.8
            continue
            
        wrapped = textwrap.wrap(line, width=width_chars)
        for wline in wrapped:
            # Subtle styling: bold the step names, dim the bullet points
            weight = "bold" if wline.startswith("Step") else "normal"
            color = TEXT_DARK if wline.startswith("Step") else TEXT_LIGHT
            fontfam = "sans-serif" if wline.startswith("Step") else "monospace"
            sz = 16 if wline.startswith("Step") else 15
            
            # Indent bullet points
            x_pos = x + 0.02 if wline.startswith("•") else x
            
            ax.text(
                x_pos, y,
                wline,
                ha="left", va="top",
                fontsize=sz, fontweight=weight, color=color,
                transform=ax.transAxes,
                fontfamily=fontfam,
            )
            y -= line_height
    return y


def create_slide(output_path: str = "blog/math_comparison_slide.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Header ---
    ax.text(
        0.05, 0.90,
        "Which Model Answered Correctly?",
        ha="left", va="center",
        fontsize=38, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    ax.text(
        0.05, 0.84,
        "A common LLM benchmark prompt (GSM8K) testing basic arithmetic and logic.",
        ha="left", va="center",
        fontsize=18, color=TEXT_LIGHT,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )

    # --- Prompt Card ---
    # The Prompt card needs to be moved up and made slightly taller to fit the wrapped text
    _draw_card(ax, 0.05, 0.60, 0.90, 0.20, PROMPT_BG, PROMPT_BORDER, PROMPT_ACCENT, "Prompt", draw_separator=False)
    
    # Use exact coordinates to force it inside the box instead of relying strictly on wrapping
    ax.text(
        0.07, 0.73,
        "Cassy packs 12 jars of jam in 10 boxes while she packs 10 jars of jam in 30 boxes.\n"
        "If she has 500 jars of jams, how many jars of jam will she have left when all the\n"
        "boxes are full?",
        ha="left", va="top",
        fontsize=18, color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
        linespacing=1.6,
    )

    # --- Model A Card (Left) ---
    _draw_card(ax, 0.05, 0.07, 0.43, 0.50, MOD_A_BG, MOD_A_BORDER, MOD_A_ACCENT, "Model A")
    _draw_text_lines(ax, 0.08, 0.47, RESPONSE_A_LINES, width_chars=40, line_height=0.035)

    # --- Model B Card (Right) ---
    _draw_card(ax, 0.52, 0.07, 0.43, 0.50, MOD_B_BG, MOD_B_BORDER, MOD_B_ACCENT, "Model B")
    _draw_text_lines(ax, 0.55, 0.47, RESPONSE_B_LINES, width_chars=40, line_height=0.035)

    # --- Footer Teaser ---
    # Add subtle teaser text at the bottom
    ax.text(
        0.50, 0.025,
        "Hint: One model costs 24× more per request than the other.",
        ha="center", va="center",
        fontsize=16, color=TEXT_LIGHT, fontweight="bold", style="italic",
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_slide()
    print(f"Saved: {path}")
