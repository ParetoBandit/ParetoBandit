"""Generate a professional presentation slide explaining why Bandits fit LLM Routing.

Creates a 16:9 Google Slides compatible image with three modern UI cards
detailing the theoretical fit between Contextual Bandits and LLM routing.
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
TEXT_DARK = "#111827" 
TEXT_LIGHT = "#4b5563"
BG_WHITE = "#ffffff"

# Card 1 (Feedback) - Blue
C1_BG = "#eff6ff"
C1_BORDER = "#bfdbfe"
C1_ACCENT = "#3b82f6"

# Card 2 (Drift) - Coral/Red
C2_BG = "#fff1f2"
C2_BORDER = "#fecdd3"
C2_ACCENT = "#f43f5e"

# Card 3 (Trade-off) - Teal/Green
C3_BG = "#f0fdf4"
C3_BORDER = "#bbf7d0"
C3_ACCENT = "#10b981"


def _draw_numbered_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    bg_color: str,
    border_color: str,
    accent_color: str,
    number: str,
    title: str,
    subtitle: str,
    body_text: str,
) -> None:
    # Main card body
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc=bg_color, ec=border_color, lw=1.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(box)
    
    # Top accent bar
    accent = FancyBboxPatch(
        (x, y + h - 0.02), w, 0.02,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc=accent_color, ec="none",
        transform=ax.transAxes, clip_on=False,
    )
    rect = Rectangle(
        (x, y + h - 0.025), w, 0.025,
        fc=accent_color, ec="none",
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(accent)
    ax.add_patch(rect)
    
    # Large Number Watermark (Subtle)
    ax.text(
        x + w - 0.02, y + h - 0.06,
        number,
        ha="right", va="top",
        fontsize=72, fontweight="bold", color=accent_color,
        alpha=0.15,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    
    # Card Title
    ax.text(
        x + 0.03, y + h - 0.08,
        title,
        ha="left", va="center",
        fontsize=22, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    
    # Card Subtitle
    ax.text(
        x + 0.03, y + h - 0.13,
        subtitle,
        ha="left", va="center",
        fontsize=14, fontweight="bold", color=accent_color,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    
    # Subtle separator line
    ax.plot([x + 0.03, x + w - 0.03], [y + h - 0.17, y + h - 0.17], 
            color=border_color, lw=2, transform=ax.transAxes, clip_on=False)
            
    # Body Text (Wrapped)
    # Wrap very tightly (26 characters) and bump text size slightly back up
    wrapped_body = "\n".join(textwrap.wrap(body_text, width=28))
    ax.text(
        x + 0.03, y + h - 0.22,
        wrapped_body,
        ha="left", va="top",
        fontsize=16, color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
        linespacing=1.6,
    )


def create_slide(output_path: str = "blog/why_bandits_slide.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Header ---
    ax.text(
        0.05, 0.88,
        "Why Contextual Bandits for LLM Routing?",
        ha="left", va="center",
        fontsize=42, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )
    ax.text(
        0.05, 0.80,
        "The routing problem isn't static classification. It is fundamentally an online learning challenge.",
        ha="left", va="center",
        fontsize=20, color=TEXT_LIGHT,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )

    # --- Three Pillars (Cards) ---
    # Moved cards lower (y=0.08) and increased height (0.65)
    card_y = 0.08
    card_h = 0.65
    card_w = 0.28
    
    # Card 1: Partial Feedback
    _draw_numbered_card(
        ax, x=0.05, y=card_y, w=card_w, h=card_h,
        bg_color=C1_BG, border_color=C1_BORDER, accent_color=C1_ACCENT,
        number="01",
        title="Partial Feedback",
        subtitle="The Counterfactual Problem",
        body_text=(
            "When we route a prompt to a cheap model, we only learn its quality score. "
            "We never see the 'correct' label for the models we didn't use. "
            "Bandits are designed exactly for this partial 'counterfactual' feedback."
        )
    )
    
    # Card 2: Non-Stationarity
    _draw_numbered_card(
        ax, x=0.36, y=card_y, w=card_w, h=card_h,
        bg_color=C2_BG, border_color=C2_BORDER, accent_color=C2_ACCENT,
        number="02",
        title="Model Drift",
        subtitle="The Non-Stationary Environment",
        body_text=(
            "LLMs degrade silently. Prices change. New models drop weekly. "
            "A static supervised router goes stale the moment it's deployed. "
            "Bandits update dynamically, self-healing and adapting in real-time."
        )
    )
    
    # Card 3: Explore vs Exploit
    _draw_numbered_card(
        ax, x=0.67, y=card_y, w=card_w, h=card_h,
        bg_color=C3_BG, border_color=C3_BORDER, accent_color=C3_ACCENT,
        number="03",
        title="Explore vs. Exploit",
        subtitle="The Cost-Discovery Tradeoff",
        body_text=(
            "To hit a budget, we must route to known cheap models. "
            "To find out if a new model is good, we must route to it blindly. "
            "Bandits use UCB algorithms to mathematically optimize this risk/reward tradeoff."
        )
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" to enforce exactly 16:9 for Google Slides
    fig.savefig(out, dpi=300, facecolor=BG_WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_slide()
    print(f"Saved: {path}")
