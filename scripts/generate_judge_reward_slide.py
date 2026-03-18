"""Generate a professional presentation slide explaining the Reward function (LLM-as-a-Judge).

Shows the three dimensions of the rubric and an actual DeepSeek-R1 evaluation
rationale from the dataset to illustrate how continuous rewards are generated.
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

# Card Colors
CARD_BG = "#f8fafc"
CARD_BORDER = "#e2e8f0"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_PURPLE = "#8b5cf6"


def _draw_metric_bar(
    ax: plt.Axes, 
    x: float, y: float, 
    width: float, 
    score: float, 
    label: str, 
    color: str
) -> None:
    """Draws a horizontal progress bar for a metric."""
    label_y = y + 0.055

    # Background bar
    ax.add_patch(FancyBboxPatch(
        (x, y), width, 0.022,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        fc="#e5e7eb", ec="none", transform=ax.transAxes, clip_on=False
    ))
    
    # Fill bar
    fill_width = width * score
    ax.add_patch(FancyBboxPatch(
        (x, y), fill_width, 0.022,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        fc=color, ec="none", transform=ax.transAxes, clip_on=False
    ))
    
    # Label
    ax.text(
        x,
        label_y,
        label,
        fontsize=17,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        x + width,
        label_y,
        f"{score:.2f}",
        ha="right",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
        bbox=dict(
            facecolor=BG_WHITE,
            edgecolor="none",
            boxstyle="round,pad=0.18",
        ),
    )


def create_slide(output_path: str = "blog/judge_reward_slide.png") -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG_WHITE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Header ---
    ax.text(
        0.05, 0.92,
        "Defining the Reward Signal",
        ha="left", va="center",
        fontsize=42, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
        fontfamily="sans-serif",
    )

    # --- Left Column: The Rubric ---
    rubric_x, rubric_y, rubric_w, rubric_h = 0.05, 0.09, 0.43, 0.72
    
    # Card Background
    ax.add_patch(FancyBboxPatch(
        (rubric_x, rubric_y), rubric_w, rubric_h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc=CARD_BG, ec=CARD_BORDER, lw=1.5, transform=ax.transAxes, clip_on=False
    ))
    
    # Top Accent
    ax.add_patch(FancyBboxPatch(
        (rubric_x, rubric_y + rubric_h - 0.015), rubric_w, 0.015,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc=ACCENT_BLUE, ec="none", transform=ax.transAxes, clip_on=False
    ))
    ax.add_patch(Rectangle(
        (rubric_x, rubric_y + rubric_h - 0.02), rubric_w, 0.02,
        fc=ACCENT_BLUE, ec="none", transform=ax.transAxes, clip_on=False
    ))
    
    # Title
    ax.text(
        rubric_x + 0.04,
        rubric_y + rubric_h - 0.07,
        "The Evaluation Rubric",
        fontsize=26,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        rubric_x + 0.04,
        rubric_y + rubric_h - 0.11,
        "Three weighted dimensions converted into a continuous reward.",
        fontsize=14,
        color=TEXT_LIGHT,
        transform=ax.transAxes,
    )
    ax.plot(
        [rubric_x + 0.04, rubric_x + rubric_w - 0.04],
        [rubric_y + rubric_h - 0.14, rubric_y + rubric_h - 0.14],
        color=CARD_BORDER,
        lw=2,
        transform=ax.transAxes,
        clip_on=False,
    )
    
    # Metrics
    bar_x = rubric_x + 0.04
    bar_w = rubric_w - 0.08
    _draw_metric_bar(ax, bar_x, rubric_y + 0.45, bar_w, 0.20, "Reasoning Quality (40%)", ACCENT_BLUE)
    _draw_metric_bar(ax, bar_x, rubric_y + 0.31, bar_w, 0.30, "Instruction Following (30%)", ACCENT_GREEN)
    _draw_metric_bar(ax, bar_x, rubric_y + 0.17, bar_w, 0.70, "Communication Quality (30%)", ACCENT_PURPLE)

    # Formula
    formula = "Reward = 0.4 x 0.2 + 0.3 x 0.3 + 0.3 x 0.7\n= 0.38"
    ax.text(
        rubric_x + rubric_w / 2,
        rubric_y + 0.055,
        formula,
        ha="center", va="center", fontsize=18, fontweight="bold", color=TEXT_DARK,
        bbox=dict(facecolor="#f3f4f6", edgecolor="#d1d5db", boxstyle="round,pad=0.45")
    )


    # --- Right Column: The Judge Rationale (Actual Data) ---
    judge_x, judge_y, judge_w, judge_h = 0.51, 0.09, 0.44, 0.72
    
    # Card Background
    ax.add_patch(FancyBboxPatch(
        (judge_x, judge_y), judge_w, judge_h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc="#fef2f2", ec="#fecdd3", lw=1.5, transform=ax.transAxes, clip_on=False
    ))
    
    # Top Accent
    ax.add_patch(FancyBboxPatch(
        (judge_x, judge_y + judge_h - 0.015), judge_w, 0.015,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        fc="#e50914", ec="none", transform=ax.transAxes, clip_on=False
    ))
    ax.add_patch(Rectangle(
        (judge_x, judge_y + judge_h - 0.02), judge_w, 0.02,
        fc="#e50914", ec="none", transform=ax.transAxes, clip_on=False
    ))
    
    # Title
    ax.text(
        judge_x + 0.04,
        judge_y + judge_h - 0.07,
        "DeepSeek-R1 Judge Evaluation",
        fontsize=26,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        judge_x + 0.04,
        judge_y + judge_h - 0.115,
        "Example: scoring Gemini-pro on the Cassy jam word problem.",
        fontsize=13,
        color=TEXT_LIGHT,
        transform=ax.transAxes,
    )
    ax.plot(
        [judge_x + 0.04, judge_x + judge_w - 0.04],
        [judge_y + judge_h - 0.145, judge_y + judge_h - 0.145],
        color="#fecdd3",
        lw=2,
        transform=ax.transAxes,
        clip_on=False,
    )

    prompt_x = judge_x + 0.04
    prompt_y = judge_y + 0.43
    prompt_w = judge_w - 0.08
    prompt_h = 0.10
    ax.add_patch(FancyBboxPatch(
        (prompt_x, prompt_y),
        prompt_w,
        prompt_h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        fc="#fff7f7",
        ec="#fecdd3",
        lw=1.2,
        transform=ax.transAxes,
        clip_on=False,
    ))
    prompt_text = (
        '"Cassy packs 12 jars of jam in 10 boxes while she packs 10 jars of jam '
        'in 30 boxes. If she has 500 jars of jams, how many jars of jam will she '
        'have left when all the boxes are full?"'
    )
    ax.text(
        prompt_x + 0.006,
        prompt_y + prompt_h / 2,
        "\n".join(textwrap.wrap(prompt_text, width=54)),
        ha="left",
        va="center",
        fontsize=14,
        style="italic",
        color=TEXT_LIGHT,
        transform=ax.transAxes,
        linespacing=1.18,
    )

    rationale_x = judge_x + 0.04
    rationale_y = judge_y + 0.165
    rationale_w = judge_w - 0.08
    rationale_h = 0.23
    ax.add_patch(FancyBboxPatch(
        (rationale_x, rationale_y),
        rationale_w,
        rationale_h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        fc=BG_WHITE,
        ec="#fecdd3",
        lw=1.2,
        transform=ax.transAxes,
        clip_on=False,
    ))
    ax.text(
        rationale_x + 0.02,
        rationale_y + rationale_h - 0.025,
        "Judge's Chain-of-Thought Reasoning:",
        ha="left", va="top", fontsize=17, fontweight="bold", color=TEXT_DARK,
        transform=ax.transAxes,
    )

    rationale = (
        "\"Critical arithmetic error: the response adds 12 + 10 instead of computing "
        "jars packed across both box types. That leads to 22 jars used rather than "
        "420, so the final answer is incorrect despite clean formatting.\""
    )
    wrapped_rationale = "\n".join(textwrap.wrap(rationale, width=46))
    ax.text(
        rationale_x + 0.02,
        rationale_y + rationale_h - 0.062,
        wrapped_rationale,
        ha="left", va="top",
        fontsize=16,
        color="#7f1d1d",
        transform=ax.transAxes,
        fontfamily="serif",
        linespacing=1.45,
    )
    
    takeaway_x = judge_x + 0.04
    takeaway_y = judge_y + 0.05
    takeaway_w = judge_w - 0.08
    takeaway_h = 0.08
    ax.add_patch(FancyBboxPatch(
        (takeaway_x, takeaway_y),
        takeaway_w,
        takeaway_h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        fc="#fee2e2",
        ec="none",
        transform=ax.transAxes,
        clip_on=False,
    ))
    ax.text(
        takeaway_x + takeaway_w / 2,
        takeaway_y + takeaway_h / 2,
        "Continuous rewards let bandits learn nuanced\nquality differences beyond win/loss labels.",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#991b1b",
        transform=ax.transAxes,
    )


    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=BG_WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_slide()
    print(f"Saved: {path}")
