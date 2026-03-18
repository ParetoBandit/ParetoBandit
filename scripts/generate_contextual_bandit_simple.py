"""Generate a professional contextual bandit feedback-loop diagram.

This script creates a clean 16:9 presentation graphic for explaining the
contextual bandit cycle. The output is designed for slide decks and replaces
the previous clip-art style image with a more polished vector schematic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


TEXT_DARK = "#0f172a"
TEXT_LIGHT = "#475569"
BG = "#ffffff"
SURFACE = "#f8fafc"
BORDER = "#e2e8f0"

BLUE = "#3b82f6"
TEAL = "#14b8a6"
GREEN = "#22c55e"
AMBER = "#f59e0b"


def _add_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    accent: str,
    step: str,
    title: str,
    body: str,
) -> None:
    """Draw a rounded card with a step badge and short explanatory text.

    Args:
        ax: Matplotlib axes receiving the drawing commands.
        x: Left coordinate in axis space.
        y: Bottom coordinate in axis space.
        w: Card width in axis space.
        h: Card height in axis space.
        accent: Accent color used for the top rule and step badge.
        step: Step label shown in the badge.
        title: Short card title.
        body: Supporting description.
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=SURFACE,
            ec=BORDER,
            lw=1.6,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.018),
            w,
            0.018,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=accent,
            ec="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + h - 0.025),
            w,
            0.025,
            fc=accent,
            ec="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        Circle(
            (x + 0.055, y + h - 0.09),
            0.03,
            fc=accent,
            ec="none",
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.055,
        y + h - 0.09,
        step,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=BG,
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.10,
        y + h - 0.09,
        title,
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.04,
        y + h - 0.155,
        body,
        ha="left",
        va="top",
        fontsize=13,
        color=TEXT_LIGHT,
        transform=ax.transAxes,
        linespacing=1.4,
    )


def _add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str,
    rad: float,
) -> None:
    """Draw a smooth cycle arrow between stages.

    Args:
        ax: Matplotlib axes receiving the drawing commands.
        start: Arrow starting point in axis space.
        end: Arrow end point in axis space.
        color: Arrow fill color.
        rad: Curvature parameter for connectionstyle.
    """
    patch = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="Simple,head_width=22,head_length=24,tail_width=7",
        fc=color,
        ec="#1e3a8a" if color == BLUE else color,
        lw=1.4,
        alpha=0.9,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)


def _add_context_icon(ax: plt.Axes, x: float, y: float, accent: str) -> None:
    """Draw a minimal context icon with prompt and user signals."""
    ax.add_patch(Circle((x + 0.022, y + 0.032), 0.014, fc="#dbeafe", ec="none", transform=ax.transAxes))
    ax.add_patch(Circle((x + 0.022, y + 0.032), 0.006, fc=accent, ec="none", transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((x + 0.050, y + 0.005), 0.050, 0.040, boxstyle="round,pad=0.01,rounding_size=0.01",
                                fc="#eef2ff", ec="#c7d2fe", lw=1, transform=ax.transAxes))
    ax.text(x + 0.075, y + 0.025, "prompt", ha="center", va="center", fontsize=7.5, color=TEXT_LIGHT, transform=ax.transAxes)
    ax.add_patch(Circle((x + 0.120, y + 0.032), 0.012, fc="#dcfce7", ec="none", transform=ax.transAxes))
    ax.text(x + 0.120, y + 0.032, "$", ha="center", va="center", fontsize=8.5, color=GREEN, transform=ax.transAxes)


def _add_action_icon(ax: plt.Axes, x: float, y: float, accent: str) -> None:
    """Draw a routing decision icon with two model choices."""
    for idx, label in enumerate(("Model A", "Model B")):
        box_y = y + 0.030 - idx * 0.034
        ax.add_patch(
            FancyBboxPatch(
                (x, box_y),
                0.075,
                0.022,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                fc="#eff6ff" if idx == 0 else "#ecfdf5",
                ec="#bfdbfe" if idx == 0 else "#bbf7d0",
                lw=1,
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.0375, box_y + 0.011, label, ha="center", va="center", fontsize=7.5, color=TEXT_DARK, transform=ax.transAxes)
    ax.add_patch(Circle((x - 0.022, y + 0.012), 0.011, fc=accent, ec="none", transform=ax.transAxes))
    ax.text(x - 0.022, y + 0.012, ">", ha="center", va="center", fontsize=8.5, fontweight="bold", color=BG, transform=ax.transAxes)


def _add_reward_icon(ax: plt.Axes, x: float, y: float, accent: str) -> None:
    """Draw a minimal reward icon with stars and feedback."""
    for idx in range(5):
        ax.text(x + idx * 0.015, y + 0.028, "★", fontsize=12, color=AMBER if idx < 4 else "#cbd5e1", transform=ax.transAxes)
    ax.add_patch(Circle((x + 0.03, y), 0.012, fc="#dcfce7", ec="none", transform=ax.transAxes))
    ax.text(x + 0.03, y, "+", ha="center", va="center", fontsize=9, fontweight="bold", color=GREEN, transform=ax.transAxes)
    ax.add_patch(Circle((x + 0.07, y), 0.012, fc="#fee2e2", ec="none", transform=ax.transAxes))
    ax.text(x + 0.07, y, "-", ha="center", va="center", fontsize=9, fontweight="bold", color="#ef4444", transform=ax.transAxes)


def _add_update_icon(ax: plt.Axes, x: float, y: float, accent: str) -> None:
    """Draw a policy update icon with parameter blocks."""
    for idx, height in enumerate((0.016, 0.028, 0.040)):
        ax.add_patch(
            FancyBboxPatch(
                (x + idx * 0.025, y),
                0.015,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.005",
                fc="#ecfeff",
                ec="#99f6e4",
                lw=1,
                transform=ax.transAxes,
            )
        )
    ax.add_patch(Circle((x + 0.09, y + 0.022), 0.015, fc=accent, ec="none", transform=ax.transAxes))
    ax.text(x + 0.09, y + 0.022, "↻", ha="center", va="center", fontsize=10, fontweight="bold", color=BG, transform=ax.transAxes)


def create_slide(output_path: str = "blog/contextual_bandit_simple.png") -> Path:
    """Create the contextual bandit cycle slide.

    Args:
        output_path: Destination path for the rendered PNG.

    Returns:
        Absolute path to the written image.
    """
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.05,
        0.91,
        "Contextual Bandit Feedback Loop",
        ha="left",
        va="center",
        fontsize=40,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.84,
        "The policy observes context, chooses an action, receives reward, and updates online.",
        ha="left",
        va="center",
        fontsize=19,
        color=TEXT_LIGHT,
        transform=ax.transAxes,
    )

    ax.add_patch(
        Circle(
            (0.50, 0.47),
            0.125,
            fc="#f8fafc",
            ec="#dbeafe",
            lw=2,
            ls=(0, (4, 4)),
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.50,
        0.50,
        "Online learning\nwith partial\nfeedback",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
        linespacing=1.35,
    )

    card_w = 0.28
    card_h = 0.19

    _add_card(
        ax,
        x=0.07,
        y=0.53,
        w=card_w,
        h=card_h,
        accent=BLUE,
        step="1",
        title="Observe Context",
        body="Read the prompt, user state,\nlatency budget, and recent model behavior.",
    )
    _add_context_icon(ax, 0.15, 0.565, BLUE)

    _add_card(
        ax,
        x=0.65,
        y=0.53,
        w=card_w,
        h=card_h,
        accent=TEAL,
        step="2",
        title="Choose Action",
        body="Select the model or tool with the\nhighest current expected value.",
    )
    _add_action_icon(ax, 0.82, 0.565, TEAL)

    _add_card(
        ax,
        x=0.65,
        y=0.19,
        w=card_w,
        h=card_h,
        accent=AMBER,
        step="3",
        title="Observe Reward",
        body="Measure the outcome from the chosen\naction only: quality, cost, or utility.",
    )
    _add_reward_icon(ax, 0.81, 0.245, AMBER)

    _add_card(
        ax,
        x=0.07,
        y=0.19,
        w=card_w,
        h=card_h,
        accent=GREEN,
        step="4",
        title="Update Policy",
        body="Refine estimates and exploration so\nthe next routing decision improves.",
    )
    _add_update_icon(ax, 0.15, 0.235, GREEN)

    _add_arrow(ax, (0.36, 0.66), (0.64, 0.66), BLUE, 0.12)
    _add_arrow(ax, (0.79, 0.51), (0.79, 0.40), TEAL, -0.05)
    _add_arrow(ax, (0.64, 0.28), (0.36, 0.28), AMBER, 0.12)
    _add_arrow(ax, (0.21, 0.39), (0.21, 0.51), GREEN, -0.05)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=BG)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    result = create_slide()
    print(f"Saved: {result}")
