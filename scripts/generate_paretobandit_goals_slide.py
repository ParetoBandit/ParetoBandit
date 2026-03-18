"""Generate a presentation slide summarizing ParetoBandit's core goals.

The slide visualizes a single decision loop:
1) Maximize response quality by selecting the best model for each request.
2) Keep expected spend under a target dollars-per-request budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Visual palette aligned with existing blog diagrams.
DARK_NAVY = "#1B2A4A"
TEAL = "#2A9D8F"
TEAL_LIGHT = "#D4F0EC"
CORAL = "#E76F51"
CORAL_LIGHT = "#FDEAE4"
AMBER = "#D97706"
AMBER_LIGHT = "#FEF3C7"
MUTED_BLUE = "#3B82F6"
MUTED_BLUE_LIGHT = "#DBEAFE"
SLATE = "#4A5568"
WHITE = "#FFFFFF"


def _add_rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    face_color: str,
    edge_color: str,
    title_size: int = 14,
    subtitle_size: int = 10,
) -> None:
    """Draw a rounded rectangle with title/subtitle text.

    Args:
        ax: Target matplotlib axis.
        xy: Bottom-left corner `(x, y)` in axis coordinates.
        width: Box width.
        height: Box height.
        title: Main label shown in bold.
        subtitle: Supporting text shown below the title.
        face_color: Fill color.
        edge_color: Border color.
        title_size: Title font size.
        subtitle_size: Subtitle font size.
    """
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.28",
        fc=face_color,
        ec=edge_color,
        lw=2.0,
    )
    ax.add_patch(box)

    cx = xy[0] + width / 2.0
    cy = xy[1] + height / 2.0
    ax.text(
        cx,
        cy + 0.35,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=edge_color,
    )
    ax.text(
        cx,
        cy - 0.35,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=SLATE,
    )


def _add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = DARK_NAVY,
    style: str = "-|>",
    linewidth: float = 2.0,
    connection_style: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> None:
    """Add a directional arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=20,
        lw=linewidth,
        color=color,
        connectionstyle=connection_style,
        linestyle=linestyle,
    )
    ax.add_patch(arrow)


def create_paretobandit_goals_slide(
    output_path: str = "blog/paretobandit_goals_slide.png",
) -> Path:
    """Render a single-slide ParetoBandit goals diagram.

    Args:
        output_path: Destination image path. Parent directories are created if
            they do not already exist.

    Returns:
        The resolved path to the written PNG file.

    Side Effects:
        Writes a high-resolution PNG diagram to disk.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Slide heading.
    ax.text(
        8.0,
        8.35,
        "ParetoBandit Goal: Quality-Optimal Routing Under Budget",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=DARK_NAVY,
    )
    ax.text(
        8.0,
        7.8,
        "Choose the best LLM response quality while respecting a $/request constraint",
        ha="center",
        va="center",
        fontsize=13,
        color=SLATE,
        style="italic",
    )

    # Central router node.
    _add_rounded_box(
        ax=ax,
        xy=(6.2, 3.2),
        width=3.6,
        height=2.1,
        title="ParetoBandit Router",
        subtitle="Contextual bandit with online updates",
        face_color=TEAL_LIGHT,
        edge_color=TEAL,
        title_size=16,
        subtitle_size=11,
    )

    # Left objective: quality.
    _add_rounded_box(
        ax=ax,
        xy=(1.0, 4.9),
        width=4.2,
        height=2.0,
        title="Objective 1: Maximize Quality",
        subtitle="Route each prompt to the model with the highest expected utility",
        face_color=MUTED_BLUE_LIGHT,
        edge_color=MUTED_BLUE,
        title_size=14,
        subtitle_size=10,
    )

    # Right objective: budget.
    _add_rounded_box(
        ax=ax,
        xy=(10.8, 4.9),
        width=4.2,
        height=2.0,
        title="Objective 2: Enforce Budget",
        subtitle="Keep mean cost per request <= target budget B",
        face_color=AMBER_LIGHT,
        edge_color=AMBER,
        title_size=14,
        subtitle_size=10,
    )

    # Bottom candidate model pool.
    _add_rounded_box(
        ax=ax,
        xy=(1.0, 0.9),
        width=14.0,
        height=1.6,
        title="Candidate LLM Pool (cheap -> expensive)",
        subtitle="Router balances quality gain against incremental cost at decision time",
        face_color=CORAL_LIGHT,
        edge_color=CORAL,
        title_size=14,
        subtitle_size=10,
    )

    # Arrows linking objectives to router.
    _add_arrow(ax, (5.2, 5.6), (6.2, 4.8), color=MUTED_BLUE)
    ax.text(
        5.35,
        5.1,
        "quality signal",
        fontsize=10,
        color=MUTED_BLUE,
        rotation=-18,
        ha="center",
        va="center",
        bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.9, pad=0.8),
    )

    _add_arrow(ax, (10.8, 4.8), (9.8, 4.8), color=AMBER)
    ax.text(
        10.25,
        5.2,
        "cost pressure",
        fontsize=10,
        color=AMBER,
        ha="center",
        va="center",
        bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.9, pad=0.8),
    )

    # Router to LLM pool and feedback loop.
    _add_arrow(ax, (8.0, 3.2), (8.0, 2.5), color=TEAL)
    _add_arrow(
        ax,
        (11.8, 2.5),
        (9.5, 3.2),
        color=TEAL,
        linestyle="--",
        connection_style="arc3,rad=0.20",
    )
    ax.text(
        10.95,
        3.05,
        "online reward + cost feedback",
        fontsize=9,
        color=TEAL,
        ha="center",
        va="center",
        bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.9, pad=0.8),
    )

    # Footer takeaway.
    ax.text(
        8.0,
        0.35,
        "Outcome: Higher response quality at the same budget, or lower cost at the same quality.",
        ha="center",
        va="center",
        fontsize=12,
        color=DARK_NAVY,
        fontweight="bold",
        bbox=dict(
            facecolor=TEAL_LIGHT,
            edgecolor=TEAL,
            boxstyle="round,pad=0.35",
            alpha=0.65,
        ),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_paretobandit_goals_slide()
    print(f"Saved slide to {path}")
