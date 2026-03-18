"""Generate a presentation slide summarising the evaluation benchmark dataset.

The slide communicates:
- 9 established public benchmarks used as data sources
- Prompt counts per source (horizontal bar chart)
- Train / Val / Holdout splits with sizes and intended use
- Key properties: real prompts, no overlap, stratified split
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Palette (aligned with existing blog slides)
# ---------------------------------------------------------------------------
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
LIGHT_GRAY = "#F3F4F6"
WHITE = "#FFFFFF"

PURPLE = "#7C3AED"
PURPLE_LIGHT = "#EDE9FE"

# Per-source bar colours — loosely grouped by domain.
BAR_COLORS: List[str] = [
    "#3B82F6",  # MMLU (knowledge)
    "#10B981",  # GSM8K (math)
    "#F59E0B",  # OpenBookQA (reasoning)
    "#8B5CF6",  # HellaSwag (language)
    "#EF4444",  # ARC-Challenge (reasoning)
    "#EC4899",  # BIG-Bench Hard (diverse)
    "#06B6D4",  # TruthfulQA (truthfulness)
    "#6366F1",  # WinoGrande (commonsense)
    "#14B8A6",  # MBPP (code)
]

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
SOURCES: List[Tuple[str, int]] = [
    ("MMLU", 2_651),
    ("GSM8K", 2_400),
    ("BIG-Bench Hard", 1_935),
    ("OpenBookQA", 1_151),
    ("HellaSwag", 1_134),
    ("ARC-Challenge", 1_060),
    ("TruthfulQA", 745),
    ("WinoGrande", 584),
    ("MBPP", 323),
]

TOTAL_PROMPTS = 11_983

SPLITS = [
    {
        "name": "Train",
        "count": 8_374,
        "pct": "70%",
        "use": "Warmup Priors",
        "desc": "Pre-compute arm statistics for cold start",
        "face": MUTED_BLUE_LIGHT,
        "edge": MUTED_BLUE,
    },
    {
        "name": "Validation",
        "count": 1_785,
        "pct": "15%",
        "use": "Online Learning",
        "desc": "Sequential bandit stream for adaptation",
        "face": TEAL_LIGHT,
        "edge": TEAL,
    },
    {
        "name": "Holdout",
        "count": 1_824,
        "pct": "15%",
        "use": "Evaluation",
        "desc": "Unseen prompts for final reporting",
        "face": CORAL_LIGHT,
        "edge": CORAL,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rounded_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    fc: str,
    ec: str,
    lw: float = 1.5,
    rounding: float = 0.25,
) -> FancyBboxPatch:
    """Draw a rounded rectangle and return the patch."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.15,rounding_size={rounding}",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# Main slide builder
# ---------------------------------------------------------------------------
def create_dataset_overview_slide(
    output_path: str = "blog/dataset_overview_slide.png",
) -> Path:
    """Render the dataset overview slide to *output_path*.

    Args:
        output_path: Destination PNG path.  Parent dirs are created as needed.

    Returns:
        Resolved path to the saved image.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.text(
        8.0,
        8.45,
        "Evaluation Benchmark: Real-World LLM Prompts",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=DARK_NAVY,
        fontfamily="sans-serif",
    )
    ax.text(
        8.0,
        7.9,
        f"{TOTAL_PROMPTS:,} prompts from 9 established public benchmarks  "
        "\u2014  all real, zero synthetic data",
        ha="center",
        va="center",
        fontsize=16,
        color=SLATE,
        style="italic",
        fontfamily="sans-serif",
    )

    # ------------------------------------------------------------------
    # LEFT PANEL – Data sources bar chart (x: 0.6–7.4, y: 1.6–7.4)
    # ------------------------------------------------------------------
    panel_l_x, panel_l_y = 0.6, 1.6
    panel_l_w, panel_l_h = 6.8, 5.7
    _rounded_box(ax, (panel_l_x, panel_l_y), panel_l_w, panel_l_h,
                 fc=LIGHT_GRAY, ec="#E5E7EB", lw=1.2, rounding=0.3)

    ax.text(
        panel_l_x + panel_l_w / 2,
        panel_l_y + panel_l_h - 0.35,
        "Data Sources",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=DARK_NAVY,
    )
    ax.text(
        panel_l_x + panel_l_w / 2,
        panel_l_y + panel_l_h - 0.75,
        "Curated from widely-adopted HuggingFace benchmarks",
        ha="center",
        va="center",
        fontsize=14,
        color=SLATE,
        style="italic",
    )

    bar_left = panel_l_x + 2.0
    bar_max_w = 4.2
    max_count = max(c for _, c in SOURCES)
    bar_h = 0.38
    bar_gap = 0.11
    top_y = panel_l_y + panel_l_h - 1.35

    for idx, ((name, count), color) in enumerate(zip(SOURCES, BAR_COLORS)):
        y = top_y - idx * (bar_h + bar_gap)
        w = (count / max_count) * bar_max_w

        bar = FancyBboxPatch(
            (bar_left, y),
            w,
            bar_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color,
            ec="none",
            alpha=0.85,
        )
        ax.add_patch(bar)

        ax.text(
            bar_left - 0.12,
            y + bar_h / 2,
            name,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=DARK_NAVY,
        )

        ax.text(
            bar_left + w + 0.12,
            y + bar_h / 2,
            f"{count:,}",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    # ------------------------------------------------------------------
    # RIGHT PANEL – Data splits & pipeline (x: 8.0–15.4, y: 1.6–7.4)
    # ------------------------------------------------------------------
    panel_r_x, panel_r_y = 8.0, 1.6
    panel_r_w, panel_r_h = 7.4, 5.7
    _rounded_box(ax, (panel_r_x, panel_r_y), panel_r_w, panel_r_h,
                 fc=LIGHT_GRAY, ec="#E5E7EB", lw=1.2, rounding=0.3)

    ax.text(
        panel_r_x + panel_r_w / 2,
        panel_r_y + panel_r_h - 0.35,
        "Data Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=DARK_NAVY,
    )


    card_w = 6.4
    card_h = 1.0
    card_x = panel_r_x + (panel_r_w - card_w) / 2
    card_top = panel_r_y + panel_r_h - 1.90

    for idx, split in enumerate(SPLITS):
        y = card_top - idx * (card_h + 0.60)

        _rounded_box(
            ax,
            (card_x, y),
            card_w,
            card_h,
            fc=split["face"],
            ec=split["edge"],
            lw=2.0,
            rounding=0.18,
        )

        # Top line: split name + arrow + purpose.
        ax.text(
            card_x + 0.30,
            y + card_h / 2 + 0.22,
            split["name"],
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=split["edge"],
        )
        ax.text(
            card_x + 2.4,
            y + card_h / 2 + 0.22,
            f"\u2192  {split['use']}",
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=DARK_NAVY,
        )

        # Bottom line: count + description (full width).
        ax.text(
            card_x + 0.30,
            y + card_h / 2 - 0.20,
            f"{split['count']:,} prompts ({split['pct']})  \u2022  {split['desc']}",
            ha="left",
            va="center",
            fontsize=13,
            color=SLATE,
        )

        # Connecting arrows between cards.
        if idx < len(SPLITS) - 1:
            ax.annotate(
                "",
                xy=(card_x + card_w / 2, y - 0.07),
                xytext=(card_x + card_w / 2, y + 0.01),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#CBD5E1",
                    lw=1.8,
                    mutation_scale=14,
                ),
            )

    # ------------------------------------------------------------------
    # Bottom highlight banner
    # ------------------------------------------------------------------
    banner_y = 0.35
    highlights = [
        ("\u2714  All real prompts", "No synthetic or paraphrased data"),
        ("\u2714  Standard benchmarks", "Widely used in LLM evaluation"),
        ("\u2714  Strict disjoint splits", "No prompt appears in > 1 partition"),
        ("\u2714  Reproducible", "Stratified split, fixed seed = 42"),
    ]

    banner_w = 3.5
    banner_h = 0.85
    total_w = len(highlights) * banner_w + (len(highlights) - 1) * 0.2
    start_x = (16 - total_w) / 2

    for i, (title, subtitle) in enumerate(highlights):
        bx = start_x + i * (banner_w + 0.2)
        _rounded_box(
            ax,
            (bx, banner_y),
            banner_w,
            banner_h,
            fc=PURPLE_LIGHT,
            ec=PURPLE,
            lw=1.2,
            rounding=0.15,
        )
        ax.text(
            bx + banner_w / 2,
            banner_y + banner_h / 2 + 0.12,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=PURPLE,
        )
        ax.text(
            bx + banner_w / 2,
            banner_y + banner_h / 2 - 0.18,
            subtitle,
            ha="center",
            va="center",
            fontsize=13,
            color=SLATE,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=WHITE)
    plt.close(fig)
    return out.resolve()


if __name__ == "__main__":
    path = create_dataset_overview_slide()
    print(f"Saved slide to {path}")
