"""
Generate a two-panel presentation diagram contrasting:
  (A) How most teams deploy contextual bandits today (batch / offline),
  (B) A recommended production architecture for a contextual-bandit LLM router.

Designed for a mixed-expertise audience: uses plain-language labels with
optional technical annotations.

Visual style matches the existing blog diagrams (dark-navy/teal palette,
white background, clean box-and-arrow flow).
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Colour palette (consistent with the other blog figures)
# ---------------------------------------------------------------------------
DARK_NAVY = "#1B2A4A"
TEAL = "#2A9D8F"
TEAL_LIGHT = "#D4F0EC"
SLATE = "#4A5568"
LIGHT_BG = "#F7FAFC"
WHITE = "#FFFFFF"
CORAL = "#E76F51"
CORAL_LIGHT = "#FDEAE4"
MUTED_BLUE = "#3B82F6"
MUTED_BLUE_LIGHT = "#DBEAFE"
GREEN = "#16A34A"
GREEN_LIGHT = "#DCFCE7"
AMBER = "#D97706"
AMBER_LIGHT = "#FEF3C7"
GRAY = "#9CA3AF"
GRAY_LIGHT = "#F3F4F6"


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    fc: str = LIGHT_BG,
    ec: str = DARK_NAVY,
    lw: float = 1.8,
    fontsize: int = 10,
    fontweight: str = "bold",
    fontcolor: str = DARK_NAVY,
    sublabel: Optional[str] = None,
    sublabel_size: int = 8,
    boxstyle: str = "round,pad=0.25",
    text_y_offset: float = 0.0,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    """Draw a rounded rectangle with centred label and optional sub-label."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=boxstyle, fc=fc, ec=ec, lw=lw, alpha=alpha,
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2 + text_y_offset

    if sublabel:
        cy += 0.18
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=fontcolor)
        ax.text(cx, cy - 0.42, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=SLATE, style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=fontcolor)
    return box


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = DARK_NAVY,
    lw: float = 1.6,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
    mutation_scale: int = 16,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=connectionstyle,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        lw=lw, color=color, linestyle=linestyle,
    )
    ax.add_patch(arrow)
    return arrow


def _label(
    ax: plt.Axes,
    x: float, y: float,
    text: str,
    *,
    fontsize: int = 8,
    color: str = SLATE,
    fontweight: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0,
    bbox: Optional[dict] = None,
) -> None:
    if bbox is None:
        bbox = dict(facecolor=WHITE, edgecolor="none", alpha=0.85, pad=1.0)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight=fontweight, rotation=rotation, bbox=bbox)


# ═══════════════════════════════════════════════════════════════════════════
# Panel A – "How Most Teams Deploy Today"
# ═══════════════════════════════════════════════════════════════════════════
def _draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    ax.text(
        7, 8.55,
        "How Most Teams Deploy Today",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=CORAL,
    )
    ax.text(
        7, 8.0,
        "Offline / Batch  —  the policy is frozen at deploy time",
        ha="center", va="center",
        fontsize=10, color=SLATE, style="italic",
    )

    # --- Training pipeline (top row) ---
    top_y = 5.6
    box_h = 1.5

    _rounded_box(ax, (0.3, top_y), 2.7, box_h, "Historical\nLogs / Data",
                 fc=GRAY_LIGHT, ec=GRAY, fontsize=10)

    _rounded_box(ax, (4.3, top_y), 2.7, box_h, "Offline Training",
                 fc=CORAL_LIGHT, ec=CORAL, fontsize=10,
                 sublabel="Train bandit on static dataset")

    _rounded_box(ax, (8.3, top_y), 2.7, box_h, "Frozen Policy\nArtifact",
                 fc=CORAL_LIGHT, ec=CORAL, fontsize=10,
                 sublabel="Serialized model weights")

    _rounded_box(ax, (11.8, top_y), 2.0, box_h, "Production\nAPI",
                 fc=GRAY_LIGHT, ec=GRAY, fontsize=10)

    arrow_y = top_y + box_h / 2
    _arrow(ax, (3.0, arrow_y), (4.3, arrow_y), color=GRAY)
    _arrow(ax, (7.0, arrow_y), (8.3, arrow_y), color=CORAL)
    _arrow(ax, (11.0, arrow_y), (11.8, arrow_y), color=GRAY)

    _label(ax, 3.65, arrow_y + 0.45, "train", fontsize=8, color=GRAY)
    _label(ax, 7.65, arrow_y + 0.45, "export", fontsize=8, color=CORAL)
    _label(ax, 11.4, arrow_y + 0.45, "deploy", fontsize=8, color=GRAY)

    # --- Production request flow (bottom row) ---
    bot_y = 2.8
    bot_h = 1.3

    _rounded_box(ax, (0.3, bot_y), 2.3, bot_h, "User\nRequest",
                 fc=MUTED_BLUE_LIGHT, ec=MUTED_BLUE, fontsize=11)

    _rounded_box(ax, (3.7, bot_y), 3.0, bot_h, "Frozen Router",
                 fc=CORAL_LIGHT, ec=CORAL, fontsize=11,
                 sublabel="Same policy forever")

    _rounded_box(ax, (7.9, bot_y + 0.7), 2.2, 0.75, "Model A",
                 fc=TEAL_LIGHT, ec=TEAL, fontsize=10)
    _rounded_box(ax, (7.9, bot_y - 0.15), 2.2, 0.75, "Model B",
                 fc=TEAL_LIGHT, ec=TEAL, fontsize=10)

    _rounded_box(ax, (11.2, bot_y), 2.5, bot_h, "Response\n→ User",
                 fc=MUTED_BLUE_LIGHT, ec=MUTED_BLUE, fontsize=11)

    mid_y = bot_y + bot_h / 2
    _arrow(ax, (2.6, mid_y), (3.7, mid_y), color=DARK_NAVY)
    _arrow(ax, (6.7, mid_y + 0.3), (7.9, bot_y + 1.05), color=DARK_NAVY)
    _arrow(ax, (6.7, mid_y - 0.3), (7.9, bot_y + 0.23), color=DARK_NAVY)
    _arrow(ax, (10.1, mid_y), (11.2, mid_y), color=DARK_NAVY)

    # "No feedback loop" — prominent X with dashed arrow blocked
    x_center = 5.2
    _arrow(ax, (x_center, bot_y), (x_center, 1.6), color=CORAL, lw=1.5,
           linestyle="--")
    cross_sz = 0.25
    cx, cy_cross = x_center, 2.15
    ax.plot([cx - cross_sz, cx + cross_sz], [cy_cross - cross_sz, cy_cross + cross_sz],
            color=CORAL, lw=3, solid_capstyle="round")
    ax.plot([cx - cross_sz, cx + cross_sz], [cy_cross + cross_sz, cy_cross - cross_sz],
            color=CORAL, lw=3, solid_capstyle="round")

    ax.text(
        x_center, 1.15, "No feedback loop",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=CORAL,
    )
    ax.text(
        7, 0.55,
        "Policy drifts silently  •  New models ignored  •  Budget not enforced online",
        ha="center", va="center", fontsize=9, color=SLATE,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Panel B – "Recommended Production Architecture"
# ═══════════════════════════════════════════════════════════════════════════
def _draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-0.6, 10.0)
    ax.axis("off")

    ax.text(
        7, 9.6,
        "Recommended: Online Learning Architecture",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=TEAL,
    )
    ax.text(
        7, 9.1,
        "The router learns and adapts from every request in production",
        ha="center", va="center",
        fontsize=10, color=SLATE, style="italic",
    )

    # ── SERVING PATH (top row, y ≈ 7) ──
    srv_y = 6.6
    srv_h = 1.3

    _rounded_box(ax, (0.0, srv_y), 2.1, srv_h, "User\nRequest",
                 fc=MUTED_BLUE_LIGHT, ec=MUTED_BLUE, fontsize=11)

    _rounded_box(ax, (2.7, srv_y), 2.5, srv_h, "Feature\nExtraction",
                 fc=LIGHT_BG, ec=DARK_NAVY, fontsize=10,
                 sublabel="Embed prompt context")

    # Central router — taller to emphasise importance
    router_y = 6.2
    router_h = 2.0
    _rounded_box(ax, (5.8, router_y), 2.7, router_h, "Bandit\nRouter",
                 fc=TEAL_LIGHT, ec=TEAL, fontsize=13,
                 sublabel="Score → Select → Route", lw=2.2)

    # LLM pool
    pool_x = 9.4
    pool_w = 2.2
    _rounded_box(ax, (pool_x, 7.6), pool_w, 0.75, "Small  — cheap",
                 fc=GREEN_LIGHT, ec=GREEN, fontsize=9)
    _rounded_box(ax, (pool_x, 6.6), pool_w, 0.75, "Medium  — mid",
                 fc=AMBER_LIGHT, ec=AMBER, fontsize=9)
    _rounded_box(ax, (pool_x, 5.6), pool_w, 0.75, "Large  — costly",
                 fc=CORAL_LIGHT, ec=CORAL, fontsize=9)

    # Response
    _rounded_box(ax, (12.2, srv_y), 1.9, srv_h, "Response\n→ User",
                 fc=MUTED_BLUE_LIGHT, ec=MUTED_BLUE, fontsize=11)

    # Arrows: request → features → router
    srv_mid = srv_y + srv_h / 2
    _arrow(ax, (2.1, srv_mid), (2.7, srv_mid), color=DARK_NAVY)
    _arrow(ax, (5.2, srv_mid), (5.8, srv_mid), color=DARK_NAVY)

    # Router → models (fan-out)
    _arrow(ax, (8.5, 7.55), (pool_x, 7.95), color=DARK_NAVY)
    _arrow(ax, (8.5, 7.2), (pool_x, 6.97), color=DARK_NAVY)
    _arrow(ax, (8.5, 6.85), (pool_x, 6.0), color=DARK_NAVY)

    # Models → response (converge)
    _arrow(ax, (pool_x + pool_w, 7.2), (12.2, srv_mid + 0.15), color=DARK_NAVY,
           connectionstyle="arc3,rad=-0.05")
    _arrow(ax, (pool_x + pool_w, 6.35), (12.2, srv_mid - 0.15), color=DARK_NAVY,
           connectionstyle="arc3,rad=0.05")

    _label(ax, 4.65, srv_mid + 0.5, "context\nvector", fontsize=7, color=SLATE)

    # ── FEEDBACK / LEARNING LOOP (middle row, y ≈ 3.6) ──
    fb_y = 3.6
    fb_h = 1.3

    _rounded_box(ax, (9.2, fb_y), 2.8, fb_h, "Reward Signal",
                 fc=TEAL_LIGHT, ec=TEAL, fontsize=10,
                 sublabel="LLM-as-Judge / user rating")

    _rounded_box(ax, (5.8, fb_y), 2.8, fb_h, "Online Model\nUpdate",
                 fc=TEAL_LIGHT, ec=TEAL, fontsize=10,
                 sublabel="LinUCB / Thompson Sampling")

    # Response → reward signal (down-and-left)
    resp_bottom = srv_y
    _arrow(ax, (13.15, resp_bottom), (13.15, fb_y + fb_h),
           color=TEAL, lw=1.5)
    _arrow(ax, (13.15, fb_y + fb_h), (12.0, fb_y + fb_h / 2),
           color=TEAL, lw=1.5, connectionstyle="arc3,rad=0.2")
    _label(ax, 13.55, 5.5, "response\n+ cost", fontsize=7, color=TEAL)

    # Reward signal → model update
    _arrow(ax, (9.2, fb_y + fb_h / 2), (8.6, fb_y + fb_h / 2), color=TEAL, lw=1.5)
    _label(ax, 8.9, fb_y + fb_h + 0.15, "reward", fontsize=7, color=TEAL)

    # Model update → router (the key feedback loop arrow, going up)
    _arrow(ax, (7.1, fb_y + fb_h), (7.1, router_y), color=TEAL, lw=2.0)
    _label(ax, 7.65, 5.55, "update\nweights", fontsize=8, color=TEAL, fontweight="bold")

    # ── OPERATIONAL LAYER (bottom row, y ≈ 1.0) ──
    op_y = 0.8
    op_h = 1.3

    _rounded_box(ax, (0.0, fb_y), 2.5, fb_h, "Budget\nController",
                 fc=AMBER_LIGHT, ec=AMBER, fontsize=10,
                 sublabel="Dynamic cost penalty")

    _rounded_box(ax, (0.0, op_y), 2.5, op_h, "Monitoring &\nObservability",
                 fc=MUTED_BLUE_LIGHT, ec=MUTED_BLUE, fontsize=10,
                 sublabel="Dashboards / alerts")

    _rounded_box(ax, (3.2, op_y), 2.7, op_h, "Decision Log\n& Replay Buffer",
                 fc=GRAY_LIGHT, ec=SLATE, fontsize=10,
                 sublabel="Context, action, reward")

    _rounded_box(ax, (6.5, op_y), 2.8, op_h, "Shadow Mode &\nA/B Testing",
                 fc=GREEN_LIGHT, ec=GREEN, fontsize=10,
                 sublabel="Safe rollout of new policies")

    _rounded_box(ax, (10.0, op_y), 2.7, op_h, "Model Registry\n& Hot-Swap",
                 fc=GRAY_LIGHT, ec=SLATE, fontsize=10,
                 sublabel="Add/remove LLMs live")

    # Budget controller → router (dashed)
    _arrow(ax, (2.5, fb_y + fb_h / 2 + 0.3), (5.8, router_y + 0.3),
           color=AMBER, lw=1.3, linestyle="--", connectionstyle="arc3,rad=0.15")
    _label(ax, 3.5, 5.6, "cost\npressure", fontsize=7, color=AMBER)

    # Budget controller ← monitoring
    _arrow(ax, (1.25, fb_y), (1.25, op_y + op_h), color=MUTED_BLUE, lw=1.0,
           linestyle="--")

    # Model update → decision log
    _arrow(ax, (7.2, fb_y), (5.2, op_y + op_h), color=SLATE, lw=1.0,
           linestyle="--")
    _label(ax, 6.5, 2.9, "log", fontsize=7, color=SLATE)

    # Shadow mode ← router (comparison traffic)
    _arrow(ax, (7.9, op_y + op_h), (7.9, fb_y), color=GREEN, lw=1.0,
           linestyle="--")

    # Model registry → LLM pool (hot-swap new models in)
    _arrow(ax, (11.35, op_y + op_h), (11.0, 5.6),
           color=SLATE, lw=1.0, linestyle="--",
           connectionstyle="arc3,rad=-0.1")
    _label(ax, 11.8, 3.5, "add / swap\nmodels", fontsize=7, color=SLATE)

    # ── Benefits bar ──
    benefits = (
        "Learns from every request    "
        "•  Adapts to drift    "
        "•  Enforces budget online    "
        "•  Safe rollout via shadow mode"
    )
    ax.text(
        7, -0.3, benefits,
        ha="center", va="center",
        fontsize=9, color=TEAL, fontweight="bold",
        bbox=dict(facecolor=TEAL_LIGHT, edgecolor=TEAL,
                  boxstyle="round,pad=0.35", alpha=0.6),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main figure assembly
# ═══════════════════════════════════════════════════════════════════════════
def create_production_architecture_diagram(
    save_path: str = "blog/production_architecture.png",
) -> None:
    """Render the two-panel production-architecture diagram and save to *save_path*."""
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(15, 18),
        gridspec_kw={"height_ratios": [0.38, 0.62]},
    )
    fig.patch.set_facecolor(WHITE)
    fig.subplots_adjust(hspace=0.08)

    _draw_panel_a(ax_a)
    _draw_panel_b(ax_b)

    # Thin separator between panels — computed from axes positions
    ax_a_bb = ax_a.get_position()
    ax_b_bb = ax_b.get_position()
    sep_y = (ax_a_bb.y0 + ax_b_bb.y1) / 2
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (0.04, sep_y), 0.92, 0.002,
            transform=fig.transFigure,
            boxstyle="round,pad=0.001",
            fc=GRAY, ec="none", alpha=0.4,
        )
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved diagram -> {save_path}")


if __name__ == "__main__":
    create_production_architecture_diagram()
