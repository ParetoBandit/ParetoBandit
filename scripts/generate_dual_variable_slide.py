"""Generate a presentation slide explaining the dual variable concept and updates.

Matches the visual style of the LinUCB equations slide: white background, bold title,
color-coded equation parts, clean rounded boxes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette (matching reference slide)
# ---------------------------------------------------------------------------
ORANGE = "#E87722"
TEAL = "#2E8B8B"
DARK_TEXT = "#1A1A1A"
GREEN = "#2D8E3C"
BLUE = "#1565C0"
LIGHT_GRAY_BG = "#F5F5F5"
MID_GRAY = "#666666"
WHITE = "#FFFFFF"
LIGHT_ORANGE_BG = "#FFF7F0"

FIG_W, FIG_H = 14.0, 8.6


def _box(ax, x, y, w, h, fc=LIGHT_GRAY_BG, ec="#CCCCCC", lw=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _txt(ax, x, y, s, **kw):
    defaults = dict(transform=ax.transAxes, va="center", fontsize=14,
                    color=DARK_TEXT, family="sans-serif", zorder=5)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def build_slide(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # ── Title ─────────────────────────────────────────────────────────
    _txt(ax, 0.04, 0.960,
         r"Budget Cruise Control: The Dual Variable $\lambda_t$",
         fontsize=28, fontweight="bold")

    # ── Goal statement ────────────────────────────────────────────────
    _box(ax, 0.04, 0.870, 0.92, 0.055, fc="#EAF4F4", ec=TEAL, lw=1.5)
    _txt(ax, 0.50, 0.897,
         r"Goal:  Route each query to an LLM while keeping average spend "
         r"per request at target $B$",
         fontsize=13.5, fontweight="bold", color=TEAL, ha="center")

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — EMA smoothed cost
    # ══════════════════════════════════════════════════════════════════
    _txt(ax, 0.04, 0.795, "Step 1: Smooth the noisy cost signal",
         fontsize=18, fontweight="bold", color=ORANGE)

    eq1_y, eq1_h = 0.700, 0.07
    _box(ax, 0.04, eq1_y, 0.92, eq1_h)

    eq1c = eq1_y + eq1_h / 2
    _txt(ax, 0.06, eq1c,
         r"$\bar{c}_t$", fontsize=20, fontweight="bold", color=BLUE)
    _txt(ax, 0.12, eq1c,
         r"$= \;(1 - \alpha_{\mathrm{ema}})\;\cdot$", fontsize=18)
    _txt(ax, 0.34, eq1c,
         r"$\bar{c}_{t-1}$", fontsize=20, fontweight="bold", color=BLUE)
    _txt(ax, 0.43, eq1c,
         r"$+ \;\;\alpha_{\mathrm{ema}}\;\cdot$", fontsize=18)
    _txt(ax, 0.59, eq1c,
         r"$c_t$", fontsize=20, fontweight="bold", color=GREEN)

    _txt(ax, 0.06, eq1_y - 0.030,
         "A single expensive request (\\$0.015) can spike 500× above budget — "
         "the EMA filters this noise into a stable signal.",
         fontsize=11.5, color=MID_GRAY, style="italic")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — Dual variable update
    # ══════════════════════════════════════════════════════════════════
    _txt(ax, 0.04, 0.610, "Step 2: Update the dual variable",
         fontsize=18, fontweight="bold", color=ORANGE)

    eq2_y, eq2_h = 0.515, 0.075
    _box(ax, 0.04, eq2_y, 0.92, eq2_h)

    eq2c = eq2_y + eq2_h / 2
    _txt(ax, 0.06, eq2c,
         r"$\lambda_{t+1}$", fontsize=20, fontweight="bold", color=ORANGE)
    _txt(ax, 0.145, eq2c,
         r"$= \;\mathrm{clip}($", fontsize=18)
    _txt(ax, 0.275, eq2c,
         r"$\lambda_t$", fontsize=20, fontweight="bold", color=ORANGE)
    _txt(ax, 0.34, eq2c,
         r"$+ \;\eta\;\cdot\;($", fontsize=18)
    _txt(ax, 0.465, eq2c,
         r"$\bar{c}_t \,/\, B$", fontsize=18, fontweight="bold", color=BLUE)
    _txt(ax, 0.575, eq2c,
         r"$- \;1),\;\;0,\;\;\bar{\lambda}\;)$", fontsize=18)

    _txt(ax, 0.06, eq2_y - 0.030,
         r"If average cost $\bar{c}_t$ exceeds budget $B$, the gradient is positive "
         r"and $\lambda$ rises; if under budget, $\lambda$ falls.",
         fontsize=11.5, color=MID_GRAY, style="italic")

    # ══════════════════════════════════════════════════════════════════
    # THE FEEDBACK LOOP — side-by-side boxes
    # ══════════════════════════════════════════════════════════════════
    loop_y, loop_h = 0.260, 0.14
    _box(ax, 0.04, loop_y, 0.44, loop_h, fc=LIGHT_ORANGE_BG, ec=ORANGE, lw=1.5)
    _box(ax, 0.52, loop_y, 0.44, loop_h, fc=LIGHT_ORANGE_BG, ec=ORANGE, lw=1.5)

    # Left: above budget
    mid_left = 0.26
    _txt(ax, mid_left, loop_y + loop_h - 0.025,
         "Spending above budget", fontsize=14, fontweight="bold", ha="center")
    _txt(ax, mid_left, loop_y + loop_h - 0.060,
         r"$\bar{c}_t / B > 1$", fontsize=15, fontweight="bold", color=BLUE,
         ha="center")
    _txt(ax, mid_left, loop_y + loop_h - 0.090,
         r"$\lambda$ increases  →  expensive models penalized more",
         fontsize=12, color=DARK_TEXT, ha="center")
    _txt(ax, mid_left, loop_y + loop_h - 0.117,
         "→  router shifts to cheaper models  →  costs drop",
         fontsize=11, color=MID_GRAY, style="italic", ha="center")

    # Right: below budget
    mid_right = 0.74
    _txt(ax, mid_right, loop_y + loop_h - 0.025,
         "Spending below budget", fontsize=14, fontweight="bold", ha="center")
    _txt(ax, mid_right, loop_y + loop_h - 0.060,
         r"$\bar{c}_t / B < 1$", fontsize=15, fontweight="bold", color=BLUE,
         ha="center")
    _txt(ax, mid_right, loop_y + loop_h - 0.090,
         r"$\lambda$ decreases  →  cost penalty relaxed",
         fontsize=12, color=DARK_TEXT, ha="center")
    _txt(ax, mid_right, loop_y + loop_h - 0.117,
         "→  router freed to pick higher-quality models",
         fontsize=11, color=MID_GRAY, style="italic", ha="center")

    # Tagline
    _txt(ax, 0.50, loop_y - 0.030,
         "Self-correcting:  like cruise control for your LLM spend",
         fontsize=12, color=ORANGE, fontweight="bold", ha="center")

    # ── Save ──────────────────────────────────────────────────────────
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight",
                pad_inches=0.3, facecolor=WHITE)
    plt.close(fig)
    print(f"Saved slide to {output_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "blog" / "dual_variable_slide.png"
    build_slide(out)
