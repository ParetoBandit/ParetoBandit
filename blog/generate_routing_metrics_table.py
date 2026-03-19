#!/usr/bin/env python3
"""Generate a blog-ready image of the Experiment 01 routing metrics table.

Renders the Static vs BudgetPacer routing evaluation metrics as a
publication-quality matplotlib figure suitable for embedding in the
blog post.  Uses manual axes drawing for full control over layout,
typography, and color.

Reads metric values from the experiment results JSON (via
``generate_latex.compute_routing_metrics``) so the blog image stays
in sync with the paper tables.

Usage:
    python blog/generate_routing_metrics_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

OUT_DIR = Path(__file__).parent
EXP01_RESULTS = (
    PROJECT_ROOT / "experiments" / "01_stationary_budget_pacing"
    / "results" / "budget_pacing_results.json"
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
DARK = "#111827"
ACCENT = "#1D4ED8"
ACCENT_LIGHT = "#DBEAFE"
WIN_BG = "#EFF6FF"
WIN_BORDER = "#1D4ED8"
LOSE_BG = "#FFFFFF"
HEADER_BG = "#111827"
HEADER_TEXT = "#FFFFFF"
ROW_LABEL_BG = "#F8FAFC"
DIVIDER = "#CBD5E1"
SUBTLE_TEXT = "#374151"
BODY_TEXT = "#111827"
DELTA_POS = "#1E40AF"
DELTA_NEG = "#1E40AF"


# ---------------------------------------------------------------------------
# Data — read from JSON via generate_latex helpers
# ---------------------------------------------------------------------------
COLUMNS = [
    ("Pareto\nAUC", True),
    ("AUCPC", True),
    ("cost @90%\nquality", False),
    ("cost @95%\nquality", False),
    ("qual @50%\ncost", True),
    ("qual @25%\ncost", True),
    ("Save\n@95%", True),
]

_METRIC_KEYS = [
    "pareto_auc_mean",
    "aucpc_mean",
    "cost_at_90_mean",
    "cost_at_95_mean",
    "quality_at_50_mean",
    "quality_at_25_mean",
    "cost_save_95_mean",
]


def _fmt_cost_sci(v: float) -> str:
    """Format a cost value as $X.Y×10⁻ⁿ using Unicode superscripts."""
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    exp = int(np.floor(np.log10(abs(v))))
    mantissa = v / 10**exp
    exp_str = str(exp).translate(sup)
    return f"${mantissa:.1f}×10{exp_str}"


def _fmt_cell(key: str, v: float) -> str:
    """Format a metric value for display in the blog table."""
    if "cost_at" in key:
        return _fmt_cost_sci(v)
    if "cost_save" in key:
        return f"{v * 100:.1f}%"
    if "pareto_auc" in key:
        return f"{v:.4f}"
    if "aucpc" in key:
        return f"{v:.3f}"
    return f"{v:.3f}"


def _load_routing_metrics() -> Tuple[
    List[float], List[float], List[str], List[str], float,
]:
    """Load routing metrics from the experiment JSON.

    Returns:
        static_raw, pacer_raw, static_fmt, pacer_fmt, p_value
    """
    sys.path.insert(
        0,
        str(PROJECT_ROOT / "experiments" / "01_stationary_budget_pacing"),
    )
    from generate_latex import compute_routing_metrics, load_results

    data = load_results(EXP01_RESULTS)
    rm = compute_routing_metrics(data)

    sm = rm["static"]
    pm = rm["pacer"]

    static_raw = [sm[k] for k in _METRIC_KEYS]
    pacer_raw = [pm[k] for k in _METRIC_KEYS]

    static_fmt = [_fmt_cell(k, sm[k]) for k in _METRIC_KEYS]
    pacer_fmt = [_fmt_cell(k, pm[k]) for k in _METRIC_KEYS]

    p_value = data.get("dominance_test", {}).get("p_value", 1.0)
    return static_raw, pacer_raw, static_fmt, pacer_fmt, p_value


def _delta_label(s: float, p: float, higher_better: bool, is_cost: bool) -> str:
    """Produce a human-readable delta string."""
    if is_cost:
        ratio = s / p if p != 0 else float("inf")
        return f"{ratio:.1f}× cheaper"
    diff_pct = (p - s) / s * 100 if s != 0 else 0.0
    if abs(diff_pct) < 1.0:
        return f"+{(p - s):.4f}"
    return f"+{diff_pct:.1f}%"


def _rounded_rect(
    ax: plt.Axes,
    x: float, y: float, w: float, h: float,
    color: str,
    border_color: str = "none",
    radius: float = 0.008,
    lw: float = 0,
) -> None:
    """Draw a rounded rectangle on *ax*."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color,
        edgecolor=border_color,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)


def main() -> None:
    static_raw, pacer_raw, static_fmt, pacer_fmt, p_value = (
        _load_routing_metrics()
    )
    n_cols = len(COLUMNS)

    fig, ax = plt.subplots(figsize=(18, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    left_margin = 0.115
    right_margin = 0.01
    table_width = 1.0 - left_margin - right_margin
    col_w = table_width / n_cols

    row_height = 0.18
    header_y = 0.68
    static_y = header_y - row_height - 0.015
    pacer_y = static_y - row_height - 0.01
    delta_y = pacer_y - row_height - 0.01

    p_str = f"{p_value:.1e}".replace("e-0", " × 10⁻").replace("e-", " × 10⁻")

    ax.text(
        0.5, 0.98,
        "Routing Evaluation Metrics",
        ha="center", va="top",
        fontsize=26, fontweight="bold", color=DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.90,
        f"Static Cost Penalty  vs.  BudgetPacer   ·   K = 3 models   ·   20 seeds   ·   Wilcoxon p = {p_str}",
        ha="center", va="top",
        fontsize=14, fontweight="medium", color=SUBTLE_TEXT,
        transform=ax.transAxes,
    )

    _rounded_rect(ax, left_margin - 0.005, header_y, table_width + 0.01, row_height,
                  HEADER_BG, radius=0.012)

    for j, (label, _) in enumerate(COLUMNS):
        cx = left_margin + (j + 0.5) * col_w
        cy = header_y + row_height * 0.5
        ax.text(
            cx, cy, label,
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=HEADER_TEXT,
            linespacing=1.25,
            transform=ax.transAxes,
        )

    label_x = left_margin * 0.5

    _rounded_rect(ax, left_margin - 0.005, static_y, table_width + 0.01, row_height,
                  "#F1F5F9", border_color=DIVIDER, lw=1, radius=0.012)
    ax.text(
        label_x, static_y + row_height * 0.5,
        "Static\nPenalty",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=BODY_TEXT,
        linespacing=1.3,
        transform=ax.transAxes,
    )
    for j in range(n_cols):
        cx = left_margin + (j + 0.5) * col_w
        cy = static_y + row_height * 0.5
        ax.text(
            cx, cy, static_fmt[j],
            ha="center", va="center",
            fontsize=16, color=BODY_TEXT,
            fontfamily="monospace",
            transform=ax.transAxes,
        )

    _rounded_rect(ax, left_margin - 0.005, pacer_y, table_width + 0.01, row_height,
                  WIN_BG, border_color=WIN_BORDER, lw=2.2, radius=0.012)
    ax.text(
        label_x, pacer_y + row_height * 0.5,
        "Budget\nPacer",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#1E3A8A",
        linespacing=1.3,
        transform=ax.transAxes,
    )
    for j in range(n_cols):
        cx = left_margin + (j + 0.5) * col_w
        cy = pacer_y + row_height * 0.5
        ax.text(
            cx, cy, pacer_fmt[j],
            ha="center", va="center",
            fontsize=16, fontweight="bold", color="#1E3A8A",
            fontfamily="monospace",
            transform=ax.transAxes,
        )

    is_cost_col = [False, False, True, True, False, False, False]
    for j, ((_, higher_better), is_cost) in enumerate(zip(COLUMNS, is_cost_col)):
        cx = left_margin + (j + 0.5) * col_w
        cy = delta_y + row_height * 0.5

        s, p = static_raw[j], pacer_raw[j]
        label = _delta_label(s, p, higher_better, is_cost)

        pill_w = col_w * 0.85
        pill_h = row_height * 0.58
        _rounded_rect(
            ax,
            cx - pill_w / 2, cy - pill_h / 2,
            pill_w, pill_h,
            "#EFF6FF", border_color="#93C5FD", lw=1.2, radius=0.008,
        )
        ax.text(
            cx, cy, f"▲ {label}",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=DELTA_POS,
            transform=ax.transAxes,
        )

    # -- Thin vertical dividers within rows --
    for j in range(1, n_cols):
        x = left_margin + j * col_w
        for y_start, h in [(static_y, row_height), (pacer_y, row_height)]:
            ax.plot(
                [x, x],
                [y_start + 0.02, y_start + h - 0.02],
                color=DIVIDER, lw=0.6,
                transform=ax.transAxes,
                clip_on=False,
            )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = OUT_DIR / "experiment_01_routing_metrics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
