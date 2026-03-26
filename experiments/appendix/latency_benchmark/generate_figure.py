#!/usr/bin/env python3
"""Generate latency benchmark figure for the appendix.

Produces a three-panel figure:
  1. Route latency (p50 + p95) for all eight configurations.
  2. Update latency (p50 + p95) for all eight configurations.
  3. Throughput (req/s) for all eight configurations.

The layout makes the key insight visually immediate: route latency is
nearly identical for ParetoBandit and the Cached Inv. baseline (both
use a cached inverse), while the update panel exposes the O(d^2) vs.
O(d^3) difference that is Sherman-Morrison's actual contribution.

Usage:
    python experiments/appendix/latency_benchmark/generate_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "latency_benchmark_results.json"

_CONFIG_COLORS: dict[str, str] = {
    "ParetoBandit": "#1565C0",
    "Bare SM": "#64B5F6",
    "Cached Inv.": "#4CAF50",
    "Per-Route Inv.": "#FF9800",
}

_DIM_HATCHES: dict[int, str] = {26: "", 385: "//"}


def _style_for(name: str, dim: int) -> tuple[str, str]:
    """Return (color, hatch) for a configuration."""
    for prefix, color in _CONFIG_COLORS.items():
        if name.startswith(prefix):
            return color, _DIM_HATCHES.get(dim, "")
    return "#9E9E9E", ""


def main() -> None:
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    results = data["results"]

    labels = [r["name"] for r in results]
    route_p50 = [r["route_p50_us"] for r in results]
    route_p95 = [r["route_p95_us"] for r in results]
    update_p50 = [r["update_p50_us"] for r in results]
    update_p95 = [r["update_p95_us"] for r in results]
    throughput = [r["throughput_rps"] for r in results]

    plt.rcParams.update({"font.size": 16})

    fig, (ax_route, ax_update, ax_tp) = plt.subplots(
        1, 3, figsize=(21, 7.8),
        gridspec_kw={"width_ratios": [2, 2, 1.2]},
    )

    x = np.arange(len(labels))
    width = 0.35

    colors = [_style_for(r["name"], r["dimension"])[0] for r in results]
    hatches = [_style_for(r["name"], r["dimension"])[1] for r in results]

    # --- Route latency panel ---
    for i in range(len(labels)):
        ax_route.bar(
            x[i] - width / 2, route_p50[i], width,
            color=colors[i], hatch=hatches[i],
            edgecolor="white", linewidth=0.5,
            label="p50" if i == 0 else None,
        )
        ax_route.bar(
            x[i] + width / 2, route_p95[i], width,
            color=colors[i], hatch=hatches[i], alpha=0.5,
            edgecolor="white", linewidth=0.5,
            label="p95" if i == 0 else None,
        )

    ax_route.set_yscale("log")
    ax_route.set_ylabel("Latency (µs, log scale)", fontsize=17)
    ax_route.set_xticks(x)
    ax_route.set_xticklabels(labels, fontsize=14, rotation=35, ha="right")
    ax_route.legend(fontsize=14, loc="upper left")
    ax_route.set_title("Route Latency", fontsize=18, fontweight="bold")
    ax_route.tick_params(axis="y", labelsize=14)
    ax_route.grid(axis="y", alpha=0.3, which="both")
    ax_route.set_axisbelow(True)

    # --- Update latency panel ---
    for i in range(len(labels)):
        ax_update.bar(
            x[i] - width / 2, update_p50[i], width,
            color=colors[i], hatch=hatches[i],
            edgecolor="white", linewidth=0.5,
            label="p50" if i == 0 else None,
        )
        ax_update.bar(
            x[i] + width / 2, update_p95[i], width,
            color=colors[i], hatch=hatches[i], alpha=0.5,
            edgecolor="white", linewidth=0.5,
            label="p95" if i == 0 else None,
        )

    ax_update.set_yscale("log")
    ax_update.set_ylabel("Latency (µs, log scale)", fontsize=17)
    ax_update.set_xticks(x)
    ax_update.set_xticklabels(labels, fontsize=14, rotation=35, ha="right")
    ax_update.legend(fontsize=14, loc="upper left")
    ax_update.set_title("Update Latency", fontsize=18, fontweight="bold")
    ax_update.tick_params(axis="y", labelsize=14)
    ax_update.grid(axis="y", alpha=0.3, which="both")
    ax_update.set_axisbelow(True)

    # --- Throughput panel ---
    for i, (tp, c, h) in enumerate(zip(throughput, colors, hatches)):
        ax_tp.bar(i, tp, color=c, hatch=h, edgecolor="white", linewidth=0.5)
        ax_tp.text(
            i, tp * 1.08, f"{tp:,.0f}",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold",
        )

    ax_tp.set_yscale("log")
    ax_tp.set_ylabel("Throughput (req/s, log scale)", fontsize=17)
    ax_tp.set_xticks(range(len(labels)))
    ax_tp.set_xticklabels(labels, fontsize=14, rotation=35, ha="right")
    ax_tp.set_title("Throughput", fontsize=18, fontweight="bold")
    ax_tp.tick_params(axis="y", labelsize=14)
    ax_tp.grid(axis="y", alpha=0.3, which="both")
    ax_tp.set_axisbelow(True)

    fig.tight_layout()

    for fmt in ("png", "pdf"):
        out = RESULTS_DIR / f"latency_benchmark.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
