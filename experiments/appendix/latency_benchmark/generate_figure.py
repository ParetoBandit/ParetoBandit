#!/usr/bin/env python3
"""Generate latency benchmark figure for the appendix.

Produces a grouped bar chart comparing route and update p50 latencies
across all four configurations, with throughput annotated.

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

    fig, (ax_lat, ax_tp) = plt.subplots(
        1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [3, 1.2]},
    )

    x = np.arange(len(labels))
    width = 0.2

    colors = {
        "route_p50": "#2196F3",
        "route_p95": "#90CAF9",
        "update_p50": "#FF9800",
        "update_p95": "#FFE0B2",
    }

    bars_rp50 = ax_lat.bar(
        x - 1.5 * width, route_p50, width,
        label="Route p50", color=colors["route_p50"], edgecolor="white", linewidth=0.5,
    )
    ax_lat.bar(
        x - 0.5 * width, route_p95, width,
        label="Route p95", color=colors["route_p95"], edgecolor="white", linewidth=0.5,
    )
    bars_up50 = ax_lat.bar(
        x + 0.5 * width, update_p50, width,
        label="Update p50", color=colors["update_p50"], edgecolor="white", linewidth=0.5,
    )
    ax_lat.bar(
        x + 1.5 * width, update_p95, width,
        label="Update p95", color=colors["update_p95"], edgecolor="white", linewidth=0.5,
    )

    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("Latency (µs, log scale)")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax_lat.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax_lat.set_title("Per-Request Latency", fontsize=10, fontweight="bold")
    ax_lat.grid(axis="y", alpha=0.3, which="both")
    ax_lat.set_axisbelow(True)

    # Throughput bar chart
    bar_colors = ["#4CAF50", "#A5D6A7", "#4CAF50", "#A5D6A7"]
    hatches = ["", "", "//", "//"]
    for i, (tp, c, h) in enumerate(zip(throughput, bar_colors, hatches)):
        ax_tp.bar(
            i, tp, color=c, edgecolor="white", linewidth=0.5, hatch=h,
        )
        ax_tp.text(
            i, tp * 1.05, f"{tp:,.0f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    ax_tp.set_yscale("log")
    ax_tp.set_ylabel("Throughput (req/s, log scale)")
    ax_tp.set_xticks(range(len(labels)))
    ax_tp.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax_tp.set_title("Throughput", fontsize=10, fontweight="bold")
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
