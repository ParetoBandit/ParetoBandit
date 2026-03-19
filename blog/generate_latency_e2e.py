#!/usr/bin/env python3
"""Generate the sharpened E2E latency comparison blog image.

Horizontal log-scale bar chart comparing ParetoBandit's full critical-path
latency against other LLM routing systems (both peer-reviewed and practitioner),
with a shaded LLM inference region grounded in real API benchmarks.

Usage:
    python blog/generate_latency_e2e.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    systems = [
        ("ParetoBandit  (CPU only)", 8.3, 9.3, "#2E7D32"),
        ("NadirClaw  (CPU, practitioner)", 8, 12, "#66BB6A"),
        ("Orq.ai Auto Router  (Cloud, commercial)", 30, 40, "#FFA726"),
        ("vLLM Semantic Router — 1 clf.  (MI300X GPU)", 9, 14, "#EF5350"),
        ("vLLM Semantic Router — 3 clf.  (MI300X GPU)", 22, 22, "#E53935"),
        ("vLLM Semantic Router — 8K tok.  (MI300X GPU)", 50, 50, "#C62828"),
    ]

    y_positions = np.arange(len(systems))[::-1]

    for i, (label, lo, hi, color) in enumerate(systems):
        y = y_positions[i]
        ax.barh(y, hi, height=0.55, color=color, edgecolor="white",
                linewidth=0.8, alpha=0.92, zorder=3)
        txt = f"{lo}–{hi} ms" if lo != hi else f"{hi} ms"
        ax.text(hi * 1.12, y, txt,
                va="center", ha="left", fontsize=11, fontweight="bold",
                color="#333")

    ax.set_xscale("log")
    ax.set_xlim(0.01, 50_000)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in systems], fontsize=10.5)
    ax.set_xlabel("Latency (ms, log scale)", fontsize=12)

    # LLM inference region: 600 ms TTFT to 20 s total (real benchmark data)
    ax.axvspan(600, 50_000, alpha=0.14, color="#42A5F5", zorder=1)

    ref_lines = [952, 3_954, 18_075]
    for xval in ref_lines:
        ax.axvline(xval, color="#5C9BD5", ls="--", lw=1.0, alpha=0.6, zorder=1)

    ax.text(4500, y_positions[3],
            "LLM Inference\n952 ms – 20 s\n(Ganglani 2026)",
            ha="center", va="center", fontsize=10.5, color="#0D47A1",
            fontweight="bold", style="italic", zorder=2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.8))

    # route() call-out on ParetoBandit bar
    route_bar_y = y_positions[0]
    route_ms = 0.058
    ax.barh(route_bar_y, route_ms, height=0.55, color="#1B5E20",
            edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate("route() = 0.06 ms\n(<1% of E2E routing)",
                xy=(route_ms, route_bar_y + 0.30),
                xytext=(0.015, route_bar_y + 1.15),
                fontsize=9.5, fontweight="bold", color="#1B5E20",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.3),
                zorder=5)

    ax.annotate("embedding = 8 ms",
                xy=(2.0, route_bar_y),
                fontsize=8.5, color="white", fontweight="bold",
                ha="center", va="center", zorder=5)

    fig.suptitle("End-to-End Routing Overhead vs. LLM Inference",
                 fontsize=18, fontweight="bold", y=0.97)
    ax.set_title("Full critical-path latency: routing is invisible next to inference",
                 fontsize=12, color="#333", pad=18)

    ax.grid(axis="x", alpha=0.25, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    takeaways = [
        "Routing math (0.06 ms) is <1% of the 8 ms E2E pipeline — effectively free",
        "Full E2E routing (8 ms) adds <1% overhead to fastest LLM call (Haiku 4.5: 952 ms)",
        "For typical workloads (2–5 s total), routing is 0.2–0.4%; for long-form (18 s), <0.05%",
        "No peer-reviewed LLM router publishes per-decision or E2E latency",
    ]
    for j, line in enumerate(takeaways):
        fig.text(0.06, 0.13 - j * 0.032, f"•  {line}",
                 fontsize=10.5, color="#222", va="top")

    fig.tight_layout(rect=[0, 0.0, 1, 0.94])
    fig.subplots_adjust(bottom=0.22)

    for fmt in ("png",):
        out = OUT_DIR / f"latency_spectrum_e2e.{fmt}"
        fig.savefig(out, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
