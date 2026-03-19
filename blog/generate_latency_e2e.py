#!/usr/bin/env python3
"""Generate the E2E latency comparison blog/slide image.

Horizontal log-scale bar chart comparing single-query **p50** latencies
across LLM routing systems.  All bars use the same metric for
apples-to-apples comparison.

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
PAPER_DIR = (
    Path(__file__).resolve().parent.parent
    / "experiments" / "appendix" / "latency_benchmark" / "results"
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 9.2))

    # ── System data ──────────────────────────────────────────────────
    # Each tuple: (label, p50_ms, color, annotation_or_None)
    #   annotation: extra text placed to the right of the p50 label
    systems: list[tuple[str, float, str, str | None]] = [
        ("ParetoBandit  (CPU only)",
         8.3, "#2E7D32", "p95: 9.3 ms"),
        ("PROTEUS  (A100 GPU, peer-reviewed)",
         8.7, "#5C6BC0", "batch-32: 2.6 ms"),
        ("NadirClaw  (CPU, practitioner)",
         10.0, "#66BB6A", "range: 8–12 ms"),
        ("vLLM Semantic Router — 1 clf.  (MI300X GPU)",
         9.0, "#EF5350", "range: 9–14 ms"),
        ("vLLM Semantic Router — 3 clf.  (MI300X GPU)",
         22.0, "#E53935", None),
        ("Orq.ai Auto Router  (Cloud, commercial)",
         40.0, "#FFA726", "reported as <40 ms"),
        ("vLLM Semantic Router — 8K tok.  (MI300X GPU)",
         50.0, "#C62828", None),
    ]
    systems.sort(key=lambda s: s[1])

    y_positions = np.arange(len(systems))[::-1]

    # ── Draw bars (all single-query p50) ─────────────────────────────
    for i, (label, p50, color, note) in enumerate(systems):
        y = y_positions[i]
        ax.barh(y, p50, height=0.55, color=color, edgecolor="white",
                linewidth=0.8, alpha=0.92, zorder=3)

        main_txt = f"{p50:g} ms"
        ax.text(p50 * 1.12, y, main_txt,
                va="center", ha="left", fontsize=11.5, fontweight="bold",
                color="#333")

        if note:
            ax.text(p50 * 1.12, y - 0.24, note,
                    va="center", ha="left", fontsize=9, color="#777",
                    style="italic")

    # ── Axes ─────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50_000)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in systems], fontsize=13)
    ax.set_xlabel("Latency (ms, log scale)", fontsize=12)

    # ── LLM inference region ─────────────────────────────────────────
    ax.axvspan(600, 50_000, alpha=0.14, color="#42A5F5", zorder=1)
    for xval in [952, 3_954, 18_075]:
        ax.axvline(xval, color="#5C9BD5", ls="--", lw=1.0, alpha=0.6,
                   zorder=1)
    ax.text(4500, y_positions[4],
            "LLM Inference\n952 ms – 20 s\n(Ganglani 2026)",
            ha="center", va="center", fontsize=10.5, color="#0D47A1",
            fontweight="bold", style="italic", zorder=2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none",
                      alpha=0.8))

    # ── ParetoBandit decomposition ───────────────────────────────────
    route_bar_y = y_positions[0]
    route_ms = 0.058
    ax.barh(route_bar_y, route_ms, height=0.55, color="#1B5E20",
            edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate("route() = 0.06 ms\n(<1% of E2E)",
                xy=(route_ms, route_bar_y + 0.30),
                xytext=(0.015, route_bar_y + 1.15),
                fontsize=9.5, fontweight="bold", color="#1B5E20",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.3),
                zorder=5)
    ax.annotate("embedding = 8.1 ms",
                xy=(2.0, route_bar_y),
                fontsize=9, color="#1B5E20", fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#1B5E20", alpha=0.85, lw=0.8))

    # ── Titles ───────────────────────────────────────────────────────
    fig.suptitle("End-to-End Routing Latency vs. LLM Inference",
                 fontsize=18, fontweight="bold", y=0.97)
    ax.set_title(
        "All bars show single-query p50 (median) latency"
        " — routing is invisible next to inference",
        fontsize=12, color="#555", pad=18,
    )

    ax.grid(axis="x", alpha=0.25, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Takeaway bullets ─────────────────────────────────────────────
    takeaways = [
        "All bars: single-query p50 (median). Secondary metrics noted in grey italic.",
        "Routing decision (0.06 ms) is <1% of the 8.3 ms E2E pipeline — effectively free.",
        "Full E2E routing (8.3 ms on CPU) adds <1% overhead to fastest LLM call"
        " (Haiku 4.5: 952 ms).",
        "ParetoBandit matches PROTEUS p50 (8.3 vs 8.7 ms) on CPU alone — no GPU required.",
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

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    paper_out = PAPER_DIR / f"latency_spectrum_e2e.{fmt}"
    fig.savefig(paper_out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {paper_out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
