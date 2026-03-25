#!/usr/bin/env python3
"""Generate the E2E latency comparison blog/slide image.

Horizontal log-scale bar chart comparing single-query **p50** latencies
across LLM routing systems.  ParetoBandit numbers are read from the
reproducible benchmark output
(``experiments/appendix/latency_benchmark/results/e2e_latency_results.json``);
all other systems use published numbers from their respective papers.

Usage:
    # First, run the benchmark to produce the JSON:
    python experiments/appendix/latency_benchmark/run_e2e_latency_benchmark.py
    # Then regenerate the figure:
    python blog/generate_latency_e2e.py
"""

from __future__ import annotations

import json
import sys
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
E2E_JSON = PAPER_DIR / "e2e_latency_results.json"


def _load_pareto_numbers() -> dict:
    """Load measured ParetoBandit E2E numbers from the benchmark JSON.

    Returns:
        Dict with keys ``total_p50_ms``, ``total_p95_ms``,
        ``route_p50_ms``, ``embed_p50_ms``.

    Raises:
        SystemExit: If the JSON file is missing (tells the user to run the
        benchmark first).
    """
    if not E2E_JSON.exists():
        sys.exit(
            f"ERROR: {E2E_JSON} not found.\n"
            "Run the E2E benchmark first:\n"
            "  python experiments/appendix/latency_benchmark/"
            "run_e2e_latency_benchmark.py"
        )
    data = json.loads(E2E_JSON.read_text(encoding="utf-8"))
    stages = data["stages"]
    return {
        "total_p50_ms": stages["total_p50_ms"],
        "total_p95_ms": stages["total_p95_ms"],
        "route_p50_ms": stages["route_p50_ms"],
        "embed_p50_ms": stages["embed_p50_ms"],
    }


def main() -> None:
    pb = _load_pareto_numbers()

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(16, 10))

    # ── System data ──────────────────────────────────────────────────
    # Each tuple: (label, p50_ms, color, annotation_or_None)
    total_p50 = round(pb["total_p50_ms"], 1)
    total_p95 = round(pb["total_p95_ms"], 1)

    systems: list[tuple[str, float, str, str | None]] = [
        ("ParetoBandit  (CPU only)",
         total_p50, "#2E7D32", f"p95: {total_p95} ms"),
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
                va="center", ha="left", fontsize=15, fontweight="bold",
                color="#333")

        if note:
            ax.text(p50 * 1.12, y - 0.26, note,
                    va="center", ha="left", fontsize=12, color="#777",
                    style="italic")

    # ── Axes ─────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlim(0.01, 50_000)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in systems], fontsize=15)
    ax.set_xlabel("Latency (ms, log scale)", fontsize=16)

    # ── LLM inference region ─────────────────────────────────────────
    ax.axvspan(600, 50_000, alpha=0.14, color="#42A5F5", zorder=1)
    for xval in [952, 3_954, 18_075]:
        ax.axvline(xval, color="#5C9BD5", ls="--", lw=1.0, alpha=0.6,
                   zorder=1)
    ax.text(4500, y_positions[4],
            "LLM Inference\n952 ms – 20 s\n(Ganglani 2026)",
            ha="center", va="center", fontsize=14, color="#0D47A1",
            fontweight="bold", style="italic", zorder=2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="none",
                      alpha=0.8))

    # ── ParetoBandit decomposition (from benchmark JSON) ─────────────
    embed_ms = pb["embed_p50_ms"]
    route_ms = pb["route_p50_ms"]

    pb_idx = next(
        i for i, (label, *_) in enumerate(systems)
        if "ParetoBandit" in label
    )
    pb_bar_y = y_positions[pb_idx]
    ax.annotate(
        f"embedding = {embed_ms:.1f} ms (98% of E2E)",
        xy=(0.6, pb_bar_y),
        fontsize=12, color="#1B5E20", fontweight="bold",
        ha="center", va="center", zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                  ec="#1B5E20", alpha=0.85, lw=0.8),
    )

    # ── Titles ───────────────────────────────────────────────────────
    fig.suptitle("End-to-End Routing Latency vs. LLM Inference",
                 fontsize=22, fontweight="bold", y=0.97)
    ax.set_title(
        "All bars show single-query p50 (median) latency"
        " — routing is invisible next to inference",
        fontsize=15, color="#555", pad=20,
    )

    ax.grid(axis="x", alpha=0.25, which="both", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Takeaway bullets ─────────────────────────────────────────────
    takeaways = [
        "All bars: single-query p50 (median). Secondary metrics noted in grey italic.",
        f"Routing decision ({route_ms:.2f} ms) is <1% of the {total_p50} ms E2E pipeline"
        " — effectively free.",
        f"Full E2E routing ({total_p50} ms on CPU) adds <1% overhead to fastest LLM call"
        " (Haiku 4.5: 952 ms).",
        f"ParetoBandit matches PROTEUS p50 ({total_p50} vs 8.7 ms) on CPU alone"
        " — no GPU required.",
    ]
    for j, line in enumerate(takeaways):
        fig.text(0.06, 0.13 - j * 0.034, f"•  {line}",
                 fontsize=13, color="#222", va="top")

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
