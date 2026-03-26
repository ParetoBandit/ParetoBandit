"""Generate LaTeX commands from latency benchmark results.

Reads ``results/latency_benchmark_results.json`` (component-level),
``results/e2e_latency_results.json`` (end-to-end pipeline), and
``results/inference_latency_results.json`` (LLM inference latency),
then emits ``_autogen.tex`` with ``\\lat``-prefixed commands.

Usage::

    python experiments/appendix/latency_benchmark/generate_latex.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from utils.latex_gen import CommandSet, fmt_int, fmt_num, load_json

LABEL_TO_SHORT: Dict[str, str] = {
    "ParetoBandit (d=26)": "PBdTwentySix",
    "ParetoBandit (d=385)": "PBdFull",
    "Bare SM (d=26)": "SMdTwentySix",
    "Bare SM (d=385)": "SMdFull",
    "Cached Inv. (d=26)": "CachedDTwentySix",
    "Cached Inv. (d=385)": "CachedDFull",
    "Per-Route Inv. (d=26)": "PerRouteDTwentySix",
    "Per-Route Inv. (d=385)": "PerRouteDFull",
}

MODEL_ID_TO_SHORT: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "mistralai/mistral-large-2512": "Mistral",
    "google/gemini-2.5-flash": "GemFlash",
    "google/gemini-2.5-pro": "GemPro",
}

LENGTH_TO_SHORT: Dict[str, str] = {
    "short": "Short",
    "medium": "Med",
    "long": "Long",
}


def build_command_set(
    bench_data: Dict[str, Any],
    e2e_data: Dict[str, Any],
) -> CommandSet:
    """Build the full ``CommandSet`` from JSON data."""
    cs = CommandSet(prefix="lat")

    cs.raw("Rounds", fmt_int(bench_data["rounds"]))
    cs.raw("Warmup", fmt_int(bench_data["warmup"]))

    for result in bench_data.get("results", []):
        label = result.get("label", result.get("name", ""))
        short = LABEL_TO_SHORT.get(label)
        if short is None:
            continue

        cs.num(f"{short}RouteMedian", result["route_p50_us"], digits=1)
        cs.num(f"{short}RoutePNineFive", result["route_p95_us"], digits=1)
        cs.num(f"{short}RoutePNineNine", result["route_p99_us"], digits=1)
        cs.num(f"{short}UpdateMedian", result["update_p50_us"], digits=1)
        cs.num(f"{short}UpdatePNineFive", result["update_p95_us"], digits=1)
        cs.num(f"{short}UpdatePNineNine", result["update_p99_us"], digits=1)
        cs.num(f"{short}TotalMedian", result["total_p50_us"], digits=1)
        cs.num(f"{short}TotalPNineFive", result["total_p95_us"], digits=1)
        cs.raw(f"{short}Throughput", fmt_int(round(result["throughput_rps"])))

    speedups = bench_data.get("speedups", {})
    for k, v in speedups.items():
        parts = k.split("_")
        cs.num(f"Speedup{''.join(p.title() for p in parts)}", v, digits=1)

    stages = e2e_data.get("stages", {})
    fracs = e2e_data.get("fractions", {})

    cs.num("EteEmbedMedianMs", stages.get("embed_p50_ms", 0), digits=1)
    cs.num("EteEmbedPNineFiveMs", stages.get("embed_p95_ms", 0), digits=1)
    cs.num("EteEmbedPNineNineMs", stages.get("embed_p99_ms", 0), digits=1)
    cs.num("EtePcaMedianMs", stages.get("pca_p50_ms", 0), digits=2)
    cs.num("EtePcaPNineFiveMs", stages.get("pca_p95_ms", 0), digits=2)
    cs.num("EteRouteMedianMs", stages.get("route_p50_ms", 0), digits=3)
    cs.num("EteRoutePNineFiveMs", stages.get("route_p95_ms", 0), digits=3)
    cs.num("EteRoutePNineNineMs", stages.get("route_p99_ms", 0), digits=3)
    cs.num("EteTotalMedianMs", stages.get("total_p50_ms", 0), digits=1)
    cs.num("EteTotalPNineFiveMs", stages.get("total_p95_ms", 0), digits=1)
    cs.num("EteTotalPNineNineMs", stages.get("total_p99_ms", 0), digits=1)

    cs.num("EteEmbedPctOfTotal", fracs.get("embed_pct_of_total_p50", 0), digits=1)
    cs.num("EteRoutePctOfTotal", fracs.get("route_pct_of_total_p50", 0), digits=1)

    cs.raw("EteRounds", fmt_int(e2e_data.get("rounds", 0)))
    cs.raw("EtePromptPool", fmt_int(e2e_data.get("prompt_pool_size", 0)))

    return cs


def _add_inference_latency_commands(
    cs: CommandSet,
    inf_data: Dict[str, Any],
    e2e_total_ms: float,
) -> None:
    """Add inference latency macros from the K=4 portfolio benchmark.

    Emits macros like ``\\latInfLlamaTtftShort``, ``\\latInfLlamaTotalShort``,
    ``\\latInfLlamaTtftShortCILo``, ``\\latInfLlamaTtftShortCIHi``,
    and ``\\latInfLlamaRoutingPctShort`` for each (model, prompt-length).

    Args:
        cs: The ``CommandSet`` to add macros to.
        inf_data: Parsed ``inference_latency_results.json``.
        e2e_total_ms: ParetoBandit E2E routing latency in ms (for ratio column).
    """
    cs.raw("InfTrials", fmt_int(inf_data.get("trials_per_config", 0)))

    results = inf_data.get("results", {})
    for model_id, lengths in results.items():
        m_short = MODEL_ID_TO_SHORT.get(model_id)
        if m_short is None:
            continue
        for length_label, metrics in lengths.items():
            l_short = LENGTH_TO_SHORT.get(length_label, length_label.title())
            tag = f"Inf{m_short}{l_short}"

            ttft = metrics.get("ttft_ms", {})
            total = metrics.get("total_ms", {})

            cs.raw(f"{tag}Ttft", fmt_int(round(ttft.get("mean", 0))))
            cs.ci_bounds(f"{tag}Ttft", ttft.get("ci_lo", 0), ttft.get("ci_hi", 0), digits=0)

            cs.raw(f"{tag}Total", f"{total.get('mean', 0):,.0f}")
            cs.ci_bounds(f"{tag}Total", total.get("ci_lo", 0), total.get("ci_hi", 0), digits=0)

            total_mean = total.get("mean", 0)
            if total_mean > 0:
                routing_pct = e2e_total_ms / total_mean * 100.0
                cs.num(f"{tag}RoutingPct", routing_pct, digits=2)


def main() -> None:
    """Load JSON files, emit ``_autogen.tex``."""
    exp_dir = Path(__file__).resolve().parent
    bench_path = exp_dir / "results" / "latency_benchmark_results.json"
    e2e_path = exp_dir / "results" / "e2e_latency_results.json"
    inf_path = exp_dir / "results" / "inference_latency_results.json"

    if not bench_path.exists():
        print(f"Error: {bench_path} not found.")
        sys.exit(1)
    if not e2e_path.exists():
        print(f"Error: {e2e_path} not found.")
        sys.exit(1)

    bench_data = load_json(bench_path)
    e2e_data = load_json(e2e_path)
    cs = build_command_set(bench_data, e2e_data)

    e2e_total_ms = e2e_data.get("stages", {}).get("total_p50_ms", 0)

    if inf_path.exists():
        inf_data = load_json(inf_path)
        _add_inference_latency_commands(cs, inf_data, e2e_total_ms)
    else:
        print(f"Warning: {inf_path} not found; skipping inference latency macros.")

    autogen_path = exp_dir / "_autogen.tex"
    cs.write(autogen_path, header="Appendix: latency benchmark")


if __name__ == "__main__":
    main()
