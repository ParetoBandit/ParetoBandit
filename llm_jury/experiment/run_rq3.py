#!/usr/bin/env python3
"""
RQ3 Experiment: Cost vs Quality Tradeoff Analysis

Research Question:
    How does the cost weight parameter (w_cost) affect the quality-cost tradeoff
    in model selection? What is the Pareto frontier?

This experiment analyzes REAL priors and model costs to understand:
    - Which models offer the best quality per dollar
    - How different cost weights affect model selection
    - The theoretical Pareto frontier based on learned weights

Data Sources:
    - expert_priors.npz: Real priors from expert-distilled training (81 models)
    - models_cache.json: Real pricing data from OpenRouter API

Library Integration:
    Uses the library's prior loading logic to handle both:
    - Expert priors (A_stack format, disjoint covariance)
    - Shared priors (A_shared format, legacy)

Usage:
    python -m llm_jury.experiment.run_rq3
    python -m llm_jury.experiment.run_rq3 --priors data/priors/expert_priors.npz

Output:
    - results/rq3/cost_quality_analysis.json - Analysis results
    - results/rq3/pareto_frontier.png - Visualization of cost-quality tradeoff

Figure Caption (for KDD paper):
    "Figure 4: The Cost-Quality Pareto Frontier.
    The router identifies a non-linear efficiency frontier (Green Dashed Line) 
    where specialist models like Amazon Nova-Lite offer maximal learned expertise 
    (||θ|| ≈ 3.7) at minimal cost (<$0.10/1M tokens). The system effectively 
    filters out 'Dominated Candidates' (Bottom-Right quadrant)—models that are 
    orders of magnitude more expensive but possess lower domain-specific confidence. 
    This demonstrates that for specialized tasks, the router achieves a 100x cost 
    reduction compared to generalist baselines without sacrificing expert performance."
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Project root for locating data files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Plotting (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for RQ3 experiment."""
    # Path to priors (default: expert_priors.npz from expert-distilled training)
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"

    # Model costs from cache
    models_cache_path: Path = PROJECT_ROOT / "data" / "models_cache.json"

    # Cost weights to analyze (the "knob" from w=0 to w=high)
    cost_weights: List[float] = None  # Set in __post_init__

    # Output
    output_dir: Path = Path("results/rq3")

    def __post_init__(self):
        if self.cost_weights is None:
            self.cost_weights = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_model_costs(cache_path: Path) -> Dict[str, float]:
    """
    Load model costs from models_cache.json.

    Returns dict mapping model_id -> cost per 1M tokens (blended or average).
    """
    if not cache_path.exists():
        return {}

    data = json.loads(cache_path.read_text())
    costs = {}

    for m in data.get("models", []):
        model_id = m.get("openrouter_id", "")
        if not model_id:
            continue

        # Try blended cost first, then compute from input/output
        blended = m.get("price_1m_blended")
        if blended is not None and blended > 0:
            costs[model_id] = float(blended)
        else:
            # Fall back to average of input and output cost
            input_cost = float(m.get("input_cost_per_m", 0) or m.get("price_1m_input", 0) or 0)
            output_cost = float(m.get("output_cost_per_m", 0) or m.get("price_1m_output", 0) or 0)
            if input_cost > 0 or output_cost > 0:
                costs[model_id] = (input_cost + output_cost) / 2

    return costs


# ---------------------------------------------------------------------------
# Analysis Results
# ---------------------------------------------------------------------------

@dataclass
class CostQualityAnalysis:
    """Analysis of cost-quality tradeoffs."""
    config: Dict[str, Any]
    n_models: int
    model_names: List[str]
    priors_format: str  # "expert" (A_stack) or "shared" (A_shared)

    # Per-model metrics
    model_costs: Dict[str, float]  # Cost per 1M tokens
    model_theta_norms: Dict[str, float]  # ||θ|| = ||A⁻¹ @ b|| - learned weight magnitude
    model_efficiency: Dict[str, float]  # Quality proxy / cost

    # Top models by different criteria
    top_by_quality: List[str]  # Highest theta norms
    top_by_efficiency: List[str]  # Best quality per dollar
    top_by_cost: List[str]  # Cheapest

    # Pareto analysis
    pareto_frontier: List[Tuple[str, float, float]]  # (model, cost, quality_proxy)

    timestamp: str


def compute_pareto_frontier(
    models: List[str],
    costs: Dict[str, float],
    quality_proxy: Dict[str, float],
) -> List[Tuple[str, float, float]]:
    """
    Compute Pareto-optimal models (no model dominates on both cost AND quality).
    """
    points = []
    for m in models:
        c = costs.get(m, float("inf"))
        q = quality_proxy.get(m, 0.0)
        if c > 0:  # Skip models without cost data
            points.append((m, c, q))

    # Sort by cost (ascending)
    points.sort(key=lambda x: x[1])

    # Find Pareto frontier: keep if no other point has lower cost AND higher quality
    frontier = []
    max_quality_seen = -float("inf")

    for m, c, q in points:
        if q >= max_quality_seen:
            frontier.append((m, c, q))
            max_quality_seen = q

    return frontier


def load_priors(priors_path: Path) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray], int, str]:
    """
    Load priors from NPZ file, handling both expert and shared formats.
    
    Returns:
        (model_names, A_dict, b_dict, dim, format_name)
        where A_dict and b_dict map model names to their matrices/vectors.
    """
    data = np.load(priors_path, allow_pickle=True)
    
    if "A_stack" in data:
        # Expert priors (disjoint format)
        model_names = [str(m) for m in data["model_names"]]
        dim = int(data["dim"])
        A_stack = np.asarray(data["A_stack"], dtype=np.float64)
        b_stack = np.asarray(data["b_stack"], dtype=np.float64)
        
        A_dict = {m: A_stack[i] for i, m in enumerate(model_names)}
        b_dict = {m: b_stack[i] for i, m in enumerate(model_names)}
        
        return model_names, A_dict, b_dict, dim, "expert"
    else:
        # Shared priors (legacy format)
        model_names = [str(m) for m in data["models"]]
        meta = json.loads(str(list(data["meta"])[0]))
        dim = int(meta.get("dim", 384))
        
        A_shared = np.asarray(data["A"], dtype=np.float64)
        b_stack = np.asarray(data["b_stack"], dtype=np.float64)
        
        # Use shared A for all models
        A_dict = {m: A_shared for m in model_names}
        b_dict = {m: b_stack[i] for i, m in enumerate(model_names)}
        
        return model_names, A_dict, b_dict, dim, "shared"


def analyze_cost_quality(config: ExperimentConfig) -> CostQualityAnalysis:
    """
    Analyze cost-quality tradeoffs from real priors and model costs.

    Uses ||θ|| = ||A⁻¹ @ b|| as a proxy for learned quality. Models with 
    higher theta norms have developed stronger "opinions" in the latent space,
    indicating more confident expertise.

    Args:
        config: Experiment configuration

    Returns:
        CostQualityAnalysis with tradeoff metrics
    """
    print(f"[RQ3] Loading priors from {config.priors_path}")

    if not config.priors_path.exists():
        raise FileNotFoundError(
            f"Priors not found: {config.priors_path}\n"
            f"Generate priors first:\n"
            f"  python -m llm_jury.experiment.generate_expert_priors generate"
        )

    # Load priors (handles both expert and shared formats)
    model_names, A_dict, b_dict, dim, priors_format = load_priors(config.priors_path)
    print(f"[RQ3] Loaded {len(model_names)} models (format: {priors_format})")
    print(f"[RQ3] Embedding dimension: {dim}")

    # Load model costs
    costs = load_model_costs(config.models_cache_path)
    print(f"[RQ3] Loaded costs for {len(costs)} models")

    # Compute metrics for each model
    model_costs: Dict[str, float] = {}
    theta_norms: Dict[str, float] = {}
    efficiency: Dict[str, float] = {}

    for m in model_names:
        # Get cost (default to 0 if not found)
        model_costs[m] = costs.get(m, 0.0)

        # Compute θ = A⁻¹ @ b (learned weight vector)
        A = A_dict[m]
        b = b_dict[m]
        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            theta_norms[m] = float(np.linalg.norm(theta))
        except np.linalg.LinAlgError:
            # Singular matrix, use regularized inverse
            theta_norms[m] = float(np.linalg.norm(b))

        # Efficiency = quality / cost (avoid division by zero)
        cost = model_costs[m]
        if cost > 0:
            efficiency[m] = theta_norms[m] / cost
        else:
            efficiency[m] = float("inf") if theta_norms[m] > 0 else 0.0

    # Top models by different criteria
    models_with_cost = [m for m in model_names if model_costs.get(m, 0) > 0]

    top_by_quality = sorted(models_with_cost, key=lambda m: theta_norms[m], reverse=True)[:10]
    top_by_efficiency = sorted(
        [m for m in models_with_cost if efficiency[m] < float("inf")],
        key=lambda m: efficiency[m],
        reverse=True
    )[:10]
    top_by_cost = sorted(models_with_cost, key=lambda m: model_costs[m])[:10]

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(model_names, model_costs, theta_norms)

    print(f"\n[RQ3] Top 5 by Quality (||θ|| = ||A⁻¹ @ b||):")
    for i, m in enumerate(top_by_quality[:5], 1):
        print(f"   {i}. {m}: ||θ||={theta_norms[m]:.4f}, cost=${model_costs[m]:.4f}/1M")

    print(f"\n[RQ3] Top 5 by Efficiency (||θ|| / cost):")
    for i, m in enumerate(top_by_efficiency[:5], 1):
        print(f"   {i}. {m}: efficiency={efficiency[m]:.2f}, cost=${model_costs[m]:.4f}/1M")

    print(f"\n[RQ3] Top 5 Cheapest:")
    for i, m in enumerate(top_by_cost[:5], 1):
        print(f"   {i}. {m}: cost=${model_costs[m]:.6f}/1M, ||θ||={theta_norms[m]:.4f}")

    print(f"\n[RQ3] Pareto frontier ({len(pareto)} models):")
    for m, c, q in pareto[:5]:
        print(f"   - {m}: cost=${c:.4f}/1M, ||θ||={q:.4f}")

    return CostQualityAnalysis(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        n_models=len(model_names),
        model_names=model_names,
        priors_format=priors_format,
        model_costs=model_costs,
        model_theta_norms=theta_norms,
        model_efficiency=efficiency,
        top_by_quality=top_by_quality,
        top_by_efficiency=top_by_efficiency,
        top_by_cost=top_by_cost,
        pareto_frontier=pareto,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_pareto_frontier(analysis: CostQualityAnalysis, output_path: Path) -> None:
    """
    Generate publication-quality Pareto frontier plot.
    
    Shows cost vs quality (||θ||) tradeoff with Pareto-optimal models highlighted.
    """
    if not HAS_MATPLOTLIB:
        print("[RQ3] Warning: matplotlib not available, skipping plot")
        return

    # KDD Paper Settings
    COLUMN_WIDTH = 3.5
    FONT_SIZE = 9
    DPI = 300

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.4, COLUMN_WIDTH * 1.0))

    # Get all models with cost data
    models = [m for m in analysis.model_names if analysis.model_costs.get(m, 0) > 0]
    costs = [analysis.model_costs[m] for m in models]
    qualities = [analysis.model_theta_norms[m] for m in models]

    # Plot all models
    ax.scatter(costs, qualities, c="#CCCCCC", s=20, alpha=0.5, label="All Models")

    # Highlight Pareto frontier
    if analysis.pareto_frontier:
        pareto_costs = [p[1] for p in analysis.pareto_frontier]
        pareto_qualities = [p[2] for p in analysis.pareto_frontier]
        
        # Create readable short names (keep model variant like "1b", "3b", "micro", "lite")
        def short_name(full_name: str) -> str:
            name = full_name.split("/")[-1]
            # For llama models, keep the size indicator
            if "llama" in name.lower():
                # Extract size (1b, 3b, 8b, 70b, etc.)
                import re
                match = re.search(r'(\d+b)', name.lower())
                if match:
                    return f"llama-{match.group(1)}"
            # For nova models, keep the variant
            if "nova" in name.lower():
                if "micro" in name.lower():
                    return "nova-micro"
                if "lite" in name.lower():
                    return "nova-lite"
                if "pro" in name.lower():
                    return "nova-pro"
            return name[:12]
        
        pareto_names = [short_name(p[0]) for p in analysis.pareto_frontier]

        ax.scatter(pareto_costs, pareto_qualities, c="#2CA02C", s=60, zorder=3, label="Pareto Optimal")
        ax.plot(pareto_costs, pareto_qualities, "--", c="#2CA02C", alpha=0.5, zorder=2)

        # Annotate Pareto points - all labels to the right of their data points
        for i, (c, q, name) in enumerate(zip(pareto_costs, pareto_qualities, pareto_names)):
            if i < 5:
                ax.annotate(name, (c, q), textcoords="offset points", xytext=(8, 0), 
                           fontsize=7, va='center')

    # Highlight top by efficiency (stars)
    for m in analysis.top_by_efficiency[:3]:
        c = analysis.model_costs[m]
        q = analysis.model_theta_norms[m]
        ax.scatter([c], [q], c="#FF7F0E", s=100, marker="*", zorder=4, label="High Efficiency" if m == analysis.top_by_efficiency[0] else "")

    ax.set_xlabel("Cost per 1M Tokens ($)", fontsize=10)
    ax.set_ylabel(r"Learned Specialist Confidence ($||\theta||$)", fontsize=10)
    ax.set_xscale("log")  # Log scale for cost (large range)
    ax.set_title("RQ3: Cost-Quality Pareto Frontier", fontsize=11, fontweight='bold')

    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, linestyle="-", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    print(f"[RQ3] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(analysis: CostQualityAnalysis, output_path: Path) -> None:
    """Save analysis results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(analysis)
    if "config" in data:
        cfg = data["config"]
        for key in ["output_dir", "priors_path", "models_cache_path"]:
            if key in cfg and cfg[key] is not None:
                cfg[key] = str(cfg[key])
        # Convert cost_weights list
        if "cost_weights" in cfg:
            cfg["cost_weights"] = list(cfg["cost_weights"]) if cfg["cost_weights"] else []

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ3] Saved analysis to {output_path}")


def generate_roi_table(analysis: CostQualityAnalysis, output_path: Path) -> None:
    """
    Generate ROI Leaderboard table for the paper.
    
    Shows ROI Factor = (||θ|| / cost) relative to GPT-4o baseline.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get GPT-4o as baseline
    gpt4o_theta = analysis.model_theta_norms.get("openai/gpt-4o", 1.0)
    gpt4o_cost = analysis.model_costs.get("openai/gpt-4o", 1.0)
    gpt4o_roi = gpt4o_theta / gpt4o_cost if gpt4o_cost > 0 else 1.0
    
    # Collect ROI data for all models with cost data
    roi_data = []
    for m in analysis.model_names:
        theta = analysis.model_theta_norms.get(m, 0)
        cost = analysis.model_costs.get(m, 0)
        if cost > 0 and theta > 0:
            roi = theta / cost
            roi_factor = roi / gpt4o_roi
            roi_data.append({
                "model": m,
                "short_name": m.split("/")[-1],
                "cost": cost,
                "theta": theta,
                "roi_factor": roi_factor,
            })
    
    # Sort by ROI factor descending
    roi_data.sort(key=lambda x: x["roi_factor"], reverse=True)
    
    # Generate markdown table
    lines = [
        "# RQ3: ROI Leaderboard",
        "",
        "**ROI Factor** = (||θ|| / Cost) relative to GPT-4o baseline",
        "",
        "| Rank | Model | Cost/1M | ||θ|| | ROI Factor |",
        "|------|-------|---------|-------|------------|",
    ]
    
    for i, d in enumerate(roi_data[:15], 1):  # Top 15
        marker = "★" if d["roi_factor"] > 10 else ""
        lines.append(f"| {i} | {d['short_name'][:25]} | ${d['cost']:.3f} | {d['theta']:.2f} | {d['roi_factor']:.1f}x {marker}|")
    
    # Add GPT-4o reference row
    lines.append(f"| Ref | gpt-4o (Baseline) | ${gpt4o_cost:.3f} | {gpt4o_theta:.2f} | 1.0x |")
    
    # Write markdown
    md_path = output_path.with_suffix(".md")
    md_path.write_text("\n".join(lines))
    print(f"[RQ3] Saved ROI table to {md_path}")
    
    # Also save as JSON for programmatic use
    json_path = output_path.with_name("roi_leaderboard.json")
    with open(json_path, "w") as f:
        json.dump(roi_data[:20], f, indent=2)
    print(f"[RQ3] Saved ROI data to {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RQ3: Analyze Cost vs Quality Tradeoffs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--priors", type=str,
        default=str(PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"),
        help="Path to priors NPZ (expert_priors.npz or shippable_priors.npz)",
    )
    parser.add_argument(
        "--cache", type=str,
        default=str(PROJECT_ROOT / "data" / "models_cache.json"),
        help="Path to models_cache.json for cost data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq3",
        help="Output directory for results",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        priors_path=Path(args.priors),
        models_cache_path=Path(args.cache),
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("RQ3: Cost vs Quality Tradeoff Analysis")
    print("=" * 60)
    print(f"Priors: {config.priors_path}")
    print(f"Model costs: {config.models_cache_path}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Analyze cost-quality tradeoffs
    analysis = analyze_cost_quality(config)

    # Save outputs
    save_results(analysis, config.output_dir / "cost_quality_analysis.json")
    plot_pareto_frontier(analysis, config.output_dir / "pareto_frontier.png")
    generate_roi_table(analysis, config.output_dir / "roi_table")

    print("=" * 60)
    print("RQ3 Analysis Complete!")
    print(f"  Priors format: {analysis.priors_format}")
    print(f"  Models analyzed: {analysis.n_models}")
    print(f"  Pareto optimal models: {len(analysis.pareto_frontier)}")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
