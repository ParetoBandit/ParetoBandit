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

Usage:
    python -m llm_jury.experiment.run_rq3
    python -m llm_jury.experiment.run_rq3 --priors data/priors/shippable_priors.npz

Output:
    - results/rq3/cost_quality_analysis.json - Analysis results
    - results/rq3/pareto_frontier.png - Visualization of cost-quality tradeoff
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

# Use project's actual bandit implementation
from llm_jury.async_bandit.bandit_router import SharedCovarianceLinUCBPolicy

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
    # REQUIRED: Path to shippable priors from archetype grid
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"

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

    Returns dict mapping model_id -> cost per 1M tokens (input + output average).
    """
    if not cache_path.exists():
        return {}

    data = json.loads(cache_path.read_text())
    costs = {}

    for m in data.get("models", []):
        model_id = m.get("openrouter_id", "")
        if not model_id:
            continue

        # Get pricing info
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", 0) or 0)
        completion_cost = float(pricing.get("completion", 0) or 0)

        # Average cost per token (simplified)
        avg_cost = (prompt_cost + completion_cost) / 2
        costs[model_id] = avg_cost

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

    # Per-model metrics
    model_costs: Dict[str, float]  # Cost per token
    model_weight_norms: Dict[str, float]  # Proxy for learned quality
    model_update_counts: Dict[str, int]  # Training activity
    model_efficiency: Dict[str, float]  # Quality proxy / cost

    # Top models by different criteria
    top_by_quality: List[str]  # Highest weight norms
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


def analyze_cost_quality(config: ExperimentConfig) -> CostQualityAnalysis:
    """
    Analyze cost-quality tradeoffs from real priors and model costs.

    Uses weight norms as a proxy for learned quality (models with higher
    weight norms have received more positive feedback during training).

    Args:
        config: Experiment configuration

    Returns:
        CostQualityAnalysis with tradeoff metrics
    """
    print(f"[RQ3] Loading priors from {config.priors_path}")

    if not config.priors_path.exists():
        raise FileNotFoundError(
            f"Priors not found: {config.priors_path}\n"
            f"Run the archetype grid first:\n"
            f"  python -m llm_jury.async_bandit.archetype_grid_dense_run"
        )

    # Load priors
    policy = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(config.priors_path)
    print(f"[RQ3] Loaded {len(policy.models)} models")

    # Load model costs
    costs = load_model_costs(config.models_cache_path)
    print(f"[RQ3] Loaded costs for {len(costs)} models")

    # Compute metrics for each model
    model_costs: Dict[str, float] = {}
    weight_norms: Dict[str, float] = {}
    update_counts: Dict[str, int] = {}
    efficiency: Dict[str, float] = {}

    for m in policy.models:
        # Get cost (default to 0 if not found)
        model_costs[m] = costs.get(m, 0.0)

        # Weight norm as quality proxy
        b_vec = policy.b.get(m, np.zeros(policy.dim))
        weight_norms[m] = float(np.linalg.norm(b_vec))

        # Update counts
        update_counts[m] = policy._updates.get(m, 0)

        # Efficiency = quality / cost (avoid division by zero)
        cost = model_costs[m]
        if cost > 0:
            efficiency[m] = weight_norms[m] / cost
        else:
            efficiency[m] = float("inf") if weight_norms[m] > 0 else 0.0

    # Top models by different criteria
    models_with_data = [m for m in policy.models if model_costs.get(m, 0) > 0]

    top_by_quality = sorted(models_with_data, key=lambda m: weight_norms[m], reverse=True)[:10]
    top_by_efficiency = sorted(
        [m for m in models_with_data if efficiency[m] < float("inf")],
        key=lambda m: efficiency[m],
        reverse=True
    )[:10]
    top_by_cost = sorted(models_with_data, key=lambda m: model_costs[m])[:10]

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(policy.models, model_costs, weight_norms)

    print(f"\n[RQ3] Top 5 by Quality (weight norm):")
    for i, m in enumerate(top_by_quality[:5], 1):
        print(f"   {i}. {m}: ||b||={weight_norms[m]:.4f}, cost=${model_costs[m]:.6f}")

    print(f"\n[RQ3] Top 5 by Efficiency (quality/cost):")
    for i, m in enumerate(top_by_efficiency[:5], 1):
        print(f"   {i}. {m}: efficiency={efficiency[m]:.2f}, cost=${model_costs[m]:.6f}")

    print(f"\n[RQ3] Top 5 Cheapest:")
    for i, m in enumerate(top_by_cost[:5], 1):
        print(f"   {i}. {m}: cost=${model_costs[m]:.6f}, ||b||={weight_norms[m]:.4f}")

    print(f"\n[RQ3] Pareto frontier ({len(pareto)} models):")
    for m, c, q in pareto[:5]:
        print(f"   - {m}: cost=${c:.6f}, quality={q:.4f}")

    return CostQualityAnalysis(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        n_models=len(policy.models),
        model_names=policy.models,
        model_costs=model_costs,
        model_weight_norms=weight_norms,
        model_update_counts=update_counts,
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

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 1.2, COLUMN_WIDTH * 0.9))

    # Get all models with cost data
    models = [m for m in analysis.model_names if analysis.model_costs.get(m, 0) > 0]
    costs = [analysis.model_costs[m] for m in models]
    qualities = [analysis.model_weight_norms[m] for m in models]

    # Plot all models
    ax.scatter(costs, qualities, c="#CCCCCC", s=20, alpha=0.5, label="All Models")

    # Highlight Pareto frontier
    if analysis.pareto_frontier:
        pareto_costs = [p[1] for p in analysis.pareto_frontier]
        pareto_qualities = [p[2] for p in analysis.pareto_frontier]
        pareto_names = [p[0].split("/")[-1][:8] for p in analysis.pareto_frontier]

        ax.scatter(pareto_costs, pareto_qualities, c="#2CA02C", s=60, zorder=3, label="Pareto Optimal")
        ax.plot(pareto_costs, pareto_qualities, "--", c="#2CA02C", alpha=0.5, zorder=2)

        # Annotate top Pareto points
        for i, (c, q, name) in enumerate(zip(pareto_costs[:3], pareto_qualities[:3], pareto_names[:3])):
            ax.annotate(name, (c, q), textcoords="offset points", xytext=(5, 5), fontsize=6)

    # Highlight top by efficiency
    for m in analysis.top_by_efficiency[:3]:
        c = analysis.model_costs[m]
        q = analysis.model_weight_norms[m]
        ax.scatter([c], [q], c="#FF7F0E", s=80, marker="*", zorder=4)

    ax.set_xlabel("Cost per Token ($)")
    ax.set_ylabel("Learned Weight Norm (Quality Proxy)")
    ax.set_xscale("log")  # Log scale for cost (large range)

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
        default=str(PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"),
        help="Path to shippable_priors.npz from archetype grid",
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

    print("=" * 60)
    print("Analysis complete!")
    print(f"  Models analyzed: {analysis.n_models}")
    print(f"  Pareto optimal models: {len(analysis.pareto_frontier)}")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
