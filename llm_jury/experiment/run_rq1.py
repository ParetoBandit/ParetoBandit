#!/usr/bin/env python3
"""
RQ1 Experiment: The "Shippable Brain" Advantage

Research Question:
    Does pre-training on clustered prompt archetypes reduce regret compared to
    a cold-start bandit during user deployment?

This experiment loads REAL priors from the archetype grid and analyzes them.
For actual regret comparison, integrate with production traffic or use 
historical logs with ground-truth quality labels.

Usage:
    python -m llm_jury.experiment.run_rq1
    python -m llm_jury.experiment.run_rq1 --priors data/priors/shippable_priors.npz

Output:
    - results/rq1/prior_analysis.json - Analysis of loaded priors
    - results/rq1/prior_weights.png - Visualization of learned weights
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Use project's actual bandit implementation
from llm_jury.async_bandit.bandit_router import SharedCovarianceLinUCBPolicy

# Project root for locating data files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Plotting (optional, for headless servers)
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
    """Configuration for RQ1 experiment."""
    # REQUIRED: Path to shippable priors from archetype grid
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"

    # Output
    output_dir: Path = Path("results/rq1")


# ---------------------------------------------------------------------------
# Analysis Results
# ---------------------------------------------------------------------------

@dataclass
class PriorAnalysis:
    """Analysis of loaded priors."""
    config: Dict[str, Any]
    n_models: int
    model_names: List[str]
    embedding_dim: int
    alpha: float
    # Per-model statistics
    model_update_counts: Dict[str, int]
    model_weight_norms: Dict[str, float]
    # Shared covariance analysis
    covariance_condition_number: float
    covariance_trace: float
    # Top models by activity
    top_models_by_updates: List[str]
    timestamp: str


def analyze_priors(config: ExperimentConfig) -> PriorAnalysis:
    """
    Load and analyze the shippable priors.

    This provides insight into what the bandit has learned without
    requiring synthetic simulation.

    Args:
        config: Experiment configuration

    Returns:
        PriorAnalysis with statistics about the loaded priors
    """
    print(f"[RQ1] Loading priors from {config.priors_path}")

    if not config.priors_path.exists():
        raise FileNotFoundError(
            f"Priors not found: {config.priors_path}\n"
            f"Run the archetype grid first:\n"
            f"  python -m llm_jury.async_bandit.archetype_grid_dense_run"
        )

    # Load priors
    policy = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(config.priors_path)

    print(f"[RQ1] Loaded {len(policy.models)} models")
    print(f"[RQ1] Embedding dimension: {policy.dim}")
    print(f"[RQ1] Alpha: {policy.alpha}")

    # Analyze update counts per model
    update_counts = {m: policy._updates.get(m, 0) for m in policy.models}
    total_updates = sum(update_counts.values())
    print(f"[RQ1] Total updates across all models: {total_updates}")

    # Analyze weight norms (b vectors)
    weight_norms = {}
    for m in policy.models:
        b_vec = policy.b.get(m, np.zeros(policy.dim))
        weight_norms[m] = float(np.linalg.norm(b_vec))

    # Analyze shared covariance matrix
    A = policy.A
    try:
        eigenvalues = np.linalg.eigvalsh(A)
        condition_number = float(eigenvalues.max() / max(eigenvalues.min(), 1e-10))
    except Exception:
        condition_number = float("inf")

    covariance_trace = float(np.trace(A))

    # Top models by update count
    sorted_models = sorted(policy.models, key=lambda m: update_counts[m], reverse=True)
    top_models = sorted_models[:10]

    print(f"\n[RQ1] Top 10 models by update count:")
    for i, m in enumerate(top_models, 1):
        print(f"   {i}. {m}: {update_counts[m]} updates, ||b||={weight_norms[m]:.4f}")

    print(f"\n[RQ1] Covariance matrix analysis:")
    print(f"   Condition number: {condition_number:.2e}")
    print(f"   Trace: {covariance_trace:.2f}")

    return PriorAnalysis(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        n_models=len(policy.models),
        model_names=policy.models,
        embedding_dim=policy.dim,
        alpha=policy.alpha,
        model_update_counts=update_counts,
        model_weight_norms=weight_norms,
        covariance_condition_number=condition_number,
        covariance_trace=covariance_trace,
        top_models_by_updates=top_models,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_prior_analysis(analysis: PriorAnalysis, output_path: Path) -> None:
    """
    Generate publication-quality visualization of prior weights.
    """
    if not HAS_MATPLOTLIB:
        print("[RQ1] Warning: matplotlib not available, skipping plot")
        return

    # KDD Paper Settings
    COLUMN_WIDTH = 3.5
    FONT_SIZE = 9
    DPI = 300

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 2,
        "ytick.labelsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, COLUMN_WIDTH * 0.6))

    # Plot 1: Update counts (top 15 models)
    top_models = analysis.top_models_by_updates[:15]
    counts = [analysis.model_update_counts[m] for m in top_models]
    short_names = [m.split("/")[-1][:12] for m in top_models]

    ax1.barh(range(len(top_models)), counts, color="#1F77B4")
    ax1.set_yticks(range(len(top_models)))
    ax1.set_yticklabels(short_names)
    ax1.set_xlabel("Update Count")
    ax1.set_title("Training Activity by Model")
    ax1.invert_yaxis()

    # Plot 2: Weight norms
    norms = [analysis.model_weight_norms[m] for m in top_models]
    ax2.barh(range(len(top_models)), norms, color="#2CA02C")
    ax2.set_yticks(range(len(top_models)))
    ax2.set_yticklabels(short_names)
    ax2.set_xlabel("Weight Norm ||b||")
    ax2.set_title("Learned Weight Magnitudes")
    ax2.invert_yaxis()

    plt.tight_layout(pad=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    print(f"[RQ1] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(analysis: PriorAnalysis, output_path: Path) -> None:
    """Save analysis results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(analysis)
    if "config" in data:
        cfg = data["config"]
        for key in ["output_dir", "priors_path"]:
            if key in cfg and cfg[key] is not None:
                cfg[key] = str(cfg[key])

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ1] Saved analysis to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RQ1: Analyze Shippable Priors (The 'Shippable Brain')",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--priors", type=str,
        default=str(PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"),
        help="Path to shippable_priors.npz from archetype grid",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq1",
        help="Output directory for results",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        priors_path=Path(args.priors),
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("RQ1: The 'Shippable Brain' - Prior Analysis")
    print("=" * 60)
    print(f"Priors: {config.priors_path}")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Analyze priors
    analysis = analyze_priors(config)

    # Save outputs
    save_results(analysis, config.output_dir / "prior_analysis.json")
    plot_prior_analysis(analysis, config.output_dir / "prior_weights.png")

    print("=" * 60)
    print("Analysis complete!")
    print(f"  Models in priors: {analysis.n_models}")
    print(f"  Embedding dim: {analysis.embedding_dim}")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
