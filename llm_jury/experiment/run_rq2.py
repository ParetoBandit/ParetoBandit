#!/usr/bin/env python3
"""
RQ2 Experiment: Local Adaptation Analysis

Research Question:
    Can the bandit discover that a "niche" model outperforms on a specific
    user distribution, even when generic benchmarks say otherwise?

This experiment analyzes the REAL priors to understand model specialization.
For actual adaptation testing, integrate with production traffic that has
distribution shifts (e.g., user switches from Python to KQL queries).

Analysis Approach:
    - Load expert-distilled priors from archetype grid
    - Analyze which models have learned weights for different embedding regions
    - Identify potential specialists vs generalists based on weight patterns
    - Supports both expert (A_stack) and legacy shared (A_shared) formats

Usage:
    python -m llm_jury.experiment.run_rq2
    python -m llm_jury.experiment.run_rq2 --priors data/priors/expert_priors.npz

Output:
    - results/rq2/specialization_analysis.json - Model specialization metrics
    - results/rq2/model_coverage.png - Visualization of model weight coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from llm_jury.async_bandit.bandit_router import (
    DisjointLinUCBPolicy,
    SharedCovarianceLinUCBPolicy,
)

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
    """Configuration for RQ2 experiment."""
    # Use expert priors by default (generated via Expert Distillation)
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"

    # Number of top models to analyze in detail
    n_top_models: int = 20

    # Output
    output_dir: Path = Path("results/rq2")


# ---------------------------------------------------------------------------
# Prior Loading (supports both expert and shared formats)
# ---------------------------------------------------------------------------

def load_priors(priors_path: Path) -> DisjointLinUCBPolicy:
    """
    Load priors from NPZ file, supporting both formats:
    - Expert: A_stack (N, D, D) and b_stack (N, D)
    - Shared: A_shared (D, D) and b vectors per model
    
    Returns a DisjointLinUCBPolicy for unified analysis.
    """
    data = np.load(priors_path, allow_pickle=True)
    
    if "A_stack" in data:
        # Expert-distilled priors (already disjoint format)
        model_names = [str(m) for m in data["model_names"]]
        dim = int(data["dim"])
        alpha = float(data.get("alpha", 0.5))
        A_stack = np.asarray(data["A_stack"], dtype=np.float64)
        b_stack = np.asarray(data["b_stack"], dtype=np.float64)
        
        policy = DisjointLinUCBPolicy(model_names, dim=dim, alpha=alpha)
        for i, m in enumerate(model_names):
            policy.A[m] = A_stack[i]
            policy.b[m] = b_stack[i]
            policy.A_inv[m] = np.linalg.inv(A_stack[i])
        
        return policy
    else:
        # Shared covariance priors (need inflation)
        shared = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(priors_path)
        policy = DisjointLinUCBPolicy(shared.models, dim=shared.dim, alpha=0.5)
        
        for m in shared.models:
            policy.A[m] = np.asarray(shared.A, dtype=np.float64).copy()
            policy.b[m] = np.asarray(shared.b.get(m, np.zeros(shared.dim)), dtype=np.float64)
            policy.A_inv[m] = np.linalg.inv(policy.A[m])
        
        return policy


# ---------------------------------------------------------------------------
# Analysis Results
# ---------------------------------------------------------------------------

@dataclass
class SpecializationAnalysis:
    """Analysis of model specialization from priors."""
    config: Dict[str, Any]
    n_models: int
    model_names: List[str]
    embedding_dim: int
    priors_format: str  # "expert" or "shared"

    # Per-model specialization metrics
    model_theta_norms: Dict[str, float]  # ||θ|| = ||A^-1 @ b|| (learned weight magnitude)
    model_weight_entropies: Dict[str, float]  # Weight distribution entropy
    model_confidence_scores: Dict[str, float]  # Trace of A (higher = more updates)

    # Clustering analysis
    model_similarity_matrix: List[List[float]]  # Cosine similarity of θ vectors
    specialist_candidates: List[str]  # Models with focused weight distributions
    generalist_candidates: List[str]  # Models with spread weight distributions

    timestamp: str


def compute_weight_entropy(theta: np.ndarray) -> float:
    """
    Compute entropy of weight distribution.

    Lower entropy = more specialized (weights concentrated)
    Higher entropy = more generalist (weights spread out)
    """
    # Normalize to probability-like distribution
    abs_weights = np.abs(theta) + 1e-10
    probs = abs_weights / abs_weights.sum()

    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(theta))
    return float(entropy / max_entropy)


def analyze_specialization(config: ExperimentConfig) -> SpecializationAnalysis:
    """
    Analyze model specialization patterns from the priors.

    This identifies which models have learned specialized weights
    (potential niche experts) vs generalized weights.

    Args:
        config: Experiment configuration

    Returns:
        SpecializationAnalysis with model specialization metrics
    """
    print(f"[RQ2] Loading priors from {config.priors_path}")

    if not config.priors_path.exists():
        raise FileNotFoundError(
            f"Priors not found: {config.priors_path}\n"
            f"Generate expert priors first:\n"
            f"  python -m llm_jury.experiment.generate_expert_priors generate"
        )

    # Detect format
    data = np.load(config.priors_path, allow_pickle=True)
    priors_format = "expert" if "A_stack" in data else "shared"
    print(f"[RQ2] Detected priors format: {priors_format}")

    # Load priors as DisjointLinUCBPolicy
    policy = load_priors(config.priors_path)

    print(f"[RQ2] Loaded {len(policy.models)} models")
    print(f"[RQ2] Embedding dimension: {policy.dim}")

    # Compute θ = A^-1 @ b for each model (the learned weight vector)
    thetas: Dict[str, np.ndarray] = {}
    theta_norms: Dict[str, float] = {}
    confidence_scores: Dict[str, float] = {}
    
    for m in policy.models:
        theta = policy.A_inv[m] @ policy.b[m]
        thetas[m] = theta
        theta_norms[m] = float(np.linalg.norm(theta))
        # Trace of A indicates total updates (higher = more confident)
        confidence_scores[m] = float(np.trace(policy.A[m]))

    # Sort by theta norm (models with strongest learned preferences)
    sorted_models = sorted(policy.models, key=lambda m: theta_norms[m], reverse=True)
    top_models = sorted_models[:config.n_top_models]

    print(f"[RQ2] Analyzing top {len(top_models)} models by learned weight magnitude")

    # Compute entropies for top models
    weight_entropies: Dict[str, float] = {}
    for m in top_models:
        weight_entropies[m] = compute_weight_entropy(thetas[m])

    # Compute similarity matrix (cosine similarity of θ vectors)
    similarity_matrix: List[List[float]] = []
    for m1 in top_models:
        row = []
        theta1 = thetas[m1]
        norm1 = np.linalg.norm(theta1)
        for m2 in top_models:
            theta2 = thetas[m2]
            norm2 = np.linalg.norm(theta2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                sim = float(np.dot(theta1, theta2) / (norm1 * norm2))
            else:
                sim = 0.0
            row.append(sim)
        similarity_matrix.append(row)

    # Identify specialists (low entropy) and generalists (high entropy)
    entropy_sorted = sorted(top_models, key=lambda m: weight_entropies[m])
    specialist_candidates = entropy_sorted[:5]  # Lowest entropy = most specialized
    generalist_candidates = entropy_sorted[-5:]  # Highest entropy = most general

    print(f"\n[RQ2] Top Models by Learned Weight Magnitude ||θ||:")
    for i, m in enumerate(top_models[:10]):
        print(f"   {i+1}. {m}: ||θ||={theta_norms[m]:.4f}, entropy={weight_entropies[m]:.4f}")

    print(f"\n[RQ2] Specialist Candidates (low entropy = focused weights):")
    for m in specialist_candidates:
        print(f"      - {m}: entropy={weight_entropies[m]:.4f}")

    print(f"\n[RQ2] Generalist Candidates (high entropy = spread weights):")
    for m in generalist_candidates:
        print(f"      - {m}: entropy={weight_entropies[m]:.4f}")

    return SpecializationAnalysis(
        config={k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        n_models=len(policy.models),
        model_names=top_models,
        embedding_dim=policy.dim,
        priors_format=priors_format,
        model_theta_norms={m: theta_norms[m] for m in top_models},
        model_weight_entropies=weight_entropies,
        model_confidence_scores={m: confidence_scores[m] for m in top_models},
        model_similarity_matrix=similarity_matrix,
        specialist_candidates=specialist_candidates,
        generalist_candidates=generalist_candidates,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_specialization(analysis: SpecializationAnalysis, output_path: Path) -> None:
    """
    Generate publication-quality visualization of model specialization.
    """
    if not HAS_MATPLOTLIB:
        print("[RQ2] Warning: matplotlib not available, skipping plot")
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
        "ytick.labelsize": FONT_SIZE - 2,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })

    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH * 2.5, COLUMN_WIDTH * 0.8))
    ax1, ax2, ax3 = axes

    models = analysis.model_names[:15]  # Top 15 for readability
    short_names = [m.split("/")[-1][:12] for m in models]

    # Plot 1: Theta Norm (learned weight magnitude)
    norms = [analysis.model_theta_norms[m] for m in models]
    colors1 = ["#1F77B4"] * len(models)
    ax1.barh(range(len(models)), norms, color=colors1)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(short_names)
    ax1.set_xlabel("||θ|| (learned weight magnitude)")
    ax1.set_title("Model Expertise")
    ax1.invert_yaxis()

    # Plot 2: Entropy (specialization measure)
    entropies = [analysis.model_weight_entropies[m] for m in models]
    colors2 = ["#D62728" if m in analysis.specialist_candidates else
               "#2CA02C" if m in analysis.generalist_candidates else
               "#7F7F7F" for m in models]

    ax2.barh(range(len(models)), entropies, color=colors2)
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(short_names)
    ax2.set_xlabel("Weight Entropy")
    ax2.set_title("Specialization")
    ax2.invert_yaxis()
    ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # Plot 3: Model similarity heatmap
    sim_matrix = np.array(analysis.model_similarity_matrix[:15])[:, :15]
    im = ax3.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(models)))
    ax3.set_yticks(range(len(models)))
    ax3.set_xticklabels(short_names, rotation=45, ha="right")
    ax3.set_yticklabels(short_names)
    ax3.set_title("θ Similarity")
    plt.colorbar(im, ax=ax3, shrink=0.8)

    plt.tight_layout(pad=1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    print(f"[RQ2] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(analysis: SpecializationAnalysis, output_path: Path) -> None:
    """Save analysis results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(analysis)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ2] Saved analysis to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RQ2: Analyze Model Specialization from Priors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--priors", type=str,
        default=str(PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"),
        help="Path to priors file (supports expert_priors.npz or shippable_priors.npz)",
    )
    parser.add_argument(
        "--n-top-models", type=int, default=20,
        help="Number of top models to analyze in detail",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/rq2",
        help="Output directory for results",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        priors_path=Path(args.priors),
        n_top_models=args.n_top_models,
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("RQ2: Local Adaptation - Specialization Analysis")
    print("=" * 60)
    print(f"Priors: {config.priors_path}")
    print(f"Analyzing top: {config.n_top_models} models")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Analyze specialization
    analysis = analyze_specialization(config)

    # Save outputs
    save_results(analysis, config.output_dir / "specialization_analysis.json")
    plot_specialization(analysis, config.output_dir / "model_coverage.png")

    print("=" * 60)
    print("Analysis complete!")
    print(f"  Priors format: {analysis.priors_format}")
    print(f"  Models analyzed: {len(analysis.model_names)}")
    print(f"  Specialist candidates: {len(analysis.specialist_candidates)}")
    print(f"  Generalist candidates: {len(analysis.generalist_candidates)}")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
