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
    - Load priors from archetype grid
    - Analyze which models have learned weights for different embedding regions
    - Identify potential specialists vs generalists based on weight patterns

Usage:
    python -m llm_jury.experiment.run_rq2
    python -m llm_jury.experiment.run_rq2 --priors data/priors/shippable_priors.npz

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
from typing import Any, Dict, List, Tuple

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
    """Configuration for RQ2 experiment."""
    # REQUIRED: Path to shippable priors from archetype grid
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"

    # Number of top models to analyze in detail
    n_top_models: int = 20

    # Output
    output_dir: Path = Path("results/rq2")


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

    # Per-model specialization metrics
    model_weight_directions: Dict[str, List[float]]  # Top principal components
    model_weight_entropies: Dict[str, float]  # Weight distribution entropy
    model_update_counts: Dict[str, int]

    # Clustering analysis
    model_similarity_matrix: List[List[float]]  # Cosine similarity of b vectors
    specialist_candidates: List[str]  # Models with focused weight distributions
    generalist_candidates: List[str]  # Models with spread weight distributions

    timestamp: str


def compute_weight_entropy(b_vec: np.ndarray) -> float:
    """
    Compute entropy of weight distribution.

    Lower entropy = more specialized (weights concentrated)
    Higher entropy = more generalist (weights spread out)
    """
    # Normalize to probability-like distribution
    abs_weights = np.abs(b_vec) + 1e-10
    probs = abs_weights / abs_weights.sum()

    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(b_vec))
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
            f"Run the archetype grid first:\n"
            f"  python -m llm_jury.async_bandit.archetype_grid_dense_run"
        )

    # Load priors
    policy = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(config.priors_path)

    print(f"[RQ2] Loaded {len(policy.models)} models")
    print(f"[RQ2] Embedding dimension: {policy.dim}")

    # Get models with most updates (most informative)
    update_counts = {m: policy._updates.get(m, 0) for m in policy.models}
    sorted_models = sorted(policy.models, key=lambda m: update_counts[m], reverse=True)
    top_models = sorted_models[:config.n_top_models]

    print(f"[RQ2] Analyzing top {len(top_models)} models by training activity")

    # Analyze weight vectors
    weight_directions: Dict[str, List[float]] = {}
    weight_entropies: Dict[str, float] = {}
    b_vectors: Dict[str, np.ndarray] = {}

    for m in top_models:
        b_vec = policy.b.get(m, np.zeros(policy.dim))
        b_vectors[m] = b_vec

        # Get top 5 principal directions (indices of largest absolute weights)
        top_indices = np.argsort(np.abs(b_vec))[-5:][::-1]
        weight_directions[m] = [float(b_vec[i]) for i in top_indices]

        # Compute specialization entropy
        weight_entropies[m] = compute_weight_entropy(b_vec)

    # Compute similarity matrix (cosine similarity of b vectors)
    similarity_matrix: List[List[float]] = []
    for m1 in top_models:
        row = []
        b1 = b_vectors[m1]
        norm1 = np.linalg.norm(b1)
        for m2 in top_models:
            b2 = b_vectors[m2]
            norm2 = np.linalg.norm(b2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                sim = float(np.dot(b1, b2) / (norm1 * norm2))
            else:
                sim = 0.0
            row.append(sim)
        similarity_matrix.append(row)

    # Identify specialists (low entropy) and generalists (high entropy)
    entropy_sorted = sorted(top_models, key=lambda m: weight_entropies[m])
    specialist_candidates = entropy_sorted[:5]  # Lowest entropy = most specialized
    generalist_candidates = entropy_sorted[-5:]  # Highest entropy = most general

    print(f"\n[RQ2] Specialization Analysis:")
    print(f"   Specialist candidates (low entropy = focused weights):")
    for m in specialist_candidates:
        print(f"      - {m}: entropy={weight_entropies[m]:.4f}")

    print(f"\n   Generalist candidates (high entropy = spread weights):")
    for m in generalist_candidates:
        print(f"      - {m}: entropy={weight_entropies[m]:.4f}")

    return SpecializationAnalysis(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        n_models=len(policy.models),
        model_names=top_models,
        embedding_dim=policy.dim,
        model_weight_directions=weight_directions,
        model_weight_entropies=weight_entropies,
        model_update_counts={m: update_counts[m] for m in top_models},
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH * 2, COLUMN_WIDTH * 0.7))

    models = analysis.model_names[:15]  # Top 15 for readability
    short_names = [m.split("/")[-1][:10] for m in models]

    # Plot 1: Entropy (specialization measure)
    entropies = [analysis.model_weight_entropies[m] for m in models]
    colors = ["#D62728" if m in analysis.specialist_candidates else
              "#2CA02C" if m in analysis.generalist_candidates else
              "#1F77B4" for m in models]

    ax1.barh(range(len(models)), entropies, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(short_names)
    ax1.set_xlabel("Weight Entropy (lower = more specialized)")
    ax1.set_title("Model Specialization")
    ax1.invert_yaxis()
    ax1.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Model similarity heatmap
    sim_matrix = np.array(analysis.model_similarity_matrix[:15])[:, :15]
    im = ax2.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(models)))
    ax2.set_yticks(range(len(models)))
    ax2.set_xticklabels(short_names, rotation=45, ha="right")
    ax2.set_yticklabels(short_names)
    ax2.set_title("Weight Similarity")
    plt.colorbar(im, ax=ax2, shrink=0.8)

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
    if "config" in data:
        cfg = data["config"]
        for key in ["output_dir", "priors_path"]:
            if key in cfg and cfg[key] is not None:
                cfg[key] = str(cfg[key])

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
        default=str(PROJECT_ROOT / "data" / "priors" / "shippable_priors.npz"),
        help="Path to shippable_priors.npz from archetype grid",
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
    print(f"  Models analyzed: {len(analysis.model_names)}")
    print(f"  Specialist candidates: {len(analysis.specialist_candidates)}")
    print(f"  Generalist candidates: {len(analysis.generalist_candidates)}")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
