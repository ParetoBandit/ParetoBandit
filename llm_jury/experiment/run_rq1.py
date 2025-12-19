#!/usr/bin/env python3
"""
RQ1 Experiment: The "Shippable Brain" Advantage

Research Question:
    Does pre-training on clustered prompt archetypes reduce regret compared to
    a cold-start bandit during user deployment?

Experiment Design:
    - Environment: Simulated LLM routing with clustered prompt distributions
    - Treatment A: Cold Start Bandit (Identity A, zero b vectors)
    - Treatment B: Warm Start Bandit (Pre-trained on public data)
    - Metric: Cumulative Regret over N user requests

Key Insights:
    - The warm-start agent should show near-zero regret from Day 1
    - The cold-start agent pays a "learning tax" before converging
    - The gap represents the business value of shippable priors

Usage:
    # Simulation mode (default)
    python -m llm_jury.experiment.run_rq1

    # With real priors (if available)
    python -m llm_jury.experiment.run_rq1 --priors data/priors/shippable_priors.npz

    # Custom configuration
    python -m llm_jury.experiment.run_rq1 --n-pretrain 1000 --n-test 5000 --seed 42

Output:
    - results/rq1/regret_curve.png - Publication-ready figure
    - results/rq1/metrics.json - Raw experimental data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use project's actual bandit implementation (same as BanditRouter uses)
from llm_jury.async_bandit.bandit_router import DisjointLinUCBPolicy

# Project root for locating data files
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Model Loading from Cache
# ---------------------------------------------------------------------------

def load_models_from_cache(cache_path: Path, max_models: int = 0) -> List[str]:
    """
    Load model IDs from the project's models_cache.json.

    This uses the same format as the production BanditRouter.

    Args:
        cache_path: Path to models_cache.json
        max_models: Maximum number of models to load (0 = all)

    Returns:
        List of OpenRouter model IDs (e.g., "anthropic/claude-3.5-haiku")
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Models cache not found: {cache_path}")

    data = json.loads(cache_path.read_text())
    models = data.get("models", [])

    # Extract OpenRouter IDs
    model_ids: List[str] = []
    for m in models:
        oid = (m or {}).get("openrouter_id")
        if isinstance(oid, str) and oid.strip():
            model_ids.append(oid.strip())

    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for mid in model_ids:
        if mid not in seen:
            seen.add(mid)
            unique.append(mid)

    if max_models > 0:
        unique = unique[:max_models]

    return unique

# Plotting (optional, for headless servers)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
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
    # Dimensions
    dim: int = 384  # Match sentence-transformers/all-MiniLM-L6-v2
    n_models: int = 0  # Number of models (0 = all from cache)
    n_clusters: int = 5  # Latent task clusters (Code, Math, Creative, Factual, Chat)

    # Model source
    models_cache: Path = PROJECT_ROOT / "data" / "models_cache.json"

    # Experiment size
    n_pretrain: int = 500  # Size of "public" pretraining set
    n_test: int = 2000  # Size of user deployment simulation

    # Bandit parameters (same defaults as BanditRouter)
    alpha: float = 0.5  # UCB exploration parameter
    ridge_lambda: float = 1.0  # Regularization
    recompute_inv_every: int = 50  # How often to recompute A^-1

    # Noise
    noise_level: float = 0.1  # Reward noise (std)

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: Path = Path("results/rq1")

    # Optional: Use real priors instead of synthetic
    priors_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Simulated Environment
# ---------------------------------------------------------------------------

class SimulatedRoutingEnvironment:
    """
    Simulates ground truth for LLM routing experiments.

    This environment models:
    - K latent task clusters (e.g., Code, Math, Creative)
    - M models with different competencies per cluster
    - Prompts sampled from cluster distributions
    - Rewards based on model-cluster affinity

    The environment is calibrated to match real-world observations:
    - Some models are specialists (high competence on 1-2 clusters)
    - Some models are generalists (moderate competence everywhere)
    - Reward noise reflects real grading uncertainty

    Model names can be provided from the project's models_cache.json to use
    real OpenRouter model IDs (e.g., "anthropic/claude-3.5-haiku").
    """

    def __init__(
        self,
        n_clusters: int,
        dim: int,
        model_names: List[str],
        noise_level: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_clusters = n_clusters
        self.dim = dim
        self.noise_level = noise_level
        self.rng = rng or np.random.default_rng()

        # Use provided model names (from models_cache.json or simulated)
        self.model_names = list(model_names)
        self.n_models = len(self.model_names)

        # Generate cluster centers in embedding space
        # These represent the "meaning" of each task type
        self.cluster_centers = self._generate_cluster_centers()

        # Generate model competencies
        # competencies[m, c] = base quality of model m on cluster c
        self.competencies = self._generate_model_competencies()

    def _generate_cluster_centers(self) -> np.ndarray:
        """Generate orthogonal-ish cluster centers."""
        # Start with random vectors
        centers = self.rng.standard_normal((self.n_clusters, self.dim))
        # Normalize to unit vectors
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(norms, 1e-8)
        return centers

    def _generate_model_competencies(self) -> np.ndarray:
        """
        Generate model competency matrix.

        Strategy:
        - First few models are "specialists" (one cluster expert)
        - Remaining models are "generalists" with moderate scores
        """
        competencies = self.rng.uniform(0.3, 0.6, (self.n_models, self.n_clusters))

        # Make some models specialists (high on one cluster)
        n_specialists = min(self.n_clusters, self.n_models)
        for i in range(n_specialists):
            competencies[i, i % self.n_clusters] = self.rng.uniform(0.85, 0.95)

        return competencies

    def sample_context(self) -> Tuple[np.ndarray, int]:
        """
        Sample a prompt embedding and its latent cluster.

        Returns:
            (context_vector, cluster_id)
        """
        # Sample cluster
        cluster_id = self.rng.integers(0, self.n_clusters)
        center = self.cluster_centers[cluster_id]

        # Add noise to create variance within cluster
        noise = self.rng.standard_normal(self.dim) * 0.15
        context = center + noise

        # Normalize
        context = context / np.linalg.norm(context)
        return context, cluster_id

    def get_reward(self, model_idx: int, cluster_id: int) -> float:
        """
        Get noisy reward for model on cluster.

        Returns:
            Reward in [0, 1] range
        """
        base = self.competencies[model_idx, cluster_id]
        noise = self.rng.standard_normal() * self.noise_level
        return float(np.clip(base + noise, 0.0, 1.0))

    def get_optimal_reward(self, cluster_id: int) -> float:
        """Get best possible reward for cluster (oracle)."""
        return float(np.max(self.competencies[:, cluster_id]))


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config: Dict[str, Any]
    regret_cold: List[float]
    regret_warm: List[float]
    final_regret_cold: float
    final_regret_warm: float
    regret_reduction_pct: float
    timestamp: str


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Run the RQ1 experiment: Cold Start vs Warm Start.

    Uses the project's DisjointLinUCBPolicy - the same algorithm
    that powers the production BanditRouter.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResults with regret curves and summary metrics
    """
    print(f"[RQ1] Starting experiment with seed={config.seed}")
    rng = np.random.default_rng(config.seed)

    # Load model names from cache (same as production BanditRouter)
    if config.models_cache.exists():
        model_names = load_models_from_cache(config.models_cache, max_models=config.n_models)
        print(f"[RQ1] Loaded {len(model_names)} models from {config.models_cache.name}")
    else:
        # Fallback to simulated names if cache doesn't exist
        model_names = [f"model_{i}" for i in range(config.n_models)]
        print(f"[RQ1] Using {len(model_names)} simulated model names (cache not found)")

    # Create environment with real model names
    print(f"[RQ1] Creating environment: {len(model_names)} models, {config.n_clusters} clusters")
    env = SimulatedRoutingEnvironment(
        n_clusters=config.n_clusters,
        dim=config.dim,
        model_names=model_names,
        noise_level=config.noise_level,
        rng=rng,
    )

    # Initialize warm-start agent using project's DisjointLinUCBPolicy
    print(f"[RQ1] Initializing warm-start agent (DisjointLinUCBPolicy)...")
    agent_warm = DisjointLinUCBPolicy(
        model_names=env.model_names,
        dim=config.dim,
        alpha=config.alpha,
        ridge_lambda=config.ridge_lambda,
        recompute_inv_every=config.recompute_inv_every,
    )

    # Pre-train warm agent (or load real priors)
    if config.priors_path and config.priors_path.exists():
        print(f"[RQ1] Loading real priors from {config.priors_path}")
        # Load priors using project's NPZ format
        meta = {"dim": config.dim, "alpha": config.alpha, "ridge_lambda": config.ridge_lambda}
        agent_warm = DisjointLinUCBPolicy.from_meta_and_npz(meta, config.priors_path)
        # Update environment to match loaded models
        env.model_names = agent_warm.models
        env.n_models = len(agent_warm.models)
        # Regenerate competencies for new model count
        env.competencies = env._generate_model_competencies()
    else:
        print(f"[RQ1] Pre-training on {config.n_pretrain} synthetic samples...")
        for _ in range(config.n_pretrain):
            ctx, cluster_id = env.sample_context()
            # Random exploration during pretraining (mimics archetype grid)
            model_idx = rng.integers(0, env.n_models)
            reward = env.get_reward(model_idx, cluster_id)
            # Use project's update method
            agent_warm.update(env.model_names[model_idx], ctx, reward)

    # Initialize cold-start agent (fresh) using same policy class
    print(f"[RQ1] Initializing cold-start agent (empty DisjointLinUCBPolicy)...")
    agent_cold = DisjointLinUCBPolicy(
        model_names=env.model_names,
        dim=config.dim,
        alpha=config.alpha,
        ridge_lambda=config.ridge_lambda,
        recompute_inv_every=config.recompute_inv_every,
    )

    # Run deployment simulation
    print(f"[RQ1] Running deployment simulation ({config.n_test} requests)...")
    regret_warm: List[float] = []
    regret_cold: List[float] = []
    cum_regret_warm = 0.0
    cum_regret_cold = 0.0

    for t in range(config.n_test):
        ctx, cluster_id = env.sample_context()
        optimal_reward = env.get_optimal_reward(cluster_id)

        # Warm agent decision using project's select_arm method
        # Returns: (model_name, ucb_score, propensity)
        model_warm, _, _ = agent_warm.select_arm(ctx, rng=rng)
        model_idx_warm = env.model_names.index(model_warm)
        reward_warm = env.get_reward(model_idx_warm, cluster_id)
        # Update using project's update method
        agent_warm.update(model_warm, ctx, reward_warm)

        # Cold agent decision using project's select_arm method
        model_cold, _, _ = agent_cold.select_arm(ctx, rng=rng)
        model_idx_cold = env.model_names.index(model_cold)
        reward_cold = env.get_reward(model_idx_cold, cluster_id)
        # Update using project's update method
        agent_cold.update(model_cold, ctx, reward_cold)

        # Compute regret (using expected reward, not noisy)
        expected_warm = env.competencies[model_idx_warm, cluster_id]
        expected_cold = env.competencies[model_idx_cold, cluster_id]

        cum_regret_warm += (optimal_reward - expected_warm)
        cum_regret_cold += (optimal_reward - expected_cold)

        regret_warm.append(cum_regret_warm)
        regret_cold.append(cum_regret_cold)

        if (t + 1) % 500 == 0:
            print(f"   Step {t+1}: Cold={cum_regret_cold:.1f}, Warm={cum_regret_warm:.1f}")

    # Compute summary statistics
    final_cold = regret_cold[-1]
    final_warm = regret_warm[-1]
    reduction_pct = 100.0 * (final_cold - final_warm) / max(final_cold, 1e-8)

    print(f"[RQ1] Final Results:")
    print(f"   Cold Start Regret: {final_cold:.2f}")
    print(f"   Warm Start Regret: {final_warm:.2f}")
    print(f"   Regret Reduction: {reduction_pct:.1f}%")

    return ExperimentResults(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
        regret_cold=regret_cold,
        regret_warm=regret_warm,
        final_regret_cold=final_cold,
        final_regret_warm=final_warm,
        regret_reduction_pct=reduction_pct,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results: ExperimentResults, output_path: Path) -> None:
    """
    Generate KDD publication-quality regret curve plot.

    Args:
        results: Experiment results
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        print("[RQ1] Warning: matplotlib not available, skipping plot")
        return

    # ---------------------------------------------------------------------------
    # KDD Paper Figure Settings
    # ---------------------------------------------------------------------------
    # KDD uses two-column format. Single column width ~3.33", double ~7"
    # We use single-column width for clarity
    COLUMN_WIDTH = 3.5  # inches
    FONT_SIZE = 9
    LEGEND_SIZE = 8
    LINE_WIDTH = 1.5
    DPI = 300  # Publication quality

    # Set matplotlib parameters for publication
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": LEGEND_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": LINE_WIDTH,
    })

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))

    n_test = len(results.regret_cold)
    x = np.arange(1, n_test + 1)

    # Plot lines with grayscale-friendly colors
    ax.plot(
        x, results.regret_cold,
        label="Cold Start",
        color="#D62728",  # Red - visible in grayscale as darker
        linestyle="--",
        linewidth=LINE_WIDTH,
    )
    ax.plot(
        x, results.regret_warm,
        label="Warm Start (Ours)",
        color="#1F77B4",  # Blue - visible in grayscale as lighter
        linestyle="-",
        linewidth=LINE_WIDTH + 0.5,
    )

    # Subtle fill to show improvement
    ax.fill_between(
        x,
        results.regret_cold,
        results.regret_warm,
        alpha=0.12,
        color="#1F77B4",
    )

    # Labels (no title - KDD uses figure captions)
    ax.set_xlabel("Number of Requests")
    ax.set_ylabel("Cumulative Regret")

    # Clean axis formatting
    ax.set_xlim(0, n_test)
    ax.set_ylim(0, None)

    # Add final regret annotation
    final_gap = results.final_regret_cold - results.final_regret_warm
    ax.annotate(
        f"Δ = {final_gap:.0f}\n({results.regret_reduction_pct:.0f}% reduction)",
        xy=(n_test * 0.95, (results.final_regret_cold + results.final_regret_warm) / 2),
        fontsize=FONT_SIZE - 1,
        ha="right",
        va="center",
    )

    # Legend - outside or inside based on space
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.95,
    )

    # Minimal grid
    ax.grid(True, linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    # Save as both PNG and PDF (PDF for paper submission)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")

    # Also save PDF version for LaTeX
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"[RQ1] Saved plot to {output_path} and {pdf_path}")
    plt.close()

    # Reset rcParams to defaults
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(results: ExperimentResults, output_path: Path) -> None:
    """Save experiment results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config Path objects to strings for JSON serialization
    data = asdict(results)
    if "config" in data:
        cfg = data["config"]
        if "output_dir" in cfg:
            cfg["output_dir"] = str(cfg["output_dir"])
        if "priors_path" in cfg:
            cfg["priors_path"] = str(cfg["priors_path"]) if cfg["priors_path"] else None
        if "models_cache" in cfg:
            cfg["models_cache"] = str(cfg["models_cache"])

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ1] Saved results to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RQ1: Cold Start vs Warm Start Bandit Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment size
    parser.add_argument(
        "--n-pretrain", type=int, default=500,
        help="Number of pretraining samples for warm agent",
    )
    parser.add_argument(
        "--n-test", type=int, default=2000,
        help="Number of test (deployment) samples",
    )

    # Environment
    parser.add_argument(
        "--n-models", type=int, default=0,
        help="Number of models to use from cache (0 = all models)",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of latent task clusters",
    )
    parser.add_argument(
        "--dim", type=int, default=384,
        help="Embedding dimension (384 matches sentence-transformers)",
    )
    parser.add_argument(
        "--noise", type=float, default=0.1,
        help="Reward noise standard deviation",
    )

    # Model cache (same as production BanditRouter)
    parser.add_argument(
        "--cache", type=str, default=str(PROJECT_ROOT / "data" / "models_cache.json"),
        help="Path to models_cache.json for loading real model IDs",
    )

    # Bandit parameters
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="UCB exploration parameter",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # Real priors
    parser.add_argument(
        "--priors", type=str, default=None,
        help="Path to real priors NPZ file (overrides synthetic pretraining)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results/rq1",
        help="Output directory for results and plots",
    )

    args = parser.parse_args()

    return ExperimentConfig(
        dim=args.dim,
        n_models=args.n_models,
        n_clusters=args.n_clusters,
        models_cache=Path(args.cache),
        n_pretrain=args.n_pretrain,
        n_test=args.n_test,
        alpha=args.alpha,
        noise_level=args.noise,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        priors_path=Path(args.priors) if args.priors else None,
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()

    print("=" * 60)
    print("RQ1: The 'Shippable Brain' Advantage")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Models: {config.n_models}")
    print(f"  Clusters: {config.n_clusters}")
    print(f"  Pretrain samples: {config.n_pretrain}")
    print(f"  Test samples: {config.n_test}")
    print(f"  Embedding dim: {config.dim}")
    print(f"  Alpha (UCB): {config.alpha}")
    print(f"  Seed: {config.seed}")
    print(f"  Output: {config.output_dir}")
    if config.priors_path:
        print(f"  Real priors: {config.priors_path}")
    print("=" * 60)
    print(f"Using: DisjointLinUCBPolicy (same as BanditRouter)")
    print("=" * 60)

    # Run experiment
    results = run_experiment(config)

    # Save outputs
    save_results(results, config.output_dir / "metrics.json")
    plot_results(results, config.output_dir / "regret_curve.png")

    print("=" * 60)
    print("Experiment complete!")
    print(f"  Regret reduction: {results.regret_reduction_pct:.1f}%")
    print(f"  Results saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
