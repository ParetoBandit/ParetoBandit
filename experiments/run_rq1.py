#!/usr/bin/env python3
"""
RQ1 Experiment: The "Shippable Brain" Advantage (Trace-Driven)

Research Question:
    Does shipping pre-trained priors reduce regret compared to a cold-start bandit?

Experiment Design (Trace-Driven with REAL Embeddings):
    - Uses REAL priors from archetype grid
    - Embeds prompts using the SAME model used for training (all-MiniLM-L6-v2)
    - Ground truth rewards from actual model grading
    - Fair comparison: same embedding space for training and testing

Critical Requirements:
    1. DIM must match priors (384 for all-MiniLM-L6-v2)
    2. Embeddings must use same model as priors training
    3. Use real prompts, not np.random vectors
    4. Use DisjointLinUCB (not SharedCovariance) for proper per-model uncertainty

Why DisjointLinUCB?
    SharedCovariance forces identical uncertainty for all models (same A^-1).
    This breaks the warm-start advantage because the bandit can't say
    "I'm confident about GPT-4 but uncertain about Llama-3".
    
    DisjointLinUCB gives each model its own A matrix, enabling differential
    exploration and proper exploitation of learned priors.

Usage:
    python -m banditgpt.experiment.run_rq1

Output:
    - results/rq1/regret_curve.png - Publication-ready figure
    - results/rq1/metrics.json - Raw data
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

from banditgpt.core.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    DisjointLinUCBPolicy,
    SharedCovarianceLinUCBPolicy,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

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
    # Use expert priors by default (generated via Expert Distillation)
    priors_path: Path = PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"
    prompts_path: Path = PROJECT_ROOT / "data" / "priors" / "archetype_grid_prompts.jsonl"
    rewards_path: Path = PROJECT_ROOT / "data" / "priors" / "archetype_grid_dense_run.jsonl"
    embeddings_cache: Optional[Path] = PROJECT_ROOT / "data" / "priors" / "prompt_embeddings.npy"
    context_model: str = DEFAULT_CONTEXT_MODEL  # MUST match priors training
    n_test: int = 2000
    alpha: float = 0.5
    prior_strength: float = 50.0  # Confidence boost (50x works well for expert priors)
    seed: int = 42
    output_dir: Path = Path("results/rq1")


# ---------------------------------------------------------------------------
# Prior Inflation: SharedCovariance -> Disjoint
# ---------------------------------------------------------------------------

def inflate_shared_to_disjoint(
    shared: SharedCovarianceLinUCBPolicy,
    alpha: float = 0.5,
    strength: float = 1.0,
) -> DisjointLinUCBPolicy:
    """
    Convert SharedCovarianceLinUCBPolicy priors to DisjointLinUCBPolicy.
    
    This "inflates" the shared A matrix into per-model A matrices,
    enabling differential uncertainty per model (critical for warm-start advantage).
    
    Args:
        shared: Loaded SharedCovarianceLinUCBPolicy with priors
        alpha: UCB exploration parameter
        strength: Confidence boost multiplier for priors.
                  1.0 = Use priors as-is (weak confidence)
                  10.0 = Trust priors 10x more (reduces exploration)
                  100.0 = Very high confidence (near-zero exploration)
                  
                  This effectively tells the bandit: "Treat this prior data
                  as if it came from N*strength users, not just N users."
        
    Returns:
        DisjointLinUCBPolicy with inflated and boosted priors
    """
    disjoint = DisjointLinUCBPolicy(
        model_names=shared.models,
        dim=shared.dim,
        alpha=alpha,
    )
    
    # Copy shared A to each model, copy per-model b
    # Apply confidence boost to both A and b
    for m in shared.models:
        # Scale A and b by strength factor
        # This reduces uncertainty (A larger -> A_inv smaller -> lower UCB bonus)
        disjoint.A[m] = np.asarray(shared.A, dtype=np.float64).copy() * strength
        disjoint.b[m] = np.asarray(shared.b[m], dtype=np.float64).copy() * strength
        disjoint.A_inv[m] = np.linalg.inv(disjoint.A[m])
    
    if strength != 1.0:
        print(f"   [Boost] Applied {strength}x confidence multiplier to priors")
    
    return disjoint


def load_expert_priors(
    path: Path,
    alpha: float = 0.5,
    strength: float = 1.0,
) -> DisjointLinUCBPolicy:
    """
    Load Expert-Distilled priors (already in Disjoint format).
    
    Expert priors are generated via teacher demonstration (80% optimal picks)
    rather than uniform exploration. This encodes "expert intuition".
    
    Args:
        path: Path to expert_priors.npz
        alpha: UCB exploration parameter
        strength: Confidence boost multiplier
        
    Returns:
        DisjointLinUCBPolicy with expert priors
    """
    data = np.load(path, allow_pickle=True)
    
    model_names = [str(m) for m in data["model_names"]]
    dim = int(data["dim"])
    A_stack = np.asarray(data["A_stack"], dtype=np.float64)
    b_stack = np.asarray(data["b_stack"], dtype=np.float64)
    
    policy = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=alpha,
    )
    
    # Load and optionally boost priors
    for i, m in enumerate(model_names):
        policy.A[m] = A_stack[i] * strength
        policy.b[m] = b_stack[i] * strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    if strength != 1.0:
        print(f"   [Boost] Applied {strength}x confidence multiplier to expert priors")
    
    return policy


# ---------------------------------------------------------------------------
# Trace-Driven Environment with REAL Embeddings
# ---------------------------------------------------------------------------

class TraceEnvironment:
    """
    Trace-driven environment using REAL embeddings.

    Critical: Uses the SAME embedding model that was used to train the priors.
    This ensures the embedding geometry matches.

    Performance: Caches embeddings to disk (.npy) to avoid recomputing on every run.
    This is essential for scaling to 100k+ prompts.
    """

    def __init__(
        self,
        prompts_path: Path,
        rewards_path: Path,
        model_names: List[str],
        context_model: str,
        embeddings_cache: Optional[Path] = None,
        seed: int = 42,
    ):
        self.model_names = model_names
        self.rng = np.random.default_rng(seed)

        # Load prompts first (needed for cache validation)
        self.prompts, self.cluster_ids = self._load_prompts(prompts_path)
        print(f"   [Env] Loaded {len(self.prompts)} prompts")

        # Load or compute embeddings (with disk caching for performance)
        self.embeddings = self._load_or_compute_embeddings(
            context_model=context_model,
            cache_path=embeddings_cache,
        )
        print(f"   [Env] Embeddings ready: {len(self.embeddings)} x {self.embeddings.shape[1]}")

        # Load ground truth rewards from archetype grid run
        self.rewards = self._load_rewards(rewards_path)
        print(f"   [Env] Loaded {len(self.rewards)} (model, cluster) rewards")

        self.cursor = 0

    def _load_prompts(self, path: Path) -> Tuple[List[str], List[int]]:
        """Load prompts and cluster IDs."""
        prompts = []
        clusters = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
                clusters.append(data["cluster_id"])
        return prompts, clusters

    def _load_or_compute_embeddings(
        self,
        context_model: str,
        cache_path: Optional[Path],
    ) -> np.ndarray:
        """
        Load embeddings from cache or compute fresh.

        Cache validation: checks that cached embeddings have correct shape.
        This avoids using stale cache if prompts file changes.
        """
        n_prompts = len(self.prompts)

        # Try loading from cache
        if cache_path and cache_path.exists():
            try:
                cached = np.load(cache_path)
                if cached.shape[0] == n_prompts:
                    print(f"   [Env] Loaded cached embeddings from {cache_path.name}")
                    return np.asarray(cached, dtype=np.float64)
                else:
                    print(f"   [Env] Cache size mismatch ({cached.shape[0]} vs {n_prompts}), recomputing...")
            except Exception as e:
                print(f"   [Env] Cache load failed ({e}), recomputing...")

        # Compute fresh embeddings
        print(f"   [Env] Loading embedding model: {context_model}")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(context_model)

        print(f"   [Env] Embedding {n_prompts} prompts (this may take a moment)...")
        embeddings = encoder.encode(
            self.prompts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float64)

        # Save to cache for future runs
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"   [Env] Cached embeddings to {cache_path.name}")

        return embeddings

    def _load_rewards(self, path: Path) -> Dict[Tuple[str, int], float]:
        """Load (model, cluster) -> reward from dense run."""
        rewards = {}
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("ok", False):
                    model = data["model_id"]
                    cluster = data["cluster_id"]
                    logit = data.get("reward_logit", 0.0)
                    # Convert logit to [0, 1] reward
                    reward = 1.0 / (1.0 + np.exp(-logit))
                    rewards[(model, cluster)] = reward
        return rewards

    def get_next_request(self) -> Tuple[np.ndarray, int, str]:
        """
        Get next request from trace (cycles through prompts).

        Returns:
            (embedding_vector, cluster_id, prompt_text)
        """
        idx = self.cursor % len(self.prompts)
        self.cursor += 1

        # Shuffle order after each complete pass
        if self.cursor % len(self.prompts) == 0:
            perm = self.rng.permutation(len(self.prompts))
            self.prompts = [self.prompts[i] for i in perm]
            self.cluster_ids = [self.cluster_ids[i] for i in perm]
            self.embeddings = self.embeddings[perm]

        return self.embeddings[idx], self.cluster_ids[idx], self.prompts[idx]

    def get_reward(self, model_name: str, cluster_id: int) -> float:
        """Get reward with small noise."""
        base = self.rewards.get((model_name, cluster_id), 0.5)
        noise = self.rng.standard_normal() * 0.02
        return float(np.clip(base + noise, 0.0, 1.0))

    def get_expected_reward(self, model_name: str, cluster_id: int) -> float:
        """Get expected (noise-free) reward."""
        return self.rewards.get((model_name, cluster_id), 0.5)

    def get_optimal_reward(self, cluster_id: int) -> float:
        """Get best possible reward for cluster."""
        best = 0.0
        for model in self.model_names:
            r = self.rewards.get((model, cluster_id), 0.0)
            if r > best:
                best = r
        return max(best, 0.5)


# ---------------------------------------------------------------------------
# Bandit Selection (DisjointLinUCB)
# ---------------------------------------------------------------------------

def select_arm(
    policy: DisjointLinUCBPolicy,
    ctx: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """
    Select best arm using UCB with randomized tie-breaking.
    
    DisjointLinUCB has per-model A matrices, so each model has
    different uncertainty. This enables proper exploitation of priors.
    
    We add small random noise to break ties fairly (important when
    cold-start agent has identical UCBs for all models).
    """
    rng = rng or np.random.default_rng()
    
    best_model = policy.models[0]
    best_ucb = -float("inf")
    
    for m in policy.models:
        # Compute UCB
        theta = policy.A_inv[m] @ policy.b[m]
        mean = float(theta.dot(ctx))
        var = float(ctx.dot(policy.A_inv[m]).dot(ctx))
        std = float(np.sqrt(max(var, 1e-12)))
        ucb = mean + policy.alpha * std
        
        # Add tiny noise for fair tie-breaking
        ucb += rng.random() * 1e-8
        
        if ucb > best_ucb:
            best_ucb = ucb
            best_model = m
    
    return best_model


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResults:
    config: Dict[str, Any]
    regret_cold: List[float]  # Per-step regret
    regret_warm: List[float]  # Per-step regret
    cumulative_regret_cold: List[float]
    cumulative_regret_warm: List[float]
    final_regret_cold: float
    final_regret_warm: float
    regret_reduction_pct: float
    n_models: int
    n_prompts: int
    embedding_model: str
    timestamp: str


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """
    Run trace-driven cold vs warm comparison using DisjointLinUCB.

    Uses REAL priors and REAL embeddings from the same model.
    Inflates SharedCovariance priors to Disjoint for proper per-model uncertainty.
    """
    print(f"[RQ1] Loading priors from {config.priors_path}")

    if not config.priors_path.exists():
        raise FileNotFoundError(f"Priors not found: {config.priors_path}")

    # Detect prior type and load accordingly
    priors_data = np.load(config.priors_path, allow_pickle=True)
    
    if "A_stack" in priors_data:
        # Expert-Distilled priors (already Disjoint format)
        print(f"   [Expert] Detected Expert-Distilled priors")
        agent_warm = load_expert_priors(
            config.priors_path,
            alpha=config.alpha,
            strength=config.prior_strength,
        )
        model_names = agent_warm.models
        dim = agent_warm.dim
        print(f"   [Warm] Loaded {len(model_names)} models, dim={dim}")
    else:
        # Shared Covariance priors (need inflation)
        print(f"   [Shared] Detected Shared Covariance priors")
        shared_priors = SharedCovarianceLinUCBPolicy.from_shippable_priors_npz(config.priors_path)
        model_names = shared_priors.models
        dim = shared_priors.dim
        print(f"   [Shared] Loaded {len(model_names)} models, dim={dim}")

        # Inflate to DisjointLinUCB (enables per-model uncertainty)
        print(f"   [Inflate] Converting to DisjointLinUCB...")
        agent_warm = inflate_shared_to_disjoint(
            shared_priors,
            alpha=config.alpha,
            strength=config.prior_strength,
        )
    
    print(f"   [Warm] DisjointLinUCB ready (strength={config.prior_strength}x)")

    # Verify dimension matches embedding model
    print(f"   [Check] Embedding model: {config.context_model}")
    print(f"   [Check] Priors dim: {dim} (should be 384 for all-MiniLM-L6-v2)")

    # Create cold start agent (DisjointLinUCB with identity A per model)
    agent_cold = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=config.alpha,
    )
    print(f"   [Cold] Fresh DisjointLinUCB (no priors)")

    # Create trace-driven environment with REAL embeddings
    print(f"[RQ1] Creating trace-driven environment with REAL embeddings...")
    env = TraceEnvironment(
        prompts_path=config.prompts_path,
        rewards_path=config.rewards_path,
        model_names=model_names,
        context_model=config.context_model,
        embeddings_cache=config.embeddings_cache,
        seed=config.seed,
    )

    # Run simulation
    print(f"[RQ1] Running {config.n_test} requests...")
    step_regret_warm: List[float] = []  # Per-step regret
    step_regret_cold: List[float] = []  # Per-step regret
    cumulative_warm: List[float] = []
    cumulative_cold: List[float] = []
    cum_warm = 0.0
    cum_cold = 0.0
    
    # Separate RNGs for fair comparison
    rng_warm = np.random.default_rng(config.seed)
    rng_cold = np.random.default_rng(config.seed + 1)

    for t in range(config.n_test):
        ctx, cluster_id, _ = env.get_next_request()
        optimal = env.get_optimal_reward(cluster_id)

        # Warm agent (has real priors, DisjointLinUCB)
        model_w = select_arm(agent_warm, ctx, rng_warm)
        reward_w = env.get_reward(model_w, cluster_id)
        agent_warm.update(model_w, ctx, reward_w)
        r_warm = optimal - env.get_expected_reward(model_w, cluster_id)
        cum_warm += r_warm
        step_regret_warm.append(r_warm)
        cumulative_warm.append(cum_warm)

        # Cold agent (no priors, DisjointLinUCB)
        model_c = select_arm(agent_cold, ctx, rng_cold)
        reward_c = env.get_reward(model_c, cluster_id)
        agent_cold.update(model_c, ctx, reward_c)
        r_cold = optimal - env.get_expected_reward(model_c, cluster_id)
        cum_cold += r_cold
        step_regret_cold.append(r_cold)
        cumulative_cold.append(cum_cold)

        if (t + 1) % 500 == 0:
            print(f"   Step {t+1}: Cold={cum_cold:.1f}, Warm={cum_warm:.1f}")

    # Compute reduction (handle edge case where cold regret is zero/negative)
    if cum_cold > 0:
        reduction = 100.0 * (cum_cold - cum_warm) / cum_cold
    else:
        reduction = 0.0

    print(f"\n[RQ1] Final Results:")
    print(f"   Cold Start Regret: {cum_cold:.1f}")
    print(f"   Warm Start Regret: {cum_warm:.1f}")
    print(f"   Regret Reduction: {reduction:.1f}%")

    return ExperimentResults(
        config=asdict(config),
        regret_cold=step_regret_cold,
        regret_warm=step_regret_warm,
        cumulative_regret_cold=cumulative_cold,
        cumulative_regret_warm=cumulative_warm,
        final_regret_cold=cum_cold,
        final_regret_warm=cum_warm,
        regret_reduction_pct=reduction,
        n_models=len(model_names),
        n_prompts=len(env.prompts),
        embedding_model=config.context_model,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results: ExperimentResults, output_path: Path) -> None:
    """Plot cumulative regret vs requests."""
    if not HAS_MATPLOTLIB:
        return

    COLUMN_WIDTH = 3.5
    FONT_SIZE = 9
    DPI = 300

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))

    n = len(results.cumulative_regret_cold)
    x = np.arange(1, n + 1)

    ax.plot(x, results.cumulative_regret_cold, label="Cold Start (No Priors)",
            color="#D62728", linestyle="--", linewidth=1.5)
    ax.plot(x, results.cumulative_regret_warm, label="Warm Start (Shippable Brain)",
            color="#1F77B4", linestyle="-", linewidth=2.0)

    ax.fill_between(x, results.cumulative_regret_cold, results.cumulative_regret_warm,
                    alpha=0.15, color="#1F77B4")

    gap = results.final_regret_cold - results.final_regret_warm
    ax.annotate(
        f"Δ = {gap:.0f}\n({results.regret_reduction_pct:.0f}% reduction)",
        xy=(n * 0.65, (results.final_regret_cold + results.final_regret_warm) / 2),
        fontsize=FONT_SIZE - 1,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("User Requests")
    ax.set_ylabel("Cumulative Regret")
    ax.set_xlim(0, n)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")

    print(f"[RQ1] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(results: ExperimentResults, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(results)
    if "config" in data:
        for key in ["output_dir", "priors_path", "prompts_path", "rewards_path", "embeddings_cache"]:
            if key in data["config"] and data["config"][key]:
                data["config"][key] = str(data["config"][key])

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ1] Saved results to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="RQ1: Trace-Driven Regret Comparison (DisjointLinUCB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--priors", type=str,
                        default=str(PROJECT_ROOT / "data" / "priors" / "expert_priors.npz"))
    parser.add_argument("--context-model", type=str, default=DEFAULT_CONTEXT_MODEL,
                        help="Embedding model (MUST match priors training)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable embedding cache (recompute every run)")
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prior-strength", type=float, default=50.0,
                        help="Confidence multiplier for priors (50x optimal for expert priors)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/rq1")

    args = parser.parse_args()

    # Determine cache path (None if disabled)
    cache_path: Optional[Path] = None
    if not args.no_cache:
        cache_path = PROJECT_ROOT / "data" / "priors" / "prompt_embeddings.npy"

    return ExperimentConfig(
        priors_path=Path(args.priors),
        embeddings_cache=cache_path,
        context_model=args.context_model,
        n_test=args.n_test,
        alpha=args.alpha,
        prior_strength=args.prior_strength,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    config = parse_args()

    print("=" * 60)
    print("RQ1: The 'Shippable Brain' Advantage")
    print("=" * 60)
    print("Trace-Driven Evaluation with DisjointLinUCB")
    print(f"  Embedding Model: {config.context_model}")
    print(f"  Priors: {config.priors_path.name}")
    print(f"  Prior Strength: {config.prior_strength}x (confidence boost)")
    print(f"  Policy: DisjointLinUCB (per-model uncertainty)")
    print(f"  Requests: {config.n_test}")
    print("=" * 60)

    results = run_experiment(config)

    save_results(results, config.output_dir / "metrics.json")
    plot_results(results, config.output_dir / "regret_curve.png")

    print("=" * 60)
    print("Complete!")
    print(f"  Models: {results.n_models}")
    print(f"  Prompts: {results.n_prompts}")
    print(f"  Embedding: {results.embedding_model}")
    print(f"  Regret reduction: {results.regret_reduction_pct:.1f}%")
    print(f"  Saved to: {config.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
