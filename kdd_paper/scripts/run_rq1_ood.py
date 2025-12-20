#!/usr/bin/env python3
"""
RQ1-OOD: Out-of-Distribution Generalization Evaluation

This script addresses the KDD reviewer's critique about data leakage:
    "The priors are trained on 497 archetypes derived from LMSYS. The evaluation 
     samples 2,000 queries from these same archetypes... This tests interpolation, 
     not generalization."

Solution:
    - Evaluate on TRULY held-out datasets (GSM8K, HumanEval, MMLU)
    - Use published benchmark scores as ground truth (not GPT-4o judge)
    - Prove that LMSYS-trained priors generalize to unseen domains

Research Question:
    Do priors learned on LMSYS conversational data transfer to specialized 
    benchmark domains (math, code, knowledge)?

Experimental Design:
    1. Load priors trained on LMSYS archetypes (unchanged from RQ1)
    2. Embed OOD prompts (GSM8K, HumanEval, MMLU) using same embedding model
    3. For each prompt, bandit selects a model based on learned priors
    4. "Reward" = model's benchmark score on that domain
    5. Compare warm-start vs cold-start regret

Key Insight:
    If warm-start shows ANY improvement on OOD data, it proves the priors
    capture generalizable model quality correlations, not just LMSYS memorization.

Usage:
    # Math domain (GSM8K prompts, MATH-500 benchmark)
    python kdd_paper/scripts/run_rq1_ood.py --domain math

    # Code domain (HumanEval prompts, HumanEval benchmark)
    python kdd_paper/scripts/run_rq1_ood.py --domain code

    # Knowledge domain (MMLU prompts, MMLU-Pro benchmark)
    python kdd_paper/scripts/run_rq1_ood.py --domain knowledge

    # All domains
    python kdd_paper/scripts/run_rq1_ood.py --domain all

Output:
    - results/rq1_ood/{domain}_regret_curve.png
    - results/rq1_ood/{domain}_metrics.json
    - results/rq1_ood/summary.json (when --domain all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.core.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    DisjointLinUCBPolicy,
)
from banditgpt._resources import get_priors_path

# Import OOD datasets
from kdd_paper.scripts.ood_datasets import get_domain_prompts, DOMAIN_CONFIG

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
class OODExperimentConfig:
    """Configuration for OOD evaluation experiment."""
    # Domain to evaluate
    domain: str = "math"  # math, code, knowledge, graduate, all
    
    # Priors (trained on LMSYS)
    priors_path: Path = field(default_factory=lambda: get_priors_path("expert_priors.npz"))
    benchmarks_path: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "banditgpt" / "data" / "models_cache.json")
    
    # Embedding model (must match priors training)
    context_model: str = DEFAULT_CONTEXT_MODEL
    
    # Experiment parameters
    n_test: Optional[int] = None  # None = use domain default
    alpha: float = 0.5
    prior_strength: float = 50.0
    seed: int = 42
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("results/rq1_ood"))


# ---------------------------------------------------------------------------
# Benchmark Data Loading
# ---------------------------------------------------------------------------

def load_benchmarks(path: Path) -> Dict[str, Dict[str, float]]:
    """Load benchmark scores from models_cache.json file."""
    if not path.exists():
        raise FileNotFoundError(f"Models cache file not found: {path}")
    
    data = json.loads(path.read_text())
    
    # models_cache.json has a list of models, convert to dict by openrouter_id
    models = data.get("models", [])
    return {
        m["openrouter_id"]: m 
        for m in models 
        if "openrouter_id" in m
    }


# ---------------------------------------------------------------------------
# OOD Benchmark Environment
# ---------------------------------------------------------------------------

class OODBenchmarkEnvironment:
    """
    Out-of-Distribution evaluation environment using benchmark scores.
    
    Unlike TraceEnvironment (which uses GPT-4o judge on LMSYS prompts),
    this uses published benchmark scores on held-out datasets.
    
    This addresses two reviewer critiques:
    1. Data Leakage: OOD prompts are NOT from LMSYS training data
    2. Circular Judge: Ground truth is benchmark scores, not GPT-4o
    
    Attributes:
        domain: The evaluation domain (math, code, knowledge)
        benchmark_key: Which benchmark score to use as reward
        model_rewards: Dict mapping model_id -> normalized reward
    """
    
    def __init__(
        self,
        domain: str,
        benchmarks: Dict[str, Dict[str, float]],
        model_names: List[str],
        prompts: List[str],
        context_model: str,
        seed: int = 42,
    ):
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        
        # Get domain configuration
        config = DOMAIN_CONFIG[domain]
        self.benchmark_key = config["benchmark_key"]
        self.benchmark_scale = config.get("benchmark_scale", 1.0)
        
        # Build model -> reward mapping
        self.model_rewards: Dict[str, float] = {}
        self.model_names = model_names
        
        for model_id in model_names:
            model_data = benchmarks.get(model_id, {})
            raw_score = model_data.get(self.benchmark_key)
            
            if raw_score is not None:
                # Normalize to [0, 1] if needed
                if self.benchmark_scale > 1:
                    normalized = float(raw_score) / self.benchmark_scale
                else:
                    normalized = float(raw_score)
                self.model_rewards[model_id] = np.clip(normalized, 0.0, 1.0)
            else:
                # Default for missing benchmark data
                self.model_rewards[model_id] = 0.5
        
        # Embed prompts
        self.prompts = prompts
        self.embeddings = self._embed_prompts(prompts, context_model)
        
        # Track iteration
        self.cursor = 0
        self._order = list(range(len(prompts)))
        
        print(f"   [OOD] Domain: {domain}")
        print(f"   [OOD] Benchmark: {self.benchmark_key}")
        print(f"   [OOD] Prompts: {len(prompts)}")
        print(f"   [OOD] Models with benchmark: {sum(1 for r in self.model_rewards.values() if r != 0.5)}/{len(model_names)}")
    
    def _embed_prompts(self, prompts: List[str], model_name: str) -> np.ndarray:
        """Embed prompts using sentence transformer."""
        print(f"   [OOD] Embedding {len(prompts)} prompts with {model_name}...")
        
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        
        embeddings = encoder.encode(
            prompts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        
        return np.asarray(embeddings, dtype=np.float64)
    
    def get_next_request(self) -> Tuple[np.ndarray, int]:
        """
        Get next OOD prompt embedding.
        
        Returns:
            (embedding_vector, prompt_index)
        """
        if self.cursor >= len(self._order):
            # Reshuffle for next epoch
            self.rng.shuffle(self._order)
            self.cursor = 0
        
        idx = self._order[self.cursor]
        self.cursor += 1
        
        return self.embeddings[idx], idx
    
    def get_reward(self, model_name: str, noise_std: float = 0.02) -> float:
        """
        Get reward with small noise.
        
        For OOD, reward is the model's benchmark score (same for all prompts
        in a domain, since benchmark scores are per-model, not per-prompt).
        """
        base = self.model_rewards.get(model_name, 0.5)
        noise = self.rng.standard_normal() * noise_std
        return float(np.clip(base + noise, 0.0, 1.0))
    
    def get_expected_reward(self, model_name: str) -> float:
        """Get expected (noise-free) reward."""
        return self.model_rewards.get(model_name, 0.5)
    
    def get_optimal_reward(self) -> float:
        """Get best possible reward (best model's benchmark score)."""
        if not self.model_rewards:
            return 0.5
        return max(self.model_rewards.values())
    
    def get_optimal_model(self) -> str:
        """Get the model with highest benchmark score."""
        return max(self.model_rewards.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Prior Loading (from run_rq1.py)
# ---------------------------------------------------------------------------

def load_expert_priors(
    path: Path,
    alpha: float = 0.5,
    strength: float = 1.0,
) -> DisjointLinUCBPolicy:
    """Load Expert-Distilled priors."""
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
    
    for i, m in enumerate(model_names):
        policy.A[m] = A_stack[i] * strength
        policy.b[m] = b_stack[i] * strength
        policy.A_inv[m] = np.linalg.inv(policy.A[m])
    
    if strength != 1.0:
        print(f"   [Boost] Applied {strength}x confidence multiplier to expert priors")
    
    return policy


# ---------------------------------------------------------------------------
# Bandit Selection
# ---------------------------------------------------------------------------

def select_arm(
    policy: DisjointLinUCBPolicy,
    ctx: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> str:
    """Select best arm using UCB with randomized tie-breaking."""
    rng = rng or np.random.default_rng()
    
    best_model = policy.models[0]
    best_ucb = -float("inf")
    
    for m in policy.models:
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
class OODExperimentResults:
    """Results from OOD evaluation."""
    config: Dict[str, Any]
    domain: str
    benchmark_key: str
    
    # Regret curves
    regret_cold: List[float]
    regret_warm: List[float]
    cumulative_regret_cold: List[float]
    cumulative_regret_warm: List[float]
    
    # Summary metrics
    final_regret_cold: float
    final_regret_warm: float
    regret_reduction_pct: float
    
    # OOD-specific
    optimal_model: str
    optimal_reward: float
    n_models: int
    n_prompts: int
    
    timestamp: str


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_ood_experiment(config: OODExperimentConfig) -> OODExperimentResults:
    """
    Run OOD evaluation comparing warm-start vs cold-start.
    
    This is the key experiment for the KDD rebuttal:
    - Priors trained on LMSYS (in-distribution)
    - Tested on GSM8K/HumanEval/MMLU (out-of-distribution)
    - Ground truth from benchmark scores (not GPT-4o)
    """
    print(f"[RQ1-OOD] Loading priors from {config.priors_path}")
    
    if not config.priors_path.exists():
        raise FileNotFoundError(f"Priors not found: {config.priors_path}")
    
    # Load warm-start agent with LMSYS-trained priors
    agent_warm = load_expert_priors(
        config.priors_path,
        alpha=config.alpha,
        strength=config.prior_strength,
    )
    model_names = agent_warm.models
    dim = agent_warm.dim
    print(f"   [Warm] Loaded {len(model_names)} models, dim={dim}")
    
    # Load benchmark data
    print(f"[RQ1-OOD] Loading benchmarks from {config.benchmarks_path}")
    benchmarks = load_benchmarks(config.benchmarks_path)
    print(f"   [Benchmarks] {len(benchmarks)} models with benchmark data")
    
    # Load OOD prompts
    print(f"[RQ1-OOD] Loading {config.domain} prompts...")
    prompts, domain_config = get_domain_prompts(
        config.domain,
        n=config.n_test,
        seed=config.seed,
    )
    print(f"   [OOD] Loaded {len(prompts)} prompts")
    
    # Create OOD environment
    print(f"[RQ1-OOD] Creating OOD environment...")
    env = OODBenchmarkEnvironment(
        domain=config.domain,
        benchmarks=benchmarks,
        model_names=model_names,
        prompts=prompts,
        context_model=config.context_model,
        seed=config.seed,
    )
    
    # Create cold-start agent
    agent_cold = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=config.alpha,
    )
    print(f"   [Cold] Fresh DisjointLinUCB (no priors)")
    
    # Run simulation
    n_steps = len(prompts)
    print(f"[RQ1-OOD] Running {n_steps} requests...")
    
    step_regret_warm: List[float] = []
    step_regret_cold: List[float] = []
    cumulative_warm: List[float] = []
    cumulative_cold: List[float] = []
    cum_warm = 0.0
    cum_cold = 0.0
    
    # Separate RNGs
    rng_warm = np.random.default_rng(config.seed)
    rng_cold = np.random.default_rng(config.seed + 1)
    
    optimal_reward = env.get_optimal_reward()
    
    for t in range(n_steps):
        ctx, _ = env.get_next_request()
        
        # Warm agent
        model_w = select_arm(agent_warm, ctx, rng_warm)
        reward_w = env.get_reward(model_w)
        agent_warm.update(model_w, ctx, reward_w)
        r_warm = optimal_reward - env.get_expected_reward(model_w)
        cum_warm += r_warm
        step_regret_warm.append(r_warm)
        cumulative_warm.append(cum_warm)
        
        # Cold agent
        model_c = select_arm(agent_cold, ctx, rng_cold)
        reward_c = env.get_reward(model_c)
        agent_cold.update(model_c, ctx, reward_c)
        r_cold = optimal_reward - env.get_expected_reward(model_c)
        cum_cold += r_cold
        step_regret_cold.append(r_cold)
        cumulative_cold.append(cum_cold)
        
        if (t + 1) % 100 == 0:
            print(f"   Step {t+1}: Cold={cum_cold:.2f}, Warm={cum_warm:.2f}")
    
    # Compute reduction
    if cum_cold > 0:
        reduction = 100.0 * (cum_cold - cum_warm) / cum_cold
    else:
        reduction = 0.0
    
    print(f"\n[RQ1-OOD] Final Results ({config.domain}):")
    print(f"   Cold Start Regret: {cum_cold:.2f}")
    print(f"   Warm Start Regret: {cum_warm:.2f}")
    print(f"   Regret Reduction: {reduction:.1f}%")
    print(f"   Optimal Model: {env.get_optimal_model()}")
    print(f"   Optimal Reward: {optimal_reward:.3f}")
    
    return OODExperimentResults(
        config=asdict(config),
        domain=config.domain,
        benchmark_key=domain_config["benchmark_key"],
        regret_cold=step_regret_cold,
        regret_warm=step_regret_warm,
        cumulative_regret_cold=cumulative_cold,
        cumulative_regret_warm=cumulative_warm,
        final_regret_cold=cum_cold,
        final_regret_warm=cum_warm,
        regret_reduction_pct=reduction,
        optimal_model=env.get_optimal_model(),
        optimal_reward=optimal_reward,
        n_models=len(model_names),
        n_prompts=len(prompts),
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ood_results(results: OODExperimentResults, output_path: Path) -> None:
    """Plot OOD regret curve."""
    if not HAS_MATPLOTLIB:
        print("[RQ1-OOD] Warning: matplotlib not available, skipping plot")
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
    ax.plot(x, results.cumulative_regret_warm, label="Warm Start (LMSYS Priors)",
            color="#1F77B4", linestyle="-", linewidth=2.0)
    
    ax.fill_between(x, results.cumulative_regret_cold, results.cumulative_regret_warm,
                    alpha=0.15, color="#1F77B4")
    
    gap = results.final_regret_cold - results.final_regret_warm
    ax.annotate(
        f"Δ = {gap:.1f}\n({results.regret_reduction_pct:.0f}% reduction)",
        xy=(n * 0.65, (results.final_regret_cold + results.final_regret_warm) / 2),
        fontsize=FONT_SIZE - 1,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    
    domain_titles = {
        "math": "Math (GSM8K → MATH-500)",
        "code": "Code (HumanEval)",
        "code_live": "Code (LiveCodeBench)",
        "knowledge": "Knowledge (MMLU → MMLU-Pro)",
        "graduate": "Graduate QA (GPQA)",
    }
    
    ax.set_xlabel("OOD Prompts")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"RQ1-OOD: {domain_titles.get(results.domain, results.domain)}", fontsize=10)
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
    
    print(f"[RQ1-OOD] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def plot_combined_results(all_results: Dict[str, OODExperimentResults], output_path: Path) -> None:
    """Plot combined regret curves for all domains."""
    if not HAS_MATPLOTLIB:
        return
    
    COLUMN_WIDTH = 7.0
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
    
    fig, axes = plt.subplots(1, len(all_results), figsize=(COLUMN_WIDTH, 2.5))
    if len(all_results) == 1:
        axes = [axes]
    
    domain_titles = {
        "math": "Math (GSM8K)",
        "code": "Code (HumanEval)",
        "knowledge": "Knowledge (MMLU)",
    }
    
    for ax, (domain, results) in zip(axes, all_results.items()):
        n = len(results.cumulative_regret_cold)
        x = np.arange(1, n + 1)
        
        ax.plot(x, results.cumulative_regret_cold, label="Cold Start (Fresh)",
                color="#2CA02C", linestyle="--", linewidth=1.5)
        ax.plot(x, results.cumulative_regret_warm, label="LMSYS Priors (Wrong!)",
                color="#D62728", linestyle="-", linewidth=2.0)
        
        ax.fill_between(x, results.cumulative_regret_cold, results.cumulative_regret_warm,
                        alpha=0.15, color="#D62728")  # Red fill = priors hurt
        
        ax.set_xlabel("OOD Prompts")
        if ax == axes[0]:
            ax.set_ylabel("Cumulative Regret")
        # For OOD, negative reduction = priors HURT performance (Negative Transfer)
        if results.regret_reduction_pct < 0:
            title_suffix = f"({abs(results.regret_reduction_pct):.0f}% worse)"
        else:
            title_suffix = f"({results.regret_reduction_pct:.0f}% better)"
        ax.set_title(f"{domain_titles.get(domain, domain)}\n{title_suffix}", fontsize=9)
        ax.set_xlim(0, n)
        ax.set_ylim(0, None)
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    axes[0].legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.8")
    
    plt.tight_layout(pad=0.5)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    
    print(f"[RQ1-OOD] Saved combined plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def save_results(results: OODExperimentResults, output_path: Path) -> None:
    """Save results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = asdict(results)
    # Convert Path objects to strings
    if "config" in data:
        for key in ["output_dir", "priors_path", "benchmarks_path"]:
            if key in data["config"] and data["config"][key]:
                data["config"][key] = str(data["config"][key])
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[RQ1-OOD] Saved results to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> OODExperimentConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RQ1-OOD: Out-of-Distribution Generalization Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--domain", type=str, default="math",
                        choices=["math", "code", "code_live", "knowledge", "graduate", "all"],
                        help="Domain to evaluate")
    parser.add_argument("--priors", type=str,
                        default=str(get_priors_path("expert_priors.npz")))
    parser.add_argument("--benchmarks", type=str,
                        default=str(Path(__file__).parent.parent.parent / "banditgpt" / "data" / "models_cache.json"))
    parser.add_argument("--context-model", type=str, default=DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--n-test", type=int, default=None,
                        help="Number of test prompts (None = use domain default)")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/rq1_ood")
    
    args = parser.parse_args()
    
    return OODExperimentConfig(
        domain=args.domain,
        priors_path=Path(args.priors),
        benchmarks_path=Path(args.benchmarks),
        context_model=args.context_model,
        n_test=args.n_test,
        alpha=args.alpha,
        prior_strength=args.prior_strength,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


def main() -> int:
    """Main entry point."""
    config = parse_args()
    
    print("=" * 70)
    print("RQ1-OOD: Out-of-Distribution Generalization Evaluation")
    print("=" * 70)
    print("Addressing KDD Reviewer Critique: Data Leakage / Interpolation")
    print("")
    print("Key Points:")
    print("  - Priors trained on LMSYS (conversational data)")
    print("  - Testing on GSM8K/HumanEval/MMLU (specialized benchmarks)")
    print("  - Ground truth: Published benchmark scores (NOT GPT-4o judge)")
    print("=" * 70)
    print(f"Domain: {config.domain}")
    print(f"Priors: {config.priors_path.name}")
    print(f"Embedding: {config.context_model}")
    print("=" * 70)
    
    if config.domain == "all":
        # Run all domains
        domains = ["math", "code", "knowledge"]
        all_results: Dict[str, OODExperimentResults] = {}
        
        for domain in domains:
            print(f"\n{'='*70}")
            print(f"Running domain: {domain}")
            print(f"{'='*70}\n")
            
            domain_config = OODExperimentConfig(
                domain=domain,
                priors_path=config.priors_path,
                benchmarks_path=config.benchmarks_path,
                context_model=config.context_model,
                n_test=config.n_test,
                alpha=config.alpha,
                prior_strength=config.prior_strength,
                seed=config.seed,
                output_dir=config.output_dir,
            )
            
            results = run_ood_experiment(domain_config)
            all_results[domain] = results
            
            # Save individual results
            save_results(results, config.output_dir / f"{domain}_metrics.json")
            plot_ood_results(results, config.output_dir / f"{domain}_regret_curve.png")
        
        # Generate combined plot and summary
        plot_combined_results(all_results, config.output_dir / "combined_regret_curves.png")
        
        # Summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "domains": {
                domain: {
                    "regret_reduction_pct": r.regret_reduction_pct,
                    "final_regret_cold": r.final_regret_cold,
                    "final_regret_warm": r.final_regret_warm,
                    "n_prompts": r.n_prompts,
                    "optimal_model": r.optimal_model,
                }
                for domain, r in all_results.items()
            },
            "average_regret_reduction_pct": np.mean([r.regret_reduction_pct for r in all_results.values()]),
        }
        
        summary_path = config.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 70)
        print("RQ1-OOD Summary: Out-of-Distribution Generalization")
        print("=" * 70)
        for domain, r in all_results.items():
            print(f"  {domain}: {r.regret_reduction_pct:.1f}% regret reduction")
        print(f"  Average: {summary['average_regret_reduction_pct']:.1f}% regret reduction")
        print("=" * 70)
        print(f"Results saved to: {config.output_dir}")
        print("=" * 70)
        
    else:
        # Run single domain
        results = run_ood_experiment(config)
        
        save_results(results, config.output_dir / f"{config.domain}_metrics.json")
        plot_ood_results(results, config.output_dir / f"{config.domain}_regret_curve.png")
        
        print("\n" + "=" * 70)
        print("RQ1-OOD Complete!")
        print(f"  Domain: {config.domain}")
        print(f"  Regret Reduction: {results.regret_reduction_pct:.1f}%")
        print(f"  (Compare to 63.6% on in-distribution LMSYS)")
        print(f"  Results saved to: {config.output_dir}")
        print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
