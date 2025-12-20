#!/usr/bin/env python3
"""
RQ1-Transfer: Category-Split Transfer Learning Experiment

This script implements the "Category Split" approach suggested for KDD rebuttal:
- Split LMSYS archetypes by inferred category (Chat/Creative vs Math/Code)
- Train priors on source domain (general chat)
- Test on target domain (specialized tasks)

This is COMPLEMENTARY to run_rq1_ood.py:
- run_rq1_ood.py: Uses external datasets + benchmark scores (addresses both critiques)
- run_rq1_transfer.py: Uses category split within LMSYS (simpler, likely positive transfer)

Key Insight:
If priors from "Chat/Creative" help learn "Math/Code" faster, it proves the 
covariance structure captures generalizable model quality correlations.

Usage:
    python kdd_paper/scripts/run_rq1_transfer.py

Output:
    - results/rq1_transfer/transfer_regret_curve.png
    - results/rq1_transfer/metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from banditgpt.core.bandit_router import (
    DEFAULT_CONTEXT_MODEL,
    DisjointLinUCBPolicy,
)
from banditgpt._resources import get_priors_path

# Plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ---------------------------------------------------------------------------
# Category Classification (Heuristic)
# ---------------------------------------------------------------------------

def classify_prompt_category(prompt: str) -> str:
    """
    Heuristically classify a prompt into a category based on keywords.
    
    Categories:
        - math: Contains math symbols, numbers, equations
        - code: Contains code keywords, function definitions, programming
        - reasoning: Logic puzzles, analytical questions
        - creative: Stories, roleplay, poetry
        - general: General knowledge, chat, Q&A
    """
    prompt_lower = prompt.lower()
    
    # Math indicators
    math_patterns = [
        r'\d+\s*[\+\-\*\/\=]\s*\d+',  # arithmetic
        r'calculate|solve|equation|formula|derivative|integral',
        r'how many|what is \d+|sum of|product of',
        r'mathematics|algebra|geometry|calculus',
    ]
    for pattern in math_patterns:
        if re.search(pattern, prompt_lower):
            return "math"
    
    # Code indicators
    code_patterns = [
        r'def |function |class |import |from .* import',
        r'```|code|programming|python|java|javascript|c\+\+|rust',
        r'algorithm|debug|compile|execute|variable|loop',
        r'write a (function|program|script|code)',
    ]
    for pattern in code_patterns:
        if re.search(pattern, prompt_lower):
            return "code"
    
    # Reasoning indicators
    reasoning_patterns = [
        r'logic|puzzle|riddle|brain teaser',
        r'if .* then|therefore|conclude|deduce',
        r'what comes next|pattern|sequence',
        r'analyze|compare and contrast|pros and cons',
    ]
    for pattern in reasoning_patterns:
        if re.search(pattern, prompt_lower):
            return "reasoning"
    
    # Creative indicators
    creative_patterns = [
        r'story|poem|write a|creative|fiction|narrative',
        r'roleplay|pretend|imagine|character',
        r'song|lyrics|haiku|limerick',
    ]
    for pattern in creative_patterns:
        if re.search(pattern, prompt_lower):
            return "creative"
    
    # Default: general
    return "general"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_archetype_data(
    prompts_path: Path,
    rewards_path: Path,
    embeddings_path: Path,
) -> Tuple[List[Dict], Dict[Tuple[str, int], float]]:
    """
    Load archetype prompts with embeddings and rewards.
    
    Returns:
        (archetypes, rewards_dict)
        - archetypes: List of {cluster_id, prompt, category, embedding}
        - rewards_dict: (model_id, cluster_id) -> reward
    """
    # Load prompts
    prompts_data = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts_data.append({
                "cluster_id": data["cluster_id"],
                "prompt": data["prompt"],
            })
    
    # Load embeddings (precomputed)
    embeddings = np.load(embeddings_path)
    
    # Add embeddings and classify categories
    for i, p in enumerate(prompts_data):
        p["embedding"] = embeddings[i]
        p["category"] = classify_prompt_category(p["prompt"])
    
    # Load rewards
    rewards_dict = {}
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                rewards_dict[(model, cluster)] = reward
    
    return prompts_data, rewards_dict


# ---------------------------------------------------------------------------
# Transfer Experiment
# ---------------------------------------------------------------------------

@dataclass
class TransferExperimentConfig:
    """Configuration for transfer learning experiment."""
    prompts_path: Path = field(default_factory=lambda: get_priors_path("archetype_grid_prompts.jsonl"))
    rewards_path: Path = field(default_factory=lambda: get_priors_path("archetype_grid_dense_run.jsonl"))
    embeddings_path: Path = field(default_factory=lambda: get_priors_path("prompt_embeddings.npy"))
    
    # Source/target category split
    source_categories: List[str] = field(default_factory=lambda: ["general", "creative"])
    target_categories: List[str] = field(default_factory=lambda: ["math", "code", "reasoning"])
    
    alpha: float = 0.5
    prior_strength: float = 20.0  # Lower than OOD since within-dataset
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/rq1_transfer"))


def run_transfer_experiment(config: TransferExperimentConfig) -> Dict[str, Any]:
    """
    Run the category-split transfer learning experiment.
    
    1. Split data by category (source: chat/creative, target: math/code)
    2. Train priors from source domain
    3. Test on target domain
    4. Compare warm-start vs cold-start
    """
    print("[Transfer] Loading archetype data...")
    archetypes, rewards_dict = load_archetype_data(
        config.prompts_path,
        config.rewards_path,
        config.embeddings_path,
    )
    
    # Get model names from rewards
    model_names = sorted(set(m for m, _ in rewards_dict.keys()))
    dim = archetypes[0]["embedding"].shape[0]
    
    print(f"   Loaded {len(archetypes)} archetypes, {len(model_names)} models, dim={dim}")
    
    # Classify and split by category
    category_counts = {}
    for a in archetypes:
        cat = a["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\n[Transfer] Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count}")
    
    source_data = [a for a in archetypes if a["category"] in config.source_categories]
    target_data = [a for a in archetypes if a["category"] in config.target_categories]
    
    print(f"\n[Transfer] Split:")
    print(f"   Source ({config.source_categories}): {len(source_data)} samples")
    print(f"   Target ({config.target_categories}): {len(target_data)} samples")
    
    if len(target_data) == 0:
        raise ValueError("No target data found! Check category classification.")
    
    # Step 1: Build priors from source domain
    print("\n[Transfer] Step 1: Distilling priors from source domain...")
    
    # Initialize A matrices
    A_source = {m: np.eye(dim) for m in model_names}
    b_source = {m: np.zeros(dim) for m in model_names}
    
    # Accumulate covariance from source data
    for a in source_data:
        emb = a["embedding"]
        cluster_id = a["cluster_id"]
        outer = np.outer(emb, emb)
        
        for m in model_names:
            reward = rewards_dict.get((m, cluster_id), 0.5)
            A_source[m] += outer
            b_source[m] += emb * reward
    
    # Create warm-start policy with source priors
    policy_warm = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=config.alpha,
    )
    
    for m in model_names:
        policy_warm.A[m] = A_source[m] * config.prior_strength
        policy_warm.b[m] = b_source[m] * config.prior_strength
        policy_warm.A_inv[m] = np.linalg.inv(policy_warm.A[m])
    
    print(f"   Priors distilled from {len(source_data)} source samples")
    
    # Step 2: Create cold-start policy
    policy_cold = DisjointLinUCBPolicy(
        model_names=model_names,
        dim=dim,
        alpha=config.alpha,
    )
    
    # Step 3: Test on target domain
    print("\n[Transfer] Step 2: Testing on target domain...")
    
    rng = np.random.default_rng(config.seed)
    rng.shuffle(target_data)
    
    regret_cold = []
    regret_warm = []
    cumulative_cold = []
    cumulative_warm = []
    cum_cold = 0.0
    cum_warm = 0.0
    
    for i, a in enumerate(target_data):
        emb = a["embedding"]
        cluster_id = a["cluster_id"]
        
        # Get optimal reward for this cluster
        optimal = max(rewards_dict.get((m, cluster_id), 0.0) for m in model_names)
        
        # Cold-start selection
        best_cold, best_ucb_cold = None, -float("inf")
        for m in model_names:
            theta = policy_cold.A_inv[m] @ policy_cold.b[m]
            mean = float(theta.dot(emb))
            var = float(emb.dot(policy_cold.A_inv[m]).dot(emb))
            ucb = mean + config.alpha * np.sqrt(max(var, 1e-12))
            if ucb > best_ucb_cold:
                best_ucb_cold = ucb
                best_cold = m
        
        reward_cold = rewards_dict.get((best_cold, cluster_id), 0.5)
        policy_cold.update(best_cold, emb, reward_cold)
        r_cold = optimal - reward_cold
        cum_cold += r_cold
        regret_cold.append(r_cold)
        cumulative_cold.append(cum_cold)
        
        # Warm-start selection
        best_warm, best_ucb_warm = None, -float("inf")
        for m in model_names:
            theta = policy_warm.A_inv[m] @ policy_warm.b[m]
            mean = float(theta.dot(emb))
            var = float(emb.dot(policy_warm.A_inv[m]).dot(emb))
            ucb = mean + config.alpha * np.sqrt(max(var, 1e-12))
            if ucb > best_ucb_warm:
                best_ucb_warm = ucb
                best_warm = m
        
        reward_warm = rewards_dict.get((best_warm, cluster_id), 0.5)
        policy_warm.update(best_warm, emb, reward_warm)
        r_warm = optimal - reward_warm
        cum_warm += r_warm
        regret_warm.append(r_warm)
        cumulative_warm.append(cum_warm)
        
        if (i + 1) % 50 == 0:
            print(f"   Step {i+1}: Cold={cum_cold:.2f}, Warm={cum_warm:.2f}")
    
    # Compute reduction
    if cum_cold > 0:
        reduction = 100.0 * (cum_cold - cum_warm) / cum_cold
    else:
        reduction = 0.0
    
    print(f"\n[Transfer] Results:")
    print(f"   Cold Start Regret: {cum_cold:.2f}")
    print(f"   Warm Start Regret: {cum_warm:.2f}")
    print(f"   Regret Reduction: {reduction:.1f}%")
    
    results = {
        "config": {
            "source_categories": config.source_categories,
            "target_categories": config.target_categories,
            "prior_strength": config.prior_strength,
            "n_source": len(source_data),
            "n_target": len(target_data),
        },
        "regret_cold": regret_cold,
        "regret_warm": regret_warm,
        "cumulative_cold": cumulative_cold,
        "cumulative_warm": cumulative_warm,
        "final_regret_cold": cum_cold,
        "final_regret_warm": cum_warm,
        "regret_reduction_pct": reduction,
        "category_counts": category_counts,
        "timestamp": datetime.now().isoformat(),
    }
    
    return results


def plot_transfer_results(results: Dict, output_path: Path) -> None:
    """Plot transfer learning regret curves."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "figure.dpi": 300,
    })
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    n = len(results["cumulative_cold"])
    x = np.arange(1, n + 1)
    
    ax.plot(x, results["cumulative_cold"], label="Cold Start",
            color="#888888", linestyle="--", linewidth=1.5)
    ax.plot(x, results["cumulative_warm"], label="Transfer (Chat → Math/Code)",
            color="#2CA02C", linestyle="-", linewidth=2.0)
    
    ax.fill_between(x, results["cumulative_cold"], results["cumulative_warm"],
                    alpha=0.15, color="#2CA02C", 
                    where=[c > w for c, w in zip(results["cumulative_cold"], results["cumulative_warm"])])
    
    reduction = results["regret_reduction_pct"]
    ax.annotate(
        f"{reduction:.0f}% reduction",
        xy=(n * 0.7, (results["final_regret_cold"] + results["final_regret_warm"]) / 2),
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )
    
    ax.set_xlabel("Target Domain Interactions")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("RQ1-Transfer: Category-Split Learning", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[Transfer] Saved plot to {output_path}")
    plt.close()


def main() -> int:
    config = TransferExperimentConfig()
    
    print("=" * 60)
    print("RQ1-Transfer: Category-Split Transfer Learning")
    print("=" * 60)
    print("Source: Chat, Creative, General")
    print("Target: Math, Code, Reasoning")
    print("=" * 60)
    
    results = run_transfer_experiment(config)
    
    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config.output_dir / "metrics.json", "w") as f:
        # Convert numpy arrays to lists for JSON
        results_json = results.copy()
        for key in ["regret_cold", "regret_warm", "cumulative_cold", "cumulative_warm"]:
            if key in results_json:
                results_json[key] = [float(x) for x in results_json[key]]
        json.dump(results_json, f, indent=2)
    
    plot_transfer_results(results, config.output_dir / "transfer_regret_curve.png")
    
    print("\n" + "=" * 60)
    print("Transfer Learning Experiment Complete!")
    print(f"   Regret Reduction: {results['regret_reduction_pct']:.1f}%")
    print(f"   Results saved to: {config.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
