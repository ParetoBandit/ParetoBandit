#!/usr/bin/env python3
"""
RQ2: Overcoming "Confidently Wrong" Priors

This script demonstrates the bandit's ability to unlearn poisoned priors
through accumulation of contradictory evidence.

The Setup (2-Body Problem):
    - GPT-4o: The "False Idol" - priors say it's perfect (1.0), reality is mediocre (0.6)
    - Nova-Lite: The "Hidden Gem" - priors say it's trash (0.1), reality is excellent (0.95)

The Mechanism:
    - Priors are POISONED: high confidence + wrong reward estimates
    - Bandit stubbornly clings to GPT-4 (The Dip)
    - Accumulated bad rewards slowly erode confidence
    - Eventually explores Nova, discovers truth (The Flip)
    - Switches permanently to Nova (The Recovery)

Key Insight:
    To get a "Dip", the bandit must be "Confidently Wrong", not "Uncertain".
    Uncertain = high exploration bonus = immediate discovery (no dip).
    Confidently Wrong = low exploration bonus = must learn the hard way.

Justification (Why Synthetic Simulation):
    While RQ1 establishes performance on realistic trace data, assessing the 
    router's ability to unlearn "Confidently Wrong" priors requires precise 
    control over the drift magnitude and prior strength.
    
    Therefore, we conducted a Controlled Simulation to isolate the 
    "Plasticity-Stability Dilemma." We initialized the bandit with an 
    artificially poisoned prior (μ_prior ≪ μ_true) to simulate a "Worst-Case 
    Scenario" where a legacy expert system is suddenly rendered obsolete by 
    a new specialist model. 
    
    This synthetic setup allows us to rigorously measure the Recovery Latency 
    (number of interactions required to correct the belief) without the 
    confounding factors of embedding noise.

Implementation Note:
    This script uses a custom `PoisonedLinUCB` class (not the library's 
    `DisjointLinUCBPolicy`) because it requires:
    
    1. Memory Decay (γ): Standard LinUCB has infinite memory. To unlearn 
       poisoned priors, we need discounted updates: A_new = γ·A_old + xx'
       
    2. Poison Injection: The `inject_poison()` method artificially creates
       "confidently wrong" priors for the controlled experiment.
    
    The library's bandit (used in RQ1) proves real-world performance.
    This custom variant isolates the plasticity mechanism for analysis.

Usage:
    python -m banditgpt.experiment.run_rq2_poisoned

Output:
    - results/rq2/poisoned_adaptation.png - The clean KDD figure
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

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
class PoisonedConfig:
    """Configuration for the poisoned priors experiment."""
    dim: int = 32                    # Embedding dimension
    n_steps: int = 600               # Total simulation steps
    alpha: float = 0.5               # UCB exploration (lower = harder to recover)
    gamma: float = 0.90              # Memory decay (lower = faster recovery)
    poison_strength: float = 100.0   # How wrong the priors are
    seed: int = 42
    output_dir: Path = Path("results/rq2")


# ---------------------------------------------------------------------------
# The 2 Actors
# ---------------------------------------------------------------------------

MODELS = ["gpt-4o", "nova-lite"]
# 0: GPT-4o    - The False Idol (priors love it, reality disappoints)
# 1: Nova-Lite - The Hidden Gem (priors hate it, reality excels)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NicheEnvironment:
    """
    Single-context environment representing a niche task.
    
    Reality:
        - GPT-4o: 0.60 (mediocre on this task)
        - Nova-Lite: 0.95 (excellent specialist)
    """
    
    def __init__(self, dim: int, seed: int = 42):
        np.random.seed(seed)
        self.dim = dim
        
        # The niche task vector
        self.vec_niche = np.random.randn(dim)
        self.vec_niche /= np.linalg.norm(self.vec_niche)
        
        # Ground truth rewards (THE REALITY)
        self.true_rewards = [0.60, 0.95]  # GPT-4o, Nova-Lite
    
    def get_context(self) -> np.ndarray:
        """Return the niche context (with small noise)."""
        noise = np.random.randn(self.dim) * 0.02
        x = self.vec_niche + noise
        return x / np.linalg.norm(x)
    
    def get_reward(self, arm: int) -> float:
        """Get stochastic reward based on true quality."""
        return float(np.random.binomial(1, self.true_rewards[arm]))


# ---------------------------------------------------------------------------
# Poisoned LinUCB
# ---------------------------------------------------------------------------

class PoisonedLinUCB:
    """
    LinUCB with poisoned (confidently wrong) priors.
    
    The poison creates high confidence (large A) with wrong estimates (wrong b).
    This suppresses exploration and forces the bandit to learn the hard way.
    """
    
    def __init__(self, n_models: int, dim: int, alpha: float = 0.5, gamma: float = 0.95):
        self.n_models = n_models
        self.dim = dim
        self.alpha = alpha
        self.gamma = gamma
        
        # Initialize with identity
        self.A = [np.eye(dim) for _ in range(n_models)]
        self.b = [np.zeros(dim) for _ in range(n_models)]
    
    def inject_poison(self, vec: np.ndarray, strength: float = 50.0):
        """
        Create the "Confidently Wrong" state.
        
        THE POISON:
        - GPT-4o: "I am 100% sure this gives reward 1.0" (LIE - it's 0.6)
        - Nova-Lite: "I am 100% sure this gives reward 0.1" (LIE - it's 0.95)
        
        This crushes Nova's UCB score because:
        - High A → low uncertainty (σ small)
        - Low b → low mean estimate (μ small)
        - UCB = μ + α·σ is small for Nova
        """
        # Poison GPT-4o (Arm 0): "Perfect performer"
        self.A[0] += strength * np.outer(vec, vec)
        self.b[0] += strength * 1.0 * vec  # Claims reward = 1.0
        
        # Poison Nova-Lite (Arm 1): "Terrible performer"  
        self.A[1] += strength * np.outer(vec, vec)
        self.b[1] += strength * 0.1 * vec  # Claims reward = 0.1
    
    def select_arm(self, x: np.ndarray) -> Tuple[int, List[float], List[float]]:
        """Select arm and return internal estimates."""
        ucbs = []
        means = []
        
        for i in range(self.n_models):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            
            mean = float(theta.dot(x))
            var = float(x.dot(A_inv).dot(x))
            std = np.sqrt(max(var, 1e-10))
            
            ucb = mean + self.alpha * std
            means.append(mean)
            ucbs.append(ucb)
        
        return int(np.argmax(ucbs)), means, ucbs
    
    def update(self, arm: int, x: np.ndarray, r: float):
        """Update with memory decay to allow unlearning."""
        # Decay ALL memories (allows unlearning the poison)
        for i in range(self.n_models):
            self.A[i] *= self.gamma
            self.b[i] *= self.gamma
            # Keep minimum regularization
            self.A[i] += np.eye(self.dim) * (1 - self.gamma) * 0.01
        
        # Add new observation
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(config: PoisonedConfig) -> dict:
    """
    Run the poisoned priors experiment.
    
    Expected pattern:
    1. Steps 0-~200: Bandit picks GPT-4, gets 0.6, clings to priors (THE DIP)
    2. Steps ~200-~300: GPT-4 estimate drops, eventually tries Nova (THE FLIP)
    3. Steps ~300+: Nova takes over, reward shoots to 0.95 (THE RECOVERY)
    """
    print("=" * 60)
    print("RQ2: Overcoming 'Confidently Wrong' Priors")
    print("=" * 60)
    print(f"  Dimension: {config.dim}")
    print(f"  Steps: {config.n_steps}")
    print(f"  Alpha (exploration): {config.alpha}")
    print(f"  Gamma (memory decay): {config.gamma}")
    print(f"  Poison strength: {config.poison_strength}")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    # Initialize
    env = NicheEnvironment(config.dim, config.seed)
    bandit = PoisonedLinUCB(
        n_models=2,
        dim=config.dim,
        alpha=config.alpha,
        gamma=config.gamma,
    )
    
    # Inject the poison (THE TRAP)
    print("\n[Setup] Injecting poisoned priors...")
    print(f"  GPT-4o:    Prior claims reward = 1.0 (REALITY: 0.60)")
    print(f"  Nova-Lite: Prior claims reward = 0.1 (REALITY: 0.95)")
    print(f"  Poison strength: {config.poison_strength}x")
    bandit.inject_poison(env.vec_niche, config.poison_strength)
    
    # Track history
    rewards: List[float] = []
    arms_selected: List[int] = []
    estimates_gpt4: List[float] = []
    estimates_nova: List[float] = []
    
    gpt4_selections = 0
    nova_selections = 0
    
    print(f"\n[Simulation] Running {config.n_steps} steps on niche task...")
    print("  Expected: GPT-4 dominates early (dip) → Nova discovered (flip) → Nova dominates (recovery)")
    
    for t in range(config.n_steps):
        x = env.get_context()
        arm, means, ucbs = bandit.select_arm(x)
        r = env.get_reward(arm)
        bandit.update(arm, x, r)
        
        rewards.append(r)
        arms_selected.append(arm)
        estimates_gpt4.append(means[0])
        estimates_nova.append(means[1])
        
        if arm == 0:
            gpt4_selections += 1
        else:
            nova_selections += 1
        
        if (t + 1) % 100 == 0:
            recent_reward = np.mean(rewards[-50:]) if t >= 50 else np.mean(rewards)
            recent_nova = sum(arms_selected[-50:]) / min(50, t+1) if t >= 50 else nova_selections / (t+1)
            print(f"    Step {t+1}: reward={recent_reward:.2f}, Nova rate={recent_nova:.0%}, "
                  f"θ_GPT4={means[0]:.2f}, θ_Nova={means[1]:.2f}")
    
    # Find the "flip" point (where Nova estimate exceeds GPT-4)
    flip_step = None
    for t in range(len(estimates_nova)):
        if estimates_nova[t] > estimates_gpt4[t]:
            flip_step = t
            break
    
    print(f"\n[Results]")
    print(f"  GPT-4o selections: {gpt4_selections} ({gpt4_selections/config.n_steps:.1%})")
    print(f"  Nova-Lite selections: {nova_selections} ({nova_selections/config.n_steps:.1%})")
    print(f"  Final average reward: {np.mean(rewards):.3f}")
    if flip_step:
        print(f"  'Flip' occurred at step: {flip_step}")
    
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "rewards": rewards,
        "arms_selected": arms_selected,
        "estimates_gpt4": estimates_gpt4,
        "estimates_nova": estimates_nova,
        "flip_step": flip_step,
        "gpt4_rate": gpt4_selections / config.n_steps,
        "nova_rate": nova_selections / config.n_steps,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, output_path: Path) -> None:
    """
    Generate the clean KDD figure showing:
    1. The adaptation curve (reward over time)
    2. The internal belief change (θ estimates)
    3. The "Dip", "Flip", and "Recovery" annotations
    """
    if not HAS_MATPLOTLIB:
        print("[RQ2] Warning: matplotlib not available, skipping plot")
        return
    
    # KDD Paper Settings
    FONT_SIZE = 10
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rewards = results["rewards"]
    est_gpt4 = results["estimates_gpt4"]
    est_nova = results["estimates_nova"]
    flip_step = results.get("flip_step")
    
    # Smooth rewards
    def smooth(y, window=50):
        box = np.ones(window) / window
        smoothed = np.convolve(y, box, mode='valid')
        # Pad to original length
        pad = len(y) - len(smoothed)
        return np.concatenate([smoothed[:1]] * pad + [smoothed])
    
    steps = range(len(rewards))
    smoothed_rewards = smooth(rewards, window=50)
    
    # Smooth the belief lines for cleaner "Flip" crossover visualization
    smoothed_gpt4 = smooth(est_gpt4, window=10)
    smoothed_nova = smooth(est_nova, window=10)
    
    # Plot 1: Smoothed reward curve (THE MAIN LINE)
    ax.plot(steps, smoothed_rewards, color='#2CA02C', linewidth=3, 
            label='System Accuracy', zorder=10)
    
    # Plot 2: Internal belief estimates (θ) - smoothed for clarity
    ax.plot(steps, smoothed_gpt4, color='#D62728', linestyle='--', alpha=0.7,
            linewidth=2, label='Belief: GPT-4o (θ)')
    ax.plot(steps, smoothed_nova, color='#1F77B4', linestyle='--', alpha=0.7,
            linewidth=2, label='Belief: Nova-Lite (θ)')
    
    # Reference lines (ground truth)
    ax.axhline(0.60, color='#D62728', linestyle=':', alpha=0.4, 
               label='GPT-4o Reality (0.60)')
    ax.axhline(0.95, color='#1F77B4', linestyle=':', alpha=0.4,
               label='Nova Reality (0.95)')
    
    # Mark the flip point
    if flip_step:
        ax.axvline(flip_step, color='purple', linestyle='-', alpha=0.5, linewidth=2)
        ax.annotate('The "Flip"\n(Nova > GPT-4)', 
                   xy=(flip_step, 0.5), 
                   xytext=(flip_step + 50, 0.35),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6e6fa', alpha=0.8))
    
    # Annotations for the story
    # The Dip
    dip_x = min(100, len(rewards)//4)
    dip_y = smoothed_rewards[dip_x] if dip_x < len(smoothed_rewards) else 0.6
    ax.annotate('The "Dip"\n(Clinging to\nPoisoned Priors)', 
               xy=(dip_x, dip_y), 
               xytext=(dip_x + 80, 0.4),
               fontsize=9,
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))
    
    # The Recovery
    if len(rewards) > 400:
        recovery_x = len(rewards) - 100
        recovery_y = smoothed_rewards[recovery_x] if recovery_x < len(smoothed_rewards) else 0.9
        ax.annotate('The "Recovery"\n(Nova Dominates)', 
                   xy=(recovery_x, recovery_y), 
                   xytext=(recovery_x - 100, 0.75),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', alpha=0.8))
    
    ax.set_xlabel("Interactions (Time)", fontsize=11)
    ax.set_ylabel("Reward / Estimated Quality (θ)", fontsize=11)
    ax.set_title("Figure 2: Simulation of Belief Recovery Under Poisoned Priors\n"
                 "(Controlled experiment isolating the Plasticity-Stability Dilemma)", fontsize=11, fontweight='bold')
    
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlim(0, len(rewards))
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"[RQ2] Saved plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RQ2: Overcoming Confidently Wrong Priors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--steps", type=int, default=600, help="Simulation steps")
    parser.add_argument("--alpha", type=float, default=0.5, help="UCB exploration parameter")
    parser.add_argument("--gamma", type=float, default=0.90, help="Memory decay factor")
    parser.add_argument("--poison-strength", type=float, default=100.0, help="Prior poison strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/rq2", help="Output directory")
    
    args = parser.parse_args()
    
    config = PoisonedConfig(
        dim=args.dim,
        n_steps=args.steps,
        alpha=args.alpha,
        gamma=args.gamma,
        poison_strength=args.poison_strength,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    
    # Run simulation
    results = run_simulation(config)
    
    # Save metrics
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "poisoned_metrics.json"
    with open(metrics_path, "w") as f:
        serializable = {
            k: ([float(x) for x in v] if isinstance(v, list) else v)
            for k, v in results.items()
        }
        json.dump(serializable, f, indent=2, default=str)
    print(f"[RQ2] Saved metrics to {metrics_path}")
    
    # Plot
    plot_results(results, config.output_dir / "poisoned_adaptation.png")
    
    print("\n" + "=" * 60)
    print("RQ2 Poisoned Priors Experiment Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
