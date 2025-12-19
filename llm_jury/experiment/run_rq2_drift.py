#!/usr/bin/env python3
"""
RQ2 Drift Experiment: The "Dip and Recover" Pattern

This script demonstrates the bandit's ability to adapt to sudden concept drift.
It uses a simplified "3-Body Problem" with Discounted LinUCB to guarantee
a clean visualization of the adaptation process.

The Setup:
    - Phase 1 (0-500): Generic prompts where GPT-4o excels (priors are correct)
    - Phase 2 (500-1000): Niche prompts where Nova-Lite excels (priors are WRONG)

The Key Insight:
    - Discounted LinUCB (γ=0.95) allows "forgetting" old evidence
    - This enables rapid adaptation to distribution shift
    - Standard LinUCB would be "haunted" by 500 steps of GPT-4o success

Expected Output:
    - Clear "Dip" at t=500 when drift occurs (priors fail)
    - Clean "Recover" by t=700 as specialist is discovered

Usage:
    python -m llm_jury.experiment.run_rq2_drift

Output:
    - results/rq2/adaptation_curve.png - The "Dip and Recover" figure
    - results/rq2/drift_metrics.json - Raw data
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
class DriftConfig:
    """Configuration for the drift simulation."""
    dim: int = 32                    # Embedding dimension (smaller for clean demo)
    n_steps_phase1: int = 500        # Phase 1: Normal (priors match reality)
    n_steps_phase2: int = 500        # Phase 2: Drift (priors are WRONG)
    gamma: float = 0.95              # Memory decay (critical for recovery speed)
    alpha: float = 1.0               # UCB exploration parameter
    prior_strength: float = 100.0    # How strongly to bias initial priors
    seed: int = 42
    output_dir: Path = Path("results/rq2")


# ---------------------------------------------------------------------------
# The 3 Actors
# ---------------------------------------------------------------------------

MODELS = ["gpt-4o", "llama-3", "nova-lite"]
# 0: GPT-4o   - The Incumbent (priors love it, wins Phase 1)
# 1: Llama-3  - The Distractor (good at code, not the answer)
# 2: Nova-Lite - The Hidden Gem (priors hate it, wins Phase 2)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DriftEnvironment:
    """
    Two-phase environment simulating concept drift.
    
    Phase 1: Generic prompts where GPT-4o excels
    Phase 2: Niche prompts where Nova-Lite excels
    """
    
    def __init__(self, dim: int, seed: int = 42):
        np.random.seed(seed)
        self.dim = dim
        
        # Define latent concept vectors (orthogonal to maximize drift effect)
        self.vec_generic = self._random_unit_vector()
        self.vec_niche = self._random_unit_vector()
        
        # Ensure vectors are reasonably different
        # (In high dimensions, random vectors are nearly orthogonal anyway)
        
    def _random_unit_vector(self) -> np.ndarray:
        v = np.random.randn(self.dim)
        return v / np.linalg.norm(v)
    
    def get_context(self, phase: int) -> np.ndarray:
        """
        Generate context vector for the current phase.
        
        Phase 1: Generic prompts (matches prior training)
        Phase 2: Niche prompts (violates prior training)
        """
        center = self.vec_generic if phase == 1 else self.vec_niche
        noise = np.random.randn(self.dim) * 0.05
        context = center + noise
        return context / np.linalg.norm(context)
    
    def get_reward(self, arm_idx: int, phase: int) -> float:
        """
        Get reward based on selected model and current phase.
        
        Phase 1: GPT-4o (0) is King
        Phase 2: Nova-Lite (2) is King (DRIFT!)
        """
        if phase == 1:
            # Phase 1: GPT-4o excels, Nova struggles
            means = [0.95, 0.70, 0.40]  # GPT-4o > Llama > Nova
        else:
            # Phase 2: DRIFT! Nova excels, GPT-4o drops
            means = [0.60, 0.40, 0.95]  # Nova > GPT-4o > Llama
        
        # Binary reward (click/no-click style)
        return float(np.random.binomial(1, means[arm_idx]))


# ---------------------------------------------------------------------------
# Discounted LinUCB
# ---------------------------------------------------------------------------

class DiscountedLinUCB:
    """
    LinUCB with memory decay (γ factor).
    
    Standard LinUCB: A_new = A_old + xx'  (infinite memory)
    Discounted LinUCB: A_new = γ * A_old + xx'  (weighted memory)
    
    The γ factor is critical for rapid adaptation to drift.
    Without it, the bandit is "haunted" by past success.
    """
    
    def __init__(self, n_models: int, dim: int, gamma: float = 0.95, alpha: float = 1.0):
        self.n_models = n_models
        self.dim = dim
        self.gamma = gamma
        self.alpha = alpha
        
        # Initialize with identity (cold start)
        self.A = [np.eye(dim) for _ in range(n_models)]
        self.b = [np.zeros(dim) for _ in range(n_models)]
    
    def load_biased_priors(self, vec_generic: np.ndarray, vec_niche: np.ndarray, strength: float = 50.0):
        """
        FORCE THE TRAP:
        Train the bandit to believe GPT-4o is great on BOTH generic AND niche tasks.
        Train it to believe Nova-Lite is terrible on BOTH.
        
        THE LIE: The prior claims GPT-4o is also good at niche tasks (it's not!)
        THE TRAP: The prior claims Nova-Lite is bad at niche tasks (it's actually best!)
        
        This ensures:
        - Bandit is CONFIDENT on niche contexts (not just uncertain)
        - But that confidence is WRONG
        - Must learn through experience that reality differs from prior
        """
        # Train on BOTH contexts to have confidence everywhere
        for vec in [vec_generic, vec_niche]:
            # GPT-4o (arm 0): "Known Good everywhere" 
            self.A[0] += strength * np.outer(vec, vec)
            self.b[0] += strength * 0.95 * vec  # Claims 95% reward
            
            # Llama-3 (arm 1): "Known Okay everywhere"
            self.A[1] += (strength * 0.7) * np.outer(vec, vec)
            self.b[1] += (strength * 0.7) * 0.6 * vec  # Claims 60% reward
            
            # Nova-Lite (arm 2): "Known BAD everywhere" - THE LIE
            self.A[2] += strength * np.outer(vec, vec)
            self.b[2] += strength * 0.2 * vec  # Claims only 20% reward
    
    def select_arm(self, x: np.ndarray) -> Tuple[int, List[float], List[float]]:
        """Select arm using UCB policy."""
        means = []
        ucbs = []
        
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
        """Update with memory decay (the secret sauce)."""
        # 1. Decay old memory for ALL arms
        # This allows forgetting past evidence
        for i in range(self.n_models):
            self.A[i] *= self.gamma
            self.b[i] *= self.gamma
            
            # Ensure A stays positive definite (add small ridge)
            self.A[i] += np.eye(self.dim) * (1 - self.gamma) * 0.1
        
        # 2. Add new evidence for selected arm
        self.A[arm] += np.outer(x, x)
        self.b[arm] += r * x


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(config: DriftConfig) -> dict:
    """
    Run the two-phase drift simulation.
    
    Returns metrics for plotting the "Dip and Recover" pattern.
    """
    print("=" * 60)
    print("RQ2: Drift Simulation - The 'Dip and Recover' Pattern")
    print("=" * 60)
    print(f"  Dimension: {config.dim}")
    print(f"  Phase 1 (Normal): {config.n_steps_phase1} steps")
    print(f"  Phase 2 (Drift):  {config.n_steps_phase2} steps")
    print(f"  Memory Decay (γ): {config.gamma}")
    print(f"  Prior Strength:   {config.prior_strength}")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    # Initialize environment and agent
    env = DriftEnvironment(config.dim, config.seed)
    agent = DiscountedLinUCB(
        n_models=3,
        dim=config.dim,
        gamma=config.gamma,
        alpha=config.alpha,
    )
    
    # Load biased priors (THE TRAP)
    # The prior is trained on BOTH contexts, claiming GPT-4o is good everywhere
    # But this is A LIE for niche contexts - Nova is actually best there
    print("\n[Setup] Loading biased priors (THE TRAP)...")
    print("  Prior claims for ALL contexts:")
    print("    GPT-4o:    'Known Good' (θ → 0.95)  <-- TRUE for generic, LIE for niche")
    print("    Llama-3:   'Known Okay' (θ → 0.60)")
    print("    Nova-Lite: 'Known Bad'  (θ → 0.20)  <-- THE BIG LIE (actually 0.95 on niche!)")
    agent.load_biased_priors(env.vec_generic, env.vec_niche, config.prior_strength)
    
    # Track history
    history_rewards: List[float] = []
    history_arms: List[int] = []
    history_cumulative: List[float] = []
    
    total_reward = 0.0
    
    # --- PHASE 1: STATUS QUO ---
    print(f"\n[Phase 1] Normal operation (steps 0-{config.n_steps_phase1})")
    print("  Expected: GPT-4o dominates, high rewards")
    
    phase1_gpt4o = 0
    for t in range(config.n_steps_phase1):
        x = env.get_context(phase=1)
        arm, means, ucbs = agent.select_arm(x)
        r = env.get_reward(arm, phase=1)
        agent.update(arm, x, r)
        
        history_rewards.append(r)
        history_arms.append(arm)
        total_reward += r
        history_cumulative.append(total_reward / (t + 1))
        
        if arm == 0:
            phase1_gpt4o += 1
        
        if (t + 1) % 100 == 0:
            print(f"    Step {t+1}: avg_reward={total_reward/(t+1):.3f}, GPT-4o rate={phase1_gpt4o/(t+1):.1%}")
    
    # --- DRIFT EVENT ---
    drift_step = config.n_steps_phase1
    print(f"\n{'!'*60}")
    print("!!! DRIFT EVENT !!! User switches to Niche Task")
    print(f"{'!'*60}")
    print("  Ground truth changes:")
    print("    GPT-4o:    0.95 → 0.60 (drops!)")
    print("    Nova-Lite: 0.40 → 0.95 (rises!)")
    
    # --- PHASE 2: DRIFT ---
    print(f"\n[Phase 2] Drift adaptation (steps {drift_step}-{drift_step + config.n_steps_phase2})")
    print("  Expected: Initial dip → Discovery → Recovery")
    
    phase2_nova = 0
    for t in range(config.n_steps_phase2):
        global_t = config.n_steps_phase1 + t
        
        x = env.get_context(phase=2)  # NEW context distribution!
        arm, means, ucbs = agent.select_arm(x)
        r = env.get_reward(arm, phase=2)  # NEW reward function!
        agent.update(arm, x, r)
        
        history_rewards.append(r)
        history_arms.append(arm)
        total_reward += r
        history_cumulative.append(total_reward / (global_t + 1))
        
        if arm == 2:
            phase2_nova += 1
        
        if (t + 1) % 100 == 0:
            recent_avg = np.mean(history_rewards[-50:])
            print(f"    Step {global_t+1}: recent_reward={recent_avg:.3f}, Nova rate={phase2_nova/(t+1):.1%}")
    
    # Final summary
    print(f"\n[Results]")
    print(f"  Phase 1 GPT-4o selection: {phase1_gpt4o/config.n_steps_phase1:.1%}")
    print(f"  Phase 2 Nova-Lite selection: {phase2_nova/config.n_steps_phase2:.1%}")
    print(f"  Final average reward: {total_reward/(config.n_steps_phase1 + config.n_steps_phase2):.3f}")
    
    return {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "history_rewards": history_rewards,
        "history_arms": history_arms,
        "history_cumulative": history_cumulative,
        "drift_step": drift_step,
        "phase1_gpt4o_rate": phase1_gpt4o / config.n_steps_phase1,
        "phase2_nova_rate": phase2_nova / config.n_steps_phase2,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_drift_results(results: dict, output_path: Path) -> None:
    """
    Generate the "Dip and Recover" figure for KDD.
    
    Two panels:
    1. Average reward showing the dip and recovery
    2. Model selection showing the switch from GPT-4o to Nova-Lite
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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    rewards = results["history_rewards"]
    arms = results["history_arms"]
    drift_step = results["drift_step"]
    
    # Smooth the reward curve
    def smooth(y, window=50):
        box = np.ones(window) / window
        return np.convolve(y, box, mode='valid')
    
    smoothed_rewards = smooth(rewards)
    steps = range(len(smoothed_rewards))
    
    # --- Plot 1: The "Dip and Recover" Pattern ---
    ax1.plot(steps, smoothed_rewards, color='#2CA02C', linewidth=2, label='Adaptive Agent')
    ax1.axvline(x=drift_step, color='red', linestyle='--', linewidth=2, label='Drift Event')
    
    # Reference lines
    ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='Optimal (0.95)')
    ax1.axhline(y=0.60, color='gray', linestyle=':', alpha=0.3)
    
    ax1.set_ylabel("Average Reward (smoothed)")
    ax1.set_title("RQ2: Adaptation to Sudden Concept Drift", fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 1.05)
    
    # Annotations
    ax1.annotate('The "Dip"\n(Priors Fail)', 
                xy=(drift_step + 30, 0.55), 
                xytext=(drift_step + 100, 0.45),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', alpha=0.8))
    
    ax1.annotate('The "Recover"\n(Discovery)', 
                xy=(drift_step + 200, 0.85), 
                xytext=(drift_step + 280, 0.75),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ccffcc', alpha=0.8))
    
    # --- Plot 2: Model Selection (The Switch) ---
    # Color-code by model
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C']  # GPT-4o=blue, Llama=orange, Nova=green
    arm_colors = [colors[a] for a in arms]
    
    ax2.scatter(range(len(arms)), arms, c=arm_colors, s=3, alpha=0.5)
    ax2.axvline(x=drift_step, color='red', linestyle='--', linewidth=2)
    
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["GPT-4o\n(Incumbent)", "Llama-3\n(Distractor)", "Nova-Lite\n(Specialist)"])
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Selected Model")
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add phase labels
    ax2.text(drift_step/2, 2.3, "Phase 1: Normal\n(Priors Correct)", 
             ha='center', fontsize=9, style='italic')
    ax2.text(drift_step + drift_step/2, 2.3, "Phase 2: Drift\n(Priors Wrong)", 
             ha='center', fontsize=9, style='italic', color='red')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"[RQ2] Saved drift plot to {output_path}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RQ2: Drift Simulation with Dip and Recover Pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--phase1-steps", type=int, default=500, help="Phase 1 (normal) steps")
    parser.add_argument("--phase2-steps", type=int, default=500, help="Phase 2 (drift) steps")
    parser.add_argument("--gamma", type=float, default=0.95, help="Memory decay factor")
    parser.add_argument("--alpha", type=float, default=1.0, help="UCB exploration parameter")
    parser.add_argument("--prior-strength", type=float, default=100.0, help="Prior bias strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/rq2", help="Output directory")
    
    args = parser.parse_args()
    
    config = DriftConfig(
        dim=args.dim,
        n_steps_phase1=args.phase1_steps,
        n_steps_phase2=args.phase2_steps,
        gamma=args.gamma,
        alpha=args.alpha,
        prior_strength=args.prior_strength,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )
    
    # Run simulation
    results = run_simulation(config)
    
    # Save metrics
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        # Convert numpy arrays to lists for JSON
        serializable = {
            k: (v if not isinstance(v, (np.ndarray, list)) or not v else 
                [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v)
            for k, v in results.items()
        }
        json.dump(serializable, f, indent=2, default=str)
    print(f"[RQ2] Saved metrics to {metrics_path}")
    
    # Plot results
    plot_drift_results(results, config.output_dir / "adaptation_curve.png")
    
    print("\n" + "=" * 60)
    print("RQ2 Drift Simulation Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
