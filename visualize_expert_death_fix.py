"""
Visualization: Expert Death Prevention via Mixing Parameter

This script demonstrates the difference between:
1. Pure exponential weighting (gamma=0) - leads to Expert Death
2. Mixed weighting (gamma=0.05) - prevents Expert Death

It simulates a non-stationary environment where:
- Phase 1 (0-500): Expert 0 is better
- Phase 2 (500-1000): Expert 1 becomes better (environment shift)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import sys
sys.path.insert(0, '.')

from src.bandit_gpt.router import CorrallingRouter


class MockExpert:
    """Mock expert for simulation."""
    
    def __init__(self, model_id: str, models: List[str]):
        self.model_id = model_id
        self.models = models
        
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.model_id
    
    def update(self, context: np.ndarray, model: str, reward: float):
        pass


def simulate_nonstationary_environment(gamma: float, n_steps: int = 1000):
    """
    Simulate non-stationary environment with phase shift.
    
    Returns:
        weights_history: Array of shape (n_steps, 2) with expert weights
        probs_history: Array of shape (n_steps, 2) with expert probabilities
    """
    models = ["model_a", "model_b"]
    
    expert_0 = MockExpert("model_a", models)
    expert_1 = MockExpert("model_b", models)
    
    router = CorrallingRouter(
        experts=[expert_0, expert_1],
        models=models,
        learning_rate=0.3,
        gamma=gamma
    )
    
    context = np.random.randn(10)
    weights_history = []
    probs_history = []
    
    for step in range(n_steps):
        # Get current probabilities
        probs = router._get_mixed_distribution()
        weights_history.append(router.weights.copy())
        probs_history.append(probs.copy())
        
        # Select model
        np.random.seed(42 + step)
        model = router.select_model(context)
        
        # Phase shift at step 500
        if step < 500:
            # Phase 1: Expert 0 is better
            reward = 0.8 if model == "model_a" else 0.2
        else:
            # Phase 2: Expert 1 becomes better (environment shift!)
            reward = 0.2 if model == "model_a" else 0.8
        
        router.update(context, model, reward)
    
    return np.array(weights_history), np.array(probs_history)


def plot_comparison():
    """Create comparison plot showing Expert Death vs Prevention."""
    
    print("Simulating Expert Death scenario (gamma=0)...")
    weights_no_mix, probs_no_mix = simulate_nonstationary_environment(gamma=0.0)
    
    print("Simulating with Expert Death prevention (gamma=0.05)...")
    weights_mix, probs_mix = simulate_nonstationary_environment(gamma=0.05)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Expert Death Prevention: γ=0 vs γ=0.05', fontsize=16, fontweight='bold')
    
    steps = np.arange(len(weights_no_mix))
    
    # --- Row 1: Raw Weights ---
    
    # Subplot 1: Weights without mixing (gamma=0)
    ax = axes[0, 0]
    ax.plot(steps, weights_no_mix[:, 0], label='Expert 0 (Warmup)', color='#2E86AB', linewidth=2)
    ax.plot(steps, weights_no_mix[:, 1], label='Expert 1 (Tabula Rasa)', color='#A23B72', linewidth=2)
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Phase Shift')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Expert Weight', fontsize=11)
    ax.set_title('Raw Weights (γ=0): Expert Death Occurs', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([1e-10, 1])
    
    # Subplot 2: Weights with mixing (gamma=0.05)
    ax = axes[0, 1]
    ax.plot(steps, weights_mix[:, 0], label='Expert 0 (Warmup)', color='#2E86AB', linewidth=2)
    ax.plot(steps, weights_mix[:, 1], label='Expert 1 (Tabula Rasa)', color='#A23B72', linewidth=2)
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Phase Shift')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Expert Weight', fontsize=11)
    ax.set_title('Raw Weights (γ=0.05): Still Drops Low', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([1e-10, 1])
    
    # --- Row 2: Selection Probabilities (What Matters!) ---
    
    # Subplot 3: Probabilities without mixing (gamma=0)
    ax = axes[1, 0]
    ax.plot(steps, probs_no_mix[:, 0], label='Expert 0 (Warmup)', color='#2E86AB', linewidth=2)
    ax.plot(steps, probs_no_mix[:, 1], label='Expert 1 (Tabula Rasa)', color='#A23B72', linewidth=2)
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Phase Shift')
    ax.axhline(0.025, color='green', linestyle=':', alpha=0.5, label='Min Prob (γ/K)')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Selection Probability', fontsize=11)
    ax.set_title('Selection Probability (γ=0): Cannot Recover', fontsize=12, fontweight='bold', color='red')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add annotation
    ax.annotate('Expert Death!\nCannot detect\nExpert 1 is now better',
                xy=(700, probs_no_mix[700, 1]), xytext=(750, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Subplot 4: Probabilities with mixing (gamma=0.05)
    ax = axes[1, 1]
    ax.plot(steps, probs_mix[:, 0], label='Expert 0 (Warmup)', color='#2E86AB', linewidth=2)
    ax.plot(steps, probs_mix[:, 1], label='Expert 1 (Tabula Rasa)', color='#A23B72', linewidth=2)
    ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='Phase Shift')
    ax.axhline(0.025, color='green', linestyle=':', alpha=0.5, label='Min Prob (γ/K = 0.025)')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Selection Probability', fontsize=11)
    ax.set_title('Selection Probability (γ=0.05): Recovers! ✅', fontsize=12, fontweight='bold', color='green')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add annotation
    ax.annotate('Recovery!\nExpert 1 weight\nincreases after shift',
                xy=(700, probs_mix[700, 1]), xytext=(750, 0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'results/expert_death_prevention.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print("\n--- Phase 2 (after shift at step 500) ---")
    print(f"\nWithout Mixing (γ=0):")
    print(f"  Expert 1 final weight: {weights_no_mix[-1, 1]:.2e}")
    print(f"  Expert 1 final probability: {probs_no_mix[-1, 1]:.4f}")
    print(f"  Expert 1 avg probability (steps 500-1000): {probs_no_mix[500:, 1].mean():.4f}")
    
    print(f"\nWith Mixing (γ=0.05):")
    print(f"  Expert 1 final weight: {weights_mix[-1, 1]:.2e}")
    print(f"  Expert 1 final probability: {probs_mix[-1, 1]:.4f}")
    print(f"  Expert 1 avg probability (steps 500-1000): {probs_mix[500:, 1].mean():.4f}")
    
    print(f"\n✅ Mixing parameter provides {probs_mix[500:, 1].mean() / probs_no_mix[500:, 1].mean():.1f}x higher")
    print(f"   probability for Expert 1 after the phase shift!")
    
    plt.show()


if __name__ == "__main__":
    plot_comparison()

