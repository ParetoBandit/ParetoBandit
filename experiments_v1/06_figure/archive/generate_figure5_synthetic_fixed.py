"""
Figure 5 Generator: Synthetic Stress Test (FIXED)
=================================================
Forces a DETERMINISTIC "Worst-Case Mismatch" scenario.

CRITICAL FIX:
Previous version had a mathematical flaw - adding bias to b-vectors with random 
zero-mean contexts resulted in random predictions (not systematic bias).

This version uses stubborn mock experts to GUARANTEE the behavior we need to test:
- Stubborn Expert: ALWAYS picks GPT-4 (simulates wrong prior)
- Smart Expert: ALWAYS picks Mixtral (simulates correct learning)
- Environment: Mixtral gets 0.9 reward, GPT-4 gets 0.2 reward
- Result: Corralling MUST decommission the Stubborn Expert

This is honest scientific practice: we're testing the ALGORITHM'S properties
(can it detect and decommission a failing expert?), not making claims about
real LinUCB dynamics.

Scenario:
- Stubborn Expert (Warmup): ALWAYS picks GPT-4 (The "Bad" Model)
- Smart Expert (Tabula Rasa): Mostly picks Mixtral (The "Good" Model)
- Synthetic Environment: Mixtral=0.9, GPT-4=0.2
- Corralling coordinates and adapts weights based on observed losses
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK EXPERTS (Deterministic Behavior for Stress Test)
# ============================================================================

class StubbornExpert:
    """
    Simulates a misspecified warmup prior that stubbornly prefers expensive models.
    
    This represents a prior learned on hard reasoning tasks (where GPT-4 excels)
    being applied to a chat-heavy distribution (where Mixtral excels).
    
    Always selects the expensive model, ignoring all evidence.
    """
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
        self.cumulative_regret = 0.0
        self.t = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        """Always pick the favorite model (GPT-4)."""
        return self.favorite_model
    
    def update(self, context, model, reward):
        """Stubborn expert never learns."""
        self.t += 1


class SmartExpert:
    """
    Simulates a tabula rasa expert that has learned the correct model.
    
    Mostly picks the good model (Mixtral) but explores occasionally.
    This represents an unbiased learner that has discovered the quality inversion.
    """
    def __init__(self, name: str, best_model: str, exploration_rate: float = 0.05):
        self.name = name
        self.best_model = best_model
        self.exploration_rate = exploration_rate
        self.cumulative_regret = 0.0
        self.t = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        """Mostly pick best model, occasionally explore."""
        if np.random.random() < self.exploration_rate:
            # Explore the other model
            return "openai/gpt-4-turbo" if self.best_model != "openai/gpt-4-turbo" else "mistralai/mixtral-8x7b-instruct"
        return self.best_model
    
    def update(self, context, model, reward):
        """Smart expert continues learning (though behavior is mostly fixed)."""
        self.t += 1


# ============================================================================
# SYNTHETIC REWARD ENVIRONMENT
# ============================================================================

class SyntheticEnvironment:
    """
    Generates rewards for the stress test scenario.
    
    Quality inversion: Cheap model (Mixtral) outperforms expensive model (GPT-4).
    This represents a distribution shift from the training data.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reward_params = {
            "mistralai/mixtral-8x7b-instruct": (0.90, 0.05),  # High quality, low variance
            "openai/gpt-4-turbo": (0.20, 0.08),               # Low quality (distribution shift!)
        }
    
    def get_reward(self, model: str) -> float:
        """Sample reward from the model's distribution."""
        mean, std = self.reward_params.get(model, (0.5, 0.1))
        reward = self.rng.normal(mean, std)
        return np.clip(reward, 0.0, 1.0)


# ============================================================================
# STRESS TEST RUNNER
# ============================================================================

def run_synthetic_stress_test(
    n_steps: int = 500,
    learning_rate: float = 1.0,
    gamma: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    Run the corralling stress test with deterministic experts.
    
    Args:
        n_steps: Number of routing decisions
        learning_rate: Corralling η parameter (higher = faster adaptation)
        gamma: Exploration floor (prevents expert death)
        seed: Random seed
        
    Returns:
        Dictionary with trajectories
    """
    np.random.seed(seed)
    logger.info("="*70)
    logger.info("SYNTHETIC STRESS TEST: Deterministic Experts")
    logger.info("="*70)
    logger.info(f"Configuration: n={n_steps}, η={learning_rate}, γ={gamma}")
    
    # Setup models
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    # Create experts
    logger.info("\nCreating Experts:")
    warmup_expert = StubbornExpert(
        name="Warmup Prior (Stubborn)",
        favorite_model="openai/gpt-4-turbo"  # Always picks the BAD model
    )
    logger.info("  ✓ Stubborn Expert: ALWAYS picks GPT-4 (simulates wrong prior)")
    
    tabula_expert = SmartExpert(
        name="Tabula Rasa (Smart)",
        best_model="mistralai/mixtral-8x7b-instruct",  # Picks the GOOD model
        exploration_rate=0.05
    )
    logger.info("  ✓ Smart Expert: Mostly picks Mixtral (95%), explores 5%")
    
    # Create Corralling coordinator
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_expert],
        models=models,
        learning_rate=learning_rate,
        gamma=gamma
    )
    logger.info(f"\n✓ CorrallingRouter initialized (η={learning_rate}, γ={gamma})")
    
    # Create environment
    env = SyntheticEnvironment(seed=seed)
    logger.info("\n✓ Synthetic Environment: Mixtral μ=0.9, GPT-4 μ=0.2")
    
    # Run simulation
    logger.info(f"\nRunning {n_steps} routing decisions...\n")
    
    history = {
        "weights": [],
        "losses": {"warmup": [], "tabula_rasa": []},
        "expert_selections": [],
        "model_selections": []
    }
    
    for t in range(n_steps):
        # Dummy context (not used by stubborn experts, but needed for API)
        context = np.random.randn(10)
        
        # Router selects expert, expert selects model
        selected_model = router.select_model(context)
        expert_idx = router.last_expert_idx
        
        # Get reward from synthetic environment
        reward = env.get_reward(selected_model)
        
        # Update router (triggers weight update)
        router.update(context, selected_model, reward)
        
        # Track history
        history["weights"].append(router.weights.copy())
        history["losses"]["warmup"].append(router.cumulative_losses[0])
        history["losses"]["tabula_rasa"].append(router.cumulative_losses[1])
        history["expert_selections"].append(expert_idx)
        history["model_selections"].append(selected_model)
        
        # Progress logging
        if (t + 1) % 100 == 0:
            weights = router.weights
            logger.info(
                f"  Step {t+1:3d}/{n_steps} | "
                f"Weights: Warmup={weights[0]:.3f}, TR={weights[1]:.3f} | "
                f"Losses: W={router.cumulative_losses[0]:.1f}, TR={router.cumulative_losses[1]:.1f}"
            )
    
    # Convert to numpy
    history["weights"] = np.array(history["weights"])
    
    # Find key transition points
    warmup_weights = history["weights"][:, 0]
    decom_idx = np.where(warmup_weights < 0.1)[0]
    crossover_idx = np.where(history["weights"][:, 1] > history["weights"][:, 0])[0]
    
    decom_step = decom_idx[0] if len(decom_idx) > 0 else n_steps
    crossover_step = crossover_idx[0] if len(crossover_idx) > 0 else 0
    
    history["decom_step"] = decom_step
    history["crossover_step"] = crossover_step
    history["learning_rate"] = learning_rate
    history["gamma"] = gamma
    history["n_steps"] = n_steps
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    logger.info(f"Crossover point (TR > Warmup): t = {crossover_step}")
    logger.info(f"Decommissioning point (Warmup < 10%): t = {decom_step}")
    logger.info(f"Final weights: Warmup={warmup_weights[-1]:.4f}, TR={history['weights'][-1, 1]:.4f}")
    logger.info(f"Expert selections: Warmup={router.expert_selections[0]}, TR={router.expert_selections[1]}")
    
    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(history: Dict, output_dir: Path):
    """Generate Figure 5 with clean, publication-quality visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights = history["weights"]
    n_steps = len(weights)
    t = np.arange(n_steps)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # =========== Plot 1: Weight Evolution ===========
    ax1.plot(t, weights[:, 0], color='#e74c3c', linewidth=2.5, label='Warmup Expert (Stubborn)')
    ax1.plot(t, weights[:, 1], color='#27ae60', linewidth=2.5, label='Tabula Rasa (Smart)')
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Uniform (50/50)')
    ax1.axhline(y=0.1, color='#e74c3c', linestyle=':', alpha=0.5, label='Decommission Threshold')
    
    # Annotate key transition points
    crossover_step = history["crossover_step"]
    decom_step = history["decom_step"]
    
    if crossover_step < n_steps:
        ax1.axvline(x=crossover_step, color='#3498db', linestyle='--', alpha=0.6)
        ax1.text(crossover_step + 10, 0.9, f'Crossover\n(t={crossover_step})', 
                 fontsize=10, color='#2980b9', fontweight='bold')
    
    if decom_step < n_steps:
        ax1.axvline(x=decom_step, color='#e74c3c', linestyle='--', alpha=0.6)
        ax1.annotate(
            f'Decisive Decommissioning\n(Warmup < 10% at t={decom_step})',
            xy=(decom_step, 0.1),
            xytext=(decom_step + 80, 0.35),
            fontsize=11,
            fontweight='bold',
            color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#c0392b', alpha=0.95)
        )
    
    ax1.set_ylabel("Expert Weight $p_{i,t}$", fontsize=14, fontweight='bold')
    ax1.set_title(
        "Corralling Algorithm: Synthetic Stress Test (Deterministic Experts)\n"
        f"(Mixtral μ=0.9, GPT-4 μ=0.2, η={history['learning_rate']}, γ={history['gamma']})",
        fontsize=16, fontweight='bold', pad=15
    )
    ax1.legend(loc='right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(-0.05, 1.05)
    
    # =========== Plot 2: Cumulative Loss ===========
    ax2.plot(t, history["losses"]["warmup"], 
             color='#e74c3c', linewidth=2.5, label='Warmup Cumulative Loss')
    ax2.plot(t, history["losses"]["tabula_rasa"], 
             color='#27ae60', linewidth=2.5, label='Tabula Rasa Cumulative Loss')
    
    # Loss gap annotation
    final_warmup = history["losses"]["warmup"][-1]
    final_tr = history["losses"]["tabula_rasa"][-1]
    loss_gap = final_warmup - final_tr
    
    if loss_gap > 0:
        ax2.annotate(
            f"Loss Gap: +{loss_gap:.1f}\n(Warmup incurs {loss_gap/max(final_tr, 0.01)*100:.0f}% more loss)",
            xy=(n_steps * 0.7, (final_warmup + final_tr) / 2),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95)
        )
    
    ax2.set_xlabel("Routing Step (t)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Cumulative Importance-Weighted Loss", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    # Save
    out_pdf = output_dir / "figure5_corralling_weights.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PDF: {out_pdf}")
    
    out_png = output_dir / "figure5_corralling_weights.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PNG: {out_png}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("FIGURE 5: Corralling Stress Test (Deterministic Experts)")
    logger.info("="*70)
    logger.info("\nThis is a CONTROLLED EXPERIMENT using stubborn mock experts")
    logger.info("to test the Corralling algorithm's decommissioning behavior.")
    logger.info("This is NOT a claim about real LinUCB dynamics.\n")
    
    # Run stress test with aggressive learning rate for clear visualization
    results = run_synthetic_stress_test(
        n_steps=500,
        learning_rate=1.0,  # Aggressive for visible step function
        gamma=0.05,  # Standard exploration floor
        seed=42
    )
    
    # Generate visualization
    output_dir = Path(__file__).parent / "results"
    plot_results(results, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()

