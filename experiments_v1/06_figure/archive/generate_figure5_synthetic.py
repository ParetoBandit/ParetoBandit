"""
Figure 5 Generator: Phased Stress Test (FIXED)
==============================================
Visualizes "Reaction to Distribution Shift".

Scenario:
- t=0 to 50: "Easy" prompts. Both models work well. System maintains 50/50 trust.
- t=50+: "Hard/Strict" prompts arrive. Warmup Expert (GPT-4) fails.
- Result: Visible "Knee" in the curve where the system detects the shift and decommissions.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# MOCK EXPERTS (Deterministic Behavior for Stress Test)
# ============================================================================

class StubbornExpert:
    """Simulates a Warmup Prior that stubbornly loves the expensive model."""
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
        self.cumulative_regret = 0.0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, cost=0.0):
        pass

class SmartExpert:
    """Simulates a Tabula Rasa expert that has found the cheap model."""
    def __init__(self, name: str, best_model: str):
        self.name = name
        self.best_model = best_model
        self.cumulative_regret = 0.0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        # Adds slight noise to simulate exploration, but mostly picks best
        if np.random.random() < 0.05: # 5% exploration
            return "openai/gpt-4-turbo"
        return self.best_model
    
    def update(self, context, model, reward, cost=0.0):
        pass

# ============================================================================
# PHASED ENVIRONMENT
# ============================================================================

class PhasedEnvironment:
    """
    Simulates a distribution shift at t=shift_step.
    """
    def __init__(self, shift_step: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.shift_step = shift_step
        self.t = 0
        
    def get_reward(self, model: str) -> float:
        self.t += 1
        
        # Phase 1: Neutral Zone (Both models are good)
        if self.t < self.shift_step:
            # Both get high reward (simulating easy tasks)
            reward = self.rng.normal(0.85, 0.05)
            return np.clip(reward, 0.0, 1.0)
            
        # Phase 2: Alignment Tax Zone (Mismatch)
        else:
            if model == "mistralai/mixtral-8x7b-instruct":
                reward = self.rng.normal(0.9, 0.05) # Cheap model stays good
            else:
                reward = self.rng.normal(0.2, 0.08) # Expensive model fails
            return np.clip(reward, 0.0, 1.0)

# ============================================================================
# RUNNER
# ============================================================================

def run_phased_stress_test(n_steps=300, shift_step=50, learning_rate=0.3):
    # Setup
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    # Experts
    warmup = StubbornExpert("Warmup Expert (Stubborn)", "openai/gpt-4-turbo") # Always picks GPT-4
    tabula = SmartExpert("Tabula Rasa (Smart)", "mistralai/mixtral-8x7b-instruct") # Picks Mixtral
    
    # Router
    router = CorrallingRouter(experts=[warmup, tabula], models=models, 
                              learning_rate=learning_rate, gamma=0.05)
    
    env = PhasedEnvironment(shift_step=shift_step)
    
    history = {"weights": [], "losses": {"warmup": [], "tabula": []}}
    
    logger.info(f"Running Phased Stress Test: Shift at t={shift_step}, η={learning_rate}, γ=0.05")
    
    for t in range(n_steps):
        context = np.random.randn(10)
        selected_model = router.select_model(context)
        reward = env.get_reward(selected_model)
        
        router.update(context, selected_model, reward)
        
        history["weights"].append(router.weights.copy())
        history["losses"]["warmup"].append(router.cumulative_losses[0])
        history["losses"]["tabula"].append(router.cumulative_losses[1])
        
        if (t + 1) % 100 == 0:
            logger.info(f"  Step {t+1}/{n_steps} | Weights: W={router.weights[0]:.3f}, TR={router.weights[1]:.3f}")

    # Summary
    weights = np.array(history["weights"])
    decom_idx = np.where((np.arange(len(weights)) >= shift_step) & (weights[:, 0] < 0.1))[0]
    if len(decom_idx) > 0:
        decom_step = decom_idx[0]
        reaction_time = decom_step - shift_step
        logger.info(f"  Decommissioning at t={decom_step} (Δt={reaction_time} after shift)")
    
    logger.info(f"  Final weights: W={weights[-1, 0]:.4f}, TR={weights[-1, 1]:.4f}")
    logger.info(f"  Loss gap: +{history['losses']['warmup'][-1] - history['losses']['tabula'][-1]:.1f}")

    return history, shift_step

def plot_phased_results(history, shift_step, output_dir):
    weights = np.array(history["weights"])
    t = np.arange(len(weights))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- Plot 1: Weights ---
    ax1.plot(t, weights[:, 0], color='#e74c3c', linewidth=3, label='Warmup Expert (Prior)')
    ax1.plot(t, weights[:, 1], color='#27ae60', linewidth=3, label='Tabula Rasa (Online)')
    
    # Visualizing Phases
    ax1.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add colored background for phases
    ax1.axvspan(0, shift_step, color='gray', alpha=0.05)
    ax1.axvspan(shift_step, len(t), color='#e74c3c', alpha=0.05)
    
    ax1.text(shift_step/2, 0.8, "Phase 1: Exploration\n(Both Models Good)", 
             ha='center', va='center', color='gray', fontweight='bold', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
             
    # Annotation: Distribution Shift
    ax1.annotate('Distribution Shift\n(Alignment Tax Emerges)', 
                 xy=(shift_step, 0.5), 
                 xytext=(shift_step+35, 0.6),
                 arrowprops=dict(facecolor='#2980b9', shrink=0.05, lw=2),
                 fontsize=10, fontweight='bold', color='#2980b9',
                 bbox=dict(facecolor='lightyellow', edgecolor='#2980b9', boxstyle='round,pad=0.5'))
    
    # Annotation: The Drop
    drop_idx = np.where(weights[:, 0] < 0.1)[0]
    if len(drop_idx) > 0:
        actual_drop = drop_idx[0]
        if actual_drop > shift_step:
            reaction_time = actual_drop - shift_step
            ax1.annotate(f'Decisive Decommissioning\n(t={actual_drop}, Δt={reaction_time} after shift)', 
                         xy=(actual_drop, 0.1), 
                         xytext=(actual_drop+50, 0.35),
                         arrowprops=dict(facecolor='#c0392b', shrink=0.05, lw=2),
                         fontsize=10, fontweight='bold', color='#c0392b',
                         bbox=dict(facecolor='white', edgecolor='#c0392b', boxstyle='round,pad=0.5'))

    ax1.set_ylabel("Expert Weight $p_{i,t}$", fontsize=12, fontweight='bold')
    ax1.set_title("Corralling Algorithm: Phased Stress Test (Distribution Shift Detection)\n(Shift at t=50, η=0.3, γ=0.05)", 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.1, color='#e74c3c', linestyle=':', alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # --- Plot 2: Loss ---
    ax2.plot(t, history["losses"]["warmup"], color='#e74c3c', linewidth=2.5, label='Warmup Cumulative Loss')
    ax2.plot(t, history["losses"]["tabula"], color='#27ae60', linewidth=2.5, label='Tabula Rasa Cumulative Loss')
    
    # Show Divergence Point
    ax2.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvspan(shift_step, len(t), color='#e74c3c', alpha=0.05)
    
    # Annotation: Loss Divergence
    warmup_loss_at_shift = history["losses"]["warmup"][shift_step]
    ax2.annotate('Losses Diverge\n(After Shift)', 
                 xy=(shift_step, warmup_loss_at_shift), 
                 xytext=(shift_step+35, warmup_loss_at_shift+25),
                 arrowprops=dict(facecolor='#2980b9', shrink=0.05, lw=2),
                 fontsize=10, fontweight='bold', color='#2980b9',
                 bbox=dict(facecolor='white', edgecolor='#2980b9', boxstyle='round,pad=0.5'))
    
    # Loss gap
    final_warmup = history["losses"]["warmup"][-1]
    final_tr = history["losses"]["tabula"][-1]
    loss_gap = final_warmup - final_tr
    
    ax2.annotate(
        f"Loss Gap: +{loss_gap:.1f}\n(Warmup incurs {(loss_gap/final_tr)*100:.0f}% more loss)",
        xy=(len(t)*0.75, (final_warmup + final_tr)/2),
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95)
    )
                 
    ax2.set_xlabel("Routing Step (t)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Cumulative Importance-Weighted Loss", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    # Save outputs
    plt.savefig(output_dir / "figure5_corralling_weights.png", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PNG: {output_dir / 'figure5_corralling_weights.png'}")
    
    plt.savefig(output_dir / "figure5_corralling_weights.pdf", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PDF: {output_dir / 'figure5_corralling_weights.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("FIGURE 5: Phased Stress Test (Distribution Shift Detection)")
    logger.info("="*70)
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # η=0.3 provides a nice visible curve after the shift
    results, shift_step = run_phased_stress_test(n_steps=300, shift_step=50, learning_rate=0.3)
    plot_phased_results(results, shift_step, output_dir)
    
    logger.info("="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
