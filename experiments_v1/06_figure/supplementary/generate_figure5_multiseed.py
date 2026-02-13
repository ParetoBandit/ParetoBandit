"""
Figure 5 Generator: Multi-Seed Statistical Analysis
====================================================
Runs the phased stress test with multiple seeds to provide statistical validation.

This addresses the KDD reviewer concern: "No confidence intervals or multiple seeds"

Reports:
- Mean ± std for decommission time
- Mean ± std for final weights
- Mean ± std for cumulative losses
- Distribution of reaction times
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK EXPERTS (Same as main script)
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
    """Simulates a distribution shift at t=shift_step."""
    def __init__(self, shift_step: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.shift_step = shift_step
        self.t = 0
        
    def get_reward(self, model: str) -> float:
        self.t += 1
        
        # Phase 1: Neutral Zone (Both models are good)
        if self.t < self.shift_step:
            reward = self.rng.normal(0.85, 0.05)
            return np.clip(reward, 0.0, 1.0)
            
        # Phase 2: Alignment Tax Zone (Mismatch)
        else:
            if model == "mistralai/mixtral-8x7b-instruct":
                reward = self.rng.normal(0.9, 0.05)
            else:
                reward = self.rng.normal(0.2, 0.08)
            return np.clip(reward, 0.0, 1.0)


# ============================================================================
# RUNNER (Single Seed)
# ============================================================================

def run_single_seed(seed: int, n_steps=300, shift_step=50, learning_rate=0.3) -> Dict:
    """Run one trial with a specific seed."""
    np.random.seed(seed)
    
    # Setup
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    # Experts
    warmup = StubbornExpert("Warmup Expert (Stubborn)", "openai/gpt-4-turbo")
    tabula = SmartExpert("Tabula Rasa (Smart)", "mistralai/mixtral-8x7b-instruct")
    
    # Router
    router = CorrallingRouter(experts=[warmup, tabula], models=models, 
                              learning_rate=learning_rate, gamma=0.05)
    
    env = PhasedEnvironment(shift_step=shift_step, seed=seed)
    
    history = {"weights": [], "losses": {"warmup": [], "tabula": []}}
    
    for t in range(n_steps):
        context = np.random.randn(10)
        selected_model = router.select_model(context)
        reward = env.get_reward(selected_model)
        
        router.update(context, selected_model, reward)
        
        history["weights"].append(router.weights.copy())
        history["losses"]["warmup"].append(router.cumulative_losses[0])
        history["losses"]["tabula"].append(router.cumulative_losses[1])
    
    # Compute metrics
    weights = np.array(history["weights"])
    decom_idx = np.where((np.arange(len(weights)) >= shift_step) & (weights[:, 0] < 0.1))[0]
    
    result = {
        "seed": seed,
        "final_warmup_weight": weights[-1, 0],
        "final_tr_weight": weights[-1, 1],
        "final_warmup_loss": history["losses"]["warmup"][-1],
        "final_tr_loss": history["losses"]["tabula"][-1],
        "decommission_step": decom_idx[0] if len(decom_idx) > 0 else n_steps,
        "reaction_time": (decom_idx[0] - shift_step) if len(decom_idx) > 0 else (n_steps - shift_step),
        "warmup_selections": router.expert_selections[0],
        "tr_selections": router.expert_selections[1],
    }
    
    return result


# ============================================================================
# MULTI-SEED ANALYSIS
# ============================================================================

def run_multiseed_analysis(n_seeds=20, n_steps=300, shift_step=50, learning_rate=0.3):
    """Run multiple trials and compute statistics."""
    logger.info("="*70)
    logger.info("MULTI-SEED STATISTICAL ANALYSIS")
    logger.info("="*70)
    logger.info(f"Configuration: {n_seeds} seeds, n={n_steps}, shift={shift_step}, η={learning_rate}, γ=0.05\n")
    
    results = []
    
    for seed in range(n_seeds):
        result = run_single_seed(seed, n_steps, shift_step, learning_rate)
        results.append(result)
        
        if (seed + 1) % 5 == 0:
            logger.info(f"  Completed {seed+1}/{n_seeds} trials...")
    
    logger.info(f"\n✅ Completed all {n_seeds} trials\n")
    
    # Compute statistics
    decom_times = [r["decommission_step"] for r in results]
    reaction_times = [r["reaction_time"] for r in results]
    final_warmup_weights = [r["final_warmup_weight"] for r in results]
    final_tr_weights = [r["final_tr_weight"] for r in results]
    warmup_losses = [r["final_warmup_loss"] for r in results]
    tr_losses = [r["final_tr_loss"] for r in results]
    
    stats = {
        "decom_mean": np.mean(decom_times),
        "decom_std": np.std(decom_times),
        "decom_min": np.min(decom_times),
        "decom_max": np.max(decom_times),
        "reaction_mean": np.mean(reaction_times),
        "reaction_std": np.std(reaction_times),
        "warmup_weight_mean": np.mean(final_warmup_weights),
        "warmup_weight_std": np.std(final_warmup_weights),
        "tr_weight_mean": np.mean(final_tr_weights),
        "tr_weight_std": np.std(final_tr_weights),
        "warmup_loss_mean": np.mean(warmup_losses),
        "warmup_loss_std": np.std(warmup_losses),
        "tr_loss_mean": np.mean(tr_losses),
        "tr_loss_std": np.std(tr_losses),
        "loss_gap_mean": np.mean([w - t for w, t in zip(warmup_losses, tr_losses)]),
        "loss_gap_std": np.std([w - t for w, t in zip(warmup_losses, tr_losses)]),
    }
    
    # Report
    logger.info("="*70)
    logger.info("STATISTICAL RESULTS")
    logger.info("="*70)
    logger.info(f"\n📊 Decommissioning Time (< 10% threshold):")
    logger.info(f"   Mean: {stats['decom_mean']:.1f} ± {stats['decom_std']:.1f} steps")
    logger.info(f"   Range: [{stats['decom_min']:.0f}, {stats['decom_max']:.0f}]")
    
    logger.info(f"\n📊 Reaction Time (shift → decommission):")
    logger.info(f"   Mean: {stats['reaction_mean']:.1f} ± {stats['reaction_std']:.1f} steps")
    
    logger.info(f"\n📊 Final Weights (at t={n_steps}):")
    logger.info(f"   Warmup: {stats['warmup_weight_mean']:.4f} ± {stats['warmup_weight_std']:.4f}")
    logger.info(f"   Tabula Rasa: {stats['tr_weight_mean']:.4f} ± {stats['tr_weight_std']:.4f}")
    
    logger.info(f"\n📊 Cumulative Losses:")
    logger.info(f"   Warmup: {stats['warmup_loss_mean']:.1f} ± {stats['warmup_loss_std']:.1f}")
    logger.info(f"   Tabula Rasa: {stats['tr_loss_mean']:.1f} ± {stats['tr_loss_std']:.1f}")
    logger.info(f"   Loss Gap: {stats['loss_gap_mean']:.1f} ± {stats['loss_gap_std']:.1f}")
    
    return results, stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_multiseed_results(results: List[Dict], stats: Dict, output_dir: Path):
    """Generate visualization showing distribution of key metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Decommissioning Time Distribution
    ax1 = axes[0, 0]
    decom_times = [r["decommission_step"] for r in results]
    ax1.hist(decom_times, bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.axvline(stats["decom_mean"], color='darkred', linestyle='--', linewidth=2, 
                label=f'Mean: {stats["decom_mean"]:.1f} ± {stats["decom_std"]:.1f}')
    ax1.set_xlabel("Decommissioning Time (steps)", fontweight='bold')
    ax1.set_ylabel("Frequency", fontweight='bold')
    ax1.set_title("Distribution of Decommissioning Times", fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reaction Time Distribution
    ax2 = axes[0, 1]
    reaction_times = [r["reaction_time"] for r in results]
    ax2.hist(reaction_times, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(stats["reaction_mean"], color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {stats["reaction_mean"]:.1f} ± {stats["reaction_std"]:.1f}')
    ax2.set_xlabel("Reaction Time (shift → decommission)", fontweight='bold')
    ax2.set_ylabel("Frequency", fontweight='bold')
    ax2.set_title("Distribution of Reaction Times", fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Weights
    ax3 = axes[1, 0]
    warmup_weights = [r["final_warmup_weight"] for r in results]
    tr_weights = [r["final_tr_weight"] for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    ax3.bar(x - width/2, warmup_weights, width, label='Warmup', color='#e74c3c', alpha=0.7)
    ax3.bar(x + width/2, tr_weights, width, label='Tabula Rasa', color='#27ae60', alpha=0.7)
    ax3.axhline(stats["warmup_weight_mean"], color='darkred', linestyle='--', alpha=0.5)
    ax3.axhline(stats["tr_weight_mean"], color='darkgreen', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Trial Number", fontweight='bold')
    ax3.set_ylabel("Final Weight", fontweight='bold')
    ax3.set_title(f"Final Weights Across {len(results)} Trials", fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Loss Gap Distribution
    ax4 = axes[1, 1]
    loss_gaps = [r["final_warmup_loss"] - r["final_tr_loss"] for r in results]
    ax4.hist(loss_gaps, bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.axvline(stats["loss_gap_mean"], color='darkorange', linestyle='--', linewidth=2,
                label=f'Mean: {stats["loss_gap_mean"]:.1f} ± {stats["loss_gap_std"]:.1f}')
    ax4.set_xlabel("Loss Gap (Warmup - Tabula Rasa)", fontweight='bold')
    ax4.set_ylabel("Frequency", fontweight='bold')
    ax4.set_title("Distribution of Cumulative Loss Gaps", fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    out_png = output_dir / "figure5_multiseed_statistics.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved PNG: {out_png}")
    
    out_pdf = output_dir / "figure5_multiseed_statistics.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PDF: {out_pdf}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("FIGURE 5: Multi-Seed Statistical Validation")
    logger.info("="*70)
    logger.info("\nThis addresses the KDD reviewer concern:")
    logger.info("'No confidence intervals or multiple seeds'\n")
    
    # Run multi-seed analysis
    results, stats = run_multiseed_analysis(
        n_seeds=20,  # 20 trials for robust statistics
        n_steps=300,
        shift_step=50,
        learning_rate=0.3
    )
    
    # Generate visualization
    output_dir = Path(__file__).parent / "results"
    plot_multiseed_results(results, stats, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    logger.info("\n💡 Key Findings:")
    logger.info(f"   - Decommissioning is robust: {stats['decom_mean']:.1f} ± {stats['decom_std']:.1f} steps")
    logger.info(f"   - Reaction time is consistent: {stats['reaction_mean']:.1f} ± {stats['reaction_std']:.1f} steps")
    logger.info(f"   - Final convergence is decisive: Warmup {stats['warmup_weight_mean']:.4f}, TR {stats['tr_weight_mean']:.4f}")
    logger.info(f"   - Loss gap validates decommission: {stats['loss_gap_mean']:.1f} ± {stats['loss_gap_std']:.1f}\n")


if __name__ == "__main__":
    main()
