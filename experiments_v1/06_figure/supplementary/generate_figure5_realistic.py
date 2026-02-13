"""
Figure 5 Alternative: Realistic Reward Scenario
================================================
Tests Corralling with realistic LMSYS-like reward distributions.

Real LMSYS data shows:
- GPT-4-Turbo: μ=0.812, σ≈0.10
- Mixtral-8x7B: μ=0.823, σ≈0.09
- Difference: +0.011 (very small!)

This tests: Can Corralling detect and decommission with realistic effect sizes?

Cohen's d = (0.823 - 0.812) / 0.095 ≈ 0.12 (very small effect)
vs synthetic extreme: d = (0.9 - 0.2) / 0.065 ≈ 10.8 (huge effect)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK EXPERTS (Same as before)
# ============================================================================

class StubbornExpert:
    """Always picks GPT-4 (simulates rigid prior)."""
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
        self.cumulative_regret = 0.0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, cost=0.0):
        pass


class SmartExpert:
    """Mostly picks Mixtral (simulates adaptive learner)."""
    def __init__(self, name: str, best_model: str):
        self.name = name
        self.best_model = best_model
        self.cumulative_regret = 0.0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        if np.random.random() < 0.05:  # 5% exploration
            return "openai/gpt-4-turbo"
        return self.best_model
    
    def update(self, context, model, reward, cost=0.0):
        pass


# ============================================================================
# REALISTIC ENVIRONMENT (LMSYS-like distributions)
# ============================================================================

class RealisticEnvironment:
    """
    Uses realistic LMSYS reward distributions with small effect size.
    
    Real data:
    - GPT-4-Turbo: μ=0.812, σ≈0.10
    - Mixtral-8x7B: μ=0.823, σ≈0.09
    - Effect size: d ≈ 0.12 (Cohen's d)
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Phase 1: Both models similar (neutral baseline)
        self.phase1_params = {
            "mistralai/mixtral-8x7b-instruct": (0.82, 0.09),
            "openai/gpt-4-turbo": (0.81, 0.10),
        }
        # Phase 2: Small but consistent difference (realistic scenario)
        self.phase2_params = {
            "mistralai/mixtral-8x7b-instruct": (0.823, 0.09),  # Real LMSYS
            "openai/gpt-4-turbo": (0.812, 0.10),               # Real LMSYS
        }
        self.shift_step = 100  # More samples needed to detect small effect
        self.t = 0
    
    def get_reward(self, model: str) -> float:
        self.t += 1
        
        # Choose phase
        if self.t < self.shift_step:
            params = self.phase1_params
        else:
            params = self.phase2_params
        
        mean, std = params.get(model, (0.5, 0.1))
        reward = self.rng.normal(mean, std)
        return np.clip(reward, 0.0, 1.0)


# ============================================================================
# RUNNER
# ============================================================================

def run_realistic_test(seed: int, n_steps=1000, learning_rate=0.1) -> Dict:
    """
    Run with realistic reward distributions.
    
    Key differences from synthetic:
    - Smaller effect size: d ≈ 0.12 (not d ≈ 10.8)
    - Higher variance: σ ≈ 0.10 (not σ ≈ 0.05)
    - More samples needed: 1000 steps (not 300)
    - Lower learning rate: η = 0.1 (not η = 0.3) to avoid oscillations from noise
    """
    np.random.seed(seed)
    
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    warmup = StubbornExpert("Warmup (GPT-4)", "openai/gpt-4-turbo")
    tabula = SmartExpert("Tabula Rasa (Mixtral)", "mistralai/mixtral-8x7b-instruct")
    
    router = CorrallingRouter(experts=[warmup, tabula], models=models,
                              learning_rate=learning_rate, gamma=0.05)
    
    env = RealisticEnvironment(seed=seed)
    
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
    shift_step = 100
    decom_idx = np.where((np.arange(len(weights)) >= shift_step) & (weights[:, 0] < 0.1))[0]
    
    result = {
        "seed": seed,
        "decommissioned": len(decom_idx) > 0,
        "decommission_step": decom_idx[0] if len(decom_idx) > 0 else None,
        "reaction_time": (decom_idx[0] - shift_step) if len(decom_idx) > 0 else None,
        "final_warmup_weight": weights[-1, 0],
        "final_tr_weight": weights[-1, 1],
        "final_warmup_loss": history["losses"]["warmup"][-1],
        "final_tr_loss": history["losses"]["tabula"][-1],
    }
    
    return result, history


# ============================================================================
# MULTI-SEED ANALYSIS
# ============================================================================

def run_realistic_multiseed(n_seeds=20, n_steps=1000, learning_rate=0.1):
    """Test realistic scenario across multiple seeds."""
    logger.info("="*70)
    logger.info("REALISTIC SCENARIO: LMSYS-like Reward Distributions")
    logger.info("="*70)
    logger.info(f"Configuration: {n_seeds} seeds, n={n_steps}, shift=100, η={learning_rate}")
    logger.info(f"Phase 1 (t<100): Both ≈μ=0.81-0.82 (neutral)")
    logger.info(f"Phase 2 (t≥100): Mixtral μ=0.823, GPT-4 μ=0.812 (d≈0.12)")
    logger.info(f"Effect size: SMALL (real LMSYS data)\n")
    
    results = []
    histories = []
    
    for seed in range(n_seeds):
        result, history = run_realistic_test(seed, n_steps, learning_rate)
        results.append(result)
        histories.append(history)
        
        if (seed + 1) % 5 == 0:
            logger.info(f"  Completed {seed+1}/{n_seeds} trials...")
    
    logger.info(f"\n✅ Completed all {n_seeds} trials\n")
    
    # Stats
    decommissioned_trials = [r for r in results if r["decommissioned"]]
    n_decom = len(decommissioned_trials)
    
    logger.info("="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"\n📊 Decommissioning Success Rate:")
    logger.info(f"   {n_decom}/{n_seeds} trials ({n_decom/n_seeds*100:.1f}%) decommissioned failing expert")
    
    if n_decom > 0:
        decom_times = [r["decommission_step"] for r in decommissioned_trials]
        reaction_times = [r["reaction_time"] for r in decommissioned_trials]
        
        logger.info(f"\n📊 Decommissioning Time (successful trials only):")
        logger.info(f"   Mean: {np.mean(decom_times):.1f} ± {np.std(decom_times):.1f} steps")
        logger.info(f"   Range: [{np.min(decom_times):.0f}, {np.max(decom_times):.0f}]")
        
        logger.info(f"\n📊 Reaction Time (shift → decommission):")
        logger.info(f"   Mean: {np.mean(reaction_times):.1f} ± {np.std(reaction_times):.1f} steps")
    
    # Final weights for ALL trials (including non-decommissioned)
    final_warmup = [r["final_warmup_weight"] for r in results]
    final_tr = [r["final_tr_weight"] for r in results]
    
    logger.info(f"\n📊 Final Weights (all trials, t={n_steps}):")
    logger.info(f"   Warmup: {np.mean(final_warmup):.3f} ± {np.std(final_warmup):.3f}")
    logger.info(f"   Tabula Rasa: {np.mean(final_tr):.3f} ± {np.std(final_tr):.3f}")
    
    logger.info(f"\n💡 Interpretation:")
    if n_decom / n_seeds > 0.5:
        logger.info(f"   ✅ Corralling can detect small effect sizes (d≈0.12) in majority of cases")
        logger.info(f"   ⚠️  Requires more samples: mean reaction ~{np.mean(reaction_times) if n_decom > 0 else 'N/A':.0f} steps")
    else:
        logger.info(f"   ⚠️  Small effect size (d≈0.12) is challenging to detect reliably")
        logger.info(f"   💡 May need: (1) more samples, (2) lower η, or (3) domain-specific tuning")
    
    return results, histories


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_realistic_comparison(results, histories, output_dir: Path):
    """Plot one successful trial to compare with synthetic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find a trial that decommissioned
    decom_idx = next((i for i, r in enumerate(results) if r["decommissioned"]), None)
    
    if decom_idx is None:
        logger.warning("No trial decommissioned. Plotting first trial anyway.")
        decom_idx = 0
    
    history = histories[decom_idx]
    result = results[decom_idx]
    
    weights = np.array(history["weights"])
    t = np.arange(len(weights))
    shift_step = 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Weights ---
    ax1.plot(t, weights[:, 0], color='#e74c3c', linewidth=2.5, label='Warmup Expert (GPT-4)')
    ax1.plot(t, weights[:, 1], color='#27ae60', linewidth=2.5, label='Tabula Rasa (Mixtral)')
    
    ax1.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.1, color='#e74c3c', linestyle=':', alpha=0.5, label='Decommission Threshold')
    
    # Phase labels
    ax1.axvspan(0, shift_step, color='gray', alpha=0.05)
    ax1.text(shift_step/2, 0.85, "Phase 1: Neutral\n(Both ≈μ=0.81)", 
             ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    if result["decommissioned"]:
        decom_step = result["decommission_step"]
        ax1.axvline(x=decom_step, color='#e74c3c', linestyle='--', alpha=0.6)
        ax1.annotate(
            f'Decommissioning\n(t={decom_step}, Δt={result["reaction_time"]} after shift)',
            xy=(decom_step, 0.1),
            xytext=(decom_step + 150, 0.4),
            fontsize=10,
            fontweight='bold',
            color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#c0392b', alpha=0.95)
        )
    
    ax1.set_ylabel("Expert Weight $p_{i,t}$", fontsize=12, fontweight='bold')
    ax1.set_title(
        f"Corralling with Realistic LMSYS Reward Distributions (Seed {decom_idx})\n"
        f"Phase 2: Mixtral μ=0.823, GPT-4 μ=0.812 (d≈0.12, η=0.1, γ=0.05)",
        fontsize=13, fontweight='bold', pad=15
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(-0.05, 1.05)
    
    # --- Plot 2: Losses ---
    ax2.plot(t, history["losses"]["warmup"], color='#e74c3c', linewidth=2.5, label='Warmup Cumulative Loss')
    ax2.plot(t, history["losses"]["tabula"], color='#27ae60', linewidth=2.5, label='Tabula Rasa Cumulative Loss')
    
    ax2.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    
    final_gap = history["losses"]["warmup"][-1] - history["losses"]["tabula"][-1]
    ax2.annotate(
        f"Loss Gap: +{final_gap:.1f}",
        xy=(len(t) * 0.75, (history["losses"]["warmup"][-1] + history["losses"]["tabula"][-1]) / 2),
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95)
    )
    
    ax2.set_xlabel("Routing Step (t)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Cumulative Importance-Weighted Loss", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    out_png = output_dir / "appendixE_realistic_scenario.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved PNG: {out_png}")
    
    out_pdf = output_dir / "appendixE_realistic_scenario.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PDF: {out_pdf}")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("FIGURE 5 ALTERNATIVE: Realistic Reward Scenario")
    logger.info("="*70)
    logger.info("\nTests: Can Corralling detect small effect sizes (d≈0.12)?")
    logger.info("Context: Real LMSYS shows Mixtral μ=0.823, GPT-4 μ=0.812\n")
    
    results, histories = run_realistic_multiseed(
        n_seeds=20,
        n_steps=1000,  # More samples needed for small effect
        learning_rate=0.1  # Lower η to reduce oscillations from noise
    )
    
    output_dir = Path(__file__).parent / "results"
    plot_realistic_comparison(results, histories, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
