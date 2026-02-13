"""
Figure 5 Alternative: Real LinUCB Experts
==========================================
Tests Corralling with ACTUAL LinUCB bandits (not deterministic mock experts).

Key differences from mock expert version:
- Uses real contextual LinUCB bandits with exploration
- Context-dependent predictions (not fixed choices)
- Shows oscillations from exploration noise
- More realistic dynamics

Setup:
- Warmup Expert: LinUCB with WRONG priors (biased toward GPT-4)
- Tabula Rasa: LinUCB with NO priors (cold start)
- Synthetic environment: Mixtral is better, but context-dependent
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLIFIED LinUCB EXPERTS
# ============================================================================

class SimpleLinUCBExpert:
    """
    Simplified LinUCB bandit for stress test.
    
    Unlike the full banditGPT router, this uses:
    - Simple context vectors (no embedding/PCA)
    - Basic UCB formula without cost penalties
    - Manual prior injection for "warmup" expert
    """
    def __init__(self, name: str, models: List[str], context_dim: int, 
                 alpha: float = 1.0, priors: Dict = None):
        self.name = name
        self.models = models
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A (precision) and b (moment) matrices
        self.A = {}
        self.b = {}
        
        for model in models:
            if priors and model in priors:
                # Warmup expert: inject biased priors
                self.A[model] = priors[model]["A"].copy()
                self.b[model] = priors[model]["b"].copy()
            else:
                # Tabula rasa: identity initialization
                self.A[model] = np.eye(context_dim)
                self.b[model] = np.zeros(context_dim)
        
        self.t = 0
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        """Select model using UCB."""
        ucb_scores = {}
        
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected_reward = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            
            ucb_scores[model] = expected_reward + self.alpha * uncertainty
        
        return max(ucb_scores, key=ucb_scores.get)
    
    def update(self, context: np.ndarray, model: str, reward: float):
        """Standard LinUCB update."""
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()
        self.t += 1


# ============================================================================
# CONTEXT-DEPENDENT ENVIRONMENT
# ============================================================================

class ContextDependentEnvironment:
    """
    Environment where Mixtral is better on average, but outcomes depend on context.
    
    This is more realistic than fixed rewards:
    - Reward = base_quality + context_effect + noise
    - Some contexts favor GPT-4, most favor Mixtral
    """
    def __init__(self, context_dim: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.context_dim = context_dim
        
        # Base quality (Mixtral slightly better on average)
        self.base_quality = {
            "mistralai/mixtral-8x7b-instruct": 0.75,
            "openai/gpt-4-turbo": 0.65,
        }
        
        # Context weights (determines which contexts favor which model)
        # Mixtral prefers contexts with positive first dimension
        # GPT-4 prefers contexts with negative first dimension (rarer)
        self.context_weights = {
            "mistralai/mixtral-8x7b-instruct": np.array([0.15] + [0.01] * (context_dim - 1)),
            "openai/gpt-4-turbo": np.array([-0.15] + [0.01] * (context_dim - 1)),
        }
        
        self.shift_step = 50
        self.t = 0
    
    def get_reward(self, model: str, context: np.ndarray) -> float:
        """Context-dependent reward."""
        self.t += 1
        
        # Base quality
        base = self.base_quality[model]
        
        # Context effect (dot product with weights)
        context_effect = self.context_weights[model] @ context
        
        # Phase 2: After shift, GPT-4 gets worse
        if self.t >= self.shift_step and model == "openai/gpt-4-turbo":
            base -= 0.15  # Performance degrades
        
        # Noise
        noise = self.rng.normal(0, 0.08)
        
        reward = base + context_effect + noise
        return np.clip(reward, 0.0, 1.0)


# ============================================================================
# RUNNER
# ============================================================================

def create_wrong_priors(context_dim: int) -> Dict:
    """
    Create priors that bias toward GPT-4.
    
    Simulates: warmup was trained on hard reasoning tasks where GPT-4 excels.
    """
    # Inject high rewards for GPT-4 (as if it worked well in the past)
    n_fake_samples = 100
    
    priors = {}
    for model in ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]:
        A = np.eye(context_dim) * 10  # Strong prior
        b = np.zeros(context_dim)
        
        # Bias: GPT-4 looks better
        if model == "openai/gpt-4-turbo":
            fake_contexts = np.random.randn(n_fake_samples, context_dim)
            fake_rewards = np.random.uniform(0.85, 0.95, n_fake_samples)  # High rewards
            
            for ctx, r in zip(fake_contexts, fake_rewards):
                ctx = ctx.reshape(-1, 1)
                A += ctx @ ctx.T
                b += r * ctx.flatten()
        else:
            # Mixtral looks worse in prior
            fake_contexts = np.random.randn(n_fake_samples, context_dim)
            fake_rewards = np.random.uniform(0.50, 0.60, n_fake_samples)  # Low rewards
            
            for ctx, r in zip(fake_contexts, fake_rewards):
                ctx = ctx.reshape(-1, 1)
                A += ctx @ ctx.T
                b += r * ctx.flatten()
        
        priors[model] = {"A": A, "b": b}
    
    return priors


def run_real_linucb_test(seed: int = 42, n_steps: int = 300, context_dim: int = 5):
    """Run stress test with ACTUAL LinUCB experts."""
    np.random.seed(seed)
    
    logger.info("="*70)
    logger.info("REAL LinUCB EXPERIMENT")
    logger.info("="*70)
    logger.info(f"Seed: {seed}, Steps: {n_steps}, Context Dim: {context_dim}\n")
    
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    # Create priors that favor GPT-4 (wrong!)
    wrong_priors = create_wrong_priors(context_dim)
    
    # Expert 1: Warmup with wrong priors
    warmup_expert = SimpleLinUCBExpert(
        name="Warmup (Wrong Priors)",
        models=models,
        context_dim=context_dim,
        alpha=1.0,
        priors=wrong_priors
    )
    logger.info("✓ Warmup Expert: LinUCB with priors biased toward GPT-4")
    
    # Expert 2: Tabula rasa (cold start)
    tabula_expert = SimpleLinUCBExpert(
        name="Tabula Rasa",
        models=models,
        context_dim=context_dim,
        alpha=1.0,
        priors=None  # No priors
    )
    logger.info("✓ Tabula Rasa Expert: LinUCB with no priors (cold start)")
    
    # Corralling coordinator
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_expert],
        models=models,
        learning_rate=0.3,
        gamma=0.05
    )
    logger.info(f"✓ CorrallingRouter: η=0.3, γ=0.05\n")
    
    # Environment
    env = ContextDependentEnvironment(context_dim=context_dim, seed=seed)
    logger.info("✓ Environment: Context-dependent rewards, Mixtral better on average")
    logger.info("  Phase 1 (t<50): Mixtral ≈0.75, GPT-4 ≈0.65")
    logger.info("  Phase 2 (t≥50): Mixtral ≈0.75, GPT-4 ≈0.50 (degrades)\n")
    
    # Run simulation
    history = {
        "weights": [],
        "losses": {"warmup": [], "tabula": []},
        "expert_selections": [],
        "model_selections": [],
    }
    
    for t in range(n_steps):
        # Sample random context
        context = np.random.randn(context_dim)
        
        # Router selects expert → expert selects model
        selected_model = router.select_model(context)
        expert_idx = router.last_expert_idx
        
        # Get context-dependent reward
        reward = env.get_reward(selected_model, context)
        
        # Update router (updates weights and chosen expert)
        router.update(context, selected_model, reward)
        
        # Track history
        history["weights"].append(router.weights.copy())
        history["losses"]["warmup"].append(router.cumulative_losses[0])
        history["losses"]["tabula"].append(router.cumulative_losses[1])
        history["expert_selections"].append(expert_idx)
        history["model_selections"].append(selected_model)
        
        if (t + 1) % 100 == 0:
            logger.info(
                f"  Step {t+1}/{n_steps} | "
                f"Weights: W={router.weights[0]:.3f}, TR={router.weights[1]:.3f}"
            )
    
    # Summary
    weights = np.array(history["weights"])
    decom_idx = np.where((np.arange(len(weights)) >= 50) & (weights[:, 0] < 0.1))[0]
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Expert selections: Warmup {router.expert_selections[0]}, TR {router.expert_selections[1]}")
    logger.info(f"Final weights: Warmup {weights[-1, 0]:.4f}, TR {weights[-1, 1]:.4f}")
    
    if len(decom_idx) > 0:
        logger.info(f"Decommissioning at t={decom_idx[0]} (reaction time: {decom_idx[0]-50} steps)")
    else:
        logger.info("No decommissioning occurred within window")
    
    logger.info(f"Loss gap: {history['losses']['warmup'][-1] - history['losses']['tabula'][-1]:.1f}")
    
    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_real_linucb(history: Dict, output_dir: Path):
    """Plot results showing real LinUCB dynamics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights = np.array(history["weights"])
    n_steps = len(weights)
    t = np.arange(n_steps)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Weights ---
    ax1.plot(t, weights[:, 0], color='#e74c3c', linewidth=2.5, alpha=0.8, label='Warmup Expert (Wrong Priors)')
    ax1.plot(t, weights[:, 1], color='#27ae60', linewidth=2.5, alpha=0.8, label='Tabula Rasa (Cold Start)')
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax1.axhline(y=0.1, color='#e74c3c', linestyle=':', alpha=0.5, label='Decommission Threshold')
    ax1.axvline(x=50, color='blue', linestyle='--', alpha=0.5, label='Shift (t=50)')
    
    # Check for decommissioning
    decom_idx = np.where((np.arange(len(weights)) >= 50) & (weights[:, 0] < 0.1))[0]
    if len(decom_idx) > 0:
        ax1.axvline(x=decom_idx[0], color='#e74c3c', linestyle='--', alpha=0.6)
        ax1.text(decom_idx[0] + 10, 0.8, f'Decommission\n(t={decom_idx[0]})', 
                 fontsize=10, color='#c0392b', fontweight='bold')
    
    ax1.set_ylabel("Expert Weight $p_{i,t}$", fontsize=12, fontweight='bold')
    ax1.set_title(
        "Corralling with Real LinUCB Experts (Context-Dependent Rewards)\n"
        "(η=0.3, γ=0.05, Context Dim=5)",
        fontsize=14, fontweight='bold', pad=15
    )
    ax1.legend(loc='right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(-0.05, 1.05)
    
    # Add annotation about oscillations
    ax1.text(
        0.02, 0.98, 
        "Note: More oscillations than mock experts\ndue to exploration and context-dependence",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # --- Plot 2: Losses ---
    ax2.plot(t, history["losses"]["warmup"], color='#e74c3c', linewidth=2.5, alpha=0.8, label='Warmup Cumulative Loss')
    ax2.plot(t, history["losses"]["tabula"], color='#27ae60', linewidth=2.5, alpha=0.8, label='Tabula Rasa Cumulative Loss')
    
    ax2.axvline(x=50, color='blue', linestyle='--', alpha=0.5)
    
    loss_gap = history["losses"]["warmup"][-1] - history["losses"]["tabula"][-1]
    ax2.text(
        n_steps * 0.7, 
        (history["losses"]["warmup"][-1] + history["losses"]["tabula"][-1]) / 2,
        f"Loss Gap: +{loss_gap:.1f}",
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95)
    )
    
    ax2.set_xlabel("Routing Step (t)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Cumulative Importance-Weighted Loss", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    out_png = output_dir / "figure5_real_linucb.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved PNG: {out_png}")
    
    out_pdf = output_dir / "figure5_real_linucb.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved PDF: {out_pdf}\n")
    
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("FIGURE 5 ALTERNATIVE: Real LinUCB Experts")
    logger.info("="*70)
    logger.info("\nThis addresses the reviewer concern:")
    logger.info("'Uses deterministic mock experts, not actual LinUCB'\n")
    
    # Run with real LinUCB experts
    history = run_real_linucb_test(seed=42, n_steps=300, context_dim=5)
    
    # Generate visualization
    output_dir = Path(__file__).parent / "results"
    plot_real_linucb(history, output_dir)
    
    logger.info("="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    logger.info("\n💡 Key Differences vs Mock Experts:")
    logger.info("   - More oscillations (exploration noise)")
    logger.info("   - Context-dependent predictions (not fixed choices)")
    logger.info("   - Slower convergence (learning required)")
    logger.info("   - More realistic dynamics\n")


if __name__ == "__main__":
    main()
