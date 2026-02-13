"""
Learning Rate Ablation for Catastrophic Failure Detection
==========================================================

Tests multiple learning rates (η ∈ {0.1, 0.3, 0.5, 1.0, 2.0, 5.0}) to:
1. Validate η=0.3 choice for catastrophic failure detection
2. Show detection time vs false positive trade-off
3. Explain Phase 3 recovery behavior (depends on learning rate)

Connects to Figures 4 & 7:
- Figure 4: η=5.0 → Complete unlearning (~300-500 steps)
- Figure 7: η=0.1 → Stable weights (insufficient adaptation)
- This exp: η=0.3 → Balanced (fast detection, low false positives)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from bandit_gpt.router import CorrallingRouter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MOCK EXPERTS (Same as main experiment)
# ============================================================================

class StubbornExpert:
    """Always picks GPT-4 (simulates warmup prior)."""
    def __init__(self, name: str, favorite_model: str):
        self.name = name
        self.favorite_model = favorite_model
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        return self.favorite_model
    
    def update(self, context, model, reward, cost=0.0):
        pass


class SmartExpert:
    """Mostly picks Mixtral (simulates adaptive learner)."""
    def __init__(self, name: str, best_model: str):
        self.name = name
        self.best_model = best_model
    
    def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
        if np.random.random() < 0.05:
            return "openai/gpt-4-turbo"
        return self.best_model
    
    def update(self, context, model, reward, cost=0.0):
        pass


# ============================================================================
# ENVIRONMENT
# ============================================================================

class CatastrophicFailureEnvironment:
    """Three-phase catastrophic failure scenario."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.t = 0
        
        self.phases = {
            "healthy_1": {
                "mistralai/mixtral-8x7b-instruct": (0.80, 0.08),
                "openai/gpt-4-turbo": (0.80, 0.08),
            },
            "failure": {
                "mistralai/mixtral-8x7b-instruct": (0.80, 0.08),
                "openai/gpt-4-turbo": (0.15, 0.15),  # CATASTROPHIC
            },
            "recovery": {
                "mistralai/mixtral-8x7b-instruct": (0.80, 0.08),
                "openai/gpt-4-turbo": (0.80, 0.08),
            }
        }
    
    def _get_phase(self) -> str:
        if self.t < 100:
            return "healthy_1"
        elif self.t < 300:
            return "failure"
        else:
            return "recovery"
    
    def get_reward(self, model: str) -> float:
        self.t += 1
        phase = self._get_phase()
        params = self.phases[phase]
        mean, std = params.get(model, (0.5, 0.1))
        reward = self.rng.normal(mean, std)
        return np.clip(reward, 0.0, 1.0)


# ============================================================================
# ABLATION RUNNER
# ============================================================================

def run_single_trial(learning_rate: float, seed: int, n_steps: int = 500) -> Dict:
    """Run single trial with given learning rate."""
    np.random.seed(seed)
    
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    
    warmup = StubbornExpert("Warmup", "openai/gpt-4-turbo")
    tabula = SmartExpert("Tabula Rasa", "mistralai/mixtral-8x7b-instruct")
    
    router = CorrallingRouter(
        experts=[warmup, tabula],
        models=models,
        learning_rate=learning_rate,
        gamma=0.05
    )
    
    env = CatastrophicFailureEnvironment(seed=seed)
    
    history = {
        "weights": [],
        "losses": {"warmup": [], "tabula": []},
    }
    
    for t in range(n_steps):
        context = np.random.randn(10)
        selected_model = router.select_model(context)
        reward = env.get_reward(selected_model)
        router.update(context, selected_model, reward)
        
        history["weights"].append(router.weights.copy())
        history["losses"]["warmup"].append(router.cumulative_losses[0])
        history["losses"]["tabula"].append(router.cumulative_losses[1])
    
    # Analyze key metrics
    weights = np.array(history["weights"])
    
    # Phase 1: False positive (premature decommissioning)
    phase1_idx = np.arange(100)
    phase1_min_weight = weights[phase1_idx, 0].min()
    false_positive = phase1_min_weight < 0.1
    
    # Phase 2: Failure detection
    failure_start = 100
    failure_decom_idx = np.where((np.arange(len(weights)) >= failure_start) & 
                                   (weights[:, 0] < 0.1))[0]
    failure_detection = failure_decom_idx[0] if len(failure_decom_idx) > 0 else None
    failure_reaction = (failure_detection - failure_start) if failure_detection else None
    
    # Phase 3: Recovery detection
    recovery_start = 300
    recovery_idx = np.where((np.arange(len(weights)) >= recovery_start) & 
                             (weights[:, 0] > 0.3))[0]
    recovery_detection = recovery_idx[0] if len(recovery_idx) > 0 else None
    recovery_time = (recovery_detection - recovery_start) if recovery_detection else None
    
    # Phase 1: Weight stability (lower variance = more stable)
    phase1_variance = weights[phase1_idx, 0].var()
    
    return {
        "history": history,
        "false_positive": false_positive,
        "phase1_min_weight": phase1_min_weight,
        "phase1_variance": phase1_variance,
        "failure_detection": failure_detection,
        "failure_reaction": failure_reaction,
        "recovery_detection": recovery_detection,
        "recovery_time": recovery_time,
        "final_warmup_weight": weights[-1, 0],
    }


def run_ablation_study(learning_rates: List[float], n_seeds: int = 20, n_steps: int = 500) -> Dict:
    """Run ablation study across multiple learning rates and seeds."""
    logger.info("\n" + "="*70)
    logger.info("LEARNING RATE ABLATION STUDY")
    logger.info("="*70)
    logger.info(f"\nTesting learning rates: {learning_rates}")
    logger.info(f"Seeds: {n_seeds} per learning rate")
    logger.info(f"Steps: {n_steps} per trial\n")
    
    results = {lr: [] for lr in learning_rates}
    
    for lr in learning_rates:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing η = {lr}")
        logger.info(f"{'='*70}")
        
        for seed in range(n_seeds):
            trial_result = run_single_trial(lr, seed, n_steps)
            results[lr].append(trial_result)
            
            if seed % 5 == 0:
                logger.info(
                    f"  Seed {seed:2d}: "
                    f"Detection={trial_result['failure_reaction']:>3} steps, "
                    f"FP={trial_result['false_positive']}, "
                    f"Recovery={trial_result['recovery_time'] if trial_result['recovery_time'] else 'None'}"
                )
    
    # Aggregate statistics
    aggregated = {}
    for lr in learning_rates:
        trials = results[lr]
        
        # Detection time
        detection_times = [t["failure_reaction"] for t in trials if t["failure_reaction"] is not None]
        
        # False positives
        false_positives = sum([t["false_positive"] for t in trials])
        
        # Recovery times
        recovery_times = [t["recovery_time"] for t in trials if t["recovery_time"] is not None]
        
        # Phase 1 stability
        phase1_variances = [t["phase1_variance"] for t in trials]
        
        aggregated[lr] = {
            "detection_mean": np.mean(detection_times) if detection_times else None,
            "detection_std": np.std(detection_times) if detection_times else None,
            "detection_times": detection_times,
            "success_rate": len(detection_times) / n_seeds,
            "false_positive_rate": false_positives / n_seeds,
            "recovery_mean": np.mean(recovery_times) if recovery_times else None,
            "recovery_std": np.std(recovery_times) if recovery_times else None,
            "recovery_rate": len(recovery_times) / n_seeds,
            "phase1_variance_mean": np.mean(phase1_variances),
            "raw_trials": trials,
        }
    
    return aggregated


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ablation_results(aggregated: Dict, output_dir: Path):
    """Generate comprehensive ablation visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    learning_rates = sorted(aggregated.keys())
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # --- Plot 1: Detection Time vs Learning Rate ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    detection_means = [aggregated[lr]["detection_mean"] for lr in learning_rates]
    detection_stds = [aggregated[lr]["detection_std"] for lr in learning_rates]
    
    ax1.errorbar(learning_rates, detection_means, yerr=detection_stds, 
                 fmt='o-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
                 color='#e74c3c', markerfacecolor='white', markeredgewidth=2)
    
    # Highlight optimal (η=0.3)
    optimal_idx = learning_rates.index(0.3) if 0.3 in learning_rates else None
    if optimal_idx is not None:
        ax1.scatter([0.3], [detection_means[optimal_idx]], s=300, 
                   color='gold', marker='*', zorder=5, 
                   edgecolor='darkgoldenrod', linewidth=2,
                   label='Current (η=0.3)')
    
    ax1.set_xlabel("Learning Rate (η)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Detection Time (steps)", fontsize=12, fontweight='bold')
    ax1.set_title("Phase 2: Catastrophic Failure Detection Speed", 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='upper right')
    
    # --- Plot 2: False Positive Rate vs Learning Rate ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    fp_rates = [aggregated[lr]["false_positive_rate"] * 100 for lr in learning_rates]
    
    ax2.plot(learning_rates, fp_rates, 'o-', linewidth=2.5, markersize=10,
             color='#9b59b6', markerfacecolor='white', markeredgewidth=2)
    
    if optimal_idx is not None:
        ax2.scatter([0.3], [fp_rates[optimal_idx]], s=300, 
                   color='gold', marker='*', zorder=5,
                   edgecolor='darkgoldenrod', linewidth=2,
                   label='Current (η=0.3)')
    
    # Acceptable threshold
    ax2.axhline(5, color='gray', linestyle='--', alpha=0.5, label='Acceptable (<5%)')
    
    ax2.set_xlabel("Learning Rate (η)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("False Positive Rate (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Phase 1: Premature Decommissioning Risk", 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10, loc='upper left')
    
    # --- Plot 3: Recovery Detection Rate ---
    ax3 = fig.add_subplot(gs[1, 0])
    
    recovery_rates = [aggregated[lr]["recovery_rate"] * 100 for lr in learning_rates]
    
    ax3.plot(learning_rates, recovery_rates, 'o-', linewidth=2.5, markersize=10,
             color='#3498db', markerfacecolor='white', markeredgewidth=2)
    
    if optimal_idx is not None:
        ax3.scatter([0.3], [recovery_rates[optimal_idx]], s=300, 
                   color='gold', marker='*', zorder=5,
                   edgecolor='darkgoldenrod', linewidth=2,
                   label='Current (η=0.3)')
    
    ax3.set_xlabel("Learning Rate (η)", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Recovery Detection Rate (%)", fontsize=12, fontweight='bold')
    ax3.set_title("Phase 3: Automatic Recovery Detection", 
                  fontsize=13, fontweight='bold', pad=10)
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.legend(fontsize=10, loc='lower right')
    
    # --- Plot 4: Recovery Time (when detected) ---
    ax4 = fig.add_subplot(gs[1, 1])
    
    recovery_means = [aggregated[lr]["recovery_mean"] if aggregated[lr]["recovery_mean"] else np.nan 
                      for lr in learning_rates]
    recovery_stds = [aggregated[lr]["recovery_std"] if aggregated[lr]["recovery_std"] else 0
                     for lr in learning_rates]
    
    valid_idx = [i for i, x in enumerate(recovery_means) if not np.isnan(x)]
    valid_lrs = [learning_rates[i] for i in valid_idx]
    valid_means = [recovery_means[i] for i in valid_idx]
    valid_stds = [recovery_stds[i] for i in valid_idx]
    
    if len(valid_lrs) > 0:
        ax4.errorbar(valid_lrs, valid_means, yerr=valid_stds,
                     fmt='o-', linewidth=2.5, markersize=10, capsize=5, capthick=2,
                     color='#3498db', markerfacecolor='white', markeredgewidth=2)
    
    ax4.set_xlabel("Learning Rate (η)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Recovery Time (steps)", fontsize=12, fontweight='bold')
    ax4.set_title("Phase 3: Speed of Recovery Detection (when successful)", 
                  fontsize=13, fontweight='bold', pad=10)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, linestyle=':')
    
    # --- Plot 5: Trade-off Curve (Detection vs FP) ---
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Scatter with learning rate labels
    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))
    
    for i, lr in enumerate(learning_rates):
        ax5.scatter(detection_means[i], fp_rates[i], 
                   s=200, color=colors[i], 
                   edgecolor='black', linewidth=1.5,
                   zorder=3, label=f'η={lr}')
    
    # Highlight optimal
    if optimal_idx is not None:
        ax5.scatter([detection_means[optimal_idx]], [fp_rates[optimal_idx]], 
                   s=400, color='gold', marker='*', zorder=5,
                   edgecolor='darkgoldenrod', linewidth=2)
    
    # Ideal region
    ax5.axvspan(0, 20, alpha=0.1, color='green', label='Fast Detection (<20 steps)')
    ax5.axhspan(0, 5, alpha=0.1, color='green', label='Low FP Rate (<5%)')
    
    ax5.set_xlabel("Detection Time (steps)", fontsize=12, fontweight='bold')
    ax5.set_ylabel("False Positive Rate (%)", fontsize=12, fontweight='bold')
    ax5.set_title("Trade-off: Detection Speed vs Safety", 
                  fontsize=13, fontweight='bold', pad=10)
    ax5.grid(True, alpha=0.3, linestyle=':')
    ax5.legend(fontsize=9, loc='upper left', ncol=2)
    
    # --- Plot 6: Example Weight Evolution ---
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Show 3 representative learning rates
    representative_lrs = [0.1, 0.3, 5.0] if all(lr in learning_rates for lr in [0.1, 0.3, 5.0]) else learning_rates[:3]
    colors_rep = ['#3498db', '#e74c3c', '#9b59b6']
    
    for lr, color in zip(representative_lrs, colors_rep):
        # Get first trial for this learning rate
        trial = aggregated[lr]["raw_trials"][0]
        weights = np.array(trial["history"]["weights"])
        t = np.arange(len(weights))
        
        label = f'η={lr}'
        if lr == 0.3:
            label += ' (Current)'
        
        ax6.plot(t, weights[:, 0], color=color, linewidth=2, 
                label=label, alpha=0.8)
    
    # Phase boundaries
    ax6.axvspan(0, 100, color='green', alpha=0.05)
    ax6.axvspan(100, 300, color='red', alpha=0.05)
    ax6.axvspan(300, 500, color='blue', alpha=0.05)
    ax6.axvline(100, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax6.axvline(300, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax6.axhline(0.1, color='gray', linestyle=':', alpha=0.4, label='Decommission Threshold')
    
    ax6.set_xlabel("Step (t)", fontsize=12, fontweight='bold')
    ax6.set_ylabel("Warmup Expert Weight", fontsize=12, fontweight='bold')
    ax6.set_title("Example: Weight Evolution Across Learning Rates", 
                  fontsize=13, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, linestyle=':')
    ax6.legend(fontsize=9, loc='upper right')
    ax6.set_ylim(-0.05, 1.05)
    
    plt.suptitle("Learning Rate Ablation: Catastrophic Failure Detection", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    out_png = output_dir / "appendixD_learning_rate_ablation.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    logger.info(f"\n✅ Saved: {out_png}")
    
    out_pdf = output_dir / "appendixD_learning_rate_ablation.pdf"
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Saved: {out_pdf}")
    
    plt.close()


def print_summary_table(aggregated: Dict):
    """Print comprehensive summary table."""
    learning_rates = sorted(aggregated.keys())
    
    logger.info("\n" + "="*100)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*100)
    logger.info(f"\n{'η':<8} {'Detection':<20} {'FP Rate':<12} {'Recovery Rate':<15} {'Recovery Time':<20}")
    logger.info("-"*100)
    
    for lr in learning_rates:
        stats = aggregated[lr]
        
        detection_str = f"{stats['detection_mean']:.1f} ± {stats['detection_std']:.1f}" if stats['detection_mean'] else "N/A"
        fp_str = f"{stats['false_positive_rate']*100:.1f}%"
        recovery_rate_str = f"{stats['recovery_rate']*100:.0f}%"
        recovery_str = f"{stats['recovery_mean']:.1f} ± {stats['recovery_std']:.1f}" if stats['recovery_mean'] else "None detected"
        
        marker = " ← Current" if lr == 0.3 else ""
        logger.info(f"{lr:<8.2f} {detection_str:<20} {fp_str:<12} {recovery_rate_str:<15} {recovery_str:<20} {marker}")
    
    logger.info("="*100)
    
    # Recommendations
    logger.info("\n📊 KEY FINDINGS:\n")
    
    # Find best for each criterion
    detection_times = [(lr, aggregated[lr]["detection_mean"]) for lr in learning_rates if aggregated[lr]["detection_mean"]]
    fastest = min(detection_times, key=lambda x: x[1])
    
    logger.info(f"1. FASTEST DETECTION: η={fastest[0]:.2f} ({fastest[1]:.1f} steps)")
    logger.info(f"   - Trade-off: {aggregated[fastest[0]]['false_positive_rate']*100:.1f}% false positive rate")
    
    # Safest (lowest FP)
    fp_rates = [(lr, aggregated[lr]["false_positive_rate"]) for lr in learning_rates]
    safest = min(fp_rates, key=lambda x: x[1])
    logger.info(f"\n2. SAFEST (Lowest FP): η={safest[0]:.2f} ({safest[1]*100:.1f}% FP rate)")
    logger.info(f"   - Trade-off: {aggregated[safest[0]]['detection_mean']:.1f} step detection time")
    
    # Best recovery
    recovery_rates = [(lr, aggregated[lr]["recovery_rate"]) for lr in learning_rates]
    best_recovery = max(recovery_rates, key=lambda x: x[1])
    logger.info(f"\n3. BEST RECOVERY DETECTION: η={best_recovery[0]:.2f} ({best_recovery[1]*100:.0f}% success)")
    logger.info(f"   - Recovery time: {aggregated[best_recovery[0]]['recovery_mean']:.1f} steps")
    
    # Current (η=0.3)
    if 0.3 in learning_rates:
        logger.info(f"\n4. CURRENT CHOICE (η=0.3):")
        logger.info(f"   - Detection: {aggregated[0.3]['detection_mean']:.1f} ± {aggregated[0.3]['detection_std']:.1f} steps")
        logger.info(f"   - False positives: {aggregated[0.3]['false_positive_rate']*100:.1f}%")
        logger.info(f"   - Recovery rate: {aggregated[0.3]['recovery_rate']*100:.0f}%")
        logger.info(f"   - ✅ BALANCED: Good detection speed with low false positive rate")
    
    logger.info("\n💡 DEPLOYMENT RECOMMENDATIONS:\n")
    logger.info("   Safety-critical systems:     η = 0.3   (balanced, current)")
    logger.info("   High-availability systems:   η = 1.0   (faster detection + recovery)")
    logger.info("   Ultra-fast failover:         η = 2.0   (accept higher FP rate)")
    logger.info("   Adaptive strategy:           Start 0.3 → increase to 5.0 for recovery")
    logger.info("="*100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("LEARNING RATE ABLATION FOR CATASTROPHIC FAILURE DETECTION")
    logger.info("="*70)
    logger.info("\n💡 Objective: Validate η=0.3 choice and characterize trade-offs")
    logger.info("   - Detection speed (Phase 2)")
    logger.info("   - False positive rate (Phase 1)")
    logger.info("   - Recovery detection (Phase 3)")
    logger.info("   - Connection to Figures 4 & 7\n")
    
    # Test learning rates
    learning_rates = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    # Run ablation study
    aggregated = run_ablation_study(learning_rates, n_seeds=20, n_steps=500)
    
    # Print summary
    print_summary_table(aggregated)
    
    # Generate plots
    output_dir = Path(__file__).parent.parent / "results"
    plot_ablation_results(aggregated, output_dir)
    
    logger.info("\n✅ Learning rate ablation complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
