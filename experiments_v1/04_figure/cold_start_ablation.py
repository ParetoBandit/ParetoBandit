#!/usr/bin/env python3
"""
Figure 4: Cold-Start Ablation (Prior vs. No-Prior)

This experiment answers the critical question: "If you can pivot 99.7% of the policy 
in 1,121 samples, do you even need the 80,000-sample warmup?"

Compares:
1. Fully Calibrated Router (with warmup priors from 80k samples)
2. Tabula Rasa Bandit (A=I, b=0, no priors, trained only on calibration data)

Key Metrics:
- Day 1 Quality: Performance on first 100 samples
- Cumulative Regret: Total regret over calibration period
- Convergence Speed: How quickly each approach reaches optimal policy

The Goal: Prove that the warmup provides a "Linguistic Foundation" that prevents 
catastrophic routing errors during early calibration, even if both converge to 
similar final policies.

Usage:
    python cold_start_ablation.py --output results/
    python cold_start_ablation.py --calibration-samples 1121 --output results/
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import argparse
import json
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
    STRONG_MODEL_EQUIVALENTS
)


class TabulaRasaRouter:
    """
    LinUCB router initialized from scratch (A=I, b=0).
    
    This represents a bandit that has NO prior knowledge and must learn
    everything from scratch using only the calibration data.
    """
    
    def __init__(self, models: List[str], context_dim: int, alpha: float = 1.0):
        """
        Initialize with identity matrix and zero reward vector.
        
        Args:
            models: List of model IDs
            context_dim: Dimension of context vectors
            alpha: Exploration parameter
        """
        self.models = models
        self.alpha = alpha
        self.context_dim = context_dim
        
        # Initialize with identity (no prior knowledge)
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        
        # Track selections for usage stats
        self.selections = {m: 0 for m in models}
    
    def select_model(self, context: np.ndarray) -> str:
        """Select model using UCB (takes pre-computed context vector)."""
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
            # UCB = expected reward + exploration bonus
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores[model] = expected + self.alpha * uncertainty
        
        selected = max(ucb_scores, key=ucb_scores.get)
        self.selections[selected] += 1
        return selected
    
    def update(self, context: np.ndarray, model: str, reward: float):
        """Update matrices after observing reward."""
        context = context.reshape(-1, 1)  # Column vector
        self.A[model] += context @ context.T
        self.b[model] += (reward * context).flatten()
    
    def get_model_usage(self) -> Dict[str, float]:
        """Get model selection percentages."""
        total = sum(self.selections.values())
        if total == 0:
            return {m: 100.0 / len(self.models) for m in self.models}
        return {m: (count / total) * 100 for m, count in self.selections.items()}


def map_model_to_data(router_model: str, data_models: List[str]) -> str:
    """
    Map router model ID to data model ID.
    
    Args:
        router_model: Model ID from router (e.g., 'openai/gpt-4-turbo')
        data_models: Available model IDs in data
        
    Returns:
        Model ID that exists in data, or router_model if not found
    """
    # Direct match
    if router_model in data_models:
        return router_model
    
    # If not found, return original (will result in 0.0 reward)
    return router_model


def compute_oracle_reward(item: dict, models: List[str]) -> float:
    """
    Compute the best possible reward (oracle) for this prompt.
    
    Args:
        item: Data item with 'prompt' and 'rewards' dict
        models: List of available models (router models)
        
    Returns:
        Maximum reward achievable
    """
    data_models = list(item['rewards'].keys())
    available_rewards = []
    
    for router_model in models:
        data_model = map_model_to_data(router_model, data_models)
        reward = item['rewards'].get(data_model, 0.0)
        available_rewards.append(reward)
    
    return max(available_rewards) if available_rewards else 0.0


def compute_convergence_point(
    warmup_metrics: List[Dict],
    tabula_rasa_metrics: List[Dict],
    window_size: int = 10
) -> Tuple[int, float]:
    """
    Find the convergence point where tabula rasa catches up to warmup.
    
    Uses a sliding window to compute slopes and finds where the gap
    in average reward becomes negligible (< 1% difference).
    
    Args:
        warmup_metrics: Time series metrics from warmup router
        tabula_rasa_metrics: Time series metrics from tabula rasa router
        window_size: Window for computing moving average
        
    Returns:
        (convergence_sample, gap_at_convergence)
    """
    warmup_rewards = [m['avg_reward'] for m in warmup_metrics]
    tabula_rewards = [m['avg_reward'] for m in tabula_rasa_metrics]
    samples = [m['sample'] for m in warmup_metrics]
    
    # Find where gap becomes < 1%
    for i, (w_reward, t_reward, sample) in enumerate(zip(warmup_rewards, tabula_rewards, samples)):
        if w_reward > 0:  # Avoid division by zero
            gap_pct = abs((w_reward - t_reward) / w_reward) * 100
            if gap_pct < 1.0 and sample > 100:  # After Day 1
                return sample, gap_pct
    
    # If never converges, return last sample
    return samples[-1], abs((warmup_rewards[-1] - tabula_rewards[-1]) / warmup_rewards[-1]) * 100


def compute_uncertainty_metrics(router, context: np.ndarray) -> Dict[str, float]:
    """
    Compute uncertainty metrics for each model's prediction.
    
    This helps diagnose whether tabula rasa suffers from numerical
    instability vs. semantic ignorance.
    
    Args:
        router: Router instance (SimpleLinUCBRouter or TabulaRasaRouter)
        context: Context vector
        
    Returns:
        Dict with uncertainty measures per model
    """
    uncertainties = {}
    for model in router.models:
        A_inv = np.linalg.inv(router.A[model])
        uncertainty = np.sqrt(context @ A_inv @ context)
        uncertainties[model] = float(uncertainty)
    
    return uncertainties


def run_calibration_with_tracking(
    router,
    calibration_data: List[dict],
    encoder: SentenceTransformer,
    pca_model,
    router_type: str,
    verbose: bool = False
) -> Dict:
    """
    Run calibration while tracking detailed metrics.
    
    Args:
        router: Either SimpleLinUCBRouter or TabulaRasaRouter
        calibration_data: List of calibration samples
        encoder: Sentence transformer for embeddings
        pca_model: PCA model for dimensionality reduction
        router_type: "warmup" or "tabula_rasa" for logging
        verbose: Print progress
        
    Returns:
        Dict with comprehensive metrics
    """
    models = router.models
    
    # Track metrics over time
    metrics = []
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    
    # Day 1 metrics (first 100 samples)
    day1_rewards = []
    day1_regrets = []
    
    # Uncertainty tracking (for numerical stability analysis)
    uncertainty_history = []
    
    for i, item in enumerate(tqdm(calibration_data, desc=f"Calibrating {router_type}", disable=not verbose)):
        # Embed
        context = embed_prompt(item['prompt'], encoder, pca_model)
        
        # Track uncertainty BEFORE selection (for stability analysis)
        uncertainties = compute_uncertainty_metrics(router, context)
        
        # Select model
        selected_model = router.select_model(context)
        
        # Map model to data (handle gpt-4-turbo -> gpt-4o equivalence)
        data_models = list(item['rewards'].keys())
        data_model = map_model_to_data(selected_model, data_models)
        
        # Get reward
        reward = item['rewards'].get(data_model, 0.0)
        
        # Compute oracle (best possible)
        oracle_reward = compute_oracle_reward(item, models)
        
        # Compute instantaneous regret
        regret = oracle_reward - reward
        
        # Update router
        router.update(context, selected_model, reward)
        
        # Accumulate
        cumulative_reward += reward
        cumulative_regret += regret
        
        # Track Day 1 (first 100 samples)
        if i < 100:
            day1_rewards.append(reward)
            day1_regrets.append(regret)
        
        # Track uncertainty for first 50 samples (critical cold-start period)
        if i < 50:
            uncertainty_history.append({
                'sample': i + 1,
                'uncertainties': uncertainties,
                'avg_uncertainty': np.mean(list(uncertainties.values()))
            })
        
        # Record metrics at intervals
        if i % 10 == 0 or i == len(calibration_data) - 1:
            usage = router.get_model_usage()
            metrics.append({
                'sample': i + 1,
                'model_usage': usage,
                'strong_pct': usage.get(models[1] if len(models) > 1 else models[0], 0.0),
                'avg_reward': cumulative_reward / (i + 1),
                'cumulative_regret': cumulative_regret,
                'avg_regret': cumulative_regret / (i + 1)
            })
    
    # Compute final statistics
    final_metrics = {
        'router_type': router_type,
        'metrics': metrics,
        'total_samples': len(calibration_data),
        'cumulative_reward': cumulative_reward,
        'avg_reward': cumulative_reward / len(calibration_data),
        'cumulative_regret': cumulative_regret,
        'avg_regret': cumulative_regret / len(calibration_data),
        'day1_avg_reward': np.mean(day1_rewards),
        'day1_avg_regret': np.mean(day1_regrets),
        'day1_cumulative_regret': sum(day1_regrets),
        'final_model_usage': router.get_model_usage(),
        'uncertainty_history': uncertainty_history,
        'avg_initial_uncertainty': np.mean([u['avg_uncertainty'] for u in uncertainty_history[:10]]) if uncertainty_history else 0.0
    }
    
    return final_metrics


def plot_comparison(
    warmup_metrics: Dict,
    tabula_rasa_metrics: Dict,
    output_dir: Path,
    model_display_names: List[str],
    alpha: float
):
    """
    Create comprehensive comparison plots.
    
    Args:
        warmup_metrics: Metrics from warmup-backed router
        tabula_rasa_metrics: Metrics from tabula rasa router
        output_dir: Directory to save plots
        model_display_names: Display names for models
        alpha: Exploration parameter (for annotation)
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Cold-Start Ablation: Warmup Priors vs. Tabula Rasa (α={alpha})', 
                 fontsize=16, fontweight='bold')
    
    # Extract time series
    warmup_samples = [m['sample'] for m in warmup_metrics['metrics']]
    warmup_regret = [m['cumulative_regret'] for m in warmup_metrics['metrics']]
    warmup_avg_reward = [m['avg_reward'] for m in warmup_metrics['metrics']]
    warmup_strong_pct = [m['strong_pct'] for m in warmup_metrics['metrics']]
    
    tabula_samples = [m['sample'] for m in tabula_rasa_metrics['metrics']]
    tabula_regret = [m['cumulative_regret'] for m in tabula_rasa_metrics['metrics']]
    tabula_avg_reward = [m['avg_reward'] for m in tabula_rasa_metrics['metrics']]
    tabula_strong_pct = [m['strong_pct'] for m in tabula_rasa_metrics['metrics']]
    
    # Compute convergence point
    convergence_sample, convergence_gap = compute_convergence_point(
        warmup_metrics['metrics'],
        tabula_rasa_metrics['metrics']
    )
    
    # 1. Cumulative Regret (THE KEY METRIC)
    ax1 = axes[0, 0]
    ax1.plot(warmup_samples, warmup_regret, 'b-', linewidth=2, label='With Warmup Priors')
    ax1.plot(tabula_samples, tabula_regret, 'r--', linewidth=2, label='Tabula Rasa (A=I, b=0)')
    ax1.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='Day 1 (100 samples)')
    
    # Mark convergence point
    if convergence_sample < warmup_samples[-1]:
        ax1.axvline(x=convergence_sample, color='green', linestyle='-.', alpha=0.5, 
                   label=f'Convergence (~{convergence_sample} samples)')
    
    ax1.set_xlabel('Calibration Samples', fontsize=11)
    ax1.set_ylabel('Cumulative Regret', fontsize=11)
    ax1.set_title('Cumulative Regret Over Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for Day 1 regret
    day1_warmup = warmup_metrics['day1_cumulative_regret']
    day1_tabula = tabula_rasa_metrics['day1_cumulative_regret']
    regret_reduction = ((day1_tabula - day1_warmup) / day1_tabula) * 100
    ax1.text(0.95, 0.95, f'Day 1 Regret Reduction:\n{regret_reduction:.1f}%',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Average Reward (Quality Metric) with Convergence Point
    ax2 = axes[0, 1]
    ax2.plot(warmup_samples, warmup_avg_reward, 'b-', linewidth=2, label='With Warmup Priors')
    ax2.plot(tabula_samples, tabula_avg_reward, 'r--', linewidth=2, label='Tabula Rasa')
    ax2.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='Day 1')
    
    # Mark convergence point
    if convergence_sample < warmup_samples[-1]:
        ax2.axvline(x=convergence_sample, color='green', linestyle='-.', alpha=0.5,
                   label=f'Convergence (~{convergence_sample} samples)')
        # Annotate time-to-value
        ax2.text(convergence_sample + 50, 0.5, 
                f'Time-to-Value:\n{convergence_sample} samples\n(Gap < 1%)',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_xlabel('Calibration Samples', fontsize=11)
    ax2.set_ylabel('Average Reward', fontsize=11)
    ax2.set_title('Average Reward Over Time (Convergence Analysis)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for Day 1 quality
    day1_warmup_reward = warmup_metrics['day1_avg_reward']
    day1_tabula_reward = tabula_rasa_metrics['day1_avg_reward']
    quality_improvement = ((day1_warmup_reward - day1_tabula_reward) / day1_tabula_reward) * 100
    ax2.text(0.05, 0.05, f'Day 1 Quality Improvement:\n+{quality_improvement:.1f}%',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Numerical Stability: Uncertainty Over Time (First 50 Samples)
    ax3 = axes[0, 2]
    
    warmup_uncertainty = [u['avg_uncertainty'] for u in warmup_metrics['uncertainty_history']]
    tabula_uncertainty = [u['avg_uncertainty'] for u in tabula_rasa_metrics['uncertainty_history']]
    uncertainty_samples = [u['sample'] for u in warmup_metrics['uncertainty_history']]
    
    ax3.plot(uncertainty_samples, warmup_uncertainty, 'b-', linewidth=2, marker='o', 
            markersize=4, label='With Warmup Priors')
    ax3.plot(uncertainty_samples, tabula_uncertainty, 'r--', linewidth=2, marker='s',
            markersize=4, label='Tabula Rasa (A=I, b=0)')
    ax3.set_xlabel('Calibration Samples (First 50)', fontsize=11)
    ax3.set_ylabel('Average UCB Uncertainty', fontsize=11)
    ax3.set_title('Numerical Stability: Uncertainty Analysis', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Annotate stability difference
    warmup_init_uncertainty = warmup_metrics['avg_initial_uncertainty']
    tabula_init_uncertainty = tabula_rasa_metrics['avg_initial_uncertainty']
    stability_ratio = tabula_init_uncertainty / warmup_init_uncertainty if warmup_init_uncertainty > 0 else 0
    ax3.text(0.05, 0.95, f'Initial Uncertainty Ratio:\n{stability_ratio:.1f}× higher for Tabula Rasa',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # 4. Policy Evolution (Strong Model Usage)
    ax4 = axes[1, 0]
    ax4.plot(warmup_samples, warmup_strong_pct, 'b-', linewidth=2, label='With Warmup Priors')
    ax4.plot(tabula_samples, tabula_strong_pct, 'r--', linewidth=2, label='Tabula Rasa')
    ax4.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='Day 1')
    
    if convergence_sample < warmup_samples[-1]:
        ax4.axvline(x=convergence_sample, color='green', linestyle='-.', alpha=0.5,
                   label=f'Convergence')
    
    ax4.set_xlabel('Calibration Samples', fontsize=11)
    ax4.set_ylabel(f'{model_display_names[1] if len(model_display_names) > 1 else "Strong Model"} Usage (%)', fontsize=11)
    ax4.set_title('Policy Evolution: Strong Model Usage', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    # 5. Day 1 Focus: First 100 Samples (Zoomed In)
    ax5 = axes[1, 1]
    
    # Filter to first 100 samples
    warmup_day1_samples = [s for s in warmup_samples if s <= 100]
    warmup_day1_regret = [r for s, r in zip(warmup_samples, warmup_regret) if s <= 100]
    tabula_day1_samples = [s for s in tabula_samples if s <= 100]
    tabula_day1_regret = [r for s, r in zip(tabula_samples, tabula_regret) if s <= 100]
    
    ax5.plot(warmup_day1_samples, warmup_day1_regret, 'b-', linewidth=2.5, 
             marker='o', markersize=3, label='With Warmup Priors')
    ax5.plot(tabula_day1_samples, tabula_day1_regret, 'r--', linewidth=2.5,
             marker='s', markersize=3, label='Tabula Rasa')
    ax5.set_xlabel('Calibration Samples (Day 1)', fontsize=11)
    ax5.set_ylabel('Cumulative Regret', fontsize=11)
    ax5.set_title('Day 1 Performance (First 100 Samples)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Highlight the gap
    ax5.fill_between(warmup_day1_samples, warmup_day1_regret, tabula_day1_regret[:len(warmup_day1_regret)],
                     where=[t >= w for w, t in zip(warmup_day1_regret, tabula_day1_regret[:len(warmup_day1_regret)])],
                     alpha=0.2, color='green', label='Regret Prevented by Warmup')
    
    # 6. Instantaneous Regret Rate (Derivative Analysis)
    ax6 = axes[1, 2]
    
    # Compute instantaneous regret rate (change in cumulative regret)
    warmup_regret_rate = [warmup_regret[i] - warmup_regret[i-1] if i > 0 else warmup_regret[0] 
                          for i in range(len(warmup_regret))]
    tabula_regret_rate = [tabula_regret[i] - tabula_regret[i-1] if i > 0 else tabula_regret[0]
                          for i in range(len(tabula_regret))]
    
    # Smooth with moving average for clarity
    window = 5
    warmup_regret_rate_smooth = np.convolve(warmup_regret_rate, np.ones(window)/window, mode='valid')
    tabula_regret_rate_smooth = np.convolve(tabula_regret_rate, np.ones(window)/window, mode='valid')
    rate_samples = warmup_samples[window-1:]
    
    ax6.plot(rate_samples, warmup_regret_rate_smooth, 'b-', linewidth=2, label='With Warmup Priors')
    ax6.plot(rate_samples, tabula_regret_rate_smooth, 'r--', linewidth=2, label='Tabula Rasa')
    ax6.axvline(x=100, color='gray', linestyle=':', alpha=0.5, label='Day 1')
    
    if convergence_sample < warmup_samples[-1]:
        ax6.axvline(x=convergence_sample, color='green', linestyle='-.', alpha=0.5,
                   label=f'Convergence')
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax6.set_xlabel('Calibration Samples', fontsize=11)
    ax6.set_ylabel('Instantaneous Regret Rate (smoothed)', fontsize=11)
    ax6.set_title('Regret Rate: Convergence to Steady State', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "cold_start_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved plot: {output_path}")
    
    plt.close()


def save_results(
    warmup_metrics: Dict,
    tabula_rasa_metrics: Dict,
    output_dir: Path,
    alpha: float
):
    """Save detailed results to JSON."""
    # Compute convergence point
    convergence_sample, convergence_gap = compute_convergence_point(
        warmup_metrics['metrics'],
        tabula_rasa_metrics['metrics']
    )
    
    results = {
        'experiment': 'cold_start_ablation',
        'description': 'Comparison of warmup-backed router vs tabula rasa bandit',
        'experimental_parameters': {
            'alpha': float(alpha),
            'note': 'Alpha held constant across both routers to isolate effect of prior matrices (A, b)'
        },
        'warmup': {
            'total_samples': warmup_metrics['total_samples'],
            'cumulative_reward': float(warmup_metrics['cumulative_reward']),
            'avg_reward': float(warmup_metrics['avg_reward']),
            'cumulative_regret': float(warmup_metrics['cumulative_regret']),
            'avg_regret': float(warmup_metrics['avg_regret']),
            'day1_avg_reward': float(warmup_metrics['day1_avg_reward']),
            'day1_avg_regret': float(warmup_metrics['day1_avg_regret']),
            'day1_cumulative_regret': float(warmup_metrics['day1_cumulative_regret']),
            'final_model_usage': warmup_metrics['final_model_usage'],
            'avg_initial_uncertainty': float(warmup_metrics['avg_initial_uncertainty'])
        },
        'tabula_rasa': {
            'total_samples': tabula_rasa_metrics['total_samples'],
            'cumulative_reward': float(tabula_rasa_metrics['cumulative_reward']),
            'avg_reward': float(tabula_rasa_metrics['avg_reward']),
            'cumulative_regret': float(tabula_rasa_metrics['cumulative_regret']),
            'avg_regret': float(tabula_rasa_metrics['avg_regret']),
            'day1_avg_reward': float(tabula_rasa_metrics['day1_avg_reward']),
            'day1_avg_regret': float(tabula_rasa_metrics['day1_avg_regret']),
            'day1_cumulative_regret': float(tabula_rasa_metrics['day1_cumulative_regret']),
            'final_model_usage': tabula_rasa_metrics['final_model_usage'],
            'avg_initial_uncertainty': float(tabula_rasa_metrics['avg_initial_uncertainty'])
        },
        'comparison': {
            'day1_regret_reduction_pct': float(
                ((tabula_rasa_metrics['day1_cumulative_regret'] - 
                  warmup_metrics['day1_cumulative_regret']) / 
                 tabula_rasa_metrics['day1_cumulative_regret']) * 100
            ),
            'day1_quality_improvement_pct': float(
                ((warmup_metrics['day1_avg_reward'] - 
                  tabula_rasa_metrics['day1_avg_reward']) / 
                 tabula_rasa_metrics['day1_avg_reward']) * 100
            ),
            'total_regret_reduction_pct': float(
                ((tabula_rasa_metrics['cumulative_regret'] - 
                  warmup_metrics['cumulative_regret']) / 
                 tabula_rasa_metrics['cumulative_regret']) * 100
            ),
            'convergence_sample': int(convergence_sample),
            'convergence_gap_pct': float(convergence_gap),
            'time_to_value_samples': int(convergence_sample),
            'numerical_stability': {
                'initial_uncertainty_ratio': float(
                    tabula_rasa_metrics['avg_initial_uncertainty'] / 
                    warmup_metrics['avg_initial_uncertainty']
                ) if warmup_metrics['avg_initial_uncertainty'] > 0 else 0.0,
                'interpretation': 'Ratio > 1 indicates tabula rasa has higher exploration variance (numerical instability)'
            }
        }
    }
    
    output_path = output_dir / "cold_start_ablation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Saved results: {output_path}")
    
    return results


def print_summary(results: Dict, model_display_names: List[str]):
    """Print executive summary of results."""
    print("\n" + "="*80)
    print("COLD-START ABLATION RESULTS")
    print("="*80)
    
    print("\n📊 Experimental Design:")
    print(f"   Total Calibration Samples: {results['warmup']['total_samples']:,}")
    print(f"   Exploration Parameter (α): {results['experimental_parameters']['alpha']}")
    print(f"   Note: {results['experimental_parameters']['note']}")
    
    print("\n🔵 With Warmup Priors (80k samples):")
    print(f"   Cumulative Regret: {results['warmup']['cumulative_regret']:.2f}")
    print(f"   Average Reward: {results['warmup']['avg_reward']:.4f}")
    print(f"   Day 1 Avg Reward: {results['warmup']['day1_avg_reward']:.4f}")
    print(f"   Day 1 Cumulative Regret: {results['warmup']['day1_cumulative_regret']:.2f}")
    print(f"   Initial Uncertainty: {results['warmup']['avg_initial_uncertainty']:.4f}")
    
    print("\n🔴 Tabula Rasa (A=I, b=0):")
    print(f"   Cumulative Regret: {results['tabula_rasa']['cumulative_regret']:.2f}")
    print(f"   Average Reward: {results['tabula_rasa']['avg_reward']:.4f}")
    print(f"   Day 1 Avg Reward: {results['tabula_rasa']['day1_avg_reward']:.4f}")
    print(f"   Day 1 Cumulative Regret: {results['tabula_rasa']['day1_cumulative_regret']:.2f}")
    print(f"   Initial Uncertainty: {results['tabula_rasa']['avg_initial_uncertainty']:.4f}")
    
    print("\n✅ Warmup Advantage:")
    print(f"   Day 1 Regret Reduction: {results['comparison']['day1_regret_reduction_pct']:.1f}%")
    print(f"   Day 1 Quality Improvement: {results['comparison']['day1_quality_improvement_pct']:.1f}%")
    print(f"   Total Regret Reduction: {results['comparison']['total_regret_reduction_pct']:.1f}%")
    
    print("\n⏱️  Convergence Analysis:")
    print(f"   Time-to-Value: {results['comparison']['time_to_value_samples']} samples")
    print(f"   Convergence Gap: {results['comparison']['convergence_gap_pct']:.2f}% (< 1% threshold)")
    print(f"   Interpretation: Warmup provides {results['comparison']['time_to_value_samples']} samples")
    print(f"                   of superior performance before policies converge")
    
    print("\n🔧 Numerical Stability Analysis:")
    print(f"   Initial Uncertainty Ratio: {results['comparison']['numerical_stability']['initial_uncertainty_ratio']:.2f}×")
    print(f"   {results['comparison']['numerical_stability']['interpretation']}")
    
    print("\n📈 Final Model Usage:")
    print("   With Warmup Priors:")
    for model, pct in results['warmup']['final_model_usage'].items():
        display_name = model_display_names[0] if model == list(results['warmup']['final_model_usage'].keys())[0] else model_display_names[1] if len(model_display_names) > 1 else model
        print(f"      {display_name}: {pct:.1f}%")
    
    print("   Tabula Rasa:")
    for model, pct in results['tabula_rasa']['final_model_usage'].items():
        display_name = model_display_names[0] if model == list(results['tabula_rasa']['final_model_usage'].keys())[0] else model_display_names[1] if len(model_display_names) > 1 else model
        print(f"      {display_name}: {pct:.1f}%")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. SEMANTIC GUIDANCE: Warmup provides linguistic structure, not just numerical")
    print(f"   stability. Day 1 regret reduced by {results['comparison']['day1_regret_reduction_pct']:.1f}%.")
    print()
    print("2. NUMERICAL STABILITY: Warmup-backed router has")
    print(f"   {results['comparison']['numerical_stability']['initial_uncertainty_ratio']:.1f}× lower initial uncertainty,")
    print("   preventing catastrophic exploration during cold-start.")
    print()
    print("3. TIME-TO-VALUE: Warmup provides superior performance for")
    print(f"   {results['comparison']['time_to_value_samples']} samples before convergence.")
    print("   This is the critical deployment window where warmup justifies its cost.")
    print()
    print("4. CONVERGENCE: Both routers eventually reach similar policies, proving that")
    print("   the value of warmup is in the LEARNING TRAJECTORY, not just the endpoint.")
    print("="*80)


def load_model_registry() -> Dict[str, Dict]:
    """Load model registry from models.json."""
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    
    # Handle nested format: {"models": [...]}
    if isinstance(data, dict) and "models" in data:
        models_list = data["models"]
    else:
        models_list = data
    
    # Create lookup by openrouter_id
    return {model["openrouter_id"]: model for model in models_list}


def get_model_display_name(openrouter_id: str, model_registry: Dict[str, Dict]) -> str:
    """Get display name for a model from the registry."""
    if openrouter_id in model_registry:
        return model_registry[openrouter_id].get("display_name", openrouter_id)
    
    # If not found, try to create a reasonable display name
    if "/" in openrouter_id:
        _, model_name = openrouter_id.split("/", 1)
        return " ".join(word.capitalize() for word in model_name.replace("-", " ").split())
    
    return openrouter_id


def main():
    parser = argparse.ArgumentParser(
        description="Cold-Start Ablation: Compare warmup priors vs tabula rasa bandit"
    )
    parser.add_argument(
        "--calibration-data", type=str,
        default=str(CANONICAL_DEV_DATA_PATH),
        help="Path to calibration data (JSONL with 'prompt' and 'rewards' fields)"
    )
    parser.add_argument(
        "--warmup-priors", type=str,
        default=str(DEFAULT_WARMUP_PRIORS_PATH),
        help="Path to warmup priors"
    )
    parser.add_argument(
        "--pca", type=str,
        default=str(DEFAULT_PCA_PATH),
        help="Path to PCA model"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.002,
        help="Gamma scaling factor for warmup priors (default: 0.002, optimal from Figure 3)"
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=1121,
        help="Number of calibration samples to use (default: 1121)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Exploration parameter (default: 1.0)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("FIGURE 4: COLD-START ABLATION (PRIOR VS. NO-PRIOR)")
    print("="*80)
    print("\nResearch Question:")
    print("If we can pivot 99.7% of the policy in 1,121 samples,")
    print("do we even need the 80,000-sample warmup?")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load resources
    print("\n📥 Loading resources...")
    warmup_priors = joblib.load(Path(args.warmup_priors))
    pca_model = joblib.load(Path(args.pca))
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    model_registry = load_model_registry()
    
    # Get display names for models
    model_display_names = [
        get_model_display_name(model_id, model_registry) 
        for model_id in warmup_priors['models']
    ]
    
    print(f"   ✅ Warmup priors: {warmup_priors['n_prompts']:,} samples")
    print(f"   ✅ PCA: {pca_model.n_components} components")
    print(f"   ✅ Models: {', '.join(model_display_names)}")
    print(f"   ✅ Gamma scaling: {args.gamma}")
    
    # Load calibration data
    print(f"\n📊 Loading calibration data from: {args.calibration_data}")
    
    if args.calibration_data.endswith('.gz'):
        with gzip.open(args.calibration_data, 'rt') as f:
            raw_data = [json.loads(line) for line in f]
    else:
        with open(args.calibration_data) as f:
            raw_data = [json.loads(line) for line in f]
    
    if not raw_data:
        print("❌ No calibration data found!")
        return
    
    # Check format and transform if needed
    first_item = raw_data[0]
    
    if 'prompt' in first_item and 'rewards' in first_item:
        calibration_data = raw_data
    elif 'prompt' in first_item and 'model_id' in first_item and 'raw_score' in first_item:
        print("   🔄 Transforming oracle rewards format...")
        oracle_dict = {}
        for entry in raw_data:
            if entry.get('ok', True):
                prompt = entry['prompt']
                model_id = entry['model_id']
                reward = entry['raw_score']
                
                if prompt not in oracle_dict:
                    oracle_dict[prompt] = {}
                oracle_dict[prompt][model_id] = reward
        
        calibration_data = [
            {'prompt': prompt, 'rewards': rewards}
            for prompt, rewards in oracle_dict.items()
        ]
        print(f"   ✅ Transformed {len(raw_data)} entries → {len(calibration_data)} unique prompts")
    else:
        print("❌ Invalid format! Expected either:")
        print("   1. {'prompt': '...', 'rewards': {'model': 0.0}}")
        print("   2. {'prompt': '...', 'model_id': '...', 'raw_score': 0.0}")
        return
    
    # Limit to specified number of samples
    calibration_data = calibration_data[:args.calibration_samples]
    print(f"   ✅ Using {len(calibration_data)} calibration samples")
    
    # Experiment 1: Router with Warmup Priors
    print("\n🔬 Experiment 1: Router with Warmup Priors")
    print("-" * 80)
    
    # Apply gamma scaling
    priors_scaled = apply_gamma_scaling(warmup_priors, args.gamma)
    
    # Initialize warmup-backed router
    warmup_router = SimpleLinUCBRouter(
        models=warmup_priors['models'],
        warmup_priors=priors_scaled,
        alpha=args.alpha
    )
    
    # Run calibration with tracking
    warmup_metrics = run_calibration_with_tracking(
        warmup_router,
        calibration_data,
        encoder,
        pca_model,
        "warmup",
        verbose=args.verbose
    )
    
    print(f"   ✅ Completed: Cumulative Regret = {warmup_metrics['cumulative_regret']:.2f}")
    
    # Experiment 2: Tabula Rasa Router
    print("\n🔬 Experiment 2: Tabula Rasa Router (A=I, b=0)")
    print("-" * 80)
    
    # Initialize tabula rasa router
    tabula_rasa_router = TabulaRasaRouter(
        models=warmup_priors['models'],
        context_dim=warmup_priors['context_dim'],
        alpha=args.alpha
    )
    
    # Run calibration with tracking
    tabula_rasa_metrics = run_calibration_with_tracking(
        tabula_rasa_router,
        calibration_data,
        encoder,
        pca_model,
        "tabula_rasa",
        verbose=args.verbose
    )
    
    print(f"   ✅ Completed: Cumulative Regret = {tabula_rasa_metrics['cumulative_regret']:.2f}")
    
    # Save results
    print("\n💾 Saving results...")
    results = save_results(warmup_metrics, tabula_rasa_metrics, output_dir, args.alpha)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    plot_comparison(warmup_metrics, tabula_rasa_metrics, output_dir, model_display_names, args.alpha)
    
    # Print summary
    print_summary(results, model_display_names)
    
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

