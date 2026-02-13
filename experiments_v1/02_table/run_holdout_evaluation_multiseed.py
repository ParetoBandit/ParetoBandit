#!/usr/bin/env python3
"""
Run Corralling evaluation on HOLDOUT set with multiple seeds for statistical validation.

This script addresses the statistical rigor concerns by:
1. Running experiments with multiple random seeds (default: 10)
2. Computing confidence intervals (95% CI)
3. Performing statistical significance tests (paired t-test, Wilcoxon)
4. Reporting effect sizes (Cohen's d)
5. Extracting early-phase regret (0-500) metrics

Usage:
    python run_holdout_evaluation_multiseed.py --learning-rate 0.1 --num-seeds 10 --output data/eta_0.1_holdout_multiseed
    python run_holdout_evaluation_multiseed.py --learning-rate 1.0 --num-seeds 10 --output data/eta_1.0_holdout_multiseed
"""

import sys
from pathlib import Path

# Add src to path
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
from scipy import stats

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEFAULT_MODEL_REGISTRY_PATH,
)


class TabulaRasaRouter:
    """LinUCB router initialized from scratch (A=I, b=0)."""
    
    def __init__(self, models: List[str], context_dim: int, alpha: float = 1.0):
        self.models = models
        self.alpha = alpha
        self.context_dim = context_dim
        
        # Initialize with identity (no prior knowledge)
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        
        # Track selections
        self.selections = {m: 0 for m in models}
    
    def select_model(self, context: np.ndarray, total_steps: int = None) -> str:
        """Select model using UCB."""
        ucb_scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            
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
        self.b[model] += reward * context.flatten()


def load_data(data_path: Path) -> Dict[str, Dict]:
    """
    Load evaluation data and group by prompt.
    
    Returns:
        Dict mapping prompt_id -> {prompt: str, scores: {model_id: score}}
    """
    # Load all entries
    entries = []
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Group by prompt
    prompt_data = {}
    for entry in entries:
        prompt = entry['prompt']
        model_id = entry['model_id']
        score = entry.get('raw_score', 0.0)
        
        if prompt not in prompt_data:
            prompt_data[prompt] = {'prompt': prompt, 'scores': {}}
        prompt_data[prompt]['scores'][model_id] = score
    
    # Convert to list - keep ALL prompts (no sampling)
    prompts_list = list(prompt_data.values())
    
    return {i: data for i, data in enumerate(prompts_list)}


def run_single_seed_experiment(
    data: Dict,
    encoder: SentenceTransformer,
    pca,
    warmup_priors: Dict,
    models: List[str],
    context_dim: int,
    learning_rate: float,
    seed: int,
    name: str
) -> Dict:
    """
    Run routing experiment for a single seed.
    
    Returns:
        Dict with cumulative_regret, avg_reward, model_usage, regret_history, early_regret
    """
    # Set seed for this run (controls both expert selection AND data ordering)
    np.random.seed(seed)
    
    # Shuffle data ordering to test sensitivity to arrival order
    # Online bandit regret depends on which prompts arrive early vs late
    data_items = list(data.items())
    np.random.shuffle(data_items)
    data = dict(data_items)
    
    # Initialize routers for this seed
    if "Warmup" in name:
        router = SimpleLinUCBRouter(
            models=models,
            warmup_priors=warmup_priors,
            alpha=1.0
        )
    elif "Tabula Rasa" in name:
        router = TabulaRasaRouter(
            models=models,
            context_dim=context_dim,
            alpha=1.0
        )
    elif "Hybrid" in name or "Corralling" in name:
        warmup_expert = SimpleLinUCBRouter(
            models=models,
            warmup_priors=warmup_priors,
            alpha=1.0
        )
        
        tabula_rasa_expert = TabulaRasaRouter(
            models=models,
            context_dim=context_dim,
            alpha=1.0
        )
        
        router = CorrallingRouter(
            experts=[warmup_expert, tabula_rasa_expert],
            models=models,
            learning_rate=learning_rate
        )
    else:
        raise ValueError(f"Unknown router type: {name}")
    
    cumulative_regret = 0.0
    total_reward = 0.0
    model_usage = {}
    regret_history = []
    
    # Get oracle (best possible model per prompt)
    oracle_rewards = {}
    for prompt_id, prompt_data in data.items():
        scores = prompt_data['scores']
        if scores:
            oracle_rewards[prompt_id] = max(scores.values())
        else:
            oracle_rewards[prompt_id] = 0.0
    
    # Run simulation
    for i, (prompt_id, prompt_data) in enumerate(data.items()):
        prompt = prompt_data['prompt']
        scores = prompt_data['scores']
        
        # Get context
        context = embed_prompt(prompt, encoder, pca)
        
        # Router selects model
        selected_model = router.select_model(context)
        
        # Get reward
        reward = scores.get(selected_model, 0.0)
        
        # Calculate regret
        oracle_reward = oracle_rewards[prompt_id]
        regret = oracle_reward - reward
        cumulative_regret += regret
        total_reward += reward
        
        # Track model usage
        if selected_model not in model_usage:
            model_usage[selected_model] = 0
        model_usage[selected_model] += 1
        
        # Update router
        router.update(context, selected_model, reward)
        
        # Store regret history
        regret_history.append(cumulative_regret)
    
    avg_reward = total_reward / len(data) if data else 0.0
    
    # Compute early regret (first 500 samples, or 67% of 750)
    early_cutoff = min(500, len(data))
    early_regret = regret_history[early_cutoff - 1] if len(regret_history) >= early_cutoff else cumulative_regret
    
    return {
        'cumulative_regret': cumulative_regret,
        'avg_reward': avg_reward,
        'model_usage': model_usage,
        'total_samples': len(data),
        'regret_history': regret_history,
        'early_regret': early_regret,
        'seed': seed
    }


def compute_statistics(results_list: List[Dict]) -> Dict:
    """Compute statistics across multiple seeds."""
    
    # Extract metrics
    cum_regrets = [r['cumulative_regret'] for r in results_list]
    avg_rewards = [r['avg_reward'] for r in results_list]
    early_regrets = [r['early_regret'] for r in results_list]
    
    # Compute statistics
    stats_dict = {
        'cumulative_regret': {
            'mean': float(np.mean(cum_regrets)),
            'std': float(np.std(cum_regrets, ddof=1)),
            'sem': float(stats.sem(cum_regrets)),
            'ci_95': (
                float(np.mean(cum_regrets) - 1.96 * stats.sem(cum_regrets)),
                float(np.mean(cum_regrets) + 1.96 * stats.sem(cum_regrets))
            ),
            'median': float(np.median(cum_regrets)),
            'min': float(np.min(cum_regrets)),
            'max': float(np.max(cum_regrets))
        },
        'avg_reward': {
            'mean': float(np.mean(avg_rewards)),
            'std': float(np.std(avg_rewards, ddof=1)),
            'sem': float(stats.sem(avg_rewards)),
            'ci_95': (
                float(np.mean(avg_rewards) - 1.96 * stats.sem(avg_rewards)),
                float(np.mean(avg_rewards) + 1.96 * stats.sem(avg_rewards))
            ),
        },
        'early_regret': {
            'mean': float(np.mean(early_regrets)),
            'std': float(np.std(early_regrets, ddof=1)),
            'sem': float(stats.sem(early_regrets)),
            'ci_95': (
                float(np.mean(early_regrets) - 1.96 * stats.sem(early_regrets)),
                float(np.mean(early_regrets) + 1.96 * stats.sem(early_regrets))
            ),
        },
        'num_seeds': len(results_list),
        'raw_values': {
            'cumulative_regret': cum_regrets,
            'avg_reward': avg_rewards,
            'early_regret': early_regrets
        }
    }
    
    # Aggregate model usage
    all_models = set()
    for r in results_list:
        all_models.update(r['model_usage'].keys())
    
    model_usage_agg = {}
    for model in all_models:
        counts = [r['model_usage'].get(model, 0) for r in results_list]
        model_usage_agg[model] = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts, ddof=1))
        }
    
    stats_dict['model_usage'] = model_usage_agg
    
    return stats_dict


def perform_statistical_tests(baseline_results: List[Dict], treatment_results: List[Dict]) -> Dict:
    """
    Perform statistical significance tests comparing two strategies.
    
    Args:
        baseline_results: List of results from baseline strategy (e.g., η=0.1)
        treatment_results: List of results from treatment strategy (e.g., η=1.0)
    
    Returns:
        Dict with test results
    """
    baseline_regrets = [r['cumulative_regret'] for r in baseline_results]
    treatment_regrets = [r['cumulative_regret'] for r in treatment_results]
    
    # Paired t-test (assumes normality)
    t_stat, t_pvalue = stats.ttest_rel(baseline_regrets, treatment_regrets)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(baseline_regrets, treatment_regrets)
    
    # Effect size (Cohen's d for paired samples)
    differences = np.array(baseline_regrets) - np.array(treatment_regrets)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    
    # Mean improvement
    mean_baseline = np.mean(baseline_regrets)
    mean_treatment = np.mean(treatment_regrets)
    improvement = mean_baseline - mean_treatment
    improvement_pct = 100 * improvement / mean_baseline if mean_baseline != 0 else 0
    
    return {
        'paired_t_test': {
            'statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant_at_0.05': bool(t_pvalue < 0.05),
            'significant_at_0.01': bool(t_pvalue < 0.01)
        },
        'wilcoxon_test': {
            'statistic': float(wilcoxon_stat),
            'p_value': float(wilcoxon_pvalue),
            'significant_at_0.05': bool(wilcoxon_pvalue < 0.05),
            'significant_at_0.01': bool(wilcoxon_pvalue < 0.01)
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': (
                'negligible' if abs(cohens_d) < 0.2 else
                'small' if abs(cohens_d) < 0.5 else
                'medium' if abs(cohens_d) < 0.8 else
                'large'
            )
        },
        'improvement': {
            'absolute': float(improvement),
            'percentage': float(improvement_pct),
            'direction': 'treatment_better' if improvement > 0 else 'baseline_better'
        }
    }


def plot_results_multiseed(all_results: Dict, output_dir: Path):
    """Generate comparison plots with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = list(all_results.keys())
    
    # Plot 1: Cumulative Regret with Error Bars
    ax1 = axes[0, 0]
    x = np.arange(len(strategies))
    means = [all_results[s]['statistics']['cumulative_regret']['mean'] for s in strategies]
    stds = [all_results[s]['statistics']['cumulative_regret']['std'] for s in strategies]
    
    ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha='right')
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('Total Regret (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Early Regret (0-500)
    ax2 = axes[0, 1]
    early_means = [all_results[s]['statistics']['early_regret']['mean'] for s in strategies]
    early_stds = [all_results[s]['statistics']['early_regret']['std'] for s in strategies]
    
    ax2.bar(x, early_means, yerr=early_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    ax2.set_ylabel('Early Regret (0-500)', fontsize=12)
    ax2.set_title('Early Phase Regret (Mean ± Std)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Average Reward
    ax3 = axes[1, 0]
    reward_means = [all_results[s]['statistics']['avg_reward']['mean'] for s in strategies]
    reward_stds = [all_results[s]['statistics']['avg_reward']['std'] for s in strategies]
    
    ax3.bar(x, reward_means, yerr=reward_stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=15, ha='right')
    ax3.set_ylabel('Average Reward', fontsize=12)
    ax3.set_title('Average Reward (Mean ± Std)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Regret Distribution (Box Plot)
    ax4 = axes[1, 1]
    regret_data = [all_results[s]['results'] for s in strategies]
    # Extract cumulative_regret from each result
    regret_values = [[r['cumulative_regret'] for r in data] for data in regret_data]
    bp = ax4.boxplot(regret_values, labels=strategies, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xticklabels(strategies, rotation=15, ha='right')
    ax4.set_ylabel('Cumulative Regret', fontsize=12)
    ax4.set_title('Regret Distribution Across Seeds', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multiseed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir / 'multiseed_comparison.png'}")


def print_summary_multiseed(all_results: Dict, statistical_tests: Dict = None):
    """Print results summary with statistics."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS (HOLDOUT SET - MULTI-SEED)")
    print("="*80)
    
    print(f"\n{'Strategy':<20} {'Cum. Regret':<25} {'Early Regret (0-500)':<25} {'Avg. Reward':<20}")
    print("-"*90)
    
    for name, results in all_results.items():
        stats_dict = results['statistics']
        
        cum_mean = stats_dict['cumulative_regret']['mean']
        cum_ci = stats_dict['cumulative_regret']['ci_95']
        
        early_mean = stats_dict['early_regret']['mean']
        early_ci = stats_dict['early_regret']['ci_95']
        
        reward_mean = stats_dict['avg_reward']['mean']
        reward_ci = stats_dict['avg_reward']['ci_95']
        
        print(
            f"{name:<20} "
            f"{cum_mean:.1f} [{cum_ci[0]:.1f}, {cum_ci[1]:.1f}]    "
            f"{early_mean:.1f} [{early_ci[0]:.1f}, {early_ci[1]:.1f}]    "
            f"{reward_mean:.4f} [{reward_ci[0]:.4f}, {reward_ci[1]:.4f}]"
        )
    
    # Print statistical tests if provided
    if statistical_tests:
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        for comparison, test_results in statistical_tests.items():
            print(f"\n{comparison}:")
            print(f"  Paired t-test: t={test_results['paired_t_test']['statistic']:.3f}, "
                  f"p={test_results['paired_t_test']['p_value']:.4f} "
                  f"{'***' if test_results['paired_t_test']['significant_at_0.01'] else '**' if test_results['paired_t_test']['significant_at_0.05'] else 'ns'}")
            print(f"  Wilcoxon test: W={test_results['wilcoxon_test']['statistic']:.0f}, "
                  f"p={test_results['wilcoxon_test']['p_value']:.4f} "
                  f"{'***' if test_results['wilcoxon_test']['significant_at_0.01'] else '**' if test_results['wilcoxon_test']['significant_at_0.05'] else 'ns'}")
            print(f"  Effect size (Cohen's d): {test_results['effect_size']['cohens_d']:.3f} "
                  f"({test_results['effect_size']['interpretation']})")
            print(f"  Improvement: {test_results['improvement']['absolute']:.2f} "
                  f"({test_results['improvement']['percentage']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Corralling on Holdout Set (Multi-Seed)')
    parser.add_argument('--gamma', type=float, default=0.05, help='Gamma scaling for warmup priors')
    parser.add_argument('--learning-rate', type=float, required=True, help='Corralling learning rate (0.1 or 1.0)')
    parser.add_argument('--num-seeds', type=int, default=30, help='Number of random seeds to run')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CORRALLING EVALUATION ON HOLDOUT SET (MULTI-SEED)")
    print("="*80)
    print(f"⚠️  IMPORTANT: Using HOLDOUT set for out-of-sample evaluation")
    print(f"   Running with {args.num_seeds} random seeds for statistical rigor")
    print("="*80)
    print(f"Gamma: {args.gamma}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Number of Seeds: {args.num_seeds}")
    print(f"Output: {output_dir}")
    
    # Load resources
    print("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # Get model list
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    print(f"   Models: {models}")
    print(f"   Context Dim: {context_dim}")
    
    # Load HOLDOUT data
    print("\n📊 Loading HOLDOUT evaluation data...")
    data = load_data(Path(CANONICAL_HOLDOUT_DATA_PATH))
    print(f"   Loaded {len(data)} samples from HOLDOUT set")
    
    # Scale priors
    priors_scaled = apply_gamma_scaling(warmup_priors, gamma=args.gamma)
    
    # Run experiments for all strategies across multiple seeds
    print(f"\n🔬 Running experiments with {args.num_seeds} seeds...")
    
    all_results = {
        'Warmup': {'results': []},
        'Tabula Rasa': {'results': []},
        'Hybrid (Corralling)': {'results': []}
    }
    
    for seed in range(args.num_seeds):
        print(f"\n  Seed {seed + 1}/{args.num_seeds}")
        
        # Warmup
        result = run_single_seed_experiment(
            data, encoder, pca, priors_scaled, models, context_dim, 
            args.learning_rate, seed, "Warmup"
        )
        all_results['Warmup']['results'].append(result)
        
        # Tabula Rasa
        result = run_single_seed_experiment(
            data, encoder, pca, priors_scaled, models, context_dim,
            args.learning_rate, seed, "Tabula Rasa"
        )
        all_results['Tabula Rasa']['results'].append(result)
        
        # Hybrid (Corralling)
        result = run_single_seed_experiment(
            data, encoder, pca, priors_scaled, models, context_dim,
            args.learning_rate, seed, "Hybrid (Corralling)"
        )
        all_results['Hybrid (Corralling)']['results'].append(result)
    
    # Compute statistics for each strategy
    print("\n📊 Computing statistics...")
    for strategy_name in all_results:
        all_results[strategy_name]['statistics'] = compute_statistics(
            all_results[strategy_name]['results']
        )
    
    # Perform statistical tests (if we have another learning rate to compare)
    statistical_tests = {}
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save raw results and statistics
    results_to_save = {}
    for strategy, data in all_results.items():
        results_to_save[strategy] = {
            'statistics': data['statistics'],
            'num_seeds': args.num_seeds
        }
    
    with open(output_dir / 'results_multiseed.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save detailed per-seed results
    with open(output_dir / 'results_per_seed.json', 'w') as f:
        per_seed_data = {}
        for strategy, data in all_results.items():
            per_seed_data[strategy] = [
                {k: v for k, v in r.items() if k != 'regret_history'}  # Exclude history for brevity
                for r in data['results']
            ]
        json.dump(per_seed_data, f, indent=2)
    
    # Generate plots
    plot_results_multiseed(all_results, output_dir)
    
    # Print summary
    print_summary_multiseed(all_results, statistical_tests)
    
    print("\n✅ Multi-seed evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Compare η=0.1 vs η=1.0 using statistical tests")
    print(f"  2. Run: python compare_learning_rates.py")


if __name__ == '__main__':
    main()
