#!/usr/bin/env python3
"""
Issue #2: Ablation Studies for Corralling Hyperparameters

Tests robustness across different hyperparameter settings:
1. Learning rate (η): {0.1, 0.5, 1.0, 2.0, 5.0}
2. Mixing parameter (γ): {0.0, 0.05, 0.10}
3. Multiple random seeds for statistical significance

Conference Reviewer Requirement: "Add ablation studies (vary learning rate η, compare γ=0 vs γ=0.05)"
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    OFFLINE_DATASET_DIR,
)

# Import helper functions
sys.path.insert(0, str(Path(__file__).parent))
from corralled_semantic_analysis import (
    TabulaRasaRouter,
    load_labeled_data,
    compute_oracle_reward,
    extend_priors_with_semantic_transfer
)

CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"


def run_single_experiment(
    data,
    encoder,
    pca,
    warmup_priors,
    learning_rate: float,
    gamma: float,
    seed: int = 42
):
    """
    Run a single Corralling experiment with given hyperparameters.
    
    Returns:
        dict with results
    """
    np.random.seed(seed)
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    # Initialize experts
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
    
    # Initialize Corralling with custom hyperparameters
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    # Training loop
    cumulative_regret = 0.0
    total_reward = 0.0
    regret_history = []
    reward_history = []
    expert_weights_history = []
    
    for i, sample in enumerate(data):
        prompt = sample['prompt']
        context = embed_prompt(prompt, encoder, pca)
        
        # CorrallingRouter.select_model returns (model_id, selection_token)
        selected_model, selection_token = router.select_model(context)
        model_reward, oracle_reward = compute_oracle_reward(sample, selected_model)
        
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        regret_history.append(cumulative_regret)
        reward_history.append(total_reward / (i + 1))
        expert_weights_history.append(router.weights.copy())
        
        # Pass selection_token so the meta-weight update is applied
        router.update(context, selected_model, model_reward, selection_token=selection_token)
    
    return {
        'learning_rate': learning_rate,
        'gamma': gamma,
        'seed': seed,
        'cumulative_regret': float(cumulative_regret),
        'avg_reward': float(total_reward / len(data)),
        'final_expert_weights': router.weights.tolist(),
        'model_usage': router.selections,
        'regret_history': regret_history,
        'reward_history': reward_history,
        'expert_weights_history': [w.tolist() for w in expert_weights_history]
    }


def run_ablation_experiments(
    data,
    encoder,
    pca,
    warmup_priors,
    learning_rates=[0.1, 0.5, 1.0, 2.0, 5.0],
    gammas=[0.0, 0.05, 0.10],
    seeds=[42, 43, 44]
):
    """
    Run full ablation study across hyperparameters and seeds.
    """
    results = []
    total_experiments = len(learning_rates) * len(gammas) * len(seeds)
    
    print(f"\n🔬 Running {total_experiments} experiments...")
    print(f"   Learning rates: {learning_rates}")
    print(f"   Gammas: {gammas}")
    print(f"   Seeds: {seeds}")
    
    with tqdm(total=total_experiments, desc="   Progress") as pbar:
        for eta in learning_rates:
            for gamma in gammas:
                for seed in seeds:
                    result = run_single_experiment(
                        data, encoder, pca, warmup_priors,
                        learning_rate=eta,
                        gamma=gamma,
                        seed=seed
                    )
                    results.append(result)
                    pbar.update(1)
    
    return results


def analyze_ablation_results(results):
    """
    Analyze ablation results and compute statistics.
    """
    print("\n📊 Analyzing results...")
    
    # Group by hyperparameters
    grouped = {}
    for r in results:
        key = (r['learning_rate'], r['gamma'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    # Compute statistics for each configuration
    summary = []
    for (eta, gamma), runs in grouped.items():
        regrets = [r['cumulative_regret'] for r in runs]
        rewards = [r['avg_reward'] for r in runs]
        
        summary.append({
            'learning_rate': eta,
            'gamma': gamma,
            'n_seeds': len(runs),
            'mean_regret': np.mean(regrets),
            'std_regret': np.std(regrets),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_regret': np.min(regrets),
            'max_regret': np.max(regrets),
        })
    
    return summary


def plot_ablation_results(results, summary, output_dir):
    """
    Create visualizations for ablation study.
    """
    print("\n🎨 Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========================================================================
    # Plot 1: Regret vs Learning Rate (for different gammas)
    # ========================================================================
    ax1 = axes[0, 0]
    
    gammas_unique = sorted(set(s['gamma'] for s in summary))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for gamma, color in zip(gammas_unique, colors):
        data = [s for s in summary if s['gamma'] == gamma]
        data = sorted(data, key=lambda x: x['learning_rate'])
        
        etas = [d['learning_rate'] for d in data]
        means = [d['mean_regret'] for d in data]
        stds = [d['std_regret'] for d in data]
        
        ax1.plot(etas, means, 'o-', linewidth=2.5, markersize=8, 
                color=color, label=f'γ={gamma}')
        ax1.fill_between(etas, 
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, color=color)
    
    ax1.set_xlabel('Learning Rate (η)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Regret (mean ± std)', fontsize=13, fontweight='bold')
    ax1.set_title('Regret vs Learning Rate', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # ========================================================================
    # Plot 2: Reward vs Learning Rate
    # ========================================================================
    ax2 = axes[0, 1]
    
    for gamma, color in zip(gammas_unique, colors):
        data = [s for s in summary if s['gamma'] == gamma]
        data = sorted(data, key=lambda x: x['learning_rate'])
        
        etas = [d['learning_rate'] for d in data]
        means = [d['mean_reward'] for d in data]
        stds = [d['std_reward'] for d in data]
        
        ax2.plot(etas, means, 'o-', linewidth=2.5, markersize=8,
                color=color, label=f'γ={gamma}')
        ax2.fill_between(etas,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, color=color)
    
    ax2.set_xlabel('Learning Rate (η)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Reward (mean ± std)', fontsize=13, fontweight='bold')
    ax2.set_title('Reward vs Learning Rate', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # ========================================================================
    # Plot 3: Heatmap of Regret (η vs γ)
    # ========================================================================
    ax3 = axes[1, 0]
    
    # Create pivot table
    etas_unique = sorted(set(s['learning_rate'] for s in summary))
    heatmap_data = np.zeros((len(gammas_unique), len(etas_unique)))
    
    for i, gamma in enumerate(gammas_unique):
        for j, eta in enumerate(etas_unique):
            matching = [s for s in summary if s['gamma'] == gamma and s['learning_rate'] == eta]
            if matching:
                heatmap_data[i, j] = matching[0]['mean_regret']
    
    im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r')
    ax3.set_xticks(range(len(etas_unique)))
    ax3.set_xticklabels([f'{eta:.1f}' for eta in etas_unique])
    ax3.set_yticks(range(len(gammas_unique)))
    ax3.set_yticklabels([f'{gamma:.2f}' for gamma in gammas_unique])
    ax3.set_xlabel('Learning Rate (η)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Mixing Parameter (γ)', fontsize=13, fontweight='bold')
    ax3.set_title('Regret Heatmap (Lower is Better)', fontsize=15, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Cumulative Regret', fontsize=11)
    
    # Annotate cells with values
    for i in range(len(gammas_unique)):
        for j in range(len(etas_unique)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    # ========================================================================
    # Plot 4: Regret Convergence Curves (for different ηs, fixed γ=0.05)
    # ========================================================================
    ax4 = axes[1, 1]
    
    gamma_fixed = 0.05
    etas_to_plot = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors_eta = plt.cm.viridis(np.linspace(0, 1, len(etas_to_plot)))
    
    for eta, color in zip(etas_to_plot, colors_eta):
        matching = [r for r in results if r['learning_rate'] == eta and r['gamma'] == gamma_fixed]
        if matching:
            # Average across seeds
            histories = [r['regret_history'] for r in matching]
            mean_history = np.mean(histories, axis=0)
            std_history = np.std(histories, axis=0)
            
            T = len(mean_history)
            time_steps = np.arange(1, T + 1)
            
            ax4.plot(time_steps, mean_history, linewidth=2, color=color, label=f'η={eta}')
            ax4.fill_between(time_steps,
                            mean_history - std_history,
                            mean_history + std_history,
                            alpha=0.15, color=color)
    
    ax4.set_xlabel('Time Steps (T)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax4.set_title(f'Regret Curves (γ={gamma_fixed})', fontsize=15, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / 'ablation_study.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved: {output_file}")
    
    # High-res version
    output_file_hires = output_dir / 'ablation_study_hires.png'
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved high-res: {output_file_hires}")
    
    plt.close()


def print_summary_table(summary):
    """Print formatted summary table."""
    print("\n" + "="*120)
    print("ABLATION STUDY RESULTS")
    print("="*120)
    
    # Sort by learning rate, then gamma
    summary = sorted(summary, key=lambda x: (x['learning_rate'], x['gamma']))
    
    print(f"\n{'η':<8} {'γ':<8} {'Seeds':<8} {'Regret (mean±std)':<25} {'Reward (mean±std)':<25} {'Best?':<10}")
    print("-" * 120)
    
    # Find best configuration (lowest mean regret)
    best_config = min(summary, key=lambda x: x['mean_regret'])
    
    for s in summary:
        is_best = (s == best_config)
        mark = "✅ BEST" if is_best else ""
        
        print(f"{s['learning_rate']:<8.1f} {s['gamma']:<8.2f} {s['n_seeds']:<8} "
              f"{s['mean_regret']:>7.2f} ± {s['std_regret']:>6.2f}      "
              f"{s['mean_reward']:>7.4f} ± {s['std_reward']:>7.4f}      "
              f"{mark}")
    
    print("\n" + "="*120)
    print(f"BEST CONFIGURATION:")
    print(f"   η = {best_config['learning_rate']}")
    print(f"   γ = {best_config['gamma']}")
    print(f"   Mean Regret: {best_config['mean_regret']:.2f} ± {best_config['std_regret']:.2f}")
    print(f"   Mean Reward: {best_config['mean_reward']:.4f} ± {best_config['std_reward']:.4f}")
    print("="*120)


def main():
    print("="*80)
    print("ABLATION STUDY: Corralling Hyperparameters")
    print("="*80)
    
    # Load resources
    print("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    # Load data
    print("\n📊 Loading labeled data...")
    labeled_data = load_labeled_data(Path(CANONICAL_DEV_DATA_PATH), sample_size=1121)
    print(f"   ✅ Loaded {len(labeled_data)} samples")
    
    # Step 1: Extend priors for GPT-4o BEFORE scaling (avoids double-scaling)
    all_models_in_data = set()
    for sample in labeled_data:
        all_models_in_data.update(sample['scores'].keys())
    all_models_in_data = sorted(all_models_in_data)
    
    missing_models = [m for m in all_models_in_data if m not in warmup_priors['models']]
    if missing_models:
        print(f"\n🔄 Extending priors for: {missing_models}")
        transfer_mapping = {'openai/gpt-4o': 'openai/gpt-4-turbo'}
        warmup_priors = extend_priors_with_semantic_transfer(
            warmup_priors,
            new_models=missing_models,
            transfer_mapping=transfer_mapping,
            gamma=0.05
        )
    
    # Step 2: Apply gamma scaling uniformly to ALL models (including transferred)
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=0.05)
    
    # Run ablation experiments
    results = run_ablation_experiments(
        data=labeled_data,
        encoder=encoder,
        pca=pca,
        warmup_priors=warmup_priors_scaled,
        learning_rates=[0.1, 0.5, 1.0, 2.0, 5.0],
        gammas=[0.0, 0.05, 0.10],
        seeds=[42, 43, 44]  # 3 seeds for statistical significance
    )
    
    # Analyze results
    summary = analyze_ablation_results(results)
    
    # Print summary table
    print_summary_table(summary)
    
    # Save results
    output_dir = Path(__file__).parent / "results_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved raw results to: {output_dir}/ablation_results.json")
    
    with open(output_dir / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary to: {output_dir}/ablation_summary.json")
    
    # Create visualizations
    plot_ablation_results(results, summary, output_dir)
    
    print("\n" + "="*80)
    print("✅ ABLATION STUDY COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
