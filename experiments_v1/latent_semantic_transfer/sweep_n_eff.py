#!/usr/bin/env python3
"""
Hyperparameter Sweep: Find Optimal n_eff for LST

Tests multiple n_eff values to empirically validate the choice of prior strength.
Runs the same regret waterfall experiment with different n_eff settings.

Saves:
- results/sweep_n_eff_results.json: Raw data for each n_eff value
- results/sweep_n_eff_plot.png: Visualization comparing all n_eff values
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gzip
import json
import hashlib
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.router import BanditRouter


def load_shared_prompts(rewards_file: Path) -> List[Dict[str, float]]:
    """Load prompts that have rewards for all three models."""
    import hashlib
    
    # First pass: organize by prompt hash
    prompt_data = defaultdict(dict)
    
    with gzip.open(rewards_file, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            model_id = entry['model_id']
            prompt = entry['prompt']
            reward = entry['raw_score']
            
            # Use hash of prompt as unique ID
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            if model_id in ['openai/gpt-4o', 'mistralai/mixtral-8x7b-instruct', 'openai/gpt-5-chat']:
                if 'prompt_text' not in prompt_data[prompt_hash]:
                    prompt_data[prompt_hash]['prompt_text'] = prompt
                prompt_data[prompt_hash][model_id] = reward
    
    # Second pass: keep only prompts with all three models
    shared_prompts = []
    for prompt_hash, data in prompt_data.items():
        if all(m in data for m in ['openai/gpt-4o', 'mistralai/mixtral-8x7b-instruct', 'openai/gpt-5-chat']):
            shared_prompts.append({
                'prompt_id': prompt_hash,
                'gpt4o_reward': data['openai/gpt-4o'],
                'mixtral_reward': data['mistralai/mixtral-8x7b-instruct'],
                'gpt5_reward': data['openai/gpt-5-chat'],
                'prompt_text': data['prompt_text']
            })
    
    return shared_prompts


def create_base_router(registry: Dict[str, Dict]) -> BanditRouter:
    """Create router with base models (GPT-4o, Mixtral) and warmup priors."""
    router = BanditRouter(
        model_registry=registry,
        alpha=0.05,
        init_lambda=1.0,
        verbose_routing=False
    )
    
    # Load priors for base models
    from bandit_gpt.config_legacy import DEFAULT_WARMUP_PRIORS_PATH
    priors_path = DEFAULT_WARMUP_PRIORS_PATH
    priors_data = joblib.load(priors_path)
    A_matrices = priors_data['A']
    b_vectors = priors_data['b']
    
    # Load for Mixtral (if available)
    if "mistralai/mixtral-8x7b-instruct" in router.bandit.models and "mistralai/mixtral-8x7b-instruct" in A_matrices:
        router.bandit.A["mistralai/mixtral-8x7b-instruct"] = A_matrices["mistralai/mixtral-8x7b-instruct"].copy()
        router.bandit.b["mistralai/mixtral-8x7b-instruct"] = b_vectors["mistralai/mixtral-8x7b-instruct"].copy()
        router.bandit.A_inv["mistralai/mixtral-8x7b-instruct"] = np.linalg.inv(router.bandit.A["mistralai/mixtral-8x7b-instruct"])
    
    # For GPT-4o, use GPT-4-turbo priors as surrogate
    if "openai/gpt-4o" in router.bandit.models and "openai/gpt-4-turbo" in A_matrices:
        router.bandit.A["openai/gpt-4o"] = A_matrices["openai/gpt-4-turbo"].copy()
        router.bandit.b["openai/gpt-4o"] = b_vectors["openai/gpt-4-turbo"].copy()
        router.bandit.A_inv["openai/gpt-4o"] = np.linalg.inv(router.bandit.A["openai/gpt-4o"])
    
    return router


def register_gpt5_with_n_eff(router: BanditRouter, n_eff: float) -> None:
    """Register GPT-5 with specified n_eff value."""
    # Manually compute transfer from GPT-4o
    A_inv_gpt4o = router.bandit.A_inv["openai/gpt-4o"]
    b_gpt4o = router.bandit.b["openai/gpt-4o"]
    theta_gpt4o = A_inv_gpt4o @ b_gpt4o
    
    # Initialize with specified n_eff
    A_init = np.eye(router.bandit.dim) * router.bandit.init_lambda
    b_init = (router.bandit.init_lambda * theta_gpt4o) * n_eff
    
    # Add to bandit (properly initialize all tracking dicts)
    model_id = "openai/gpt-5-chat"
    router.bandit.models.append(model_id)
    router.bandit.A[model_id] = A_init
    router.bandit.b[model_id] = b_init
    router.bandit.A_inv[model_id] = np.linalg.inv(A_init)
    
    # Initialize tracking dictionaries
    if hasattr(router.bandit, 'last_update'):
        router.bandit.last_update[model_id] = 0
    if hasattr(router.bandit, 'reward_history'):
        router.bandit.reward_history[model_id] = []


def run_online_routing(
    router: BanditRouter,
    shared_prompts: List[Dict],
    n_samples: int = 500,
) -> Tuple[List[float], List[str]]:
    """Run online routing and return cumulative regret."""
    cumulative_regret_list = [0.0]
    selected_models_list = []
    cumulative_regret = 0.0
    
    for t in range(min(n_samples, len(shared_prompts))):
        prompt_data = shared_prompts[t]
        
        # Use REAL prompt features
        prompt_text = prompt_data['prompt_text']
        context, _ = router._build_routing_features(prompt_text)
        
        # Bandit selects model
        selected_model, selected_ucb = router.bandit.select_arm(context)
        selected_models_list.append(selected_model)
        
        # Get reward for selected model
        if selected_model == "openai/gpt-4o":
            reward = prompt_data['gpt4o_reward']
        elif selected_model == "mistralai/mixtral-8x7b-instruct":
            reward = prompt_data['mixtral_reward']
        else:  # openai/gpt-5-chat
            reward = prompt_data['gpt5_reward']
        
        # Oracle reward (best possible)
        oracle_reward = max(prompt_data['gpt4o_reward'], 
                           prompt_data['mixtral_reward'],
                           prompt_data['gpt5_reward'])
        
        # Update bandit
        router.bandit.update(selected_model, context, reward)
        
        # Track regret
        regret = oracle_reward - reward
        cumulative_regret += regret
        cumulative_regret_list.append(cumulative_regret)
    
    return cumulative_regret_list, selected_models_list


def run_sweep(
    shared_prompts: List[Dict],
    n_eff_values: List[float],
    n_trials: int = 5,
    n_samples: int = 500
) -> Dict:
    """Run sweep over n_eff values."""
    
    registry = {
        "openai/gpt-4o": {
            "cost_per_1m_tokens": 10000.0,
            "median_latency_s": 2.0,
            "capabilities": ["reasoning", "coding"],
            "speed_profile": "balanced"
        },
        "mistralai/mixtral-8x7b-instruct": {
            "cost_per_1m_tokens": 500.0,
            "median_latency_s": 0.8,
            "capabilities": ["general", "coding"],
            "speed_profile": "fast"
        }
    }
    
    results = {}
    
    for n_eff in n_eff_values:
        print(f"\n{'='*80}")
        print(f"Testing n_eff = {n_eff}")
        print("="*80)
        
        trial_regrets = []
        trial_models = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}...", end=" ")
            
            # Shuffle prompts for this trial
            np.random.seed(42 + trial)
            trial_prompts = shared_prompts.copy()
            np.random.shuffle(trial_prompts)
            
            # Create router and register GPT-5 with this n_eff
            router = create_base_router(registry)
            register_gpt5_with_n_eff(router, n_eff)
            
            # Run online routing
            regret, models = run_online_routing(router, trial_prompts, n_samples)
            
            trial_regrets.append(regret)
            trial_models.append(models)
            
            print(f"Final regret: {regret[-1]:.2f}")
        
        # Compute statistics
        final_regrets = [r[-1] for r in trial_regrets]
        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        
        # Count model selections
        all_selections = [m for trial in trial_models for m in trial]
        gpt5_count = sum(1 for m in all_selections if 'gpt-5' in m)
        gpt4o_count = sum(1 for m in all_selections if 'gpt-4o' in m)
        mixtral_count = sum(1 for m in all_selections if 'mixtral' in m)
        
        results[n_eff] = {
            'mean_regret': mean_regret,
            'std_regret': std_regret,
            'final_regrets': final_regrets,
            'all_regrets': trial_regrets,  # Full time series for each trial
            'gpt5_selections': gpt5_count,
            'gpt4o_selections': gpt4o_count,
            'mixtral_selections': mixtral_count,
            'total_selections': len(all_selections)
        }
        
        print(f"  → Mean: {mean_regret:.2f} ± {std_regret:.2f}")
        print(f"  → GPT-5 selections: {gpt5_count}/{len(all_selections)} ({gpt5_count/len(all_selections)*100:.1f}%)")
    
    return results


def plot_sweep_results(results: Dict, output_path: Path) -> None:
    """Create visualization of sweep results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    n_eff_values = sorted(results.keys())
    means = [results[n]['mean_regret'] for n in n_eff_values]
    stds = [results[n]['std_regret'] for n in n_eff_values]
    
    # Plot 1: Mean regret vs n_eff
    ax1.errorbar(n_eff_values, means, yerr=stds, marker='o', capsize=5, 
                linewidth=2, markersize=8, color='#2c3e50')
    ax1.axhline(y=min(means), color='#27ae60', linestyle='--', alpha=0.5, label='Optimal')
    ax1.set_xlabel('$n_{eff}$ (Prior Strength)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Final Cumulative Regret', fontsize=13, fontweight='bold')
    ax1.set_title('Hyperparameter Sweep: $n_{eff}$ Optimization', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Annotate optimal
    optimal_idx = np.argmin(means)
    optimal_n_eff = n_eff_values[optimal_idx]
    ax1.annotate(f'Optimal: {optimal_n_eff}', 
                xy=(optimal_n_eff, means[optimal_idx]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#27ae60'),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    # Plot 2: Learning curves for each n_eff
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_eff_values)))
    for i, n_eff in enumerate(n_eff_values):
        # Average across trials
        all_regrets = results[n_eff]['all_regrets']
        mean_curve = np.mean(all_regrets, axis=0)
        std_curve = np.std(all_regrets, axis=0)
        
        samples = np.arange(len(mean_curve))
        ax2.plot(samples, mean_curve, label=f'$n_{{eff}}$={n_eff}', 
                color=colors[i], linewidth=2, alpha=0.8)
        ax2.fill_between(samples, mean_curve - std_curve, mean_curve + std_curve,
                        color=colors[i], alpha=0.15)
    
    ax2.set_xlabel('Samples', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax2.set_title('Learning Curves by $n_{eff}$', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model selection distribution
    gpt5_pcts = [results[n]['gpt5_selections'] / results[n]['total_selections'] * 100 
                 for n in n_eff_values]
    gpt4o_pcts = [results[n]['gpt4o_selections'] / results[n]['total_selections'] * 100 
                  for n in n_eff_values]
    
    ax3.bar(range(len(n_eff_values)), gpt5_pcts, label='GPT-5', color='#27ae60', alpha=0.8)
    ax3.bar(range(len(n_eff_values)), gpt4o_pcts, bottom=gpt5_pcts, 
           label='GPT-4o', color='#e74c3c', alpha=0.8)
    
    ax3.set_xlabel('$n_{eff}$', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Selection %', fontsize=13, fontweight='bold')
    ax3.set_title('Model Selection Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(n_eff_values)))
    ax3.set_xticklabels([str(n) for n in n_eff_values])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Statistical summary table
    ax4.axis('off')
    
    table_data = [['$n_{eff}$', 'Mean Regret', 'Std', 'GPT-5 %']]
    for n_eff in n_eff_values:
        r = results[n_eff]
        gpt5_pct = r['gpt5_selections'] / r['total_selections'] * 100
        table_data.append([
            f'{n_eff:.1f}',
            f'{r["mean_regret"]:.2f}',
            f'±{r["std_regret"]:.2f}',
            f'{gpt5_pct:.1f}%'
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center',
                     bbox=[0.1, 0.2, 0.8, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight optimal row
    optimal_row = optimal_idx + 1
    for i in range(4):
        table[(optimal_row, i)].set_facecolor('#d5f4e6')
        table[(optimal_row, i)].set_text_props(weight='bold')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")


def main():
    print("="*80)
    print("HYPERPARAMETER SWEEP: n_eff Optimization")
    print("="*80)
    print("\nGoal: Find optimal prior strength for Latent Semantic Transfer")
    print("Method: Test n_eff ∈ {1, 3, 5, 7, 10, 15, 20} over 5 trials each")
    
    # Load data
    base_path = Path(__file__).parent.parent.parent / "src" / "bandit_gpt" / "data" / "offline_dataset"
    dev_file = base_path / "dev_rewards_complete.jsonl.gz"
    holdout_file = base_path / "holdout_rewards_complete.jsonl.gz"
    
    print(f"\n📂 Loading prompts...")
    dev_prompts = load_shared_prompts(dev_file)
    holdout_prompts = load_shared_prompts(holdout_file)
    shared_prompts = dev_prompts + holdout_prompts
    print(f"   ✓ Total: {len(shared_prompts)} prompts (dev + holdout)")
    
    # Define sweep range
    n_eff_values = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    
    # Run sweep
    results = run_sweep(shared_prompts, n_eff_values, n_trials=5, n_samples=500)
    
    # Find optimal
    optimal_n_eff = min(results.keys(), key=lambda n: results[n]['mean_regret'])
    optimal_regret = results[optimal_n_eff]['mean_regret']
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\n🏆 Optimal n_eff: {optimal_n_eff}")
    print(f"   Regret: {optimal_regret:.2f} ± {results[optimal_n_eff]['std_regret']:.2f}")
    
    # Compare to current default (10.0)
    if 10.0 in results:
        current_regret = results[10.0]['mean_regret']
        improvement = ((current_regret - optimal_regret) / current_regret) * 100
        print(f"\n📊 Current default (n_eff=10.0): {current_regret:.2f}")
        if optimal_n_eff != 10.0:
            print(f"   → Switching to {optimal_n_eff} would improve by {improvement:.1f}%")
        else:
            print(f"   → Current default is optimal! ✅")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_path = output_dir / "sweep_n_eff_results.json"
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for n_eff, data in results.items():
            json_results[str(n_eff)] = {
                'mean_regret': float(data['mean_regret']),
                'std_regret': float(data['std_regret']),
                'final_regrets': [float(r) for r in data['final_regrets']],
                'gpt5_selections': int(data['gpt5_selections']),
                'gpt4o_selections': int(data['gpt4o_selections']),
                'mixtral_selections': int(data['mixtral_selections']),
                'total_selections': int(data['total_selections'])
            }
        
        json.dump({
            'optimal_n_eff': float(optimal_n_eff),
            'n_eff_values': [float(n) for n in n_eff_values],
            'results': json_results,
            'metadata': {
                'n_trials': 5,
                'n_samples': 500,
                'n_prompts': len(shared_prompts)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved: {json_path}")
    
    # Plot
    plot_path = output_dir / "sweep_n_eff_plot.png"
    plot_sweep_results(results, plot_path)
    
    print("\n✅ Sweep complete!")


if __name__ == "__main__":
    main()

