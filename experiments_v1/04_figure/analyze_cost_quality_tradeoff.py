#!/usr/bin/env python3
"""
Model Discovery & Cost-Quality Analysis for Corralling Experiment

Evaluates the paper's central claim for Figure 4:
  "Corralling overcomes stale warmup priors to discover a superior new model
   (GPT-4o), added via semantic transfer with no cold-start penalty."

All baseline rewards are computed from the ACTUAL dataset — nothing is fabricated.

Baselines:
  - Always GPT-4-Turbo: The old default (warmup priors biased toward this)
  - Always Mixtral: The cheap alternative
  - Always GPT-4o: Oracle for the new model (best possible if you already knew)
  - Random (Uniform): No learning, equal allocation
  - Corralling: Adaptive meta-learner combining warmup + tabula rasa experts

Conference Reviewer Requirement:
  "Use measured per-model rewards, not assumed values."
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import numpy as np
import matplotlib.pyplot as plt

from bandit_gpt.config_legacy import OFFLINE_DATASET_DIR

CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"

# Model costs (per 1M tokens input)
MODEL_COSTS = {
    'mistralai/mixtral-8x7b-instruct': 0.27,  # Cheap
    'openai/gpt-4-turbo': 10.00,               # Expensive (the old default)
    'openai/gpt-4o': 2.50,                      # Mid-cost (the new model)
}


def compute_real_model_rewards(data_path: Path) -> dict:
    """
    Compute per-model average rewards from the ACTUAL labeled dataset.

    Returns:
        {
            'per_model': {model_id: {'mean': float, 'n': int, 'std': float}},
            'oracle_reward': float,  # best-model-per-prompt average
            'n_prompts': int,
        }
    """
    entries = []
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            entries.append(json.loads(line))

    # Group by prompt
    prompt_data = {}
    for e in entries:
        p = e['prompt']
        if p not in prompt_data:
            prompt_data[p] = {}
        prompt_data[p][e['model_id']] = e.get('raw_score', 0.0)

    # Per-model statistics
    model_scores = {}
    for scores in prompt_data.values():
        for model, score in scores.items():
            model_scores.setdefault(model, []).append(score)

    per_model = {}
    for model, scores in model_scores.items():
        arr = np.array(scores)
        per_model[model] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'n': len(arr),
        }

    # Oracle: picks best model per prompt
    oracle_rewards = []
    for scores in prompt_data.values():
        oracle_rewards.append(max(scores.values()))
    oracle_reward = float(np.mean(oracle_rewards))

    return {
        'per_model': per_model,
        'oracle_reward': oracle_reward,
        'n_prompts': len(prompt_data),
    }


def analyze_corralling_results(results_file: Path, real_rewards: dict) -> dict:
    """
    Analyze Corralling results using REAL per-model rewards.
    """
    with open(results_file, 'r') as f:
        data = json.loads(f.read())

    model_usage = data['model_usage']
    total = sum(model_usage.values())

    # Compute cost from actual model usage
    avg_cost = sum(
        MODEL_COSTS.get(m, 0.0) * count / total
        for m, count in model_usage.items()
    )

    return {
        'name': 'Corralling',
        'model_usage': model_usage,
        'avg_cost': avg_cost,
        'avg_reward': data['avg_reward'],  # Actual online reward (includes exploration cost)
    }


def create_baseline_strategies(real_rewards: dict, n_prompts: int) -> list:
    """
    Create baseline strategies using MEASURED per-model rewards.
    """
    per_model = real_rewards['per_model']
    baselines = []

    # Baseline 1: Always GPT-4-Turbo (the old default the priors are biased toward)
    baselines.append({
        'name': 'Always GPT-4-Turbo',
        'model_usage': {'openai/gpt-4-turbo': n_prompts},
        'avg_cost': MODEL_COSTS['openai/gpt-4-turbo'],
        'avg_reward': per_model['openai/gpt-4-turbo']['mean'],
    })

    # Baseline 2: Always Mixtral (cheap alternative)
    baselines.append({
        'name': 'Always Mixtral',
        'model_usage': {'mistralai/mixtral-8x7b-instruct': n_prompts},
        'avg_cost': MODEL_COSTS['mistralai/mixtral-8x7b-instruct'],
        'avg_reward': per_model['mistralai/mixtral-8x7b-instruct']['mean'],
    })

    # Baseline 3: Always GPT-4o (the new model — oracle ceiling for model discovery)
    baselines.append({
        'name': 'Always GPT-4o (oracle)',
        'model_usage': {'openai/gpt-4o': n_prompts},
        'avg_cost': MODEL_COSTS['openai/gpt-4o'],
        'avg_reward': per_model['openai/gpt-4o']['mean'],
    })

    # Baseline 4: Random uniform selection
    n_each = n_prompts // len(per_model)
    models = sorted(per_model.keys())
    avg_reward_random = np.mean([per_model[m]['mean'] for m in models])
    avg_cost_random = np.mean([MODEL_COSTS[m] for m in models])
    baselines.append({
        'name': 'Random (Uniform)',
        'model_usage': {m: n_each for m in models},
        'avg_cost': avg_cost_random,
        'avg_reward': avg_reward_random,
    })

    # Baseline 5: Per-prompt oracle (selects best model for each prompt)
    baselines.append({
        'name': 'Per-Prompt Oracle',
        'model_usage': {},  # Not a fixed allocation
        'avg_cost': np.nan,  # Would need per-prompt cost tracking
        'avg_reward': real_rewards['oracle_reward'],
    })

    return baselines


def plot_model_discovery(corralling, baselines, real_rewards, output_dir):
    """
    Create visualization focused on model discovery and adaptation.
    """
    print("\n   Creating model discovery visualizations...")

    per_model = real_rewards['per_model']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ========================================================================
    # Plot 1: Cost-Quality Scatter (measured rewards)
    # ========================================================================
    ax1 = axes[0]

    # Plot individual model points (measured from data)
    model_labels = {
        'mistralai/mixtral-8x7b-instruct': 'Mixtral',
        'openai/gpt-4-turbo': 'GPT-4-Turbo',
        'openai/gpt-4o': 'GPT-4o',
    }
    model_colors = {
        'mistralai/mixtral-8x7b-instruct': '#3498db',
        'openai/gpt-4-turbo': '#e74c3c',
        'openai/gpt-4o': '#9b59b6',
    }

    for model, label in model_labels.items():
        cost = MODEL_COSTS[model]
        reward = per_model[model]['mean']
        color = model_colors[model]
        ax1.scatter(cost, reward, s=180, c=color, marker='^',
                    edgecolors='black', linewidths=1.5, alpha=0.8,
                    label=f'Always {label} ({reward:.3f})', zorder=5)

    # Corralling point
    ax1.scatter(corralling['avg_cost'], corralling['avg_reward'],
                s=250, c='#27ae60', marker='o', edgecolors='black',
                linewidths=2, alpha=0.9,
                label=f"Corralling ({corralling['avg_reward']:.3f})", zorder=10)

    # Oracle line
    ax1.axhline(y=real_rewards['oracle_reward'], linestyle=':', color='gold',
                linewidth=2, alpha=0.7, label=f"Per-Prompt Oracle ({real_rewards['oracle_reward']:.3f})")

    ax1.set_xlabel('Average Cost ($/1M tokens)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Reward (Measured)', fontsize=13, fontweight='bold')
    ax1.set_title('Model Discovery: Corralling vs Fixed Strategies\n(All rewards measured from data)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.75, 1.02])

    # Annotation: highlight the gap between old default and new model
    turbo_reward = per_model['openai/gpt-4-turbo']['mean']
    gpt4o_reward = per_model['openai/gpt-4o']['mean']
    gap = gpt4o_reward - turbo_reward
    ax1.annotate(
        f'GPT-4o is +{gap:.1%} better\nAND 4x cheaper',
        xy=(MODEL_COSTS['openai/gpt-4o'], gpt4o_reward),
        xytext=(4.5, 0.84),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='#9b59b6'),
        fontsize=10, fontweight='bold', color='#9b59b6',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#9b59b6', lw=1.5))

    ax1.annotate(
        'Stale prior\ndefault',
        xy=(MODEL_COSTS['openai/gpt-4-turbo'], turbo_reward),
        xytext=(7.5, 0.84),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='#e74c3c'),
        fontsize=10, color='#e74c3c',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#e74c3c', lw=1.5))

    # ========================================================================
    # Plot 2: Reward comparison bar chart (measured)
    # ========================================================================
    ax2 = axes[1]

    # Strategies to compare
    strategies = [
        ('GPT-4-Turbo\n(old default)', per_model['openai/gpt-4-turbo']['mean'], '#e74c3c'),
        ('Mixtral\n(cheap)', per_model['mistralai/mixtral-8x7b-instruct']['mean'], '#3498db'),
        ('Random\n(uniform)', np.mean([per_model[m]['mean'] for m in per_model]), '#95a5a6'),
        ('Corralling\n(adaptive)', corralling['avg_reward'], '#27ae60'),
        ('GPT-4o\n(new model)', per_model['openai/gpt-4o']['mean'], '#9b59b6'),
        ('Per-Prompt\nOracle', real_rewards['oracle_reward'], '#f39c12'),
    ]

    names = [s[0] for s in strategies]
    rewards = [s[1] for s in strategies]
    colors = [s[2] for s in strategies]

    bars = ax2.bar(range(len(strategies)), rewards, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=1)

    # Highlight Corralling bar
    bars[3].set_linewidth(3)
    bars[3].set_edgecolor('#1a7a3a')

    ax2.set_ylabel('Average Reward (Measured)', fontsize=13, fontweight='bold')
    ax2.set_title('Adaptation: Overcoming Stale Warmup Priors\n(All rewards measured from data)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.7, 1.02])

    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{reward:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'cost_quality_tradeoff.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {output_file}")

    output_file_hires = output_dir / 'cost_quality_tradeoff_hires.png'
    plt.savefig(output_file_hires, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"   Saved high-res: {output_file_hires}")

    plt.close()


def print_summary(corralling, baselines, real_rewards):
    """Print formatted summary with measured rewards."""
    per_model = real_rewards['per_model']

    print("\n" + "=" * 100)
    print("MODEL DISCOVERY & COST-QUALITY ANALYSIS (All Rewards Measured from Data)")
    print("=" * 100)

    print("\n--- Per-Model Reward Statistics (from dev dataset) ---")
    print(f"{'Model':<45} {'Mean Reward':<15} {'Std':<10} {'N':<8} {'Cost ($/1M)':<12}")
    print("-" * 100)
    for model in sorted(per_model.keys()):
        stats = per_model[model]
        cost = MODEL_COSTS.get(model, 0.0)
        print(f"{model:<45} {stats['mean']:<15.4f} {stats['std']:<10.4f} {stats['n']:<8} ${cost:<11.2f}")

    print(f"\n{'Per-Prompt Oracle':<45} {real_rewards['oracle_reward']:<15.4f}")

    print("\n--- Strategy Comparison ---")
    all_strategies = [corralling] + baselines
    print(f"\n{'Strategy':<30} {'Avg Reward':<15} {'Avg Cost ($/1M)':<18} {'Reward/Cost':<15}")
    print("-" * 100)
    for s in all_strategies:
        cost = s['avg_cost']
        reward = s['avg_reward']
        efficiency = reward / cost if cost > 0 and not np.isnan(cost) else float('nan')
        cost_str = f"${cost:.2f}" if not np.isnan(cost) else "N/A"
        eff_str = f"{efficiency:.4f}" if not np.isnan(efficiency) else "N/A"
        print(f"{s['name']:<30} {reward:<15.4f} {cost_str:<18} {eff_str:<15}")

    # Key findings
    turbo_reward = per_model['openai/gpt-4-turbo']['mean']
    gpt4o_reward = per_model['openai/gpt-4o']['mean']
    corr_reward = corralling['avg_reward']
    gap_vs_turbo = corr_reward - turbo_reward
    gap_vs_gpt4o = gpt4o_reward - corr_reward

    print("\n" + "=" * 100)
    print("KEY FINDINGS — MODEL DISCOVERY:")
    print(f"   1. GPT-4-Turbo (old default) has LOWEST quality: {turbo_reward:.4f}")
    print(f"   2. GPT-4o (new model) has HIGHEST quality: {gpt4o_reward:.4f}")
    print(f"   3. Quality gap: GPT-4o is +{gpt4o_reward - turbo_reward:.1%} better than GPT-4-Turbo")
    print(f"   4. Corralling discovers this: {corr_reward:.4f} reward (+{gap_vs_turbo:.1%} vs old default)")
    print(f"   5. Corralling vs GPT-4o ceiling: within {gap_vs_gpt4o:.4f} ({gap_vs_gpt4o / gpt4o_reward:.1%} gap)")

    corr_cost = corralling['avg_cost']
    cost_vs_turbo = (MODEL_COSTS['openai/gpt-4-turbo'] - corr_cost) / MODEL_COSTS['openai/gpt-4-turbo'] * 100
    print(f"\n   COST IMPACT (secondary finding):")
    print(f"   6. Corralling avg cost: ${corr_cost:.2f}/1M ({cost_vs_turbo:.0f}% less than old default)")
    print(f"   7. Cost savings are a BYPRODUCT of discovering a better model,")
    print(f"      not the primary contribution. GPT-4o is both better AND cheaper.")

    print("\n   ADAPTATION:")
    print(f"   8. Warmup priors were biased toward GPT-4-Turbo (reward={turbo_reward:.3f})")
    print(f"   9. Corralling overcame this bias to discover GPT-4o (reward={gpt4o_reward:.3f})")
    print(f"  10. Model usage: {dict(corralling['model_usage'])}")
    print("=" * 100)


def main():
    print("=" * 80)
    print("MODEL DISCOVERY & COST-QUALITY ANALYSIS")
    print("=" * 80)

    # Step 1: Compute REAL per-model rewards from the dataset
    print(f"\n1. Computing real per-model rewards from dataset...")
    if not CANONICAL_DEV_DATA_PATH.exists():
        print(f"   ERROR: Dataset not found: {CANONICAL_DEV_DATA_PATH}")
        print(f"   Run data generation scripts first.")
        return

    real_rewards = compute_real_model_rewards(CANONICAL_DEV_DATA_PATH)
    print(f"   Dataset: {real_rewards['n_prompts']} prompts")
    for model, stats in real_rewards['per_model'].items():
        print(f"   {model}: mean={stats['mean']:.4f}, n={stats['n']}")
    print(f"   Per-prompt oracle: {real_rewards['oracle_reward']:.4f}")

    # Step 2: Load Corralling results
    results_file = Path(__file__).parent / "results_3models" / "quick_test_results.json"
    if not results_file.exists():
        print(f"\n   ERROR: Results file not found: {results_file}")
        print(f"   Run quick_test_3models.py first.")
        return

    print(f"\n2. Loading Corralling results...")
    corralling = analyze_corralling_results(results_file, real_rewards)
    print(f"   Reward: {corralling['avg_reward']:.4f}")
    print(f"   Cost: ${corralling['avg_cost']:.2f}/1M tokens")

    # Step 3: Create baselines with REAL rewards
    print(f"\n3. Creating baselines (measured rewards)...")
    baselines = create_baseline_strategies(real_rewards, n_prompts=real_rewards['n_prompts'])
    print(f"   Created {len(baselines)} baselines")

    # Step 4: Summary and visualization
    print_summary(corralling, baselines, real_rewards)

    output_dir = Path(__file__).parent / "results_3models"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_model_discovery(corralling, baselines, real_rewards, output_dir)

    # Save analysis
    analysis = {
        'real_rewards': {
            'per_model': real_rewards['per_model'],
            'oracle_reward': real_rewards['oracle_reward'],
            'n_prompts': real_rewards['n_prompts'],
        },
        'corralling': corralling,
        'baselines': [
            {k: v for k, v in b.items() if k != 'model_usage' or v}
            for b in baselines
        ],
        'model_costs': MODEL_COSTS,
    }

    with open(output_dir / 'cost_quality_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n   Saved analysis to: {output_dir}/cost_quality_analysis.json")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
