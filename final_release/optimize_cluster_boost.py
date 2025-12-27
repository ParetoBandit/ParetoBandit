#!/usr/bin/env python3
"""
5-Fold Cross-Validation for Cluster Boost Weight Optimization

Uses training data with 5-fold CV to find optimal cluster_boost_weight.
Optimizes for cumulative regret (lower is better).
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from bandit import BanditRouter

def load_training_data() -> Tuple[List[Dict], Dict]:
    """
    Load training prompts and rewards.
    
    Returns:
        prompts: List of prompt dicts with cluster_id
        rewards: Dict[prompt_text -> Dict[model_id -> reward]]
    """
    base_dir = Path(__file__).parent / "data"
    
    # Load prompts
    prompts = []
    with open(base_dir / "train_prompts.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line))
    
    # Load rewards - indexed by prompt text
    reward_lookup = defaultdict(dict)
    with open(base_dir / "train_rewards.jsonl") as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt']
            model = data['model_id']
            
            # Convert logit to probability
            reward_logit = data['reward_logit']
            reward = 1.0 / (1.0 + np.exp(-reward_logit))
            
            reward_lookup[prompt][model] = reward
    
    print(f"✓ Loaded {len(prompts)} prompts")
    print(f"✓ Loaded rewards for {len(reward_lookup)} unique prompts")
    
    return prompts, reward_lookup

def create_folds(prompts: List[Dict], n_folds: int = 5) -> List[Tuple[List[int], List[int]]]:
    """
    Create stratified folds (stratify by cluster_id if possible).
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    n = len(prompts)
    indices = np.arange(n)
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    folds = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        folds.append((train_indices.tolist(), val_indices.tolist()))
    
    return folds

def simulate_fold(
    prompts: List[Dict],
    reward_lookup: Dict,
    train_indices: List[int],
    val_indices: List[int],
    cluster_boost_weight: float
) -> Dict:
    """
    Simulate one fold: train on train_indices, evaluate on val_indices.
    
    Returns metrics from validation set.
    """
    # Create fresh router with benchmark priors
    router = BanditRouter.create(
        priors="benchmark",
        cluster_boost_weight=cluster_boost_weight
    )
    
    # Verify priors are loaded (check that b vectors are non-zero)
    sample_model = list(router.registry.keys())[0]
    b_norm = float(np.linalg.norm(router.bandit.b[sample_model]))
    if b_norm < 1.0:
        print(f"  WARNING: Priors may not be loaded! b_norm = {b_norm}")
    
    
    # Training phase: process feedback to learn
    for idx in train_indices:
        prompt_data = prompts[idx]
        prompt = prompt_data['prompt']
        
        # Route (use balanced profile for cost/quality tradeoff)
        selected_model, log = router.route(prompt, profile="balanced")
        
        # Get reward
        rewards = reward_lookup.get(prompt, {})
        if selected_model in rewards:
            reward = rewards[selected_model]
            router.process_feedback(log.request_id, reward, cluster_boost=True)
    
    # Validation phase: measure performance
    cumulative_regret = 0.0
    optimal_selections = 0
    total_evaluated = 0
    
    for idx in val_indices:
        prompt_data = prompts[idx]
        prompt = prompt_data['prompt']
        
        # Get ground truth rewards
        rewards = reward_lookup.get(prompt, {})
        if not rewards:
            continue
        
        # Route (use balanced profile)
        selected_model, log = router.route(prompt, profile="balanced")
        
        if selected_model not in rewards:
            continue
        
        # Calculate regret
        actual_reward = rewards[selected_model]
        optimal_reward = max(rewards.values())
        regret = optimal_reward - actual_reward
        
        cumulative_regret += regret
        
        # Track accuracy
        optimal_model = max(rewards.items(), key=lambda x: x[1])[0]
        if selected_model == optimal_model:
            optimal_selections += 1
        
        total_evaluated += 1
    
    return {
        'cumulative_regret': cumulative_regret,
        'mean_regret': cumulative_regret / total_evaluated if total_evaluated > 0 else float('inf'),
        'accuracy': optimal_selections / total_evaluated if total_evaluated > 0 else 0.0,
        'n_evaluated': total_evaluated
    }

def cross_validate_weight(
    prompts: List[Dict],
    reward_lookup: Dict,
    folds: List[Tuple[List[int], List[int]]],
    weight: float
) -> Dict:
    """
    Run 5-fold CV for a given weight.
    
    Returns averaged metrics across folds.
    """
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        result = simulate_fold(prompts, reward_lookup, train_idx, val_idx, weight)
        fold_results.append(result)
    
    # Average across folds
    avg_results = {
        'weight': weight,
        'cumulative_regret': np.mean([r['cumulative_regret'] for r in fold_results]),
        'cumulative_regret_std': np.std([r['cumulative_regret'] for r in fold_results]),
        'mean_regret': np.mean([r['mean_regret'] for r in fold_results]),
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'n_evaluated': np.mean([r['n_evaluated'] for r in fold_results])
    }
    
    return avg_results

def grid_search(
    weight_range: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
) -> Dict:
    """
    Grid search over cluster boost weights using 5-fold CV.
    """
    print("="*70)
    print("5-FOLD CROSS-VALIDATION: CLUSTER BOOST WEIGHT OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\n📂 Loading training data...")
    prompts, reward_lookup = load_training_data()
    
    # Create folds
    print(f"\n🔀 Creating 5 folds...")
    folds = create_folds(prompts, n_folds=5)
    
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} validation")
    
    # Grid search
    print(f"\n🔍 Testing {len(weight_range)} weights with 5-fold CV...")
    print("-"*70)
    
    all_results = {}
    
    for weight in weight_range:
        print(f"\n📊 Weight = {weight}")
        
        results = cross_validate_weight(prompts, reward_lookup, folds, weight)
        all_results[weight] = results
        
        print(f"  Cumulative Regret: {results['cumulative_regret']:.3f} ± {results['cumulative_regret_std']:.3f}")
        print(f"  Mean Regret: {results['mean_regret']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.1%} ± {results['accuracy_std']:.1%}")
        print(f"  Avg Evaluated: {results['n_evaluated']:.0f} prompts/fold")
    
    return all_results

def plot_results(results: Dict):
    """Visualize grid search results."""
    weights = sorted(results.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('5-Fold CV: Cluster Boost Weight Optimization', fontsize=14, fontweight='bold')
    
    # Cumulative Regret
    regrets = [results[w]['cumulative_regret'] for w in weights]
    regret_stds = [results[w]['cumulative_regret_std'] for w in weights]
    ax1.errorbar(weights, regrets, yerr=regret_stds, marker='o', linewidth=2, 
                 markersize=10, capsize=5, color='#2E86AB')
    ax1.set_xlabel('Cluster Boost Weight', fontsize=12)
    ax1.set_ylabel('Cumulative Regret (5-Fold Avg)', fontsize=12)
    ax1.set_title('Lower is Better', fontsize=11, style='italic')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Default (0.1)')
    
    # Mark best
    best_weight = min(weights, key=lambda w: results[w]['cumulative_regret'])
    best_regret = results[best_weight]['cumulative_regret']
    ax1.plot(best_weight, best_regret, 'g*', markersize=20, label=f'Best ({best_weight})')
    ax1.legend()
    
    # Accuracy
    accuracies = [results[w]['accuracy'] for w in weights]
    accuracy_stds = [results[w]['accuracy_std'] for w in weights]
    ax2.errorbar(weights, accuracies, yerr=accuracy_stds, marker='s', linewidth=2,
                 markersize=10, capsize=5, color='#A23B72')
    ax2.set_xlabel('Cluster Boost Weight', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (5-Fold Avg)', fontsize=12)
    ax2.set_title('Higher is Better', fontsize=11, style='italic')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Default (0.1)')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Mark best
    best_acc_weight = max(weights, key=lambda w: results[w]['accuracy'])
    best_acc = results[best_acc_weight]['accuracy']
    ax2.plot(best_acc_weight, best_acc, 'g*', markersize=20, label=f'Best ({best_acc_weight})')
    ax2.legend()
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent / "cluster_boost_5fold_cv.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_path}")

def main():
    """Run 5-fold CV grid search."""
    
    # Configuration
    weight_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    # Run grid search
    results = grid_search(weight_range)
    
    # Find best
    best_regret = min(results.items(), key=lambda x: x[1]['cumulative_regret'])
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n🏆 Best by Cumulative Regret (Primary Metric):")
    print(f"   Weight: {best_regret[0]}")
    print(f"   Regret: {best_regret[1]['cumulative_regret']:.3f} ± {best_regret[1]['cumulative_regret_std']:.3f}")
    print(f"   Accuracy: {best_regret[1]['accuracy']:.1%} ± {best_regret[1]['accuracy_std']:.1%}")
    
    print(f"\n🏆 Best by Accuracy:")
    print(f"   Weight: {best_accuracy[0]}")
    print(f"   Accuracy: {best_accuracy[1]['accuracy']:.1%} ± {best_accuracy[1]['accuracy_std']:.1%}")
    print(f"   Regret: {best_accuracy[1]['cumulative_regret']:.3f} ± {best_accuracy[1]['cumulative_regret_std']:.3f}")
    
    print(f"\n📊 Default (0.1) Performance:")
    default = results[0.1]
    print(f"   Regret: {default['cumulative_regret']:.3f} ± {default['cumulative_regret_std']:.3f}")
    print(f"   Accuracy: {default['accuracy']:.1%} ± {default['accuracy_std']:.1%}")
    
    if best_regret[0] != 0.1:
        improvement = ((default['cumulative_regret'] - best_regret[1]['cumulative_regret']) 
                      / default['cumulative_regret'] * 100)
        print(f"\n✨ Improvement over default: {improvement:+.1f}%")
    
    # Generate plot
    plot_results(results)
    
    print("\n" + "="*70)
    print(f"✅ RECOMMENDATION: cluster_boost_weight = {best_regret[0]}")
    print("="*70)

if __name__ == "__main__":
    main()
