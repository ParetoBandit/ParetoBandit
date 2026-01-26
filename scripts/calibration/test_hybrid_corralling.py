#!/usr/bin/env python3
"""
Test Hybrid/Corralling Router for Robust Warmup.

This script evaluates three strategies:
1. **Warmup**: Standard warmup priors (may suffer from negative transfer)
2. **Tabula Rasa**: No priors, learn from scratch
3. **Hybrid (Corralling)**: Adaptively combine warmup and tabula rasa

The Corralling algorithm provides safety guarantees: if warmup priors are
harmful, the algorithm will downweight them and rely on tabula rasa.

Usage:
    python test_hybrid_corralling.py --gamma 0.05
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

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
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


def load_data(data_path: Path, sample_size: int = None) -> Dict[str, Dict]:
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
            prompt_data[prompt] = {
                'prompt': prompt,
                'scores': {}
            }
        
        prompt_data[prompt]['scores'][model_id] = score
    
    # Convert to list and sample if needed
    data_list = list(prompt_data.values())
    
    if sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(data_list), size=min(sample_size, len(data_list)), replace=False)
        data_list = [data_list[i] for i in indices]
    
    return data_list


def compute_oracle_reward(sample: Dict, model: str) -> Tuple[float, float]:
    """
    Compute oracle reward for a model on a sample.
    
    Oracle chooses the best model available in the data for each prompt.
    Reward is the model's actual score from the dataset.
    
    Args:
        sample: Dict with 'prompt' and 'scores' keys
        model: Model ID that was selected
    
    Returns:
        Tuple of (model_reward, oracle_reward)
    """
    scores = sample.get('scores', {})
    
    if not scores:
        return 0.0, 0.0
    
    # Oracle always picks the best model
    oracle_model = max(scores, key=scores.get)
    oracle_reward = scores[oracle_model]
    
    # Return the selected model's reward and oracle reward
    return scores.get(model, 0.0), oracle_reward


def run_experiment(
    router,
    data: List[Dict],
    encoder,
    pca,
    router_name: str
) -> Dict:
    """
    Run evaluation for a single router.
    
    Returns:
        Dict with metrics (cumulative_regret, avg_reward, model_usage, etc.)
    """
    cumulative_regret = 0.0
    total_reward = 0.0
    regret_history = []
    reward_history = []
    
    # Track selections manually
    selection_counts = {}
    
    # Track expert weights over time (for hybrid)
    expert_weights_history = []
    
    for i, sample in enumerate(tqdm(data, desc=f"Evaluating {router_name}")):
        prompt = sample['prompt']
        
        # Embed prompt
        context = embed_prompt(prompt, encoder, pca)
        
        # Select model
        selected_model = router.select_model(context)
        
        # Track selection
        selection_counts[selected_model] = selection_counts.get(selected_model, 0) + 1
        
        # Get reward and oracle
        model_reward, oracle_reward = compute_oracle_reward(sample, selected_model)
        
        # Compute regret
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        regret_history.append(cumulative_regret)
        reward_history.append(total_reward / (i + 1))
        
        # Track expert weights (if applicable)
        if hasattr(router, 'weights'):
            expert_weights_history.append(router.weights.copy())
        
        # Update router
        router.update(context, selected_model, model_reward)
    
    result = {
        'cumulative_regret': cumulative_regret,
        'avg_reward': total_reward / len(data),
        'regret_history': regret_history,
        'reward_history': reward_history,
        'model_usage': selection_counts,
        'total_samples': len(data)
    }
    
    # Add expert weights for hybrid router
    if expert_weights_history:
        result['expert_weights_history'] = expert_weights_history
        result['final_expert_weights'] = expert_weights_history[-1] if expert_weights_history else None
    
    return result


def plot_results(results: Dict[str, Dict], output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Cumulative Regret & Average Reward
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Cumulative Regret
    for name, metrics in results.items():
        axes[0].plot(metrics['regret_history'], label=name, linewidth=2)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Cumulative Regret')
    axes[0].set_title('Cumulative Regret Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Average Reward
    for name, metrics in results.items():
        axes[1].plot(metrics['reward_history'], label=name, linewidth=2)
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Average Reward Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plots saved to {output_dir}/hybrid_comparison.png")
    
    # Plot 2: Expert Weights Evolution (if available)
    if 'Hybrid (Corralling)' in results:
        hybrid_metrics = results['Hybrid (Corralling)']
        if 'expert_weights_history' in hybrid_metrics:
            weights_history = np.array(hybrid_metrics['expert_weights_history'])
            
            plt.figure(figsize=(10, 6))
            plt.plot(weights_history[:, 0], label='Warmup Expert', linewidth=2, color='orange')
            plt.plot(weights_history[:, 1], label='Tabula Rasa Expert', linewidth=2, color='green')
            plt.xlabel('Samples')
            plt.ylabel('Expert Weight')
            plt.title('Corralling: Expert Weight Evolution Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            
            # Add final weights annotation
            final_warmup = weights_history[-1, 0]
            final_tabula = weights_history[-1, 1]
            plt.annotate(
                f'Final:\nWarmup: {final_warmup:.3f}\nTabula Rasa: {final_tabula:.3f}',
                xy=(len(weights_history)-1, 0.5),
                xytext=(len(weights_history)*0.7, 0.5),
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10
            )
            
            plt.tight_layout()
            plt.savefig(output_dir / 'expert_weights_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Expert weights plot saved to {output_dir}/expert_weights_evolution.png")


def print_summary(results: Dict[str, Dict]):
    """Print summary table."""
    print("\n" + "="*80)
    print("HYBRID/CORRALLING EVALUATION RESULTS")
    print("="*80)
    
    # Summary table
    print(f"\n{'Strategy':<20} {'Cumul. Regret':<15} {'Avg Reward':<15} {'Winner'}")
    print("-" * 80)
    
    best_regret = min(r['cumulative_regret'] for r in results.values())
    best_reward = max(r['avg_reward'] for r in results.values())
    
    for name, metrics in results.items():
        is_best_regret = metrics['cumulative_regret'] == best_regret
        is_best_reward = metrics['avg_reward'] == best_reward
        
        winner = ""
        if is_best_regret and is_best_reward:
            winner = "🏆 WINNER"
        elif is_best_regret:
            winner = "✅ Best Regret"
        elif is_best_reward:
            winner = "✅ Best Reward"
        
        print(
            f"{name:<20} "
            f"{metrics['cumulative_regret']:<15.2f} "
            f"{metrics['avg_reward']:<15.4f} "
            f"{winner}"
        )
    
    # Detailed model usage
    print("\n" + "="*80)
    print("MODEL USAGE BREAKDOWN")
    print("="*80)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        usage = metrics['model_usage']
        total = sum(usage.values())
        for model, count in sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]:
            pct = 100.0 * count / total if total > 0 else 0.0
            print(f"  {model:<40} {count:>5} ({pct:>5.1f}%)")
    
    # Hybrid-specific metrics
    if 'Hybrid (Corralling)' in results:
        hybrid_results = results['Hybrid (Corralling)']
        if 'final_expert_weights' in hybrid_results and hybrid_results['final_expert_weights'] is not None:
            print("\n" + "="*80)
            print("HYBRID EXPERT WEIGHTS (Final)")
            print("="*80)
            weights = hybrid_results['final_expert_weights']
            print(f"  Expert 0 (Warmup):       {weights[0]:.4f} ({weights[0]*100:.1f}%)")
            print(f"  Expert 1 (Tabula Rasa):  {weights[1]:.4f} ({weights[1]*100:.1f}%)")
            
            if weights[1] > weights[0]:
                print(f"\n  ✅ Tabula Rasa won: {weights[1]/weights[0]:.2f}x more weight than Warmup")
            else:
                print(f"\n  ⚠️  Warmup still dominates: {weights[0]/weights[1]:.2f}x more weight than Tabula Rasa")


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid/Corralling Router')
    parser.add_argument('--gamma', type=float, default=0.05, help='Gamma scaling for warmup priors')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Corralling learning rate')
    parser.add_argument('--output', type=str, default='results/hybrid_corralling', help='Output directory')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'holdout'], help='Data split to use')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HYBRID/CORRALLING ROUTER EVALUATION")
    print("="*80)
    print(f"Data Split: {args.split.upper()}")
    print(f"Gamma: {args.gamma}")
    print(f"Sample Size: {args.sample_size or 'ALL'}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output: {output_dir}")
    
    if args.split == 'holdout':
        print("\n⚠️  IMPORTANT: Using HOLDOUT set for out-of-sample evaluation")
        print("   Dev set was used for hyperparameter tuning")
        print("   Holdout set provides unbiased performance metrics")
    
    # Load resources
    print("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    
    # Get model list
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    print(f"   Models: {models}")
    print(f"   Context Dim: {context_dim}")
    
    # Load data based on split
    print(f"\n📊 Loading {args.split.upper()} evaluation data...")
    from bandit_gpt.config_legacy import CANONICAL_HOLDOUT_DATA_PATH
    data_path = Path(CANONICAL_HOLDOUT_DATA_PATH) if args.split == 'holdout' else Path(CANONICAL_DEV_DATA_PATH)
    data = load_data(data_path, sample_size=args.sample_size)
    print(f"   Loaded {len(data)} samples from {args.split.upper()} set")
    
    # Scale priors
    priors_scaled = apply_gamma_scaling(warmup_priors, gamma=args.gamma)
    
    # Initialize routers
    print("\n🚀 Initializing routers...")
    
    # 1. Warmup Router
    warmup_router = SimpleLinUCBRouter(
        models=models,
        warmup_priors=priors_scaled,
        alpha=1.0
    )
    
    # 2. Tabula Rasa Router
    tabula_rasa_router = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    # 3. Hybrid Router (Corralling)
    # Create fresh instances for the hybrid
    warmup_expert = SimpleLinUCBRouter(
        models=models,
        warmup_priors=priors_scaled,
        alpha=1.0
    )
    
    tabula_rasa_expert = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    hybrid_router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=args.learning_rate
    )
    
    # Run experiments
    print("\n🔬 Running experiments...")
    results = {}
    
    # Experiment 1: Warmup
    results['Warmup'] = run_experiment(
        warmup_router,
        data,
        encoder,
        pca,
        "Warmup"
    )
    
    # Experiment 2: Tabula Rasa
    results['Tabula Rasa'] = run_experiment(
        tabula_rasa_router,
        data,
        encoder,
        pca,
        "Tabula Rasa"
    )
    
    # Experiment 3: Hybrid (Corralling)
    results['Hybrid (Corralling)'] = run_experiment(
        hybrid_router,
        data,
        encoder,
        pca,
        "Hybrid (Corralling)"
    )
    
    # Save results
    print("\n💾 Saving results...")
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for name, metrics in results.items():
            results_serializable[name] = {
                'cumulative_regret': float(metrics['cumulative_regret']),
                'avg_reward': float(metrics['avg_reward']),
                'model_usage': metrics['model_usage'],
                'total_samples': metrics['total_samples']
            }
        json.dump(results_serializable, f, indent=2)
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

