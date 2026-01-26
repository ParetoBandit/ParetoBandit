#!/usr/bin/env python3
"""
Run Corralling evaluation on HOLDOUT set for Table 2.

This script evaluates the corralling algorithm on the holdout set (750 samples)
to provide out-of-sample evaluation metrics for Table 2. The dev set (1,121 samples)
was used for hyperparameter tuning and should NOT be used for final reporting.

Usage:
    python run_holdout_evaluation.py --learning-rate 0.1 --output data/eta_0.1_holdout
    python run_holdout_evaluation.py --learning-rate 1.0 --output data/eta_1.0_holdout
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
    CANONICAL_HOLDOUT_DATA_PATH,  # Use HOLDOUT, not DEV
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
            prompt_data[prompt] = {'prompt': prompt, 'scores': {}}
        prompt_data[prompt]['scores'][model_id] = score
    
    # Convert to list and optionally sample
    prompts_list = list(prompt_data.values())
    
    if sample_size and sample_size < len(prompts_list):
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(prompts_list), sample_size, replace=False)
        prompts_list = [prompts_list[i] for i in sorted(indices)]
    
    return {i: data for i, data in enumerate(prompts_list)}


def run_experiment(
    router,
    data: Dict,
    encoder: SentenceTransformer,
    pca,
    name: str
) -> Dict:
    """
    Run routing experiment and track metrics.
    
    Returns:
        Dict with cumulative_regret, avg_reward, model_usage, regret_history
    """
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
    for i, (prompt_id, prompt_data) in enumerate(tqdm(data.items(), desc=f"  {name}")):
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
    
    return {
        'cumulative_regret': cumulative_regret,
        'avg_reward': avg_reward,
        'model_usage': model_usage,
        'total_samples': len(data),
        'regret_history': regret_history
    }


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Regret curves
    ax1 = axes[0]
    for name, metrics in results.items():
        if 'regret_history' in metrics:
            ax1.plot(metrics['regret_history'], label=name, linewidth=2)
    
    ax1.set_xlabel('Time Steps (Prompts)', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.set_title('Regret Accumulation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model usage
    ax2 = axes[1]
    strategies = list(results.keys())
    
    # Get top 3 models by usage
    all_models = set()
    for metrics in results.values():
        all_models.update(metrics['model_usage'].keys())
    
    model_usage_counts = {}
    for model in all_models:
        model_usage_counts[model] = sum(
            results[s]['model_usage'].get(model, 0) for s in strategies
        )
    
    top_models = sorted(model_usage_counts, key=model_usage_counts.get, reverse=True)[:3]
    
    x = np.arange(len(strategies))
    width = 0.25
    
    for i, model in enumerate(top_models):
        model_short = model.split('/')[-1][:20]
        values = [results[s]['model_usage'].get(model, 0) for s in strategies]
        ax2.bar(x + i*width, values, width, label=model_short)
    
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.set_ylabel('Selection Count', fontsize=12)
    ax2.set_title('Model Usage Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(strategies, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir / 'hybrid_comparison.png'}")


def print_summary(results: Dict):
    """Print results summary."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS (HOLDOUT SET)")
    print("="*80)
    
    # Find best performers
    best_regret = min(results.values(), key=lambda x: x['cumulative_regret'])
    best_reward = max(results.values(), key=lambda x: x['avg_reward'])
    
    print(f"\n{'Strategy':<20} {'Cum. Regret':<15} {'Avg. Reward':<15} {'Status'}")
    print("-"*80)
    
    for name, metrics in results.items():
        is_best_regret = metrics == best_regret
        is_best_reward = metrics == best_reward
        
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Corralling on Holdout Set')
    parser.add_argument('--gamma', type=float, default=0.05, help='Gamma scaling for warmup priors')
    parser.add_argument('--learning-rate', type=float, required=True, help='Corralling learning rate (0.1 or 1.0)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CORRALLING EVALUATION ON HOLDOUT SET")
    print("="*80)
    print(f"⚠️  IMPORTANT: Using HOLDOUT set for out-of-sample evaluation")
    print(f"   Dev set (1,121 samples) was used for hyperparameter tuning")
    print(f"   Holdout set (750 samples) provides unbiased performance metrics")
    print("="*80)
    print(f"Gamma: {args.gamma}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output: {output_dir}")
    
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
    
    # Load HOLDOUT data
    print("\n📊 Loading HOLDOUT evaluation data...")
    data = load_data(Path(CANONICAL_HOLDOUT_DATA_PATH))
    print(f"   Loaded {len(data)} samples from HOLDOUT set")
    
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

