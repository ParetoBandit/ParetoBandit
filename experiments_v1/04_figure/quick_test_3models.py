#!/usr/bin/env python3
"""
Quick test: Train Corralling on 3 models (no expensive projection step).
This verifies the 3-model setup works before running the full experiment.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    OFFLINE_DATASET_DIR,
)

# Import helper functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from corralled_semantic_analysis import (
    TabulaRasaRouter,
    load_labeled_data,
    compute_oracle_reward,
    extend_priors_with_semantic_transfer
)

CANONICAL_DEV_DATA_PATH = OFFLINE_DATASET_DIR / "dev_rewards_complete.jsonl.gz"


def main():
    print("="*80)
    print("QUICK TEST: 3-MODEL CORRALLING (Training Only)")
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
    
    # Check models in data
    all_models_in_data = set()
    for sample in labeled_data:
        all_models_in_data.update(sample['scores'].keys())
    all_models_in_data = sorted(all_models_in_data)
    
    print(f"\n📊 Models in data: {all_models_in_data}")
    print(f"   Models in priors: {warmup_priors['models']}")
    
    # Step 1: Extend priors for GPT-4o via semantic transfer BEFORE scaling
    # (avoids double-scaling the transferred model's priors)
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
    
    models = warmup_priors_scaled['models']
    context_dim = warmup_priors_scaled['A'][models[0]].shape[0]
    
    print(f"\n🎓 Training Corralling (3 models)...")
    print(f"   Models: {models}")
    print(f"   Context Dim: {context_dim}")
    
    # Initialize experts
    warmup_expert = SimpleLinUCBRouter(
        models=models,
        warmup_priors=warmup_priors_scaled,
        alpha=1.0
    )
    
    tabula_rasa_expert = TabulaRasaRouter(
        models=models,
        context_dim=context_dim,
        alpha=1.0
    )
    
    # Initialize Corralling
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=1.0
    )
    
    # Training loop with history tracking
    cumulative_regret = 0.0
    total_reward = 0.0
    regret_history = []
    reward_history = []
    expert_weights_history = []
    
    for i, sample in enumerate(tqdm(labeled_data, desc="   Training")):
        prompt = sample['prompt']
        context = embed_prompt(prompt, encoder, pca)
        
        # CorrallingRouter.select_model returns (model_id, selection_token)
        selected_model, selection_token = router.select_model(context)
        model_reward, oracle_reward = compute_oracle_reward(sample, selected_model)
        
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        # Track history for convergence analysis
        regret_history.append(cumulative_regret)
        reward_history.append(total_reward / (i + 1))
        expert_weights_history.append(router.weights.copy())
        
        # Pass selection_token so the meta-weight update is applied
        router.update(context, selected_model, model_reward, selection_token=selection_token)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n📊 Performance:")
    print(f"   Cumulative Regret: {cumulative_regret:.2f}")
    print(f"   Average Reward: {total_reward/len(labeled_data):.4f}")
    
    print(f"\n⚖️  Expert Weights:")
    print(f"   Warmup: {router.weights[0]:.4f} ({router.weights[0]*100:.1f}%)")
    print(f"   Tabula Rasa: {router.weights[1]:.4f} ({router.weights[1]*100:.1f}%)")
    
    if router.weights[0] > router.weights[1]:
        print(f"\n   ✅ Warmup expert dominates — semantic transfer worked")
    else:
        ratio = router.weights[1] / max(router.weights[0], 1e-8)
        print(f"\n   ✅ Tabula Rasa dominates ({ratio:.1f}x) — learned from scratch")
    
    print(f"\n🎯 Model Usage (All {len(models)} models):")
    for model, count in sorted(router.selections.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / len(labeled_data)
        print(f"   {model:<45} {count:>4} ({pct:>5.1f}%)")
    
    # Generate training plots
    output_dir = Path(__file__).parent / "results_3models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_arr = np.array(expert_weights_history)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Panel 1: Expert weight evolution
    ax1 = axes[0]
    ax1.plot(weights_arr[:, 0], label='Warmup Expert', linewidth=2.5, color='orange')
    ax1.plot(weights_arr[:, 1], label='Tabula Rasa Expert', linewidth=2.5, color='green')
    ax1.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Expert Weight', fontsize=13, fontweight='bold')
    ax1.set_title('Expert Weight Evolution\nDuring Model Discovery', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    final_warmup = weights_arr[-1, 0]
    final_tabula = weights_arr[-1, 1]
    ax1.annotate(
        f'Final Weights:\nWarmup: {final_warmup:.3f}\nTabula Rasa: {final_tabula:.3f}',
        xy=(len(weights_arr)-1, 0.5),
        xytext=(len(weights_arr)*0.5, 0.5),
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
        fontsize=11, fontweight='bold'
    )
    
    # Panel 2: Cumulative regret
    ax2 = axes[1]
    ax2.plot(regret_history, linewidth=2.5, color='#e74c3c')
    ax2.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Regret\n(Lower is Better)', fontsize=15, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average reward
    ax3 = axes[2]
    ax3.plot(reward_history, linewidth=2.5, color='#27ae60')
    ax3.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Average Reward', fontsize=13, fontweight='bold')
    ax3.set_title('Average Reward Convergence\n(Higher is Better)', fontsize=15, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    fig_path = output_dir / 'training_summary.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 Saved training plot: {fig_path}")
    
    # Also save to main results/ dir to replace stale figure
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_main = results_dir / 'figure4_training_summary.png'
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    ax1 = axes[0]
    ax1.plot(weights_arr[:, 0], label='Warmup Expert', linewidth=2.5, color='orange')
    ax1.plot(weights_arr[:, 1], label='Tabula Rasa Expert', linewidth=2.5, color='green')
    ax1.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Expert Weight', fontsize=13, fontweight='bold')
    ax1.set_title('Expert Weight Evolution\nDuring Model Discovery', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(fontsize=12, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    ax1.annotate(
        f'Final Weights:\nWarmup: {final_warmup:.3f}\nTabula Rasa: {final_tabula:.3f}',
        xy=(len(weights_arr)-1, 0.5),
        xytext=(len(weights_arr)*0.5, 0.5),
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
        fontsize=11, fontweight='bold'
    )
    
    ax2 = axes[1]
    ax2.plot(regret_history, linewidth=2.5, color='#e74c3c')
    ax2.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Regret', fontsize=13, fontweight='bold')
    ax2.set_title('Cumulative Regret\n(Lower is Better)', fontsize=15, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.plot(reward_history, linewidth=2.5, color='#27ae60')
    ax3.set_xlabel('Training Samples', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Average Reward', fontsize=13, fontweight='bold')
    ax3.set_title('Average Reward Convergence\n(Higher is Better)', fontsize=15, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(fig_main, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved to results/: {fig_main}")
    
    # Save results
    
    results = {
        'n_models': len(models),
        'models': models,
        'train_size': len(labeled_data),
        'cumulative_regret': float(cumulative_regret),
        'avg_reward': float(total_reward / len(labeled_data)),
        'final_expert_weights': router.weights.tolist(),
        'model_usage': router.selections,
        # Add history for convergence analysis
        'regret_history': regret_history,
        'reward_history': reward_history,
        'expert_weights_history': [w.tolist() for w in expert_weights_history]
    }
    
    with open(output_dir / 'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved results to: {output_dir}/quick_test_results.json")
    print("\n" + "="*80)
    print("✅ 3-MODEL TRAINING WORKS!")
    print("="*80)


if __name__ == '__main__':
    main()
