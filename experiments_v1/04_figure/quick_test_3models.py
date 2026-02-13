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
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=0.05)
    
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
    print(f"   Models in priors: {warmup_priors_scaled['models']}")
    
    # Extend priors for GPT-4o via semantic transfer
    missing_models = [m for m in all_models_in_data if m not in warmup_priors_scaled['models']]
    if missing_models:
        print(f"\n🔄 Extending priors for: {missing_models}")
        transfer_mapping = {'openai/gpt-4-turbo': 'openai/gpt-4-turbo'}
        warmup_priors_scaled = extend_priors_with_semantic_transfer(
            warmup_priors_scaled,
            new_models=missing_models,
            transfer_mapping=transfer_mapping,
            gamma=0.05
        )
    
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
        
        selected_model = router.select_model(context)
        model_reward, oracle_reward = compute_oracle_reward(sample, selected_model)
        
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        # Track history for convergence analysis
        regret_history.append(cumulative_regret)
        reward_history.append(total_reward / (i + 1))
        expert_weights_history.append(router.weights.copy())
        
        router.update(context, selected_model, model_reward)
    
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
    
    if router.weights[1] > router.weights[0]:
        ratio = router.weights[1] / router.weights[0]
        print(f"\n   ✅ Tabula Rasa WON: {ratio:.2f}x")
    
    print(f"\n🎯 Model Usage (All {len(models)} models):")
    for model, count in sorted(router.selections.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * count / len(labeled_data)
        print(f"   {model:<45} {count:>4} ({pct:>5.1f}%)")
    
    # Save results
    output_dir = Path(__file__).parent / "results_3models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
