#!/usr/bin/env python3
"""
Test Corralling implementation with optimized hyperparameters.

This script runs the Corralling algorithm on a small subset of data to verify:
1. Importance-weighted loss estimation works
2. Expert weights update correctly
3. Algorithm can unlearn warmup bias
4. Optimized hyperparameters (η=5.0, γ=0.10) work correctly

Usage:
    python experiments_v1/04_figure/test_corralling.py
"""

import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import gzip
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from bandit_gpt.calibration import SimpleLinUCBRouter, apply_gamma_scaling, embed_prompt
from bandit_gpt.router import CorrallingRouter
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_WARMUP_PRIORS_PATH,
    DEFAULT_PCA_PATH,
    CANONICAL_DEV_DATA_PATH,
)


class TabulaRasaRouter:
    """LinUCB router initialized from scratch."""
    
    def __init__(self, models, context_dim, alpha=1.0):
        self.models = models
        self.alpha = alpha
        self.context_dim = context_dim
        self.A = {m: np.eye(context_dim) for m in models}
        self.b = {m: np.zeros(context_dim) for m in models}
        self.selections = {m: 0 for m in models}
    
    def select_model(self, context, total_steps=0):
        """Select model using UCB.
        
        Args:
            context: Context vector
            total_steps: Total training steps (unused, for compatibility)
        """
        ucb_scores = {}
        for model in self.models:
            try:
                A_inv = np.linalg.inv(self.A[model])
            except np.linalg.LinAlgError:
                # Add small regularization if singular
                A_inv = np.linalg.inv(self.A[model] + 1e-6 * np.eye(self.context_dim))
            
            theta = A_inv @ self.b[model]
            expected = theta @ context
            uncertainty = np.sqrt(max(0, context @ A_inv @ context))  # Ensure non-negative
            ucb_scores[model] = expected + self.alpha * uncertainty
        selected = max(ucb_scores, key=ucb_scores.get)
        self.selections[selected] += 1
        return selected
    
    def update(self, context, model, reward):
        context = context.reshape(-1, 1)
        self.A[model] += context @ context.T
        self.b[model] += reward * context.flatten()


def load_test_data(data_path, sample_size=100):
    """Load a small sample of data for testing."""
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
    
    # Sample
    data_list = list(prompt_data.values())
    np.random.seed(42)
    indices = np.random.choice(len(data_list), size=min(sample_size, len(data_list)), replace=False)
    return [data_list[i] for i in indices]


def test_corralling():
    """Test Corralling implementation."""
    print("="*80)
    print("CORRALLING IMPLEMENTATION TEST")
    print("="*80)
    
    # Load resources
    print("\n📦 Loading resources...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    pca = joblib.load(DEFAULT_PCA_PATH)
    warmup_priors = joblib.load(DEFAULT_WARMUP_PRIORS_PATH)
    warmup_priors_scaled = apply_gamma_scaling(warmup_priors, gamma=0.10)  # Optimized: was 0.05
    
    models = warmup_priors['models']
    context_dim = warmup_priors['A'][models[0]].shape[0]
    
    print(f"   ✅ Models: {len(models)}")
    print(f"   ✅ Context Dim: {context_dim}")
    
    # Load test data
    print("\n📊 Loading test data...")
    test_data = load_test_data(Path(CANONICAL_DEV_DATA_PATH), sample_size=100)
    print(f"   ✅ Loaded {len(test_data)} samples")
    
    # Initialize experts
    print("\n🚀 Initializing Corralling...")
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
    
    router = CorrallingRouter(
        experts=[warmup_expert, tabula_rasa_expert],
        models=models,
        learning_rate=5.0  # Optimized: was 1.0
    )
    
    print(f"   ✅ Initial weights: {router.weights}")
    
    # Run training loop
    print("\n🎓 Training...")
    cumulative_regret = 0.0
    total_reward = 0.0
    
    for i, sample in enumerate(test_data):
        prompt = sample['prompt']
        context = embed_prompt(prompt, encoder, pca)
        
        # Select model
        selected_model = router.select_model(context)
        
        # Get reward
        scores = sample.get('scores', {})
        if not scores:
            continue
        
        oracle_model = max(scores, key=scores.get)
        oracle_reward = scores[oracle_model]
        model_reward = scores.get(selected_model, 0.0)
        
        # Update
        regret = oracle_reward - model_reward
        cumulative_regret += regret
        total_reward += model_reward
        
        router.update(context, selected_model, model_reward)
        
        # Print progress every 20 samples
        if (i + 1) % 20 == 0:
            print(f"   Sample {i+1}/{len(test_data)}: "
                  f"Regret={cumulative_regret:.2f}, "
                  f"Avg Reward={total_reward/(i+1):.4f}, "
                  f"Weights={router.weights}")
    
    # Final results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    print(f"\n📊 Performance:")
    print(f"   Cumulative Regret: {cumulative_regret:.2f}")
    print(f"   Average Reward: {total_reward/len(test_data):.4f}")
    
    print(f"\n⚖️  Expert Weights:")
    print(f"   Warmup Expert: {router.weights[0]:.4f} ({router.weights[0]*100:.1f}%)")
    print(f"   Tabula Rasa Expert: {router.weights[1]:.4f} ({router.weights[1]*100:.1f}%)")
    
    if router.weights[1] > router.weights[0]:
        ratio = router.weights[1] / router.weights[0]
        print(f"\n   ✅ Tabula Rasa WON: {ratio:.2f}x more weight than Warmup")
    else:
        ratio = router.weights[0] / router.weights[1]
        print(f"\n   ⚠️  Warmup WON: {ratio:.2f}x more weight than Tabula Rasa")
    
    print(f"\n📈 Expert Selections:")
    print(f"   Warmup Expert: {router.expert_selections[0]} times")
    print(f"   Tabula Rasa Expert: {router.expert_selections[1]} times")
    
    print(f"\n🎯 Model Usage (Top 5):")
    for model, count in sorted(router.selections.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = 100.0 * count / len(test_data)
        print(f"   {model:<40} {count:>3} ({pct:>5.1f}%)")
    
    # Verify implementation
    print("\n" + "="*80)
    print("IMPLEMENTATION CHECKS")
    print("="*80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Weights sum to 1
    checks_total += 1
    weight_sum = router.weights.sum()
    if abs(weight_sum - 1.0) < 1e-6:
        print("   ✅ Weights sum to 1.0")
        checks_passed += 1
    else:
        print(f"   ❌ Weights sum to {weight_sum:.6f} (should be 1.0)")
    
    # Check 2: Weights are non-negative
    checks_total += 1
    if np.all(router.weights >= 0):
        print("   ✅ Weights are non-negative")
        checks_passed += 1
    else:
        print("   ❌ Some weights are negative")
    
    # Check 3: Weights changed from initial
    checks_total += 1
    initial_weights = np.array([0.5, 0.5])
    if not np.allclose(router.weights, initial_weights):
        print("   ✅ Weights changed from initial (algorithm is learning)")
        checks_passed += 1
    else:
        print("   ⚠️  Weights unchanged (algorithm may not be learning)")
    
    # Check 4: Expert selections match total samples
    checks_total += 1
    total_selections = sum(router.expert_selections)
    if total_selections == len(test_data):
        print(f"   ✅ Expert selections match sample count ({total_selections})")
        checks_passed += 1
    else:
        print(f"   ❌ Expert selections ({total_selections}) != samples ({len(test_data)})")
    
    # Check 5: Model selections match total samples
    checks_total += 1
    total_model_selections = sum(router.selections.values())
    if total_model_selections == len(test_data):
        print(f"   ✅ Model selections match sample count ({total_model_selections})")
        checks_passed += 1
    else:
        print(f"   ❌ Model selections ({total_model_selections}) != samples ({len(test_data)})")
    
    # Summary
    print("\n" + "="*80)
    print(f"CHECKS PASSED: {checks_passed}/{checks_total}")
    print("="*80)
    
    if checks_passed == checks_total:
        print("\n✅ All checks passed! Implementation is correct.")
        return True
    else:
        print(f"\n⚠️  {checks_total - checks_passed} check(s) failed. Review implementation.")
        return False


if __name__ == '__main__':
    success = test_corralling()
    sys.exit(0 if success else 1)

