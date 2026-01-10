#!/usr/bin/env python3
"""
Analyze model selection performance by prompt complexity.

Buckets prompts into Easy/Medium/Hard (Frontier) based on router's
complexity vector projection, then analyzes which models were selected
and how well they performed in each bucket.
"""

import sys
import gzip
import json
import numpy as np
import math
from pathlib import Path
from collections import defaultdict, Counter

# Setup paths
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from bandit_gpt.router import BanditRouter

def compute_difficulty(router: BanditRouter, prompt: str) -> float:
    """
    Compute normalized difficulty using router's complexity vector.
    Returns difficulty score in [0.0, 1.0].
    """
    emb = router.encoder.encode(prompt, normalize_embeddings=True)
    raw_projection = float(np.dot(emb, router.complexity_vector))
    
   # Apply sigmoid normalization
    COMPLEXITY_MU = getattr(router, 'calibrated_complexity_mu', -0.0037)
    COMPLEXITY_SIGMA = getattr(router, 'calibrated_complexity_sigma', 0.0950)
    k = 1.0 / COMPLEXITY_SIGMA
    
    z_score = k * (raw_projection - COMPLEXITY_MU)
    difficulty_normalized = 1.0 / (1.0 + math.exp(-z_score))
    
    return difficulty_normalized


def classify_complexity(difficulty: float) -> str:
    """Classify difficulty into Easy/Medium/Hard buckets."""
    if difficulty < 0.33:
        return "Easy"
    elif difficulty < 0.67:
        return "Medium"
    else:
        return "Hard (Frontier)"


def main():
    print("=" * 70)
    print("COMPLEXITY BUCKETING ANALYSIS")
    print("=" * 70)
    print()
    
    # Initialize router to get complexity vector
    print("🔧 Initializing router for complexity calculation...")
    router = BanditRouter.create(model_registry=None, priors="hle")
    print()
    
    # Load test data
    data_path = repo_root / "src/bandit_gpt/data/offline_dataset/lmsys_test_final_rewards_1k_clean.jsonl.gz"
    
    print(f"📦 Loading test data from {data_path.name}...")
    
    # Group entries by prompt
    prompt_data = defaultdict(lambda: {"models": {}, "complexity": None})
    
    with gzip.open(data_path, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get('ok'):
                prompt = entry['prompt']
                model_id = entry['model_id']
                reward = entry['raw_score']
                
                prompt_data[prompt]["models"][model_id] = reward
    
    print(f"✓ Loaded {len(prompt_data)} unique prompts")
    print()
    
    # Compute complexity for each prompt
    print("🧮 Computing complexity scores...")
    complexity_buckets = {"Easy": [], "Medium": [], "Hard (Frontier)": []}
    
    for prompt, data in prompt_data.items():
        difficulty = compute_difficulty(router, prompt)
        bucket = classify_complexity(difficulty)
        
        data["complexity"] = difficulty
        data["bucket"] = bucket
        complexity_buckets[bucket].append(prompt)
    
    # Print bucket sizes
    print("\n📊 Prompt Distribution by Complexity:")
    for bucket in ["Easy", "Medium", "Hard (Frontier)"]:
        count = len(complexity_buckets[bucket])
        pct = 100 * count / len(prompt_data)
        print(f"  {bucket:20s}: {count:4d} prompts ({pct:5.1f}%)")
    
    # Analyze what models perform best in each bucket
    print("\n" + "=" * 70)
    print("TOP MODELS BY COMPLEXITY BUCKET")
    print("=" * 70)
    
    for bucket in ["Easy", "Medium", "Hard (Frontier)"]:
        print(f"\n{bucket}:")
        print("-" * 70)
        
        # Collect all rewards for each model in this bucket
        model_rewards = defaultdict(list)
        
        for prompt in complexity_buckets[bucket]:
            for model_id, reward in prompt_data[prompt]["models"].items():
                model_rewards[model_id].append(reward)
        
        # Calculate mean reward for each model
        model_stats = []
        for model, rewards in model_rewards.items():
            mean_reward = np.mean(rewards)
            count = len(rewards)
            model_stats.append((model, mean_reward, count))
        
        # Sort by mean reward
        model_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 5
        print(f"{'Model':45s} | {'Mean Reward':12s} | {'Count':6s}")
        print("-" * 70)
        for model, mean, count in model_stats[:5]:
            print(f"{model[:45]:45s} | {mean:12.4f} | {count:6d}")
    
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
