#!/usr/bin/env python3
"""
Empirical Validation of Complexity Score Normalization Bounds

Purpose: Validate the hardcoded bounds [-0.15, 0.25] used in router.py
by projecting real prompts onto the complexity vector and analyzing the
statistical distribution.

KDD Review Requirement: "You must run a histogram of the Complexity Vector
projection across the entire LMSYS dataset to find the true statistical
μ ± 3σ bounds. Hardcoding from 4 samples is 'hacking', not data science."
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from bandit_gpt.config_legacy import DEFAULT_SENTENCE_TRANSFORMER
import json
from typing import List, Dict
import matplotlib.pyplot as plt

# Paths
COMPLEXITY_VECTOR_PATH = Path("banditgpt/priors/complexity_vector.npz")
TEST_PROMPTS_PATH = Path("banditgpt/data/offline_dataset/test_prompts.jsonl")
TRAIN_PROMPTS_PATH = Path("banditgpt/data/offline_dataset/train_prompts.jsonl")

def load_prompts(path: Path, max_samples: int = 1000) -> List[str]:
    """Load prompts from JSONL file."""
    prompts = []
    if not path.exists():
        print(f"Warning: {path} not found")
        return prompts
    
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data = json.loads(line)
            prompt = data.get("prompt", data.get("text", ""))
            if prompt:
                prompts.append(prompt)
    return prompts

def main():
    print("=" * 70)
    print("COMPLEXITY SCORE NORMALIZATION VALIDATION")
    print("=" * 70)
    
    # Load complexity vector
    if not COMPLEXITY_VECTOR_PATH.exists():
        print(f"ERROR: {COMPLEXITY_VECTOR_PATH} not found")
        return
    
    data = np.load(COMPLEXITY_VECTOR_PATH)
    complexity_vector = data["complexity_vector"]
    print(f"✓ Loaded complexity vector (dim={len(complexity_vector)})")
    
    # Initialize encoder
    print("Loading sentence encoder...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print("✓ Encoder loaded")
    
    # Load prompts
    print("\nLoading prompts...")
    # CRITICAL: Use ONLY train prompts for calibration to avoid data leakage!
    # Test prompts should remain unseen for proper evaluation.
    train_prompts = load_prompts(TRAIN_PROMPTS_PATH, max_samples=1000)
    all_prompts = train_prompts
    
    print(f"✓ Loaded {len(all_prompts)} TRAIN prompts for calibration")

    
    if len(all_prompts) == 0:
        print("\nERROR: No prompts loaded. Using synthetic samples...")
        all_prompts = [
            "Hello, how are you?",
            "Tell me a joke",
            "Write python code to sort a list",
            "Compute the integral of x^2 * sin(x) dx",
            "Explain quantum mechanics",
            "What is the capital of France?",
            "Solve this differential equation: dy/dx = x^2 + y",
            "Write a poem about nature",
            "Debug this Python code: def foo(): return bar",
            "Derive the Schrödinger equation"
        ] * 100  # Repeat to get 1000 samples
    
    # Project prompts onto complexity vector
    print("\nProjecting prompts onto complexity vector...")
    projections = []
    
    for i, prompt in enumerate(all_prompts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_prompts)}")
        
        emb = encoder.encode(prompt, normalize_embeddings=True)
        projection = float(np.dot(emb, complexity_vector))
        projections.append(projection)
    
    projections = np.array(projections)
    
    # Statistical Analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    mean = np.mean(projections)
    std = np.std(projections)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_values = np.percentile(projections, percentiles)
    
    print(f"\nBasic Statistics:")
    print(f"  Mean:     {mean:7.4f}")
    print(f"  Std Dev:  {std:7.4f}")
    print(f"  Min:      {np.min(projections):7.4f}")
    print(f"  Max:      {np.max(projections):7.4f}")
    print(f"  Range:    {np.max(projections) - np.min(projections):7.4f}")
    
    print(f"\nPercentiles:")
    for p, val in zip(percentiles, p_values):
        print(f"  P{p:2d}:  {val:7.4f}")
    
    print(f"\nEmpirical Bounds (μ ± 3σ):")
    print(f"  Lower: {mean - 3*std:7.4f}")
    print(f"  Upper: {mean + 3*std:7.4f}")
    
    # Compare with hardcoded bounds
    HARDCODED_MIN = -0.15
    HARDCODED_MAX = 0.25
    
    print("\n" + "=" * 70)
    print("VALIDATION OF HARDCODED BOUNDS")
    print("=" * 70)
    
    print(f"\nCurrent hardcoded bounds: [{HARDCODED_MIN}, {HARDCODED_MAX}]")
    print(f"Range coverage: {HARDCODED_MAX - HARDCODED_MIN:.4f}")
    
    # Check coverage
    below_min = np.sum(projections < HARDCODED_MIN)
    above_max = np.sum(projections > HARDCODED_MAX)
    within_bounds = len(projections) - below_min - above_max
    
    print(f"\nCoverage Analysis:")
    print(f"  Within bounds: {within_bounds:4d} ({100*within_bounds/len(projections):.1f}%)")
    print(f"  Below min:     {below_min:4d} ({100*below_min/len(projections):.1f}%)")
    print(f"  Above max:     {above_max:4d} ({100*above_max/len(projections):.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    # Use P1 and P99 for conservative bounds
    recommended_min = p_values[0]  # P1
    recommended_max = p_values[-1]  # P99
    
    print(f"\nRecommended bounds (P1-P99):")
    print(f"  COMPLEXITY_MIN = {recommended_min:.4f}  # P1")
    print(f"  COMPLEXITY_MAX = {recommended_max:.4f}  # P99")
    
    print(f"\nAlternative (μ ± 2σ, ~95% coverage):")
    print(f"  COMPLEXITY_MIN = {mean - 2*std:.4f}")
    print(f"  COMPLEXITY_MAX = {mean + 2*std:.4f}")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(projections, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(HARDCODED_MIN, color='r', linestyle='--', label=f'Hardcoded Min ({HARDCODED_MIN})')
    plt.axvline(HARDCODED_MAX, color='r', linestyle='--', label=f'Hardcoded Max ({HARDCODED_MAX})')
    plt.axvline(mean, color='g', linestyle='-', label=f'Mean ({mean:.3f})')
    plt.axvline(mean - 2*std, color='orange', linestyle=':', label=f'μ - 2σ ({mean - 2*std:.3f})')
    plt.axvline(mean + 2*std, color='orange', linestyle=':', label=f'μ + 2σ ({mean + 2*std:.3f})')
    plt.xlabel('Complexity Score Projection')
    plt.ylabel('Frequency')
    plt.title('Distribution of Complexity Score Projections')
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_path = Path("complexity_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Histogram saved to: {output_path}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
