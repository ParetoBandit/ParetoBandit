#!/usr/bin/env python3
"""
Diagnose: Test different prior strengths for shared covariance

Hypothesis: λ=5 is too high, causing over-confident wrong predictions.
Test: λ = 0.5, 1, 2, 3, 5, 10
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from shared_covariance_policy import SharedCovarianceLinUCBPolicy
from banditgpt._resources import get_priors_path
from run_5fold_shared_dense import load_full_dataset, train_shared_dense, evaluate_fold


def main():
    print("=" * 70)
    print("Diagnosing Shared Covariance: Prior Strength Sweep")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    all_embeddings, all_clusters, all_rewards, model_names = load_full_dataset()
    
    # Use fold 1 for quick test
    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(len(all_clusters))
    all_embeddings = all_embeddings[perm]
    all_clusters = [all_clusters[i] for i in perm]
    
    # Split
    n_test = 99
    train_embeddings = all_embeddings[n_test:]
    train_clusters = all_clusters[n_test:]
    test_embeddings = all_embeddings[:n_test]
    test_clusters = all_clusters[:n_test]
    
    # PCA
    print("Fitting PCA...")
    pca = PCA(n_components=16, random_state=42)
    train_pca = pca.fit_transform(train_embeddings)
    test_pca = pca.transform(test_embeddings)
    
    # Train priors
    print("Training shared policy...")
    policy_base = train_shared_dense(train_pca, train_clusters, all_rewards, model_names, epochs=3, seed=42)
    print()
    
    # Test different strengths
    strengths = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    print("Testing prior strengths...")
    print(f"{'λ':<8} {'Cold':<12} {'Warm':<12} {'Reduction':<12} {'Status'}")
    print("-" * 70)
    
    results = []
    for strength in strengths:
        # Make a copy of the policy
        import copy
        policy = copy.deepcopy(policy_base)
        
        cold, warm, reduction = evaluate_fold(
            policy,
            test_pca,
            test_clusters,
            all_rewards,
            model_names,
            strength,
            seed=42,
        )
        
        status = "✓" if reduction > 0 else "✗"
        print(f"{strength:<8.1f} {cold:<12.1f} {warm:<12.1f} {reduction:>+10.1f}%  {status}")
        
        results.append((strength, reduction))
    
    print("=" * 70)
    
    best_strength, best_reduction = max(results, key=lambda x: x[1])
    print(f"\nBest: λ={best_strength} → {best_reduction:+.1f}%")
    
    if best_reduction < 0:
        print("\n⚠️  ALL STRENGTHS ARE NEGATIVE!")
        print("This suggests a fundamental issue with the shared covariance approach.")
        print("\nPossible causes:")
        print("1. d=16 loses too much signal (only 28.5% variance)")
        print("2. Shared A forces wrong uncertainty structure")
        print("3. Dense training on low-quality models adds noise")


if __name__ == "__main__":
    sys.exit(main())

