#!/usr/bin/env python3
"""
Apply PCA Dimensionality Reduction to Fix Overfitting

Problem: 384 dimensions × 81 models = ~147k parameters per model
         With only ~5 samples per model, this causes severe overfitting.

Solution: Reduce to d=32 dimensions via PCA
          → ~1k parameters per model
          → Data-to-parameter ratio improves from 1:300 to 1:2

This is a principled fix that preserves the "difficulty signal" while
removing irrelevant semantic nuances.

Usage:
    python experiments/apply_pca_reduction.py --n-components 32

Output:
    - banditgpt/data/priors/pca_model_d32.pkl (fitted PCA)
    - banditgpt/data/priors/train_embeddings_pca32.npy
    - banditgpt/data/priors/test_embeddings_pca32.npy
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt._resources import get_priors_path
from banditgpt.core.bandit_router import DEFAULT_CONTEXT_MODEL


def load_and_embed_prompts(prompts_path: Path, cache_path: Path = None):
    """Load prompts and embed them."""
    # Load prompts
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    
    # Check cache first
    if cache_path and cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(prompts):
            print(f"   Loaded cached embeddings: {embeddings.shape}")
            return np.asarray(embeddings, dtype=np.float64), prompts
    
    # Compute embeddings
    print(f"   Embedding {len(prompts)} prompts...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    
    # Cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"   Cached embeddings to {cache_path.name}")
    
    return embeddings, prompts


def main():
    parser = argparse.ArgumentParser(description="Apply PCA to reduce embedding dimensions")
    parser.add_argument("--n-components", type=int, default=32,
                        help="Number of PCA components to keep (default: 32)")
    args = parser.parse_args()
    
    n_components = args.n_components
    
    print("=" * 70)
    print("PCA Dimensionality Reduction for Routing")
    print("=" * 70)
    print(f"Target dimensions: {n_components}")
    print("=" * 70)
    print()
    
    # Paths
    train_prompts = get_priors_path("train_archetypes.jsonl")
    test_prompts = get_priors_path("test_archetypes.jsonl")
    
    train_emb_cache = get_priors_path("train_embeddings_full.npy")
    test_emb_cache = get_priors_path("test_embeddings_full.npy")
    
    pca_model_path = get_priors_path(f"pca_model_d{n_components}.pkl")
    train_pca_path = get_priors_path(f"train_embeddings_pca{n_components}.npy")
    test_pca_path = get_priors_path(f"test_embeddings_pca{n_components}.npy")
    
    # Load embeddings
    print("[1/4] Loading training embeddings...")
    train_emb, train_prompts_list = load_and_embed_prompts(train_prompts, train_emb_cache)
    print(f"   Shape: {train_emb.shape}")
    
    print("\n[2/4] Fitting PCA on training data...")
    pca = PCA(n_components=n_components, random_state=42)
    train_emb_pca = pca.fit_transform(train_emb)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"   ✓ PCA fitted")
    print(f"   Explained variance: {explained_var:.1%}")
    print(f"   Reduced: {train_emb.shape[1]} → {train_emb_pca.shape[1]} dims")
    
    # Parameter reduction
    orig_params = train_emb.shape[1] * train_emb.shape[1]  # A matrix size
    new_params = n_components * n_components
    reduction = orig_params / new_params
    print(f"   Parameters per model: {orig_params:,} → {new_params:,} ({reduction:.0f}x reduction)")
    
    # Data-to-parameter ratio
    n_train = train_emb.shape[0]
    orig_ratio = n_train / orig_params
    new_ratio = n_train / new_params
    print(f"   Data-to-parameter ratio: {orig_ratio:.4f} → {new_ratio:.2f}")
    
    # Save PCA model
    with open(pca_model_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"   ✓ Saved PCA model to {pca_model_path.name}")
    
    # Save reduced training embeddings
    np.save(train_pca_path, train_emb_pca)
    print(f"   ✓ Saved reduced train embeddings to {train_pca_path.name}")
    
    print("\n[3/4] Transforming test embeddings...")
    test_emb, test_prompts_list = load_and_embed_prompts(test_prompts, test_emb_cache)
    test_emb_pca = pca.transform(test_emb)
    
    np.save(test_pca_path, test_emb_pca)
    print(f"   ✓ Saved reduced test embeddings to {test_pca_path.name}")
    print(f"   Shape: {test_emb_pca.shape}")
    
    print("\n[4/4] Analysis of PCA components...")
    # Show how much variance each component captures
    print("   Top 5 components:")
    for i in range(min(5, n_components)):
        print(f"      PC{i+1}: {pca.explained_variance_ratio_[i]:.1%}")
    
    print("\n" + "=" * 70)
    print("PCA Reduction Complete!")
    print("=" * 70)
    print(f"✓ Reduced from 384 → {n_components} dimensions")
    print(f"✓ Explained variance: {explained_var:.1%}")
    print(f"✓ Parameters reduced {reduction:.0f}x")
    print(f"✓ Data-to-parameter ratio improved from {orig_ratio:.4f} to {new_ratio:.2f}")
    print()
    print("Next Steps:")
    print("1. Regenerate priors with PCA-reduced embeddings:")
    print("   python experiments/generate_expert_priors_pca.py")
    print()
    print("2. Test on held-out data:")
    print("   python experiments/run_rq1_pca.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

