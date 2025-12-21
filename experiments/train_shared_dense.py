#!/usr/bin/env python3
"""
Train Shared Covariance Policy on DENSE Evaluations

Key improvements:
1. Shared A matrix (1 matrix instead of 81) → 144x parameter reduction
2. Dense training (all 81 models graded per prompt) → 81x more data
3. PCA d=16 (captures core routing factors) → further reduction

Data efficiency:
- Old: 497 samples / 12M params = 0.00004
- New: 40,257 samples / 1,552 params = 25.9 samples/param ✓
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from shared_covariance_policy import SharedCovarianceLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_dense_data():
    """Load ALL evaluations (not just winners)."""
    # Load prompts
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
    # Load DENSE rewards (all model evaluations)
    rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    
    # Organize as: rewards[cluster][model] = reward
    rewards = {}
    models = set()
    
    with open(rewards_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("ok", False):
                model = data["model_id"]
                cluster = data["cluster_id"]
                logit = data.get("reward_logit", 0.0)
                reward = 1.0 / (1.0 + np.exp(-logit))
                
                if cluster not in rewards:
                    rewards[cluster] = {}
                rewards[cluster][model] = reward
                models.add(model)
    
    model_names = sorted(models)
    
    n_dense = sum(len(rewards[c]) for c in cluster_ids if c in rewards)
    
    print(f"  Prompts: {len(cluster_ids)}")
    print(f"  Models: {len(model_names)}")
    print(f"  Dense evaluations: {n_dense} (vs {len(cluster_ids)} winner-only)")
    print(f"  Data increase: {n_dense / len(cluster_ids):.1f}x")
    
    return cluster_ids, rewards, model_names


def train_shared_policy(embeddings_pca, cluster_ids, rewards, model_names, epochs=3):
    """
    Train shared covariance policy on DENSE data.
    
    Args:
        embeddings_pca: PCA-reduced embeddings
        cluster_ids: Cluster ID for each prompt
        rewards: Dense reward dict[cluster][model] = reward
        model_names: List of all models
        epochs: Number of training epochs (fewer needed with dense data)
    
    Returns:
        Trained SharedCovarianceLinUCBPolicy
    """
    dim = embeddings_pca.shape[1]
    policy = SharedCovarianceLinUCBPolicy(model_names, dim, alpha=0.5)
    
    rng = np.random.default_rng(42)
    n_prompts = len(cluster_ids)
    
    total_updates = 0
    
    print(f"  Training with dense updates (epochs={epochs})...")
    for epoch in range(epochs):
        perm = rng.permutation(n_prompts)
        
        for idx in perm:
            embedding = embeddings_pca[idx]
            cluster = cluster_ids[idx]
            
            if cluster not in rewards:
                continue
            
            # Update with ALL model evaluations (dense training)
            for model in model_names:
                if model in rewards[cluster]:
                    reward = rewards[cluster][model]
                    policy.update(model, embedding, reward)
                    total_updates += 1
        
        if (epoch + 1) % 1 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: {total_updates} updates")
    
    print(f"  ✓ Training complete: {total_updates} total updates")
    print(f"  ✓ Shared A updated {total_updates} times")
    print(f"  ✓ Each model's b updated ~{total_updates // len(model_names)} times")
    
    return policy


def main():
    print("=" * 70)
    print("Training Shared Covariance Policy with Dense Data")
    print("=" * 70)
    print("Configuration:")
    print("  PCA dimensions: d=16 (aggressive reduction)")
    print("  Policy: SharedCovarianceLinUCB")
    print("  Training: Dense (all models per prompt)")
    print("  Epochs: 3 (dense data converges faster)")
    print("=" * 70)
    print()
    
    # Load dense data
    print("[1/4] Loading dense evaluation data...")
    cluster_ids, rewards, model_names = load_dense_data()
    print()
    
    # Load or compute embeddings
    print("[2/4] Loading/computing embeddings...")
    full_emb_cache = get_priors_path("full_embeddings_384.npy")
    
    if full_emb_cache.exists():
        print(f"  Loading cached embeddings...")
        embeddings_full = np.load(full_emb_cache)
    else:
        print(f"  Computing embeddings...")
        from sentence_transformers import SentenceTransformer
        from banditgpt.core.bandit_router import DEFAULT_CONTEXT_MODEL
        
        prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
        prompts = []
        with open(prompts_path) as f:
            for line in f:
                prompts.append(json.loads(line)["prompt"])
        
        encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
        embeddings_full = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
        embeddings_full = np.asarray(embeddings_full, dtype=np.float64)
        np.save(full_emb_cache, embeddings_full)
    
    print(f"  Full embeddings: {embeddings_full.shape}")
    print()
    
    # Apply PCA d=16
    print("[3/4] Applying PCA (d=16)...")
    pca = PCA(n_components=16, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_full)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"  Reduced: {embeddings_full.shape[1]} → {embeddings_pca.shape[1]} dims")
    print(f"  Explained variance: {explained_var:.1%}")
    
    # Save PCA and embeddings
    pca_path = get_priors_path("pca_model_d16.pkl")
    import pickle
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  ✓ Saved PCA model to {pca_path.name}")
    
    np.save(get_priors_path("full_embeddings_pca16.npy"), embeddings_pca)
    print(f"  ✓ Saved PCA embeddings")
    print()
    
    # Train policy
    print("[4/4] Training shared covariance policy...")
    policy = train_shared_policy(embeddings_pca, cluster_ids, rewards, model_names, epochs=3)
    print()
    
    # Save priors
    print("Saving priors...")
    output_path = get_priors_path("shared_priors_dense_d16.npz")
    
    np.savez(
        output_path,
        model_names=np.array(model_names, dtype=object),
        dim=16,
        A_shared=policy.A_shared,
        b_dict_keys=np.array(list(policy.b.keys()), dtype=object),
        b_dict_values=np.array(list(policy.b.values())),
        counts=np.array([policy.counts[m] for m in model_names]),
        total_updates=policy.total_updates,
    )
    
    print(f"  ✓ Saved to {output_path.name}")
    print(f"  ✓ Size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    
    # Parameter analysis
    print("=" * 70)
    print("PARAMETER EFFICIENCY")
    print("=" * 70)
    
    n_prompts = len(cluster_ids)
    n_models = len(model_names)
    n_dense = policy.total_updates
    
    params_A = 16 * 16
    params_b = n_models * 16
    total_params = params_A + params_b
    
    print(f"Shared A matrix: {params_A} parameters")
    print(f"Model b vectors: {params_b} parameters ({n_models} × 16)")
    print(f"Total parameters: {total_params}")
    print(f"Training samples: {n_dense}")
    print(f"Samples per parameter: {n_dense / total_params:.1f}")
    print()
    
    print("Comparison:")
    print(f"  Old (Disjoint, d=384): {n_models * 384 * 384:,} params, ~{n_prompts} samples")
    print(f"  Old ratio: {n_prompts / (n_models * 384 * 384):.6f} samples/param")
    print(f"  New (Shared, d=16): {total_params:,} params, {n_dense:,} samples")
    print(f"  New ratio: {n_dense / total_params:.1f} samples/param")
    print(f"  Improvement: {(n_dense / total_params) / (n_prompts / (n_models * 384 * 384)):.0f}x better!")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

