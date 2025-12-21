#!/usr/bin/env python3
"""
Train Disjoint Priors with Dense Data + PCA d=16

The "Goldilocks" solution:
- Disjoint A matrices: Prevents "herd suppression" of specialists
- Dense training: ALL 40k interactions (failures teach too!)
- PCA d=16: 497 samples/model ÷ 256 params = 1.9 samples/param

Key insight: When 80 models fail a math problem, that's VALUABLE
data for EACH of those 80 models' individual uncertainty matrices.
The one specialist (DeepSeek-Math) maintains its exploration bonus
because it has its own A matrix.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from banditgpt.core.bandit_router import DisjointLinUCBPolicy
from banditgpt._resources import get_priors_path


def load_dense_data():
    """Load ALL evaluations (not just winners)."""
    # Load prompts
    prompts_path = get_priors_path("archetype_grid_prompts.jsonl")
    cluster_ids = []
    with open(prompts_path) as f:
        for line in f:
            cluster_ids.append(json.loads(line)["cluster_id"])
    
    # Load DENSE rewards
    rewards_path = get_priors_path("archetype_grid_dense_run.jsonl")
    
    rewards = {}  # rewards[cluster][model] = reward
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
    print(f"  Dense evaluations: {n_dense}")
    print(f"  Avg per model: {n_dense / len(model_names):.0f}")
    
    return cluster_ids, rewards, model_names


def train_disjoint_dense(embeddings_pca, cluster_ids, rewards, model_names, epochs=3):
    """
    Train disjoint policies with dense data.
    
    Critical difference from winner-only:
    - Winner-only: 5 updates per model (just winners)
    - Dense: ~497 updates per model (ALL interactions, including failures)
    
    Why failures matter: When GPT-3.5 fails a calculus problem, we learn
    "GPT-3.5's A matrix should mark this region as explored but low-reward"
    """
    dim = embeddings_pca.shape[1]
    policy = DisjointLinUCBPolicy(model_names=model_names, dim=dim, alpha=0.5)
    
    rng = np.random.default_rng(42)
    n_prompts = len(cluster_ids)
    
    # Track updates per model
    updates_per_model = {m: 0 for m in model_names}
    
    print(f"  Training disjoint priors (epochs={epochs})...")
    for epoch in range(epochs):
        perm = rng.permutation(n_prompts)
        
        for idx in perm:
            embedding = embeddings_pca[idx]
            cluster = cluster_ids[idx]
            
            if cluster not in rewards:
                continue
            
            # DENSE TRAINING: Update with ALL model evaluations
            for model in model_names:
                if model in rewards[cluster]:
                    reward = rewards[cluster][model]
                    policy.update(model, embedding, reward)
                    updates_per_model[model] += 1
        
        if (epoch + 1) % 1 == 0:
            total = sum(updates_per_model.values())
            print(f"    Epoch {epoch+1}/{epochs}: {total:,} updates")
    
    # Report distribution
    min_updates = min(updates_per_model.values())
    max_updates = max(updates_per_model.values())
    avg_updates = np.mean(list(updates_per_model.values()))
    
    print(f"  ✓ Training complete")
    print(f"  Updates per model: min={min_updates}, avg={avg_updates:.0f}, max={max_updates}")
    
    return policy


def main():
    print("=" * 70)
    print("Training Disjoint Priors with Dense Data (d=16)")
    print("=" * 70)
    print("The Goldilocks Solution:")
    print("  ✓ Disjoint: Each model keeps its own uncertainty")
    print("  ✓ Dense: Learn from ALL interactions (including failures)")
    print("  ✓ Low-rank: d=16 PCA for 1.9 samples/param ratio")
    print("=" * 70)
    print()
    
    # Load dense data
    print("[1/4] Loading dense evaluation data...")
    cluster_ids, rewards, model_names = load_dense_data()
    print()
    
    # Load embeddings
    print("[2/4] Loading/computing embeddings...")
    full_emb_cache = get_priors_path("full_embeddings_384.npy")
    
    if full_emb_cache.exists():
        embeddings_full = np.load(full_emb_cache)
    else:
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
    
    # Save PCA
    pca_path = get_priors_path("pca_d16_for_disjoint.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  ✓ Saved PCA model")
    
    np.save(get_priors_path("full_embeddings_pca16_disjoint.npy"), embeddings_pca)
    print()
    
    # Train disjoint policy with dense data
    print("[4/4] Training disjoint priors with DENSE data...")
    policy = train_disjoint_dense(embeddings_pca, cluster_ids, rewards, model_names, epochs=3)
    print()
    
    # Save priors
    print("Saving priors...")
    output_path = get_priors_path("disjoint_priors_dense_d16.npz")
    
    A_stack = np.array([policy.A[m] for m in model_names])
    b_stack = np.array([policy.b[m] for m in model_names])
    
    np.savez(
        output_path,
        model_names=np.array(model_names, dtype=object),
        dim=16,
        A_stack=A_stack,
        b_stack=b_stack,
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
    
    # Calculate actual dense samples per model
    samples_per_model = {}
    for cluster in cluster_ids:
        if cluster in rewards:
            for model in model_names:
                if model in rewards[cluster]:
                    samples_per_model[model] = samples_per_model.get(model, 0) + 1
    
    avg_samples = np.mean(list(samples_per_model.values()))
    
    params_per_model = 16 * 16 + 16  # A + b
    total_params = n_models * params_per_model
    
    print(f"Per-model parameters: {params_per_model} (256 for A, 16 for b)")
    print(f"Total parameters: {total_params:,} ({n_models} models)")
    print(f"Training samples per model: ~{avg_samples:.0f}")
    print(f"Samples per param (per model): {avg_samples / params_per_model:.2f}")
    print()
    
    print("Why this works:")
    print(f"  ✓ Each model gets ~{avg_samples:.0f} training examples")
    print(f"  ✓ Each model has {params_per_model} parameters")
    print(f"  ✓ Ratio: {avg_samples / params_per_model:.2f} samples/param (> 1.0 threshold)")
    print(f"  ✓ Disjoint: Specialists maintain exploration bonus")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())

