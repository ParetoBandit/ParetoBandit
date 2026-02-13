#!/usr/bin/env python3
"""
Validate PCA Transfer Quality Across Distributions

This script validates whether PCA trained on warmup distribution transfers
well to evaluation distribution by comparing:
1. Reconstruction error (warmup vs. evaluation)
2. Explained variance (warmup vs. evaluation)
3. Per-category reconstruction (if categories still tracked)

Scientific question: Do principal components optimized for warmup variance
structure adequately capture variance in evaluation data?
"""

import sys
import json
import gzip
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
from sentence_transformers import SentenceTransformer

# Paths
PCA_MODEL_PATH = PROJECT_ROOT / "src" / "artifacts" / "pca_32.joblib"
WARMUP_DATA = PROJECT_ROOT / "src" / "bandit_gpt" / "data" / "offline_dataset" / "routellm_battles_rewards.jsonl"
DEV_PROMPTS = PROJECT_ROOT / "data" / "dev_prompts_for_rejudge.jsonl"
HOLDOUT_PROMPTS = PROJECT_ROOT / "data" / "holdout_prompts_for_rejudge.jsonl"


def load_pca():
    """Load trained PCA model."""
    if not PCA_MODEL_PATH.exists():
        raise FileNotFoundError(f"PCA model not found at {PCA_MODEL_PATH}")
    
    pca = joblib.load(PCA_MODEL_PATH)
    print(f"✓ Loaded PCA model: {PCA_MODEL_PATH}")
    print(f"  - Components: {pca.n_components}")
    print(f"  - Input dim: {pca.n_features_in_}")
    return pca


def load_prompts(filepath: Path) -> List[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append(data['prompt'])
    return prompts


def compute_embeddings(prompts: List[str], batch_size: int = 32) -> np.ndarray:
    """Compute sentence embeddings for prompts."""
    print(f"  Computing embeddings for {len(prompts)} prompts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim model
    embeddings = model.encode(prompts, batch_size=batch_size, show_progress_bar=True)
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    return embeddings


def compute_reconstruction_error(pca, embeddings: np.ndarray) -> float:
    """Compute mean squared reconstruction error."""
    # Project to low-dim
    low_dim = pca.transform(embeddings)
    
    # Reconstruct to high-dim
    reconstructed = pca.inverse_transform(low_dim)
    
    # Compute MSE
    mse = np.mean((embeddings - reconstructed) ** 2)
    return mse


def compute_explained_variance(pca, embeddings: np.ndarray) -> float:
    """Compute explained variance ratio for given data."""
    # Project to low-dim and back to high-dim
    low_dim = pca.transform(embeddings)
    reconstructed = pca.inverse_transform(low_dim)
    
    # Center the data
    centered = embeddings - embeddings.mean(axis=0)
    centered_reconstructed = reconstructed - embeddings.mean(axis=0)
    
    # Total variance in original data (sum across all dimensions)
    total_var = np.sum(np.var(centered, axis=0))
    
    # Variance captured in reconstruction
    reconstruction_var = np.sum(np.var(centered_reconstructed, axis=0))
    
    # Explained variance ratio
    explained = reconstruction_var / total_var
    return explained


def main():
    print("=" * 70)
    print("PCA TRANSFER QUALITY VALIDATION")
    print("=" * 70)
    print("\nValidating whether PCA trained on warmup distribution")
    print("transfers well to evaluation distribution.")
    print()
    
    # Load PCA
    print("\n1. Loading PCA model...")
    pca = load_pca()
    
    # Load prompts
    print("\n2. Loading prompts...")
    warmup_prompts = load_prompts(WARMUP_DATA)
    dev_prompts = load_prompts(DEV_PROMPTS)
    holdout_prompts = load_prompts(HOLDOUT_PROMPTS)
    eval_prompts = dev_prompts + holdout_prompts
    
    print(f"  ✓ Warmup: {len(warmup_prompts)} prompts")
    print(f"  ✓ Dev: {len(dev_prompts)} prompts")
    print(f"  ✓ Holdout: {len(holdout_prompts)} prompts")
    print(f"  ✓ Evaluation total: {len(eval_prompts)} prompts")
    
    # Compute embeddings
    print("\n3. Computing embeddings...")
    print("\n  Warmup embeddings:")
    warmup_embeddings = compute_embeddings(warmup_prompts)
    
    print("\n  Evaluation embeddings:")
    eval_embeddings = compute_embeddings(eval_prompts)
    
    # Validation 1: Reconstruction Error
    print("\n4. Computing reconstruction error...")
    warmup_error = compute_reconstruction_error(pca, warmup_embeddings)
    eval_error = compute_reconstruction_error(pca, eval_embeddings)
    error_ratio = eval_error / warmup_error
    
    print(f"\n  Warmup MSE:     {warmup_error:.6f}")
    print(f"  Evaluation MSE: {eval_error:.6f}")
    print(f"  Ratio:          {error_ratio:.2f}x")
    
    # Interpret
    if error_ratio < 1.2:
        print(f"  ✓ GOOD: PCA transfers well (ratio < 1.2x)")
    elif error_ratio < 1.5:
        print(f"  ⚠️  MODERATE: PCA transfers reasonably (1.2x < ratio < 1.5x)")
    else:
        print(f"  ✗ POOR: PCA doesn't transfer well (ratio > 1.5x)")
    
    # Validation 2: Explained Variance
    print("\n5. Computing explained variance...")
    
    # Get explained variance from training (standard PCA metric)
    warmup_explained_train = pca.explained_variance_ratio_.sum()
    
    # Compute how much variance captured in evaluation data
    eval_explained = compute_explained_variance(pca, eval_embeddings)
    
    print(f"\n  Warmup (from training): {warmup_explained_train:.2%}")
    print(f"  Evaluation (computed):  {eval_explained:.2%}")
    print(f"  Difference:             {(warmup_explained_train - eval_explained):.2%}")
    
    if abs(warmup_explained_train - eval_explained) < 0.05:
        print(f"  ✓ GOOD: Similar explained variance (< 5% difference)")
    elif abs(warmup_explained_train - eval_explained) < 0.10:
        print(f"  ⚠️  MODERATE: Some variance loss (5-10% difference)")
    else:
        print(f"  ✗ POOR: Significant variance loss (> 10% difference)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n1. Reconstruction Error Ratio: {error_ratio:.2f}x")
    print(f"2. Explained Variance Drop: {(warmup_explained_train - eval_explained):.2%}")
    print()
    
    # Overall assessment
    if error_ratio < 1.2 and abs(warmup_explained_train - eval_explained) < 0.05:
        print("✓ OVERALL: PCA transfers well across distributions")
        print("  → Feature-space distribution shift is minor")
        print("  → Using PCA trained on warmup is justified")
    elif error_ratio < 1.5 and abs(warmup_explained_train - eval_explained) < 0.10:
        print("⚠️  OVERALL: PCA transfers reasonably")
        print("  → Some feature-space distribution shift")
        print("  → Using warmup PCA is acceptable but not optimal")
    else:
        print("✗ OVERALL: PCA transfer quality is concerning")
        print("  → Significant feature-space distribution shift")
        print("  → Recommend training PCA on combined warmup+eval data")
    
    print()
    print("=" * 70)
    print("\nResults saved to: experiments_v1/01_table/pca_validation_results.txt")
    
    # Save results
    results_file = PROJECT_ROOT / "experiments_v1" / "01_table" / "pca_validation_results.txt"
    with open(results_file, 'w') as f:
        f.write("PCA Transfer Quality Validation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: 2026-02-13\n\n")
        f.write(f"Reconstruction Error:\n")
        f.write(f"  Warmup MSE:     {warmup_error:.6f}\n")
        f.write(f"  Evaluation MSE: {eval_error:.6f}\n")
        f.write(f"  Ratio:          {error_ratio:.2f}x\n\n")
        f.write(f"Explained Variance:\n")
        f.write(f"  Warmup:     {warmup_explained_train:.2%}\n")
        f.write(f"  Evaluation: {eval_explained:.2%}\n")
        f.write(f"  Difference: {(warmup_explained_train - eval_explained):.2%}\n\n")
        
        if error_ratio < 1.2 and abs(warmup_explained_train - eval_explained) < 0.05:
            f.write("Overall: PCA transfers well\n")
        elif error_ratio < 1.5 and abs(warmup_explained_train - eval_explained) < 0.10:
            f.write("Overall: PCA transfers reasonably\n")
        else:
            f.write("Overall: PCA transfer quality is concerning\n")
    
    print(f"✓ Saved to {results_file}")


if __name__ == "__main__":
    main()
