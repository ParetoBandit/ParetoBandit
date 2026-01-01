#!/usr/bin/env python3
"""
Generate PCA-based priors for BanditGPT (Reproducible)

This script generates the official PCA-based prior covariance matrix
used by the BanditRouter. It ensures reproducibility by:
- Using fixed random seeds
- Excluding train/test sets (zero data leakage)
- Generating deterministic PCA transformation

Output: banditgpt/priors/priors_meta_pca.npz (54KB)
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from banditgpt.data.scripts.pca_manager import train_pca_pipeline

def main():
    # Paths
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "banditgpt" / "data"
    priors_dir = repo_root / "banditgpt" / "priors"
    
    source_prompts = data_dir / "lmsys_all_prompts_clustered.jsonl"
    exclusion_paths = [
        data_dir / "test_prompts.jsonl",
        data_dir / "train_prompts.jsonl"
    ]
    
    print("=" * 70)
    print("GENERATING PCA-BASED PRIORS FOR BANDITGPT")
    print("=" * 70)
    print(f"\nSource: {source_prompts}")
    print(f"Exclusions: {len(exclusion_paths)} files (train/test sets)")
    print(f"Output: {priors_dir}/")
    print("\n" + "-" * 70)
    
    # Run pipeline
    train_pca_pipeline(
        source_prompts_path=source_prompts,
        exclusion_paths=exclusion_paths,
        output_dir=priors_dir,
        n_components=32,
        max_prompts=25000
    )
    
    print("\n" + "=" * 70)
    print("✓ PCA PRIORS GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - pca_32.joblib (51KB) - PCA transformation model")
    print(f"  - priors_meta_pca.npz (54KB) - Prior covariance matrix")
    print(f"\nThese files enable:")
    print(f"  • Dimensionality reduction: 384D → 32D")
    print(f"  • Hybrid features: 32 PCA + 8 handcrafted + 5 cluster = 45D")
    print(f"  • Fast, lightweight routing with strong priors")

if __name__ == "__main__":
    main()
