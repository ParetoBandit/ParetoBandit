#!/usr/bin/env python3
"""
Train PCA Model from RouteLLM Battle Data

This script:
1. Loads prompts from the RouteLLM battles dataset (80K prompts)
2. Embeds them using SentenceTransformer (all-MiniLM-L6-v2)
3. Trains a PCA model to reduce dimensionality (384 -> 23 components)
4. Saves the PCA model for use in routing

The PCA model is critical for:
- Reducing embedding size for faster LinUCB updates
- Capturing semantic structure in prompts
- Ensuring consistency between warmup and live routing

Usage:
    python3 scripts/train_pca_from_routellm.py
    
    # With custom settings:
    python3 scripts/train_pca_from_routellm.py \\
        --input <path-to-battles-file> \\
        --output src/artifacts/pca_23.joblib \\
        --n-components 23 \\
        --max-prompts 80000
"""

import sys
from pathlib import Path

# Add project root and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from bandit_gpt.config_legacy import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    ARTIFACTS_DIR,
)


def load_holdout_prompts(holdout_file: Path) -> set:
    """
    Load holdout prompts to exclude from PCA training (decontamination).

    Args:
        holdout_file: Path to holdout JSONL.gz file

    Returns:
        Set of holdout prompt strings
    """
    import gzip

    holdout_prompts = set()
    if not holdout_file.exists():
        print(f"   ⚠️  Holdout file not found: {holdout_file}")
        return holdout_prompts

    with gzip.open(holdout_file, 'rt') as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry.get('prompt', '').strip()
                if prompt:
                    holdout_prompts.add(prompt)
            except Exception:
                continue

    print(f"   Loaded {len(holdout_prompts):,} holdout prompts for exclusion")
    return holdout_prompts


def load_prompts_from_battles(
    battles_file: Path,
    max_prompts: int = 80000,
    exclude_prompts: set = None,
) -> list:
    """
    Load unique prompts from RouteLLM battles dataset.

    Args:
        battles_file: Path to battles JSONL file
        max_prompts: Maximum number of prompts to load
        exclude_prompts: Set of prompts to exclude (e.g. holdout set)

    Returns:
        List of unique prompt strings
    """
    print(f"\n📥 Loading prompts from: {battles_file}")

    if not battles_file.exists():
        raise FileNotFoundError(f"File not found: {battles_file}")

    if exclude_prompts is None:
        exclude_prompts = set()

    prompts_seen = set()
    prompts = []
    n_excluded = 0

    with open(battles_file, 'r') as f:
        for line in tqdm(f, desc="   Reading", total=max_prompts):
            if len(prompts) >= max_prompts:
                break

            try:
                battle = json.loads(line)
                prompt = battle['prompt']

                # Handle list-formatted prompts
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""

                # Handle stringified list prompts
                if isinstance(prompt, str) and prompt.startswith('["'):
                    try:
                        prompt_list = json.loads(prompt)
                        prompt = prompt_list[0] if prompt_list else ""
                    except:
                        pass

                prompt = prompt.strip()

                # Skip if we've seen this prompt or it's invalid
                if not prompt or prompt in prompts_seen:
                    continue

                # Skip if in exclusion set (holdout decontamination)
                if prompt in exclude_prompts:
                    n_excluded += 1
                    continue

                prompts_seen.add(prompt)
                prompts.append(prompt)

            except Exception as e:
                print(f"   ⚠️  Skipping line: {e}")
                continue

    print(f"   ✅ Loaded {len(prompts):,} unique prompts")
    if n_excluded > 0:
        print(f"   🔒 Excluded {n_excluded:,} holdout prompts (decontamination)")
    return prompts


def train_pca(prompts: list, n_components: int, output_path: Path, batch_size: int = 64):
    """
    Train PCA model on prompt embeddings.
    
    Args:
        prompts: List of prompt strings
        n_components: Number of PCA components (e.g., 23)
        output_path: Path to save PCA model
        batch_size: Batch size for embedding
    """
    print(f"\n🔤 Loading sentence encoder...")
    print(f"   Model: {DEFAULT_SENTENCE_TRANSFORMER}")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    print(f"   ✅ Encoder loaded")
    
    # Embed prompts
    print(f"\n🧮 Embedding {len(prompts):,} prompts...")
    print(f"   Batch size: {batch_size}")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True
    )
    
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    print(f"   Original dimension: {embeddings.shape[1]}")
    
    # Train PCA
    print(f"\n📐 Training PCA...")
    print(f"   Components: {n_components}")
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    # Statistics
    explained_var = np.sum(pca.explained_variance_ratio_)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"\n   ✅ PCA trained successfully")
    print(f"   Total explained variance: {explained_var:.2%}")
    print(f"   Variance by component:")
    for i in range(min(5, n_components)):
        print(f"      PC{i+1}: {pca.explained_variance_ratio_[i]:.3%} (cumulative: {cumulative_var[i]:.3%})")
    if n_components > 5:
        print(f"      ...")
        print(f"      PC{n_components}: {pca.explained_variance_ratio_[-1]:.3%} (cumulative: {cumulative_var[-1]:.3%})")
    
    # Save PCA model
    print(f"\n💾 Saving PCA model...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    
    size_kb = output_path.stat().st_size / 1024
    print(f"   ✅ Saved to: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    
    return pca


def verify_pca(pca_path: Path, test_prompts: list):
    """
    Verify PCA model works correctly.
    
    Args:
        pca_path: Path to saved PCA model
        test_prompts: List of test prompts
    """
    print(f"\n🔍 Verifying PCA model...")
    
    # Load PCA
    pca = joblib.load(pca_path)
    print(f"   ✅ Loaded PCA: {pca.n_components_} components")
    
    # Load encoder and embed test prompts
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(test_prompts, normalize_embeddings=True, convert_to_numpy=True)
    
    # Transform with PCA
    reduced = pca.transform(embeddings)
    
    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {reduced.shape}")
    print(f"   ✅ PCA transform works correctly")
    
    # Show sample output
    print(f"\n   Sample PCA features (first prompt, first 5 components):")
    for i in range(min(5, pca.n_components_)):
        print(f"      PC{i+1}: {reduced[0, i]:+.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train PCA model from RouteLLM battle data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (default: 23 components, 80K prompts)
    python3 scripts/train_pca_from_routellm.py
    
    # Custom settings
    python3 scripts/train_pca_from_routellm.py \\
        --n-components 32 \\
        --max-prompts 50000
    
    # Different input/output
    python3 scripts/train_pca_from_routellm.py \\
        --input data/my_battles.jsonl \\
        --output artifacts/pca_custom.joblib

Why PCA?
    - Reduces embedding size: 384 → 23 dimensions
    - Faster LinUCB updates (matrix operations)
    - Captures 80-90% of semantic variance
    - Consistent representation for warmup + live routing
        """
    )
    
    parser.add_argument(
        "--input", type=str,
        default=str(ROUTELLM_BATTLES_REWARDS_PATH),
        help="Input battles JSONL file (default: canonical RouteLLM battles rewards path)"
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DEFAULT_PCA_PATH),
        help=f"Output PCA model path (default: {DEFAULT_PCA_PATH})"
    )
    parser.add_argument(
        "--n-components", type=int, default=23,
        help="Number of PCA components (default: 23)"
    )
    parser.add_argument(
        "--max-prompts", type=int, default=80000,
        help="Maximum prompts to use for training (default: 80000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for embedding (default: 64)"
    )
    parser.add_argument(
        "--exclude-holdout", action="store_true",
        help="Exclude holdout prompts from PCA training (decontamination). "
             "Produces an uncontaminated PCA artifact for fair evaluation."
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    # If excluding holdout, default output to a decontaminated artifact path
    if args.exclude_holdout and args.output == str(DEFAULT_PCA_PATH):
        output_file = ARTIFACTS_DIR / f"pca_{args.n_components}_decontaminated.joblib"
    else:
        output_file = Path(args.output)
    
    print("="*80)
    print("TRAIN PCA MODEL FROM ROUTELLM BATTLES")
    print("="*80)
    
    print(f"\n📋 Configuration:")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print(f"   PCA components: {args.n_components}")
    print(f"   Max prompts: {args.max_prompts:,}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Exclude holdout: {args.exclude_holdout}")

    # Step 0 (optional): Load holdout prompts for exclusion
    exclude_prompts = set()
    if args.exclude_holdout:
        print(f"\n🔒 Decontamination mode: excluding holdout prompts")
        exclude_prompts = load_holdout_prompts(CANONICAL_HOLDOUT_DATA_PATH)

    # Step 1: Load prompts
    prompts = load_prompts_from_battles(
        input_file, args.max_prompts, exclude_prompts=exclude_prompts
    )
    
    if len(prompts) == 0:
        print("\n❌ No prompts loaded! Check input file.")
        return
    
    # Step 2: Train PCA
    pca = train_pca(prompts, args.n_components, output_file, args.batch_size)
    
    # Step 3: Verify
    test_prompts = prompts[:3]  # Use first 3 prompts for verification
    verify_pca(output_file, test_prompts)
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("✅ PCA TRAINING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"   PCA model: {output_file}")
    print(f"   Components: {args.n_components}")
    print(f"   Training samples: {len(prompts):,}")
    print(f"   Explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print(f"\n🚀 Next Steps:")
    print(f"\n   1. Generate warmup priors using this PCA:")
    print(f"      python3 scripts/generate_warmup_priors.py \\")
    print(f"          --rewards-file {input_file} \\")
    print(f"          --pca {output_file} \\")
    print(f"          --output artifacts/priors_warmup.joblib")
    
    print(f"\n   2. Use in calibration:")
    print(f"      python3 scripts/calibration/find_gamma.py \\")
    print(f"          --pca {output_file} \\")
    print(f"          --calibration-data your_data.jsonl")
    
    print(f"\n   3. Load in Python:")
    print(f"      import joblib")
    print(f"      pca = joblib.load('{output_file}')")
    print(f"      # pca.transform(embeddings)  # 384 → {args.n_components}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

