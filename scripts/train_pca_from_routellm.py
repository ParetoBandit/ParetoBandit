#!/usr/bin/env python3
"""
Train PCA Model from RouteLLM Battle Data

This script:
1. Loads prompts from the RouteLLM battles dataset (80K prompts)
2. Embeds them using the default SentenceTransformer (see `bandit_gpt.config.DEFAULT_SENTENCE_TRANSFORMER`)
3. Trains a PCA model to reduce dimensionality (1024 -> 32 components for `BAAI/bge-m3`)
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
        --output src/bandit_gpt/data/artifacts/pca_32.joblib \\
        --n-components 32 \\
        --max-prompts 80000
"""

import sys
from pathlib import Path

# Add project root and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import json
import gzip
import argparse
import joblib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from bandit_gpt.config import (
    DEFAULT_SENTENCE_TRANSFORMER,
    DEFAULT_PCA_PATH,
    ROUTELLM_BATTLES_REWARDS_PATH,
    CANONICAL_DEV_DATA_PATH,
    CANONICAL_HOLDOUT_DATA_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
    THREE_WAY_SPLITS_PATH,
)

from utils.multimodel import PORTFOLIO_K10

def load_prompts_from_battles(
    battles_file: Path,
    max_prompts: int = 80000,
    *,
    exclude_prompts: set[str] | None = None,
) -> list:
    """
    Load unique prompts from RouteLLM battles dataset.

    Args:
        battles_file: Path to battles JSONL file
        max_prompts: Maximum number of prompts to load

    Returns:
        List of unique prompt strings
    """
    print(f"\n📥 Loading prompts from: {battles_file}")

    if not battles_file.exists():
        raise FileNotFoundError(f"File not found: {battles_file}")

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
                if exclude_prompts is not None and prompt in exclude_prompts:
                    n_excluded += 1
                    continue

                prompts_seen.add(prompt)
                prompts.append(prompt)

            except Exception as e:
                print(f"   ⚠️  Skipping line: {e}")
                continue

    if exclude_prompts is not None:
        print(f"   Excluded (overlaps removed): {n_excluded:,}")
    print(f"   ✅ Loaded {len(prompts):,} unique prompts")
    return prompts


def _load_prompts_with_full_model_coverage(
    rewards_gz_path: Path,
    *,
    models: list[str],
    prompt_filter: set[str] | None = None,
) -> set[str]:
    """Load prompt strings that have rewards for all requested models."""
    model_set = set(models)
    by_prompt: dict[str, set[str]] = {}
    with gzip.open(rewards_gz_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            if not entry.get("ok"):
                continue
            prompt = entry["prompt"]
            if prompt_filter is not None and prompt not in prompt_filter:
                continue
            model_id = entry["model_id"]
            if model_id not in model_set:
                continue
            by_prompt.setdefault(prompt, set()).add(model_id)
    return {p for p, ms in by_prompt.items() if ms == model_set}


def build_overlap_exclusion_set(
    *,
    exclude_k2_dev_holdout: bool,
    exclude_k10_dev_holdout: bool,
) -> set[str]:
    """Return prompt strings to exclude from the external PCA corpus."""
    exclude: set[str] = set()

    if exclude_k2_dev_holdout:
        k2_models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
        exclude |= _load_prompts_with_full_model_coverage(Path(CANONICAL_DEV_DATA_PATH), models=k2_models)
        exclude |= _load_prompts_with_full_model_coverage(Path(CANONICAL_HOLDOUT_DATA_PATH), models=k2_models)

    if exclude_k10_dev_holdout:
        splits = json.loads(Path(THREE_WAY_SPLITS_PATH).read_text())
        prior_pool = set(splits["prior_train_pool"])
        online_pool = set(splits["online_learn_pool"])
        dev_pool = prior_pool | online_pool

        exclude |= _load_prompts_with_full_model_coverage(
            Path(DEV_DATA_PATH_ALL_MODELS),
            models=list(PORTFOLIO_K10),
            prompt_filter=dev_pool,
        )
        exclude |= _load_prompts_with_full_model_coverage(
            Path(HOLDOUT_DATA_PATH_ALL_MODELS),
            models=list(PORTFOLIO_K10),
        )

    return exclude

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
    # Basic usage (default: 32 components, 80K prompts)
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
    - Reduces embedding size: 1024 → 32 dimensions (for `BAAI/bge-m3`)
    - Faster LinUCB updates (matrix operations)
    - Captures a substantial fraction of embedding variance in a compact representation
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
        "--n-components", type=int, default=32,
        help="Number of PCA components (default: 32)"
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
        "--exclude-k2-overlaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude prompts that appear in canonical K=2 dev/holdout (default: True).",
    )
    parser.add_argument(
        "--exclude-k10-overlaps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude prompts that appear in K=10 dev pools or holdout (default: True).",
    )
    parser.add_argument(
        "--save-filtered-prompts",
        type=str,
        default="",
        help="Optional path to save the filtered prompt list as JSONL.",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
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

    # Step 1: Load prompts
    exclude = build_overlap_exclusion_set(
        exclude_k2_dev_holdout=bool(args.exclude_k2_overlaps),
        exclude_k10_dev_holdout=bool(args.exclude_k10_overlaps),
    )
    if args.exclude_k2_overlaps or args.exclude_k10_overlaps:
        print(f"\n🧹 Exclusion set size (unique prompts): {len(exclude):,}")
    prompts = load_prompts_from_battles(
        input_file, args.max_prompts, exclude_prompts=exclude
    )

    if str(args.save_filtered_prompts).strip():
        out_path = Path(args.save_filtered_prompts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
        print(f"   ✅ Saved filtered prompts to: {out_path}")
    
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
    print(f"\n   1. Generate 43-model warmup priors using this PCA:")
    print(f"      python3 scripts/generate_multimodel_warmup_priors.py \\")
    print(f"          --pca {output_file}")
    print(f"\n      Then extract K=2 priors:")
    print(f"      python3 scripts/extract_k2_warmup_from_multimodel.py")

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

