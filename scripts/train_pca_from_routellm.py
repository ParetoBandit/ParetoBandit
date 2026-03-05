#!/usr/bin/env python3
"""
Train PCA Model from LMSYS Chatbot Arena Data

Trains an unsupervised PCA projection on prompts drawn from the same
distribution used throughout the BanditGPT pipeline (LMSYS Chatbot Arena).

Pipeline
--------
1. Loads unique prompts from the LMSYS battles corpus (~50K English prompts).
2. Excludes any prompts that appear in the experimental dev or holdout
   reward files to keep the PCA fitting data strictly independent of
   all evaluation data.
3. Embeds prompts using ``DEFAULT_SENTENCE_TRANSFORMER`` (``BAAI/bge-m3``).
4. Fits a PCA model (1024 -> N components) and saves the artifact.

Usage
-----
    python3 scripts/train_pca_from_routellm.py          # defaults: 15 comp
    python3 scripts/train_pca_from_routellm.py --n-components 32
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
    LMSYS_BATTLES_PATH,
    DEV_DATA_PATH_ALL_MODELS,
    HOLDOUT_DATA_PATH_ALL_MODELS,
)

def load_prompts_from_battles(
    battles_file: Path,
    max_prompts: int = 80000,
    *,
    exclude_prompts: set[str] | None = None,
) -> list:
    """
    Load unique prompts from the offline battles dataset.

    Args:
        battles_file: Path to battles JSONL file
        max_prompts: Maximum number of prompts to load

    Returns:
        List of unique prompt strings
    """
    print(f"\n  Loading prompts from: {battles_file}")

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
                print(f"  Skipping line: {e}")
                continue

    if exclude_prompts is not None:
        print(f"  Excluded (overlaps removed): {n_excluded:,}")
    print(f"  Loaded {len(prompts):,} unique prompts")
    return prompts


def _load_all_prompts_from_rewards(rewards_gz_path: Path) -> set[str]:
    """Return every unique prompt string appearing in a gzipped rewards file."""
    prompts: set[str] = set()
    with gzip.open(rewards_gz_path, "rt") as f:
        for line in f:
            entry = json.loads(line)
            prompts.add(entry["prompt"])
    return prompts


def build_experimental_exclusion_set() -> set[str]:
    """Collect all prompts used in any experimental split (dev + holdout).

    PCA is unsupervised, so including these prompts would not cause label
    leakage.  We still exclude them for methodological cleanliness: the
    PCA fitting data is then strictly disjoint from all evaluation data.
    """
    exclude: set[str] = set()
    for path in (DEV_DATA_PATH_ALL_MODELS, HOLDOUT_DATA_PATH_ALL_MODELS):
        p = Path(path)
        if p.exists():
            exclude |= _load_all_prompts_from_rewards(p)
            print(f"   Loaded {len(exclude):,} exclusion prompts so far "
                  f"(after {p.name})")
    return exclude

def train_pca(prompts: list, n_components: int, output_path: Path,
              batch_size: int = 32, max_seq_length: int = 512):
    """Train PCA on SentenceTransformer embeddings of *prompts*.

    Args:
        prompts: Prompt strings to embed.
        n_components: Number of PCA components to retain.
        output_path: Where to save the fitted PCA artifact.
        batch_size: Encoding batch size (keep low for long-prompt safety).
        max_seq_length: Token-level truncation applied to the encoder to
            prevent OOM on extremely long prompts.
    """
    print(f"\n  Loading sentence encoder: {DEFAULT_SENTENCE_TRANSFORMER}")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    encoder.max_seq_length = max_seq_length
    print(f"  Encoder loaded (max_seq_length={max_seq_length})")

    print(f"\n  Embedding {len(prompts):,} prompts (batch_size={batch_size}) ...")
    embeddings = encoder.encode(
        prompts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"\n  Fitting PCA ({n_components} components) ...")
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    explained_var = np.sum(pca.explained_variance_ratio_)
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)

    print(f"  Total explained variance: {explained_var:.2%}")
    for i in range(min(5, n_components)):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.3%} "
              f"(cumul. {cumulative_var[i]:.3%})")
    if n_components > 5:
        print(f"    ...")
        print(f"    PC{n_components}: "
              f"{pca.explained_variance_ratio_[-1]:.3%} "
              f"(cumul. {cumulative_var[-1]:.3%})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    print(f"  Saved to: {output_path} "
          f"({output_path.stat().st_size / 1024:.1f} KB)")

    return pca


def verify_pca(pca_path: Path, test_prompts: list):
    """
    Verify PCA model works correctly.
    
    Args:
        pca_path: Path to saved PCA model
        test_prompts: List of test prompts
    """
    print(f"\n  Verifying PCA round-trip ...")
    pca = joblib.load(pca_path)
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(
        test_prompts, normalize_embeddings=True, convert_to_numpy=True,
    )
    reduced = pca.transform(embeddings)
    print(f"  Input {embeddings.shape} -> Output {reduced.shape}  OK")


def main():
    parser = argparse.ArgumentParser(
        description="Train PCA on LMSYS Chatbot Arena prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 15 components from LMSYS battles
    python3 scripts/train_pca_from_routellm.py

    # 32-component variant for ablation
    python3 scripts/train_pca_from_routellm.py --n-components 32

    # Custom input/output
    python3 scripts/train_pca_from_routellm.py \\
        --input data/my_prompts.jsonl --output artifacts/pca_custom.joblib
        """
    )

    parser.add_argument(
        "--input", type=str,
        default=str(LMSYS_BATTLES_PATH),
        help="Input JSONL file with a 'prompt' field per line "
             "(default: LMSYS Chatbot Arena battles)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(DEFAULT_PCA_PATH),
        help=f"Output PCA model path (default: {DEFAULT_PCA_PATH})",
    )
    parser.add_argument(
        "--n-components", type=int, default=15,
        help="Number of PCA components (default: 15)",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=60000,
        help="Maximum unique prompts to use (default: 60000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for SentenceTransformer encoding (default: 64)",
    )
    parser.add_argument(
        "--no-exclude-experimental",
        action="store_true",
        help="Skip exclusion of dev/holdout experimental prompts.",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    print("=" * 80)
    print("TRAIN PCA MODEL — LMSYS Chatbot Arena")
    print("=" * 80)

    print(f"\n  Input : {input_file}")
    print(f"  Output: {output_file}")
    print(f"  PCA components : {args.n_components}")
    print(f"  Max prompts    : {args.max_prompts:,}")

    # Collect experimental prompts to exclude
    if args.no_exclude_experimental:
        exclude: set[str] = set()
    else:
        print("\n  Building experimental-prompt exclusion set ...")
        exclude = build_experimental_exclusion_set()
        print(f"  Excluding {len(exclude):,} experimental prompts")

    prompts = load_prompts_from_battles(
        input_file, args.max_prompts, exclude_prompts=exclude,
    )

    if len(prompts) == 0:
        print("\n  ERROR: No prompts loaded — check input file.")
        return

    # Train
    pca = train_pca(prompts, args.n_components, output_file, args.batch_size)

    # Verify round-trip
    verify_pca(output_file, prompts[:3])

    # Summary
    print("\n" + "=" * 80)
    print("PCA TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Artifact          : {output_file}")
    print(f"  Components        : {args.n_components}")
    print(f"  Training samples  : {len(prompts):,}")
    print(f"  Explained variance: "
          f"{np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"\nNext: regenerate warmup priors with this PCA:")
    print(f"  python3 scripts/generate_multimodel_warmup_priors.py "
          f"--pca {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

