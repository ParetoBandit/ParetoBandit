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
    K4_TRAIN_DATA_PATH,
    K4_CAL_DATA_PATH,
    K4_HOLDOUT_DATA_PATH,
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
    """Collect all prompts used in the K=4 experimental splits.

    PCA is unsupervised, so including these prompts would not cause label
    leakage.  We still exclude them for methodological cleanliness: the
    PCA fitting data is then strictly disjoint from all evaluation data.
    """
    exclude: set[str] = set()
    for path in (K4_TRAIN_DATA_PATH, K4_CAL_DATA_PATH, K4_HOLDOUT_DATA_PATH):
        p = Path(path)
        if p.exists():
            exclude |= _load_all_prompts_from_rewards(p)
            print(f"   Loaded {len(exclude):,} exclusion prompts so far "
                  f"(after {p.name})")
    return exclude

LMSYS_EMBEDDINGS_CACHE = (
    Path(__file__).parent.parent / "data_collection" / "embeddings"
    / "lmsys_pca_training_embeddings.npz"
)


def embed_prompts(
    prompts: list[str],
    *,
    batch_size: int = 64,
    max_seq_length: int = 512,
    cache_path: Path | None = LMSYS_EMBEDDINGS_CACHE,
) -> np.ndarray:
    """Embed *prompts* with the canonical encoder, using a disk cache.

    If *cache_path* exists and contains embeddings for the same prompt
    set (matched by count + first/last hash), the cached matrix is
    returned directly.  Otherwise, prompts are encoded, the result is
    saved to *cache_path*, and the matrix is returned.

    Returns:
        ``np.ndarray`` of shape ``(len(prompts), encoder_dim)``.
    """
    import hashlib

    def _fingerprint(ps: list[str]) -> str:
        h = hashlib.sha256()
        h.update(str(len(ps)).encode())
        h.update(ps[0].encode())
        h.update(ps[-1].encode())
        return h.hexdigest()[:16]

    fp = _fingerprint(prompts)

    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        if data.get("fingerprint", None) is not None:
            cached_fp = str(data["fingerprint"])
        else:
            cached_fp = None
        cached_emb = data["embeddings"]
        if cached_fp == fp and cached_emb.shape[0] == len(prompts):
            print(f"  Loaded cached embeddings: {cache_path.name} "
                  f"({cached_emb.shape})")
            return cached_emb
        print(f"  Cache fingerprint mismatch — re-encoding.")

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

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, embeddings=embeddings, fingerprint=np.array(fp),
        )
        print(f"  Saved embeddings cache: {cache_path} "
              f"({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return embeddings


def fit_pca(
    embeddings: np.ndarray,
    n_components: int,
    output_path: Path,
) -> PCA:
    """Fit PCA on *embeddings* and persist to *output_path*.

    Args:
        embeddings: ``(n_samples, n_features)`` matrix.
        n_components: Number of PCA components to retain.
        output_path: Where to save the fitted PCA artifact.

    Returns:
        The fitted ``sklearn.decomposition.PCA`` object.
    """
    print(f"\n  Fitting PCA ({n_components} components) on "
          f"{embeddings.shape[0]:,} samples ...")
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
    # Default: 15 + 32 components from LMSYS battles
    python3 scripts/train_pca_from_routellm.py

    # Single variant only
    python3 scripts/train_pca_from_routellm.py --n-components 15

    # Custom input/output
    python3 scripts/train_pca_from_routellm.py \\
        --input data/my_prompts.jsonl --output artifacts/pca_custom.joblib \\
        --n-components 15
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
        default=None,
        help="Output PCA model path (default: auto-named per variant)",
    )
    parser.add_argument(
        "--n-components", type=int, nargs="+", default=[15, 32],
        help="PCA component counts to train (default: 15 32)",
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
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip embedding cache (always re-encode).",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    artifacts_dir = Path(DEFAULT_PCA_PATH).parent

    print("=" * 80)
    print("TRAIN PCA MODEL — LMSYS Chatbot Arena")
    print("=" * 80)

    print(f"\n  Input : {input_file}")
    print(f"  PCA variants   : {args.n_components}")
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

    # Embed once (with disk cache)
    cache_path = None if args.no_cache else LMSYS_EMBEDDINGS_CACHE
    embeddings = embed_prompts(
        prompts, batch_size=args.batch_size, cache_path=cache_path,
    )

    # Fit each PCA variant from the same embeddings
    results: list[tuple[int, Path, PCA]] = []
    for nc in args.n_components:
        if args.output and len(args.n_components) == 1:
            out = Path(args.output)
        else:
            out = artifacts_dir / f"pca_{nc}.joblib"
        pca = fit_pca(embeddings, nc, out)
        results.append((nc, out, pca))

    # Verify round-trip with the first variant
    verify_pca(results[0][1], prompts[:3])

    # Summary
    print("\n" + "=" * 80)
    print("PCA TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Training samples  : {len(prompts):,}")
    for nc, out, pca in results:
        ev = np.sum(pca.explained_variance_ratio_)
        print(f"  PCA-{nc:>2d}: {ev:.2%} variance  -> {out}")
    print(f"\nNext: regenerate warmup priors with the canonical PCA:")
    print(f"  python3 scripts/generate_multimodel_warmup_priors.py "
          f"--pca {results[0][1]}")
    print("=" * 80)


if __name__ == "__main__":
    main()

