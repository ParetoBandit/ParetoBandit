"""
Generate prior covariance matrix from prompts NOT in train/test sets.

Ensures zero data leakage by excluding all evaluation prompts.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    base_dir = Path(__file__).parent
    
    # Load all clustered prompts
    all_prompts_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    print(f"Loading all prompts from {all_prompts_path}...")
    all_prompts = []
    with open(all_prompts_path) as f:
        for line in f:
            all_prompts.append(json.loads(line))
    print(f"Loaded {len(all_prompts)} total prompts")
    
    # Load evaluation prompts (test + train)
    eval_prompts = set()
    
    for eval_file in ["test_prompts.jsonl", "train_prompts.jsonl"]:
        eval_path = base_dir / eval_file
        if eval_path.exists():
            with open(eval_path) as f:
                for line in f:
                    data = json.loads(line)
                    eval_prompts.add(data["prompt"])
    
    print(f"Loaded {len(eval_prompts)} evaluation prompts to exclude")
    
    # Filter to get ONLY prior prompts
    prior_prompts = []
    for p in all_prompts:
        if p["prompt"] not in eval_prompts:
            prior_prompts.append(p["prompt"])
    
    print(f"Filtered to {len(prior_prompts)} prompts for prior covariance matrix")
    print(f"Verification: {len(all_prompts)} total = {len(prior_prompts)} prior + {len(eval_prompts)} eval")
    
    # Embed
    print("Embedding prompts...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        prior_prompts, 
        normalize_embeddings=True, 
        show_progress_bar=True,
        batch_size=128
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute statistics for LinUCB prior
    print("Computing covariance and sum...")
    sum_vec = np.sum(embeddings, axis=0)
    cov_matrix = embeddings.T @ embeddings
    
    # Save
    output_path = base_dir / "priors_meta_large.npz"
    print(f"Saving to {output_path}...")
    np.savez(
        output_path, 
        sum_vec=sum_vec, 
        cov_matrix=cov_matrix,
        n_samples=len(prior_prompts)
    )
    
    print("Done!")
    print(f"Prior covariance matrix:")
    print(f"  - Based on {len(prior_prompts)} prompts")
    print(f"  - Sum vector shape: {sum_vec.shape}")
    print(f"  - Covariance matrix shape: {cov_matrix.shape}")
    print(f"  - Saved to: {output_path}")

if __name__ == "__main__":
    main()
