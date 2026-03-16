import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pareto_bandit.config import DEFAULT_SENTENCE_TRANSFORMER

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    # Load prompts
    print("Loading training prompts...")
    prompts = []
    with open(data_dir / "train_prompts.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            
    # Embed
    print(f"Embedding {len(prompts)} prompts...")
    encoder = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER)
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
    
    # Calculate Statistics
    # Sum Vector: sum(x)
    sum_vec = np.sum(embeddings, axis=0)
    
    # Covariance Matrix (uncentered, technically Gram matrix sum): sum(x x^T)
    # This is what LinUCB's A matrix accumulates.
    cov_matrix = embeddings.T @ embeddings
    
    output_path = base_dir / "data" / "priors_meta.npz"
    np.savez(output_path, sum_vec=sum_vec, cov_matrix=cov_matrix, n_samples=len(prompts))
    
    print(f"Saved priors metadata to {output_path}")
    print(f"Sum Vector: {sum_vec.shape}")
    print(f"Cov Matrix: {cov_matrix.shape}")

if __name__ == "__main__":
    main()
