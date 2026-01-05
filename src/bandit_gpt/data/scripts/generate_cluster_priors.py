import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def main():
    base_dir = Path(__file__).parent
    
    # Input file
    input_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # Load prompts and organize by cluster
    print("Loading clustered prompts...")
    prompts = []
    cluster_ids = []
    
    cluster_map = defaultdict(list)
    
    with open(input_path) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            p_text = data["prompt"]
            c_id = int(data["cluster_id"])
            
            prompts.append(p_text)
            cluster_ids.append(c_id)
            cluster_map[c_id].append(i) # Track index of prompt
            
    print(f"Loaded {len(prompts)} prompts across {len(cluster_map)} clusters.")

    # Embed all prompts
    print(f"Embedding {len(prompts)} prompts...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Optimize: Batch encoding
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    
    # Calculate Cluster Sums
    print("Calculating cluster sum vectors...")
    n_clusters = 100 # We know there are 100 clusters (0-99)
    embedding_dim = embeddings.shape[1]
    
    cluster_sums = np.zeros((n_clusters, embedding_dim))
    cluster_counts = np.zeros(n_clusters)
    
    for c_id, indices in cluster_map.items():
        if c_id >= n_clusters:
            print(f"Warning: Cluster ID {c_id} exceeds limit {n_clusters}")
            continue
            
        # Get embeddings for this cluster
        cluster_embs = embeddings[indices]
        
        # Sum them up
        cluster_sums[c_id] = np.sum(cluster_embs, axis=0)
        cluster_counts[c_id] = len(indices)
        
    # Calculate Global Covariance (for A matrix)
    # This remains the same: sum(x x^T) over all prompts
    print("Calculating global covariance matrix...")
    cov_matrix = embeddings.T @ embeddings
    
    # Calculate Global Sum (for fallback/sanity check)
    global_sum = np.sum(embeddings, axis=0)
    
    # Save
    output_path = base_dir / "priors_meta_clusters.npz"
    np.savez(
        output_path, 
        cluster_sums=cluster_sums, 
        cluster_counts=cluster_counts,
        cov_matrix=cov_matrix, 
        global_sum=global_sum,
        n_samples=len(prompts)
    )
    
    print(f"Saved cluster priors to {output_path}")
    print(f"Cluster Sums Shape: {cluster_sums.shape}")
    print(f"Cov Matrix Shape: {cov_matrix.shape}")

if __name__ == "__main__":
    main()
