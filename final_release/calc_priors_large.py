import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    # Paths
    base_dir = Path(__file__).parent
    prompts_path = Path("banditgpt/data/priors/lmsys_all_prompts.jsonl")
    output_path = base_dir / "data/priors_meta_large.npz"
    
    print(f"Loading prompts from {prompts_path}...")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                # Handle different formats if necessary, assuming "prompt" key or raw text
                if isinstance(data, dict):
                    text = data.get("prompt") or data.get("text") or data.get("content")
                else:
                    text = str(data)
                
                if text:
                    prompts.append(text)
            except Exception as e:
                print(f"Skipping bad line: {e}")
                
    print(f"Loaded {len(prompts)} prompts.")
    
    # Embed
    print("Embedding prompts (this may take a while)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    
    # Compute Statistics
    print("Computing covariance and sum...")
    sum_vec = np.sum(embeddings, axis=0)
    cov_matrix = embeddings.T @ embeddings
    
    # Save
    print(f"Saving to {output_path}...")
    np.savez(output_path, sum_vec=sum_vec, cov_matrix=cov_matrix)
    print("Done.")

if __name__ == "__main__":
    main()
