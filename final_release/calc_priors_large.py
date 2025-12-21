import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    # Paths
    base_dir = Path(__file__).parent
    prompts_path = Path("banditgpt/data/priors/lmsys_all_prompts.jsonl")
    output_path = base_dir / "data/priors_meta_large.npz"
    
    # Load Evaluation Prompts to exclude (Data Leakage Fix)
    eval_prompts = set()
    for p_file in ["data/train_prompts.jsonl", "data/test_prompts.jsonl"]:
        p_path = base_dir / p_file
        if p_path.exists():
            with open(p_path) as f:
                for line in f:
                    eval_prompts.add(json.loads(line)["prompt"].strip())
    
    print(f"Loaded {len(eval_prompts)} evaluation prompts to exclude.")
    
    print(f"Loading prompts from {prompts_path}...")
    prompts = []
    seen = set()
    excluded_count = 0
    duplicate_count = 0
    
    with open(prompts_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    text = data.get("prompt") or data.get("text") or data.get("content")
                else:
                    text = str(data)
                
                if text:
                    clean_text = text.strip()
                    # 1. Check for Data Leakage (Evaluation Prompts)
                    if clean_text in eval_prompts:
                        excluded_count += 1
                        continue
                    
                    # 2. Check for Duplicates
                    if clean_text in seen:
                        duplicate_count += 1
                        continue
                    
                    seen.add(clean_text)
                    prompts.append(clean_text)
            except Exception as e:
                print(f"Skipping bad line: {e}")
                
    print(f"Loaded {len(prompts)} unique prompts.")
    print(f" - Excluded (Data Leakage): {excluded_count}")
    print(f" - Excluded (Duplicates): {duplicate_count}")
    
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
