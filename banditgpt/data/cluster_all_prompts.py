import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

def main():
    base_dir = Path(__file__).parent
    source_path = base_dir / "lmsys_all_prompts.jsonl"
    output_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    
    # Configuration
    n_clusters = 100  # Optimal based on elbow/silhouette analysis
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"Loading prompts from {source_path}...")
    prompts_data = []
    seen_prompts = set()
    
    with open(source_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "prompt" in data:
                    prompt_text = data["prompt"].strip()
                elif isinstance(data, str):
                    prompt_text = data.strip()
                else:
                    text = data.get("prompt") or data.get("text") or data.get("content")
                    if text:
                        prompt_text = text.strip()
                    else:
                        continue
                
                # Deduplicate
                if prompt_text not in seen_prompts:
                    seen_prompts.add(prompt_text)
                    prompts_data.append({"prompt": prompt_text})
            except:
                pass
    
    print(f"Loaded {len(prompts_data)} unique prompts (deduplicated from {source_path})")
    
    # Extract prompt texts
    prompts = [p["prompt"] for p in prompts_data]
    
    # Embed
    print(f"Embedding {len(prompts)} prompts...")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    
    # Cluster
    print(f"Fitting KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, verbose=1)
    cluster_ids = kmeans.fit_predict(embeddings)
    
    # Calculate similarities to centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    print("Assigning cluster IDs and similarities...")
    for i, data in enumerate(tqdm(prompts_data)):
        cid = int(cluster_ids[i])
        centroid = centroids[cid]
        emb = embeddings[i]
        similarity = float(np.dot(emb, centroid))
        
        data['cluster_id'] = cid
        data['similarity'] = similarity
    
    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for data in prompts_data:
            f.write(json.dumps(data) + "\n")
    
    print(f"Done! Created {output_path} with {len(prompts_data)} clustered prompts")
    print(f"Cluster IDs range from 0 to {n_clusters-1}")

if __name__ == "__main__":
    main()
