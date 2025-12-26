
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import shutil

def main():
    base_dir = Path(__file__).parent
    
    # Configuration
    n_clusters = 500
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    files_to_process = [
        "train_prompts.jsonl",
        "test_prompts.jsonl"
    ]
    
    print(f"Loading SentenceTransformer: {model_name}...")
    encoder = SentenceTransformer(model_name)
    
    # We ideally fit clusters on TRAIN and then predict on TEST
    # Step 1: Load Train Data to fit KMeans
    train_path = base_dir / "train_prompts.jsonl"
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return

    print("Loading training prompts...")
    train_data = []
    with open(train_path) as f:
        for line in f:
            train_data.append(json.loads(line))
            
    train_prompts = [x['prompt'] for x in train_data]
    
    print(f"Embedding {len(train_prompts)} training prompts...")
    train_embeddings = encoder.encode(train_prompts, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"Fitting KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(train_embeddings)
    
    # Step 2: Process Clusters for Both Train and Test
    for fname in files_to_process:
        fpath = base_dir / fname
        if not fpath.exists():
            continue
            
        print(f"Processing {fname}...")
        
        # Reload to ensure clean slate (though train is already loaded)
        items = []
        with open(fpath) as f:
            for line in f:
                items.append(json.loads(line))
        
        prompts = [x['prompt'] for x in items]
        embeddings = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=True)
        
        # Predict Clusters
        cluster_ids = kmeans.predict(embeddings)
        
        # Calculate Similarity (Cosine Similarity to Centroid)
        # Since embeddings are normalized, similarity is dot product
        # Distance = ||x - c||^2 = 2 - 2(x.c). So closer is higher dot product.
        centroids = kmeans.cluster_centers_
        # Normalize centroids just in case sklearn doesn't perfectly maintain it
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        updated_items = []
        for i, item in enumerate(items):
            cid = int(cluster_ids[i])
            centroid = centroids[cid]
            emb = embeddings[i]
            
            # Cosine similarity
            sim = float(np.dot(emb, centroid))
            
            # Update Item
            item['cluster_id'] = cid
            item['similarity'] = sim
            updated_items.append(item)
            
        # Write back
        # Backup original
        shutil.copy(fpath, fpath.with_suffix('.jsonl.bak'))
        
        with open(fpath, 'w') as f:
            for item in updated_items:
                f.write(json.dumps(item) + "\n")
                
        print(f"Updated {fname} with cluster IDs.")

if __name__ == "__main__":
    main()
