"""
Cluster Detection for Semantic Prompt Routing

Detects which of the 100 semantic clusters a user prompt belongs to,
enabling cluster-aware reward boosting in the bandit router.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer

class ClusterDetector:
    """
    Detects the semantic cluster of a prompt using pre-computed centroids.
    
    Uses cosine similarity to find the nearest cluster centroid from training data.
    """
    
    def __init__(self, 
                 centroids_path: Optional[Path] = None,
                 encoder: Optional[SentenceTransformer] = None,
                 encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize cluster detector.
        
        Args:
            centroids_path: Path to cluster centroids file (will be computed if None)
            encoder: Pre-loaded sentence transformer (shares with BanditRouter)
            encoder_model: Model name if encoder not provided
        """
        self.encoder = encoder or SentenceTransformer(encoder_model)
        
        # Load or compute centroids
        if centroids_path and centroids_path.exists():
            self.centroids = self._load_centroids(centroids_path)
        else:
            # Compute from golden prompts
            self.centroids = self._compute_centroids_from_golden_prompts()
        
        self.n_clusters = len(self.centroids)
    
    def _compute_centroids_from_golden_prompts(self) -> np.ndarray:
        """
        Compute cluster centroids from golden prompts.
        
        Golden prompts are the representatives closest to each cluster center.
        """
        # Try to load golden prompts
        base_dir = Path(__file__).parent / "data"
        golden_path = base_dir / "golden_prompts.jsonl"
        
        if not golden_path.exists():
            # Fallback: use training data
            return self._compute_centroids_from_training()
        
        print(f"Loading golden prompts from {golden_path}...")
        golden_prompts = []
        cluster_ids = []
        
        with open(golden_path) as f:
            for line in f:
                data = json.loads(line)
                golden_prompts.append(data['prompt'])
                cluster_ids.append(data['cluster_id'])
        
        # Encode golden prompts (these are cluster representatives)
        print(f"Encoding {len(golden_prompts)} golden prompts...")
        embeddings = self.encoder.encode(golden_prompts, normalize_embeddings=True, 
                                        show_progress_bar=False)
        
        # Sort by cluster_id to ensure alignment
        sorted_pairs = sorted(zip(cluster_ids, embeddings), key=lambda x: x[0])
        centroids = np.array([emb for _, emb in sorted_pairs])
        
        print(f"✓ Loaded {len(centroids)} cluster centroids")
        return centroids
    
    def _compute_centroids_from_training(self) -> np.ndarray:
        """
        Compute centroids by averaging all prompts in each cluster from training data.
        """
        base_dir = Path(__file__).parent / "data"
        train_path = base_dir / "train_prompts.jsonl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Cannot find training data at {train_path}")
        
        print(f"Computing centroids from {train_path}...")
        
        # Group prompts by cluster
        from collections import defaultdict
        cluster_prompts = defaultdict(list)
        
        with open(train_path) as f:
            for line in f:
                data = json.loads(line)
                cluster_prompts[data['cluster_id']].append(data['prompt'])
        
        # Compute centroid for each cluster
        centroids = []
        for cluster_id in sorted(cluster_prompts.keys()):
            prompts = cluster_prompts[cluster_id]
            embeddings = self.encoder.encode(prompts[:100], normalize_embeddings=True,
                                            show_progress_bar=False)  # Limit to 100 for speed
            centroid = np.mean(embeddings, axis=0)
            # Re-normalize
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        print(f"✓ Computed {len(centroids)} cluster centroids")
        return centroids
    
    def _load_centroids(self, path: Path) -> np.ndarray:
        """Load pre-computed centroids from file."""
        data = np.load(path)
        return data['centroids']
    
    def save_centroids(self, path: Path):
        """Save centroids to file for faster loading."""
        np.savez_compressed(path, centroids=self.centroids)
        print(f"✓ Saved centroids to {path}")
    
    def detect_cluster(self, prompt: str) -> Tuple[int, float]:
        """
        Detect which cluster a prompt belongs to.
        
        Args:
            prompt: User input text
            
        Returns:
            (cluster_id, similarity): Cluster ID and cosine similarity to centroid
        """
        # Encode prompt
        embedding = self.encoder.encode([prompt], normalize_embeddings=True, 
                                       show_progress_bar=False)[0]
        
        # Compute cosine similarity to all centroids
        similarities = np.dot(self.centroids, embedding)
        
        # Find nearest cluster
        cluster_id = int(np.argmax(similarities))
        similarity = float(similarities[cluster_id])
        
        return cluster_id, similarity
    
    def detect_top_k_clusters(self, prompt: str, k: int = 3) -> list:
        """
        Return top K most similar clusters.
        
        Args:
            prompt: User input text
            k: Number of top clusters to return
            
        Returns:
            List of (cluster_id, similarity) tuples, sorted by similarity
        """
        embedding = self.encoder.encode([prompt], normalize_embeddings=True,
                                       show_progress_bar=False)[0]
        similarities = np.dot(self.centroids, embedding)
        
        # Get top K
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


def main():
    """Demo: Test cluster detection."""
    print("=== Cluster Detector Demo ===\n")
    
    # Initialize
    detector = ClusterDetector()
    
    # Test prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum entanglement to a 5 year old",
        "What are the best restaurants in Tokyo?",
        "Debug this JavaScript code: const x = [1,2,3]; x.push(4",
        "Write a haiku about machine learning"
    ]
    
    print(f"Testing {len(test_prompts)} prompts:\n")
    
    for prompt in test_prompts:
        cluster_id, similarity = detector.detect_cluster(prompt)
        top_3 = detector.detect_top_k_clusters(prompt, k=3)
        
        print(f"Prompt: \"{prompt[:60]}...\"")
        print(f"  → Cluster {cluster_id} (similarity: {similarity:.3f})")
        print(f"  Top 3: {[(c, f'{s:.3f}') for c, s in top_3]}")
        print()
    
    # Save centroids for faster future loading
    output_path = Path(__file__).parent / "data" / "cluster_centroids.npz"
    detector.save_centroids(output_path)
    
    print(f"\n✓ Cluster detection ready for production use!")

if __name__ == "__main__":
    main()
