"""
Data Management for ParetoBandit Evaluation Pipeline

This module provides classes for:
1. Clustering prompts using sentence transformers
2. Stratified train/test splitting
3. Dataset sampling and validation
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from pareto_bandit.config import DEFAULT_SENTENCE_TRANSFORMER


class PromptClusterer:
    """Handles clustering of prompts using sentence embeddings"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER,
        n_clusters: int = 100,
    ):
        """
        Args:
            model_name: SentenceTransformer model name
            n_clusters: Number of clusters to create
        """
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.encoder = None
        self.clusterer = None
        
    def load_encoder(self):
        """Lazy load the sentence transformer model"""
        if self.encoder is None:
            print(f"Loading encoder: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
    
    def cluster_prompts(self, prompts: List[str], batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster prompts and return cluster assignments and centroids
        
        Args:
            prompts: List of prompt strings
            batch_size: Batch size for encoding
            
        Returns:
            (cluster_ids, embeddings) tuple
        """
        self.load_encoder()
        
        print(f"Encoding {len(prompts)} prompts...")
        embeddings = self.encoder.encode(prompts, batch_size=batch_size, show_progress_bar=True)
        
        print(f"Clustering into {self.n_clusters} clusters...")
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=batch_size * 10,
            n_init=3
        )
        cluster_ids = self.clusterer.fit_predict(embeddings)
        
        # Print cluster distribution
        cluster_counts = Counter(cluster_ids)
        print(f"\nCluster distribution:")
        print(f"  Min size: {min(cluster_counts.values())}")
        print(f"  Max size: {max(cluster_counts.values())}")
        print(f"  Mean size: {np.mean(list(cluster_counts.values())):.1f}")
        
        return cluster_ids, embeddings
    
    def cluster_file(self, input_path: Path, output_path: Path):
        """
        Cluster prompts from a JSONL file and save with cluster IDs
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file with cluster_id added
        """
        print(f"Loading prompts from {input_path}")
        prompts = []
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get("prompt", ""))
        
        cluster_ids, embeddings = self.cluster_prompts(prompts)
        
        print(f"\nSaving clustered data to {output_path}")
        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for i, line in enumerate(fin):
                data = json.loads(line)
                data["cluster_id"] = int(cluster_ids[i])
                fout.write(json.dumps(data) + '\n')
        
        print("✓ Done!")


class DataSplitter:
    """Handles stratified splitting of clustered data into train/test sets"""
    
    def __init__(self, test_size: int = 1000, train_size: int = 4000, random_seed: int = 42):
        """
        Args:
            test_size: Number of test samples
            train_size: Number of train samples
            random_seed: Random seed for reproducibility
        """
        self.test_size = test_size
        self.train_size = train_size
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def stratified_split(self, prompts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform stratified split by cluster_id
        
        Args:
            prompts: List of prompt dictionaries with cluster_id field
            
        Returns:
            (test_prompts, train_prompts) tuple
        """
        # Group by cluster
        clusters = defaultdict(list)
        for p in prompts:
            clusters[p['cluster_id']].append(p)
        
        print(f"Found {len(clusters)} clusters")
        
        test_prompts = []
        train_prompts = []
        
        # Stratified sampling within each cluster
        for cluster_id, cluster_items in clusters.items():
            random.shuffle(cluster_items)
            
            cluster_size = len(cluster_items)
            
            # Proportional allocation
            n_test = int(self.test_size * cluster_size / len(prompts))
            n_train = int(self.train_size * cluster_size / len(prompts))
            
            # Ensure at least 1 from each cluster if possible
            if cluster_size >= 2:
                n_test = max(1, n_test)
                n_train = max(1, n_train)
            
            # Don't exceed cluster size
            n_test = min(n_test, cluster_size)
            n_train = min(n_train, cluster_size - n_test)
            
            # Allocate
            test_prompts.extend(cluster_items[:n_test])
            train_prompts.extend(cluster_items[n_test:n_test + n_train])
        
        # Adjust to exact sizes
        random.shuffle(test_prompts)
        random.shuffle(train_prompts)
        
        # Handle size mismatches
        if len(test_prompts) < self.test_size:
            remaining = [p for cid, ps in clusters.items() 
                        for p in ps if p not in test_prompts and p not in train_prompts]
            random.shuffle(remaining)
            test_prompts.extend(remaining[:self.test_size - len(test_prompts)])
        
        if len(train_prompts) < self.train_size:
            remaining = [p for cid, ps in clusters.items() 
                        for p in ps if p not in test_prompts and p not in train_prompts]
            random.shuffle(remaining)
            train_prompts.extend(remaining[:self.train_size - len(train_prompts)])
        
        # Trim to exact sizes
        test_prompts = test_prompts[:self.test_size]
        train_prompts = train_prompts[:self.train_size]
        
        # Verify
        test_clusters = set(p['cluster_id'] for p in test_prompts)
        train_clusters = set(p['cluster_id'] for p in train_prompts)
        
        print(f"\nSplit complete:")
        print(f"  Test: {len(test_prompts)} prompts across {len(test_clusters)} clusters")
        print(f"  Train: {len(train_prompts)} prompts across {len(train_clusters)} clusters")
        print(f"  Overlap: {len(test_clusters & train_clusters)} clusters")
        
        return test_prompts, train_prompts
    
    def split_file(self, input_path: Path, test_path: Path, train_path: Path):
        """
        Split a clustered JSONL file into train and test sets
        
        Args:
            input_path: Path to clustered input file
            test_path: Path to save test set
            train_path: Path to save train set
        """
        print(f"Loading from {input_path}")
        prompts = []
        with open(input_path, 'r') as f:
            for line in f:
                prompts.append(json.loads(line))
        
        test_prompts, train_prompts = self.stratified_split(prompts)
        
        print(f"\nSaving test set to {test_path}")
        with open(test_path, 'w') as f:
            for p in test_prompts:
                f.write(json.dumps(p) + '\n')
        
        print(f"Saving train set to {train_path}")
        with open(train_path, 'w') as f:
            for p in train_prompts:
                f.write(json.dumps(p) + '\n')
        
        print("\n✓ Done!")


class DatasetSampler:
    """Handles sampling operations on datasets"""
    
    @staticmethod
    def cluster_stratified_sample(prompts: List[Dict], target_size: int, random_seed: int = 42) -> List[Dict]:
        """
        Sample prompts maintaining cluster distribution
        
        Args:
            prompts: List of prompt dicts with cluster_id
            target_size: Target number of samples
            random_seed: Random seed
            
        Returns:
            Sampled prompts
        """
        random.seed(random_seed)
        
        clusters = defaultdict(list)
        for p in prompts:
            clusters[p['cluster_id']].append(p)
        
        sampled = []
        total = len(prompts)
        
        for cluster_id, cluster_items in clusters.items():
            # Proportional sampling
            proportion = len(cluster_items) / total
            n_samples = max(1, int(target_size * proportion))
            n_samples = min(n_samples, len(cluster_items))
            
            random.shuffle(cluster_items)
            sampled.extend(cluster_items[:n_samples])
        
        # Adjust to exact size
        random.shuffle(sampled)
        return sampled[:target_size]


# Convenience function
def prepare_dataset(source_file: Path, output_dir: Path, 
                   n_clusters: int = 100, test_size: int = 1000, train_size: int = 4000):
    """
    Complete dataset preparation pipeline
    
    Args:
        source_file: Path to raw prompts JSONL
        output_dir: Directory to save outputs
        n_clusters: Number of clusters
        test_size: Test set size
        train_size: Train set size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Cluster
    clusterer = PromptClusterer(n_clusters=n_clusters)
    clustered_file = output_dir / f"{source_file.stem}_clustered.jsonl"
    clusterer.cluster_file(source_file, clustered_file)
    
    # Step 2: Split
    splitter = DataSplitter(test_size=test_size, train_size=train_size)
    test_file = output_dir / "test_prompts.jsonl"
    train_file = output_dir / "train_prompts.jsonl"
    splitter.split_file(clustered_file, test_file, train_file)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Clustered data: {clustered_file}")
    print(f"  Test set: {test_file}")
    print(f"  Train set: {train_file}")
