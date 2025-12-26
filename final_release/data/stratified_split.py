import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

def main():
    base_dir = Path(__file__).parent
    source_path = base_dir / "lmsys_all_prompts_clustered.jsonl"
    
    # Output paths
    test_path = base_dir / "test_prompts.jsonl"
    train_path = base_dir / "train_prompts.jsonl"
    
    # Target sizes
    TEST_SIZE = 1000
    TRAIN_SIZE = 4000
    TOTAL_EVAL = TEST_SIZE + TRAIN_SIZE  # 5000
    
    print(f"Loading clustered prompts from {source_path}...")
    prompts = []
    with open(source_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Group by cluster_id
    clusters = defaultdict(list)
    for p in prompts:
        clusters[p['cluster_id']].append(p)
    
    print(f"Found {len(clusters)} unique clusters")
    print(f"Cluster sizes: min={min(len(v) for v in clusters.values())}, max={max(len(v) for v in clusters.values())}, avg={sum(len(v) for v in clusters.values())/len(clusters):.1f}")
    
    # Stratified sampling
    random.seed(42)
    test_prompts = []
    train_prompts = []
    
    print("\nPerforming stratified sampling...")
    
    for cluster_id, cluster_prompts in clusters.items():
        # Shuffle within cluster
        random.shuffle(cluster_prompts)
        
        # Calculate how many from this cluster for test/train
        cluster_size = len(cluster_prompts)
        
        # Proportional allocation
        n_test = int(TEST_SIZE * cluster_size / len(prompts))
        n_train = int(TRAIN_SIZE * cluster_size / len(prompts))
        
        # Ensure at least 1 from each cluster if possible
        if cluster_size >= 2:
            n_test = max(1, n_test)
            n_train = max(1, n_train)
        
        # Ensure we don't exceed cluster size
        n_test = min(n_test, cluster_size)
        n_train = min(n_train, cluster_size - n_test)
        
        # Allocate
        test_prompts.extend(cluster_prompts[:n_test])
        train_prompts.extend(cluster_prompts[n_test:n_test+n_train])
    
    # Adjust to exact sizes by adding/removing prompts
    random.shuffle(test_prompts)
    random.shuffle(train_prompts)
    
    # If we're short, sample more from remaining pool
    if len(test_prompts) < TEST_SIZE:
        remaining = [p for cid, ps in clusters.items() for p in ps if p not in test_prompts and p not in train_prompts]
        random.shuffle(remaining)
        needed = TEST_SIZE - len(test_prompts)
        test_prompts.extend(remaining[:needed])
    
    if len(train_prompts) < TRAIN_SIZE:
        remaining = [p for cid, ps in clusters.items() for p in ps if p not in test_prompts and p not in train_prompts]
        random.shuffle(remaining)
        needed = TRAIN_SIZE - len(train_prompts)
        train_prompts.extend(remaining[:needed])
    
    # Trim to exact sizes
    test_prompts = test_prompts[:TEST_SIZE]
    train_prompts = train_prompts[:TRAIN_SIZE]
    
    print(f"\nFinal sizes:")
    print(f"  Test: {len(test_prompts)}")
    print(f"  Train: {len(train_prompts)}")
    
    # Verify stratification
    test_clusters = set(p['cluster_id'] for p in test_prompts)
    train_clusters = set(p['cluster_id'] for p in train_prompts)
    print(f"\nCluster coverage:")
    print(f"  Test covers {len(test_clusters)}/{len(clusters)} clusters")
    print(f"  Train covers {len(train_clusters)}/{len(clusters)} clusters")
    print(f"  Overlap: {len(test_clusters & train_clusters)} clusters")
    
    # Save
    print(f"\nSaving to {test_path}...")
    with open(test_path, 'w') as f:
        for p in test_prompts:
            f.write(json.dumps(p) + "\n")
    
    print(f"Saving to {train_path}...")
    with open(train_path, 'w') as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    
    print("\nDone!")
    print(f"\nRemaining {len(prompts) - len(test_prompts) - len(train_prompts)} prompts for prior covariance matrix")

if __name__ == "__main__":
    main()
