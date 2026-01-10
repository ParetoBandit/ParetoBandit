
import json
import sys
from pathlib import Path

def verify_splits():
    # Define verified absolute path
    splits_path = Path("/Users/annette/repostitories/banditGPT/experiments/01_effectiveness/results/splits.json")
    
    print(f"🔍 Verifying splits file: {splits_path}")
    
    if not splits_path.exists():
        print("❌ splits.json not found!")
        sys.exit(1)
        
    with open(splits_path, 'r') as f:
        data = json.load(f)
        
    dev_set = set(data['dev_pool'])
    holdout_set = set(data['holdout_pool'])
    
    print(f"📦 Dev Set Size:     {len(dev_set)}")
    print(f"📦 Holdout Set Size: {len(holdout_set)}")
    
    # Check Intersection
    intersection = dev_set.intersection(holdout_set)
    intersection_size = len(intersection)
    
    print("-" * 30)
    print(f"⚠️  Intersection Size: {intersection_size}")
    
    if intersection_size == 0:
        print("✅ SUCCESS: Sets are strictly disjoint. No data leakage detected.")
        sys.exit(0)
    else:
        print("❌ CRITICAL FAILURE: Data leakage detected!")
        print(f"   Overlapping IDs: {list(intersection)[:5]}...")
        sys.exit(1)

if __name__ == "__main__":
    verify_splits()
