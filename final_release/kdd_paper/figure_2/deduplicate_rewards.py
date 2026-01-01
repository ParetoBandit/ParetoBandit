#!/usr/bin/env python3
"""
Deduplicate rewards.jsonl files by (model_id, prompt) calling.
Keeps the LATEST entry (highest timestamp).
"""

import json
import shutil
from pathlib import Path
import time

DATA_DIR = Path('/Users/annette/repostitories/llm_jury/final_release/data')

def deduplicate_file(filename):
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"Skipping {filename} (not found)")
        return

    print(f"Processing {filename}...")
    
    # Read all records
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
    
    print(f"  Total records: {len(records)}")
    
    # Deduplicate
    # Key: (model_id, prompt, cluster_id) -> record
    # Actually, uniqueness should probably be (model_id, prompt). Cluster ID shouldn't change for the same prompt.
    
    unique_map = {}
    duplicates = 0
    
    for r in records:
        key = (r.get('model_id'), r.get('prompt'))
        if not key[0] or not key[1]:
            continue
            
        # If exists, keep the one with higher timestamp
        if key in unique_map:
            duplicates += 1
            existing = unique_map[key]
            # Use 'ts' if available, else default to 0
            ts_new = r.get('ts', 0)
            ts_old = existing.get('ts', 0)
            
            if ts_new > ts_old:
                unique_map[key] = r
        else:
            unique_map[key] = r
            
    print(f"  Unique records: {len(unique_map)}")
    print(f"  Duplicates removed: {duplicates}")
    
    # Write back
    backup_path = file_path.with_suffix('.jsonl.bak')
    shutil.copy(file_path, backup_path)
    print(f"  Backup saved to {backup_path.name}")
    
    with open(file_path, 'w') as f:
        for r in unique_map.values():
            f.write(json.dumps(r) + '\n')
            
    print(f"  ✓ Saved deduplicated file")

if __name__ == "__main__":
    deduplicate_file('train_rewards.jsonl')
    print("-" * 30)
    deduplicate_file('test_rewards.jsonl')
