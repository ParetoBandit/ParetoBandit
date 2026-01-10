#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

def verify():
    path = Path('data/lmsys_distribution_1k.jsonl')
    if not path.exists():
        print(f"❌ Error: {path} not found")
        return

    counts = Counter()
    total = 0
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            cat = data.get('target_category', 'Unknown')
            counts[cat] += 1
            total += 1

    print(f"📊 Distribution for {total} prompts:")
    for cat, count in counts.items():
        pct = (count / total) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Check targets
    targets = {
        'Frontier (Hard)': 200,
        'Contentious': 300,
        'Commodity (Easy)': 500
    }
    
    all_ok = True
    for cat, target in targets.items():
        if counts[cat] != target:
            print(f"⚠️  Mismatch in {cat}: expected {target}, got {counts[cat]}")
            all_ok = False
            
    if all_ok and total == 1000:
        print("\n✅ Distribution verified successfully!")
    else:
        print("\n❌ Distribution verification failed.")

if __name__ == "__main__":
    verify()
