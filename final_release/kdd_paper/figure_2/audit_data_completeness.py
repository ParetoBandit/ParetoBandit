
import json
import pandas as pd
from pathlib import Path

def main():
    root = Path('/Users/annette/repostitories/llm_jury')
    data_dir = root / 'banditgpt' / 'data'
    
    # 1. Load Expected Models (Registry)
    with open(root / 'banditgpt' / 'models.json') as f:
        models_data = json.load(f)['models']
        registry_ids = sorted([m['openrouter_id'] for m in models_data])
    
    print(f"[{len(registry_ids)}] MOdels in Registry (Expected):")
    # print(registry_ids)
    
    # 2. Load Actual Rewards Data
    model_counts = {mid: 0 for mid in registry_ids}
    total_rewards = 0
    
    print("\nScanning test_rewards.jsonl...")
    with open(data_dir / 'test_rewards.jsonl') as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get('ok'): continue
                
                mid = r.get('model_id')
                if mid in model_counts:
                    model_counts[mid] += 1
                total_rewards += 1
            except:
                continue
                
    # 3. Report
    print(f"Total Reward Records Found: {total_rewards}")
    
    print("\n--- Model Coverage Report ---")
    print(f"{'Model ID':<40} | {'Count':<10} | {'Status'}")
    print("-" * 65)
    
    warnings = 0
    zeros = 0
    
    for mid in registry_ids:
        count = model_counts[mid]
        status = "OK"
        if count == 0:
            status = "MISSING (0)"
            zeros += 1
        elif count < 100:
            status = "LOW DATA"
            warnings += 1
            
        print(f"{mid:<40} | {count:<10} | {status}")
        
    print("-" * 65)
    print(f"Fully Missing Models: {zeros}")
    print(f"Low Data Models (<100): {warnings}")

if __name__ == "__main__":
    main()
