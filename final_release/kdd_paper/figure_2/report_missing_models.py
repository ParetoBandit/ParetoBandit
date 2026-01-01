
import json
from pathlib import Path

def main():
    root = Path('/Users/annette/repostitories/llm_jury')
    data_dir = root / 'banditgpt' / 'data'
    
    # 1. Load Registry
    with open(root / 'banditgpt' / 'models.json') as f:
        registry_models = sorted([m['openrouter_id'] for m in json.load(f)['models']])

    # 2. Load Prompts
    prompts = {36: [], 80: []}
    with open(data_dir / 'test_prompts.jsonl') as f:
        for l in f:
            p = json.loads(l)
            cid = p.get('cluster_id')
            if cid in [36, 80]: prompts[cid].append(p['prompt'])

    # 3. Load Available Pairs
    pairs = set()
    with open(data_dir / 'test_rewards.jsonl') as f:
        for l in f:
            r = json.loads(l)
            cid = r.get('cluster_id')
            if cid in [36, 80] and r.get('ok'):
                pairs.add((r['prompt'], r['model_id']))

    print(f"Checking Coverage for:")
    print(f"  Cluster 36 (Phase 1): {len(prompts[36])} prompts")
    print(f"  Cluster 80 (Phase 2): {len(prompts[80])} prompts")
    print("-" * 60)
    print(f"{'Model ID':<40} | {'Missing (C36)':<15} | {'Missing (C80)':<15}")
    print("-" * 60)
    
    gappy_models = []
    
    for m in registry_models:
        missing_36 = sum(1 for p in prompts[36] if (p, m) not in pairs)
        missing_80 = sum(1 for p in prompts[80] if (p, m) not in pairs)
        
        if missing_36 > 0 or missing_80 > 0:
            gappy_models.append(m)
            mark = "(!)" if missing_36 > 0 else ""
            print(f"{m:<40} | {missing_36}/{len(prompts[36])} {mark:<8} | {missing_80}/{len(prompts[80])}")

    print("-" * 60)
    print(f"Total Gappy Models: {len(gappy_models)}")

if __name__ == "__main__":
    main()
