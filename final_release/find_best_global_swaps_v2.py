import json
from collections import defaultdict
import numpy as np

files = [
    '/Users/annette/repostitories/llm_jury/banditgpt/data/train_rewards.jsonl',
    '/Users/annette/repostitories/llm_jury/banditgpt/data/test_rewards.jsonl'
]

cluster_perf = defaultdict(lambda: defaultdict(list))

print("Merging reward data...")
for fpath in files:
    with open(fpath) as f:
        for line in f:
            r = json.loads(line)
            if 'cluster_id' in r and 'model_id' in r:
                logit = r.get('reward_logit')
                if logit is not None:
                    cluster_perf[r['cluster_id']][r['model_id']].append(logit)

# Filter for models with high coverage
model_counts = defaultdict(int)
for cid, models in cluster_perf.items():
    for mid in models:
        model_counts[mid] += 1

# Reliable models are those appearing in many clusters
reliable_models = [mid for mid, count in model_counts.items() if count > 70]
print(f"Analyzing {len(reliable_models)} reliable models across {len(cluster_perf)} clusters...")

swaps = []
for m1 in reliable_models:
    for m2 in reliable_models:
        if m1 >= m2: continue  # Avoid duplicates
        
        m1_wins = []
        m2_wins = []
        for cid in cluster_perf:
            if m1 in cluster_perf[cid] and m2 in cluster_perf[cid]:
                v1 = np.mean([x for x in cluster_perf[cid][m1] if x is not None])
                v2 = np.mean([x for x in cluster_perf[cid][m2] if x is not None])
                diff = v1 - v2
                if diff > 1.2: m1_wins.append((cid, diff))
                if diff < -1.2: m2_wins.append((cid, diff))
        
        if m1_wins and m2_wins:
            swaps.append({
                'm1': m1,
                'm2': m2,
                'm1_wins': m1_wins,
                'm2_wins': m2_wins,
                'score': len(m1_wins) * len(m2_wins) * (max([w[1] for w in m1_wins]) + max([-w[1] for w in m2_wins]))
            })

swaps.sort(key=lambda x: x['score'], reverse=True)

for s in swaps[:10]:
    print(f"\n--- {s['m1']} vs {s['m2']} ---")
    print(f"M1 wins in {len(s['m1_wins'])} clusters. Top: {sorted(s['m1_wins'], key=lambda x: x[1], reverse=True)[:3]}")
    print(f"M2 wins in {len(s['m2_wins'])} clusters. Top: {sorted(s['m2_wins'], key=lambda x: -x[1], reverse=True)[:3]}")
