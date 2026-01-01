import json
from collections import defaultdict
import numpy as np

rewards_path = '/Users/annette/repostitories/llm_jury/banditgpt/data/test_rewards.jsonl'
cluster_perf = defaultdict(lambda: defaultdict(list))

with open(rewards_path) as f:
    for line in f:
        r = json.loads(line)
        if 'cluster_id' in r and 'model_id' in r:
            cluster_perf[r['cluster_id']][r['model_id']].append(r['reward_logit'])

# Filter for models with enough coverage
model_counts = defaultdict(int)
for cid, models in cluster_perf.items():
    for mid in models:
        model_counts[mid] += 1

reliable_models = [mid for mid, count in model_counts.items() if count > 50]
print(f"Analyzing {len(reliable_models)} reliable models...")

swaps = []
for m1 in reliable_models:
    for m2 in reliable_models:
        if m1 == m2: continue
        
        m1_wins = []
        m2_wins = []
        for cid in cluster_perf:
            if m1 in cluster_perf[cid] and m2 in cluster_perf[cid]:
                diff = np.mean(cluster_perf[cid][m1]) - np.mean(cluster_perf[cid][m2])
                if diff > 1.0: m1_wins.append(cid)
                if diff < -1.0: m2_wins.append(cid)
        
        if len(m1_wins) >= 1 and len(m2_wins) >= 1:
            swaps.append((m1, m2, m1_wins, m2_wins))

# Sort by total difference to find the most dramatic swap
swaps.sort(key=lambda x: len(x[2]) + len(x[3]), reverse=True)

for m1, m2, w1, w2 in swaps[:5]:
    print(f"\n--- {m1} vs {m2} ---")
    print(f"M1 wins in {len(w1)} clusters: {w1[:5]}")
    print(f"M2 wins in {len(w2)} clusters: {w2[:5]}")
