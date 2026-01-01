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

gemini = 'google/gemini-3-pro-preview'
claude = 'anthropic/claude-3.7-sonnet:thinking'

results = []
for cid in sorted(cluster_perf.keys()):
    perf = cluster_perf[cid]
    if gemini in perf and claude in perf:
        g = np.mean(perf[gemini])
        c = np.mean(perf[claude])
        if g > c:
            results.append((cid, g, c))

if not results:
    print("Gemini never beats Claude in the test set.")
else:
    print("Clusters where Gemini > Claude:")
    for cid, g, c in results:
        print(f"Cluster {cid}: Gemini={g:.2f}, Claude={c:.2f}, Diff={g-c:+.2f}")
