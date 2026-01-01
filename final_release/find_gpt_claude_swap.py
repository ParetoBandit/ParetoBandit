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

m1 = 'openai/gpt-4o'
m2 = 'anthropic/claude-3.7-sonnet:thinking'

m1_wins = []
m2_wins = []

for cid in sorted(cluster_perf.keys()):
    if m1 in cluster_perf[cid] and m2 in cluster_perf[cid]:
        v1 = np.mean(cluster_perf[cid][m1])
        v2 = np.mean(cluster_perf[cid][m2])
        if v1 > v2 + 0.5: m1_wins.append((cid, v1, v2))
        elif v2 > v1 + 0.5: m2_wins.append((cid, v1, v2))

print(f"GPT-4o wins in {len(m1_wins)} clusters:")
for cid, v1, v2 in m1_wins[:10]:
    print(f"  Cluster {cid}: GPT-4o={v1:.2f}, Claude={v2:.2f}")

print(f"\nClaude wins in {len(m2_wins)} clusters:")
for cid, v1, v2 in m2_wins[:10]:
    print(f"  Cluster {cid}: GPT-4o={v1:.2f}, Claude={v2:.2f}")
