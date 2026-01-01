import json
from collections import defaultdict
import numpy as np

rewards_path = '/Users/annette/repostitories/llm_jury/banditgpt/data/test_rewards.jsonl'
cluster_perf = defaultdict(lambda: defaultdict(list))

print("Loading rewards...")
with open(rewards_path) as f:
    for line in f:
        r = json.loads(line)
        if 'cluster_id' in r and 'model_id' in r:
            cluster_perf[r['cluster_id']][r['model_id']].append(r['reward_logit'])

gemini = 'google/gemini-3-pro-preview'
claude = 'anthropic/claude-3.7-sonnet:thinking'

results = []
for cid in sorted(cluster_perf.keys()):
    models = cluster_perf[cid]
    if gemini in models and claude in models:
        g_avg = np.mean(models[gemini])
        c_avg = np.mean(models[claude])
        results.append((cid, g_avg, c_avg, g_avg - c_avg))

print("\n--- Gemini Favored ---")
for cid, g, c, diff in sorted(results, key=lambda x: x[3], reverse=True)[:10]:
    print(f"Cluster {cid}: Gemini={g:.2f}, Claude={c:.2f}, Diff={diff:+.2f}")

print("\n--- Claude Favored ---")
for cid, g, c, diff in sorted(results, key=lambda x: x[3])[:10]:
    print(f"Cluster {cid}: Gemini={g:.2f}, Claude={c:.2f}, Diff={diff:+.2f}")
