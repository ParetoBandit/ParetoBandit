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

m1 = 'google/gemini-3-pro-preview'
m2 = 'anthropic/claude-3.7-sonnet:thinking'
m3 = 'openai/gpt-4o'

print("Searching for Gemini > Others...")
for cid in range(100):
     if m1 in cluster_perf[cid] and m3 in cluster_perf[cid]:
          g = np.mean(cluster_perf[cid][m1])
          gp = np.mean(cluster_perf[cid][m3])
          if g > gp:
               print(f"Cluster {cid}: Gemini={g:.2f}, GPT-4o={gp:.2f}")

print("\nSearching for Claude > Others...")
for cid in range(100):
     if m2 in cluster_perf[cid] and m1 in cluster_perf[cid]:
          c = np.mean(cluster_perf[cid][m2])
          g = np.mean(cluster_perf[cid][m1])
          if c > g + 2.0:
               print(f"Cluster {cid}: Claude={c:.2f}, Gemini={g:.2f}")
