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

models_of_interest = [
    'google/gemini-3-pro-preview',
    'anthropic/claude-3.7-sonnet:thinking',
    'openai/gpt-4o',
    'meta-llama/llama-3-70b-instruct'
]

results = []
for cid in sorted(cluster_perf.keys()):
    perf = {m: np.mean(cluster_perf[cid][m]) for m in models_of_interest if m in cluster_perf[cid]}
    if len(perf) >= 2:
        best_model = max(perf, key=perf.get)
        worst_model = min(perf, key=perf.get)
        results.append((cid, best_model, perf[best_model], worst_model, perf[worst_model], perf))

# Find Phase 1 clusters where Gemini is BEST
gemini = 'google/gemini-3-pro-preview'
print("Clusters where Gemini is STRONG:")
for cid, best_m, best_v, worst_m, worst_v, perf in results:
    if best_m == gemini:
        print(f"Cluster {cid}: Gemini={perf[gemini]:.2f}, Best={best_m}, Others: { {k: f'{v:.2f}' for k,v in perf.items() if k != gemini} }")

# Find Phase 2 clusters where Claude is BEST and Gemini is WEAK
print("\nClusters where Claude is STRONG and Gemini is WEAK:")
for cid, best_m, best_v, worst_m, worst_v, perf in results:
    if best_m == 'anthropic/claude-3.7-sonnet:thinking' and gemini in perf and perf[gemini] < (best_v - 2.0):
         print(f"Cluster {cid}: Claude={perf['anthropic/claude-3.7-sonnet:thinking']:.2f}, Gemini={perf[gemini]:.2f}")
