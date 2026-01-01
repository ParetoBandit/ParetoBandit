import json
from collections import defaultdict
import numpy as np

files = [
    '/Users/annette/repostitories/llm_jury/banditgpt/data/train_rewards.jsonl',
    '/Users/annette/repostitories/llm_jury/banditgpt/data/test_rewards.jsonl'
]

cluster_perf = defaultdict(lambda: defaultdict(list))

for fpath in files:
    with open(fpath) as f:
        for line in f:
            r = json.loads(line)
            if 'cluster_id' in r and 'model_id' in r:
                logit = r.get('reward_logit')
                if logit is not None:
                    cluster_perf[r['cluster_id']][r['model_id']].append(logit)

for cid in [37, 80]:
    m1 = 'openai/gpt-4o'
    m2 = 'anthropic/claude-3.7-sonnet:thinking'
    m3 = 'google/gemini-3-pro-preview'
    print(f"Cluster {cid}:")
    for mid in [m1, m2, m3]:
        vals = cluster_perf[cid].get(mid, [])
        if vals:
            mean_logit = np.mean(vals)
            sigmoid_mean = np.mean([1/(1+np.exp(-v)) for v in vals])
            print(f"  {mid}: count={len(vals)}, mean_logit={mean_logit:.2f}, sigmoid_mean={sigmoid_mean:.3f}")
