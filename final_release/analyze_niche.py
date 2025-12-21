import json
from collections import defaultdict

coding_clusters = [350, 72, 412, 118, 161, 453, 41, 164, 36, 463]

rewards = defaultdict(list)

with open("data/test_rewards.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if data["cluster_id"] in coding_clusters:
            if data.get("reward_logit") is not None:
                rewards[data["model_id"]].append(data["reward_logit"])

avg_rewards = {m: sum(r)/len(r) for m, r in rewards.items() if len(r) > 0}
sorted_models = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)

print("Top models in Coding Niche:")
for m, r in sorted_models[:10]:
    print(f"{m}: {r:.4f}")

gemini_id = "google/gemini-3-pro-preview"
if gemini_id in avg_rewards:
    print(f"\nGemini 3 Pro Reward: {avg_rewards[gemini_id]:.4f}")
else:
    # Try to find a similar gemini
    for m in avg_rewards:
        if "gemini" in m:
            print(f"Found {m}: {avg_rewards[m]:.4f}")
