import json

with open("models.json", "r") as f:
    models = json.load(f)["models"]

hle_rankings = []
for m in models:
    hle = m.get("hle", 0)
    hle_rankings.append((m["openrouter_id"], hle))

hle_rankings.sort(key=lambda x: x[1], reverse=True)

print("Top 10 Models by HLE Score:")
for i, (m_id, hle) in enumerate(hle_rankings[:10]):
    print(f"{i+1}. {m_id}: {hle}")
