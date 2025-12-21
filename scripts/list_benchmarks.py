import json

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

model = data['models'][0]
print("Available benchmarks in models.json:")
for key in sorted(model.keys()):
    if any(suffix in key for suffix in ['score', 'pro', '500', 'bench', 'aime', 'gpqa', 'hle']):
        print(f"- {key}: {model[key]}")
