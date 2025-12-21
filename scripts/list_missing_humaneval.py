import json

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']
missing_humaneval = [m['name'] for m in models if m.get('humaneval_score') is None]

print(f"Models missing HumanEval ({len(missing_humaneval)}):")
for name in sorted(missing_humaneval):
    print(f"- {name}")
