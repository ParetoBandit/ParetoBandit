import json

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']
total_models = len(models)

humaneval_count = sum(1 for m in models if m.get('humaneval_score') is not None)
livecodebench_count = sum(1 for m in models if m.get('livecodebench') is not None)
mbpp_count = sum(1 for m in models if m.get('mbpp_score') is not None)

print(f"Total models: {total_models}")
print(f"HumanEval coverage: {humaneval_count}/80")
print(f"LiveCodeBench coverage: {livecodebench_count}/80")
print(f"MBPP coverage: {mbpp_count}/80")

missing_lcb = [m['name'] for m in models if m.get('livecodebench') is None]
print(f"\nMissing LiveCodeBench ({len(missing_lcb)}):")
for name in sorted(missing_lcb):
    print(f"- {name}")
