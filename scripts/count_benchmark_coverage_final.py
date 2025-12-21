import json

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']
total_models = len(models)

math_500_count = sum(1 for m in models if m.get('math_500') is not None)
humaneval_count = sum(1 for m in models if m.get('humaneval_score') is not None)
mmlu_pro_count = sum(1 for m in models if m.get('mmlu_pro') is not None)

all_three_count = sum(1 for m in models if 
                      m.get('math_500') is not None and 
                      m.get('humaneval_score') is not None and 
                      m.get('mmlu_pro') is not None)

print(f"Total models in final_release/models.json: {total_models}")
print(f"Models with MATH-500: {math_500_count}")
print(f"Models with HumanEval: {humaneval_count}")
print(f"Models with MMLU-Pro: {mmlu_pro_count}")
print(f"Models with all three: {all_three_count}")
