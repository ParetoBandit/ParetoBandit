import json
import numpy as np
from scipy.stats import pearsonr

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']

def normalize(val, scale=1.0):
    if val is None: return None
    return np.clip(float(val) / scale, 0.0, 1.0)

q_indices = []
reasoning_scores = []
math_ood = []
code_ood = []
knowledge_ood = []

for m in models:
    q = m.get('aa_quality_index')
    r = m.get('reasoning_score')
    ma = normalize(m.get('math_500'))
    co = normalize(m.get('livecodebench'))
    kn = normalize(m.get('mmlu_pro'))
    
    if all(x is not None for x in [q, r, ma, co, kn]):
        q_indices.append(q)
        reasoning_scores.append(r)
        math_ood.append(ma)
        code_ood.append(co)
        knowledge_ood.append(kn)

print(f"Correlation Analysis ({len(q_indices)} models):")
print("-" * 30)
print("vs MATH-500:")
print(f"  aa_quality_index: {pearsonr(q_indices, math_ood)[0]:.4f}")
print(f"  reasoning_score: {pearsonr(reasoning_scores, math_ood)[0]:.4f}")
print("-" * 30)
print("vs LiveCodeBench:")
print(f"  aa_quality_index: {pearsonr(q_indices, code_ood)[0]:.4f}")
print(f"  reasoning_score: {pearsonr(reasoning_scores, code_ood)[0]:.4f}")
print("-" * 30)
print("vs MMLU-Pro:")
print(f"  aa_quality_index: {pearsonr(q_indices, knowledge_ood)[0]:.4f}")
print(f"  reasoning_score: {pearsonr(reasoning_scores, knowledge_ood)[0]:.4f}")
