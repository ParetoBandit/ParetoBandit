import json
import numpy as np
from scipy.stats import pearsonr

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']

# Normalize all scores to [0, 1]
def normalize(val, scale=1.0):
    if val is None: return None
    return np.clip(float(val) / scale, 0.0, 1.0)

hle = []
math_ood = []
code_ood = []
knowledge_ood = []
composite_1 = [] # Mean of HLE, MMLU-Pro, GPQA (General Reasoning)
composite_2 = [] # Mean of all available (Broad)

for m in models:
    h = normalize(m.get('hle'))
    ma = normalize(m.get('math_500'))
    co = normalize(m.get('livecodebench'))
    kn = normalize(m.get('mmlu_pro'))
    gpqa = normalize(m.get('gpqa'))
    mix = normalize(m.get('mixeval_score'), 100.0)
    
    if all(x is not None for x in [h, ma, co, kn, gpqa, mix]):
        hle.append(h)
        math_ood.append(ma)
        code_ood.append(co)
        knowledge_ood.append(kn)
        
        # Composite 1: General Reasoning (excluding the OOD targets themselves)
        c1 = (h + normalize(m.get('mmlu_pro')) + gpqa) / 3.0
        composite_1.append(c1)
        
        # Composite 2: Broad Quality (including MixEval)
        c2 = (h + normalize(m.get('mmlu_pro')) + gpqa + mix) / 4.0
        composite_2.append(c2)

print(f"Analyzing {len(hle)} models...")
print("-" * 30)
print("Correlation with MATH-500:")
print(f"  HLE: {pearsonr(hle, math_ood)[0]:.4f}")
print(f"  Comp1 (HLE+MMLU+GPQA): {pearsonr(composite_1, math_ood)[0]:.4f}")
print(f"  Comp2 (HLE+MMLU+GPQA+Mix): {pearsonr(composite_2, math_ood)[0]:.4f}")
print("-" * 30)
print("Correlation with LiveCodeBench:")
print(f"  HLE: {pearsonr(hle, code_ood)[0]:.4f}")
print(f"  Comp1: {pearsonr(composite_1, code_ood)[0]:.4f}")
print(f"  Comp2: {pearsonr(composite_2, code_ood)[0]:.4f}")
print("-" * 30)
print("Correlation with MMLU-Pro:")
print(f"  HLE: {pearsonr(hle, knowledge_ood)[0]:.4f}")
print(f"  Comp1: {pearsonr(composite_1, knowledge_ood)[0]:.4f}")
print(f"  Comp2: {pearsonr(composite_2, knowledge_ood)[0]:.4f}")
