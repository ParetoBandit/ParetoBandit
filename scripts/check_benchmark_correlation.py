import json
import numpy as np
from scipy.stats import pearsonr

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']

hle = []
math = []
code = []
knowledge = []

for m in models:
    h = m.get('hle')
    ma = m.get('math_500')
    co = m.get('livecodebench')
    kn = m.get('mmlu_pro')
    
    if all(x is not None for x in [h, ma, co, kn]):
        hle.append(h)
        math.append(ma)
        code.append(co)
        knowledge.append(kn)

print(f"Analyzing {len(hle)} models with full benchmark data...")
print(f"Correlation (HLE vs MATH-500): {pearsonr(hle, math)[0]:.4f}")
print(f"Correlation (HLE vs LiveCodeBench): {pearsonr(hle, code)[0]:.4f}")
print(f"Correlation (HLE vs MMLU-Pro): {pearsonr(hle, knowledge)[0]:.4f}")
