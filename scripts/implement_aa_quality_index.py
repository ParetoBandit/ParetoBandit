import json
import numpy as np
from scipy.stats import pearsonr

with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'r') as f:
    data = json.load(f)

models = data['models']

def normalize(val, scale=1.0):
    if val is None: return 0.0 # Treat missing as 0 for composite
    return np.clip(float(val) / scale, 0.0, 1.0)

results = []
for m in models:
    # Core Reasoning
    hle = normalize(m.get('hle'))
    mmlu = normalize(m.get('mmlu_pro'))
    gpqa = normalize(m.get('gpqa'))
    
    # Specialized
    math = normalize(m.get('math_500'))
    code = normalize(m.get('livecodebench'))
    
    # General/Hard
    mix_hard = normalize(m.get('mixeval_hard_score'), 100.0)
    
    # AA Quality Index (Composite)
    # We weight reasoning and hard benchmarks more heavily
    q_index = (
        0.25 * hle + 
        0.20 * mmlu + 
        0.15 * gpqa + 
        0.15 * math + 
        0.15 * code + 
        0.10 * mix_hard
    )
    
    m['aa_quality_index'] = q_index
    results.append(m)

# Check correlations of this new index
q_indices = [m['aa_quality_index'] for m in results]
math_ood = [normalize(m.get('math_500')) for m in results]
code_ood = [normalize(m.get('livecodebench')) for m in results]
knowledge_ood = [normalize(m.get('mmlu_pro')) for m in results]

print(f"AA Quality Index Correlations:")
print(f"  vs MATH-500: {pearsonr(q_indices, math_ood)[0]:.4f}")
print(f"  vs LiveCodeBench: {pearsonr(q_indices, code_ood)[0]:.4f}")
print(f"  vs MMLU-Pro: {pearsonr(q_indices, knowledge_ood)[0]:.4f}")

# Save the updated models.json with the new index
with open('/Users/annette/repostitories/llm_jury/final_release/models.json', 'w') as f:
    json.dump({"models": results}, f, indent=2)

print("\nUpdated final_release/models.json with 'aa_quality_index'")
