import json
from pathlib import Path

# Historical HLE scores from commit 6d10673 (80-model registry)
historical_hle = {
    'mistralai/ministral-3b': 0.053,
    'google/gemma-3-4b-it': 0.052,
    'google/gemma-3-12b-it': 0.048,
    'openai/gpt-oss-20b': 0.098,
    'google/gemma-3-27b-it': 0.047,
    'x-ai/grok-3-mini': 0.111,
    'google/gemini-2.5-flash-preview': None,  # Not in old registry
    'anthropic/claude-opus-4.5': 0.284,
    'openai/gpt-4.1': 0.046
}

# Update models.json
models_path = Path('src/bandit_gpt/config/models.json')
with open(models_path) as f:
    data = json.load(f)

updated_count = 0
for model in data['models']:
    model_id = model['openrouter_id']
    if model_id in historical_hle and historical_hle[model_id] is not None:
        old_hle = model.get('hle')
        model['hle'] = historical_hle[model_id]
        print(f"Updated {model_id}: {old_hle} → {historical_hle[model_id]}")
        updated_count += 1

with open(models_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Updated {updated_count} models in models.json")

# Update models_full.json
models_full_path = Path('src/bandit_gpt/config/models_full.json')
with open(models_full_path) as f:
    data = json.load(f)

updated_count = 0
for model in data['models']:
    model_id = model['openrouter_id']
    if model_id in historical_hle and historical_hle[model_id] is not None:
        old_hle = model.get('hle')
        model['hle'] = historical_hle[model_id]
        updated_count += 1

with open(models_full_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ Updated {updated_count} models in models_full.json")
