import json
from pathlib import Path

# Success rates from DEV set (no data leakage)
empirical_hle = {
    'mistralai/ministral-3b': 0.7665,
    'google/gemma-3-4b-it': 0.8967,
    'google/gemma-3-12b-it': 0.9483,
    'openai/gpt-oss-20b': 0.9497,
    'google/gemma-3-27b-it': 0.9550,
    'x-ai/grok-3-mini': 0.9792,
    'anthropic/claude-opus-4.5': 0.9775,
    'openai/gpt-4.1': 0.9808,
}

# Update models.json
for path in ['src/bandit_gpt/config/models.json', 'src/bandit_gpt/config/models_full.json']:
    with open(path) as f:
        data = json.load(f)
    
    for model in data['models']:
        model_id = model['openrouter_id']
        if model_id in empirical_hle:
            old_hle = model.get('hle')
            model['hle'] = empirical_hle[model_id]
            print(f"Updated {model_id}: {old_hle} → {empirical_hle[model_id]}")
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {path}")
