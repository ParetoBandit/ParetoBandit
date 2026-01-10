#!/usr/bin/env python3
"""
Add raw_hle and empirical_hle fields to models.json and models_full.json.

- raw_hle: Humanity's Last Exam benchmark scores (0.04-0.28 range)
- empirical_hle: Success rates from dev set oracle rewards (0.76-0.98 range)
"""
import json
from pathlib import Path

# Historical HLE benchmark scores (from commit 6d10673)
raw_hle_scores = {
    'mistralai/ministral-3b': 0.053,
    'google/gemma-3-4b-it': 0.052,
    'google/gemma-3-12b-it': 0.048,
    'openai/gpt-oss-20b': 0.098,
    'google/gemma-3-27b-it': 0.047,
    'x-ai/grok-3-mini': 0.111,
    'google/gemini-2.5-flash-preview': 0.127,  # From gemini-2.5-flash-preview-09-2025
    'anthropic/claude-opus-4.5': 0.284,
    'openai/gpt-4.1': 0.046
}

# Empirical success rates from dev set (no data leakage)
empirical_hle_scores = {
    'mistralai/ministral-3b': 0.7665,
    'google/gemma-3-4b-it': 0.8967,
    'google/gemma-3-12b-it': 0.9483,
    'openai/gpt-oss-20b': 0.9497,
    'google/gemma-3-27b-it': 0.9550,
    'x-ai/grok-3-mini': 0.9792,
    'google/gemini-2.5-flash-preview': None,  # No data in oracle rewards
    'anthropic/claude-opus-4.5': 0.9775,
    'openai/gpt-4.1': 0.9808
}

def update_models_file(path):
    """Update a models JSON file with raw_hle and empirical_hle fields."""
    with open(path) as f:
        data = json.load(f)
    
    updated_count = 0
    for model in data['models']:
        model_id = model['openrouter_id']
        
        # Add raw_hle
        if model_id in raw_hle_scores:
            model['raw_hle'] = raw_hle_scores[model_id]
            updated_count += 1
        
        # Add empirical_hle (clipped to [0.01, 0.99] for bandit safety)
        if model_id in empirical_hle_scores and empirical_hle_scores[model_id] is not None:
            # Clip to avoid 0.0 and 1.0 (bandits hate absolute certainty)
            clipped = max(0.01, min(0.99, empirical_hle_scores[model_id]))
            model['empirical_hle'] = clipped
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated {path}")
    print(f"   Added raw_hle and empirical_hle to {updated_count} models")
    return updated_count

def main():
    print("=" * 70)
    print("ADDING raw_hle AND empirical_hle FIELDS")
    print("=" * 70)
    
    # Update both files
    total_updated = 0
    total_updated += update_models_file(Path('src/bandit_gpt/config/models.json'))
    total_updated += update_models_file(Path('src/bandit_gpt/config/models_full.json'))
    
    print("\n" + "=" * 70)
    print("FIELD DESCRIPTIONS")
    print("=" * 70)
    print("raw_hle:        Humanity's Last Exam benchmark scores (0.04-0.28)")
    print("empirical_hle:  Dev set success rates (0.76-0.98, clipped to [0.01, 0.99])")
    print("hle:            Legacy field (currently holds various values)")
    print("\n💡 Recommendation: Use 'empirical_hle' for warmup priors (Bernoulli success probability)")
    
if __name__ == "__main__":
    main()
