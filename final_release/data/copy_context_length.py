#!/usr/bin/env python3
"""
Copy context_length from models_cache_with_hle.json to models.json
"""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    
    # Load both files
    models_path = base_dir / 'models.json'
    cache_path = base_dir / 'data' / 'models_cache_with_hle.json'
    
    with open(models_path) as f:
        models_data = json.load(f)
    
    with open(cache_path) as f:
        cache_data = json.load(f)
    
    print(f"Models in models.json: {len(models_data['models'])}")
    print(f"Models in cache: {len(cache_data['models'])}")
    
    # Create lookup by openrouter_id
    cache_lookup = {}
    for model in cache_data['models']:
        if 'openrouter_id' in model:
            cache_lookup[model['openrouter_id']] = model
        elif 'id' in model:
            cache_lookup[model['id']] = model
    
    # Copy context_length
    updated_count = 0
    for model in models_data['models']:
        model_id = model.get('openrouter_id')
        if model_id and model_id in cache_lookup:
            cache_model = cache_lookup[model_id]
            if 'context_length' in cache_model and cache_model['context_length']:
                model['context_length'] = cache_model['context_length']
                updated_count += 1
    
    print(f"\nUpdated {updated_count} models with context_length")
    
    # Show distribution
    context_lengths = [m['context_length'] for m in models_data['models'] if 'context_length' in m]
    if context_lengths:
        print(f"Context length range: {min(context_lengths)} - {max(context_lengths)}")
        
        # Group by size
        small = sum(1 for c in context_lengths if c <= 8192)
        medium = sum(1 for c in context_lengths if 8192 < c <= 32768)
        large = sum(1 for c in context_lengths if 32768 < c <= 128000)
        xlarge = sum(1 for c in context_lengths if c > 128000)
        
        print(f"\nDistribution:")
        print(f"  Small (≤8K): {small}")
        print(f"  Medium (8K-32K): {medium}")
        print(f"  Large (32K-128K): {large}")
        print(f"  XLarge (>128K): {xlarge}")
    
    # Backup and save
    backup_path = models_path.with_suffix('.json.backup3')
    models_path.rename(backup_path)
    print(f"\nCreated backup: {backup_path}")
    
    with open(models_path, 'w') as f:
        json.dump(models_data, f, indent=2)
    
    print(f"✓ Updated {models_path}")

if __name__ == '__main__':
    main()
