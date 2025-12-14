#!/usr/bin/env python3
"""
Map OpenCompass model names to models_cache.json names.
"""

from huggingface_hub import list_repo_files
import os
from dotenv import load_dotenv
import json
import re
from pathlib import Path

# Load environment
load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')

def normalize_name(s):
    """Normalize name for matching"""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def fuzzy_match(opencompass_name, cache_name):
    """Check if names likely refer to same model"""
    oc_norm = normalize_name(opencompass_name)
    cache_norm = normalize_name(cache_name)
    
    # Exact match after normalization
    if oc_norm == cache_norm:
        return 10  # High confidence
    
    # Check key model families
    families = {
        'claude': ['claude'],
        'gpt': ['gpt'],
        'gemini': ['gemini'],
        'llama': ['llama', 'meta'],
        'qwen': ['qwen', 'qwq'],
        'deepseek': ['deepseek'],
        'mistral': ['mistral', 'mixtral'],
        'glm': ['glm'],
        'gemma': ['gemma'],
    }
    
    for family, keywords in families.items():
        oc_has = any(kw in oc_norm for kw in keywords)
        cache_has = any(kw in cache_norm for kw in keywords)
        
        if oc_has and cache_has:
            # Same family - check version/size
            # Extract numbers
            oc_nums = re.findall(r'\d+', opencompass_name)
            cache_nums = re.findall(r'\d+', cache_name)
            
            # If same numbers, likely same model
            if oc_nums and cache_nums and oc_nums[0] == cache_nums[0]:
                return 8  # Good confidence
            
            # Partial match
            return 5
    
    return 0  # No match

def main():
    # Get OpenCompass models
    print("Loading OpenCompass models...")
    files = list(list_repo_files(
        'opencompass/compass_academic_predictions',
        repo_type='dataset',
        token=HF_TOKEN
    ))
    gpqa_files = [f for f in files if 'GPQA_diamond' in f and f.endswith('.json')]
    opencompass_models = [f.split('/')[-1].replace('.json', '') for f in gpqa_files]
    
    # Get cache models
    cache_path = Path(__file__).parent.parent.parent / "data" / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    cache_models = {m['name']: m for m in data['models']}
    
    print(f"✓ Found {len(opencompass_models)} OpenCompass models")
    print(f"✓ Found {len(cache_models)} models in cache\n")
    
    # Find matches
    mappings = {}
    confidence_scores = {}
    
    for oc_model in opencompass_models:
        best_match = None
        best_score = 0
        
        for cache_name in cache_models.keys():
            score = fuzzy_match(oc_model, cache_name)
            if score > best_score:
                best_score = score
                best_match = cache_name
        
        if best_score >= 5:  # Threshold for accepting match
            mappings[oc_model] = best_match
            confidence_scores[oc_model] = best_score
    
    # Display results
    print("="*80)
    print(f"MATCHED: {len(mappings)}/{len(opencompass_models)} models")
    print("="*80)
    print()
    
    # Group by family
    families = {}
    for oc, cache in sorted(mappings.items()):
        family = oc.split('-')[0].split('_')[0]
        if family not in families:
            families[family] = []
        families[family].append((oc, cache, confidence_scores[oc]))
    
    for family, models in sorted(families.items()):
        print(f"{family.upper()}:")
        for oc, cache, conf in sorted(models, key=lambda x: -x[2])[:10]:
            conf_str = "★" * (conf // 2)
            print(f"  {oc:50s} → {cache:40s} {conf_str}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        print()
    
    # Save Python mapping code
    output_path = Path(__file__).parent / "opencompass_name_mappings.py"
    with open(output_path, 'w') as f:
        f.write("# Auto-generated model name mappings\n\n")
        f.write("OPENCOMPASS_TO_CACHE = {\n")
        for oc, cache in sorted(mappings.items()):
            f.write(f"    '{oc}': '{cache}',\n")
        f.write("}\n")
    
    print(f"✓ Saved mappings to: {output_path}")
    print(f"\n{'='*80}")
    print(f"SUMMARY: {len(mappings)} models matched")
    print(f"{'='*80}")
    
    # Unmatched
    unmatched = [m for m in opencompass_models if m not in mappings]
    if unmatched:
        print(f"\n⚠️  {len(unmatched)} unmatched OpenCompass models:")
        for model in sorted(unmatched)[:15]:
            print(f"  - {model}")
        if len(unmatched) > 15:
            print(f"  ... and {len(unmatched) - 15} more")

if __name__ == '__main__':
    main()
