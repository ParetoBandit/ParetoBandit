"""
Deduplicate models cache by keeping only the latest version within each model family.

For reasoning vs non-reasoning variants, we keep the reasoning version as it typically
represents the more capable configuration.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def get_model_priority(name: str) -> tuple:
    """
    Return a priority tuple for sorting models within a family.
    Higher values = keep this one (latest/best).
    
    Returns (version_score, reasoning_score, preview_score)
    """
    version_score = 0
    reasoning_score = 0
    preview_score = 0
    
    # Version scoring - higher versions preferred
    if '4.5' in name:
        version_score = 45
    elif '4.1' in name:
        version_score = 41
    elif '5.1' in name:
        version_score = 51
    elif '3.7' in name:
        version_score = 37
    elif '3.6' in name:
        version_score = 36
    elif '3.5' in name:
        version_score = 35
    elif '3.2' in name:
        version_score = 32
    elif '3.1' in name:
        version_score = 31
    elif '4' in name:
        version_score = 40
    elif '5' in name:
        version_score = 50
    elif '3' in name:
        version_score = 30
    elif '2.5' in name:
        version_score = 25
    elif '2.0' in name:
        version_score = 20
    
    # Reasoning preferred over non-reasoning
    if '(Reasoning)' in name or 'Reasoning' in name:
        reasoning_score = 1
    elif '(Non-reasoning)' in name:
        reasoning_score = 0
    else:
        reasoning_score = 0.5  # Unknown, middle priority
    
    # Latest preview/dated versions preferred
    if "Sep '25" in name or "2507" in name:
        preview_score = 3
    elif "May '25" in name or "0528" in name:
        preview_score = 2
    elif "Jan '25" in name:
        preview_score = 1
    elif "Nov '24" in name or "Dec '24" in name:
        preview_score = 0
    elif "Preview" in name:
        preview_score = 2
    
    # High effort reasoning preferred
    if '(high)' in name:
        reasoning_score += 0.3
    elif '(medium)' in name:
        reasoning_score += 0.2
    elif '(minimal)' in name or '(low)' in name:
        reasoning_score += 0.1
    
    return (version_score, reasoning_score, preview_score)


def get_model_family(name: str) -> str:
    """Extract the model family name for grouping."""
    
    # Gemini families
    if 'Gemini' in name:
        if '3 Pro' in name:
            return 'Gemini 3 Pro'
        elif '2.5 Flash-Lite' in name:
            return 'Gemini 2.5 Flash-Lite'
        elif '2.5 Flash' in name:
            return 'Gemini 2.5 Flash'
        elif '2.5 Pro' in name:
            return 'Gemini 2.5 Pro'
        elif '2.0' in name:
            return 'Gemini 2.0'
        return 'Gemini Other'
    
    # Claude families
    if 'Claude' in name:
        if 'Opus 4.5' in name or '4 Opus' in name:
            return 'Claude Opus 4'
        elif '4.5 Sonnet' in name or '4 Sonnet' in name:
            return 'Claude Sonnet 4'
        elif '4.5 Haiku' in name:
            return 'Claude Haiku 4'
        elif '3.7' in name:
            return 'Claude 3.7'
        elif '3.5' in name:
            if 'Sonnet' in name:
                return 'Claude 3.5 Sonnet'
            elif 'Haiku' in name:
                return 'Claude 3.5 Haiku'
        return 'Claude Other'
    
    # GPT families
    if 'GPT' in name or 'gpt' in name:
        if '5.1' in name:
            return 'GPT-5.1'
        elif '5 nano' in name:
            return 'GPT-5 nano'
        elif '5 mini' in name:
            return 'GPT-5 mini'
        elif 'GPT-5' in name:
            return 'GPT-5'
        elif 'gpt-oss' in name:
            return 'gpt-oss-120B'
        elif '4o' in name:
            return 'GPT-4o'
        elif '4.1' in name:
            return 'GPT-4.1'
        return 'GPT Other'
    
    # OpenAI reasoning models
    if name.startswith('o3') or name == 'o3':
        return 'o3'
    if name.startswith('o4'):
        return 'o4'
    if name.startswith('o1'):
        return 'o1'
    
    # Grok families
    if 'Grok' in name:
        if '4.1' in name:
            return 'Grok 4.1'
        elif '4' in name and 'Fast' in name:
            return 'Grok 4 Fast'
        elif '4' in name:
            return 'Grok 4'
        elif '3 mini' in name:
            return 'Grok 3 mini'
        elif '3' in name:
            return 'Grok 3'
        return 'Grok Other'
    
    # DeepSeek families
    if 'DeepSeek' in name:
        if 'R1 Distill' in name:
            return 'DeepSeek R1 Distill'
        elif 'R1 0528' in name:
            return 'DeepSeek R1 0528'
        elif 'R1' in name:
            return 'DeepSeek R1'
        elif 'V3.2' in name:
            return 'DeepSeek V3.2'
        elif 'V3.1 Terminus' in name:
            return 'DeepSeek V3.1 Terminus'
        elif 'V3.1' in name:
            return 'DeepSeek V3.1'
        elif 'V3 0324' in name:
            return 'DeepSeek V3 0324'
        elif 'V3' in name:
            return 'DeepSeek V3'
        return 'DeepSeek Other'
    
    # Llama families
    if 'Llama' in name or 'llama' in name:
        if '4 Maverick' in name:
            return 'Llama 4 Maverick'
        elif '4 Scout' in name:
            return 'Llama 4 Scout'
        elif '3.3' in name:
            return 'Llama 3.3'
        return 'Llama Other'
    
    # GLM families
    if 'GLM' in name:
        if '4.6' in name:
            return 'GLM-4.6'
        elif '4.5' in name:
            if 'Air' in name:
                return 'GLM-4.5-Air'
            return 'GLM-4.5'
        return 'GLM Other'
    
    # Qwen families
    if 'Qwen' in name:
        if '32B' in name:
            return 'Qwen3 32B'
        elif '14B' in name:
            return 'Qwen3 14B'
        elif '8B' in name:
            return 'Qwen3 8B'
        elif '4B' in name:
            return 'Qwen3 4B'
        return 'Qwen Other'
    
    # Gemma families
    if 'Gemma' in name:
        if '27B' in name:
            return 'Gemma 3 27B'
        elif '12B' in name:
            return 'Gemma 3 12B'
        elif '4B' in name:
            return 'Gemma 3 4B'
        return 'Gemma Other'
    
    # Mistral families
    if 'Mistral' in name or 'Ministral' in name:
        if 'Large' in name:
            return 'Mistral Large'
        elif 'Small 3.2' in name:
            return 'Mistral Small 3.2'
        elif 'Small 3.1' in name:
            return 'Mistral Small 3.1'
        elif 'Ministral 8B' in name:
            return 'Ministral 8B'
        elif 'Ministral 3B' in name:
            return 'Ministral 3B'
        return 'Mistral Other'
    
    # Phi families
    if 'Phi' in name:
        if 'Mini' in name:
            return 'Phi-4 Mini'
        return 'Phi-4'
    
    # Default: use the name itself
    return name


def deduplicate_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate models by keeping only the best version within each family.
    """
    from collections import defaultdict
    
    # Group by family
    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for model in models:
        name = model.get('name', '')
        family = get_model_family(name)
        families[family].append(model)
    
    # For each family, pick the best one
    deduplicated = []
    kept = []
    removed = []
    
    for family, family_models in sorted(families.items()):
        if len(family_models) == 1:
            # Only one model in family, keep it
            deduplicated.append(family_models[0])
            kept.append((family, family_models[0]['name']))
        else:
            # Multiple models - sort by priority and pick the best
            sorted_models = sorted(
                family_models,
                key=lambda m: get_model_priority(m['name']),
                reverse=True  # Higher priority first
            )
            best = sorted_models[0]
            deduplicated.append(best)
            kept.append((family, best['name']))
            for m in sorted_models[1:]:
                removed.append((family, m['name']))
    
    return deduplicated, kept, removed


def main():
    # Load models
    cache_path = Path('data/models_cache.json')
    with open(cache_path) as f:
        models = json.load(f)
    
    print(f"Original model count: {len(models)}")
    
    # Deduplicate
    deduplicated, kept, removed = deduplicate_models(models)
    
    print(f"Deduplicated model count: {len(deduplicated)}")
    print(f"\n{'='*60}")
    print("KEPT (one per family):")
    print('='*60)
    for family, name in sorted(kept):
        print(f"  {family}: {name}")
    
    print(f"\n{'='*60}")
    print("REMOVED (duplicates):")
    print('='*60)
    for family, name in sorted(removed):
        print(f"  {family}: {name}")
    
    # Backup original
    backup_path = Path('data/models_cache_backup.json')
    with open(backup_path, 'w') as f:
        json.dump(models, f, indent=2)
    print(f"\n✓ Original backed up to: {backup_path}")
    
    # Save deduplicated
    with open(cache_path, 'w') as f:
        json.dump(deduplicated, f, indent=2)
    print(f"✓ Deduplicated cache saved to: {cache_path}")


if __name__ == '__main__':
    main()

