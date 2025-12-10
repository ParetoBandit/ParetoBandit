#!/usr/bin/env python3
"""
Check which models don't have complete SummEdits scores across all domains.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT))

# Available SummEdits domains
SUMMEDITS_DOMAINS = [
    "news",
    "podcast", 
    "billsum",
    "samsum",
    "sales_call",
    "sales_email",
    "shakespeare",
    "scitldr",
    "qmsumm",
    "ectsum"
]


def load_models_cache():
    """Load models from cache."""
    cache_path = DATA_PATH / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    
    models_list = data.get("models", data)
    models = {}
    
    for model in models_list:
        openrouter_id = model.get("openrouter_id")
        if openrouter_id:
            models[openrouter_id] = {
                "name": model.get("name", ""),
                "slug": model.get("slug", ""),
                "openrouter_id": openrouter_id,
            }
    
    return models


def load_domain_scores(domain):
    """Load scores for a specific domain."""
    scores_file = f"summedits_{domain}_scores.json"
    scores_path = DATA_PATH / scores_file
    
    if scores_path.exists():
        with open(scores_path) as f:
            return json.load(f)
    return {}


def main():
    print("="*80)
    print("SummEdits Coverage Analysis")
    print("="*80)
    
    # Load all models
    all_models = load_models_cache()
    print(f"\nTotal models with OpenRouter ID: {len(all_models)}")
    
    # Load scores for each domain
    domain_scores = {}
    for domain in SUMMEDITS_DOMAINS:
        domain_scores[domain] = load_domain_scores(domain)
    
    print(f"\nDomains: {len(SUMMEDITS_DOMAINS)}")
    for domain in SUMMEDITS_DOMAINS:
        count = len(domain_scores[domain])
        print(f"  {domain:<15} {count:>4} models evaluated")
    
    # Build coverage map
    model_coverage = defaultdict(list)
    
    # Find which domains each model has
    for openrouter_id in all_models.keys():
        for domain in SUMMEDITS_DOMAINS:
            if openrouter_id in domain_scores[domain]:
                model_coverage[openrouter_id].append(domain)
    
    # Categorize models
    complete_models = []
    incomplete_models = []
    no_scores = []
    
    for openrouter_id, model_info in all_models.items():
        covered = model_coverage.get(openrouter_id, [])
        coverage_count = len(covered)
        
        if coverage_count == len(SUMMEDITS_DOMAINS):
            complete_models.append((openrouter_id, model_info, covered))
        elif coverage_count > 0:
            incomplete_models.append((openrouter_id, model_info, covered))
        else:
            no_scores.append((openrouter_id, model_info, covered))
    
    # Sort by coverage (descending)
    incomplete_models.sort(key=lambda x: len(x[2]), reverse=True)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Complete (all {len(SUMMEDITS_DOMAINS)} domains): {len(complete_models)} models")
    print(f"⚠️  Incomplete (some domains):  {len(incomplete_models)} models")
    print(f"❌ No scores yet:            {len(no_scores)} models")
    
    # Show incomplete models
    if incomplete_models:
        print("\n" + "="*80)
        print("INCOMPLETE MODELS (have some but not all domains)")
        print("="*80)
        
        for openrouter_id, model_info, covered in incomplete_models[:30]:  # Show top 30
            missing = [d for d in SUMMEDITS_DOMAINS if d not in covered]
            coverage_pct = len(covered) / len(SUMMEDITS_DOMAINS) * 100
            
            print(f"\n{model_info['name']}")
            print(f"  ID: {openrouter_id}")
            print(f"  Coverage: {len(covered)}/{len(SUMMEDITS_DOMAINS)} ({coverage_pct:.0f}%)")
            print(f"  Missing: {', '.join(missing)}")
        
        if len(incomplete_models) > 30:
            print(f"\n... and {len(incomplete_models) - 30} more")
    
    # Show models with no scores
    if no_scores:
        print("\n" + "="*80)
        print(f"MODELS WITH NO SCORES ({len(no_scores)} total)")
        print("="*80)
        
        print("\nFirst 20:")
        for openrouter_id, model_info, _ in no_scores[:20]:
            print(f"  • {model_info['name']} ({openrouter_id})")
        
        if len(no_scores) > 20:
            print(f"\n... and {len(no_scores) - 20} more")
    
    # Show complete models
    if complete_models:
        print("\n" + "="*80)
        print(f"✅ COMPLETE MODELS ({len(complete_models)} total)")
        print("="*80)
        
        print("\nFirst 20:")
        for openrouter_id, model_info, _ in complete_models[:20]:
            print(f"  • {model_info['name']} ({openrouter_id})")
        
        if len(complete_models) > 20:
            print(f"\n... and {len(complete_models) - 20} more")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
