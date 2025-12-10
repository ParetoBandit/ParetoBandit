#!/usr/bin/env python3
"""
Aggregate SummEdits scores across all domains with confidence intervals.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

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


def load_all_scores() -> Dict[str, Dict[str, float]]:
    """Load scores from all domains."""
    all_scores = {}
    
    for domain in SUMMEDITS_DOMAINS:
        scores_file = f"summedits_{domain}_scores.json"
        scores_path = DATA_PATH / scores_file
        
        if scores_path.exists():
            with open(scores_path) as f:
                domain_scores = json.load(f)
                all_scores[domain] = domain_scores
    
    return all_scores


def aggregate_scores_with_ci(all_scores: Dict[str, Dict[str, float]]) -> List[Tuple[str, float, float, float, int]]:
    """
    Aggregate scores across domains with confidence intervals.
    
    Returns:
        List of (model_id, mean_score, ci_lower, ci_upper, num_domains)
    """
    # Collect scores by model
    model_scores = {}
    
    for domain, scores in all_scores.items():
        for model_id, score in scores.items():
            if model_id not in model_scores:
                model_scores[model_id] = []
            model_scores[model_id].append(score)
    
    # Calculate aggregate scores with CIs
    results = []
    
    for model_id, scores in model_scores.items():
        if len(scores) < 2:  # Need at least 2 domains for CI
            continue
        
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        
        # Calculate 95% confidence interval using t-distribution
        n = len(scores)
        std_err = stats.sem(scores_array)
        ci = stats.t.interval(0.95, n-1, loc=mean_score, scale=std_err)
        
        results.append((model_id, mean_score, ci[0], ci[1], n))
    
    # Sort by mean score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def load_model_names() -> Dict[str, str]:
    """Load model names from cache."""
    cache_path = DATA_PATH / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    
    models = data.get("models", data)
    return {m.get("openrouter_id"): m.get("name") for m in models if m.get("openrouter_id")}


def main():
    print("="*80)
    print("SUMMEDITS AGGREGATE SCORES WITH CONFIDENCE INTERVALS")
    print("="*80)
    
    # Load all scores
    all_scores = load_all_scores()
    print(f"\nLoaded scores from {len(all_scores)} domains:")
    for domain, scores in all_scores.items():
        print(f"  {domain:<15} {len(scores):>3} models")
    
    # Get aggregate scores
    results = aggregate_scores_with_ci(all_scores)
    
    # Load model names
    model_names = load_model_names()
    
    # Save aggregate scores to file
    aggregate_dict = {}
    for model_id, mean_score, ci_lower, ci_upper, num_domains in results:
        aggregate_dict[model_id] = {
            "mean_score": round(mean_score, 2),
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "num_domains": num_domains,
            "model_name": model_names.get(model_id, "Unknown")
        }
    
    output_path = DATA_PATH / "summedits_aggregate_scores.json"
    with open(output_path, "w") as f:
        json.dump(aggregate_dict, f, indent=2)
    
    print(f"\n✅ Saved aggregate scores to {output_path.name}")
    
    # Display results
    print("\n" + "="*80)
    print("TOP 30 MODELS BY AGGREGATE SCORE")
    print("="*80)
    print(f"{'Rank':<5} {'Model':<45} {'Score':<8} {'95% CI':<20} {'Domains':<8}")
    print("-"*80)
    
    for i, (model_id, mean_score, ci_lower, ci_upper, num_domains) in enumerate(results[:30], 1):
        model_name = model_names.get(model_id, model_id)
        # Truncate long names
        if len(model_name) > 43:
            model_name = model_name[:40] + "..."
        
        ci_str = f"[{ci_lower:.1f}, {ci_upper:.1f}]"
        print(f"{i:<5} {model_name:<45} {mean_score:>6.1f}%  {ci_str:<20} {num_domains}/10")
    
    # Show models with complete coverage (all 10 domains)
    complete_models = [(m, s, l, u, n) for m, s, l, u, n in results if n == 10]
    
    print("\n" + "="*80)
    print(f"COMPLETE COVERAGE ({len(complete_models)} models with all 10 domains)")
    print("="*80)
    print(f"{'Rank':<5} {'Model':<45} {'Score':<8} {'95% CI':<20}")
    print("-"*80)
    
    for i, (model_id, mean_score, ci_lower, ci_upper, _) in enumerate(complete_models[:20], 1):
        model_name = model_names.get(model_id, model_id)
        if len(model_name) > 43:
            model_name = model_name[:40] + "..."
        
        ci_str = f"[{ci_lower:.1f}, {ci_upper:.1f}]"
        print(f"{i:<5} {model_name:<45} {mean_score:>6.1f}%  {ci_str:<20}")
    
    if len(complete_models) > 20:
        print(f"\n... and {len(complete_models) - 20} more with complete coverage")
    
    # Show newly scored models (those with fewer domains)
    new_models = [(m, s, l, u, n) for m, s, l, u, n in results if 1 <= n < 10]
    
    if new_models:
        print("\n" + "="*80)
        print(f"NEWLY/PARTIALLY SCORED ({len(new_models)} models)")
        print("="*80)
        print(f"{'Model':<45} {'Score':<8} {'95% CI':<20} {'Domains':<8}")
        print("-"*80)
        
        for model_id, mean_score, ci_lower, ci_upper, num_domains in new_models[:20]:
            model_name = model_names.get(model_id, model_id)
            if len(model_name) > 43:
                model_name = model_name[:40] + "..."
            
            ci_str = f"[{ci_lower:.1f}, {ci_upper:.1f}]"
            print(f"{model_name:<45} {mean_score:>6.1f}%  {ci_str:<20} {num_domains}/10")
        
        if len(new_models) > 20:
            print(f"\n... and {len(new_models) - 20} more")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
