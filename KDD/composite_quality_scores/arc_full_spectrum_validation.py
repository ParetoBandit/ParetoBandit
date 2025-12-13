#!/usr/bin/env python3
"""
ARC Challenge Validation Across Full CRS Spectrum

Tests models across LOW, MID, and HIGH CRS ranges to understand the full
correlation between CRS scores and actual reasoning accuracy.

Strategy:
- Select ~30 models: 10 low, 10 mid, 10 high CRS
- Test on ARC-Challenge (harder than Easy)
- Analyze CRS vs accuracy correlation with full range
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ModelSelection:
    """Model selected for testing."""
    name: str
    openrouter_id: str
    crs_score: float
    crs_rank: int
    crs_tier: str  # "low", "mid", "high"


def load_all_models_with_crs() -> List[Dict]:
    """Load all models with CRS scores and OpenRouter access."""
    
    # Load models cache (contains benchmarks)
    models_cache_path = PROJECT_ROOT / "data" / "models_cache.json"
    
    if not models_cache_path.exists():
        print(f"❌ Models cache not found: {models_cache_path}")
        sys.exit(1)
    
    with open(models_cache_path, 'r') as f:
        data = json.load(f)
    
    # The cache has a "models" key with list of models
    models = data.get('models', [])
    
    # Filter: has CRS and OpenRouter ID
    valid_models = []
    for m in models:
        if m.get('crs') is not None and m.get('openrouter_id'):
            valid_models.append(m)
    
    # Sort by CRS
    valid_models.sort(key=lambda m: m['crs'], reverse=True)
    
    return valid_models


def select_stratified_models(
    models: List[Dict],
    n_per_tier: int = 10,
    exclude_tested: bool = True
) -> Tuple[List[ModelSelection], List[ModelSelection], List[ModelSelection]]:
    """
    Select models stratified across CRS range.
    
    Returns:
        (high_crs_models, mid_crs_models, low_crs_models)
    """
    
    print(f"\n📊 Stratified Model Selection")
    print(f"{'='*80}")
    
    # Load already tested models (if excluding)
    tested_models = set()
    if exclude_tested:
        results_path = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results" / "arc_easy_vs_challenge_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            tested_models = {m['openrouter_id'] for m in results['models']}
            print(f"   Excluding {len(tested_models)} already tested models")
    
    # Filter out tested models
    available = [m for m in models if m['openrouter_id'] not in tested_models]
    
    print(f"   Available models: {len(available)}")
    print(f"   CRS range: {available[-1]['crs']:.2f} to {available[0]['crs']:.2f}")
    
    # Divide into thirds
    n = len(available)
    high_end = n // 3
    low_start = 2 * n // 3
    
    high_pool = available[:high_end]
    mid_pool = available[high_end:low_start]
    low_pool = available[low_start:]
    
    print(f"\n   High CRS pool: {len(high_pool)} models (CRS: {high_pool[-1]['crs']:.2f} to {high_pool[0]['crs']:.2f})")
    print(f"   Mid CRS pool:  {len(mid_pool)} models (CRS: {mid_pool[-1]['crs']:.2f} to {mid_pool[0]['crs']:.2f})")
    print(f"   Low CRS pool:  {len(low_pool)} models (CRS: {low_pool[-1]['crs']:.2f} to {low_pool[0]['crs']:.2f})")
    
    # Select from each tier
    def select_from_pool(pool: List[Dict], tier: str, n: int, start_rank: int) -> List[ModelSelection]:
        # Sample evenly across the pool
        if len(pool) <= n:
            selected = pool
        else:
            indices = np.linspace(0, len(pool)-1, n, dtype=int)
            selected = [pool[i] for i in indices]
        
        return [
            ModelSelection(
                name=m['name'],
                openrouter_id=m['openrouter_id'],
                crs_score=m['crs'],
                crs_rank=start_rank + i,
                crs_tier=tier
            )
            for i, m in enumerate(selected)
        ]
    
    high_models = select_from_pool(high_pool, "high", n_per_tier, 1)
    mid_models = select_from_pool(mid_pool, "mid", n_per_tier, high_end + 1)
    low_models = select_from_pool(low_pool, "low", n_per_tier, low_start + 1)
    
    print(f"\n✓ Selected Models:")
    print(f"   High CRS: {len(high_models)} models (CRS: {high_models[-1].crs_score:.2f} to {high_models[0].crs_score:.2f})")
    print(f"   Mid CRS:  {len(mid_models)} models (CRS: {mid_models[-1].crs_score:.2f} to {mid_models[0].crs_score:.2f})")
    print(f"   Low CRS:  {len(low_models)} models (CRS: {low_models[-1].crs_score:.2f} to {low_models[0].crs_score:.2f})")
    
    return high_models, mid_models, low_models


def preview_test_plan(high: List[ModelSelection], mid: List[ModelSelection], low: List[ModelSelection]):
    """Show what will be tested."""
    
    all_models = high + mid + low
    
    print(f"\n{'='*80}")
    print(f"TEST PLAN PREVIEW")
    print(f"{'='*80}")
    print(f"\nTotal models to test: {len(all_models)}")
    print(f"Problems per model: 50 (ARC-Challenge)")
    print(f"Total API calls: {len(all_models) * 50}")
    
    print(f"\n{'Tier':<10} {'#':<5} {'CRS Range':<20} {'Models'}")
    print(f"{'-'*10} {'-'*5} {'-'*20} {'-'*40}")
    print(f"{'HIGH':<10} {len(high):<5} {high[-1].crs_score:.2f} to {high[0].crs_score:.2f}   {', '.join([m.name[:15] for m in high[:3]])}...")
    print(f"{'MID':<10} {len(mid):<5} {mid[-1].crs_score:.2f} to {mid[0].crs_score:.2f}   {', '.join([m.name[:15] for m in mid[:3]])}...")
    print(f"{'LOW':<10} {len(low):<5} {low[-1].crs_score:.2f} to {low[0].crs_score:.2f}   {', '.join([m.name[:15] for m in low[:3]])}...")
    
    print(f"\n{'Model Details:'}")
    print(f"{'─'*80}")
    print(f"{'Tier':<8} {'Rank':<6} {'CRS':<8} {'Model Name':<45} {'OpenRouter ID'}")
    print(f"{'─'*80}")
    
    for tier_name, tier_models in [("HIGH", high), ("MID", mid), ("LOW", low)]:
        for m in tier_models:
            print(f"{tier_name:<8} {m.crs_rank:<6} {m.crs_score:>6.2f}  {m.name[:43]:<45} {m.openrouter_id}")


def estimate_cost_and_time(n_models: int, n_problems: int = 50):
    """Estimate cost and time for the test."""
    
    total_calls = n_models * n_problems
    
    # Rough estimates
    avg_input_tokens = 300  # prompt
    avg_output_tokens = 50   # short answer
    
    total_input_tokens = total_calls * avg_input_tokens
    total_output_tokens = total_calls * avg_output_tokens
    
    # OpenRouter pricing (very rough average)
    avg_cost_per_1k_input = 0.0015  # $1.50 per 1M tokens
    avg_cost_per_1k_output = 0.0050  # $5.00 per 1M tokens
    
    estimated_cost = (total_input_tokens / 1000 * avg_cost_per_1k_input + 
                     total_output_tokens / 1000 * avg_cost_per_1k_output)
    
    # Time estimate (with retries, rate limits, etc.)
    avg_time_per_call = 3  # seconds
    total_time_seconds = total_calls * avg_time_per_call
    total_hours = total_time_seconds / 3600
    
    print(f"\n{'='*80}")
    print(f"COST & TIME ESTIMATES")
    print(f"{'='*80}")
    print(f"\nAPI Calls:")
    print(f"   Total calls: {total_calls:,}")
    print(f"   Input tokens (est): {total_input_tokens:,}")
    print(f"   Output tokens (est): {total_output_tokens:,}")
    
    print(f"\nCost Estimate:")
    print(f"   ~${estimated_cost:.2f} (rough average across models)")
    print(f"   Note: Actual cost varies by model tier")
    
    print(f"\nTime Estimate:")
    print(f"   Sequential: ~{total_hours:.1f} hours")
    print(f"   Note: Includes retries, rate limits, API delays")
    

def main():
    print("="*80)
    print("ARC CHALLENGE: FULL CRS SPECTRUM VALIDATION")
    print("="*80)
    
    # Load models
    print(f"\n📂 Loading models...")
    all_models = load_all_models_with_crs()
    print(f"   ✓ Found {len(all_models)} models with CRS scores and API access")
    
    # Select stratified sample
    high, mid, low = select_stratified_models(all_models, n_per_tier=10, exclude_tested=True)
    
    # Preview
    preview_test_plan(high, mid, low)
    
    # Estimates
    estimate_cost_and_time(len(high) + len(mid) + len(low), n_problems=50)
    
    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"\n1. Review the model selection above")
    print(f"2. Adjust n_per_tier if needed (currently 10 per tier)")
    print(f"3. Run the actual validation:")
    print(f"   python3 KDD/composite_quality_scores/run_arc_full_spectrum.py")
    print(f"\n4. Analysis will show:")
    print(f"   • CRS vs Accuracy scatter plot (full range)")
    print(f"   • Correlation coefficients")
    print(f"   • Performance by tier (low/mid/high)")
    print(f"   • Identification of over/under performers")
    
    # Save selection for actual run
    output_dir = PROJECT_ROOT / "KDD" / "composite_quality_scores" / "llm_judge_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selection_path = output_dir / "arc_full_spectrum_model_selection.json"
    
    selection_data = {
        'metadata': {
            'total_models': len(high) + len(mid) + len(low),
            'n_per_tier': 10,
            'excluded_already_tested': True,
        },
        'tiers': {
            'high': [
                {
                    'name': m.name,
                    'openrouter_id': m.openrouter_id,
                    'crs_score': m.crs_score,
                    'crs_rank': m.crs_rank,
                }
                for m in high
            ],
            'mid': [
                {
                    'name': m.name,
                    'openrouter_id': m.openrouter_id,
                    'crs_score': m.crs_score,
                    'crs_rank': m.crs_rank,
                }
                for m in mid
            ],
            'low': [
                {
                    'name': m.name,
                    'openrouter_id': m.openrouter_id,
                    'crs_score': m.crs_score,
                    'crs_rank': m.crs_rank,
                }
                for m in low
            ],
        }
    }
    
    with open(selection_path, 'w') as f:
        json.dump(selection_data, f, indent=2)
    
    print(f"\n💾 Model selection saved to: {selection_path}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
