#!/usr/bin/env python3
"""
Analyze Creative Writing scores and correlate with other quality metrics.

This script loads Creative Writing Elo scores and compares them
with other quality signals like hallucination rates, intelligence indices, etc.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"


def load_creative_writing_scores() -> Dict[str, float]:
    """Load Creative Writing Elo scores."""
    scores_file = DATA_PATH / "creative_writing_scores.json"
    if scores_file.exists():
        with open(scores_file) as f:
            return json.load(f)
    return {}


def load_models_cache() -> List[Dict]:
    """Load models from cache."""
    cache_path = DATA_PATH / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    return data.get("models", data)


def analyze_correlations(scores: Dict[str, float], models_cache: List[Dict]):
    """Analyze correlations between Creative Writing and other metrics."""
    
    # Build model lookup
    model_lookup = {}
    for m in models_cache:
        openrouter_id = m.get('openrouter_id', '')
        if openrouter_id:
            model_lookup[openrouter_id] = m
    
    # Collect paired data
    data_points = []
    for model_id, creative_score in scores.items():
        if model_id in model_lookup:
            m = model_lookup[model_id]
            data_points.append({
                'model_id': model_id,
                'name': m.get('name', ''),
                'creative': creative_score,
                'hallucination': float(m.get('hallucination_rate', 0)),
                'intelligence': float(m.get('intelligence_index', 0)),
                'coding': float(m.get('coding_index', 0)),
                'math': float(m.get('math_index', 0)),
            })
    
    return data_points


def print_report(scores: Dict[str, float], data_points: List[Dict]):
    """Print analysis report."""
    
    print("=" * 80)
    print("CREATIVE WRITING ANALYSIS REPORT")
    print("=" * 80)
    
    if not scores:
        print("\n❌ No Creative Writing scores found yet")
        print("   Run: python kdd_paper/run_creative_writing.py --all")
        return
    
    # Top performers
    print("\n1. TOP PERFORMERS (Elo Score)")
    print("-" * 80)
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_id, score) in enumerate(sorted_models[:15], 1):
        # Find model name
        name = model_id
        for dp in data_points:
            if dp['model_id'] == model_id:
                name = dp['name']
                break
        print(f"  {i:2d}. {name:<45} {score:>6.1f}")
    
    # Score distribution
    score_vals = list(scores.values())
    print("\n2. SCORE DISTRIBUTION")
    print("-" * 80)
    print(f"  Mean:   {statistics.mean(score_vals):.1f}")
    print(f"  Median: {statistics.median(score_vals):.1f}")
    print(f"  Std:    {statistics.stdev(score_vals) if len(score_vals) > 1 else 0:.1f}")
    print(f"  Min:    {min(score_vals):.1f}")
    print(f"  Max:    {max(score_vals):.1f}")
    
    # Performance tiers
    print("\n3. PERFORMANCE TIERS")
    print("-" * 80)
    excellent = [s for s in score_vals if s >= 1400]
    good = [s for s in score_vals if 1200 <= s < 1400]
    fair = [s for s in score_vals if 1000 <= s < 1200]
    poor = [s for s in score_vals if s < 1000]
    
    print(f"  🟢 Excellent (1400+):  {len(excellent):>3} models ({len(excellent)/len(score_vals)*100:>5.1f}%)")
    print(f"  🟡 Good (1200-1399):   {len(good):>3} models ({len(good)/len(score_vals)*100:>5.1f}%)")
    print(f"  🟠 Fair (1000-1199):   {len(fair):>3} models ({len(fair)/len(score_vals)*100:>5.1f}%)")
    print(f"  🔴 Poor (<1000):       {len(poor):>3} models ({len(poor)/len(score_vals)*100:>5.1f}%)")
    
    # Correlations
    if len(data_points) >= 5:
        print("\n4. CORRELATION WITH OTHER METRICS")
        print("-" * 80)
        
        def simple_correlation(x_vals, y_vals):
            """Calculate Pearson correlation coefficient."""
            if len(x_vals) < 2:
                return 0.0
            
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
            sum_x2 = sum(x * x for x in x_vals)
            sum_y2 = sum(y * y for y in y_vals)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        creative_vals = [dp['creative'] for dp in data_points]
        
        # Hallucination (negative correlation expected)
        halluc_vals = [dp['hallucination'] for dp in data_points if dp['hallucination'] > 0]
        creative_halluc = [dp['creative'] for dp in data_points if dp['hallucination'] > 0]
        if len(halluc_vals) >= 2:
            corr = simple_correlation(halluc_vals, creative_halluc)
            print(f"  Hallucination Rate:    r = {corr:>6.3f} {'✓ (negative as expected)' if corr < 0 else '⚠ (should be negative)'}")
        
        # Intelligence
        intel_vals = [dp['intelligence'] for dp in data_points if dp['intelligence'] > 0]
        creative_intel = [dp['creative'] for dp in data_points if dp['intelligence'] > 0]
        if len(intel_vals) >= 2:
            corr = simple_correlation(intel_vals, creative_intel)
            print(f"  Intelligence Index:    r = {corr:>6.3f} {'✓ (positive)' if corr > 0 else ''}")
        
        # Coding
        coding_vals = [dp['coding'] for dp in data_points if dp['coding'] > 0]
        creative_coding = [dp['creative'] for dp in data_points if dp['coding'] > 0]
        if len(coding_vals) >= 2:
            corr = simple_correlation(coding_vals, creative_coding)
            print(f"  Coding Index:          r = {corr:>6.3f}")
        
        # Math
        math_vals = [dp['math'] for dp in data_points if dp['math'] > 0]
        creative_math = [dp['creative'] for dp in data_points if dp['math'] > 0]
        if len(math_vals) >= 2:
            corr = simple_correlation(math_vals, creative_math)
            print(f"  Math Index:            r = {corr:>6.3f}")
        
        print("\n  Note: Creative writing may have weak correlation with technical metrics")
        print("        This is expected - creative ability is a distinct skill")
    
    # Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    
    total_models = len([m for m in load_models_cache() if m.get('openrouter_id')])
    if len(scores) < total_models * 0.5:
        print(f"  ℹ Only {len(scores)} / {total_models} models evaluated")
        print(f"  ➜ Run more evaluations: python kdd_paper/run_creative_writing.py --all")
    else:
        print(f"  ✓ Good coverage: {len(scores)} / {total_models} models evaluated")
    
    print("\n  💡 Use Case Recommendations:")
    print("     • High Elo (1400+): Excellent for creative writing, storytelling, RP")
    print("     • Medium Elo (1200-1399): Good for general creative tasks")
    print("     • Low Elo (<1200): Better for technical/analytical tasks")
    
    print("\n" + "=" * 80)


def main():
    print("Loading Creative Writing data...")
    
    # Load data
    scores = load_creative_writing_scores()
    models_cache = load_models_cache()
    
    # Analyze
    data_points = analyze_correlations(scores, models_cache)
    
    # Print report
    print_report(scores, data_points)
    
    print(f"\n✓ Scores loaded from: {DATA_PATH / 'creative_writing_scores.json'}")


if __name__ == "__main__":
    main()

