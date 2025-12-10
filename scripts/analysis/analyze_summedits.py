#!/usr/bin/env python3
"""
Analyze SummEdits scores and correlate with other quality metrics.

This script loads SummEdits scores across all domains and compares them
with other quality signals like hallucination rates, intelligence indices, etc.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/analysis -> scripts -> project root
DATA_PATH = PROJECT_ROOT / "data"

SUMMEDITS_DOMAINS = [
    "news", "podcast", "billsum", "samsum",
    "sales_call", "sales_email", "shakespeare",
    "scitldr", "qmsumm", "ectsum"
]


def load_summedits_scores() -> Dict[str, Dict[str, float]]:
    """Load SummEdits scores for all domains."""
    scores = {}
    for domain in SUMMEDITS_DOMAINS:
        scores_file = DATA_PATH / f"summedits_{domain}_scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                scores[domain] = json.load(f)
        else:
            scores[domain] = {}
    return scores


def load_models_cache() -> List[Dict]:
    """Load models from cache."""
    cache_path = DATA_PATH / "models_cache.json"
    with open(cache_path) as f:
        data = json.load(f)
    return data.get("models", data)


def calculate_aggregate_scores(domain_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
    """Calculate aggregate SummEdits score across all domains for each model.
    
    Returns dict with mean, std, ci_lower, ci_upper for each model.
    """
    import math
    
    model_aggregates = {}
    
    # Find all models
    all_models = set()
    for domain_data in domain_scores.values():
        all_models.update(domain_data.keys())
    
    # Calculate stats across domains for each model
    for model in all_models:
        scores = []
        for domain, domain_data in domain_scores.items():
            if model in domain_data:
                scores.append(domain_data[model])
        
        if scores:
            n = len(scores)
            mean_score = statistics.mean(scores)
            
            if n >= 2:
                std = statistics.stdev(scores)
                se = std / math.sqrt(n)
                # 95% CI using t-distribution approximation (t ~ 1.96 for large n)
                t_value = 1.96 if n > 30 else 2.0  # Conservative for smaller samples
                ci_lower = mean_score - t_value * se
                ci_upper = mean_score + t_value * se
            else:
                std = 0
                ci_lower = mean_score
                ci_upper = mean_score
            
            model_aggregates[model] = {
                'mean': mean_score,
                'std': std,
                'n_domains': n,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    return model_aggregates


def analyze_correlations(aggregate_scores: Dict[str, Dict], models_cache: List[Dict]):
    """Analyze correlations between SummEdits and other metrics."""
    
    # Build model lookup
    model_lookup = {}
    for m in models_cache:
        openrouter_id = m.get('openrouter_id', '')
        if openrouter_id:
            model_lookup[openrouter_id] = m
    
    # Collect paired data
    data_points = []
    for model_id, score_data in aggregate_scores.items():
        if model_id in model_lookup:
            m = model_lookup[model_id]
            data_points.append({
                'model_id': model_id,
                'name': m.get('name', ''),
                'summedits': score_data['mean'],
                'summedits_ci_lower': score_data['ci_lower'],
                'summedits_ci_upper': score_data['ci_upper'],
                'summedits_std': score_data['std'],
                'n_domains': score_data['n_domains'],
                'hallucination': float(m.get('hallucination_rate', 0) or 0),
                'intelligence': float(m.get('intelligence_index', 0) or 0),
                'coding': float(m.get('coding_index', 0) or 0),
                'math': float(m.get('math_index', 0) or 0),
            })
    
    return data_points


def print_report(domain_scores: Dict[str, Dict[str, float]], aggregate_scores: Dict[str, Dict], data_points: List[Dict]):
    """Print analysis report."""
    
    print("=" * 80)
    print("SUMMEDITS ANALYSIS REPORT")
    print("=" * 80)
    
    # Domain coverage
    print("\n1. DOMAIN COVERAGE")
    print("-" * 80)
    for domain in SUMMEDITS_DOMAINS:
        count = len(domain_scores.get(domain, {}))
        if count > 0:
            avg_score = statistics.mean(domain_scores[domain].values())
            print(f"  {domain:<15} {count:>3} models evaluated (avg: {avg_score:.1f}%)")
        else:
            print(f"  {domain:<15} No evaluations yet")
    
    # Top performers with confidence intervals
    if aggregate_scores:
        print("\n2. TOP PERFORMERS (Overall Balanced Accuracy with 95% CI)")
        print("-" * 80)
        sorted_models = sorted(aggregate_scores.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for i, (model_id, score_data) in enumerate(sorted_models[:15], 1):
            # Find model name
            name = model_id
            for dp in data_points:
                if dp['model_id'] == model_id:
                    name = dp['name']
                    break
            mean = score_data['mean']
            ci_lower = score_data['ci_lower']
            ci_upper = score_data['ci_upper']
            n = score_data['n_domains']
            print(f"  {i:2d}. {name:<40} {mean:>5.1f}% [{ci_lower:.1f}, {ci_upper:.1f}] ({n} domains)")
    
    # Score distribution
    if aggregate_scores:
        scores = [s['mean'] for s in aggregate_scores.values()]
        print("\n3. SCORE DISTRIBUTION")
        print("-" * 80)
        print(f"  Mean:   {statistics.mean(scores):.1f}%")
        print(f"  Median: {statistics.median(scores):.1f}%")
        print(f"  Std:    {statistics.stdev(scores) if len(scores) > 1 else 0:.1f}%")
        print(f"  Min:    {min(scores):.1f}%")
        print(f"  Max:    {max(scores):.1f}%")
    
    # Correlations
    if len(data_points) >= 5:
        print("\n4. CORRELATION WITH OTHER METRICS")
        print("-" * 80)
        
        # Calculate simple correlations
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
        
        # Hallucination (should be negative correlation - higher hallucination = lower consistency)
        halluc_vals = [dp['hallucination'] for dp in data_points if dp['hallucination'] > 0]
        summedits_halluc = [dp['summedits'] for dp in data_points if dp['hallucination'] > 0]
        if len(halluc_vals) >= 2:
            corr = simple_correlation(halluc_vals, summedits_halluc)
            print(f"  Hallucination Rate:    r = {corr:>6.3f} {'✓ (negative as expected)' if corr < 0 else '⚠ (should be negative)'}")
        
        # Intelligence
        intel_vals = [dp['intelligence'] for dp in data_points if dp['intelligence'] > 0]
        summedits_intel = [dp['summedits'] for dp in data_points if dp['intelligence'] > 0]
        if len(intel_vals) >= 2:
            corr = simple_correlation(intel_vals, summedits_intel)
            print(f"  Intelligence Index:    r = {corr:>6.3f} {'✓ (positive)' if corr > 0 else ''}")
        
        # Coding
        coding_vals = [dp['coding'] for dp in data_points if dp['coding'] > 0]
        summedits_coding = [dp['summedits'] for dp in data_points if dp['coding'] > 0]
        if len(coding_vals) >= 2:
            corr = simple_correlation(coding_vals, summedits_coding)
            print(f"  Coding Index:          r = {corr:>6.3f}")
        
        # Math
        math_vals = [dp['math'] for dp in data_points if dp['math'] > 0]
        summedits_math = [dp['summedits'] for dp in data_points if dp['math'] > 0]
        if len(math_vals) >= 2:
            corr = simple_correlation(math_vals, summedits_math)
            print(f"  Math Index:            r = {corr:>6.3f}")
        
        print("\n  Note: Correlation coefficients range from -1 to +1")
        print("        Values > 0.5 indicate strong positive correlation")
        print("        Values < -0.5 indicate strong negative correlation")
    
    # Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    
    if not any(domain_scores.values()):
        print("  ⚠ No SummEdits scores found yet")
        print("  ➜ Run: python kdd_paper/run_summedits.py --all --domains all")
    else:
        incomplete_domains = [d for d in SUMMEDITS_DOMAINS if not domain_scores.get(d)]
        if incomplete_domains:
            print(f"  ⚠ {len(incomplete_domains)} domains not yet evaluated: {', '.join(incomplete_domains[:5])}")
            print(f"  ➜ Run: python kdd_paper/run_summedits.py --all --domains {' '.join(incomplete_domains)}")
        else:
            print("  ✓ All domains evaluated!")
        
        if aggregate_scores:
            avg_coverage = len(aggregate_scores) / len([m for m in load_models_cache() if m.get('openrouter_id')])
            if avg_coverage < 0.5:
                print(f"  ℹ Only {len(aggregate_scores)} models evaluated")
                print(f"  ➜ Run more evaluations: python kdd_paper/run_summedits.py --all")
    
    print("\n" + "=" * 80)


def main():
    print("Loading SummEdits data...")
    
    # Load data
    domain_scores = load_summedits_scores()
    aggregate_scores = calculate_aggregate_scores(domain_scores)
    models_cache = load_models_cache()
    
    # Analyze
    data_points = analyze_correlations(aggregate_scores, models_cache)
    
    # Print report
    print_report(domain_scores, aggregate_scores, data_points)
    
    # Export aggregate scores
    if aggregate_scores:
        # Save simple mean scores for backward compatibility
        simple_scores = {model_id: data['mean'] for model_id, data in aggregate_scores.items()}
        output_path = DATA_PATH / "summedits_aggregate_scores.json"
        with open(output_path, 'w') as f:
            json.dump(simple_scores, f, indent=2)
        print(f"\n✓ Aggregate scores saved to: {output_path}")
        
        # Save detailed scores with confidence intervals
        detailed_path = DATA_PATH / "summedits_aggregate_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(aggregate_scores, f, indent=2)
        print(f"✓ Detailed scores (with CI) saved to: {detailed_path}")


if __name__ == "__main__":
    main()

