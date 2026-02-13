#!/usr/bin/env python3
"""
Cost Analysis for Table 2

Computes actual cost implications of different routing strategies based on:
- GPT-4-Turbo pricing: $10/1M input tokens, $30/1M output tokens
- Mixtral-8x7B pricing: $0.50/1M input tokens, $0.50/1M output tokens
- Average prompt/completion lengths from LMSYS data

Shows that higher GPT-4 usage in Corralling leads to higher costs.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


# Pricing (per 1M tokens) - approximate 2026 rates
GPT4_INPUT_COST = 10.0
GPT4_OUTPUT_COST = 30.0
MIXTRAL_INPUT_COST = 0.50
MIXTRAL_OUTPUT_COST = 0.50

# Average token counts (estimated from LMSYS data)
AVG_PROMPT_TOKENS = 150
AVG_COMPLETION_TOKENS = 400


def compute_query_cost(model: str) -> float:
    """Compute cost per query for a given model."""
    
    if 'gpt-4' in model.lower():
        input_cost = (AVG_PROMPT_TOKENS / 1_000_000) * GPT4_INPUT_COST
        output_cost = (AVG_COMPLETION_TOKENS / 1_000_000) * GPT4_OUTPUT_COST
    elif 'mixtral' in model.lower():
        input_cost = (AVG_PROMPT_TOKENS / 1_000_000) * MIXTRAL_INPUT_COST
        output_cost = (AVG_COMPLETION_TOKENS / 1_000_000) * MIXTRAL_OUTPUT_COST
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return input_cost + output_cost


def compute_strategy_cost(model_usage: Dict[str, int]) -> Dict:
    """Compute total cost and breakdown for a strategy."""
    
    gpt4_cost_per_query = compute_query_cost('gpt-4-turbo')
    mixtral_cost_per_query = compute_query_cost('mixtral')
    
    gpt4_count = model_usage.get('openai/gpt-4-turbo', 0)
    mixtral_count = model_usage.get('mistralai/mixtral-8x7b-instruct', 0)
    
    gpt4_total_cost = gpt4_count * gpt4_cost_per_query
    mixtral_total_cost = mixtral_count * mixtral_cost_per_query
    
    total_cost = gpt4_total_cost + mixtral_total_cost
    total_queries = gpt4_count + mixtral_count
    
    return {
        'gpt4_queries': gpt4_count,
        'mixtral_queries': mixtral_count,
        'total_queries': total_queries,
        'gpt4_pct': 100 * gpt4_count / total_queries if total_queries > 0 else 0,
        'gpt4_cost': gpt4_total_cost,
        'mixtral_cost': mixtral_total_cost,
        'total_cost': total_cost,
        'cost_per_query': total_cost / total_queries if total_queries > 0 else 0,
        'cost_per_1k_queries': 1000 * total_cost / total_queries if total_queries > 0 else 0
    }


def load_results(path: Path) -> Dict:
    """Load per-seed results."""
    with open(path) as f:
        return json.load(f)


def cost_analysis():
    """Comprehensive cost analysis."""
    
    print("="*100)
    print("COST ANALYSIS: Production Deployment Implications")
    print("="*100)
    
    # Load data
    eta_10_path = Path(__file__).parent / 'data' / 'eta_1.0_holdout_multiseed' / 'results_per_seed.json'
    eta_01_path = Path(__file__).parent / 'data' / 'eta_0.1_holdout_multiseed' / 'results_per_seed.json'
    
    eta_10_data = load_results(eta_10_path)
    eta_01_data = load_results(eta_01_path)
    
    # 1. Pricing assumptions
    print("\n1. PRICING ASSUMPTIONS (2026 Approximate Rates)")
    print("-"*100)
    
    print(f"\nGPT-4-Turbo:")
    print(f"  Input:  ${GPT4_INPUT_COST}/1M tokens")
    print(f"  Output: ${GPT4_OUTPUT_COST}/1M tokens")
    print(f"  Cost per query: ${compute_query_cost('gpt-4-turbo'):.6f}")
    
    print(f"\nMixtral-8x7B-Instruct:")
    print(f"  Input:  ${MIXTRAL_INPUT_COST}/1M tokens")
    print(f"  Output: ${MIXTRAL_OUTPUT_COST}/1M tokens")
    print(f"  Cost per query: ${compute_query_cost('mixtral'):.6f}")
    
    cost_ratio = compute_query_cost('gpt-4-turbo') / compute_query_cost('mixtral')
    print(f"\nCost Ratio: GPT-4-Turbo is {cost_ratio:.1f}× more expensive than Mixtral")
    
    print(f"\nToken Assumptions:")
    print(f"  Average prompt: {AVG_PROMPT_TOKENS} tokens")
    print(f"  Average completion: {AVG_COMPLETION_TOKENS} tokens")
    
    # 2. Baseline costs
    print("\n2. BASELINE STRATEGY COSTS")
    print("-"*100)
    
    warmup = eta_10_data['Warmup'][0]
    tabula_rasa = eta_10_data['Tabula Rasa'][0]
    
    warmup_cost = compute_strategy_cost(warmup['model_usage'])
    tr_cost = compute_strategy_cost(tabula_rasa['model_usage'])
    
    print(f"\n{'Strategy':<25} {'GPT-4-Turbo %':<15} {'Cost/Query':<15} {'Cost/1K':<15} {'Regret':<10}")
    print("-"*100)
    print(f"{'Warmup (Harmful)':<25} {warmup_cost['gpt4_pct']:>14.1f}% "
          f"${warmup_cost['cost_per_query']:>14.6f} ${warmup_cost['cost_per_1k_queries']:>14.4f} "
          f"{warmup['cumulative_regret']:>9.1f}")
    print(f"{'Tabula Rasa (Optimal)':<25} {tr_cost['gpt4_pct']:>14.1f}% "
          f"${tr_cost['cost_per_query']:>14.6f} ${tr_cost['cost_per_1k_queries']:>14.4f} "
          f"{tabula_rasa['cumulative_regret']:>9.1f}")
    
    # 3. Corralling costs (all seeds)
    print("\n3. CORRALLING COSTS: η=1.0 (Aggressive Learning)")
    print("-"*100)
    
    corralling_10 = eta_10_data['Hybrid (Corralling)']
    
    print(f"\n{'Seed':<8} {'GPT-4-Turbo %':<15} {'Cost/Query':<15} {'Cost/1K':<15} {'Regret':<10} {'Category':<20}")
    print("-"*100)
    
    costs_10 = []
    for seed_data in corralling_10:
        cost = compute_strategy_cost(seed_data['model_usage'])
        costs_10.append(cost)
        
        category = (
            'CATASTROPHIC' if seed_data['cumulative_regret'] > 70 else
            'Poor' if seed_data['cumulative_regret'] > 50 else
            'Good' if seed_data['cumulative_regret'] > 40 else
            'Excellent'
        )
        
        print(f"Seed {seed_data['seed']:<3} {cost['gpt4_pct']:>14.1f}% "
              f"${cost['cost_per_query']:>14.6f} ${cost['cost_per_1k_queries']:>14.4f} "
              f"{seed_data['cumulative_regret']:>9.1f} {category:<20}")
    
    # Mean costs
    mean_cost_per_query_10 = np.mean([c['cost_per_query'] for c in costs_10])
    mean_cost_per_1k_10 = np.mean([c['cost_per_1k_queries'] for c in costs_10])
    mean_gpt4_pct_10 = np.mean([c['gpt4_pct'] for c in costs_10])
    
    print("-"*100)
    print(f"{'MEAN':<8} {mean_gpt4_pct_10:>14.1f}% "
          f"${mean_cost_per_query_10:>14.6f} ${mean_cost_per_1k_10:>14.4f}")
    
    # 4. Corralling costs: η=0.1
    print("\n4. CORRALLING COSTS: η=0.1 (Conservative Learning)")
    print("-"*100)
    
    corralling_01 = eta_01_data['Hybrid (Corralling)']
    
    print(f"\n{'Seed':<8} {'GPT-4-Turbo %':<15} {'Cost/Query':<15} {'Cost/1K':<15} {'Regret':<10}")
    print("-"*100)
    
    costs_01 = []
    for seed_data in corralling_01:
        cost = compute_strategy_cost(seed_data['model_usage'])
        costs_01.append(cost)
        
        print(f"Seed {seed_data['seed']:<3} {cost['gpt4_pct']:>14.1f}% "
              f"${cost['cost_per_query']:>14.6f} ${cost['cost_per_1k_queries']:>14.4f} "
              f"{seed_data['cumulative_regret']:>9.1f}")
    
    mean_cost_per_query_01 = np.mean([c['cost_per_query'] for c in costs_01])
    mean_cost_per_1k_01 = np.mean([c['cost_per_1k_queries'] for c in costs_01])
    mean_gpt4_pct_01 = np.mean([c['gpt4_pct'] for c in costs_01])
    
    print("-"*100)
    print(f"{'MEAN':<8} {mean_gpt4_pct_01:>14.1f}% "
          f"${mean_cost_per_query_01:>14.6f} ${mean_cost_per_1k_01:>14.4f}")
    
    # 5. Cost comparison
    print("\n5. COST COMPARISON: SUMMARY")
    print("-"*100)
    
    print(f"\n{'Strategy':<30} {'GPT-4-Turbo %':<15} {'Cost/1K':<15} {'vs Tabula Rasa':<20} {'Regret':<10}")
    print("-"*100)
    
    strategies = [
        ('Tabula Rasa (Baseline)', tr_cost, tabula_rasa['cumulative_regret']),
        ('Warmup (Harmful)', warmup_cost, warmup['cumulative_regret']),
        ('Corralling η=0.1 (Mean)', 
         {'gpt4_pct': mean_gpt4_pct_01, 'cost_per_1k_queries': mean_cost_per_1k_01}, 
         np.mean([s['cumulative_regret'] for s in corralling_01])),
        ('Corralling η=1.0 (Mean)', 
         {'gpt4_pct': mean_gpt4_pct_10, 'cost_per_1k_queries': mean_cost_per_1k_10}, 
         np.mean([s['cumulative_regret'] for s in corralling_10])),
    ]
    
    for name, cost_dict, regret in strategies:
        vs_tr = 100 * (cost_dict['cost_per_1k_queries'] / tr_cost['cost_per_1k_queries'] - 1)
        print(f"{name:<30} {cost_dict['gpt4_pct']:>14.1f}% "
              f"${cost_dict['cost_per_1k_queries']:>14.4f} {vs_tr:>+19.1f}% "
              f"{regret:>9.1f}")
    
    # 6. Key findings
    print("\n6. KEY FINDINGS")
    print("-"*100)
    
    print("\n🔴 COST IMPLICATIONS:")
    print()
    print(f"1. Corralling uses MORE GPT-4-Turbo than Tabula Rasa:")
    print(f"   • Tabula Rasa: {tr_cost['gpt4_pct']:.1f}% GPT-4-Turbo")
    print(f"   • η=0.1: {mean_gpt4_pct_01:.1f}% GPT-4-Turbo (+{mean_gpt4_pct_01 - tr_cost['gpt4_pct']:.1f} pp)")
    print(f"   • η=1.0: {mean_gpt4_pct_10:.1f}% GPT-4-Turbo (+{mean_gpt4_pct_10 - tr_cost['gpt4_pct']:.1f} pp)")
    print()
    print(f"2. Corralling is MORE EXPENSIVE than Tabula Rasa:")
    cost_increase_01 = 100 * (mean_cost_per_1k_01 / tr_cost['cost_per_1k_queries'] - 1)
    cost_increase_10 = 100 * (mean_cost_per_1k_10 / tr_cost['cost_per_1k_queries'] - 1)
    print(f"   • η=0.1: +{cost_increase_01:.1f}% cost vs Tabula Rasa")
    print(f"   • η=1.0: +{cost_increase_10:.1f}% cost vs Tabula Rasa")
    print()
    print(f"3. Cost-Quality Tradeoff:")
    print(f"   • Tabula Rasa: ${tr_cost['cost_per_1k_queries']:.4f}/1K, 40 regret")
    print(f"   • η=0.1: ${mean_cost_per_1k_01:.4f}/1K (+{cost_increase_01:.1f}%), "
          f"{np.mean([s['cumulative_regret'] for s in corralling_01]):.1f} regret")
    print(f"   • η=1.0: ${mean_cost_per_1k_10:.4f}/1K (+{cost_increase_10:.1f}%), "
          f"{np.mean([s['cumulative_regret'] for s in corralling_10]):.1f} regret")
    print()
    print("💡 INTERPRETATION:")
    print("   • Corralling provides SAFETY (protects against harmful warmup)")
    print("   • But at a COST: Higher GPT-4-Turbo usage than optimal baseline")
    print("   • This is the 'insurance premium' for robustness to domain mismatch")
    
    # 7. Production scale projections
    print("\n7. PRODUCTION SCALE PROJECTIONS")
    print("-"*100)
    
    scales = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    
    print(f"\n{'Queries/Month':<20} {'Tabula Rasa':<15} {'η=0.1':<15} {'η=1.0':<15} {'Extra Cost (η=1.0)':<20}")
    print("-"*100)
    
    for scale in scales:
        tr_monthly = scale * tr_cost['cost_per_query']
        corralling_01_monthly = scale * mean_cost_per_query_01
        corralling_10_monthly = scale * mean_cost_per_query_10
        extra_10 = corralling_10_monthly - tr_monthly
        
        print(f"{scale:>19,} ${tr_monthly:>14,.2f} ${corralling_01_monthly:>14,.2f} "
              f"${corralling_10_monthly:>14,.2f} ${extra_10:>+19,.2f}/mo")
    
    # 8. Save report
    save_cost_analysis_report(
        warmup_cost, tr_cost,
        costs_01, costs_10,
        mean_cost_per_1k_01, mean_cost_per_1k_10
    )
    
    print("\n" + "="*100)
    print("COST ANALYSIS COMPLETE")
    print("="*100)


def save_cost_analysis_report(warmup_cost, tr_cost, costs_01, costs_10, 
                               mean_cost_01, mean_cost_10):
    """Save machine-readable cost report."""
    
    report = {
        'pricing_assumptions': {
            'gpt4_input_per_1m': GPT4_INPUT_COST,
            'gpt4_output_per_1m': GPT4_OUTPUT_COST,
            'mixtral_input_per_1m': MIXTRAL_INPUT_COST,
            'mixtral_output_per_1m': MIXTRAL_OUTPUT_COST,
            'avg_prompt_tokens': AVG_PROMPT_TOKENS,
            'avg_completion_tokens': AVG_COMPLETION_TOKENS,
            'gpt4_cost_per_query': compute_query_cost('gpt-4-turbo'),
            'mixtral_cost_per_query': compute_query_cost('mixtral')
        },
        'baselines': {
            'warmup': warmup_cost,
            'tabula_rasa': tr_cost
        },
        'corralling_eta_0.1': {
            'mean_cost_per_1k': float(mean_cost_01),
            'mean_gpt4_pct': float(np.mean([c['gpt4_pct'] for c in costs_01])),
            'cost_increase_vs_tr': float(100 * (mean_cost_01 / tr_cost['cost_per_1k_queries'] - 1)),
            'per_seed': costs_01
        },
        'corralling_eta_1.0': {
            'mean_cost_per_1k': float(mean_cost_10),
            'mean_gpt4_pct': float(np.mean([c['gpt4_pct'] for c in costs_10])),
            'cost_increase_vs_tr': float(100 * (mean_cost_10 / tr_cost['cost_per_1k_queries'] - 1)),
            'per_seed': costs_10
        },
        'key_findings': {
            'corralling_more_expensive_than_tabula_rasa': True,
            'cost_increase_eta_0.1_pct': float(100 * (mean_cost_01 / tr_cost['cost_per_1k_queries'] - 1)),
            'cost_increase_eta_1.0_pct': float(100 * (mean_cost_10 / tr_cost['cost_per_1k_queries'] - 1)),
            'interpretation': 'Corralling provides safety against harmful warmup at cost of higher GPT-4 usage'
        }
    }
    
    output_file = Path(__file__).parent / 'data' / 'cost_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Saved cost analysis report: {output_file}")


if __name__ == '__main__':
    cost_analysis()
