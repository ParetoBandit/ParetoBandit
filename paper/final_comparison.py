#!/usr/bin/env python3
"""
Final comparison: FrugalGPT vs LLM Jury
Using FrugalGPT's published data and Gemini 3 as LLM Jury baseline.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_jury import get_best_models_for_budget

PAPER_DIR = Path(__file__).parent

# FrugalGPT results from their HEADLINES evaluation
FRUGALGPT_RESULTS = {
    # model: (accuracy%, cost_per_1k_queries, cost_per_M_tokens_approx)
    'textsynth/gptj_6B': (77.5, 0.06, 0.09),
    'textsynth/fairseq_gpt_13B': (72.4, 0.16, 0.24),
    'ai21/j1-large': (75.3, 0.36, 0.30),
    'textsynth/gptneox_20B': (82.3, 0.39, 0.58),
    'openaichat/gpt-3.5-turbo': (81.0, 0.44, 2.0),
    'openai/text-curie-001': (69.8, 0.46, 2.0),
    'ai21/j1-grande': (78.8, 0.96, 0.80),
    'cohere/xlarge': (78.0, 1.79, 2.50),
    'cohere/medium': (79.5, 1.79, 2.50),
    'openai/text-davinci-002': (83.4, 4.56, 20.0),
    'ai21/j1-jumbo': (82.5, 5.50, 25.0),
    'openaichat/gpt-4': (85.6, 6.66, 30.0),
}

def get_frugalgpt_best_at_budget(max_cost_per_m):
    """Get best FrugalGPT model within budget."""
    best = None
    best_acc = 0
    for model, (acc, cost_1k, cost_m) in FRUGALGPT_RESULTS.items():
        if cost_m <= max_cost_per_m and acc > best_acc:
            best = model
            best_acc = acc
    return best

def main():
    print("=" * 90)
    print("COMPARISON: FrugalGPT (2023) vs LLM Jury (2025)")
    print("Dataset: HEADLINES (5,000 test queries)")
    print("LLM Jury Baseline: Gemini 3 Pro Preview")
    print("=" * 90)
    
    # Budget tiers to compare ($/M tokens)
    budgets = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    
    # Get Gemini 3 baseline stats
    baseline_name = "Gemini 3 Pro Preview (high)"
    
    print(f"\nGetting LLM Jury recommendations (baseline: {baseline_name})...")
    
    comparison_data = []
    
    for budget in budgets:
        print(f"\n--- Budget: ${budget}/M ---")
        
        # LLM Jury recommendation
        results = get_best_models_for_budget(
            max_budget=budget,
            baseline_model_name=baseline_name,
            top_k=1,
            verbose=False
        )
        
        if results:
            llm_model = results[0].model_name
            llm_reasoning = results[0].reasoning
            # Parse reasoning for quality, latency, cost
            parts = llm_reasoning.split('|')
            llm_quality = float(parts[0].split(':')[1].strip())
            llm_latency = float(parts[1].split(':')[1].replace('ms', '').strip())
            llm_cost = float(parts[2].split('$')[1].split('/')[0].strip())
        else:
            llm_model = "No model found"
            llm_quality = 0
            llm_latency = 0
            llm_cost = 0
        
        # Best FrugalGPT model at this budget
        frugal_model = get_frugalgpt_best_at_budget(budget)
        if frugal_model:
            frugal_acc, frugal_cost_1k, frugal_cost_m = FRUGALGPT_RESULTS[frugal_model]
            frugal_model_short = frugal_model.split('/')[-1]
        else:
            frugal_model_short = "None"
            frugal_acc = 0
            frugal_cost_m = 0
        
        comparison_data.append({
            'budget': budget,
            'frugalgpt_model': frugal_model_short,
            'frugalgpt_accuracy': frugal_acc,
            'frugalgpt_cost': frugal_cost_m,
            'llm_jury_model': llm_model,
            'llm_jury_quality': llm_quality,
            'llm_jury_cost': llm_cost,
            'llm_jury_latency': llm_latency,
        })
        
        print(f"  FrugalGPT: {frugal_model_short} ({frugal_acc:.1f}% acc, ${frugal_cost_m:.2f}/M)")
        print(f"  LLM Jury:  {llm_model} (Quality: {llm_quality:.1f}, ${llm_cost:.2f}/M, {llm_latency:.0f}ms)")
    
    # Print comparison table
    print("\n" + "=" * 120)
    print("COMPARISON TABLE")
    print("=" * 120)
    
    # Header
    print(f"\n{'Budget':<10} | {'FrugalGPT Selection':<20} {'Acc%':<8} {'$/M':<8} | "
          f"{'LLM Jury Selection':<25} {'Quality':<10} {'$/M':<8} {'Latency':<10}")
    print("-" * 120)
    
    for row in comparison_data:
        print(f"${row['budget']:<9.1f} | {row['frugalgpt_model']:<20} {row['frugalgpt_accuracy']:<8.1f} "
              f"${row['frugalgpt_cost']:<7.2f} | {row['llm_jury_model']:<25} {row['llm_jury_quality']:<10.1f} "
              f"${row['llm_jury_cost']:<7.2f} {row['llm_jury_latency']:<10.0f}ms")
    
    print("-" * 120)
    
    # Key insights
    print("\n" + "=" * 90)
    print("KEY INSIGHTS")
    print("=" * 90)
    print("""
1. FrugalGPT (2023): Uses cascade routing trained on labeled data
   - Requires labeled training data for each task
   - Multiple model calls per query (cascade)
   - Optimizes for accuracy within cost budget

2. LLM Jury (2025): Uses target-driven optimization with NO labeled data
   - Zero labeled data required
   - Single model call (no cascade overhead)
   - Optimizes for quality, cost, AND latency simultaneously
   - Modern models with higher benchmark scores

3. Model Landscape Evolution:
   - 2023 models (GPT-4, GPT-3.5, J1) have been superseded
   - 2025 models (GPT-5.1, Gemini 3) offer better quality/cost ratios
   - Direct accuracy comparison is limited by model availability
""")
    
    # Save results
    output_path = PAPER_DIR / "figures" / "final_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

