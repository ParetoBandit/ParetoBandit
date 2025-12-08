#!/usr/bin/env python3
"""
Test configurable baseline/reference model.

Demonstrates how users can specify different reference models
to find value-optimized alternatives relative to their current model.
"""

import json
from pathlib import Path

from llm_jury.ranking.chebyshev import ChebyshevRanker, RankingStrategy
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype


def create_model(d):
    """Create ModelMetadata from dict."""
    m = ModelMetadata(
        name=d['name'], archetype=ProductArchetype.FRONTIER,
        input_cost_per_m=d.get('price_1m_input', 1.0),
        output_cost_per_m=d.get('price_1m_output', 2.0),
        median_latency_ms=1000.0, context_window_k=128, param_count_b=70.0,
        mmlu_score=0, gpqa_score=0, math_score=0, ifeval_score=0, tool_use_ability=0.5
    )
    for key in ['intelligence_index', 'coding_index', 'math_index', 'output_tokens_per_second']:
        setattr(m, key, d.get(key))
    return m


def test_baseline(baseline_name, models, valid_data):
    """Test optimization with a specific baseline."""
    baseline = next((m for m in models if baseline_name.lower() in m.name.lower()), None)
    
    if baseline is None:
        print(f"⚠️  Baseline '{baseline_name}' not found. Skipping.")
        return
    
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    print(f"\n{'=' * 100}")
    print(f"BASELINE: {baseline.name}")
    print(f"Cost: ${base_cost:.2f}/M tokens")
    print('=' * 100)
    
    # Find sweet spot models (80-95% quality, 10-30% cost)
    ranker = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=valid_data,
        strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 0.95),
        cost_range=(0.10, 0.30)
    )
    
    decision = RoutingDecision(
        category=PromptCategory.CODING, 
        archetype=ProductArchetype.FRONTIER, 
        reason="Coding task"
    )
    
    results = ranker.rank(models, decision, top_k=8, return_detailed=True)
    
    if not results:
        print("\n⚠️  No models found matching sweet spot constraints.")
        return
    
    print(f"\nTop {len(results)} Sweet Spot Alternatives (80-95% quality, 10-30% cost):")
    print("-" * 100)
    
    for i, r in enumerate(results, 1):
        cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
        savings = base_cost - cost
        savings_pct = (savings / base_cost) * 100
        
        print(f"  {i}. {r.name:45s} "
              f"Q:{r.quality_score:5.1f}  "
              f"Cost:${cost:5.2f}  "
              f"Save: ${savings:5.2f} ({savings_pct:3.0f}%)  "
              f"Cheb:{r.chebyshev_score:.4f}")


def main():
    # Load data
    cache_path = Path("data/models_complete_composite_indices.json")
    
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    valid_data = [m for m in raw_data if m.get('price_1m_input') and m.get('price_1m_input') > 0]
    models = [create_model(d) for d in valid_data]
    
    print("CONFIGURABLE BASELINE/REFERENCE MODEL")
    print("=" * 100)
    print("Demonstration: Find value alternatives relative to DIFFERENT baseline models")
    print("=" * 100)
    
    # Test different baselines
    baselines = [
        "GPT-5.1 (high)",
        "Claude 3.5 Sonnet (new)",
        "Gemini 2.5 Pro",
        "GPT-4o",
    ]
    
    for baseline_name in baselines:
        test_baseline(baseline_name, models, valid_data)
    
    # Summary
    print("\n" + "=" * 100)
    print("KEY INSIGHT: Different Baselines → Different Sweet Spots")
    print("=" * 100)
    print("""
When you change the baseline/reference model, the "sweet spot" changes too:

1. **Expensive Baseline** (e.g., GPT-5.1 high @ $3.44):
   → Sweet spot: $0.34-$1.03 models with 80-95% quality
   → Large absolute savings ($2-3 per M tokens)

2. **Mid-Tier Baseline** (e.g., Claude 3.5 Sonnet @ $3.00):
   → Sweet spot: $0.30-$0.90 models with 80-95% quality
   → Moderate absolute savings

3. **Cheaper Baseline** (e.g., GPT-4o @ $2.50):
   → Sweet spot: $0.25-$0.75 models with 80-95% quality
   → Smaller absolute savings, but still significant percentages

USAGE:
    from llm_jury import get_recommendations
    from llm_jury.ranking.chebyshev import RankingStrategy
    
    # Find alternatives to your current model
    results = get_recommendations(
        prompt="Your task here",
        baseline_model_name="YOUR_CURRENT_MODEL",  # ← Configurable!
        ranking_strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 0.95),
        cost_range=(0.10, 0.30)
    )

This allows users to find cost-effective alternatives to THEIR specific model,
not just a hardcoded default!
    """)


if __name__ == "__main__":
    main()

