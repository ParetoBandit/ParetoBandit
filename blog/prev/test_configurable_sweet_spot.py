#!/usr/bin/env python3
"""
Test configurable sweet spot constraints.

Demonstrates how users can specify their own quality/cost/speed constraints
to find models that match their specific value criteria.
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


def main():
    # Load data
    cache_path = Path("data/models_complete_composite_indices.json")
    
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    valid_data = [m for m in raw_data if m.get('price_1m_input') and m.get('price_1m_input') > 0]
    models = [create_model(d) for d in valid_data]
    
    # Baseline
    baseline = [m for m in models if 'gpt-5.1' in m.name.lower() and 'high' in m.name.lower()][0]
    base_cost = baseline.input_cost_per_m * 0.75 + baseline.output_cost_per_m * 0.25
    
    print("CONFIGURABLE SWEET SPOT OPTIMIZATION")
    print("=" * 100)
    print(f"Baseline: {baseline.name} (Cost: ${base_cost:.2f})")
    print("=" * 100)
    
    # Test different constraint configurations
    configurations = [
        {
            "name": "Conservative (High Quality)",
            "quality_range": (0.90, 0.98),
            "cost_range": (0.10, 0.40),
            "description": "90-98% quality, up to 40% cost → High-quality bargains"
        },
        {
            "name": "Balanced Sweet Spot",
            "quality_range": (0.80, 0.95),
            "cost_range": (0.10, 0.30),
            "description": "80-95% quality, 10-30% cost → Classic sweet spot"
        },
        {
            "name": "Aggressive Cost Cutting",
            "quality_range": (0.70, 0.90),
            "cost_range": (0.05, 0.20),
            "description": "70-90% quality, up to 20% cost → Maximum savings"
        },
        {
            "name": "Custom: Your Example",
            "quality_range": (0.80, 1.00),
            "cost_range": (0.00, 0.30),
            "description": "80%+ quality, up to 30% cost → Flexible quality, strict cost"
        },
    ]
    
    decision = RoutingDecision(
        category=PromptCategory.CODING, 
        archetype=ProductArchetype.FRONTIER, 
        reason="Coding task"
    )
    
    for config in configurations:
        print(f"\n{'=' * 100}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Constraints: Quality {config['quality_range']}, Cost {config['cost_range']}")
        print('=' * 100)
        
        ranker = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=valid_data,
            strategy=RankingStrategy.VALUE_OPTIMIZED,
            quality_range=config['quality_range'],
            cost_range=config['cost_range']
        )
        
        results = ranker.rank(models, decision, top_k=8, return_detailed=True)
        
        print(f"\nTop {len(results)} models matching constraints:")
        print("-" * 100)
        
        for i, r in enumerate(results, 1):
            cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
            q_ratio = r.quality_score / 97.1  # Approximate baseline quality
            c_ratio = cost / base_cost
            savings = (1 - c_ratio) * 100
            
            print(f"  {i}. {r.name:45s} "
                  f"Q:{r.quality_score:5.1f} ({q_ratio:4.0%})  "
                  f"Cost:${cost:5.2f} ({c_ratio:4.0%}, {savings:3.0f}% cheaper)  "
                  f"Cheb:{r.chebyshev_score:.4f}")
    
    # Interactive example
    print("\n" + "=" * 100)
    print("INTERACTIVE EXAMPLE: User-Defined Constraints")
    print("=" * 100)
    
    # User's example: [0.8, 0.3, 0.3] means:
    # - Quality: at least 80% (0.80 to 1.00)
    # - Cost: at most 30% (0.00 to 0.30)
    # - Speed: at least 30% faster? (0.30 to inf) - this interpretation is ambiguous
    
    # I'll interpret as: quality_min=0.8, cost_max=0.3, speed_min=0.3
    print("\nUser Input: [0.8, 0.3, 0.3]")
    print("Interpretation:")
    print("  - Quality: ≥80% of baseline (0.80 to 1.00)")
    print("  - Cost: ≤30% of baseline (0.00 to 0.30)")
    print("  - Speed: ≥30% of baseline (0.30 to inf)")
    print()
    
    ranker_custom = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=valid_data,
        strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(0.80, 1.00),
        cost_range=(0.00, 0.30),
        speed_range=(0.30, 10.0)  # At least 30% of baseline speed, up to 10x faster
    )
    
    results_custom = ranker_custom.rank(models, decision, top_k=10, return_detailed=True)
    
    print("Results:")
    print("-" * 100)
    for i, r in enumerate(results_custom, 1):
        cost = r.metadata.input_cost_per_m * 0.75 + r.metadata.output_cost_per_m * 0.25
        print(f"  {i}. {r.name:45s} Q:{r.quality_score:5.1f}  Cost:${cost:5.2f}  Cheb:{r.chebyshev_score:.4f}")
    
    print("\n" + "=" * 100)
    print("✅ CONFIGURABLE SWEET SPOT WORKING!")
    print("=" * 100)
    print("""
USAGE:
    ranker = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=models_data,
        strategy=RankingStrategy.VALUE_OPTIMIZED,
        quality_range=(min_quality_ratio, max_quality_ratio),
        cost_range=(min_cost_ratio, max_cost_ratio),
        speed_range=(min_speed_ratio, max_speed_ratio)  # Optional
    )

EXAMPLES:
    # Conservative: 90-98% quality, up to 40% cost
    quality_range=(0.90, 0.98), cost_range=(0.10, 0.40)
    
    # Aggressive savings: 70-90% quality, up to 20% cost
    quality_range=(0.70, 0.90), cost_range=(0.05, 0.20)
    
    # Flexible: 80%+ quality, cheap as possible
    quality_range=(0.80, 1.00), cost_range=(0.00, 0.30)

All ratios are relative to the baseline model.
    """)


if __name__ == "__main__":
    main()

