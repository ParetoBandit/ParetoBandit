#!/usr/bin/env python3
"""Test updated Chebyshev ranker with AA benchmark data."""

import json
from pathlib import Path

from llm_jury.ranking.chebyshev import ChebyshevRanker, RankingStrategy
from llm_jury.core.models import ModelMetadata, RoutingDecision, PromptCategory, ProductArchetype
from llm_jury.data.registry import ModelRegistry


def main():
    # Load models from cache
    cache_path = Path("data/models_complete_composite_indices.json")
    
    if not cache_path.exists():
        print(f"❌ Cache not found: {cache_path}")
        print("Run: python run_etl.py --complete-only")
        return
    
    print("Loading model data...")
    with open(cache_path) as f:
        raw_data = json.load(f)
    
    print(f"✓ Loaded {len(raw_data)} models\n")
    
    # Convert raw data to simple ModelMetadata objects for testing
    models = []
    for model_dict in raw_data:
        # Create minimal ModelMetadata for testing
        model = ModelMetadata(
            name=model_dict['name'],
            archetype=ProductArchetype.FRONTIER,
            input_cost_per_m=model_dict.get('price_1m_input', 1.0),
            output_cost_per_m=model_dict.get('price_1m_output', 2.0),
            median_latency_ms=1000.0,  # Default
            context_window_k=128,
            param_count_b=70.0,
            mmlu_score=0,
            gpqa_score=0,
            math_score=0,
            ifeval_score=0,
            tool_use_ability=0.5
        )
        
        # Add AA benchmark attributes dynamically
        for key in ['intelligence_index', 'coding_index', 'math_index', 
                     'mmlu_pro', 'gpqa', 'hle', 'livecodebench', 'scicode', 
                     'math_500', 'aime', 'output_tokens_per_second']:
            setattr(model, key, model_dict.get(key))
        
        models.append(model)
    
    print(f"✓ Converted {len(models)} models to ModelMetadata\n")
    
    # Find baseline model (e.g., GPT-4)
    baseline = None
    for m in models:
        if 'gpt-4' in m.name.lower() and 'mini' not in m.name.lower():
            baseline = m
            break
    
    if not baseline:
        # Fallback to first model with good data
        baseline = [m for m in models if m.input_cost_per_m and m.input_cost_per_m > 0][0]
        print(f"⚠️ Using {baseline.name} as baseline (GPT-4 not found)")
    else:
        print(f"✓ Using {baseline.name} as baseline\n")
    
    # Initialize ranker
    print("="*80)
    print("Initializing ChebyshevRanker with AA benchmark data...")
    print("="*80)
    ranker = ChebyshevRanker(
        baseline_model=baseline,
        all_models_data=raw_data,
        strategy=RankingStrategy.BALANCED
    )
    print("✓ Ranker initialized\n")
    
    # Test different scenarios
    scenarios = [
        ("General Use", RoutingDecision(
            category=PromptCategory.GENERAL,
            archetype=ProductArchetype.FRONTIER,
            reason="General task"
        )),
        ("Coding Task", RoutingDecision(
            category=PromptCategory.CODING,
            archetype=ProductArchetype.REASONING_SPECIALIST,
            reason="Coding task"
        )),
        ("Data Science", RoutingDecision(
            category=PromptCategory.DATA_SCIENCE,
            archetype=ProductArchetype.REASONING_SPECIALIST,
            reason="Data science task"
        )),
        ("Creative Writing", RoutingDecision(
            category=PromptCategory.CREATIVE,
            archetype=ProductArchetype.FRONTIER,
            reason="Creative task"
        )),
    ]
    
    for scenario_name, decision in scenarios:
        print("\n" + "="*80)
        print(f"📊 Scenario: {scenario_name}")
        print("="*80)
        
        # Rank models
        top_models = ranker.rank(
            models=models,
            decision=decision,
            top_k=10,
            return_detailed=False
        )
        
        print(f"\nTop 10 models (Strategy: {ranker.strategy.value}):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<45} {'Cheb Score':<12} {'Summary'}")
        print("-" * 80)
        
        for result in top_models:
            print(f"{result.rank:<6} {result.model_name:<45} {result.score:<12.4f} {result.reasoning}")
    
    # Test different strategies for one scenario
    print("\n" + "="*80)
    print("📊 Strategy Comparison: Coding Task")
    print("="*80)
    
    coding_decision = RoutingDecision(
        category=PromptCategory.CODING,
        archetype=ProductArchetype.REASONING_SPECIALIST,
        reason="Coding task"
    )
    
    strategies = [
        RankingStrategy.QUALITY_FOCUSED,
        RankingStrategy.COST_FOCUSED,
        RankingStrategy.SPEED_FOCUSED,
        RankingStrategy.BALANCED
    ]
    
    for strategy in strategies:
        ranker_test = ChebyshevRanker(
            baseline_model=baseline,
            all_models_data=raw_data,
            strategy=strategy
        )
        
        top_3 = ranker_test.rank(
            models=models,
            decision=coding_decision,
            top_k=3,
            return_detailed=False
        )
        
        print(f"\n{strategy.value.upper()} Strategy:")
        for i, result in enumerate(top_3, 1):
            print(f"  {i}. {result.model_name} (Cheb: {result.score:.4f})")
    
    print("\n" + "="*80)
    print("✅ Chebyshev Ranker Test Complete!")
    print("="*80)
    print("""
KEY FEATURES:
  ✓ Uses AA benchmark data for quality scoring
  ✓ Task-specific quality assessment (coding vs creative vs data science)
  ✓ Multi-objective optimization (quality + cost + speed)
  ✓ Chebyshev scalarization (minimizes worst-case regret)
  ✓ Multiple strategy profiles (quality/cost/speed/balanced)
    """)


if __name__ == "__main__":
    main()

