#!/usr/bin/env python3
"""
Generate real HYBRID rankings for the frontend demo.

This script runs the actual optimizer algorithm to produce genuine rankings
for each use case at various constraint levels.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_jury.core.models import PromptCategory, RoutingDecision, ModelMetadata
from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy
from llm_jury.ranking.quality_scorer import QualityScorer


def load_models():
    """Load models from cache."""
    cache_path = Path(__file__).parent.parent / 'data' / 'models_cache.json'
    with open(cache_path) as f:
        return json.load(f)


def model_dict_to_metadata(m: dict) -> ModelMetadata:
    """Convert model dict to ModelMetadata."""
    return ModelMetadata(
        name=m.get('name', ''),
        intelligence_index=m.get('intelligence_index'),
        coding_index=m.get('coding_index'),
        math_index=m.get('math_index'),
        mmlu_pro=m.get('mmlu_pro'),
        gpqa=m.get('gpqa'),
        hle=m.get('hle'),
        livecodebench=m.get('livecodebench'),
        scicode=m.get('scicode'),
        math_500=m.get('math_500'),
        aime=m.get('aime'),
        input_cost_per_m=m.get('price_1m_input') or m.get('input_cost_per_m'),
        output_cost_per_m=m.get('price_1m_output') or m.get('output_cost_per_m'),
        hallucination_rate=m.get('hallucination_rate'),
        refusal_rate=m.get('refusal_rate'),
        measured_ttft_seconds=m.get('measured_ttft_seconds') or m.get('time_to_first_token_seconds'),
        context_window_k=m.get('context_window_k') or (m.get('context_length', 0) // 1000 if m.get('context_length') else None),
    )


# Use case mapping - now using real PromptCategory enum values
USE_CASE_MAP = {
    'coding': PromptCategory.CODING,
    'data_science': PromptCategory.DATA_SCIENCE,
    'creative': PromptCategory.CREATIVE,
    'general': PromptCategory.GENERAL,
    'qa': PromptCategory.QA,          # Q&A - trust/accuracy focused (real category)
    'rag': PromptCategory.RAG,        # RAG - context window + cost focused (real category)
    'chatbot': PromptCategory.CHATBOT,  # Chatbot - latency + cost focused (real category)
}

# Use-case specific baseline models (the "premium incumbent" for comparison)
# These represent the premium model users might be using for each task
USE_CASE_BASELINES = {
    'coding': "GPT-5.1 (high)",              # Premium coding model ($7.81/M, 91.6 quality)
    'data_science': "GPT-5.1 (high)",        # Strong math/reasoning ($7.81/M, 93.7 quality)
    'creative': "Claude Opus 4.5 (Reasoning)",  # Known for creative writing ($20/M, 89.5 quality)
    'general': "GPT-5.1 (high)",             # Premium all-rounder ($7.81/M, 79.5 quality)
    'qa': "Claude Opus 4.5 (Reasoning)",     # Highest trust score ($20/M, 72.6 quality)
    'rag': "Gemini 3 Pro Preview (high)",    # Large context + high quality ($9.50/M, 1M context)
    'chatbot': "GPT-5 mini (high)",          # Fast, conversational baseline ($1.56/M, good latency)
}

# Budget values
BUDGET_VALUES = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00]
BUDGET_KEYS = {0.10: '0_1', 0.25: '0_25', 0.50: '0_5', 1.00: '1_0', 2.00: '2_0', 5.00: '5_0'}


def get_blended_cost(m: dict) -> float:
    """Get blended cost for a model."""
    if m.get('price_1m_blended'):
        return m['price_1m_blended']
    input_cost = m.get('price_1m_input') or m.get('input_cost_per_m') or 0
    output_cost = m.get('price_1m_output') or m.get('output_cost_per_m') or 0
    return (input_cost + output_cost * 3) / 4  # 3:1 output:input ratio


def generate_hybrid_top10(models_data: list, use_case: str, baseline_name: str = None):
    """
    Generate top 10 HYBRID rankings for a use case.
    
    Uses the REAL optimizer algorithm with task-specific quality scoring:
    - CODING: coding_index, livecodebench, scicode weights (baseline: GPT-5.1)
    - DATA_SCIENCE: math_index, math_500, aime weights (baseline: GPT-5.1)
    - CREATIVE: intelligence_index, hle weights (baseline: Claude Opus 4.5)
    - GENERAL: intelligence + trust_score weights (baseline: GPT-5.1)
    - QA: trust_score (30%), intelligence, mmlu_pro weights (baseline: Claude Opus 4.5)
    - RAG: context_score (35%), intelligence, trust weights (baseline: Gemini 3 Pro)
    """
    
    # Use use-case specific baseline if not provided
    if baseline_name is None:
        baseline_name = USE_CASE_BASELINES.get(use_case, "GPT-5.1 (high)")
    
    # Find baseline
    baseline_dict = next((m for m in models_data if baseline_name in m.get('name', '')), None)
    if not baseline_dict:
        print(f"Warning: Baseline '{baseline_name}' not found, using first model")
        baseline_dict = models_data[0]
    
    baseline = model_dict_to_metadata(baseline_dict)
    baseline_cost = get_blended_cost(baseline_dict)
    baseline_ttft = baseline_dict.get('measured_ttft_seconds') or baseline_dict.get('time_to_first_token_seconds') or 32.254
    
    # Get quality scorer and category
    scorer = QualityScorer(models_data)
    category = USE_CASE_MAP.get(use_case, PromptCategory.GENERAL)
    baseline_quality = scorer.calculate_quality_score(baseline_dict, category)
    
    # Use standard HYBRID optimization for ALL use cases (including RAG/QA)
    # The task-specific scoring is now handled by QualityScorer weights
    optimizer = Optimizer(
        baseline_model=baseline,
        all_models_data=models_data,
        strategy=OptimizationStrategy.HYBRID,
    )
    
    # Convert models to metadata
    model_metas = [model_dict_to_metadata(m) for m in models_data]
    
    # Create decision
    decision = RoutingDecision(
        archetype=None,
        category=category,
        reason=f"Frontend demo - {use_case}"
    )
    
    # Rank models using the REAL algorithm
    try:
        results = optimizer.rank(model_metas, decision, top_k=10, verbose=False)
    except Exception as e:
        print(f"  Error ranking for {use_case}: {e}")
        return []
    
    # Build result list
    rankings = []
    for r in results:
        # Find original model data
        model_dict = next((m for m in models_data if m.get('name') == r.model_name), None)
        if not model_dict:
            continue
            
        cost = get_blended_cost(model_dict)
        ttft = model_dict.get('measured_ttft_seconds') or model_dict.get('time_to_first_token_seconds') or 1.0
        quality = scorer.calculate_quality_score(model_dict, category)
        
        # Calculate percentages vs baseline
        quality_pct = (quality / baseline_quality * 100) if baseline_quality > 0 else 0
        cost_pct = (cost / baseline_cost * 100) if baseline_cost > 0 else 0
        ttft_pct = (ttft / baseline_ttft * 100) if baseline_ttft > 0 else 0
        
        result = {
            "name": r.model_name,
            "quality": round(quality_pct, 1),
            "cost": round(cost, 2),
            "ttft": round(ttft, 2),
            "cost_pct": round(cost_pct, 1),
            "ttft_pct": round(ttft_pct, 1),
        }
        
        # Include context window for RAG use case
        if use_case == 'rag':
            ctx_k = model_dict.get('context_window_k') or (model_dict.get('context_length', 0) / 1000)
            result["context_k"] = int(ctx_k) if ctx_k else 32
        
        rankings.append(result)
    
    return rankings


def generate_budget_rankings(models_data: list, use_case: str, max_budget: float, baseline_name: str = None):
    """
    Generate HYBRID rankings for models under a budget.
    
    Uses the REAL optimizer algorithm with budget constraints.
    Uses use-case specific baselines for accurate comparisons.
    """
    
    # Filter models by budget
    budget_models = [m for m in models_data if get_blended_cost(m) <= max_budget]
    
    if not budget_models:
        return []
    
    # Use use-case specific baseline if not provided
    if baseline_name is None:
        baseline_name = USE_CASE_BASELINES.get(use_case, "GPT-5.1 (high)")
    
    # Find baseline (may not be in budget, that's ok)
    baseline_dict = next((m for m in models_data if baseline_name in m.get('name', '')), None)
    if not baseline_dict:
        baseline_dict = models_data[0]
    
    baseline = model_dict_to_metadata(baseline_dict)
    baseline_cost = get_blended_cost(baseline_dict)
    baseline_ttft = baseline_dict.get('measured_ttft_seconds') or baseline_dict.get('time_to_first_token_seconds') or 32.254
    
    scorer = QualityScorer(models_data)
    category = USE_CASE_MAP.get(use_case, PromptCategory.GENERAL)
    baseline_quality = scorer.calculate_quality_score(baseline_dict, category)
    
    # Use standard HYBRID optimization for ALL use cases
    optimizer = Optimizer(
        baseline_model=baseline,
        all_models_data=models_data,
        strategy=OptimizationStrategy.HYBRID,
    )
    
    # Convert budget models to metadata
    model_metas = [model_dict_to_metadata(m) for m in budget_models]
    
    decision = RoutingDecision(
        archetype=None,
        category=category,
        reason=f"Budget mode - {use_case}"
    )
    
    try:
        results = optimizer.rank(model_metas, decision, top_k=10, verbose=False)
    except Exception as e:
        print(f"  Error in budget ranking for {use_case} at ${max_budget}: {e}")
        return []
    
    rankings = []
    for r in results:
        model_dict = next((m for m in budget_models if m.get('name') == r.model_name), None)
        if not model_dict:
            continue
            
        cost = get_blended_cost(model_dict)
        ttft = model_dict.get('measured_ttft_seconds') or model_dict.get('time_to_first_token_seconds') or 1.0
        quality = scorer.calculate_quality_score(model_dict, category)
        
        quality_pct = (quality / baseline_quality * 100) if baseline_quality > 0 else 0
        cost_pct = (cost / baseline_cost * 100) if baseline_cost > 0 else 0
        ttft_pct = (ttft / baseline_ttft * 100) if baseline_ttft > 0 else 0
        
        result = {
            "name": r.model_name,
            "quality": round(quality_pct, 1),
            "cost": round(cost, 2),
            "ttft": round(ttft, 2),
            "cost_pct": round(cost_pct, 1),
            "ttft_pct": round(ttft_pct, 1),
            "score": round(r.score, 4) if hasattr(r, 'score') else 0.1,
        }
        
        # Include context window for RAG use case
        if use_case == 'rag':
            ctx_k = model_dict.get('context_window_k') or (model_dict.get('context_length', 0) / 1000)
            result["context_k"] = int(ctx_k) if ctx_k else 32
        
        rankings.append(result)
    
    return rankings


def main():
    print("=" * 70)
    print("GENERATING REAL HYBRID RANKINGS FOR FRONTEND")
    print("=" * 70)
    
    models_data = load_models()
    print(f"Loaded {len(models_data)} models\n")
    
    use_cases = ['coding', 'data_science', 'creative', 'general', 'qa', 'rag', 'chatbot']
    
    print("Use-case specific baselines:")
    for uc in use_cases:
        print(f"  {uc}: {USE_CASE_BASELINES.get(uc, 'default')}")
    print()
    
    # Generate HYBRID_TOP_10
    print("Generating HYBRID_TOP_10...")
    hybrid_top_10 = {}
    for uc in use_cases:
        baseline = USE_CASE_BASELINES.get(uc, 'default')
        print(f"  Processing {uc} (baseline: {baseline})...")
        rankings = generate_hybrid_top10(models_data, uc)
        hybrid_top_10[uc] = rankings
        if rankings:
            print(f"    Top model: {rankings[0]['name']} (Q:{rankings[0]['quality']:.0f}%)")
    
    # Generate BUDGET_RANKINGS
    print("\nGenerating BUDGET_RANKINGS...")
    budget_rankings = {}
    for uc in use_cases:
        print(f"  Processing {uc}...")
        for budget in BUDGET_VALUES:
            key = f"{uc}_{BUDGET_KEYS[budget]}"
            rankings = generate_budget_rankings(models_data, uc, budget)
            budget_rankings[key] = rankings
            if rankings:
                print(f"    ${budget:.2f}: {len(rankings)} models, top={rankings[0]['name']}")
    
    # Output JavaScript
    print("\n" + "=" * 70)
    print("JAVASCRIPT OUTPUT (copy to app.js)")
    print("=" * 70)
    
    print("\n// ============================================")
    print("// Top 10 HYBRID Rankings per Use Case (REAL DATA from optimizer)")
    print("// ============================================")
    print("const HYBRID_TOP_10 = " + json.dumps(hybrid_top_10, indent=4) + ";")
    
    print("\n// Pre-computed HYBRID rankings for each budget level and use case (REAL DATA)")
    print("const BUDGET_RANKINGS = " + json.dumps(budget_rankings, indent=4) + ";")
    
    # Also save to file for easier copy
    output_path = Path(__file__).parent / 'frontend_rankings_output.js'
    with open(output_path, 'w') as f:
        f.write("// Auto-generated by generate_frontend_rankings.py\n")
        f.write("// This is REAL data from the HYBRID optimizer algorithm\n\n")
        f.write("const HYBRID_TOP_10 = " + json.dumps(hybrid_top_10, indent=4) + ";\n\n")
        f.write("const BUDGET_RANKINGS = " + json.dumps(budget_rankings, indent=4) + ";\n")
    
    print(f"\n✅ Output also saved to: {output_path}")


if __name__ == "__main__":
    main()

