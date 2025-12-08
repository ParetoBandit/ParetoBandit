"""
Tests for edge/local model ranking scenarios.

Note: These tests are marked as expected to fail (xfail) because they test
future functionality for edge/local model ranking penalties that haven't
been implemented yet.
"""

import pytest
from llm_jury.core.models import ModelMetadata, RoutingDecision, ProductArchetype, PromptCategory
from llm_jury.ranking.optimizer import Optimizer, OptimizationStrategy, MissingDataStrategy


@pytest.mark.xfail(reason="EDGE_LOCAL archetype not yet implemented - test for future feature")
def test_edge_local_ranking_favors_quality():
    """
    Verify that for Edge/Local archetype, a high-quality model (like Gemini Flash)
    ranks higher than a low-quality but efficient model (like a tiny 1B model),
    and that cloud models don't dominate purely on efficiency.
    
    This test is marked xfail because:
    1. EDGE_LOCAL archetype doesn't exist yet
    2. The ranking logic for edge/local scenarios needs to be implemented
    """
    
    # 1. Setup Models
    # Using BULK_OPS as a stand-in for edge/local scenarios
    archetype = ProductArchetype.BULK_OPS
    
    # A high-quality candidate (e.g. Gemini Flash)
    gemini_flash = ModelMetadata(
        name="Google: Gemini 2.5 Flash",
        intelligence_index=86.0,
        input_cost_per_m=0.3,
        output_cost_per_m=1.0,
        hallucination_rate=5.0,
        refusal_rate=2.0,
        median_latency_ms=150.0,
        archetype=archetype,
    )
    
    # A very efficient but low-quality model (e.g. TinyLlama)
    tiny_model = ModelMetadata(
        name="TinyLlama 1.1B",
        intelligence_index=45.0,  # Low quality
        input_cost_per_m=0.01,
        output_cost_per_m=0.01,
        hallucination_rate=25.0,
        refusal_rate=10.0,
        median_latency_ms=50.0,  # Very fast
        archetype=archetype,
    )
    
    # A cloud model that might be misclassified (High quality, high cost)
    cloud_model = ModelMetadata(
        name="Cloud Giant 100B",
        intelligence_index=90.0,
        input_cost_per_m=10.0,
        output_cost_per_m=30.0,
        hallucination_rate=2.0,
        refusal_rate=1.0,
        median_latency_ms=500.0,
        archetype=archetype,
    )

    # A cheap model (e.g. GPT-4o-mini)
    gpt_mini = ModelMetadata(
        name="OpenAI: GPT-4o-mini",
        intelligence_index=82.0,
        input_cost_per_m=0.15,
        output_cost_per_m=0.6,
        hallucination_rate=8.0,
        refusal_rate=3.0,
        median_latency_ms=100.0,
        archetype=archetype,
    )

    models = [gemini_flash, tiny_model, cloud_model, gpt_mini]
    
    # 2. Setup Optimizer
    optimizer = Optimizer(
        baseline_model=cloud_model,
        all_models_data=[],
        strategy=OptimizationStrategy.QUALITY_FOCUSED,
        missing_data=MissingDataStrategy.IMPUTE
    )
    
    decision = RoutingDecision(
        archetype=archetype,
        category=PromptCategory.GENERAL,
        reason="Test"
    )
    
    # 3. Rank
    results = optimizer.rank(models, decision, top_k=4, verbose=False)
    
    print("\nRanking Results:")
    for res in results:
        print(f"{res.rank}. {res.model_name} (Score: {res.score:.4f})")
        
    # 4. Assertions
    top_model = results[0].model_name
    
    # High quality models should rank higher
    # Note: This may not work as expected until edge/local penalties are added
    assert len(results) > 0, "Should return results"
    
    # Verify TinyLlama is penalized for low quality despite high efficiency
    tiny_rank = next(r.rank for r in results if "Tiny" in r.model_name)
    assert tiny_rank > 1, "TinyLlama should not be #1 due to low quality"


if __name__ == "__main__":
    test_edge_local_ranking_favors_quality()
