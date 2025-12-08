#!/usr/bin/env python3
"""
Quick Start Example for LLM Jury

This example demonstrates the basic usage of LLM Jury:
1. Getting model recommendations
2. Routing prompts
3. Filtering models
4. Cost optimization
"""

from llm_jury import (
    get_recommendations,
    PromptRouter,
    PromptCategory,
    OptimizationStrategy,
    ModelRegistry,
    ProductArchetype,
)


def example_1_basic_recommendations():
    """Example 1: Get basic model recommendations."""
    print("=" * 60)
    print("Example 1: Basic Recommendations")
    print("=" * 60)

    recommendations = get_recommendations(
        prompt="Explain quantum computing in simple terms",
        category=PromptCategory.SIMPLE_QA,
        max_recommendations=3,
    )

    print("\nTop 3 models for simple Q&A:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.model.name}")
        print(f"   Quality Score: {rec.quality_score:.2f}")
        print(f"   Estimated Cost: ${rec.estimated_cost:.4f}")
        print(f"   Rationale: {rec.rationale[:80]}...")


def example_2_prompt_routing():
    """Example 2: Route different types of prompts."""
    print("\n\n" + "=" * 60)
    print("Example 2: Prompt Routing")
    print("=" * 60)

    router = PromptRouter()

    prompts = [
        ("What is 2+2?", PromptCategory.SIMPLE_QA),
        ("Write a detailed business analysis report", PromptCategory.CONTENT_GENERATION),
        ("Debug this Python code: def func():", PromptCategory.CODE_GENERATION),
        ("Solve this calculus problem: ∫x²dx", PromptCategory.REASONING),
    ]

    for prompt, category in prompts:
        result = router.route(prompt, category)
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"Category: {category.value}")
        print(f"Recommended: {result.recommended_model.name}")
        print(f"Estimated Cost: ${result.estimated_cost:.4f}")


def example_3_cost_optimization():
    """Example 3: Cost-optimized model selection."""
    print("\n\n" + "=" * 60)
    print("Example 3: Cost Optimization")
    print("=" * 60)

    print("\nCost-Optimized Strategy:")
    cost_optimized = get_recommendations(
        prompt="Summarize this article",
        ranking_strategy=OptimizationStrategy.COST_FOCUSED,
        max_cost_per_m=5.0,  # Max $5 per million tokens
        max_recommendations=5,
    )

    for i, rec in enumerate(cost_optimized, 1):
        print(f"{i}. {rec.model.name}")
        print(f"   Input Cost: ${rec.model.input_cost_per_m:.2f}/M")
        print(f"   Output Cost: ${rec.model.output_cost_per_m:.2f}/M")
        print(f"   Quality: {rec.quality_score:.2f}")


def example_4_quality_focus():
    """Example 4: Quality-focused model selection."""
    print("\n\n" + "=" * 60)
    print("Example 4: Quality-Focused Selection")
    print("=" * 60)

    print("\nHighest Quality Models:")
    quality_focused = get_recommendations(
        prompt="Solve this complex physics problem",
        ranking_strategy=OptimizationStrategy.QUALITY_FOCUSED,
        min_quality_score=0.8,
        max_recommendations=3,
    )

    for i, rec in enumerate(quality_focused, 1):
        print(f"{i}. {rec.model.name}")
        print(f"   Quality Score: {rec.quality_score:.2f}")
        print(f"   Chebyshev Score: {rec.model.chebyshev_score:.3f}")
        print(f"   Cost: ${rec.estimated_cost:.4f}")


def example_5_model_filtering():
    """Example 5: Custom model filtering."""
    print("\n\n" + "=" * 60)
    print("Example 5: Model Filtering")
    print("=" * 60)

    registry = ModelRegistry()

    # Filter by archetype
    print("\nElite Models:")
    elite_models = registry.get_models_by_archetype(ProductArchetype.ELITE)
    for model in elite_models[:3]:
        print(f"  - {model.name}")

    # Filter by specs
    print("\nAffordable Models (< $10/M tokens):")
    affordable = registry.filter_models(
        max_input_cost=10.0,
        max_output_cost=10.0,
    )
    for model in affordable[:3]:
        print(f"  - {model.name}: ${model.input_cost_per_m:.2f}/M input")

    # Filter by context length
    print("\nLong Context Models (> 100K tokens):")
    long_context = registry.filter_models(min_context_length=100000)
    for model in long_context[:3]:
        print(f"  - {model.name}: {model.context_length:,} tokens")


def example_6_balanced_strategy():
    """Example 6: Balanced cost-quality strategy."""
    print("\n\n" + "=" * 60)
    print("Example 6: Balanced Strategy")
    print("=" * 60)

    print("\nBalanced Cost-Quality:")
    balanced = get_recommendations(
        prompt="Create a comprehensive marketing strategy",
        ranking_strategy=OptimizationStrategy.BALANCED,
        max_recommendations=5,
    )

    for i, rec in enumerate(balanced, 1):
        print(f"\n{i}. {rec.model.name}")
        print(f"   Quality: {rec.quality_score:.2f}")
        print(f"   Cost: ${rec.estimated_cost:.4f}")
        print(f"   Value Score: {rec.model.value_score:.3f}")
        print(f"   Archetype: {rec.model.archetype.value}")


def main():
    """Run all examples."""
    print("\n🎯 LLM Jury - Quick Start Examples\n")

    try:
        example_1_basic_recommendations()
        example_2_prompt_routing()
        example_3_cost_optimization()
        example_4_quality_focus()
        example_5_model_filtering()
        example_6_balanced_strategy()

        print("\n\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\n📚 Next steps:")
        print("  - Read the User Guide: USER_GUIDE.md")
        print("  - Try: llm-jury update")
        print("  - Explore: llm-jury --help")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run: llm-jury init")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

