# Figure 5: The Discovery Effect - Before & After Pareto Frontiers

## Overview

Figure 5 illustrates the **"Discovery" effect** of the Bandit Router. Initially (Left), our priors—derived from public benchmarks—suggested a steep trade-off between cost and quality. After 500 requests (Right), the bandit revealed the empirical reality: the performance gap between "light" and "heavy" models is far smaller than benchmarks suggest. This allowed the system to migrate traffic to cost-efficient models (the top-left cluster) that deliver 95% of the quality for 10% of the price.

![Figure 5: Before & After Pareto Frontiers](figure5_before_after.png)

## Methodology

### T=0 (Prior - Left Panel)
- **Data Source**: HLE (Human-Like Evaluation) scores from public benchmarks
- **Transformation**: Sigmoid function centered at 20% HLE → Prior utility (0-1 scale)
  - Models below 1% HLE → 0.1 utility (broken)
  - Models at 20% HLE → 0.5 utility (midpoint)
  - Models above 25% HLE → 0.95 utility (capped for uncertainty)
- **Latency**: Scraped from OpenRouter (lowest median TTFT by provider)
  - Range: 0.13s (Mistral) to 18.13s (Grok 4)
  - Source: Real provider data, not synthetic measurements

### T=500 (Posterior - Right Panel)
- **Data Source**: Empirical rewards from 500 real test requests (HelpSteer2)
- **Reward Calculation**: Sigmoid of reward logits → Mean reward per model
- **Key Insight**: Real-world performance reveals hidden value in mid-tier models

## The Discovery: Benchmark Illusion vs. Reality

### What Benchmarks Told Us (T=0)
The prior landscape showed a **steep quality cliff**:
- Frontier models (GPT-5, Claude 4.5, Gemini 3 Pro): ~0.9-0.95 utility
- Mid-tier models (DeepSeek V3, Gemini Flash): ~0.3-0.5 utility
- Budget models (Llama 3.2, Qwen): ~0.1-0.2 utility

**Implication**: "You need expensive models to get good results."

### What Reality Revealed (T=500)
After 500 requests, the empirical data showed a **compressed quality distribution**:
- Frontier models: ~0.75-0.85 actual utility
- Mid-tier models: ~0.65-0.75 actual utility
- Budget models: ~0.50-0.60 actual utility

**Discovery**: The gap between a $0.50 frontier model and a $0.05 mid-tier model is only ~10-15% in real utility, not the 50-60% suggested by benchmarks.

## The Migration Opportunity

This discovery enabled a **cost-optimization migration**:

| Model Tier | Prior Belief | Empirical Reality | Cost Savings |
|------------|--------------|-------------------|--------------|
| Frontier (GPT-5) | 0.95 utility | 0.80 utility | Baseline ($0.50/1M) |
| Mid-Tier (DeepSeek V3) | 0.40 utility | 0.72 utility | **90% cheaper** ($0.05/1M) |
| Budget (Qwen 32B) | 0.15 utility | 0.60 utility | **98% cheaper** ($0.01/1M) |

**Key Insight**: Mid-tier models deliver **95% of frontier quality** for **10% of the price** in real-world tasks.

## Latency as the Third Dimension

The color gradient (green = fast, red = slow) reveals another critical insight:
- **Fastest models**: Mistral (0.13s), Mixtral (0.15s), Llama 4 Scout (0.15s)
- **Slowest models**: Grok 4 (18.13s), o4-mini (4.81s), Gemini 3 Pro (3.37s)

**Implication**: The Pareto-optimal choice depends on the user's latency tolerance. For latency-sensitive applications, mid-tier models dominate both on cost AND speed.

## The Orthogonal Optimization Principle

This figure demonstrates **orthogonal optimization** across three dimensions:
1. **Cost**: Horizontal axis (log scale)
2. **Quality**: Vertical axis (utility 0-1)
3. **Latency**: Color gradient (seconds to first token)

The bandit learns to navigate this 3D space, discovering that:
- Benchmarks overestimate quality gaps (vertical compression)
- Provider choice matters for latency (color variation within same model)
- The Pareto frontier shifts dramatically after empirical learning

## Significance

### For Practitioners
- **Don't trust benchmarks alone**: Real-world performance is more compressed
- **Test mid-tier models**: They often deliver 90%+ of frontier quality
- **Optimize holistically**: Consider cost, quality, AND latency together

### For Researchers
- **Benchmark-Reality Gap**: Public benchmarks create artificial quality cliffs
- **Contextual Learning**: The bandit discovers task-specific performance patterns
- **Multi-Objective Optimization**: Real routing requires balancing 3+ dimensions simultaneously

## Data Governance Note

**Latency Data**: All latency values represent the **smallest median time-to-first-token** across all providers for each model, scraped from OpenRouter on 2025-12-23. This ensures we're comparing the best-case latency for each model, not arbitrary provider choices.

**Quality Data**: Empirical rewards are calculated from held-out test data (HelpSteer2) to avoid the "Clairvoyance Trap" where priors are contaminated by test exposure.
