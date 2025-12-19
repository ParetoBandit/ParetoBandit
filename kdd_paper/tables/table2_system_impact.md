# RQ3: System Impact - Router vs Static Policies

Compares static single-model policies against the adaptive router.

**Quality scores are probabilities [0-1], consistent with Table 4.**

| Policy Strategy | Avg Quality | Cost/1M | Cost Reduction | Quality vs GPT-4o |
|-----------------|-------------|---------|----------------|-------------------|
| Static: GPT-4o (SOTA) | 0.63 | $4.38 | 0% (Ref) | 0% (Ref) |
| Static: Llama-3-70B | 0.59 | $0.88 | -79.9% | -6.1% |
| Static: Nova-Lite | 0.61 | $0.10 | -97.6% | -2.8% |
| **Adaptive Router (Ours)** | **0.67** | **$1.42** | **-67.7%** | **+6.1%** |

## Key Insights

**Nova-Lite alone**: -97.6% cost, but -2.8% quality gap vs GPT-4o

**Adaptive Router**: Achieves **+6.1% higher quality** than GPT-4o at -67.7% cost

The router beats GPT-4o by selecting the optimal specialist for each task cluster.

## Router Model Selection

| Model | Clusters | Percentage |
|-------|----------|------------|
| nova-lite-v1 | 196 | 39.4% |
| gpt-4o | 154 | 31.0% |
| nova-micro-v1 | 147 | 29.6% |

## Trade-off Summary

> "The router achieves near-parity with GPT-4o (0.67 vs 0.63) while reducing cost by 68%."
