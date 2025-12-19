# RQ3: System Impact - Router vs Static Policies

Compares static single-model policies against the adaptive router.

| Policy Strategy | Avg Quality | Cost/1M | Cost Reduction | ROI |
|-----------------|-------------|---------|----------------|-----|
| Static: GPT-4o (SOTA) | 1.235 | $4.38 | 0% (Ref) | 1.0× |
| Static: Llama-3-70B | 0.782 | $0.88 | -79.9% | 5.0× |
| Static: Nova-Lite | 0.997 | $0.10 | -97.6% | 41.7× |
| **Adaptive Router (Ours)** | **1.652** | **$1.42** | **-67.7%** | **3.1×** |

## Key Insights

**Nova-Lite alone**: -97.6% cost, but 19.3% quality gap

**Adaptive Router**: Achieves **+33.8% higher quality** than GPT-4o at -67.7% cost

The router beats GPT-4o by selecting specialists for each task type.

## Router Model Selection

| Model | Clusters | Percentage |
|-------|----------|------------|
| nova-lite-v1 | 196 | 39.4% |
| gpt-4o | 154 | 31.0% |
| nova-micro-v1 | 147 | 29.6% |