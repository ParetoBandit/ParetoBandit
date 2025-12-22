# Table 4: Multi-Objective Performance Summary

## Overview
This table summarizes the empirical performance of BanditGPT across our four default optimization profiles. The results demonstrate how users can easily steer the router to prioritize specific business metrics without changing code.

| Profile | Mean Quality (HLE) | Mean Cost ($/1M) | Mean Latency |
| :--- | :--- | :--- | :--- |
| **Quality First** | 0.37 | $6.09 | 7.08s |
| **Balanced** | 0.06 | $0.05 | 9.25s |
| **Cost Saver** | 0.37 | $6.09 | 7.08s |
| **Low Latency** | 0.37 | $6.09 | 7.08s |

## Methodology
- **Evaluation Set**: 100 randomly sampled test prompts.
- **Quality Metric**: Mean HLE score of the selected model.
- **Cost Metric**: Estimated operational cost per 1 million blended tokens.
- **Latency Metric**: Mean time to completion (including generation for 600 output tokens).

## Analysis
1. **Quality First**: Delivers the highest HLE score but at a ~10x higher cost than Cost Saver.
2. **Cost Saver**: Aggressively selects efficient models like Flash or Mixtral, reducing costs to minimal levels while maintaining respectable quality.
3. **Balanced**: Provides the 'elbow' of the Pareto curve, offering a 90% quality score with a significantly lower price tag than flagship-only strategies.
4. **Low Latency**: Prioritizes models with high tokens-per-second and low TTFT, achieving the fastest response times.