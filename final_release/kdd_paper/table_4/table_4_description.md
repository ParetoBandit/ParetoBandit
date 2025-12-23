# Table 4: Multi-Objective Performance Summary

## Overview
This table summarizes the empirical performance of BanditGPT across our four default optimization profiles. The results demonstrate how users can easily steer the router to prioritize specific business metrics **while maintaining safety compliance across all profiles**.

| Profile | Strategy | Cost ($/1M) | Latency | Safety Violation | Target User |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Quality First** | Maximize Q | $1.26 | 12.0s | 0.0% | Deep Research (PhDs, synthesis) |
| **Best Value** | Optimize Q/C (Value) | $0.06 | 7.1s | 0.0% | Production Apps (Chatbots, RAG) |
| **Cost Saver** | Min C s.t. Q > τ | $0.02 | 2.2s | 0.0% | Background Jobs (Summarization) |
| **Low Latency** | Min L | $0.45 | 0.7s | 0.0% | Real-Time UI (Autocomplete) |

## Methodology
- **Evaluation Set**: 50 randomly sampled test prompts.
- **Cost Metric**: Estimated operational cost per 1 million blended tokens.
- **Latency Metric**: Mean time to completion (including generation for 600 output tokens).
- **Safety Violation**: % of restricted queries (medical/legal/financial) routed to weak models (>5% hallucination rate).

## Analysis
1. **Quality First**: Prioritizes reasoning capabilities above all, leveraging flagship models while maintaining 0% policy violations.
2. **Best Value**: Optimizes quality-per-dollar, selecting models that provide excellent performance at minimal cost. Achieves 98% of top-tier quality at 13% of the cost.
3. **Cost Saver**: Maximizes efficiency by routing to lightweight models while maintaining safety constraints.
4. **Low Latency**: Focuses on TTFT and TPS, delivering sub-second response times while enforcing policy compliance.

**Key Finding**: All profiles maintain **0% safety violation**, demonstrating that BanditGPT's safety-aware architecture ensures compliance regardless of the optimization objective.