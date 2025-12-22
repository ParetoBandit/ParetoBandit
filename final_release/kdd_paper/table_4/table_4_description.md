# Table 4: Multi-Objective Performance Summary

## Overview
This table summarizes the empirical performance of BanditGPT across our four default optimization profiles. The results demonstrate how users can easily steer the router to prioritize specific business metrics without changing code.

| Profile | Strategy | Cost ($/1M) | Latency | Target User |
| :--- | :--- | :--- | :--- | :--- |
| **Quality First** | Maximize Q | $6.09 | 7.1s | Deep Research (PhDs, synthesis) |
| **Balanced** | Optimize U = Q - λC | $0.55 | 1.3s | Production Apps (Chatbots, RAG) |
| **Cost Saver** | Min C s.t. Q > τ | $0.03 | 0.8s | Background Jobs (Summarization) |
| **Low Latency** | Min L | $0.03 | 0.8s | Real-Time UI (Autocomplete) |

## Methodology
- **Evaluation Set**: 100 randomly sampled test prompts.
- **Cost Metric**: Estimated operational cost per 1 million blended tokens.
- **Latency Metric**: Mean time to completion (including generation for 600 output tokens).

## Analysis
1. **Quality First**: Prioritizes reasoning capabilities above all, achieving 0.37 HLE by leveraging flagship models (e.g., Gemini 1.5 Pro), though at a high premium ($6.09/1M).
2. **Balanced**: Targets the 'Value' segment ($0.55/1M), filtering out diminishing-return flagships to select capable 70B-class models. This offers substantial cost savings (-90% vs Quality First) while outperforming budget tiers.
3. **Cost Saver**: Maximizes efficiency ($0.03/1M) by routing to lightweight 7B-8B models, reducing costs by 18x compared to Balanced while maintaining baseline functionality.
4. **Low Latency**: Focuses on TTFT and TPS, delivering sub-second response times (0.76s) suitable for real-time applications, effectively converging with the efficient Cost Saver models.