# Table 2: SOTA Router Comparison

## Overview
Table 2 compares **BanditGPT** against four popular open-source LLM routers: **RouteLLM**, **FrugalGPT**, **Aurelio AI (Semantic Router)**, and **LiteLLM**. The comparison is based on a high-fidelity simulation using real model performance data from `models.json`.

| System | Accuracy | Cost ($/1k) |
| :--- | :--- | :--- |
| **BanditGPT (Ours)** | **86.6%** | **$0.000068** |
| LiteLLM (Balanced) | 86.7% | $0.000175 |
| Aurelio AI | 85.8% | $0.001924 |
| RouteLLM | 75.2% | $0.002957 |
| FrugalGPT | 71.3% | $0.001632 |

## Methodology
- **Real Data**: All routers are evaluated using ground-truth accuracy and pricing from `models.json` for 80+ models.
- **High-Fidelity Simulation**:
    - **RouteLLM**: Simulates a static classifier routing between a strong (GPT-4o) and weak (Nova-Lite) model.
    - **Aurelio AI**: Simulates manual intent mapping to specialists (DeepSeek R1 8B) with a 15% miss rate.
    - **FrugalGPT**: Implements a cascade pattern (Cheap -> Verifier -> Strong).
    - **LiteLLM**: Simulates cost-aware routing by consistently picking a balanced model (Gemini 2.0 Flash).
    - **BanditGPT (Ours)**: Uses the **actual `BanditRouter` code** with standard production defaults (`exploration="balanced"`, `forgetting_factor=0.9`).

## Significance of Results
1. **Cost Efficiency**: BanditGPT achieves **~60% cost savings** compared to the next best efficient router (LiteLLM) while maintaining parity in accuracy.
2. **Pareto Dominance**: BanditGPT strictly dominates RouteLLM and FrugalGPT in both accuracy and cost.
3. **Mechanism**: This is achieved by correctly identifying and exploiting the "Efficient Specialist" (`deepseek-r1-0528`) which offers GPT-4 class performance at a fraction of the cost.
4. **Zero Manual Overhead**: Unlike Aurelio AI, which requires manual intent mapping, BanditGPT achieves these results entirely through online exploration and pre-computed priors.
