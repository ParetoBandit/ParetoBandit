# Figure 3: Specialist Confidence vs. Cost (Pareto Frontier)

## Overview
Figure 3 visualizes the **Pareto Frontier** between the router's learned specialist confidence and the operational cost of the models. This analysis moves beyond legacy cluster-based methodologies to a more granular, weight-based evaluation of model specialization.

![Figure 3: Pareto Frontier](figure3_pareto.png)

## Methodology
-   **Learned Specialist Confidence**: Quantified by the **Euclidean norm of the learned LinUCB weights ($||\theta||$)** for each model. A higher $||\theta||$ indicates that the router has learned strong, specific associations between certain prompt features and that model's performance (i.e., the model is a "specialist" in certain domains).
-   **Cost**: Measured as the **Cost per 1M Blended Tokens ($)**, as defined in the model registry.
-   **Data Source**: The weights are derived from the **HLE (Humanity's Last Exam) Priors**, representing the router's "expert intuition" after processing 26,223 high-quality benchmark prompts.

## Key Observations
-   **The Frontier**: The red dashed line represents the Pareto optimal models—those that offer the highest confidence for a given cost bracket.
-   **High-End Specialists**: Models like **Gemini 3 Pro Preview** and **GPT-5.1 (high)** occupy the top-right, offering the highest learned confidence but at a premium price.
-   **Efficiency Leaders**: Mid-tier models like **Kimi K2 Thinking** and **gpt-oss-120B** provide a significant "confidence-per-dollar" advantage, forming the elbow of the curve.
-   **Commodity Models**: Low-cost models (left side) show lower $||\theta||$ norms, indicating they are treated more as generalists with less specific domain expertise learned so far.

## Significance
This figure demonstrates the **mathematical transparency** of the Bandit Router. By inspecting the norms of the learned weights, we can objectively identify which models the router "trusts" as specialists. This allows users to make informed decisions about their model registry, potentially pruning models that are dominated on the Pareto front to optimize for both performance and budget.
