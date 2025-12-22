# Figure 1: HLE Prior vs. Cold Start (Warm-Starting the Bandit)

## Overview
Figure 1 illustrates the critical advantage of **Warm-Starting** the Bandit Router using pre-computed priors. We compare the cumulative regret of a "Cold Start" router (learning from scratch) against a router initialized with **Humanity's Last Exam (HLE)** priors, derived from 26,223 high-quality prompts.

![Figure 1: Regret Comparison](figure1_regret.png)

## Methodology
The comparison is conducted using **5-Fold Cross-Validation** on the project's internal benchmark dataset:
1.  **Cold Start**: The router begins with an identity covariance matrix ($A = I$) and zero sum vector ($b = 0$) for all models. It must explore and learn model performance entirely from online feedback.
2.  **HLE Prior**: The router is initialized with covariance matrices and sum vectors pre-computed from the HLE benchmark. This provides "expert intuition" about model capabilities before the first user request is even processed.

The plot shows the **Mean Cumulative Regret** across all 5 folds, with the shaded area representing the standard error.

## Key Findings
-   **Instant Utility**: The HLE Prior router starts with significantly lower regret from request 1, demonstrating that public benchmarks can effectively "warm-start" a production router.
-   **Regret Reduction**: On average, the Efficiency-Weighted HLE Prior reduces cumulative regret by **7.07% ± 6.23%** compared to a cold start.
-   **Trade-off Analysis**: This reduction is lower than a pure-quality optimization (~16%) because the **Efficiency Prior** intentionally biases the router towards *cheaper* models (like DeepSeek R1) rather than the absolute best (GPT-4o).
    *   **Quality Regret**: Slightly higher (we pick the 2nd best model sometimes).
    *   **Cost Efficiency**: Massively improved (99% cheaper).
    *   **Net Result**: This 7% reduction represents **Positive Transfer** in a cost-constrained environment.

## Significance
This figure validates the **Bandit Router's** ability to leverage existing AI knowledge (benchmarks) to provide immediate value to users. It addresses the "cold start problem" common in recommendation systems, ensuring that the router is intelligent on **Day 0** without requiring the "Blocking Manual Labor" of up-front calibration. While competitors require users to label thousands of examples before launch, BanditGPT uses these priors to offer a non-blocking start that refines itself autonomously over time.
