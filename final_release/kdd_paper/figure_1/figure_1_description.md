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
-   **Regret Reduction**: On average, the HLE Prior reduces cumulative regret by **~30-40%** compared to a cold start over the first few hundred requests.
-   **Robustness**: The narrow error bands across 5 folds confirm that the performance gain is consistent across different subsets of data.

## Significance
This figure validates the **Bandit Router's** ability to leverage existing AI knowledge (benchmarks) to provide immediate value to users. It addresses the "cold start problem" common in recommendation systems, ensuring that the router is intelligent on day one while remaining flexible enough to adapt to real-world feedback (as shown in Figure 2).
