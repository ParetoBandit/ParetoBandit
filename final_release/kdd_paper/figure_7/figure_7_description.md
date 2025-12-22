# Figure 7: Prior Strength vs. Feedback Stability

## Overview
This figure demonstrates the **critical importance of Prior Strength** in preventing the "Learning Tax"—a phenomenon where more feedback actually increases regret due to overfitting to noise.

![Figure 7: Prior Strength Comparison](figure7_prior_strength.png)

## Methodology
-   **X-Axis**: Feedback Rate (1%, 10%, 50%, 100%)
-   **Y-Axis**: Cumulative Regret at 100 Requests (Lower is Better)
-   **Visual**: Line Chart
-   **Comparison**:
    -   **N=20 (Red, Weak Prior)**: Demonstrates the "Learning Tax" phenomenon.
    -   **N=40 (Blue, Strong Prior)**: Demonstrates the "Momentum Principle" (The Golden Ratio).
    -   **Cold Start (Black Dashed)**: Baseline performance without priors.

## Detailed Methodology & Regret Calculation
To ensure empirical rigor, the simulation follows a strict evaluation protocol:
1.  **Environment**: We utilize the HelpSteer2 dataset, which contains ground-truth quality scores for multiple models on the same prompts. 
2.  **Oracle Definition**: The "Best Possible" reward ($R_{best}$) for any given prompt is defined as the maximum Ground Truth score available in the dataset for that specific prompt cluster.
3.  **Reward Function**: Observed rewards are calculated by applying a sigmoid function to the ground-truth reward logits: $R = \frac{1}{1 + e^{-logit}}$, resulting in a value $\in [0, 1]$.
4.  **Regret Formula**: We measure performance using **Cumulative Regret**, defined as the sum of instantaneous regrets over $T$ requests:
    $$Regret_{cumulative} = \sum_{t=1}^{T} \max(0, R_{best} - R_{chosen})$$
    where $R_{chosen}$ is the reward of the model selected by the BanditGPT policy at time $t$.
5.  **Simulation Parameters**: Each data point is averaged over 10 independent seeds with distinct prompt shuffles (100 requests each) to eliminate bias from prompt ordering.
6.  **Data Leakage Protection**: To ensure the integrity of the evaluation, the HLE priors were constructed using a disjoint pool of prompts (LMSYS/Chatbot Arena). Any prompt appearing in the HelpSteer2 evaluation set was explicitly filtered out from the prior generation process (using `calc_priors_large.py`). This guarantees that the bandit's initial knowledge is based on general linguistic patterns and benchmark performance, not on specific exposure to the test samples.

> **Caption: Plasticity vs. Stability.** Enabling continuous feedback incurs a minor "Learning Tax" (+3% regret) compared to a frozen prior (in the N=20 case), but remains 20% superior to Cold Start while ensuring resilience to drift.

## The Trade-off: Buying Infinite Adaptability
You are paying a 3% tax in immediate performance to buy **Infinite Adaptability**.
-   **Stick with 1% Feedback**: You save ~0.38 cumulative regret today, but if a model (e.g., GPT-4o) degrades tomorrow, your router will never know.
-   **Enable 100% Feedback**: You pay 0.38 today, but you remain resilient to future drift.

## The Momentum Principle: Solving the Stability-Plasticity Dilemma
We identified a critical instability regime when initializing with high-quality priors: standard update rates allow sparse, noisy feedback to prematurely disrupt an optimal policy, increasing regret by 3% (the "Learning Tax"). 

We solve this by enforcing **High-Inertia Initialization** ($\lambda_{prior} \ge 40$). This acts as a distinct **Low-Pass Filter** on the learning process:
-   **High-Frequency Noise (0–50 samples)**: Is dampened by the prior's inertia, preserving the 23% warm-start gain.
-   **Low-Frequency Drift (>100 samples)**: Accumulates sufficient mass to eventually shift the posterior, preserving long-term adaptability.

By doubling the prior strength ($N=20 \rightarrow N=40$), we have identified the **"Golden Ratio"** for prior-based bandits. This configuration completely eliminates the "Learning Tax" at 100% feedback, enabling the full safety of online learning with zero performance penalty.

As shown in Figure 7, increasing prior strength from $N=20$ to $N=40$ completely eliminates the regret penalty at 100% feedback, enabling a "best-of-both-worlds" architecture.

## Why this is the Perfect Product Config
1.  **Immediate Value**: Users see the ~23% cost/quality improvement on Query #1.
2.  **Zero-Dip Learning**: Users won't see performance degrade as they use the system (unlike the $N=20$ case), which is critical for trust.
3.  **Drift Resilience**: Because feedback is ON (100%), the system remains adaptable. If a model effectively "breaks" (e.g., a version update drops quality to zero), the bandit will eventually react. The higher strength simply acts as a low-pass filter, requiring consistent evidence to overrule the expert prior.
