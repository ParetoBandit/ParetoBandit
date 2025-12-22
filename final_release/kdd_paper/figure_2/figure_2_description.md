# Figure 2: Real-Data Adaptation Dynamics

## Overview
Figure 2 demonstrates the **"Dip and Recover"** adaptation pattern of the Bandit Router in a non-stationary environment. This simulation uses real prompt embeddings and rewards from the project's benchmark data to showcase how the router adapts to a dramatic distribution shift in a single user session.

![Figure 2: Adaptation Curve](figure2_adaptation.png)

## Simulation Scenario
The simulation is divided into two distinct phases:
1.  **Phase 1: General Coding (Requests 1-50)**: The router handles general coding prompts. It is initialized with **HLE (Humanity's Last Exam) Priors**, which correctly identify **Gemini 3 Pro Preview** as the top performer for this domain.
2.  **Phase 2: Specialized Task (Requests 51-500)**: A distribution shift occurs. The prompts shift to a specialized domain where Gemini's performance drops significantly, while **Claude 3.7 Sonnet** becomes the optimal choice.

## Parameters for "Session-Level" Adaptation
To achieve the recovery shown in the figure, the standard production parameters were utilized:

| Parameter | Value | Significance |
| :--- | :--- | :--- |
| `prior_strength` | `40.0` | Stronger initial trust (High Confidence Anchor). |
| `forgetting_factor` ($\gamma$) | `0.96` | Tuned for $\alpha=1.0$ (Effective $N \approx 25$). |
| `exploration` | `balanced` | Optimal exploration ($\alpha = 1.0$). |

## Significance
The "Dip and Recover" pattern is a hallmark of robust online learning:
-   **The Dip**: Represents the cost of the distribution shift. The router initially relies on its strong priors (Gemini) before realizing they are no longer optimal.
-   **The Recover**: Demonstrates the router's ability to self-correct without manual intervention. By request 75, the router has successfully pivoted to Claude, restoring the rolling reward to near-optimal levels.

### Mechanism: Time-Based Variance Inflation
This recovery is guaranteed by our implementation of **Global Forgetting**. In standard LinUCB, high confidence in a suboptimal arm (strong prior) can permanently suppress exploration of better alternatives. To prevent this, BanditGPT inflates the variance of *all* arms over time, not just the selected one:
$$ \sigma_{effective}^2 = \sigma_{stored}^2 \cdot \gamma^{-t} $$
As time passes ($t$) without selecting a model, its uncertainty ($sigma$) grows exponentially. This ensures that even if the HLE Prior strongly favors Gemini, the uncertainty around the neglected Claude arm will eventually grow large enough to trigger an exploratory selection, leading to the discovery of its superior performance.

This figure proves that the Bandit Router is not just a static selection engine but a dynamic system capable of **personalization and task-specific optimization** in real-time.
