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
| `prior_strength` | `20.0` | High initial trust in HLE priors. |
| `forgetting_factor` ($\gamma$) | `0.9` | Standard adaptation rate (Discounted LinUCB). |
| `exploration` | `balanced` | Conservative exploration ($\alpha = 0.1$). |

## Significance
The "Dip and Recover" pattern is a hallmark of robust online learning:
-   **The Dip**: Represents the cost of the distribution shift. The router initially relies on its strong priors (Gemini) before realizing they are no longer optimal.
-   **The Recover**: Demonstrates the router's ability to self-correct without manual intervention. By request 75, the router has successfully pivoted to Claude, restoring the rolling reward to near-optimal levels.

This figure proves that the Bandit Router is not just a static selection engine but a dynamic system capable of **personalization and task-specific optimization** in real-time.
