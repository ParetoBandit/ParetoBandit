# Appendix A: Bandit Dynamics and Hyperparameter Analysis

## A.1 Theoretical Foundation: Discounted LinUCB

Our routing algorithm is formally an instance of **Discounted Linear UCB (D-LinUCB)**, a standard approach for non-stationary bandit problems. We cite the foundational work by Russac et al. [1] which establishes the theoretical regret bounds for this class of algorithms.

The decision rule at time $t$ is:
$$ a_t = \arg\max_{a \in \mathcal{A}} \left( x_{t,a}^\top \hat{\theta}_t + \alpha \sqrt{x_{t,a}^\top A_t^{-1} x_{t,a}} \right) $$

where the covariance matrix $A_t$ and bias vector $b_t$ are updated with a forgetting factor $\gamma \in (0, 1]$:
$$ A_t = \sum_{s=1}^{t-1} \gamma^{t-1-s} x_{s,a_s} x_{s,a_s}^\top + \lambda I $$

### A.2 Orthogonal Roles of $\alpha$ and $\gamma$

A key contribution of our analysis is the empirical decoupling of the roles of $\alpha$ (Exploration) and $\gamma$ (Forgetting). While both parameters influence the router's adaptability, they address different sources of uncertainty:

1.  **Alpha ($\alpha$) handles Stochasticity**: It governs the confidence interval width, protecting against noisy rewards within a stationary period.
2.  **Gamma ($\gamma$) handles Non-Stationarity**: It governs the effective memory length ($H \approx \frac{1}{1-\gamma}$), allowing the model to discard obsolete priors when the environment shifts (e.g., user intent change).

## A.3 Empirical Analysis: The "Abandonment Rate"

We conducted a simulation to determine how these parameters affect the **Switching Time**—the number of requests required to abandon a high-prior "Favorite" model (Prior=0.8) that begins underperforming (Reward=0.4) in favor of an "Underdog" (Prior=0.5).

### Table A.1: Impact of Hyperparameters on Switching Time

| Parameter Varied | Value | Steps to Switch | Speedup Factor | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | $\alpha=0.1, \gamma=1.0$ | **54** | 1.0x | High inertia due to strong prior. |
| **Exploration** | $\alpha=2.0, \gamma=1.0$ | **18** | 3.0x | High alpha encourages testing the underdog. |
| **Forgetting** | $\alpha=0.1, \gamma=0.7$ | **11** | **4.9x** | Forgetting discounts the old prior directly. |

### A.4 Production Parameter Selection

To determine production-quality defaults, we conducted a **48-configuration grid search** with sigmoid-transformed HLE priors:

**Search Space:**
- α (Exploration): {0.1, 0.5, 1.0, 2.0}
- Prior Strength: {20, 40, 80, 160}
- γ (Forgetting): {0.95, 0.98, 1.0}

**Evaluation Protocol:**
- Dataset: 200 HelpSteer2 prompts per configuration
- Metric: Utility = Accuracy - 0.5 × log(Cost)
- Execution: 10 parallel workers

**Optimal Configuration:**
| Parameter | Value | Rationale |
|:----------|:------|:----------|
| α | **0.1** | Low exploration sufficient with high-quality sigmoid priors |
| Prior Strength | **40.0** | Balanced between prior belief and online learning |
| γ | **0.95** | Moderate forgetting enables adaptation without instability |

**Performance:**
- **Accuracy: 84.2%**
- **Cost: $0.001543/1k**
- **Utility: 4.08** (highest among all 48 configurations)

**Key Finding:** With sigmoid-transformed priors (mapping raw HLE scores 0-40% to realistic utility 0-95%), **low exploration (α=0.1) dramatically outperforms** the previously recommended α=1.0, achieving +3% accuracy and -16% cost.

### A.5 Conclusion

Our analysis demonstrates that **Forgetting (γ) is the dominant factor for adaptation** in the presence of strong priors.
*   Increasing α yields diminishing returns because it only increases the uncertainty of the underdog.
*   Decreasing γ directly erodes the "Bayesian Inertia" of the favorite, allowing new evidence to dominate 5x faster.

The **sigmoid prior transformation** is critical: it acknowledges that even "low" HLE scores (6.5%) indicate highly capable models, preventing the bandit from wasting exploration on discovering obvious competence.

**Production Defaults:** α=0.1, prior_strength=40.0, γ=0.95

## References

[1] Russac, Y., Vernade, C., & Cappé, O. (2019). Weighted Linear Bandits for Non-Stationary Environments. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
