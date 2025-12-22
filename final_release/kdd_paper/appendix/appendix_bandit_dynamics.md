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

### A.4 Conclusion

Our analysis demonstrates that **Forgetting ($\gamma$) is the dominant factor for adaptation** in the presence of strong priors.
*   Increasing $\alpha$ yields diminishing returns because it only increases the uncertainty of the underdog.
*   Decreasing $\gamma$ directly erodes the "Bayesian Inertia" of the favorite, allowing new evidence to dominate 5x faster.

Therefore, we select a conservative exploration rate ($\alpha=0.1$) to exploit our high-quality priors, paired with a tuned forgetting factor ($\gamma=0.9$) to ensure robustness to distribution shifts.

## References

[1] Russac, Y., Vernade, C., & Cappé, O. (2019). Weighted Linear Bandits for Non-Stationary Environments. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
