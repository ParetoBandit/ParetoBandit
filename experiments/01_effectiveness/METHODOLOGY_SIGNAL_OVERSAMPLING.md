# Signal-Aware Oversampling: A Budget-Friendly Calibration Strategy

## 1. The Challenge: Information Scarcity in Real Data
Real-world prompt distributions (like LMSYS) are often dominated by "easy" queries where most models perform adequately. In our dataset, we found that ~50% of training prompts have a reward variance of $\approx 0$ (Consensus). Burning in a bandit on this raw distribution teaches it to be "lazy," as discriminative decisions are rarely reinforced.

## 2. The Solution: Signal-Aware Oversampling
To maximize the information density of the burn-in phase without incurring the cost of new data collection, we implemented a **Signal-Aware Oversampling** strategy.

We define the **"Learnable Signal"** of a prompt $x$ as the variance of its reward vector $\sigma^2(r_x)$. 
- **Consensus Prompts** ($\sigma^2 < 0.05$): Represent generic success or failure. These provide negligible gradients for discriminative learning (learning *which* model is better).
- **Contentious Prompts** ($\sigma^2 \ge 0.05$): Represent decision boundaries where the policy must actively distinguish between model capabilities (e.g., complex coding tasks where DeepSeek excels but Llama fails).

**The Protocol**:
We oversample these high-signal prompts by a factor of **$3\times$** during the burn-in phase. This artificially constructs a "Hard Curriculum" that forces the covariance matrix ($A$) to resolve feature correlations along critical decision boundaries.

## 3. Algorithm Implementation (`run_budget_experiment.py`)

1.  **Strict Data Splitting**:
    *   **Train (Burn-in)**: 40% (800 prompts)
    *   **Validation (Tuning)**: 20% (400 prompts)
    *   **Test (Hold-out)**: 40% (800 prompts)

2.  **Curriculum Construction**:
    ```python
    hard_train = [p for p in train_pool if variance(rewards[p]) > 0.05]
    easy_train = [p for p in train_pool if variance(rewards[p]) <= 0.05]
    
    # Signal-Aware Oversampling
    curriculum = []
    curriculum.extend(hard_train * 3)  # 3x boost to signal
    curriculum.extend(random.sample(easy_train, len(hard_train) * 3)) # Balance
    ```

3.  **Hyperparameter Tuning**:
    *   We use the Validation set (offline evaluation) to grid-search the optimal Prior Stiffness ($N_{eff}$) using this curriculum.

4.  **Final Evaluation**:
    *   The winning configuration is evaluated once on the Hold-out Test set.

This approach effectively turns a small, imbalanced synthetic/real dataset into a high-density training signal for the bandit.
