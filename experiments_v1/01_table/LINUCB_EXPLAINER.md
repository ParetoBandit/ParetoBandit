# LinUCB Priors Explainer

## What are "LinUCB Priors (33 dims with bias)"?

### Simple Answer
LinUCB priors are the **initialization parameters** for the contextual bandit algorithm. Think of them as the bandit's "prior knowledge" before it sees any live traffic.

### Technical Answer
LinUCB (Linear Upper Confidence Bound) maintains two matrices **per model**:

#### 1. **A** - Covariance Matrix (33×33 = 1,089 parameters)
```
A ∈ ℝ³³ˣ³³
```
- **What it captures**: Feature correlations and uncertainty
- **Purpose**: Tracks which prompt features (from PCA) predict good/bad outcomes
- **Updates**: Each time a model is selected, A is updated with the outer product of the context vector
- **Initial value**: Identity matrix I₃₃ (from warmup data)

**Intuition**: A is like a "memory" of what contexts the model has been used for. High values on the diagonal mean high confidence; off-diagonal values capture feature interactions.

#### 2. **b** - Belief Vector (33×1 = 33 parameters)
```
b ∈ ℝ³³
```
- **What it captures**: Reward expectations for different contexts
- **Purpose**: Encodes which prompt features lead to high rewards for this model
- **Updates**: Each time a model is selected, b is updated with reward × context
- **Initial value**: Zero vector (from warmup data)

**Intuition**: b is like a "scorecard" showing what types of prompts this model is good at. For example, if a coding model performs well on coding prompts, b will have high values in the "coding feature" dimensions.

### The 33 Dimensions

**32 PCA components** + **1 bias term**

```
context = [PCA₁, PCA₂, ..., PCA₃₂, 1.0]
          └────────────────────┘  └─┘
               semantic           bias
              representation
```

- **PCA components (32)**: Compressed semantic representation of the prompt
- **Bias term (1)**: Captures base quality (independent of prompt content)

### From Priors to Predictions

At inference time, LinUCB uses A and b to predict expected reward:

```python
# For each model:
theta = inv(A) @ b          # Estimate reward weights (33×1)
mu = theta.T @ context      # Expected reward (scalar)
sigma = sqrt(context.T @ inv(A) @ context)  # Uncertainty
ucb = mu + alpha * sigma    # Upper confidence bound

# Select model with highest UCB
```

**Key insight**: Models with:
- High **mu** (mean): Expected to perform well on this prompt
- High **sigma** (uncertainty): Haven't been tested much on this type of prompt (exploration bonus)

### Why Warmup with 80k Prompts?

**Cold start problem**: Without warmup, all models start with A = I and b = 0, meaning:
- No prior knowledge → random exploration
- Wastes budget on bad models
- Slow convergence

**With warmup**: Use 80k historical battles to pre-populate A and b:
```python
for prompt, reward in warmup_data:
    context = embed(prompt)  # 384 dims
    context = pca.transform(context)  # 32 dims
    context = np.append(context, 1.0)  # 33 dims
    
    A += context @ context.T  # Update covariance
    b += reward * context      # Update beliefs
```

**Result**: 
- Bandit starts with informed priors
- Good models (e.g., GPT-4) have high b values
- Weak models (e.g., Mixtral) have lower b values
- Faster convergence, better initial performance

### Storage Size

Per model:
- **A**: 33×33 = 1,089 floats × 8 bytes = **8.7 KB**
- **b**: 33 floats × 8 bytes = **264 bytes**
- **Total**: ~9 KB per model

For 2 models (mixtral + gpt-4-turbo):
- **Total size**: ~18 KB (tiny!)

### In the Paper

You can describe this as:

> "We initialize BanditGPT with warmup priors derived from 80,000 LMSYS Arena 
> battles. For each model, we maintain a covariance matrix **A** ∈ ℝ³³ˣ³³ 
> (capturing feature correlations) and belief vector **b** ∈ ℝ³³ (encoding 
> reward expectations), where context dimension is 33 (32 PCA components + 1 bias)."

### Mathematical Formulation

**LinUCB Update** (after selecting model m and observing reward r):
```
A_m ← A_m + x × x^T
b_m ← b_m + r × x
```

**LinUCB Prediction** (before selecting model m):
```
θ_m = A_m^{-1} b_m
μ_m(x) = θ_m^T x
σ_m(x) = √(x^T A_m^{-1} x)
UCB_m(x) = μ_m(x) + α × σ_m(x)
```

where:
- x ∈ ℝ³³ is the context vector (PCA embedding + bias)
- r ∈ [0,1] is the observed reward
- α > 0 is the exploration parameter

### Example Values (After Warmup)

**GPT-4-turbo** (strong model):
```
A_gpt4 = [[10.2, 0.3, ...],    # High diagonal (high confidence)
          [0.3, 9.8, ...],
          ...]

b_gpt4 = [0.85, 0.92, ..., 0.88]  # High values (good performance)
```

**Mixtral** (weak model):
```
A_mixtral = [[9.8, 0.2, ...],   # Similar diagonal
             [0.2, 10.1, ...],
             ...]

b_mixtral = [0.72, 0.68, ..., 0.65]  # Lower values (weaker performance)
```

### Why 33 Dimensions is a Sweet Spot

- **Too few** (e.g., 10): Loses semantic information, poor predictions
- **Too many** (e.g., 384): Slow matrix inversions, overfitting
- **33**: Good balance of expressiveness and efficiency
  - Matrix inversion: O(33³) ≈ 35,937 ops (fast!)
  - Captures ~90% of semantic variance

---

## Summary

**LinUCB priors (33 dims with bias)** means:
- Each model has **A** (33×33 covariance) and **b** (33×1 beliefs)
- Initialized from 80k warmup prompts
- Enables informed routing from the start
- Dimension: 32 PCA + 1 bias = 33

**Key benefit**: Bandit doesn't start "blind" - it has learned priors about which models are good for which types of prompts.

