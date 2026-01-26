# Expert Parameter Warm-Start Architecture

## Overview

The `CostAwareLinUCBRouter` implements **Expert Parameter Warm-Start** - a critical architectural pattern where bandit experts are initialized with pre-trained knowledge instead of cold-start identity matrices.

## Why Warm-Start in `__init__`?

### 1. **Hybrid Effectiveness**
For Corralling/Hybrid methods to work effectively, individual experts must be "pre-informed" so the Corralling Master has meaningful choices from day one.

```python
# BAD: Cold-start experts
warmup_expert = CostAwareLinUCBRouter(models, identity_priors, ...)  # A = I
tabula_rasa_expert = TabulaRasaRouter(models, ...)                    # A = I
# Master has no meaningful signal to choose between experts initially

# GOOD: Warm-start experts
warmup_expert = CostAwareLinUCBRouter(models, warmup_priors, ...)    # A = 80k battles
tabula_rasa_expert = TabulaRasaRouter(models, ridge_priors, ...)     # A = λI (regularized)
# Master can immediately distinguish expert strengths
```

### 2. **Bayesian Grounding**
By copying `warmup_priors['A']` and `warmup_priors['b']` directly in `__init__`, the expert starts with high-confidence beliefs from 80k RouteLLM battles instead of uninformed identity matrices.

```python
def __init__(self, models, warmup_priors, model_costs, ...):
    # EXPERT PARAMETER WARM-START (Core Architecture)
    # Initialize from warmup priors (80k RouteLLM battles)
    # - A matrices: Confidence/precision (inverse covariance)
    # - b vectors: Reward-weighted context sums
    self.A = {m: warmup_priors['A'][m].copy() for m in models}
    self.b = {m: warmup_priors['b'][m].copy() for m in models}
```

### 3. **Why Not Delay Until First Routing?**
Delaying warmup until first routing would defeat the purpose:
- Corralling Master needs informed experts **immediately**
- First routing decision sets trajectory for adaptive weight updates
- Expert differentiation must be present from timestep t=0

## The "First-Child" Bias Correction

### Problem: Confident Transfer Trap
When dynamically adding models via `register_model()`, naive transfer causes:

```python
# BAD: Transfer both A and b from neighbor
new_model.A = neighbor.A.copy()  # Inherits 1M samples of confidence!
new_model.b = neighbor.b.copy()
# Result: New model thinks it has 1M samples → tiny confidence intervals → no exploration
```

### Solution: Transfer θ (Preferences), Reset A (Confidence)

```python
# GOOD: Latent Semantic Transfer (admix_theta_from_neighbors)
neighbor_theta = neighbor.A_inv @ neighbor.b  # Extract preferences
new_model.A = λ * I                           # Fresh exploration potential
new_model.b = λ * neighbor_theta * n_eff      # Inherit domain knowledge
# Result: Same preferences, but high exploration potential
```

### Implementation in BanditRouter

The `BanditRouter.register_model()` method implements this automatically:

```python
def register_model(self, model_id: str, capabilities: List[str], speed: str, ...):
    # 1. Find semantic neighbor via embedding similarity
    neighbor, similarity = self._find_semantic_neighbor(model_id, dna_str)
    
    # 2. Bootstrap with θ-only transfer (fixed in KDD review)
    A_init, b_init = self.admix_theta_from_neighbors(
        model_id=model_id,
        registry=self.registry,
        bandit=self.bandit,
        encoder=self.encoder,
        n_effective=5.0  # Tunable prior strength
    )
    
    # 3. Add arm with transferred preferences but fresh confidence
    self.bandit.models.append(model_id)
    self.bandit.A[model_id] = A_init  # A = λI (maximum uncertainty)
    self.bandit.b[model_id] = b_init  # b = λ × θ_neighbor (preferences)
```

## Expert Configurations

### Warmup Expert (Conservative)
```python
warmup_expert = CostAwareLinUCBRouter(
    models=models,
    warmup_priors=scaled_priors,  # 80k battles, high confidence
    model_costs=model_costs,
    alpha_start=2.0,  # High initial exploration
    alpha_end=0.1,    # Low final exploitation
    cost_penalty=λ
)
```

**Characteristics:**
- Starts with high-confidence priors (large A values)
- Decays exploration over burn-in (2.0 → 0.1)
- Trusts prior knowledge but adapts to 94.2% Easy Cluster

### Tabula Rasa Expert (Aggressive Learner)
```python
tabula_rasa_expert = CostAwareTabulaRasaRouter(
    models=models,
    context_dim=24,
    model_costs=model_costs,
    alpha_start=2.0,
    alpha_end=0.1,
    cost_penalty=λ,
    ridge_lambda=ridge_lambda  # Regularization for stability
)
```

**Characteristics:**
- Starts from identity (A = λI) with Tikhonov regularization
- Same exploration schedule as warmup expert
- Can deviate quickly when encountering domain mismatch

## Dynamic Prior Management

### load_priors() Method
For flexible prior updates after initialization:

```python
# Reduce prior strength for faster adaptation
router.load_priors(new_priors, scale=0.5)

# Transfer priors from related domain
router.load_priors(coding_priors, scale=0.3)  # Weak transfer
```

**Mathematical Effect:**
Scaling both A and b by the same factor preserves θ = A^(-1)b:
- θ_new = (scale×A)^(-1) @ (scale×b) = θ_old
- But confidence changes: Smaller scale → wider confidence intervals → more exploration

## Empirical Validation

### Figure 4 Results (N=1,871 prompts)
**banditGPT Hybrid with Warm-Start:**
- Successfully exploited 94.2% Easy Cluster (Mixtral)
- Maintained quality on hard tasks (GPT-4-turbo)
- Achieved Pareto dominance across all budget tiers

**Key Insight:**
Without warm-start (cold-start identity matrices), the hybrid method requires 10x more samples to learn the same routing policy.

### Production Deployment (η=1.0, γ=0.01)
```python
# Aggressive Corralling learning (η=1.0)
router = CorrallingRouter(
    experts=[warmup_expert, tabula_rasa_expert],
    models=models,
    learning_rate=1.0  # Fast adaptation to expert performance
)
```

**Result:**
- Warmup expert dominated early (leveraged 80k battles)
- Tabula rasa expert adapted quickly to Easy Cluster
- Master dynamically balanced between experts

## References

1. **Warmup Priors Generation**: `scripts/generate_warmup_priors.py`
   - Trains on 80k RouteLLM battles (LMSYS Arena data)
   - Produces `priors_warmup.joblib` with A/b matrices

2. **Semantic Transfer**: `BanditRouter.admix_theta_from_neighbors()`
   - Implements θ-only transfer (fixes "confident transfer trap")
   - Used in `register_model()` for dynamic model admission

3. **Pareto Sweeps**: `experiments_v1/04_figure/generate_pareto_frontier.py`
   - Demonstrates warm-start effectiveness across cost penalties
   - Shows 4.3x cost reduction at production quality level

## Best Practices

### ✅ DO:
- Load warmup priors in `__init__` for immediate expert differentiation
- Use `load_priors(scale=0.5)` to reduce prior strength for faster adaptation
- Transfer θ only (not A) when adding semantically similar models

### ❌ DON'T:
- Delay warmup until first routing (defeats hybrid effectiveness)
- Transfer both A and b from neighbors (causes confident transfer trap)
- Use identity initialization for warmup expert (loses 80k battles of knowledge)

## Conclusion

Expert Parameter Warm-Start is a **core architectural pattern** in banditGPT:
1. Happens in `__init__` for immediate effect
2. Provides Bayesian grounding from offline data
3. Enables effective hybrid/corralling methods
4. Supports semantic transfer for dynamic model registration

The "First-Child" Bias Correction ensures that dynamically added models benefit from transfer learning while maintaining exploration potential.

