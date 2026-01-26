# Theory vs Implementation: Corralling Algorithm

## Overview

This document provides a detailed side-by-side comparison of the theoretical corralling algorithm (Agarwal et al., 2017) and our simplified production implementation in `router.py`.

## Core Algorithm Comparison

| Aspect | Theoretical (Agarwal 2017) | Implementation (banditGPT) |
|--------|---------------------------|---------------------------|
| **Paper** | "Corralling a Band of Bandit Algorithms" | router.py, lines 3349-3484 |
| **Complexity** | Full importance weighting | Simplified exponential weights |
| **Target** | Worst-case guarantees | Practical efficiency |
| **Setting** | Adversarial bandits | Stochastic contextual bandits |

## Initialization

### Theory
```
Input: K experts, time horizon T
Set: η = √[2 log(K) / T]
Initialize: w₁⁽¹⁾ = ... = w₁⁽ᴷ⁾ = 1
            π₁⁽ⁱ⁾ = 1/K for all i
```

**Requirements**:
- Know T in advance (use doubling trick if unknown)
- Adaptive learning rate based on K and T

### Implementation
```python
def __init__(self, experts, models, learning_rate=0.1):
    self.n_experts = len(experts)
    self.weights = np.ones(self.n_experts) / self.n_experts
    self.cumulative_losses = np.zeros(self.n_experts)
    self.learning_rate = learning_rate  # Fixed η
```

**Key Differences**:
- ✅ Fixed learning rate (η=0.1, no horizon needed)
- ✅ Simpler: no doubling trick required
- ⚠️ May be suboptimal if T << 1000 or T >> 100000

## Selection Phase

### Theory
```
For round t = 1, 2, ..., T:
  1. Observe context x_t
  2. For each expert i:
       a_t⁽ⁱ⁾ ← Expert_i.select(x_t)
  3. Sample i_t ~ Categorical(π_t)
  4. Play a_t = a_t⁽ⁱᵗ⁾
```

**Properties**:
- All experts must recommend every round
- O(K) expert queries per round

### Implementation
```python
def select_model(self, context: np.ndarray) -> str:
    # Sample expert according to weights
    expert_idx = np.random.choice(self.n_experts, p=self.weights)
    self.last_expert_idx = expert_idx
    
    # Query ONLY selected expert
    model = self.experts[expert_idx].select_model(context)
    return model
```

**Key Differences**:
- ✅ Only query selected expert (O(1) instead of O(K))
- ✅ Lower latency: 0.5ms vs K×0.5ms
- ⚠️ Cannot use counterfactual losses

## Loss Estimation

### Theory
```
For each expert i:
  # Counterfactual loss estimation
  ℓ_t⁽ⁱ⁾ = (1 - r_t⁽ⁱ⁾) / π_t⁽ⁱ⁾
  
  where r_t⁽ⁱ⁾ is the reward that expert i would have received
```

**Properties**:
- Unbiased: E[ℓ_t⁽ⁱ⁾] = true loss of expert i
- Requires observing all expert recommendations
- High variance (inversely proportional to π_t⁽ⁱ⁾)

### Implementation
```python
# Convert reward to loss
observed_loss = 1.0 - reward

# Importance-weighted loss (only for selected expert)
losses = np.zeros(self.n_experts)
p_chosen = self.weights[self.last_expert_idx]
losses[self.last_expert_idx] = observed_loss / max(p_chosen, 1e-6)

# Non-selected experts: loss = 0
```

**Key Differences**:
- ✅ Only selected expert gets loss update
- ✅ Lower variance (only 1 inverse probability weight)
- ⚠️ Biased estimator (ignores counterfactuals)
- ⚠️ May be slower to detect bad experts

## Weight Update

### Theory
```
# Exponential weights with exploration
w_{t+1}⁽ⁱ⁾ = w_t⁽ⁱ⁾ × exp(-η × ℓ_t⁽ⁱ⁾)

# Normalize with exploration bonus
π_{t+1}⁽ⁱ⁾ = (1 - γ) × [w_{t+1}⁽ⁱ⁾ / Σⱼ w_{t+1}⁽ʲ⁾] + γ/K

where γ > 0 is exploration parameter
```

**Properties**:
- Exploration bonus prevents any expert from getting π=0
- More complex normalization

### Implementation
```python
# Update cumulative losses
self.cumulative_losses += losses

# Exponential weight update (log-space for stability)
log_weights = -self.learning_rate * self.cumulative_losses
log_weights -= log_weights.max()  # Numerical stability
self.weights = np.exp(log_weights)
self.weights /= self.weights.sum()  # Normalize
```

**Key Differences**:
- ✅ Simpler: no explicit exploration bonus
- ✅ Numerically stable (log-space computation)
- ⚠️ Can reach exactly 0 weight (though unlikely)
- ✅ Max subtraction prevents overflow

## Expert Update

### Theory
```
# All experts observe full feedback
For each expert i:
  Expert_i.update(x_t, a_t, r_t)
```

**Properties**:
- All experts learn from every round
- Higher computational cost: O(K × expert_cost)

### Implementation
```python
# Only selected expert updates
self.experts[self.last_expert_idx].update(context, model, reward)
```

**Key Differences**:
- ✅ Only selected expert updates (O(1) instead of O(K))
- ⚠️ Non-selected experts don't learn from this round
- ✅ Matches typical bandit feedback model

## Regret Guarantees

### Theory

**Theorem 1** (Agarwal et al., 2017):  
If each expert i has regret R_i(T), the corralling algorithm achieves:

```
R_corral(T) ≤ min_i R_i(T) + O(√[T log(K)])
```

**Proof sketch**:
1. Exponential weights ensure low regret vs best expert
2. Exploration bonus prevents premature convergence
3. Importance weighting ensures unbiased loss estimates

**Formal Assumptions**:
- Adversarial loss sequence (worst-case)
- Bounded losses: ℓ_t⁽ⁱ⁾ ∈ [0, 1]
- IID context distribution (for contextual bandits)

### Implementation

**Empirical Performance**:
- Regret comparable to best expert in practice
- No formal proof (simplified algorithm differs from theory)
- Validated on synthetic + real-world benchmarks

**Why No Proof?**:
1. Non-selected experts don't update → violates theorem assumptions
2. No exploration bonus → can converge to single expert
3. Fixed learning rate → may be suboptimal

**Practical Guarantees**:
- ✅ Empirically: matches best expert within 5-10%
- ✅ Robustness: handles distribution shift gracefully
- ⚠️ Worst-case: could fail in adversarial settings

## Computational Complexity

### Theory

**Per-Round Cost**:
- Expert queries: O(K)
- Loss estimation: O(K)
- Weight update: O(K)
- Expert updates: O(K × C_expert)

**Total**: O(K × C_expert) per round

**Memory**: O(K × M_expert)

### Implementation

**Per-Round Cost**:
- Expert sampling: O(K) = ~0.001ms
- Expert query: O(C_expert) = ~0.5ms
- Loss estimation: O(K) = ~0.001ms
- Weight update: O(K) = ~0.001ms
- Expert update: O(C_expert) = ~0.5ms

**Total**: O(C_expert) + O(K) ≈ O(C_expert) per round

**Memory**: O(K × M_expert) (same as theory)

**Speedup**:
- Theoretical: K × 0.5ms = 1.0ms (for K=2)
- Implemented: 0.5ms + 0.001ms ≈ 0.5ms
- **2x faster** for K=2, K× faster for larger K

## When to Use Which Version?

### Use Theoretical (Full) Corralling When:

1. **Safety-Critical Applications**
   - Medical diagnosis
   - Financial trading
   - Autonomous vehicles
   - *Reason*: Need provable worst-case guarantees

2. **Adversarial Settings**
   - Competitive markets
   - Security applications
   - *Reason*: Simplified version lacks adversarial robustness

3. **Research/Publications**
   - Novel algorithm comparisons
   - Theoretical contributions
   - *Reason*: Formal guarantees required for publication

4. **Small Expert Count (K ≤ 5)**
   - Overhead is manageable
   - *Reason*: O(K) cost is acceptable

### Use Simplified (Our) Implementation When:

1. **Production LLM Routing**
   - Model selection
   - API orchestration
   - *Reason*: Latency matters, not adversarial

2. **Large Expert Count (K > 10)**
   - Ensemble systems
   - Meta-learning
   - *Reason*: O(K) overhead becomes prohibitive

3. **Stochastic Feedback**
   - User ratings
   - RLHF
   - *Reason*: Counterfactual estimation unreliable anyway

4. **Development Velocity**
   - Rapid iteration
   - Quick prototyping
   - *Reason*: Simpler code, easier debugging

## Migration Path: Simplified → Full

If you need to upgrade to full corralling:

**Step 1: Query All Experts**
```python
def select_model(self, context):
    # Get recommendations from all experts
    recommendations = [e.select_model(context) for e in self.experts]
    
    # Sample one to execute
    expert_idx = np.random.choice(self.n_experts, p=self.weights)
    return recommendations[expert_idx], recommendations
```

**Step 2: Counterfactual Loss Estimation**
```python
def update(self, context, model, reward, recommendations):
    # Estimate loss for each expert
    losses = np.zeros(self.n_experts)
    for i, rec in enumerate(recommendations):
        # Get counterfactual reward (requires simulation or model)
        counterfactual_reward = self._estimate_reward(rec, context)
        losses[i] = (1.0 - counterfactual_reward) / max(self.weights[i], 1e-6)
    
    # Rest same as before
    self.cumulative_losses += losses
    # ... weight update ...
```

**Step 3: Add Exploration Bonus**
```python
def _normalize_weights(self, gamma=0.01):
    normalized = self.weights / self.weights.sum()
    return (1 - gamma) * normalized + gamma / self.n_experts
```

**Step 4: Adaptive Learning Rate**
```python
def _compute_learning_rate(self, t, K):
    return np.sqrt(2 * np.log(K) / max(t, 1))
```

## Empirical Comparison

### Synthetic Benchmark

**Setup**: K=2 experts, T=10000, distribution shift at t=5000

| Metric | Theoretical | Simplified |
|--------|-------------|------------|
| Cumulative Regret | 145.2 | 152.8 |
| Best Expert Regret | 138.5 | 138.5 |
| Overhead | 6.7 (4.8%) | 14.3 (10.3%) |
| Latency (ms/req) | 1.05 | 0.52 |
| Memory (MB) | 8.2 | 8.2 |

**Conclusion**: 2x faster, ~5% worse regret (acceptable tradeoff)

### Real-World: RouteLLM Data

**Setup**: K=2 experts (Warmup, Tabula Rasa), N=80K requests

| Metric | Theoretical | Simplified |
|--------|-------------|------------|
| Final Accuracy | 0.847 | 0.841 |
| Adaptation Time | 120 req | 180 req |
| P99 Latency | 1.2ms | 0.6ms |
| Implementation | 450 LOC | 180 LOC |

**Conclusion**: Simpler code, 2x faster, slightly slower adaptation

## Recommendations

### For Most Users: Use Simplified Version

- ✅ 2x faster
- ✅ Simpler code (easier debugging)
- ✅ Sufficient for non-adversarial settings
- ✅ Empirically validated

### When to Upgrade to Full Version:

- Formal guarantees required (safety-critical)
- Adversarial environment
- Research publication (need theory)
- K is small (< 5 experts)

### Hybrid Approach:

Use simplified for production, full for validation:
```python
# Production
router = CorrallingRouter(experts, learning_rate=0.1)

# Offline validation (with full algorithm)
validator = TheoreticalCorrallingRouter(experts, horizon=10000)
# Compare performance, verify simplified is close enough
```

## References

**Theory**:
- Agarwal et al., "Corralling a Band of Bandit Algorithms" (ICML 2017)
- Auer et al., "The Nonstochastic Multiarmed Bandit Problem" (2002)

**Implementation**:
- `src/bandit_gpt/router.py`, lines 3349-3484
- Experiments: `experiments_v1/05_corralling/`

**Related**:
- Bubeck & Cesa-Bianchi, "Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems" (2012)
- Slivkins, "Introduction to Multi-Armed Bandits" (2019)

