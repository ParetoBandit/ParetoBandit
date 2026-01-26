# Mathematical Appendix: Corralling Algorithm

## Formal Problem Setup

### Notation

- **K**: Number of expert policies (K=2 in our case)
- **T**: Time horizon (number of routing decisions)
- **A_t**: Set of available models at time t
- **x_t**: Context vector at time t (d-dimensional)
- **a_t**: Selected model at time t
- **r_t**: Observed reward at time t ∈ [0,1]
- **π_i**: Expert policy i ∈ {1,...,K}
- **p_{i,t}**: Probability of selecting expert i at time t
- **ℓ_{i,t}**: Loss incurred by expert i at time t

### Expert Policies

**Expert 1 (Warmup)**:
```
π_1(x_t) = argmax_{a ∈ A_t} [θ_a^T x_t + α_t · √(x_t^T A_a^{-1} x_t)]
```
Where:
- θ_a = A_a^{-1} b_a (learned from 80k RouteLLM battles)
- A_a initialized with warmup covariance (high confidence)
- α_t decays from 0.5 to 0.1 (low exploration)

**Expert 2 (Tabula Rasa)**:
```
π_2(x_t) = argmax_{a ∈ A_t} [θ_a^T x_t + α_t · √(x_t^T A_a^{-1} x_t)]
```
Where:
- θ_a = A_a^{-1} b_a (learned online from t=0)
- A_a initialized with λI (identity, maximum uncertainty)
- α_t decays from 2.0 to 0.5 (high exploration)

### Objective

Minimize regret relative to the best expert in hindsight:
```
Regret(T) = max_{i∈{1,...,K}} ∑_{t=1}^T r_{i,t} - ∑_{t=1}^T r_t
```
Where r_{i,t} is the reward expert i would have obtained at time t.

## The Corralling Algorithm

### Initialization

```
p_{i,0} = 1/K  for all i ∈ {1,...,K}    (uniform distribution)
L_{i,0} = 0    for all i ∈ {1,...,K}    (cumulative loss)
```

### Selection Rule

At each time t:

1. **Sample expert**: Draw i_t ~ p_t (probability vector over experts)
2. **Query expert**: Get action a_t = π_{i_t}(x_t)
3. **Observe reward**: Receive r_t ∈ [0,1]
4. **Convert to loss**: ℓ_t = 1 - r_t

### Update Rule (Exponential Weights)

**Step 1**: Importance-weighted loss estimation
```
ℓ̂_{i,t} = {
    ℓ_t / p_{i_t,t}   if i = i_t  (chosen expert)
    0                 if i ≠ i_t  (unchosen expert)
}
```

**Justification**: This estimator is unbiased:
```
E[ℓ̂_{i,t} | F_t] = p_{i,t} · (ℓ_t / p_{i,t}) = ℓ_t
```

**Step 2**: Accumulate losses
```
L_{i,t} = L_{i,t-1} + ℓ̂_{i,t}
```

**Step 3**: Exponential weight update
```
w_{i,t+1} = exp(-η · L_{i,t})
```

**Step 4**: Normalize to probability distribution
```
p_{i,t+1} = w_{i,t+1} / ∑_{j=1}^K w_{j,t+1}
```

### Implementation (Numerically Stable)

```python
# Log-space computation (avoids underflow)
log_weights = -η * cumulative_losses

# Shift by max (prevents overflow in exp)
log_weights -= log_weights.max()

# Exponentiate and normalize
weights = np.exp(log_weights)
weights /= weights.sum()
```

## Theoretical Guarantees

### Theorem 1 (Agarwal et al., 2017): Regret Bound

**Statement**: For any sequence of contexts and rewards, the Corralling algorithm with learning rate η > 0 satisfies:

```
E[Regret(T)] ≤ (ln K) / η + η · ∑_{t=1}^T E[max_{i,j} (ℓ_{i,t} - ℓ_{j,t})²] / 8
```

**Proof Sketch**:

1. **Potential function**: Define Φ_t = -ln(∑_{i=1}^K w_{i,t})

2. **Potential difference**: 
   ```
   Φ_{t+1} - Φ_t = η · ∑_{i=1}^K p_{i,t} ℓ̂_{i,t} - ln(E[exp(-η ℓ̂_{i_t,t})])
   ```

3. **Hoeffding's lemma** (for bounded losses):
   ```
   ln(E[exp(-η ℓ̂)]) ≤ -η E[ℓ̂] + η² E[ℓ̂²] / 8
   ```

4. **Telescope sum**:
   ```
   ∑_{t=1}^T [∑_{i} p_{i,t} ℓ̂_{i,t} - ℓ̂_{i_t,t}] ≤ Φ_T - Φ_0 + η·T·variance / 8
   ```

5. **Initial/final potential**:
   ```
   Φ_0 = ln K  (uniform start)
   Φ_T ≥ 0     (probabilities sum to 1)
   ```

**Corollary (Our Setting)**: For K=2, bounded rewards in [0,1], and η=1.0:
```
E[Regret(500)] ≤ ln(2) / 1.0 + 1.0 · 500 · (1/4) / 8
               ≤ 0.693 + 15.625
               ≤ 16.3
```

**Interpretation**: Even if one expert is completely wrong, we lose at most ~16 rewards over 500 steps compared to always using the best expert.

### Theorem 2: Adaptive Rate

**Statement**: Let Δ_t = max_i L_{i,t} - min_i L_{i,t} be the loss gap at time t. Then:

```
p_{best,t} ≥ exp(-η · Δ_t) / (1 + exp(-η · Δ_t))
```

Where "best" is the expert with minimum cumulative loss.

**Proof**:
```
p_{best,t} = exp(-η · L_{best,t}) / ∑_j exp(-η · L_{j,t})
           = exp(-η · L_{best,t}) / [exp(-η · L_{best,t}) + ∑_{j≠best} exp(-η · L_{j,t})]
           = 1 / [1 + ∑_{j≠best} exp(-η · (L_{j,t} - L_{best,t}))]
           ≥ 1 / [1 + (K-1) · exp(-η · Δ_t)]     (since L_{j,t} - L_{best,t} ≤ Δ_t)
```

For K=2:
```
p_{best,t} ≥ 1 / [1 + exp(-η · Δ_t)] = exp(η · Δ_t) / [exp(η · Δ_t) + 1]
```

**Interpretation**: The probability of selecting the best expert grows exponentially with the loss gap Δ_t.

### Corollary: Decommissioning Rate

If expert 1 (warmup) is consistently worse by margin ε > 0:
```
L_{1,t} - L_{2,t} ≈ ε · t    (loss gap grows linearly)
```

Then:
```
p_{2,t} ≥ exp(η · ε · t) / [exp(η · ε · t) + 1] → 1  as t → ∞
```

**Rate of convergence**: For η=1.0, ε=0.1 (10% reward difference):
```
p_{2,t} ≥ 0.5    when t ≥ 0     (starts at 50%)
p_{2,t} ≥ 0.9    when t ≥ 22    (90% after 22 steps)
p_{2,t} ≥ 0.99   when t ≥ 44    (99% after 44 steps)
p_{2,t} ≥ 0.999  when t ≥ 66    (99.9% after 66 steps)
```

**This is the "decisive decommissioning" we observe in Figure 5.**

## Comparison with Alternatives

### Naive Approach: Pick Best Expert A Priori

**Problem**: No way to detect prior mismatch
```
Regret = {
    0           if warmup is correct
    Ω(T)        if warmup is wrong (linear regret!)
}
```

### Alternative: Uniform Mixing

**Problem**: Wastes resources on bad expert
```
Regret = Θ(√T)  (always, even when one expert is clearly better)
```

### Alternative: ε-Greedy

**Problem**: Requires manual tuning of ε
```
Regret = O(√T)  if ε set correctly
Regret = Ω(T)   if ε set incorrectly
```

### Corralling: Automatic Adaptation

**Advantage**: Adapts automatically without hyperparameters
```
Regret = O(√T)  in worst case (adaptive)
Regret = O(1)   if one expert is clearly better (decisive)
```

## Numerical Stability Analysis

### Overflow Prevention

**Problem**: For large T, exp(-η · L_{i,T}) can underflow to 0

**Solution**: Shift log-weights by maximum before exponentiation
```python
log_w = -η * L
log_w -= log_w.max()  # Now max(log_w) = 0
w = exp(log_w)        # Largest weight is exp(0) = 1
```

**Justification**: This is equivalent to multiplying all weights by the same constant:
```
w'_i = w_i / max_j w_j
p_i = w_i / ∑_j w_j = w'_i / ∑_j w'_j
```

### Division by Zero Prevention

**Problem**: If p_{i,t} → 0, then ℓ̂_{i,t} = ℓ_t / p_{i,t} → ∞

**Solution**: Clip probabilities to minimum value
```python
losses[chosen_idx] = observed_loss / max(p_chosen, 1e-6)
```

**Impact**: Negligible for reasonable learning rates (η ≤ 5)

## Extensions and Variants

### Variant 1: Adaptive Learning Rate

Instead of fixed η, use time-varying η_t:
```
η_t = √(2 ln K / t)    (optimal for stochastic setting)
```

**Trade-off**: Better worst-case bound, but slower adaptation when gap is large

### Variant 2: Sleeping Experts

Allow experts to abstain ("sleep") on certain contexts:
```
π_i(x_t) = ∅    (expert i sleeps, not selected)
```

**Modification**: Normalize only over awake experts
```
p_{i,t} = w_{i,t} / ∑_{j∈awake(t)} w_{j,t}
```

### Variant 3: Bandit Feedback

Only observe reward for chosen action (standard bandit setting):
```
ℓ̂_{i,t} = ℓ(a_t) · I{i = i_t} / p_{i,t}
```

**Our setting**: Full information (we know optimal reward via judge)

### Variant 4: Best-of-Both-Worlds

Achieve O(√T) regret in adversarial setting AND O(log T) in stochastic:
```
Use Follow-the-Regularized-Leader with Tsallis entropy
```

**Reference**: Zimmert & Seldin (2021)

## Connection to Statistical Learning

### View 1: Online Mirror Descent

Corralling is equivalent to mirror descent with KL divergence:
```
p_{t+1} = argmin_{p} [⟨p, ℓ̂_t⟩ + (1/η) · KL(p || p_t)]
```

**Closed form**: Exponential weights update

### View 2: Bayesian Posterior

If losses are i.i.d. from distributions μ_i:
```
p_{i,t} ∝ exp(-η · L_{i,t}) = ∏_{s=1}^t exp(-η · ℓ_{i,s})
```

**Interpretation**: η=1/temperature in Bayesian posterior with exponential likelihood

### View 3: Multiplicative Weights

Each expert has a "budget" w_{i,t}:
```
w_{i,t+1} = w_{i,t} · (1 - η · ℓ̂_{i,t})    (multiplicative update)
```

**Equivalence**: For small η, this is first-order approximation to exp(-η · ℓ̂)

## Experimental Design: Isolating Prior Misalignment

### Quality-Only Mode (cost_penalty=0.0)

The default experiment uses **quality-only optimization** to isolate prediction error from cost-quality trade-offs.

**Utility Function (Both Experts)**:
```
U(a_t | x_t) = θ_a^T x_t + α_t · √(x_t^T A_a^{-1} x_t)
              └─ predicted reward ─┘   └─ exploration bonus ─┘
```

**With Cost Penalty (Alternative)**:
```
U(a_t | x_t) = [θ_a^T x_t + α_t · √(x_t^T A_a^{-1} x_t)] - λ · cost(a_t)
              └────────── quality component ─────────────┘   └─ cost penalty ─┘
```

### Why Zero Cost Penalty by Default?

**Reason 1: Isolates Prediction Error**

Loss decomposition with cost_penalty=0:
```
ℓ_t = 1 - r_t = 1 - (true quality of selected model)
```

Loss is purely from wrong quality predictions, not cost miscalibration.

**Reason 2: Fair Expert Comparison**

Both experts optimize the same objective:
```
Expert 1 (Warmup):      max E[reward | warmup beliefs]
Expert 2 (Tabula Rasa): max E[reward | learned beliefs]
```

Only difference: initialization (prior vs cold start)

**Reason 3: Clean Causal Interpretation**

If decommissioning occurs:
```
Cause: P_warmup(quality | context) ≠ P_true(quality | context)
       └─ domain shift / distribution mismatch
```

NOT:
```
Cause: λ_warmup ≠ λ_optimal  (wrong cost sensitivity)
```

### Mathematical Analysis: Quality Inversion

**Setup**: Suppose warmup was trained on distribution P_train where:
```
E_train[r | GPT-4]   = 0.90  (expensive, excellent)
E_train[r | Mixtral] = 0.70  (cheap, mediocre)
```

But production distribution P_prod shows:
```
E_prod[r | GPT-4]   = 0.75  (expensive, good)
E_prod[r | Mixtral] = 0.85  (cheap, excellent)
```

**Warmup Expert Loss** (repeatedly picks GPT-4):
```
L_1,t ≈ ∑_{s=1}^t (1 - 0.75) = 0.25t
```

**Tabula Rasa Loss** (learns Mixtral is better):
```
L_2,t ≈ ∑_{s=1}^t (1 - 0.85) = 0.15t
```

**Loss Gap**:
```
Δ_t = L_1,t - L_2,t ≈ 0.10t
```

**Weight Evolution** (from Theorem 2):
```
p_1,t ≈ exp(-η · 0.25t) / Z_t
p_2,t ≈ exp(-η · 0.15t) / Z_t

p_2,t / p_1,t ≈ exp(η · 0.10t) → ∞  as t → ∞
```

For η=1.0:
```
p_2,100 / p_1,100 ≈ exp(10) ≈ 22,000
→ p_2,100 ≈ 99.995%
```

**Decisive decommissioning occurs by t≈100.**

## Experimental Validation

### Hypothesis 1: Exponential Decommissioning

**H₁**: When warmup expert incurs higher loss, its weight should decay exponentially

**Test**: Measure weight ratio log(p_1 / p_2) over time

**Expected**: Linear decrease with slope ≈ -η · ε (where ε is reward gap)

**Validation (Quality-Only Mode)**:
- If ε=0.10 (Mixtral 10% better than GPT-4) and η=1.0
- Slope should be ≈ -0.10 per step
- By t=100: log(p_1/p_2) ≈ -10 → p_1/p_2 ≈ 0.00005

### Hypothesis 2: Importance Weighting

**H₂**: Without importance weighting (using ℓ̂_{i,t} = ℓ_t always), convergence should be slower

**Test**: Compare standard vs naive estimator on same data

**Expected**: 2-3x slower convergence without importance weighting

### Hypothesis 3: Learning Rate Sensitivity

**H₃**: Higher η should cause faster but noisier adaptation

**Test**: Run with η ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, measure time to 90% weight

**Expected**: Time ∝ 1/η, but variance ∝ η

### Hypothesis 4: Cost Penalty Impact (Extension)

**H₄**: With non-zero cost penalty, decommissioning can occur even with correct quality predictions

**Setup**: Both experts predict quality equally well, but:
- Warmup: cost_penalty=0.0 (cost-blind)
- Tabula Rasa: cost_penalty=0.5 (cost-aware)

**Test**: Run on dataset where cheap model (Mixtral) has slightly lower quality but much lower cost

**Expected**: TR achieves better cost-adjusted utility, warmup gets decommissioned despite correct quality predictions

**Validation**: Loss gap driven by objective mismatch, not prediction error

## Practical Guidelines

### Choosing Learning Rate

| η | Use Case | Characteristics |
|---|----------|----------------|
| 0.1 | High noise, long horizon | Stable, slow adaptation |
| 0.5 | Moderate noise | Balanced |
| 1.0 | **Default** (paper) | Fast adaptation, some noise |
| 2.0 | Low noise, short horizon | Aggressive |
| 5.0 | Nearly deterministic | May overreact |

### When to Use Corralling

✅ **Use when**:
- Prior source is uncertain (domain mismatch risk)
- Need worst-case guarantees
- Can afford 2x memory (two sets of parameters)
- Want automatic adaptation (no manual tuning)

❌ **Skip when**:
- Prior is highly trusted (same domain, validated)
- Memory budget is extremely tight
- Only care about expected case (not worst-case)

## Open Questions

1. **Optimal α-decay**: Should warmup/tabula experts use different exploration schedules?

2. **Prior strength tuning**: Can we automatically adjust prior confidence based on early evidence?

3. **Multi-objective**: How to extend Corralling to cost-quality trade-offs?

4. **Non-stationary**: How to handle distribution shift over time?

## References

### Core Algorithm

- Agarwal, A., Luo, H., Neyshabur, B., & Schapire, R. E. (2017). Corralling a band of bandit algorithms. In Conference on Learning Theory (pp. 12-38). PMLR.

### Extensions

- Agarwal, A., & Zhang, T. (2022). Corralling a larger band of bandits: A case study on switching regret for linear bandits. In Conference on Uncertainty in Artificial Intelligence (pp. 38-47). PMLR.

### Related Work

- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1), 119-139.

- Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, learning, and games. Cambridge University Press.

- Zimmert, J., & Seldin, Y. (2021). Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28), 1-49.

### LinUCB Foundation

- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web (pp. 661-670).

- Abbasi-Yadkori, Y., Pál, D., & Szepesvári, C. (2011). Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems (pp. 2312-2320).

