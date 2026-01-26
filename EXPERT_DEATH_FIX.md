# Expert Death Prevention Fix (KDD Reviewer Feedback)

## Problem Statement

The KDD reviewer identified a critical theoretical vulnerability in our Corralling implementation:

**Expert Death in Non-Stationary Environments**

In non-stationary bandits (like LLM routing where new models appear or data shifts), pure exponential weighting can cause "Expert Death." If the Warmup Expert starts strong but later degrades (or if Tabula Rasa needs time to learn), the exponential weighting can drive the Tabula Rasa weight to essentially zero (~10^-16). Once that happens, the router effectively "stops listening" to that expert forever, making it impossible to adapt to future changes.

### Mathematical Issue

With pure exponential weighting:
```
w_i(t+1) ∝ exp(-η * L_i(t))
```

If Expert 1 accumulates high loss early on, its weight can become arbitrarily small. In a non-stationary environment where Expert 1 later becomes better, the router cannot recover because it never samples Expert 1 anymore.

## Solution: Mixing Parameter (γ)

We introduce a **Mixing Parameter** (γ) that adds a uniform exploration floor to the master's decision, ensuring every expert—no matter how bad—always has a non-zero chance (γ/K) of being tested.

### Theoretical Guarantee

The mixed distribution is:
```
P_t = (1-γ) * w_t + γ/K
```

Where:
- `w_t` is the learned exponential weight distribution
- `K` is the number of experts
- `γ` is the mixing parameter (default: 0.05)

This ensures:
1. **Minimum Probability**: Every expert has probability ≥ γ/K
2. **Recovery Capability**: Even if an expert performs poorly initially, it will still be sampled enough to detect if it becomes better later
3. **Bounded Loss**: Since p_t ≥ γ/K, the importance-weighted loss is bounded: max loss ≤ K/γ

## Implementation Changes

### 1. Added Mixing Parameter to Constructor

```python
def __init__(
    self,
    experts: List,
    models: List[str],
    learning_rate: float = 0.1,
    gamma: float = 0.05  # NEW: Mixing parameter
):
    self.gamma = gamma  # The "Life Support" parameter
    self.last_expert_prob = None  # Track actual prob used for selection
```

### 2. Added Mixed Distribution Method

```python
def _get_mixed_distribution(self) -> np.ndarray:
    """
    Compute P_t = (1-γ) * w_t + γ/K
    This mixes the learned policy (w_t) with uniform exploration (1/K).
    """
    uniform_dist = np.ones(self.n_experts) / self.n_experts
    return (1 - self.gamma) * self.weights + self.gamma * uniform_dist
```

### 3. Updated Selection to Use Mixed Distribution

```python
def select_model(self, context: np.ndarray, total_steps: int = 0) -> str:
    # [FIX] Use mixed distribution instead of raw weights
    probs = self._get_mixed_distribution()
    
    expert_idx = np.random.choice(self.n_experts, p=probs)
    
    self.last_expert_idx = expert_idx
    self.last_expert_prob = probs[expert_idx]  # Save for unbiased update
    ...
```

### 4. Updated Loss Estimator to Use Mixed Probability

**Critical**: The importance-weighted loss estimator MUST use the mixed probability, not the raw weight. Otherwise, the estimator would be biased.

```python
def update(self, context: np.ndarray, model: str, reward: float):
    observed_loss = 1.0 - reward
    losses = np.zeros(self.n_experts)
    
    # [FIX] Use the MIXED probability (p_t) for the estimator denominator
    # If we used raw weights, the estimator would be biased.
    # Since p_t >= gamma/K, this term is bounded (max loss <= K/gamma).
    p_chosen = self.last_expert_prob
    
    # Importance-Weighted Estimator: l_hat = l_obs / p_chosen
    losses[self.last_expert_idx] = observed_loss / p_chosen
    ...
```

## Backward Compatibility

✅ **Fully Backward Compatible**

All existing code continues to work without modification because:
- The `gamma` parameter has a sensible default value (0.05)
- All existing instantiations use keyword arguments or rely on defaults
- The API signature is unchanged (only added an optional parameter)

## Test Coverage

We added comprehensive tests in `tests/test_expert_death_fix.py`:

1. **`test_mixing_parameter_prevents_zero_probability`**: Verifies that even after 1000 iterations with consistent bad performance, Expert 1 maintains probability ≥ γ/K

2. **`test_recovery_in_nonstationary_environment`**: Simulates a phase shift where Expert 1 starts bad but becomes good. Verifies the router can recover and increase Expert 1's weight.

3. **`test_gamma_zero_causes_expert_death`**: Confirms that γ=0 leads to much lower probabilities than γ>0, demonstrating the problem.

4. **`test_importance_weighting_uses_mixed_probability`**: Verifies that the loss estimator uses the mixed probability (unbiased).

5. **`test_gamma_parameter_bounds`**: Tests various gamma values and verifies correctness.

All tests pass ✅

## Performance Impact

**Negligible**

- **Memory**: +1 float (gamma) + 1 float (last_expert_prob) per router instance
- **Computation**: +O(K) per selection (computing mixed distribution)
- **Practical Impact**: ~0.01ms overhead vs ~100ms LLM inference time

## Theoretical References

This fix is based on the **Exp4** algorithm (Auer et al., 2002) and the **Corralling** algorithm (Agarwal et al., 2017), which both use mixing parameters to ensure exploration.

- Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1), 48-77.
- Agarwal, A., Luo, H., Neyshabur, B., & Schapire, R. E. (2017). Corralling a band of bandit algorithms. In Conference on Learning Theory (pp. 12-38).

## Recommendation

**Default Configuration**: Use γ=0.05 (5% uniform exploration)

This provides:
- 95% exploitation of learned policy
- 5% guaranteed exploration across all experts
- Minimum probability per expert: 2.5% (with 2 experts)

For more aggressive exploration in highly non-stationary environments, consider γ=0.1 or higher.

## Files Modified

1. **`src/bandit_gpt/router.py`**: Updated `CorrallingRouter` class
   - Added `gamma` parameter
   - Added `_get_mixed_distribution()` method
   - Updated `select_model()` to use mixed distribution
   - Updated `update()` to use mixed probability in loss estimator

2. **`tests/test_expert_death_fix.py`**: New comprehensive test suite

## Verification

Run tests:
```bash
pytest tests/test_expert_death_fix.py -v
```

All 5 tests pass ✅

## Impact on Paper

This fix should be mentioned in the paper's methodology section:

> "To prevent Expert Death in non-stationary environments, we implement a mixing parameter γ=0.05 that ensures every expert maintains a minimum selection probability of γ/K. This allows the router to recover if an initially poor expert becomes better over time, which is critical for adapting to new models or data shifts."

The fix strengthens our theoretical guarantees and addresses a key concern from the KDD reviewer.

