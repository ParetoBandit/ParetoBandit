# Implementation Guide: Using CorrallingRouter

## Quick Start

### Basic Usage

```python
from bandit_gpt.router import BanditRouter, CorrallingRouter

# 1. Create two expert routers
warmup_router = BanditRouter.create(
    model_registry=registry,
    priors="warmup",  # Load warmup priors
    alpha=1.0
)

tabula_rasa_router = BanditRouter.create(
    model_registry=registry,
    priors=None,  # No priors
    alpha=1.0
)

# 2. Wrap in Corralling
hybrid = CorrallingRouter(
    experts=[warmup_router, tabula_rasa_router],
    models=list(registry.keys()),
    learning_rate=0.1
)

# 3. Use like any router
context = "Write a Python function to parse JSON"
selected_model = hybrid.select_model(context)

# 4. Provide feedback
reward = 0.85  # From your reward function
hybrid.update(context, selected_model, reward)
```

## Configuration

### Learning Rate Selection

The learning rate η controls how quickly the coordinator adapts trust weights:

| η Value | Behavior | Use Case |
|---------|----------|----------|
| 0.01 | Conservative | Stable environments, avoid volatility |
| 0.1 | Balanced (default) | General purpose, robust to noise |
| 0.5 | Aggressive | Rapid distribution shifts, quick adaptation |
| 1.0 | Very aggressive | Research/debugging, fast convergence |

**Rule of Thumb**: Start with η=0.1, increase if you observe slow adaptation to distribution shifts.

### Expert Configuration

**Warmup Expert**:
```python
warmup_router = BanditRouter.create(
    model_registry=registry,
    priors="warmup",           # Load offline priors
    alpha=1.0,                 # Moderate exploration
    prior_n_effective=100.0    # Prior strength (pseudocounts)
)
```

**Tabula Rasa Expert**:
```python
tabula_rasa_router = BanditRouter.create(
    model_registry=registry,
    priors=None,               # No priors
    alpha=1.0,                 # Moderate exploration
    init_lambda=1.0            # Regularization only
)
```

**Key Differences**:
- Warmup: b_0 ≠ 0 (contains prior beliefs)
- Tabula Rasa: b_0 = 0 (learns from scratch)

## Monitoring and Diagnostics

### Check Trust Weights

```python
# Get current trust distribution
weights = hybrid.get_expert_weights()
print(weights)
# Output: {
#   'expert_0 (BanditRouter)': 0.72,
#   'expert_1 (BanditRouter)': 0.28
# }
```

**Interpretation**:
- 0.72 → Warmup expert is trusted 72% of the time
- 0.28 → Tabula Rasa expert is trusted 28% of the time
- Sum always equals 1.0 (probability distribution)

### Track Selection Frequency

```python
# How many times each expert was selected
print(hybrid.expert_selections)
# Output: [720, 280]

# How many times each model was selected
print(hybrid.selections)
# Output: {'gpt-4': 450, 'claude-opus': 350, 'mixtral': 200}
```

### Visualize Trust Evolution

```python
import matplotlib.pyplot as plt

# Log weights over time
trust_history = []
for t in range(1000):
    # ... routing and feedback loop ...
    weights = hybrid.get_expert_weights()
    trust_history.append([
        weights['expert_0 (BanditRouter)'],
        weights['expert_1 (BanditRouter)']
    ])

# Plot
plt.plot(trust_history)
plt.xlabel('Requests')
plt.ylabel('Trust Weight')
plt.legend(['Warmup', 'Tabula Rasa'])
plt.title('Expert Trust Evolution')
plt.show()
```

**Expected Patterns**:
- **Good Transfer**: Warmup stays high (0.7-0.9)
- **Zero Transfer**: Tabula Rasa rises to dominate (0.6-0.8)
- **Partial Transfer**: Both maintain ~0.5 (balanced)

### Cumulative Loss Tracking

```python
# Lower loss = better performance
print(hybrid.cumulative_losses)
# Output: [45.2, 89.7]
# Interpretation: Warmup has lower loss (45.2 < 89.7) → gets more trust
```

## Advanced Usage

### Custom Experts

You can wrap any routing algorithm that implements `select_model` and `update`:

```python
class MyCustomExpert:
    def select_model(self, context: np.ndarray) -> str:
        # Your custom logic
        return "gpt-4"
    
    def update(self, context: np.ndarray, model: str, reward: float):
        # Your custom update logic
        pass

custom_expert = MyCustomExpert()
hybrid = CorrallingRouter(
    experts=[warmup_router, custom_expert],
    models=models,
    learning_rate=0.1
)
```

### More Than Two Experts

```python
# Example: Three-way corralling
hybrid = CorrallingRouter(
    experts=[warmup_router, tabula_rasa_router, heuristic_router],
    models=models,
    learning_rate=0.1
)

# Weights will sum to 1.0 across all three experts
# Output: {'expert_0': 0.5, 'expert_1': 0.3, 'expert_2': 0.2}
```

**Note**: With K experts, convergence may be slower. Consider reducing learning_rate.

### Delayed Feedback

If rewards arrive asynchronously, buffer them:

```python
feedback_buffer = []

# At routing time
context = encode_prompt("...")
model = hybrid.select_model(context)
feedback_buffer.append((context, model, hybrid.last_expert_idx))

# Later, when reward arrives
for context, model, expert_idx in feedback_buffer:
    reward = get_reward(...)  # From your system
    hybrid.last_expert_idx = expert_idx  # Restore state
    hybrid.update(context, model, reward)

feedback_buffer.clear()
```

## Troubleshooting

### Problem: One Expert Dominates Immediately

**Symptoms**: Weight goes to 0.99 after 10 requests

**Causes**:
1. Learning rate too high (η > 1.0)
2. Reward function has high variance
3. One expert is genuinely much better

**Solutions**:
- Reduce learning_rate to 0.01-0.05
- Smooth rewards (moving average)
- Check reward function calibration

### Problem: Weights Oscillate Wildly

**Symptoms**: Weight swings between 0.2 and 0.8 rapidly

**Causes**:
1. Learning rate too high
2. Noisy reward signal
3. Small sample size (< 100 requests)

**Solutions**:
- Reduce learning_rate to 0.05
- Increase reward smoothing window
- Wait for more samples before analyzing

### Problem: No Adaptation to Distribution Shift

**Symptoms**: Trust doesn't shift when deployment domain changes

**Causes**:
1. Learning rate too low (η < 0.01)
2. Shift is subtle (experts perform similarly)
3. Insufficient samples after shift

**Solutions**:
- Increase learning_rate to 0.1-0.2
- Verify shift actually exists (manual inspection)
- Wait 100-200 requests post-shift

### Problem: High Memory Usage

**Symptoms**: RAM consumption doubles vs single router

**Expected**: This is normal! Corralling stores two sets of matrices.

**If Excessive**:
- Check dimensionality (d=384 is large)
- Consider PCA to reduce d to 32-64
- Use float32 instead of float64

## Performance Optimization

### Reduce Latency

**Default Overhead**: ~0.5ms per request

**Optimization 1: Precompute Expert Decisions**
```python
# If experts are deterministic, cache their recommendations
cache = {}
model_a = experts[0].select_model(context)
model_b = experts[1].select_model(context)
cache[(hash(context), 0)] = model_a
cache[(hash(context), 1)] = model_b

# Fast selection
expert_idx = sample_categorical(weights)
model = cache[(hash(context), expert_idx)]
```

**Optimization 2: Batch Updates**
```python
# Instead of updating after each request
contexts = []
models = []
rewards = []

# ... collect batch ...

for ctx, mdl, rew in zip(contexts, models, rewards):
    hybrid.update(ctx, mdl, rew)
```

### Reduce Memory

**Option 1: Share Encoder Between Experts**
```python
# Both experts use same SentenceTransformer
shared_encoder = SentenceTransformer('all-MiniLM-L6-v2')

warmup_router = BanditRouter(
    ...,
    context_encoder=shared_encoder  # Reuse
)

tabula_rasa_router = BanditRouter(
    ...,
    context_encoder=shared_encoder  # Reuse
)
```

**Option 2: Use Lower-Dimensional Features**
```python
# PCA to d=32 instead of d=384
pca = PCA(n_components=32)
# ... train PCA ...

warmup_router = BanditRouter(..., pca_path="pca_32.joblib")
tabula_rasa_router = BanditRouter(..., pca_path="pca_32.joblib")
```

## Testing and Validation

### Unit Test: Weight Updates

```python
def test_weight_updates():
    hybrid = CorrallingRouter(
        experts=[expert_a, expert_b],
        models=['gpt-4'],
        learning_rate=0.1
    )
    
    # Initial: equal weights
    assert hybrid.weights[0] == 0.5
    assert hybrid.weights[1] == 0.5
    
    # Expert 0 performs poorly
    hybrid.last_expert_idx = 0
    hybrid.update(context, 'gpt-4', reward=0.2)  # Low reward → high loss
    
    # Expert 0 should be downweighted
    assert hybrid.weights[0] < 0.5
    assert hybrid.weights[1] > 0.5
```

### Integration Test: Distribution Shift

```python
def test_distribution_shift():
    # Warmup trained on domain A
    warmup = BanditRouter.create(priors="warmup_domain_a")
    tabula_rasa = BanditRouter.create(priors=None)
    
    hybrid = CorrallingRouter([warmup, tabula_rasa], models)
    
    # Phase 1: Domain A (warmup should dominate)
    for prompt in domain_a_prompts:
        model = hybrid.select_model(prompt)
        reward = judge(model, prompt)
        hybrid.update(prompt, model, reward)
    
    weights_a = hybrid.weights.copy()
    assert weights_a[0] > 0.6  # Warmup dominant
    
    # Phase 2: Domain B (tabula rasa should catch up)
    for prompt in domain_b_prompts:
        model = hybrid.select_model(prompt)
        reward = judge(model, prompt)
        hybrid.update(prompt, model, reward)
    
    weights_b = hybrid.weights.copy()
    assert weights_b[1] > weights_a[1]  # Tabula Rasa increased trust
```

## Production Deployment Checklist

- [ ] Set learning_rate based on expected shift frequency
- [ ] Configure logging for trust weights (track every 100 requests)
- [ ] Set up alerts for extreme weights (< 0.05 or > 0.95)
- [ ] Implement reward function with proper normalization [0, 1]
- [ ] Test with synthetic distribution shift before production
- [ ] Monitor cumulative losses for debugging
- [ ] Plan for 2x memory overhead vs single router
- [ ] Document expected expert behaviors for on-call team

## References

- Implementation: `src/bandit_gpt/router.py`, lines 3349-3484
- Theory: Agarwal et al., "Corralling a Band of Bandit Algorithms" (2017)
- Alternative: Full importance-weighted corralling (research only)

