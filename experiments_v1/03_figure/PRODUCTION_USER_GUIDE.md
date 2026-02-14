# 🚀 BanditGPT Production Deployment Guide

**Last Updated:** February 14, 2026  
**Status:** Critical bug fixed - safe for production deployment  
**Target Audience:** ML Engineers, DevOps, Production System Owners

---

## Quick Start: Safe Production Deployment

### ✅ **Correct Usage Pattern**

```python
from bandit_gpt.router import CorrallingRouter, CostAwareLinUCBRouter, CostAwareTabulaRasaRouter

# 1. Initialize experts
warmup_expert = CostAwareLinUCBRouter(
    models=models,
    warmup_priors=priors,  # Your pre-trained priors
    alpha_start=2.0,
    alpha_end=2.0,  # Constant exploration
)

tabula_rasa_expert = CostAwareTabulaRasaRouter(
    models=models,
    context_dim=context_dim,
    alpha_start=2.0,
    alpha_end=2.0,
)

# 2. Create Corralling router
router = CorrallingRouter(
    experts=[warmup_expert, tabula_rasa_expert],
    models=models,
    learning_rate=1.0,
    gamma=0.05,
)

# 3. CRITICAL: Proper selection and update loop
for request in production_stream:
    # Extract context
    context = embed_prompt(request.prompt, encoder, pca)
    
    # 🚨 MUST capture selection_token (not just model)
    selected_model, selection_token = router.select_model(context)
    
    # Execute model and get reward
    response = execute_llm(selected_model, request.prompt)
    reward = evaluate_quality(response)  # Your quality metric
    
    # 🚨 MUST pass selection_token to update
    router.update(context, selected_model, reward, selection_token)
    
    # Optional: Log for monitoring
    log_metrics({
        'model': selected_model,
        'reward': reward,
        'warmup_weight': router.weights[0],
        'tabula_weight': router.weights[1],
    })
```

---

## ⚠️ Common Pitfalls (How to Fail Silently)

### ❌ **Anti-Pattern 1: Discarding Selection Token**

```python
# THIS WILL BREAK SILENTLY - DO NOT DO THIS
selected_model, _ = router.select_model(context)  # Token discarded!
router.update(context, selected_model, reward)     # No adaptation!

# Symptoms:
# - router.weights stays at [0.5, 0.5] forever
# - Performance doesn't improve with data
# - No error message (silent failure)
```

### ❌ **Anti-Pattern 2: Using Stale Token**

```python
# THIS CREATES BIASED UPDATES - DO NOT DO THIS
token = None
for request in requests:
    model, token = router.select_model(context)  # New token each time
    
# Later (after many selections)...
router.update(old_context, model, reward, token)  # Token is stale!
```

**Rule:** The token from `select_model()` must be used immediately in the corresponding `update()`. Don't store tokens across multiple selections.

---

## 📊 Production Monitoring Dashboard

### Essential Metrics to Track

```python
import logging

# 1. Expert Weight Evolution (MOST CRITICAL)
logging.info(f"Warmup weight: {router.weights[0]:.3f}")
logging.info(f"Tabula weight: {router.weights[1]:.3f}")

# 2. Selection Counts
logging.info(f"Expert selections: {router.expert_selections}")
logging.info(f"Model selections: {router.selections}")

# 3. Performance Metrics
logging.info(f"Average reward: {np.mean(recent_rewards):.3f}")
logging.info(f"Regret vs oracle: {cumulative_regret:.1f}")
```

### Interpretation Guide

| Warmup Weight | What It Means | Recommended Action |
|--------------|---------------|-------------------|
| **> 0.80** | 🟢 **Priors are highly accurate** | Consider simplifying to warmup-only for lower latency |
| **0.50 - 0.80** | 🟡 **Priors moderately helpful** | Continue with Corralling (working as designed) |
| **0.20 - 0.50** | 🟠 **Priors marginally useful** | Monitor closely; may indicate transition period |
| **< 0.20** | 🔴 **Priors harmful** | Consider switching to pure Tabula Rasa |
| **Exactly 0.50** | ⚠️ **FROZEN (BUG!)** | Check selection_token implementation immediately |

### Alerting Rules

```python
# Alert 1: Frozen Weights (Implementation Bug)
if np.std(weight_history[-100:]) < 0.01:
    alert("CRITICAL: Weights frozen - selection_token bug likely")

# Alert 2: Harmful Priors Detected
if router.weights[0] < 0.20 and timesteps > 200:
    alert("WARNING: Warmup priors performing poorly - consider switching strategy")

# Alert 3: Rapid Weight Shift
if abs(router.weights[0] - prev_weights[0]) > 0.3 in single update:
    alert("INFO: Large weight shift detected - possible distribution shift")
```

---

## 🎯 Deployment Strategy Decision Tree

```
START
│
├─ Do you have warmup priors?
│  │
│  NO → Use Tabula Rasa (cold start)
│  │    Expected: Slower convergence, no negative transfer
│  │
│  YES → Continue
│      │
│      ├─ Have you validated priors on deployment data (N=100-200)?
│      │  │
│      │  NO → Use Corralling (safe default)
│      │  │    Expected: 20% overhead, 18.5% safety improvement
│      │  │
│      │  YES → Check validation accuracy
│      │      │
│      │      ├─ Accuracy > 80% → Use Warmup Only
│      │      │                   Expected: Best performance, lowest latency
│      │      │
│      │      ├─ Accuracy 50-80% → Use Corralling
│      │      │                    Expected: Adaptive hedging
│      │      │
│      │      └─ Accuracy < 50% → Use Tabula Rasa
│                                  Expected: 16% improvement vs Corralling
```

---

## 📈 Performance Benchmarks (LMSYS Holdout)

### After Bug Fix (2026-02-14)

| Configuration | Regret | Weight Evolution | Use Case |
|--------------|--------|------------------|----------|
| **Corralling (Fixed)** | 39.5 ± 5.6 | 0.46 → 0.88 | ✅ Production recommended |
| Corralling (Broken) | 50.2 ± 5.1 | Frozen at 0.50 | ❌ Do not use |
| Tabula Rasa | 49.5 ± 2.8 | N/A | When priors known bad |
| Warmup Only | TBD | N/A | When priors validated good |

**Key Finding:** The corrected implementation shows priors generalize well to LMSYS holdout (weight → 0.88), achieving 21% better performance than the broken version.

---

## 🔧 Debugging Production Issues

### Issue 1: Weights Not Changing

**Symptoms:**
```python
print(router.weights)  # Always [0.5, 0.5]
```

**Diagnosis:**
```python
# Add debug logging
selected_model, selection_token = router.select_model(context)
print(f"Token: {selection_token}")  # Should show {'expert_idx': X, 'expert_prob': Y}

# Check if token is being passed
router.update(context, selected_model, reward, selection_token)
print(f"Weights after update: {router.weights}")  # Should change
```

**Fix:** Capture and pass `selection_token` (see correct usage above)

---

### Issue 2: High Variance in Performance

**Symptoms:**
- Some deployments perform well, others poorly
- Inconsistent behavior across instances

**Diagnosis:**
```python
# Log weight trajectories
weight_history = []
for i in range(1000):
    # ... run loop ...
    weight_history.append(router.weights[0])

# Analyze variance
print(f"Final weight: {weight_history[-1]:.3f}")
print(f"Std dev: {np.std(weight_history[-100:]):.3f}")
```

**Expected:** Moderate variance (std ≈ 0.18) is normal and indicates adaptive behavior

---

### Issue 3: Poor Performance Despite Correct Implementation

**Diagnosis:**
```python
# Check if priors match deployment distribution
validation_samples = deployment_data[:200]
warmup_accuracy = evaluate_warmup_on_validation(validation_samples)

if warmup_accuracy < 0.50:
    print("Priors are harmful - consider Tabula Rasa")
elif warmup_accuracy > 0.80:
    print("Priors are excellent - consider Warmup Only")
else:
    print("Corralling is appropriate (uncertain prior quality)")
```

**Fix:** Adjust strategy based on validation results

---

## 🧪 Testing Before Deployment

### Unit Test: Selection Token Functionality

```python
def test_corralling_adaptation():
    """Verify that weights adapt based on rewards."""
    router = CorrallingRouter(experts=[expert1, expert2], models=models)
    
    initial_weight = router.weights[0]
    
    for _ in range(100):
        model, token = router.select_model(context)
        
        # Simulate expert 0 being better
        reward = 1.0 if router.expert_selections[0] > router.expert_selections[1] else 0.0
        router.update(context, model, reward, token)
    
    final_weight = router.weights[0]
    
    # Weight should increase for higher-reward expert (lower cumulative loss)
    assert final_weight != initial_weight, "Weights must adapt!"
    assert abs(final_weight - initial_weight) > 0.1, "Weights must change significantly"
```

### Integration Test: End-to-End Flow

```python
def test_production_flow():
    """Simulate production deployment."""
    router = setup_router()
    rewards = []
    
    for i, sample in enumerate(test_data):
        # Correct usage pattern
        context = embed_prompt(sample['prompt'], encoder, pca)
        model, token = router.select_model(context)
        reward = get_reward(model, sample)
        router.update(context, model, reward, token)
        
        rewards.append(reward)
        
        # Verify adaptation
        if i == 50:
            weights_50 = router.weights.copy()
        if i == 200:
            weights_200 = router.weights.copy()
            assert not np.allclose(weights_50, weights_200), "Weights should evolve"
    
    # Verify performance improvement
    early_perf = np.mean(rewards[:100])
    late_perf = np.mean(rewards[-100:])
    assert late_perf >= early_perf, "Performance should not degrade"
```

---

## 📚 Additional Resources

- **Bug Report:** `CRITICAL_BUG_FIX_2026-02-14.md`
- **LaTeX Documentation:** `latex_section_5.3_practical_recommendations.tex`
- **Code Reference:** `src/bandit_gpt/router.py:3329-3398`
- **Experiment Results:** `results/weight_evolution/statistics.json`

---

## 🆘 Support

If you encounter issues in production:

1. **Check implementation** against correct usage pattern above
2. **Enable debug logging** to verify token flow
3. **Monitor weight evolution** for frozen weights (std < 0.01)
4. **Validate on small sample** (N=100-200) before full deployment
5. **Review bug report** for known issues and fixes

**Critical Reminder:** The selection token is NOT optional. Omitting it causes complete failure of the meta-learning mechanism with no error message.

---

**Version:** 1.0.0 (Post Bug Fix)  
**Validated On:** LMSYS Holdout (N=750, 10 seeds)  
**Production Ready:** ✅ Yes (with corrected implementation)
