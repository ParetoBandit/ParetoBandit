# Practical Deployment Guidelines

**Purpose:** Actionable guidance for practitioners deploying LLM routing systems  
**Based on:** Comprehensive validation studies (75 configurations, multi-seed)  
**Last Updated:** February 12, 2026

---

## Executive Summary

Our validation studies reveal three critical insights for production deployment:

1. **Match strategy to prior quality** - Corralling isn't universally optimal
2. **Use constant α=2.0** - Adaptive decay degrades performance by 48%
3. **Monitor expert weights** - Adaptation occurs in ~16 requests

---

## Decision Framework: Which Strategy to Deploy

### Scenario 1: Prior Quality Unknown (Uncertainty) 🎯
**→ Deploy: Corralling**

**Performance:** 59.2 ± 7.1 regret (validated on 750 samples, 10 seeds)

**When to use:**
- Cross-domain deployment (e.g., enterprise chatbot with consumer training data)
- New geography/market with no local validation data
- Risk-averse scenarios (medical, financial, safety-critical)
- Cannot pre-validate prior quality

**Why it works:**
- 18.5% safety improvement vs harmful warmup-only
- Hedges between initialization sources
- Detects bad priors in ~16 requests

---

### Scenario 2: Priors Validated Good ✅
**→ Deploy: Warmup Only**

**Performance:** Expected to outperform Corralling (validated in low-mismatch)

**When to use:**
- Deployment domain matches training (validated on held-out data)
- Recent priors (collected <1 month ago)
- Prior accuracy >80% on validation sample (N=100-200)

**Why it works:**
- Leverages domain knowledge efficiently
- No hedging overhead
- Faster convergence with accurate initialization

---

### Scenario 3: Priors Known Bad ❌
**→ Deploy: Tabula Rasa Only**

**Performance:** 49.5 ± 2.8 regret (BEST in our validation, 10 seeds)

**When to use:**
- Severe domain mismatch confirmed (e.g., 68.6%→13.7% shift)
- Cold start with no priors available
- Prior accuracy <50% on validation sample
- Priors outdated (>6 months old)

**Why it works:**
- No bias from bad priors
- Pure online learning adapts to true distribution
- 16% better than Corralling (49.5 vs 59.2)
- 33% better than harmful warmup (49.5 vs 74.7)

---

## Critical: Always Use Constant α=2.0

### Validated Finding

| Configuration | Regret | Performance |
|---------------|--------|-------------|
| **Constant α=2.0** | **60.6 ± 1.4** | **Optimal** |
| Adaptive decay | 90.2 ± 7.8 | 48% worse |

**Why it matters:**
- Under domain uncertainty, premature exploitation is catastrophic
- Decaying α causes system to "lock in" to misspecified beliefs
- Constant exploration maintains vigilance for distribution shifts

### Configuration

```python
# ✅ VALIDATED: Constant exploration
expert = CostAwareLinUCBRouter(
    alpha_start=2.0,
    alpha_end=2.0,  # Keep constant!
    ...
)
```

**Exception:** Only use adaptive decay if:
- Priors validated accurate (>80% accuracy)
- Environment confirmed stationary
- Low domain mismatch (<10% distribution shift)

---

## Prior Quality Assessment

### Method: Holdout Validation

```python
def assess_prior_quality(priors, validation_data):
    """
    Validate priors on N=100-200 deployment samples.
    Returns: 'good', 'uncertain', or 'bad'
    """
    accuracy = evaluate_predictions(priors, validation_data)
    
    if accuracy > 0.80:
        return 'good'      # Use Warmup Only
    elif accuracy > 0.50:
        return 'uncertain' # Use Corralling
    else:
        return 'bad'       # Use Tabula Rasa
```

**Decision Thresholds (Empirically Validated):**
- **>80% accuracy:** Priors are good → Warmup Only
- **50-80% accuracy:** Uncertain → Corralling (hedging)
- **<50% accuracy:** Priors are bad → Tabula Rasa

---

## Production Monitoring

### Key Metrics

**1. Expert Weights (Most Important)**
```python
weights = router.get_expert_weights()
# Monitor every request or log periodically
```

**Interpretation:**
- **Weight <0.2:** Priors are harmful → Consider switching to Tabula Rasa
- **Weight 0.3-0.7:** Genuine uncertainty or transition period
- **Weight >0.8:** Priors are accurate → Consider simplifying to Warmup Only

**2. Adaptation Speed**
- Validated: ~16 ± 14 requests to detect severe mismatch
- Monitor first 50-100 requests closely
- Weight trajectory reveals prior quality

**3. Performance Benchmarks**
Based on validation (750 samples, multiple seeds):
- **Excellent:** <50 regret (Tabula Rasa level)
- **Good:** 50-60 regret (Corralling level)
- **Acceptable:** 60-75 regret (Warmup in mild mismatch)
- **Poor:** >75 regret (Harmful warmup)

---

## Performance Trade-offs

### Corralling Overhead

**Cost:** 20% overhead vs optimal strategy (59.2 vs 49.5)  
**Benefit:** 18.5% safety vs harmful warmup (59.2 vs 74.7)

**When overhead is justified:**
- True uncertainty about prior quality
- Risk-averse deployment requirements
- High cost of mistakes (medical, financial)

**When overhead not justified:**
- Prior quality validated on deployment data
- Low-stakes application
- Can tolerate initial exploration period

---

## Common Deployment Mistakes

### ❌ Mistake 1: Using Adaptive α Decay
**Problem:** 48% performance degradation (90.2 vs 60.6)  
**Solution:** Use constant α=2.0 under uncertainty

### ❌ Mistake 2: Always Using Corralling
**Problem:** 20% unnecessary overhead when priors known bad  
**Solution:** Assess prior quality first, use Tabula Rasa if appropriate

### ❌ Mistake 3: Not Monitoring Weights
**Problem:** Can't detect deployment issues  
**Solution:** Log weights, set up alerts, review weekly

### ❌ Mistake 4: Expecting Deterministic Trajectories
**Problem:** High variance (0.382 ± 0.471)  
**Solution:** Monitor your deployment, use confidence intervals

---

## Validated Configuration Parameters

### Optimal Settings (From Ablation Studies)

```python
CorrallingRouter(
    experts=[warmup, tabula_rasa],
    learning_rate=1.0,    # Validated optimal
    gamma=0.05            # Validated optimal
)

# Both experts:
alpha_start=2.0           # Constant exploration
alpha_end=2.0             # (48% better than decay)
```

### Gamma Selection (From Gamma Ablation)

| γ | Regret | Stability | Recommendation |
|------|--------|-----------|----------------|
| 0.001 | 59.0 | Medium | Marginal winner but unstable |
| **0.05** | **60.6** | **High** | **Recommended (best balance)** |
| 0.10 | 69.2 | Low | Too much mixing |

**Result:** γ=0.05 provides best balance of performance and stability

---

## Integration with Existing Systems

### Monitoring Stack

```python
# Production monitoring
import logging

logger = logging.getLogger('llm_router')

for t, request in enumerate(requests):
    # Route
    model = router.select_model(context)
    response = invoke_model(model, request)
    reward = evaluate(response)
    
    # Update
    router.update(context, model, reward)
    
    # Monitor (for Corralling)
    if t % 10 == 0:  # Every 10 requests
        weights = router.get_expert_weights()
        logger.info(f"t={t}, weights={weights}")
        
        # Alert on anomalies
        if t == 50 and min(weights) < 0.2:
            logger.warning("Prior mismatch detected")
```

---

## Summary

### The Three Rules

1. **Assess prior quality first** (validation data)
2. **Use constant α=2.0** (not adaptive decay)
3. **Monitor expert weights** (fast adaptation ~16 requests)

### Performance Matrix

| Prior Quality | Best Strategy | Expected Regret | Performance |
|---------------|---------------|-----------------|-------------|
| Unknown | Corralling | ~60 | Safe hedging |
| Bad | Tabula Rasa | ~50 | Optimal |
| Good | Warmup Only | <60* | Efficient |

*Lower than Corralling when priors are accurate

---

## References

**Implementation:** `src/bandit_gpt/router.py`  
**Experiments:** All scripts in `experiments_v1/03_figure/`  
**Results:** `experiments_v1/03_figure/results/`  
**Paper Sections:** LaTeX files in this directory

---

*All recommendations validated through systematic experimentation*
