# Experimental Design Choices: Figure 5

## Key Decision: Quality-Only Mode (cost_penalty=0.0)

### What We Did

Both experts configured with **zero cost penalty**:

```python
warmup_expert = CostAwareLinUCBRouter(..., cost_penalty=0.0)
tabula_rasa_expert = CostAwareTabulaRasaRouter(..., cost_penalty=0.0)
```

### Why This Matters

This isolates **pure prior misalignment** (wrong quality predictions) from **objective mismatch** (wrong cost-quality trade-offs).

### Scientific Justification

| Aspect | With cost_penalty=0.0 | With cost_penalty>0 |
|--------|----------------------|---------------------|
| **What drives loss** | Prediction error only | Prediction error + cost efficiency |
| **Decommissioning cause** | "Prior wrong about quality" | "Prior wrong about quality OR cost trade-off" |
| **Interpretation** | Clean: domain shift | Confounded: multiple causes |
| **Fair comparison** | ✅ Same objective | ❌ Different objectives possible |

## The Three Variables We Control

### 1. Expert Initialization (The Comparison)

| Expert | Initialization | Confidence | Belief Source |
|--------|---------------|------------|---------------|
| **Warmup** | 80k RouteLLM battles | High (large A) | Historical data |
| **Tabula Rasa** | Identity matrices | Low (A=λI) | None (cold start) |

**What we're testing**: Does historical knowledge transfer to new domain?

### 2. Cost Penalty (The Objective)

| Setting | Optimizes For | Use Case |
|---------|--------------|----------|
| **0.0 (default)** | Pure quality | Isolate prediction error |
| **0.0 vs 0.5** | Different objectives | Show objective mismatch matters |
| **0.5 (both)** | Cost efficiency | Show cost-aware learning |

**What we're testing**: Whether experts optimize the same goal.

### 3. Learning Rate (The Adaptation Speed)

| η | Adaptation | Stability | Use Case |
|---|-----------|-----------|----------|
| **0.1** | Slow | High | Long horizon, noisy data |
| **1.0 (default)** | Fast | Medium | Standard setting |
| **5.0** | Aggressive | Low | Short horizon, clear signal |

**What we're testing**: How quickly to downweight bad expert.

## Design Choice Rationale

### Why cost_penalty=0.0 is the Right Default

**Scenario A: Quality Inversion (What We Expect)**

Suppose warmup was trained on hard reasoning tasks:
```
P_train: GPT-4 (0.90) > Mixtral (0.70)
```

But production is simple chat:
```
P_prod: Mixtral (0.85) > GPT-4 (0.75)
```

**With cost_penalty=0.0**:
- Warmup picks GPT-4 → Loss = 0.25
- Tabula picks Mixtral → Loss = 0.15
- **Clear signal**: Warmup has wrong quality beliefs
- **Result**: Decommissioning happens cleanly

**With cost_penalty=0.5** (contaminated experiment):
- Warmup picks GPT-4 → Loss = 0.25 + (0.5 × high_cost)
- Tabula picks Mixtral → Loss = 0.15 + (0.5 × low_cost)
- **Confounded signal**: Is loss from wrong quality OR wrong cost sensitivity?
- **Result**: Can't tell if prior is wrong about quality or just cost-blind

### Why Equal Cost Penalties is Critical

**Scenario B: Asymmetric Cost Penalties (Broken Comparison)**

```python
warmup_expert = CostAwareLinUCBRouter(..., cost_penalty=0.0)
tabula_rasa = CostAwareTabulaRasaRouter(..., cost_penalty=0.5)
```

**Problem**: Different objectives!
- Warmup optimizes: max E[quality]
- Tabula Rasa optimizes: max E[quality - 0.5·cost]

**Result**: Even if warmup has PERFECT quality predictions, it might lose because it's cost-blind. This doesn't demonstrate "prior misalignment" - it demonstrates "they're playing different games."

## Follow-Up Experiments

### Experiment 1: Cost Sensitivity Misalignment

**Goal**: Show decommissioning can happen from objective mismatch

**Setup**:
```python
warmup = CostAwareLinUCBRouter(..., cost_penalty=0.0)   # Cost-blind
tabula = CostAwareTabulaRasaRouter(..., cost_penalty=0.5)  # Cost-aware
```

**Expected**: If both predict quality equally well, but TR discovers cost savings, warmup gets decommissioned despite correct predictions.

**Interpretation**: Demonstrates decommissioning from **utility function mismatch**, not quality prediction error.

### Experiment 2: Learning Rate Sweep

**Goal**: Validate adaptation speed theory

**Setup**: Run with η ∈ {0.1, 0.5, 1.0, 2.0, 5.0}

**Expected**: Time to 90% weight scales as O(1/η)

**Interpretation**: Confirms exponential decay theory.

### Experiment 3: Prior Strength Ablation

**Goal**: Test robustness to overconfident priors

**Setup**: Scale warmup priors by factors {0.1, 0.5, 1.0, 2.0, 10.0}

**Expected**: Stronger priors (larger scale) → slower decommissioning

**Interpretation**: Highly confident wrong beliefs take longer to overturn.

## Common Misunderstandings

### ❌ "Cost penalty should be high to show real-world relevance"

**Counter**: Cost-only experiments confound variables. To isolate prior quality beliefs, use cost_penalty=0. Then run separate experiment with cost_penalty>0 to show cost sensitivity matters.

### ❌ "Asymmetric cost penalties show which approach is better"

**Counter**: Asymmetric penalties mean different objectives. Not a fair comparison. Use symmetric penalties to isolate initialization (warmup vs cold start).

### ❌ "If no decommissioning, the experiment failed"

**Counter**: No decommissioning means prior was CORRECT for your domain! This is a valid result showing warmup effectiveness. The safety mechanism is that decommissioning happens when NEEDED.

## Validation Checklist

When running the experiment, verify:

- [ ] Both experts have **same cost_penalty** (usually 0.0)
- [ ] Both experts optimize **same objective** (quality-only or cost-aware)
- [ ] Only difference is **initialization** (warmup vs cold start)
- [ ] Learning rate η ∈ [0.1, 5.0] (reasonable range)
- [ ] At least 200 samples (enough for signal)

## Interpretation Guide

### Result 1: Warmup Weight <20% by t=500

**Interpretation**: ✅ Prior was misspecified (domain shift detected)

**Cause**: Warmup trained on different distribution than production

**Action**: Use Corralling in production (safety mechanism works)

### Result 2: Warmup Weight >80% by t=500

**Interpretation**: ✅ Prior was correct (warmup validated)

**Cause**: Historical data matches production distribution

**Action**: Can skip Corralling overhead (warmup alone is sufficient)

### Result 3: Both Weights 40-60% by t=500

**Interpretation**: ⚖️ Both experts contribute value

**Cause**: Warmup partially correct, TR adds complementary information

**Action**: Keep Corralling (optimal mixing achieved)

## Summary: The Design Philosophy

**Core Principle**: Control for confounds by isolating ONE variable at a time.

**Variable 1 (Primary)**: Initialization
- Warmup: Strong priors from history
- Tabula Rasa: No priors (cold start)
- **Test**: Does historical knowledge transfer?

**Variable 2 (Controlled)**: Objective
- Both: cost_penalty=0.0 (quality-only)
- **Reason**: Ensure fair comparison (same goal)

**Variable 3 (Tunable)**: Adaptation Speed
- Learning rate η (default 1.0)
- **Effect**: Controls how fast to downweight bad expert

By setting cost_penalty=0.0, we get a **clean experiment** where:
- Decommissioning = "Prior has wrong quality beliefs"
- No decommissioning = "Prior has correct quality beliefs"

Any other configuration risks confounding these interpretations.

## References for Reviewers

**Why cost_penalty=0 is scientifically sound**:

1. **Ablation Study Best Practice**: Change one variable at a time
   - Vary initialization (warmup vs cold), hold objective constant
   - Separate experiment for cost sensitivity analysis

2. **Causal Inference**: Isolate the treatment effect
   - Treatment: Using warmup priors vs cold start
   - Outcome: Cumulative loss over time
   - Confounder: Different cost sensitivities → Must control

3. **Theory Validation**: Test the mathematical claim
   - Claim: Exponential weights adapt to best expert
   - Test: Both experts optimize quality, differ only in initialization
   - Result: Clean validation of adaptation mechanism

**Bottom line**: cost_penalty=0.0 is a deliberate design choice for experimental clarity, not an oversight.

