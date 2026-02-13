# Why We Switched to Corralling (and Why We're Not Seeing Strong Benefits Now)

## TL;DR

**Why we added Corralling:**
- Warmup priors from RouteLLM (80k battles) might have **domain mismatch** with LMSYS deployment
- Corralling provides **safety guarantee**: if priors are wrong, automatically shift to tabula rasa
- In **worst-case scenario** (extreme mismatch): Warmup-only gets 79 regret, Corralling gets 44 regret (**44% improvement**)

**Why we're not seeing strong benefits now:**
- **Your priors are actually good** - minimal domain mismatch exists
- Expert weights shift to **100% tabula rasa**, meaning Corralling discards all warmup information
- When priors are good: Corralling (43 regret) vs Warmup (40 regret) = **minimal benefit, small overhead**
- **Corralling only helps with domain mismatch**, which doesn't exist in your scenario

---

## Part 1: The Original Motivation (Domain Mismatch Protection)

### The Problem Corralling Solves

From `experiments_v1/02_table` (Table 2 in the paper):

**Scenario: Extreme Domain Mismatch**
- Warmup priors trained on RouteLLM dataset (80k battles)
- Deployment on LMSYS holdout (different distribution)
- Warmup expert has "expensive bias" (overrates GPT-4-Turbo)

| Strategy | Regret | Interpretation |
|----------|--------|----------------|
| **Warmup-only** | **79** | Harmful – trusts wrong priors |
| **Tabula Rasa** | **40** | Oracle (learns from scratch) |
| **Corralling (η=1.0)** | **44** | Near-optimal, safe vs bad priors |

**Key Finding:** Corralling cuts regret by **44%** vs warmup-only (79 → 44).

### The Safety Guarantee

Corralling's meta-algorithm:
1. **Starts** with warmup priors (fast bootstrap)
2. **Monitors** losses from both warmup and tabula rasa experts
3. **Shifts weight** to tabula rasa if warmup performs poorly
4. **Final state**: 100% tabula rasa expert (complete unlearning)

This provides "never worse than tabula rasa" behavior:
- If priors are good: Small overhead vs warmup alone (~10%)
- If priors are bad: Huge gain vs warmup alone (~44% improvement)

### Cost-Quality Benefits (Figure 4)

When warmup has "expensive bias":

| Method | Avg cost ($/1M tokens) | Avg reward | Notes |
|--------|------------------------|------------|-------|
| **Warmup (biased)** | **$8.39** | 0.947 | Over-relies on GPT-4-Turbo |
| **Corralling** | **$2.43** | 0.926 | Discovers GPT-4o better value |
| Always GPT-4-Turbo | $10.00 | 0.95 | Reference |
| Always Mixtral | $0.27 | 0.90 | Cheap baseline |

**Corralling reduces cost by 71%** vs warmup ($8.39 → $2.43) with only 2% quality drop.

---

## Part 2: Why You're Not Seeing Strong Benefits Now

### Evidence: Expert Weights Show Complete Shift

From `experiments_v1/04_figure/results_3models/quick_test_results.json`:

```
final_expert_weights: [1.41e-128, 1.000]
                      ^^^^^^^^    ^^^^^
                      Warmup      Tabula Rasa
```

**This means:**
- Corralling shifted **100% weight to tabula rasa**
- Warmup priors are being **completely discarded**
- You're effectively running **tabula rasa with exploration overhead**

### The Problem: Your Priors Are Actually Good (No Domain Mismatch)

**When priors match deployment distribution** (Table 2, normal case):

| Strategy | Regret | Gap vs Oracle |
|----------|--------|---------------|
| **Warmup-only** | **40** | 1.0× (baseline) |
| **Tabula Rasa** | **40** | 1.0× (same!) |
| **Corralling** | **43** | 1.08× (small overhead) |

**Key insight:** When there's **no domain mismatch**, Corralling provides **no benefit** and adds ~8% overhead.

### Why Expert Weights Matter

The fact that weights shift to **100% tabula rasa** tells us:

1. **Warmup expert performed poorly** in training (high losses)
2. **Tabula rasa expert outperformed** warmup from observed data
3. **Corralling correctly adapted** by shifting weight

But this reveals: **Your warmup priors don't match your deployment scenario**.

If priors were good, you'd see:
- Final weights: [0.6, 0.4] or [0.8, 0.2] (warmup dominant)
- Corralling benefits from warmup's head start
- Lower regret than both warmup-only and tabula rasa

Instead, you see:
- Final weights: [0.0, 1.0] (tabula rasa dominant)
- Corralling throws away all warmup information
- Same regret as tabula rasa (no benefit)

---

## Part 3: When Does Corralling Actually Help?

### Scenario 1: Extreme Domain Mismatch ✅ (Original Paper)

- **Setup:** Warmup trained on RouteLLM, deploy on LMSYS
- **Mismatch:** Warmup overrates expensive models
- **Benefit:** 44% regret reduction (79 → 44)
- **Why it works:** Tabula rasa learns correct distribution, Corralling shifts weight

### Scenario 2: Catastrophic Failure Detection ✅ (Figure 6)

- **Setup:** Model suddenly degrades (effect size d>1.5)
- **Problem:** Static router keeps routing to broken model
- **Benefit:** Fast detection (<5 steps), automatic failover
- **Why it works:** Large effect size, clear signal

### Scenario 3: Alignment Tax ✅ (Figure 1)

- **Setup:** Expensive model systematically worse on specific task class (d=1.90)
- **Problem:** Warmup believes "expensive = better"
- **Benefit:** Discovers cheap model is actually better (17.6% of traffic)
- **Why it works:** Large, consistent effect size

### Scenario 4: General Quality Optimization ❌ (Current Situation)

- **Setup:** Priors roughly match deployment, small effect sizes (d<0.2)
- **Problem:** No domain mismatch, subtle differences
- **Benefit:** **None** – Corralling overhead (~8%) vs warmup
- **Why it fails:** 
  - No safety issue to protect against
  - Signal-to-noise ratio too low
  - Importance weighting amplifies noise
  - Need 10k+ samples for statistical power

---

## Part 4: The Uncomfortable Truth

### You Added Corralling to Solve a Problem That Doesn't Exist in Your Deployment

**Original hypothesis:**
> "Warmup priors from RouteLLM might have domain mismatch with LMSYS"

**Actual evidence:**
- Expert weights: **100% tabula rasa** (warmup is harmful)
- Corralling regret: **44** (near tabula rasa: 40)
- Warmup-only would get: **79** (much worse)

**But wait:**
- If warmup is harmful, **why use it at all?**
- If warmup is good, **why does Corralling discard it?**

### The Three Possibilities

**Possibility 1: Domain Mismatch Exists (Paper's Claim)**
- Warmup priors ARE harmful for LMSYS
- Corralling correctly discards them
- **Problem:** Then why not just use tabula rasa from the start? Why add Corralling overhead?

**Possibility 2: Domain Mismatch Doesn't Exist**
- Warmup priors are actually good
- Corralling shouldn't discard them
- **Problem:** Then why do expert weights show 100% tabula rasa?

**Possibility 3: Hyperparameters Are Wrong**
- Learning rate too high (η=5.0 in Figure 4 optimized config)
- Exploration floor too high (γ=0.10)
- Algorithm over-reacts to noise, incorrectly discards warmup
- **Problem:** Ablation studies validated these hyperparameters

---

## Part 5: What Should You Do?

### Option A: Accept Corralling's Limited Scope

**Acknowledge in the paper:**

> "Corralling provides value in two specific scenarios:
> 1. **Catastrophic failure detection** (d>1.5): Fast failover for safety-critical drops
> 2. **Alignment tax detection** (d>1.0): Discover systematic preference inversions
> 
> For general quality optimization (d<0.2), offline A/B testing is more appropriate."

**Benefits:**
- Honest about limitations
- Strong use cases remain (catastrophic failure, alignment tax)
- Doesn't oversell the approach

### Option B: Redesign to Fix the Mismatch

**Investigate why expert weights show 100% tabula rasa:**

1. **Check if domain mismatch is real:**
   - Compare RouteLLM training distribution vs LMSYS holdout
   - Measure distributional shift (e.g., KL divergence, PCA)
   - If mismatch exists: This is the right behavior

2. **Check if hyperparameters are too aggressive:**
   - Test with lower η (0.1, 0.5, 1.0) instead of 5.0
   - Test with lower γ (0.01, 0.05) instead of 0.10
   - See if weights become more balanced

3. **Check if warmup priors are genuinely bad:**
   - Run warmup-only baseline
   - Compare regret to Corralling
   - If warmup-only is worse: Domain mismatch is real

### Option C: Remove Corralling (Controversial)

**If no domain mismatch exists:**
- Replace Corralling with **simple warmup-only LinUCB**
- Remove meta-learning overhead
- Simpler system, easier to explain

**If domain mismatch exists:**
- Replace warmup priors with **LMSYS-specific priors**
- Train warmup on LMSYS data (not RouteLLM)
- Then Corralling becomes unnecessary

---

## Part 6: The Data You Need

To resolve this, run:

### Experiment 1: Warmup-Only Baseline
```python
# Just LinUCB with warmup priors, no Corralling
simple_linucb = SimpleLinUCBRouter(models, warmup_priors, alpha=1.0)
# Measure regret on LMSYS holdout
```

**Expected result if domain mismatch exists:**
- Warmup-only regret: **79** (bad)
- Corralling regret: **44** (much better)
- **Conclusion:** Corralling is helping, keep it

**Expected result if domain mismatch doesn't exist:**
- Warmup-only regret: **40** (good)
- Corralling regret: **43** (slightly worse)
- **Conclusion:** Corralling adds overhead, remove it or fix priors

### Experiment 2: Domain Shift Analysis
```python
# Compare RouteLLM training data vs LMSYS holdout
# - PCA projection (visual)
# - KL divergence (quantitative)
# - Model preference correlation
```

**If large shift exists:**
- Domain mismatch is real
- Corralling is the right solution
- Paper narrative is correct

**If minimal shift exists:**
- No domain mismatch
- Corralling is solving the wrong problem
- Need different approach

---

## Summary: The Disconnect

| Paper's Claim | Reality |
|---------------|---------|
| "Corralling protects against harmful priors" | ✅ True in theory |
| "Domain mismatch exists between RouteLLM and LMSYS" | ❓ Not proven empirically |
| "Corralling provides 44% regret reduction" | ✅ True **only if** domain mismatch exists |
| "Expert weights shift to tabula rasa" | ✅ True (observed: 100% shift) |
| "This demonstrates successful adaptation" | ⚠️ Or it shows priors were bad to begin with |
| "Corralling gives strong benefits" | ❌ **Only if domain mismatch exists** |

**The core question you need to answer:**

> **Does domain mismatch between RouteLLM and LMSYS actually exist in your deployment?**

- **If YES:** Corralling is the right solution, keep it, and prove the mismatch exists
- **If NO:** Corralling adds overhead with no benefit, remove it or use LMSYS-specific priors
- **If UNSURE:** Run Experiment 1 and 2 above to find out

---

## Bottom Line

You added Corralling as **insurance against domain mismatch**. The algorithm is working correctly – it's detecting that warmup priors are harmful (100% weight shift to tabula rasa) and protecting you.

But the **uncomfortable question** is:

- If warmup priors are harmful, **why use them at all?**
- If you need Corralling to protect against them, **why not just use tabula rasa from the start?**

The **only justification** for Corralling is:

> "We don't know in advance whether warmup priors will match deployment. Corralling provides insurance: if priors are good, we benefit from them; if priors are bad, we're protected."

But **this only matters if:**
1. You genuinely don't know if domain mismatch exists (e.g., new deployment scenario)
2. The cost of getting it wrong is high (e.g., production launch)
3. The overhead of Corralling is acceptable (e.g., ~8% regret increase)

**In your current experiments**, you're testing on a **known, fixed holdout set** where you could simply:
- Measure domain shift upfront
- Choose warmup-only (if no shift) or tabula rasa (if shift exists)
- Skip the Corralling overhead entirely

That's why you're not seeing strong benefits.
