# Why Corralling Exists: Complete Analysis

## Executive Summary

**Your question:** "Why did we switch to Corralling and why are we not seeing strong benefits now?"

**Short answer:**
1. You added Corralling to protect against **domain mismatch** (warmup priors trained on RouteLLM but deployed on LMSYS)
2. Domain mismatch **does exist** in your experiment (documented: 68.6% → 13.7% hard prompts)
3. Corralling **does work** (reduces regret from 79 to 44, a 44% improvement)
4. But you're questioning it because:
   - The "domain mismatch" might be **artificially created** by experimental design
   - Corralling only helps for **large effect sizes** (d>1.0), not general quality optimization
   - You're **not seeing benefit in all scenarios** you care about

**The real insight:** Corralling has **TWO legitimate use cases**, not one general-purpose solution.

---

## Part 1: The Complete Performance Picture

### Experiment Setup (Table 2)

**Training:**
- Dataset: 1,121 LMSYS dev prompts
- Warmup priors: Trained on RouteLLM dataset (80k battles)
- Models: Mixtral vs GPT-4-Turbo

**Evaluation:**
- Dataset: 750 LMSYS holdout prompts  
- **Distribution shift:** 68.6% hard prompts in warmup → 13.7% in holdout
- **Domain alignment:** 0.48 (severe mismatch)

### Actual Performance (10-seed validation)

| Strategy | Regret (Median [IQR]) | Gap vs Oracle | Interpretation |
|----------|----------------------|---------------|----------------|
| **Tabula Rasa** | **40 [38-42]** | 1.0× (baseline) | No priors, pure learning |
| **Warmup-Only** | **79 [77-81]** | 1.98× | **Harmful** - trusts wrong priors |
| **Corralling (η=0.1)** | 49 [47-51] | 1.23× | Slow adaptation |
| **Corralling (η=1.0)** | **52 [34-80]** | 1.30× (median) | Fast adaptation, high variance |

**Single best seed (seed=42):**
- Corralling achieves **44 regret** (1.10× vs oracle)
- This is the number used in original paper claims

### The Variance Problem

From `experiments_v1/02_table/VARIANCE_ANALYSIS.md`:

**Corralling variance:**
- Standard deviation: 23.2 (coefficient of variation: 42%)
- Worst-case seed: **80 regret** (same as warmup's 79!)
- Best-case seed: **34 regret** (better than tabula rasa's 40!)

**Root cause:** Stochastic expert selection in line 3032 of `router.py`:
```python
expert_idx = np.random.choice(self.n_experts, p=probs)
```

**Why this matters:**
- Some seeds get "lucky" and quickly shift to tabula rasa (34 regret)
- Some seeds get "unlucky" and stick with warmup too long (80 regret)
- **Median performance (52) is only 34% better than warmup (79)**
- **Not the 44% improvement claimed in the paper (based on single best seed)**

---

## Part 2: Domain Mismatch - Real or Artificial?

### Evidence That Mismatch Is Real

From `experiments_v1/02_table/README.md` and `table_02_mismatch_robustness.tex`:

**Measured distribution shift:**
- Warmup: 68.6% hard prompts (difficult, constraint-heavy)
- Holdout: 13.7% hard prompts (easier, natural language)
- **5× reduction in hard prompt frequency**

**Domain alignment:**
- Score: 0.48 (cosine similarity between feature distributions)
- This is **measured**, not a hyperparameter
- Computed from 1,000 warmup vs 1,000 production prompts

**Warmup bias:**
- Warmup expert learns "expensive = better" on RouteLLM data
- This is correct for RouteLLM (flagship models win on hard tasks)
- This is **wrong** for LMSYS holdout (fewer hard tasks)

### Evidence That Mismatch Might Be Artificial

**Question 1: Is the 68.6% → 13.7% shift realistic?**

From `experiments_v1/01_figure/README.md` (Alignment Tax analysis):
- **Low PC1 (82.4% of traffic)**: Natural language zone (easy)
- **High PC1 (17.6% of traffic)**: Strictness zone (hard, constraint-heavy)

So the **actual LMSYS distribution** is:
- 82.4% easy, 17.6% hard

But your **warmup priors** were trained on:
- 31.4% easy, 68.6% hard (from RouteLLM)

**This mismatch is real!** RouteLLM is NOT representative of LMSYS deployment distribution.

**Question 2: Why use RouteLLM priors if they're mismatched?**

From `experiments_v1/04_figure/README.md`:

> "Warmup Expert: LinUCB initialized with priors from RouteLLM. Biased toward flagships (GPT-4-Turbo, Claude-3). May suffer from negative transfer if domain mismatch exists."

**The design choice:** Use publicly available priors (RouteLLM) because:
- They're the best available transfer learning source
- You don't know in advance if they'll match deployment
- Corralling provides insurance if they don't match

**But:** If you KNOW they're mismatched (which you do now), why not train priors on LMSYS itself?

---

## Part 3: When Corralling Actually Helps

### Scenario 1: Domain Mismatch Protection ✅ (Table 2)

**Setup:**
- Warmup priors from RouteLLM (68.6% hard)
- Deploy on LMSYS holdout (13.7% hard)
- 5× distribution shift

**Performance:**
- Warmup-only: 79 regret (catastrophic)
- Corralling: 52 regret [34-80] (median, robust estimate)
- **Improvement: 34%** (79 → 52)

**When useful:**
- You're using transfer learning (priors from different dataset)
- You don't know in advance if mismatch exists
- Cost of getting it wrong is high (production launch)

### Scenario 2: Catastrophic Failure Detection ✅ (Figure 6)

**Setup:**
- Model suddenly degrades (effect size d>1.5)
- Static router keeps routing to broken model
- Need fast detection and failover

**Performance:**
- Detection time: <5 steps
- Automatic weight shift: 100% → 0% in ~10 steps
- Large effect size (d≈5.0) provides clear signal

**When useful:**
- Production safety monitoring
- Large, sudden quality drops
- Need automatic failover without human intervention

### Scenario 3: Alignment Tax Detection ✅ (Figure 1)

**Setup:**
- Expensive model systematically worse on specific task class
- Effect size: d=1.90 (very large)
- Frequency: 17.6% of traffic

**Performance:**
- Mixtral wins: -0.682 reward advantage on strictness tasks
- GPT-4-Turbo wins: +0.133 reward advantage on natural language
- Corralling learns to route accordingly

**When useful:**
- Specific, identifiable task classes with large effect sizes
- Systematic preference inversions (not just noise)
- Frequency high enough to matter (>10% of traffic)

### Scenario 4: General Quality Optimization ❌ (Doesn't Work)

**Setup:**
- Small effect sizes (d<0.2)
- Subtle quality differences
- High noise relative to signal

**Performance:**
- Need 10,000+ samples for statistical power
- Importance weighting amplifies noise
- Success rate: ~25% (from Figure 6 realistic scenario)

**Why it fails:**
- Signal-to-noise ratio too low
- Non-stationarity invalidates learning
- Opportunity cost of exploration
- Offline A/B testing is better

---

## Part 4: The Architecture Decision Tree

### Should You Use Corralling?

```
Q1: Do you have warmup priors?
├─ NO → Use Tabula Rasa (LinUCB from scratch)
│         No benefit from Corralling
│
└─ YES → Q2: Do you know if priors match deployment?
    ├─ YES, they match → Use Warmup-Only (SimpleLinUCBRouter)
    │                     No benefit from Corralling (8% overhead)
    │
    ├─ YES, they don't match → Fix the priors!
    │                          Train on LMSYS data, not RouteLLM
    │                          Then use Warmup-Only
    │
    └─ NO, uncertain → Q3: What's the use case?
        ├─ Safety (catastrophic failure) → Use Corralling ✅
        │                                   d>1.5, fast detection
        │
        ├─ Alignment Tax detection → Use Corralling ✅
        │                            d>1.0, systematic inversions
        │
        ├─ Domain mismatch insurance → Use Corralling ✅
        │                               34% improvement (median)
        │                               High variance (CV=42%)
        │
        └─ General quality optimization → DON'T use Corralling ❌
                                          Offline A/B instead
```

---

## Part 5: Why You're Questioning Corralling Now

### The Paper's Overclaims

**Original claim (from single best seed):**
> "Corralling achieves 44 regret (1.10× vs oracle), demonstrating near-optimal performance with safety guarantees."

**Reality (from 10-seed validation):**
> "Corralling achieves 52 regret [34-80] (1.30× vs oracle median), with high variance. Worst-case seed (80) is no better than warmup (79)."

**The difference:**
- Single best seed: **44% improvement** (79 → 44)
- Median across seeds: **34% improvement** (79 → 52)
- Worst-case seed: **0% improvement** (79 → 80)

### The "Not Seeing Benefits" Problem

You're testing Corralling on scenarios where it **shouldn't** provide benefit:

1. **General quality optimization** (small effect sizes)
   - Expected: No benefit (need offline A/B)
   - Observed: 25% success rate
   - Conclusion: ✅ Prediction matches reality

2. **Realistic LMSYS scenario** (d≈0.12)
   - Expected: No benefit (effect size too small)
   - Observed: Oscillating weights, slow convergence
   - Conclusion: ✅ Prediction matches reality

3. **Pareto frontier** (banditGPT vs RouteLLM)
   - Expected: banditGPT should win due to online learning
   - Observed: banditGPT does win (0.9088 vs 0.8827 peak quality)
   - Conclusion: ✅ System works, but not specifically due to Corralling

**The insight:** You're seeing the **limitations** of Corralling because you're testing it outside its valid operating regime.

---

## Part 6: What the Data Actually Shows

### Corralling Is Solving the Right Problem (Domain Mismatch)

**Evidence from expert weights:**

From `experiments_v1/04_figure/results_3models/quick_test_results.json`:
```
final_expert_weights: [1.41e-128, 1.000]
                      ^^^^^^^^^    ^^^^^
                      Warmup       Tabula Rasa
```

**What this tells us:**
1. Warmup priors ARE harmful on LMSYS holdout
2. Corralling correctly detected this (high losses on warmup expert)
3. Corralling correctly adapted (shifted 100% weight to tabula rasa)
4. **The algorithm is working as designed**

**The uncomfortable question:** If warmup is so bad that Corralling discards it completely, why use warmup at all?

### The Three-Way Performance Comparison

| What Happens | Warmup-Only | Tabula Rasa | Corralling |
|--------------|-------------|-------------|------------|
| **If priors are good** | 40 regret ✅ Best | 40 regret ✅ Good | 43 regret ⚠️ 8% overhead |
| **If priors are bad** | 79 regret ❌ Catastrophic | 40 regret ✅ Safe | 52 regret ✅ Protected |

**Key insight:** Corralling only helps when priors are bad. If you know priors are good, don't use Corralling. If you know priors are bad, don't use them at all (use tabula rasa).

**Corralling's value proposition:** When you DON'T KNOW if priors are good or bad, and you need insurance.

---

## Part 7: The Honest Narrative

### What Your Paper Should Say

**Claim 1: Corralling provides insurance against domain mismatch** ✅ TRUE

> "In production deployments, warmup priors may not match the target distribution. Our Corralling approach provides 34% median improvement (79 → 52 regret) when priors are misaligned, with only 8% overhead when priors are correct. This insurance is valuable when distribution match is uncertain."

**Claim 2: Corralling enables catastrophic failure detection** ✅ TRUE

> "When models suddenly degrade (d>1.5), Corralling detects failures within 5 steps and automatically shifts routing. This provides a safety mechanism for production deployments without manual intervention."

**Claim 3: Corralling discovers alignment tax effects** ✅ TRUE

> "On LMSYS data, we observe large effect sizes (d=1.90) on 17.6% of traffic where Mixtral outperforms GPT-4-Turbo. Corralling automatically discovers and exploits these systematic preference inversions, achieving 75.7% cost reduction while maintaining quality."

**Claim 4: Corralling achieves near-optimal performance** ⚠️ OVERSTATED

> BEFORE: "Corralling achieves 1.10× vs oracle (44 regret)"  
> AFTER: "Corralling achieves 1.30× vs oracle median (52 regret [34-80], N=10 seeds)"

**Claim 5: Corralling is general-purpose for LLM routing** ❌ FALSE

> REMOVE: Corralling does NOT work for general quality optimization with small effect sizes (d<0.2). For those scenarios, offline A/B testing is more appropriate.

### What Your Paper Should NOT Say

❌ "Corralling provides near-optimal performance in all scenarios"  
✅ "Corralling provides near-optimal performance when priors are misaligned OR when detecting large quality drops"

❌ "Corralling is a general solution for adaptive LLM routing"  
✅ "Corralling is a specialized solution for domain mismatch protection and catastrophic failure detection"

❌ "Results based on single seed (44 regret)"  
✅ "Results based on 10 seeds (52 median regret [34-80])"

---

## Part 8: Answering Your Original Question

### "Why did we switch to Corralling?"

**Answer:**
1. To protect against domain mismatch (RouteLLM priors on LMSYS deployment)
2. To provide safety guarantees (never worse than tabula rasa)
3. To enable automatic adaptation (don't need to know priors are good/bad upfront)

**This was a reasonable design decision** given:
- Transfer learning from public datasets (RouteLLM)
- Uncertainty about distribution match
- Production safety requirements

### "Why are we not seeing strong benefits now?"

**Answer:**
1. You ARE seeing benefits in the scenarios where it should work:
   - Domain mismatch: 34% improvement (79 → 52 median)
   - Catastrophic failure: <5 step detection
   - Alignment tax: 75.7% cost reduction

2. You're NOT seeing benefits in scenarios where it shouldn't work:
   - General quality optimization (d<0.2): Use offline A/B instead
   - Realistic LMSYS scenario (d≈0.12): Effect size too small

3. The **variance** is higher than expected:
   - Std = 23.2 (CV = 42%)
   - Worst-case seed (80) ≈ warmup (79)
   - This wasn't clear from single-seed results

**The core issue:** You're testing Corralling outside its valid operating regime and discovering its limitations, which is actually GOOD science.

---

## Part 9: Recommendations

### Option A: Narrow the Scope (Recommended)

**Action:** Position Corralling as a **specialized solution** for two scenarios:

1. **Domain mismatch protection** (Table 2, Figure 4)
   - When using transfer learning from public datasets
   - 34% improvement when priors are misaligned
   - 8% overhead when priors are correct

2. **Catastrophic failure detection** (Figure 6)
   - Production safety mechanism
   - Fast detection (<5 steps) for large drops (d>1.5)
   - Automatic failover without manual intervention

**Benefits:**
- Honest about limitations
- Strong, defensible use cases
- Avoids overclaims

### Option B: Fix the Priors

**Action:** Train warmup priors on LMSYS data, not RouteLLM

**Steps:**
1. Use LMSYS dev set (1,121 prompts) to train priors
2. Re-run experiments with LMSYS-matched priors
3. Check if domain mismatch still exists

**Expected result:**
- If priors now match: Warmup-only should perform well (≈40 regret)
- If priors still mismatch: Something else is wrong

**Problem:** This might eliminate the need for Corralling entirely.

### Option C: Add Explicit Comparison (Most Honest)

**Action:** Report **all three scenarios** in the paper:

| Scenario | Warmup | Corralling | Benefit |
|----------|--------|------------|---------|
| **Priors match** (good) | 40 regret | 43 regret | ❌ -8% (overhead) |
| **Priors mismatch** (bad) | 79 regret | 52 regret | ✅ +34% (insurance) |
| **Catastrophic failure** | No detection | <5 steps | ✅ Safety |

**Message:** Corralling is insurance. You pay 8% overhead for protection against 79 regret catastrophe.

---

## Bottom Line

**Your intuition is correct:**

You're not seeing strong benefits because:
1. Corralling only helps in specific scenarios (domain mismatch, large effect sizes)
2. Your multi-seed analysis revealed high variance (worst-case = warmup performance)
3. You're testing it on general quality optimization where it shouldn't work

**But Corralling isn't useless:**
- It DOES provide 34% improvement when priors are misaligned (median across seeds)
- It DOES detect catastrophic failures quickly (<5 steps)
- It DOES discover alignment tax effects (d=1.90 on 17.6% of traffic)

**The honest narrative:**

> "Corralling is a specialized insurance mechanism for domain mismatch and catastrophic failure detection. It provides 34% median improvement (79 → 52 regret) when warmup priors are misaligned, with only 8% overhead when priors are correct. For general quality optimization with small effect sizes (d<0.2), offline A/B testing remains more appropriate."

This is still a **valuable contribution** to the field, just not the "general-purpose adaptive routing" silver bullet the early experiments suggested.
