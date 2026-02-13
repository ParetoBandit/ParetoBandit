# Corralling Reality Check: What the Data Actually Shows

## TL;DR

**Q: Why did we switch to Corralling?**  
**A:** To protect against domain mismatch (RouteLLM priors → LMSYS deployment). Warmup-only gets 79 regret, Corralling gets 52 regret (34% improvement).

**Q: Why are we not seeing strong benefits now?**  
**A:** Because:
1. **High variance**: Worst seed (80) ≈ warmup (79), best seed (34) << tabula rasa (40)
2. **Wrong regime**: Testing on small effect sizes (d<0.2) where Corralling shouldn't work
3. **Overclaimed initially**: Single best seed (44 regret) → median across 10 seeds (52 regret)

---

## The Performance Reality

### Original Paper Claim (Single Seed)

> "Corralling achieves **44 regret** (1.10× vs oracle), demonstrating near-optimal performance."

**Based on:** seed=42, single lucky run

### Multi-Seed Reality

| Strategy | Regret | Gap vs Oracle |
|----------|--------|---------------|
| Tabula Rasa (no priors) | 40 [38-42] | 1.0× (baseline) |
| **Warmup-Only (harmful)** | **79 [77-81]** | 1.98× |
| Corralling (η=1.0) | **52 [34-80]** | 1.30× |

**Based on:** 10 random seeds, median [IQR]

### The Variance Problem

- **Best seed:** 34 regret (better than tabula rasa!)
- **Median seed:** 52 regret (34% better than warmup)
- **Worst seed:** 80 regret (NO better than warmup!)
- **Standard deviation:** 23.2 (CV = 42%)

**Root cause:** Stochastic expert selection creates lucky/unlucky runs

**Implication:** Corralling's benefit is **NOT guaranteed**, it's probabilistic

---

## Where Corralling Actually Works

### ✅ Scenario 1: Domain Mismatch Protection (Table 2, Figure 4)

**Setup:**
- Warmup priors from RouteLLM (68.6% hard prompts)
- Deploy on LMSYS (13.7% hard prompts)
- 5× distribution shift is REAL and MEASURED

**Performance:**
- Warmup-only: 79 regret (catastrophic)
- Corralling: 52 regret [34-80] (median)
- **Improvement: 34% median, 44% best-case**

**When to use:**
- Transfer learning from public datasets
- Don't know if priors will match deployment
- Cost of wrong priors is high

**Trade-off:**
- 34% improvement when priors are bad (79 → 52)
- 8% overhead when priors are good (40 → 43)

### ✅ Scenario 2: Catastrophic Failure Detection (Figure 6)

**Setup:**
- Model suddenly degrades (d>1.5)
- Need fast detection and automatic failover

**Performance:**
- Detection time: **<5 steps**
- Weight shift: 100% → 0% in ~10 steps
- Success rate: ~100% (large effect size)

**When to use:**
- Production safety monitoring
- Large, sudden quality drops
- Automatic failover without human intervention

### ✅ Scenario 3: Alignment Tax Discovery (Figure 1)

**Setup:**
- Expensive model systematically worse on specific tasks
- Effect size: **d=1.90** (very large)
- Frequency: **17.6% of traffic**

**Performance:**
- Cost reduction: **75.7%** ($8.39 → $2.43 per 1M tokens)
- Quality maintained: 0.926 reward (97.5% of max)
- Mixtral wins by -0.682 on strictness tasks

**When to use:**
- Specific task classes with large effect sizes (d>1.0)
- Systematic preference inversions
- High enough frequency to matter (>10%)

### ❌ Scenario 4: General Quality Optimization (Doesn't Work)

**Setup:**
- Small effect sizes (d<0.2)
- Subtle quality differences
- High noise relative to signal

**Performance:**
- Success rate: **~25%** (Figure 6 realistic scenario)
- Need: **10,000+ samples** for statistical power
- Problem: Importance weighting amplifies noise

**When NOT to use:**
- General quality optimization
- Small effect sizes (d<0.2)
- Production constraints (time, cost, non-stationarity)

**Use instead:** Offline A/B testing

---

## The Domain Mismatch Is Real (But Specific)

### Measured Distribution Shift

From `experiments_v1/02_table/`:

| Dataset | Hard Prompts (%) | Easy Prompts (%) | Source |
|---------|------------------|------------------|--------|
| **RouteLLM (warmup)** | **68.6%** | 31.4% | Training priors |
| **LMSYS (holdout)** | **13.7%** | 86.3% | Deployment |
| **Shift** | **5× reduction** | 2.7× increase | Mismatch |

**Domain alignment score:** 0.48 (cosine similarity)
- This is **measured**, not a hyperparameter
- Computed from 1,000 prompts from each distribution

### Why RouteLLM Priors Are Biased

**RouteLLM training:**
- Dataset: Augment-100k (curated hard tasks)
- Purpose: Test model reasoning capabilities
- Selection bias: Overrepresents difficult, constraint-heavy prompts

**LMSYS deployment:**
- Dataset: Real user traffic (chat logs)
- Distribution: 82.4% natural language, 17.6% strictness tasks
- Represents actual usage patterns

**Result:** Warmup expert learns "expensive = better" (correct for RouteLLM, wrong for LMSYS)

### The Uncomfortable Question

**If domain mismatch is known, why not just train priors on LMSYS?**

**Current approach:**
- Use public priors (RouteLLM)
- Add Corralling for insurance
- Accept 34% improvement when priors are wrong, 8% overhead when priors are right

**Alternative approach:**
- Train priors on LMSYS dev set (1,121 samples)
- Use warmup-only (no Corralling needed)
- Achieve 40 regret directly (no insurance cost)

**The trade-off:**
- Current: More complex (Corralling), but transfers from public data
- Alternative: Simpler (warmup-only), but requires domain-specific training

---

## Expert Weight Evolution: Corralling Is Working Correctly

### Figure 4 Results

From `experiments_v1/04_figure/results_3models/quick_test_results.json`:

```
final_expert_weights: [1.41e-128, 1.000]
                      ^^^^^^^^^    ^^^^^
                      Warmup       Tabula Rasa
```

**What this means:**
1. Warmup expert had **consistently high losses** (suboptimal selections)
2. Tabula rasa expert had **consistently low losses** (better selections)
3. Corralling **correctly detected** this through importance-weighted updates
4. Corralling **correctly adapted** by shifting 100% weight to tabula rasa

**This is the CORRECT behavior** when priors are misaligned.

### The Two Interpretations

**Interpretation A: Corralling is saving you**
- Warmup-only would get 79 regret (catastrophic)
- Corralling detects this and shifts to tabula rasa
- Achieves 52 regret (34% improvement)
- **Corralling is the hero**

**Interpretation B: Priors are useless**
- Expert weights → 100% tabula rasa
- You're effectively running tabula rasa with extra steps
- Could have just used tabula rasa from the start
- **Corralling is overhead**

**Which is correct?** **Both**, depending on what you knew upfront:
- If you **didn't know** priors were bad → Interpretation A (insurance worked)
- If you **did know** priors were bad → Interpretation B (why use them?)

---

## The Cost-Quality Story (Figure 4)

### The "Expensive Bias" Discovery

| Method | Cost ($/1M tokens) | Reward | Model Distribution |
|--------|-------------------|--------|-------------------|
| **Warmup (biased)** | **$8.39** | 0.947 | 6% Mixtral, 94% flagships |
| **Corralling** | **$2.43** | 0.926 | 23% Mixtral, 77% flagships |
| Always GPT-4-Turbo | $10.00 | 0.95 | 100% expensive |
| Always Mixtral | $0.27 | 0.90 | 100% cheap |

**Key findings:**
1. Warmup overrates expensive models (trained on hard RouteLLM tasks)
2. Corralling discovers GPT-4o ≈ GPT-4-Turbo quality at 4× lower cost
3. **71% cost reduction** ($8.39 → $2.43) with only 2% quality drop (0.947 → 0.926)

**Important:** This is **not** optimizing for cost (λ_cost=0). Cost savings emerge naturally from correcting **quality prediction errors**.

**The narrative:** Warmup believes expensive models are necessary for high quality. Corralling learns this is false by observing actual rewards, discovers cheaper models work just as well.

---

## Pareto Frontier (Figure 5): banditGPT vs RouteLLM

### The "Negative Intelligence Tax"

From `experiments_v1/05_figure/README.md`:

**Shocking finding:**
- GPT-4-Turbo: $0.013/request, 0.812 reward
- Mixtral: $0.000294/request, 0.823 reward
- **GPT-4-Turbo costs 43× more but performs 1.3% WORSE**

### Performance Comparison

| Method | Peak Quality | Peak Cost | Pareto-Optimal Points |
|--------|-------------|-----------|----------------------|
| **banditGPT-Hybrid** | **0.9088** | $0.00954 | 6/10 (60%) |
| RouteLLM-MF | 0.8827 | $0.00651 | 10/28 (36%) |
| Oracle | 0.9533 | $0.00195 | 1/1 (100%) |

**Gap closure:**
- banditGPT: **66.2%** of gap to oracle
- RouteLLM: 46.2% of gap to oracle

**Interpretation:** banditGPT's online learning (which includes Corralling) does provide benefit over pre-trained routing (RouteLLM).

**But:** This comparison is "banditGPT WITH Corralling" vs "RouteLLM". It's NOT "banditGPT with Corralling" vs "banditGPT without Corralling".

**Missing experiment:** What if you ran SimpleLinUCBRouter (warmup-only) on the Pareto sweep? Would it do better or worse than Corralling?

---

## Why You're Questioning It Now

### What Changed

**Original experiments (single seed):**
- Corralling: **44 regret** (1.10× vs oracle)
- Narrative: "Near-optimal performance with safety guarantees"
- Seemed like a slam dunk

**Multi-seed validation:**
- Corralling: **52 regret [34-80]** (1.30× vs oracle median)
- Worst seed: **80 regret** (no better than warmup's 79)
- Narrative: "Probabilistic improvement with high variance"

**Realistic scenarios:**
- Small effect sizes (d≈0.12): **25% success rate**
- Need 10k+ samples for statistical power
- Production constraints make this infeasible

### The Reality Check

You're discovering that Corralling:
1. **Is NOT general-purpose** (only works for d>1.0)
2. **Has high variance** (worst-case ≈ warmup)
3. **Only helps in specific scenarios** (domain mismatch, catastrophic failure, alignment tax)
4. **Was over-claimed initially** (single seed cherry-picking)

**This is GOOD science!** You're:
- Running proper multi-seed validation
- Testing realistic scenarios
- Honestly documenting limitations
- Questioning initial assumptions

---

## The Honest Assessment

### What Corralling IS Good For

✅ **Domain mismatch protection** (34% median improvement)
- When using transfer learning from public datasets
- When distribution match is uncertain
- Trade-off: 8% overhead if priors are good, 34% improvement if priors are bad

✅ **Catastrophic failure detection** (<5 step detection)
- Production safety mechanism
- Large, sudden quality drops (d>1.5)
- Automatic failover

✅ **Alignment tax exploitation** (d=1.90 on 17.6% of traffic)
- Systematic preference inversions
- Large effect sizes on specific task classes
- Cost savings emerge from correcting quality predictions

### What Corralling IS NOT Good For

❌ **General quality optimization** (d<0.2)
- Need 10k+ samples for statistical power
- Importance weighting amplifies noise
- Offline A/B testing is better

❌ **Guaranteed performance improvement**
- High variance (worst-case = warmup)
- Probabilistic benefit, not deterministic
- Need to run multiple seeds

❌ **Simple warmup enhancement**
- If priors are good: 8% overhead for no benefit
- If priors are bad: Why use them at all?

---

## Recommendations

### For the Paper

**Option 1: Narrow the scope (recommended)**

Position Corralling as **insurance for domain mismatch + safety mechanism for catastrophic failures**, not general-purpose routing.

**Claims to keep:**
- ✅ "34% median improvement when priors are misaligned"
- ✅ "<5 step detection for catastrophic failures"
- ✅ "Discovers alignment tax (d=1.90) on 17.6% of traffic"
- ✅ "71% cost reduction from correcting warmup bias"

**Claims to revise:**
- ⚠️ "Near-optimal performance" → "Competitive performance (1.30× vs oracle median)"
- ⚠️ "44 regret" → "52 regret [34-80] (N=10 seeds)"
- ❌ Remove "general-purpose adaptive routing"

**Option 2: Add warmup-only comparison**

Run the missing ablation: SimpleLinUCBRouter (warmup-only) on all experiments.

**This answers:**
- Is Corralling better than just using warmup?
- How much does the meta-algorithm contribute?
- Is the overhead (8% when priors are good) worth the insurance (34% when priors are bad)?

### For the System

**Option 1: Keep Corralling for insurance**

If you value:
- Protection against unknown domain mismatch
- Catastrophic failure detection
- Alignment tax discovery

Then keep Corralling, but:
- Document the variance (report median + IQR)
- Clarify it's insurance, not performance optimization
- Set expectations about when it helps vs doesn't

**Option 2: Remove Corralling, use warmup-only**

If you:
- Know priors match deployment (no mismatch)
- Don't need catastrophic failure detection
- Want simpler system

Then use SimpleLinUCBRouter:
- Simpler implementation
- No stochastic variance
- No 8% overhead when priors are good

**Option 3: Hybrid approach**

Use Corralling for:
- ✅ Production safety monitoring (catastrophic failure detection)
- ✅ New model rollouts (domain mismatch uncertainty)

Use warmup-only for:
- ✅ Stable deployments (known good priors)
- ✅ Cost-sensitive scenarios (minimize overhead)

---

## Bottom Line

**Corralling is not broken, it's specialized.**

It works exactly as designed:
- Detects domain mismatch (expert weights → 100% tabula rasa)
- Provides 34% median improvement vs harmful priors
- Enables fast catastrophic failure detection (<5 steps)
- Discovers alignment tax effects (d=1.90 on 17.6% of traffic)

But it's NOT:
- General-purpose quality optimizer
- Low-variance (CV=42%)
- Better than warmup when priors are good (8% overhead)

**The honest message:**

> "Corralling provides insurance against domain mismatch and catastrophic failures. It improves median regret by 34% (79 → 52) when warmup priors are misaligned, with <5 step detection for large quality drops. However, it has high variance (worst-case seed performs no better than warmup) and requires large effect sizes (d>1.0) to be effective. For general quality optimization with small effect sizes, offline A/B testing remains more appropriate."

This is still a **valuable contribution**, just not the general-purpose solution it initially appeared to be.

Your skepticism is warranted and scientifically rigorous. The multi-seed analysis and realistic scenario testing revealed important limitations that weren't visible in the initial single-seed experiments.
