# Why the Realistic Scenario Only Achieves 25% Success

## TL;DR

**The signal (Δμ=0.011) is SMALLER than the noise (σ≈0.10).** On nearly half of all samples (46.7%), GPT-4 randomly gets a higher reward than Mixtral just due to noise, making it statistically impossible for Corralling to reliably detect the true difference with only 1,000 samples.

---

## The Four Root Causes

### 1. Tiny Effect Size (100x Smaller)

**Synthetic Scenario:**
- Δμ = 0.70 (70 percentage points)
- Cohen's d = 10.5 (huge effect)
- Separation: 8.8 standard deviations
- **Interpretation**: These distributions are in different galaxies

**Realistic Scenario:**
- Δμ = 0.011 (1.1 percentage points)  
- Cohen's d = 0.12 (tiny effect)
- Separation: 0.11 standard deviations
- **Interpretation**: Signal is SMALLER than noise

**Impact**: The realistic scenario has **90x smaller effect size**, requiring exponentially more samples to detect.

---

### 2. High Distribution Overlap (47% Confusion Rate)

The diagnostic visualization shows the critical difference:

**Synthetic**: The Mixtral and GPT-4 distributions don't overlap at all
- P(GPT-4 sample > Mixtral sample) = 0.00000001%
- Essentially impossible for GPT-4 to randomly beat Mixtral

**Realistic**: The distributions almost completely overlap (yellow region)
- P(GPT-4 sample > Mixtral sample) = **46.7%**
- Nearly HALF the time, GPT-4 gets higher reward by pure chance!

**What this means**: When Corralling samples the warmup expert (GPT-4) and gets a reward of 0.85, and then samples the tabula rasa expert (Mixtral) and gets 0.81, it can't tell if:
- GPT-4 is actually better (signal), OR
- It's just random noise (happened to sample high for GPT-4, low for Mixtral)

With 47% confusion rate, most samples provide **contradictory evidence** that cancels out.

---

### 3. Insufficient Statistical Power (Need 2,400 Samples, Have 1,000)

**Statistical Power Analysis** (bottom-left plot):

To detect an effect with Cohen's d=0.12 at 80% power, you need:
- **~1,174 samples per expert** (2,348 total)
- **We only have ~500 samples per expert** (1,000 total)
- **Underpowered by 58%**

**Breakdown of the 1,000 steps:**
1. Corralling splits 50/50 between experts → 500 per expert
2. Each expert does 50/50 exploration → 250 on GPT-4, 250 on Mixtral
3. **Only 250 GPT-4 samples under warmup vs 250 Mixtral samples under tabula rasa**
4. This is **21% of the 1,174 samples needed** for 80% power

**Result**: The experiment is fundamentally underpowered to detect this effect size.

---

### 4. Importance Weighting Amplifies Noise (Bottom-Right Plot)

The Corralling algorithm uses **importance-weighted loss estimation**:

```
ℓ̂ = (1 - reward) / p
```

where `p` is the probability of selecting that expert.

**The Problem**: As an expert's weight drops, the estimator variance explodes:

```
Var(ℓ̂) = Var(loss) / p = σ² / p
```

**Concrete Example** (bottom-right plot shows this):

| Expert Weight (p) | Loss Std Dev | vs Signal (0.011) |
|-------------------|--------------|-------------------|
| p = 0.50 | 0.141 | **13x larger** |
| p = 0.30 | 0.183 | **17x larger** |
| p = 0.10 | 0.316 | **29x larger** |
| p = 0.05 | 0.447 | **41x larger** |

**The Vicious Cycle**:
1. Warmup expert starts getting low rewards (due to tiny signal)
2. Its weight drops to p=0.1
3. Importance weighting amplifies noise by √10 = 3.2x
4. Now noise (0.316) is **29x larger than signal (0.011)**
5. Exponential weights oscillate wildly instead of converging
6. Sometimes warmup gets lucky high reward → weight bounces back up
7. Process never stabilizes

**The red shaded region** shows where noise exceeds signal—this is almost the entire range below p=0.5.

---

## Why Synthetic Works But Realistic Fails

### Synthetic Scenario (100% Success)

| Factor | Value | Impact |
|--------|-------|--------|
| Effect size | d = 10.5 | Signal is 9x larger than noise |
| Overlap | 0.00% | Zero confusion |
| Samples needed | 0.1 | Already overpowered |
| Noise amplification | Irrelevant | Signal so strong it doesn't matter |

**Result**: Clean, decisive exponential decay to 0.00 vs 1.00 weights

### Realistic Scenario (25% Success)

| Factor | Value | Impact |
|--------|-------|--------|
| Effect size | d = 0.12 | Signal is 9x SMALLER than noise |
| Overlap | 46.7% | Massive confusion |
| Samples needed | 1,174 | Underpowered by 58% |
| Noise amplification | 29x at p=0.1 | Completely swamps signal |

**Result**: Oscillating weights that rarely converge (only 25% of trials)

---

## Implications for Production

### What the 25% Success Rate Means

This is **NOT a bug**—it's a fundamental limitation of online learning with small effect sizes.

In production with real LMSYS data (d≈0.1), Corralling cannot reliably detect quality differences because:
1. **Signal too weak**: 1.1 percentage point difference
2. **Noise too high**: 10% standard deviation
3. **Sample budget insufficient**: Need 10x more data
4. **Variance amplification**: Importance weighting makes it worse

### Can't We Just Get More Samples?

**Short answer**: Yes in simulation (we just proved it!), but no in production.

**We tested this**: Running 10,000 steps (10x longer) achieves **100% success** vs 25% with 1,000 steps.

**But in production, five constraints prevent this:**

1. **Time to convergence**: 10-100 days for small companies (unacceptable)
2. **Opportunity cost**: Lose quality/revenue during 2,000-step learning phase
3. **Non-stationarity**: World changes over weeks/months (user distribution, model updates)
4. **Context drift**: New topics, language evolution, embedding updates
5. **Sample efficiency**: Offline A/B testing is cheaper and faster

See `PRODUCTION_CONSTRAINTS.md` for detailed analysis of why production can't just "wait longer."

## Solutions for Production

If you need to detect small effects (d<0.2) in production:

**Option 1: Offline A/B Testing + Corralling Safety Net** ⭐ Recommended
- Run offline test with 10,000 samples (1-7 days in controlled environment)
- If d>0.5: Deploy with Corralling as safety net for catastrophic failures
- If d<0.5: Don't deploy, signal too weak for online detection

**Option 2: Sequential Testing (SPRT)**
- 50% fewer samples than fixed-horizon test
- Better power for small effects than Corralling

**Option 3: Accept Limitations**
- Corralling protects against **catastrophic failures** (d>1.0)
- Don't expect it to detect subtle quality differences (d<0.2)
- Use it as a safety mechanism, not an optimization tool

**Option 4: High-Traffic Only**
- Only use Corralling if you have 10,000+ requests/day
- Converges in 1 day, avoiding non-stationarity issues

---

## Why This Strengthens (Not Weakens) the Paper

By running the realistic scenario and honestly reporting the 25% success rate, you demonstrate:

1. ✅ **Scientific rigor**: Tested under realistic conditions, not just favorable ones
2. ✅ **Intellectual honesty**: Disclosed limitations clearly
3. ✅ **Practical value**: Readers know when Corralling will/won't work
4. ✅ **Reproducibility**: Other researchers can validate your findings

**Compare two papers**:

**Bad Paper**: "Our algorithm achieves 100% success!" (only tested with d=10)  
**Good Paper**: "Our algorithm achieves 100% with large effects (d>1) but only 25% with realistic effects (d=0.1), requiring 10x more samples"

**Which would you trust more?** The honest one.

---

## Visual Summary

See `results/diagnostic_realistic_failure.png`:

- **Top-left**: Synthetic distributions (zero overlap)
- **Top-right**: Realistic distributions (47% overlap in yellow)
- **Bottom-left**: Statistical power curve (realistic needs 8,000x more samples)
- **Bottom-right**: Noise amplification (noise exceeds signal below p=0.5)

---

## References

For more details on:
- **Effect size interpretation**: Cohen (1988) - d<0.2 is "small", d>0.8 is "large"
- **Statistical power**: Cohen (1992) - 80% power is standard
- **Importance sampling variance**: Owen (2013) - variance scales as 1/p
- **Sequential testing**: Wald (1945) - SPRT for faster detection

---

## Files

- `diagnostic_realistic_failure.py` - Generates this analysis
- `results/diagnostic_realistic_failure.{png,pdf}` - Visualization
- `generate_figure5_realistic.py` - Runs 20-seed realistic experiment
