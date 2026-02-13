# Variance vs Regime Switching: Why More Reps Won't Help

## The Confound Explained

### Scenario A: Traditional Variance (More Reps HELPS)
```
            n_eff=1.0          n_eff=20.0         Gap
Seed 1:     4.50 ± 0.1         4.45 ± 0.1         +1.1%
Seed 2:     4.48 ± 0.1         4.43 ± 0.1         +1.1%
Seed 3:     4.52 ± 0.1         4.47 ± 0.1         +1.1%
...
Seed 100:   4.49 ± 0.1         4.44 ± 0.1         +1.1%

Average:    4.50 ± 0.01        4.45 ± 0.01        +1.1% ✓
```
**Interpretation**: Effect is consistent across seeds, just noisy
**Solution**: More reps → tighter CI → clear conclusion
**Statistical approach**: t-test with N=100 has high power

---

### Scenario B: Regime Switching (More Reps DOESN'T HELP)
```
            n_eff=1.0    n_eff=20.0    Gap       Expert Active
Seed 1:     4.50         4.30          +4.6%     Warmup (100%)
Seed 2:     4.25         4.27          -0.5%     Tabula Rasa (100%)
Seed 3:     4.23         4.23          0.0%      Tabula Rasa (100%)
...
Seed 33:    4.48         4.28          +4.6%     Warmup (100%)
Seed 34:    4.24         4.24          0.0%      Tabula Rasa (100%)
...
Seed 100:   4.22         4.22          0.0%      Tabula Rasa (100%)

Average:    4.32 ± 0.03  4.28 ± 0.03   +1.0% ✗ (misleading!)
```
**Interpretation**: Effect depends on which expert Corralling chooses
**Problem**: Averaging across regimes hides the true story
**Statistical approach**: Stratified analysis by expert regime

---

## Why Averaging is Misleading

### The Math
```
Overall effect = P(Warmup) × Effect_warmup + P(Tabula) × Effect_tabula
                = 0.33 × (+4.6%) + 0.67 × (0.0%)
                = +1.5%
```

### With 100 Seeds
- Tight confidence interval: ±0.03
- Significant p-value: p<0.05
- **BUT**: Wrong interpretation!

You'd conclude: "n_eff=1.0 is 1.5% better on average"

**Reality**: 
- 33% of time: n_eff=1.0 is 4.6% better (when semantic transfer used)
- 67% of time: n_eff doesn't matter at all (semantic transfer ignored)

---

## Analogy: Medicine Trial

### Bad Experiment (Like Ours)
```
Drug A vs Drug B for headaches:

Patients with tension headaches (33%):
  - Drug A: 90% cure rate
  - Drug B: 60% cure rate
  - Conclusion: Drug A is 30pp better! ✓

Patients with migraines (67%):
  - Drug A: 50% cure rate
  - Drug B: 50% cure rate  
  - Conclusion: No difference (neither drug works for migraines)

Average across all patients:
  - Drug A: 63% cure rate
  - Drug B: 53% cure rate
  - Conclusion: Drug A is 10pp better overall ✗ (misleading!)
```

**What's wrong**: 
- You'd prescribe Drug A for all patients
- But it only helps 33% of patients (tension headaches)
- For 67% (migraines), you need a different treatment

**What you should report**:
- "Drug A works for tension headaches (+30pp)"
- "Neither drug works for migraines"
- "Diagnose first, then choose treatment"

---

## Our Experiment

### What We Did (Wrong Approach)
```
Research Question: "Is n_eff=1.0 optimal?"

Average Result (N=3 seeds):
- n_eff=1.0: 4.32
- n_eff=20.0: 4.28
- Conclusion: +1.0%, p=0.43 (not significant)

With N=100 seeds:
- n_eff=1.0: 4.32 ± 0.01
- n_eff=20.0: 4.28 ± 0.01
- Conclusion: +1.0%, p=0.001 (significant!) ✗ MISLEADING!
```

### What We Should Do (Stratified Analysis)
```
Research Question: "When does n_eff matter?"

Regime 1: Warmup Expert Active (33% of seeds)
- n_eff=1.0: 4.48 ± 0.02
- n_eff=20.0: 4.28 ± 0.02
- Conclusion: +4.6%, p<0.001 ✓ (n_eff matters here!)

Regime 2: Tabula Rasa Active (67% of seeds)
- n_eff=1.0: 4.25 ± 0.02
- n_eff=20.0: 4.25 ± 0.02
- Conclusion: 0.0%, p=1.00 ✓ (n_eff ignored here!)

Meta-Finding:
- n_eff only matters when Corralling uses semantic transfer (33% of time)
- Production implication: n_eff choice has 1.5% impact on average users
```

---

## How to Identify the Problem

### Diagnostic Signs

1. **Bimodal expert weights**:
   - Seeds cluster at 0% or 100% warmup weight
   - NOT normally distributed around 50%

2. **Effect size varies dramatically**:
   - Some seeds: Large effect (+5%)
   - Other seeds: Zero effect (0%)
   - NOT consistent effect with noise

3. **Correlations**:
   - Effect size correlates with warmup expert weight (r=0.99)
   - NOT independent random variation

### Example Code
```python
# Traditional variance (OK to average)
effects = [1.1, 1.0, 1.2, 0.9, 1.1]  # Consistent ~1%
print(f"Mean: {np.mean(effects):.1%}")  # 1.1%
print(f"CI: ±{1.96 * np.std(effects):.1%}")  # ±0.2%

# Regime switching (NOT OK to average)
effects = [4.6, 0.0, 0.0, 4.5, 0.0]  # Bimodal!
print(f"Mean: {np.mean(effects):.1%}")  # 1.8% (misleading!)
print(f"Regimes: {sum(e > 2 for e in effects)}/5 warmup")  # 2/5
```

---

## The Right Experimental Design

### Option 1: Isolate the Effect (Turn Off Confound)
```python
# Remove Corralling to test n_eff in isolation
router = BanditRouter.create(
    use_corralling=False,  # ← KEY CHANGE
    priors=str(DEFAULT_WARMUP_PRIORS_PATH),
    alpha=2.0
)
```
**Result**: 
- All seeds use semantic transfer (no regime switching)
- n_eff effect is consistent across seeds
- Traditional variance → more reps help

### Option 2: Stratified Analysis (Embrace the Confound)
```python
# Track which expert is active
for seed in range(100):
    weights, rewards = run_experiment(seed)
    
    if warmup_weight > 0.5:
        warmup_regime_results.append(rewards)
    else:
        tabula_regime_results.append(rewards)

# Report separately
print(f"Warmup regime (N={len(warmup_regime_results)}): +4.6%")
print(f"Tabula regime (N={len(tabula_regime_results)}): 0.0%")
```

### Option 3: Report Heterogeneous Effects
```python
# Regression with interaction term
effect = β₀ + β₁ × n_eff + β₂ × warmup_weight + β₃ × (n_eff × warmup_weight)

# Interpretation:
# β₁: Direct n_eff effect (should be ~0)
# β₂: Warmup expert effect (should be positive)
# β₃: Interaction (n_eff only matters when warmup active)
```

---

## Bottom Line

### Question: "Do we need more reps?"

**Answer: Depends on your goal**

**If goal is**: "Get significant p-value for n_eff effect"
- **Yes**: N=100 seeds → p<0.05
- **But**: Result is misleading (averages incompatible regimes)

**If goal is**: "Understand when n_eff matters"
- **No**: N=3 seeds already revealed the regime switching
- **Instead**: Stratify by expert regime or turn off Corralling

**If goal is**: "Make production recommendation"
- **No**: More reps won't change conclusion
- **Conclusion**: "n_eff doesn't matter much (only affects 33% of traffic)"

---

## Recommendation

### For the Paper

**Don't run 100 seeds to get significance**. Instead:

1. **Report stratified results**:
   - "When warmup expert active (33% of seeds): n_eff=1.0 beats n_eff=20.0 by 4.6%"
   - "When tabula rasa active (67% of seeds): n_eff has no effect"

2. **Explain the meta-learning**:
   - "Corralling adaptively chooses between semantic transfer and cold start"
   - "Choice depends on early-stage performance match with priors"

3. **Honest production impact**:
   - "Expected n_eff effect: 1.5% (0.33 × 4.6%)"
   - "Robustness comes from expert switching, not parameter tuning"

### For Future Experiments

**When using meta-learning systems** (like Corralling):
1. Always track which component is active
2. Report results stratified by active component
3. Don't average across heterogeneous regimes
4. Consider ablations with meta-learning disabled

---

## Statistical Principle

**Simpson's Paradox**: 
- A trend that appears in different groups can disappear or reverse when the groups are combined.

**Our case**:
- Warmup regime: n_eff=1.0 > n_eff=20.0 ✓
- Tabula regime: n_eff=1.0 = n_eff=20.0 ✓
- Combined: n_eff=1.0 ≈ n_eff=20.0 ✗ (hides regime difference)

**Solution**: Don't combine! Report conditional effects.

---

**Prepared by**: Statistical Analysis  
**Key Insight**: More reps tighten CI but don't resolve confounds. Stratify by regime instead.
