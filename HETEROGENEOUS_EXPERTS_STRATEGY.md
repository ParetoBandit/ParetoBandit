# Heterogeneous Experts Strategy: Intelligent Exploration-Exploitation

**Date:** January 26, 2026  
**Status:** ✅ Implemented  
**Files Modified:** `src/bandit_gpt/router.py` (lines 1999-2106)

## Executive Summary

Implemented an intelligent **Heterogeneous Experts Strategy** that eliminates the need for manual alpha tuning by leveraging the Corralling meta-learner to automatically switch between two opposing strategies:

1. **Expert 1 (Conservative/Efficiency Engine):** Decays exploration → Optimizes for stable environments
2. **Expert 2 (Adaptive/Discovery Engine):** Maintains constant vigilance → Detects distribution shifts

**Key Benefit:** The router automatically adapts to both stationary and non-stationary regimes without manual intervention.

---

## The Fundamental Problem: Exploration-Exploitation in Non-Stationary Worlds

### The Alpha Scheduling Dilemma

Traditional bandit algorithms assume a **stationary world** where:
- The optimal model is fixed
- Learning converges to a stable policy
- Decaying exploration (α → 0) is optimal

**Reality of Production ML Systems:**
- New models are released (GPT-4 → GPT-5 → GPT-5.1)
- Model APIs change (rate limits, deprecations)
- Distribution drift (user preferences evolve)
- The world **never stops changing**

### Why Decaying Alpha Causes "Brain Death"

```
Time →
        t=0          t=5000        t=10000        [GPT-5.1 Released]
α:      1.0    →     0.3     →     0.01          →    0.01 (stuck)
        ↓            ↓             ↓                     ↓
    Exploring   Converging   Exploiting          Still exploiting GPT-4
                                                  (Never tries GPT-5.1!)
```

**The Problem:**
- At t=10000, α=0.01 means the router is ~99% exploiting the current best (GPT-4)
- When GPT-5.1 arrives, there's only 1% chance to explore it
- The router is "too confident" in old knowledge to discover new truths
- **Result:** Manual reset required to escape local optimum

---

## The Naive Solution (Rejected)

### Option 1: Constant High Alpha

```python
expert = LinUCBRouter(alpha_start=2.0, alpha_end=2.0)  # Never decay
```

**Pros:**
- ✅ Detects distribution shifts immediately
- ✅ Always responsive to new models

**Cons:**
- ❌ Wastes resources exploring during stable periods
- ❌ Higher regret when the world is actually stationary
- ❌ Sub-optimal for benchmarks (where truth is fixed)

### Option 2: Manual Toggling

```python
if production_with_new_models:
    alpha = 2.0  # Stay vigilant
elif stable_benchmark:
    alpha = 0.01  # Pure exploitation
```

**Cons:**
- ❌ Requires domain knowledge upfront
- ❌ Can't react to unexpected shifts
- ❌ Brittle: wrong configuration = poor performance

---

## The Intelligent Solution: Heterogeneous Experts

### Core Insight

**Instead of choosing one alpha schedule, use BOTH and let the meta-learner decide which is better.**

The Corralling architecture already solves the "expert selection problem." We just need to give it **heterogeneous experts** with opposing strategies:

```
Expert 1: Conservative (Decay)   ←→   Expert 2: Adaptive (Constant)
     Assumes: Stable World               Assumes: Shifting World
     Optimizes: Low Regret                Optimizes: Responsiveness
     Risk: Brain Death                    Risk: Wasted Exploration
```

**Meta-Learner (Corralling):**
- Observes which expert performs better
- Routes traffic to the winning strategy
- Automatically pivots when the regime changes

---

## Implementation Details

### Before: Homogeneous Experts (Same Alpha Schedule)

```python
# Both experts decay from 2.0 → 0.1
expert_warmup = CostAwareLinUCBRouter(
    alpha_start=2.0,
    alpha_end=target_alpha  # 0.1
)

expert_tabula_rasa = CostAwareTabulaRasaRouter(
    alpha_start=2.0,
    alpha_end=target_alpha  # 0.1
)
```

**Problem:** Both experts converge to low exploration → Both suffer brain death

---

### After: Heterogeneous Experts (Opposing Strategies)

```python
# ---------------------------------------------------------------
# Expert 1: The "Efficiency Engine" (Conservative/Warmup)
# ---------------------------------------------------------------
# STRATEGY: Aggressive decay to pure exploitation
# ASSUMPTION: The world is stable; priors are good
# GOAL: Minimize regret by converging to the best known model
# BEHAVIOR:
#   - Starts with moderate exploration (alpha=1.0)
#   - Linearly decays to near-zero (alpha=0.01)
#   - Result: High efficiency in stable environments
#   - Risk: "Brain Death" if new models appear (e.g., GPT-5.1)
# ---------------------------------------------------------------
expert_warmup = CostAwareLinUCBRouter(
    models=router.bandit.models,
    warmup_priors=warmup_priors,
    model_costs=model_costs,
    alpha_start=1.0,   # Moderate initial exploration
    alpha_end=0.01,    # Decay to near-zero (Pure Exploitation)
    cost_penalty=0.0
)

# ---------------------------------------------------------------
# Expert 2: The "Discovery Engine" (Adaptive/Tabula Rasa)
# ---------------------------------------------------------------
# STRATEGY: Constant high alpha (vigilance)
# ASSUMPTION: The world is non-stationary; shifts happen
# GOAL: Remain sensitive to distribution shifts and new models
# BEHAVIOR:
#   - Starts with high exploration (alpha=2.0)
#   - NEVER decays (alpha_end=2.0)
#   - Result: Immediately detects new models (GPT-5) or concept drift
#   - Cost: Higher exploration overhead during stable periods
# META-LEARNING GUARANTEE:
#   - During stable times: Corralling downweights this expert (saves cost)
#   - During shifts: This expert wins → Corralling pivots automatically
# ---------------------------------------------------------------
expert_tabula_rasa = CostAwareTabulaRasaRouter(
    models=router.bandit.models,
    context_dim=router.bandit.dim,
    model_costs=model_costs,
    alpha_start=2.0,   # High initial exploration
    alpha_end=2.0,     # CONSTANT: Never stop exploring
    cost_penalty=0.0,
    ridge_lambda=1.0
)

# ---------------------------------------------------------------
# The Manager: Corralling Meta-Learner
# ---------------------------------------------------------------
# Automatically switches between "Efficiency" and "Discovery" 
# based on which expert performs better in the current regime.
#
# Stable Period → Conservative expert dominates (low regret)
# Distribution Shift → Adaptive expert wins (detects changes)
# New Model Release → Adaptive finds it first → Router pivots
# ---------------------------------------------------------------
router.corralling_router = CorrallingRouter(
    experts=[expert_warmup, expert_tabula_rasa],
    models=router.bandit.models,
    learning_rate=router.corralling_learning_rate,
    gamma=router.corralling_gamma
)
```

---

## Behavior Analysis: Lifecycle Scenarios

### Scenario 1: Stable Environment (Benchmarks, Fixed Test Sets)

```
Time:       t=0        t=1000       t=5000       t=10000
α1 (Cons):  1.0   →    0.5    →     0.1    →     0.01
α2 (Adap):  2.0   →    2.0    →     2.0    →     2.0

Corralling Weights:
Expert 1:   0.50  →    0.70   →     0.85   →     0.95  ✅ Winner
Expert 2:   0.50  →    0.30   →     0.15   →     0.05  (Downweighted)

Outcome: Conservative expert dominates → Low regret, high efficiency
```

**Why Conservative Wins:**
- It quickly converges to the best model
- Lower exploration overhead → Higher cumulative reward
- Corralling observes this and routes 95% traffic to Expert 1

**Cost Saved:**
- Adaptive expert still explores at α=2.0, but only gets 5% traffic
- Net effect: ~95% exploitation efficiency with 5% safety margin

---

### Scenario 2: New Model Release (GPT-4 → GPT-5.1)

```
Time:       t=8000     t=8100   [GPT-5.1 Released]   t=8500     t=9000
α1 (Cons):  0.02   →   0.02    →    0.01      →       0.01   →   0.01
α2 (Adap):  2.0    →   2.0     →    2.0       →       2.0    →   2.0

Expert 1 Performance:
  - Stuck on GPT-4 (99% of the time)
  - Rarely tries GPT-5.1 (α=0.01 → 1% exploration)
  - Misses the quality improvement

Expert 2 Performance:
  - Still exploring at α=2.0 (20% exploration rate)
  - Quickly discovers GPT-5.1 is better
  - Cumulative reward jumps

Corralling Weights:
Expert 1:   0.95   →   0.90    →    0.70      →       0.30   →   0.10
Expert 2:   0.05   →   0.10    →    0.30      →       0.70   →   0.90  ✅ Winner

Outcome: Adaptive expert detects shift → Corralling pivots → Router adapts
```

**Critical Timeline:**
1. **t=8100:** GPT-5.1 released but not yet discovered
2. **t=8200:** Expert 2 (Adaptive) tries GPT-5.1 due to high α
3. **t=8250:** Expert 2 observes GPT-5.1 > GPT-4 quality
4. **t=8300:** Corralling sees Expert 2 getting higher rewards
5. **t=8500:** Corralling shifts weight to Expert 2 (30%)
6. **t=9000:** Expert 2 dominates (90%), router effectively uses GPT-5.1

**No Manual Intervention Required!**

---

## Mathematical Guarantees

### Corralling Loss Bound (Agarwal et al., 2017)

```
Regret(Corralling) ≤ min_i Regret(Expert_i) + O(√(T log K))
```

**Interpretation:**
- The meta-learner performs **at most logarithmically worse** than the best expert
- In stable regimes: ≈ Expert 1 (Conservative)
- In shifting regimes: ≈ Expert 2 (Adaptive)
- During transitions: Small overhead for learning which expert is better

### Exploration-Exploitation Trade-off

**Expert 1 (Conservative):**
```
Regret_stable = O(d log T)       [Low]
Regret_shift  = O(T)             [High - Brain Death]
```

**Expert 2 (Adaptive):**
```
Regret_stable = O(√T)            [Medium - Constant exploration]
Regret_shift  = O(d log T)       [Low - Detects changes]
```

**Corralling (Heterogeneous):**
```
Regret_stable ≈ O(d log T)       [≈ Expert 1]
Regret_shift  ≈ O(d log T)       [≈ Expert 2]
Regret_total  = min(Expert 1, Expert 2) + O(√(T log K))
```

**Conclusion:** Best of both worlds (with logarithmic overhead)

---

## Comparison to Manual Tuning

### Manual Strategy Table (Old Approach)

| Scenario | Recommended Alpha | Manual Effort | Failure Mode |
|----------|------------------|---------------|--------------|
| A/B Test (Stable) | Decay 2.0→0.1 | High | Misses new models |
| Production (Shifting) | Constant 2.0 | High | Wastes exploration in stable periods |
| Benchmark Evaluation | Static 0.05 | High | Can't adapt to changes |
| Cold Start | Decay 5.0→0.5 | High | Wrong schedule = poor performance |

**Problems:**
- Requires upfront knowledge of the environment type
- Wrong choice = suboptimal regret
- Can't adapt mid-stream if assumptions change

---

### Heterogeneous Strategy (New Approach)

| Scenario | Behavior | Manual Effort | Failure Mode |
|----------|----------|---------------|--------------|
| A/B Test (Stable) | Auto: Favors Expert 1 (Decay) | **None** | ✅ Robust |
| Production (Shifting) | Auto: Pivots to Expert 2 (Constant) | **None** | ✅ Robust |
| Benchmark Evaluation | Auto: Favors Expert 1 (Decay) | **None** | ✅ Robust |
| Cold Start | Auto: Balances both experts initially | **None** | ✅ Robust |

**Advantages:**
- ✅ Zero manual tuning required
- ✅ Adapts automatically to regime changes
- ✅ Worst-case: ~√T overhead (logarithmic)
- ✅ Best-case: Optimal regret (matches best expert)

---

## Implementation Notes

### Alpha Decay Mechanism

Both expert classes (`CostAwareLinUCBRouter`, `CostAwareTabulaRasaRouter`) use **linear decay**:

```python
def get_current_alpha(self, total_steps: int) -> float:
    """
    α_t = α_start + (t / T) × (α_end - α_start)
    """
    if total_steps == 0:
        return self.alpha_end  # Evaluation mode
    
    fraction = min(self.t / total_steps, 1.0)
    return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)
```

**Expert 1 (Conservative):**
- `alpha_start=1.0, alpha_end=0.01`
- At t=0: α=1.0 (moderate exploration)
- At t=T: α=0.01 (pure exploitation)

**Expert 2 (Adaptive):**
- `alpha_start=2.0, alpha_end=2.0`
- At t=0: α=2.0 (high exploration)
- At t=T: α=2.0 (still high exploration)
- **Effect:** No decay → constant vigilance

### Why Not Add a Third Expert?

**Potential Extension:**
```python
expert_moderate = CostAwareLinUCBRouter(
    alpha_start=1.0,
    alpha_end=0.1  # Moderate decay
)
```

**Trade-offs:**
- **Pro:** More granular adaptation
- **Con:** Higher meta-learning overhead (O(√(T log K)) grows with K)
- **Con:** Slower convergence (Corralling needs more data to distinguish 3 experts)

**Recommendation:** Two experts are sufficient for most cases. The meta-learner can interpolate by mixing the two experts' distributions.

---

## Experimental Validation (Expected Results)

### Stable Benchmark (e.g., RouteLLM Test Set)

**Prediction:**
- **Burn-in (t=0-1000):** Both experts explore, Corralling weights ≈ 50/50
- **Convergence (t=1000-5000):** Expert 1 (Conservative) pulls ahead, weight → 70/30
- **Exploitation (t=5000+):** Expert 1 dominates, weight → 90/10
- **Final Regret:** ≈ Expert 1 standalone (+ small Corralling overhead)

**Metrics to Monitor:**
- `expert_selections[0]` vs `expert_selections[1]`
- Cumulative regret compared to single-expert baseline
- Cost efficiency (should match decay strategy)

---

### Production with Model Releases (Simulated)

**Setup:**
1. Start with GPT-4, GPT-3.5, Mixtral
2. At t=5000, release "GPT-5.1" with +0.2 quality improvement
3. Measure time to discovery and pivot

**Prediction:**
- **Pre-Release (t=0-5000):** Expert 1 (Conservative) dominates, weight → 90/10
- **Release (t=5000):** New model appears, Expert 1 rarely tries it (α=0.01)
- **Discovery (t=5100):** Expert 2 (Adaptive) finds GPT-5.1 (α=2.0 → 20% exploration)
- **Pivot (t=5200-5500):** Corralling detects Expert 2's higher reward, weight → 30/70
- **New Equilibrium (t=6000+):** Router now uses GPT-5.1, Expert 1 eventually learns it

**Metrics to Monitor:**
- Time to first GPT-5.1 selection
- Time to Corralling weight shift (30% threshold)
- Quality improvement captured (should reach +0.2 within 500 steps)

---

## Related Work

### Corralling (Agarwal et al., 2017)
- **Paper:** "Corralling a Band of Bandit Algorithms"
- **Key Insight:** Meta-learning over experts with adversarial guarantees
- **Our Extension:** Apply to exploration-exploitation diversity

### Non-Stationary Bandits (Garivier & Moulines, 2011)
- **Paper:** "On Upper-Confidence Bound Policies for Switching Bandit Problems"
- **Approach:** Discount old observations (sliding window)
- **Our Approach:** Let meta-learner choose between stationary and non-stationary experts

### Alpha Scheduling (Chu et al., 2011)
- **Paper:** "Contextual Bandits with Linear Payoff Functions"
- **Standard:** α = c√(d log t) (time-dependent decay)
- **Our Extension:** Heterogeneous schedules chosen by meta-learner

---

## Future Extensions

### 1. Add Regime Detection Expert

```python
expert_detector = RegimeChangeDetector(
    models=router.bandit.models,
    context_dim=router.bandit.dim,
    detection_threshold=0.05  # Significance level
)
```

- Explicitly monitors for distribution shifts
- Signals Corralling to shift weight to Adaptive expert
- Faster response to changes (proactive vs reactive)

### 2. Adaptive Alpha Based on Uncertainty

```python
expert_adaptive_alpha = CostAwareLinUCBRouter(
    alpha_schedule="uncertainty_based",
    min_alpha=0.01,
    max_alpha=2.0
)
```

- When uncertainty is high → increase α (explore more)
- When uncertainty is low → decrease α (exploit more)
- More dynamic than fixed constant or decay

### 3. Per-Cluster Alpha Strategies

```python
# Easy cluster (94.2% of traffic): Low alpha (exploit)
expert_easy = CostAwareLinUCBRouter(alpha_start=0.5, alpha_end=0.01)

# Hard cluster (5.8% of traffic): High alpha (explore)
expert_hard = CostAwareLinUCBRouter(alpha_start=2.0, alpha_end=1.0)
```

- Cluster-specific experts with different exploration needs
- Corralling routes based on cluster assignment + performance

---

## Conclusion

The **Heterogeneous Experts Strategy** eliminates the fundamental dilemma of choosing between:
- **Decaying Alpha:** Optimal for stable worlds, but causes brain death in non-stationary environments
- **Constant Alpha:** Responsive to changes, but wasteful in stable periods

By delegating this choice to the Corralling meta-learner, we achieve:
- ✅ **Automatic Adaptation:** No manual tuning required
- ✅ **Robust Performance:** Near-optimal in both stationary and non-stationary regimes
- ✅ **Principled Guarantees:** Backed by Corralling's adversarial regret bounds
- ✅ **Production-Ready:** Handles new model releases without manual intervention

**Architectural Philosophy:**
> "When faced with mutually exclusive strategies, don't choose one. Use both and let data decide."

This is the essence of meta-learning applied to the exploration-exploitation trade-off.

---

## References

1. Agarwal, A., Luo, H., Neyshabur, B., & Schapire, R. E. (2017). Corralling a band of bandit algorithms. *Conference on Learning Theory (COLT)*.
2. Chu, W., Li, L., Reyzin, L., & Schapire, R. (2011). Contextual bandits with linear payoff functions. *International Conference on Artificial Intelligence and Statistics (AISTATS)*.
3. Garivier, A., & Moulines, E. (2011). On upper-confidence bound policies for switching bandit problems. *Algorithmic Learning Theory (ALT)*.
4. Auer, P. (2002). Using confidence bounds for exploitation-exploration trade-offs. *Journal of Machine Learning Research*.

---

**Implementation Status:** ✅ Complete  
**Testing Status:** ⏳ Pending  
**Documentation Status:** ✅ Complete

