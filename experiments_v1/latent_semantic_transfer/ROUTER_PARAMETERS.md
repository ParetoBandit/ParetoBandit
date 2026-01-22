# Router Parameters & Profiles

## TL;DR

**For the Regret Waterfall Experiment**, we use:
- **`alpha = 0.05`**: UCB exploration bonus (controls exploration vs exploitation)
- **`init_lambda = 1.0`**: Ridge regularization strength
- **No profile/weights**: Direct UCB selection (pure bandit, no economic utility)

**For Production Routing** (via `router.route()`), the default profile is:
- **`profile = "auto"`**: Pareto-optimal routing
- **`lambda = 0.02`**: Quality-cost trade-off parameter
- **`exploration_alpha = 2.0`**: UCB exploration constant (Pareto mode)

---

## Experiment Parameters (Regret Waterfall)

### What We Use

```python
router = BanditRouter(
    model_registry=registry,
    alpha=0.05,           # ← UCB exploration parameter
    init_lambda=1.0,      # ← Ridge regularization
    verbose_routing=False
)

# Direct bandit selection (no profile/utility function)
selected_model, ucb_score = router.bandit.select_arm(context)
```

### Parameter Definitions

#### 1. `alpha = 0.05` (UCB Exploration)

**What it controls**: The exploration-exploitation trade-off in UCB.

**Formula**:
```
UCB(model, context) = θ^T · x + α · √(x^T · A^{-1} · x)
                      └─mean─┘   └──uncertainty───┘
```

**Interpretation**:
- **α = 0.0**: Pure exploitation (always pick model with highest mean reward)
- **α = 0.05**: Modest exploration (slightly favor uncertain models)
- **α = 2.0**: High exploration (strongly favor uncertain models)

**Why 0.05?**
- Standard value in LinUCB literature
- Balances exploration (trying uncertain models) and exploitation (using best known model)
- Too low → under-explore (miss better models)
- Too high → over-explore (waste samples on bad models)

---

#### 2. `init_lambda = 1.0` (Ridge Regularization)

**What it controls**: Prior strength and initial uncertainty.

**Where it's used**:
```python
A_init = λ · I  # Initial covariance matrix (identity scaled by lambda)
```

**Interpretation**:
- **λ = 0.1**: Weak regularization (high initial uncertainty → more exploration)
- **λ = 1.0**: Moderate regularization (balanced initialization)
- **λ = 10.0**: Strong regularization (low initial uncertainty → more exploitation)

**Why 1.0?**
- Standard choice in ridge regression
- Provides reasonable "prior confidence" before seeing data
- Not too weak (unstable estimates) or too strong (over-regularized)

**Relationship to uncertainty**:
```
Uncertainty = √(x^T · A^{-1} · x)
            = √(x^T · (λI)^{-1} · x)  [initially]
            = √(1/λ) · ||x||

Higher λ → Lower initial uncertainty → Less exploration
```

---

#### 3. No Profile (Pure Bandit)

**What we DON'T use in the experiment**:
- ❌ Quality/cost/latency weights
- ❌ Pareto optimization
- ❌ Economic utility functions

**Why?**
The Regret Waterfall is designed to measure **pure reward maximization**:
- Oracle = best reward (regardless of cost)
- Regret = missed reward (not missed utility)
- Goal: Show that LST learns faster, not that it's more cost-efficient

This isolates the **learning efficiency** of LST from economic trade-offs.

---

## Production Parameters (router.route())

### Default Configuration

```python
model_id, log = router.route(
    prompt,
    profile="auto"  # ← Default: Pareto-optimal routing
)
```

### Profile System

#### 1. Default Profile: `"auto"`

**Parameters**:
```python
lambda = 0.02            # Quality-cost trade-off
exploration_alpha = 2.0  # UCB exploration (Pareto mode)
```

**Utility Function**:
```
Utility(model) = Quality - (λ · Cost)
               = (μ_quality + α · σ_quality) - (0.02 · Cost)
```

**Interpretation**:
- **λ = 0.02**: "I'm willing to pay $1 extra for 50% quality improvement"
- Or equivalently: "Quality improvement of 2% is worth $1"

**Behavior**:
- Cheap models win **easy tasks** (where quality gap is small)
- Expensive models win **hard tasks** (where quality gap is large)
- Naturally Pareto-optimal: never picks strictly dominated models

---

#### 2. Custom Profile

```python
model_id, log = router.route(
    prompt,
    profile={
        "w_q": 10.0,  # Quality weight
        "w_c": 1.0,   # Cost weight (base currency)
        "w_l": 0.5    # Latency weight
    }
)
```

**Utility Function**:
```
Utility = (w_q · Quality) - (w_c · Cost) - (w_l · Latency)
```

**Example Interpretations**:

| Profile | Meaning |
|---------|---------|
| `{w_q: 10.0, w_c: 1.0, w_l: 0.0}` | "10% quality gain worth $1, ignore latency" |
| `{w_q: 5.0, w_c: 1.0, w_l: 2.0}` | "5% quality gain OR 0.5s latency reduction worth $1" |
| `{w_q: 1.0, w_c: 1.0, w_l: 0.0}` | "Cost-quality equality (100% quality gain worth $1)" |

---

### Exploration Constants

#### 1. UCB Exploration (`alpha`)

**Experiment value**: `0.05`

```python
router = BanditRouter(alpha=0.05)
```

This is used in **direct bandit selection** (not Pareto mode).

---

#### 2. Pareto Exploration (`exploration_alpha`)

**Production value**: `2.0`

```python
self.PARETO_EXPLORATION_CONSTANT = 2.0
```

**Why 2.0 > 0.05?**
- Pareto mode uses a **two-stage filter**: 
  1. **Gatekeeper**: Filter out non-competitive models (uses UCB with α=2.0)
  2. **Judge**: Select best from survivors (uses UCB with α=2.0)
- Higher α ensures uncertain models get a chance to compete
- Prevents "Explore-then-Exploit Disconnect" (KDD fix)

**Effect**:
```
α = 0.05 → Conservative exploration (favor proven models)
α = 2.0  → Generous exploration (give uncertain models a chance)
```

---

## Summary Table

| Parameter | Experiment | Production | Purpose |
|-----------|------------|------------|---------|
| **α (UCB)** | 0.05 | 0.05 | Exploration bonus (direct bandit mode) |
| **λ (init)** | 1.0 | 1.0 | Ridge regularization strength |
| **λ (Pareto)** | N/A | 0.02 | Quality-cost trade-off |
| **α (Pareto)** | N/A | 2.0 | UCB exploration (Pareto mode) |
| **Profile** | None (pure UCB) | "auto" | Economic utility function |

---

## Why Different Parameters for Experiment vs Production?

### Experiment (Regret Waterfall)

**Goal**: Measure **learning efficiency**
- Pure reward maximization (no cost/latency trade-offs)
- Isolate the effect of LST initialization
- Scientific comparison: Cold Start vs LST

**Design**:
- Direct UCB selection (α=0.05)
- Binary rewards (0 or 1)
- Oracle = max reward (cost-agnostic)

---

### Production (router.route())

**Goal**: Optimize **economic utility**
- Balance quality, cost, and latency
- Pareto-optimal model selection
- Real-world routing decisions

**Design**:
- Pareto utility function (λ=0.02)
- Higher exploration (α=2.0) to avoid premature convergence
- Context-aware routing (exploit prompt features)

---

## How LST Uses These Parameters

### Initialization

When GPT-5 is registered via LST:

1. **Find semantic neighbor** (GPT-4-Turbo)
2. **Compute similarity** (cosine of model DNA embeddings)
3. **Set `n_effective`** based on similarity:

```python
if similarity > 0.8:
    n_effective = 5.0   # Strong match (empirically optimal)
elif similarity > 0.6:
    n_effective = 3.0   # Moderate match
else:
    n_effective = 1.0   # Weak match (revert to cold start)
```

4. **Initialize prior**:

```python
A_new = init_lambda · I                        # Reset uncertainty (λ=1.0)
b_new = (init_lambda · θ_neighbor) · n_eff     # Scaled knowledge transfer
```

### Why This Works

**Initial θ estimate**:
```
θ_initial = A^{-1} · b 
          = (λI)^{-1} · (λ · θ_neighbor · n_eff)
          = θ_neighbor · n_eff
```

**Effect**:
- **Direction**: Same as neighbor (inherits preferences)
- **Magnitude**: Scaled by `n_eff` (prior strength)
- **Uncertainty**: High (A is reset to λI, not inherited)

**Result**:
- GPT-5 starts with GPT-4-Turbo's intuition
- But maintains high uncertainty → willing to explore
- UCB will initially favor GPT-5 (high mean + high uncertainty)
- As data accumulates, uncertainty shrinks → converges to true performance

---

## Key Takeaways

1. **Experiment uses pure UCB** (α=0.05, no profile)
   - Goal: Measure learning efficiency (regret)
   - LST reduces regret by 52.8% (14.4 → 6.8)

2. **Production uses Pareto optimization** (λ=0.02, α=2.0)
   - Goal: Optimize economic utility (quality - cost)
   - LST provides zero-day utility (no warmup needed)

3. **LST is parameter-agnostic**
   - Works with any α, λ, or profile
   - Core idea: Transfer semantic knowledge via `n_effective`
   - Benefits are **additive** to baseline UCB/Pareto performance

4. **Empirically tuned `n_effective`** via hyperparameter sweep:
   - High similarity (>0.8): `n_eff = 5.0` (optimal)
   - Moderate similarity (0.6-0.8): `n_eff = 3.0`
   - Low similarity (<0.6): `n_eff = 1.0` (safety fallback)

---

## References

- Router code: `src/bandit_gpt/router.py`
- Experiment: `experiments_v1/latent_semantic_transfer/regret_waterfall_v2.py`
- Hyperparameter sweep: `experiments_v1/latent_semantic_transfer/sweep_n_eff.py`
- Sweep results: `experiments_v1/latent_semantic_transfer/SWEEP_FINDINGS.md`

