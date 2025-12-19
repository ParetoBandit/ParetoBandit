# BanditGPT Architecture

This document explains how the async bandit router selects the optimal LLM model for each prompt.

## Overview

BanditGPT uses a **contextual bandit** (LinUCB) to route prompts to the best model based on:
- **Quality**: Learned from past interactions
- **Cost**: Token pricing from the model registry
- **Latency**: Estimated response time

The system operates in two paths:
- **Hot Path** (Router): Millisecond-scale model selection
- **Cold Path** (Grader): Asynchronous quality assessment for learning

## Why "Bandit"?

The name comes from the slang term for a slot machine: the **"One-Armed Bandit."**

### The Casino Analogy

Imagine you are in a casino with 80 slot machines (models). You have a limited budget (prompts). Your goal is to figure out which machine pays out the most money (Quality), but you have to pay every time you pull the handle.

**The Problem**: Do you keep pulling the handle of the machine that paid you $5 yesterday (**Exploit**)? Or do you risk money on a machine you've never touched, hoping it pays $100 (**Explore**)?

**The Context**: In our case, it's a "**Contextual Bandit**" because the "winning machine" changes based on the weather (the Prompt). GPT-4 might win for poetry, but Llama-3 might win for code.

### Where is the Randomness?

The utility formula `U = Q - C - L` looks like a deterministic equation. So where's the "gambling"?

In sophisticated bandits like LinUCB, we don't use random dice rolls (like Epsilon-Greedy). Instead, we use **"Optimism in the Face of Uncertainty"**:

```
Q_final = μ(x) + α · σ(x)
          ────   ─────────
          Mean   Uncertainty Bonus
```

- **μ (Mean)**: "I think Llama-3's quality is 0.70."
- **σ (Uncertainty)**: "But I haven't used Llama-3 much, so I might be wrong by ±0.40."
- **α (Alpha)**: Your "Curiosity Setting" (e.g., 1.0)

**Example calculation:**

| Model | Mean (μ) | Uncertainty (σ) | α | Score |
|-------|----------|-----------------|---|-------|
| Llama-3 | 0.70 | 0.40 | 1.0 | 0.70 + 0.40 = **1.10** |
| GPT-4 | 0.95 | 0.01 | 1.0 | 0.95 + 0.01 = **0.96** |

**Result**: The router picks Llama-3 (1.10) over GPT-4 (0.96).

- Is it random? **No**, it's calculated.
- Does it look random? **Yes**. To a user, it looks like the router "randomly" picked a worse model.
- Why did it do it? Because the math said: "The potential upside of Llama-3 being amazing is worth the risk."

### Why LinUCB Over True Randomness

Old-school bandits (Epsilon-Greedy) literally flip a coin: "10% of the time, pick a random model."

**Problem**: This is dangerous. It might randomly route a "Medical Diagnosis" prompt to a tiny 1B model just for fun.

**LinUCB is safer:**
- It only explores when there is **statistical ambiguity**
- It will **never** randomly pick a model it knows is bad (low mean, low uncertainty)
- It only picks models it doesn't know enough about (high uncertainty)

### Code Verification

This is exactly what the code does:

```python
# From bandit_router.py
theta = self.A_inv[m] @ self.b[m]      # θ = learned weights
mean = float(theta.dot(x))              # μ = θ · x
var = float(x.dot(self.A_inv[m]).dot(x))
std = float(np.sqrt(max(var, 1e-12)))   # σ = √(x'A⁻¹x)
ucb = mean + self.alpha * std           # Q = μ + α·σ
```

### Summary

- **"Bandit"**: Because you are gambling on which model gives the best reward
- **"Randomness"**: Replaced by **uncertainty**. The router "hallucinates" that unknown models are better than they are, forcing it to test them
- **Safety**: Unlike coin-flip exploration, LinUCB never picks a model it *knows* is bad

## How Routing Works: Prompt → Prediction

### 1. The Mapper: Embedding Model

When a user sends the prompt "Write a Python script to parse JSON," the system does **not** categorize it as "Coding" using rules or keywords.

Instead, the **Sentence Transformer** (`all-MiniLM-L6-v2`) converts that text into a list of 384 numbers — the **Context Vector** (x):

```
Prompt: "Write Python..."
Vector (x): [0.05, -0.92, 0.44, 0.12, -0.33, ...]  (384 dimensions)
```

This vector describes the prompt's position in **"Meaning Space."** The numbers essentially encode:
- "This is very close to Coding"
- "This is far from Poetry"
- "This is somewhat close to Logic"

The embedding captures semantic meaning, so "Write Python code" and "Create a Python script" produce similar vectors.

### 2. The Memory: Weight Vector (θ)

Each model (e.g., Llama-3) has its own **Weight Vector** (θ) that acts as its "Profile."

```
θ_llama3 = [0.8, 0.2, 0.9, -0.1, ...]  (384 dimensions)
```

This vector was **learned** during the prior generation phase (archetype grid or synthetic warmup). Because Llama-3 performed well on coding prompts during training, its weights are high in the dimensions that correspond to coding.

Mathematically, θ is computed from the bandit's learned parameters:

```
θ = A⁻¹ @ b
```

Where:
- **A**: Covariance matrix (tracks feature correlations)
- **b**: Reward accumulator (tracks which features predict success)

### 3. The Lookup: Dot Product

The bandit calculates the **predicted quality** using a simple dot product:

```
predicted_score = θ · x = Σ(θᵢ × xᵢ)
```

**If the vectors align** (prompt asks for Python, Llama-3's weights "point" towards Python):
- The math produces a **high score** (e.g., 0.95)
- The model is a good match

**If they oppose** (prompt asks for French History, Llama-3's weights point away from History):
- The math produces a **low score** (e.g., 0.20)
- The model is a poor match

### 4. The Decision: UCB with Utility

We don't just pick the model with the highest predicted score. We also consider:

1. **Exploration Bonus** (UCB): Try uncertain models to learn about them
2. **Cost Penalty**: Expensive models get a penalty
3. **Latency Penalty**: Slow models get a penalty

```
UCB = (θ·x + prior) + α·√(x'A⁻¹x)
          ↑              ↑
     exploitation    exploration

Utility = UCB - λ_cost·Cost - λ_latency·Latency
```

The model with the **highest Utility** wins.

## Visual Example

```
Prompt: "Write a recursive fibonacci function in Python"
                    ↓
         ┌─────────────────────┐
         │  Sentence Transformer │
         │  (all-MiniLM-L6-v2)   │
         └──────────┬───────────┘
                    ↓
         x = [0.9, 0.7, 0.3, ...]  (384-dim "coding" vector)
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
┌────────┐    ┌────────┐      ┌────────┐
│ GPT-4o │    │ Llama-3│      │Claude-3│
│ θ·x=0.8│    │ θ·x=0.9│      │ θ·x=0.7│
│ Cost=$2│    │ Cost=$1│      │Cost=$1.5│
└────────┘    └────────┘      └────────┘
    ↓               ↓               ↓
U = 0.7       U = 0.85        U = 0.6
                    ↓
              ✓ WINNER: Llama-3
```

## The Learning Loop

After the model responds, the **Grader** (TieredGrader) evaluates the quality:

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ User Prompt  │ →  │ LLM Response│ →  │ TieredGrader │
└──────────────┘    └─────────────┘    └──────┬───────┘
                                              ↓
                                       reward = 0.92
                                              ↓
                                    ┌─────────────────┐
                                    │  Bandit Update  │
                                    │  A ← A + x·x'   │
                                    │  b ← b + r·x    │
                                    └─────────────────┘
```

### The Rank-One Update (Real-Time Learning)

When feedback arrives, the library does **not** retrain a neural network (which would take hours). Instead, it performs a simple **Rank-One Update** that takes **microseconds**.

Each model has two variables representing its "brain":

| Variable | Shape | Purpose |
|----------|-------|---------|
| **A** (Confidence Matrix) | 384×384 | Where in vector space we have data |
| **b** (Reward Vector) | 384×1 | How much reward we got in each direction |

When feedback arrives with prompt vector `x` and reward `r`:

```
A_new = A_old + (x · x')    # Outer product
b_new = b_old + (r · x)     # Scaled direction
```

**What this physically does:**

- **A Update**: Adds the "shape" of the prompt to the matrix. This tells the router: *"I have now seen this type of request. I am no longer guessing."* (Uncertainty ↓)

- **b Update**: Pushes the model's weight vector based on the reward:
  - If reward is **+1.0**: Push θ *towards* this prompt type. *"Do more of this."*
  - If reward is **-1.0**: Push θ *away* from this prompt type. *"Never do this again."*

### Code Verification

This is exactly what the code does:

```python
# From DisjointLinUCBPolicy.update()
def update(self, model: str, x: np.ndarray, reward: float) -> None:
    # 1. Update Confidence Matrix (A) - "I've seen this type of prompt"
    self.A[model] += np.outer(x, x)   # A_new = A_old + x·x'
    
    # 2. Update Reward Vector (b) - "This worked well/poorly"
    self.b[model] += reward * x       # b_new = b_old + r·x
    
    # 3. Periodically recompute inverse for speed
    if self._updates[model] % self.recompute_inv_every == 0:
        self.A_inv[model] = np.linalg.inv(self.A[model])
```

### Persistence: User State vs Bundled Priors

The library **never overwrites** the read-only `shippable_priors.npz` that ships with the package.

Instead, it creates a separate `user_priors.npz`:

| File | Location | Writable | Purpose |
|------|----------|----------|---------|
| `shippable_priors.npz` | `<package>/data/priors/` | No | Library defaults (frozen) |
| `user_priors.npz` | `~/.banditgpt/priors/` | Yes | User's learned updates |

**On startup:** Load bundled priors, then overlay user state if it exists.
**On shutdown:** Save current matrices to user state.

This ensures users never lose their training, even when upgrading the library.

### The Result: Next Request

The very next time `route()` is called:

1. It uses the **new A** and **new b**
2. Because **b** was nudged towards the prompt direction, `θ·x` gives a higher score for similar prompts
3. Because **A** was increased, uncertainty **σ** is lower (more confident)

**Summary:** The update is simply adding numbers to a saved state file. It is efficient, instant, and permanent.

## Prior Generation

To avoid cold-start issues, we pre-train the bandit on representative prompts:

### Archetype Grid Strategy

1. **Cluster** 50,000 prompts into 500 representative "archetypes"
2. **Dense Run**: All 81 models × 500 archetypes = 40,500 evaluations
3. **Grade** each response with TieredGrader
4. **Export** compressed priors (`shippable_priors.npz`)

This gives the router "day-1 intelligence" about which models excel at which tasks.

### Why 500 Clusters?

Research shows that LLM task diversity saturates around 500-1,000 archetypes:

- **~50 clusters**: Broad topics (Coding, Math, History)
- **~500 clusters**: Specific tasks (Python Debugging, SQL, Calculus)
- **~5,000 clusters**: Specific entities (Django v4, Taylor Swift)

For routing, the **task level (~500)** is optimal. A model good at "Django v4" is almost certainly good at "Django v5."

## Utility Equation

The full utility equation used by the router:

```
U_model = q̂_model - λ_cost·Cost - λ_latency·Latency

where:
  q̂ = (θ·x + prior) + α·√(x'A⁻¹x)    # Quality estimate with exploration
  λ_cost = cost penalty weight         # Quality per dollar
  λ_latency = latency penalty weight   # Quality per second
```

## Optimization Profiles

The weights `λ_cost` and `λ_latency` act as **exchange rates** converting money and time into "quality points":

- **λ_cost**: "How much quality am I willing to sacrifice to save $1.00?"
- **λ_latency**: "How much quality am I willing to sacrifice to save 1 second?"

### Named Presets

Instead of tuning raw floats, use named profiles:

| Profile | λ_cost | λ_latency | Behavior |
|---------|--------|-----------|----------|
| `quality_first` | 0.1 | 0.05 | Maximize quality, ignore cost |
| `balanced` | 10.0 | 0.10 | Reasonable trade-off |
| `cost_saver` | 50.0 | 0.20 | Aggressive cost optimization |
| `low_latency` | 1.0 | 0.50 | Prioritize speed |

### Usage

```python
# Using a named profile (recommended)
model, log = router.route("Write code", profile="balanced")

# Using explicit weights (power users)
model, log = router.route("Write code", lambda_cost=25.0, lambda_latency=0.15)
```

### CLI

```bash
# Use a profile
python -m banditgpt.async_bandit.cli recommend \
    --prompt "Write a fibonacci function" \
    --profile balanced

# Override with explicit weights
python -m banditgpt.async_bandit.cli recommend \
    --prompt "Write a fibonacci function" \
    --lambda-cost 25.0 --lambda-latency 0.15
```

### The "Penalty" Analogy

Think of weights as **penalties**:

- Quality is 0.0 to 1.0 (a score of 0.9 is an A-grade student)
- **Cost Penalty**: If λ_cost = 20, for every $0.01 spent, penalize the score by 0.2

## Exploration Rate

The exploration rate (α) controls the router's **risk appetite** — how often it tries unproven models to discover better options.

### Why Different Users Need Different Settings

| User Type | Need | Setting |
|-----------|------|---------|
| **Day-1 User** | "Test all models to learn what works. I don't care about a few bad answers today." | `aggressive` |
| **Production App** | "Reasonable exploration, but don't break things." | `safe` |
| **Fintech/Bank** | "NEVER route to an unproven model. Zero risk." | `static` |

If you hardcode α=2.0, the Bank User will uninstall your library.
If you hardcode α=0.0, the Day-1 User's router will never learn.

### Named Presets

| Setting | Alpha | Behavior |
|---------|-------|----------|
| `static` | 0.0 | Pure exploitation. Trust mean only. (Bank Mode) |
| `safe` | 0.1 | Minimal exploration. Only explore if upside is huge. (**DEFAULT**) |
| `balanced` | 0.5 | Standard bandit behavior. |
| `aggressive` | 2.0 | Try everything! Fast learning. (Day-1/Shadow Mode) |

### Usage

```python
# At router creation
router = BanditRouter.create(registry, exploration="safe")

# Override per-request
model, log = router.route("Analyze risk", exploration="static")

# Day-1 calibration mode
model, log = router.route("Test this", exploration="aggressive")
```

### CLI

```bash
# Safe exploration (production default)
python -m banditgpt.async_bandit.cli recommend \
    --prompt "Analyze risk" \
    --exploration safe

# Aggressive exploration (calibration)
python -m banditgpt.async_bandit.cli recommend \
    --prompt "Test this" \
    --exploration aggressive
```

### Shadow Mode Tip

If running in "shadow mode" (responses logged but not shown to users), automatically boost to `aggressive`:

```python
exploration = "aggressive" if shadow_mode else "safe"
model, log = router.route(prompt, exploration=exploration)
```

Since no real user sees the answer, you should explore wildly to learn faster
- **Latency Penalty**: If λ_latency = 0.2, for every second waited, penalize the score by 0.2

## Code References

| Component | File | Description |
|-----------|------|-------------|
| Router | `bandit_router.py` | Main BanditRouter class |
| Policy | `bandit_router.py` | DisjointLinUCBPolicy |
| Grader | `tiered_grader.py` | TieredGrader (soft + hard) |
| Priors | `judge.py` | PriorManager |
| CLI | `cli.py` | Unified command-line interface |

## Quick Start

```python
from banditgpt.async_bandit import BanditRouter, OptimizationProfile

# Create router with automatic prior detection
router = BanditRouter.create(model_registry, priors="merged")

# Route using a named profile (recommended)
model_id, log = router.route("Write a fibonacci function", profile="balanced")

# Route with explicit weights (power users)
model_id, log = router.route(
    "Write a fibonacci function",
    lambda_cost=25.0,
    lambda_latency=0.15,
)

# Get top-k recommendations
recommendations = router.rank_prompt(
    "Explain quantum computing",
    top_k=5,
    profile="quality_first",
)

# List available profiles
print(OptimizationProfile.list_profiles())
# ['quality_first', 'balanced', 'cost_saver', 'low_latency']
```
