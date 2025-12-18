# LLM Jury Architecture

This document explains how the async bandit router selects the optimal LLM model for each prompt.

## Overview

LLM Jury uses a **contextual bandit** (LinUCB) to route prompts to the best model based on:
- **Quality**: Learned from past interactions
- **Cost**: Token pricing from the model registry
- **Latency**: Estimated response time

The system operates in two paths:
- **Hot Path** (Router): Millisecond-scale model selection
- **Cold Path** (Grader): Asynchronous quality assessment for learning

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

The bandit updates its parameters so future similar prompts will route better.

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
  λ_cost ≈ 50                          # Cost sensitivity
  λ_latency ≈ 0.05                     # Latency sensitivity (per second)
```

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
from llm_jury.async_bandit import BanditRouter, PriorManager

# Create router with automatic prior detection
router = BanditRouter.create(model_registry, priors="merged")

# Get the best model for a prompt
model_id, log = router.route("Write a recursive fibonacci function")

# Get top-k recommendations
recommendations = router.rank_prompt("Explain quantum computing", top_k=5)
```
