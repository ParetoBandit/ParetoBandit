# Cost-Aware banditGPT: Proper Pareto Frontier

## The Problem (Reviewer Critique)

### Issue 1: Learning Rate Sweep is Wrong ❌
```python
# WRONG: Sweeping η doesn't create a Pareto frontier
for learning_rate in [0.1, 0.5, 1.0, 5.0, 10.0]:
    reward, cost = banditgpt_routing(..., learning_rate=learning_rate)
```

**Why wrong:**
- η controls **convergence speed**, not cost-quality trade-off
- All values of η converge to similar policies
- Just shows "how fast does it learn", not "what can it achieve"

### Issue 2: No Cost-Aware Policy ❌
```python
# Standard UCB (quality-only)
UCB = expected_reward + α * uncertainty
```

**Why wrong:**
- Optimizes for quality only
- No mechanism to prefer cheaper models
- Can't create cost-quality trade-off curve

## The Solution ✅

### 1. Cost-Aware UCB Formula

```python
# Cost-aware UCB
UCB = expected_reward - λ * cost + α * uncertainty
```

**Parameters:**
- `expected_reward`: Predicted quality
- `λ` (lambda): **Cost penalty** - controls trade-off
- `cost`: Model cost per request
- `α`: Exploration bonus
- `uncertainty`: Confidence interval width

**How it works:**
- **λ = 0**: Pure quality optimization → expensive models
- **λ = 50**: Balanced → mix of models
- **λ = 200**: Cost-conscious → cheap models

### 2. Sweep Cost Penalty λ

```python
# CORRECT: Sweep λ to create Pareto frontier
cost_penalties = [0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]

for lambda_val in cost_penalties:
    reward, cost = banditgpt_routing(..., cost_penalty=lambda_val)
    pareto_points.append((cost, reward))
```

**Expected behavior:**
| λ | Behavior | Cost | Quality |
|---|----------|------|---------|
| 0 | No penalty | High | High |
| 50 | Balanced | Med | Med |
| 200 | Cost-conscious | Low | Low |

This creates a **true Pareto frontier**!

## Implementation Details

### Using Existing Routers ✅

```python
# Use EXISTING CorrallingRouter from router.py
from bandit_gpt.router import CorrallingRouter

# Create cost-aware expert wrappers
warmup_expert = CostAwareLinUCBRouter(
    models=models,
    warmup_priors=warmup_priors,
    model_costs=model_costs,
    cost_penalty=lambda_val  # ← Controls trade-off
)

tabula_rasa_expert = CostAwareTabulaRasaRouter(
    models=models,
    context_dim=context_dim,
    model_costs=model_costs,
    cost_penalty=lambda_val  # ← Same penalty
)

# Combine with existing Corralling
router = CorrallingRouter(
    experts=[warmup_expert, tabula_rasa_expert],
    models=models,
    learning_rate=1.0  # ← Fixed at 1.0
)
```

**Key points:**
1. ✅ Uses existing `CorrallingRouter` (no recreation)
2. ✅ Wraps experts with cost-awareness (minimal wrappers)
3. ✅ Sweeps λ, not η
4. ✅ Fixed η=1.0 (standard convergence)

### Cost-Aware Expert Wrapper

```python
class CostAwareLinUCBRouter:
    """Wraps SimpleLinUCBRouter with cost penalty."""
    
    def select_model(self, context):
        ucb_scores = {}
        for model in self.models:
            # Standard LinUCB components
            theta = A_inv @ b
            expected_reward = theta @ context
            uncertainty = sqrt(context @ A_inv @ context)
            
            # Add cost penalty
            model_cost = self.model_costs[model]["cost"]
            ucb_scores[model] = (
                expected_reward 
                - self.cost_penalty * model_cost  # ← Cost-awareness
                + self.alpha * uncertainty
            )
        
        return max(ucb_scores, key=ucb_scores.get)
```

**What changed:**
- Added `- λ * cost` term to UCB
- Higher λ → penalize expensive models more
- Creates cost-quality trade-off

## Comparison to RouteLLM

### RouteLLM Approach
```python
# Threshold-based routing
if routing_score > threshold:
    use_expensive_model()
else:
    use_cheap_model()

# Sweep threshold to create Pareto frontier
thresholds = [0.0, 0.1, 0.2, ..., 1.0]
```

**Characteristics:**
- ✅ Simple and interpretable
- ✅ Natural Pareto frontier (sweep threshold)
- ❌ Binary decision (no learning)
- ❌ No adaptation over time

### banditGPT Approach
```python
# Cost-aware UCB
UCB = reward - λ * cost + α * uncertainty

# Sweep cost penalty to create Pareto frontier
lambdas = [0.0, 5.0, 10.0, ..., 200.0]
```

**Characteristics:**
- ✅ Learns from feedback
- ✅ Adapts to data distribution
- ✅ Natural Pareto frontier (sweep λ)
- ✅ Probabilistic (explores multiple models)
- ⚠️ More complex

## Expected Results

### Pareto Frontier Shape

```
Quality (Reward)
    ^
1.0 |        * Oracle
    |       /
0.9 |      /  
    |     /  banditGPT (λ sweep)
0.85|    /  /  
    |   /  /  RouteLLM (threshold sweep)
0.8 |  /  /___
    | /  /
    |/__/________> Cost ($)
   0.0  0.005  0.01
```

### Example Points

**RouteLLM:**
- Threshold 0.0: All GPT-4 → Cost=$0.013, Reward=0.812
- Threshold 0.5: Mixed → Cost=$0.005, Reward=0.870
- Threshold 1.0: All Mixtral → Cost=$0.0003, Reward=0.823

**banditGPT:**
- λ=0: Quality-focused → Cost=$0.011, Reward=0.890
- λ=50: Balanced → Cost=$0.005, Reward=0.860
- λ=200: Cost-focused → Cost=$0.001, Reward=0.830

## Why This is Correct

### 1. λ Controls Trade-Off ✅
- **Not a convergence parameter** (like η)
- **Directly affects routing decisions**
- **Creates different policies** at different values

### 2. Creates True Pareto Frontier ✅
- Each λ value → different strategy
- Low λ → expensive, high quality
- High λ → cheap, lower quality
- Smooth curve from one extreme to other

### 3. Uses Existing Code ✅
- Leverages `CorrallingRouter` from `router.py`
- Minimal wrappers for cost-awareness
- No code duplication

### 4. Fair Comparison ✅
- RouteLLM sweeps threshold
- banditGPT sweeps cost penalty
- Both create valid Pareto frontiers
- Both optimize different objectives at different settings

## Summary

**Before (Wrong):**
- ❌ Swept learning rate η (convergence speed)
- ❌ No cost-aware policy
- ❌ Recreated router code

**After (Correct):**
- ✅ Sweep cost penalty λ (trade-off parameter)
- ✅ Cost-aware UCB formula
- ✅ Uses existing CorrallingRouter
- ✅ Creates true Pareto frontier

**Key Formula:**
```
UCB = expected_reward - λ * cost + α * uncertainty
       ↑                ↑              ↑
    Quality      Cost penalty    Exploration
```

This is the **correct** way to generate a Pareto frontier for banditGPT!

