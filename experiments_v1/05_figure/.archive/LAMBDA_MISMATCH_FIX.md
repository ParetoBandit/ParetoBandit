# The λ-Penalty Mismatch: Critical Flaw and Fix

## The Critical Flaw

**What was wrong:** Training with λ=0, then evaluating with λ=specified

```python
# WRONG IMPLEMENTATION
# Phase 1: Train with λ=0 (learn pure rewards)
for prompt_data in train_data:
    selected_model = router.select_model(context)  # Using λ=0
    reward = rewards[selected_model]
    router.update(context, selected_model, reward)

# Phase 2: Evaluate with λ=5.0 (suddenly apply cost penalty)
warmup_expert.cost_penalty = 5.0  # ❌ Experts never trained for this!
tabula_rasa_expert.cost_penalty = 5.0
for prompt_data in eval_data:
    selected_model = router.select_model(context)  # Using λ=5.0
```

## Why This Failed

**Analogy:** Training a driver on a race track for pure speed, then handing them a fuel-efficiency manual during the final race without practice.

**Technical Issue:**
1. The experts' reward estimates (θ) were calibrated for **maximizing quality**
2. During evaluation, we asked them to **maximize (quality - λ × cost)**
3. The utility function changed, but the learned parameters didn't adapt

**Observed Symptoms:**
- Non-monotonic curve (λ=2.0 outperforming λ=1.0)
- Huge gap in Pareto frontier ($0.003 → $0.009 with no intermediate points)
- High variance between trials

## The Correct Protocol

**Each λ value gets its own trained model:**

```python
# CORRECT IMPLEMENTATION
def banditgpt_hybrid_routing(..., cost_penalty):
    # Phase 1: Train WITH cost penalty λ
    warmup_expert = CostAwareLinUCBRouter(..., cost_penalty=cost_penalty)
    tabula_rasa_expert = CostAwareTabulaRasaRouter(..., cost_penalty=cost_penalty)
    
    for prompt_data in train_data:
        selected_model = router.select_model(context)  # ✓ Using λ
        reward = rewards[selected_model]
        router.update(context, selected_model, reward)  # ✓ Learn cost-aware utility
    
    # Phase 2: Evaluate with SAME λ (no updates)
    for prompt_data in eval_data:
        selected_model = router.select_model(context)  # ✓ Using same λ
        # NO UPDATE - pure evaluation
```

## What This Fixes

1. **Monotonic Curve**: Each λ learns its own cost-quality trade-off
2. **Smoother Pareto Frontier**: No sudden jumps between points
3. **Lower Variance**: Model is calibrated for the λ it's evaluated on
4. **Fair Comparison**: Like RouteLLM, each "model" (λ value) is trained then tested

## Expected Results

**Before Fix (λ mismatch):**
- Gap: $0.003 → $0.009 (no intermediate points)
- banditGPT @ 0.87 quality: $0.0093 (23% MORE than RouteLLM)

**After Fix (λ consistency):**
- Smooth curve: Each λ produces a calibrated cost-quality point
- banditGPT should dominate across budget tiers
- Cost-aware training enables better exploration-exploitation

## Key Insight

**The bandit needs to "practice" with the cost penalty it will be evaluated on.**

Just like:
- You can't train a chess AI with one evaluation function and expect it to optimize a different one
- You can't train a student for math exams and expect them to ace physics tests

Each budget constraint (λ) requires its own training regime.

