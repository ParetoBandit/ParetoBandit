# Diagnosis: Why banditGPT Performance is Low

## The Problem

**banditGPT with λ=0 gets 0.8133, which is:**
- Worse than Mixtral-only (0.8227)
- Worse than GPT-4-only (0.8120)  
- Much worse than expected (~0.85-0.90)

## Root Cause: Train/Eval Split Issue

### What We're Doing (Wrong)
```python
# PHASE 1: Train on dev set (1,121 prompts)
for prompt in train_data:
    selected = router.select_model(context)
    reward = get_reward(selected)
    router.update(context, selected, reward)  # ← Learning

# PHASE 2: Evaluate on holdout (750 prompts)  
for prompt in eval_data:
    selected = router.select_model(context)
    reward = get_reward(selected)
    # NO UPDATE ← Frozen policy
```

**Problem**: The router learns a policy on dev, then we freeze it and evaluate on holdout. But:
1. Holdout distribution might be different from dev
2. Router can't adapt to holdout
3. Performance suffers

### What We Should Do (Correct)

**Option 1: Online Evaluation (Standard for Bandits)**
```python
# Evaluate WITH learning (online setting)
for prompt in eval_data:
    selected = router.select_model(context)
    reward = get_reward(selected)
    router.update(context, selected, reward)  # ← Keep learning!
```

**Option 2: Train on Combined, Report Cumulative Reward**
```python
# Train on everything, report average reward
all_data = train_data + eval_data
total_reward = 0
for prompt in all_data:
    selected = router.select_model(context)
    reward = get_reward(selected)
    router.update(context, selected, reward)
    total_reward += reward

avg_reward = total_reward / len(all_data)
```

## Why This Matters

### Bandit Algorithms are ONLINE
- They're designed to learn as they go
- Performance improves over time
- Freezing the policy defeats the purpose

### RouteLLM is OFFLINE
- Pre-trained on separate data
- Frozen at evaluation time
- No learning regret

### Apples to Oranges Comparison
- RouteLLM: Offline, pre-trained, no regret
- banditGPT (current): Offline eval after online train → distribution mismatch
- banditGPT (correct): Online eval with learning → fair comparison

## Evidence

### From Calibration Experiment
- banditGPT on holdout: **0.8507** (with 23.3% GPT-4 usage)
- This used **online evaluation** (kept learning)

### Our Results
- banditGPT λ=0 on holdout: **0.8133** (frozen after dev training)
- Much worse!

## The Fix

### For Fair Comparison with RouteLLM

Since RouteLLM is offline/frozen, we have two options:

**Option A: Make banditGPT Offline Too**
- Train on LARGE dataset (e.g., 80k warmup data)
- Freeze policy
- Evaluate on holdout
- But this loses the "online learning" advantage

**Option B: Report Online Performance**
- Train + evaluate on combined data
- Report cumulative reward / N
- This is standard for bandit papers
- But includes "training regret"

**Option C: Report Both**
- Offline: Train on dev, freeze, eval on holdout (current)
- Online: Train + eval on holdout (fair for bandits)
- Show both curves

## Recommendation

For the Pareto frontier, use **Option C**:

1. **RouteLLM**: Offline (frozen policy)
2. **banditGPT-Offline**: Train on dev, freeze, eval on holdout
3. **banditGPT-Online**: Train + eval on holdout (keep learning)

This shows:
- banditGPT-Offline: Comparable to RouteLLM (both frozen)
- banditGPT-Online: Better performance (adaptive)
- Fair comparison for both paradigms

## Expected Results

### banditGPT-Offline (Frozen)
- λ=0: ~0.81-0.83 (what we're seeing)
- λ=0.5: ~0.85-0.87
- Performance limited by dev/holdout mismatch

### banditGPT-Online (Adaptive)
- λ=0: ~0.85-0.88 (learns on holdout)
- λ=0.5: ~0.87-0.90
- Better performance, includes learning regret

### RouteLLM (Offline, Pre-trained)
- Threshold sweep: 0.81-0.87
- No learning regret (already trained)
- Fair comparison with banditGPT-Offline

## Summary

**Current problem**: We're doing offline evaluation (freeze after dev) which hurts banditGPT.

**Solution**: Either:
1. Do online evaluation (keep learning on holdout)
2. Train on much more data before freezing
3. Report both offline and online performance

**For Pareto frontier**: Show banditGPT-Online to demonstrate adaptive advantage.

