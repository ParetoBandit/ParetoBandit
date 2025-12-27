#!/usr/bin/env python3
"""Quick diagnostic: Does the router learn from feedback?"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bandit import BanditRouter
import numpy as np

# Load some test data
base_dir = Path(__file__).parent / "data"
prompts = []
with open(base_dir / "train_prompts.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        prompts.append(json.loads(line))

# Load rewards
reward_lookup = {}
with open(base_dir / "train_rewards.jsonl") as f:
    for line in f:
        data = json.loads(line)
        prompt = data['prompt']
        model = data['model_id']
        reward_logit = data['reward_logit']
        reward = 1.0 / (1.0 + np.exp(-reward_logit))
        if prompt not in reward_lookup:
            reward_lookup[prompt] = {}
        reward_lookup[prompt][model] = reward

print("Testing router learning...")
print("="*60)

# Create router with benchmark priors
router = BanditRouter.create(priors="benchmark", cluster_boost_weight=0.0)

print(f"\n✓ Router created with {len(router.registry)} models")
print(f"✓ Using priors: benchmark")
print(f"✓ Cluster boost weight: {router.cluster_boost_weight}")

# Test on first 20 prompts
optimal_selections_before = 0
optimal_selections_after = 0
total = 0

for prompt_data in prompts[:20]:
    prompt = prompt_data['prompt']
    rewards = reward_lookup.get(prompt, {})
    
    if not rewards:
        continue
    
    # Route BEFORE feedback
    selected, log = router.route(prompt)
    
    if selected in rewards:
        actual_reward = rewards[selected]
        optimal_model = max(rewards.items(), key=lambda x: x[1])[0]
        
        if selected == optimal_model:
            optimal_selections_before += 1
        
        # Give feedback
        router.process_feedback(log.request_id, actual_reward, cluster_boost=False)
        
        total += 1

print(f"\n📊 Initial Performance (first 20 prompts):")
print(f"   Accuracy: {optimal_selections_before}/{total} = {optimal_selections_before/total:.1%}")

# Now test on next 20 prompts (after learning)
for prompt_data in prompts[20:40]:
    prompt = prompt_data['prompt']
    rewards = reward_lookup.get(prompt, {})
    
    if not rewards:
        continue
    
    selected, log = router.route(prompt)
    
    if selected in rewards:
        optimal_model = max(rewards.items(), key=lambda x: x[1])[0]
        if selected == optimal_model:
            optimal_selections_after += 1

print(f"\n📊 After Learning (next 20 prompts):")
print(f"   Accuracy: {optimal_selections_after}/{20} = {optimal_selections_after/20:.1%}")

# Check what models are being selected
print(f"\n🔍 Sample selections from first 5 prompts:")
for i, prompt_data in enumerate(prompts[20:25]):
    prompt = prompt_data['prompt']
    rewards = reward_lookup.get(prompt, {})
    
    if not rewards:
        continue
    
    selected, log = router.route(prompt)
    optimal = max(rewards.items(), key=lambda x: x[1])[0]
    
    print(f"\n  Prompt {i+1}: \"{prompt[:40]}...\"")
    print(f"    Selected: {selected[:35]}")
    print(f"    Optimal:  {optimal[:35]}")
    print(f"    Match: {'✓' if selected == optimal else '✗'}")
    if selected in rewards:
        print(f"    Rewards: selected={rewards[selected]:.3f}, optimal={rewards[optimal]:.3f}")

print("\n" + "="*60)
