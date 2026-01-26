#!/usr/bin/env python3
"""Test parallel RouteLLM routing"""

import sys
from pathlib import Path
import json
import gzip
import time
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.config_legacy import CANONICAL_HOLDOUT_DATA_PATH, DEFAULT_MODEL_REGISTRY_PATH
from routellm.controller import Controller
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

def load_model_costs():
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        data = json.load(f)
    costs = {}
    for model in data["models"]:
        model_id = model["openrouter_id"]
        input_cost = model.get("price_1m_input", 0.0)
        output_cost = model.get("price_1m_output", 0.0)
        cost_per_request = (100 * input_cost + 400 * output_cost) / 1_000_000
        costs[model_id] = {"name": model.get("display_name", model_id), "cost": cost_per_request}
    return costs

def load_small_data(n=100):
    prompt_rewards = defaultdict(lambda: {})
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                score = entry.get("raw_score", 0.0)
                prompt_rewards[prompt][model_id] = score
    
    prompts_data = []
    for prompt, rewards in prompt_rewards.items():
        if len(rewards) == 2:
            prompts_data.append({"prompt": prompt, "rewards": rewards})
        if len(prompts_data) >= n:
            break
    return prompts_data

def route_single(prompt_data, controller, router_name, threshold, model_costs):
    prompt = prompt_data["prompt"]
    rewards = prompt_data["rewards"]
    selected_model = controller.route(prompt, router=router_name, threshold=threshold)
    if selected_model in rewards:
        return rewards[selected_model], model_costs[selected_model]["cost"]
    return 0.0, 0.0

def test_parallel(prompts_data, controller, model_costs, n_threads):
    print(f"\nTesting with {n_threads} threads on {len(prompts_data)} prompts...")
    
    start = time.time()
    total_reward = 0.0
    total_cost = 0.0
    
    route_func = partial(route_single, controller=controller, router_name='mf', 
                        threshold=0.5, model_costs=model_costs)
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(route_func, pd) for pd in prompts_data]
        for future in as_completed(futures):
            reward, cost = future.result()
            total_reward += reward
            total_cost += cost
    
    elapsed = time.time() - start
    avg_reward = total_reward / len(prompts_data)
    avg_cost = total_cost / len(prompts_data)
    
    print(f"  Time: {elapsed:.2f}s ({elapsed/len(prompts_data):.3f}s per prompt)")
    print(f"  Reward: {avg_reward:.4f}, Cost: ${avg_cost:.6f}")
    return elapsed

print("Loading data...")
model_costs = load_model_costs()
prompts_data = load_small_data(100)
print(f"Loaded {len(prompts_data)} prompts")

print("\nInitializing RouteLLM...")
controller = Controller(
    routers=['mf'],
    strong_model='openai/gpt-4-turbo',
    weak_model='mistralai/mixtral-8x7b-instruct'
)
print("✓ Initialized")

# Test different thread counts
for n_threads in [1, 8, 16, 32]:
    test_parallel(prompts_data, controller, model_costs, n_threads)

print("\n✅ Parallel processing works!")

