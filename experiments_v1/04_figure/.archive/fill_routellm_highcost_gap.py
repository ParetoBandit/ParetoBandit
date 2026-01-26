#!/usr/bin/env python3
"""
Fill the critical high-cost gap in RouteLLM curve ($0.008 - $0.012).
Target thresholds in the 0.15-0.25 range to get more data points.
"""

import sys
from pathlib import Path
import json
import numpy as np
import time
from collections import defaultdict
import gzip

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from bandit_gpt.config_legacy import (
    DEFAULT_MODEL_REGISTRY_PATH,
    CANONICAL_HOLDOUT_DATA_PATH
)
from routellm.controller import Controller

def load_eval_data():
    prompt_rewards = defaultdict(lambda: {})
    with gzip.open(CANONICAL_HOLDOUT_DATA_PATH, 'rt') as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("ok"):
                prompt = entry["prompt"]
                model_id = entry["model_id"]
                score = entry["raw_score"]
                prompt_rewards[prompt][model_id] = score
    
    return [{"prompt": p, "rewards": r} for p, r in prompt_rewards.items() if len(r) == 2]

def load_model_costs():
    with open(DEFAULT_MODEL_REGISTRY_PATH) as f:
        models_data = json.load(f)
    
    model_costs = {}
    for model in models_data["models"]:
        model_id = model["openrouter_id"]
        input_cost = model["price_1m_input"]
        output_cost = model["price_1m_output"]
        cost_per_request = (100 * input_cost + 400 * output_cost) / 1_000_000
        model_costs[model_id] = {"cost": cost_per_request}
    
    return model_costs

def route_with_retry(controller, prompt, threshold, max_retries=5):
    for attempt in range(max_retries):
        try:
            return controller.route(prompt, router='mf', threshold=threshold)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = (2 ** attempt) * 0.5
                print(f"        Rate limit, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries")

def main():
    print("="*70)
    print("ROUTELLM HIGH-COST GAP FILLER")
    print("="*70)
    print("\nTargeting the $0.008 - $0.012 gap with fine-grained thresholds")
    
    # Load existing results
    results_path = Path(__file__).parent / "results" / "pareto_results.json"
    with open(results_path) as f:
        data = json.load(f)
    
    existing_points = [(p["cost"], p["reward"]) for p in data["strategies"]["RouteLLM-MF"]]
    print(f"  Current RouteLLM points: {len(existing_points)}")
    
    # Find the gap
    sorted_points = sorted(existing_points, key=lambda x: x[0])
    print("\n  Current cost distribution:")
    for c, r in sorted_points[-5:]:
        print(f"    ${c:.6f}: R={r:.4f}")
    
    # Target thresholds between 0.10 and 0.25 (fine-grained)
    target_thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
    
    print(f"\n  Will evaluate {len(target_thresholds)} fine-grained thresholds")
    print("  Expected to fill the $0.008-$0.012 range")
    
    # Load data
    eval_data = load_eval_data()
    model_costs = load_model_costs()
    print(f"\n  Loaded {len(eval_data)} evaluation prompts")
    
    # Initialize controller
    models = ["mistralai/mixtral-8x7b-instruct", "openai/gpt-4-turbo"]
    controller = Controller(
        routers=['mf'],
        strong_model=models[1],
        weak_model=models[0]
    )
    print("  ✓ RouteLLM controller initialized")
    
    # Evaluate new thresholds
    print(f"\n  Evaluating thresholds (sequential, rate-limit safe)...")
    new_points = []
    
    for i, threshold in enumerate(target_thresholds, 1):
        print(f"\n  [{i}/{len(target_thresholds)}] Threshold {threshold:.3f}...")
        
        total_reward = 0.0
        total_cost = 0.0
        count = 0
        
        for prompt_data in eval_data:
            try:
                selected = route_with_retry(controller, prompt_data["prompt"], threshold)
                if selected in prompt_data["rewards"]:
                    total_reward += prompt_data["rewards"][selected]
                    total_cost += model_costs[selected]["cost"]
                    count += 1
            except Exception as e:
                print(f"        Failed: {e}")
                continue
        
        if count > 0:
            avg_reward = total_reward / count
            avg_cost = total_cost / count
            new_points.append({"cost": avg_cost, "reward": avg_reward})
            print(f"      ✓ Cost: ${avg_cost:.6f}, Reward: {avg_reward:.4f}")
        
        time.sleep(0.3)  # Rate limit safety
    
    # Merge and save
    all_points = data["strategies"]["RouteLLM-MF"] + new_points
    data["strategies"]["RouteLLM-MF"] = all_points
    
    output_path = Path(__file__).parent / "results" / "pareto_results_filled.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ SUCCESS")
    print("="*70)
    print(f"  Added {len(new_points)} new points")
    print(f"  Total RouteLLM points: {len(all_points)}")
    print(f"  Saved to: {output_path}")
    print("\nRegenerate plot with:")
    print("  python generate_pareto_frontier.py")

if __name__ == "__main__":
    main()

