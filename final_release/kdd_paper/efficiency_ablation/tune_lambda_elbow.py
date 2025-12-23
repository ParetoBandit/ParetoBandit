
import sys
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from final_release.bandit import BanditRouter, l2_normalize, OptimizationProfile

# ==============================================================================
# 1. SETUP & DATA
# ==============================================================================
def load_data():
    base_dir = Path("final_release")
    with open(base_dir / "data" / "models_cache_with_hle.json") as f:
        m_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in m_data["models"] if "openrouter_id" in m and m.get("price_1m_blended", 0) > 0.01}
    
    # Validation Set: Mixed Scenarios
    scenarios = []
    # 500 "Easy" prompts (Simulated)
    for i in range(500):
        scenarios.append({
            "type": "easy", 
            "prompt": f"Easy prompt {i}", 
            "vec": np.random.normal(0, 0.1, 384) 
        })
    # 100 "Hard" prompts (Simulated)
    for i in range(100):
         scenarios.append({
            "type": "hard", 
            "prompt": f"Hard prompt {i}", 
            "vec": np.random.normal(1, 0.1, 384)
        })
    return registry, scenarios

def get_simulated_metrics(model_id, registry, task_type):
    m = registry[model_id]
    cost = m.get("price_1m_blended", 1.0)
    
    if task_type == "hard":
        acc = m.get("hle", 0.0) or (m.get("math_500", 0.0) * 0.3)
    else:
        # Easy task
        base = m.get("quality_score", 0.5)
        acc = min(0.98, base * 1.5)
    return acc, cost

# ==============================================================================
# 2. EVALUATION FUNCTION (Stage 2: Steering)
# ==============================================================================
def evaluate_lambda(args):
    lambda_cost, registry, scenarios = args
    
    # FIX ENGINE PARAMS (Stage 1 Done)
    alpha = 1.0 
    
    router = BanditRouter.create(model_registry=registry, exploration="balanced")
    router.bandit.alpha = alpha 
    
    # Pre-compute metrics
    candidates = list(router.bandit.models)
    costs = {m: registry[m].get("price_1m_blended", 1.0) for m in candidates}
    log_costs = {m: np.log(max(costs[m], 1e-9)) for m in candidates}
    min_c, max_c = min(log_costs.values()), max(log_costs.values())
    range_c = max_c - min_c if max_c > min_c else 1.0
    
    total_acc = 0.0
    total_cost = 0.0
    
    for s in scenarios:
        # Select Arm (Contextual with Lambda)
        x = s["vec"]
        x = l2_normalize(x)
        x = np.append(x, 1.0)
        
        best_m = None
        best_util = -float("inf")
        
        for m in candidates:
             _, ucb = router.bandit.select_arm(x, candidates=[m])
             
             norm_cost = (log_costs[m] - min_c) / range_c
             util = ucb - (lambda_cost * norm_cost)
             if util > best_util:
                 best_util = util
                 best_m = m
        
        # Evaluate
        acc, cost = get_simulated_metrics(best_m, registry, s["type"])
        
        # Update (Simulated Feedback)
        norm_c = (log_costs[best_m] - min_c) / range_c
        reward = acc - (lambda_cost * norm_c)
        router.bandit.update(best_m, x, reward)
        
        total_acc += acc
        total_cost += cost
        
    avg_acc = total_acc / len(scenarios)
    avg_cost = total_cost / len(scenarios)
    
    return {
        "lambda": lambda_cost, 
        "quality": avg_acc, 
        "cost": avg_cost
    }

# ==============================================================================
# 3. PARETO SCAN (Kneedle Algorithm)
# ==============================================================================
def find_optimal_lambda(results):
    """
    Identifies the 'Elbow' (Best Value) configuration from a sweep.
    """
    # 1. Sort by Cost
    points = sorted(results, key=lambda x: x['cost'])
    
    if len(points) < 2: return points[0]['lambda']
    
    # 2. Get the line from Cheapest (First) to Most Expensive (Last)
    # We use Quality vs Cost plot logic.
    p1 = np.array([points[0]['cost'], points[0]['quality']])
    p2 = np.array([points[-1]['cost'], points[-1]['quality']])
    
    best_lambda = None
    max_distance = -1
    
    print("\n--- Elbow Analysis ---")
    
    # 3. Find the point furthest from that line (The Elbow)
    for p in points:
        p_curr = np.array([p['cost'], p['quality']])
        
        # Distance from point to line formula
        # In 2D: |Cross product| / Length
        # (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1) ...
        # Standard vector cross method:
        
        vec_line = p2 - p1
        vec_point = p1 - p_curr
        
        # Cross product of 2D vectors (z-component)
        cross_prod = vec_line[0]*vec_point[1] - vec_line[1]*vec_point[0]
        distance = np.abs(cross_prod) / np.linalg.norm(vec_line)
        
        print(f"Lambda={p['lambda']}: Cost={p['cost']:.3f}, Qual={p['quality']:.3f}, Dist={distance:.4f}")
        
        if distance > max_distance:
            max_distance = distance
            best_lambda = p['lambda']
            
    return best_lambda

def run_tuning():
    print("Loading data...")
    registry, scenarios = load_data()
    print(f"Data loaded: {len(registry)} models, {len(scenarios)} scenarios.")
    
    # Lambda Sweep
    lambdas = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0]
    
    print(f"Running Pareto Scan (Lambdas={lambdas})...")
    
    results = []
    for lam in lambdas:
        res = evaluate_lambda((lam, registry, scenarios))
        print(f"Lambda={res['lambda']:<4} -> Qual={res['quality']:.3f}, Cost=${res['cost']:.3f}")
        results.append(res)
        
    optimal_lambda = find_optimal_lambda(results)
    
    print(f"\n✅ Optimal Lambda (Elbow): {optimal_lambda}")
    print("Action: Update 'VALUE_EFFICIENT' profile in bandit.py with this value.")

if __name__ == "__main__":
    run_tuning()
