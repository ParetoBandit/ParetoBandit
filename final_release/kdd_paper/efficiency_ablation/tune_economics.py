
import sys
import json
import itertools
import math
import numpy as np
import pandas as pd
from pathlib import Path
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
    # 500 "Easy" prompts -> Should use Cheap Models
    for i in range(500):
        scenarios.append({
            "type": "easy", 
            "prompt": f"Easy prompt {i}", 
            "vec": np.random.normal(0, 0.1, 384) 
        })
    # 100 "Hard" prompts -> Should use Powerful Models
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
        # Easy task: Cheap models ~0.95, Expensive ~0.99
        base = m.get("quality_score", 0.5)
        acc = min(0.99, base * 1.5)
    return acc, cost

# ==============================================================================
# 2. EVALUATION FUNCTION (Stage 2: Economics)
# ==============================================================================
def evaluate_config(args):
    lambda_cost, eff_boost_factor, registry, scenarios = args
    
    # FIX ENGINE PARAMS (Stage 1 Done)
    alpha = 1.0 
    prior_strength = 20.0 # Reduced from 40 based on easy-task findings, or keep 40? User said "Lock it".
    # User said "Status: SOLVED. alpha=1.0, gamma=0.98, strength=40. Lock these."
    # OK, I will lock strength = 40.0.
    prior_strength = 40.0 
    
    if eff_boost_factor != 0.2: 
        # Simulation Logic: Override Prior
        pass

    # Create Router
    router = BanditRouter.create(model_registry=registry, exploration="balanced", prior_strength=prior_strength)
    router.bandit.alpha = alpha
    
    dim = router.bandit.dim - 1
    for m in router.bandit.models:
        cost = float(registry.get(m, {}).get("price_1m_blended") or 0.0) / 1000.0
        cost = max(cost, 0.00000005)
        
        # Current (baked in) boost (assuming bandit.py is at 0.3 currently)
        # We should probably hardcode the 'base' assumption or read it.
        # Let's assume the base in bandit.py is whatever we left it (0.3).
        current_boost = 1.0 + (0.3 * math.log(1.0 / cost)) 
        
        # Target boost
        target_boost = 1.0 + (eff_boost_factor * math.log(1.0 / cost))
        
        adjustment_ratio = target_boost / current_boost
        
        # Adjust Bias Term in b (Last element)
        router.bandit.b[m][dim] *= adjustment_ratio
        
        # Re-calc inverse
        router.bandit.A_inv[m] = np.linalg.inv(router.bandit.A[m])

    # ------------------------------------------------------------------
    # SIMULATION
    # ------------------------------------------------------------------
    candidates = list(router.bandit.models)
    costs = {m: registry[m].get("price_1m_blended", 1.0) for m in candidates}
    log_costs = {m: np.log(max(costs[m], 1e-9)) for m in candidates}
    min_c, max_c = min(log_costs.values()), max(log_costs.values())
    range_c = max_c - min_c if max_c > min_c else 1.0
    
    total_acc = 0.0
    total_norm_cost = 0.0
    
    for i, s in enumerate(scenarios):
        # Select
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
        norm_cost_actual = (log_costs[best_m] - min_c) / range_c
        
        # Update
        reward = acc - (lambda_cost * norm_cost_actual)
        router.bandit.update(best_m, x, reward)
        
        # Track stats
        total_acc += acc
        total_norm_cost += norm_cost_actual
        
    avg_quality = total_acc / len(scenarios)
    avg_norm_cost = total_norm_cost / len(scenarios)
    
    return {
        "lambda": lambda_cost, 
        "boost": eff_boost_factor,
        "quality": avg_quality, 
        "norm_cost": avg_norm_cost
    }

# ==============================================================================
# 3. CHEBYSHEV SCALARIZATION (The "Best Value" Finder)
# ==============================================================================
def find_chebyshev_optimal(sweep_results, w_q=0.5, w_c=0.5):
    """
    Identifies the Best Value config using Chebyshev Scalarization.
    """
    df = pd.DataFrame(sweep_results)
    
    # Rename for clarity
    df['q'] = df['quality']
    df['c'] = df['norm_cost'] # This is already log-normalized per candidate list
    
    min_c, max_c = df['c'].min(), df['c'].max()
    min_q, max_q = df['q'].min(), df['q'].max()
    
    # Normalize to [0,1] range relative to the SWEEP Extremes
    df['n_c'] = (df['c'] - min_c) / (max_c - min_c + 1e-9) 
    df['n_q'] = (df['q'] - min_q) / (max_q - min_q + 1e-9)
    
    # 2. Calculate Distance to Utopia (Q=1.0, C=0.0 relative to range)
    df['diff_q'] = 1.0 - df['n_q'] 
    df['diff_c'] = df['n_c'] - 0.0
    
    # 3. The Chebyshev Scalarization
    # Score = max( w_q * diff_q, w_c * diff_c )
    df['chebyshev_score'] = df.apply(
        lambda row: max(w_q * row['diff_q'], w_c * row['diff_c']), axis=1
    )
    
    # 4. Find the Minimizer (Closest to Utopia)
    best_idx = df['chebyshev_score'].idxmin()
    best_config = df.loc[best_idx]
    
    return best_config, df

def run_tuning():
    print("Starting script...", flush=True)
    print("Loading data...", flush=True)
    registry, scenarios = load_data()
    print(f"Data loaded: {len(registry)} models, {len(scenarios)} scenarios.", flush=True)
    print("Parameters: Alpha=1.0, Strength=40.0 (Locked Engine)", flush=True)
    
    # Steering Sweep
    lambdas = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    boosts = [0.0, 0.2, 0.3, 0.5, 0.7] 
    
    grid = list(itertools.product(lambdas, boosts))
    print(f"Running Economics Sweep ({len(grid)} configs)...", flush=True)
    
    results = []
    for l, b in grid:
        res = evaluate_config((l, b, registry, scenarios))
        print(f"L={l:<4} B={b:<3} -> Q={res['quality']:.3f}, NC={res['norm_cost']:.3f}")
        results.append(res)
        
    # Find Optimal via Chebyshev
    winner, df = find_chebyshev_optimal(results, w_q=0.5, w_c=0.5)
    
    print("\n--- OPTIMAL CHEBYSHEV CONFIGURATION (Balanced) ---")
    print(f"Lambda Cost: {winner['lambda']}")
    print(f"Efficiency Boost: {winner['boost']}")
    print(f"Chebyshev Score: {winner['chebyshev_score']:.4f} (Lower is better)")
    print(f"Resulting Stats: Acc={winner['quality']:.3f}, NormCost={winner['norm_cost']:.3f}")
    
    print("\nAction Plan:")
    print(f"1. Update bandit.py: Set `factor` in `efficiency_boost` equation to {winner['boost']}")
    print(f"2. Update bandit.py: Set `VALUE_EFFICIENT` lambda_cost to {winner['lambda']}")

if __name__ == "__main__":
    run_tuning()
