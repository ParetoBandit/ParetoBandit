
import sys
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
try:
    from banditgpt import BanditRouter, l2_normalize, OptimizationProfile
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from banditgpt import BanditRouter, l2_normalize, OptimizationProfile

# ==============================================================================
# 1. SETUP & DATA
# ==============================================================================
def load_data():
    # 1. Models
    project_root = Path(__file__).parent.parent.parent.parent
    data_dir = project_root / "banditgpt" / "data"
    with open(data_dir / "models_cache_with_hle.json") as f:
        m_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in m_data["models"] if "openrouter_id" in m and m.get("price_1m_blended", 0) > 0.01}
    
    # 2. Scenarios (Validation Set)
    # We need a mix of Easy and Hard to check adaptability
    scenarios = []
    
    # Generate 500 "Easy" prompts (Simulated)
    # Ground Truth: Cheap models work (Acc ~0.95), Expensive models work (Acc ~0.98)
    for i in range(500):
        scenarios.append({
            "type": "easy", 
            "prompt": f"Easy prompt {i}", 
            "vec": np.random.normal(0, 0.1, 384) # Dummy vectors close to 0
        })
        
    # Generate 100 "Hard" prompts (Simulated)
    # Ground Truth: Cheap models fail (Acc ~0.05), Expensive models work (Acc ~0.40)
    for i in range(100):
         scenarios.append({
            "type": "hard", 
            "prompt": f"Hard prompt {i}", 
            "vec": np.random.normal(1, 0.1, 384) # Dummy vectors shifted
        })
        
    return registry, scenarios

def get_simulated_metrics(model_id, registry, task_type):
    m = registry[model_id]
    cost = m.get("price_1m_blended", 1.0)
    
    if task_type == "hard":
        acc = m.get("hle", 0.0) or (m.get("math_500", 0.0) * 0.3)
    else:
        # Easy task: Cap at 0.98, cheap models get ~0.95
        base = m.get("quality_score", 0.5)
        acc = min(0.98, base * 1.5)
    return acc, cost

# ==============================================================================
# 2. EVALUATION FUNCTION
# ==============================================================================
def evaluate_config(args):
    alpha, lambda_cost, registry, scenarios = args
    
    # Override Profile weights dynamically for the test
    # (We can't easily pass this into 'route', so we'll hack it or add a custom profile)
    # Better: We'll modify the router's profile dict or pass manual args if supported.
    # For now, let's assume we use 'value_efficient' but manually patch the class.
    
    # Actually, simpler: We will just reproduce the routing logic here
    # to avoid modifying the global class state in parallel
    
    router = BanditRouter.create(model_registry=registry, exploration="balanced")
    router.bandit.alpha = alpha # Set Alpha
    
    # Pre-compute metrics for speed
    candidates = list(router.bandit.models)
    costs = {m: registry[m].get("price_1m_blended", 1.0) for m in candidates}
    log_costs = {m: np.log(max(costs[m], 1e-9)) for m in candidates}
    min_c, max_c = min(log_costs.values()), max(log_costs.values())
    range_c = max_c - min_c if max_c > min_c else 1.0
    
    total_value = 0.0
    total_acc = 0.0
    total_cost = 0.0
    
    for s in scenarios:
        # 1. Select
        # x = ... (We use pre-computed dummy vecs for speed if possible, else encode)
        # For simulation speed, we just use random vectors or simplified logic
        # But to be fair, we should use the router.
        
        # Mock embedding (speedup)
        x = s["vec"]
        x = l2_normalize(x)
        x = np.append(x, 1.0)
        
        # UCB
        best_m = None
        best_util = -float("inf")
        
        for m in candidates:
             _, ucb = router.bandit.select_arm(x, candidates=[m])
             
             norm_cost = (log_costs[m] - min_c) / range_c
             util = ucb - (lambda_cost * norm_cost)
             if util > best_util:
                 best_util = util
                 best_m = m
        
        # 2. Evaluate
        acc, cost = get_simulated_metrics(best_m, registry, s["type"])
        
        # 3. Update (Online Learning)
        # We assume we get the true Acc back as reward (Or shaped reward)
        # Shaped Reward = Acc - cost_penalty?
        # Bandit learns VALUE.
        norm_c = (log_costs[best_m] - min_c) / range_c
        reward = acc - (lambda_cost * norm_cost)
        
        router.bandit.update(best_m, x, reward)
        
        total_acc += acc
        total_cost += cost
        total_value += (acc - 0.01 * cost) # Global "Business Value" (1% Acc ~= $0.01/1M)
        
    avg_acc = total_acc / len(scenarios)
    avg_cost = total_cost / len(scenarios)
    avg_val = total_value / len(scenarios)
    
    return {
        "alpha": alpha, 
        "lambda_cost": lambda_cost, 
        "avg_acc": avg_acc, 
        "avg_cost": avg_cost, 
        "avg_val": avg_val
    }

# ==============================================================================
# 3. MAIN GRID SEARCH
# ==============================================================================
def run_tuning():
    print("Loading data...")
    registry, scenarios = load_data()
    print(f"Data loaded: {len(registry)} models, {len(scenarios)} scenarios.")
    
    # Grid
    alphas = [0.1, 0.5, 1.0, 1.5, 2.0]
    lambdas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    grid = list(itertools.product(alphas, lambdas))
    print(f"Running Grid Search ({len(grid)} configs)...")
    
    # Prepare args
    # Note: multiprocess might arguably be slower due to pickle overhead for registry
    # We'll run serial for simplicity unless slow.
    results = []
    for alpha, lam in grid:
        res = evaluate_config((alpha, lam, registry, scenarios))
        print(f"Config(a={alpha}, l={lam}) -> Acc={res['avg_acc']:.3f}, Cost=${res['avg_cost']:.3f}, Val={res['avg_val']:.3f}")
        results.append(res)
        
    # Analyze
    df = pd.DataFrame(results)
    best = df.loc[df['avg_val'].idxmax()]
    
    print("\n--- Tuning Results ---")
    print(f"Best Configuration:")
    print(f"  Alpha: {best['alpha']}")
    print(f"  Lambda Cost: {best['lambda_cost']}")
    print(f"  Result: Acc={best['avg_acc']:.3f}, Cost=${best['avg_cost']:.3f}")
    
    # Heatmap (Text)
    print("\nValue Heatmap (Alpha x Lambda):")
    pivot = df.pivot(index="alpha", columns="lambda_cost", values="avg_val")
    print(pivot.round(3))
    
    # Suggest specific fix
    print("\nRecommended Action:")
    print(f"Update `OptimizationProfile.VALUE_EFFICIENT` with lambda_cost={best['lambda_cost']}")
    print(f"Update `BanditRouter` default exploraton to alpha={best['alpha']}")

if __name__ == "__main__":
    run_tuning()
