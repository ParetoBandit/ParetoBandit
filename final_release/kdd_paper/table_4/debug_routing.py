import json
import numpy as np
from pathlib import Path

try:
    from final_release.bandit import BanditRouter, estimate_tokens_rough
except (ImportError, ValueError):
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from bandit import BanditRouter, estimate_tokens_rough

def debug_routing():
    router = BanditRouter.create()
    prompt = "Hello, how are you?"
    profile = "balanced"
    l_cost = 30.0
    l_lat = 0.1
    
    # 1. Embed
    x = router._get_context_vector(prompt)
    
    in_tok = estimate_tokens_rough(prompt)
    out_tok = 600
    
    candidates = list(router.registry.keys())
    
    costs = {m: router._estimate_cost(m, in_tok, out_tok) for m in candidates}
    lats = {m: router._estimate_latency(m, out_tok) for m in candidates}
    
    min_cost, max_cost = min(costs.values()), max(costs.values())
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
    
    min_lat, max_lat = min(lats.values()), max(lats.values())
    lat_range = max_lat - min_lat if max_lat > min_lat else 1.0
    
    # Get top 10 by utility
    scorings = []
    for m in candidates:
        _, ucb = router.bandit.select_arm(x, candidates=[m])
        norm_cost = (costs[m] - min_cost) / cost_range
        norm_lat = (lats[m] - min_lat) / lat_range
        utility = ucb - (l_cost * norm_cost) - (l_lat * norm_lat)
        
        scorings.append({
            "model": m,
            "ucb": ucb,
            "cost": costs[m],
            "norm_cost": norm_cost,
            "utility": utility,
            "hle": router.registry[m].get("hle", 0.0)
        })
    
    scorings.sort(key=lambda x: x["utility"], reverse=True)
    
    print(f"Profile: {profile} (lambda_cost={l_cost})")
    print(f"Cost Range: {min_cost:.6f} to {max_cost:.6f} (Range: {cost_range:.6f})")
    print(f"Lat Range:  {min_lat:.2f} to {max_lat:.2f} (Range: {lat_range:.2f})")
    print(f"{'Model':<40} | {'UCB':>6} | {'Cost':>8} | {'Utility':>8} | {'HLE':>6}")
    print("-" * 80)
    for s in scorings[:15]:
        print(f"{s['model']:<40} | {s['ucb']:>6.3f} | {s['cost']:>8.4f} | {s['utility']:>8.4f} | {s['hle']:>6.3f}")

if __name__ == "__main__":
    debug_routing()
