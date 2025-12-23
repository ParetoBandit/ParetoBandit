
from final_release.bandit import BanditRouter, OptimizationProfile
import numpy as np
import time

def main():
    router = BanditRouter.create()
    
    # Use a dummy prompt
    prompt = "Explain quantum computing briefly."
    
    # Profile to debug
    # Let's debug Balanced and Low Latency
    
    in_tok = 20
    out_tok = 600
    
    candidates = list(router.registry.keys())
    
    # 1. Get UCBs 
    x = router._get_context_vector(prompt)
    ucbs = {}
    for m in candidates:
        _, ucb = router.bandit.select_arm(x, candidates=[m])
        ucbs[m] = ucb
        
    # 2. Get Costs/Latencies
    costs = {m: router._estimate_cost(m, in_tok, out_tok) for m in candidates}
    lats = {m: router._estimate_latency(m, out_tok) for m in candidates}
    
    # LOG MIN MAX
    EPS = 1e-9
    log_costs = {m: np.log(max(costs[m], EPS)) for m in candidates}
    log_lats = {m: np.log(max(lats[m], EPS)) for m in candidates}
    
    min_c, max_c = min(log_costs.values()), max(log_costs.values())
    range_c = max_c - min_c if max_c > min_c else 1.0
    
    min_l, max_l = min(log_lats.values()), max(log_lats.values())
    range_l = max_l - min_l if max_l > min_l else 1.0
    
    print(f"LogCost Range: {min_c:.2f} to {max_c:.2f}")

    for profile_name in ["balanced", "low_latency"]:
        weights = OptimizationProfile.get(profile_name)
        lambda_cost = weights["lambda_cost"]
        lambda_latency = weights["lambda_latency"]
        
        print(f"\nDEBUGGING PROFILE: {profile_name}")
        print(f"Weights: Cost={lambda_cost}, Latency={lambda_latency}")
        
        print(f"{'Model':<35} | {'Util':<6} | {'Qual':<5} | {'Cost':<7} | {'LogNC':<5} | {'Lat':<5} | {'LogNL':<5}")
        print("-" * 90)
        
        results = []
        for m in candidates:
            quality = ucbs[m]
            norm_cost = (log_costs[m] - min_c) / range_c
            norm_lat = (log_lats[m] - min_l) / range_l
            
            utility = quality - (lambda_cost * norm_cost) - (lambda_latency * norm_lat)
            results.append((m, utility, quality, costs[m], norm_cost, lats[m], norm_lat))
            
        results.sort(key=lambda x: x[1], reverse=True)
        
        for r in results[:10]:
            m, u, q, c, nc, l, nl = r
            print(f"{m:<35} | {u:.3f}  | {q:.3f} | ${c:<6.2f} | {nc:.3f} | {l:<5.1f} | {nl:.3f}")

if __name__ == "__main__":
    main()
