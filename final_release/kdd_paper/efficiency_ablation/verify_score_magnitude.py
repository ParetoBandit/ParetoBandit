import math

def calculate_scores():
    # Constants
    LAMBDA_COST = 50.0
    PRIOR_STRENGTH = 20.0 # Not used directly in utility comparison, but affects initial belief magnitude?
    # Actually, the router selects based on UCB. Initial UCB ~ Mean + Alpha*Std.
    # Initial Mean (theta*x) depends on A_inv * b.
    # If A = I*lambda, b = score * bias_vec.
    # Mean ~ score.
    # So we can compare scores directly.

    # Models
    # Llama 3.2 1B (Cheapest, General)
    llama = {
        "name": "Llama 3.2 1B",
        "cost": 0.00005, # $/1k
        "base_hle": 0.053,
        "is_math": False
    }
    # DeepSeek R1 (Cheap, Math Specialist)
    deepseek = {
        "name": "DeepSeek R1",
        "cost": 0.00006, # $/1k (Slightly more than Llama)
        "base_hle": 0.053, # Assuming same base
        "is_math": True # Cluster Boost!
    }
    
    # Cost Range (Realistic - includes Claude 3 Opus at $15/1M)
    min_cost = 0.00005
    max_cost = 0.01500 # $0.015/1k
    cost_range = max_cost - min_cost
    
    print(f"Cost Penalty Factor (Lambda): {LAMBDA_COST}")
    print(f"Cost Range: {min_cost} to {max_cost} (Range={cost_range:.5f})")
    print("-" * 60)

    def get_metrics(enable_efficiency):
        print(f"Scenario: Efficiency Boost {'ON' if enable_efficiency else 'OFF'}")
        
        results = {}
        for m in [llama, deepseek]:
            # 1. Efficiency Boost
            if enable_efficiency:
                eff_boost = 1.0 + (0.2 * math.log(1.0 / m["cost"]))
            else:
                eff_boost = 1.0
            
            # 2. Cluster Boost
            cluster_boost = 1.5 if m["is_math"] else 1.0
            
            # 3. Final Score
            score = m["base_hle"] * eff_boost * cluster_boost
            
            # 4. Normalized Cost
            norm_cost = (m["cost"] - min_cost) / cost_range
            
            # 5. Utility (Quality - CostPenalty)
            cost_penalty = LAMBDA_COST * norm_cost
            utility = score - cost_penalty
            
            results[m["name"]] = {
                "score": score,
                "eff_boost": eff_boost,
                "cluster_boost": cluster_boost,
                "norm_cost": norm_cost,
                "cost_penalty": cost_penalty,
                "utility": utility
            }
            
            print(f"  {m['name']:<15} | Score: {score:.4f} (Eff={eff_boost:.2f}, Clust={cluster_boost:.1f}) | CostPen: {cost_penalty:.4f} | Util: {utility:.4f}")
            
        # Compare
        u_llama = results["Llama 3.2 1B"]["utility"]
        u_ds = results["DeepSeek R1"]["utility"]
        winner = "DeepSeek R1" if u_ds > u_llama else "Llama 3.2 1B"
        margin = abs(u_ds - u_llama)
        print(f"  Winner: {winner} (Margin: {margin:.4f})")
        print("-" * 60)

    get_metrics(enable_efficiency=False)
    get_metrics(enable_efficiency=True)

if __name__ == "__main__":
    calculate_scores()
