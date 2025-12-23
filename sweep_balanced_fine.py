
import json
import numpy as np
from pathlib import Path
from final_release.bandit import BanditRouter, OptimizationProfile

def main():
    router = BanditRouter.create()
    
    # Load prompts
    base_dir = Path(__file__).parent
    data_dir = base_dir / "final_release/data"
    if not data_dir.exists():
        data_dir = base_dir / "data"

    prompts = []
    with open(data_dir / "test_prompts.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            
    # Sample 50 prompts
    np.random.seed(42)
    selected_prompts = np.random.choice(prompts, 50, replace=False)
    
    lambdas = [1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58, 1.60]
    
    print(f"{'Lambda':<6} | {'Qual (HLE)':<10} | {'Cost ($/1M)':<12} | {'Lat (s)':<8}")
    print("-" * 50)
    
    for l_cost in lambdas:
        OptimizationProfile._PROFILES["balanced"] = {
            "lambda_cost": l_cost,
            "lambda_latency": 0.1
        }
        
        total_q = 0.0
        total_c = 0.0
        total_l = 0.0
        count = 0
        
        for p in selected_prompts:
            model, log = router.route(p, profile="balanced")
            q = float(router.registry[model].get("hle", 0.0))
            
            total_q += q
            total_c += log.cost_usd
            total_l += log.latency_s
            count += 1
            
        avg_q = total_q / count
        avg_c_per_m = (total_c / count) * (1000000.0 / 1200.0)
        avg_l = total_l / count
        
        print(f"{l_cost:<6.2f} | {avg_q:<10.3f} | ${avg_c_per_m:<11.2f} | {avg_l:<8.2f}")

if __name__ == "__main__":
    main()
