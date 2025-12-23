
from final_release.bandit import BanditRouter
import json

def main():
    router = BanditRouter.create()
    print(f"{'Model ID':<35} | {'HLE':<5} | {'$/1M (in)':<10} | {'Lat(s)':<6}")
    print("-" * 70)
    
    # Sort by HLE descending
    sorted_models = sorted(router.registry.items(), key=lambda x: float(x[1].get("hle", 0)), reverse=True)
    
    for mid, m in sorted_models:
        hle = float(m.get("hle", 0))
        cost = float(m.get("input_cost_per_m", 0))
        lat = float(m.get("latency_s", 0)) # Default static latency if available
        # Estimate latency using router logic roughly
        est_lat = router._estimate_latency(mid, 600)
        
        print(f"{mid:<35} | {hle:.3f} | ${cost:<9.2f} | {est_lat:.2f}s")
        
if __name__ == "__main__":
    main()
