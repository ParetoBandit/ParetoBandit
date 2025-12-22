import time
import numpy as np
import json
from pathlib import Path
import sys

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

try:
    from bandit import BanditRouter
except ImportError:
    from final_release.bandit import BanditRouter

def benchmark_latency():
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    router = BanditRouter.create(
        model_registry=registry
    )
    
    prompt = "Explain the difference between a B-tree and a LSM-tree in the context of TimescaleDB."
    
    # Warmup
    for _ in range(5):
        router.route(prompt)
        
    n_iters = 100
    times = {
        "embedding": [],
        "filtering": [],
        "scoring": [],
        "total": []
    }
    
    for _ in range(n_iters):
        start_total = time.perf_counter()
        
        # 1. Embedding
        start = time.perf_counter()
        x = router.encoder.encode(prompt)
        from final_release.bandit import l2_normalize
        x = l2_normalize(x)
        times["embedding"].append(time.perf_counter() - start)
        
        # 2. Filtering
        start = time.perf_counter()
        candidates = list(router.registry.keys())
        filtered = []
        for m in candidates:
            cost = router._estimate_cost(m, 20, 600)
            lat = router._estimate_latency(m, 600)
            filtered.append(m)
        times["filtering"].append(time.perf_counter() - start)
        
        # 3. Scoring
        start = time.perf_counter()
        for m in filtered:
            x_bias = np.append(x, 1.0)
        router.bandit.select_arm(x_bias, candidates=[m])
        times["scoring"].append(time.perf_counter() - start)
        
        times["total"].append(time.perf_counter() - start_total)

    print(f"BanditRouter Latency Breakdown ({len(registry)} models):")
    print(f"{'Component':<15} | {'Mean (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 40)
    for comp, vals in times.items():
        mean_ms = np.mean(vals) * 1000
        p95_ms = np.percentile(vals, 95) * 1000
        print(f"{comp.capitalize():<15} | {mean_ms:<10.2f} | {p95_ms:<10.2f}")

if __name__ == "__main__":
    benchmark_latency()
