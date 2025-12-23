#!/usr/bin/env python3
"""
Benchmark BanditRouter latency overhead per request.
Measures embedding, filtering, and scoring components.
"""
import time
import numpy as np
import json
from pathlib import Path
import sys

# Setup paths
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent.parent  # Go to llm_jury root
sys.path.insert(0, str(repo_root / "final_release"))

from bandit import BanditRouter, l2_normalize

def benchmark_latency():
    # Load real model registry  
    # repo_root = llm_jury/, models.json is in llm_jury/final_release/
    models_path = repo_root / "final_release" / "models.json"
    with open(models_path) as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    print(f"Loaded {len(registry)} models from registry")
    
    # Create real router
    router = BanditRouter.create(model_registry=registry)
    
    prompt = "Explain the difference between a B-tree and a LSM-tree in the context of TimescaleDB."
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        router.route(prompt)
        
    n_iters = 100
    times = {
        "embedding": [],
        "filtering": [],
        "scoring": [],
        "total": []
    }
    
    print(f"Benchmarking {n_iters} iterations...")
    for _ in range(n_iters):
        start_total = time.perf_counter()
        
        # 1. Embedding
        start = time.perf_counter()
        x = router.encoder.encode(prompt)
        x = l2_normalize(x)
        times["embedding"].append(time.perf_counter() - start)
        
        # 2. Filtering (simulate constraint checking)
        start = time.perf_counter()
        candidates = list(router.registry.keys())
        filtered = []
        for m in candidates:
            cost = router._estimate_cost(m, 20, 600)
            lat = router._estimate_latency(m, 600)
            filtered.append(m)
        times["filtering"].append(time.perf_counter() - start)
        
        # 3. Scoring (LinUCB selection)
        start = time.perf_counter()
        x_bias = np.append(x, 1.0)
        for m in filtered[:10]:  # Sample 10 models for speed
            router.bandit.select_arm(x_bias, candidates=[m])
        times["scoring"].append(time.perf_counter() - start)
        
        times["total"].append(time.perf_counter() - start_total)

    print(f"\nBanditRouter Latency Breakdown ({len(registry)} models):")
    print(f"{'Component':<15} | {'Mean (ms)':<10} | {'P95 (ms)':<10}")
    print("-" * 40)
    for comp, vals in times.items():
        mean_ms = np.mean(vals) * 1000
        p95_ms = np.percentile(vals, 95) * 1000
        print(f"{comp.capitalize():<15} | {mean_ms:<10.2f} | {p95_ms:<10.2f}")

if __name__ == "__main__":
    benchmark_latency()
