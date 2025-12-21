import json
import numpy as np
from pathlib import Path

try:
    from bandit import BanditRouter
except ImportError:
    import sys
    sys.path.append("/Users/annette/repostitories/llm_jury/final_release")
    from bandit import BanditRouter

def main():
    root_dir = Path("/Users/annette/repostitories/llm_jury/final_release")
    
    # 1. Load Models and Costs
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # 2. Initialize Router with HLE Priors
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5,
        prior_strength=20.0,
        priors_meta_path=priors_meta_path
    )
    
    # 3. Extract Confidence (||theta||) and Cost
    baseline_cost = 4.5
    results = []
    
    for m_id in router.bandit.models:
        if m_id not in registry: continue
        
        cost = registry[m_id].get("price_1m_blended", 0.0)
        if cost <= baseline_cost: continue
        
        # Calculate theta: theta = A_inv @ b
        A_inv = router.bandit.A_inv[m_id]
        b = router.bandit.b[m_id]
        theta = A_inv @ b
        confidence = np.linalg.norm(theta)
        
        results.append({
            "name": registry[m_id].get("name", m_id),
            "cost": cost,
            "confidence": confidence
        })
    
    # Sort by cost descending
    results.sort(key=lambda x: x["cost"], reverse=True)
    
    print("| Model Name | Cost ($/1M) | Specialist Confidence (||\u03b8||) |")
    print("| :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['name']} | {r['cost']:.2f} | {r['confidence']:.4f} |")

if __name__ == "__main__":
    main()
