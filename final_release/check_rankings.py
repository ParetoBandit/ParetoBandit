import numpy as np
import json
from pathlib import Path
from bandit import BanditRouter

def check_rankings():
    root_dir = Path("/Users/annette/repostitories/llm_jury/final_release")
    priors_path = root_dir / "data" / "priors_meta_large.npz"
    models_path = root_dir / "models.json"
    
    with open(models_path, "r") as f:
        models_list = json.load(f)["models"]
    
    model_registry = {m["openrouter_id"]: m for m in models_list}
    
    router = BanditRouter.load_from_benchmark(
        model_registry=model_registry,
        context_model="all-MiniLM-L6-v2",
        alpha=1.0,
        prior_strength=10.0,
        priors_meta_path=priors_path
    )
    
    rankings = []
    for m_id in router.bandit.models:
        theta = router.bandit.A_inv[m_id] @ router.bandit.b[m_id]
        conf = np.linalg.norm(theta)
        rankings.append((m_id, conf))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 Models by Prior Confidence:")
    for i, (m_id, conf) in enumerate(rankings[:10]):
        print(f"{i+1}. {m_id}: {conf:.4f}")

if __name__ == "__main__":
    check_rankings()
