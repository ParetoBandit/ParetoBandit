import json
import numpy as np
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from final_release.bandit import BanditRouter

def run_regret_comparison():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load data
    with open(data_dir / "train_prompts.jsonl") as f:
        prompts = [json.loads(line) for line in f]
    
    # Encode embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Encoding embeddings...")
    embeddings = model.encode([p["prompt"] for p in prompts])
    cluster_ids = [p["cluster_id"] for p in prompts]
    
    priors_meta_path = data_dir / "priors_meta_large.npz"
    
    def run_sim(router, embs, c_ids):
        cum_regret = 0
        regrets = []
        for i in range(len(embs)):
            x = embs[i]
            c_id = c_ids[i]
            chosen, _ = router.bandit.select_arm(x)
            
            # Reward is accuracy in that cluster
            reward = registry[chosen].get(f"acc_{c_id}", 0.0)
            # Optimal reward
            opt_reward = max(registry[m].get(f"acc_{c_id}", 0.0) for m in router.bandit.models)
            
            regret = opt_reward - reward
            router.bandit.update(chosen, x, reward)
            cum_regret += regret
        return cum_regret

    results = {}
    for key, strength in [("hle", 20.0), ("hle", 200.0), ("math_500", 20.0), ("math_500", 200.0), (None, 0)]:
        print(f"Testing Prior: {key}, Strength: {strength}")
        if key is None:
            router = BanditRouter(
                model_registry=registry,
                context_model="sentence-transformers/all-MiniLM-L6-v2",
                alpha=0.5,
                embedding_dim=embeddings.shape[1]
            )
        else:
            router = BanditRouter.load_from_benchmark(
                model_registry=registry,
                context_model="sentence-transformers/all-MiniLM-L6-v2",
                alpha=0.5,
                prior_strength=strength,
                priors_meta_path=priors_meta_path,
                benchmark_key=key
            )
        
        regret = run_sim(router, embeddings, cluster_ids)
        results[f"{key}_{strength}"] = regret
        print(f"  Final Cum Regret: {regret:.4f}")

    cold_regret = results["None_0"]
    print("\nRegret Reduction vs Cold Start:")
    for k, v in results.items():
        if k == "None_0": continue
        reduction = (cold_regret - v) / cold_regret * 100
        print(f"  {k}: {reduction:.2f}%")

if __name__ == "__main__":
    run_regret_comparison()
