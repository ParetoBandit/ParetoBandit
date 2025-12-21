import json
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from .bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    # Load Models
    print("Loading models...")
    with open(base_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load All Data (Train + Test)
    print("Loading all data...")
    prompts = []
    rewards = []
    
    files = [
        ("train_prompts.jsonl", "train_rewards.jsonl"),
        ("test_prompts.jsonl", "test_rewards.jsonl")
    ]
    
    for p_file, r_file in files:
        with open(data_dir / p_file) as f:
            for line in f:
                prompts.append(json.loads(line))
        with open(data_dir / r_file) as f:
            for line in f:
                rewards.append(json.loads(line))
                
    # Build Truth
    truth = {}
    for r in rewards:
        if r.get("ok"):
            c = r["cluster_id"]
            m = r["model_id"]
            logit = r.get("reward_logit", 0.0)
            val = 1.0 / (1.0 + np.exp(-logit))
            if c not in truth: truth[c] = {}
            truth[c][m] = val
            
    # Embed All Prompts
    print(f"Embedding {len(prompts)} total prompts...")
    prompt_texts = [p["prompt"] for p in prompts]
    cluster_ids = [p["cluster_id"] for p in prompts]
    
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(prompt_texts, normalize_embeddings=True)
    
    # Shuffle Data
    indices = np.arange(len(embeddings))
    np.random.seed(42)
    np.random.shuffle(indices)
    embeddings = embeddings[indices]
    cluster_ids = [cluster_ids[i] for i in indices]
    
    # Run Simulation on Full Dataset (Prior from 33k Prompts)
    print("Running Simulation with Large Priors (33k prompts)...")
    
    # Load Large Priors Metadata
    priors_meta_path = base_dir / "data/priors_meta_large.npz"
    
    # Initialize Routers
    # Cold Start
    cold_router = BanditRouter(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5,
        embedding_dim=embeddings.shape[1]
    )
    
    # HLE Prior (using large covariance)
    hle_router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5,
        prior_strength=20.0,
        reward_mode="logit",
        priors_meta_path=priors_meta_path
    )
    
    def run_sim(router):
        regrets = []
        cum_regret = 0.0
        for i, x in enumerate(embeddings):
            cid = cluster_ids[i]
            cluster_rewards = truth.get(cid, {})
            best_reward = max(cluster_rewards.values()) if cluster_rewards else 0.0
            
            # Select
            chosen, _, _ = router.bandit.select_arm(x)
            observed = cluster_rewards.get(chosen, 0.0)
            
            # Update
            router.bandit.update(chosen, x, observed)
            
            # Regret
            regret = best_reward - observed
            cum_regret += regret
            regrets.append(cum_regret)
        return regrets

    cold_curve = run_sim(cold_router)
    hle_curve = run_sim(hle_router)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cold_curve, label="Cold Start", linestyle="--", color="gray")
    plt.plot(hle_curve, label="HLE Prior (33k Prompts Covariance)", linewidth=2, color="blue")
    plt.xlabel("Requests")
    plt.ylabel("Cumulative Regret")
    plt.title("Figure 1: HLE Prior vs Cold Start (Full Dataset, Large Prior)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = base_dir / "figure1_regret.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    
    final_red = (cold_curve[-1] - hle_curve[-1]) / cold_curve[-1] * 100
    print(f"Final Regret Reduction: {final_red:.2f}%")

if __name__ == "__main__":
    main()
