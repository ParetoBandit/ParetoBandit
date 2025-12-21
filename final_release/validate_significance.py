import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from scipy import stats
from .bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_file = base_dir / "significance_results.txt"
    
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
    
    # 5-Fold Cross Validation
    print("Running 5-Fold CV with Large Prior...")
    kf = KFold(n_splits=5)
    
    cold_regrets = []
    hle_regrets = []
    
    priors_meta_path = base_dir / "data/priors_meta_large.npz"
    if not priors_meta_path.exists():
        raise FileNotFoundError(f"Missing {priors_meta_path}. Run calc_priors_large.py first.")

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        print(f"Fold {fold+1}/5")
        
        # Test Data for this fold
        test_emb = embeddings[test_idx]
        test_clusters = [cluster_ids[i] for i in test_idx]
        
        # 1. Initialize Routers
        # Cold Start
        cold_router = BanditRouter(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=0.5,
            embedding_dim=embeddings.shape[1]
        )
        
        # HLE Prior (using fixed large covariance)
        hle_router = BanditRouter.load_from_benchmark(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=0.5,
            prior_strength=20.0,
            reward_mode="logit",
            priors_meta_path=priors_meta_path
        )
        
        # 2. Run Simulation
        c_regret = 0.0
        h_regret = 0.0
        
        for i, x in enumerate(test_emb):
            cid = test_clusters[i]
            cluster_rewards = truth.get(cid, {})
            best_reward = max(cluster_rewards.values()) if cluster_rewards else 0.0
            
            # Cold
            c_chosen, _, _ = cold_router.bandit.select_arm(x)
            c_obs = cluster_rewards.get(c_chosen, 0.0)
            cold_router.bandit.update(c_chosen, x, c_obs)
            c_regret += (best_reward - c_obs)
            
            # HLE
            h_chosen, _, _ = hle_router.bandit.select_arm(x)
            h_obs = cluster_rewards.get(h_chosen, 0.0)
            hle_router.bandit.update(h_chosen, x, h_obs)
            h_regret += (best_reward - h_obs)
            
        cold_regrets.append(c_regret)
        hle_regrets.append(h_regret)
        print(f"  Cold Regret: {c_regret:.2f}")
        print(f"  HLE Regret:  {h_regret:.2f}")

    # Statistical Analysis
    cold_regrets = np.array(cold_regrets)
    hle_regrets = np.array(hle_regrets)
    
    reductions = (cold_regrets - hle_regrets) / cold_regrets * 100
    mean_red = np.mean(reductions)
    std_red = np.std(reductions)
    
    t_stat, p_val = stats.ttest_rel(cold_regrets, hle_regrets)
    
    output = []
    output.append("=" * 50)
    output.append("5-Fold Cross-Validation Results (Large Prior)")
    output.append("=" * 50)
    output.append(f"Mean Regret Reduction: {mean_red:.2f}% ± {std_red:.2f}%")
    output.append(f"P-Value (paired t-test): {p_val:.5f}")
    if p_val < 0.05:
        output.append("RESULT: Statistically Significant (p < 0.05)")
    else:
        output.append("RESULT: Not Significant")
    output.append("=" * 50)
    
    output_str = "\n".join(output)
    print(output_str)
    
    with open(results_file, "w") as f:
        f.write(output_str)
    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()
