import json
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
try:
    from .bandit import BanditRouter
except (ImportError, ValueError):
    try:
        from final_release.bandit import BanditRouter
    except (ImportError, ValueError):
        from bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    root_dir = base_dir.parent.parent
    data_dir = root_dir / "data"
    
    # Load Models
    print("Loading models...")
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "models.json") as f:
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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def run_sim(router, fold_embeddings, fold_cluster_ids):
        regrets = []
        cum_regret = 0.0
        for i, x in enumerate(fold_embeddings):
            cid = fold_cluster_ids[i]
            cluster_rewards = truth.get(cid, {})
            best_reward = max(cluster_rewards.values()) if cluster_rewards else 0.0
            
            # Select
            # Append bias term to match BanditRouter's internal dimension (384 + 1)
            x_bias = np.append(x, 1.0)
            chosen, _ = router.bandit.select_arm(x_bias)
            observed = cluster_rewards.get(chosen, 0.0)
            
            # Update
            router.bandit.update(chosen, x_bias, observed)
            
            # Regret
            regret = best_reward - observed
            cum_regret += regret
            regrets.append(cum_regret)
        return regrets

    alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    results = {}

    print(f"Tuning Alpha for COLD START: {alphas}")
    
    for alpha in alphas:
        print(f"\nTesting Alpha = {alpha}...")
        all_cold_curves = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
            # We use the test_idx for each fold to simulate a fresh run on a subset
            fold_embeddings = embeddings[test_idx]
            fold_cluster_ids = [cluster_ids[i] for i in test_idx]
            
            # COLD START ROUTER (No Priors)
            cold_router = BanditRouter(
                model_registry=registry,
                context_model="sentence-transformers/all-MiniLM-L6-v2",
                alpha=alpha,
                embedding_dim=embeddings.shape[1]
            )
            
            cold_curve = run_sim(cold_router, fold_embeddings, fold_cluster_ids)
            all_cold_curves.append(cold_curve)
            
        # Aggregate Results
        min_len = min(len(c) for c in all_cold_curves)
        all_cold_curves = np.array([c[:min_len] for c in all_cold_curves])
        
        mean_cold = np.mean(all_cold_curves, axis=0)
        final_regret = mean_cold[-1]
        results[alpha] = final_regret
        print(f"  -> Final Cumulative Regret: {final_regret:.4f}")

    print("\n" + "="*40)
    print("FINAL RESULTS (COLD START)")
    print("="*40)
    best_alpha = None
    min_regret = float('inf')
    max_regret = float('-inf')
    
    for alpha, regret in results.items():
        print(f"Alpha {alpha:<4}: {regret:.4f}")
        if regret < min_regret:
            min_regret = regret
            best_alpha = alpha
        if regret > max_regret:
            max_regret = regret
            
    print("-" * 40)
    print(f"OPTIMAL ALPHA: {best_alpha} (Regret: {min_regret:.4f})")
    print(f"SPREAD: {max_regret - min_regret:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
