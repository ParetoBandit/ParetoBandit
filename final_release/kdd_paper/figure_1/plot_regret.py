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
        ("train_prompts.jsonl", "train_rewards.jsonl")
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
    print("Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_cold_curves = []
    all_hle_curves = []
    reductions = []
    
    # Load Large Priors Metadata
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    
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

    for fold, (train_idx, test_idx) in enumerate(kf.split(embeddings)):
        print(f" Processing Fold {fold+1}/5...")
        
        # We use the test_idx for each fold to simulate a fresh run on a subset
        fold_embeddings = embeddings[test_idx]
        fold_cluster_ids = [cluster_ids[i] for i in test_idx]
        
        # Initialize Routers for this fold
        cold_router = BanditRouter(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=1.0,
            embedding_dim=embeddings.shape[1]
        )
        
        hle_router = BanditRouter.load_from_benchmark(
            model_registry=registry,
            context_model="sentence-transformers/all-MiniLM-L6-v2",
            alpha=1.0,
            prior_strength=40.0,
            priors_meta_path=priors_meta_path
        )
        
        cold_curve = run_sim(cold_router, fold_embeddings, fold_cluster_ids)
        hle_curve = run_sim(hle_router, fold_embeddings, fold_cluster_ids)
        
        all_cold_curves.append(cold_curve)
        all_hle_curves.append(hle_curve)
        
        red = (cold_curve[-1] - hle_curve[-1]) / cold_curve[-1] * 100 if cold_curve[-1] > 0 else 0
        reductions.append(red)

    # Aggregate Results
    # Since folds might have slightly different sizes, we truncate to the minimum length
    min_len = min(len(c) for c in all_cold_curves)
    all_cold_curves = np.array([c[:min_len] for c in all_cold_curves])
    all_hle_curves = np.array([c[:min_len] for c in all_hle_curves])
    
    mean_cold = np.mean(all_cold_curves, axis=0)
    std_cold = np.std(all_cold_curves, axis=0) / np.sqrt(5)
    
    mean_hle = np.mean(all_hle_curves, axis=0)
    std_hle = np.std(all_hle_curves, axis=0) / np.sqrt(5)
    
    print("\nDEBUG DATA (First 50 Requests):")
    print(f"{'Req':<4} | {'Cold':<10} | {'HLE':<10} | {'Diff':<10}")
    print("-" * 40)
    for i in range(min(50, len(mean_cold))):
        print(f"{i:<4} | {mean_cold[i]:<10.4f} | {mean_hle[i]:<10.4f} | {mean_cold[i] - mean_hle[i]:<10.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(min_len)
    
    # Cold Start
    plt.plot(x_axis, mean_cold, label="Cold Start (Mean)", linestyle="--", color="gray")
    plt.fill_between(x_axis, mean_cold - std_cold, mean_cold + std_cold, color="gray", alpha=0.2)
    
    # HLE Prior
    plt.plot(x_axis, mean_hle, label="HLE Prior (26k Prompts, Mean)", linewidth=2, color="blue")
    plt.fill_between(x_axis, mean_hle - std_hle, mean_hle + std_hle, color="blue", alpha=0.1)
    
    plt.xlabel("Requests")
    plt.ylabel("Cumulative Regret")
    plt.title("Figure 1: HLE Prior vs Cold Start (5-Fold Cross Validation)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.axvline(x=35, color='black', linestyle='--', alpha=0.5, label="Phase Shift")
    
    # Use axes coordinates for robust positioning
    ylim = plt.ylim()
    y_pos = ylim[1] * 0.95  # Moved up to top 5% to avoid overlap with high-regret curve
    
    plt.text(17.5, y_pos, r"Exploration Dominance" + "\n" + r"($\alpha \sigma \gg \mu$)", 
             ha='center', va='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
             
    plt.text(min_len - 15, y_pos, "Prior Advantage\n(HLE Converged)", 
             ha='center', va='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    out_file = base_dir / "figure1_regret.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    
    mean_red = np.mean(reductions)
    std_red = np.std(reductions) / np.sqrt(5)
    print(f"Final Mean Regret Reduction: {mean_red:.2f}% ± {std_red:.2f}%")

if __name__ == "__main__":
    main()
