import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    from final_release.bandit import BanditRouter
except (ImportError, ValueError):
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    project_dir = base_dir.parent.parent
    data_dir = project_dir / "data"
    
    # 1. Load Data
    print("Loading data...")
    with open(project_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: dict(m) for m in models_data["models"]}
    
    # NOTE: We use raw HLE scores (0.0-0.4) as requested, despite the potential scale mismatch
    # with sigmoid rewards (0.5-1.0). The bandit will learn to adjust.
    
    prompts = []
    rewards = []
    
    for prefix in ["train", "test"]:
        p_path = data_dir / f"{prefix}_prompts.jsonl"
        r_path = data_dir / f"{prefix}_rewards.jsonl"
        if p_path.exists():
            with open(p_path) as f:
                for line in f: prompts.append(json.loads(line))
        if r_path.exists():
            with open(r_path) as f:
                for line in f: rewards.append(json.loads(line))
            
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
            
    # Embed
    print("Embedding prompts...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Take a robust sample
    np.random.seed(42)
    sample_indices = np.random.choice(len(prompts), min(2000, len(prompts)), replace=False)
    sampled_prompts = [prompts[i] for i in sample_indices]
    
    embeddings = encoder.encode([p["prompt"] for p in sampled_prompts], normalize_embeddings=True)
    cluster_ids = [p["cluster_id"] for p in sampled_prompts]
    
    # 2. Simulation Setup
    modes = [
        ("Cold Start (100% Feedback)", False, 1.0, "gray", "--", 1.5, 1.0),
        ("HLE Priors (No Feedback)", True, 0.0, "red", "-", 4.0, 0.25),
        ("HLE Priors + 1% Feedback", True, 0.01, "orange", "-", 3.0, 0.6),
        ("HLE Priors + 10% Feedback", True, 0.1, "green", "-", 2.0, 0.8),
        ("HLE Priors + 50% Feedback", True, 0.5, "purple", "-", 1.8, 0.9),
        ("HLE Priors + 100% Feedback", True, 1.0, "blue", "-", 1.2, 1.0),
    ]
    
    priors_meta_path = data_dir / "priors_meta_large.npz"
    num_seeds = 10
    target_requests = 100
    
    # Consistent prompt sequences across ALL runs
    sequences = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        idx = np.arange(len(embeddings))
        np.random.shuffle(idx)
        valid_seq = []
        for i in idx:
            if cluster_ids[i] in truth:
                valid_seq.append(i)
                if len(valid_seq) >= target_requests: break
        sequences.append(valid_seq)
    
    results = {}
    
    for name, use_priors, freq, color, style, lw, line_alpha in modes:
        print(f"Simulating: {name}...")
        all_runs = []
        
        for seed_idx, seq in enumerate(sequences):
            np.random.seed(seed_idx)
            
            if use_priors:
                router = BanditRouter.load_from_benchmark(
                    model_registry=registry,
                    context_model="sentence-transformers/all-MiniLM-L6-v2",
                    alpha=0.1, 
                    prior_strength=40.0,
                    forgetting_factor=1.0,
                    priors_meta_path=priors_meta_path
                )
            else:
                router = BanditRouter(
                    model_registry=registry,
                    context_model="sentence-transformers/all-MiniLM-L6-v2",
                    alpha=0.1,
                    embedding_dim=384,
                    forgetting_factor=1.0
                )
            
            cum_regret = 0.0
            regrets = []
            
            for idx in seq:
                x = embeddings[idx]
                cid = cluster_ids[idx]
                cluster_rewards = truth[cid]
                
                valid_candidates = list(cluster_rewards.keys())
                best_reward = max(cluster_rewards.values())
                
                x_with_bias = np.append(x, 1.0)
                # Pure selection
                chosen, _ = router.bandit.select_arm(x_with_bias, candidates=valid_candidates)
                
                observed = cluster_rewards.get(chosen, 0.0)
                
                if freq > 0 and np.random.random() < freq:
                    router.bandit.update(chosen, x_with_bias, observed)
                
                regret = max(0, best_reward - observed)
                cum_regret += regret
                regrets.append(cum_regret)
            all_runs.append(regrets)
            
        min_len = min(len(r) for r in all_runs)
        results[name] = (np.mean([r[:min_len] for r in all_runs], axis=0), color, style, lw, line_alpha)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    print("\nSummary (Avg Cumulative Regret):")
    checkpoints = [1, 10, 50, 100]
    print(f"{'Mode':<30} | " + " | ".join([f"Req {c:<4}" for c in checkpoints]))
    print("-" * 80)

    for name, (curve, color, style, lw, line_alpha) in results.items():
        plt.plot(curve, label=name, color=color, linestyle=style, linewidth=lw, alpha=line_alpha)
        stats = [f"{curve[min(c-1, len(curve)-1)]:8.2f}" for c in checkpoints]
        print(f"{name:<30} | " + " | ".join(stats))
        
    plt.xlabel("Number of Requests")
    plt.ylabel("Cumulative Regret")
    plt.title("Figure 6: Robustness to Sparse Feedback (Priors vs Cold Start)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = base_dir / "figure6_sparsity.png"
    plt.savefig(out_file)
    print(f"\nSaved plot to {out_file}")

if __name__ == "__main__":
    main()
