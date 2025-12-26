"""
Figure 7: Prior Strength vs. Feedback Stability (The Learning Tax)

Clean, rigorous evaluation showing:
- Real BanditRouter with actual LinUCB implementation
- Test set ONLY (1,000 prompts, strict hold-out)
- Individual ground truth rewards
- Comparison of N=0 (Cold), N=5 (Warm), N=40 (Stubborn)
- 50 seeds per data point for statistical significance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from banditgpt import BanditRouter
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from banditgpt import BanditRouter

def main():
    base_dir = Path(__file__).parent
    root_dir = base_dir.parent.parent
    project_root = root_dir.parent
    data_dir = project_root / "banditgpt" / "data"
    
    print("="*60)
    print("FIGURE 7: PRIOR STRENGTH VS FEEDBACK STABILITY")
    print("="*60)
    
    # Configuration
    FEEDBACK_RATES = [0.01, 0.1, 0.5, 1.0]  # 1%, 10%, 50%, 100%
    SEEDS = 50                              # For statistical significance
    REQUESTS = 500                          # Per seed
    
    # Load Models
    print("\n[1/5] Loading model registry...")
    with open(project_root / "banditgpt" / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load Test Data
    print("\n[2/5] Loading TEST data...")
    test_prompts = []
    with open(data_dir / "test_prompts.jsonl") as f:
        for line in f:
            test_prompts.append(json.loads(line))
            
    test_rewards = []
    with open(data_dir / "test_rewards.jsonl") as f:
        for line in f:
            test_rewards.append(json.loads(line))
            
    # Build ground truth lookup
    ground_truth = {}
    for r in test_rewards:
        if not r.get("ok"):
            continue
            
        # Use prompt if available, fallback to (cluster_id, model_id)
        if "prompt" in r:
            lookup_key = (r["prompt"], r["model_id"])
        else:
            lookup_key = (r["cluster_id"], r["model_id"])
            
        ground_truth[lookup_key] = r["reward_logit"]
            
    print(f"  Loaded {len(test_prompts)} prompts and {len(ground_truth)} reward entries")

    # [3/5] Pre-compute Embeddings
    print("\n[3/5] Computing embeddings for eval set...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    prompt_texts = [p["prompt"] for p in test_prompts]
    all_embeddings = encoder.encode(prompt_texts, normalize_embeddings=True, show_progress_bar=True)
    cluster_ids = [p["cluster_id"] for p in test_prompts]

    # [4/5] Run Simulations
    print(f"\n[4/5] Running simulations ({len(FEEDBACK_RATES)} rates × 3 N-values × {SEEDS} seeds)...")
    
    results = {
        "N=0 (Cold)": [],
        "N=5 (Warm)": [],
        "N=40 (Stubborn)": []
    }
    
    # Simulation configurations
    configs = [
        ("N=0 (Cold)", "none", 0.0),      # No priors
        ("N=5 (Warm)", "large", 5.0),     # Weak priors
        ("N=40 (Stubborn)", "large", 40.0) # Strong priors
    ]
    
    for label, prior_type, n_val in configs:
        print(f"\n  Testing {label}...")
        rate_results = []
        
        for rate in FEEDBACK_RATES:
            print(f"    Feedback Rate: {rate*100:.0f}%...")
            seed_regrets = []
            
            for seed in range(SEEDS):
                np.random.seed(seed)
                
                # Sample 500 requests for this seed
                indices = np.random.choice(len(all_embeddings), REQUESTS, replace=False)
                fold_embeddings = all_embeddings[indices]
                fold_cluster_ids = [cluster_ids[i] for i in indices]
                fold_prompts = [prompt_texts[i] for i in indices]
                
                # Initialize router
                router = BanditRouter.create(
                    model_registry=registry,
                    priors=prior_type,
                    prior_strength=n_val,
                    exploration="safe"
                )
                
                # Run session
                regret = run_session(router, fold_embeddings, fold_prompts, fold_cluster_ids, ground_truth, feedback_rate=rate)
                seed_regrets.append(regret)
            
            rate_results.append(np.mean(seed_regrets))
            
        results[label] = rate_results

    # [5/5] Plot Results
    print("\n[5/5] Generating Figure 7...")
    plt.figure(figsize=(10, 6))
    
    styles = {
        "N=0 (Cold)": ('k--', 'o'),
        "N=5 (Warm)": ('g-', 's'),
        "N=40 (Stubborn)": ('r-', '^')
    }
    
    x = [r * 100 for r in FEEDBACK_RATES]
    for label, y_values in results.items():
        ls, marker = styles[label]
        plt.plot(x, y_values, ls, marker=marker, linewidth=2, markersize=8, label=label)

    plt.xlabel('Feedback Rate (%)', fontsize=12)
    plt.ylabel(f'Mean Cumulative Regret (T={REQUESTS})', fontsize=12)
    plt.title('Figure 7: Prior Strength vs. Feedback Stability\nPreventing the "Learning Tax"', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = base_dir / "figure7_prior_strength.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    print("\n✅ COMPLETE!")

def run_session(router, embeddings, prompt_texts, cluster_ids, ground_truth, feedback_rate):
    """Run a single session and return final cumulative regret."""
    cumulative_regret = 0.0
    
    for embedding, prompt, cluster_id in zip(embeddings, prompt_texts, cluster_ids):
        # Route
        selected_model_id, _ = router.route(embedding.tolist())
        
        # Get ground truth
        oracle_reward, selected_reward = get_rewards(prompt, cluster_id, selected_model_id, ground_truth)
        
        # Accumulate regret
        cumulative_regret += (oracle_reward - selected_reward)
        
        # Feedback (Probabilistic)
        if np.random.random() < feedback_rate:
            # Try prompt-level first, then cluster
            reward_logit = ground_truth.get((prompt, selected_model_id))
            if reward_logit is None:
                reward_logit = ground_truth.get((cluster_id, selected_model_id), 0.0)
                
            trace_id = router.routing_logs[-1].trace_id
            router.process_feedback(trace_id, reward_logit)
            
    return cumulative_regret

def get_rewards(prompt, cluster_id, selected_model_id, ground_truth):
    """Find oracle and selected rewards [0, 1] for a prompt."""
    
    # Extract rewards for this specific prompt (with cluster fallback)
    possible_rewards = {}
    
    # Get all model IDs available in ground truth
    all_model_ids = set(m for _, m in ground_truth.keys())
    
    for mid in all_model_ids:
        # Try prompt-level first
        logit = ground_truth.get((prompt, mid))
        if logit is None:
            # Fallback to cluster-level
            logit = ground_truth.get((cluster_id, mid))
            
        if logit is not None:
            possible_rewards[mid] = 1 / (1 + np.exp(-logit))
            
    if not possible_rewards:
        # Fallback to neutral reward if no data at all
        return 0.5, 0.5
        
    oracle_reward = max(possible_rewards.values())
    selected_reward = possible_rewards.get(selected_model_id, 0.5) 
    
    return oracle_reward, selected_reward

if __name__ == "__main__":
    main()
