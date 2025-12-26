"""
Figure 10: Prior Sensitivity Analysis (Tuning N-Value)

Clean, rigorous hyperparameter validation using:
- Real BanditRouter with actual LinUCB implementation
- TRAIN set for tuning (4,000 prompts)
- Sweep of Prior Strength (N) from 1 to 150
- 10 seeds per data point for statistical significance
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
    print("FIGURE 10: PRIOR SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Configuration
    STRENGTH_VALUES = [1, 5, 10, 20, 40, 60, 80, 100, 150]
    SEEDS = 10                              # For hyperparameter tuning
    REQUESTS = 500                          # Per seed
    
    # Load Models
    print("\n[1/5] Loading model registry...")
    with open(project_root / "banditgpt" / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # Load TRAIN Data (for tuning)
    print("\n[2/5] Loading TRAIN data (for hyperparameter tuning)...")
    train_prompts = []
    with open(data_dir / "train_prompts.jsonl") as f:
        for line in f:
            train_prompts.append(json.loads(line))
            
    train_rewards = []
    with open(data_dir / "train_rewards.jsonl") as f:
        for line in f:
            train_rewards.append(json.loads(line))
            
    # Build ground truth lookup
    ground_truth = {}
    for r in train_rewards:
        if not r.get("ok"):
            continue
            
        # Use prompt if available, fallback to (cluster_id, model_id)
        if "prompt" in r:
            lookup_key = (r["prompt"], r["model_id"])
        else:
            lookup_key = (r["cluster_id"], r["model_id"])
            
        ground_truth[lookup_key] = r["reward_logit"]
            
    print(f"  Loaded {len(train_prompts)} prompts and {len(ground_truth)} reward entries")

    # [3/5] Pre-compute Embeddings
    print("\n[3/5] Computing embeddings for train set...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    prompt_texts = [p["prompt"] for p in train_prompts]
    all_embeddings = encoder.encode(prompt_texts, normalize_embeddings=True, show_progress_bar=True)
    cluster_ids = [p["cluster_id"] for p in train_prompts]

    # [4/5] Run Sweep
    print(f"\n[4/5] Running N-value sweep ({len(STRENGTH_VALUES)} values × {SEEDS} seeds)...")
    
    mean_regrets = []
    std_regrets = []
    
    for n_val in STRENGTH_VALUES:
        print(f"  Testing Prior Strength N={n_val}...")
        seed_regrets = []
        
        for seed in range(SEEDS):
            np.random.seed(seed)
            
            # Sample prompts for this trial
            indices = np.random.choice(len(all_embeddings), REQUESTS, replace=False)
            fold_embeddings = all_embeddings[indices]
            fold_cluster_ids = [cluster_ids[i] for i in indices]
            fold_prompts = [prompt_texts[i] for i in indices]
            
            # Initialize router
            router = BanditRouter.create(
                model_registry=registry,
                priors="large",
                prior_strength=float(n_val),
                exploration="safe"
            )
            
            # Run session
            regret = run_session(router, fold_embeddings, fold_prompts, fold_cluster_ids, ground_truth)
            seed_regrets.append(regret)
        
        mean_regrets.append(np.mean(seed_regrets))
        std_regrets.append(np.std(seed_regrets))

    # [5/5] Plot Results
    print("\n[5/5] Generating Figure 10...")
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(STRENGTH_VALUES, mean_regrets, yerr=std_regrets, fmt='-o', color='b', 
                 capsize=5, capthick=2, linewidth=2, label='Empirical Regret (Training Set)')
    
    # Highlight the chosen value (N=40)
    plt.axvline(x=40, color='r', linestyle='--', alpha=0.7, label='Production Setting (N=40)')
    
    plt.xlabel('Prior Strength (N)', fontsize=12)
    plt.ylabel(f'Mean Cumulative Regret (T={REQUESTS})', fontsize=12)
    plt.title('Figure 10: Prior Sensitivity Analysis\nOptimizing for Initial Performance vs. Adaptability', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = base_dir / "figure_10_prior_tuning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    print("\n✅ COMPLETE!")

def run_session(router, embeddings, prompt_texts, cluster_ids, ground_truth):
    """Run a single session (10% feedback rate) and return final cumulative regret."""
    cumulative_regret = 0.0
    feedback_rate = 0.1 # Standard feedback for tuning
    
    for embedding, prompt, cluster_id in zip(embeddings, prompt_texts, cluster_ids):
        # Route
        selected_model_id, _ = router.route(embedding.tolist())
        
        # Get ground truth
        oracle_reward, selected_reward = get_rewards(prompt, cluster_id, selected_model_id, ground_truth)
        
        # Accumulate regret
        cumulative_regret += (oracle_reward - selected_reward)
        
        # Feedback (Probabilistic)
        if np.random.random() < feedback_rate:
            # Try prompt-level first
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
        # For training/tuning, we might have missing clusters. 
        # But for rigorous KDD, we expect full coverage.
        return 0.9, 0.1 # Extreme penalty for missing data to flag it
        
    oracle_reward = max(possible_rewards.values())
    selected_reward = possible_rewards.get(selected_model_id, 0.5) 
    
    return oracle_reward, selected_reward

if __name__ == "__main__":
    main()
