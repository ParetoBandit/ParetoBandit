import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
try:
    from banditgpt import BanditRouter, transform_hle_to_prior, sigmoid
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from banditgpt import BanditRouter, transform_hle_to_prior, sigmoid

def load_ground_truth(reward_file: str):
    """Calculates mean empirical reward (0-1) for each model from test set."""
    print(f"Loading ground truth from {reward_file}...")
    model_scores = defaultdict(list)
    
    path = Path(reward_file)
    if not path.exists():
        raise FileNotFoundError(f"CRITICAL: Ground truth reward file {reward_file} not found. Cannot proceed with Figure 5 evaluation.")
        
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            val = item.get("reward_logit")
            model_id = item["model_id"]
            
            if not item.get("ok") or val is None:
                continue # Strict: Skip failed or missing evaluations
            
            logit = float(val)
            reward = sigmoid(logit)
            model_scores[model_id].append(reward)

    # Calculate mean
    means = {}
    for m, scores in model_scores.items():
        if len(scores) >= 1: # Require at least one ground truth sample
            means[m] = np.mean(scores)
    
    if not means:
        raise ValueError(f"CRITICAL: No valid rewards found in {reward_file}.")
        
    return means

def get_frontier(costs, qualities, ids):
    """Calculates Pareto Frontier indices."""
    sorted_indices = np.argsort(costs)
    s_costs = np.array(costs)[sorted_indices]
    s_quals = np.array(qualities)[sorted_indices]
    s_ids = np.array(ids)[sorted_indices]
    
    frontier_x = []
    frontier_y = []
    current_max_q = -1.0
    
    for c, q in zip(s_costs, s_quals):
        if q > current_max_q:
            frontier_x.append(c)
            frontier_y.append(q)
            current_max_q = q
            
    return frontier_x, frontier_y

def plot_landscape(ax, costs, qualities, latencies, ids, models, title, y_label):
    """Helper to plot one landscape."""
    # Scatter
    sc = ax.scatter(costs, qualities, c=latencies, cmap='RdYlGn_r', s=100, alpha=0.8, edgecolors='black')
    
    # Frontier
    fx, fy = get_frontier(costs, qualities, ids)
    ax.plot(fx, fy, 'k--', linewidth=1.5, alpha=0.6, label="Efficient Frontier")
    
    # Annotate Frontier
    processed = set()
    for cx, cy in zip(fx, fy):
        for i, mid in enumerate(ids):
            if abs(costs[i] - cx) < 1e-9 and abs(qualities[i] - cy) < 1e-9:
                label = models[i].replace("Google ", "").replace("Anthropic ", "").replace("DeepSeek ", "")
                ax.annotate(label, (cx, cy), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
                processed.add(mid)
                break
                
    # Annotate DeepSeek/Gemini if not on frontier
    targets = ["deepseek-chat-v3", "gemini-3-pro"]
    for t in targets:
        for i, mid in enumerate(ids):
            if t in mid and mid not in processed:
                 label = models[i].replace("Google ", "").replace("Anthropic ", "").replace("DeepSeek ", "")
                 ax.annotate(label, (costs[i], qualities[i]), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color='blue')

    ax.set_xscale('log')
    ax.set_xlabel('Cost ($/1M) - Log Scale')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    return sc

def main():
    root_dir = Path(__file__).parent.parent.parent
    project_root = root_dir.parent
    data_dir = project_root / "banditgpt" / "data"
    
    # 1. Load Registry and Initialize Router
    print("Loading model registry...")
    with open(project_root / "banditgpt" / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    router = BanditRouter.create(model_registry=registry) # Uses defaults
    
    # Load Ground Truth (Posterior) from test_rewards
    gt_rewards = load_ground_truth(data_dir / "test_rewards.jsonl")

    # Arrays
    p_costs, p_priors, p_lat, p_ids, p_names = [], [], [], [], []
    r_costs, r_rewards, r_lat, r_ids, r_names = [], [], [], [], [] # For Posterior

    for mid, meta in registry.items():
        # Cost validation
        cost = meta.get("price_1m_blended")
        if cost is None or cost <= 0:
            i = meta.get("input_cost_per_m")
            o = meta.get("output_cost_per_m")
            if i and o: cost = (i*3 + o)/4
            else: continue
            
        # Latency validation - Use scraped OpenRouter data (lowest median TTFT by provider)
        lat = meta.get("lowest_latency_seconds")
        if lat is None or lat <= 0:
            # Fallback to old field if new data not available
            lat = meta.get("time_to_first_token_seconds")
            if lat is None or lat <= 0:
                continue
            
        name = meta.get("display_name", mid)
        
        # 1. T=0 Prior (HLE Transformed)
        hle = meta.get("hle")
        if hle and hle > 0:
            prior = transform_hle_to_prior(hle) # Uses updated Sigmoid (20% Center)
            p_costs.append(cost)
            p_priors.append(prior)
            p_lat.append(lat)
            p_ids.append(mid)
            p_names.append(name)
            
        # 2. T=500 Posterior (Real Mean)
        if mid in gt_rewards:
            mean_r = gt_rewards[mid]
            r_costs.append(cost)
            r_rewards.append(mean_r)
            r_lat.append(lat)
            r_ids.append(mid)
            r_names.append(name)

    # --- PLOT 3: Quality Constraints (Figure 5c) ---
    # "What is the Cheapest Model that is Smart Enough (MMLU > X)?"
    
    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=False) # Separate Y-scales
    
    # Plot 1: T=0 (Prior)
    sc1 = plot_landscape(ax1, p_costs, p_priors, p_lat, p_ids, p_names, 
                         "T=0: Benchmark Estimates (Prior)\nSigmoid Center = 20% HLE", 
                         "Predicted Utility (Prior Probability)")
    
    # Plot 2: T=500 (Posterior)
    sc2 = plot_landscape(ax2, r_costs, r_rewards, r_lat, r_ids, r_names, 
                         "T=500: Real-World Performance (Posterior)\nEmpirical Mean Reward", 
                         "Actual Utility")

    # Shared Colorbar
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], fraction=0.02, pad=0.04)
    cbar.set_label('Latency (TTFT, sec) - Green=Fast', fontsize=10)
    
    plt.suptitle("Figure 5: The 'Before & After' Strategy - Optimization Landscape Shift", fontsize=16)
    
    # Save to specific folder
    output_dir = Path("final_release/kdd_paper/figure_5")
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "figure5_before_after.png"
    
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"Saved {outfile}")

if __name__ == "__main__":
    main()
