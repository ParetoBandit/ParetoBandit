import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    
    # 1. Load Router
    print("Initializing router...")
    router = BanditRouter.create()
    
    # 2. Load Sample Prompts
    print("Loading test prompts...")
    prompts = []
    with open(data_dir / "test_prompts.jsonl") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    
    # Take a representative sample
    sample_size = 90
    np.random.seed(42)
    selected_prompts = np.random.choice(prompts, sample_size, replace=False)
    
    # 3. Define Constraints to vary
    # We'll vary max_cost and see the shift
    # Expanded range to see extremely low-cost model transitions
    cost_limits = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5] # $/request
    
    results = {} # limit -> [model_counts]
    
    print("Varying cost constraints...")
    for limit in cost_limits:
        selected_models = []
        for p in selected_prompts:
            try:
                model, _ = router.route(p, profile="quality_first", max_cost=limit)
                selected_models.append(model)
            except ValueError:
                # No model satisfies constraint
                selected_models.append("None")
        
        counts = {}
        for m in selected_models:
            counts[m] = counts.get(m, 0) + 1
        results[limit] = counts

    # 4. Process for Plotting
    # Identify top N models across all runs to keep the legend clean
    all_models = set()
    for counts in results.values():
        all_models.update(counts.keys())
    
    # Sort models by total popularity
    total_popularity = {m: sum(c.get(m, 0) for c in results.values()) for m in all_models}
    top_models = sorted(all_models, key=lambda m: total_popularity[m], reverse=True)[:6]
    if "None" not in top_models:
        top_models.append("None")

    # Prepare data for stacked bar plot
    plot_data = []
    for limit in cost_limits:
        counts = results[limit]
        row = []
        other_sum = 0
        for m in top_models:
            val = counts.get(m, 0)
            row.append(val)
        
        # Add models not in top_models to "Other"
        for m, val in counts.items():
            if m not in top_models:
                other_sum += val
        row.append(other_sum)
        plot_data.append(row)
    
    labels = top_models + ["Other"]
    plot_data = np.array(plot_data).T # Models x Limits
    
    # 5. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(cost_limits))
    
    colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(labels)))
    
    for i, model_row in enumerate(plot_data):
        ax.bar([str(l) for l in cost_limits], model_row, bottom=bottom, label=labels[i], color=colors[i])
        bottom += model_row
        
    ax.set_xlabel("Max Cost Constraint ($ per request)")
    ax.set_ylabel("Number of Requests")
    ax.set_title("Figure 5: Model Selection Shift vs. Cost Constraints")
    ax.legend(title="Selected Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_file = base_dir / "figure5_constraints.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

    # Save summary for description
    summary = {
        "cost_limits": cost_limits,
        "results": results
    }
    with open(base_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
