import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .bandit import BanditRouter
except (ImportError, ValueError):
    try:
        from final_release.bandit import BanditRouter
    except (ImportError, ValueError):
        from bandit import BanditRouter

def main():
    base_dir = Path(__file__).parent
    
    # 1. Load Models and Costs
    print("Loading models and costs...")
    root_dir = Path(__file__).parent.parent.parent
    with open(root_dir / "models.json") as f:
        models_data = json.load(f)
    registry = {m["openrouter_id"]: m for m in models_data["models"]}
    
    # 2. Initialize Router with HLE Priors
    print("Initializing router with HLE priors...")
    priors_meta_path = root_dir / "data/priors_meta_large.npz"
    router = BanditRouter.load_from_benchmark(
        model_registry=registry,
        context_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5,
        prior_strength=20.0,
        priors_meta_path=priors_meta_path
    )
    
    # 3. Extract Confidence (||theta||) and Cost
    print("Extracting confidence and cost for all models...")
    data = []
    for m_id in router.bandit.models:
        if m_id not in registry: continue
        
        # Calculate theta: theta = A_inv @ b
        A_inv = router.bandit.A_inv[m_id]
        b = router.bandit.b[m_id]
        theta = A_inv @ b
        confidence = np.linalg.norm(theta)
        
        cost = registry[m_id].get("price_1m_blended", 0.0)
        
        # Skip models with zero confidence (not learned/no data)
        if confidence < 1e-6: continue
        
        data.append({
            "id": m_id,
            "name": registry[m_id].get("name", m_id),
            "confidence": confidence,
            "cost": cost
        })
    
    # 4. Identify Pareto Front
    # Objectives: Maximize Confidence, Minimize Cost
    # Sort by cost ascending
    data.sort(key=lambda x: x["cost"])
    
    pareto_points = []
    max_conf = -1.0
    for d in data:
        if d["confidence"] > max_conf:
            pareto_points.append(d)
            max_conf = d["confidence"]
            d["is_pareto"] = True
        else:
            d["is_pareto"] = False

    print("Pareto Optimal Models:")
    for p in pareto_points:
        print(f"  {p['name']} ({p['id']}): Cost={p['cost']:.4f}, Confidence={p['confidence']:.4f}")

    # 5. Plot
    print("Generating Figure 3...")
    plt.figure(figsize=(10, 6))
    
    costs = [d["cost"] for d in data]
    confs = [d["confidence"] for d in data]
    
    plt.scatter(costs, confs, alpha=0.3, color='#7f8c8d', s=20, label='All Models (80+)')
    
    # Plot Pareto Front line
    p_costs = [d["cost"] for d in pareto_points]
    p_confs = [d["confidence"] for d in pareto_points]
    
    # Add a step line to show the frontier
    plt.step(p_costs, p_confs, where='post', color='#e74c3c', linewidth=2, alpha=0.8, label='Pareto Frontier')
    plt.scatter(p_costs, p_confs, color='#e74c3c', s=100, edgecolors='black', zorder=5, label='Pareto Optimal Models')
    
    # Label Pareto Models
    for i, d in enumerate(pareto_points):
        # Use a more robust labeling strategy with high zorder
        plt.annotate(
            d["name"].split('(')[0].strip(),
            xy=(d["cost"], d["confidence"]),
            xytext=(5, 8),
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            zorder=10, # Ensure labels are in front
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec='gray', lw=0.5)
        )
    
    plt.xscale('log')
    plt.xlim(left=min(costs) * 0.5, right=max(costs) * 1.5)
    plt.xlabel("Cost ($ per 1M blended tokens) - Log Scale", fontsize=10)
    plt.ylabel("Expert Confidence (||θ||)", fontsize=10)
    plt.title("Figure 3: The Pareto Frontier of Expert Intuition", fontsize=12, fontweight='bold')
    
    # Further expand Y-axis to avoid "squished" look
    min_c, max_c = min(confs), max(confs)
    padding = (max_c - min_c) * 0.2
    plt.ylim(min_c - padding, max_c + padding)
    
    plt.legend(loc='lower right', fontsize=9, framealpha=0.9)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    
    # Add a note about methodology
    plt.figtext(0.15, 0.02, "Note: Confidence is measured by the Euclidean norm of the learned LinUCB weights (θ) derived from HLE priors.", 
                fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_file = base_dir / "figure3_pareto.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()
