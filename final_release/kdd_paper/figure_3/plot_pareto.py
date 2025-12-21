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
    # Look in the final_release root (two levels up)
    root_dir = base_dir.parent.parent
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

    # 5. Plot
    print("Generating Figure 3...")
    plt.figure(figsize=(12, 8))
    
    costs = [d["cost"] for d in data]
    confs = [d["confidence"] for d in data]
    
    plt.scatter(costs, confs, alpha=0.4, color='#7f8c8d', label='All Models (80+)')
    
    # Plot Pareto Front line
    p_costs = [d["cost"] for d in pareto_points]
    p_confs = [d["confidence"] for d in pareto_points]
    
    # Add a step line to show the frontier
    plt.step(p_costs, p_confs, where='post', color='#e74c3c', linestyle='--', alpha=0.6, label='Pareto Frontier')
    plt.scatter(p_costs, p_confs, color='#e74c3c', s=80, edgecolors='black', label='Pareto Optimal Models')
    
    # Label Pareto Models
    for i, d in enumerate(pareto_points):
        # Manual offsets and alignment for the first few models which are very crowded
        ha = 'left'
        if i == 0: # Gemma
            y_off, x_off, ha = 10, -10, 'right'
        elif i == 1: # Llama 3.2
            y_off, x_off, ha = -25, 5, 'left'
        elif i == 2: # DeepSeek
            y_off, x_off, ha = 20, 5, 'left'
        elif i == 3: # Qwen
            y_off, x_off, ha = -15, 10, 'left'
        elif i == 4: # gpt-oss-20B
            y_off, x_off, ha = 15, 10, 'left'
        else:
            # Alternate others
            y_off, x_off = (5, 5) if i % 2 == 0 else (-12, 5)
            
        plt.annotate(d["name"], (d["cost"], d["confidence"]), 
                     xytext=(x_off, y_off), textcoords='offset points', 
                     fontsize=8, fontweight='bold', ha=ha)
    
    plt.xscale('log')
    plt.xlabel("Cost per 1M Blended Tokens ($)", fontsize=12)
    plt.ylabel("Learned Specialist Confidence (||\u03b8||)", fontsize=12)
    plt.title("Figure 3: Specialist Confidence vs. Cost (Pareto Frontier)", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    # Add a note about methodology
    plt.figtext(0.15, 0.02, "Note: Confidence is measured by the Euclidean norm of the learned LinUCB weights (\u03b8) derived from HLE priors.", 
                fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = base_dir / "figure3_pareto.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
