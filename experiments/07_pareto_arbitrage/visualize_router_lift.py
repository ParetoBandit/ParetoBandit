#!/usr/bin/env python3
"""
visualize_router_lift.py

The "Money Chart": Demonstrates the "Router Lift" by plotting BanditGPT profiles 
against the Pareto frontier of a 9-model portfolio.
"""

import sys
import json
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root and experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments"))

from src.bandit_gpt.router import BanditRouter, DEFAULT_CONTEXT_MODEL, OptimizationProfile
from src.bandit_gpt.utils import ExperimentBurnIn
from utils.data_loader import load_oracle_rewards, load_model_registry
from sentence_transformers import SentenceTransformer

def get_model_cost_1k(model: dict) -> float:
    """Calculate blended cost per 1k tokens in USD."""
    # Models registry uses cost per 1M
    input_cost = model.get("price_1m_input") or model.get("input_cost_per_m") or 0.0
    output_cost = model.get("price_1m_output") or model.get("output_cost_per_m") or 0.0
    cost_per_1m = 0.5 * input_cost + 0.5 * output_cost
    return cost_per_1m / 1000.0

def compute_pareto_frontier(points: list) -> list:
    """Compute the Pareto frontier (convex hull)."""
    # Sort by cost ascending
    sorted_points = sorted(points, key=lambda x: x["cost"])
    frontier = []
    max_quality = -float('inf')
    for p in sorted_points:
        if p["quality"] > max_quality:
            frontier.append(p)
            max_quality = p["quality"]
    return frontier

def main(prior_n_effective: float = 20.0, alpha: float = 0.1):
    print("="*70)
    print("ROUTER LIFT VISUALIZATION: THE MONEY CHART")
    print(f"Parameters: N={prior_n_effective}, alpha={alpha}")
    print("="*70)

    # 1. Load Data
    print("📦 Loading model registry and setup...")
    project_root = Path(__file__).parent.parent.parent
    full_registry = load_model_registry()
    splits_path = project_root / "experiments" / "01_effectiveness" / "results" / "splits.json"
    priors_file = project_root / "data" / "priors_warmup.joblib"
    
    # Initialize ExperimentBurnIn
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    burner = ExperimentBurnIn(full_registry, {}, splits_path, encoder=encoder)
    
    # 2. Get dev and test prompts WITH rewards automatically joined
    print("📊 Loading canonical splits with rewards...")
    (dev_prompts, dev_rewards), (test_prompts_pool, test_rewards) = burner.get_splits(load_rewards=True)
    print(f"  ✓ Dev prompts: {len(dev_prompts)} with {len(dev_rewards)} reward entries")
    print(f"  ✓ Test prompts: {len(test_prompts_pool)} with {len(test_rewards)} reward entries")
    
    # Combine for backward compatibility with rest of script
    all_rewards = {**dev_rewards, **test_rewards}


    # 3. Identify 9-Model Portfolio (>=50% coverage in test set)
    print("⚖️ Identifying 9-model portfolio...")
    
    model_coverage = defaultdict(int)
    for p in test_prompts_pool:
        rewards = all_rewards.get(p, {})
        for m in rewards:
            model_coverage[m] += 1
            
    min_coverage = len(test_prompts_pool) * 0.5
    available_models = [m for m in full_registry if model_coverage[m] >= min_coverage]
    registry = {m: full_registry[m] for m in available_models}
    
    print(f"  ✓ {len(available_models)} models identified.")
    
    # 3. Compute Baseline Stats (Using ACTUAL Test Distribution)
    print("📉 Computing static model baselines on TEST set...")
    model_points = []
    
    for m_id in available_models:
        # Get actual rewards and token counts for this model on the test set
        relevant_prompts = [p for p in test_prompts_pool if m_id in all_rewards.get(p, {})]
        
        if not relevant_prompts:
            continue
            
        qualities = []
        costs = []
        
        for p in relevant_prompts:
            # 1. Get Reward
            qualities.append(all_rewards[p][m_id])
            
            # 2. Get ACTUAL Cost (Not Heuristic)
            # Use rough estimation based on word count for consistency with router logic
            # (src.bandit_gpt.router.estimate_tokens_rough)
            est_input = len(str(p).split()) * 1.3 
            est_output = 600 # Standard output assumption for evaluation
            
            m_data = registry[m_id]
            in_rate = (m_data.get("input_cost_per_m") or 0) / 1_000_000
            out_rate = (m_data.get("output_cost_per_m") or 0) / 1_000_000
            
            real_cost = (est_input * in_rate) + (est_output * out_rate)
            costs.append(real_cost * 1000) # Scale to $/1k
            
        model_points.append({
            "id": m_id,
            "name": registry[m_id].get("display_name", m_id),
            "cost": np.mean(costs),     # Actual average cost on this test set
            "quality": np.mean(qualities)
        })
    
    frontier = compute_pareto_frontier(model_points)
    
    # 4. Burn in Once, Evaluate with Multiple Profiles
    print("\n🚀 Burning in router...")
    
    # Manual Burn-In with Production Router
    # FIX: Use Randomized Profiles to ensure exploration of the full Pareto frontier.
    
    # 2. Get splits ONCE and freeze them
    # (Already loaded at line 69, but we verify disjointness here)
    assert set(dev_prompts).isdisjoint(set(test_prompts_pool)), "CRITICAL: Train/Test contamination detected!"
    
    # Do NOT call generate_curriculum(dev_prompts) again if it re-shuffles.
    # Instead, iterate directly over the dev_prompts list we already have.
    print(f"  🔥 Executing Heterogeneous Burn-in on {len(dev_prompts)} prompts...")
    
    # Shuffle dev prompts manually to ensure random order
    curriculum = list(dev_prompts)
    random.shuffle(curriculum)
    
    # Explicitly use pca_23.joblib
    pca_path = project_root / "artifacts" / "pca_23.joblib"
    
    router = BanditRouter.create(
        burner.registry,
        context_encoder=burner.encoder,
        priors=str(priors_file),
        prior_n_effective=prior_n_effective,
        pca_path=str(pca_path)
    )
    
    # FIX: Force Aggressive Exploration during Burn-in
    # Set alpha=2.0 (High Exploration) to ensure unvisited arms get tried.
    # This prevents the "Rich Get Richer" problem where one lucky model dominates early.
    router.bandit.alpha = 2.0  
    
    print("  🔥 Executing Heterogeneous Burn-in (Randomized Profiles + High Alpha)...")
    training_profiles = ["cost_saver", "arbitrage", "max_quality"]
    
    # We essentially reimplement perform_burn_in but with random profiles
    for prompt in tqdm(curriculum, desc="  Burn-in"):
        # Simulate diverse traffic: Pick a random user profile for this request
        p_name = random.choice(training_profiles)
        
        # Route (Debug prints suppressed by counter > 3)
        model_id, _ = router.route(prompt, profile=p_name)
        
        # Oracle Reward
        reward = all_rewards.get(prompt, {}).get(model_id, 0.0)
        router.update(model_id, prompt, reward)
    
    # Reset Alpha for Evaluation (User specified parameter)
    router.bandit.alpha = alpha
    
    actual_test_prompts = test_prompts_pool
    # ---------------------------------
    
    # router, actual_test_prompts = burner.create_burned_in_router(...) -> REPLACED BY ABOVE

    
    print(f"✓ Burn-in complete. Evaluating {len(actual_test_prompts)} test prompts with different profiles...")
    
    profiles = [
        ("Cost Saver", "cost_saver", "green", "D"),
        ("Arbitrage", "arbitrage", "blue", "P"),
        ("Max Quality", "max_quality", "red", "*")
    ]
    
    router_results = []
    
    # Evaluate the SAME router with different profiles
    # This is the realistic deployment scenario: one router, multiple user preferences
    for label, profile_name, color, marker in profiles:
        print(f"  Evaluating {label} profile...")
        
        # CRITICAL: Do NOT set alpha=0 here! 
        # With alpha=0 (pure exploitation), the bandit ignores profile weights and just
        # picks the model with highest learned mean. Keeping alpha>0 allows uncertainty 
        # to propagate and enables different profiles to steer selection via their weight ratios.
        costs = []
        qualities = []
        
        for p in tqdm(actual_test_prompts, desc=f"Eval {label}", leave=False):
            model_id, _ = router.route(p, profile=profile_name)
            if model_id in all_rewards.get(p, {}):
                model_data = registry[model_id]
                
                # Calculate ACTUAL cost (using same logic as baselines)
                est_input = len(str(p).split()) * 1.3 
                est_output = 600 # Standard output assumption
                
                in_rate = (model_data.get("input_cost_per_m") or 0) / 1_000_000
                out_rate = (model_data.get("output_cost_per_m") or 0) / 1_000_000
                
                real_cost = (est_input * in_rate) + (est_output * out_rate)
                costs.append(real_cost * 1000) # Scale to $/1k
                
                qualities.append(all_rewards[p][model_id])
        
        
        router_results.append({
            "label": label,
            "cost": np.mean(costs),
            "quality": np.mean(qualities),
            "color": color,
            "marker": marker
        })
        print(f"    - Cost: ${np.mean(costs):.4f}/1k, Quality: {np.mean(qualities)*100:.2f}%")

    # 5. Plotting
    print("\n🎨 Generating plot...")
    plt.figure(figsize=(10, 7))
    
    # Static Models
    costs = [p["cost"] for p in model_points]
    qualities = [p["quality"] for p in model_points]
    plt.scatter(costs, qualities, color="grey", alpha=0.5, label="Static Models")
    
    for p in model_points:
        plt.annotate(p["name"][:15], (p["cost"], p["quality"]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, alpha=0.7)

    # Static Frontier
    f_costs = [p["cost"] for p in frontier]
    f_qualities = [p["quality"] for p in frontier]
    plt.plot(f_costs, f_qualities, "--", color="black", alpha=0.6, label="Static Pareto Frontier")
    
    # Router Points
    for res in router_results:
        plt.scatter(res["cost"], res["quality"], color=res["color"], marker=res["marker"], 
                    s=200, label=f"BanditGPT: {res['label']}", edgecolors="black", zorder=5)
        
        # Add a subtle arrow from fixed frontier to router if lift is positive
        # (Find model at similar or lower cost)
        f_at_cost = [p for p in frontier if p["cost"] <= res["cost"]]
        if f_at_cost:
            best_f = max(f_at_cost, key=lambda x: x["quality"])
            if res["quality"] > best_f["quality"]:
                plt.annotate("", xy=(res["cost"], res["quality"]), xytext=(res["cost"], best_f["quality"]),
                             arrowprops=dict(arrowstyle="->", color=res["color"], lw=1.5, alpha=0.5))

    # Formatting
    plt.xscale('log')
    plt.xlabel("Average Cost ($ / 1k tokens) [Log Scale]", fontsize=12)
    plt.ylabel("Quality (HLE Accuracy)", fontsize=12)
    plt.title("The 'Money Chart': Router Lift over Static Frontier", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    
    # Set limits to focus on the action
    all_costs = costs + [r["cost"] for r in router_results]
    plt.xlim(min(all_costs) * 0.5, max(all_costs) * 2.0)
    plt.ylim(0, 1.05) # Accuracy is 0-1
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "router_lift_chart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Success! Chart saved to: {plot_path}")
    
    # --- PRINT REQUESTED TABLES ---
    print("\n" + "="*60)
    print("TABLE 1: PARETO FRONTIER MODELS (STATIC)")
    print("="*60)
    print(f"| {'Model Name':<25} | {'Cost ($/1k)':<12} | {'Quality':<8} |")
    print(f"| {'-'*25} | {'-'*12} | {'-'*8} |")
    
    # Sort frontier by cost for readability
    sorted_frontier = sorted(frontier, key=lambda x: x["cost"])
    for p in sorted_frontier:
        print(f"| {p['name'][:25]:<25} | ${p['cost']:<11.4f} | {p['quality']:.2%} |")
        
    print("\n" + "="*60)
    print("TABLE 2: BANDIT PROFILE PERFORMANCE")
    print("="*60)
    print(f"| {'Profile':<25} | {'Cost ($/1k)':<12} | {'Quality':<8} |")
    print(f"| {'-'*25} | {'-'*12} | {'-'*8} |")
    
    for res in router_results:
        print(f"| {res['label']:<25} | ${res['cost']:<11.4f} | {res['quality']:.2%} |")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize router lift vs Pareto frontier")
    parser.add_argument("--N", type=float, default=100.0, help="Prior N effective (default: 100.0)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Exploration alpha (default: 0.05)")
    args = parser.parse_args()
    main(prior_n_effective=args.N, alpha=args.alpha)
