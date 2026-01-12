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

def estimate_prompt_cost(prompt: str, model_data: dict, cost_per_1k: float = True, 
                        prompt_id: str = None, model_id: str = None, 
                        oracle_rewards: dict = None) -> float:
    """Estimate the cost for a single prompt execution.
    
    Args:
        prompt: The prompt text
        model_data: Model configuration dict with pricing info
        cost_per_1k: If True, return cost scaled to $/1k, else raw cost
        prompt_id: Optional prompt identifier for looking up real usage
        model_id: Optional model identifier for looking up real usage
        oracle_rewards: Optional oracle data with real token counts
    
    Returns:
        Estimated cost in dollars (scaled to /1k if cost_per_1k=True)
    """
    # Try to use real token counts if available
    if prompt_id and model_id and oracle_rewards:
        prompt_data = oracle_rewards.get(prompt_id, {})
        if model_id in prompt_data and isinstance(prompt_data[model_id], dict):
            usage = prompt_data[model_id].get('usage', {})
            if 'input_tokens' in usage and 'output_tokens' in usage:
                est_input = usage['input_tokens']
                est_output = usage['output_tokens']
            else:
                # Fallback to heuristic
                est_input = len(str(prompt).split()) * 1.3
                est_output = 600
        else:
            # Fallback to heuristic
            est_input = len(str(prompt).split()) * 1.3
            est_output = 600
    else:
        # Heuristic estimation (conservative)
        est_input = len(str(prompt).split()) * 1.3
        est_output = 600  # Standard output assumption for evaluation
    
    in_rate = (model_data.get("input_cost_per_m") or 0) / 1_000_000
    out_rate = (model_data.get("output_cost_per_m") or 0) / 1_000_000
    
    raw_cost = (est_input * in_rate) + (est_output * out_rate)
    return raw_cost * 1000 if cost_per_1k else raw_cost

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

def main(prior_n_effective: float = 20.0, alpha: float = 0.1, num_runs: int = 1,
         warmup_path: str = None, splits_path: str = None, pca_path: str = None):
    print("="*70)
    print("ROUTER LIFT VISUALIZATION: THE MONEY CHART")
    print(f"Parameters: N={prior_n_effective}, alpha={alpha}, Runs={num_runs}")
    print("="*70)

    # 1. Load Data
    print("📦 Loading model registry and setup...")
    project_root = Path(__file__).parent.parent.parent
    full_registry = load_model_registry()
    
    # Use provided paths or fall back to defaults
    if splits_path is None:
        splits_path = project_root / "experiments" / "01_effectiveness" / "results" / "splits.json"
    else:
        splits_path = Path(splits_path)
    
    if warmup_path is None:
        warmup_path = project_root / "data" / "priors_warmup_9_models.joblib"
    else:
        warmup_path = Path(warmup_path)
        
    if pca_path is None:
        pca_path = project_root / "artifacts" / "pca_23.joblib"
    else:
        pca_path = Path(pca_path)
    
    # Initialize ExperimentBurnIn
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    burner = ExperimentBurnIn(full_registry, {}, splits_path, encoder=encoder)
    
    # 2. Get dev and test prompts WITH rewards automatically joined
    print("📊 Loading canonical splits with rewards...")
    (dev_prompts, dev_rewards), (test_prompts_pool, test_rewards) = burner.get_splits(load_rewards=True)
    print(f"  ✓ Dev prompts: {len(dev_prompts)} with {len(dev_rewards)} reward entries")
    print(f"  ✓ Test prompts: {len(test_prompts_pool)} with {len(test_rewards)} reward entries")
    

    # 3. Use Full Model Portfolio (Including Specialists)
    # CRITICAL FIX: Remove arbitrary 50% coverage filter to allow specialist models.
    # Pessimistic imputation handles specialists fairly: quality=0.0 on unevaluated prompts.
    # This demonstrates the bandit's true strength: learning when to leverage sparse specialists.
    print("⚖️ Using full model portfolio (including specialists)...")
    registry = full_registry
    available_models = list(full_registry.keys())
    
    print(f"  ✓ {len(available_models)} models in portfolio.")
    
    # 3. Compute Baseline Stats with Pessimistic Imputation
    # CRITICAL FIX: Use pessimistic imputation (quality=0.0 for missing rewards)
    # to eliminate survivorship bias that artificially inflates weak model scores
    print("📉 Computing static model baselines on TEST set (with pessimistic imputation)...")
    model_points = []
    
    for m_id in available_models:
        qualities = []
        costs = []
        
        # EVALUATE ON ALL TEST PROMPTS (not just those with recorded rewards)
        for p in test_prompts_pool:
            m_data = registry[m_id]
            
            # 1. Get Reward (Impute 0.0 if missing - PENALIZE MISSING DATA)
            rewards = burner.oracle_rewards.get(p, {})
            quality = rewards.get(m_id, 0.0)
            qualities.append(quality)
            
            # 2. Estimate cost (charged even on failure in production)
            costs.append(estimate_prompt_cost(p, m_data, cost_per_1k=True))
            
        model_points.append({
            "id": m_id,
            "name": registry[m_id].get("display_name", m_id),
            "cost": np.mean(costs),     # Average cost across ALL test prompts
            "quality": np.mean(qualities)  # Average quality with failures counted as 0.0
        })
    
    frontier = compute_pareto_frontier(model_points)
    
    # 4. Monte Carlo Loop over Simulations
    print(f"\n� Running {num_runs} simulations for stability...")
    
    # Use default profile weights from OptimizationProfile
    profiles = [
        ("Cost Saver", OptimizationProfile.COST_SAVER, "green", "D"),
        ("Smart Shopper", OptimizationProfile.ARBITRAGE, "blue", "P"),
        ("Rational Luxury", OptimizationProfile.MAX_QUALITY, "red", "*")
    ]
    
    # Aggregate results: {profile_name: {"costs": [], "qualities": []}}
    # Create results dict using profile dict as key
    mc_results = {str(p_dict): {"costs": [], "qualities": [], "label": label} for label, p_dict, _, _ in profiles}
    
    for run_i in range(1, num_runs + 1):
        # Set deterministic seeds that vary by run
        seed = 42 + run_i
        random.seed(seed)
        np.random.seed(seed)
        
        # Burn in Once per Run
        # Manual Burn-In with Production Router
        router = BanditRouter.create(
            burner.registry,
            context_encoder=burner.encoder,
            priors="warmup",
            warmup_path=str(warmup_path),
            prior_n_effective=prior_n_effective,
            pca_path=str(pca_path)
        )
        
        router.bandit.alpha = 2.0  # High Exploration during Burn-in
        training_profiles = [OptimizationProfile.COST_SAVER, OptimizationProfile.ARBITRAGE, OptimizationProfile.MAX_QUALITY]
        
        # Shuffle curriculum for this run
        curriculum = list(dev_prompts)
        random.shuffle(curriculum)
        
        for prompt in curriculum:
            p_name = random.choice(training_profiles)
            model_id, _ = router.route(prompt, profile=p_name)
            reward = burner.oracle_rewards.get(prompt, {}).get(model_id, 0.0)
            router.update(model_id, prompt, reward)
        
        # Reset Alpha for Evaluation
        router.bandit.alpha = alpha
        
        # Evaluate multiple profiles on the same burned-out router with pessimistic imputation
        for label, profile_dict, color, marker in profiles:
            run_costs = []
            run_qualities = []
            
            # EVALUATE ON ALL TEST PROMPTS (not just those with recorded rewards)
            for p in test_prompts_pool:
                model_id, _ = router.route(p, profile=profile_dict)
                model_data = registry[model_id]
                
                # 1. Get Reward (Impute 0.0 if missing - PENALIZE MISSING DATA)
                rewards = burner.oracle_rewards.get(p, {})
                quality = rewards.get(model_id, 0.0)
                run_qualities.append(quality)
                
                # 2. Estimate cost (charged even on failure)
                run_costs.append(estimate_prompt_cost(p, model_data, cost_per_1k=True))
            
            profile_key = str(profile_dict)
            mc_results[profile_key]["costs"].append(np.mean(run_costs))
            mc_results[profile_key]["qualities"].append(np.mean(run_qualities))
        
        print(f"  ✓ Run {run_i}/{num_runs} complete.")

    # Calculate average router results with error bars
    router_results = []
    for label, profile_dict, color, marker in profiles:
        profile_key = str(profile_dict)
        avg_cost = np.mean(mc_results[profile_key]["costs"])
        avg_quality = np.mean(mc_results[profile_key]["qualities"])
        std_cost = np.std(mc_results[profile_key]["costs"])
        std_quality = np.std(mc_results[profile_key]["qualities"])
        
        router_results.append({
            "label": label,
            "cost": avg_cost,
            "quality": avg_quality,
            "std_cost": std_cost,
            "std_quality": std_quality,
            "color": color,
            "marker": marker
        })
        print(f"    - {label}: Cost: ${avg_cost:.4f}/1k ± ${std_cost:.4f}, Quality: {avg_quality*100:.2f}% ± {std_quality*100:.2f}%")

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
    
    # Router Points with Error Bars (Statistical Significance)
    for res in router_results:
        # Plot error bars for both cost (x-axis) and quality (y-axis)
        plt.errorbar(res["cost"], res["quality"], 
                     xerr=res["std_cost"], yerr=res["std_quality"],
                     fmt=res["marker"], color=res["color"], markersize=12,
                     capsize=5, capthick=2, elinewidth=2,
                     label=f"BanditGPT: {res['label']}", 
                     markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        
        # Add a subtle arrow from fixed frontier to router if lift is positive
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
    plt.title(f"The 'Money Chart': Router Lift (Avg of {num_runs} Runs)", fontsize=14, fontweight='bold')
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
        
    print("\n" + "="*80)
    print("TABLE 2: BANDIT PROFILE PERFORMANCE (Monte Carlo Avg ± Std)")
    print("="*80)
    print(f"| {'Profile':<25} | {'Cost ($/1k)':<20} | {'Quality':<20} |")
    print(f"| {'-'*25} | {'-'*20} | {'-'*20} |")
    
    for res in router_results:
        cost_str = f"${res['cost']:.4f} ± ${res['std_cost']:.4f}"
        q_str = f"{res['quality']:.2%} ± {res['std_quality']:.2%}"
        print(f"| {res['label']:<25} | {cost_str:<20} | {q_str:<20} |")
    print("="*80 + "\n")

    print("\n" + "="*60)
    print("TABLE 3: ALL PORTFOLIO MODELS (SORTED BY COST)")
    print("="*60)
    print(f"| {'Model Name':<25} | {'Cost ($/1k)':<12} | {'Quality':<8} |")
    print(f"| {'-'*25} | {'-'*12} | {'-'*8} |")
    
    sorted_all = sorted(model_points, key=lambda x: x["cost"])
    for p in sorted_all:
        print(f"| {p['name'][:25]:<25} | ${p['cost']:<11.4f} | {p['quality']:.2%} |")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize router lift vs Pareto frontier")
    parser.add_argument("--N", type=float, default=20.0, help="Prior N effective (default: 20.0)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Exploration alpha (default: 0.1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of Monte Carlo runs (default: 1)")
    parser.add_argument("--warmup-path", type=str, default=None, 
                        help="Path to warmup priors .joblib file (default: data/priors_warmup_9_models.joblib)")
    parser.add_argument("--splits-path", type=str, default=None,
                        help="Path to splits.json file (default: experiments/01_effectiveness/results/splits.json)")
    parser.add_argument("--pca-path", type=str, default=None,
                        help="Path to PCA model .joblib file (default: artifacts/pca_23.joblib)")
    args = parser.parse_args()
    main(prior_n_effective=args.N, alpha=args.alpha, num_runs=args.runs,
         warmup_path=args.warmup_path, splits_path=args.splits_path, pca_path=args.pca_path)
