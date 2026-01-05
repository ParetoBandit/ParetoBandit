#!/usr/bin/env python3
"""
Transformed HLE Pareto Analysis (KDD-Grade)

Key Metric: Predicted Success Probability (Utility)
- Y-axis = "Predicted Success Probability" (0-100%)
- This visualizes "the router's brain" - what it sees internally
- Uses transform_hle_to_prior() to show the barbell dynamics:
  - Easy Mode: All models cluster at 95-99% (cost decides)
  - Hard Mode: Massive spread 1-99% (quality decides)

Evaluation Protocol:
1. BURN-IN: Train on training set
2. EVALUATE: For each test prompt, compute the transformed HLE of selected model
3. AGGREGATE: Average transformed HLE = "Expected Utility"
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import logging
import argparse

logging.getLogger("banditgpt").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter, DEFAULT_CONTEXT_MODEL, transform_hle_to_prior


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_with_hle():
    """
    Load prompts and registry with HLE scores.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    train_rewards_path = data_dir / "train_rewards_1k.jsonl"
    models_path = Path(__file__).parent.parent.parent / "models.json"
    
    # Load registry with HLE scores
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Build HLE lookup (raw scores)
    hle_lookup = {model_id: model.get("hle", 0) for model_id, model in registry.items()}
    avg_hle = np.mean(list(hle_lookup.values()))
    print(f"  HLE scores: {len(hle_lookup)} models, avg={avg_hle:.3f}")
    
    def load_prompts(path, label):
        """Load unique prompts from reward file"""
        prompts = set()
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    prompts.add(entry["prompt"])
        print(f"  {label}: {len(prompts)} prompts")
        return list(prompts)
    
    train_prompts = load_prompts(train_rewards_path, "Training")
    test_prompts = load_prompts(test_rewards_path, "Test")
    
    return train_prompts, test_prompts, registry, hle_lookup


def get_model_cost(model, input_tokens=100, output_tokens=200):
    if "price_1m_input" not in model or "price_1m_output" not in model:
        return None
    return (input_tokens * model["price_1m_input"] + output_tokens * model["price_1m_output"]) / 1_000_000


# =============================================================================
# TRANSFORMED HLE BASELINES
# =============================================================================

def compute_utility_oracle(registry, is_hard=True):
    """
    Oracle: Best possible model for EACH prompt.
    For mixed analysis, weighted average of best-possible in each tier.
    """
    best_hle = max(m.get("hle", 0) for m in registry.values())
    u_hard = transform_hle_to_prior(best_hle, is_hard_prompt=True)
    u_easy = transform_hle_to_prior(best_hle, is_hard_prompt=False)
    
    if is_hard:
        return u_hard
    
    # MIXED REALITY: 10.6% Hard, 89.4% Easy
    return (0.106 * u_hard) + (0.894 * u_easy)


def compute_utility_random(registry, is_hard=True):
    """
    Random router: Average transformed HLE when picking uniformly.
    """
    u_hard_list = [transform_hle_to_prior(m.get("hle", 0), is_hard_prompt=True) 
                 for m in registry.values()]
    u_easy_list = [transform_hle_to_prior(m.get("hle", 0), is_hard_prompt=False) 
                 for m in registry.values()]
    
    if is_hard:
        return np.mean(u_hard_list) if u_hard_list else 0
    
    # MIXED REALITY
    return (0.106 * np.mean(u_hard_list)) + (0.894 * np.mean(u_easy_list))


def compute_cheapest_utility(registry, is_hard=True):
    """
    Cheapest router: Utility of the cheapest model.
    """
    cheapest = min(
        [(m_id, get_model_cost(m), m.get("hle", 0)) for m_id, m in registry.items() if get_model_cost(m)],
        key=lambda x: x[1]
    )
    
    u_hard = transform_hle_to_prior(cheapest[2], is_hard_prompt=True)
    u_easy = transform_hle_to_prior(cheapest[2], is_hard_prompt=False)
    
    if is_hard:
        return u_hard, cheapest[1] * 1000, cheapest[0]
    
    # MIXED REALITY
    mixed_u = (0.106 * u_hard) + (0.894 * u_easy)
    return mixed_u, cheapest[1] * 1000, cheapest[0]


def compute_quality_first_utility(registry, is_hard=True):
    """
    Quality-first: Utility of the highest HLE model.
    """
    best = max(registry.items(), key=lambda x: x[1].get("hle", 0))
    cost = get_model_cost(best[1])
    hle = best[1].get("hle", 0)
    
    u_hard = transform_hle_to_prior(hle, is_hard_prompt=True)
    u_easy = transform_hle_to_prior(hle, is_hard_prompt=False)
    
    if is_hard:
        return u_hard, cost * 1000 if cost else 0, best[0]
    
    # MIXED REALITY
    mixed_u = (0.106 * u_hard) + (0.894 * u_easy)
    return mixed_u, cost * 1000 if cost else 0, best[0]


# =============================================================================
# BANDIT WITH TRANSFORMED HLE EVALUATION
# =============================================================================

def simulate_bandit_utility(train_prompts, test_prompts, registry, hle_lookup, 
                            profile_name="quality_first", seed=42, encoder=None):
    """
    Two-Phase Bandit evaluation using Transformed HLE as utility metric.
    Returns (avg_cost, avg_utility, model_selections)
    """
    router = BanditRouter.create(
        registry,
        exploration="safe",
        priors="hle",
        prior_n_effective=60.0,
        prior_structure_n_effective=10.0,
        context_encoder=encoder
    )
    
    # Phase 1: Burn-in (learning phase)
    random.seed(seed)
    shuffled_train = train_prompts.copy()
    random.shuffle(shuffled_train)
    
    for prompt in shuffled_train:
        selected, log = router.route(prompt, profile=profile_name, input_tokens=100)
        # Simulate reward based on transformed HLE
        hle_score = hle_lookup.get(selected, 0)
        # Use the router's own is_hard detection for consistency
        is_hard = router._is_hard_cluster(log.cluster_id) if hasattr(log, 'cluster_id') else False
        simulated_reward = transform_hle_to_prior(hle_score, is_hard_prompt=is_hard)
        router.process_feedback(log.request_id, simulated_reward)
    
    # Phase 2: Evaluate (GREEDY MODE)
    shuffled_test = test_prompts.copy()
    random.shuffle(shuffled_test)
    
    total_cost = 0.0
    total_utility = 0.0
    count = 0
    selections = defaultdict(int)
    easy_utilities = []
    hard_utilities = []
    
    original_alpha = router.bandit.alpha
    router.bandit.alpha = 0.0  # Greedy
    
    for prompt in shuffled_test:
        selected, log = router.route(prompt, profile=profile_name, input_tokens=100)
        
        if selected in registry:
            hle = hle_lookup.get(selected, 0)
            # Use the router's is_hard detection
            is_hard = router._is_hard_cluster(log.cluster_id) if hasattr(log, 'cluster_id') else False
            utility = transform_hle_to_prior(hle, is_hard_prompt=is_hard)
            
            model = registry[selected]
            cost = get_model_cost(model)
            
            if cost:
                total_cost += cost
                total_utility += utility
                count += 1
                selections[selected] += 1
                
                if is_hard:
                    hard_utilities.append(utility)
                else:
                    easy_utilities.append(utility)
    
    router.bandit.alpha = original_alpha
    
    if count > 0:
        return (total_cost / count * 1000, total_utility / count, dict(selections),
                np.mean(easy_utilities) if easy_utilities else 0,
                np.mean(hard_utilities) if hard_utilities else 0)
    return None, None, {}, 0, 0


def simulate_bandit_frontier_utility(train_prompts, test_prompts, registry, hle_lookup, encoder=None, n_trials=3):
    """Generate BanditRouter frontier using Transformed HLE as utility metric"""
    print(f"\n[4/4] Generating BanditRouter Utility Frontier...")
    
    frontier_points = []
    
    lambda_configs = [
        {"name": "Max Quality",  "w_q": 0.97, "w_c": 0.02, "w_l": 0.01},
        {"name": "Arbitrage",    "w_q": 0.65, "w_c": 0.30, "w_l": 0.05},
        {"name": "Budget",       "w_q": 0.10, "w_c": 0.85, "w_l": 0.05},
        {"name": "Ultra Cheap",  "w_q": 0.02, "w_c": 0.97, "w_l": 0.01},
    ]
    
    for config in lambda_configs:
        costs, utilities = [], []
        easy_utils, hard_utils = [], []
        all_selections = defaultdict(int)
        
        profile = {"w_q": config["w_q"], "w_c": config["w_c"], "w_l": config["w_l"]}
        
        for trial in range(n_trials):
            result = simulate_bandit_utility(
                train_prompts, test_prompts, registry, hle_lookup,
                profile_name=profile, seed=42+trial, encoder=encoder
            )
            if result[0] is not None:
                costs.append(result[0])
                utilities.append(result[1])
                easy_utils.append(result[3])
                hard_utils.append(result[4])
                for m, c in result[2].items():
                    all_selections[m] += c
        
        if costs:
            avg_util = np.mean(utilities)
            avg_easy = np.mean(easy_utils)
            avg_hard = np.mean(hard_utils)
            print(f"  {config['name']:15} [Q:{config['w_q']:.2f}, C:{config['w_c']:.2f}, L:{config['w_l']:.2f}] -> ${np.mean(costs):.4f}, "
                  f"Utility={avg_util*100:.1f}% (Easy={avg_easy*100:.1f}%, Hard={avg_hard*100:.1f}%)")
            frontier_points.append({
                "profile": config["name"],
                "w_q": config["w_q"],
                "w_c": config["w_c"],
                "w_l": config["w_l"],
                "cost_mean": np.mean(costs),
                "cost_std": np.std(costs),
                "utility_mean": np.mean(utilities),
                "utility_std": np.std(utilities),
                "easy_utility": avg_easy,
                "hard_utility": avg_hard,
                "selections": dict(all_selections)
            })
    
    return sorted(frontier_points, key=lambda x: x["cost_mean"])


# =============================================================================
# PLOTTING
# =============================================================================

COLORS = {
    "Budget": "#e74c3c",      # Red
    "Arbitrage": "#2980b9",   # Blue
    "Max Quality": "#95a5a6", # Gray
    "Oracle": "#27ae60",      # Green
    "random": "#bdc3c7",      # Light Gray
    "model_noise": "#dcdde1", 
    "model_signal": "#7f8c8d",
}
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D


def create_utility_plot(registry, bandit_frontier, oracle_utility, random_utility, output_path):
    """
    Create Pareto plot using Vertical Intervals for "Capability Gaps".
    - Top Marker (Easy): Empty circle (Ceiling)
    - Bottom Marker (Hard): Solid shape (Floor)
    - Vertical Line: The "Capability Gap" (Fragility)
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 1. Plot individual models as faint background intervals
    model_data = []
    for model_id, model in registry.items():
        cost = get_model_cost(model)
        hle = model.get("hle", 0)
        if cost and hle > 0:
            u_hard = transform_hle_to_prior(hle, is_hard_prompt=True)
            u_easy = transform_hle_to_prior(hle, is_hard_prompt=False)
            mixed_cost = cost * 1000
            
            # Capability Gap Line (Faint)
            plt.plot([mixed_cost, mixed_cost], [u_hard*100, u_easy*100], '-', 
                     color=COLORS["model_noise"], alpha=0.3, linewidth=1, zorder=1)
            # Easy/Hard markers (Tiny)
            plt.scatter(mixed_cost, u_easy*100, color=COLORS["model_noise"], s=10, alpha=0.3, zorder=1)
            plt.scatter(mixed_cost, u_hard*100, color=COLORS["model_signal"], s=10, alpha=0.3, zorder=1)
            
            model_data.append({"id": model_id, "cost": mixed_cost, "u_hard": u_hard})

    # Label top Hard performance models (Background only)
    sorted_models = sorted(model_data, key=lambda x: x["u_hard"], reverse=True)[:3]
    for m in sorted_models:
        plt.annotate(m["id"][:15], (m["cost"], m["u_hard"]*100), 
                    xytext=(3, -3), textcoords='offset points', fontsize=7, alpha=0.4)
    
    # 2. Plot Bandit Profiles as primary intervals
    if bandit_frontier:
        # Sort for line plotting
        frontier_sorted = sorted(bandit_frontier, key=lambda x: x["cost_mean"])
        b_costs = [p["cost_mean"] for p in frontier_sorted if p["profile"] != "Max Quality"]
        b_hards = [p["hard_utility"] * 100 for p in frontier_sorted if p["profile"] != "Max Quality"]
        
        # Hard Task Frontier Line (Connecting Budget -> Arbitrage)
        plt.plot(b_costs, b_hards, '--', color='gray', alpha=0.4, linewidth=1, zorder=1)
        
        for p in bandit_frontier:
            name = p["profile"]
            cost = p["cost_mean"]
            h = p["hard_utility"] * 100
            e = p["easy_utility"] * 100
            col = COLORS.get(name, "#7f8c8d")
            
            # A. The Interval Line (The Capability Gap)
            plt.plot([cost, cost], [h, e], '-', color=col, alpha=0.8, linewidth=4, zorder=2)
            
            # B. The Ceiling (Easy Tasks) - Empty Circle
            plt.scatter(cost, e, s=180, facecolors='white', edgecolors=col, linewidth=3, zorder=3)
            
            # C. The Floor (Hard Tasks) - Solid Shape
            marker = '*' if name == "Arbitrage" else 'o'
            size = 450 if name == "Arbitrage" else 180
            plt.scatter(cost, h, s=size, color=col, marker=marker, zorder=4, edgecolors='white', linewidths=1.5)
            
            # D. Annotations (The Story)
            gap = e - h
            if name == "Budget":
                plt.text(cost * 1.2, (h + e)/2, f"High Fragility\n(-{gap:.0f}% drop)", 
                         color=col, fontsize=10, va='center', fontweight='bold')
            elif name == "Arbitrage":
                plt.text(cost * 0.85, (h + e)/2, f"Robust\n(-{gap:.0f}%)", 
                         color=col, fontsize=11, ha='right', va='center', fontweight='bold')
            elif name == "Max Quality":
                plt.text(cost * 1.1, h - 5, "Sub-optimal\nSwitching", 
                         color=col, fontsize=9, ha='center', va='top')

    # 3. Baselines
    # Hard Oracle
    plt.axhline(y=oracle_utility*100, color=COLORS["Oracle"], linestyle='--', lw=2, alpha=0.8,
                label=f'Oracle Hard Ceiling: {oracle_utility*100:.1f}%', zorder=3)
    # Hard Random
    plt.axhline(y=random_utility*100, color=COLORS["random"], linestyle=':', lw=1.5,
                label=f'Random Hard: {random_utility*100:.1f}%', zorder=3)

    # 4. Formatting
    plt.xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    plt.xlabel('Average Cost ($/1k Tokens) [Log Scale]', fontsize=12, fontweight='bold')
    plt.ylabel('Success Probability (%)', fontsize=12, fontweight='bold')
    plt.title('Router Capability Gap: Reliability vs. Complexity', fontsize=18, fontweight='bold', loc='left')

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='gray', markerfacecolor='white', markersize=10, markeredgewidth=2, label='Success on Easy Prompts (Ceiling)'),
        Line2D([0], [0], marker='o', color='gray', label='Success on Hard Prompts (Floor)'),
        Line2D([0], [0], color='gray', lw=2, label='Capability Gap (Fragility)'),
        Line2D([0], [0], marker='*', color=COLORS["Arbitrage"], linestyle='None', markersize=15, label='Arbitrage Profile (Target)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)

    plt.ylim(0, 105)
    plt.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    # Also save as the specific requested name
    capability_path = Path(output_path).parent / 'router_capability_gap.png'
    plt.savefig(capability_path, dpi=300)
    
    print(f"\n✓ Capability Gap Plot saved to: {output_path}")
    print(f"✓ Capability Gap Plot (Secondary) saved to: {capability_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Transformed HLE Pareto Analysis")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation and just re-plot from pareto_data.json")
    args = parser.parse_args()

    print("=" * 80)
    print("TRANSFORMED HLE PARETO ANALYSIS")
    print("Y-axis = Predicted Success Probability (Utility)")
    print("=" * 80)
    
    data_file = Path(__file__).parent / "pareto_data.json"
    
    # Load data
    print("\n[1/4] Loading registry...")
    train_prompts, test_prompts, registry, hle_lookup = load_data_with_hle()
    
    if args.plot_only and data_file.exists():
        print(f"\n[SKIP] Simulation skipped. Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            saved = json.load(f)
            bandit_frontier = saved["bandit_frontier"]
            oracle_utility = saved["oracle_utility"]
            random_utility = saved["random_utility"]
    else:
        # Baselines (HARD MODE for high resolution)
        print("\n[2/4] Computing utility baselines (Hard Mode)...")
        oracle_utility = compute_utility_oracle(registry, is_hard=True)
        random_utility = compute_utility_random(registry, is_hard=True)
        
        # BanditRouter
        print("\n[3/4] Initializing encoder and running simulations...")
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
        
        bandit_frontier = simulate_bandit_frontier_utility(
            train_prompts, test_prompts, registry, hle_lookup, encoder=encoder, n_trials=3
        )
        
        # Save for fast iteration
        print(f"\n[SAVE] Saving simulation results to {data_file}...")
        with open(data_file, 'w') as f:
            json.dump({
                "bandit_frontier": bandit_frontier,
                "oracle_utility": float(oracle_utility),
                "random_utility": float(random_utility)
            }, f, indent=2)
    
    # Print model selections for best profile
    if bandit_frontier:
        best = max(bandit_frontier, key=lambda x: x["utility_mean"])
        print(f"\n  Best profile ({best['profile']}) model selections:")
        sorted_sel = sorted(best["selections"].items(), key=lambda x: -x[1])[:5]
        for model_id, cnt in sorted_sel:
            model = registry.get(model_id, {})
            name = model.get("display_name", model_id)[:25]
            hle = model.get("hle", 0) * 100
            util = transform_hle_to_prior(model.get("hle", 0), is_hard_prompt=True) * 100
            print(f"    {name}: {cnt}x (HLE={hle:.1f}% -> Utility={util:.1f}%)")
    
    # Create plot
    output_path = Path(__file__).parent / "pareto_frontier.png"
    create_utility_plot(registry, bandit_frontier, oracle_utility, random_utility, output_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Utility Analysis")
    print("=" * 80)
    
    if bandit_frontier:
        best = max(bandit_frontier, key=lambda x: x["utility_mean"])
        gap = oracle_utility - best["utility_mean"]
        
        print(f"\nOracle ceiling: {oracle_utility*100:.1f}%")
        print(f"BanditRouter best ({best['profile']}): {best['utility_mean']*100:.1f}% at ${best['cost_mean']:.4f}")
        print(f"  Easy prompts: {best['easy_utility']*100:.1f}%")
        print(f"  Hard prompts: {best['hard_utility']*100:.1f}%")
        print(f"Gap to Oracle: {gap*100:.1f}%")
        print(f"Advantage over Random: {(best['utility_mean'] - random_utility)*100:+.1f}%")
    
    print("\n✅ UTILITY PARETO ANALYSIS COMPLETE!")


if __name__ == "__main__":
    main()
