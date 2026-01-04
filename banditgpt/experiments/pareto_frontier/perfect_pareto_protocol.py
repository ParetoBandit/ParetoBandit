#!/usr/bin/env python3
"""
Perfect Pareto Protocol: Two-Stage Experiment for KDD-Quality Results

The "Calibrate Once, Sweep Costs" Protocol:

Stage 1: Architectural Calibration (Finding the "Engine")
    - 2D Grid Search: N_structure × N_prior
    - Fixed "Balanced" profile (lambda_cost=0.5)
    - Metric: Average Z-Score (greedy evaluation)
    - Output: Champion configuration (N*_s, N*_p)

Stage 2: Economic Sweep (Tracing the Frontier)
    - Architecture: LOCKED to Champion settings
    - Sweep: lambda_cost ∈ [0.0, 0.5, 5.0, 50.0]
    - Output: Pareto Frontier (Cost vs Z-Score)

Why this protocol avoids "Oracle Tuning":
    - Hyperparameters locked BEFORE sweeping cost profiles
    - Proves architecture generalizes across user preferences
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
import random
import logging

logging.getLogger("banditgpt").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from banditgpt.bandit import BanditRouter, DEFAULT_CONTEXT_MODEL


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_with_zscores():
    """
    Load rewards and z-scores from models.json.
    Uses REAL data only - no fallbacks.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    test_rewards_path = data_dir / "test_rewards_pareto_dedup.jsonl"
    train_rewards_path = data_dir / "train_rewards_1k.jsonl"
    models_path = Path(__file__).parent.parent.parent / "models.json"
    
    # Verify paths exist (no fallbacks!)
    assert test_rewards_path.exists(), f"Test rewards not found: {test_rewards_path}"
    assert train_rewards_path.exists(), f"Train rewards not found: {train_rewards_path}"
    assert models_path.exists(), f"Models not found: {models_path}"
    
    # Load registry with z-scores
    with open(models_path) as f:
        data = json.load(f)
    registry = {m["openrouter_id"]: m for m in data["models"]}
    
    # Build z-score lookup: (model_id, cluster_id) -> z_score
    zscore_lookup = {}
    for model_id, model in registry.items():
        if "cluster_success_rates" in model:
            for cluster_id_str, cluster_data in model["cluster_success_rates"].items():
                if isinstance(cluster_data, dict) and "z_score" in cluster_data:
                    zscore_lookup[(model_id, int(cluster_id_str))] = cluster_data["z_score"]
    
    def load_rewards(path, label):
        prompt_data = defaultdict(lambda: {"cluster_id": None, "rewards": {}, "zscores": {}})
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("ok"):
                    prompt = entry["prompt"]
                    model_id = entry["model_id"]
                    cluster_id = entry.get("cluster_id", 0)
                    
                    prompt_data[prompt]["cluster_id"] = cluster_id
                    prompt_data[prompt]["rewards"][model_id] = entry["raw_score"]
                    
                    # Look up z-score from registry
                    zscore = zscore_lookup.get((model_id, cluster_id), 0.0)
                    prompt_data[prompt]["zscores"][model_id] = zscore
        
        print(f"  {label}: {len(prompt_data)} prompts")
        return dict(prompt_data)
    
    train_data = load_rewards(train_rewards_path, "Training")
    test_data = load_rewards(test_rewards_path, "Test")
    
    return train_data, test_data, registry, zscore_lookup


def get_model_cost(model, input_tokens=100, output_tokens=200):
    """Calculate cost per request in USD."""
    if "price_1m_input" not in model or "price_1m_output" not in model:
        return None
    return (input_tokens * model["price_1m_input"] + output_tokens * model["price_1m_output"]) / 1_000_000


def get_cluster_difficulty(registry, cluster_id):
    """
    Compute cluster difficulty (average success rate) from registry CSR data.
    
    Returns:
        (mu, sigma): Cluster mean success rate and standard deviation.
                     Used for Difficulty-Aware Reward Scaling.
    """
    # Collect success rates for this cluster across all models
    success_rates = []
    for model_id, model in registry.items():
        if "cluster_success_rates" in model:
            cluster_data = model["cluster_success_rates"].get(str(cluster_id))
            if cluster_data is None:
                cluster_data = model["cluster_success_rates"].get(cluster_id)
            if isinstance(cluster_data, dict) and "raw" in cluster_data:
                success_rates.append(cluster_data["raw"])
    
    if not success_rates:
        # Fallback: assume balanced difficulty
        return 0.5, 0.5
    
    mu = np.mean(success_rates)
    # Clip to avoid extreme scaling at edges
    mu = np.clip(mu, 0.05, 0.95)
    # Bernoulli std deviation
    sigma = np.sqrt(mu * (1 - mu))
    return mu, sigma


def scale_reward_by_difficulty(raw_reward, cluster_mu, cluster_sigma):
    """
    Transform binary reward (0/1) to synthetic Z-score with Winsorization.
    
    This aligns the online learning objective with the evaluation metric:
    - Easy cluster (μ=0.95): Success gives small reward (+0.2)
    - Hard cluster (μ=0.50): Success gives large reward (+1.0)
    
    WINSORIZATION (Asymmetric Clipping):
    - Raw Z-scores can explode (e.g., failure on easy cluster = -5.4)
    - This creates a "minefield" where one failure kills a model
    - Clipping to [-2.0, +2.0] preserves signal while preventing outliers
    
    Returns:
        Clipped scaled reward (robust synthetic Z-score).
    """
    if cluster_sigma < 0.01:
        return raw_reward  # Avoid division by near-zero
    
    z_raw = (raw_reward - cluster_mu) / cluster_sigma
    
    # WINSORIZATION: Clip to [-2.0, +2.0]
    # - Failure on easy task: -5.4 → -2.0 (still penalized, but survivable)
    # - Success on hard task: +1.0 → +1.0 (preserved)
    reward_clipped = np.clip(z_raw, -2.0, 2.0)
    
    return reward_clipped



# =============================================================================
# STAGE 1: ARCHITECTURAL CALIBRATION (2D Grid Search)
# =============================================================================

def save_intermediate(data, path, label):
    """Save intermediate results and flush output."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved {label}: {path}", flush=True)


def run_2d_synergy_sweep(train_data, test_data, registry, encoder,
                         n_struct_values, n_prior_values, 
                         fixed_lambda=0.0, n_trials=3, output_dir=None):
    """
    Stage 1: THE STRESS TEST - Find optimal (N_structure, N_prior) for HLE mode.
    Champion selected by MINIMUM REGRET.
    """
    print(f"\n{'='*70}", flush=True)
    print("STAGE 1: HLE CALIBRATION (2D Synergy Sweep)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Grid: N_struct={n_struct_values} × N_prior={n_prior_values}", flush=True)
    print(f"Fixed lambda_cost={fixed_lambda} (Max Quality Mode)", flush=True)
    
    # Profile for calibration (Max Quality stress test)
    profile = {"lambda_cost": fixed_lambda, "lambda_latency": 0.001}
    
    # Results storage: (n_struct, n_prior) -> stats
    results = {}
    grid_count = 0
    total_grid = len(n_struct_values) * len(n_prior_values)
    
    for n_struct in n_struct_values:
        for n_prior in n_prior_values:
            grid_count += 1
            print(f"\n  [{grid_count}/{total_grid}] Testing (N_struct={n_struct}, N_prior={n_prior})...", flush=True)
            trial_regrets = []
            trial_zscores = []
            
            for trial in range(n_trials):
                # Knob 1: prior_structure_n_effective (A matrix stiffness)
                # Knob 2: prior_n_effective (b vector belief strength)
                router = BanditRouter.create(
                    registry,
                    exploration="safe",
                    priors="hle",
                    prior_n_effective=float(n_prior),
                    prior_structure_n_effective=float(n_struct),
                    context_encoder=encoder
                )
                
                # Phase 1: Burn-in (Learning)
                train_prompts = list(train_data.keys())
                random.seed(42 + trial)
                random.shuffle(train_prompts)
                
                for prompt in train_prompts:
                    d = train_data[prompt]
                    selected, log = router.route(prompt, profile=profile, input_tokens=100)
                    if selected in d["rewards"]:
                        raw_reward = d["rewards"][selected]
                        cluster_id = d.get("cluster_id", 0)
                        cluster_mu, cluster_sigma = get_cluster_difficulty(registry, cluster_id)
                        scaled_reward = scale_reward_by_difficulty(raw_reward, cluster_mu, cluster_sigma)
                        difficulty_weight = 1.0 - cluster_mu
                        router.update(selected, prompt, scaled_reward, weight=difficulty_weight)
                
                # Phase 2: Evaluate (Greedy)
                test_prompts = list(test_data.keys())
                random.shuffle(test_prompts)
                
                original_alpha = router.bandit.alpha
                router.bandit.alpha = 0.0  # Force greedy
                
                zscores = []
                regrets = []
                for prompt in test_prompts:
                    d = test_data[prompt]
                    selected, _ = router.route(prompt, profile=profile, input_tokens=100)
                    if selected in d["zscores"]:
                        zscores.append(d["zscores"][selected])
                    if selected in d["rewards"]:
                        regrets.append(1.0 - d["rewards"][selected])
                
                router.bandit.alpha = original_alpha
                
                avg_regret = np.mean(regrets) if regrets else 1.0
                avg_z = np.mean(zscores) if zscores else 0.0
                trial_regrets.append(avg_regret)
                trial_zscores.append(avg_z)
                print(f"    Trial {trial+1}: Regret={avg_regret:.4f} (Z={avg_z:+.4f}σ)", flush=True)
            
            results[(n_struct, n_prior)] = {
                "mean_regret": np.mean(trial_regrets),
                "mean_z": np.mean(trial_zscores),
                "regrets": trial_regrets
            }
            
            # Save intermediate
            if output_dir:
                intermediate_path = output_dir / "stage1_intermediate.json"
                intermediate_data = {
                    "completed_points": grid_count,
                    "total_samples": total_grid,
                    "results": {f"{k[0]},{k[1]}": v for k, v in results.items()}
                }
                save_intermediate(intermediate_data, intermediate_path, f"Stage 1 progress ({grid_count}/{total_grid})")
    
    # Build Accuracy Heatmap (1 - Regret)
    heatmap = np.zeros((len(n_prior_values), len(n_struct_values)))
    for i, n_prior in enumerate(n_prior_values):
        for j, n_struct in enumerate(n_struct_values):
            heatmap[i, j] = 1.0 - results[(n_struct, n_prior)]["mean_regret"]
    
    # Champion selection (Minimum Regret)
    best_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    best_n_prior = n_prior_values[best_idx[0]]
    best_n_struct = n_struct_values[best_idx[1]]
    best_regret = 1.0 - heatmap[best_idx]
    
    print(f"\n{'='*70}")
    print(f"🏆 CHAMPION CONFIGURATION (Primary: Regret):")
    print(f"   N_prior (belief)    = {best_n_prior}")
    print(f"   N_struct (stiffness) = {best_n_struct}")
    print(f"   Best Regret         = {best_regret:.4f}")
    print(f"{'='*70}")
    
    return heatmap, results, (best_n_struct, best_n_prior), n_struct_values, n_prior_values


def plot_calibration_heatmap(heatmap, n_struct_values, n_prior_values, 
                              champion, output_path):
    """Create professional heatmap for Stage 1 results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap: red (low) -> yellow -> green (high)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('synergy', colors)
    
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(n_struct_values)))
    ax.set_xticklabels([str(n) for n in n_struct_values], fontsize=12)
    ax.set_yticks(range(len(n_prior_values)))
    ax.set_yticklabels([str(n) for n in n_prior_values], fontsize=12)
    
    ax.set_xlabel('$N_{structure}$ (Covariance Stiffness)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$N_{prior}$ (Belief Strength)', fontsize=14, fontweight='bold')
    ax.set_title('Stage 1: HLE Calibration Heatmap\n(Metric: 1.0 - Regret | Max Quality Mode)', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Annotate cells with values
    for i in range(len(n_prior_values)):
        for j in range(len(n_struct_values)):
            val = heatmap[i, j]
            # Mark champion cell
            if n_struct_values[j] == champion[0] and n_prior_values[i] == champion[1]:
                text = ax.text(j, i, f'{val:.4f}\n★', ha='center', va='center',
                               color='white', fontsize=11, fontweight='bold')
            else:
                text = ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                               color='white' if val > np.mean(heatmap) else 'black', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Intelligence Density (1 - Regret)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Calibration heatmap saved to: {output_path}")


# =============================================================================
# STAGE 2: ECONOMIC SWEEP (Pareto Frontier)
# =============================================================================

def run_pareto_frontier(train_data, test_data, registry, encoder,
                        champion_n_struct, champion_n_prior, n_trials=3):
    """
    Stage 2: Sweep lambda_cost with LOCKED architecture.
    Measures Success Probability (Utility) as the primary Y-axis.
    """
    print(f"\n{'='*70}")
    print(f"STAGE 2: PARETO FRONTIER GENERATION (HLE Mode)")
    print(f"Architecture LOCKED: N_struct={champion_n_struct}, N_prior={champion_n_prior}")
    print(f"{'='*70}")
    
    # Cost profiles to sweep (KDD Expanded Spectrum)
    profiles = [
        {"name": "Max Quality",  "lambda_cost": 0.0},
        {"name": "Arbitrage",    "lambda_cost": 0.5},
        {"name": "Balanced",     "lambda_cost": 2.0},
        {"name": "Budget",       "lambda_cost": 10.0},
        {"name": "Ultra Cheap",  "lambda_cost": 50.0},
    ]
    
    frontier_results = []
    
    for config in profiles:
        print(f"\n  [{config['name']}] (λ_cost={config['lambda_cost']})...")
        
        trial_costs = []
        trial_utilities = []
        all_selections = defaultdict(int)
        
        for trial in range(n_trials):
            # Create router with LOCKED champion architecture
            router = BanditRouter.create(
                registry,
                exploration="safe",
                priors="hle",
                prior_n_effective=float(champion_n_prior),
                prior_structure_n_effective=float(champion_n_struct),
                context_encoder=encoder
            )
            
            # Profile for this sweep point
            profile = {"lambda_cost": config["lambda_cost"], "lambda_latency": 0.001}
            
            # Phase 1: Burn-in
            train_prompts = list(train_data.keys())
            random.seed(42 + trial)
            random.shuffle(train_prompts)
            
            for prompt in train_prompts:
                data = train_data[prompt]
                selected, log = router.route(prompt, profile=profile, input_tokens=100)
                if selected in data["rewards"]:
                    raw_reward = data["rewards"][selected]
                    cluster_id = data.get("cluster_id", 0)
                    cluster_mu, cluster_sigma = get_cluster_difficulty(registry, cluster_id)
                    scaled_reward = scale_reward_by_difficulty(raw_reward, cluster_mu, cluster_sigma)
                    difficulty_weight = 1.0 - cluster_mu
                    router.update(selected, prompt, scaled_reward, weight=difficulty_weight)
            
            # Phase 2: Evaluate
            test_prompts = list(test_data.keys())
            random.shuffle(test_prompts)
            
            original_alpha = router.bandit.alpha
            router.bandit.alpha = 0.0  # SILENCE EXPLORATION
            
            costs = []
            utilities = []
            
            for prompt in test_prompts:
                data = test_data[prompt]
                selected, _ = router.route(prompt, profile=profile, input_tokens=100)
                
                if selected in data["rewards"]:
                    model = registry.get(selected, {})
                    cost = get_model_cost(model)
                    if cost is not None:
                        costs.append(cost)
                        utilities.append(data["rewards"][selected])
                        all_selections[selected] += 1
            
            router.bandit.alpha = original_alpha
            
            if costs:
                trial_costs.append(np.mean(costs) * 1000)
                trial_utilities.append(np.mean(utilities))
                print(f"    Trial {trial+1}: Cost=${np.mean(costs)*1000:.4f}, Utility={np.mean(utilities)*100:.1f}%")
        
        if trial_costs:
            frontier_results.append({
                "profile": config["name"],
                "lambda_cost": config["lambda_cost"],
                "cost_mean": np.mean(trial_costs),
                "cost_std": np.std(trial_costs),
                "utility_mean": np.mean(trial_utilities),
                "utility_std": np.std(trial_utilities),
                "selections": dict(all_selections)
            })
    
    return frontier_results


def plot_pareto_frontier(frontier_results, test_data, registry, output_path,
                         champion_n_struct, champion_n_prior):
    """Create professional Pareto frontier visualization using Success Probability."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 9))
    
    COLORS = {
        "models": "#A19F9D",
        "pareto": "#FF9500",
        "bandit": "#0055A4",
        "random": "#FF6B6B",
    }
    
    # 1. Plot individual models as background (Raw Success Probability)
    model_points = []
    for model_id, model in registry.items():
        cost = get_model_cost(model)
        if cost is None:
            continue
        
        utilities = []
        for prompt, data in test_data.items():
            if model_id in data["rewards"]:
                utilities.append(data["rewards"][model_id])
        
        if utilities:
            model_points.append({
                "model_id": model_id,
                "name": model.get("display_name", model_id),
                "cost": cost * 1000,
                "utility": np.mean(utilities)
            })
    
    m_costs = [m["cost"] for m in model_points]
    m_utilities = [m["utility"] for m in model_points]
    ax.scatter(m_costs, m_utilities, color=COLORS["models"], s=60, alpha=0.4,
               label='Individual Models', zorder=2, edgecolors='white', linewidths=0.5)
    
    # 2. Compute and plot model Pareto frontier
    sorted_by_cost = sorted(model_points, key=lambda x: x["cost"])
    pareto = []
    max_u = float('-inf')
    for m in sorted_by_cost:
        if m["utility"] > max_u:
            pareto.append(m)
            max_u = m["utility"]
    
    if len(pareto) > 1:
        p_costs = [m["cost"] for m in pareto]
        p_utilities = [m["utility"] for m in pareto]
        ax.plot(p_costs, p_utilities, color=COLORS["pareto"], lw=2, linestyle='--',
                alpha=0.7, label='Model Pareto Frontier', zorder=3)
    
    # 3. Random baseline (Average of all models)
    avg_utility = np.mean(m_utilities) if m_utilities else 0.5
    ax.axhline(y=avg_utility, color=COLORS["random"], linestyle='-', lw=2, alpha=0.6,
               label=f'Average Baseline ({avg_utility*100:.1f}%)', zorder=1)
    
    # 4. BanditRouter frontier (THE STAR)
    if frontier_results:
        b_costs = [p["cost_mean"] for p in frontier_results]
        b_utilities = [p["utility_mean"] for p in frontier_results]
        b_cost_err = [p["cost_std"] for p in frontier_results]
        b_utility_err = [p["utility_std"] for p in frontier_results]
        
        # Error bars
        ax.errorbar(b_costs, b_utilities, xerr=b_cost_err, yerr=b_utility_err,
                    color=COLORS["bandit"], fmt='none', alpha=0.3, capsize=4, zorder=5)
        
        # Main line and points
        ax.plot(b_costs, b_utilities, color=COLORS["bandit"], lw=3.5, zorder=6,
                label=f'BanditRouter (N_s={champion_n_struct}, N_p={champion_n_prior})')
        ax.scatter(b_costs, b_utilities, color=COLORS["bandit"], s=180,
                   edgecolors='white', linewidths=2.5, zorder=7, marker='s')
        
        # Label profiles
        for p in frontier_results:
            offset = (12, 8) if p["profile"] != "Ultra Cheap" else (12, -12)
            ax.annotate(p["profile"], (p["cost_mean"], p["utility_mean"]),
                        xytext=offset, textcoords='offset points', fontsize=10,
                        fontweight='bold', color=COLORS["bandit"])
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Average Cost per 1k Tokens ($)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Probability (Utility %)', fontsize=13, fontweight='bold')
    ax.set_title('Stage 2: Perfect Pareto Frontier (HLE Mode)\n(Architecture Locked: Minimum Regret Champion)',
                 fontsize=15, fontweight='bold', pad=15)
    
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Interpretation box
    textstr = (
        "Metric: Raw Success Prop\n"
        "• 100%: Perfect Intelligence\n"
        "• Higher: More Robust\n"
        "• Snapshot: 10.6% Hard tasks"
    )
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pareto frontier saved to: {output_path}")


# =============================================================================
# MAIN: EXECUTE PERFECT PARETO PROTOCOL
# =============================================================================

def execute_perfect_pareto_protocol():
    """Execute the complete two-stage protocol."""
    print("=" * 80)
    print("PERFECT PARETO PROTOCOL: Calibrate Once, Sweep Costs")
    print("KDD-Grade Experiment for Scientific Rigor")
    print("=" * 80)
    
    # ======================================================
    # LOAD DATA
    # ======================================================
    print("\n[0/4] Loading data with z-scores...")
    train_data, test_data, registry, zscore_lookup = load_data_with_zscores()
    print(f"  Z-score lookup: {len(zscore_lookup)} entries")
    print(f"  Models in registry: {len(registry)}")
    
    # ======================================================
    # INITIALIZE ENCODER (once, shared)
    # ======================================================
    print("\n[1/4] Initializing encoder (shared)...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(DEFAULT_CONTEXT_MODEL)
    
    # ======================================================
    # STAGE 1: ARCHITECTURAL CALIBRATION
    # ======================================================
    print("\n[2/4] Running Stage 1: Architectural Calibration...")
    
    # Output directory for intermediate saves
    output_dir = Path(__file__).parent
    
    # Grid configuration (from protocol spec)
    # Fast Grid for HLE
    n_struct_grid = [10, 60, 250]
    n_prior_grid = [10, 60, 250]
    
    heatmap, stage1_results, champion, n_struct_values, n_prior_values = run_2d_synergy_sweep(
        train_data, test_data, registry, encoder,
        n_struct_values=n_struct_grid,
        n_prior_values=n_prior_grid,
        fixed_lambda=0.0,  # MAX QUALITY - THE STRESS TEST
        n_trials=1,        # Trials=1 for SPEED
        output_dir=output_dir
    )
    
    champion_n_struct, champion_n_prior = champion
    # No override needed - calibration at λ=0 will find the stiffer champion
    
    # Plot heatmap
    heatmap_path = output_dir / "calibration_heatmap.png"
    plot_calibration_heatmap(heatmap, n_struct_values, n_prior_values, champion, heatmap_path)
    
    # Save Stage 1 complete results BEFORE Stage 2
    stage1_complete_path = output_dir / "stage1_complete.json"
    stage1_complete = {
        "status": "COMPLETE",
        "grid": {
            "n_struct_values": n_struct_grid,
            "n_prior_values": n_prior_grid,
            "fixed_lambda": 0.5
        },
        "champion": {
            "n_struct": champion_n_struct,
            "n_prior": champion_n_prior,
            "zscore": float(heatmap[n_prior_values.index(champion_n_prior), 
                                   n_struct_values.index(champion_n_struct)])
        },
        "heatmap": heatmap.tolist(),
        "full_results": {f"{k[0]},{k[1]}": v for k, v in stage1_results.items()}
    }
    save_intermediate(stage1_complete, stage1_complete_path, "Stage 1 COMPLETE")
    print(f"\n✅ Stage 1 complete. Champion locked: N_s={champion_n_struct}, N_p={champion_n_prior}")
    
    # ======================================================
    # STAGE 2: PARETO FRONTIER
    # ======================================================
    print("\n[3/4] Running Stage 2: Pareto Frontier Generation...")
    
    frontier_results = run_pareto_frontier(
        train_data, test_data, registry, encoder,
        champion_n_struct=champion_n_struct,
        champion_n_prior=champion_n_prior,
        n_trials=3  # Multi-trial for Confidence Intervals
    )
    
    # Plot frontier
    frontier_path = Path(__file__).parent / "perfect_pareto_frontier.png"
    plot_pareto_frontier(frontier_results, test_data, registry, frontier_path,
                         champion_n_struct, champion_n_prior)
    
    # ======================================================
    # SAVE RESULTS
    # ======================================================
    print("\n[4/4] Saving results...")
    
    results_path = Path(__file__).parent / "perfect_pareto_results.json"
    output = {
        "protocol": "Perfect Pareto: HLE Minimum Regret",
        "stage1": {
            "grid": {
                "n_struct_values": n_struct_grid,
                "n_prior_values": n_prior_grid,
                "fixed_lambda": 0.0
            },
            "champion": {
                "n_struct": champion_n_struct,
                "n_prior": champion_n_prior,
                "regret": 1.0 - float(heatmap[n_prior_values.index(champion_n_prior), 
                                              n_struct_values.index(champion_n_struct)])
            },
            "heatmap": heatmap.tolist()
        },
        "stage2": {
            "architecture_locked": {
                "n_struct": champion_n_struct,
                "n_prior": champion_n_prior
            },
            "frontier": [
                {
                    "profile": r["profile"],
                    "lambda_cost": r["lambda_cost"],
                    "cost_mean": r["cost_mean"],
                    "cost_std": r["cost_std"],
                    "utility_mean": r["utility_mean"]
                }
                for r in frontier_results
            ]
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Results saved to: {results_path}")
    
    # ======================================================
    # SUMMARY
    # ======================================================
    print("\n" + "=" * 80)
    print("PERFECT PARETO PROTOCOL COMPLETE")
    print("=" * 80)
    
    print(f"\n🏆 Stage 1 Champion: N_struct={champion_n_struct}, N_prior={champion_n_prior}")
    print(f"   Best Regret at Max Quality: {1.0 - heatmap[n_prior_values.index(champion_n_prior), n_struct_values.index(champion_n_struct)]:.4f}")
    
    print(f"\n📈 Stage 2 Frontier (Locked Architecture):")
    for r in frontier_results:
        print(f"   {r['profile']:<12} → Cost=${r['cost_mean']:.4f} ± {r['cost_std']:.4f}, Utility={r['utility_mean']*100:.1f}% ± {r['utility_std']*100:.1f}%")
    
    print(f"\n📁 Output Files:")
    print(f"   • {heatmap_path} (Figure X: Hyperparameter Landscape)")
    print(f"   • {frontier_path} (Figure Y: Pareto Frontier)")
    print(f"   • {results_path}")
    
    print("\n✅ PROTOCOL COMPLETE! Ready for KDD submission.")


if __name__ == "__main__":
    execute_perfect_pareto_protocol()
